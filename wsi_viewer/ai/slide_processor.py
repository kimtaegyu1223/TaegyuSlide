from __future__ import annotations
import math
import logging
from typing import List, Tuple, Iterator, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from .tissue_detector import TissueDetector, TissueRegion
from ..config import CONFIG

@dataclass
class PatchInfo:
    """패치 정보를 저장하는 클래스"""
    x: int                # Level 0에서의 x 좌표
    y: int                # Level 0에서의 y 좌표
    width: int            # 패치 너비
    height: int           # 패치 높이
    level: int            # 추출할 레벨
    patch_id: int         # 패치 고유 ID
    row: int              # 격자에서의 행
    col: int              # 격자에서의 열

@dataclass
class SlideAnalysis:
    """슬라이드 분석 결과"""
    total_patches: int    # 전체 패치 수
    tissue_patches: int   # 티슈 영역 패치 수
    optimal_level: int    # 최적 레벨
    patch_size: int       # 패치 크기 (픽셀)
    overlap: int          # 오버랩 크기 (픽셀)
    grid_rows: int        # 격자 행 수
    grid_cols: int        # 격자 열 수
    mpp: float           # 마이크로미터/픽셀
    recommended_magnification: str  # 권장 배율
    tissue_regions: List[TissueRegion]  # 감지된 티슈 영역들
    tissue_coverage: float  # 티슈 영역 커버리지 (0-1)

class SlideProcessor:
    """전체 슬라이드를 패치로 분할하고 처리하는 클래스"""

    def __init__(self, backend, target_magnification: str = None,
                 patch_size: int = None, overlap_ratio: float = None,
                 enable_tissue_detection: bool = True):
        self.backend = backend
        self.target_magnification = target_magnification or CONFIG.ai.default_magnification
        self.patch_size = patch_size or CONFIG.ai.default_patch_size
        self.overlap_ratio = overlap_ratio or CONFIG.ai.overlap_ratio
        self.enable_tissue_detection = enable_tissue_detection
        self.logger = logging.getLogger(__name__)

        # 티슈 감지기 초기화
        self.tissue_detector = TissueDetector() if enable_tissue_detection else None

        # 배율별 추천 MPP (마이크로미터/픽셀)
        self.magnification_mpp = {
            "40x": 0.25,   # 40배율
            "20x": 0.5,    # 20배율
            "10x": 1.0,    # 10배율
            "5x": 2.0      # 5배율
        }

    def analyze_slide(self) -> SlideAnalysis:
        """슬라이드를 분석하여 최적의 처리 방법 결정"""
        if not self.backend:
            raise ValueError("Backend가 초기화되지 않았습니다")

        self.logger.info("Starting slide analysis...")

        # 슬라이드 MPP 정보 가져오기
        slide_mpp_x = self.backend.mpp_x or 0.25  # 기본값: 40배율
        slide_mpp_y = self.backend.mpp_y or 0.25
        slide_mpp = (slide_mpp_x + slide_mpp_y) / 2

        # 목표 배율에 맞는 최적 레벨 찾기
        target_mpp = self.magnification_mpp.get(self.target_magnification, 0.25)
        optimal_level = self._find_optimal_level(slide_mpp, target_mpp)

        # 해당 레벨에서의 슬라이드 크기
        level_width, level_height = self.backend.dimensions[optimal_level]

        # 오버랩 계산
        overlap = int(self.patch_size * self.overlap_ratio)
        step_size = self.patch_size - overlap

        # 격자 계산
        grid_cols = math.ceil(level_width / step_size)
        grid_rows = math.ceil(level_height / step_size)
        total_patches = grid_cols * grid_rows

        # 실제 배율 계산
        actual_magnification = self._calculate_magnification(slide_mpp, optimal_level)

        # 티슈 영역 감지
        tissue_regions = []
        tissue_patches = total_patches
        tissue_coverage = 1.0

        if self.tissue_detector:
            self.logger.info("Detecting tissue regions...")
            tissue_regions = self.tissue_detector.detect_tissue_regions(self.backend)

            if tissue_regions:
                # 티슈 영역과 겹치는 패치만 계산
                all_patches = list(self.generate_patches_basic(
                    optimal_level, grid_rows, grid_cols, step_size))
                filtered_patches = self.tissue_detector.filter_patches_by_tissue(
                    all_patches, tissue_regions)
                tissue_patches = len(filtered_patches)

                # 티슈 커버리지 계산
                total_tissue_area = sum(region.area for region in tissue_regions)
                slide_area = level_width * level_height * (self.backend.level_downsamples[optimal_level] ** 2)
                tissue_coverage = min(total_tissue_area / slide_area, 1.0) if slide_area > 0 else 0.0
            else:
                self.logger.warning("No tissue regions detected - will process entire slide")

        analysis = SlideAnalysis(
            total_patches=total_patches,
            tissue_patches=tissue_patches,
            optimal_level=optimal_level,
            patch_size=self.patch_size,
            overlap=overlap,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            mpp=slide_mpp,
            recommended_magnification=actual_magnification,
            tissue_regions=tissue_regions,
            tissue_coverage=tissue_coverage
        )

        efficiency = (tissue_patches / total_patches * 100) if total_patches > 0 else 0
        self.logger.info(f"Analysis complete: {grid_cols}x{grid_rows} total patches, "
                        f"{tissue_patches} tissue patches ({efficiency:.1f}% efficiency)")

        return analysis

    def _find_optimal_level(self, slide_mpp: float, target_mpp: float) -> int:
        """목표 MPP에 가장 가까운 레벨 찾기"""
        best_level = 0
        best_diff = float('inf')

        for level in range(self.backend.levels):
            level_ds = self.backend.level_downsamples[level]
            level_mpp = slide_mpp * level_ds
            diff = abs(level_mpp - target_mpp)

            if diff < best_diff:
                best_diff = diff
                best_level = level

        return best_level

    def _calculate_magnification(self, slide_mpp: float, level: int) -> str:
        """실제 배율 계산"""
        level_ds = self.backend.level_downsamples[level]
        actual_mpp = slide_mpp * level_ds

        # 가장 가까운 표준 배율 찾기
        standard_mags = {
            0.25: "40x",
            0.5: "20x",
            1.0: "10x",
            2.0: "5x"
        }

        closest_mpp = min(standard_mags.keys(), key=lambda x: abs(x - actual_mpp))
        return standard_mags[closest_mpp]

    def generate_patches_basic(self, optimal_level: int, grid_rows: int,
                             grid_cols: int, step_size: int) -> Iterator[PatchInfo]:
        """기본 패치 정보를 순차적으로 생성 (티슈 필터링 없이)"""
        level_width, level_height = self.backend.dimensions[optimal_level]
        downsample = self.backend.level_downsamples[optimal_level]

        patch_id = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Level 좌표에서의 위치
                level_x = col * step_size
                level_y = row * step_size

                # 패치 크기 조정 (슬라이드 경계 고려)
                patch_width = min(self.patch_size, level_width - level_x)
                patch_height = min(self.patch_size, level_height - level_y)

                # 너무 작은 패치는 건너뛰기
                if patch_width < self.patch_size // 2 or patch_height < self.patch_size // 2:
                    continue

                # Level 0 좌표로 변환
                level0_x = int(level_x * downsample)
                level0_y = int(level_y * downsample)
                level0_width = int(patch_width * downsample)
                level0_height = int(patch_height * downsample)

                yield PatchInfo(
                    x=level0_x,
                    y=level0_y,
                    width=level0_width,
                    height=level0_height,
                    level=optimal_level,
                    patch_id=patch_id,
                    row=row,
                    col=col
                )

                patch_id += 1

    def generate_patches(self, analysis: SlideAnalysis) -> Iterator[PatchInfo]:
        """티슈 영역을 고려한 패치 정보를 순차적으로 생성"""
        step_size = analysis.patch_size - analysis.overlap

        # 모든 패치 생성
        all_patches = list(self.generate_patches_basic(
            analysis.optimal_level, analysis.grid_rows, analysis.grid_cols, step_size))

        # 티슈 영역 필터링 적용
        if self.tissue_detector and analysis.tissue_regions:
            filtered_patches = self.tissue_detector.filter_patches_by_tissue(
                all_patches, analysis.tissue_regions)

            # 패치 ID 재할당
            for i, patch in enumerate(filtered_patches):
                patch.patch_id = i

            for patch in filtered_patches:
                yield patch
        else:
            # 티슈 감지 없이 모든 패치 반환
            for patch in all_patches:
                yield patch

    def extract_patch(self, patch_info: PatchInfo) -> Image.Image:
        """단일 패치 추출"""
        try:
            # OpenSlide는 level 0 좌표와 목표 레벨을 받음
            image = self.backend.slide.read_region(
                (patch_info.x, patch_info.y),
                patch_info.level,
                (patch_info.width // self.backend.level_downsamples[patch_info.level],
                 patch_info.height // self.backend.level_downsamples[patch_info.level])
            ).convert('RGB')

            return image

        except Exception as e:
            self.logger.error(f"패치 추출 실패 {patch_info.patch_id}: {e}")
            raise

    def get_patch_batches(self, analysis: SlideAnalysis, batch_size: int) -> Iterator[List[PatchInfo]]:
        """패치들을 배치로 묶어서 반환"""
        batch = []
        for patch_info in self.generate_patches(analysis):
            batch.append(patch_info)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # 마지막 배치
        if batch:
            yield batch

    def estimate_processing_time(self, analysis: SlideAnalysis,
                               patches_per_second: float = 1.0) -> Dict[str, Any]:
        """처리 시간 예상 (티슈 패치 기준)"""
        effective_patches = analysis.tissue_patches
        estimated_seconds = effective_patches / patches_per_second

        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        seconds = int(estimated_seconds % 60)

        efficiency = (analysis.tissue_patches / analysis.total_patches * 100) if analysis.total_patches > 0 else 0

        return {
            "total_patches": analysis.total_patches,
            "tissue_patches": analysis.tissue_patches,
            "estimated_seconds": estimated_seconds,
            "formatted_time": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "patches_per_second": patches_per_second,
            "efficiency_percent": efficiency,
            "tissue_regions_count": len(analysis.tissue_regions),
            "tissue_coverage": analysis.tissue_coverage
        }