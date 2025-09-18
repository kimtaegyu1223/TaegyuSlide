from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass

@dataclass
class TissueRegion:
    """티슈 영역 정보"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) - level 0 좌표
    area: int                        # 영역 면적 (픽셀)
    confidence: float                # 신뢰도 (0-1)

class TissueDetector:
    """썸네일을 사용한 티슈 영역 감지"""

    def __init__(self,
                 thumbnail_level: int = -1,  # 가장 낮은 해상도 레벨
                 min_tissue_area: int = 50,  # 최소 티슈 영역 크기
                 saturation_threshold: int = 15,  # 채도 임계값
                 gaussian_blur_kernel: int = 5):   # 가우시안 블러 커널 크기

        self.thumbnail_level = thumbnail_level
        self.min_tissue_area = min_tissue_area
        self.saturation_threshold = saturation_threshold
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.logger = logging.getLogger(__name__)

    def detect_tissue_regions(self, backend) -> List[TissueRegion]:
        """슬라이드에서 티슈 영역들을 감지"""

        # 썸네일 레벨 결정 (가장 낮은 해상도)
        if self.thumbnail_level == -1:
            thumbnail_level = backend.levels - 1
        else:
            thumbnail_level = min(self.thumbnail_level, backend.levels - 1)

        self.logger.info(f"Using thumbnail level {thumbnail_level} for tissue detection")

        # 썸네일 이미지 추출
        thumbnail_width, thumbnail_height = backend.dimensions[thumbnail_level]

        try:
            thumbnail_img = backend.slide.read_region(
                (0, 0), thumbnail_level, (thumbnail_width, thumbnail_height)
            ).convert('RGB')

            # PIL → OpenCV 형식 변환
            thumbnail_cv = cv2.cvtColor(np.array(thumbnail_img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            self.logger.error(f"Failed to extract thumbnail: {e}")
            return []

        # 티슈 마스크 생성
        tissue_mask = self._create_tissue_mask(thumbnail_cv)

        # 연결된 컴포넌트 찾기
        tissue_regions = self._find_tissue_components(tissue_mask, thumbnail_level, backend)

        self.logger.info(f"Found {len(tissue_regions)} tissue regions")

        return tissue_regions

    def _create_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        """이미지에서 티슈 마스크 생성"""

        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 채도(Saturation) 기반 티슈 감지
        # 배경(흰색/회색)은 채도가 낮고, 티슈는 채도가 높음
        saturation = hsv[:, :, 1]

        # 채도 임계값 적용
        tissue_mask = saturation > self.saturation_threshold

        # 추가적인 색상 기반 필터링 (선택사항)
        # 너무 밝거나 어두운 영역 제거
        value = hsv[:, :, 2]
        brightness_mask = (value > 20) & (value < 240)  # 너무 어둡거나 밝은 픽셀 제거

        # 최종 마스크 결합
        tissue_mask = tissue_mask & brightness_mask

        # 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

        # 가우시안 블러로 부드럽게 만들기
        if self.gaussian_blur_kernel > 1:
            tissue_mask = cv2.GaussianBlur(tissue_mask,
                                         (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)
            tissue_mask = tissue_mask > 127  # 이진화

        return tissue_mask.astype(np.uint8)

    def _find_tissue_components(self, tissue_mask: np.ndarray,
                              thumbnail_level: int, backend) -> List[TissueRegion]:
        """티슈 마스크에서 연결된 컴포넌트 찾기"""

        # 연결된 컴포넌트 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            tissue_mask, connectivity=8
        )

        tissue_regions = []
        downsample = backend.level_downsamples[thumbnail_level]

        # 각 컴포넌트 처리 (배경 제외)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            # 최소 크기 필터링
            if area < self.min_tissue_area:
                continue

            # 썸네일 좌표를 level 0 좌표로 변환
            level0_x = int(x * downsample)
            level0_y = int(y * downsample)
            level0_w = int(w * downsample)
            level0_h = int(h * downsample)

            # 신뢰도 계산 (영역 크기와 모양을 기반으로)
            aspect_ratio = w / h if h > 0 else 1.0
            shape_score = 1.0 / (1.0 + abs(aspect_ratio - 1.0))  # 정사각형에 가까울수록 높은 점수
            size_score = min(area / (tissue_mask.shape[0] * tissue_mask.shape[1]), 1.0)
            confidence = (shape_score + size_score) / 2.0

            tissue_region = TissueRegion(
                bbox=(level0_x, level0_y, level0_w, level0_h),
                area=int(area * downsample * downsample),  # level 0 기준 면적
                confidence=confidence
            )

            tissue_regions.append(tissue_region)

        # 면적 기준 내림차순 정렬
        tissue_regions.sort(key=lambda r: r.area, reverse=True)

        return tissue_regions

    def visualize_tissue_detection(self, image: Image.Image,
                                 tissue_regions: List[TissueRegion],
                                 thumbnail_level: int, backend) -> Image.Image:
        """티슈 감지 결과를 시각화"""

        # PIL → OpenCV 변환
        vis_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        downsample = backend.level_downsamples[thumbnail_level]

        # 각 티슈 영역에 바운딩 박스 그리기
        for i, region in enumerate(tissue_regions):
            # Level 0 좌표를 썸네일 좌표로 변환
            x = int(region.bbox[0] / downsample)
            y = int(region.bbox[1] / downsample)
            w = int(region.bbox[2] / downsample)
            h = int(region.bbox[3] / downsample)

            # 색상 (티슈 영역별로 다른 색상)
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
            color = colors[i % len(colors)]

            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # 영역 정보 텍스트
            text = f"Region {i+1}: {region.area//1000}K px"
            cv2.putText(vis_image, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # OpenCV → PIL 변환
        return Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

    def filter_patches_by_tissue(self, patch_infos: List,
                               tissue_regions: List[TissueRegion],
                               overlap_threshold: float = 0.1) -> List:
        """패치 목록을 티슈 영역과 겹치는 것들만 필터링"""

        filtered_patches = []

        for patch in patch_infos:
            patch_bbox = (patch.x, patch.y, patch.width, patch.height)

            # 각 티슈 영역과의 겹침 검사
            for tissue_region in tissue_regions:
                overlap_ratio = self._calculate_overlap_ratio(patch_bbox, tissue_region.bbox)

                if overlap_ratio > overlap_threshold:
                    filtered_patches.append(patch)
                    break  # 하나라도 겹치면 포함

        self.logger.info(f"Filtered patches: {len(patch_infos)} → {len(filtered_patches)} "
                        f"({len(filtered_patches)/len(patch_infos)*100:.1f}%)")

        return filtered_patches

    def _calculate_overlap_ratio(self, bbox1: Tuple[int, int, int, int],
                               bbox2: Tuple[int, int, int, int]) -> float:
        """두 바운딩 박스의 겹침 비율 계산"""

        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # 교집합 영역 계산
        intersect_x1 = max(x1_1, x1_2)
        intersect_y1 = max(y1_1, y1_2)
        intersect_x2 = min(x2_1, x2_2)
        intersect_y2 = min(y2_1, y2_2)

        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return 0.0

        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        patch_area = w1 * h1

        return intersect_area / patch_area if patch_area > 0 else 0.0