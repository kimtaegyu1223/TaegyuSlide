from __future__ import annotations
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from PySide6.QtCore import QThread, Signal

from .multiprocess_patch_extractor import MultiprocessPatchProcessor, ProcessingResult
from .slide_processor import SlideProcessor, SlideAnalysis
from .mitosis_detector import DetectionResult
from ..config import CONFIG


@dataclass
class RealTimeStats:
    """실시간 처리 통계"""
    total_patches: int
    processed_patches: int
    detected_mitosis: int
    processing_speed: float  # patches/second
    elapsed_time: float      # seconds
    estimated_remaining: float  # seconds
    extract_workers: int
    detection_workers: int
    current_batch_detections: int = 0


class RealTimeDetectionWorker(QThread):
    """실시간 Mitosis 감지 워커 - 멀티프로세스 + 실시간 표시"""

    # 시그널 정의 - 실시간 업데이트용
    analysis_completed = Signal(object)  # SlideAnalysis
    progress_updated = Signal(str, float)  # message, progress (0-100)
    stats_updated = Signal(object)  # RealTimeStats

    # 실시간 결과 시그널
    patch_result_ready = Signal(object)  # ProcessingResult (패치별 즉시 표시)
    detection_batch_ready = Signal(list)  # List[DetectionResult] (배치 단위 업데이트)

    # 완료/에러 시그널
    detection_completed = Signal(list, object)  # List[DetectionResult], final_stats
    detection_failed = Signal(str)  # error message

    def __init__(self, backend, target_magnification: str = None,
                 patch_size: int = None, num_extract_workers: int = None,
                 num_detection_workers: int = None):
        super().__init__()
        self.backend = backend
        self.target_magnification = target_magnification or CONFIG.ai.default_magnification
        self.patch_size = patch_size or CONFIG.ai.default_patch_size
        self.num_extract_workers = num_extract_workers or CONFIG.ai.default_worker_count
        self.num_detection_workers = num_detection_workers or max(1, CONFIG.ai.default_worker_count // 2)

        self.logger = logging.getLogger(__name__)

        # 처리 상태
        self.should_stop = False
        self.start_time = 0
        self.all_results: List[DetectionResult] = []
        self.processor: MultiprocessPatchProcessor = None

        # 실시간 표시 설정
        self.enable_real_time = CONFIG.ai.enable_real_time_display
        self.batch_update_size = 10  # 몇 개 패치마다 배치 업데이트를 할지

    def stop_processing(self):
        """처리 중단 요청"""
        self.should_stop = True
        if self.processor:
            self.processor.stop_processing()

    def run(self):
        """메인 실행 함수"""
        try:
            self.start_time = time.time()
            self.all_results = []

            # 1. 슬라이드 분석
            self.progress_updated.emit("슬라이드 구조 분석 중...", 5)
            slide_processor = SlideProcessor(
                self.backend,
                target_magnification=self.target_magnification,
                patch_size=self.patch_size
            )
            analysis = slide_processor.analyze_slide()
            self.analysis_completed.emit(analysis)

            if analysis.tissue_patches == 0:
                self.detection_failed.emit("슬라이드에서 조직 영역을 찾을 수 없습니다")
                return

            tissue_regions_count = len(analysis.tissue_regions) if analysis.tissue_regions else 0
            self.progress_updated.emit(
                f"{tissue_regions_count}개 조직 영역, {analysis.tissue_patches}개 패치 처리 예정", 10
            )

            # 2. 멀티프로세스 처리기 초기화
            self.progress_updated.emit("멀티프로세스 처리기 초기화 중...", 15)
            self.processor = MultiprocessPatchProcessor(
                self.backend,
                target_magnification=self.target_magnification,
                patch_size=self.patch_size,
                num_extract_workers=self.num_extract_workers,
                num_detection_workers=self.num_detection_workers
            )

            # 3. 실시간 처리 시작
            self.progress_updated.emit("실시간 처리 시작...", 20)
            self._process_with_real_time_updates(analysis)

            if not self.should_stop:
                final_stats = self._create_final_stats(analysis)
                self.progress_updated.emit("처리 완료!", 100)
                self.detection_completed.emit(self.all_results, final_stats)

        except Exception as e:
            error_msg = f"처리 실패: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.detection_failed.emit(error_msg)

    def _process_with_real_time_updates(self, analysis: SlideAnalysis):
        """실시간 업데이트와 함께 처리"""
        processed_count = 0
        batch_results = []
        last_update_time = time.time()
        update_interval = 1.0  # 1초마다 통계 업데이트

        try:
            # 멀티프로세스 처리 시작
            for result in self.processor.start_processing(analysis):
                if self.should_stop:
                    break

                processed_count += 1

                # 결과 수집
                if result.success:
                    self.all_results.extend(result.detections)
                    batch_results.append(result)

                    # 실시간 개별 결과 표시 (설정된 경우)
                    if self.enable_real_time:
                        self.patch_result_ready.emit(result)

                # 배치 단위 업데이트
                if len(batch_results) >= self.batch_update_size:
                    batch_detections = []
                    for batch_result in batch_results:
                        batch_detections.extend(batch_result.detections)

                    if batch_detections:
                        self.detection_batch_ready.emit(batch_detections)

                    batch_results = []

                # 통계 업데이트 (시간 간격 고려)
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    self._update_stats(analysis, processed_count)
                    last_update_time = current_time

            # 마지막 배치 처리
            if batch_results:
                batch_detections = []
                for batch_result in batch_results:
                    batch_detections.extend(batch_result.detections)

                if batch_detections:
                    self.detection_batch_ready.emit(batch_detections)

        except Exception as e:
            self.logger.error(f"실시간 처리 오류: {e}")
            raise

    def _update_stats(self, analysis: SlideAnalysis, processed_count: int):
        """통계 업데이트"""
        elapsed_time = time.time() - self.start_time
        speed = processed_count / elapsed_time if elapsed_time > 0 else 0
        remaining_patches = analysis.tissue_patches - processed_count
        estimated_remaining = remaining_patches / speed if speed > 0 else 0

        progress = 20 + (processed_count / analysis.tissue_patches) * 75 if analysis.tissue_patches > 0 else 20

        stats = RealTimeStats(
            total_patches=analysis.tissue_patches,
            processed_patches=processed_count,
            detected_mitosis=len(self.all_results),
            processing_speed=speed,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
            extract_workers=self.num_extract_workers,
            detection_workers=self.num_detection_workers
        )

        # 시그널 발생
        self.stats_updated.emit(stats)
        self.progress_updated.emit(
            f"처리 중: {processed_count}/{analysis.tissue_patches} 패치 "
            f"({len(self.all_results)}개 검출, {speed:.1f} 패치/초)",
            progress
        )

    def _create_final_stats(self, analysis: SlideAnalysis) -> RealTimeStats:
        """최종 통계 생성"""
        elapsed_time = time.time() - self.start_time
        final_speed = analysis.tissue_patches / elapsed_time if elapsed_time > 0 else 0

        return RealTimeStats(
            total_patches=analysis.tissue_patches,
            processed_patches=analysis.tissue_patches,
            detected_mitosis=len(self.all_results),
            processing_speed=final_speed,
            elapsed_time=elapsed_time,
            estimated_remaining=0,
            extract_workers=self.num_extract_workers,
            detection_workers=self.num_detection_workers
        )