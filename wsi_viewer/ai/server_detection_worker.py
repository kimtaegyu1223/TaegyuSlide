from __future__ import annotations
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from PySide6.QtCore import QThread, Signal
from PIL import Image

from .api_client import ObjectDetectionAPIClient, DetectionResult, APIConfig
from .slide_processor import SlideProcessor, SlideAnalysis, PatchInfo
from ..overlay import ObjectDetection

@dataclass
class ProcessingStats:
    """처리 통계"""
    processed_patches: int = 0
    total_patches: int = 0
    detected_objects: int = 0
    processing_speed: float = 0.0  # patches/second
    estimated_remaining: float = 0.0  # seconds
    start_time: float = 0.0

class ServerBasedDetectionWorker(QThread):
    """서버 API를 사용한 전체 슬라이드 감지 워커"""

    # 시그널 정의
    analysis_completed = Signal(object)  # SlideAnalysis
    progress_updated = Signal(str, float)  # message, progress(0-100)
    stats_updated = Signal(object)  # ProcessingStats
    detection_completed = Signal(list)  # List[ObjectDetection]
    detection_failed = Signal(str)  # error message

    def __init__(self, backend, target_magnification: str = "20x",
                 patch_size: int = 512, api_config: APIConfig = None):
        super().__init__()
        self.backend = backend
        self.target_magnification = target_magnification
        self.patch_size = patch_size
        self.api_config = api_config
        self.logger = logging.getLogger(__name__)

        # 처리 상태
        self.should_stop = False
        self.stats = ProcessingStats()

    def stop(self):
        """처리 중단"""
        self.should_stop = True

    def run(self):
        """메인 실행 함수"""
        try:
            self.stats.start_time = time.time()

            # 1. 슬라이드 분석
            self.progress_updated.emit("Analyzing slide structure...", 5)
            processor = SlideProcessor(
                self.backend,
                target_magnification=self.target_magnification,
                patch_size=self.patch_size
            )

            analysis = processor.analyze_slide()
            self.analysis_completed.emit(analysis)

            if self.should_stop:
                return

            # 2. API 클라이언트 초기화
            self.progress_updated.emit("Connecting to detection server...", 10)
            client = ObjectDetectionAPIClient(config=self.api_config)

            if not client.is_ready():
                raise RuntimeError("Detection server is not available")

            # 3. 패치 처리
            self.progress_updated.emit("Processing tissue patches...", 15)

            self.stats.total_patches = analysis.tissue_patches
            all_detections = []

            patch_generator = processor.extract_tissue_patches()

            for i, (patch_info, patch_image) in enumerate(patch_generator):
                if self.should_stop:
                    break

                try:
                    # 서버에 패치 전송 및 감지
                    patch_results = client.detect_from_pil(patch_image)

                    # 좌표 변환 (패치 좌표 → Level 0 좌표)
                    for result in patch_results:
                        x1, y1, x2, y2 = result.bbox

                        # 패치 내 좌표를 Level 0 좌표로 변환
                        global_x1 = patch_info.x + x1
                        global_y1 = patch_info.y + y1
                        global_x2 = patch_info.x + x2
                        global_y2 = patch_info.y + y2

                        detection = ObjectDetection(
                            bbox=(global_x1, global_y1, global_x2, global_y2),
                            confidence=result.confidence,
                            level0_coords=True
                        )
                        all_detections.append(detection)

                    self.stats.detected_objects = len(all_detections)

                except Exception as e:
                    self.logger.warning(f"Failed to process patch {patch_info.patch_id}: {e}")

                # 통계 업데이트
                self.stats.processed_patches = i + 1
                elapsed_time = time.time() - self.stats.start_time

                if elapsed_time > 0:
                    self.stats.processing_speed = self.stats.processed_patches / elapsed_time

                if self.stats.processing_speed > 0:
                    remaining_patches = self.stats.total_patches - self.stats.processed_patches
                    self.stats.estimated_remaining = remaining_patches / self.stats.processing_speed

                # 진행률 계산 (15% ~ 95%)
                progress = 15 + (80 * self.stats.processed_patches / self.stats.total_patches)

                message = f"Processing patch {self.stats.processed_patches}/{self.stats.total_patches}"
                self.progress_updated.emit(message, progress)
                self.stats_updated.emit(self.stats)

            # 완료
            if not self.should_stop:
                self.progress_updated.emit("Detection completed!", 100)
                self.detection_completed.emit(all_detections)

        except Exception as e:
            error_msg = f"Detection processing failed: {str(e)}"
            self.logger.error(error_msg)
            self.detection_failed.emit(error_msg)

class BatchDetectionWorker(QThread):
    """배치 단위 감지를 위한 워커 (대용량 슬라이드용)"""

    # 시그널 정의
    analysis_completed = Signal(object)
    progress_updated = Signal(str, float)
    stats_updated = Signal(object)
    batch_result_ready = Signal(list)  # 배치 결과
    detection_completed = Signal(list)
    detection_failed = Signal(str)

    def __init__(self, backend, target_magnification: str = "20x",
                 patch_size: int = 512, batch_size: int = 8, api_config: APIConfig = None):
        super().__init__()
        self.backend = backend
        self.target_magnification = target_magnification
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.api_config = api_config
        self.logger = logging.getLogger(__name__)

        self.should_stop = False
        self.stats = ProcessingStats()

    def stop(self):
        self.should_stop = True

    def run(self):
        """배치 단위로 처리"""
        try:
            self.stats.start_time = time.time()

            # 슬라이드 분석
            self.progress_updated.emit("Analyzing slide...", 5)
            processor = SlideProcessor(
                self.backend,
                target_magnification=self.target_magnification,
                patch_size=self.patch_size
            )

            analysis = processor.analyze_slide()
            self.analysis_completed.emit(analysis)

            if self.should_stop:
                return

            # API 클라이언트 초기화
            client = ObjectDetectionAPIClient(config=self.api_config)
            if not client.is_ready():
                raise RuntimeError("Detection server is not available")

            # 배치 처리
            self.stats.total_patches = analysis.tissue_patches
            all_detections = []

            patch_generator = processor.extract_tissue_patches()
            batch_patches = []
            batch_infos = []

            for patch_info, patch_image in patch_generator:
                if self.should_stop:
                    break

                batch_patches.append(patch_image)
                batch_infos.append(patch_info)

                # 배치 크기에 도달하면 처리
                if len(batch_patches) >= self.batch_size:
                    batch_detections = self._process_batch(client, batch_patches, batch_infos)
                    all_detections.extend(batch_detections)

                    # 배치 결과 즉시 전송
                    self.batch_result_ready.emit(batch_detections)

                    self._update_stats(len(batch_patches))

                    # 배치 초기화
                    batch_patches = []
                    batch_infos = []

            # 남은 패치 처리
            if batch_patches and not self.should_stop:
                batch_detections = self._process_batch(client, batch_patches, batch_infos)
                all_detections.extend(batch_detections)
                self.batch_result_ready.emit(batch_detections)

            # 완료
            if not self.should_stop:
                self.progress_updated.emit("Batch processing completed!", 100)
                self.detection_completed.emit(all_detections)

        except Exception as e:
            error_msg = f"Batch detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.detection_failed.emit(error_msg)

    def _process_batch(self, client: ObjectDetectionAPIClient,
                      batch_patches: List[Image.Image],
                      batch_infos: List[PatchInfo]) -> List[ObjectDetection]:
        """배치 처리"""
        batch_detections = []

        try:
            # 배치 감지 수행
            batch_results = client.detect_batch(batch_patches)

            # 결과 변환
            for patch_info, patch_results in zip(batch_infos, batch_results):
                for result in patch_results:
                    x1, y1, x2, y2 = result.bbox

                    global_x1 = patch_info.x + x1
                    global_y1 = patch_info.y + y1
                    global_x2 = patch_info.x + x2
                    global_y2 = patch_info.y + y2

                    detection = MitosisDetection(
                        bbox=(global_x1, global_y1, global_x2, global_y2),
                        confidence=result.confidence,
                        level0_coords=True
                    )
                    batch_detections.append(detection)

        except Exception as e:
            self.logger.warning(f"Batch processing failed: {e}")

        return batch_detections

    def _update_stats(self, batch_size: int):
        """통계 업데이트"""
        self.stats.processed_patches += batch_size
        elapsed_time = time.time() - self.stats.start_time

        if elapsed_time > 0:
            self.stats.processing_speed = self.stats.processed_patches / elapsed_time

        if self.stats.processing_speed > 0:
            remaining_patches = self.stats.total_patches - self.stats.processed_patches
            self.stats.estimated_remaining = remaining_patches / self.stats.processing_speed

        progress = 15 + (80 * self.stats.processed_patches / self.stats.total_patches)
        message = f"Processed {self.stats.processed_patches}/{self.stats.total_patches} patches"

        self.progress_updated.emit(message, progress)
        self.stats_updated.emit(self.stats)