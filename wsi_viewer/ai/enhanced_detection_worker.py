from __future__ import annotations
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from PySide6.QtCore import QThread, Signal

from .mitosis_detector import MitosisDetector, DetectionResult
from .slide_processor import SlideProcessor, SlideAnalysis, PatchInfo
from .gpu_manager import GPUManager

@dataclass
class ProcessingStats:
    """처리 통계 정보"""
    total_patches: int
    processed_patches: int
    detected_mitosis: int
    processing_speed: float  # patches/second
    elapsed_time: float      # seconds
    estimated_remaining: float  # seconds

class EnhancedMitosisDetectionWorker(QThread):
    """향상된 Mitosis 감지 워커 - 전체 슬라이드 처리"""

    # 시그널 정의
    analysis_completed = Signal(object)  # SlideAnalysis
    progress_updated = Signal(str, float)  # message, progress (0-100)
    stats_updated = Signal(object)  # ProcessingStats
    patch_completed = Signal(int, int)  # patch_id, detections_count
    detection_completed = Signal(list)  # List[DetectionResult] (모든 결과)
    detection_failed = Signal(str)  # error message

    def __init__(self, backend, target_magnification: str = "40x",
                 patch_size: int = None, custom_batch_size: int = None, model_path: str = None):
        super().__init__()
        self.backend = backend
        self.target_magnification = target_magnification
        self.patch_size = patch_size
        self.custom_batch_size = custom_batch_size
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

        # GPU 관리자
        self.gpu_manager = GPUManager()

        # 처리 상태
        self.should_stop = False
        self.start_time = 0
        self.all_results: List[DetectionResult] = []

    def stop_processing(self):
        """처리 중단 요청"""
        self.should_stop = True

    def run(self):
        """메인 실행 함수"""
        try:
            self.start_time = time.time()
            self.all_results = []

            # 1. GPU 설정 최적화
            self.progress_updated.emit("Optimizing GPU settings...", 0)
            gpu_config = self.gpu_manager.get_optimal_config(self.custom_batch_size)
            batch_size = gpu_config["batch_size"]

            self.logger.info(f"Using GPU config: {gpu_config['description']}, batch_size={batch_size}")

            # 2. 슬라이드 분석
            self.progress_updated.emit("Analyzing slide structure...", 5)
            processor = SlideProcessor(
                self.backend,
                target_magnification=self.target_magnification,
                patch_size=self.patch_size,
                enable_tissue_detection=True
            )
            analysis = processor.analyze_slide()
            self.analysis_completed.emit(analysis)

            if analysis.tissue_patches == 0:
                self.detection_failed.emit("No tissue regions found in the slide")
                return

            self.progress_updated.emit(
                f"Found {analysis.tissue_regions_count} tissue regions, "
                f"{analysis.tissue_patches} patches to process", 10
            )

            # 3. 모델 초기화
            self.progress_updated.emit("Initializing detection model...", 15)
            detector = MitosisDetector(model_path=self.model_path)

            if not detector.is_ready():
                self.detection_failed.emit("Failed to initialize detection model")
                return

            # 4. 배치 처리
            self.progress_updated.emit("Starting inference...", 20)
            self._process_batches(processor, analysis, detector, batch_size)

            if not self.should_stop:
                self.progress_updated.emit("Detection completed!", 100)
                self.detection_completed.emit(self.all_results)

        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.detection_failed.emit(error_msg)

    def _process_batches(self, processor: SlideProcessor, analysis: SlideAnalysis,
                        detector: MitosisDetector, batch_size: int):
        """배치 단위로 패치 처리"""

        processed_count = 0
        total_patches = analysis.tissue_patches

        # 패치를 배치로 나누어 처리
        for batch_patches in processor.get_patch_batches(analysis, batch_size):
            if self.should_stop:
                break

            batch_start_time = time.time()

            # 배치의 이미지들 추출
            batch_images = []
            valid_patches = []

            for patch_info in batch_patches:
                try:
                    image = processor.extract_patch(patch_info)
                    batch_images.append(image)
                    valid_patches.append(patch_info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract patch {patch_info.patch_id}: {e}")

            if not batch_images:
                continue

            # 배치 추론 실행
            batch_results = self._process_image_batch(detector, batch_images, valid_patches)

            # 결과 수집
            batch_detections = 0
            for patch_results in batch_results:
                self.all_results.extend(patch_results)
                batch_detections += len(patch_results)

            # 진행률 업데이트
            processed_count += len(valid_patches)
            progress = 20 + (processed_count / total_patches) * 75  # 20-95% 범위

            batch_time = time.time() - batch_start_time
            speed = len(valid_patches) / batch_time if batch_time > 0 else 0

            # 통계 업데이트
            elapsed_time = time.time() - self.start_time
            remaining_patches = total_patches - processed_count
            estimated_remaining = remaining_patches / speed if speed > 0 else 0

            stats = ProcessingStats(
                total_patches=total_patches,
                processed_patches=processed_count,
                detected_mitosis=len(self.all_results),
                processing_speed=speed,
                elapsed_time=elapsed_time,
                estimated_remaining=estimated_remaining
            )

            # 시그널 발생
            self.stats_updated.emit(stats)
            self.progress_updated.emit(
                f"Processed {processed_count}/{total_patches} patches "
                f"({len(self.all_results)} detections, {speed:.1f} patches/sec)",
                progress
            )

            # 배치별 결과 알림
            for i, patch_results in enumerate(batch_results):
                if i < len(valid_patches):
                    self.patch_completed.emit(valid_patches[i].patch_id, len(patch_results))

    def _process_image_batch(self, detector: MitosisDetector,
                           batch_images: List, batch_patches: List[PatchInfo]) -> List[List[DetectionResult]]:
        """이미지 배치를 처리하여 감지 결과 반환"""

        batch_results = []

        # 개별 이미지 처리 (현재는 배치 추론 미지원)
        for i, image in enumerate(batch_images):
            try:
                results = detector.detect_from_pil(image)

                # 패치 좌표를 level 0 좌표로 변환
                patch_info = batch_patches[i]
                converted_results = []

                for detection in results:
                    # 패치 내 상대 좌표를 절대 좌표로 변환
                    rel_x1, rel_y1, rel_x2, rel_y2 = detection.bbox

                    # 패치 내 좌표를 level 0 절대 좌표로 변환
                    # detection.bbox는 이미 패치 이미지 픽셀 좌표 (0 ~ patch_image_size)
                    # 이를 슬라이드 좌표로 변환
                    scale_x = patch_info.width / image.width
                    scale_y = patch_info.height / image.height

                    abs_x1 = float(patch_info.x + (rel_x1 * scale_x))
                    abs_y1 = float(patch_info.y + (rel_y1 * scale_y))
                    abs_x2 = float(patch_info.x + (rel_x2 * scale_x))
                    abs_y2 = float(patch_info.y + (rel_y2 * scale_y))

                    print(f"패치 {patch_info.patch_id}: 원본좌표=({rel_x1:.1f}, {rel_y1:.1f}, {rel_x2:.1f}, {rel_y2:.1f})")
                    print(f"패치 정보: x={patch_info.x}, y={patch_info.y}, w={patch_info.width}, h={patch_info.height}")
                    print(f"이미지 크기: {image.width}x{image.height}")
                    print(f"스케일: {scale_x:.3f}, {scale_y:.3f}")
                    print(f"최종 좌표: ({abs_x1:.1f}, {abs_y1:.1f}, {abs_x2:.1f}, {abs_y2:.1f})")

                    # 새로운 DetectionResult 생성
                    converted_result = DetectionResult(
                        bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                        confidence=float(detection.confidence),
                        class_id=int(detection.class_id)
                    )
                    converted_results.append(converted_result)

                batch_results.append(converted_results)

            except Exception as e:
                self.logger.error(f"Failed to process image {i}: {e}")
                batch_results.append([])  # 빈 결과

        return batch_results