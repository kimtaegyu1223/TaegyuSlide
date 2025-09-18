from __future__ import annotations
import time
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
from threading import Thread
import numpy as np
from PIL import Image

from .slide_processor import SlideProcessor, SlideAnalysis, PatchInfo
from .mitosis_detector import MitosisDetector, DetectionResult
from ..config import CONFIG


@dataclass
class PatchData:
    """패치 데이터와 메타정보를 포함하는 클래스"""
    patch_info: PatchInfo
    image: Image.Image
    extraction_time: float


@dataclass
class ProcessingResult:
    """처리 결과를 포함하는 클래스"""
    patch_info: PatchInfo
    detections: List[DetectionResult]
    processing_time: float
    success: bool = True
    error_message: str = ""


def extract_patch_worker(backend_params: Dict, patch_queue: mp.Queue,
                        result_queue: mp.Queue, stop_event: mp.Event):
    """패치 추출 워커 프로세스"""
    logger = logging.getLogger(f"PatchExtractor-{mp.current_process().pid}")

    try:
        # Backend 재생성 (프로세스간 공유 불가)
        import openslide
        slide = openslide.OpenSlide(backend_params['slide_path'])

        while not stop_event.is_set():
            try:
                # 타임아웃을 사용하여 패치 정보 가져오기
                patch_info = patch_queue.get(timeout=1.0)
                if patch_info is None:  # 종료 신호
                    break

                start_time = time.time()

                # 패치 추출
                image = slide.read_region(
                    (patch_info.x, patch_info.y),
                    patch_info.level,
                    (patch_info.width // int(slide.level_downsamples[patch_info.level]),
                     patch_info.height // int(slide.level_downsamples[patch_info.level]))
                ).convert('RGB')

                extraction_time = time.time() - start_time

                patch_data = PatchData(
                    patch_info=patch_info,
                    image=image,
                    extraction_time=extraction_time
                )

                result_queue.put(patch_data)

            except mp.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"패치 추출 실패 {patch_info.patch_id if 'patch_info' in locals() else 'unknown'}: {e}")

    except Exception as e:
        logger.error(f"패치 추출 워커 오류: {e}")
    finally:
        if 'slide' in locals():
            slide.close()


def detection_worker(model_path: str, patch_queue: mp.Queue,
                    result_queue: mp.Queue, stop_event: mp.Event):
    """AI 추론 워커 프로세스"""
    logger = logging.getLogger(f"DetectionWorker-{mp.current_process().pid}")

    try:
        # 각 프로세스에서 모델 초기화
        detector = MitosisDetector(model_path=model_path)

        while not stop_event.is_set():
            try:
                # 타임아웃을 사용하여 패치 데이터 가져오기
                patch_data = patch_queue.get(timeout=1.0)
                if patch_data is None:  # 종료 신호
                    break

                start_time = time.time()

                # AI 추론 실행
                detections = detector.detect_from_pil(patch_data.image)
                processing_time = time.time() - start_time

                # 패치 좌표를 절대 좌표로 변환
                converted_detections = []
                for detection in detections:
                    rel_x1, rel_y1, rel_x2, rel_y2 = detection.bbox

                    # 패치 내 상대 좌표를 절대 좌표로 변환
                    abs_x1 = patch_data.patch_info.x + (rel_x1 / patch_data.image.width) * patch_data.patch_info.width
                    abs_y1 = patch_data.patch_info.y + (rel_y1 / patch_data.image.height) * patch_data.patch_info.height
                    abs_x2 = patch_data.patch_info.x + (rel_x2 / patch_data.image.width) * patch_data.patch_info.width
                    abs_y2 = patch_data.patch_info.y + (rel_y2 / patch_data.image.height) * patch_data.patch_info.height

                    converted_detection = DetectionResult(
                        bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                        confidence=detection.confidence,
                        class_id=detection.class_id
                    )
                    converted_detections.append(converted_detection)

                result = ProcessingResult(
                    patch_info=patch_data.patch_info,
                    detections=converted_detections,
                    processing_time=processing_time,
                    success=True
                )

                result_queue.put(result)

            except mp.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"AI 추론 실패: {e}")
                if 'patch_data' in locals():
                    error_result = ProcessingResult(
                        patch_info=patch_data.patch_info,
                        detections=[],
                        processing_time=0,
                        success=False,
                        error_message=str(e)
                    )
                    result_queue.put(error_result)

    except Exception as e:
        logger.error(f"AI 추론 워커 오류: {e}")


class MultiprocessPatchProcessor:
    """멀티프로세스 패치 처리기"""

    def __init__(self, backend, target_magnification: str = None,
                 patch_size: int = None, num_extract_workers: int = None,
                 num_detection_workers: int = None):
        self.backend = backend
        self.target_magnification = target_magnification or CONFIG.ai.default_magnification
        self.patch_size = patch_size or CONFIG.ai.default_patch_size
        self.num_extract_workers = num_extract_workers or CONFIG.ai.default_worker_count
        self.num_detection_workers = num_detection_workers or max(1, CONFIG.ai.default_worker_count // 2)

        self.logger = logging.getLogger(__name__)

        # 큐 및 프로세스 관리
        self.patch_info_queue = None
        self.extracted_patch_queue = None
        self.result_queue = None
        self.extract_workers = []
        self.detection_workers = []
        self.stop_event = None

        # 통계
        self.total_patches = 0
        self.processed_patches = 0
        self.total_detections = 0

    def start_processing(self, analysis: SlideAnalysis) -> Iterator[ProcessingResult]:
        """멀티프로세스 처리 시작"""
        self.logger.info(f"멀티프로세스 처리 시작: {self.num_extract_workers}개 추출 워커, "
                        f"{self.num_detection_workers}개 추론 워커")

        # 큐 초기화
        self.patch_info_queue = mp.Queue(maxsize=CONFIG.ai.patch_queue_size)
        self.extracted_patch_queue = mp.Queue(maxsize=CONFIG.ai.patch_queue_size)
        self.result_queue = mp.Queue(maxsize=CONFIG.ai.result_queue_size)
        self.stop_event = mp.Event()

        # Backend 정보 준비
        backend_params = {
            'slide_path': str(self.backend.slide._filename)
        }

        try:
            # 워커 프로세스 시작
            self._start_workers(backend_params)

            # 패치 정보 제공 스레드 시작
            patch_feeder = Thread(target=self._feed_patches, args=(analysis,))
            patch_feeder.daemon = True
            patch_feeder.start()

            # 결과 수집
            self.total_patches = analysis.tissue_patches
            for result in self._collect_results():
                yield result

        finally:
            self._cleanup_workers()

    def _start_workers(self, backend_params: Dict):
        """워커 프로세스 시작"""
        # 패치 추출 워커들
        for i in range(self.num_extract_workers):
            worker = mp.Process(
                target=extract_patch_worker,
                args=(backend_params, self.patch_info_queue,
                     self.extracted_patch_queue, self.stop_event)
            )
            worker.start()
            self.extract_workers.append(worker)

        # AI 추론 워커들
        model_path = str(CONFIG.ai.model_path)
        for i in range(self.num_detection_workers):
            worker = mp.Process(
                target=detection_worker,
                args=(model_path, self.extracted_patch_queue,
                     self.result_queue, self.stop_event)
            )
            worker.start()
            self.detection_workers.append(worker)

    def _feed_patches(self, analysis: SlideAnalysis):
        """패치 정보를 큐에 공급"""
        processor = SlideProcessor(
            self.backend,
            target_magnification=self.target_magnification,
            patch_size=self.patch_size
        )

        try:
            for patch_info in processor.generate_patches(analysis):
                if self.stop_event.is_set():
                    break
                self.patch_info_queue.put(patch_info)

            # 종료 신호 전송
            for _ in range(self.num_extract_workers):
                self.patch_info_queue.put(None)

        except Exception as e:
            self.logger.error(f"패치 공급 오류: {e}")

    def _collect_results(self) -> Iterator[ProcessingResult]:
        """결과 수집"""
        finished_extract_workers = 0

        while self.processed_patches < self.total_patches:
            try:
                if not self.result_queue.empty():
                    result = self.result_queue.get(timeout=1.0)
                    self.processed_patches += 1

                    if result.success:
                        self.total_detections += len(result.detections)

                    yield result

                # 추출 워커가 모두 완료되었는지 확인
                if finished_extract_workers < self.num_extract_workers:
                    finished_count = sum(1 for w in self.extract_workers if not w.is_alive())
                    if finished_count == self.num_extract_workers:
                        finished_extract_workers = finished_count
                        # AI 워커들에 종료 신호
                        for _ in range(self.num_detection_workers):
                            self.extracted_patch_queue.put(None)

                time.sleep(0.01)  # CPU 사용률 조절

            except mp.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"결과 수집 오류: {e}")
                break

    def _cleanup_workers(self):
        """워커 프로세스 정리"""
        self.logger.info("워커 프로세스 정리 중...")

        # 중단 이벤트 설정
        if self.stop_event:
            self.stop_event.set()

        # 모든 워커 종료 대기
        all_workers = self.extract_workers + self.detection_workers
        for worker in all_workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

        # 큐 정리
        for queue in [self.patch_info_queue, self.extracted_patch_queue, self.result_queue]:
            if queue:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except:
                        break

    def stop_processing(self):
        """처리 중단"""
        if self.stop_event:
            self.stop_event.set()

    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            'total_patches': self.total_patches,
            'processed_patches': self.processed_patches,
            'total_detections': self.total_detections,
            'progress_percent': (self.processed_patches / self.total_patches * 100) if self.total_patches > 0 else 0,
            'extract_workers': self.num_extract_workers,
            'detection_workers': self.num_detection_workers
        }