from __future__ import annotations
import logging
from typing import List
from PySide6.QtCore import QThread, Signal
from PIL import Image

from .mitosis_detector import MitosisDetector, DetectionResult

class MitosisDetectionWorker(QThread):
    """Mitosis 감지를 위한 백그라운드 워커 쓰레드"""

    # 시그널 정의
    progress_updated = Signal(str)  # 진행 상태 메시지
    detection_completed = Signal(list)  # 감지 결과 (List[DetectionResult])
    detection_failed = Signal(str)  # 에러 메시지

    def __init__(self, image: Image.Image, model_path: str = None):
        super().__init__()
        self.image = image
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

    def run(self):
        """메인 실행 함수"""
        try:
            self.progress_updated.emit("Initializing detector...")

            # 감지기 초기화
            detector = MitosisDetector(model_path=self.model_path)

            if not detector.is_ready():
                raise RuntimeError("감지기 초기화 실패 - 모델 파일을 확인하세요")

            self.progress_updated.emit("Running inference...")

            # 감지 수행
            results = detector.detect_from_pil(self.image)

            self.progress_updated.emit(f"Detection complete: {len(results)} mitosis found")

            # 결과 반환
            self.detection_completed.emit(results)

        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.detection_failed.emit(error_msg)