from __future__ import annotations
import logging
from typing import List
from PySide6.QtCore import QThread, Signal
from PIL import Image

from .api_client import MitosisAPIClient, DetectionResult, APIConfig

class MitosisDetectionWorker(QThread):
    """서버 API를 사용한 Mitosis 감지 백그라운드 워커 쓰레드"""

    # 시그널 정의
    progress_updated = Signal(str)  # 진행 상태 메시지
    detection_completed = Signal(list)  # 감지 결과 (List[DetectionResult])
    detection_failed = Signal(str)  # 에러 메시지

    def __init__(self, image: Image.Image, api_config: APIConfig = None):
        super().__init__()
        self.image = image
        self.api_config = api_config
        self.logger = logging.getLogger(__name__)

    def run(self):
        """메인 실행 함수"""
        try:
            self.progress_updated.emit("Connecting to detection server...")

            # API 클라이언트 초기화
            client = MitosisAPIClient(config=self.api_config)

            if not client.is_ready():
                raise RuntimeError("Detection server is not available. Please check server connection.")

            self.progress_updated.emit("Sending image to server for analysis...")

            # 감지 수행
            results = client.detect_from_pil(self.image)

            self.progress_updated.emit(f"Detection complete: {len(results)} mitosis found")

            # 결과 반환
            self.detection_completed.emit(results)

        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.detection_failed.emit(error_msg)