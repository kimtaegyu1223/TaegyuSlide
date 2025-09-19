from __future__ import annotations
import logging
import base64
import io
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import requests
import json

@dataclass
class DetectionResult:
    """감지 결과를 저장하는 클래스"""
    def __init__(self, bbox: tuple[float, float, float, float], confidence: float, class_id: int = 0):
        self.bbox = bbox  # (x1, y1, x2, y2) in original image coordinates
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = "mitosis"

    def __repr__(self):
        return f"DetectionResult(bbox={self.bbox}, conf={self.confidence:.3f})"

@dataclass
class APIConfig:
    """API 설정"""
    base_url: str = "http://localhost:8000"
    detection_endpoint: str = "/api/v1/detect"
    timeout: int = 30
    max_retries: int = 3

class MitosisAPIClient:
    """서버 기반 Mitosis 감지 API 클라이언트"""

    def __init__(self, config: APIConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or APIConfig()
        self.session = requests.Session()

        # 세션 설정
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _image_to_base64(self, image: Image.Image) -> str:
        """PIL 이미지를 base64 문자열로 변환"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def _parse_response(self, response_data: Dict[str, Any]) -> List[DetectionResult]:
        """API 응답을 DetectionResult 리스트로 변환"""
        results = []

        if 'detections' not in response_data:
            return results

        for detection in response_data['detections']:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            class_id = detection.get('class_id', 0)

            # bbox 형식이 [x, y, w, h]인 경우 [x1, y1, x2, y2]로 변환
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox = (x, y, x + w, y + h)

            result = DetectionResult(
                bbox=tuple(bbox),
                confidence=confidence,
                class_id=class_id
            )
            results.append(result)

        return results

    def detect_from_pil(self, image: Image.Image, **kwargs) -> List[DetectionResult]:
        """PIL 이미지에서 mitosis 감지"""
        try:
            # 이미지를 base64로 인코딩
            image_b64 = self._image_to_base64(image)

            # API 요청 데이터 준비
            request_data = {
                'image': image_b64,
                'image_format': 'jpeg',
                'confidence_threshold': kwargs.get('confidence_threshold', 0.5),
                'nms_threshold': kwargs.get('nms_threshold', 0.4)
            }

            # API 요청
            url = f"{self.config.base_url}{self.config.detection_endpoint}"

            self.logger.info(f"Sending detection request to {url}")

            response = self.session.post(
                url,
                json=request_data,
                timeout=self.config.timeout
            )

            response.raise_for_status()

            # 응답 파싱
            response_data = response.json()
            results = self._parse_response(response_data)

            self.logger.info(f"Received {len(results)} detections from server")
            return results

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise RuntimeError(f"Detection API request failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            raise RuntimeError(f"Invalid API response format: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during detection: {e}")
            raise RuntimeError(f"Detection failed: {e}")

    def detect_batch(self, images: List[Image.Image], **kwargs) -> List[List[DetectionResult]]:
        """여러 이미지에 대한 배치 감지"""
        results = []

        for i, image in enumerate(images):
            try:
                self.logger.debug(f"Processing image {i+1}/{len(images)}")
                image_results = self.detect_from_pil(image, **kwargs)
                results.append(image_results)
            except Exception as e:
                self.logger.error(f"Failed to process image {i+1}: {e}")
                results.append([])  # 빈 결과 추가

        return results

    def is_ready(self) -> bool:
        """API 서버 연결 상태 확인"""
        try:
            health_url = f"{self.config.base_url}/health"
            response = self.session.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """서버 정보 조회"""
        try:
            info_url = f"{self.config.base_url}/api/v1/info"
            response = self.session.get(info_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get server info: {e}")
            return {}