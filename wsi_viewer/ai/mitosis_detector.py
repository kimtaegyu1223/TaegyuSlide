from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2
from ..config import CONFIG

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class DetectionResult:
    """감지 결과를 저장하는 클래스"""
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, class_id: int = 0):
        self.bbox = bbox  # (x1, y1, x2, y2) in original image coordinates
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = "mitosis"

    def __repr__(self):
        return f"DetectionResult(bbox={self.bbox}, conf={self.confidence:.3f})"

class MitosisDetector:
    """YOLOv12 ONNX 모델을 사용한 Mitosis 감지기"""

    def __init__(self, model_path: str = None, config_path: str = None):
        self.logger = logging.getLogger(__name__)

        # config에서 기본 경로 설정
        if model_path is None:
            model_path = CONFIG.ai.model_path
        if config_path is None:
            config_path = CONFIG.ai.config_path

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        # 설정 로드
        self.config = self._load_config()
        self.session = None
        self.input_name = None
        self.output_names = None

        # 모델 초기화
        self._initialize_model()

    def _load_config(self) -> Dict[str, Any]:
        """모델 설정 파일 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return config.get("mitosis_yolov12", {})
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                # 기본 설정 반환
                return {
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.4,
                    "class_names": ["mitosis"],
                    "preprocessing": {
                        "normalize": True,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    }
                }
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    def _initialize_model(self):
        """ONNX 모델 초기화"""
        if ort is None:
            raise RuntimeError(
                "onnxruntime가 설치되지 않았습니다. "
                "다음 명령으로 설치하세요: pip install onnxruntime-gpu"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        try:
            # ONNX Runtime 세션 생성
            providers = self._get_providers()
            self.logger.info(f"Using providers: {providers}")

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )

            # 입력/출력 정보 가져오기
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]

            input_shape = self.session.get_inputs()[0].shape
            self.logger.info(f"Model initialized. Input shape: {input_shape}")
            self.logger.info(f"Input name: {self.input_name}")
            self.logger.info(f"Output names: {self.output_names}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX model: {e}")
            raise

    def _get_providers(self) -> List[str]:
        """사용 가능한 실행 제공자 반환"""
        providers = []

        # TensorRT 사용 가능한지 확인
        available_providers = ort.get_available_providers()

        if "TensorrtExecutionProvider" in available_providers:
            providers.append("TensorrtExecutionProvider")
            self.logger.info("TensorRT provider available")

        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            self.logger.info("CUDA provider available")

        # CPU는 항상 사용 가능
        providers.append("CPUExecutionProvider")

        return providers

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # config에서 모델 입력 크기 가져오기, 없으면 설정에서 가져오기
        from ..config import CONFIG
        model_input_size = CONFIG.ai.model_input_size
        input_size = self.config.get("input_size", [model_input_size, model_input_size])
        preprocess_config = self.config.get("preprocessing", {})

        # 크기 조정
        h, w = image.shape[:2]
        target_h, target_w = input_size

        # 종횡비 유지하며 리사이즈
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 패딩 추가 (회색으로)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # RGB로 변환 및 정규화
        processed = padded.astype(np.float32) / 255.0

        if preprocess_config.get("normalize", False):
            mean = np.array(preprocess_config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.array(preprocess_config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
            processed = (processed - mean) / std

        # CHW 형식으로 변환 및 배치 차원 추가
        processed = processed.transpose(2, 0, 1)
        processed = np.expand_dims(processed, axis=0)

        # 데이터 타입을 float32로 확실히 보장
        processed = processed.astype(np.float32)

        return processed, scale, (pad_x, pad_y)

    def postprocess_outputs(self, outputs: List[np.ndarray], scale: float,
                          pad_offset: Tuple[int, int], original_size: Tuple[int, int]) -> List[DetectionResult]:
        """모델 출력 후처리"""
        if not outputs or len(outputs) == 0:
            return []

        predictions = outputs[0]  # Shape: (1, num_detections, 85) for YOLOv12

        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension

        confidence_threshold = self.config.get("confidence_threshold", 0.5)
        nms_threshold = self.config.get("nms_threshold", 0.4)

        # 신뢰도 필터링
        confidences = predictions[:, 4]
        valid_indices = confidences > confidence_threshold
        valid_predictions = predictions[valid_indices]

        if len(valid_predictions) == 0:
            return []

        # 바운딩 박스 변환
        boxes = valid_predictions[:, :4]  # (center_x, center_y, width, height)
        confidences = valid_predictions[:, 4]

        # 중심좌표 + 크기 → 모서리 좌표
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        boxes = np.column_stack([x1, y1, x2, y2])

        # NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            confidence_threshold,
            nms_threshold
        )

        results = []
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            pad_x, pad_y = pad_offset
            original_w, original_h = original_size

            for i in indices:
                box = boxes[i]
                conf = confidences[i]

                # 좌표를 원본 이미지 크기로 변환
                x1 = (box[0] - pad_x) / scale
                y1 = (box[1] - pad_y) / scale
                x2 = (box[2] - pad_x) / scale
                y2 = (box[3] - pad_y) / scale

                # 좌표 범위 제한
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))

                results.append(DetectionResult((x1, y1, x2, y2), conf))

        return results

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """이미지에서 mitosis 감지"""
        if self.session is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")

        original_size = (image.shape[1], image.shape[0])  # (width, height)

        try:
            # 전처리
            processed_image, scale, pad_offset = self.preprocess_image(image)

            # 추론
            outputs = self.session.run(
                self.output_names,
                {self.input_name: processed_image}
            )

            # 후처리
            results = self.postprocess_outputs(outputs, scale, pad_offset, original_size)

            self.logger.info(f"Detected {len(results)} mitosis candidates")
            return results

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise

    def detect_from_pil(self, pil_image: Image.Image) -> List[DetectionResult]:
        """PIL 이미지에서 mitosis 감지"""
        # PIL → OpenCV 형식 변환
        image_array = np.array(pil_image.convert('RGB'))
        # RGB → BGR 변환
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return self.detect(image_array)

    def is_ready(self) -> bool:
        """모델이 사용 가능한 상태인지 확인"""
        return self.session is not None and self.model_path.exists()