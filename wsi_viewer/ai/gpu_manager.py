from __future__ import annotations
import logging
import platform
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class GPUInfo:
    """GPU 정보"""
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: str
    driver_version: str
    recommended_batch_size: int

class GPUManager:
    """GPU 정보 관리 및 성능 최적화"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._gpu_info: Optional[GPUInfo] = None
        self._gpu_configs = self._load_gpu_configs()

    def _load_gpu_configs(self) -> Dict[str, Dict[str, Any]]:
        """미리 정의된 GPU별 최적 설정"""
        return {
            # NVIDIA RTX 시리즈
            "rtx 4090": {
                "batch_size": 16,
                "vram_limit_mb": 20000,
                "threads": 8,
                "description": "RTX 4090 - High-end"
            },
            "rtx 4080": {
                "batch_size": 12,
                "vram_limit_mb": 14000,
                "threads": 6,
                "description": "RTX 4080 - High-end"
            },
            "rtx 4070": {
                "batch_size": 8,
                "vram_limit_mb": 10000,
                "threads": 4,
                "description": "RTX 4070 - Mid-high"
            },
            "rtx 3090": {
                "batch_size": 14,
                "vram_limit_mb": 20000,
                "threads": 8,
                "description": "RTX 3090 - High-end"
            },
            "rtx 3080": {
                "batch_size": 10,
                "vram_limit_mb": 8000,
                "threads": 6,
                "description": "RTX 3080 - High-end"
            },
            "rtx 3070": {
                "batch_size": 6,
                "vram_limit_mb": 6000,
                "threads": 4,
                "description": "RTX 3070 - Mid-high"
            },
            "rtx 3060": {
                "batch_size": 4,
                "vram_limit_mb": 10000,
                "threads": 4,
                "description": "RTX 3060 - Mid-range"
            },
            # GTX 시리즈
            "gtx 1660": {
                "batch_size": 2,
                "vram_limit_mb": 5000,
                "threads": 2,
                "description": "GTX 1660 - Entry-level"
            },
            "gtx 1080": {
                "batch_size": 4,
                "vram_limit_mb": 7000,
                "threads": 4,
                "description": "GTX 1080 - Legacy high-end"
            },
            # 기본값 (GPU 감지 실패 시)
            "default": {
                "batch_size": 2,
                "vram_limit_mb": 4000,
                "threads": 2,
                "description": "Default settings"
            }
        }

    def detect_gpu(self) -> GPUInfo:
        """현재 시스템의 GPU 정보 감지"""
        if self._gpu_info is not None:
            return self._gpu_info

        gpu_info = self._detect_gpu_info()
        self._gpu_info = gpu_info

        self.logger.info(f"Detected GPU: {gpu_info.name} "
                        f"({gpu_info.memory_available}MB available)")

        return gpu_info

    def _detect_gpu_info(self) -> GPUInfo:
        """실제 GPU 정보 감지 로직"""
        try:
            # NVIDIA GPU 감지 시도
            gpu_info = self._detect_nvidia_gpu()
            if gpu_info:
                return gpu_info
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU detection failed: {e}")

        try:
            # ONNX Runtime을 통한 GPU 정보 감지
            gpu_info = self._detect_onnx_gpu()
            if gpu_info:
                return gpu_info
        except Exception as e:
            self.logger.debug(f"ONNX GPU detection failed: {e}")

        # 기본값 반환
        return self._create_default_gpu_info()

    def _detect_nvidia_gpu(self) -> Optional[GPUInfo]:
        """NVIDIA GPU 감지 (nvidia-ml-py 사용)"""
        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None

            # 첫 번째 GPU 정보 가져오기
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

            # 메모리 정보
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory_info.total // (1024 * 1024)  # MB
            memory_available = memory_info.free // (1024 * 1024)  # MB

            # 기타 정보
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                driver_version = "Unknown"

            # 권장 배치 크기 계산
            recommended_batch = self._calculate_batch_size(name, memory_available)

            return GPUInfo(
                name=name,
                memory_total=memory_total,
                memory_available=memory_available,
                compute_capability="Unknown",
                driver_version=driver_version,
                recommended_batch_size=recommended_batch
            )

        except ImportError:
            self.logger.debug("pynvml not available")
            return None
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU detection error: {e}")
            return None

    def _detect_onnx_gpu(self) -> Optional[GPUInfo]:
        """ONNX Runtime을 통한 GPU 감지"""
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()

            if "CUDAExecutionProvider" in available_providers:
                # CUDA 사용 가능
                return GPUInfo(
                    name="CUDA Device",
                    memory_total=8000,  # 추정값
                    memory_available=6000,  # 추정값
                    compute_capability="Unknown",
                    driver_version="Unknown",
                    recommended_batch_size=4
                )
            elif "TensorrtExecutionProvider" in available_providers:
                # TensorRT 사용 가능
                return GPUInfo(
                    name="TensorRT Device",
                    memory_total=8000,  # 추정값
                    memory_available=6000,  # 추정값
                    compute_capability="Unknown",
                    driver_version="Unknown",
                    recommended_batch_size=6
                )
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"ONNX GPU detection error: {e}")

        return None

    def _create_default_gpu_info(self) -> GPUInfo:
        """기본 GPU 정보 생성"""
        return GPUInfo(
            name="Unknown GPU",
            memory_total=4000,
            memory_available=3000,
            compute_capability="Unknown",
            driver_version="Unknown",
            recommended_batch_size=2
        )

    def _calculate_batch_size(self, gpu_name: str, available_memory_mb: int) -> int:
        """GPU 이름과 메모리를 기반으로 권장 배치 크기 계산"""
        gpu_name_lower = gpu_name.lower()

        # 미리 정의된 GPU 설정에서 찾기
        for gpu_key, config in self._gpu_configs.items():
            if gpu_key in gpu_name_lower:
                # 메모리 여유도 확인
                if available_memory_mb >= config["vram_limit_mb"]:
                    return config["batch_size"]
                else:
                    # 메모리가 부족하면 배치 크기 줄이기
                    ratio = available_memory_mb / config["vram_limit_mb"]
                    return max(1, int(config["batch_size"] * ratio))

        # 알 수 없는 GPU의 경우 메모리 기반 추정
        if available_memory_mb >= 16000:
            return 12  # 16GB+
        elif available_memory_mb >= 10000:
            return 8   # 10-16GB
        elif available_memory_mb >= 6000:
            return 4   # 6-10GB
        elif available_memory_mb >= 4000:
            return 2   # 4-6GB
        else:
            return 1   # 4GB 미만

    def get_optimal_config(self, custom_batch_size: Optional[int] = None) -> Dict[str, Any]:
        """최적 설정 반환"""
        gpu_info = self.detect_gpu()
        gpu_name_lower = gpu_info.name.lower()

        # GPU별 설정 찾기
        config = self._gpu_configs.get("default").copy()
        for gpu_key, gpu_config in self._gpu_configs.items():
            if gpu_key in gpu_name_lower and gpu_key != "default":
                config.update(gpu_config)
                break

        # 사용자 지정 배치 크기 적용
        if custom_batch_size is not None:
            config["batch_size"] = custom_batch_size

        # GPU 정보 추가
        config.update({
            "gpu_name": gpu_info.name,
            "gpu_memory_total": gpu_info.memory_total,
            "gpu_memory_available": gpu_info.memory_available,
            "auto_detected": True
        })

        return config

    def estimate_memory_usage(self, batch_size: int, input_size: tuple = (896, 896)) -> Dict[str, int]:
        """예상 메모리 사용량 계산 (MB)"""
        # YOLOv12 모델 크기 추정
        model_memory = 200  # 모델 자체: ~200MB

        # 입력 텐서 크기 (RGB, float32)
        h, w = input_size
        input_tensor_size = batch_size * 3 * h * w * 4 / (1024 * 1024)  # MB

        # 출력 텐서 크기 (추정)
        output_tensor_size = batch_size * 84 * 8400 * 4 / (1024 * 1024)  # MB

        # 중간 레이어 메모리 (추정)
        intermediate_memory = input_tensor_size * 3

        total_memory = model_memory + input_tensor_size + output_tensor_size + intermediate_memory

        return {
            "model_memory": int(model_memory),
            "input_tensor": int(input_tensor_size),
            "output_tensor": int(output_tensor_size),
            "intermediate": int(intermediate_memory),
            "total_estimated": int(total_memory)
        }

    def validate_batch_size(self, batch_size: int, input_size: tuple = (896, 896)) -> bool:
        """배치 크기가 현재 GPU에서 실행 가능한지 확인"""
        gpu_info = self.detect_gpu()
        memory_usage = self.estimate_memory_usage(batch_size, input_size)

        available_memory = gpu_info.memory_available
        estimated_usage = memory_usage["total_estimated"]

        # 안전 여유분 20% 고려
        return estimated_usage * 1.2 < available_memory