from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class ViewerConfig:
    tile_size: int = 1024
    padding: int = 1000
    cache_max_tiles: int = 4096
    zoom_factor_step: float = 1.35
    min_scale_l0: float = 0.001
    max_scale_l0: float = 50.0
    cleanup_delay_ms: int = 10
    use_opengl_viewport: bool = True

@dataclass(frozen=True)
class AIConfig:
    # 패치 설정
    patch_sizes: List[int] = (512, 768, 896)
    default_patch_size: int = 512
    overlap_ratio: float = 0.05

    # 모델 입력 크기 (패치 크기와 다를 수 있음)
    model_input_size: int = 896  # 모델이 기대하는 입력 크기 (일반적으로 640x640)

    # 모델 설정
    model_base_path: str = "models"
    mitosis_model_filename: str = "mitosis_yolov12_896.onnx"
    model_config_filename: str = "model_config.json"

    # 처리 설정
    default_batch_size: int = 4
    max_batch_size: int = 32
    default_worker_count: int = 4
    max_worker_count: int = 6

    # 성능 설정
    enable_multiprocessing: bool = True
    enable_real_time_display: bool = True
    patch_queue_size: int = 100
    result_queue_size: int = 1000

    # 성능 최적화 설정
    optimize_for_speed: bool = True
    prefetch_patches: bool = True
    use_shared_memory: bool = False  # 실험적 기능
    memory_limit_mb: int = 8000  # 최대 메모리 사용량

    # 배율 설정
    target_magnifications: List[str] = ("5x", "10x", "20x", "40x")
    default_magnification: str = "40x"

    @property
    def model_path(self) -> Path:
        """모델 파일의 전체 경로 반환"""
        return Path(self.model_base_path) / self.mitosis_model_filename

    @property
    def config_path(self) -> Path:
        """설정 파일의 전체 경로 반환"""
        return Path(self.model_base_path) / self.model_config_filename

@dataclass(frozen=True)
class AppConfig:
    viewer: ViewerConfig = ViewerConfig()
    ai: AIConfig = AIConfig()

CONFIG = AppConfig()