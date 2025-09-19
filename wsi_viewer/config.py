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
    overlap_ratio: float = 0.00

    # 서버 API 설정
    server_base_url: str = "http://localhost:8000"
    detection_endpoint: str = "/api/v1/detect"
    health_endpoint: str = "/health"
    info_endpoint: str = "/api/v1/info"
    api_timeout: int = 30
    max_retries: int = 3

    # 처리 설정
    default_batch_size: int = 8
    max_batch_size: int = 32

    # 성능 설정
    enable_batch_processing: bool = True
    enable_real_time_display: bool = True

    # 배율 설정
    target_magnifications: List[str] = ("5x", "10x", "20x", "40x")
    default_magnification: str = "20x"

    # 감지 설정
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4

@dataclass(frozen=True)
class AppConfig:
    viewer: ViewerConfig = ViewerConfig()
    ai: AIConfig = AIConfig()

CONFIG = AppConfig()