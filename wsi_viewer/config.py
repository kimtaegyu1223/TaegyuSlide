from __future__ import annotations
from dataclasses import dataclass

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

CONFIG = ViewerConfig()