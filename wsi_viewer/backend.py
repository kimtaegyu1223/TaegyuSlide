from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
from PIL import Image
import logging

try:
    import openslide  # type: ignore
except ImportError:
    openslide = None

Size = Tuple[int, int]

class SlideLoadError(Exception):
    """슬라이드 로딩 관련 에러"""
    pass

class SlideReadError(Exception):
    """슬라이드 읽기 관련 에러"""
    pass

class OpenSlideBackend:
    def __init__(self, slide_path: str):
        if openslide is None:
            raise SlideLoadError(
                "OpenSlide 라이브러리를 찾을 수 없습니다. "
                "pip install openslide-python을 실행하여 설치해주세요."
            )

        slide_file = Path(slide_path)
        if not slide_file.exists():
            raise SlideLoadError(f"슬라이드 파일을 찾을 수 없습니다: {slide_path}")

        if not slide_file.is_file():
            raise SlideLoadError(f"올바른 파일이 아닙니다: {slide_path}")

        try:
            self.slide = openslide.OpenSlide(slide_path)
            if self.slide.level_count == 0:
                raise SlideLoadError(f"유효하지 않은 슬라이드 파일입니다: {slide_path}")

        except openslide.OpenSlideError as e:
            raise SlideLoadError(f"슬라이드 열기 실패: {e}")
        except Exception as e:
            raise SlideLoadError(f"예상치 못한 오류가 발생했습니다: {e}")

        try:
            self.levels = self.slide.level_count
            self.level_downsamples: List[float] = [
                float(self.slide.level_downsamples[i]) for i in range(self.levels)
            ]
            self.dimensions: List[Size] = [
                self.slide.level_dimensions[i] for i in range(self.levels)
            ]

            prop = self.slide.properties
            self.mpp_x = float(prop.get("openslide.mpp-x", 0) or 0)
            self.mpp_y = float(prop.get("openslide.mpp-y", 0) or 0)
            self.objective_power = prop.get("openslide.objective-power", "")

            logging.info(f"슬라이드 로딩 완료: {slide_path} ({self.levels}레벨)")

        except Exception as e:
            self.slide.close()
            raise SlideLoadError(f"슬라이드 메타데이터 읽기 실패: {e}")

    def close(self) -> None:
        """슬라이드 리소스 정리"""
        try:
            if hasattr(self, 'slide') and self.slide:
                self.slide.close()
                logging.debug("슬라이드 리소스 정리 완료")
        except Exception as e:
            logging.warning(f"슬라이드 리소스 정리 중 오류: {e}")

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """주어진 다운샘플 비율에 최적화된 레벨 반환"""
        try:
            return self.slide.get_best_level_for_downsample(downsample)
        except Exception as e:
            logging.warning(f"최적 레벨 계산 실패, 기본값 사용: {e}")
            return min(max(0, int(downsample)), self.levels - 1)

    def read_region(self, level: int, x: int, y: int, w: int, h: int) -> Image.Image:
        """지정된 레벨에서 이미지 영역 읽기"""
        if not (0 <= level < self.levels):
            raise SlideReadError(f"유효하지 않은 레벨: {level} (가능 범위: 0-{self.levels-1})")

        if w <= 0 or h <= 0:
            raise SlideReadError(f"유효하지 않은 크기: {w}x{h}")

        level_w, level_h = self.dimensions[level]
        if x < 0 or y < 0 or x >= level_w or y >= level_h:
            raise SlideReadError(f"레벨 {level}에서 유효하지 않은 좌표: ({x}, {y}), 최대: ({level_w}, {level_h})")

        try:
            ds = self.level_downsamples[level]
            x0, y0 = int(x * ds), int(y * ds)
            img = self.slide.read_region((x0, y0), level, (w, h)).convert("RGB")

            if img.size != (w, h):
                logging.warning(f"요청 크기와 실제 크기 불일치: 요청={w}x{h}, 실제={img.size}")

            return img

        except openslide.OpenSlideError as e:
            raise SlideReadError(f"이미지 읽기 실패 (레벨={level}, 좌표=({x},{y}), 크기={w}x{h}): {e}")
        except Exception as e:
            raise SlideReadError(f"예상치 못한 읽기 오류: {e}")
