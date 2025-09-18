from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import logging
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage
from PIL import Image
from .backend import SlideReadError

TileKey = Tuple[int, int, int]  # (level, col, row)

class TileTaskSignals(QObject):
    done = Signal(int, int, int, QImage)
    error = Signal(int, int, int, str)

@dataclass
class TileRequest:
    backend: "OpenSlideBackend"
    level: int
    col: int
    row: int
    tile_size: int

class TileTask(QRunnable):
    def __init__(self, req: TileRequest):
        super().__init__()
        self.req = req
        self.signals = TileTaskSignals()

    def run(self) -> None:
        """타일 로딩 작업 실행"""
        req = self.req
        tile_id = f"레벨{req.level}_타일({req.col},{req.row})"

        try:
            logging.debug(f"타일 로딩 시작: {tile_id}")

            # 타일 좌표 계산
            x = req.col * req.tile_size
            y = req.row * req.tile_size

            # 백엔드에서 이미지 읽기
            pil_image: Image.Image = req.backend.read_region(
                req.level, x, y, req.tile_size, req.tile_size
            )

            # PIL → QImage 변환
            if pil_image.size[0] == 0 or pil_image.size[1] == 0:
                raise ValueError(f"빈 이미지가 반환됨: {pil_image.size}")

            # RGB 포맷으로 변환
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # QImage 생성
            bytes_data = pil_image.tobytes()
            w, h = pil_image.size
            qimg = QImage(bytes_data, w, h, w * 3, QImage.Format_RGB888).copy()

            if qimg.isNull():
                raise ValueError("QImage 변환 실패")

            logging.debug(f"타일 로딩 완료: {tile_id} ({w}x{h})")
            self.signals.done.emit(req.level, req.col, req.row, qimg)

        except SlideReadError as e:
            error_msg = f"슬라이드 읽기 오류 ({tile_id}): {e}"
            logging.error(error_msg)
            self.signals.error.emit(req.level, req.col, req.row, error_msg)

        except ValueError as e:
            error_msg = f"이미지 변환 오류 ({tile_id}): {e}"
            logging.error(error_msg)
            self.signals.error.emit(req.level, req.col, req.row, error_msg)

        except MemoryError as e:
            error_msg = f"메모리 부족 ({tile_id}): 타일 크기를 줄여주세요"
            logging.error(error_msg)
            self.signals.error.emit(req.level, req.col, req.row, error_msg)

        except Exception as e:
            error_msg = f"예상치 못한 오류 ({tile_id}): {e}"
            logging.error(error_msg)
            self.signals.error.emit(req.level, req.col, req.row, error_msg)

class TileScheduler:
    def __init__(self, pool: QThreadPool | None = None):
        self.pool = pool or QThreadPool.globalInstance()

    def request(self, req: TileRequest, on_done, on_error) -> None:
        task = TileTask(req)
        task.signals.done.connect(on_done)
        task.signals.error.connect(on_error)
        self.pool.start(task)
