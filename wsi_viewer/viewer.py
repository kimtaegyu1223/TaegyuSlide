from __future__ import annotations
import math
from typing import Dict, Set
from cachetools import LRUCache
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QPainter, QPixmap, QTransform, QBrush
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView
import logging
from .config import CONFIG
from .backend import OpenSlideBackend
from .tiling import TileRequest, TileScheduler, TileKey
from .overlay import OverlayItem

class SlideViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        # OpenGL 뷰포트 여부
        if CONFIG.viewer.use_opengl_viewport:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
            self.setViewport(QOpenGLWidget())

        # 기본 렌더링 설정
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Scene 초기화
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(Qt.white))
        self.setScene(self.scene)

        # Backend & 캐시
        self.backend: OpenSlideBackend | None = None
        self.cur_level: int = 0
        self.cache: LRUCache[TileKey, QPixmap] = LRUCache(maxsize=CONFIG.viewer.cache_max_tiles)
        self.tile_items: Dict[TileKey, QGraphicsPixmapItem] = {}
        self.scheduler = TileScheduler()

        # Overlay
        self.overlay: OverlayItem | None = None
        self._create_overlay()

        # Cleanup timer
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.setSingleShot(True)
        self.cleanup_timer.timeout.connect(self.cleanup_old_tiles)

        # 패딩 값
        self._padding = CONFIG.viewer.padding

    def _view_scale_x(self) -> float:
        return float(self.transform().m11())

    def level0_to_view_scale(self) -> float:
        if not self.backend:
            return 1.0
        ds = self.backend.level_downsamples[self.cur_level]
        return self._view_scale_x() / ds

    def load_slide(self, path: str):
        for item in self.tile_items.values():
            self.scene.removeItem(item)
        self.tile_items.clear()
        self.cache.clear()
        self.scene.clear()

        if self.backend:
            self.backend.close()

        self.backend = OpenSlideBackend(path)
        self.cur_level = self.backend.levels - 1
        w, h = self.backend.dimensions[self.cur_level]

        self.scene.setSceneRect(
            -self._padding, -self._padding,
            w + 2*self._padding, h + 2*self._padding
        )
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.update_visible_tiles()

    def current_scale_from_level0(self) -> float:
        if not self.backend:
            return 1.0
        ds = self.backend.level_downsamples[self.cur_level]
        return 1.0 / ds

    def wheelEvent(self, event):
        if not self.backend:
            return

        anchor_scene_before = self.mapToScene(event.position().toPoint())
        zf = CONFIG.viewer.zoom_factor_step if event.angleDelta().y() > 0 else 1.0 / CONFIG.viewer.zoom_factor_step

        cur_l0_scale = self.level0_to_view_scale()
        target_l0_scale = cur_l0_scale * zf

        if not (CONFIG.viewer.min_scale_l0 <= target_l0_scale <= CONFIG.viewer.max_scale_l0):
            return

        new_level = self.backend.get_best_level_for_downsample(1.0 / target_l0_scale)
        level_changed = (new_level != self.cur_level)

        if not level_changed:
            self.scale(zf, zf)
        else:
            self.cleanup_timer.stop()
            for item in self.tile_items.values():
                item.setZValue(-1)

            old_ds = self.backend.level_downsamples[self.cur_level]
            self.cur_level = new_level
            w, h = self.backend.dimensions[self.cur_level]
            self.scene.setSceneRect(
                -self._padding, -self._padding,
                w + 2*self._padding, h + 2*self._padding
            )
            new_ds = self.backend.level_downsamples[self.cur_level]
            new_view_scale = target_l0_scale * new_ds
            self.setTransform(QTransform().scale(new_view_scale, new_view_scale))

            level0_pos = anchor_scene_before * old_ds
            anchor_scene_after = level0_pos / new_ds
            self.centerOn(anchor_scene_after)

        self.update_visible_tiles()
        if level_changed:
            self.cleanup_timer.start(CONFIG.viewer.cleanup_delay_ms)

    def cleanup_old_tiles(self):
        if not self.backend:
            return
        keys_to_remove = [k for k in self.tile_items if k[0] != self.cur_level]
        for key in keys_to_remove:
            item = self.tile_items.pop(key)
            self.scene.removeItem(item)

    def update_visible_tiles(self):
        if not self.backend:
            return
        vr = self.mapToScene(self.viewport().rect()).boundingRect()
        w, h = self.backend.dimensions[self.cur_level]
        vr = vr.intersected(QRectF(0, 0, w, h))
        if vr.isEmpty():
            return

        ts = CONFIG.viewer.tile_size
        col0 = max(int(math.floor(vr.left()/ts)), 0)
        row0 = max(int(math.floor(vr.top()/ts)), 0)
        col1 = min(int(math.ceil(vr.right()/ts)), (w-1)//ts)
        row1 = min(int(math.ceil(vr.bottom()/ts)), (h-1)//ts)

        visible: Set[TileKey] = set()
        for r in range(row0, row1+1):
            for c in range(col0, col1+1):
                visible.add((self.cur_level, c, r))

        for key in set(self.tile_items.keys()) - visible:
            item = self.tile_items.pop(key)
            self.scene.removeItem(item)

        for r in range(row0, row1+1):
            for c in range(col0, col1+1):
                key = (self.cur_level, c, r)
                if key in self.tile_items:
                    continue
                if key in self.cache:
                    pm = self.cache[key]
                    item = QGraphicsPixmapItem(pm)
                    item.setPos(c*ts, r*ts)
                    item.setZValue(0)
                    self.scene.addItem(item)
                    self.tile_items[key] = item
                else:
                    req = TileRequest(self.backend, self.cur_level, c, r, ts)
                    self.scheduler.request(req, self._on_tile_done, self._on_tile_error)

    def _on_tile_done(self, level, col, row, qimg):
        key = (level, col, row)
        if key in self.cache:
            return
        pm = QPixmap.fromImage(qimg)
        self.cache[key] = pm
        if not self.backend or level != self.cur_level or key in self.tile_items:
            return
        ts = CONFIG.viewer.tile_size
        tile_rect = QRectF(col*ts, row*ts, ts, ts)
        viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if tile_rect.intersects(viewport_rect):
            item = QGraphicsPixmapItem(pm)
            item.setPos(col*ts, row*ts)
            item.setZValue(0)
            self.scene.addItem(item)
            self.tile_items[key] = item

    def _on_tile_error(self, level, col, row, msg: str):
        pass

    def get_viewport_image(self, target_level: int = None) -> "Image.Image" | None:
        """현재 뷰포트 영역의 이미지를 추출"""
        if not self.backend:
            return None

        from PIL import Image

        if target_level is None:
            target_level = 0  # 최고 해상도

        viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        current_ds = self.backend.level_downsamples[self.cur_level]
        target_ds = self.backend.level_downsamples[target_level]
        scale_factor = current_ds / target_ds

        x0 = int(viewport_rect.left() * scale_factor)
        y0 = int(viewport_rect.top() * scale_factor)
        w0 = int(viewport_rect.width() * scale_factor)
        h0 = int(viewport_rect.height() * scale_factor)

        level_w, level_h = self.backend.dimensions[target_level]
        target_scale = self.backend.level_downsamples[target_level]

        x0 = max(0, min(x0, int(level_w * target_scale)))
        y0 = max(0, min(y0, int(level_h * target_scale)))
        w0 = min(w0, int(level_w * target_scale) - x0)
        h0 = min(h0, int(level_h * target_scale) - y0)

        if w0 <= 0 or h0 <= 0:
            return None

        try:
            image = self.backend.slide.read_region(
                (x0, y0), target_level,
                (int(w0 / target_scale), int(h0 / target_scale))
            ).convert('RGB')

            logging.debug(f"Extracted image: {image.size} at level {target_level}")
            return image

        except Exception as e:
            logging.error(f"Failed to extract viewport image: {e}")
            return None

    def _create_overlay(self):
        if self.overlay:
            self.scene.removeItem(self.overlay)

        self.overlay = OverlayItem(get_scale_func=self.level0_to_view_scale)
        self.scene.addItem(self.overlay)

    def add_mitosis_detections(self, detections: List["MitosisDetection"]):
        if not self.overlay:
            self._create_overlay()

        from .overlay import MitosisDetection

        overlay_detections = []
        for detection in detections:
            overlay_detection = MitosisDetection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                level0_coords=True
            )
            overlay_detections.append(overlay_detection)

        self.overlay.add_mitosis_detections(overlay_detections)

    def clear_mitosis_detections(self):
        import shiboken6
        if self.overlay and shiboken6.isValid(self.overlay):
            self.overlay.clear_mitosis_detections()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update_visible_tiles()
