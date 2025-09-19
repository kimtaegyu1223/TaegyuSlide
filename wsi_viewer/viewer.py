# filename: slide_viewer_refactored.py
from __future__ import annotations
import math
import logging
from typing import Dict, Set, Optional
from cachetools import LRUCache
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QPainter, QPixmap, QTransform, QBrush, QImage
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

# 외부 모듈 (원본 구조 유지)
from .config import CONFIG as _DEFAULT_CONFIG
from .backend import OpenSlideBackend
from .tiling import TileRequest, TileScheduler, TileKey
from .overlay import OverlayItem

log = logging.getLogger(__name__)

class SlideViewer(QGraphicsView):
    """WSI 타일 뷰어 (리팩토링 버전)

    - scene 인스턴스 속성명을 _scene 으로 변경해 QGraphicsView.scene()과 충돌 방지
    - get_viewport_image(): OpenSlide Level-0 좌표 기준으로 재구현
    - load_slide() 정리 루틴 단순화
    - Overlay zValue 고정(최상단)
    - print → logging
    - 테스트 주입성: scheduler/backend/config 인자 허용
    """

    def __init__(self,
                 scheduler: Optional[TileScheduler] = None,
                 backend: Optional[OpenSlideBackend] = None,
                 config: Optional[object] = None):
        super().__init__()

        # --- Config 주입 (테스트 편의) ---
        self.CFG = config or _DEFAULT_CONFIG

        # --- GPU 가속 뷰포트 (옵션) ---
        if getattr(self.CFG.viewer, 'use_opengl_viewport', False):
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
            self.setViewport(QOpenGLWidget())

        # --- 렌더링 설정 ---
        self.setRenderHints(QPainter.SmoothPixmapTransform)  # AA는 옵션화 가능
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # --- Scene 초기화 (이름 충돌 방지) ---
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(Qt.white))
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(Qt.white))

        # --- Backend & 상태 ---
        self.backend: OpenSlideBackend | None = backend
        self.cur_level: int = 0

        # --- 캐시/타일 ---
        self.cache: LRUCache[TileKey, QPixmap] = LRUCache(maxsize=self.CFG.viewer.cache_max_tiles)
        self.tile_items: Dict[TileKey, QGraphicsPixmapItem] = {}
        self.scheduler: TileScheduler = scheduler or TileScheduler()

        # --- Overlay ---
        self.overlay: OverlayItem | None = None
        self._create_overlay()

        # --- Timers ---
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.setSingleShot(True)
        self.cleanup_timer.timeout.connect(self.cleanup_old_tiles)

        self.scroll_update_timer = QTimer(self)
        self.scroll_update_timer.setSingleShot(True)
        self.scroll_update_timer.timeout.connect(self.update_visible_tiles)

        # --- 패딩 ---
        self._padding = self.CFG.viewer.padding

        # 초기 backend가 주입된 경우 sceneRect 설정
        if self.backend:
            self.cur_level = self.backend.levels - 1
            w, h = self.backend.dimensions[self.cur_level]
            self._scene.setSceneRect(-self._padding, -self._padding, w + 2*self._padding, h + 2*self._padding)
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # -------------------- 유틸 --------------------
    def _view_scale_x(self) -> float:
        return float(self.transform().m11())

    def level0_to_view_scale(self) -> float:
        if not self.backend:
            return 1.0
        ds = self.backend.level_downsamples[self.cur_level]
        return self._view_scale_x() / ds

    def current_scale_from_level0(self) -> float:
        if not self.backend:
            return 1.0
        ds = self.backend.level_downsamples[self.cur_level]
        return 1.0 / ds

    # -------------------- 로딩 --------------------
    def load_slide(self, path: str):
        # 기존 오버레이 데이터 백업 (데이터만)
        mitosis_backup = []
        if self.overlay and hasattr(self.overlay, 'mitosis_detections'):
            mitosis_backup = list(self.overlay.mitosis_detections)
            log.debug("감지 결과 백업: %d", len(mitosis_backup))

        # 전체 정리 (중복 제거 루틴 간소화)
        self._scene.clear()
        self.tile_items.clear()
        self.cache.clear()

        if self.backend:
            try:
                self.backend.close()
            except Exception:  # 방어적
                pass

        # 백엔드 로드 및 레벨 초기화
        self.backend = OpenSlideBackend(path)
        self.cur_level = self.backend.levels - 1

        # 오버레이 재생성
        self._create_overlay()

        # 백업 감지 결과 복구
        if mitosis_backup:
            self.overlay.mitosis_detections = mitosis_backup
            log.debug("감지 결과 복원: %d", len(mitosis_backup))

        # SceneRect & 화면 맞춤
        w, h = self.backend.dimensions[self.cur_level]
        self._scene.setSceneRect(-self._padding, -self._padding, w + 2*self._padding, h + 2*self._padding)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.update_visible_tiles()

    # -------------------- 입력(줌/스크롤) --------------------
    def wheelEvent(self, event):
        if not self.backend:
            return

        anchor_scene_before = self.mapToScene(event.position().toPoint())
        zf = self.CFG.viewer.zoom_factor_step if event.angleDelta().y() > 0 else 1.0 / self.CFG.viewer.zoom_factor_step

        cur_l0_scale = self.level0_to_view_scale()
        target_l0_scale = cur_l0_scale * zf
        if not (self.CFG.viewer.min_scale_l0 <= target_l0_scale <= self.CFG.viewer.max_scale_l0):
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
            self._scene.setSceneRect(-self._padding, -self._padding, w + 2*self._padding, h + 2*self._padding)

            new_ds = self.backend.level_downsamples[self.cur_level]
            new_view_scale = target_l0_scale * new_ds
            self.setTransform(QTransform().scale(new_view_scale, new_view_scale))

            level0_pos = anchor_scene_before * old_ds
            anchor_scene_after = level0_pos / new_ds
            self.centerOn(anchor_scene_after)

        self.update_visible_tiles()
        if level_changed:
            self.cleanup_timer.start(self.CFG.viewer.cleanup_delay_ms)

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.scroll_update_timer.start(50)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update_visible_tiles()

    # -------------------- 타일 관리 --------------------
    def cleanup_old_tiles(self):
        if not self.backend:
            return
        keys_to_remove = [k for k in self.tile_items if k[0] != self.cur_level]
        for key in keys_to_remove:
            item = self.tile_items.pop(key)
            self._scene.removeItem(item)

    def update_visible_tiles(self):
        if not self.backend:
            return

        vr = self.mapToScene(self.viewport().rect()).boundingRect()
        w, h = self.backend.dimensions[self.cur_level]
        vr = vr.intersected(QRectF(0, 0, w, h))
        if vr.isEmpty():
            return

        ts = self.CFG.viewer.tile_size
        margin_tiles = getattr(self.CFG.viewer, 'prefetch_margin', 0)
        if margin_tiles:
            expand = QRectF(vr)
            expand.adjust(-margin_tiles*ts, -margin_tiles*ts, margin_tiles*ts, margin_tiles*ts)
            expand = expand.intersected(QRectF(0, 0, w, h))
        else:
            expand = vr

        col0 = max(int(math.floor(expand.left()/ts)), 0)
        row0 = max(int(math.floor(expand.top()/ts)), 0)
        col1 = min(int(math.ceil(expand.right()/ts)), (w-1)//ts)
        row1 = min(int(math.ceil(expand.bottom()/ts)), (h-1)//ts)

        visible: Set[TileKey] = set()
        for r in range(row0, row1+1):
            for c in range(col0, col1+1):
                visible.add((self.cur_level, c, r))

        # 제거 (가시집합 밖)
        for key in set(self.tile_items.keys()) - visible:
            item = self.tile_items.pop(key)
            self._scene.removeItem(item)

        # 추가/요청
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
                    self._scene.addItem(item)
                    self.tile_items[key] = item
                else:
                    req = TileRequest(self.backend, self.cur_level, c, r, ts)
                    self.scheduler.request(req, self._on_tile_done, self._on_tile_error)

    def _on_tile_done(self, level, col, row, qimg: QImage):
        key = (level, col, row)
        if key in self.cache:
            return
        pm = QPixmap.fromImage(qimg)
        self.cache[key] = pm
        if not self.backend or level != self.cur_level or key in self.tile_items:
            return
        ts = self.CFG.viewer.tile_size
        tile_rect = QRectF(col*ts, row*ts, ts, ts)
        viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if tile_rect.intersects(viewport_rect):
            item = QGraphicsPixmapItem(pm)
            item.setPos(col*ts, row*ts)
            item.setZValue(0)
            self._scene.addItem(item)
            self.tile_items[key] = item

    def _on_tile_error(self, level, col, row, msg: str):
        log.warning("Tile load error (L%d, c%d, r%d): %s", level, col, row, msg)


    # -------------------- 오버레이 --------------------
    def _create_overlay(self):
        # 안전 제거
        try:
            import shiboken6
            if getattr(self, 'overlay', None) is not None and shiboken6.isValid(self.overlay):
                self._scene.removeItem(self.overlay)
        except Exception:
            pass

        self.overlay = OverlayItem(get_scale_func=self.level0_to_view_scale)
        self.overlay.setZValue(1000)
        self._scene.addItem(self.overlay)
        log.debug("오버레이 생성 완료, Z=%s", self.overlay.zValue())

    def add_mitosis_detections(self, detections):
        if not self.overlay:
            self._create_overlay()

        try:
            from .overlay import MitosisDetection
        except Exception:
            log.exception("overlay.MitosisDetection import 실패")
            return

        overlay_detections = []
        current_view = self.mapToScene(self.viewport().rect()).boundingRect()
        current_scale = self.level0_to_view_scale()
        view_scale = self._view_scale_x()

        log.debug("뷰포트=%s, L0스케일=%.6f, 뷰스케일=%.6f, cur_level=%d", current_view, current_scale, view_scale, self.cur_level)

        for det in detections:
            if hasattr(det, 'bbox') and hasattr(det, 'confidence'):
                overlay_detections.append(MitosisDetection(bbox=det.bbox, confidence=det.confidence, level0_coords=True))

        if overlay_detections:
            self.overlay.add_mitosis_detections(overlay_detections)
            self.viewport().update()
            log.debug("오버레이에 %d개 결과 추가", len(overlay_detections))
        else:
            log.debug("추가할 감지 결과 없음")

    def clear_mitosis_detections(self):
        try:
            import shiboken6
            if self.overlay and shiboken6.isValid(self.overlay):
                self.overlay.clear_mitosis_detections()
        except Exception:
            pass