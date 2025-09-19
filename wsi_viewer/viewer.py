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

        # 스크롤 시 타일 업데이트용 타이머 (성능 최적화)
        self.scroll_update_timer = QTimer(self)
        self.scroll_update_timer.setSingleShot(True)
        self.scroll_update_timer.timeout.connect(self.update_visible_tiles)

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
        # 기존 오버레이 데이터 백업 (객체가 아닌 데이터만)
        mitosis_backup = []
        if hasattr(self, 'overlay') and self.overlay and hasattr(self.overlay, 'mitosis_detections'):
            mitosis_backup = self.overlay.mitosis_detections.copy()
            print(f"감지 결과 백업: {len(mitosis_backup)}개")

        for item in self.tile_items.values():
            self.scene.removeItem(item)
        self.tile_items.clear()
        self.cache.clear()
        self.scene.clear()  # 이때 오버레이도 함께 삭제됨

        if self.backend:
            self.backend.close()

        self.backend = OpenSlideBackend(path)
        self.cur_level = self.backend.levels - 1

        # 오버레이 재생성
        self._create_overlay()

        # 백업된 감지 결과 복원
        if mitosis_backup:
            print(f"감지 결과 복원: {len(mitosis_backup)}개")
            self.overlay.mitosis_detections = mitosis_backup
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
        # 안전하게 기존 오버레이 제거
        if hasattr(self, 'overlay') and self.overlay:
            try:
                import shiboken6
                if shiboken6.isValid(self.overlay):
                    self.scene.removeItem(self.overlay)
            except (RuntimeError, AttributeError):
                pass  # 이미 삭제된 경우 무시

        self.overlay = OverlayItem(get_scale_func=self.level0_to_view_scale)
        self.scene.addItem(self.overlay)
        print(f"오버레이 생성 완료, Z값: {self.overlay.zValue()}")

    def add_mitosis_detections(self, detections):
        print(f"add_mitosis_detections 호출됨: {len(detections)}개 감지 결과")

        if not self.overlay:
            self._create_overlay()

        from .overlay import MitosisDetection

        overlay_detections = []

        # 현재 뷰포트 및 스케일 정보 확인
        current_view = self.mapToScene(self.viewport().rect()).boundingRect()
        current_scale = self.level0_to_view_scale()
        view_scale = self._view_scale_x()

        print(f"=== 스케일링 디버깅 ===")
        print(f"현재 뷰포트: {current_view}")
        print(f"Level0 스케일: {current_scale}")
        print(f"뷰 스케일: {view_scale}")
        print(f"현재 레벨: {self.cur_level}")
        if self.backend:
            print(f"다운샘플: {self.backend.level_downsamples[self.cur_level]}")
            print(f"Level 0 크기: {self.backend.dimensions[0]}")
            print(f"현재 레벨 크기: {self.backend.dimensions[self.cur_level]}")
        print(f"패딩: {self._padding}")

        # Scene 정보도 출력
        if hasattr(self, 'scene') and self.scene():
            scene_rect = self.scene().sceneRect()
            print(f"Scene 크기: {scene_rect}")

        # Transform 정보
        transform = self.transform()
        print(f"Transform: m11={transform.m11():.6f}, m22={transform.m22():.6f}")
        print(f"Transform: dx={transform.dx():.1f}, dy={transform.dy():.1f}")

        for detection in detections:
            # DetectionResult 또는 MitosisDetection 객체 모두 처리
            if hasattr(detection, 'bbox') and hasattr(detection, 'confidence'):
                x1, y1, x2, y2 = detection.bbox
                print(f"\n감지 결과 #{len(overlay_detections)+1}:")
                print(f"  원본 bbox: ({x1}, {y1}, {x2}, {y2})")
                print(f"  박스 크기: {x2-x1:.1f} x {y2-y1:.1f}")
                print(f"  confidence: {detection.confidence}")

                # 다양한 스케일 변환 시도
                print(f"  스케일 변환 시도:")
                print(f"    current_scale 적용: ({x1*current_scale:.1f}, {y1*current_scale:.1f}, {x2*current_scale:.1f}, {y2*current_scale:.1f})")
                print(f"    view_scale 적용: ({x1*view_scale:.1f}, {y1*view_scale:.1f}, {x2*view_scale:.1f}, {y2*view_scale:.1f})")
                print(f"    스케일 없음: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

                # Level 0 대비 크기 비교
                if self.backend:
                    level0_w, level0_h = self.backend.dimensions[0]
                    print(f"  Level 0 대비 위치: x={x1/level0_w*100:.2f}%, y={y1/level0_h*100:.2f}%")

                # 현재 뷰포트와 비교
                print(f"  뷰포트 vs 박스:")
                print(f"    뷰포트: left={current_view.left():.1f}, top={current_view.top():.1f}")
                print(f"    뷰포트: right={current_view.right():.1f}, bottom={current_view.bottom():.1f}")
                print(f"    박스(원본): left={x1:.1f}, top={y1:.1f}, right={x2:.1f}, bottom={y2:.1f}")

                # 겹침 확인
                overlap_x = not (x2 < current_view.left() or x1 > current_view.right())
                overlap_y = not (y2 < current_view.top() or y1 > current_view.bottom())
                print(f"    X축 겹침: {overlap_x}, Y축 겹침: {overlap_y}")

                # Level 0 좌표계 사용 (enhanced_detection_worker에서 변환된 절대 좌표)
                overlay_detection = MitosisDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    level0_coords=True  # Level 0 좌표계 사용
                )
                overlay_detections.append(overlay_detection)

        if overlay_detections:
            print(f"오버레이에 {len(overlay_detections)}개 결과 추가")
            self.overlay.add_mitosis_detections(overlay_detections)
            # 강제로 뷰 업데이트
            self.viewport().update()
        else:
            print("변환된 오버레이 감지 결과가 없음")

    def clear_mitosis_detections(self):
        import shiboken6
        if self.overlay and shiboken6.isValid(self.overlay):
            self.overlay.clear_mitosis_detections()

    def fit_detections_to_view(self):
        """모든 감지 결과가 보이도록 화면 조정"""
        if not self.overlay or not self.overlay.mitosis_detections:
            return

        from PySide6.QtCore import QRectF

        # 모든 감지 결과의 바운딩 박스 계산
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for detection in self.overlay.mitosis_detections:
            x1, y1, x2, y2 = detection.bbox
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

        if min_x != float('inf'):
            # 여백 추가 (20%)
            margin_x = (max_x - min_x) * 0.2
            margin_y = (max_y - min_y) * 0.2

            fit_rect = QRectF(
                min_x - margin_x,
                min_y - margin_y,
                (max_x - min_x) + 2 * margin_x,
                (max_y - min_y) + 2 * margin_y
            )

            self.fitInView(fit_rect, Qt.KeepAspectRatio)
            print(f"모든 감지 결과에 맞춰 화면 조정: {fit_rect}")

    def scrollContentsBy(self, dx, dy):
        """뷰포트가 스크롤(드래그)될 때 호출됨"""
        super().scrollContentsBy(dx, dy)

        # 성능 최적화: 드래그 중에는 타이머로 지연시켜서 업데이트
        # 드래그가 끝나고 50ms 후에 타일 업데이트 수행
        self.scroll_update_timer.start(50)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update_visible_tiles()
