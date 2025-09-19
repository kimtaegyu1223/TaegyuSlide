from __future__ import annotations
from typing import Callable, Iterable, Tuple, List

from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QPen, QFont, QColor
from PySide6.QtWidgets import QGraphicsItem

# -------------------------------
# Types
# -------------------------------
Point = Tuple[float, float]
Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) in level0 coords by default


class ObjectDetection:
    """Container for a single object detection."""

    def __init__(self, bbox: Box, confidence: float, level0_coords: bool = True):
        self.bbox = bbox
        self.confidence = float(confidence)
        self.level0_coords = bool(level0_coords)

    def __repr__(self) -> str:  # debug-friendly
        return f"ObjectDetection(bbox={self.bbox}, conf={self.confidence:.3f}, level0={self.level0_coords})"


class OverlayItem(QGraphicsItem):
    """
    QGraphicsItem overlay for object detection results.

    ✅ Key features
    - Draws detection boxes anchored on scene positions, but with **fixed pixel size** on screen
      (zooming the view won't change the on-screen box size).
    - Uses **cosmetic pens** so line thickness stays constant.
    - Safe guards against double-add and GC issues by encouraging caller to keep a strong ref.

    Parameters
    ----------
    level0_points : Iterable[Point]
        Optional debug points in level0 coordinates.
    level0_boxes : Iterable[Box]
        Optional debug boxes in level0 coordinates (x1,y1,x2,y2).
    get_scale_func : Callable[[], float]
        Function returning the scale from **level0 coords -> current scene coords** (e.g., downsample factor inverse).
        If your scene already uses level0 coordinates directly, return 1.0.
    box_pixel_size : int
        On-screen pixel size of each drawn detection rectangle.
    pen : QPen
        Optional base pen.
    """

    def __init__(
        self,
        level0_points: Iterable[Point] | None = None,
        level0_boxes: Iterable[Box] | None = None,
        get_scale_func: Callable[[], float] | None = None,
        box_pixel_size: int = 26,
        pen: QPen | None = None,
    ) -> None:
        super().__init__()

        self.level0_points = list(level0_points or [])
        self.level0_boxes = list(level0_boxes or [])
        self.object_detections: List[ObjectDetection] = []
        self.get_scale = get_scale_func or (lambda: 1.0)
        self.box_px = int(box_pixel_size)

        # Make sure we're on top
        self.setZValue(9999)

        # Pens/fonts — cosmetic=True keeps width in **device pixels** regardless of view transform
        self.pen = (pen or QPen(Qt.red, 2, Qt.SolidLine))
        self.pen.setCosmetic(True)

        self.object_pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
        self.object_pen.setCosmetic(True)

        self.text_bg = QColor(255, 255, 0, 200)
        self.text_pen = QPen(Qt.black, 1)
        self.text_pen.setCosmetic(True)
        self.font = QFont("Arial", 11, QFont.Bold)

    # -------------------------------
    # Public API
    # -------------------------------
    def add_object_detections(self, detections: List[ObjectDetection]) -> None:
        if not detections:
            return
        existing = {(d.bbox, round(d.confidence, 3)) for d in self.object_detections}
        appended = False
        for d in detections:
            key = (d.bbox, round(d.confidence, 3))
            if key not in existing:
                self.object_detections.append(d)
                appended = True
        if appended:
            try:
                self.update()
            except RuntimeError:
                pass  # Item may already be deleted

    def set_object_detections(self, detections: List[ObjectDetection]) -> None:
        """Replace all detections at once."""
        self.object_detections = list(detections or [])
        try:
            self.update()
        except RuntimeError:
            pass

    def clear_object_detections(self) -> None:
        if self.object_detections:
            self.object_detections.clear()
            try:
                self.update()
            except RuntimeError:
                pass

    # -------------------------------
    # QGraphicsItem overrides
    # -------------------------------
    def boundingRect(self) -> QRectF:
        # Very large rect so we always get paint() calls without recomputing geometry
        return QRectF(-1e6, -1e6, 2e6, 2e6)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        # --- Debug: if paint is called, we can draw a viewport-wide cross in device space ---
        # (Uncomment if you need to sanity-check the rendering path.)
        # self._debug_device_cross(painter)

        s_level = float(self.get_scale())  # level0 -> scene

        # Draw optional debug primitives in SCENE space (they will scale with zoom)
        painter.setPen(self.pen)
        for (x0, y0) in self.level0_points:
            painter.drawEllipse(QPointF(x0 * s_level, y0 * s_level), 4, 4)
        for (x1, y1, x2, y2) in self.level0_boxes:
            painter.drawRect(x1 * s_level, y1 * s_level, (x2 - x1) * s_level, (y2 - y1) * s_level)

        if not self.object_detections:
            return

        # Object boxes: draw with **fixed pixel size** using device-space painting.
        for det in self.object_detections:
            x1, y1, x2, y2 = map(float, det.bbox)

            # Anchor at bbox center in SCENE coordinates
            if det.level0_coords:
                cx = (x1 + x2) * 0.5 * s_level
                cy = (y1 + y2) * 0.5 * s_level
            else:  # already scene coords
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

            scene_pt = QPointF(cx, cy)
            self._draw_fixed_pixel_box(painter, scene_pt, det)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _draw_fixed_pixel_box(self, painter: QPainter, scene_pt: QPointF, det: ObjectDetection) -> None:
        """
        Draw a box centered at `scene_pt` whose **on-screen size is constant** in pixels.
        Implementation: map scene->device, reset transform, draw in device pixels, restore.
        """
        # Map anchor point to device coords under current world transform
        device_pt = painter.worldTransform().map(scene_pt)

        half = self.box_px * 0.5
        rect_px = QRectF(device_pt.x() - half, device_pt.y() - half, self.box_px, self.box_px)

        painter.save()
        painter.resetTransform()  # from here, we paint in device pixels

        # Filled rectangle + border (cosmetic pens)
        painter.fillRect(rect_px, QColor(255, 0, 0, 80))
        painter.setPen(self.object_pen)
        painter.drawRect(rect_px)

        # Cross mark
        painter.drawLine(rect_px.topLeft(), rect_px.bottomRight())
        painter.drawLine(rect_px.topRight(), rect_px.bottomLeft())

        # Confidence label background
        label_h = 18
        text_rect = QRectF(rect_px.x(), rect_px.y() - (label_h + 4), max(rect_px.width(), 120.0), label_h)
        painter.fillRect(text_rect, self.text_bg)
        painter.setPen(self.text_pen)
        painter.setFont(self.font)
        painter.drawText(text_rect, Qt.AlignCenter, f"OBJECT: {det.confidence:.3f}")

        painter.restore()

    def _debug_device_cross(self, painter: QPainter) -> None:
        """Draws a large cross in device space to confirm that paint() is being invoked and visible."""
        painter.save()
        painter.resetTransform()
        painter.setPen(QPen(Qt.green, 1))
        painter.drawLine(0, 0, 2000, 2000)
        painter.drawLine(2000, 0, 0, 2000)
        painter.restore()


# -------------------------------
# Usage notes
# -------------------------------
# 1) Keep a strong reference on the overlay (e.g., self.overlay = OverlayItem(...))
#    and add it to the scene **once**: scene.addItem(self.overlay)
#    If you lose the Python reference, the C++ item may be GC'ed, leading to
#    "Internal C++ object already deleted" on later calls.
#
# 2) Provide a proper scale function:
#    - If your scene coordinates are already level0, use: get_scale_func=lambda: 1.0
#    - If you render a downsampled level L where level_downsample[L] = dL, and your scene
#      uses level L coordinates, then level0->scene scale is s = 1/dL.
#
# 3) To update from worker threads, emit a Qt signal carrying detections and connect it to
#    OverlayItem.set_object_detections on the GUI thread. QGraphicsItem is **not** thread-safe.
