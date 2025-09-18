from __future__ import annotations
from typing import Callable, Iterable, Tuple, List
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QPen, QFont, QColor
from PySide6.QtWidgets import QGraphicsItem

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

class MitosisDetection:
    """Mitosis 감지 결과를 저장하는 클래스"""
    def __init__(self, bbox: Box, confidence: float, level0_coords: bool = True):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.level0_coords = level0_coords  # Level 0 좌표계 여부

    def __repr__(self):
        return f"MitosisDetection(bbox={self.bbox}, conf={self.confidence:.3f})"
class OverlayItem(QGraphicsItem):
    def __init__(self, level0_points: Iterable[Point] = None, level0_boxes: Iterable[Box] = None,
                 get_scale_func: Callable[[], float] | None = None, pen: QPen | None = None):
        super().__init__()
        self.level0_points = list(level0_points or [])
        self.level0_boxes = list(level0_boxes or [])
        self.mitosis_detections: List[MitosisDetection] = []
        self.get_scale = get_scale_func
        self.setZValue(10)

        # 펜/폰트 설정
        self.pen = pen or QPen(Qt.red, 2, Qt.SolidLine)
        self.mitosis_pen = QPen(QColor(255, 0, 0), 3, Qt.SolidLine)  # 빨간색 굵은 선
        self.font = QFont("Arial", 12, QFont.Bold)

    def add_mitosis_detections(self, detections: List[MitosisDetection]):
        """Mitosis 감지 결과 추가"""
        if not detections:
            return
        # 중복 방지 (bbox + confidence 기준)
        existing = {(d.bbox, round(d.confidence, 3)) for d in self.mitosis_detections}
        for d in detections:
            if (d.bbox, round(d.confidence, 3)) not in existing:
                self.mitosis_detections.append(d)
        self.update()  # 화면 리프레시 트리거

    def clear_mitosis_detections(self):
        """Mitosis 감지 결과 제거"""
        if self.mitosis_detections:
            self.mitosis_detections.clear()
            self.update()

    def boundingRect(self) -> QRectF:
        # 아주 큰 rect 반환해서 항상 paint 호출되게 함
        return QRectF(-1e6, -1e6, 2e6, 2e6)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self.get_scale:
            return
        s = float(self.get_scale())

        # --- 기존 오버레이 ---
        painter.setPen(self.pen)
        for (x0, y0) in self.level0_points:
            painter.drawEllipse(QPointF(x0 * s, y0 * s), 6, 6)
        for (x0, y0, w0, h0) in self.level0_boxes:
            painter.drawRect(x0 * s, y0 * s, w0 * s, h0 * s)

        # --- Mitosis Detection ---
        if not self.mitosis_detections:
            return

        painter.setPen(self.mitosis_pen)
        painter.setFont(self.font)

        for detection in self.mitosis_detections:
            x1, y1, x2, y2 = detection.bbox
            if detection.level0_coords:
                rect_x = x1 * s
                rect_y = y1 * s
                rect_w = (x2 - x1) * s
                rect_h = (y2 - y1) * s
            else:
                rect_x = x1
                rect_y = y1
                rect_w = x2 - x1
                rect_h = y2 - y1

            # 박스 그리기
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)

            # 점수 표시 (반투명 흰 배경 + 텍스트)
            confidence_text = f"M: {detection.confidence:.2f}"
            text_rect = QRectF(rect_x, rect_y - 25, rect_w, 20)
            painter.fillRect(text_rect, QColor(255, 255, 255, 180))
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(text_rect, Qt.AlignCenter, confidence_text)

            # 다시 빨간색 펜으로 복원
            painter.setPen(self.mitosis_pen)
