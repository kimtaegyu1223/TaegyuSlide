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
        self.setZValue(9999)  # 매우 높은 Z값으로 설정해서 확실히 맨 앞에 표시

        # 펜/폰트 설정 - 가시성 강화
        self.pen = pen or QPen(Qt.red, 3, Qt.SolidLine)
        self.mitosis_pen = QPen(QColor(255, 0, 0), 8, Qt.SolidLine)  # 빨간색 매우 굵은 선
        self.font = QFont("Arial", 16, QFont.Bold)

    def add_mitosis_detections(self, detections: List[MitosisDetection]):
        """Mitosis 감지 결과 추가"""
        if not detections:
            return
        # 중복 방지 (bbox + confidence 기준)
        existing = {(d.bbox, round(d.confidence, 3)) for d in self.mitosis_detections}
        for d in detections:
            if (d.bbox, round(d.confidence, 3)) not in existing:
                self.mitosis_detections.append(d)

        # 안전하게 업데이트 호출
        try:
            if hasattr(self, 'scene') and self.scene():
                self.update()  # 화면 리프레시 트리거
        except RuntimeError:
            # Qt 객체가 이미 삭제된 경우 무시
            pass

    def clear_mitosis_detections(self):
        """Mitosis 감지 결과 제거"""
        if self.mitosis_detections:
            self.mitosis_detections.clear()
            # 안전하게 업데이트 호출
            try:
                if hasattr(self, 'scene') and self.scene():
                    self.update()
            except RuntimeError:
                # Qt 객체가 이미 삭제된 경우 무시
                pass

    def boundingRect(self) -> QRectF:
        # 아주 큰 rect 반환해서 항상 paint 호출되게 함
        return QRectF(-1e6, -1e6, 2e6, 2e6)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        from PySide6.QtCore import QRectF

        if not self.get_scale:
            return
        s = float(self.get_scale())

        print(f"=== PAINT 호출 ===")
        print(f"스케일: {s}")
        print(f"Mitosis 감지 결과 개수: {len(self.mitosis_detections)}")

        # === 강제 테스트 박스 (화면 중앙에 고정) ===
        try:
            scene_rect = self.scene().sceneRect() if self.scene() else QRectF(0, 0, 10000, 10000)
            center_x = scene_rect.center().x()
            center_y = scene_rect.center().y()

            # 화면 중앙에 큰 테스트 박스
            painter.setPen(QPen(QColor(0, 255, 0), 10, Qt.SolidLine))  # 녹색 매우 굵은 선
            test_rect = QRectF(center_x - 200, center_y - 200, 400, 400)
            painter.drawRect(test_rect)
            painter.fillRect(test_rect, QColor(0, 255, 0, 50))  # 반투명 녹색 채우기

            painter.setPen(QPen(Qt.black, 2))
            painter.drawText(test_rect, Qt.AlignCenter, "TEST BOX\nVISIBLE?")
            print(f"강제 테스트 박스 그리기: {test_rect}")
        except Exception as e:
            print(f"테스트 박스 실패: {e}")

        # --- 기존 오버레이 ---
        painter.setPen(self.pen)
        for (x0, y0) in self.level0_points:
            painter.drawEllipse(QPointF(x0 * s, y0 * s), 6, 6)
        for (x0, y0, w0, h0) in self.level0_boxes:
            painter.drawRect(x0 * s, y0 * s, w0 * s, h0 * s)

        # --- Mitosis Detection ---
        if not self.mitosis_detections:
            print("감지 결과 없음 - paint 종료")
            return

        painter.setPen(self.mitosis_pen)
        painter.setFont(self.font)

        for i, detection in enumerate(self.mitosis_detections):
            x1, y1, x2, y2 = detection.bbox

            # numpy 타입을 Python float로 변환
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            # level0_coords=True인 경우 스케일 적용, False인 경우 현재 레벨 좌표로 간주
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

            # Qt에서 사용할 수 있도록 float로 변환
            rect_x, rect_y, rect_w, rect_h = float(rect_x), float(rect_y), float(rect_w), float(rect_h)

            print(f"감지박스 {i}: 원본=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            print(f"감지박스 {i}: level0_coords={detection.level0_coords}, 스케일={s:.4f}")
            print(f"감지박스 {i}: 렌더링=({rect_x:.1f}, {rect_y:.1f}, {rect_w:.1f}, {rect_h:.1f})")

            # 박스 그리기 - 가시성 강화
            rect = QRectF(rect_x, rect_y, rect_w, rect_h)

            # 반투명 채우기로 더 눈에 띄게
            painter.fillRect(rect, QColor(255, 0, 0, 80))  # 반투명 빨간색 채우기
            painter.drawRect(rect)  # 테두리

            # X 표시로 중심점 강조
            painter.drawLine(rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)
            painter.drawLine(rect_x + rect_w, rect_y, rect_x, rect_y + rect_h)

            # 점수 표시 - 더 큰 텍스트
            confidence_text = f"MITOSIS: {float(detection.confidence):.3f}"
            text_rect = QRectF(rect_x, rect_y - 40, max(rect_w, 150), 30)
            painter.fillRect(text_rect, QColor(255, 255, 0, 200))  # 노란색 배경
            painter.setPen(QPen(Qt.black, 2))
            painter.drawText(text_rect, Qt.AlignCenter, confidence_text)
            painter.setPen(self.mitosis_pen)

        print(f"=== PAINT 완료 ===")
        print(f"{len(self.mitosis_detections)}개 박스 그리기 완료")
