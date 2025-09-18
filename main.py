import sys
import logging
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QPushButton, QMessageBox
from wsi_viewer.viewer import SlideViewer
from wsi_viewer.ai import MitosisDetectionWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WSI Viewer with AI Mitosis Detection")
        self.viewer = SlideViewer()
        self.setCentralWidget(self.viewer)
        self._build_menu()
        self._build_dock()

        # AI 감지 워커
        self.detection_worker = None

    def _build_menu(self):
        # File 메뉴
        act_open = QAction("Open...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_file)
        menu_file = self.menuBar().addMenu("File")
        menu_file.addAction(act_open)

        # AI 메뉴
        act_detect_mitosis = QAction("Detect Mitosis", self)
        act_detect_mitosis.setShortcut("Ctrl+M")
        act_detect_mitosis.triggered.connect(self.detect_mitosis)
        menu_ai = self.menuBar().addMenu("AI")
        menu_ai.addAction(act_detect_mitosis)

    def _build_dock(self):
        dock = QDockWidget("Controls", self)
        w = QWidget(); lay = QVBoxLayout(w)

        # 슬라이드 정보
        self.info = QLabel("No slide")
        lay.addWidget(self.info)

        # Mitosis 감지 버튼
        self.btn_detect = QPushButton("🔍 Detect Mitosis")
        self.btn_detect.setEnabled(False)  # 슬라이드가 로드되기 전까지 비활성화
        self.btn_detect.clicked.connect(self.detect_mitosis)
        lay.addWidget(self.btn_detect)

        # 결과 정보
        self.result_info = QLabel("Ready")
        lay.addWidget(self.result_info)

        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open slide", "", "Slides (*.svs *.ndpi *.scn *.mrxs *.tiff *.tif)")
        if not path:
            return
        self.viewer.load_slide(path)
        b = self.viewer.backend
        if b:
            txt = f"Levels: {b.levels}\\nDims: {b.dimensions}\\nMPP: {b.mpp_x} x {b.mpp_y}\\nObjective: {b.objective_power}"
            self.info.setText(txt)
            # 슬라이드가 로드되면 감지 버튼 활성화
            self.btn_detect.setEnabled(True)

    def detect_mitosis(self):
        """Mitosis 감지 실행"""
        if not self.viewer.backend:
            self.result_info.setText("No slide loaded")
            return

        # 현재 뷰포트 이미지 추출
        image = self.viewer.get_viewport_image(target_level=0)  # 최고 해상도
        if image is None:
            QMessageBox.warning(self, "Warning", "Failed to extract image from current viewport")
            return

        # 기존 감지 결과 제거
        self.viewer.clear_mitosis_detections()

        # 버튼 비활성화
        self.btn_detect.setEnabled(False)
        self.result_info.setText("Initializing detector...")

        # 백그라운드 워커 시작
        self.detection_worker = MitosisDetectionWorker(image)
        self.detection_worker.progress_updated.connect(self.on_detection_progress)
        self.detection_worker.detection_completed.connect(self.on_detection_completed)
        self.detection_worker.detection_failed.connect(self.on_detection_failed)
        self.detection_worker.start()

    def on_detection_progress(self, message: str):
        """감지 진행 상황 업데이트"""
        self.result_info.setText(message)

    def on_detection_completed(self, results):
        """감지 완료 처리"""
        self.btn_detect.setEnabled(True)

        if len(results) > 0:
            # 결과를 뷰어에 표시
            self.viewer.add_mitosis_detections(results)
            self.result_info.setText(f"Found {len(results)} mitosis candidates")
        else:
            self.result_info.setText("No mitosis detected")

        # 워커 정리
        self.detection_worker = None

    def on_detection_failed(self, error_message: str):
        """감지 실패 처리"""
        self.btn_detect.setEnabled(True)
        self.result_info.setText("Detection failed")

        QMessageBox.critical(self, "Detection Error", f"Mitosis detection failed:\n{error_message}")

        # 워커 정리
        self.detection_worker = None

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(1800, 1100); win.show()
    sys.exit(app.exec())
