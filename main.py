import sys
import logging
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QDockWidget, QLabel, QMainWindow, QFileDialog, QWidget,
                               QVBoxLayout, QPushButton, QMessageBox, QSplitter, QProgressBar,
                               QTextEdit, QTabWidget, QHBoxLayout, QSpinBox, QCheckBox, QComboBox)
from wsi_viewer.viewer import SlideViewer
from wsi_viewer.ai import MitosisDetectionWorker, GPUManager
from wsi_viewer.ai.enhanced_detection_worker import EnhancedMitosisDetectionWorker
from wsi_viewer.ai.real_time_detection_worker import RealTimeDetectionWorker

# ---- put this at the VERY TOP of main.py ----
import os, sys
# TensorRT/ CUDA bin 경로를 ASCII 경로로 옮겼다고 가정
TRT_BIN = r"C:\Users\keyce\OneDrive\TensorRT-10.13.3.9\bin"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(TRT_BIN)
    os.add_dll_directory(CUDA_BIN)
# ------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WSI Viewer with AI Mitosis Detection")

        # 메인 레이아웃: 스플리터로 뷰어와 대시보드 분할
        self.splitter = QSplitter(Qt.Horizontal)
        self.viewer = SlideViewer()
        self.splitter.addWidget(self.viewer)

        # 대시보드 (초기에는 숨김)
        self.dashboard = None
        self.dashboard_visible = True

        self.setCentralWidget(self.splitter)
        self._build_menu()
        self._build_dashboard()

        # GPU 관리자
        self.gpu_manager = GPUManager()

        # AI 감지 워커
        self.detection_worker = None
        self.analysis_result = None

    def _build_menu(self):
        # File 메뉴
        act_open = QAction("Open...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_file)
        menu_file = self.menuBar().addMenu("File")
        menu_file.addAction(act_open)

        # AI 메뉴
        act_detect_mitosis = QAction("Detect Mitosis (Full Slide)", self)
        act_detect_mitosis.setShortcut("Ctrl+M")
        act_detect_mitosis.triggered.connect(self.detect_mitosis_full_slide)
        menu_ai = self.menuBar().addMenu("AI")
        menu_ai.addAction(act_detect_mitosis)

        # View 메뉴
        act_toggle_dashboard = QAction("Toggle Dashboard", self)
        act_toggle_dashboard.setShortcut("F9")
        act_toggle_dashboard.triggered.connect(self.toggle_dashboard)
        menu_view = self.menuBar().addMenu("View")
        menu_view.addAction(act_toggle_dashboard)

    def _build_dashboard(self):
        """대시보드 패널 구성"""
        self.dashboard = QWidget()
        self.dashboard.setMinimumWidth(350)
        self.dashboard.setMaximumWidth(500)

        # 탭 위젯
        self.tab_widget = QTabWidget()

        # 1. 슬라이드 정보 탭
        self._build_slide_info_tab()

        # 2. AI 설정 탭
        self._build_ai_settings_tab()

        # 3. 결과 탭
        self._build_results_tab()

        # 대시보드 메인 레이아웃
        main_layout = QVBoxLayout(self.dashboard)
        main_layout.addWidget(self.tab_widget)

        # 스플리터에 추가
        self.splitter.addWidget(self.dashboard)
        self.splitter.setStretchFactor(0, 3)  # 뷰어가 3/4
        self.splitter.setStretchFactor(1, 1)  # 대시보드가 1/4

    def _build_slide_info_tab(self):
        """슬라이드 정보 탭"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 슬라이드 정보
        self.info = QLabel("No slide loaded")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        # 빠른 감지 버튼 (현재 뷰포트)
        self.btn_detect_viewport = QPushButton("🔍 Detect (Current View)")
        self.btn_detect_viewport.setEnabled(False)
        self.btn_detect_viewport.clicked.connect(self.detect_mitosis_viewport)
        layout.addWidget(self.btn_detect_viewport)

        # 전체 슬라이드 감지 버튼
        self.btn_detect_full = QPushButton("🔬 Detect (Full Slide)")
        self.btn_detect_full.setEnabled(False)
        self.btn_detect_full.clicked.connect(self.detect_mitosis_full_slide)
        layout.addWidget(self.btn_detect_full)

        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 상태 정보
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Slide Info")

    def _build_ai_settings_tab(self):
        """AI 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # GPU 정보
        layout.addWidget(QLabel("GPU Settings:"))
        self.gpu_info_label = QLabel("Detecting GPU...")
        self.gpu_info_label.setWordWrap(True)
        layout.addWidget(self.gpu_info_label)

        # 패치 크기 설정
        patch_layout = QHBoxLayout()
        patch_layout.addWidget(QLabel("Patch Size:"))
        self.patch_size_combo = QComboBox()
        self.patch_size_combo.addItems(["512", "768", "896"])
        self.patch_size_combo.setCurrentText("896")  # 기본값
        patch_layout.addWidget(self.patch_size_combo)
        layout.addLayout(patch_layout)

        # 배치 크기 설정
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        # 배율 설정
        mag_layout = QHBoxLayout()
        mag_layout.addWidget(QLabel("Magnification:"))
        self.magnification_combo = QComboBox()
        self.magnification_combo.addItems(["40x", "20x", "10x", "5x"])
        mag_layout.addWidget(self.magnification_combo)
        layout.addLayout(mag_layout)

        # 티슈 감지 활성화
        self.tissue_detection_check = QCheckBox("Enable tissue detection")
        self.tissue_detection_check.setChecked(True)
        layout.addWidget(self.tissue_detection_check)

        # 실시간 결과 표시 활성화
        self.real_time_check = QCheckBox("Enable real-time results (multiprocessing)")
        self.real_time_check.setChecked(True)
        layout.addWidget(self.real_time_check)

        # GPU 정보 새로고침 버튼
        self.btn_refresh_gpu = QPushButton("Refresh GPU Info")
        self.btn_refresh_gpu.clicked.connect(self.refresh_gpu_info)
        layout.addWidget(self.btn_refresh_gpu)

        layout.addStretch()
        self.tab_widget.addTab(tab, "AI Settings")

        # 초기 GPU 정보 로드
        self.refresh_gpu_info()

    def _build_results_tab(self):
        """결과 탭"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 결과 통계
        self.results_stats = QLabel("No results yet")
        self.results_stats.setWordWrap(True)
        layout.addWidget(self.results_stats)

        # 결과 로그
        layout.addWidget(QLabel("Processing Log:"))
        self.results_log = QTextEdit()
        self.results_log.setMaximumHeight(200)
        layout.addWidget(self.results_log)

        # 결과 제거 버튼
        self.btn_clear_results = QPushButton("Clear Results")
        self.btn_clear_results.clicked.connect(self.clear_results)
        layout.addWidget(self.btn_clear_results)

        # 감지 결과에 맞춰 화면 조정 버튼
        self.btn_fit_detections = QPushButton("Fit Detections to View")
        self.btn_fit_detections.clicked.connect(self.fit_detections_to_view)
        layout.addWidget(self.btn_fit_detections)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Results")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open slide", "", "Slides (*.svs *.ndpi *.scn *.mrxs *.tiff *.tif)")
        if not path:
            return

        try:
            self.viewer.load_slide(path)
            b = self.viewer.backend
            if b:
                txt = f"Levels: {b.levels}\\nDims: {b.dimensions}\\nMPP: {b.mpp_x:.3f} x {b.mpp_y:.3f}\\nObjective: {b.objective_power}"
                self.info.setText(txt)
                # 슬라이드가 로드되면 감지 버튼 활성화
                self.btn_detect_viewport.setEnabled(True)
                self.btn_detect_full.setEnabled(True)
                print("슬라이드 로딩 완료, 버튼 활성화됨")
        except Exception as e:
            print(f"슬라이드 로딩 오류: {e}")
            QMessageBox.critical(self, "오류", f"슬라이드 로딩 실패:\\n{e}")

    def toggle_dashboard(self):
        """대시보드 표시/숨김 토글"""
        self.dashboard_visible = not self.dashboard_visible
        self.dashboard.setVisible(self.dashboard_visible)

    def refresh_gpu_info(self):
        """GPU 정보 새로고침"""
        try:
            gpu_info = self.gpu_manager.detect_gpu()
            gpu_text = f"GPU: {gpu_info.name}\\nMemory: {gpu_info.memory_available}MB available\\nRecommended Batch: {gpu_info.recommended_batch_size}"
            self.gpu_info_label.setText(gpu_text)
            self.batch_size_spin.setValue(gpu_info.recommended_batch_size)
        except Exception as e:
            self.gpu_info_label.setText(f"GPU detection failed: {e}")

    def detect_mitosis_viewport(self):
        """현재 뷰포트에서 빠른 감지"""
        if not self.viewer.backend:
            self.status_label.setText("No slide loaded")
            return

        # 현재 뷰포트 이미지 추출
        image = self.viewer.get_viewport_image(target_level=0)
        if image is None:
            QMessageBox.warning(self, "Warning", "Failed to extract image from current viewport")
            return

        # 기존 결과 제거
        self.viewer.clear_mitosis_detections()

        # 버튼 비활성화
        self.btn_detect_viewport.setEnabled(False)
        self.status_label.setText("Processing viewport...")

        # 간단한 워커 시작
        self.detection_worker = MitosisDetectionWorker(image)
        self.detection_worker.progress_updated.connect(self.on_viewport_progress)
        self.detection_worker.detection_completed.connect(self.on_viewport_completed)
        self.detection_worker.detection_failed.connect(self.on_viewport_failed)
        self.detection_worker.start()

    def detect_mitosis_full_slide(self):
        """전체 슬라이드 감지"""
        if not self.viewer.backend:
            self.status_label.setText("No slide loaded")
            return

        # 기존 결과 제거
        self.viewer.clear_mitosis_detections()

        # 버튼 비활성화
        self.btn_detect_full.setEnabled(False)
        self.btn_detect_viewport.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 설정 가져오기
        batch_size = self.batch_size_spin.value()
        magnification = self.magnification_combo.currentText()
        patch_size = int(self.patch_size_combo.currentText())
        use_real_time = self.real_time_check.isChecked()

        # 워커 선택 및 시작
        if use_real_time:
            self.detection_worker = RealTimeDetectionWorker(
                self.viewer.backend,
                target_magnification=magnification,
                patch_size=patch_size
            )
            # 실시간 시그널 추가 연결
            self.detection_worker.patch_result_ready.connect(self.on_patch_result_ready)
            self.detection_worker.detection_batch_ready.connect(self.on_batch_result_ready)
        else:
            self.detection_worker = EnhancedMitosisDetectionWorker(
                self.viewer.backend,
                target_magnification=magnification,
                patch_size=patch_size,
                custom_batch_size=batch_size
            )

        # 시그널 연결
        self.detection_worker.analysis_completed.connect(self.on_analysis_completed)
        self.detection_worker.progress_updated.connect(self.on_full_progress)
        self.detection_worker.stats_updated.connect(self.on_stats_updated)
        self.detection_worker.detection_completed.connect(self.on_full_completed)
        self.detection_worker.detection_failed.connect(self.on_full_failed)

        self.detection_worker.start()

    def clear_results(self):
        """결과 제거"""
        self.viewer.clear_mitosis_detections()
        self.results_stats.setText("No results")
        self.results_log.clear()

    def fit_detections_to_view(self):
        """모든 감지 결과가 보이도록 화면 조정"""
        self.viewer.fit_detections_to_view()
        self.status_label.setText("View fitted to all detections")

    # 뷰포트 감지 이벤트 핸들러
    def on_viewport_progress(self, message: str):
        self.status_label.setText(message)

    def on_viewport_completed(self, results):
        self.btn_detect_viewport.setEnabled(True)
        if results:
            self.viewer.add_mitosis_detections(results)
            self.status_label.setText(f"Found {len(results)} mitosis in viewport")
            self.results_stats.setText(f"Viewport: {len(results)} detections")
        else:
            self.status_label.setText("No mitosis detected in viewport")
        self.detection_worker = None

    def on_viewport_failed(self, error_message: str):
        self.btn_detect_viewport.setEnabled(True)
        self.status_label.setText("Viewport detection failed")
        QMessageBox.critical(self, "Detection Error", error_message)
        self.detection_worker = None

    # 전체 슬라이드 감지 이벤트 핸들러
    def on_analysis_completed(self, analysis):
        """슬라이드 분석 완료"""
        self.analysis_result = analysis
        efficiency = (analysis.tissue_patches / analysis.total_patches * 100) if analysis.total_patches > 0 else 0
        log_msg = f"Analysis: {analysis.total_patches} total patches, {analysis.tissue_patches} tissue patches ({efficiency:.1f}% efficiency)\\n"
        log_msg += f"Tissue regions: {len(analysis.tissue_regions)}, Coverage: {analysis.tissue_coverage*100:.1f}%\\n"
        log_msg += f"Optimal level: {analysis.optimal_level} ({analysis.recommended_magnification})\\n"
        self.results_log.append(log_msg)

    def on_full_progress(self, message: str, progress: float):
        """전체 감지 진행률"""
        self.status_label.setText(message)
        self.progress_bar.setValue(int(progress))

    def on_stats_updated(self, stats):
        """통계 업데이트"""
        remaining_str = f"{int(stats.estimated_remaining//60)}:{int(stats.estimated_remaining%60):02d}" if stats.estimated_remaining > 0 else "00:00"
        stats_text = f"Processed: {stats.processed_patches}/{stats.total_patches}\\n"
        stats_text += f"Detected: {stats.detected_mitosis} mitosis\\n"
        stats_text += f"Speed: {stats.processing_speed:.1f} patches/sec\\n"
        stats_text += f"Remaining: {remaining_str}"
        self.results_stats.setText(stats_text)

    def on_full_completed(self, results):
        """전체 감지 완료"""
        self.btn_detect_full.setEnabled(True)
        self.btn_detect_viewport.setEnabled(True)
        self.progress_bar.setVisible(False)

        if results:
            self.viewer.add_mitosis_detections(results)
            self.status_label.setText(f"Full slide analysis complete: {len(results)} mitosis detected")

            # 결과 탭으로 전환
            self.tab_widget.setCurrentIndex(2)

            # 로그 업데이트
            self.results_log.append(f"\\n=== DETECTION COMPLETED ===\\n")
            self.results_log.append(f"Total detections: {len(results)}\\n")
            if self.analysis_result:
                density = len(results) / (self.analysis_result.tissue_coverage * 100) if self.analysis_result.tissue_coverage > 0 else 0
                self.results_log.append(f"Density: {density:.2f} mitosis per % tissue area\\n")

        else:
            self.status_label.setText("Full slide analysis complete: No mitosis detected")

        self.detection_worker = None

    def on_full_failed(self, error_message: str):
        """전체 감지 실패"""
        self.btn_detect_full.setEnabled(True)
        self.btn_detect_viewport.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Full slide detection failed")

        QMessageBox.critical(self, "Detection Error", f"Full slide detection failed:\\n{error_message}")
        self.results_log.append(f"ERROR: {error_message}\\n")
        self.detection_worker = None

    def on_patch_result_ready(self, result):
        """개별 패치 결과가 준비됨 (실시간 표시)"""
        try:
            if result.success and result.detections:
                # 즉시 화면에 표시
                self.viewer.add_mitosis_detections(result.detections)

                # 로그에 실시간 업데이트
                self.results_log.append(f"Patch {result.patch_info.patch_id}: {len(result.detections)} detections")
        except Exception as e:
            # UI 업데이트 실패 시 로그만 기록하고 계속 진행
            print(f"실시간 결과 표시 오류: {e}")

    def on_batch_result_ready(self, batch_detections):
        """배치 결과가 준비됨 (실시간 표시)"""
        try:
            if batch_detections:
                # 배치 단위로 화면에 표시
                self.viewer.add_mitosis_detections(batch_detections)
        except Exception as e:
            # UI 업데이트 실패 시 로그만 기록하고 계속 진행
            print(f"배치 결과 표시 오류: {e}")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(1800, 1100); win.show()
    sys.exit(app.exec())
