from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
                             QHBoxLayout, QFileDialog, QCheckBox, QMessageBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QDialog, QLineEdit)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from processors.data_processor import DataProcessor
from analyzers.distribution_analyzer import DistributionAnalyzer
from utils.stats import calculate_statistics

class DataAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аналіз даних")
        self.setGeometry(100, 100, 1000, 700)
        self.processor = DataProcessor()
        self.analyzer = DistributionAnalyzer()
        self.selected_column = None
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        self.label = QLabel("Файл не завантажено")
        self.btn_load = QPushButton("Завантажити")
        self.btn_load.clicked.connect(self.load_file)
        left_panel.addWidget(self.label)
        left_panel.addWidget(self.btn_load)

        self.column_box = QComboBox()
        self.column_box.currentIndexChanged.connect(self.update_column)
        left_panel.addWidget(self.column_box)

        self.chk_log = QCheckBox("Логарифм")
        self.chk_std = QCheckBox("Стандартизація")
        self.shift_input = QLineEdit()
        self.shift_input.setPlaceholderText("Зсув")
        self.shift_input.setValidator(QDoubleValidator())

        left_panel.addWidget(self.chk_log)
        left_panel.addWidget(self.chk_std)
        left_panel.addWidget(self.shift_input)

        self.btn_hist = QPushButton("Гістограма")
        self.btn_hist.clicked.connect(self.plot_histogram)
        self.btn_ecdf = QPushButton("ECDF")
        self.btn_ecdf.clicked.connect(self.plot_ecdf)
        self.btn_exp = QPushButton("Експ. розподіл")
        self.btn_exp.clicked.connect(self.plot_exponential)

        left_panel.addWidget(self.btn_hist)
        left_panel.addWidget(self.btn_ecdf)
        left_panel.addWidget(self.btn_exp)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Характеристика", "Значення"])
        left_panel.addWidget(self.stats_table)

        layout.addLayout(left_panel)

        right_panel = QVBoxLayout()
        self.canvas = FigureCanvas(Figure(figsize=(5, 4)))
        right_panel.addWidget(self.canvas)
        layout.addLayout(right_panel)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Відкрити файл")
        if file_name:
            try:
                self.processor.load_data(file_name)
                self.label.setText(f"Завантажено: {file_name}")
                cols = self.processor.get_numeric_columns()
                self.column_box.clear()
                self.column_box.addItems(cols)
                if cols:
                    self.selected_column = cols[0]
            except Exception as e:
                QMessageBox.critical(self, "Помилка", str(e))

    def update_column(self):
        self.selected_column = self.column_box.currentText()

    def prepare_data(self):
        data = self.processor.data[self.selected_column].dropna().values
        shift_val = float(self.shift_input.text()) if self.shift_input.text() else 0.0
        try:
            return self.analyzer.transform_data(data, log=self.chk_log.isChecked(),
                                                standardize=self.chk_std.isChecked(),
                                                shift=shift_val)
        except Exception as e:
            QMessageBox.warning(self, "Помилка", str(e))
            return None

    def plot_histogram(self):
        data = self.prepare_data()
        if data is not None:
            self.analyzer.histogram(self.canvas, data, bins=20)
            self.show_stats(data)

    def plot_ecdf(self):
        data = self.prepare_data()
        if data is not None:
            self.analyzer.ecdf(self.canvas, data)
            self.show_stats(data)

    def plot_exponential(self):
        data = self.prepare_data()
        if data is not None:
            self.analyzer.theoretical_exponential(self.canvas, data)
            self.show_stats(data)

    def show_stats(self, data):
        stats = calculate_statistics(data)
        self.stats_table.setRowCount(len(stats))
        for i, (k, v) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(k))
            if isinstance(v, tuple):
                value = f"[{v[0]:.4f}, {v[1]:.4f}]"
            else:
                value = f"{v:.4f}" if isinstance(v, float) else str(v)
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))
        self.stats_table.resizeColumnsToContents()