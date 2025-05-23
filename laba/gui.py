from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
                             QDoubleSpinBox, QSpinBox, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Статистичний аналіз")
        self.init_ui()

    def init_ui(self):
        # Основний віджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Лівий блок (керування)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, 1)

        # Вкладки
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        # Вкладка 1: Основний аналіз
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.tabs.addTab(tab1, "Основний аналіз")

        # Кнопка завантаження даних
        self.load_button = QPushButton("Завантажити дані")
        tab1_layout.addWidget(self.load_button)

        # Кількість класів для гістограми
        bin_layout = QHBoxLayout()
        bin_label = QLabel("Кількість класів для гістограми:")
        self.bin_entry = QSpinBox()
        self.bin_entry.setMinimum(0)
        self.bin_entry.setValue(0)
        bin_layout.addWidget(bin_label)
        bin_layout.addWidget(self.bin_entry)
        tab1_layout.addLayout(bin_layout)

        # Кнопка оновлення гістограми
        self.update_button = QPushButton("Оновити гістограму")
        tab1_layout.addWidget(self.update_button)

        # Інформація
        self.info_label = QLabel("Кількість класів: -\nКрок розбиття: -\nРозмах: -\nКількість даних: -")
        self.info_label.setAlignment(Qt.AlignLeft)
        tab1_layout.addWidget(self.info_label)

        # Рівень довіри
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Рівень довіри (%):")
        self.confidence_entry = QDoubleSpinBox()
        self.confidence_entry.setMinimum(0)
        self.confidence_entry.setMaximum(100)
        self.confidence_entry.setValue(95.0)
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_entry)
        tab1_layout.addLayout(confidence_layout)

        # Точність
        precision_layout = QHBoxLayout()
        precision_label = QLabel("Точність (знаки після коми):")
        self.precision_entry = QSpinBox()
        self.precision_entry.setMinimum(0)
        self.precision_entry.setMaximum(10)
        self.precision_entry.setValue(4)
        precision_layout.addWidget(precision_label)
        precision_layout.addWidget(self.precision_entry)
        tab1_layout.addLayout(precision_layout)

        # Границі
        bounds_widget = QWidget()
        bounds_layout = QVBoxLayout(bounds_widget)
        bounds_layout.addWidget(QLabel("Встановлення границь"))
        lower_layout = QHBoxLayout()
        lower_label = QLabel("Нижня границя:")
        self.lower_entry = QLineEdit()
        lower_layout.addWidget(lower_label)
        lower_layout.addWidget(self.lower_entry)
        bounds_layout.addLayout(lower_layout)
        upper_layout = QHBoxLayout()
        upper_label = QLabel("Верхня границя:")
        self.upper_entry = QLineEdit()
        upper_layout.addWidget(upper_label)
        upper_layout.addWidget(self.upper_entry)
        bounds_layout.addLayout(upper_layout)
        self.apply_bounds_btn = QPushButton("Застосувати границі")
        bounds_layout.addWidget(self.apply_bounds_btn)
        tab1_layout.addWidget(bounds_widget)

        # Редагування даних
        edit_widget = QWidget()
        edit_layout = QVBoxLayout(edit_widget)
        edit_layout.addWidget(QLabel("Редагування даних"))
        self.standardize_btn = QPushButton("Стандартизувати")
        self.standardize_btn.setEnabled(False)
        edit_layout.addWidget(self.standardize_btn)
        self.log_btn = QPushButton("Логарифмувати")
        self.log_btn.setEnabled(False)
        edit_layout.addWidget(self.log_btn)
        self.shift_btn = QPushButton("Зсунути")
        self.shift_btn.setEnabled(False)
        edit_layout.addWidget(self.shift_btn)
        self.outliers_btn = QPushButton("Вилучити аномальні дані")
        self.outliers_btn.setEnabled(False)
        edit_layout.addWidget(self.outliers_btn)
        self.reset_btn = QPushButton("Скинути дані")
        self.reset_btn.setEnabled(False)
        edit_layout.addWidget(self.reset_btn)
        tab1_layout.addWidget(edit_widget)

        
       # Вибір розподілу та кнопка
        distro_layout = QHBoxLayout()
        distro_label = QLabel("Тип розподілу:")
        self.distro_combo = QComboBox()
        self.distro_combo.addItems(["Нормальний", "Експоненціальний", "Вейбулла"])  # Додаємо "Вейбулла"
        distro_layout.addWidget(distro_label)
        distro_layout.addWidget(self.distro_combo)
        tab1_layout.addLayout(distro_layout)
        self.plot_distro_btn = QPushButton("Побудувати розподіл")
        self.plot_distro_btn.setEnabled(False)
        tab1_layout.addWidget(self.plot_distro_btn)

        # Додаємо кнопку для генерації синтетичних даних Вейбулла
        self.generate_weibull_btn = QPushButton("Згенерувати вибірку Вейбулла")
        tab1_layout.addWidget(self.generate_weibull_btn)

            # Таблиця характеристик
        self.char_table = QTableWidget()
        self.char_table.setRowCount(10)
        self.char_table.setColumnCount(3)
        self.char_table.setHorizontalHeaderLabels(["Характеристика", "Зсунена", "Незсунена"])
        self.char_table.setColumnWidth(0, 150)
        self.char_table.setColumnWidth(1, 100)
        self.char_table.setColumnWidth(2, 100)
        tab1_layout.addWidget(self.char_table)

        # Текстове поле для даних
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)
        data_layout.addWidget(QLabel("Дані"))
        self.data_box = QTextEdit()
        data_layout.addWidget(self.data_box)
        self.save_btn = QPushButton("Зберегти дані")
        self.save_btn.setEnabled(False)
        data_layout.addWidget(self.save_btn)
        tab1_layout.addWidget(data_widget)

        # Правий блок (графіки)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        main_layout.addWidget(right_widget, 2)

        # Вкладки для графіків
        self.graph_tabs = QTabWidget()
        right_layout.addWidget(self.graph_tabs)

        # Вкладка з гістограмою
        self.hist_tab = QWidget()
        hist_layout = QVBoxLayout(self.hist_tab)
        self.graph_tabs.addTab(self.hist_tab, "Гістограма")

        self.hist_fig, self.hist_ax = plt.subplots(figsize=(8, 6))
        self.hist_canvas = FigureCanvas(self.hist_fig)
        hist_layout.addWidget(self.hist_canvas)

        # Вкладка з розподілами
        self.distro_tab = QWidget()
        distro_layout = QVBoxLayout(self.distro_tab)
        self.graph_tabs.addTab(self.distro_tab, "Розподіли")

        self.distro_fig, self.distro_ax = plt.subplots(figsize=(8, 6))
        self.distro_canvas = FigureCanvas(self.distro_fig)
        distro_layout.addWidget(self.distro_canvas)

        self.distro_info_label = QLabel("")
        distro_layout.addWidget(self.distro_info_label)

        # Список кнопок редагування для активації після завантаження
        self.editing_buttons = [
            self.standardize_btn, self.log_btn, self.shift_btn,
            self.outliers_btn, self.reset_btn, self.apply_bounds_btn, self.save_btn
        ]