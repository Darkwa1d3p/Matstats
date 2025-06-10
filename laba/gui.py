from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
                             QDoubleSpinBox, QSpinBox, QTextEdit, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Статистичний аналіз")
        self.samples = {}  # Словник для зберігання вибірок
        self.init_ui()

    def init_ui(self):
        # Основний віджет програми
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Лівий блок (елементи керування) з прокручуванням
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Вкладки для організації елементів керування
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        # Додаємо QScrollArea для прокручування лівої панелі
        scroll_area = QScrollArea()
        scroll_area.setWidget(left_widget)
        scroll_area.setWidgetResizable(True)  # Дозволяємо віджету розтягуватися
        main_layout.addWidget(scroll_area, 1)

        # Вкладка 1: Основний аналіз
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.tabs.addTab(tab1, "Основний аналіз")

        # Кнопка для завантаження даних
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

        # Кнопка для оновлення гістограми
        self.update_button = QPushButton("Оновити гістограму")
        tab1_layout.addWidget(self.update_button)

        # Інформація про гістограму
        self.info_label = QLabel("Кількість класів: -\nКрок розбиття: -\nРозмах: -\nКількість даних: -")
        self.info_label.setAlignment(Qt.AlignLeft)
        tab1_layout.addWidget(self.info_label)

        # Рівень довіри для статистичних обчислень
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Рівень довіри (%):")
        self.confidence_entry = QDoubleSpinBox()
        self.confidence_entry.setMinimum(0)
        self.confidence_entry.setMaximum(100)
        self.confidence_entry.setValue(95.0)
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_entry)
        tab1_layout.addLayout(confidence_layout)

        # Точність для відображення чисел
        precision_layout = QHBoxLayout()
        precision_label = QLabel("Точність (знаки після коми):")
        self.precision_entry = QSpinBox()
        self.precision_entry.setMinimum(0)
        self.precision_entry.setMaximum(10)
        self.precision_entry.setValue(4)
        precision_layout.addWidget(precision_label)
        precision_layout.addWidget(self.precision_entry)
        tab1_layout.addLayout(precision_layout)

        # Кількість експериментів для моделювання
        experiments_layout = QHBoxLayout()
        experiments_label = QLabel("Кількість експериментів:")
        self.experiments_entry = QSpinBox()
        self.experiments_entry.setMinimum(1)
        self.experiments_entry.setMaximum(1000)
        self.experiments_entry.setValue(100)
        experiments_layout.addWidget(experiments_label)
        experiments_layout.addWidget(self.experiments_entry)
        tab1_layout.addLayout(experiments_layout)

        # Встановлення границь для даних
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
        self.variation_series_btn = QPushButton("Показати варіаційний ряд")
        self.variation_series_btn.setEnabled(False)
        edit_layout.addWidget(self.variation_series_btn)
        self.reset_btn = QPushButton("Скинути дані")
        self.reset_btn.setEnabled(False)
        edit_layout.addWidget(self.reset_btn)
        tab1_layout.addWidget(edit_widget)

        # Вибір розподілу та кнопка для побудови
        distro_layout = QHBoxLayout()
        distro_label = QLabel("Тип розподілу:")
        self.distro_combo = QComboBox()
        self.distro_combo.addItems(["Нормальний", "Нормальний (гістограма)", "Експоненціальний",
                                    "Експоненціальний (гістограма)", "Вейбулла",
                                    "Вейбулла (гістограма)", "Уніформний (гістограма)", 
                                    "Лог-нормальний (гістограма)"])
        distro_layout.addWidget(distro_label)
        distro_layout.addWidget(self.distro_combo)
        tab1_layout.addLayout(distro_layout)
        self.plot_distro_btn = QPushButton("Побудувати розподіл")
        self.plot_distro_btn.setEnabled(False)
        tab1_layout.addWidget(self.plot_distro_btn)
        self.plot_polygon_btn = QPushButton("Побудувати полігон частот")
        self.plot_polygon_btn.setEnabled(False)
        tab1_layout.addWidget(self.plot_polygon_btn)

        # Кнопки для генерації синтетичних даних
        self.generate_normal_btn = QPushButton("Згенерувати нормальну вибірку")
        tab1_layout.addWidget(self.generate_normal_btn)
        self.generate_weibull_btn = QPushButton("Згенерувати вибірку Вейбулла")
        tab1_layout.addWidget(self.generate_weibull_btn)
        self.generate_exponential_btn = QPushButton("Згенерувати експоненціальну вибірку")
        tab1_layout.addWidget(self.generate_exponential_btn)
        self.generate_uniform_btn = QPushButton("Згенерувати уніформну вибірку")
        tab1_layout.addWidget(self.generate_uniform_btn)
        self.generate_lognormal_btn = QPushButton("Згенерувати лог-нормальну вибірку")
        tab1_layout.addWidget(self.generate_lognormal_btn)

        # Вибір розподілу та кнопка для t-тесту
        ttest_layout = QHBoxLayout()
        ttest_label = QLabel("Тип розподілу для t-тесту:")
        self.distro_combo_ttest = QComboBox()
        self.distro_combo_ttest.addItems(["Нормальний", "Експоненціальний", "Вейбулла",
                                          "Уніформний", "Лог-нормальний"])
        ttest_layout.addWidget(ttest_label)
        ttest_layout.addWidget(self.distro_combo_ttest)
        tab1_layout.addLayout(ttest_layout)
        self.run_ttest_btn = QPushButton("Запустити t-тест")
        self.run_ttest_btn.setEnabled(False)
        tab1_layout.addWidget(self.run_ttest_btn)
        # Кнопка для запуску експерименту
        self.run_experiment_btn = QPushButton("Запустити експеримент")
        self.run_experiment_btn.setEnabled(True)  # Доступна без завантаження даних
        tab1_layout.addWidget(self.run_experiment_btn)
        # Кнопка для перевірки гіпотези H0: μ=значення
        self.test_h0_btn = QPushButton("Перевірити H0: μ=значення")
        self.test_h0_btn.setEnabled(False)
        tab1_layout.addWidget(self.test_h0_btn)

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

        # Список кнопок редагування для активації після завантаження даних
        self.editing_buttons = [
            self.standardize_btn, self.log_btn, self.shift_btn,
            self.outliers_btn, self.reset_btn, self.apply_bounds_btn,
            self.save_btn, self.run_ttest_btn, self.variation_series_btn,
            self.test_h0_btn, self.plot_polygon_btn, self.run_experiment_btn
        ]

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

        # Додаємо QComboBox для вибору експерименту
        self.exp_combo = QComboBox()
        self.exp_combo.currentTextChanged.connect(self.update_histogram)
        hist_layout.addWidget(self.exp_combo)

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

        # Вкладка з детальними даними
        self.data_tab = QWidget()
        data_tab_layout = QVBoxLayout(self.data_tab)
        self.graph_tabs.addTab(self.data_tab, "Детальні дані")

        self.char_table = QTableWidget()
        self.char_table.setRowCount(0)  # Ініціалізуємо з нульовою кількістю рядків
        self.char_table.setColumnCount(9)  # Оновлено до 9 стовпців для t-тесту
        self.char_table.setHorizontalHeaderLabels([
            "Розмір вибірки", "Параметр", "Справжнє значення", "Оцінка", "Ст. помилка",
            "T-статистика", "Критичне значення", "p-значення", "Висновок"
        ])
        self.char_table.setColumnWidth(0, 100)
        self.char_table.setColumnWidth(1, 80)
        self.char_table.setColumnWidth(2, 120)
        self.char_table.setColumnWidth(3, 100)
        self.char_table.setColumnWidth(4, 100)
        self.char_table.setColumnWidth(5, 100)
        self.char_table.setColumnWidth(6, 120)
        self.char_table.setColumnWidth(7, 100)
        self.char_table.setColumnWidth(8, 150)
        self.char_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        data_tab_layout.addWidget(self.char_table)

    def update_histogram(self):
        """Оновлення гістограми на основі вибраного експерименту."""
        selected_exp = self.exp_combo.currentText()
        if selected_exp and selected_exp in self.samples:
            sample = self.samples[selected_exp]
            n = len(sample)
            # Нова формула для визначення кількості класів
            if n < 100:
                bins = int(np.sqrt(n))
                if n % 2 == 0:  # Якщо парне
                    bins -= 1
            else:  # n >= 100 (включно)
                bins = int(np.cbrt(n))  # Кубічний корінь
                if n % 2 == 0:  # Якщо парне
                    bins -= 1
            # Якщо вручну задано кількість класів через self.bin_entry, використовуємо її
            bins = self.bin_entry.value() if self.bin_entry.value() > 0 else max(1, bins)
            self.hist_ax.clear()
            # Додаємо контури (грані) до стовпців
            self.hist_ax.hist(sample, bins=bins, density=True, rwidth=1, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            self.hist_ax.set_title("Гістограма вибірки")
            self.hist_ax.set_xlabel("Значення")
            self.hist_ax.set_ylabel("Відносна частота")
            self.hist_ax.grid(True, linestyle='--', alpha=0.7)
            # Оновлення інформації про гістограму
            range_val = np.max(sample) - np.min(sample)
            bin_width = range_val / bins if bins > 0 else 0
            self.info_label.setText(
                f"Кількість класів: {bins}\n"
                f"Крок розбиття: {bin_width:.4f}\n"
                f"Розмах: {range_val:.4f}\n"
                f"Кількість даних: {n}"
            )
            self.hist_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())