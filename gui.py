import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QFileDialog, QWidget,
                             QMessageBox, QCheckBox, QLabel, QComboBox, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QSplitter, QGroupBox, QDialog, QLineEdit,
                             QRadioButton, QButtonGroup)
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from brain import DataProcessor

class DataAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = DataProcessor()
        self.selected_column = None
        self.shift_value = 0.0
        self.shift_direction = "Додати"
        self.shift_repeats = 1
        self.setWindowTitle("Моніторинг серверної системи")
        self.setGeometry(100, 100, 1000, 700)
        self.initUI()

    def initUI(self):
        # Основний віджет із роздільником
        splitter = QSplitter(Qt.Horizontal)

        # Ліва частина: статистики та керування
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # File info panel
        file_info_layout = QHBoxLayout()
        self.label = QLabel("Файл не завантажено")
        file_info_layout.addWidget(self.label)
        
        self.btnLoad = QPushButton("Завантажити файл")
        self.btnLoad.clicked.connect(self.loadFile)
        self.btnLoad.setFixedHeight(25)
        file_info_layout.addWidget(self.btnLoad)
        left_layout.addLayout(file_info_layout)

        # Column selection
        column_layout = QHBoxLayout()
        column_layout.addWidget(QLabel("Стовпець:"))
        self.column_selector = QComboBox()
        self.column_selector.currentIndexChanged.connect(self.updateSelectedColumn)
        column_layout.addWidget(self.column_selector)
        left_layout.addLayout(column_layout)

        # Data processing group
        data_processing_group = QGroupBox("Обробка даних")
        data_processing_layout = QHBoxLayout()
        
        self.btnFillMissing = QPushButton("Заповнити пропуски")
        self.btnFillMissing.clicked.connect(self.fillMissingValues)
        self.btnFillMissing.setFixedHeight(25)
        data_processing_layout.addWidget(self.btnFillMissing)

        self.btnDropMissing = QPushButton("Видалити пропуски")
        self.btnDropMissing.clicked.connect(self.dropMissingValues)
        self.btnDropMissing.setFixedHeight(25)
        data_processing_layout.addWidget(self.btnDropMissing)
        
        data_processing_group.setLayout(data_processing_layout)
        left_layout.addWidget(data_processing_group)

        # Transformation options
        transform_group = QGroupBox("Трансформація даних")
        transform_layout = QHBoxLayout()
        
        self.chkLog = QCheckBox("Логарифмування")
        transform_layout.addWidget(self.chkLog)

        self.chkStandardize = QCheckBox("Стандартизація")
        transform_layout.addWidget(self.chkStandardize)

        self.chkShift = QCheckBox("Зсув")
        self.chkShift.toggled.connect(self.prompt_shift_value)
        transform_layout.addWidget(self.chkShift)
        
        transform_group.setLayout(transform_layout)
        left_layout.addWidget(transform_group)

        # Histogram bins settings
        bins_group = QGroupBox("Налаштування гістограми")
        bins_layout = QVBoxLayout()
        
        self.bins_method_group = QButtonGroup()
        
        self.radio_default = QRadioButton("За замовчуванням")
        self.radio_sturges = QRadioButton("Стерджеса")
        self.radio_sqrt = QRadioButton("Кв. кореня")
        self.radio_scott = QRadioButton("Скотта")
        self.radio_custom = QRadioButton("Користувацька")
        
        self.bins_method_group.addButton(self.radio_default)
        self.bins_method_group.addButton(self.radio_sturges)
        self.bins_method_group.addButton(self.radio_sqrt)
        self.bins_method_group.addButton(self.radio_scott)
        self.bins_method_group.addButton(self.radio_custom)
        
        self.radio_default.setChecked(True)
        self.radio_custom.toggled.connect(self.toggle_custom_bins)
        
        bins_layout.addWidget(self.radio_default)
        bins_layout.addWidget(self.radio_sturges)
        bins_layout.addWidget(self.radio_sqrt)
        bins_layout.addWidget(self.radio_scott)
        bins_layout.addWidget(self.radio_custom)
        
        custom_bins_layout = QHBoxLayout()
        custom_bins_layout.addWidget(QLabel("Класів:"))
        self.custom_bins_input = QLineEdit()
        self.custom_bins_input.setValidator(QIntValidator(1, 1000))
        self.custom_bins_input.setText("10")
        self.custom_bins_input.setEnabled(False)
        self.custom_bins_input.setFixedWidth(50)
        custom_bins_layout.addWidget(self.custom_bins_input)
        bins_layout.addLayout(custom_bins_layout)
        
        bins_group.setLayout(bins_layout)
        left_layout.addWidget(bins_group)

        # Analysis buttons
        analysis_group = QGroupBox("Аналіз")
        analysis_layout = QHBoxLayout()
        
        self.btnAnalyze = QPushButton("Гістограма")
        self.btnAnalyze.clicked.connect(self.analyzeData)
        self.btnAnalyze.setFixedHeight(25)
        analysis_layout.addWidget(self.btnAnalyze)

        self.btnECDF = QPushButton("ECDF")
        self.btnECDF.clicked.connect(self.plotECDF)
        self.btnECDF.setFixedHeight(25)
        analysis_layout.addWidget(self.btnECDF)

        self.btnDetectAnomalies = QPushButton("Аномалії")
        self.btnDetectAnomalies.clicked.connect(self.detectAndRemoveAnomalies)
        self.btnDetectAnomalies.setFixedHeight(25)
        analysis_layout.addWidget(self.btnDetectAnomalies)

        # Нова кнопка для експоненціальної сітки
        self.btnExpGrid = QPushButton("Експ. сітка")
        self.btnExpGrid.clicked.connect(self.showExponentialGrid)
        self.btnExpGrid.setFixedHeight(25)
        analysis_layout.addWidget(self.btnExpGrid)
        
        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)

        # Statistics table
        stats_group = QGroupBox("Статистичні характеристики")
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(["Характеристика", "Зсунена", "Незсунена"])
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        # Права частина: графіки та таблиця даних
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        self.canvas = FigureCanvas(Figure(figsize=(8, 5)))
        right_layout.addWidget(self.canvas)

        self.data_table = QTableWidget()
        right_layout.addWidget(self.data_table)

        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        splitter.setSizes([300, 700])

        container = QWidget()
        container.setLayout(QVBoxLayout())
        container.layout().addWidget(splitter)
        self.setCentralWidget(container)

        self.updateUIState(False)

    def toggle_custom_bins(self, checked):
        self.custom_bins_input.setEnabled(checked)

    def updateUIState(self, has_data):
        widgets = [
            self.btnFillMissing, self.btnDropMissing, self.chkLog,
            self.chkStandardize, self.chkShift, self.btnAnalyze,
            self.btnECDF, self.btnDetectAnomalies, self.btnExpGrid,
            self.column_selector, self.radio_default, self.radio_sturges,
            self.radio_sqrt, self.radio_scott, self.radio_custom,
            self.custom_bins_input
        ]
        for widget in widgets:
            widget.setEnabled(has_data)

    def loadFile(self):
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, "Відкрити файл", "",
                                                    "Excel and Text Files (*.xlsx *.xls *.txt);;Excel Files (*.xlsx *.xls);;Text Files (*.txt)")
            if fileName:
                self.processor.load_data(fileName)
                self.label.setText(f"Файл завантажено: {fileName}")
                self.updateColumnSelector()
                self.displayData()
                self.updateUIState(True)
                self.updateStatsTable()
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити файл: {str(e)}")

    def updateColumnSelector(self):
        if self.processor.data is not None:
            self.column_selector.clear()
            numeric_cols = self.processor.get_numeric_columns()
            if numeric_cols:
                self.column_selector.addItems(numeric_cols)
                self.selected_column = numeric_cols[0]
            else:
                QMessageBox.warning(self, "Попередження", "У файлі відсутні числові стовпці для аналізу!")
                self.updateUIState(False)

    def updateSelectedColumn(self, index):
        if index >= 0 and self.column_selector.count() > 0:
            self.selected_column = self.column_selector.currentText()
            self.updateStatsTable()

    def displayData(self):
        if self.processor.data is not None:
            self.data_table.setRowCount(len(self.processor.data))
            self.data_table.setColumnCount(len(self.processor.data.columns))
            self.data_table.setHorizontalHeaderLabels(self.processor.data.columns)
            for i in range(len(self.processor.data)):
                for j in range(len(self.processor.data.columns)):
                    value = str(self.processor.data.iloc[i, j])
                    item = QTableWidgetItem(value)
                    self.data_table.setItem(i, j, item)
            self.data_table.resizeColumnsToContents()

    def updateStatsTable(self):
        stats = self.processor.get_stats(self.selected_column)
        if stats is None:
            self.stats_table.setRowCount(0)
            return

        stat_keys = [key for key in stats.keys() if "Незсунені" not in key and "Зсунені" not in key]
        self.stats_table.setRowCount(len(stat_keys))
        for i, key in enumerate(stat_keys):
            self.stats_table.setItem(i, 0, QTableWidgetItem(key))
            if key == "Довірчий інтервал (95%)":
                value = f"[{stats[key][0]:.4f}, {stats[key][1]:.4f}]"
            else:
                value = f"{stats[key]:.4f}" if isinstance(stats[key], float) else str(stats[key])
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))
            self.stats_table.setItem(i, 2, QTableWidgetItem(value))
        self.stats_table.resizeColumnsToContents()

    def fillMissingValues(self):
        if self.processor.fill_missing_values():
            self.displayData()
            self.updateStatsTable()

    def dropMissingValues(self):
        if self.processor.drop_missing_values():
            self.displayData()
            self.updateStatsTable()

    def prompt_shift_value(self, checked):
        if checked:
            dialog = QDialog(self)
            dialog.setWindowTitle("Введіть значення зсуву")
            dialog.setMinimumWidth(300)

            layout = QVBoxLayout()

            label = QLabel("Введіть значення зсуву (наприклад, 5.0):")
            layout.addWidget(label)

            shift_input = QLineEdit()
            shift_input.setPlaceholderText("Наприклад, 5.0 або -3.2")
            validator = QDoubleValidator()
            shift_input.setValidator(validator)
            layout.addWidget(shift_input)

            repeats_label = QLabel("Введіть кількість повторень зсуву:")
            layout.addWidget(repeats_label)

            repeats_input = QLineEdit()
            repeats_input.setPlaceholderText("Наприклад, 3")
            int_validator = QIntValidator(1, 1000)
            repeats_input.setValidator(int_validator)
            repeats_input.setText("1")
            layout.addWidget(repeats_input)

            direction_label = QLabel("Напрямок зсуву:")
            layout.addWidget(direction_label)
            direction_combo = QComboBox()
            direction_combo.addItems(["Додати", "Відняти"])
            layout.addWidget(direction_combo)

            buttons_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(lambda: self.accept_shift_value(dialog, shift_input, direction_combo, repeats_input))
            cancel_button = QPushButton("Скасувати")
            cancel_button.clicked.connect(dialog.reject)
            buttons_layout.addWidget(ok_button)
            buttons_layout.addWidget(cancel_button)
            layout.addLayout(buttons_layout)

            dialog.setLayout(layout)
            dialog.exec_()
        else:
            self.shift_value = 0.0
            self.shift_direction = "Додати"
            self.shift_repeats = 1

    def accept_shift_value(self, dialog, shift_input, direction_combo, repeats_input):
        try:
            value = float(shift_input.text())
            repeats = int(repeats_input.text())
            if repeats < 1:
                raise ValueError("Кількість повторень має бути більше 0!")
            self.shift_value = value
            self.shift_direction = direction_combo.currentText()
            self.shift_repeats = repeats
            dialog.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Помилка", f"Помилка введення: {str(e)}\nВведіть коректне числове значення!")
            self.chkShift.setChecked(False)
            dialog.reject()

    def analyzeData(self):
        if not self.selected_column:
            QMessageBox.warning(self, "Помилка", "Будь ласка, оберіть стовпець.")
            return

        if self.radio_sturges.isChecked():
            bins_method = "Стерджеса"
        elif self.radio_sqrt.isChecked():
            bins_method = "Квадратного кореня"
        elif self.radio_scott.isChecked():
            bins_method = "Скотта"
        elif self.radio_custom.isChecked():
            bins_method = "Користувацька"
        else:
            bins_method = "За замовчуванням"

        custom_bins = int(self.custom_bins_input.text()) if self.radio_custom.isChecked() else 10

        stats = self.processor.analyze_data(self.selected_column,
                                            self.chkLog.isChecked(),
                                            self.chkStandardize.isChecked(),
                                            self.chkShift.isChecked(),
                                            self.shift_value,
                                            self.shift_direction,
                                            self.shift_repeats,
                                            bins_method,
                                            custom_bins,
                                            self.canvas)
        if stats:
            stat_keys = [key.replace(" (Незсунені)", "") for key in stats.keys() if "Незсунені" in key]
            self.stats_table.setRowCount(len(stat_keys))
            for i, key in enumerate(stat_keys):
                self.stats_table.setItem(i, 0, QTableWidgetItem(key))
                shifted_key = f"{key} (Зсунені)"
                non_shifted_key = f"{key} (Незсунені)"
                if shifted_key in stats:
                    if shifted_key == "Довірчий інтервал (95%) (Зсунені)":
                        value = f"[{stats[shifted_key][0]:.4f}, {stats[shifted_key][1]:.4f}]"
                    else:
                        value = f"{stats[shifted_key]:.4f}" if isinstance(stats[shifted_key], float) else str(stats[shifted_key])
                    self.stats_table.setItem(i, 1, QTableWidgetItem(value))
                if non_shifted_key in stats:
                    if non_shifted_key == "Довірчий інтервал (95%) (Незсунені)":
                        value = f"[{stats[non_shifted_key][0]:.4f}, {stats[non_shifted_key][1]:.4f}]"
                    else:
                        value = f"{stats[non_shifted_key]:.4f}" if isinstance(stats[non_shifted_key], float) else str(stats[non_shifted_key])
                    self.stats_table.setItem(i, 2, QTableWidgetItem(value))
            self.stats_table.resizeColumnsToContents()

    def plotECDF(self):
        if not self.selected_column:
            QMessageBox.warning(self, "Помилка", "Будь ласка, оберіть стовпець.")
            return

        stats = self.processor.plot_ecdf(self.selected_column, self.canvas)
        if stats:
            self.updateStatsTable()

    def detectAndRemoveAnomalies(self):
        if self.processor.data is None or not self.selected_column:
            QMessageBox.warning(self, "Помилка", "Дані або стовпець не вибрано!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Виявлення та видалення аномалій")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Вкажіть діапазон значень для аномалій:"))

        lower_layout = QHBoxLayout()
        lower_layout.addWidget(QLabel("Нижня межа:"))
        self.lower_bound_input = QLineEdit()
        self.lower_bound_input.setPlaceholderText("Наприклад, -100")
        self.lower_bound_input.setValidator(QDoubleValidator())
        lower_layout.addWidget(self.lower_bound_input)
        layout.addLayout(lower_layout)

        upper_layout = QHBoxLayout()
        upper_layout.addWidget(QLabel("Верхня межа:"))
        self.upper_bound_input = QLineEdit()
        self.upper_bound_input.setPlaceholderText("Наприклад, 500")
        self.upper_bound_input.setValidator(QDoubleValidator())
        upper_layout.addWidget(self.upper_bound_input)
        layout.addLayout(upper_layout)

        detect_button = QPushButton("Виявити аномалії")
        detect_button.clicked.connect(lambda: self.showAnomalies(dialog))
        layout.addWidget(detect_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def showAnomalies(self, parent_dialog):
        try:
            lower_bound = float(self.lower_bound_input.text()) if self.lower_bound_input.text().strip() else float('-inf')
            upper_bound = float(self.upper_bound_input.text()) if self.upper_bound_input.text().strip() else float('inf')
        except ValueError:
            QMessageBox.warning(self, "Помилка", "Введіть коректні числові значення для меж!")
            return

        result = self.processor.detect_anomalies_with_bounds(self.selected_column, lower_bound, upper_bound, self.canvas)

        if not result["anomalies"]:
            QMessageBox.information(self, "Результат", "Аномалії не знайдено.")
            return

        reply = QMessageBox.question(self, "Аномалії знайдено",
                                     f"Знайдено {len(result['anomalies'])} аномальних значень. Видалити?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.processor.remove_anomalies(self.selected_column, result["indices"])
            self.displayData()
            self.updateStatsTable()
            QMessageBox.information(self, "Готово", "Аномалії видалено.")
            parent_dialog.accept()

    def showExponentialGrid(self):
        """Нова реалізація експоненціальної сітки"""
        if not self.selected_column:
            QMessageBox.warning(self, "Помилка", "Будь ласка, оберіть стовпець.")
            return

        # Отримуємо дані вибраного стовпця
        data = self.processor.data[self.selected_column].dropna()
        if data.empty:
            QMessageBox.warning(self, "Помилка", "Дані відсутні або всі значення є пропусками!")
            return

        # Перевіряємо, чи всі значення невід'ємні для експоненціального розподілу
        if (data < 0).any():
            QMessageBox.warning(self, "Попередження", "Експоненціальний розподіл можливий лише для невід'ємних значень!")
            return

        # Розрахунок параметру лямбда за методом моментів
        mean_value = data.mean()
        if mean_value <= 0:
            QMessageBox.warning(self, "Помилка", "Середнє значення має бути більше 0 для експоненціального розподілу!")
            return
        
        lambda_param = 1 / mean_value
        
        # Діалогове вікно для налаштування параметрів сітки
        dialog = QDialog(self)
        dialog.setWindowTitle("Параметри експоненціальної сітки")
        dialog.setMinimumWidth(350)
        
        layout = QVBoxLayout()
        
        # Параметр λ
        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(QLabel(f"Параметр λ (1/середнє = {lambda_param:.4f}):"))
        lambda_input = QLineEdit()
        lambda_input.setText(f"{lambda_param:.4f}")
        lambda_input.setValidator(QDoubleValidator(0.0001, 1000, 6))
        lambda_layout.addWidget(lambda_input)
        layout.addLayout(lambda_layout)
        
        # Кількість точок сітки
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Кількість точок:"))
        points_input = QLineEdit()
        points_input.setText("10")
        points_input.setValidator(QIntValidator(2, 100))
        points_layout.addWidget(points_input)
        layout.addLayout(points_layout)
        
        # Вибір методу розбиття діапазону
        spacing_group = QGroupBox("Метод розбиття діапазону")
        spacing_layout = QVBoxLayout()
        
        spacing_method_group = QButtonGroup()
        radio_linear = QRadioButton("Рівномірний")
        radio_exponential = QRadioButton("Експоненціальний")
        radio_custom_range = QRadioButton("Власний діапазон")
        
        radio_linear.setChecked(True)
        
        spacing_method_group.addButton(radio_linear)
        spacing_method_group.addButton(radio_exponential)
        spacing_method_group.addButton(radio_custom_range)
        
        spacing_layout.addWidget(radio_linear)
        spacing_layout.addWidget(radio_exponential)
        spacing_layout.addWidget(radio_custom_range)
        
        # Поля для власного діапазону
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Діапазон:"))
        
        min_range = QLineEdit()
        min_range.setPlaceholderText("Min")
        min_range.setValidator(QDoubleValidator())
        min_range.setText("0")
        min_range.setEnabled(False)
        range_layout.addWidget(min_range)
        
        range_layout.addWidget(QLabel("—"))
        
        max_range = QLineEdit()
        max_range.setPlaceholderText("Max")
        max_range.setValidator(QDoubleValidator())
        max_range.setText(f"{data.max() * 1.5:.2f}")
        max_range.setEnabled(False)
        range_layout.addWidget(max_range)
        
        spacing_layout.addLayout(range_layout)
        spacing_group.setLayout(spacing_layout)
        layout.addWidget(spacing_group)
        
        # Активація/деактивація полів діапазону
        def toggle_range_inputs(checked):
            min_range.setEnabled(checked)
            max_range.setEnabled(checked)
        
        radio_custom_range.toggled.connect(toggle_range_inputs)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("Побудувати")
        cancel_button = QPushButton("Скасувати")
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        dialog.setLayout(layout)
        
        # Обробники кнопок
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(lambda: self.buildExponentialGrid(
            dialog,
            float(lambda_input.text()),
            int(points_input.text()),
            "linear" if radio_linear.isChecked() else ("exponential" if radio_exponential.isChecked() else "custom"),
            float(min_range.text() if min_range.text() else 0),
            float(max_range.text() if max_range.text() else data.max() * 1.5),
            data
        ))
        
        dialog.exec_()



    def buildExponentialGrid(self, dialog, lambda_param, num_points, spacing_method, min_range, max_range, data):
        """Побудова простої експоненціальної сітки"""
        try:
            if lambda_param <= 0:
                raise ValueError("Параметр λ повинен бути більше 0")
            
            if num_points < 2:
                raise ValueError("Кількість точок повинна бути не менше 2")
            
            if min_range < 0:
                raise ValueError("Мінімальне значення діапазону не може бути від'ємним для експоненціального розподілу")
            
            if max_range <= min_range:
                raise ValueError("Максимальне значення діапазону повинно бути більше мінімального")
            
            # Визначаємо точки сітки в залежності від методу
            if spacing_method == "linear":
                x_grid = np.linspace(min_range, max_range, num_points)
            elif spacing_method == "exponential":
                base = np.linspace(0, 1, num_points)
                x_grid = min_range + (max_range - min_range) * (np.exp(base * 3) - 1) / (np.exp(3) - 1)
            else:
                x_grid = np.linspace(min_range, max_range, num_points)
            
            # Обчислюємо функцію розподілу F(x) = 1 - e^(-λx)
            F_x_grid = 1 - np.exp(-lambda_param * x_grid)
            
            # Побудова графіка
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            ax.plot(x_grid, F_x_grid, 'bo-', markersize=5)
            ax.set_xlabel('x')
            ax.set_ylabel('F(x)')
            ax.set_title('Експоненціальна сітка')
            ax.grid(True)
            
            # Виведення точок сітки в таблицю
            table_data = np.column_stack((x_grid, F_x_grid))
            self.data_table.setRowCount(len(table_data))
            self.data_table.setColumnCount(2)
            self.data_table.setHorizontalHeaderLabels(['x', 'F(x)'])
            
            for i, row in enumerate(table_data):
                self.data_table.setItem(i, 0, QTableWidgetItem(f"{row[0]:.6f}"))
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{row[1]:.6f}"))
            
            self.data_table.resizeColumnsToContents()
            self.canvas.draw()
            dialog.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Помилка", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Під час побудови сітки сталася помилка: {str(e)}")
def closeEvent(self, event):
    """Обробка закриття вікна програми"""
    reply = QMessageBox.question(self, 'Завершення роботи',
                                'Ви впевнені, що хочете вийти?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

    if reply == QMessageBox.Yes:
        event.accept()
    else:
        event.ignore()

def save_results(self):
    """Збереження результатів аналізу"""
    if self.processor.data is None:
        QMessageBox.warning(self, "Помилка", "Немає даних для збереження!")
        return
    
    try:
        fileName, _ = QFileDialog.getSaveFileName(self, "Зберегти результати", "",
                                                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)")
        
        if not fileName:
            return
        
        if fileName.endswith('.xlsx'):
            self.processor.data.to_excel(fileName, index=False)
        elif fileName.endswith('.csv'):
            self.processor.data.to_csv(fileName, index=False)
        else:
            if '.' not in fileName:
                fileName += '.xlsx'
                self.processor.data.to_excel(fileName, index=False)
        
        QMessageBox.information(self, "Успіх", f"Дані успішно збережено у файл: {fileName}")
    
    except Exception as e:
        QMessageBox.critical(self, "Помилка збереження", f"Не вдалося зберегти файл: {str(e)}")

def show_help(self):
    """Показ довідки про програму"""
    help_text = """
    <h3>Довідка з використання програми "Моніторинг серверної системи"</h3>
    
    <p><b>Основні функції:</b></p>
    <ul>
        <li>Завантаження даних з Excel або текстових файлів</li>
        <li>Обробка пропущених значень</li>
        <li>Трансформація даних (логарифмування, стандартизація, зсув)</li>
        <li>Побудова гістограм з різними методами розбиття</li>
        <li>Аналіз емпіричної функції розподілу (ECDF)</li>
        <li>Виявлення та видалення аномалій</li>
        <li>Побудова експоненціальної сітки</li>
    </ul>
    
    <p><b>Порядок роботи:</b></p>
    <ol>
        <li>Завантажте файл даних</li>
        <li>Виберіть стовпець для аналізу</li>
        <li>За необхідності обробіть пропущені значення</li>
        <li>Виберіть потрібні опції трансформації</li>
        <li>Виконайте аналіз даних або побудуйте графіки</li>
    </ol>
    
    <p><b>Контакти для підтримки:</b><br>
    example@domain.com</p>
    """
    
    help_dialog = QDialog(self)
    help_dialog.setWindowTitle("Довідка")
    help_dialog.setMinimumSize(500, 400)
    
    layout = QVBoxLayout()
    help_label = QLabel(help_text)
    help_label.setWordWrap(True)
    help_label.setOpenExternalLinks(True)
    
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(help_label)
    
    layout.addWidget(scroll_area)
    
    close_button = QPushButton("Закрити")
    close_button.clicked.connect(help_dialog.accept)
    layout.addWidget(close_button)
    
    help_dialog.setLayout(layout)
    help_dialog.exec_()

def export_statistics(self):
    """Експорт статистичних даних"""
    if self.processor.data is None or self.selected_column is None:
        QMessageBox.warning(self, "Помилка", "Немає даних для експорту!")
        return
    
    try:
        fileName, _ = QFileDialog.getSaveFileName(self, "Експорт статистики", "",
                                                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)")
        
        if not fileName:
            return
        
        # Збираємо дані з таблиці статистики
        stats_data = []
        for row in range(self.stats_table.rowCount()):
            row_data = []
            for col in range(self.stats_table.columnCount()):
                item = self.stats_table.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("")
            stats_data.append(row_data)
        
        df_stats = pd.DataFrame(stats_data, columns=["Характеристика", "Зсунена", "Незсунена"])
        
        if fileName.endswith('.xlsx'):
            df_stats.to_excel(fileName, index=False)
        elif fileName.endswith('.csv'):
            df_stats.to_csv(fileName, index=False)
        else:
            if '.' not in fileName:
                fileName += '.xlsx'
                df_stats.to_excel(fileName, index=False)
        
        QMessageBox.information(self, "Успіх", f"Статистику успішно експортовано у файл: {fileName}")
    
    except Exception as e:
        QMessageBox.critical(self, "Помилка експорту", f"Не вдалося експортувати статистику: {str(e)}")

def save_chart(self):
    """Збереження поточного графіка"""
    if self.processor.data is None:
        QMessageBox.warning(self, "Помилка", "Немає графіка для збереження!")
        return
    
    try:
        fileName, _ = QFileDialog.getSaveFileName(self, "Зберегти графік", "",
                                                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        
        if not fileName:
            return
        
        if not (fileName.endswith('.png') or fileName.endswith('.pdf')):
            fileName += '.png'
        
        self.canvas.figure.savefig(fileName, dpi=300, bbox_inches='tight')
        QMessageBox.information(self, "Успіх", f"Графік успішно збережено у файл: {fileName}")
    
    except Exception as e:
        QMessageBox.critical(self, "Помилка збереження", f"Не вдалося зберегти графік: {str(e)}")