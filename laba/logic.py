import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, t, chi2, sem, kstest, expon
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QInputDialog, QDialog, QListWidget,
                             QPushButton, QVBoxLayout, QTableWidgetItem, QLabel)  # Додано QLabel
import matplotlib.pyplot as plt

# Глобальні змінні
values = np.array([])
original_values = np.array([])
gui = None

def save_data():
    global values
    file_path, _ = QFileDialog.getSaveFileName(gui, "Зберегти дані", "", "Text files (*.txt)")
    if file_path:
        with open(file_path, 'w') as file:
            file.write('\n'.join(map(str, values)))
        QMessageBox.information(gui, "Збереження", "Дані успішно збережено")

def update_from_data_box():
    global values
    try:
        text = gui.data_box.toPlainText().strip()
        data_list = [x.strip() for x in text.split(',') if x.strip()]
        new_values = []
        for item in data_list:
            try:
                new_values.append(float(item))
            except ValueError:
                new_values.append(np.nan)
        new_values = np.array(new_values)
        
        nan_mask = pd.isna(new_values)
        if np.any(nan_mask):
            mean_value = np.nanmean(new_values)
            if np.isnan(mean_value):
                raise ValueError("Усі значення є пропущеними. Неможливо обчислити середнє.")
            new_values = np.where(nan_mask, mean_value, new_values)
            QMessageBox.information(gui, "Обробка даних", f"Замінено {np.sum(nan_mask)} пропущених значень середнім: {mean_value:.4f}")
        
        if not np.array_equal(values, new_values):
            values = new_values
            update_statistics()
            update_characteristics()
            update_data_box()
    except ValueError as e:
        QMessageBox.critical(gui, "Помилка", f"Некоректний формат даних: {str(e)}")

def calculate_confidence_interval(data, statistic, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    if statistic == "mean":
        return mean - h, mean + h
    elif statistic == "variance":
        var = np.var(data, ddof=1)
        chi2_lower = chi2.ppf((1 - confidence) / 2, n - 1)
        chi2_upper = chi2.ppf((1 + confidence) / 2, n - 1)
        return (n - 1) * var / chi2_upper, (n - 1) * var / chi2_lower
    elif statistic == "std":
        var_lower, var_upper = calculate_confidence_interval(data, "variance", confidence)
        return np.sqrt(var_lower), np.sqrt(var_upper)
    return None, None

def update_characteristics():
    global values
    if len(values) == 0:
        return
    confidence = gui.confidence_entry.value() / 100
    precision = gui.precision_entry.value()
    
    gui.char_table.clearContents()
    
    mean = np.mean(values)
    variance_unbiased = np.var(values, ddof=1)
    std_dev_unbiased = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev_unbiased / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)
    
    mean_ci_lower, mean_ci_upper = calculate_confidence_interval(values, "mean", confidence)
    var_ci_lower, var_ci_upper = calculate_confidence_interval(values, "variance", confidence)
    std_ci_lower, std_ci_upper = calculate_confidence_interval(values, "std", confidence)
    
    fmt = f".{precision}f"
    characteristics = [
        ("Середнє", mean, f"[{mean_ci_lower:{fmt}}, {mean_ci_upper:{fmt}}]"),
        ("Дисперсія", variance_unbiased, f"[{var_ci_lower:{fmt}}, {var_ci_upper:{fmt}}]"),
        ("Середньокв. відхилення", std_dev_unbiased, f"[{std_ci_lower:{fmt}}, {std_ci_upper:{fmt}}]"),
        ("Асиметрія", skewness, "-"),
        ("Ексцес", excess_kurtosis, "-"),
        ("Контрексцес", counter_kurtosis, "-"),
        ("Варіація Пірсона", pearson_var, "-"),
        ("Непарам. коеф. вар.", nonparam_var, "-"),
        ("MAD", mad_val, "-"),
        ("Медіана", med_val, "-")
    ]
    
    for row, (char, value, ci) in enumerate(characteristics):
        gui.char_table.setItem(row, 0, QTableWidgetItem(char))
        gui.char_table.setItem(row, 1, QTableWidgetItem(f"{value:{fmt}}"))
        gui.char_table.setItem(row, 2, QTableWidgetItem(ci))

def update_statistics():
    global values
    if len(values) == 0:
        return
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)

def update_data_box():
    global values
    gui.data_box.setPlainText(', '.join([f"{val:.4f}" for val in values]))
    update_histogram()

def load_data():
    global values, original_values
    file_path, _ = QFileDialog.getOpenFileName(gui, "Відкрити файл", "", "Excel files (*.xlsx *.xls);;Text files (*.txt)")
    if not file_path:
        QMessageBox.warning(gui, "Попередження", "Файл не вибрано")
        return
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            wait_time_col = "Час очікування (хв)"
            if wait_time_col not in df.columns:
                possible_cols = [col for col in df.columns if "час" in col.lower() or "wait" in col.lower()]
                if not possible_cols:
                    raise ValueError("У файлі немає стовпця з часом очікування. Перевірте структуру даних.")
                wait_time_col = possible_cols[0]
            
            df[wait_time_col] = df[wait_time_col].replace(r'^\s*$', np.nan, regex=True)
            df[wait_time_col] = pd.to_numeric(df[wait_time_col], errors='coerce')
            values = df[wait_time_col].to_numpy()
            
            nan_mask = pd.isna(values)
            if np.any(nan_mask):
                mean_value = np.nanmean(values)
                if np.isnan(mean_value):
                    raise ValueError("Усі значення в стовпці є пропущеними. Неможливо обчислити середнє.")
                values = np.where(nan_mask, mean_value, values)
                nan_count = np.sum(nan_mask)
                QMessageBox.information(gui, "Обробка даних", f"Замінено {nan_count} пропущених значень середнім: {mean_value:.4f}")
        else:
            with open(file_path, 'r') as file:
                data = file.read().replace(',', '.').strip()
            data_list = [x for x in data.split() if x]
            values = []
            for item in data_list:
                try:
                    values.append(float(item))
                except ValueError:
                    continue
            values = np.array(values)
            if len(values) == 0:
                raise ValueError("Дані порожні або некоректні")
        
        if len(values) == 0:
            raise ValueError("Дані порожні або некоректні")
        
        original_values = values.copy()
        update_statistics()
        update_characteristics()
        update_data_box()
        for btn in gui.editing_buttons:
            btn.setEnabled(True)
        gui.plot_distro_btn.setEnabled(True)
        min_val, max_val = np.min(values), np.max(values)
        gui.lower_entry.setText(str(min_val))
        gui.upper_entry.setText(str(max_val))
        
        mean_wait_time = np.mean(values)
        recommendation = ("Рекомендується збільшити кількість операторів у пікові години, "
                        "а також розглянути впровадження IVR для автоматичних відповідей.") if mean_wait_time > 5 else "Поточна кількість операторів достатня."
        QMessageBox.information(gui, "Аналіз", f"Середній час очікування: {mean_wait_time:.2f} хв\nРекомендація: {recommendation}")
    except Exception as e:
        QMessageBox.critical(gui, "Помилка", f"Не вдалося завантажити дані: {str(e)}")
        values = np.array([])
        original_values = np.array([])

def update_histogram():
    global values
    if len(values) == 0:
        return
    bin_count = gui.bin_entry.value()
    if bin_count == 0:
        bin_count = int(np.sqrt(len(values)))
    
    gui.hist_ax.clear()
    
    hist, bins, _ = gui.hist_ax.hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black', density=True)
    gui.hist_ax.set_title('Гістограма часу очікування та щільність')
    
    mean, std = np.mean(values), np.std(values)
    x = np.linspace(min(values), max(values), 100)
    density = norm.pdf(x, mean, std)
    gui.hist_ax.plot(x, density, 'r-', label='Щільність')
    
    gui.hist_ax.legend()
    
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    gui.hist_ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    
    y_max = max(np.max(hist), np.max(density))
    gui.hist_ax.set_ylim(0, y_max * 1.1)
    
    gui.hist_canvas.draw()
    
    range_val = np.ptp(values)
    bin_width = range_val / bin_count
    gui.info_label.setText(f'Кількість класів: {bin_count}\nКрок розбиття: {bin_width:.3f}\nРозмах: {range_val:.3f}\nКількість даних: {len(values)}')

def plot_distribution():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для побудови розподілу")
        return
    
    distro_type = gui.distro_combo.currentText()
    gui.distro_ax.clear()
    gui.distro_info_label.setText("")
    
    confidence = gui.confidence_entry.value() / 100
    
    n_bins = gui.bin_entry.value()
    if n_bins == 0:
        n_bins = int(np.sqrt(len(values)))
    
    if distro_type == "Нормальний":
        bin_dt, bin_gr = np.histogram(values, bins=n_bins)
        Y = np.cumsum(bin_dt) / len(values)
        
        gui.distro_ax.plot([bin_gr[0], bin_gr[0]], [0, Y[0]], color='green', linewidth=2, label='Емпіричний розподіл')
        for i in range(len(Y)):
            gui.distro_ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2)
        gui.distro_ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
        
        mean, std = np.mean(values), np.std(values)
        
        x_min, x_max = np.min(values), np.max(values)
        x_range = x_max - x_min if x_max != x_min else 1
        x_margin = 0.1 * x_range
        x_lower = x_min - x_margin
        x_upper = x_max + x_margin
        
        x_theor = np.linspace(x_lower, x_upper, 1000)
        cdf = norm.cdf(x_theor, mean, std)
        gui.distro_ax.plot(x_theor, cdf, label='Нормальний розподіл', color='red')
        
        n = len(values)
        epsilon = np.sqrt(1/(2*n) * np.log(2/(1-confidence)))
        gui.distro_ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                                   color='red', alpha=0.2, label='Довірчий інтервал')
        
        gui.distro_ax.set_title('Порівняння емпіричного та нормального розподілів')
        gui.distro_ax.set_xlabel('Час очікування (хв)')
        gui.distro_ax.set_ylabel('Ймовірність')
        ks_statistic, ks_pvalue = kstest(values, 'norm', args=(mean, std))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл нормальний" if ks_statistic < critical_value else "Розподіл не нормальний"
    else:  # Експоненціальний
        if np.any(values < 0):
            QMessageBox.critical(gui, "Помилка", "Експоненціальний розподіл можливий лише для невід’ємних значень")
            return
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        empirical_probs = np.arange(1, n + 1) / (n + 1)
        y_values = -np.log(1 - empirical_probs)
        
        mean = np.mean(values)
        if mean == 0:
            QMessageBox.critical(gui, "Помилка", "Середнє значення дорівнює нулю. Неможливо оцінити параметр.")
            return
        lambda_param = 1 / mean
        
        gui.distro_ax.scatter(sorted_values, y_values, color='green', label='Дані', s=50)
        
        x_theor = np.linspace(0, np.max(sorted_values) * 1.2, 100)
        y_theor = lambda_param * x_theor
        gui.distro_ax.plot(x_theor, y_theor, color='blue', linestyle='--', label=f'Експоненціальний розподіл (λ={lambda_param:.4f})')
        
        x_min, x_max = np.min(values), np.max(values)
        x_range = x_max - x_min if x_max != x_min else 1
        x_margin = 0.1 * x_range
        x_lower = max(0, x_min - x_margin)
        x_upper = x_max + x_margin
        y_max = np.max(y_values) * 1.2
        gui.distro_ax.set_xlim(x_lower, x_upper)
        gui.distro_ax.set_ylim(0, y_max)
        
        gui.distro_ax.set_title('Імовірнісна сітка експоненціального розподілу')
        gui.distro_ax.set_xlabel('Значення (Час очікування, хв)')
        gui.distro_ax.set_ylabel('-ln(1 - F(x))')
        ks_statistic, ks_pvalue = kstest(values, 'expon', args=(0, mean))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає експоненціальному" if ks_statistic < critical_value else "Розподіл не відповідає експоненціальному"
    
    gui.distro_ax.legend()
    gui.distro_ax.grid(True, linestyle='--', alpha=0.7)
    gui.distro_canvas.draw()
    
    ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
               f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
               f"Висновок: {conclusion}")
    gui.distro_info_label.setText(ks_text)

def standardize_data():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для стандартизації")
        return
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        QMessageBox.critical(gui, "Помилка", "Стандартне відхилення дорівнює нулю. Неможливо стандартизувати.")
        return
    values = (values - mean) / std
    update_statistics()
    update_characteristics()
    update_data_box()
    QMessageBox.information(gui, "Стандартизація", "Дані успішно стандартизовано")

def log_transform():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для логарифмування")
        return
    if np.min(values) <= 0:
        QMessageBox.critical(gui, "Помилка", "Логарифмування можливе тільки для додатних значень")
        return
    values = np.log(values)
    update_statistics()
    update_characteristics()
    update_data_box()
    QMessageBox.information(gui, "Логарифмування", "Дані успішно логарифмовано")

def shift_data():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для зсуву")
        return
    shift_value, ok = QInputDialog.getDouble(gui, "Зсув даних", "Введіть значення зсуву:")
    if ok:
        values = values + shift_value
        update_statistics()
        update_characteristics()
        update_data_box()
        QMessageBox.information(gui, "Зсув", f"Дані успішно зсунуто на {shift_value}")

def remove_outliers():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для обробки")
        return
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        QMessageBox.critical(gui, "Помилка", "Стандартне відхилення дорівнює нулю. Неможливо виявити викиди.")
        return
    z_scores = np.abs((values - mean) / std)
    threshold = 3
    outlier_indices = np.where(z_scores > threshold)[0]
    
    if len(outlier_indices) == 0:
        QMessageBox.information(gui, "Аномалії", "Аномальних значень не знайдено")
        return
    
    outlier_info = [f"Індекс: {i}, Значення: {values[i]:.4f}, Z-оцінка: {z_scores[i]:.4f}" 
                    for i in outlier_indices]
    
    dialog = QDialog(gui)
    dialog.setWindowTitle("Підтвердження видалення аномалій")
    dialog.setFixedSize(400, 300)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Виявлені аномалії. Оберіть, які видалити:"))
    
    listbox = QListWidget()
    listbox.setSelectionMode(QListWidget.MultiSelection)
    for info in outlier_info:
        listbox.addItem(info)
    layout.addWidget(listbox)
    
    selected_indices = []
    
    def confirm():
        selected_items = listbox.selectedItems()
        selected_indices.extend([outlier_indices[listbox.row(item)] for item in selected_items])
        dialog.accept()
    
    def cancel():
        dialog.reject()
    
    confirm_btn = QPushButton("Видалити обрані")
    confirm_btn.clicked.connect(confirm)
    layout.addWidget(confirm_btn)
    cancel_btn = QPushButton("Скасувати")
    cancel_btn.clicked.connect(cancel)
    layout.addWidget(cancel_btn)
    
    dialog.exec_()
    
    if selected_indices:
        original_count = len(values)
        values = np.delete(values, selected_indices)
        removed_count = original_count - len(values)
        update_statistics()
        update_characteristics()
        update_data_box()
        update_histogram()
        QMessageBox.information(gui, "Видалення викидів", f"Видалено {removed_count} викидів")
    else:
        QMessageBox.information(gui, "Видалення викидів", "Жодних викидів не видалено")

def apply_bounds():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для застосування границь")
        return
    try:
        lower = float(gui.lower_entry.text())
        upper = float(gui.upper_entry.text())
        if lower >= upper:
            QMessageBox.critical(gui, "Помилка", "Нижня границя має бути менше верхньої")
            return
        mask = (values >= lower) & (values <= upper)
        if not np.any(mask):
            QMessageBox.critical(gui, "Помилка", "За вказаними границями немає даних")
            return
        values = values[mask]
        update_statistics()
        update_characteristics()
        update_data_box()
        QMessageBox.information(gui, "Границі", f"Застосовано границі: [{lower:.4f}, {upper:.4f}]")
    except ValueError:
        QMessageBox.critical(gui, "Помилка", "Введіть числові значення для границь")

def reset_data():
    global values, original_values
    if len(original_values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає початкових даних для скидання. Завантажте дані спочатку.")
        return
    values = original_values.copy()
    min_val, max_val = np.min(values), np.max(values)
    gui.lower_entry.setText(str(min_val))
    gui.upper_entry.setText(str(max_val))
    update_statistics()
    update_characteristics()
    update_data_box()
    gui.distro_ax.clear()
    gui.distro_info_label.setText("")
    gui.distro_canvas.draw()
    QMessageBox.information(gui, "Скидання", "Дані повернуто до початкового стану")

def initialize_logic(window):
    global gui
    gui = window
    gui.load_button.clicked.connect(load_data)
    gui.data_box.textChanged.connect(update_from_data_box)
    gui.update_button.clicked.connect(update_histogram)
    gui.apply_bounds_btn.clicked.connect(apply_bounds)
    gui.standardize_btn.clicked.connect(standardize_data)
    gui.log_btn.clicked.connect(log_transform)
    gui.shift_btn.clicked.connect(shift_data)
    gui.outliers_btn.clicked.connect(remove_outliers)
    gui.reset_btn.clicked.connect(reset_data)
    gui.plot_distro_btn.clicked.connect(plot_distribution)
    gui.save_btn.clicked.connect(save_data)