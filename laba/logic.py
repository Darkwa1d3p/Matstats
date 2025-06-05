import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, t, chi2, sem, kstest, expon, weibull_min, uniform, lognorm, ttest_1samp
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QInputDialog, QDialog, QListWidget,
                             QPushButton, QVBoxLayout, QTableWidgetItem, QLabel, QTextEdit)
import matplotlib.pyplot as plt

# Глобальні змінні
values = np.array([])
original_values = np.array([])
gui = None
def plot_frequency_polygon():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для побудови полігону частот")
        return
    
    gui.distro_ax.clear()
    gui.distro_info_label.setText("")
    
    bin_count = gui.bin_entry.value() or calculate_bin_count(len(values))
    hist, bins = np.histogram(values, bins=bin_count, density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    gui.distro_ax.plot(bin_centers, hist, 'b-', marker='o', label='Полігон частот')
    gui.distro_ax.set_title('Полігон частот')
    gui.distro_ax.set_xlabel('Час очікування (хв)')
    gui.distro_ax.set_ylabel('Частота')
    gui.distro_ax.legend()
    gui.distro_ax.grid(True, linestyle='--', alpha=0.7)
    gui.distro_canvas.draw()
def test_h0_mean():
    """Перевірка гіпотези H0: μ=значення за допомогою одновибіркового t-тесту."""
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для перевірки гіпотези")
        return
    mu_0, ok = QInputDialog.getDouble(gui, "Гіпотеза про середнє", "Введіть значення гіпотези H0 (μ):", 0.0, -1000.0, 1000.0, 2)
    if not ok:
        return
    t_stat, p_value = ttest_1samp(values, mu_0)
    confidence = gui.confidence_entry.value() / 100
    alpha = 1 - confidence
    conclusion = ("H0 відхиляється: середнє значимо відрізняється від введеного значення" 
                  if p_value < alpha 
                  else "H0 не відхиляється: немає підстав вважати, що середнє відрізняється від введеного значення")
    mean = np.mean(values)
    QMessageBox.information(
        gui, 
        f"T-тест для H0: μ={mu_0}",
        f"Середнє вибірки: {mean:.4f}\n"
        f"Гіпотетичне середнє (μ0): {mu_0:.4f}\n"
        f"T-статистика: {t_stat:.4f}\n"
        f"p-значення: {p_value:.4f}\n"
        f"Рівень значущості (α): {alpha:.4f}\n"
        f"Висновок: {conclusion}"
    )
def generate_weibull_sample(size, lambda_param, k_param):
    """Генерація вибірки з розподілу Вейбулла."""
    U = np.random.uniform(0, 1, size)
    return lambda_param * (-np.log(U)) ** (1 / k_param)

def show_variation_series():
    """Відображення варіаційного ряду (відсортованих даних)."""
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для відображення варіаційного ряду")
        return
    
    sorted_values = np.sort(values)
    dialog = QDialog(gui)
    dialog.setWindowTitle("Варіаційний ряд")
    dialog.setFixedSize(400, 300)
    layout = QVBoxLayout(dialog)
    
    layout.addWidget(QLabel("Відсортовані дані:"))
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setPlainText(', '.join([f"{val:.4f}" for val in sorted_values]))
    layout.addWidget(text_edit)
    
    close_btn = QPushButton("Закрити")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn)
    
    dialog.exec_()
def simulate_and_test_weibull(size, lambda_true, k_true, num_experiments=200):
    """Моделювання та тестування вибірки Вейбулла з T-тестом."""
    lambda_estimates = []
    k_estimates = []
    
    for _ in range(num_experiments):
        sample = generate_weibull_sample(size, lambda_true, k_true)
        k_est, loc, lambda_est = weibull_min.fit(sample, floc=0)
        lambda_estimates.append(lambda_est)
        k_estimates.append(k_est)
    
    t_stat_lambda, p_value_lambda = ttest_1samp(lambda_estimates, lambda_true)
    t_stat_k, p_value_k = ttest_1samp(k_estimates, k_true)
    
    return {
        'lambda_mean': np.mean(lambda_estimates),
        'lambda_std': np.std(lambda_estimates),
        'lambda_t_stat': t_stat_lambda,
        'lambda_p_value': p_value_lambda,
        'k_mean': np.mean(k_estimates),
        'k_std': np.std(k_estimates),
        'k_t_stat': t_stat_k,
        'k_p_value': p_value_k
    }
def generate_normal_data():
    """Генерація синтетичних даних із нормального розподілу з розміром від 50 до 1000."""
    global values, original_values  # Оголосити глобальні змінні на початку
    mu, ok1 = QInputDialog.getDouble(gui, "Параметр нормального розподілу", "Введіть середнє (μ):", 0.0, -1000.0, 1000.0, 2)
    if not ok1:
        return
    sigma, ok2 = QInputDialog.getDouble(gui, "Параметр нормального розподілу", "Введіть стандартне відхилення (σ):", 1.0, 0.1, 100.0, 2)
    if not ok2:
        return
    size, ok3 = QInputDialog.getInt(gui, "Розмір вибірки", "Введіть розмір вибірки (50-1000):", 50, 50, 1000)
    if not ok3:
        return
    values = np.random.normal(mu, sigma, size)
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
    QMessageBox.information(gui, "Генерація", f"Згенеровано вибірку з нормального розподілу (μ={mu}, σ={sigma}, розмір={size})")
    update_histogram()


def simulate_and_test_weibull(size, lambda_true, k_true, num_experiments=200):
    """Моделювання та тестування вибірки Вейбулла з T-тестом."""
    lambda_estimates = []
    k_estimates = []
    
    for _ in range(num_experiments):
        sample = generate_weibull_sample(size, lambda_true, k_true)
        k_est, loc, lambda_est = weibull_min.fit(sample, floc=0)
        lambda_estimates.append(lambda_est)
        k_estimates.append(k_est)
    
    t_stat_lambda, p_value_lambda = ttest_1samp(lambda_estimates, lambda_true)
    t_stat_k, p_value_k = ttest_1samp(k_estimates, k_true)
    
    return {
        'lambda_mean': np.mean(lambda_estimates),
        'lambda_std': np.std(lambda_estimates),
        'lambda_t_stat': t_stat_lambda,
        'lambda_p_value': p_value_lambda,
        'k_mean': np.mean(k_estimates),
        'k_std': np.std(k_estimates),
        'k_t_stat': t_stat_k,
        'k_p_value': p_value_k
    }

def generate_normal_sample(size, mu, sigma):
    """Генерація вибірки з нормального розподілу."""
    return np.random.normal(mu, sigma, size)

def simulate_and_test_normal(size, mu_true, sigma_true, num_experiments=200):
    """Моделювання та тестування вибірки нормального розподілу з T-тестом."""
    mu_estimates = []
    sigma_estimates = []
    
    for _ in range(num_experiments):
        sample = generate_normal_sample(size, mu_true, sigma_true)
        mu_est = np.mean(sample)
        sigma_est = np.std(sample, ddof=1)
        mu_estimates.append(mu_est)
        sigma_estimates.append(sigma_est)
    
    t_stat_mu, p_value_mu = ttest_1samp(mu_estimates, mu_true)
    t_stat_sigma, p_value_sigma = ttest_1samp(sigma_estimates, sigma_true)
    
    return {
        'mu_mean': np.mean(mu_estimates),
        'mu_std': np.std(mu_estimates),
        'mu_t_stat': t_stat_mu,
        'mu_p_value': p_value_mu,
        'sigma_mean': np.mean(sigma_estimates),
        'sigma_std': np.std(sigma_estimates),
        'sigma_t_stat': t_stat_sigma,
        'sigma_p_value': p_value_sigma
    }

def generate_exponential_sample(size, lambda_param):
    """Генерація вибірки з експоненціального розподілу."""
    return np.random.exponential(1/lambda_param, size)

def simulate_and_test_exponential(size, lambda_true, num_experiments=200):
    """Моделювання та тестування вибірки експоненціального розподілу з T-тестом."""
    lambda_estimates = []
    
    for _ in range(num_experiments):
        sample = generate_exponential_sample(size, lambda_true)
        lambda_est = 1 / np.mean(sample)
        lambda_estimates.append(lambda_est)
    
    t_stat_lambda, p_value_lambda = ttest_1samp(lambda_estimates, lambda_true)
    
    return {
        'lambda_mean': np.mean(lambda_estimates),
        'lambda_std': np.std(lambda_estimates),
        'lambda_t_stat': t_stat_lambda,
        'lambda_p_value': p_value_lambda
    }

def generate_uniform_sample(size, a, b):
    """Генерація вибірки з уніформного розподілу."""
    return np.random.uniform(a, b, size)

def simulate_and_test_uniform(size, a_true, b_true, num_experiments=200):
    """Моделювання та тестування вибірки уніформного розподілу з T-тестом."""
    a_estimates = []
    b_estimates = []
    
    for _ in range(num_experiments):
        sample = generate_uniform_sample(size, a_true, b_true)
        a_est = np.min(sample)
        b_est = np.max(sample)
        a_estimates.append(a_est)
        b_estimates.append(b_est)
    
    t_stat_a, p_value_a = ttest_1samp(a_estimates, a_true)
    t_stat_b, p_value_b = ttest_1samp(b_estimates, b_true)
    
    return {
        'a_mean': np.mean(a_estimates),
        'a_std': np.std(a_estimates),
        'a_t_stat': t_stat_a,
        'a_p_value': p_value_a,
        'b_mean': np.mean(b_estimates),
        'b_std': np.std(b_estimates),
        'b_t_stat': t_stat_b,
        'b_p_value': p_value_b
    }

def generate_lognormal_sample(size, mu, sigma):
    """Генерація вибірки з лог-нормального розподілу."""
    return np.random.lognormal(mu, sigma, size)

def simulate_and_test_lognormal(size, mu_true, sigma_true, num_experiments=200):
    """Моделювання та тестування вибірки лог-нормального розподілу з T-тестом."""
    mu_estimates = []
    sigma_estimates = []
    
    for _ in range(num_experiments):
        sample = generate_lognormal_sample(size, mu_true, sigma_true)
        log_sample = np.log(sample)
        mu_est = np.mean(log_sample)
        sigma_est = np.std(log_sample, ddof=1)
        mu_estimates.append(mu_est)
        sigma_estimates.append(sigma_est)
    
    t_stat_mu, p_value_mu = ttest_1samp(mu_estimates, mu_true)
    t_stat_sigma, p_value_sigma = ttest_1samp(sigma_estimates, sigma_true)
    
    return {
        'mu_mean': np.mean(mu_estimates),
        'mu_std': np.std(mu_estimates),
        'mu_t_stat': t_stat_mu,
        'mu_p_value': p_value_mu,
        'sigma_mean': np.mean(sigma_estimates),
        'sigma_std': np.std(sigma_estimates),
        'sigma_t_stat': t_stat_sigma,
        'sigma_p_value': p_value_sigma
    }

def generate_weibull_data():
    """Генерація синтетичних даних із розподілу Вейбулла."""
    global values, original_values
    lambda_param, ok1 = QInputDialog.getDouble(gui, "Параметр Вейбулла", "Введіть параметр масштабу (λ):", 2.0, 0.1, 100.0, 2)
    if not ok1:
        return
    k_param, ok2 = QInputDialog.getDouble(gui, "Параметр Вейбулла", "Введіть параметр форми (k):", 1.5, 0.1, 10.0, 2)
    if not ok2:
        return
    size, ok3 = QInputDialog.getInt(gui, "Розмір вибірки", "Введіть розмір вибірки:", 1000, 1, 10000)
    if not ok3:
        return
    values = generate_weibull_sample(size, lambda_param, k_param)
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
    QMessageBox.information(gui, "Генерація", f"Згенеровано вибірку з розподілу Вейбулла (λ={lambda_param}, k={k_param})")
    update_histogram()
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
        gui.char_table.clearContents()
        gui.char_table.setRowCount(10)
        gui.char_table.setHorizontalHeaderLabels(["Характеристика", "Зсунена", "Незсунена"])
        QMessageBox.warning(gui, "Попередження", "Немає даних для обчислення характеристик. Завантажте або згенеруйте дані.")
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
    gui.data_box.setPlainText(', '.join([f"{val:.4f}" for val in values]) if len(values) > 0 else "")
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

def calculate_bin_count(N):
    """Визначає кількість класів для гістограми."""
    if N <= 100:
        bin_count = int(np.sqrt(N))
    else:
        bin_count = int(np.cbrt(N))
    if bin_count % 2 == 0:
        bin_count -= 1
    return max(1, bin_count)

def update_histogram():
    global values
    if len(values) == 0:
        QMessageBox.warning(gui, "Попередження", "Немає даних для побудови гістограми. Завантажте або згенеруйте дані.")
        gui.hist_ax.clear()
        gui.hist_canvas.draw()
        gui.info_label.setText("Кількість класів: -\nКрок розбиття: -\nРозмах: -\nКількість даних: 0")
        return
    bin_count = gui.bin_entry.value()
    if bin_count == 0:
        bin_count = calculate_bin_count(len(values))
    
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
    bin_width = range_val / bin_count if bin_count > 0 else 0
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
        n_bins = calculate_bin_count(len(values))
    
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
    
    elif distro_type == "Нормальний (гістограма)":
        mean = np.mean(values)
        std = np.std(values)
        
        n_bins = gui.bin_entry.value() if gui.bin_entry.value() > 0 else int(np.sqrt(len(values)))
        hist, bins, _ = gui.distro_ax.hist(values, bins=n_bins, color='green', alpha=0.7, edgecolor='black', density=True, label='Гістограма')
        
        x = np.linspace(min(values), max(values), 100)
        normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        gui.distro_ax.plot(x, normal_pdf, 'r-', label=f'Нормальний (μ={mean:.4f}, σ={std:.4f})')
        
        gui.distro_ax.set_title('Гістограма та щільність нормального розподілу')
        gui.distro_ax.set_xlabel('Час очікування (хв)')
        gui.distro_ax.set_ylabel('Щільність')
        gui.distro_ax.legend()
        
        ks_statistic, ks_pvalue = kstest(values, 'norm', args=(mean, std))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає нормальному" if ks_statistic < critical_value else "Розподіл не відповідає нормальному"
        
        ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                  f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                  f"Висновок: {conclusion}\n")
        
        gui.distro_info_label.setText(ks_text)
    
    elif distro_type == "Експоненціальний":
        if np.any(values < 0):
            QMessageBox.critical(gui, "Помилка", "Експоненціальний розподіл можливий лише для невід’ємних значень")
            return
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        empirical_probs = np.arange(1, n + 1) / (n + 1)
        y_values = -np.log(1 - empirical_probs)
        
        y_max = np.max(y_values)
        if y_max > 0:
            y_values_normalized = y_values / y_max
        else:
            y_values_normalized = y_values
        
        mean = np.mean(values)
        if mean == 0:
            QMessageBox.critical(gui, "Помилка", "Середнє значення дорівнює нулю. Неможливо оцінити параметр.")
            return
        lambda_param = 1 / mean
        
        gui.distro_ax.scatter(sorted_values, y_values_normalized, color='green', label='Дані', s=50)
        
        x_theor = np.linspace(0, np.max(sorted_values) * 1.2, 100)
        y_theor = lambda_param * x_theor
        if y_max > 0:
            y_theor_normalized = y_theor / y_max
        else:
            y_theor_normalized = y_theor
        gui.distro_ax.plot(x_theor, y_theor_normalized, color='blue', linestyle='--', label=f'Експоненціальний розподіл (λ={lambda_param:.4f})')
        
        x_min, x_max = np.min(values), np.max(values)
        x_range = x_max - x_min if x_max != x_min else 1
        x_margin = 0.1 * x_range
        x_lower = max(0, x_min - x_margin)
        x_upper = x_max + x_margin
        gui.distro_ax.set_xlim(x_lower, x_upper)
        gui.distro_ax.set_ylim(0, 1.1)
        
        gui.distro_ax.set_title('Імовірнісна сітка експоненціального розподілу')
        gui.distro_ax.set_xlabel('Значення (Час очікування, хв)')
        gui.distro_ax.set_ylabel('Нормалізоване -ln(1 - F(x))')
        ks_statistic, ks_pvalue = kstest(values, 'expon', args=(0, mean))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає експоненціальному" if ks_statistic < critical_value else "Розподіл не відповідає експоненціальному"
    
    elif distro_type == "Експоненціальний (гістограма)":
        if np.any(values < 0):
            QMessageBox.critical(gui, "Помилка", "Експоненціальний розподіл можливий лише для невід’ємних значень")
            return
        
        mean = np.mean(values)
        if mean == 0:
            QMessageBox.critical(gui, "Помилка", "Середнє значення дорівнює нулю. Неможливо оцінити параметр.")
            return
        lambda_param = 1 / mean
        
        n_bins = gui.bin_entry.value() if gui.bin_entry.value() > 0 else int(np.sqrt(len(values)))
        hist, bins, _ = gui.distro_ax.hist(values, bins=n_bins, color='green', alpha=0.7, edgecolor='black', density=True, label='Гістограма')
        
        x = np.linspace(min(values), max(values), 100)
        expon_pdf = lambda_param * np.exp(-lambda_param * x)
        gui.distro_ax.plot(x, expon_pdf, 'r-', label=f'Експоненціальний (λ={lambda_param:.4f})')
        
        gui.distro_ax.set_title('Гістограма та щільність експоненціального розподілу')
        gui.distro_ax.set_xlabel('Час очікування (хв)')
        gui.distro_ax.set_ylabel('Щільність')
        gui.distro_ax.legend()
        
        ks_statistic, ks_pvalue = kstest(values, 'expon', args=(0, mean))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає експоненціальному" if ks_statistic < critical_value else "Розподіл не відповідає експоненціальному"
        
        ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                  f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                  f"Висновок: {conclusion}\n")
        
        gui.distro_info_label.setText(ks_text)
    
    elif distro_type == "Вейбулла":
        if np.any(values < 0):
            QMessageBox.critical(gui, "Помилка", "Розподіл Вейбулла можливий лише для невід’ємних значень")
            return
        
        k_est, loc, lambda_est = weibull_min.fit(values, floc=0)
        
        gui.distro_ax.clear()
        
        n_bins = gui.bin_entry.value() if gui.bin_entry.value() > 0 else int(np.sqrt(len(values)))
        hist, bins, _ = gui.distro_ax.hist(values, bins=n_bins, color='green', alpha=0.7, edgecolor='black', density=True, label='Гістограма')
        
        x = np.linspace(min(values), max(values), 100)
        weibull_pdf = (k_est / lambda_est) * (x / lambda_est) ** (k_est - 1) * np.exp(-(x / lambda_est) ** k_est)
        gui.distro_ax.plot(x, weibull_pdf, 'r-', label=f'Вейбулла (λ={lambda_est:.4f}, k={k_est:.4f})')
        
        gui.distro_ax.set_title('Гістограма та щільність розподілу Вейбулла')
        gui.distro_ax.set_xlabel('Час очікування (хв)')
        gui.distro_ax.set_ylabel('Щільність')
        gui.distro_ax.legend()
        
        ks_statistic, ks_pvalue = kstest(values, 'weibull_min', args=(k_est, loc, lambda_est))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає Вейбулла" if ks_statistic < critical_value else "Розподіл не відповідає Вейбулла"
        
        sizes = [20, 50, 100, 400, 1000, 2000, 5000]
        t_test_results = []
        for size in sizes:
            result = simulate_and_test_weibull(size, lambda_est, k_est)
            t_test_results.append({
                'size': size,
                'lambda_mean': result['lambda_mean'],
                'lambda_std': result['lambda_std'],
                'lambda_t_stat': result['lambda_t_stat'],
                'lambda_p_value': result['lambda_p_value'],
                'k_mean': result['k_mean'],
                'k_std': result['k_std'],
                'k_t_stat': result['k_t_stat'],
                'k_p_value': result['k_p_value']
            })
        
        ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                  f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                  f"Висновок: {conclusion}\n\n"
                  f"T-тест для оцінки параметрів Вейбулла:\n")
        for res in t_test_results:
            ks_text += (f"\nРозмір вибірки: {res['size']}\n"
                       f"λ: середнє={res['lambda_mean']:.4f}, std={res['lambda_std']:.4f}, "
                       f"T-статистика={res['lambda_t_stat']:.4f}, p-value={res['lambda_p_value']:.4f}\n"
                       f"k: середнє={res['k_mean']:.4f}, std={res['k_std']:.4f}, "
                       f"T-статистика={res['k_t_stat']:.4f}, p-value={res['k_p_value']:.4f}\n")
        
        gui.distro_info_label.setText(ks_text)
    
    elif distro_type == "Уніформний (гістограма)":
        a = np.min(values)
        b = np.max(values)
        if a >= b:
            QMessageBox.critical(gui, "Помилка", "Мінімальне значення має бути меншим за максимальне.")
            return
        
        n_bins = gui.bin_entry.value() if gui.bin_entry.value() > 0 else int(np.sqrt(len(values)))
        hist, bins, _ = gui.distro_ax.hist(values, bins=n_bins, color='green', alpha=0.7, edgecolor='black', density=True, label='Гістограма')
        
        x = np.linspace(min(values), max(values), 100)
        uniform_pdf = np.ones_like(x) / (b - a)
        uniform_pdf[(x < a) | (x > b)] = 0
        gui.distro_ax.plot(x, uniform_pdf, 'r-', label=f'Уніформний (a={a:.4f}, b={b:.4f})')
        
        gui.distro_ax.set_title('Гістограма та щільність уніформного розподілу')
        gui.distro_ax.set_xlabel('Час очікування (хв)')
        gui.distro_ax.set_ylabel('Щільність')
        gui.distro_ax.legend()
        
        ks_statistic, ks_pvalue = kstest(values, 'uniform', args=(a, b - a))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає уніформному" if ks_statistic < critical_value else "Розподіл не відповідає уніформному"
        
        ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                   f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                   f"Висновок: {conclusion}\n")
        
        gui.distro_info_label.setText(ks_text)
    
    elif distro_type == "Лог-нормальний (гістограма)":
        if np.any(values <= 0):
            QMessageBox.critical(gui, "Помилка", "Лог-нормальний розподіл можливий лише для додатних значень")
            return
        
        log_values = np.log(values)
        mu = np.mean(log_values)
        sigma = np.std(log_values)
        
        n_bins = gui.bin_entry.value() if gui.bin_entry.value() > 0 else int(np.sqrt(len(values)))
        hist, bins, _ = gui.distro_ax.hist(values, bins=n_bins, color='green', alpha=0.7, edgecolor='black', density=True)
        
        x = np.linspace(min(values), max(values), 100)
        lognorm_pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))
        gui.distro_ax.plot(x, lognorm_pdf, 'r-', label=f'Лог-нормальний (μ={mu:.4f}, σ={sigma:.4f})')
        
        gui.distro_ax.set_title('Гістограма та щільність лог-нормального розподілу')
        gui.distro_ax.set_xlabel('Час очікування (хв)')
        gui.distro_ax.set_ylabel('Щільність')
        gui.distro_ax.legend()
        
        ks_statistic, ks_pvalue = kstest(values, 'lognorm', args=(sigma, 0, np.exp(mu)))
        critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
        conclusion = "Розподіл відповідає лог-нормальному" if ks_statistic < critical_value else "Розподіл не відповідає лог-нормальному"
        
        ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                   f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                   f"Висновок: {conclusion}\n")
        
        gui.distro_info_label.setText(ks_text)
    
    gui.distro_ax.legend()
    gui.distro_ax.grid(True, linestyle='--', alpha=0.7)
    gui.distro_canvas.draw()

def run_ttest():
    """Запуск t-тесту для обраного розподілу з введенням параметрів."""
    distro_type = gui.distro_combo_ttest.currentText()
    t_test_results = []
    sizes = [20, 50, 100, 400, 1000, 2000, 5000]
    
    if distro_type == "Нормальний":
        mu_true, ok1 = QInputDialog.getDouble(gui, "Параметр нормального розподілу", "Введіть середнє (μ):", 0.0, -1000.0, 1000.0, 2)
        if not ok1:
            return
        sigma_true, ok2 = QInputDialog.getDouble(gui, "Параметр нормального розподілу", "Введіть стандартне відхилення (σ):", 1.0, 0.1, 100.0, 2)
        if not ok2:
            return
        for size in sizes:
            result = simulate_and_test_normal(size, mu_true, sigma_true)
            t_test_results.append({
                'size': size,
                'param1_mean': result['mu_mean'],
                'param1_std': result['mu_std'],
                'param1_t_stat': result['mu_t_stat'],
                'param1_p_value': result['mu_p_value'],
                'param2_mean': result['sigma_mean'],
                'param2_std': result['sigma_std'],
                'param2_t_stat': result['sigma_t_stat'],
                'param2_p_value': result['sigma_p_value'],
                'param1_name': 'μ',
                'param2_name': 'σ'
            })

    elif distro_type == "Експоненціальний":
        lambda_true, ok1 = QInputDialog.getDouble(gui, "Параметр експоненціального розподілу", "Введіть параметр (λ):", 1.0, 0.1, 100.0, 2)
        if not ok1:
            return
        for size in sizes:
            result = simulate_and_test_exponential(size, lambda_true)
            t_test_results.append({
                'size': size,
                'param1_mean': result['lambda_mean'],
                'param1_std': result['lambda_std'],
                'param1_t_stat': result['lambda_t_stat'],
                'param1_p_value': result['lambda_p_value'],
                'param2_mean': None,
                'param2_std': None,
                'param2_t_stat': None,
                'param2_p_value': None,
                'param1_name': 'λ',
                'param2_name': None
            })

    elif distro_type == "Вейбулла":
        lambda_true, ok1 = QInputDialog.getDouble(gui, "Параметр Вейбулла", "Введіть параметр масштабу (λ):", 2.0, 0.1, 100.0, 2)
        if not ok1:
            return
        k_true, ok2 = QInputDialog.getDouble(gui, "Параметр Вейбулла", "Введіть параметр форми (k):", 1.5, 0.1, 10.0, 2)
        if not ok2:
            return
        for size in sizes:
            result = simulate_and_test_weibull(size, lambda_true, k_true)
            t_test_results.append({
                'size': size,
                'param1_mean': result['lambda_mean'],
                'param1_std': result['lambda_std'],
                'param1_t_stat': result['lambda_t_stat'],
                'param1_p_value': result['lambda_p_value'],
                'param2_mean': result['k_mean'],
                'param2_std': result['k_std'],
                'param2_t_stat': result['k_t_stat'],
                'param2_p_value': result['k_p_value'],
                'param1_name': 'λ',
                'param2_name': 'k'
            })

    elif distro_type == "Уніформний":
        a_true, ok1 = QInputDialog.getDouble(gui, "Параметр уніформного розподілу", "Введіть нижню межу (a):", 0.0, -1000.0, 1000.0, 2)
        if not ok1:
            return
        b_true, ok2 = QInputDialog.getDouble(gui, "Параметр уніформного розподілу", "Введіть верхню межу (b):", 1.0, a_true + 0.1, 1000.0, 2)
        if not ok2:
            return
        for size in sizes:
            result = simulate_and_test_uniform(size, a_true, b_true)
            t_test_results.append({
                'size': size,
                'param1_mean': result['a_mean'],
                'param1_std': result['a_std'],
                'param1_t_stat': result['a_t_stat'],
                'param1_p_value': result['a_p_value'],
                'param2_mean': result['b_mean'],
                'param2_std': result['b_std'],
                'param2_t_stat': result['b_t_stat'],
                'param2_p_value': result['b_p_value'],
                'param1_name': 'a',
                'param2_name': 'b'
            })

    elif distro_type == "Лог-нормальний":
        mu_true, ok1 = QInputDialog.getDouble(gui, "Параметр лог-нормального розподілу", "Введіть середнє (μ):", 0.0, -1000.0, 1000.0, 2)
        if not ok1:
            return
        sigma_true, ok2 = QInputDialog.getDouble(gui, "Параметр лог-нормального розподілу", "Введіть стандартне відхилення (σ):", 1.0, 0.1, 100.0, 2)
        if not ok2:
            return
        for size in sizes:
            result = simulate_and_test_lognormal(size, mu_true, sigma_true)
            t_test_results.append({
                'size': size,
                'param1_mean': result['mu_mean'],
                'param1_std': result['mu_std'],
                'param1_t_stat': result['mu_t_stat'],
                'param1_p_value': result['mu_p_value'],
                'param2_mean': result['sigma_mean'],
                'param2_std': result['sigma_std'],
                'param2_t_stat': result['sigma_t_stat'],
                'param2_p_value': result['sigma_p_value'],
                'param1_name': 'μ',
                'param2_name': 'σ'
            })

    gui.char_table.clearContents()
    gui.char_table.setRowCount(len(sizes) * (2 if t_test_results[0]['param2_name'] else 1))
    gui.char_table.setColumnCount(6)
    gui.char_table.setHorizontalHeaderLabels(["Розмір вибірки", "Середнє T", "СКВ T", "Критичне T", "α", "Параметр"])
    
    for row, res in enumerate(t_test_results):
        gui.char_table.setItem(row * 2 if res['param2_name'] else row, 0, QTableWidgetItem(str(res['size'])))
        gui.char_table.setItem(row * 2 if res['param2_name'] else row, 1, QTableWidgetItem(f"{res['param1_mean']:.4f} ({res['param1_name']})"))
        gui.char_table.setItem(row * 2 if res['param2_name'] else row, 2, QTableWidgetItem(f"{res['param1_std']:.4f}"))
        gui.char_table.setItem(row * 2 if res['param2_name'] else row, 3, QTableWidgetItem(f"{res['param1_t_stat']:.4f}"))
        gui.char_table.setItem(row * 2 if res['param2_name'] else row, 4, QTableWidgetItem(f"{res['param1_p_value']:.4f}"))
        gui.char_table.setItem(row * 2 if res['param2_name'] else row, 5, QTableWidgetItem(res['param1_name']))

        if res['param2_name']:
            gui.char_table.insertRow(row * 2 + 1)
            gui.char_table.setItem(row * 2 + 1, 0, QTableWidgetItem(str(res['size'])))
            gui.char_table.setItem(row * 2 + 1, 1, QTableWidgetItem(f"{res['param2_mean']:.4f} ({res['param2_name']})"))
            gui.char_table.setItem(row * 2 + 1, 2, QTableWidgetItem(f"{res['param2_std']:.4f}"))
            gui.char_table.setItem(row * 2 + 1, 3, QTableWidgetItem(f"{res['param2_t_stat']:.4f}"))
            gui.char_table.setItem(row * 2 + 1, 4, QTableWidgetItem(f"{res['param2_p_value']:.4f}"))
            gui.char_table.setItem(row * 2 + 1, 5, QTableWidgetItem(res['param2_name']))

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
    update_characteristics()  # Додаємо виклик
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
    gui.run_ttest_btn.clicked.connect(run_ttest)
    gui.variation_series_btn.clicked.connect(show_variation_series)
    gui.generate_normal_btn.clicked.connect(generate_normal_data)
    gui.test_h0_btn.clicked.connect(test_h0_mean)
    gui.plot_polygon_btn.clicked.connect(plot_frequency_polygon)