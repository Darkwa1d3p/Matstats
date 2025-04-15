import numpy as np
import pandas as pd
from scipy.stats import skew, t, expon, norm, kurtosis
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem

class DataProcessor:
    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        try:
            if file_path.endswith('.txt'):
                delimiters = [',', '\t', ';', '\\s+']
                self.data = None
                for sep in delimiters:
                    try:
                        self.data = pd.read_csv(file_path, sep=sep, header=0, encoding='utf-8')
                        if self.data.columns.astype(str).str.isnumeric().all():
                            self.data = pd.read_csv(file_path, sep=sep, header=None, encoding='utf-8')
                            self.data.columns = [f'Column_{i}' for i in range(len(self.data.columns))]
                        break
                    except Exception:
                        continue

                if self.data is None:
                    raise ValueError("Не вдалося визначити роздільник у файлі .txt. Спробуйте використати роздільники: пробіл, табуляція, кома або крапка з комою.")

                for col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        self.data[col] = self.data[col].astype(str).str.replace(',', '.')
                        self.data[col] = pd.to_numeric(self.data[col], errors='ignore')

            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Непідтримуваний формат файлу")

            print("Зчитаний DataFrame:")
            print(self.data.head())
            print("Інформація про DataFrame:")
            print(self.data.info())

            numeric_cols = self.get_numeric_columns()
            if not numeric_cols:
                raise ValueError("У файлі відсутні числові стовпці для аналізу! Переконайтеся, що файл містить числові дані.")

        except Exception as e:
            raise Exception(f"Помилка при завантаженні файлу: {str(e)}")

    def get_numeric_columns(self):
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def fill_missing_values(self):
        if self.data is None or self.data.empty:
            QMessageBox.warning(None, "Помилка", "Дані відсутні або файл порожній!")
            return False
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            QMessageBox.warning(None, "Помилка", "Немає числових даних для заповнення!")
            return False
        
        missing_count = self.data[numeric_cols].isna().sum().sum()
        if missing_count == 0:
            QMessageBox.information(None, "Інформація", "У даних немає пропущених значень.")
            return False
        
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        QMessageBox.information(None, "Успіх", f"Пропущені значення ({missing_count}) замінено середнім значенням!")
        return True

    def drop_missing_values(self):
        if self.data is None or self.data.empty:
            QMessageBox.warning(None, "Помилка", "Дані відсутні або файл порожній!")
            return False
        
        rows_with_missing = self.data.shape[0] - self.data.dropna().shape[0]
        if rows_with_missing == 0:
            QMessageBox.information(None, "Інформація", "У даних немає пропущених значень.")
            return False
        
        self.data.dropna(inplace=True)
        QMessageBox.information(None, "Успіх", f"Видалено {rows_with_missing} рядків з пропущеними значеннями!")
        return True

    def sturges_rule(self, n):
        import math
        return max(1, math.ceil(1 + 3.322 * math.log10(n)))

    def sqrt_rule(self, n):
        import math
        return max(1, math.ceil(math.sqrt(n)))

    def scott_rule(self, data):
        import numpy as np
        if len(data) <= 1 or np.std(data) == 0:
            return 10
        h = 3.5 * np.std(data) / (len(data) ** (1/3))
        data_range = np.max(data) - np.min(data)
        if h <= 0 or data_range <= 0:
            return 10
        return max(1, min(100, int(data_range / h)))

    def analyze_data(self, column, log_transform, standardize, shift, shift_value, shift_direction, shift_repeats, bins_method, custom_bins, canvas):
        if self.data is None or self.data.empty:
            QMessageBox.warning(None, "Помилка", "Дані відсутні або файл порожній!")
            return None

        if not column or column not in self.data.columns:
            QMessageBox.warning(None, "Помилка", "Стовпець не вибрано або не існує!")
            return None

        data = self.data[column].dropna().values
        if len(data) == 0:
            QMessageBox.warning(None, "Помилка", "Немає даних для аналізу!")
            return None

        # Зберігаємо незсунені дані для порівняння
        original_data = data.copy()
        transformed_data = data.copy()
        transformation_info = []

        if log_transform:
            if (data < 0).any():
                QMessageBox.warning(None, "Помилка", "Дані містять від'ємні значення для логарифмування!")
                return None
            transformed_data = np.log1p(transformed_data)
            transformation_info.append("Логарифмування (log(x+1))")

        if standardize:
            if np.std(transformed_data) == 0:
                QMessageBox.warning(None, "Помилка", "Стандартне відхилення дорівнює нулю!")
                return None
            mean_val = np.mean(transformed_data)
            std_val = np.std(transformed_data)
            transformed_data = (transformed_data - mean_val) / std_val
            transformation_info.append(f"Стандартизація (z = (x - {mean_val:.4f}) / {std_val:.4f})")

        if shift:
            total_shift = shift_value * shift_repeats
            if shift_direction == "Додати":
                transformed_data = transformed_data + total_shift
                transformation_info.append(f"Зсув (x + {total_shift:.4f}) [{shift_repeats} раз(и)]")
            else:
                transformed_data = transformed_data - total_shift
                transformation_info.append(f"Зсув (x - {total_shift:.4f}) [{shift_repeats} раз(и)]")

        if bins_method == "Користувацька":
            bins = custom_bins
        elif bins_method == "Стерджеса":
            bins = self.sturges_rule(len(transformed_data))
        elif bins_method == "Квадратного кореня":
            bins = self.sqrt_rule(len(transformed_data))
        elif bins_method == "Скотта":
            bins = self.scott_rule(transformed_data)
        else:
            bins = min(10, len(set(transformed_data)))

        self._plot_histogram(canvas, transformed_data, bins)
        
        # Обчислюємо статистику для незсунених і зсунених даних
        stats_original = self._calculate_stats(original_data)
        stats_transformed = self._calculate_stats(transformed_data)
        
        # Об'єднуємо статистику для відображення
        combined_stats = {}
        for key in stats_original.keys():
            combined_stats[f"{key} (Незсунені)"] = stats_original[key]
            combined_stats[f"{key} (Зсунені)"] = stats_transformed[key]

        return combined_stats

    def _calculate_stats(self, data):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        variance = np.var(data, ddof=1)  # Дисперсія
        min_val = np.min(data)
        max_val = np.max(data)
        se = std / np.sqrt(n) if n > 1 else 0
        alpha = 0.05  # 95% довірчий рівень
        t_critical = t.ppf(1 - alpha/2, df=n-1) if n > 1 else 0
        ci_lower = mean - t_critical * se if n > 1 else mean
        ci_upper = mean + t_critical * se if n > 1 else mean
        range_val = max_val - min_val
        median = np.median(data)
        cv = (std / mean) * 100 if mean != 0 else float('inf')  # Коефіцієнт варіації
        skewness = skew(data)
        kurt = kurtosis(data)  # Ексцес
        contrkurt = 1 / (kurt + 3) if (kurt + 3) != 0 else float('inf')  # Контрексцес
        # Ненарам. коэф. вар. (коефіцієнт варіації на основі медіани)
        non_param_cv = (np.median(np.abs(data - median)) / median) * 100 if median != 0 else float('inf')
        # MAD (Median Absolute Deviation)
        mad = np.median(np.abs(data - median))
        
        return {
            "Довжина": n,
            "Середнє": mean,
            "Дисперсія": variance,
            "Стандартне відхилення": std,
            "Асиметрія": skewness,
            "Ексцес": kurt,
            "Контрексцес": contrkurt,
            "Мінімум": min_val,
            "Максимум": max_val,
            "Розмах": range_val,
            "Медіана": median,
            "Коефіцієнт варіації (%)": cv,
            "Ненарам. коэф. вар.": non_param_cv,
            "MAD": mad,
            "Довірчий інтервал (95%)": (ci_lower, ci_upper),
        }

    def _plot_histogram(self, canvas, data, bins=None):
        if bins is None:
            bins = min(10, len(set(data)))
        
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        
        # Побудова стандартної гістограми з прямокутниками
        counts, bins, _ = ax.hist(data, bins=bins, color='purple', alpha=0.7, edgecolor='black', label='Частота')
        
        # Обчислення параметрів нормального розподілу
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:  # Перевірка, щоб уникнути ділення на нуль
            # Масштабування нормального розподілу до частот гістограми
            x = np.linspace(min(bins), max(bins), 100)
            bin_width = bins[1] - bins[0]  # Ширина одного біну
            total_count = len(data)  # Загальна кількість даних
            y = norm.pdf(x, mean, std) * total_count * bin_width  # Масштабування до частот
            ax.plot(x, y, 'r-', label='Щільність')
        
        ax.set_title("Гістограма частоти")
        ax.set_xlabel("Час очікування (мс)")
        ax.set_ylabel("Частота")
        ax.legend()
        ax.grid(True, alpha=0.3)
        canvas.figure.tight_layout()
        canvas.draw()

    def plot_ecdf(self, column, canvas):
        if self.data is None or self.data.empty:
            QMessageBox.warning(None, "Помилка", "Дані відсутні або файл порожній!")
            return None
        if not column or column not in self.data.columns:
            QMessageBox.warning(None, "Помилка", "Стовпець не вибрано або не існує!")
            return None
        data = self.data[column].dropna().values
        if len(data) == 0:
            QMessageBox.warning(None, "Помилка", "Немає даних для аналізу!")
            return None

        sorted_data = np.sort(data)
        n = len(sorted_data)
        y = np.arange(1, n + 1) / n

        # Обчислення довірчого інтервалу для ECDF (на основі Dvoretzky-Kiefer-Wolfowitz)
        alpha = 0.05  # 95% довірчий рівень
        epsilon = np.sqrt(np.log(2/alpha) / (2 * n))
        lower_bound = np.maximum(y - epsilon, 0)
        upper_bound = np.minimum(y + epsilon, 1)

        # Теоретичний експоненціальний розподіл
        mean = np.mean(data)
        lambda_param = 1 / mean if mean > 0 else 0
        x_theoretical = np.linspace(min(sorted_data), max(sorted_data), 100)
        y_theoretical = expon.cdf(x_theoretical, scale=1/lambda_param) if lambda_param > 0 else np.zeros_like(x_theoretical)

        # Побудова графіка
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)

        # Затінена область довірчого інтервалу
        ax.fill_between(sorted_data, lower_bound, upper_bound, color='pink', alpha=0.5, label='Довірчий інтервал')
        
        # Теоретичний розподіл
        ax.plot(x_theoretical, y_theoretical, 'r-', label='Теоретичний розподіл')
        
        # Верхня і нижня межі
        ax.plot(sorted_data, lower_bound, 'g--', label='Нижня межа')
        ax.plot(sorted_data, upper_bound, 'g--', label='Верхня межа')
        
        # Емпіричний розподіл
        ax.step(sorted_data, y, 'b-', label='Емпіричний розподіл')

        ax.set_title("Порівняння емпіричного та нормального розподілу")
        ax.set_xlabel("Час очікування (мс)")
        ax.set_ylabel("Ймовірність")
        ax.legend()
        ax.grid(True)
        canvas.figure.tight_layout()
        canvas.draw()

        stats = self._calculate_stats(data)
        return stats

    def detect_anomalies_with_bounds(self, column, lower_bound, upper_bound, canvas):
        if self.data is None or self.data.empty:
            QMessageBox.warning(None, "Помилка", "Дані відсутні або файл порожній!")
            return {"anomalies": [], "indices": []}

        if not column or column not in self.data.columns:
            QMessageBox.warning(None, "Помилка", "Стовпець не вибрано або не існує!")
            return {"anomalies": [], "indices": []}

        data = self.data[column].dropna().values
        if len(data) == 0:
            QMessageBox.warning(None, "Помилка", "Немає даних для аналізу!")
            return {"anomalies": [], "indices": []}

        anomaly_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        anomalies = data[anomaly_indices]

        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.scatter(range(len(data)), data, color='blue', alpha=0.5, label='Нормальні дані')
        
        if len(anomalies) > 0:
            ax.scatter(anomaly_indices, anomalies, color='red', label='Аномалії')
            if lower_bound != float('-inf'):
                ax.axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.5, label='Нижня межа')
            if upper_bound != float('inf'):
                ax.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.5, label='Верхня межа')
        
        ax.set_title(f"Виявлення аномалій у стовпці '{column}'")
        ax.set_xlabel("Індекс")
        ax.set_ylabel("Значення")
        ax.legend()
        canvas.draw()
        
        return {"anomalies": anomalies.tolist(), "indices": anomaly_indices.tolist()}

    def remove_anomalies(self, column, indices_to_remove):
        if self.data is None or self.data.empty:
            QMessageBox.warning(None, "Помилка", "Дані відсутні або файл порожній!")
            return
        
        self.data = self.data.drop(indices_to_remove).reset_index(drop=True)

    def get_stats(self, column):
        if self.data is None or self.data.empty:
            return None
        if not column or column not in self.data.columns:
            return None
        data = self.data[column].dropna().values
        if len(data) == 0:
            return None
        return self._calculate_stats(data)