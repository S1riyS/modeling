import numpy as np
import scipy.stats as stats 


# Загрузка данных из файла
def load_data(file_path):
    with open(file_path, "r") as file:
        data = [float(line.strip()) for line in file.readlines()]
    return np.array(data)


# Расчет доверительного интервала для среднего (нормальное распределение, неизвестная дисперсия)
def confidence_interval(data, confidence):
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)  # Стандартная ошибка среднего
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)  # t-критическое значение
    margin = t_value * se  # Полуширина интервала
    return mean, margin


# Основная функция
def main():
    # Загрузка всех данных
    data = load_data("./data/numbers.txt")
    n_total = len(data)
    print(f"Всего загружено {n_total} чисел.")

    # Размеры выборок для анализа
    sample_sizes = [10, 20, 50, 100, 200, 300]

    # Эталонные значения (по всей выборке n=300)
    reference_mean = np.mean(data[:300])
    reference_var = np.var(data[:300], ddof=1)
    reference_std = np.std(data[:300], ddof=1)
    reference_cv = reference_std / reference_mean if reference_mean != 0 else 0

    print("\n" + "=" * 80)
    print("Форма 1: Характеристики заданной ЧП")
    print("=" * 80)
    print(
        f"{'Характеристика':<20} | {'10':<10} | {'20':<10} | {'50':<10} | {'100':<10} | {'200':<10} | {'300':<10}"
    )
    print("-" * 100)

    # Заголовки для строк
    headers = [
        "Мат.ож.",
        "Дов. инт. (0.9)",
        "Дов. инт. (0.95)",
        "Дов. инт. (0.99)",
        "Дисперсия",
        "С.к.о.",
        "К-т вариации",
    ]

    # Создаем пустую таблицу для заполнения
    table_rows = [[] for _ in headers]

    # Расчет для каждой выборки
    for i, n in enumerate(sample_sizes):
        sample = data[:n]  # Берем первые n элементов

        # Мат. ожидание
        mean = np.mean(sample)
        table_rows[0].append(f"{mean:.3f}")

        # Доверительные интервалы
        conf_levels = [0.90, 0.95, 0.99]
        for j, conf in enumerate(conf_levels):
            ci_mean, ci_margin = confidence_interval(sample, conf)
            table_rows[1 + j].append(f"±{ci_margin:.3f}")

        # Дисперсия (несмещенная)
        var = np.var(sample, ddof=1)
        table_rows[4].append(f"{var:.3f}")

        # СКО
        std = np.std(sample, ddof=1)
        table_rows[5].append(f"{std:.3f}")

        # Коэффициент вариации
        cv = std / mean if mean != 0 else 0
        table_rows[6].append(f"{cv:.3f}")

    # Вывод таблицы
    for header, row in zip(headers, table_rows):
        print(f"{header:<20} | ", end="")
        for val in row:
            print(f"{val:<10} | ", end="")
        print()

    # Вывод эталонных значений
    print("\n" + "=" * 50)
    print("ЭТАЛОННЫЕ ЗНАЧЕНИЯ (n=300):")
    print(f"Мат.ож.: {reference_mean:.3f}")
    print(f"Дисперсия: {reference_var:.3f}")
    print(f"СКО: {reference_std:.3f}")
    print(f"К-т вариации: {reference_cv:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
