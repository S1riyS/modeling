# ./src/03_distribution_fitting.py

import numpy as np
import matplotlib.pyplot as plt


# Загрузка данных
def load_data(file_path):
    with open(file_path, "r") as file:
        data = [float(line.strip()) for line in file.readlines()]
    return np.array(data)


# Построение гистограммы
def plot_histogram(data, bins=20, save_path=None):
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(
        data, bins=bins, density=True, alpha=0.7, edgecolor="black"
    )
    plt.title("Гистограмма распределения частот заданной последовательности")
    plt.xlabel("Значение")
    plt.ylabel("Плотность вероятности")
    plt.grid(True, linestyle="--", alpha=0.7)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return n, bins


# Подбор закона распределения по коэффициенту вариации
def select_distribution(cv):
    if cv < 0.3:
        return "Нормированный Эрланга"
    elif 0.3 <= cv < 0.8:
        return "Гипоэкспоненциальный"
    elif 0.8 <= cv < 1.2:
        return "Экспоненциальный"
    else:
        return "Гиперэкспоненциальный"


# Основная функция
def main():
    # Загрузка данных
    data = load_data("./data/numbers.txt")

    # Основные характеристики
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    cv = std / mean

    print("ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ПОСЛЕДОВАТЕЛЬНОСТИ:")
    print(f"Мат. ожидание: {mean:.3f}")
    print(f"СКО: {std:.3f}")
    print(f"Коэффициент вариации: {cv:.3f}")
    print()

    # 1. Построение гистограммы
    print("1. Построение гистограммы...")
    plot_histogram(data, bins=20, save_path="./histogram.png")

    # 2. Подбор закона распределения
    print("2. Подбор закона распределения:")
    selected_dist = select_distribution(cv)
    print(
        f"На основе коэффициента вариации CV = {cv:.3f} рекомендуется: {selected_dist}"
    )
    print()

    # 3. Генерация последовательности по экспоненциальному распределению (пример)
    # В реальной работе здесь нужно реализовать генератор для выбранного распределения
    print(
        "3. Генерация последовательности по экспоненциальному распределению (пример):"
    )
    lambda_exp = 1 / mean  # Параметр для экспоненциального распределения
    generated_data = np.random.exponential(scale=mean, size=len(data))

    # Характеристики сгенерированной последовательности
    gen_mean = np.mean(generated_data)
    gen_std = np.std(generated_data, ddof=1)
    gen_cv = gen_std / gen_mean

    print(f"Характеристики сгенерированной последовательности:")
    print(f"Мат. ожидание: {gen_mean:.3f}")
    print(f"СКО: {gen_std:.3f}")
    print(f"Коэффициент вариации: {gen_cv:.3f}")
    print()

    # 4. Сравнение гистограмм
    print("4. Сравнение гистограмм исходной и сгенерированной последовательностей:")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(
        data, bins=20, density=True, alpha=0.7, edgecolor="black", label="Исходная"
    )
    plt.title("Исходная последовательность")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.hist(
        generated_data,
        bins=20,
        density=True,
        alpha=0.7,
        edgecolor="black",
        color="orange",
        label="Сгенерированная",
    )
    plt.title("Сгенерированная последовательность")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("./comparison_histograms.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 5. Оценка качества аппроксимации
    print("5. Оценка качества аппроксимации:")
    # Простая оценка по разнице мат. ожиданий
    mean_diff = np.abs(mean - gen_mean) / mean * 100
    print(f"Относительная ошибка мат. ожидания: {mean_diff:.2f}%")

    if mean_diff < 5:
        print("Аппроксимация удовлетворительная.")
    else:
        print(
            "Аппроксимация требует улучшения (рекомендуется другой закон распределения)."
        )


if __name__ == "__main__":
    main()
