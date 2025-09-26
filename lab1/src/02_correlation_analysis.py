# ./src/02_correlation_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Загрузка данных
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return np.array(data)

# Построение графика последовательности
def plot_sequence(data, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(data, 'b-', linewidth=0.8, markersize=2)
    plt.title('График заданной числовой последовательности')
    plt.xlabel('Индекс измерения')
    plt.ylabel('Значение')
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Автокорреляционный анализ
def autocorrelation_analysis(data, max_lag=10):
    # Вычисление коэффициентов автокорреляции
    acf_values = acf(data, nlags=max_lag, fft=True)
    
    print("Коэффициенты автокорреляции:")
    print("Lag | Value")
    print("-" * 15)
    for lag, value in enumerate(acf_values):
        print(f"{lag:3d} | {value:.4f}")
    
    # Построение графика автокорреляционной функции
    plt.figure(figsize=(10, 5))
    plt.stem(range(len(acf_values)), acf_values, basefmt=" ")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Автокорреляционная функция (ACF)')
    plt.xlabel('Лаг')
    plt.ylabel('Коэффициент автокорреляции')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return acf_values

# Основная функция
def main():
    # Загрузка данных
    data = load_data('./data/numbers.txt')
    
    # 1. График последовательности
    print("1. Анализ графика последовательности:")
    plot_sequence(data, './sequence_plot.png')
    
    # Анализ характера последовательности
    print("Характер последовательности:")
    print("- Визуально последовательность не имеет явного тренда (не возрастает и не убывает).")
    print("- Периодичность на графике не наблюдается.")
    print("- Значения колеблются вокруг некоторого среднего, есть выбросы.")
    print()
    
    # 2. Автокорреляционный анализ
    print("2. Автокорреляционный анализ:")
    acf_values = autocorrelation_analysis(data, max_lag=10)
    
    # Проверка на случайность (первые несколько лагов не должны значимо отличаться от 0)
    significant_lags = np.where(np.abs(acf_values[1:]) > 1.96/np.sqrt(len(data)))[0]
    if len(significant_lags) == 0:
        print("Вывод: Последовательность можно считать случайной (нет значимой автокорреляции).")
    else:
        print(f"Вывод: Обнаружена значимая автокорреляция на лагах: {significant_lags + 1}")

if __name__ == "__main__":
    main()