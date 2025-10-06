import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple

class Visualizer:
    """Класс для визуализации результатов"""
    
    @staticmethod
    def plot_sequence(sequence: np.ndarray, title: str = "Числовая последовательность"):
        """Построение графика числовой последовательности"""
        plt.figure(figsize=(12, 6))
        plt.plot(sequence, 'b-', linewidth=0.8)
        plt.title(title)
        plt.xlabel('Индекс')
        plt.ylabel('Значение')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_autocorrelation(autocorrelations: List[float], title: str = "Автокорреляционный анализ"):
        """Построение графика автокорреляционной функции"""
        plt.figure(figsize=(10, 6))
        lags = range(1, len(autocorrelations) + 1)
        plt.stem(lags, autocorrelations, basefmt=" ")
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(title)
        plt.xlabel('Сдвиг')
        plt.ylabel('Коэффициент автокорреляции')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def calculate_histogram_intervals(sequence: np.ndarray, bins: int = 18) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, int]]]:
        """Расчет интервалов и частот для гистограммы"""
        min_val = np.min(sequence)
        max_val = np.max(sequence)
        
        # Создание интервалов
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        frequencies, _ = np.histogram(sequence, bins=bin_edges)
        
        # Формирование таблицы интервалов
        intervals_table = []
        for i in range(len(frequencies)):
            left_bound = bin_edges[i]
            right_bound = bin_edges[i + 1]
            frequency = frequencies[i]
            intervals_table.append((left_bound, right_bound, frequency))
        
        return frequencies, bin_edges, intervals_table
    
    @staticmethod
    def plot_histogram(sequence: np.ndarray, bins: int = 18, title: str = "Гистограмма распределения частот"):
        """Построение гистограммы распределения"""
        frequencies, bin_edges, intervals_table = Visualizer.calculate_histogram_intervals(sequence, bins)
        
        plt.figure(figsize=(12, 6))
        n, bins, patches = plt.hist(sequence, bins=bin_edges, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel('Интервалы значений')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return frequencies, bin_edges, intervals_table
    
    @staticmethod
    def print_histogram_table(intervals_table: List[Tuple[float, float, int]]):
        """Печать таблицы интервалов гистограммы"""
        print("\n" + "="*80)
        print("ТАБЛИЦА ИНТЕРВАЛОВ ГИСТОГРАММЫ")
        print("="*80)
        print(f"{'№':<3} {'Левая граница':<15} {'Правая граница':<15} {'Частота':<10}")
        print("-"*50)
        
        for i, (left, right, freq) in enumerate(intervals_table, 1):
            print(f"{i:<3} {left:<15.4f} {right:<15.4f} {freq:<10}")
    
    @staticmethod
    def plot_comparison_histograms(original: np.ndarray, generated: np.ndarray, bins: int = 18):
        """Сравнение гистограмм исходной и сгенерированной последовательностей"""
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(original, bins=bins, alpha=0.7, edgecolor='black', color='blue')
        plt.title('Исходная последовательность')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        
        plt.subplot(1, 2, 2)
        plt.hist(generated, bins=bins, alpha=0.7, edgecolor='black', color='green')
        plt.title('Сгенерированная последовательность')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        
        plt.tight_layout()
        plt.show()