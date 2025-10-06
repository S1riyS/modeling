import numpy as np
import pandas as pd
from utils import read_data, calculate_relative_deviation, get_sample_sizes
from statistics import StatisticsCalculator
from distribution import DistributionApproximator
from visualization import Visualizer

def format_confidence_interval(value: float) -> str:
    """Форматирование доверительного интервала в виде ±значение"""
    return f"±{value:.4f}"

def print_form1_table(results: dict, relative_results: dict, sample_sizes: list):
    """Печать полной таблицы ФОРМА 1"""
    print("\n" + "="*100)
    print("ФОРМА 1: Характеристики заданной ЧП")
    print("="*100)
    
    # Заголовки таблицы
    headers = ["Характеристика", "10", "20", "50", "100", "200", "300"]
    print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12} {headers[6]:<12}")
    print("-"*100)
    
    # 1. Математическое ожидание
    means = [f"{results[n]['mean']:.4f}" for n in sample_sizes]
    rel_means = [f"{relative_results[n]['mean']:.2f}%" for n in sample_sizes]
    print(f"{'Мат. ож.':<20} {means[0]:<12} {means[1]:<12} {means[2]:<12} {means[3]:<12} {means[4]:<12} {means[5]:<12}")
    print(f"{'':<20} {rel_means[0]:<12} {rel_means[1]:<12} {rel_means[2]:<12} {rel_means[3]:<12} {rel_means[4]:<12} {rel_means[5]:<12}")
    print()
    
    # 2. Доверительный интервал (0.9)
    ci_90_values = [format_confidence_interval(results[n]['ci_90']) for n in sample_sizes]
    rel_ci_90 = [f"{relative_results[n].get('ci_90', 0):.2f}%" for n in sample_sizes]
    print(f"{'Дов. инт. (0,9)':<20} {ci_90_values[0]:<12} {ci_90_values[1]:<12} {ci_90_values[2]:<12} {ci_90_values[3]:<12} {ci_90_values[4]:<12} {ci_90_values[5]:<12}")
    print(f"{'':<20} {rel_ci_90[0]:<12} {rel_ci_90[1]:<12} {rel_ci_90[2]:<12} {rel_ci_90[3]:<12} {rel_ci_90[4]:<12} {rel_ci_90[5]:<12}")
    print()
    
    # 3. Доверительный интервал (0.95)
    ci_95_values = [format_confidence_interval(results[n]['ci_95']) for n in sample_sizes]
    rel_ci_95 = [f"{relative_results[n].get('ci_95', 0):.2f}%" for n in sample_sizes]
    print(f"{'Дов. инт. (0,95)':<20} {ci_95_values[0]:<12} {ci_95_values[1]:<12} {ci_95_values[2]:<12} {ci_95_values[3]:<12} {ci_95_values[4]:<12} {ci_95_values[5]:<12}")
    print(f"{'':<20} {rel_ci_95[0]:<12} {rel_ci_95[1]:<12} {rel_ci_95[2]:<12} {rel_ci_95[3]:<12} {rel_ci_95[4]:<12} {rel_ci_95[5]:<12}")
    print()
    
    # 4. Доверительный интервал (0.99)
    ci_99_values = [format_confidence_interval(results[n]['ci_99']) for n in sample_sizes]
    rel_ci_99 = [f"{relative_results[n].get('ci_99', 0):.2f}%" for n in sample_sizes]
    print(f"{'Дов. инт. (0,99)':<20} {ci_99_values[0]:<12} {ci_99_values[1]:<12} {ci_99_values[2]:<12} {ci_99_values[3]:<12} {ci_99_values[4]:<12} {ci_99_values[5]:<12}")
    print(f"{'':<20} {rel_ci_99[0]:<12} {rel_ci_99[1]:<12} {rel_ci_99[2]:<12} {rel_ci_99[3]:<12} {rel_ci_99[4]:<12} {rel_ci_99[5]:<12}")
    print()
    
    # 5. Дисперсия
    variances = [f"{results[n]['variance']:.4f}" for n in sample_sizes]
    rel_variances = [f"{relative_results[n]['variance']:.2f}%" for n in sample_sizes]
    print(f"{'Дисперсия':<20} {variances[0]:<12} {variances[1]:<12} {variances[2]:<12} {variances[3]:<12} {variances[4]:<12} {variances[5]:<12}")
    print(f"{'':<20} {rel_variances[0]:<12} {rel_variances[1]:<12} {rel_variances[2]:<12} {rel_variances[3]:<12} {rel_variances[4]:<12} {rel_variances[5]:<12}")
    print()
    
    # 6. Среднеквадратическое отклонение (СКО)
    stds = [f"{results[n]['std']:.4f}" for n in sample_sizes]
    rel_stds = [f"{relative_results[n]['std']:.2f}%" for n in sample_sizes]
    print(f"{'С. к. о.':<20} {stds[0]:<12} {stds[1]:<12} {stds[2]:<12} {stds[3]:<12} {stds[4]:<12} {stds[5]:<12}")
    print(f"{'':<20} {rel_stds[0]:<12} {rel_stds[1]:<12} {rel_stds[2]:<12} {rel_stds[3]:<12} {rel_stds[4]:<12} {rel_stds[5]:<12}")
    print()
    
    # 7. Коэффициент вариации
    variations = [f"{results[n]['variation']:.4f}" for n in sample_sizes]
    rel_variations = [f"{relative_results[n]['variation']:.2f}%" for n in sample_sizes]
    print(f"{'К-т вариации':<20} {variations[0]:<12} {variations[1]:<12} {variations[2]:<12} {variations[3]:<12} {variations[4]:<12} {variations[5]:<12}")
    print(f"{'':<20} {rel_variations[0]:<12} {rel_variations[1]:<12} {rel_variations[2]:<12} {rel_variations[3]:<12} {rel_variations[4]:<12} {rel_variations[5]:<12}")
    
    print("\nПримечание: % - относительные отклонения от значений для выборки из 300 величин")

def print_autocorrelation_comparison(original_autocorr: list, generated_autocorr: list):
    """Печать сравнения коэффициентов автокорреляции"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ КОЭФФИЦИЕНТОВ АВТОКОРРЕЛЯЦИИ")
    print("="*60)
    print(f"{'Сдвиг':<6} {'Исходная':<12} {'Сгенерированная':<16} {'Разность':<12}")
    print("-"*60)
    
    for i, (orig, gen) in enumerate(zip(original_autocorr, generated_autocorr), 1):
        difference = abs(orig - gen)
        print(f"{i:<6} {orig:<12.4f} {gen:<16.4f} {difference:<12.4f}")

def main():
    # 1. Чтение данных
    print("=== УИР 1: Статистический анализ числовой последовательности ===")
    data = read_data('data/numbers.txt')
    
    if len(data) == 0:
        print("Ошибка: не удалось загрузить данные")
        return
    
    print(f"Загружено {len(data)} значений")
    print(f"Первые 10 значений: {data[:10]}")
    
    # 2. Анализ для разных размеров выборки
    sample_sizes = get_sample_sizes()
    reference_sample = data[:300]  # Эталонная выборка (300 элементов)
    
    # Характеристики эталонной выборки
    ref_mean = StatisticsCalculator.calculate_mean(reference_sample)
    ref_variance = StatisticsCalculator.calculate_variance(reference_sample)
    ref_std = StatisticsCalculator.calculate_std(reference_sample)
    ref_variation = StatisticsCalculator.calculate_variation_coefficient(reference_sample)
    
    # Доверительные интервалы для эталонной выборки
    ref_ci_90 = StatisticsCalculator.calculate_confidence_interval(reference_sample, 0.9)
    ref_ci_95 = StatisticsCalculator.calculate_confidence_interval(reference_sample, 0.95)
    ref_ci_99 = StatisticsCalculator.calculate_confidence_interval(reference_sample, 0.99)
    
    print(f"\nЭталонные характеристики (n=300):")
    print(f"Математическое ожидание: {ref_mean:.4f}")
    print(f"Дисперсия: {ref_variance:.4f}")
    print(f"СКО: {ref_std:.4f}")
    print(f"Коэффициент вариации: {ref_variation:.4f}")
    print(f"Дов. интервал (0.9): ±{ref_ci_90:.4f}")
    print(f"Дов. интервал (0.95): ±{ref_ci_95:.4f}")
    print(f"Дов. интервал (0.99): ±{ref_ci_99:.4f}")
    
    # 3. Расчет характеристик для разных размеров выборки
    results = {}
    
    for n in sample_sizes:
        sample = data[:n]
        
        mean = StatisticsCalculator.calculate_mean(sample)
        variance = StatisticsCalculator.calculate_variance(sample)
        std = StatisticsCalculator.calculate_std(sample)
        variation = StatisticsCalculator.calculate_variation_coefficient(sample)
        
        # Доверительные интервалы
        ci_90 = StatisticsCalculator.calculate_confidence_interval(sample, 0.9)
        ci_95 = StatisticsCalculator.calculate_confidence_interval(sample, 0.95)
        ci_99 = StatisticsCalculator.calculate_confidence_interval(sample, 0.99)
        
        results[n] = {
            'mean': mean, 'variance': variance, 'std': std, 'variation': variation,
            'ci_90': ci_90, 'ci_95': ci_95, 'ci_99': ci_99
        }
    
    # 4. Относительные отклонения (включая доверительные интервалы)
    relative_results = {}
    for n in sample_sizes:
        if n == 300:  # Для эталона отклонение 0%
            relative_results[n] = {
                'mean': 0, 'variance': 0, 'std': 0, 'variation': 0,
                'ci_90': 0, 'ci_95': 0, 'ci_99': 0
            }
        else:
            rel_mean = abs((results[n]['mean'] - ref_mean) / ref_mean) * 100
            rel_variance = abs((results[n]['variance'] - ref_variance) / ref_variance) * 100
            rel_std = abs((results[n]['std'] - ref_std) / ref_std) * 100
            rel_variation = abs((results[n]['variation'] - ref_variation) / ref_variation) * 100
            rel_ci_90 = abs((results[n]['ci_90'] - ref_ci_90) / ref_ci_90) * 100 if ref_ci_90 != 0 else 0
            rel_ci_95 = abs((results[n]['ci_95'] - ref_ci_95) / ref_ci_95) * 100 if ref_ci_95 != 0 else 0
            rel_ci_99 = abs((results[n]['ci_99'] - ref_ci_99) / ref_ci_99) * 100 if ref_ci_99 != 0 else 0
            
            relative_results[n] = {
                'mean': rel_mean, 'variance': rel_variance, 'std': rel_std, 'variation': rel_variation,
                'ci_90': rel_ci_90, 'ci_95': rel_ci_95, 'ci_99': rel_ci_99
            }
    
    # 5. Вывод полной таблицы ФОРМА 1
    print_form1_table(results, relative_results, sample_sizes)
    
    # 6. Визуализация исходной последовательности
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ ИСХОДНОЙ ПОСЛЕДОВАТЕЛЬНОСТИ")
    print("="*50)
    
    # График последовательности
    Visualizer.plot_sequence(data[:300], "Исходная числовая последовательность")
    
    # Автокорреляционный анализ исходной последовательности
    original_autocorrelations = StatisticsCalculator.calculate_autocorrelation(reference_sample)
    print(f"Коэффициенты автокорреляции исходной последовательности:")
    for i, acf in enumerate(original_autocorrelations, 1):
        print(f"Сдвиг {i}: {acf:.4f}")
    Visualizer.plot_autocorrelation(original_autocorrelations, "Автокорреляционный анализ исходной последовательности")
    
    # Гистограмма с таблицей интервалов
    print("\nГИСТОГРАММА РАСПРЕДЕЛЕНИЯ ЧАСТОТ (исходная последовательность)")
    frequencies, bin_edges, intervals_table = Visualizer.plot_histogram(reference_sample)
    Visualizer.print_histogram_table(intervals_table)
    
    # 7. Аппроксимация закона распределения
    print(f"\nАппроксимация закона распределения:")
    dist_type = DistributionApproximator.determine_distribution_type(ref_variation)
    print(f"Коэффициент вариации: {ref_variation:.4f} -> Тип распределения: {dist_type}")
    
    if dist_type == "гиперэкспоненциальный":
        t1, t2 = DistributionApproximator.hyperexponential_parameters(ref_mean, ref_variation)
        print(f"Параметры гиперэкспоненциального распределения: t1={t1:.4f}, t2={t2:.4f}")
        
        # Генерация новой последовательности
        generated_sequence = DistributionApproximator.generate_hyperexponential_sequence(300, t1, t2)
        
        # 8. Анализ сгенерированной последовательности
        print("\n" + "="*50)
        print("АНАЛИЗ СГЕНЕРИРОВАННОЙ ПОСЛЕДОВАТЕЛЬНОСТИ")
        print("="*50)
        
        # График сгенерированной последовательности
        Visualizer.plot_sequence(generated_sequence, "Сгенерированная числовая последовательность")
        
        # Автокорреляционный анализ сгенерированной последовательности
        generated_autocorrelations = StatisticsCalculator.calculate_autocorrelation(generated_sequence)
        print(f"Коэффициенты автокорреляции сгенерированной последовательности:")
        for i, acf in enumerate(generated_autocorrelations, 1):
            print(f"Сдвиг {i}: {acf:.4f}")
        Visualizer.plot_autocorrelation(generated_autocorrelations, "Автокорреляционный анализ сгенерированной последовательности")
        
        # Сравнение коэффициентов автокорреляции
        print_autocorrelation_comparison(original_autocorrelations, generated_autocorrelations)
        
        # Гистограмма сгенерированной последовательности
        print("\nГИСТОГРАММА РАСПРЕДЕЛЕНИЯ ЧАСТОТ (сгенерированная последовательность)")
        gen_frequencies, gen_bin_edges, gen_intervals_table = Visualizer.plot_histogram(generated_sequence)
        Visualizer.print_histogram_table(gen_intervals_table)
        
        # Сравнение гистограмм
        print("\nСРАВНЕНИЕ ГИСТОГРАММ:")
        Visualizer.plot_comparison_histograms(reference_sample, generated_sequence)
        
        # Корреляционный анализ между последовательностями
        correlation = StatisticsCalculator.calculate_correlation(reference_sample, generated_sequence)
        print(f"Коэффициент корреляции между последовательностями: {correlation:.4f}")
        
        # Анализ характеристик сгенерированной последовательности
        gen_mean = StatisticsCalculator.calculate_mean(generated_sequence)
        gen_variance = StatisticsCalculator.calculate_variance(generated_sequence)
        gen_std = StatisticsCalculator.calculate_std(generated_sequence)
        gen_variation = StatisticsCalculator.calculate_variation_coefficient(generated_sequence)
        
        print(f"\nСравнение характеристик:")
        print(f"{'Характеристика':<25} {'Исходная':<12} {'Сгенерированная':<16} {'Отклонение %':<12}")
        print("-"*65)
        print(f"{'Мат. ожидание':<25} {ref_mean:<12.4f} {gen_mean:<16.4f} {abs((gen_mean - ref_mean)/ref_mean)*100:<12.2f}")
        print(f"{'Дисперсия':<25} {ref_variance:<12.4f} {gen_variance:<16.4f} {abs((gen_variance - ref_variance)/ref_variance)*100:<12.2f}")
        print(f"{'СКО':<25} {ref_std:<12.4f} {gen_std:<16.4f} {abs((gen_std - ref_std)/ref_std)*100:<12.2f}")
        print(f"{'Коэф. вариации':<25} {ref_variation:<12.4f} {gen_variation:<16.4f} {abs((gen_variation - ref_variation)/ref_variation)*100:<12.2f}")
    
    print("\n=== Анализ завершен ===")

if __name__ == "__main__":
    main()