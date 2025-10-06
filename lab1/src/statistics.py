import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any

class StatisticsCalculator:
    """Класс для расчета статистических характеристик"""
    
    @staticmethod
    def calculate_mean(data: np.ndarray) -> float:
        """Расчет математического ожидания"""
        return np.mean(data)
    
    @staticmethod
    def calculate_variance(data: np.ndarray) -> float:
        """Расчет несмещенной дисперсии"""
        return np.var(data, ddof=1)
    
    @staticmethod
    def calculate_std(data: np.ndarray) -> float:
        """Расчет среднеквадратического отклонения"""
        return np.std(data, ddof=1)
    
    @staticmethod
    def calculate_variation_coefficient(data: np.ndarray) -> float:
        """Расчет коэффициента вариации"""
        mean = StatisticsCalculator.calculate_mean(data)
        std = StatisticsCalculator.calculate_std(data)
        return std / mean if mean != 0 else 0
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> float:
        """Расчет доверительного интервала для математического ожидания"""
        n = len(data)
        if n < 2:
            return 0
        
        mean = StatisticsCalculator.calculate_mean(data)
        std = StatisticsCalculator.calculate_std(data)
        sem = std / np.sqrt(n)  # стандартная ошибка среднего
        
        # Используем t-распределение для малых выборок
        if n < 30:
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            t_value = stats.norm.ppf((1 + confidence) / 2)
        
        return t_value * sem
    
    @staticmethod
    def calculate_autocorrelation(data: np.ndarray, max_lag: int = 10) -> List[float]:
        """Расчет коэффициентов автокорреляции"""
        n = len(data)
        autocorrelations = []
        
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            
            # Ковариация со сдвигом
            covariance = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorrelations.append(covariance if not np.isnan(covariance) else 0)
        
        return autocorrelations
    
    @staticmethod
    def calculate_correlation(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Расчет коэффициента корреляции между двумя последовательностями"""
        if len(seq1) != len(seq2):
            min_len = min(len(seq1), len(seq2))
            seq1 = seq1[:min_len]
            seq2 = seq2[:min_len]
        
        correlation = np.corrcoef(seq1, seq2)[0, 1]
        return correlation if not np.isnan(correlation) else 0