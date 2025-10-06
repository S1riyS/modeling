import numpy as np
from typing import Tuple
from statistics import StatisticsCalculator

class DistributionApproximator:
    """Класс для аппроксимации законов распределения"""
    
    @staticmethod
    def determine_distribution_type(variation_coefficient: float) -> str:
        """Определение типа распределения по коэффициенту вариации"""
        if variation_coefficient < 1:
            return "гипоэкспоненциальный"
        elif abs(variation_coefficient - 1) < 0.1:
            return "экспоненциальный"
        else:
            return "гиперэкспоненциальный"
    
    @staticmethod
    def hyperexponential_parameters(mean: float, variation_coefficient: float, q: float = 0.3) -> Tuple[float, float]:
        """Расчет параметров гиперэкспоненциального распределения"""
        v = variation_coefficient
        
        # Проверка допустимости q
        q_max = 2 / (v**2 + 1)
        q = min(q, q_max)
        
        # Расчет параметров t1 и t2
        t1 = (1 + np.sqrt((1 - q) / (2 * q) * (v**2 - 1))) * mean
        t2 = (1 - np.sqrt(q / (2 * (1 - q)) * (v**2 - 1))) * mean
        
        return t1, t2
    
    @staticmethod
    def generate_hyperexponential_sequence(size: int, t1: float, t2: float, q: float = 0.3) -> np.ndarray:
        """Генерация последовательности по гиперэкспоненциальному закону"""
        sequence = []
        
        for _ in range(size):
            r1, r2 = np.random.random(), np.random.random()
            
            if r1 < q:
                x = t1 * (-np.log(1 - r2))
            else:
                x = t2 * (-np.log(1 - r2))
            
            sequence.append(x)
        
        return np.array(sequence)