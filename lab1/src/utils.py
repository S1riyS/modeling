import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

def read_data(file_path: str) -> np.ndarray:
    """Чтение числовой последовательности из файла"""
    try:
        with open(file_path, 'r') as file:
            data = [float(line.strip()) for line in file if line.strip()]
        return np.array(data)
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return np.array([])

def calculate_relative_deviation(values: List[float], reference: float) -> List[float]:
    """Расчет относительных отклонений от эталонного значения"""
    return [abs((val - reference) / reference) * 100 if reference != 0 else 0 
            for val in values]

def get_sample_sizes() -> List[int]:
    """Возвращает размеры выборок для анализа"""
    return [10, 20, 50, 100, 200, 300]