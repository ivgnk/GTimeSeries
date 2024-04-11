"""
psd_scedasticity.py - Тестирование гетероскедастичности

(C) 2024 Ivan Genik, Perm, Russia
Released under GNU Public License (GPL)
email igenik@rambler.ru
"""

import inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Final  # https://stackoverflow.com/questions/2682745/how-do-i-create-a-constant-in-python
import sys

# for test_of_white()
from sklearn. linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
import pandas as pd
from scipy.stats import spearmanr

ONLY_SCEDASTICITY: Final[bool] = bool(0)


def spearman_rank_correlation(x1: np.ndarray, y1: np.ndarray):
    """
    Как рассчитать ранговую корреляцию Спирмена в Python
    https://www.codecamp.ru/blog/spearman-correlation-python/
    """
    # calculate Spearman Rank correlation and corresponding p-value
    print('\nFunction', inspect.currentframe().f_code.co_name)
    rho, p = spearmanr(a=x1, b=y1)

    # напечатать ранговая корреляция Спирмена и p-значение
    print('ранговая корреляция Спирмена=',rho)
    print('p-значение=',p)
    # Однако, если p-значение корреляции не меньше 0,05, то корреляция не является статистически значимой.

def test_of_white(x1: np.ndarray, y1: np.ndarray):
    """
    Как выполнить тест Уайта в Python (шаг за шагом)
    https://www.codecamp.ru/blog/white-test-in-python/
    # в конце выход из программы
    """
    # Как преобразовать массив NumPy в Pandas DataFrame
    # https://www.codecamp.ru/blog/numpy-array-to-pandas-dataframe/
    # (1) Загрузить данные
    print('\nFunction', inspect.currentframe().f_code.co_name)
    df = pd.DataFrame()
    df['x'] = x1.tolist()
    df['y'] = y1.tolist()
    # print(df)
    # df.info()

    # (2) Подгонка регрессионной модели
    # define response variable
    y = df['y']
    # define predictor variables
    x = df['x']
    # add constant to predictor variables
    x = sm.add_constant(x)
    # fit regression model
    model = sm.OLS(y, x).fit()

    # (3) Тест Уайта
    # perform White's test
    white_test = het_white(model.resid, model.model.exog)

    # define labels to use for output of White's test
    labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

    # print results of White's test
    print(dict(zip(labels, white_test)))
    # Вот как интерпретировать вывод
    # Тестовая статистика Xi^2 = 6.240569679555077.
    # Соответствующее значение p равно 0.044144592492810615
    # В тесте Уайта используются следующие нулевая и альтернативная гипотезы:
    # Нулевая: присутствует гомоскедастичность (остатки равномерно разбросаны)
    # Альтернатива: присутствует гетероскедастичность (остатки разбросаны неравномерно)
    # Поскольку p-значение меньше 0,05, мы можем отвергнуть нулевую гипотезу
    # Это означает, что в регрессионной модели присутствует гетероскедастичность.

    # Как интерпретировать P-значение менее 0,05 (с примерами)
    # https://www.codecamp.ru/blog/p-value-less-than-0-05/
    if ONLY_SCEDASTICITY: sys.exit()


def scedasticity(x: np.ndarray, y: np.ndarray, model: np.ndarray):
    """
    Построение гравика остатков и вызов функций оценки скедастичности
    """
    print('\nFunction', inspect.currentframe().f_code.co_name)
    plt.title('График остатков (данные-модель)')
    y1 = y-model
    plt.plot(x, y1)
    plt.xlabel('Отсчеты')
    plt.ylabel('Разность, мГал')
    plt.grid()
    plt.show()
