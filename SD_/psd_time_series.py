"""
psd_time_series.py - Тестирование временных рядов

(C) 2024 Ivan Genik, Perm, Russia
Released under GNU Public License (GPL)
email igenik@rambler.ru
"""
# Стационарность
# https://www.youtube.com/watch?v=oboWrxJCj9I
# https://www.youtube.com/@user-bg8cd4fn7d/videos

import inspect
import numpy as np
import scipy.stats as stats
from scipy.stats import lognorm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import math

import matplotlib.pyplot as plt
from typing import Final  # https://stackoverflow.com/questions/2682745/how-do-i-create-a-constant-in-python
from pnumpy import *
import sys


ONLY_STATIONARITY: Final[bool] = bool(1)

def Dickey_Fuller_test(values):
    """
    https://ru.wikipedia.org/wiki/Тест_Дики_—_Фуллера
    Шаг 3: Расширенный тест Дикки-Фуллера // Step 3: Augmented Dickey-Fuller test
    https://www.geeksforgeeks.org/how-to-check-if-time-series-data-is-stationary-with-python-2/
    """
    print('\nFunction', inspect.currentframe().f_code.co_name)
    n=5
    sspl=splitting_array(values, n)
    print('Расширенный тест Дикки-Фуллера')
    for i in range(n):
        print(f'Часть {i=}')
        # passing the extracted passengers count to adfuller function.
        # result of adfuller function is stored in a res variable
        res1 = adfuller(sspl[i])
        # Printing the statistical result of the adfuller test
        print('Augmneted Dickey_fuller Statistic: %f' % res1[0])
        print('p-value: %f' % res1[1])

        # printing the critical values at different alpha levels.
        print('critical values at different levels:')
        for k, v in res1[4].items():
            print('\t%s: %.3f' % (k, v))
        print('------------------')


def Kruskal_Wallis_test(group1, group2, group3):
    """
    https://ru.wikipedia.org/wiki/Критерий_Краскела_—_Уоллиса
    2022 Как выполнить тест Крускала-Уоллиса в Python
    https://www.codecamp.ru/blog/kruskal-wallis-test-python/
    """
    print('\nFunction', inspect.currentframe().f_code.co_name)
    res=stats.kruskal(group1, group2, group3)  #
    print(res)
    print('-----------------------------------------------')


def tst_mann_whitney_u_test():
    print('\nFunction', inspect.currentframe().f_code.co_name)
    np.random.seed(124)
    n=22000
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    group1 = np.random.rand(n); sg1=set(group1)
    group2 = np.random.rand(n); sg2=set(group2)
    print(f'{len(group1)=} {len(group2)=}')
    print(f'{len(sg1)=} {len(sg2)=}')
    res=stats.mannwhitneyu(group1, group2, alternative='two-sided')
    print(res)
    # [20, 23, 21, 25, 18, 17, 18, 24, 20, 24, 23, 19]
    # [24, 25, 21, 22, 23, 18, 17, 28, 24, 27, 21, 23]

def mann_whitney_u_test(parts1:np.ndarray, parts2:np.ndarray):
    """
    проверить гипотезу о том, что две независимые выборки взяты из одного и того же распределения,
    то есть нет значимого различия между распределениями выборок
    https://ru.wikipedia.org/wiki/U-критерий_Манна_—_Уитни
    https://habr.com/ru/companies/otus/articles/805961/
    https://www.codecamp.ru/blog/mann-whitney-u-test/
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    """
    sg1=set(parts1); sg2=set(parts2)
    print(f'{len(parts1)=} {len(parts2)=}')
    print(f'{len(sg1)=} {len(sg2)=}')

    res=stats.mannwhitneyu(parts1, parts2, alternative='two-sided') #
    print(res)


def none_in_ndparray(x:np.ndarray):
    lst=[not math.isnan(x1) for x1 in x]
    print('Нет NaN = ',all(lst))


def nonpraametric_stat_tests(pparts0:np.ndarray, pparts1:np.ndarray):
    """
    Непараметрические статистические тесты
    https://habr.com/ru/companies/otus/articles/805961/
    """
    print(pparts0, len(pparts0), type(pparts0))
    none_in_ndparray(pparts0)
    print(pparts1, len(pparts1), type(pparts1))
    none_in_ndparray(pparts1)
    res=mann_whitney_u_test(pparts0,pparts1) # pparts0.tolist,pparts1.tolist

def stationarity_test(values:np.ndarray, x:np.ndarray):
    """
    Как проверить, являются ли данные временных рядов стационарными с помощью Python?
    www.geeksforgeeks.org/how-to-check-if-time-series-data-is-stationary-with-python-2
    """
    # part-1
    # разделим эти данные на разные группы и вычислим среднее значение и
    # дисперсию для разных групп и проверим согласованность
    N=3
    parts = int(len(values) / N)
    pparts=[np.zeros(2)]*N
    xx=[np.zeros(2)]*N

    # splitting the data into three parts
    part_1, part_2, part_3= values[0:parts], values[parts:(parts * 2)], values[(parts * 2):(parts * 3)]
    pparts=[part_1, part_2, part_3]
    xx[0], xx[1], xx[2] = x[0:parts], x[parts:(parts * 2)], x[(parts * 2):(parts * 3)]

    # ------- test 2 ------ begin
    # n=7 # stats.kruskal for 7 KruskalResult(statistic=26329.077623206384, pvalue=0.0)
    # n=6 # stats.kruskal for 7 KruskalResult(statistic=25874.07124380349, pvalue=0.0)
    # print('stats.kruskal for ',n)
    # vi=splitting_array(values, n)
    # res=stats.kruskal(vi[0],vi[1], vi[2],vi[3], vi[4],vi[5])  #
    # print(res)
    # sys.exit()
    # ------- test 2 ------ end



    # mann_whitney_u_test(part_1, part_2) - Не работает
    Kruskal_Wallis_test(part_1, part_2, part_3)  #
    # Для 3 частей KruskalResult(statistic=2348.80506542872, pvalue=0.0)
    #

    # 2022 Как проверить нормальность в Python (4 метода)
    # https://www.codecamp.ru/blog/normality-test-python/
    print('scipy.stats.shapiro')
    res=stats.shapiro(values[0:4999])
    print(res)
    lst=[math.log10(s) for s in values]
    print(f'{min(lst)=}')
    res=stats.shapiro(lst[0:4999])
    print(res)
    print('------------------')

    print('график QQ')
    # Как проверить нормальность в Python
    # Способ 2: создать график QQ
    # https://www.codecamp.ru/blog/normality-test-python/
    # create Q-Q plot with 45-degree line added to plot
    print(max(values))
    lst = [3+math.log10(s) for s in values]
    fig = sm.qqplot(np.array(lst), stats.norm, fit=True, line="45") # values[0:4999]
    plt.title('график QQ norm')
    plt.grid()
    fig = sm.qqplot(np.array(lst), stats.lognorm, fit=True, line="45") # values[0:4999]
    plt.title('график QQ lognorm')
    plt.grid()
    plt.show()
    print('------------------')
    Dickey_Fuller_test(values)
    # plt.figure(figsize=(14, 8))
    # for i in range(N):
    #     plt.subplot(1,3,i+1)
    #     plt.plot(xx[i], pparts[i])
    #     plt.title(f' часть {i + 1}')
    #     plt.grid()
    # plt.show()


    # calculating the mean of the separated three
    # parts of data individually.
    mean_i=[pparts[i].mean() for i in range(N)]

    # calculating the variance of the separated
    # three parts of data individually.
    var_i=[pparts[i].var() for i in range(N)]
    # var_1, var_2, var_3 = part_1.var(), part_2.var(), part_3.var()

    # printing the mean of three groups
    print('mean1=%f, mean2=%f, mean2=%f' % (mean_i[0], mean_i[1], mean_i[2]))

    # printing the variance of three groups
    print('variance1=%f, variance2=%f, variance2=%f' % (var_i[0], var_i[1], var_i[2]))

    # nonpraametric_stat_tests(pparts[0],pparts[1])

    if ONLY_STATIONARITY: sys.exit()


if __name__ == '__main__':
    tst_mann_whitney_u_test()
