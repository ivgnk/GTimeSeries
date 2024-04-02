'''
Использование разных функций из numpy и scipy.
Преимущественно фильтры
'''
import inspect
import sys

import numpy as np
import scipy as sc
# from bp_const import *
from random import *
from math import *
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Final, List
from pstring import *
import pandas as pd
# from astropy.modeling.models import *  # для задания гауссова распределения весов окна

from scipy import signal
from scipy.signal import butter, bessel, cheby1, cheby2, ellip
# from numba import njit


# Для фильтраций временных рядов
SMA_filtname =['Moving average','Moving average with weights (triangle)','Savitzky–Golay filter','Wiener filter']
SOS_filtname = ['Butterworth filter','Bessel filter']
ln_style_st = ['solid','solid','solid','solid']
ln_style = ['solid','dashed','dotted','dashdot']
SMA_all_w = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
SMA_w: Final = [3, 7, 13, 25, 51]  # SMA - Simple Moving  Average,  размер окон
SMA_wt = tuple(SMA_w)

figsize_st = (15, 12) # для качественной вставки в презентацию
# figsize_st2 = (12, 10) # для качественной вставки в презентацию


def remove_letters(s)->str:
    return ''.join(filter(str.isdigit, s))

def calc_triang_weights_for_1Dfilter(win_size:int)->np.ndarray:
    '''
    Расчет значений треугольной функции для фильтра
    не проверяем win_size, оно должно быть нечетным
    '''
    x = np.array([0.0 for i in range(win_size)])
    half_win = win_size // 2
    frst_odd = 1
    for i in range(x.size):
        if i <= half_win:  # левая половина массива, включая центр
            x[i] = frst_odd
            frst_odd += 2
        else:
            x[i] = x[half_win-(i-half_win)]  # правая половина массива, без центра
    return x

def norm_weights_for_1Dfilter(dat: np.ndarray) -> np.ndarray:
    '''
    Нормализация весов для 1D фильтра
    '''
    return dat/np.sum(dat)


def test_smoothing():
    '''
    Просто проверка работоспособности
    '''
    x = np.array([i for i in range(10)])
    y = np.array([random() for i in range(10)])
    res = work_with_smoothing(x, y)


def work_with_smoothing2sos(x: np.ndarray, y: np.ndarray, with_edges=False)->(int, int, np.ndarray, list):
    '''
    Сглаживание различными фильтрами
    Return:
    num_flt - всего вариантов
    curr_flt - всего подвариантов
    res - массив для визуализации с указанием варианта, подварианта, строкового названия, строковых параметров
    llst - список подвариатов для каждого варианта
    '''
    n1 = 200 # берем с запасом, чтобы динамически не расширять
    res = np.ndarray([n1, 5], dtype=object)
    # первый столбец (0) - номер фильтра,
    # второй столбец (1) - название фильтра
    # третий столбец (2) - номер подварианта фильтра
    # четвертый столбец (3) - результат фильтрации
    # пятый столбец (4) - параметры фильрации
    curr_flt = 0
    num_flt = 0
    llst =[]
    # ---- (00) - 'butter'
    (num_flt, curr_flt, res, llst) = prep_my_butter1D_filter(num_flt, curr_flt, y, res, llst, with_edges)
    # ---- (01) - 'bessel'
    (num_flt, curr_flt, res, llst) = prep_my_bessel1D_filter(num_flt, curr_flt, y, res, llst, with_edges)
    # # ---- (O2) -
    # (num_flt, curr_flt, res, llst) = prep_savgol(num_flt, curr_flt, y, res, llst)
    # # ---- (03) -
    # (num_flt, curr_flt, res, llst) = prep_wiener1D_filter(num_flt, curr_flt, y, res, llst)
    return num_flt, curr_flt, res, llst

def work_with_smoothing(x: np.ndarray, y: np.ndarray, with_edges=False)->(int, int, np.ndarray, list):
    '''
    Сглаживание различными фильтрами
    Return:
    num_flt - всего вариантов
    curr_flt - всего подвариантов
    res - массив для визуализации с указанием варианта, подварианта, строкового названия, строковых параметров
    llst - список подвариатов для каждого варианта
    '''
    # x фактически не нужен, т.к. считаем данные на равномерной 1-Д сетке
    # num_flt - число разных фильтров
    # curr_flt - общее число с подвариантами
    # res - массив numpy
    # list - число подвариантов для каждого варианта
    n1 = 200 # берем с запасом, чтобы динамически не расширять
    res = np.ndarray([n1, 5], dtype=object)
    # первый столбец (0) - номер фильтра,
    # второй столбец (1) - название фильтра
    # третий столбец (2) - номер подварианта фильтра
    # четвертый столбец (3) - результат фильтрации
    # пятый столбец (4) - параметры фильрации
    # print(inspect.currentframe().f_code.co_name)

    curr_flt = 0
    num_flt = 0
    llst =[]
    # ---- (00) - усреднение равновесовое
    (num_flt, curr_flt, res, llst) = prep_my_moving_average1D_filter(num_flt, curr_flt, y, res, llst, with_edges)
    # ---- (01) - усреднение c треугольными весами
    (num_flt, curr_flt, res, llst) = prep_my_moving_average1D_filter_ww(num_flt, curr_flt, y, res, llst, with_edges)
    # ---- (O2) - sc.signal.savgol_filter     https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    (num_flt, curr_flt, res, llst) = prep_savgol(num_flt, curr_flt, y, res, llst)
    # ---- (03) - wiener1D_filter
    (num_flt, curr_flt, res, llst) = prep_wiener1D_filter(num_flt, curr_flt, y, res, llst)
    # ---- (04) -
    # for i in range(curr_flt):  print(res[i])
    return num_flt, curr_flt, res, llst

# Фильтр Баттерворта
def prep_butter1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list)->(int, int, np.ndarray, list):
    '''
    Образец tst_butterworth_filter2.py - > Python_Scipy_Filters
    '''
    num_flt += 1; npvar = 1
    Wn_ = 20; btype_= 'low' # 'lowpass'
    for i in range(npvar):
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SOS_filtname[0]  # 'Butterworth filter'
        res[curr_flt, 2] = i + 1
        sos = butter(N=15, Wn=Wn_, btype=btype_, fs=2000, output='sos')  # butter, bessel
        filtd = signal.sosfilt(sos, dat)  # dat = signal_
        res[curr_flt, 3] = filtd
        # print(f'{filtd=}'); print(f'{dat=}'); input()
        res[curr_flt, 4] = SOS_filtname[0]+' '+str(i)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst

# Фильтр Бесселя
def prep_bessel1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list)->(int, int, np.ndarray, list):
    '''
    Образец tst_butterworth_filter2.py - > Python_Scipy_Filters
    '''
    num_flt += 1; npvar = 1
    Wn_ = 20; btype_= 'lowpass'
    for i in range(npvar):
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SOS_filtname[0]  # 'Butterworth filter'
        res[curr_flt, 2] = i + 1
        sos = bessel(N=15, Wn=Wn_, btype=btype_, fs=2000, output='sos')  # butter, bessel
        filtd = signal.sosfilt(sos, dat)  # dat = signal_
        res[curr_flt, 3] = filtd
        res[curr_flt, 4] = SOS_filtname[0]+' '+str(i)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst


# num_flt - число разных фильтров
# curr_flt - общее число с подвариантами
# res - массив numpy
# list - число подвариантов для каждого варианта
def prep_savgol(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list)->(int, int, np.ndarray, list):
    '''
    Сшдаживание фильтром
    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-for-a-dataset
    '''
    num_flt += 1
    npvar = len(SMA_w)# число подвариантов
    for i in range(npvar):
        win_len = SMA_w[i]
        polyorder_ = get_poly_order_for_savgol(win_len)
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SMA_filtname[2] # 'Savitzky–Golay filter'
        res[curr_flt, 2] = i + 1
        if win_len >= dat.size:
            print(inspect.currentframe().f_code.co_name)
            print('Ошибка: вариант = ',i,'длина окна = ',win_len,'длина набора данных = ',dat.size)
            res[curr_flt, 3] = None
            res[curr_flt, 4] = ''
        else:
            res[curr_flt, 3] = sc.signal.savgol_filter(dat, window_length=win_len, polyorder=polyorder_)
            res[curr_flt, 4] = 'win ' + str(win_len)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst

def prep_my_butter1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray,
                                    llst:list, with_edges=False)->(int, int, np.ndarray, list):
    '''
    'Butterworth filter'
    '''
    num_flt += 1
    npvar = 1 # число подвариантов
    Wn_ = 20;  btype_ = 'lowpass'
    for i in range(npvar):
        win_len = SMA_w[i]
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SOS_filtname[0]  # 'Butterworth filter'
        res[curr_flt, 2] = i + 1
        sos = butter(N=15, Wn=Wn_, btype=btype_, fs=2000, output='sos')  # butter, bessel
        filtd = signal.sosfilt(sos, dat)  # dat = signal_
        res[curr_flt, 3] = filtd
        res[curr_flt, 4] = 'no win'
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst

def prep_my_bessel1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray,
                                    llst:list, with_edges=False)->(int, int, np.ndarray, list):
    '''
    'Bessel filter'
    '''
    num_flt += 1
    npvar = 1 # число подвариантов
    Wn_ = 20;  btype_ = 'lowpass'
    for i in range(npvar):
        win_len = SMA_w[i]
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SOS_filtname[1]  # 'Bessel filter'
        res[curr_flt, 2] = i + 1
        sos = bessel(N=15, Wn=Wn_, btype=btype_, fs=2000, output='sos')  # butter, bessel
        filtd = signal.sosfilt(sos, dat)  # dat = signal_
        res[curr_flt, 3] = filtd
        res[curr_flt, 4] = 'no win'
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst



def prep_my_moving_average1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray,
                                    llst:list, with_edges=False)->(int, int, np.ndarray, list):
    '''
    Усреднение без весов (равновесовым фильтром)
    '''
    num_flt += 1
    npvar = len(SMA_w)# число подвариантов
    for i in range(npvar):
        win_len = SMA_w[i]
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SMA_filtname[0]  #  'Moving average'
        res[curr_flt, 2] = i + 1
        res[curr_flt, 3] = my_moving_average1D(dat, window_size=win_len, with_edges=with_edges)
        res[curr_flt, 4] = 'win ' + str(win_len)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst

def prep_wiener1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list)->(int, int, np.ndarray, list):
    '''
    Усреднение без весов (равновесовым фильтром)
    '''
    num_flt += 1
    npvar = len(SMA_w)# число подвариантов
    for i in range(npvar):
        win_len = SMA_w[i]
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SMA_filtname[3] # 'Wiener filter'
        res[curr_flt, 2] = i + 1
        res[curr_flt, 3] = sc.signal.wiener(dat, win_len)
        res[curr_flt, 4] = 'win ' + str(win_len)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst


def prep_my_moving_average1D_filter_ww(num_flt:int, curr_flt:int, dat:np.ndarray,
                                       res:np.ndarray, llst:list, with_edges=False)->(int, int, np.ndarray, list):
    '''
    Усреднение с весами (треугольным фильтром)
    '''
    num_flt += 1
    npvar = len(SMA_w)# число подвариантов
    for i in range(npvar):
        win_len = SMA_w[i]
        wght = calc_triang_weights_for_1Dfilter(win_len)
        the_weights = norm_weights_for_1Dfilter(wght)
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = SMA_filtname[1] # 'Moving average with weights (triangle)'
        res[curr_flt, 2] = i + 1
        res[curr_flt, 3] = my_moving_average1D(dat, window_size=win_len, the_weights = the_weights, with_edges=with_edges)
        res[curr_flt, 4] = 'win ' + str(win_len)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst


def get_poly_order_for_savgol(window_length: int)->int:
       if window_length>30:
           res = 6
       elif window_length>20:
           res = 5
       elif window_length>10:
           res = 4
       elif window_length > 3:
           res = 3
       else:
           res = 2
       return res

def get_subplot_num(npictures:int)->tuple:
    '''
    Задает разбиение окна на части в зависимости от числа картинок
    '''
    clmn = 4
    if npictures < 1:
        raise ValueError('Число фильтров должно быть больше 0')
    if npictures == 1:
       res = (1,1)
    elif npictures == 2:
       res = (1,2)
    elif (npictures == 3) or (npictures == 4):
       res = (2, 2)
    elif (npictures % clmn) == 0:
        res = ((npictures // clmn), clmn)
    else:
        res = ((npictures // clmn) + 1, clmn)
    return res

def get_subplot_num2(npictures:int)->tuple:
    '''
    Задает разбиение окна на части в зависимости от числа картинок
    '''
    lim = 16
    if npictures < 1:
        print('Ошибка: Число subplot должно быть больше 0')
        raise ValueError('Число subplot должно быть больше 0')
    elif npictures >lim:
        print(f'Ошибка: Число subplot должно быть меньше {lim}')
        raise ValueError(f'Число subplot должно быть меньше {lim}')

    match npictures:
        case 1: res = (1,1) # row col
        case 2: res = (1,2)
        case 3 | 4: res = (2, 2)
        case 5 | 6: res = (2, 3)
        case 7 | 8 | 9: res = (3, 3)
        case 10: res = (2, 5)
        case 11 | 12: res = (4, 3)
        case 13 | 14 | 15 : res = (5, 3)
        case 16 : res = (4, 4)
        case default: res=tuple()
    return res

def test_subplot_num():
    '''
    Проверка функции get_suplotsize
    '''
    for i in range(1,20):
        print(i,' ', get_subplot_num(i))

def test_subplot_num2():
    for i in range(1,20):
        print(i,' ', get_subplot_num2(i))


def create_signals(len_filter=140, the_test=False)->(np.ndarray, np.ndarray):
    seed(123)
    x = np.array([i for i in range(len_filter)])  # same as  x1 = np.linspace(0,len_filter,len_filter, endpoint=False)
    y = np.array([random()*10 for i in range(len_filter)])
    if the_test:
        x1 = np.linspace(0,len_filter,len_filter, endpoint=False)
        x2 = np.linspace(0,len_filter,len_filter, endpoint=True)
        x3 = np.linspace(0,len_filter,len_filter)
        print(f'{len(x)=}  {len(x1)=}  {len(x2)=}  {len(x3)=}  {np.sum(x-x1)=}')
        print(f'{x[len_filter-1]=}  {x1[len_filter-1]=}  {x2[len_filter-1]=}  {x3[len_filter-1]=}')
        print(f'{x[1]=}  {x1[1]=}  {x2[1]=}  {x3[1]=}')

        # print('len(x)=',len(x),'  len(x1)=',len(x1),'  len(x2)=',len(x2), '  len(x3)=',len(x3),' np.sum(x-x1)=',np.sum(x-x1))
        # print('x[len_filter-1]=  x1[len_filter-1]=  x2[len_filter-1]=  x3[len_filter-1]=', x[len_filter-1],x1[len_filter-1], x2[len_filter-1], x3[len_filter-1])
        # print('x[1]=  x1[1]=  x2[1]=  x3[1]=',x[1],x1[1],x2[1],x3[1])

    return x,y

def tst_create_signals():
    print('Function ',inspect.currentframe().f_code.co_name)
    create_signals(len_filter=140, the_test=True)

def test_filters_with_sos2(len_filter=140, with_legend=False, with_edges=bool(1)):
    x, y = create_signals(len_filter)
    # Это основная часть, все остальное - для визуализации
    ########################################################
    (num_flt, curr_flt, res, llst) = work_with_smoothing2sos(x, y, with_edges)
    # print(f'{num_flt=}')
    # print(f'{res=}')
    ########################################################
    currflt = 0
    res_corr_coeff=[]
    for i in range(num_flt): # перебор по всем фильтрам
        for j in range(llst[i]): # перебор по всем подвариантам фильтров
            if res[currflt,4] != '':
                curr_s = res[currflt, 4]
                # chngd_s = remove_letters(curr_s)
                # print(f'{i=} {chngd_s=}')
                # print(f'{y=}')
                # print(f'{res[currflt, 3]=}')
                corr_coeff = np.corrcoef(y, res[currflt, 3])
                # print(f'{corr_coeff=}')
                # https://www.codecamp.ru/blog/correlation-in-python/
                cc = corr_coeff[0, 1]
                # print(i, res[currflt, 1], '  corr_coeff=', cc)
                res_corr_coeff.append([len_filter,res[currflt, 1],cc])
                currflt += 1
    # print(f'Good return')
    return res_corr_coeff

def test_filters_with_len_window2(len_filter:int=140, len_flt_tpl:tuple=SMA_wt, with_legend:bool = True, with_edges:bool=False)->list:
    # print('Function = test_filters_with_len_window2')
    # print(f'{len_filter=}')
    # print('Function ',inspect.currentframe().f_code.co_name)
    x,y = create_signals(len_filter)
    ########################################################
    # Это основная часть, все остальное - для визуализации
    (num_flt, curr_flt, res, llst) = work_with_smoothing(x, y,with_edges)
    ########################################################
    currflt = 0
    res_corr_coeff=[]
    for i in range(num_flt): # перебор по всем фильтрам
        for j in range(llst[i]): # перебор по всем подвариантам фильтров
            if res[currflt,4] != '':
                curr_s = res[currflt, 4]
                chngd_s = remove_letters(curr_s)
                curr_lenflt = int(chngd_s)
                if curr_lenflt in len_flt_tpl:
                    # Пустая строка означает, что фильтр не рассчитался
                    corr_coeff = np.corrcoef(y, res[currflt, 3])
                    # https://www.codecamp.ru/blog/correlation-in-python/
                    cc = corr_coeff[0, 1]

                    corr_coeff2 = np.corrcoef(y, y-res[currflt, 3])
                    cc2 = corr_coeff2[0, 1]

                    res_corr_coeff.append([len_filter,res[currflt, 1],cc,cc2])
                currflt += 1
    return res_corr_coeff

def test_filters_with_len_window(len_filter=140, len_flt_tpl:tuple=SMA_wt, with_legend = True, with_edges=False):
    print('Function = ',inspect.currentframe().f_code.co_name)

    x,y = create_signals(len_filter)
    ########################################################
    # Это основная часть, все остальное - для визуализации
    (num_flt, curr_flt, res, llst) = work_with_smoothing(x, y,with_edges)
    ########################################################

    # print(num_flt, curr_flt, llst)
    sp = get_subplot_num(num_flt)
    plt.subplots( nrows = sp[0], ncols = sp[1], figsize=figsize_st) # для качественной вставки в презентацию
    currflt = 0
    for i in range(num_flt): # перебор по всем фильтрам
        plt.subplot(sp[0],sp[1], i + 1) #  facecolor=(0.95, 0.95, 0.95)  ,  facecolor=(0.6, 0.6, 0.6)
        # figure.add_subplot(1, 1, 1, axisbg='red')
        plt.title(res[currflt,1],color= 'b')
        plt.plot(x, y, label='ini', linewidth=4)
        for j in range(llst[i]): # перебор по всем подвариантам фильтров
            if res[currflt,4] != '':
                curr_s = res[currflt, 4]
                chngd_s = remove_letters(curr_s)
                curr_lenflt = int(chngd_s)
                if curr_lenflt in len_flt_tpl:
                    # Пустая строка означает, что фильтр не рассчитался
                    plt.plot(x, res[currflt,3], label = res[currflt,1]+' '+res[currflt,4])
                    corr_coeff = np.corrcoef(y, res[currflt,3])
                    # https://www.codecamp.ru/blog/correlation-in-python/
                    print(i, res[currflt, 1],'  corr_coeff=',corr_coeff[0,1])
                    plt.grid()
                    plt.xlabel('Номер отсчета', fontsize=9)
            currflt += 1
            # https://jenyay.net/Matplotlib/LegendPosition
            if with_legend: plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
    plt.show()
    # Обратные фильтры
    print()
    plt.subplots( nrows = sp[0], ncols = sp[1], figsize=figsize_st) # для качественной вставки в презентацию
    currflt = 0
    for i in range(num_flt): # перебор по всем фильтрам
        plt.subplot(sp[0],sp[1], i + 1) #  facecolor=(0.95, 0.95, 0.95)  ,  facecolor=(0.6, 0.6, 0.6)
        # figure.add_subplot(1, 1, 1, axisbg='red')
        plt.title(res[currflt,1],color= 'b')
        plt.plot(x, y, label='ini', linewidth=4)
        for j in range(llst[i]): # перебор по всем подвариантам фильтров
            if res[currflt,4] != '':
                curr_s = res[currflt, 4]
                chngd_s = remove_letters(curr_s)
                curr_lenflt = int(chngd_s)
                if curr_lenflt in len_flt_tpl:
                    # Пустая строка означает, что фильтр не рассчитался
                    plt.plot(x, y-res[currflt,3], label = res[currflt,1]+' '+res[currflt,4]+'_HF')
                    corr_coeff = np.corrcoef(y, y-res[currflt,3])
                    # https://www.codecamp.ru/blog/correlation-in-python/
                    print(i, res[currflt, 1],'  corr_coeff_HF=',corr_coeff[0,1])
                    plt.grid()
                    plt.xlabel('Номер отсчета', fontsize=9)
            currflt += 1
            # https://jenyay.net/Matplotlib/LegendPosition
            if with_legend: plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
    plt.show()

def test_filters_with_sos(len_filter=140, with_legend=False, with_edges=bool(1)):
    x,y = create_signals(len_filter)
    ########################################################
    # Это основная часть, все остальное - для визуализации
    (num_flt, curr_flt, res, llst) = work_with_smoothing2sos(x, y, with_edges)
    ########################################################

    # print(num_flt, curr_flt, llst)
    sp = get_subplot_num(num_flt)
    plt.subplots( nrows = sp[0], ncols = sp[1], figsize=figsize_st) # для качественной вставки в презентацию
    currflt = 0
    for i in range(num_flt): # перебор по всем фильтрам
        plt.subplot(sp[0],sp[1], i + 1) #  facecolor=(0.95, 0.95, 0.95)  ,  facecolor=(0.6, 0.6, 0.6)
        # figure.add_subplot(1, 1, 1, axisbg='red')
        plt.title(res[currflt,1],color= 'b')
        plt.plot(x, y, label='ini', linewidth=4)
        for j in range(llst[i]): # перебор по всем подвариантам фильтров.
            # Пока 1 подвариант
            if res[currflt,4] != '':
                    # Пустая строка означает, что фильтр не рассчитался
                    plt.plot(x, res[currflt,3], label = res[currflt,1]) # +' '+res[currflt,4]
                    corr_coeff = np.corrcoef(y, res[currflt,3])
                    # https://www.codecamp.ru/blog/correlation-in-python/
                    print(i, res[currflt, 1],'  corr_coeff=',corr_coeff[0,1])
                    plt.grid()
                    plt.xlabel('Номер отсчета', fontsize=9)
            currflt += 1
            # https://jenyay.net/Matplotlib/LegendPosition
            if with_legend: plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
    plt.show()


def create_edges(dat:np.ndarray, half_win:int)->np.ndarray:
    left_edg = dat[0];            right_edg = dat[-1]
    larr = [left_edg]*half_win;   rarr = [right_edg]*half_win
    b1 = np.hstack((larr, dat, rarr))
    return b1

def restore_edges(dat:np.ndarray, half_win:int)->np.ndarray:
    old_r_edg = len(dat)-half_win
    k,l,m = np.hsplit(dat, [half_win, old_r_edg])
    return l

def my_moving_average1D(dat:np.ndarray, window_size:int, the_weights = None, with_edges=False)->np.ndarray:
    # print(inspect.currentframe().f_code.co_name)
    # Скользящее среднее 1 мерного массива
    # края оставляем как есть
    # веса в окне равные
    # окно только нечетной длины
    # не проверяем длину массива и окна
    half_win = window_size // 2
    if with_edges: dat =create_edges(dat, half_win)
    moving_avg = deepcopy(dat)
    llen = dat.size
    ffirst = half_win
    llast = llen - ffirst
    # Проходим по нужным элементам в массиве, края уже заполнены deepcopy
    for i in range(ffirst,llast,1):
        # Добавляем текущий элемент к скользящему среднему
        win_dat = dat[i-half_win:i+half_win+1]
        moving_avg[i] = np.average(win_dat, weights=the_weights)
    if with_edges: moving_avg =restore_edges(moving_avg, half_win)
    return moving_avg


def my_bilinear(dat:np.ndarray)->np.ndarray:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear.html
    pass

def get_lin_arr(llen_arr:int)->np.ndarray:
    # линейно возрастающая последовательность
    x = np.array([i for i in range(llen_arr)])
    return x

def get_random_arr(llen_arr:int)->np.ndarray:
    # случайная последовательность
    arr =  np.array([random()*10 for i in range(llen_arr)])
    return arr

def test_my_moving_average1D():
    # Пример использования функции для вычисления скользящего среднего
    llen = 40
    x = get_lin_arr(llen)
    arr =  get_random_arr(llen)
    plt.subplots(figsize=(15, 8))
    plt.plot(x, arr, label = 'ini', linewidth=5)
    win_lst= get_win_fltr_lst(5)
    print(win_lst)
    for i in win_lst:
        for j in range(2):
            if j==0:
                weights =  None; s = ''
            else:
                weights = calc_triang_weights_for_1Dfilter(i)
                s =' w'
            moving_avg_arr = my_moving_average1D(arr, i, weights)
            plt.plot(x, moving_avg_arr, label='win '+str(i)+s)
    plt.legend()
    plt.show()


def get_win_fltr_lst(num:int)->list:
    # список длин фильтров
    curr_len = 3
    llst=[]
    for i in range(num):
        llst=llst+[curr_len]
        curr_len += 2
    return llst

def test_get_win_fltr_lst():
    print(inspect.currentframe().f_code.co_name)
    for i in range(1,7):
        llst = get_win_fltr_lst(i)
        print(i,' ',llst)

def test_calc_triang_weights_for_filter1D():
    # dat
    len_fltr_lst = get_win_fltr_lst(5)
    arr = get_random_arr(40)
    for i in len_fltr_lst:
        x = calc_triang_weights_for_1Dfilter(i)
        weights = norm_weights_for_1Dfilter(x)
        moving_avg_arr = my_moving_average1D(arr, i, weights)

def calc_lst_of_triang_weights(n:int, win_size_lst:list = SMA_all_w)->list:
    llst = []
    for i in range(n):
        llen = win_size_lst[i]
        x = calc_triang_weights_for_1Dfilter(llen)
        weights = norm_weights_for_1Dfilter(x)
        llst.append(weights)
    return llst


def test_calc_lst_of_triang_weights():
    n = 5
    llst = calc_lst_of_triang_weights(n)
    plt.subplots(figsize=(15, 8))
    for i in range(n):
        print(i,' ', llst[i], '  ', np.sum(llst[i]))


def test_numpy_average():
    dat_len = 40
    arr = get_random_arr(dat_len)

    for i in range(1,dat_len):
        num = i % 2
        if num !=0:
            curr_arr=arr[0:i]
            res = np.average(curr_arr)
            wght = np.ones(curr_arr.size)
            wght = calc_triang_weights_for_1Dfilter(i)

            the_weights = norm_weights_for_1Dfilter(wght)
            print(i,' ',num,' ', the_weights)
            res1 = np.average(curr_arr, weights=the_weights)
            print(res, '  ', res1)

def test_wiener_filter():
    rng = np.random.default_rng()
    n = 50
    x = np.linspace(1,n,n)
    dat = rng.random(n)  # Create a random data
    plt.figure(figsize=(12,6))
    plt.plot(x,dat,label='ini')
    for i in [5,9,13,17]:
        filtered_dat = sc.signal.wiener(dat, i)  # Filter the image
        plt.plot(x,filtered_dat,label='wiener '+str(i))
    plt.grid()
    plt.legend()
    plt.show()

def tst_test_filters_with_sos2():
    print(inspect.currentframe().f_code.co_name)
    lst = list(range(100,6000,40))
    lenlst = len(lst)
    print(lst)
    df = pd.DataFrame(columns=["Filter length", "Filter name", "Correlation coefficient"])
    print(type(df))
    for i,dat in enumerate(lst):
        if i % 20 == 0: print(f'{i} / {lenlst}')
        lst2 = test_filters_with_sos2(len_filter=dat, with_legend=False, with_edges=bool(1)) # Good
        for dat2 in lst2:
            df.loc[len(df.index)] = dat2
    workvisu_with_stat_pd(df, SOS_filtname)
    print('End of tst_test_filters_with_sos2')

def tst_test_filters_with_len_window2():
    print('Function ',inspect.currentframe().f_code.co_name)
    lst = list(range(100,6000,40))
    lenlst = len(lst)
    print(lst)
    df = pd.DataFrame(columns=["Filter length", "Filter name", "Correlation coefficient", "Correlation coefficient_obr"])
    print(type(df))
    for i,dat in enumerate(lst):
        if i % 20 == 0: print(f'{i} / {lenlst}')
        lst2 = test_filters_with_len_window2(len_filter=dat, len_flt_tpl=(13,), with_legend=False, with_edges=bool(1)) # Good
        for dat2 in lst2:
            df.loc[len(df.index)] = dat2
    workvisu_with_stat_pd(df, SMA_filtname)
    print('End of tst_test_filters_with_len_window2')

def workvisu_with_stat_pd(df, flt_names:list,linestyle_=ln_style_st):
    print(f'{len(df)}')
    plt.figure(figsize=(12, 8))
    for i,fn_ in enumerate(flt_names):
        res1 = df.loc[df['Filter name'] == fn_]
        # print(f'{i=}  {len(res1)}')
        # print(res1)
        plt.plot(res1['Filter length'], res1['Correlation coefficient'], label = fn_, linestyle=linestyle_[i])
        plt.plot(res1['Filter length'], res1['Correlation coefficient_obr'], label=fn_+'_obr', linestyle=ln_style[i])
    plt.xlabel('Длина выборки', fontsize=12) #
    plt.ylabel('Коэффициент корреляции', fontsize=12)
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # test_subplot_num2()
    # test_get_win_fltr_lst()
    # test_medfilt_filter()
    # test_my_moving_average1D()
    # test_calc_triang_weights_for_filter1D()

    # tst_test_filters_with_len_window2()

    # tst_test_filters_with_sos2()
    test_filters_with_len_window(140, len_flt_tpl=(13,), with_legend=False, with_edges=bool(1)) #

    # test_filters_with_sos(len_filter=140, with_legend=False, with_edges=bool(1))
    # test_calc_lst_of_triang_weights()
    # test_wiener_filter() # Good

    # tst_create_signals()

