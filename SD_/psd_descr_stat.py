import inspect
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
from psd_dat import *

@dataclass
class DescrStat:
    # работа с 1 мерными массивами NumPy
    name: str =''
    n: int = None # длина массива - x.size
    min_: float = None
    max_: float = None
    range_: float = None
    mean_: float = None
    st_mean_: float = None # стандартная ошибка среднего
    median_: float = None
    quant_025: float = None # 0.25 квантиль, 1 квартиль
    quant_050: float = None # 0.50 квантиль, 2 квартиль
    quant_075: float = None # 0.75 квантиль, 3 квартиль
    std_vib: float = None # стандартное отклонение выборки, ddof=1
    std_gen: float = None # стандартное отклонение ген.совокуп, ddof=0
    var_vib: float = None # дисперсия выборки, ddof=1
    var_gen: float = None # дисперсия ген.совокуп, ddof=0
    skew_biasT: float = None# Асимметрия скорректированное за статист. смещение
    skew_biasF: float = None # Асимметрия НЕскорректированное за статист. смещение
    kurt_biasT: float = None# Эксцесс скорректированн за статист. смещение
    kurt_biasF: float = None # Эксцесс НЕскорректированн за статист. смещение
    # Как рассчитать доверительные интервалы в Python, 17 авг. 2022 г.
    # https://www.codecamp.ru/blog/confidence-intervals-python/
    # считаю с использованием t-распеределения
    min95_confid: float = None # минимальное значение 95% доверительного интервала для среднего
    max95_confid: float = None # минимальное значение 95% доверительного интервала для среднего

DescrStat_lst:list[DescrStat]=[]


def calc_coeff_corr(y:np.ndarray, y1:np.ndarray,annotation:str=''):
    '''
    Calculate the Pearson Correlation Coefficient in Python
    https://datagy.io/python-pearson-correlation/
    How to Calculate Pearson Correlation Coefficient in SciPy
    '''
    print('\nFunction',inspect.currentframe().f_code.co_name)
    if annotation !='': print(annotation)
    print(np.shape(y),np.shape(y1),' y1.max()=',y1.max())
    r = stats.pearsonr(y,y1)
    rl = list(r)
    print('Коэффициент корреляции r=',r)
    print('Коэффициент корреляции rl=',rl)
    r2=rl[0]**2
    print('Коэффициент детерминации r^2=',r2)
    print('-----------------------')


# stat_after_transform(lvl=[mpstd3, mpstd6],ini=ini_sd,trf=[ma_arr3, ma_arr7],names=[lblmax3, lblmax6])
def stat_after_transform(lvl:list, ini:np.ndarray,trf:list[np.ndarray],names:list,x:np.ndarray):
    print('\nFunction', inspect.currentframe().f_code.co_name)
    # plt.plot(x,ini,label='ini')


    for i,dat in enumerate(lvl):
           print(i,f' level (mean+std*N)= ',dat)
           print('ini above level = ',len(ini[ini>dat]))
           for j,datj in enumerate(trf):
               dd=datj[datj > dat]
               namestr=f'{j} trf win {all_window_size[j]}  {names[j]}  {len(dd)=}'
               # print(j,' trf win '+str(all_window_size[j])+' '+names[j], len(dd))
               print(namestr)
               if j>0:
                   x2, y2 = extract_arr(x,ini, dat)
                   # print_2_arr(x2,y2,'ini > dat')
                   x2, y2 = extract_arr(x,datj,dat)
                   # plt.plot(x2,y2,linestyle=main_stl_[j])
                   # print_2_arr(x2,y2,namestr+f' > dat')
    # plt.legend()
    # plt.show()


def print_2_arr(x,y,name):
    print(name)
    for i in range(len(x)):
        print(i,x[i],y[i])
    print('\n')

def extract_arr(x:np.ndarray, y:np.ndarray, limit)->(np.ndarray,np.ndarray):
    x1=[];y1=[]
    for i in range(len(x)):
        if y[i]>limit:
            x1.append(x[i])
            y1.append(y[i])
    return np.array(x1), np.array(y1)


def create_stddev_lines(n:int, theDescrStat:DescrStat)->(np.ndarray,np.ndarray, float, float, str, str):
    mean_ = theDescrStat.mean_
    std_vib = theDescrStat.std_vib
    st3 = 3*std_vib
    st6 = 6*std_vib
    minl3 = np.ones(n)*(mean_ - st3); lblmin3 = 'mean-3std'
    maxl3 = np.ones(n)*(mean_ + st3); lblmax3 = 'mean+3std'
    maxl6 = np.ones(n)*(mean_+st6); lblmax6 = 'mean+6std'
    return maxl3, maxl6, mean_+st3, mean_+st6, lblmax3, lblmax6

def calc_descr_stat(x, name='', is_view=False):
    '''
    Расчет описательной статистики, результирующие параметры аналогичны Excel
    '''
    ptp_ = mean_ = st_mean_ = median_ = quant_025 = quant_050 = quant_075 = std_vib = std_gen = None
    var_vib = var_gen = min95_confid = max95_confid = None
    s = ''
    name_ = name
    n_ = x.size
    min_ = float(np.min(x))
    max_ = float(np.max(x))
    equal_data = abs(min_ - max_) < 1e-38
    if not equal_data:
        ptp_ = float(np.ptp(x))
        mean_ = float(np.mean(x))
        st_mean_ = float(scipy.stats.sem(x)) # стандартная ошибка среднего
        median_ = float(np.median(x))
        quant_025 = float(np.quantile(x, 0.25))  # linear распределение по умолчанию
        quant_050 = float(np.quantile(x, 0.50))  # linear распределение по умолчанию
        quant_075 = float(np.quantile(x, 0.75))  # linear распределение по умолчанию
        std_vib = float(np.std(x,ddof=1))       # стандартное отклонение выборки, ddof=1
        std_gen = float(np.std(x,ddof=0))       # стандартное отклонение ген.совокуп, ddof=0
        var_vib = float(np.var(x,ddof=1))       # дисперсия выборки, ddof=1
        var_gen = float(np.var(x,ddof=0))       # дисперсия ген.совокуп, ddof=0

        # skew_biasT = scipy.stats.skew(x, bias=True) # Асимметрия скорректированное за статист. смещение
        # skew_biasF = float(scipy.stats.skew(x, bias=False)) # Асимметрия НЕскорректированное за статист. смещение
        # kurt_biasT: float = float(kurtosis(x, bias=True)) # Эксцесс скорректированн за статист. смещение
        # kurt_biasF: float = float(kurtosis(x, bias=False)) # Эксцесс НЕскорректированн за статист. смещение

        ttt = scipy.stats.t.interval(confidence=0.95, df=n_-1, loc=mean_, scale=st_mean_)
        min95_confid = ttt[0] # минимальное значение 95% доверительного интервала для среднего
        max95_confid = ttt[1] # максимальное значение 95% доверительного интервала для среднего

    the_descr_stat = DescrStat(name_, n_, min_, max_, ptp_, mean_, st_mean_, median_, quant_025, quant_050,
                               quant_075, std_vib, std_gen, var_vib, var_gen, min95_confid, max95_confid)
    s +=f'Описательная статистика для {name_} \n'
    s+= f'Отсчетов = {n_} \n'
    s+= f'min      = {min_} \n'
    s+= f'max      = {max_} \n'
    if not equal_data:
        s+= f'max-min    = {max_-min_} \n'
        s+= f'range(ptp) = {ptp_} \n'
        s+= f'mean    = {mean_} \n'
        s+= f'стандартная ошибка среднего = {st_mean_} \n'
        s+= f'median  = {median_} \n'
        s+= f'quantile 0.25 (linear) = {quant_025} \n'
        s+= f'quantile 0.50 (linear) = {quant_050} \n'
        s+= f'quantile 0.75 (linear) = {quant_075} \n'

        s+= f'стандартное отклонение выборки = {std_vib} \n'
        s+= f'стандартное ген.совокупности   = {std_gen} \n'
        s+= f'дисперсия выборки              = {var_vib} \n'
        s+= f'дисперсия ген.совокупности     = {var_gen} \n'

        # s+= f'Асимметрия скорректированное за статист. смещение = {skew_biasT} \n'
        # s+= f'Асимметрия НЕскорректированное за статист. смещение = {skew_biasF} \n'
        # s+= f'Эксцесс скорректированн за статист. смещение = {kurt_biasT} \n'
        # s+= f'Эксцесс Нескорректированн за статист. смещение = {kurt_biasF} \n'
        s+= f'Минимальное значение 95% доверительного интервала для среднего = {min95_confid} \n'
        s+= f'Максимальное значение 95% доверительного интервала для среднего = {max95_confid} \n \n'

    if is_view:
        print(s)
    return the_descr_stat, s
