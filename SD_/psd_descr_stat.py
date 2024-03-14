from dataclasses import dataclass
import numpy as np
import scipy

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
