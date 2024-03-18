import numpy as np
import matplotlib.pyplot as plt

from psd_dat import *
from psd_graphics import *
from psd_proc import *
from psd_descr_stat import *
from psd_linear_regression import *
from psd_filters import *

# --(1)-- ввод и визуализация исходных данных
lenx, znp, ini_sd  = create_dat()
GraphCl_lst.append(GraphCl('ini',znp,ini_sd))


# --(2)-- Описательная статистика
the_descr_stat, s =  calc_descr_stat(ini_sd, name='Стандарное отклонение отсчетов, мГал', is_view=True)
maxl3, maxl6, mpstd3, mpstd6, lblmax3, lblmax6 = create_stddev_lines(lenx, the_descr_stat)

# print(s)
# --(3)-- Linear Regression
lst = proc_OLS(ini_sd,znp)
# gr1 = znp*lst[0]+lst[1]; GraphCl_lst.append(GraphCl('gr1',znp,gr1))
gr2 = znp*lst[1]+lst[0] # прямая линия
GraphCl_lst.append(GraphCl('gr2',znp,gr2))
GraphCl_lst.append(GraphCl(lblmax3,znp,maxl3))
GraphCl_lst.append(GraphCl(lblmax6,znp,maxl6))

window_size = 3; ma_arr3 = MA_equal_weight(ini_sd, window_size); GraphCl_lst.append(GraphCl('MA'+str(window_size),znp,ma_arr3))
window_size = 7; ma_arr7 = MA_equal_weight(ini_sd, window_size); GraphCl_lst.append(GraphCl('MA'+str(window_size),znp,ma_arr7))

stat_after_transform([mpstd3, mpstd6],ini_sd,[ma_arr3, ma_arr7],[lblmax3, lblmax6])

print(f'Параметры линии тренда \n  {np.min(gr2)=}   {np.max(gr2)=}  \n  {np.mean(gr2)=}   {np.median(gr2)=} \n')
view_datetime_sd(GraphCl_lst)


print('Normal shut down')