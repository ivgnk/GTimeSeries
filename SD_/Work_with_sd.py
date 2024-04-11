import sys

import numpy as np
import matplotlib.pyplot as plt

from psd_dat import *
from psd_graphics import *
from psd_proc import *
from psd_descr_stat import *
from psd_linear_regression import *
from psd_filters import *
from psd_scedasticity import *
from psd_time_series import *

# --(1)-- ввод и визуализация исходных данных
lenx, znp, ini_sd  = create_dat()

if ONLY_SCEDASTICITY: test_of_white(znp, ini_sd) # в конце выход из программы
if ONLY_STATIONARITY: stationarity_test(ini_sd,znp)

GraphCl_lst.append(GraphCl('St.dev.',znp,ini_sd))
view_datetime_sd(GraphCl_lst,with_inset=True)
# view_datetime_sd_parts(GraphCl_lst)

# --(2)-- Описательная статистика
the_descr_stat, s =  calc_descr_stat(ini_sd, name='Стандарное отклонение отсчетов, мГал', is_view=True)
maxl3, maxl6, mpstd3, mpstd6, lblmax3, lblmax6 = create_stddev_lines(lenx, the_descr_stat)

# print(s)
# --(3)-- Linear Regression
lst = proc_OLS(ini_sd,znp)
# gr1 = znp*lst[0]+lst[1]; GraphCl_lst.append(GraphCl('gr1',znp,gr1))
gr2 = znp*lst[1]+lst[0] # прямая линия

calc_coeff_corr(ini_sd,gr2,'НЕлогарифмированные SD')
calc_coeff_corr(ini_sd,np.log10(gr2),'Логарифмированные SD')

window_size = all_window_size[0] # 3
ma_arr3 = MA_equal_weight(ini_sd, window_size); GraphCl_lst.append(GraphCl('MA'+str(window_size),znp,ma_arr3))
window_size = all_window_size[1] # 7
ma_arr7 = MA_equal_weight(ini_sd, window_size); GraphCl_lst.append(GraphCl('MA'+str(window_size),znp,ma_arr7))

GraphCl_lst.append(GraphCl(lblmax3,znp,maxl3))
GraphCl_lst.append(GraphCl(lblmax6,znp,maxl6))

stat_after_transform(lvl=[mpstd3, mpstd6],ini=ini_sd,trf=[ma_arr3, ma_arr7],names=[lblmax3, lblmax6],x=znp)
GraphCl_lst.append(GraphCl('Linear trend',znp,gr2))
# view_datetime_sd(GraphCl_lst,with_inset=False,with_legend=True)
# sys.exit()

print(f'Параметры линии тренда \n  {np.min(gr2)=}   {np.max(gr2)=}  \n  {np.mean(gr2)=}   {np.median(gr2)=} \n')
view_datetime_sd(GraphCl_lst,True)

# scedasticity(znp,ini_sd,gr2)
test_of_white(znp,ini_sd)
spearman_rank_correlation(znp,ini_sd)

print('Normal shut down')