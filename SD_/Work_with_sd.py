import numpy as np
import matplotlib.pyplot as plt

from psd_graphics import *
from psd_proc import *
from psd_descr_stat import *
from psd_linear_regression import *

# --(1)-- ввод и визуализация исходных данных
dat = the_xls_importdat(r'dat\127_sd.xlsx',is_view=False) # вывод первых 40 строк

x = dat[:,2]; lenx=len(x) # print(f'{lenx=}')
y = dat[:,0]; leny=len(y) # print(f'{leny=}')

z =[i for i in range(lenx)]; znp = np.array(z)
GraphCl_lst.append(GraphCl('ini',znp,y))


# --(2)-- Описательная статистика
the_descr_stat, s =  calc_descr_stat(y, name='Стандарное отклонение отсчетов, мГал', is_view=True)
# print(s)
# --(3)-- Linear Regression
lst = proc_OLS(y,znp)
# gr1 = znp*lst[0]+lst[1]; GraphCl_lst.append(GraphCl('gr1',znp,gr1))
gr2 = znp*lst[1]+lst[0]; GraphCl_lst.append(GraphCl('gr2',znp,gr2))
print(f'Параметры линии тренда \n  {np.min(gr2)=}   {np.max(gr2)=}  \n  {np.mean(gr2)=}   {np.median(gr2)=} \n')
view_datetime_sd(GraphCl_lst)

print('Normal shut down')