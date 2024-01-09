'''
Вычисления, которые не идут в основной программе TimeSeries_GUI.py
'''

from copy import deepcopy
import numpy as np
import pinp_struct
import pnumpy
import pmain_proc

from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy
# from scipy.fft import rfft, rfftfreq
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   # графики асf и pacf
from statsmodels.tsa.stattools import acf, pacf  # функции асf и pacf

import pfile
dat_dir=r'J:/Work-Lang2/Python/PyCharm/GTimeSeries/dat/'

fnames = ['Grav_.txt', 'sd_.txt', 'tiltx_.txt', 'tilty_.txt', 'temp_.txt', 'tide_.txt']
n_127 = 63382
n_65536 = 65536

def grav_linear_trend():
    dat_fn = dat_dir+fnames[0]
    grav = pnumpy.read_numpy(dat_fn)
    n = grav.size               #    print(n)
    lst = [i for i in range(n)] #    print(lst)
    x = np.array(lst)           #    print(x)
    print(x.size)
    print(x.shape)
    (r_sq, model_intercept_, model_coef_, y_pred, ost_anom) = pmain_proc.linear_trend(x, grav, is_view=True)
    plt.plot(x, grav   , label ='grav')
    plt.plot(x, y_pred , label ='trend', color = 'red')
    plt.grid()
    plt.legend()
    plt.show()

# grav_linear_trend()
#################################################################################################################

def calc_skew_kurt():
    '''
    Вычисление Асимметрии и эксцесса, которые не вычисляются в основной программе
    '''
    print('Вычисление Асимметрии и эксцесса, которые не вычисляются в основной программе')
    res_fn = dat_dir+'_skew_kurt_res.txt'
    f_res = open(res_fn, 'w')
    nnn = len(fnames)
    for i in range(nnn):
        print(fnames[i])
        f_res.write(fnames[i]+'  '+pinp_struct.full_grav_data_name[i]+'\n')
        f_src = dat_dir+fnames[i]
        n = pfile.text_file_num_lines(f_src)
        a = np.zeros(n)
        f = open(f_src, 'r')
        for j in range(n):
            s = f.readline()
            a[j] = float(s)
        f.close()
        # print(a); print(type(a))
        skew_biasT: float = float(skew(a, bias=True)) # Асимметрия скорректированное за статист. смещение
        f_res.writelines(f'Асимметрия скорректированная за статист. смещение = {skew_biasT} \n')
        skew_biasF: float = float(skew(a, bias=False)) # Асимметрия НЕскорректированное за статист. смещение
        f_res.write(f'Асимметрия НЕскорректированная за статист. смещение = {skew_biasF} \n')
        kurt_biasT: float = float(kurtosis(a, bias=True)) # Эксцесс скорректированный за статист. смещение
        f_res.write(f'Эксцесс скорректированная за статист. смещение = {kurt_biasT} \n')
        kurt_biasF: float = float(kurtosis(a, bias=False)) # Эксцесс НЕскорректированный за статист. смещение
        f_res.write(f'Эксцесс НЕскорректированная за статист. смещение = {kurt_biasF} \n \n')
    f_res.close()

# calc_skew_kurt()
#################################################################################################################

def view_hist(is_correct:bool):
    '''
    Просмотр гистограмм, которые трудно отображаются в основной программе
    '''
    plt.figure(figsize=(16, 8))
    plt.title('Гистограммы данных')

    nnn = len(fnames)
    n_titl = 0
    for i in range(nnn):
        print(fnames[i])
        f_src = dat_dir+fnames[i]
        a = pnumpy.read_numpy(f_src)
        plt.subplot(2, 3, n_titl + 1)  # 2 - количество строк;

        counts, bins = np.histogram(a)
        print(counts)
        nbin = 25
        if is_correct:
            if (i == 0): # грав.поле
                lst = [i for i in range(n_127)]  # print(lst)
                x = np.array(lst)  # print(x)
                (r_sq, model_intercept_, model_coef_, y_pred, ost_anom) = pmain_proc.linear_trend(x, a, is_view=True)
                b = deepcopy(ost_anom)
                titl = f'Снят линенйный тренд с ({pinp_struct.full_grav_data_name[n_titl]})'
            elif (i == 1): # стандартное отклонение
                b = np.log(a)
                titl = f'log ({pinp_struct.full_grav_data_name[n_titl]})'
            else:
                b = deepcopy(a)
                titl = pinp_struct.full_grav_data_name[n_titl]
        else:
            b = deepcopy(a)
            titl = pinp_struct.full_grav_data_name[n_titl]
        plt.hist(b, density=False, bins=nbin)  # bins=len(counts)
        plt.title(titl)

        plt.grid()
        plt.ylabel('Numbers')
        n_titl = n_titl + 1
    plt.show()

# view_hist(False)
# view_hist(True)
#################################################################################################################

def view_ini(with_smoothing:bool):
    '''
    Просмотр гистограмм, которые трудно отображаются в основной программе
    '''
    plt.figure(figsize=(16, 8))
    plt.title('Графики исходных данных')

    nnn = len(fnames)
    n_titl = 0
    lst = [i for i in range(n_127)]  # print(lst)
    x = np.array(lst)  # print(x)

    for i in range(nnn):
        print(fnames[i])
        f_src = dat_dir+fnames[i]
        dat = pnumpy.read_numpy(f_src)
        plt.subplot(2, 3, n_titl + 1)  # 2 - количество строк;
        titl = pinp_struct.full_grav_data_name[n_titl]
        plt.plot(x, dat)  # bins=len(counts)
        if with_smoothing and ((1<=i) and (i<=4)):
            dat21 = pmain_proc.moving_averages(dat, 21)
            dat65 = pmain_proc.moving_averages(dat, 65)
            plt.plot(x, dat21)
            plt.plot(x, dat65)
        plt.title(titl)
        plt.grid()
        n_titl = n_titl + 1
    plt.show()
# view_ini(bool(1))
#################################################################################################################

def view_fft():
    '''
    Просмотр результатов БПФ, которые трудно отображаются в основной программе
    '''
    plt.figure(figsize=(16, 8))
    plt.title('Результаты преобразования Фурье, АЧХ')

    nnn = len(fnames);
    n_titl = 0
    lst = [i for i in range(n_127)]  # print(lst)
    x = np.array(lst)  # print(x)
    kk = 100
    # kk = n_127-1
    for i in range(nnn):
        print(fnames[i])
        f_src = dat_dir+fnames[i]
        dat = pnumpy.read_numpy(f_src)
        plt.subplot(2, 3, n_titl + 1)  # 2 - количество строк;
        titl = pinp_struct.full_grav_data_name[n_titl]
        yf = scipy.fft.rfft(x = dat, n = n_65536)
        yf = scipy.fft.fft(x = dat, n = n_65536)
        xf = scipy.fft.rfftfreq(n_65536, 1 / 60)
        plt.plot(xf[0:kk], np.abs(yf[0:kk]))
        plt.title(titl)
        plt.grid()
        n_titl = n_titl + 1
    plt.show()

#view_fft()
#################################################################################################################

def view_acf():
    '''
    Просмотр результатов расчета АКФ, которые трудно отображаются в основной программе
    '''
    fig = plt.figure(figsize=(16, 8))
    # plt.title('Расчет автокрреляционной функции')
    llags=n_127-1
    llags=10000
    nnn = len(fnames)
    n_titl = 0
    for i in range(nnn):
        print(fnames[i])
        f_src = dat_dir+fnames[i]
        dat = pnumpy.read_numpy(f_src)
        ax_ = fig.add_subplot(2, 3, n_titl + 1)  # 2 - количество строк;
        titl = pinp_struct.full_grav_data_name[n_titl]
        plot_acf(dat, ax=ax_, lags=llags, use_vlines=False, fft=True, auto_ylims=True, markersize= 0.25)
        plt.title(titl)
        plt.grid()
        n_titl = n_titl + 1
    plt.show()

grav_linear_trend()
# view_fft()
# view_acf()