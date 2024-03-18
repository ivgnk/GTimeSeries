'''
Использование разных функций из numpy и scipy.
Преимущественно фильтры
'''
import inspect
import numpy as np
import scipy as sc
# from bp_const import *
from random import *
from math import *
from copy import deepcopy
import matplotlib.pyplot as plt
# from astropy.modeling.models import *  # для задания гауссова распределения весов окна

# Для фильтраций временных рядов
SMA_all_w = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
SMA_w: list = [3, 7, 13, 25, 51]  # SMA - Simple Moving  Average,  размер окон

def calc_triang_weights_for_1Dfilter(win_size:int)->np.ndarray:
    '''
    Расчет значений треугольной функции для фильтра
    не проверяем win_size, оно должно быть нечетным
    '''
    x = np.array([0.0 for i in range(win_size)])
    half_win = win_size // 2
    frst_odd = 1
    for i in range(x.size):
        if i <= half_win: # левая половина массива, включая центр
            x[i] = frst_odd
            frst_odd += 2
        else:
            x[i] = x[half_win-(i-half_win)]  # правая половина массива, без центра
    return x

def norm_weights_for_1Dfilter(dat: np.ndarray)->np.ndarray:
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

def work_with_smoothing(x: np.ndarray, y: np.ndarray)->(int, int, np.ndarray, list):
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

    # ---- (1) - sc.signal.savgol_filter
    curr_flt = 0
    num_flt = 0
    llst =[]
    (num_flt, curr_flt, res, llst) = prep_savgol(num_flt, curr_flt, y, res, llst)
    # ---- (2) - усреднение равновесовое
    (num_flt, curr_flt, res, llst) = prep_my_moving_average1D_filter(num_flt, curr_flt, y, res, llst)
    # ---- (3) - усреднение c треугольными весами
    (num_flt, curr_flt, res, llst) = prep_my_moving_average1D_filter_ww(num_flt, curr_flt, y, res, llst)
    # ---- (4) - усреднение c гауссовыми весами

    # for i in range(curr_flt):  print(res[i])
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
        res[curr_flt, 1] = 'savgol_filter'
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


def prep_my_moving_average1D_filter(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list)->(int, int, np.ndarray, list):
    '''
    Усреднение без весов (равновесовым фильтром)
    '''
    num_flt += 1
    npvar = len(SMA_w)# число подвариантов
    for i in range(npvar):
        win_len = SMA_w[i]
        res[curr_flt, 0] = num_flt
        res[curr_flt, 1] = 'mma1D_filter'
        res[curr_flt, 2] = i + 1
        res[curr_flt, 3] = my_moving_average1D(dat, window_size=win_len)
        res[curr_flt, 4] = 'win ' + str(win_len)
        curr_flt += 1
    llst += [npvar]
    return num_flt, curr_flt, res, llst

def prep_my_moving_average1D_filter_ww(num_flt:int, curr_flt:int, dat:np.ndarray, res:np.ndarray, llst:list)->(int, int, np.ndarray, list):
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
        res[curr_flt, 1] = 'mma1D_filter_ww'
        res[curr_flt, 2] = i + 1
        res[curr_flt, 3] = my_moving_average1D(dat, window_size=win_len, the_weights = the_weights)
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

def get_suplotsize(npictures:int)->tuple:
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

def test_suplotsize():
    '''
    Проверка функции get_suplotsize
    '''
    for i in range(1,20):
        print(i,' ',get_suplotsize(i))

def test_filters(len_filter=140):
    x = np.array([i for i in range(len_filter)])
    y = np.array([random()*10 for i in range(len_filter)])

    ########################################################
    # Это основная часть, все остальное - для визуализации
    (num_flt, curr_flt, res, llst) = work_with_smoothing(x, y)
    ########################################################

    # print(num_flt, curr_flt, llst)
    sp = get_suplotsize(num_flt)
    plt.subplots( nrows = sp[0], ncols = sp[1], figsize=(15, 8))
    currflt = 0
    for i in range(num_flt): # перебор по всем фильтрам
        plt.subplot(sp[0],sp[1], i + 1)
        plt.title(res[currflt,1])
        plt.plot(x, y, label='ini', linewidth=5)
        for j in range(llst[i]): # перебор по всем подвариантам фильтров
            if res[currflt,4] != '':
                # Пустая строка означает, что фильтр не рассчитался
                plt.plot(x, res[currflt,3], label = res[currflt,1]+' '+res[currflt,4])
            currflt += 1
            # https://jenyay.net/Matplotlib/LegendPosition
            plt.legend(loc = 'lower center', prop={'size': 8}) #  'lower right'  'best'
    plt.show()

def my_moving_average1D(dat:np.ndarray, window_size:int, the_weights = None)->np.ndarray:
    # print(inspect.currentframe().f_code.co_name)
    # Скользящее среднее 1 мерного массива
    # края оставляем как есть
    # веса в окне равные
    # окно только нечетной длины
    # не проверяем длину массива и окна
    moving_avg = deepcopy(dat)
    llen = dat.size
    half_win = window_size // 2
    ffirst = half_win
    llast = llen - ffirst
    # Проходим по нужным элементам в массиве, края уже заполнены deepcopy
    for i in range(ffirst,llast,1):
        # Добавляем текущий элемент к скользящему среднему
        win_dat = dat[i-half_win:i+half_win+1]
        moving_avg[i] = np.average(win_dat, weights=the_weights)
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

if __name__ == "__main__":
    # test_suplotsize()
    # test_get_win_fltr_lst()
    # test_medfilt_filter()
    # test_my_moving_average1D()
    #test_calc_triang_weights_for_filter1D()
    # test_numpy_average()

    # !!!  test_filters(100)
    # test_calc_lst_of_triang_weights() `````````
    test_wiener_filter()