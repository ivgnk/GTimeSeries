# -------------------------------------------------------------------
# Разные функции общего назначения
#
# (C) 2023 Ivan Genik, Perm, Russia
# Released under GNU Public License (GPL)
# email igenik@rambler.ru
# -------------------------------------------------------------------

import math
import numpy as np
from functools import reduce


#----------------- Работа с цветами, шкалами и прочее
def calc_log_levels(dat: np.array, nlevel: int) -> list:
    llist = []
    mmin = np.min(dat); llist.append(mmin)
    mmax = np.max(dat)
    q = pow(mmax/mmin, 1.0/(nlevel-1))
    ll = np.linspace(start=2, stop=nlevel-1, num=nlevel-2)
    dat = mmin
    for i in ll:
        dat = dat*q
        llist.append(dat)
    llist.append(mmax)
    return llist

def rgb_to_hex(rgb):
    # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    # rgb_to_hex((255, 255, 255))        # ==> '#ffffff'
    return '#%02x%02x%02x' % rgb

def red2blue_21colors() -> list:
    # https://www.kite.com/python/answers/how-to-make-a-colored-scatter-plot-in-matplotlib-in-python
    llist =[]
    llist.append(rgb_to_hex((92,0,0)))  # 1
    llist.append(rgb_to_hex((160,0,0))) # 2
    llist.append(rgb_to_hex((200,0,0))) # 3
    llist.append(rgb_to_hex((255,0,0))) # 4
    llist.append(rgb_to_hex((255,60,60))) # 5
    llist.append(rgb_to_hex((255,116,116))) # 6
    llist.append(rgb_to_hex((255,148,148))) # 7
    llist.append(rgb_to_hex((255,170,170))) # 8
    llist.append(rgb_to_hex((255,208,208))) # 9
    llist.append(rgb_to_hex((255,225,225))) # 10
    llist.append(rgb_to_hex((255,255,255))) # 11
    llist.append(rgb_to_hex((225,225,255))) # 12
    llist.append(rgb_to_hex((208,208,255))) # 13
    llist.append(rgb_to_hex((170,170,255))) # 14
    llist.append(rgb_to_hex((148,148,255))) # 15
    llist.append(rgb_to_hex((116,116,255))) # 16
    llist.append(rgb_to_hex((60,60,255))) # 17
    llist.append(rgb_to_hex((0,0,255))) # 18
    llist.append(rgb_to_hex((0,0,200))) # 19
    llist.append(rgb_to_hex((0,0,160))) # 20
    llist.append(rgb_to_hex((0,0,92))) # 21
    return llist

#-----------------
def list2d3_to_3nparray(ll:list) -> (np.ndarray, np.ndarray, np.ndarray):
    llen = len(ll)
    x = np.random.random(llen)
    y = np.random.random(llen)
    z = np.random.random(llen)
    for i in range(llen):
        d = ll[i]
        x[i] = d[0]
        y[i] = d[1]
        z[i] = d[2]
    return x, y, z


def out_of_diap2(dat, dat_min, dat_max) -> (float, float):
    """
    Расчет выхода за диапазон - абсолютные и относительные значения (доли 1)
    """
    llen = dat_max-dat_min
    if dat_min > dat:
        d = dat_min - dat
        return d, d/llen
    if dat_max < dat:
        d = dat - dat_max
        return d, d/llen


def out_of_diap1proc(dat, dat_min, dat_max) -> float:
    """
    Расчет выхода за диапазон - относительные значения (доли 1)
    """
    llen = dat_max-dat_min
    if dat_min > dat:
        d = dat_min - dat
        return d/llen
    if dat_max < dat:
        d = dat - dat_max
        return d/llen


def out_of_diap2proc(dat, dat_min, dat_max) -> float:
    """
    Расчет выхода за диапазон - абсолютные и относительные значения (доли 1)
    Вычисления в логарифмическом масштабе
    """
    dat_minln = math.log10(dat_min)  # print(dat_min)
    dat_maxln = math.log10(dat_max)
    if dat < 0:
        dat_ln = -10
    else:
        dat_ln = math.log10(dat)

    llen = dat_maxln-dat_minln
    d = 0
    if dat_minln > dat_ln:
        d = dat_minln - dat_ln
    if dat_maxln < dat_ln:
        d = dat_ln - dat_max
    # print(d/llen)
    return abs(d/llen)


def dat_in_diap(dat, dat_min, dat_max) -> bool:
    return (dat_min <= dat) and (dat <= dat_max)


def dat2_in_diap(dat1, dat2, dat_min, dat_max, isview: bool = False) -> bool:
    if isview:
        print(dat1, dat2, dat_min, dat_max)
    return (dat_min <= dat1) and (dat1 <= dat_max) and (dat_min <= dat2) and (dat2 <= dat_max)

# ============ Работа с двумерными списками
def find_maxn_inlist(dlist: list, npos: int, reduce_init:float):
    # поиск в двумерном списке чисел
    n = npos
    def mmax(x, y):
        nonlocal n
        #  print('x=',x)
        #  print('y=',y)
        if x > y[n]:
            return x
        else:
            return y[n]
    return reduce(mmax, dlist, reduce_init)

def find_maxn2_inlist(dlist: list, npos: int, nposextr: int, reduce_init:list):
    # поиск в двумерном списке чисел и сохраение списка:
    # наибольшее и указаннон
    n = npos
    nextr = nposextr
    def mmax(x, y):
        nonlocal n
        nonlocal nextr
        #  print('x=',x)
        #  print('y=',y)
        if x[0] > y[n]:
            return x
        else:
            return [y[n], y[nextr]]

    return reduce(mmax, dlist, reduce_init)


# ============  Тестирование
def create_2d_nparray_r1c2(row, col, dat00=0.0, dat01=0.0) -> object:
    a = np.zeros((row, col), dtype=object)
    a[0, 0] = dat00
    a[0, 1] = dat01
    return a


def add_2d_nparray(nparr1, dat00, dat01) -> object:
    # Функции numpy.hstack() и numpy.vstack()
    # https://pythonist.ru/funkczii-numpy-hstack-i-numpy-vstack/
    nparr2 = create_2d_nparray_r1c2(1, 2, dat00, dat01)
    return np.vstack((nparr1, nparr2))

# def test_2d_nparray() -> None:
#     nparr = create_2d_nparray_r1c2(1,2)
#     print(nparr, end='\n\n')
#     nparr = add_2d_nparray(nparr, 10, 20)
#     print(nparr)
# print(dat2_in_diap(0.001, 9.999, 0.0, 10.0, isview=True))
# print(dat2_in_diap(float('nan'), float('nan'), float('nan'), float('nan'), isview=True))
# test_2d_nparray()
# print_format_examples()

# dat = np.zeros(2)
# dat[0] = 0.1; dat[1] = 1000;
# llist = calc_log_levels(dat, nlevel=12)
# for i in range(len(llist)):
#     print(llist[i])

# llist = red2blue_21colors()
# for i in range(len(llist)):
#     print( llist[i] )

#----------------- Работа с datetime
# from datetime import datetime
# ll =['03.04.17 9:33', '03.04.17 9:34', '03.04.17 9:35', '03.04.17 9:36', '03.04.17 9:37', '03.04.17 9:38', '03.04.17 9:39']
# print(ll)
# kk=[]
# for i in range(len(ll)):
#     dd = datetime.strptime(ll[i], '%d.%m.%y %H:%M')
#     kk+=[dd.timestamp()]
# print(kk)
#
# zz =[]
# for i in range(len(kk)):
#     hhh = datetime.fromtimestamp(kk[i], tz=None)
#     zz+=[hhh]
# print(zz)
