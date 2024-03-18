from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import *

@dataclass
class GraphCl:
    name: str
    x:np.ndarray
    y:np.ndarray

# https://stackoverflow.com/questions/33045222/how-do-you-alias-a-type-in-python
GraphCl_lstT: TypeAlias = list[GraphCl]
GraphCl_lst:GraphCl_lstT=[]

def lens_for_4parts(n:int):
    return ((n // 2) // 2)*4

def view_in_4parts(x1:np.ndarray, y1:np.ndarray, suptitle:str,ylabel:str):
    # показ по 4 частям
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle(suptitle, fontsize=16)
    llen = lens_for_4parts(len(x1))
    x = x1[0:llen]; y = y1[0:llen]
    xspl = np.split(x,4)
    yspl = np.split(y,4)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.ylabel(ylabel, fontsize=14)
        plt.plot(xspl[i], yspl[i], color='r',linewidth=0.5) #
        plt.grid()
        plt.title('Часть '+str(i+1))
    plt.show()

    # Matplotlib Widgets -- How to Make Your Plot Interactive With Buttons
    # https://www.youtube.com/watch?v=YmE-fL3TP64
    # 50 оттенков matplotlib — The Master Plots (с полным кодом на Python)
    # https://habr.com/ru/articles/468295/ 

def view_datetime_sd( dat:GraphCl_lstT )->None:
    fig = plt.figure(figsize=(16, 8))
    plt.title('Стандарное отклонение отсчетов, мГал', fontsize=16)
    plt.xlabel('Отсчеты', fontsize=14) # 'Дата и время'
    plt.ylabel('St.dev., мГал', fontsize=14)
    llen = len(dat)
    for i in range(llen):
        plt.plot(dat[i].x, dat[i].y,label=dat[i].name )
    plt.grid()
    plt.legend()
    # вставка самксимальными значениями
    axes = fig.add_axes(rect=(0.20, 0.48, 0.28, 0.28))
    axes.plot(dat[0].x[55843:55880],dat[0].y[55843:55880])
    plt.grid(linestyle='dotted')
    plt.show()

    # показ по 4 частям
    view_in_4parts(dat[0].x, dat[0].y, suptitle='Стандарное отклонение отсчетов, мГал',ylabel='St.dev., мГал')
