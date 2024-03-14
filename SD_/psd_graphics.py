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

    axes = fig.add_axes(rect=(0.20, 0.48, 0.28, 0.28))
    axes.plot(dat[0].x[55843:55880],dat[0].y[55843:55880])
    plt.grid(linestyle='dotted')

    plt.show()
