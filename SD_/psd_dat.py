import numpy as np
from psd_proc import *


def create_dat()->(int, np.ndarray, np.ndarray):
    dat = the_xls_importdat(r'dat\127_sd.xlsx', is_view=False)  # вывод первых 40 строк

    x = dat[:, 2];    lenx = len(x)  # print(f'{lenx=}')
    ini_sd = dat[:, 0];    leny = len(ini_sd)  # print(f'{leny=}')

    z = [i for i in range(lenx)];    znp = np.array(z)
    return lenx, znp, ini_sd

