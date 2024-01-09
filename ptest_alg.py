"""
Модельная функция для тестирования алгоритма минимизации

(C) 2021 Ivan Genik, Perm, Russia
Released under GNU Public License (GPL)
email igenik@rambler.ru
"""
# ptest_alg

import numpy as np
import numba
import codecs
import random
import pfunct
import pathlib
import copy
from pinp_struct import *
from math import *


def create_test_dict(name_sq_: str, fdat_nam_: str, wrk_dr: str, finf_nm: str):
    # заготовка для inf-файла
    test_dict = empty_inf_dict
    test_dict['name_sq'] = name_sq_  # 'Тестовая площадь на основе Новозаречного'
    test_dict['fdat_name_'] = fdat_nam_  # 'test.txt'
    test_dict['work_dir'] = wrk_dr
    test_dict['finf_name_'] = finf_nm
    test_dict['full_finf_name_'] = "\\".join([test_dict["work_dir"], test_dict["finf_name_"]])
    test_dict['npoint'] = 15*15  # 15*15

    return test_dict

def cnv2str(*args) -> str:
    s = ''
    for num in args:
        s += str(num)+' '
    return s


def write_inf(test_dict_, full_inf_name: str, lat_ini: float, lon_ini: float):
    f = open(full_inf_name, mode='w', encoding='utf-8')
    f.write(test_dict_['name_sq']+' ; название площади'+'\n')
    f.write(test_dict_['fdat_name_'] + ' ; файл данных'+'\n')
    f.close()

    with open(full_inf_name) as fh:
        data = fh.read()

    with open(full_inf_name, 'wb') as fh:
        fh.write(data.encode('cp1251'))

def write_inf2(the_dict):
    f = open(the_dict['full_finf_name_'], mode='w', encoding='utf-8')
    f.write(the_dict['name_sq']+' ; название площади'+'\n')
    f.write(the_dict['fdat_name_'] + ' ; файл данных'+'\n')
    f.close()

    with open(the_dict['full_finf_name_']) as fh:
        data = fh.read()

    with open(the_dict['full_finf_name_'], 'wb') as fh:
        fh.write(data.encode('cp1251'))
