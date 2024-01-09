"""
inf-файл - структура всех данных и ввод из них

(C) 2023 Ivan Genik, Perm, Russia
Released under GNU Public License (GPL)
email igenik@rambler.ru
"""
# pinp_struct

import numpy as np
import copy
import math
import numba
import os
import pathlib
from pfunct import *
from pstring import *
from pmain_proc import *
from tkinter import messagebox as mb
from ptkinter_menu_proc import *
import datetime
import re

# test_mode = True  # тестовый режим
datfolder: str = r"Dat"
is_txt_res_file: bool = False

# 2019_Гравиразведка Ч2_Пугин.pdf, стр.69
full_grav_data_name=['Отсчеты гравиметра, мГал','СтдОткл отсчетов,мГал','Отклонение кварцевой системы от вертикали по оси Х',
                'Отклонение кварцевой системы от вертикали по оси Y','Температурная поправка, мГал',
                'Поправка за лунно-солнечное притяжение, мГал','Продолжительность одного цикла измерений на пункте, сек',
                'Число забракованных отсчетов в течение цикла измерений', 'Дата-Время Полное'] # 9 названий

# Словари (dict) и работа с ними. Методы словарей
#  https://pythonworld.ru/tipy-dannyx-v-python/slovari-dict-funkcii-i-metody-slovarej.html

# What is the maximum float in Python? == sys.float_info.max
# https://stackoverflow.com/questions/3477283/what-is-the-maximum-float-in-python

# пустой словрь для данных из inf-файла
empty_inf_dict = dict(name_sq='',  # название площади
                      fdat_name_= '',  # имя файла данных
                      full_fdat_name_='',  # полное имя файла данных с путём
                      work_dir='',  # внутреняя информация - папка с данными,
                      finf_name_='',  # имя inf-файла
                      full_finf_name_='',  # имя inf-файла
                      # num_lines='all', # число вводимых строк без заголовка
                      npoint=float('nan'),  # внутреняя информация - число точек в файле *.txt или xlsx
                      typeof_input=0,  # 0 - ничего не введено, самое начало; 1 - введен inf; 2 - введен txt/xlsx
                      saved_in_json=0,  # 0 - текущий ввод; 1 - ввод из json
                      saved_in_matrix=0, # 0 - текущий ввод; 1 - ввод данных из компактного сохраненного файла
                      matrix_file_name='' )


# Как я могу проверить, пуст numpy или нет?
# https://techarks.ru/qa/python/kak-ya-mogu-proverit-pust-n-10/

# --------- Квази объект "Структура данных" - (Начало)
curr_nstruct = -1  # номер структуры данных; -1 - нет данных, 0 - первая и т.д.
dat_struct: np.ndarray  # пустой набор данных в начале
out_of_range = np.zeros(4, dtype=int)  # cколько раз переменная выходила за границы переменная


def add_dat_struct(the_dict, the_nparray): # добавление данных в структуру dat_struct
    global dat_struct
    global curr_nstruct
    curr_nstruct = curr_nstruct+1
    if curr_nstruct == 0:
        dat_struct = create_2d_nparray_r1c2(1, 2, the_dict, the_nparray)
    else:
        dat_struct = add_2d_nparray(dat_struct, the_dict, the_nparray)


def get_dat_struct(num_el: int):  # извлечение данных из структуры dat_struct
    global dat_struct
    a = dat_struct[num_el, 0]
    b = dat_struct[num_el, 1]
    return a, b

def get_dat_array_for_view() -> str:
    """
    Объединение данных в одну строку для просмотра в окне
    """
    # https://docs-python.ru/tutorial/vstroennye-funktsii-interpretatora-python/funktsija-format/
    (the_dict, the_arr) = get_dat_struct(curr_nstruct)
    sf = '15s'
    nn = 15
    s=''
    six_wh = ' '*6
    # p = ' '*3
    n = the_dict['npoint']
    # print(type(the_arr[2, 1]),'  ',type(the_arr[2, 2]),'  ',type(the_arr[2, 3]),'  ')
    # print(type(the_arr[2, 4]),'  ',type(the_arr[2, 5]),'  ',type(the_arr[2, 6]),'  ')
    # print(type(the_arr[2, 7]),'  ',type(the_arr[2, 8]),'  ',type(the_arr[2, 10]),'  ')
    for i in range(n):
        s0 =  format(i, '6d')+ six_wh
        #                   GRAV.                        SD.                        TILTX
        #                  TILTY                       TEMP                  TIDE
        #                  DUR                         REJ                   DATE-TIME полный
        # s += format(str(the_arr[i, 1]),sf)+format(str(the_arr[i, 2]),sf)+format(str(the_arr[i, 3]),sf) + \
        # format(str(the_arr[i, 4]),sf)+format(str(the_arr[i, 5]),sf)+format(str(the_arr[i, 6]),sf) + \
        # format(str(the_arr[i, 7]),sf)+format(str(the_arr[i, 8]),sf)+format(str(the_arr[i, 10]),sf) + '\n'
        s1 = f'   {the_arr[i, 1]:8.4f}     {the_arr[i, 2]:8.4f}     {the_arr[i, 3]:8.4f}  '
        s2 = f'   {the_arr[i, 4]:8.4f}     {the_arr[i, 5]:8.4f}       {the_arr[i, 6]:8.4f}  '
        s3 = f'{the_arr[i, 7]:13d}      {the_arr[i, 8]:11d}        {the_arr[i, 10]} \n'
        s+= s0 + s1 + s2 + s3
    return s

def print_dat_struct() -> None:
    print(dat_struct)
# --------- Квази объект "Структура данных" - (Конец)

def control_curr_dict(curr_dict: dict) -> bool:
    """
    Контроль информации из inf-файла
    """
    full_file_name: str = "\\".join([curr_dict["work_dir"], curr_dict["fdat_name_"]])
    curr_dict["full_fdat_name_"] = full_file_name
    path = pathlib.Path(full_file_name)
    if not path.exists():
        # print(ss_fdfpne, full_file_name) # 'путь к dat-файлу не существует = '
        mb.showerror(s_error, ss_fdfpne + full_file_name)
        return False
    if not path.is_file():
        # print(ss_fdfne, full_file_name) # 'dat-файл не существует = '
        mb.showerror(s_error, ss_fdfne + full_file_name)
        return False

    # Далее проверки на попадание в диапазон
    return True


def input_inf(fname, is_view=False) -> (bool, object):
    """
    Ввод информации из файлв
    """
    s: str;    s2: str; s3: str
    # if isview: print('fname =', fname)
    # https://askdev.ru/q/numpy-dobavit-stroku-v-massiv-20857/
    file_exist: bool = os.path.isfile(fname)
    if not file_exist:
        return file_exist, empty_inf_dict
    else:
        f = open(fname, 'r')
        # print('this str')
        all_lines = f.read().splitlines()  # разделяет строку по символу переноса строки \n. Возвращает список(list)
        nrow1 = len(all_lines)  # вместе со строкой заголовков
        curr_dict = copy.deepcopy(empty_inf_dict)
        for i in range(nrow1):
            # Кракозябры в PyCharm
            # https://python.su/forum/topic/27557/
            s = all_lines[i]  # all_lines[i].encode('cp1251').decode('utf-8') перекодировка из Win в UTF
            s = s.strip()  # пробелы лишние убираем
            all_lines[i] = s  # на всякий случай сохраняем
            part_lines = s.split(sep=';', maxsplit=2)  # print_string(part_lines)
            if i == 0:
                curr_dict["name_sq"] = part_lines[0].strip()
            elif i == 1:
                s3 = part_lines[0].strip()
                part_lines2 = s3.split(sep=' ', maxsplit=2)
                curr_dict["fdat_name_"] = part_lines2[0]

    # Запоминаем путь к файлу данных
        # https://python-scripts.com/pathlib
        curr_dict["work_dir"] = str(pathlib.Path(fname).parent)
        # print("work_dir", curr_dict["work_dir"])
        curr_dict["finf_name_"] = str(pathlib.Path(fname).name)
        curr_dict["full_finf_name_"] = fname
        curr_dict["full_fdat_name_"] = "\\".join([curr_dict["work_dir"], curr_dict["fdat_name_"]])
        if is_view:
            print('print curr_dict');  print(curr_dict)
        # print("work_dir = ", curr_dict["work_dir"])
        return file_exist, curr_dict