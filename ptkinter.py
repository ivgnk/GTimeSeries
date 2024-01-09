"""
Главный объект программы class MakroseisGUI(Frame):

(C) 2023 Ivan Genik, Perm, Russia
Released under GNU Public License (GPL)
email igenik@rambler.ru
"""

from tkinter import *
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import filedialog as fd
from tkinter import scrolledtext
# from tkinter.constants import END

from typing import Any
import Pmw
import ptkinter_menu_proc

from ptkintertools import *
from pmain_proc import *
from dataclasses import dataclass
from ptkinter_menu_proc import *

# --- matplotlib imports
# import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageTk
import os
import sys
import pathlib
import winsound
# ------ Из Makro_seis
import pfile
import pfunct
import pinp_proc
import pinp_struct
import ptest_alg
import json
import pprint as pp

import datetime
from sklearn.linear_model import LinearRegression

p_width: int = 1400  # ширина окна программы
p_height: int = 900  # высота окна программы

view_inf_fw: int = 1150  # ширина окна проcмотра inf-файла
view_inf_fh: int = 400  # высота окна проcмотра inf-файла

view_datt_fw: int = 1150  # ширина окна проcмотра dat-файла txt
view_datt_fh: int = 800  # высота окна проcмотра dat-файла  txt

view_datx_fw: int = 1400  # ширина окна проcмотра dat-файла  xls
view_datx_fh: int = 800  # высота окна проcмотра dat-файла  xls

view_graph2_fw: int = 1140  # ширина окна проcмотра графиков исх.данных
view_graph2_fh: int = 1000  # высота окна проcмотра графиков исх.данных

view_graph_fw: int = 1140  # ширина окна проcмотра графика
view_graph_fh: int = 780  # высота окна проcмотра графика

view_par2st_fw: int = 650  # ширина окна выбора нач.прибл. 2 стадии минимизации
view_par2st_fh: int = 180  # высота окна выбора нач.прибл. 2 стадии минимизации

view_par2sti_fw: int = 850  # ширина окна информации о inf-файле
view_par2sti_fh: int = 130  # высота окна информации о inf-файле


view_stat_fw: int = 750  # ширина окна проcмотра статистики
view_stat_fh: int = 400  # высота окна проcмотра статистики

view_res_fw: int = 450  # ширина окна проcмотра результатов минимизации
view_res_fh: int = 300  # высота окна проcмотра результатов минимизации

view_res2_fw: int = 550  # ширина окна проcмотра результатов минимизации2
view_res2_fh: int = 300  # высота окна проcмотра результатов минимизации2


view_testpar_fw: int = 450  # ширина окна проcмотра параметров минимизации
view_testpar_fh: int = 300  # высота окна проcмотра параметров минимизации

vf_width: int = 700  # ширина окна просмотра
vf_height: int = 700  # высота окна программы

status_bar_width = 220
status_bar_height = 20

#  1 июля 2018 Введение в Data classes (Python 3.7)
#  https://habr.com/ru/post/415829/

res0dict = dict(  # --- общие
                # --- конкретные
                file_res_name='' )


@dataclass
class StatusBarLabel:
    status_bar: Any
    the_status: int

# ----------- Источники
# Создание графического интерфейса https://metanit.com/python/tutorial/9.1.php
# Диалоговые (всплывающие) окна / tkinter 9 https://pythonru.com/uroki/dialogovye-vsplyvajushhie-okna-tkinter-9

# ----------- ООП
# Библиотека Tkinter - 9 - Использование ООП
# https://www.youtube.com/watch?v=GhkyLQ6A6Yw&list=PLfAlku7WMht4Vm6ewLgdP9Ou8SCk4Zhar&index=9


class GTimeSeriesGUI(Frame):
    # Константы для смены надписей в StatusBar
    SBC_st = 0    # Программа запущена
    SBC_fdni = 10  # Данные не введены
    SBC_fdi = 11  # Данные введены
    SBC_fvpar = 14  # Просмотр параметров минимизации

    SBC_cb = 21  # Вычисления начаты
    SBC_ce = 22  # Вычисления закончены
    SBC_cc = 23  # Вычисления прерваны

    SBC_hab = 41  # О программе
    SBC_hum = 42  # Руководство пользователя
    SBC_ns = 50

    is_input_inf = False;     fn_current_inf = ''    # текущий inf-файл
    is_input_txt = False;     fn_current_txt = ''   # текущий txt-файл
    is_input_xlsx = False;    fn_current_xlsx = ''  # текущий xlsx-файл
    is_exist_resf = False;    fn_current_resf = ''  # текущий res-файл
    dict_struct = pinp_struct.empty_inf_dict
    dict_test_struct = pinp_struct.empty_inf_dict
    res_dict = res0dict
    res_list = []
    numpy_arr = None

    # fn_inf = '' # имя inf-файла
    dn_current_dir = ''
    dn_current_dat_dir = ''
    dn_current_res_dir = ''

    scr_w = 0  # ширина экрана
    scr_h = 0  # высота экрана

    def __init__(self, main):
        super().__init__(main)

        # Строка статуса
        # https://www.delftstack.com/ru/tutorial/tkinter-tutorial/tkinter-status-bar/
        # the_status_bar_label1   # 0 - Начата работа программы        # 1 - Данные введены
        #                         # 2 - Вычисление закончено           # 3 - Вычисление прервано
        self.the_status_bar_label1 = StatusBarLabel(status_bar=Label(main, text=" Программа запущена ",
                                                                     bd=1, relief=SUNKEN, anchor=W), the_status=0)
        self.the_status_bar_label2 = StatusBarLabel(status_bar=Label(main, text="                    ",
                                                                     bd=1, relief=SUNKEN, anchor=W), the_status=0)
        self.the_status_bar_label3 = StatusBarLabel(status_bar=Label(main, text="                    ",
                                                                     bd=1, relief=SUNKEN, anchor=W), the_status=0)

        main.title(win_name)
        (self.scr_w, self.scr_h, form_w_, form_h_, addx, addy) = mainform_positioning(main, p_width, p_height)
        # print('self.scr_w, self.scr_h')
        # print(self.scr_w, self.scr_h)

        # main.geometry(root_geometry_string_auto(main, p_width, p_heiht)) # 1920x1080 мой монитор
        main.geometry(root_geometry_string(form_w_, form_h_, addx, addy))
        # Python Tkinter, как запретить расширять окно программы на полный экран?
        # https://otvet.mail.ru/question/191704214
        main.resizable(width=False, height=False)
        main.config(menu=self.create_menu())
        main.iconbitmap(ico_progr)

        # разлчиные варианты выхода
        # https://fooobar.com/questions/83280/how-do-i-handle-the-window-close-event-in-tkinter
        main.protocol("WM_DELETE_WINDOW", self.on_closing)
        main.bind('<Escape>', self.btn_esc)
        # main.bind('<Control-b>', self.view2_txt_dat)  #  еще раз просмотр файла данных, но собранного из массива np.
        # main.bind('<Control-i>', self.view_inf_inf_)  #  вывод служебной информации
        main.bind('<Control-g>', self.calc_view_graph_res_event_btn)  # построение графика

        # self.the_status_bar_label1.status_bar.pack(side=LEFT, fill=X)
        # self.the_status_bar_label2.status_bar.pack(side=LEFT, fill=X)
        # -------- Создание Status bar
        rest_width = form_w_ - (2*(status_bar_width+2+2))
        self.the_status_bar_label1.status_bar.place(x=1,                      y=form_h_-status_bar_height-3, width=status_bar_width, height=status_bar_height)
        self.the_status_bar_label2.status_bar.place(x=status_bar_width+2,     y=form_h_-status_bar_height-3, width=status_bar_width, height=status_bar_height)
        self.the_status_bar_label3.status_bar.place(x=2*(status_bar_width+2), y=form_h_-status_bar_height-3, width=rest_width, height=20)
        # -------- Создание панели инструментов
        # Меню, подменю и панель инструментов в Tkinter
        # https://python-scripts.com/tkinter-menu-toolbars-example

        # Pmw.Balloon
        # http://pmw.sourceforge.net/doc/Balloon.html
        self.balloon = Pmw.Balloon(main)
        self.create_tool_bar(main)

    # Определяем текущую папку с программой https://ru.stackoverflow.com/questions/535318/Текущая-директория-в-python
        self.dn_current_dir = os.getcwd()
        self.dn_current_dat_dir = "\\".join([self.dn_current_dir, dat_dir])
        self.dn_current_res_dir = "\\".join([self.dn_current_dir, res_dir])

        # проверяем наличие и при отсутствии создаем папку результатов
        path = pathlib.Path(self.dn_current_res_dir)
        if not path.is_dir():
            path.mkdir()

        # Определяем есть ли json файл
        path = pathlib.Path(json_fn)
        if path.is_file():
            self.get_from_json()
            self.dict_struct["saved_in_json"] = 1
            matrix_file_name = self.dict_struct["matrix_file_name"]
            pinp_proc.the_input(fname=self.dict_struct['full_finf_name_'], res_dir=self.dn_current_res_dir,
                                is_view=False, matrix_file_name=matrix_file_name)
        else:
            self.dict_struct["saved_in_json"] = 0
            self.dict_struct["saved_in_matrix"]=0
            self.dict_struct["matrix_file_name"]=''

    def on_closing(self):
        sys.exit(0)

    def btn_esc(self, event):
        sys.exit(0)
        # root.bind('<Escape>', btn_esc)

    def create_tool_bar(self, main):
        toolbar = Frame(main, bd=1, relief=RAISED)
        # https://icon-icons.com/ru/download/
        # полный формат ввода иконки 'E:\Work_Lang\Python\PyCharm\Tkinter\Ico\play_22349.png
        self.create_button1(ico_input_inf_, self.input_inf_, toolbar, 'Открыть файл данных')
        self.create_button1(ico_inigraph_, self.f_view_graph_sd_grav, toolbar, 'Графики исходных данных')
        self.create_button1(ico_calc_, self.c_hist_, toolbar, 'Расчет')
        self.create_button1(ico_resgraph_, self.calc_view_graph_res_event, toolbar, 'График результатов')
        self.create_button1(ico_usrmanual_, self.help_usrmanual_, toolbar, sh_usrmanual)
        toolbar.pack(side=TOP, fill=X)

    # Всплывающие окна   https://pythonru.com/uroki/vsplyvajushhie-okna-tkinter-11
    # 2020_Tkinter Программирование на GUI Python_Шапошникова.pdf  11. Окна, стр. 48, 86
    # tkinter, Усложнение диалогов window_09.py http://www.russianlutheran.org/python/nardo/nardo.html
    # https://fooobar.com/questions/14054375/grabset-in-tkinter-window

    def create_scrolledtext_win(self, main):

        # https://ru.stackoverflow.com/questions/791876/Отображение-виджетов-в-tkinter-скрыть-и-вернуть-обратно
        scrolledtext_frame = Frame(main, bd=2)
        # frame1.pack(fill='both', expand='yes')
        scrolledtext_frame.pack(side=TOP, fill=X)
        scrolledtext_win = scrolledtext.ScrolledText(scrolledtext_frame, wrap=WORD,  width=20, height=10)
        scrolledtext_win.pack(padx=10, pady=10, fill=BOTH, expand=True)
        scrolledtext_win.pack_forget()
        scrolledtext_win.insert(INSERT,
                        """\
                        Integer posuere erat a ante venenatis dapibus.
                        Posuere velit aliquet.
                        Aenean eu leo quam. Pellentesque ornare sem.
                        Lacinia quam venenatis vestibulum.
                        Nulla vitae elit libero, a pharetra augue.
                        Cum sociis natoque penatibus et magnis dis.
                        Parturient montes, nascetur ridiculus mus.
                        """)
        return scrolledtext_win

    def change_status_bar1(self, status, status_bar):
        self.the_status = status
        if status == self.SBC_st:
            status_bar['text'] = 'Программа запущена'
        elif status == self.SBC_fdi:
            status_bar['text'] = ss_fdi
        elif status == self.SBC_fdni:
            status_bar['text'] = ss_fdni
        # elif status == self.SBC_fvpar:
        #     status_bar['text'] = sf_vpar
        elif status == self.SBC_cb:
            status_bar['text'] = ss_ccb  # 'Вычисления начаты'
        elif status == self.SBC_ce:
            status_bar['text'] = ss_cce  # 'Вычисления закончены'
        elif status == self.SBC_cc:
            status_bar['text'] = ss_ccc  # 'Вычисления прерваны'
        # ---- Подменю Помощь
        elif status == self.SBC_hab:
            status_bar['text'] = sh_about
        elif status == self.SBC_hum:
            status_bar['text'] = sh_usrmanual
        else:
            status_bar['text'] = 'Неопределенный статус'

    # ---- Панель инструментов
    def create_button1(self, icofilename: str, the_command, the_toolbar, the_hint):
        """
        Создание кнопки
        """
        self.img = Image.open(icofilename)
        eimg = ImageTk.PhotoImage(self.img)
        the_button = Button(
            the_toolbar, image=eimg, relief=FLAT,
            command=the_command
        )
        the_button.image = eimg
        the_button.pack(side=LEFT, padx=2, pady=2)
        self.balloon.bind(the_button, the_hint)

    def cmd_beep(self):
        winsound.Beep(frequency=150, duration=1000)

    # def cmd_calc_resmap_event(self):
    #     self.f_view_map_res()

    def input_from_inf(self, fname: str) -> bool:
        (good_end, self.dict_struct) = pinp_proc.the_input(fname=fname,
                                                           res_dir=self.dn_current_res_dir,
                                                           is_view=bool(0),
                                                           matrix_file_name=self.dict_struct["matrix_file_name"])
        self.dict_struct["saved_in_json"] = 1
        self.dict_struct["saved_in_matrix"] = 1
        self.put_to_json()
        return good_end

    # ---- Подменю Файл

    def input_inf_(self):
        ext_type = ''
        good_new_data = ''

        fn_dat = fd.askopenfilename(initialdir=self.dn_current_dat_dir,
             defaultextension='.inf', filetypes=[('inf файлы', '*.inf')])
        # , ('txt файлы', '*.txt'), ('xlsx файлы', '*.xlsx'), ('Все файлы', '*.*')

        the_ext = pfile.gfe(fn_dat)
        if the_ext == '.inf':
            ext_type = 'i'
            good_new_data = self.input_from_inf(fn_dat) # !!!!!!!!!!!!!!!!!
            if good_new_data:
                # self.dict_struct['full_finf_name_'] = fn_dat
                # self.dict_struct['finf_name_'] =
                self.dict_struct['typeof_input'] = 1
        else:
            mb.showerror(s_error, sf_err_ext)

        if the_ext == '.inf':
            if good_new_data:
                self.change_status_bar1(self.SBC_fdi, self.the_status_bar_label1.status_bar)
                self.the_status_bar_label3.status_bar['text'] = fn_dat
                self.put_to_json()
            else:
                self.change_status_bar1(self.SBC_fdni, self.the_status_bar_label1.status_bar)
                self.the_status_bar_label3.status_bar['text'] = sf_ferror


    def put_to_json(self):  # запись dict_struct в json
        # https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
        with open(json_fn, 'w') as file:
            json.dump((self.dict_struct, self.res_dict, self.res_list), file)

    def get_from_json(self):  # чтение dict_struct из json
        # https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
        with open(json_fn, 'r') as file:
            (self.dict_struct, self.res_dict, self.res_list) = json.load(file)
            # print(self.res_dict)

    def f_view_inf(self) -> None:
        if self.dict_struct['full_finf_name_'] == '':
            mb.showerror(s_error, sf_finfni)
        else:
            ffn = self.dict_struct['full_finf_name_']
            f = open(ffn, 'r')
            s = f.read()

            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_inf_fw, view_inf_fh)
            self.dialog = CViewTxt(self.master, sf_vinf + '    ' + ffn, root_geometry_string(form_w_, form_h_, addx, addy))
            self.dialog.go(s) # self.dialog.go(s.encode('cp1251').decode('utf-8'))
            f.close()

    def f_view_dat(self):
        if (self.dict_struct["saved_in_json"] == 1) or (self.dict_struct['typeof_input'] > 0):
            dfn = self.dict_struct['full_fdat_name_']
            if dfn == '':
                mb.showerror(s_error, ss_fdfne+dfn)
            else:
                ffn = self.dict_struct['full_fdat_name_']
                ss = pfile.gfe(ffn)
                if ss == '.txt':
                    f = open(self.dict_struct['full_fdat_name_'], 'r')
                    s = f.read()
                    f.close()
                    (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_datt_fw, view_datt_fh)
                    self.dialog = CViewTxt(self.master, sf_vdat + '    ' + ffn, root_geometry_string(form_w_, form_h_, addx, addy))
                    self.dialog.go(s)  # .encode('cp1251').decode('utf-8')
                elif ss == '.xlsx':
                    # https://pythonru.com/uroki/chtenie-i-zapis-fajlov-excel-xlsx-v-python
                    # 2019_Гравиразведка Ч2_Пугин.pdf, стр.69
                    s = '      N       ГравПоле,мГал  СтдОткл,мГал  ОтклХ         ОтклY    ТемперПопр,мГал '
                    s+= 'ЛунСолнПопр,мГал   1ЦиклИзм,сек  БракОтсчв1цикл  Дата-ВремяПолн' + '\n'
                    # dt_now = datetime.datetime.now();   print(dt_now)
                    s += pinp_struct.get_dat_array_for_view()
                    # dt_now = datetime.datetime.now();   print(dt_now)
                    (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_datx_fw, view_datx_fh)
                    self.dialog = CViewTxt(self.master, sf_vdat + '    ' + ffn, root_geometry_string(form_w_, form_h_, addx, addy))
                    self.dialog.go(s)  # .encode('cp1251').decode('utf-8')
                    # mb.showerror(s_error, s_error)
        else:
            mb.showerror(s_error, ss_fdni)

    def f_view_xlsx(self):
        pass

    def view2_txt_dat(self, event):    # еще раз просмотр файла данных, но собранного из массива np.
        # mb.showerror(s_error, 'Заглушка')
        s = self.create_txt_dat_str()
        (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_datt_fw, view_datt_fh)
        self.dialog = CViewTxt(self.master, sf_vstat, root_geometry_string(form_w_, form_h_, addx, addy))
        self.dialog.go(s)  # .encode('cp1251').decode('utf-8')

    def create_txt_dat_str(self) -> str:
        s = '                  Lat              Lon      Alt        I_fact            dI            N                  Нас.пункт' + '\n    '
        s += str(pinp_struct.curr_nstruct)+'\n    '
        if pinp_struct.curr_nstruct < 0:  # не введены данные
            pass
        else:
            the_arr = pinp_struct.dat_struct[pinp_struct.curr_nstruct, 1]
            s += str(len(the_arr))
        return s

    def f_view_stat(self):
        s:str = pmain_proc.calc_stat()
        print('stat')
        if (self.dict_struct["saved_in_json"] == 1) or (self.dict_struct['typeof_input'] > 0):
            pfile.write_to_file(s, create_stat_fn(self.dict_struct['full_fdat_name_']))
            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_stat_fw, view_stat_fh)
            self.dialog = CViewTxt(self.master, sf_vstat, root_geometry_string(form_w_, form_h_, addx, addy))
            self.dialog.go(s)  # .encode('cp1251').decode('utf-8')
        else:
            if self.dict_struct['typeof_input'] == 0:  # не введено
                mb.showerror(s_error, ss_fdni)
            else:
                pass

    def  f_detrend_grav(self):
        if pinp_struct.curr_nstruct == -1:
            mb.showerror(s_error, ss_fdni)
        else:
            (grav_, sd_, tilt_x, tilt_y, temp_, tide_, dur_, rej_, date_time_, xn_, time_, date_) = pmain_proc.get_data_ini2() #  self.f_get_graph_data_ini()
            print(xn_.size)
            print(xn_.shape)
            x = xn_.reshape((-1, 1))
            # 25/04/2019 Линейная регрессия на Python: объясняем на пальцах https://proglib.io/p/linear-regression
            model = LinearRegression()
            model.fit(x, grav_)
            r_sq = model.score(x, grav_)
            print('coefficient of determination:', r_sq)
            # y = np.interp(xn_, xp, fp)  # Расчет в точках интерполяции
            print('Пересечение:', model.intercept_)
            intercept: 5.633333333333329
            print('Наклон:', model.coef_)
            y_pred = model.predict(x)

            name_='Гравитационное поле и его тренд'
            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph2_fw, view_graph2_fh)
            delta = grav_ - y_pred
            print('min остаточная аномалия =',min(delta))
            print('max остаточная аномалия =',max(delta))
            graph_name = 'Участок(файл) ' + self.dict_struct["name_sq"]+'.\n '+name_+'\n'
            self.dialog = CViewGraph1Plot(self.master, name_, graph_name, root_geometry_string(form_w_, form_h_, addx, addy),
                                      xn_, grav_, y_pred, 'Время, мин', 'Исходное поле, мГал', 'Линейный тренд, мГал',
                                      0, 63000, 6535, 6550, 6535, 6550)
            self.dialog.go()


    def f_view_graph_temp_tide(self):
        if pinp_struct.curr_nstruct == -1:
            mb.showerror(s_error, ss_fdni)
        else:
            (grav_, sd_, tilt_x, tilt_y, temp_, tide_, dur_, rej_, date_time_, xn_, time_, date_) = pmain_proc.get_data_ini2() #  self.f_get_graph_data_ini()
            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph2_fw, view_graph2_fh)
            graph_name = 'Участок(файл) ' + self.dict_struct["name_sq"]+'.\n Графики исходных данных'+'\n'
            self.dialog = CViewGraph2(self.master, sf_vigraph_sd_grav, graph_name, root_geometry_string(form_w_, form_h_, addx, addy),
                                      xn_, temp_, tide_, 'Время, мин', 'Температурная поправка, мГал', 'Поправка за лунно-солнечное притяжение, мГал',
                                      0, 63000, 0.15, 0.39, -0.11, 0.05)
            self.dialog.go()

    def f_view_graph_tiltx_tilty(self):
        if pinp_struct.curr_nstruct == -1:
            mb.showerror(s_error, ss_fdni)
        else:
            (grav_, sd_, tilt_x, tilt_y, temp_, tide_, dur_, rej_, date_time_, xn_, time_, date_) = pmain_proc.get_data_ini2() #  self.f_get_graph_data_ini()
            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph2_fw, view_graph2_fh)
            graph_name = 'Участок(файл) ' + self.dict_struct["name_sq"]+'.\n Графики исходных данных'+'\n'
            self.dialog = CViewGraph2(self.master, sf_vigraph_sd_grav, graph_name, root_geometry_string(form_w_, form_h_, addx, addy),
                                      xn_, tilt_x, tilt_y,  'Время, мин', 'Откл по Х', 'Откл по Y', 0, 63000, -6, 10, -3, 14)
            self.dialog.go()

    def f_view_graph_sd_grav(self):
        # if self.dict_struct['typeof_input'] ==  0:  # не введено
        #     mb.showerror(s_error, ss_fdni)
        # else:
        if pinp_struct.curr_nstruct == -1:
            mb.showerror(s_error, ss_fdni)
        else:
            (grav_, sd_, tilt_x, tilt_y, temp_, tide_, dur_, rej_, date_time_, xn_, time_, date_) = pmain_proc.get_data_ini2() #  self.f_get_graph_data_ini()
            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph2_fw, view_graph2_fh)
            graph_name = 'Участок(файл) ' + self.dict_struct["name_sq"]+'.\n Графики исходных данных'+'\n'
            # def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
            #              xn_: np.ndarray, dat1_: np.ndarray, dat2_: np.ndarray, dat1label: str, dat2label: str,
            #              xmin: float, xmax: float,
            #              dat1min: float, dat1max: float, dat2min: float, dat2max: float):
            if pmain_proc.DescrStat_lst == []:
                s1 = calc_stat() # внутри вычисляется DescrStat_lst
                # print('s1 = \n',s1)
                DescrStat_view_names(pmain_proc.DescrStat_lst)
                # DescrStat_lst_get_n_min_max(' ')
            else:
                DescrStat_lst_get_n_min_max(' ')
            # 0  -*-  Отсчеты гравиметра, мГал
            # 1  -*-  СтдОткл отсчетов,мГал
            dat1lbl = pmain_proc.DescrStat_lst[0].name;  dat2lbl = pmain_proc.DescrStat_lst[1].name
            xmax = pmain_proc.DescrStat_lst[0].n-1
            print(f'{grav_.min()=}  {grav_.max()=}')
            print(f'{sd_.min()=}  {sd_.max()=}')
            self.dialog = CViewGraph2(self.master, sf_vigraph_sd_grav, graph_name, root_geometry_string(form_w_, form_h_, addx, addy), # addx, addy
                                      xn_, grav_, sd_,   'Время, мин', 'Grav, мГал','sd, мГал',
                                      0, xmax,
                                      grav_.min(), grav_.max(),
                                      sd_.min(), sd_.max(),line_width=1)
            # print(' this = ')
            self.dialog.go()

    def file_exit_(self):
        # https://ru.stackoverflow.com/questions/459170
        self.destroy()
        # sys.exit(0)

    # ---- Подменю Расчет
    def c_hist_(self):
        if pinp_struct.curr_nstruct == -1:
            mb.showerror(s_error, ss_fdni)
        else:
            (grav_, sd_, tilt_x, tilt_y, temp_, tide_, dur_, rej_, date_time_, xn_, time_, date_) = pmain_proc.get_data_ini2() #  self.f_get_graph_data_ini()
            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph2_fw, view_graph2_fh)
            graph_name = 'Участок(файл) ' + self.dict_struct["name_sq"]+'.\n Гистограммы исходных данных'+'\n'
            # def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
            #              xn_: np.ndarray, dat1_: np.ndarray, dat2_: np.ndarray, dat1label: str, dat2label: str,
            #              xmin: float, xmax: float,
            #              dat1min: float, dat1max: float, dat2min: float, dat2max: float):
            self.dialog = CViewHist(self.master, sf_vigraph_sd_grav, graph_name, root_geometry_string(form_w_, form_h_, addx, addy),
                                      grav_, sd_, tilt_x, tilt_y, temp_, tide_,
                                      full_grav_data_name[0], full_grav_data_name[1],
                                      full_grav_data_name[2], full_grav_data_name[3],
                                      full_grav_data_name[4], full_grav_data_name[5])
            self.dialog.go()

    def put_to_res_dict(self, num: int, lat_: float, lon_: float, dep_: float, mag_: float, fun_: float):
        self.res_dict['num'] = num;         self.res_dict['lat_'] = lat_
        self.res_dict['lon_'] = lon_;       self.res_dict['dep_'] = dep_
        self.res_dict['mag_'] = mag_;       self.res_dict['fun_'] = fun_
        self.res_dict['file_res_name'] = pmain_proc.log_file_name

    def get_from_res_dict(self) -> (int, float, float, float, float, float):
        num  = self.res_dict['num']
        lat_ = self.res_dict['lat_']
        lon_ = self.res_dict['lon_']
        dep_ = self.res_dict['dep_']
        mag_ = self.res_dict['mag_']
        fun_ = self.res_dict['fun_']
        return num, lat_, lon_, dep_, mag_, fun_

    def c_view_all_res(self):
        # Определяем есть ли файл
        if self.res_list == []:
            mb.showerror(s_error, ss_ccne)
        else:
            fn = self.res_dict['file_res_name']
            path = pathlib.Path(fn)
            if path.is_file():
                os.startfile(fn)
                self.change_status_bar1(self.SBC_hum, self.the_status_bar_label1.status_bar)
            else:
                mb.showerror(s_error, ss_ffne_.center(70)+'\n'+fn)

    def exract_lat_lon_list(self, ini_list) -> list:
        n = len(ini_list)
        res0list = list()
        for i in range(n):
            res0list.append([ini_list[i][0], ini_list[i][1]])
        return res0list

    # def f_view_map_res(self):
    #     # if self.dict_struct['typeof_input'] ==  0:  # не введено
    #     #     mb.showerror(s_error, ss_fdni)
    #     # else:
    #     if pinp_struct.curr_nstruct == -1:
    #         mb.showerror(s_error, ss_fdni)
    #     elif self.res_list == []:
    #         mb.showerror(s_error, ss_ccne)
    #     else:
    #         xmap: np.ndarray  # Lon    из файла
    #         ymap: np.ndarray  # Lat    макросейсмического
    #         zmap: np.ndarray  # I_fact обследования
    #         # Lon   Lat   I_fact ini_lon ini_lat
    #         (xmap, ymap, zmap, xini, yini) = self.f_get_graph_data_ini()
    #         xres: float = self.res_dict['lon_']  # Результат Lon
    #         yres: float = self.res_dict['lat_']  # Результат Lat
    #
    #         lat_lon_list = self.exract_lat_lon_list(self.res_list)
    #
    #         (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph2_fw, view_graph2_fh)
    #         map_name = 'Участок ' + self.dict_struct["name_sq"]+'.\n Карта интенсивности I_fact, всех результатов и выбранного результата минимизации'+'\n'
    #
    #         # def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
    #         #              xn_: np.ndarray, dat1_: np.ndarray, dat2_: np.ndarray, dat1label: str, dat2label: str,
    #         #              xmin: float, xmax: float,
    #         #              dat1min: float, dat1max: float, dat2min: float, dat2max: float):
    #
    #         self.dialog = CViewGraph2(self.master, ss_cvrmap, map_name, root_geometry_string(form_w_, form_h_, addx, addy),
    #                                   xmap, ymap, zmap, xres, yres, lat_lon_list, False)
    #         self.dialog.go()

    def calc_view_graph_res_event(self):
        self.f_view_graph_res()

    def calc_view_graph_res_event_btn(self, event):
        self.f_view_graph_res()

    def calc_i_mod_for_res(self, r, mag_: float):

        dat = pinp_struct.makroseis_fun(a=self.dict_struct['a'], b=self.dict_struct['b'], c=self.dict_struct['c'],
                                        dist=r, mag=mag_, type_of_macro_fun_=pinp_struct.type_of_macro_fun)
        return dat

    def calc_len_2intens(self, num_res) -> list:
        n = self.dict_struct['npoint']
        spec_list = [[]]*n  # создали список из n пустых списков

        (the_dict, the_arr) = pinp_struct.get_dat_struct(pinp_struct.curr_nstruct)
        d = self.res_list[num_res]
        hypo_lat = d[0]
        hypo_lon = d[1]
        hypo_dep = d[2]
        hypo_mag = d[3]
        # hypo_lat = self.res_dict['lat_']
        # hypo_lon = self.res_dict['lon_']
        # hypo_dep = self.res_dict['dep_']
        # print(hypo_lat,' ',hypo_lon,' ', hypo_dep)
        for i in range(n):
            curr_lat = the_arr[i, 0]
            curr_lon = the_arr[i, 1]
            curr_alt = the_arr[i, 2]/1000
            hypo_len = pinp_struct.calc_distance(curr_lat, curr_lon, curr_alt, hypo_lat, hypo_lon, hypo_dep)
            I_fact = the_arr[i, 3]
            I_mod = self.calc_i_mod_for_res(hypo_len, hypo_mag)
            spec_list[i] = [hypo_len, I_fact, I_mod]
        # сортировка по длине
        # https://ru.stackoverflow.com/questions/1066887/Сортировка-двумерного-массива-по-1-элементу
        spec_list.sort(key=lambda x: x[0])
        # print(spec_list)
        return spec_list


    def f_view_graph_res_all(self):
        if self.dict_struct['typeof_input'] != 1:
            mb.showerror(s_error, ss_fdni)
        if self.res_list == []:
            mb.showerror(s_error, ss_ccne)
        else:
            num = 0
            spec_list = self.calc_len_2intens(num)
            (x_len, y_i_fact, y_i_mod) = pfunct.list2d3_to_3nparray(spec_list)

            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w,
                                                                     self.scr_h, view_graph_fw, view_graph_fh)
            map_name = 'Участок ' + self.dict_struct["name_sq"]+'.\n Графики исходной (синий) и расчитанной ' + str(num)+' (краcный) интенсивности'
            self.dialog = CViewGraph(self.master, sc_gres, map_name, root_geometry_string(form_w_, form_h_, addx, addy),
                                    x_len, y_i_fact, y_i_mod)
            self.dialog.go()

    def f_view_graph_res(self):
        if self.dict_struct['typeof_input'] != 1:
            mb.showerror(s_error, ss_fdni)
        if self.res_list == []:
            mb.showerror(s_error, ss_ccne)
        else:
            num = self.res_dict['num']
            spec_list = self.calc_len_2intens(num)
            (x_len, y_i_fact, y_i_mod) = pfunct.list2d3_to_3nparray(spec_list)

            (form_w_, form_h_, addx, addy) = center_form_positioning(self.scr_w, self.scr_h, view_graph_fw, view_graph_fh)
            map_name = 'Участок ' + self.dict_struct["name_sq"]+'.\n Графики исходной (синий) и расчитанной (краcный) итоговой интенсивности'
            self.dialog = CViewGraph(self.master, sc_gres, map_name, root_geometry_string(form_w_, form_h_, addx, addy),
                                    x_len, y_i_fact, y_i_mod)

    # ---- Подменю Помощь
    def help_about_(self):
        mb.showinfo(sh_about, sh_about1)
        self.change_status_bar1(self.SBC_hab, self.the_status_bar_label1.status_bar)

    def help_usrmanual_(self):
        # Открытие файла в оконном режиме   https://www.cyberforum.ru/python/thread2047476.html
        # Определяем есть ли файл
        path = pathlib.Path(usr_manual_fn_ini)
        if path.is_file():
            os.startfile(usr_manual_fn_ini)
            self.change_status_bar1(self.SBC_hum, self.the_status_bar_label1.status_bar)
        else:
            mb.showerror(s_error, ss_ffne_.center(70) +'\n' + usr_manual_fn_ini)

    # ---- Подменю Помощь "о программе"

    def create_menu(self):
        # ----- Меню
        # https://metanit.com/python/tutorial/9.10.php
        # TheFont = ('Arial', 14)
        main_menu = Menu()
        # main_menu.config(font=TheFont)
        # main_menu.config(self, TheFont)
        # ---- Подменю Файл
        file_menu = Menu(tearoff=0)  # font=("Verdana", 13)
        file_menu.add_command(label=sf_input, command=self.input_inf_)
        file_menu.add_separator()
        file_menu.add_command(label=sf_vinf, command=self.f_view_inf)
        file_menu.add_command(label=sf_vdat, command=self.f_view_dat)
        file_menu.add_command(label=sf_vstat, command=self.f_view_stat)
        file_menu.add_separator()
        file_menu.add_command(label=sf_vigraph_sd_grav, command=self.f_view_graph_sd_grav)
        file_menu.add_command(label=sf_vigraph_tiltx_tilty, command=self.f_view_graph_tiltx_tilty)
        file_menu.add_command(label=sf_vigraph_temp_tide, command=self.f_view_graph_temp_tide)
        # file_menu.add_command(label=sf_vimap, command=self.graph)
        file_menu.add_separator()
        file_menu.add_command(label=sf_detrend_grav, command=self.f_detrend_grav)
        file_menu.add_separator()
        file_menu.add_command(label=sf_exit, command=self.file_exit_)
        # ---- Подменю Расчет
        calc_menu = Menu(tearoff=0)
        calc_menu.add_command(label="Гистограммы", command=self.c_hist_)
        calc_menu.add_separator()
        # calc_menu.add_command(label="Карта результатов", command=self.f_view_map_res)
        calc_menu.add_command(label=sc_gres+'               Ctrl-G', command=self.f_view_graph_res)  # График расчетной интенсивности
#       calc_menu.add_command(label=sc_gres_all, command=self.f_view_graph_res_all)  # Перебор графиков расчетной интенсивности
        calc_menu.add_separator()
        calc_menu.add_command(label="Файл: все результаты минимизации", command=self.c_view_all_res)

        # ---- Подменю Настройки
        # opti_menu = Menu(tearoff=0)
        # opti_menu.add_command(label=so_menufont, command=self.o_menufont) # 'Тестирование: квадрат, равномерная сетка'
        # opti_menu.add_command(label="Настройка сохранения")
        # ---- Подменю Помощь
        help_menu = Menu(tearoff=0)
        help_menu.add_command(label=sh_usrmanual, command=self.help_usrmanual_)
        help_menu.add_separator()
        help_menu.add_command(label=sh_about, command=self.help_about_)  # , font = TheFont
        # ---- Главное меню
        main_menu.add_cascade(label="Файл", menu=file_menu)
        main_menu.add_cascade(label="Вычисления", menu=calc_menu)
        # main_menu.add_cascade(label="Тестирование", menu=test_menu)
        # main_menu.add_cascade(label="Настройки", menu=opti_menu)
        main_menu.add_cascade(label=sh_help, menu=help_menu)

        return main_menu

# ------------ class MakroseisGUI -------- END


class CViewTxt:
    def __init__(self, master, win_title: str, the_root_geometry_string: str, font_str: str =''):
        self.slave = Toplevel(master)
        self.slave.iconbitmap(ico_progr)
        self.slave.title(win_title)
        self.slave.geometry(the_root_geometry_string)
        self.frame = Frame(self.slave)
        self.frame.pack(side=BOTTOM)
        # Python - Tkinter Text
        # https://www.tutorialspoint.com/python/tk_text.htm
        if font_str =='':
            self.text = Text(self.slave, background='white', exportselection=0)
        else:
            self.text = Text(self.slave, background='white', exportselection=0, font=font_str)
        self.text.pack(side=TOP, fill=BOTH, expand=YES)

    def go(self, my_text=''):
        self.text.insert('0.0', my_text)
        self.newValue = None
        self.slave.grab_set()
        self.slave.focus_set()
        self.slave.wait_window()

class CViewGraph:
    # https://ru.stackoverflow.com/questions/602148/Отрисовка-графиков-посредством-matplotlib-в-окне-tkinter
    def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
                 x_len, y_i_fact, y_i_mod):
        self.slave = Toplevel(master)
        self.slave.iconbitmap(ico_progr)
        self.slave.title(win_title)
        self.slave.geometry(the_root_geometry_string)
        self.frame = Frame(self.slave)
        self.frame.pack(side=BOTTOM)

        # self.frame.fig = mpl.figure.Figure(figsize=(5, 5), dpi=100)
        # self.frame.a = self.frame.fig.add_subplot(111)
        # self.frame.a.plot(x_len, y_i_fact , color = 'blue')
        # self.frame.a.plot(x_len, y_i_mod , color = 'red')
        # self.frame.a.set_title(map_name)

        self.frame.fig,  self.frame.ax = plt.subplots(nrows=1)
        self.frame.ax.plot(x_len, y_i_fact, color='blue')
        self.frame.ax.plot(x_len, y_i_mod , color='red')
        self.frame.ax.set_title(map_name, fontsize=15, fontname='Times New Roman')
        self.frame.ax.set_xlabel('Расстояние от гипоцентра, км')
        self.frame.ax.set_ylabel('Интенсивность')
        self.frame.ax.grid()

        self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        self.frame.canvas.draw()
        self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

    def go(self):
        self.newValue = None
        self.slave.grab_set()
        self.slave.focus_set()
        self.slave.wait_window()


class CViewHist: # просмотр гистограмм
    # https://ru.stackoverflow.com/questions/602148/Отрисовка-графиков-посредством-matplotlib-в-окне-tkinter
    def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
                 grav_, sd_, tilt_x, tilt_y, temp_, tide_,
                 grav_name, sd_name, tilt_x_name, tilt_y_name, temp_name, tide_name):
        self.slave = Toplevel(master)
        self.slave.iconbitmap(ico_progr)
        self.slave.title(win_title)
        self.slave.geometry(the_root_geometry_string)
        self.frame = Frame(self.slave)
        self.frame.pack(side=BOTTOM)

        # self.frame.fig = mpl.figure.Figure(figsize=(5, 5), dpi=300)
        # self.frame.a = self.frame.fig.add_subplot(111)
        # self.frame.a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        # self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        # self.frame.canvas.draw()
        # self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        # self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

        # self.frame.fig,  self.frame.ax2 = plt.subplots(nrows=3,ncols=1)
        self.frame.fig = mpl.figure.Figure(figsize=(25, 25), dpi=300)
        self.frame.a1 = self.frame.fig.add_subplot(231)
        # https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/
        self.frame.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

        # self.frame.fig.ylim(0,2)
        # Расстановка дат https://pythonru.com/biblioteki/pyplot-uroki
        # ax1 = self.frame.fig.add_axes([0, 63000, 0, 2])
        # counts, bins = np.histogram(dat[i])
        # plt.hist(dat[i], density=False, bins=len(counts))
        counts, bins = np.histogram(grav_);
        self.frame.a1.hist(grav_, ensity=False, bins=len(counts), linewidth=0.25)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        # self.frame.a1.set(xlim=(xmin,xmax), ylim=(dat1min, dat1max))
        # self.frame.a.plot(xn_, sd_ ,label='sd, мГал', linewidth=0.25)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        self.frame.a1.set_title(grav_name, fontsize=4, fontname='Times New Roman')
        # self.frame.a1.set_xlabel(xlabel, fontsize=3)
        # self.frame.a1.set_ylabel(dat1label, fontsize=4)
        self.frame.a1.tick_params(axis='both', which='major', labelsize=3)
        self.frame.a1.grid(color = 'gray', linewidth = 0.25, linestyle = '--')
        self.frame.a1.grid(which='minor', color='gray', linestyle=':')
        self.frame.a1.tick_params(which='major', length=1, width=0.25)

        # self.frame.a2 = self.frame.fig.add_subplot(212)
        # self.frame.a2.plot(xn_, dat2_, label=dat2label, linewidth=0.25)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        # self.frame.a2.set(xlim=(xmin,xmax), ylim=(dat2min, dat2max))
        # self.frame.a2.set_xlabel(xlabel, fontsize=3)
        # self.frame.a2.set_ylabel(dat2label, fontsize=4)
        # self.frame.a2.tick_params(axis='both', which='major', labelsize=3)
        # self.frame.a2.grid(color = 'gray', linewidth = 0.25, linestyle = '--')
        # self.frame.a2.grid(which='minor', color='gray', linestyle=':')
        # self.frame.a2.tick_params(which='major', length=1, width=0.25)

        self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        self.frame.canvas.draw()
        self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

    def go(self):
        self.newValue = None
        self.slave.grab_set()
        self.slave.focus_set()
        self.slave.wait_window()



class CViewGraph2:
    # https://ru.stackoverflow.com/questions/602148/Отрисовка-графиков-посредством-matplotlib-в-окне-tkinter
    def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
                 xn_: np.ndarray, dat1_: np.ndarray, dat2_: np.ndarray, xlabel:str, dat1label:str, dat2label:str,
                 xmin:float, xmax:float,
                 dat1min:float, dat1max:float,
                 dat2min:float, dat2max:float,line_width=0.25):
        self.slave = Toplevel(master)
        self.slave.iconbitmap(ico_progr)
        self.slave.title(win_title)
        self.slave.geometry(the_root_geometry_string)
        self.frame = Frame(self.slave)
        self.frame.pack(side=BOTTOM)

        # self.frame.fig = mpl.figure.Figure(figsize=(5, 5), dpi=300)
        # self.frame.a = self.frame.fig.add_subplot(111)
        # self.frame.a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        # self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        # self.frame.canvas.draw()
        # self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        # self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

        # self.frame.fig,  self.frame.ax2 = plt.subplots(nrows=3,ncols=1)
        self.frame.fig = mpl.figure.Figure(figsize=(25, 25), dpi=300)
        self.frame.a1 = self.frame.fig.add_subplot(211)
        # https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/
        self.frame.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

        # self.frame.a1.ylim(dat1min, dat1max)
        # Расстановка дат https://pythonru.com/biblioteki/pyplot-uroki
        ax1 = self.frame.fig.add_axes([xmin,xmax, dat1min, dat1max])
        self.frame.a1.plot(xn_, dat1_, label=dat1label, linewidth=line_width)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        # self.frame.a1.set(xlim=(xmin,xmax), ylim=(dat1min, dat1max))
        print('-inner-')
        print(dat1_)
        print(dat1min, dat1max)
        print(dat2min, dat2max)
        print(dat1label)
        self.frame.a1.set_title(map_name, fontsize=4, fontname='Times New Roman')
        self.frame.a1.set_xlabel(xlabel, fontsize=3)
        self.frame.a1.set_ylabel(dat1label, fontsize=4)
        self.frame.a1.tick_params(axis='both', which='major', labelsize=3)
        self.frame.a1.grid(color = 'gray', linewidth = 0.25, linestyle = '--')
        self.frame.a1.grid(which='minor', color='gray', linestyle=':')
        self.frame.a1.tick_params(which='major', length=1, width=0.25)

        self.frame.a2 = self.frame.fig.add_subplot(212)
        self.frame.a2.plot(xn_, dat2_, label=dat2label, linewidth=line_width)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        self.frame.a2.set(xlim=(xmin,xmax), ylim=(dat2min, dat2max))
        self.frame.a2.set_xlabel(xlabel, fontsize=3)
        self.frame.a2.set_ylabel(dat2label, fontsize=4)
        self.frame.a2.tick_params(axis='both', which='major', labelsize=3)
        self.frame.a2.grid(color = 'gray', linewidth = 0.25, linestyle = '--')
        self.frame.a2.grid(which='minor', color='gray', linestyle=':')
        self.frame.a2.tick_params(which='major', length=1, width=0.25)

        self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        self.frame.canvas.draw()
        self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

    def go(self):
        self.newValue = None
        self.slave.grab_set()
        self.slave.focus_set()
        self.slave.wait_window()

class CViewGraph1Plot:
    # https://ru.stackoverflow.com/questions/602148/Отрисовка-графиков-посредством-matplotlib-в-окне-tkinter
    def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
                 xn_: np.ndarray, dat1_: np.ndarray, dat2_: np.ndarray, xlabel: str, dat1label: str, dat2label: str,
                 xmin: float, xmax: float,
                 dat1min: float, dat1max: float, dat2min: float, dat2max: float):
        self.slave = Toplevel(master)
        self.slave.iconbitmap(ico_progr)
        self.slave.title(win_title)
        self.slave.geometry(the_root_geometry_string)
        self.frame = Frame(self.slave)
        self.frame.pack(side=BOTTOM)

        self.frame.fig = mpl.figure.Figure(figsize=(25, 25), dpi=300)
        self.frame.a1 = self.frame.fig.add_subplot(111)
        # https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/
        self.frame.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

        # self.frame.fig.ylim(0,2)
        # Расстановка дат https://pythonru.com/biblioteki/pyplot-uroki
        # ax1 = self.frame.fig.add_axes([0, 63000, 0, 2])
        self.frame.a1.plot(xn_, dat1_, label=dat1label,
                           linewidth=0.25)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        self.frame.a1.set(xlim=(xmin, xmax), ylim=(dat1min, dat1max))
        self.frame.a1.set_title(map_name, fontsize=4, fontname='Times New Roman')
        self.frame.a1.set_xlabel(xlabel, fontsize=3)
        self.frame.a1.set_ylabel(dat1label, fontsize=4)
        self.frame.a1.tick_params(axis='both', which='major', labelsize=3)
        self.frame.a1.grid(color='gray', linewidth=0.25, linestyle='--')
        self.frame.a1.grid(which='minor', color='gray', linestyle=':')
        self.frame.a1.tick_params(which='major', length=1, width=0.25)

        self.frame.a1.plot(xn_, dat2_, color='red', label=dat2label,
                           linewidth=0.25)  # , 'o', ms=12, color="red"  ko  , linewidth= 3
        self.frame.a1.set(xlim=(xmin, xmax), ylim=(dat2min, dat2max))
        self.frame.a1.legend(fontsize = 5)

        self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        self.frame.canvas.draw()
        self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

    def go(self):
        self.newValue = None
        self.slave.grab_set()
        self.slave.focus_set()
        self.slave.wait_window()


class CViewMap3:
    # https://ru.stackoverflow.com/questions/602148/Отрисовка-графиков-посредством-matplotlib-в-окне-tkinter
    def __init__(self, master, win_title: str, map_name: str, the_root_geometry_string: str,
                 xmap: np.ndarray, ymap: np.ndarray, zmap: np.ndarray, xini: float, yini: float):
        self.slave = Toplevel(master)
        self.slave.iconbitmap(ico_progr)
        self.slave.title(win_title)
        self.slave.geometry(the_root_geometry_string)
        self.frame = Frame(self.slave)
        self.frame.pack(side=BOTTOM)

        # self.frame.fig = mpl.figure.Figure(figsize=(5, 5), dpi=300)
        # self.frame.a = self.frame.fig.add_subplot(111)
        # self.frame.a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        # self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        # self.frame.canvas.draw()
        # self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        # self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)
        ncolors = 20
        self.frame.fig,  self.frame.ax2 = plt.subplots(nrows=1)
        levels_ = pfunct.calc_log_levels(zmap, nlevel = ncolors)
        # Изолинии
        self.frame.ax2.tricontour(xmap, ymap, zmap, levels_, linewidths=0.5, colors='k')
        # cntr2 = self.frame.ax2.tricontourf(xmap, ymap, zmap, levels=24, cmap="RdBu_r")
        # Цветовые схемы
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        # cmap = plt.get_cmap('viridis', 21))
        # cmap="RdBu_r" 'RdYlGn_r
        cmap2 = 'pink'
        cntr2 = self.frame.ax2.tricontourf(xmap, ymap, zmap, levels_, cmap=cmap2) # cmap="RdBu_r" cmap=pfunct.red2blue_21colors()

        # n = len(xmap)
        # for i in range(n):
        #     self.frame.ax2.plot(xmap[i], ymap[i], 'o', ms=4, color="yellow")  # green

        # if self.is_ini_map: # Карта исходных данных
        #     (lat_arr, lon_arr, ifact_arr) = pinp_struct.get_Lat_Lon_ifact()
        #     n = len(lat_arr)
        #     for i in range(n):
        #         xmap1 = float(lon_arr[i])
        #         ymap1 = float(lat_arr[i])
        #         zmap1 = float(ifact_arr[i])
        #         # https://pythonru.com/biblioteki/pyplot-uroki
        #         self.frame.ax2.text(xmap1, ymap1, format(zmap1, '2.1f'), fontsize=6)  # green

        self.frame.fig.colorbar(cntr2, ax=self.frame.ax2, label='Значения целевой функции')
        self.frame.ax2.plot(xini, yini, 'o', ms=12, color="red")  # ko
        # self.frame.ax2.plot(xmap, ymap, 'o', ms=3, color="orange")
        self.frame.ax2.set(xlim=(min(xmap), max(xmap)), ylim=(min(ymap), max(ymap)))
        self.frame.ax2.set_title(map_name, fontsize=15, fontname='Times New Roman')
        self.frame.ax2.set_xlabel('Глубина, км')
        self.frame.ax2.set_ylabel('Магнитуда')
        self.frame.ax2.grid()
        # self.frame.a = self.frame.fig.add_subplot(111)
        # self.frame.a.draw()
        self.frame.canvas = FigureCanvasTkAgg(self.frame.fig, self.slave)
        self.frame.canvas.draw()
        self.frame.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        self.frame.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=True)

    def go(self):
        self.newValue = None
        self.slave.grab_set()
        self.slave.focus_set()
        self.slave.wait_window()



class CViewMap3__:  # Рисование на главном окне
    # https://ru.stackoverflow.com/questions/602148/Отрисовка-графиков-посредством-matplotlib-в-окне-tkinter
    def __init__(self, master):
        self.fig = mpl.figure.Figure(figsize=(5, 5), dpi=300)
        self.a = self.fig.add_subplot(111)
        self.a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


# def btn_esc(event):             ## ========= Выход
#     sys.exit(0)
#     # root.bind('<Escape>', btn_esc)

def the_program() -> None:
    # ----- Инициализация, начальные установки окна
    root = Tk()
    # https://question-it.com/questions/1334867/izmenit-shrift-po-umolchaniju-v-python-tkinter
    # default_font = font.nametofont("TkDefaultFont")
    # default_font.configure(size=12)
    # default_font = font.nametofont("TkTextFont")
    # default_font.configure(size=12)
    # default_font = font.nametofont("TkFixedFont")
    # default_font.configure(size=12)
    # root.option_add("*Font", default_font)
    # print(font.families())

#   root.bind('<Escape>', btn_esc)  ## ========= Выход
    makroseis = GTimeSeriesGUI(root)
    makroseis.mainloop()
    # root.mainloop()
