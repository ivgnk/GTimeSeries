# -------------------------------------------------------------------
# модуль pstring
# Функции для работы со строками
#
# (C) 2023 Ivan Genik, Perm, Russia
# Released under GNU Public License (GPL)
# email igenik@rambler.ru
# ------------------------------
import re

# Пример разбиения на части
# s ='18.02.17 9:52'
# s2 = s.split(sep=' ', maxsplit=2)  # print_string(part_lines)
# print(s2[0].strip())
# print(s2[1].strip())


def print_string(the_list) -> None:
    print(len(the_list))
    for stri in the_list:
        print(stri, end=' ')
    print('+')


def num_words_in_string(s: str) -> int:
    return len(re.split('\s+', s))
    # проверено см. последняя строка файла, все правильно

# print(re.sub(r'\D', '', 'Fjkoweuqe -1245 654lfr'))
# print(num_words_in_string('In the hole in the ground there lived a   hobbit')) # 10
# print(num_words_in_string('В  яме в  земле   жил   хоббит')) # 6
# print(num_words_in_string('min1')) # 1

# Проработать строки и регулярные выражения
# pythonist.ru/s/proverka-yavlyaetsya-li-stroka-palindromom/?utm_source=turbo_turbo
# https://habr.com/ru/post/349860/

# print(num_words_in_string('r4 ry 7 9f'))

# Повтор строки n раз
def repeat_chr(chr_:str, n_repeat:int)->str:
    # https://stackoverflow.com/questions/38273353/how-to-repeat-individual-characters-in-strings-in-python
    return chr_*n_repeat