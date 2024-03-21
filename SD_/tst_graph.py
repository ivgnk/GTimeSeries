'''
Использование библиотеки Matplotlib. Как рисовать графики в разных окнах
https://jenyay.net/Matplotlib/Figure
'''
import matplotlib.pyplot as plt
import numpy as np

def tst1():
    xmin = -20.0
    xmax = 20.0
    count = 200

    # Создадим список координат по оси X на отрезке [xmin; xmax], включая концы
    x = np.linspace(xmin, xmax, count)

    # Вычислим значение функции в заданных точках
    y1 = np.sinc(x / np.pi)
    y2 = np.sinc(x / np.pi * 0.2)

    # !!! Создадим первое окно и нарисуем в нем график
    plt.figure()
    plt.plot(x, y1, label="f(x)")
    plt.show()

    # !!! Создадим второе окно и нарисуем график в нем
    plt.figure()
    plt.plot(x, y2, label="f(x * 0.2)")
    plt.legend()

    # Покажем окна с нарисованными графиками
    plt.show()


if __name__ == "__main__":
    tst1()

