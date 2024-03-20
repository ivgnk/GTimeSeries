'''
Как рисовать прямоугольники в Matplotlib (с примерами)
https://www.codecamp.ru/blog/matplotlib-rectangle/
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def example1():
    # define Matplotlib figure and axis
    # fig, ax = plt.figure(figsize=(16, 8))
    fig, ax = plt.subplots(figsize=(16, 8))
    # create simple line plot
    ax.plot([0, 10], [0, 10])
    # add rectangle to plot
    ax.add_patch(Rectangle((1, 1), 2, 6))
    # display plot
    plt.show()

example1()