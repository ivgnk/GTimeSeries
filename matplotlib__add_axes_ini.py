'''
2022 What is add_axes matplotlib
https://pythonguides.com/add_axes-matplotlib/
'''

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,5))
i = 1
match i:
    case 0:
         axes = fig.add_axes(rect=(0.1, 0.1, 0.85, 0.85))
         plt.grid()
    case 1:
        # plt.figure(figsize=(12,6))
        axes1 = fig.add_axes(rect=(0.1, 0.1, 0.3, 0.85))
        plt.title('x1')
        plt.xlabel('x')  # название оси абсцисс
        plt.ylabel('y') # название оси ординат
        plt.plot([2,3,4,5,6],[12,13,14,15,16])
        plt.grid()
        axes2 = fig.add_axes(rect=(0.65, 0.1, 0.3, 0.85))
        plt.title('x2')
        plt.xlabel('x')  # название оси абсцисс
        plt.ylabel('y') # название оси ординат
        plt.plot([12,13,14,15,16],[-12,-13,-14,-15,-16])
        plt.grid()
plt.show()