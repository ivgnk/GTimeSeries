'''
2022 What is add_axes matplotlib
https://pythonguides.com/add_axes-matplotlib/
'''

import matplotlib.pyplot as plt
fig = plt.figure()
i = 1
match i:
    case 0:
         axes = fig.add_axes(rect=(0.1, 0.1, 0.85, 0.85))
         plt.grid()
    case 1:
        axes1 = fig.add_axes(rect=(0.1, 0.1, 0.35, 0.85))
        plt.grid()
        axes2 = fig.add_axes(rect=(0.6, 0.1, 0.35, 0.85))
        plt.grid()
plt.show()