'''
2022 What is add_axes matplotlib
https://pythonguides.com/add_axes-matplotlib/
!!! Описание не работает переделал сам
'''

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

x = np.arange(0, 20, 0.2)
y1 = np.sin(x)
y2 = np.sqrt(x)*250

# https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_axes
# recttuple (left, bottom, width, height)
axes1 = fig.add_axes(rect=(0.1, 0.1, 0.8, 0.4))
plt.grid()
axes1.plot(x, y1) # sin

axes2 = fig.add_axes(rect=(0.1, 0.6, 0.8, 0.4))
axes2.plot(x, y2) # sqrt
plt.grid()

plt.show()