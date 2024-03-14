'''
2022 What is add_axes matplotlib
https://pythonguides.com/add_axes-matplotlib/
!!! Описание не работает переделал сам
'''

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

x = [3, 6, 9, 12, 15]
y = [5.5, 8, 10.5, 23, 12]

plt.plot(x, y)
plt.plot(x, y,'bo')
plt.grid()

axes = fig.add_axes(rect=(0.20, 0.48, 0.28, 0.28))
axes.plot([1, 5])
# plt.grid()

# display

plt.show()