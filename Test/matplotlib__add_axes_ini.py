'''
2022 What is add_axes matplotlib
https://pythonguides.com/add_axes-matplotlib/
'''

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_axes(rect=(0.1, 0.1, 0.85, 0.85))
plt.grid()
plt.show()