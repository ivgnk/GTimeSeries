import matplotlib.pyplot as plt

fig = plt.figure()
axes = fig.add_axes(rect=(0.1, 0.1, 0.85, 0.85))
zz = [1, 5]
axes.plot(zz)
axes.plot(zz,'bo')
plt.grid()
plt.show()
