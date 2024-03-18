import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


# https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
def MA_equal_weight(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    print(window)
    return np.convolve(interval, window, 'same')



def test_MA_equal_weight():
    n = 500
    llstx = [i for i in range(0, n)]
    llsty = [np.sin(i*np.pi/30) for i in range(0,n)]
    print(llsty)
    res=MA_equal_weight(np.array(llsty),45)
    print(res)
    pl.plot(llstx, llsty)
    pl.plot(llstx, res)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test_MA_equal_weight()
