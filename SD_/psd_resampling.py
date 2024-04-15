"""
https://en.wikipedia.org/wiki/Resampling_(statistics)

1
2 https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
3
"""
import inspect
import sys
import time

from icecream import ic
from typing import Final

import numpy as np
# from numba import jit
import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.stats import bootstrap

ONLY_RESAMPLING: Final[bool] = bool(1)

def the_bootstrap(x:np.ndarray):
    """
    https://www.codecamp.ru/blog/bootstrapping-in-python/
    """
    # convert array to sequence
    data = (x,)

    # calculate 95% bootstrapped confidence interval for median
    start = time.time()
    bootstrap_ci = bootstrap(data, np.std, confidence_level=0.95,
                             random_state=1, method='percentile') #
    end = time.time()
    ic(end-start)
    # view 95% boostrapped confidence interval
    print(bootstrap_ci.confidence_interval)
    sys.exit()
    # ConfidenceInterval(low=10.0, high=20.0)
