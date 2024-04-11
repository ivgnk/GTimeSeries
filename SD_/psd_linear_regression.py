'''
On the base of
https://www.statsmodels.org/stable/regression.html
'''
import inspect
import sys

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def proc_OLS(y:np.ndarray,x:np.ndarray): #  y - A 1-d endogenous response variable. The dependent variable
    '''
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS
    '''
    print('\nFunction',inspect.currentframe().f_code.co_name)

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    print(f'{results.params=}\n')
    print('results.params')
    print(results.params)
    print(f'{results.tvalues=}')
    print()
    # print(results.t_test([1, 0]))
    return results.params

def thetst_proc_OLS():
    np.random.seed(124)
    num = 100
    x = np.linspace(1,2,num)
    # y = np.sin(x/(2*np.pi))
    rrnd = np.random.random(num)
    y = x+ rrnd
    line_coeff= proc_OLS(y, x)
    # gr1 = znp*lst[0]+lst[1]; GraphCl_lst.append(GraphCl('gr1',znp,gr1))
    gr2 = x * line_coeff[1] + line_coeff[0]  # прямая линия
    plt.plot(x,y,label='ini')
    plt.plot(x,gr2,label='regr')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    thetst_proc_OLS()
