'''
On the base of
https://www.statsmodels.org/stable/regression.html
'''
import inspect

import statsmodels.api as sm
import numpy as np

def proc_OLS(y:np.ndarray,x:np.ndarray): #  y - A 1-d endogenous response variable. The dependent variable
    '''
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS
    '''
    print('\nFunction',inspect.currentframe().f_code.co_name)

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    print(f'{results.params=}')
    print(f'{results.tvalues=}')
    # print(f'{results.t_test([1, 0])=}')
    print()
    return results.params
