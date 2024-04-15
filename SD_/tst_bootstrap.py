"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

rng = np.random.default_rng()
from scipy.stats import norm
dist = norm(loc=2, scale=4)  # our "unknown" distribution
data = dist.rvs(size=100, random_state=rng)

std_true = dist.std()      # the true value of the statistic
print(std_true) # 4.0
std_sample = np.std(data)  # the sample statistic
print(std_sample)  #3.9460644295563863

Nbins=125
data = (data,)  # samples must be in a sequence
res = bootstrap(data, np.std, confidence_level=0.9,
                random_state=rng)
fig, ax = plt.subplots()
ax.hist(res.bootstrap_distribution, bins=Nbins)
ax.set_title('Bootstrap Distribution Std')
ax.set_xlabel('statistic value')
ax.set_ylabel('frequency')
plt.grid()
plt.show()

res = bootstrap(data, np.mean, confidence_level=0.9,
                random_state=rng)
fig, ax = plt.subplots()
ax.hist(res.bootstrap_distribution, bins=Nbins)
ax.set_title('Bootstrap Distribution Mean')
ax.set_xlabel('statistic value')
ax.set_ylabel('frequency')
plt.grid()
plt.show()