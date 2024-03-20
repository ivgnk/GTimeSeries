'''
https://www.gaussianwaves.com/2010/11/moving-average-filter-ma-filter-2/
https://colab.research.google.com/drive/1vZsVJIUhET_Q2e-sOcjNWOKNqMKzp0Bm?usp=sharing
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filter_characteristics(num, den, N):
    """
    Compute following filter characteristics for the given transfer function: impulse response, amplitude/phase response, poles-zeros

    Returns:
      impulse_response: Impulse response
      z,p,k : location of zeros z, poles p and system gain k
      w,h: frequencies and complex frequency response
    """

    impulse = np.concatenate(([1.], np.zeros((N - 1)), np.zeros((N // 2))))
    impulse_response = signal.lfilter(num, den, impulse)  # drive the impulse through the filter (impulse response)
    z, p, k = signal.tf2zpk(num, den)  # Transfer function to Pole-Zero representation
    w, h = signal.freqz(num, den, whole=True, worN=512)  # frequency response
    # w, gd = signal.group_delay((num, den), w = 512)

    return impulse_response, z, p, k, w, h


def plot_characteristics(impulse_response, z, p, k, w, h):
    fig, (ax) = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))
    # Plot impulse response
    ax[0].plot(impulse_response, '-o')
    ticks = np.arange(0, len(impulse_response), 1)
    ax[0].set_xticks(ticks)
    ax[0].set(title='Impulse response', xlabel='Sample index [n]', ylabel='Amplitude')
    ax[0].axis('tight')

    # Plot pole-zeros on a z-plane
    from matplotlib import patches
    patch = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dashed')
    ax[1].add_patch(patch)
    ax[1].plot(np.real(z), np.imag(z), 'ob', label='Zeros', markersize=12, markerfacecolor='None',
               markeredgecolor='red')
    ax[1].plot(np.real(p), np.imag(p), 'x', label='Poles', markersize=12)
    ax[1].legend(loc=2)
    ax[1].set(title='Pole-Zero Plot', ylabel='Real', xlabel='Imaginary')
    ticks = np.arange(-1, 1.5, 0.5)
    ax[1].set_xticks(ticks);
    ax[1].set_yticks(ticks)
    ax[1].grid()

    # Plot Magnitude-frequency response
    from scipy.fftpack import fftshift
    ax[2].plot(w / np.pi - 1, 20 * np.log10(np.abs(fftshift(h))), 'b')
    ax[2].grid()
    ax[2].set(title='Magnitude response', xlabel=r'Frequency [$\times \pi $ radians/sample]', ylabel='Magnitude [dB]')

    # Plot Phase-frequency response
    angles = np.angle(fftshift(h))
    ax[3].plot(w / np.pi - 1, angles)
    ax[3].grid()
    ax[3].set(title='Phase response', xlabel=r'Frequency [$\times \pi$ radians/sample]', ylabel='Angles [radians]')
    plt.show()

#------------------ Part 1
# Moving average filter
L = 11  # L-point filter
num = np.ones(L) / L  # numerator co-effs of filter transfer function
den = np.ones(1)  # denominator co-effs of filter transfer function

# Plot filter characteristics
impulse_response, z, p, k, w, h = filter_characteristics(num, den, L)
plot_characteristics(impulse_response, z, p, k, w, h)

#------------------ Part 3
b1 = np.ones(L) / L
b2 = np.ones(L) / L
a = 1
from numpy.polynomial import Polynomial
num = (Polynomial(b1) * Polynomial(b2)).coef
den = np.asarray(a)

# Plot filter characteristics
impulse_response, z, p, k, w, h = filter_characteristics(num, den, L)
plot_characteristics(impulse_response, z, p, k, w, h)