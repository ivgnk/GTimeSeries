'''
Spectral Analysis using Welch’s Method
https://docs.scipy.org/doc/scipy/tutorial/signal.html
'''
import sys

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math
import cmath
from copy import deepcopy


fs = 10e3
N = 1e4
amp = 2*np.sqrt(2)
freq = 1270.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
plt.plot(time,x,  linewidth=0.4)
plt.xlabel('t [s]')
plt.ylabel('F(t)')
plt.show()
f, Pwelch_spec = signal.welch(x, fs, scaling='spectrum')
plt.semilogy(f, Pwelch_spec)
# plt.plot(f, Pwelch_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
plt.show()

# FFT from Теория поля_12_Преобразование Фурье (ДПФ). Фурье-анализ с NumPy (2024).ppt
prfft = np.fft.fft(x)  # одномерное прямое преобразование Фурье
print('--------------Действительная и мнимая часть--------------')
nel=len(x); mod_ = deepcopy(x); pha_ = deepcopy(x)
for i in range(nel):
    r_ = prfft[i].real
    i_ = prfft[i].imag
    mod_[i] = math.sqrt(r_*r_+i_**2)   # Вычисляем модуль
    pha_[i] = cmath.phase(prfft[i])    # Вычисляем фазу

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)  # левый верхний
plt.title('prfft.real')
xx= range(nel)
plt.plot(xx,prfft.real)
plt.subplot(2, 2, 2) # правый верхний
plt.title('prfft.imag')
plt.plot(xx,prfft.imag)
plt.subplot(2, 2, 3) # левый нижний
plt.title('prfft.mod')
plt.plot(xx,mod_)
plt.subplot(2, 2, 4) # правый нижний
plt.title('prfft.phase')
plt.plot(xx,pha_)
plt.show()

