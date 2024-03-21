'''
August 29, 2022 Python Scipy Butterworth Filter
https://pythonguides.com/python-scipy-butterworth-filter/
'''

from scipy import signal
from scipy.signal import butter, bessel, cheby1, cheby2, ellip
import numpy as np
import matplotlib.pyplot as plt
from sympy.core import function

from psd_filters2 import get_subplot_num2

# Const
figsize_st = (15, 12)
figsize_st2 = (10, 8)
flt_type = ['lowpass', 'highpass', 'bandpass', 'bandstop']
flt_func = [butter, bessel, cheby1, cheby2, ellip]  # , cheby2
flt_func_name = ['butter', 'bessel', 'cheby1', 'cheby2', 'ellip']  #


def Python_Scipy_Butterworth_Filter():
    # Create the time duration of the signal
    t_duration = np.linspace(0, 0.5, 2000, endpoint=False)
    #  Generate a signal of 20 and 40 Hz frequency
    sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)

    with plt.ioff():
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize_st2)
        ax1.plot(t_duration, sign)
        ax1.set_title('20 and 40 Hz Sinusoid')
        ax1.axis([0, 0.5, -2, 2.5])
        plt.grid()
        plt.show(block=False)

        # Create a Butterworth high pass filter of 25 Hz and apply it to the above-created signal using the below code.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        st=get_subplot_num2(4)
        print(st)
        fig2, ax2 = plt.subplots(st[0],st[1], sharex=True, figsize=figsize_st2)
        # ax2.axis([0, 0.5, -2, 2.5])
        for i,btype_ in enumerate(flt_type):
            plt.subplot(st[0], st[1], i + 1)
            if (i==0) or (i==1): Wn_ = 20
            else: Wn_ = [10,30]
            print(i,Wn_)
            sos = butter(N = 15, Wn = Wn_, btype=btype_, fs=2000, output='sos')
            filtd = signal.sosfilt(sos, sign)

            plt.plot(t_duration, filtd)
            plt.title('After applying '+btype_+' filter')
            plt.tight_layout()
            plt.xlabel('Time (seconds)')
            plt.grid()
        plt.show()

def Python_Scipy_Bessel_Filter():
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html

    '''
    # Create the time duration of the signal
    t_duration = np.linspace(0, 0.5, 2000, endpoint=False)
    #  Generate a signal of 20 and 40 Hz frequency
    sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)

    with plt.ioff():
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize_st2)
        ax1.plot(t_duration, sign)
        ax1.set_title('20 and 40 Hz Sinusoid')
        ax1.axis([0, 0.5, -2, 2.5])
        plt.grid()
        plt.show(block=False)

        # Create a Butterworth high pass filter of 25 Hz and apply it to the above-created signal using the below code.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        st=get_subplot_num2(4)
        print(st)
        fig2, ax2 = plt.subplots(st[0],st[1], sharex=True, figsize=figsize_st2)
        # ax2.axis([0, 0.5, -2, 2.5])
        for i,btype_ in enumerate(flt_type):
            plt.subplot(st[0], st[1], i + 1)
            if (i==0) or (i==1): Wn_ = 20
            else: Wn_ = [10,30]
            print(i,Wn_)
            sos = bessel(N = 15, Wn = Wn_, btype=btype_, fs=2000, output='sos')
            filtd = signal.sosfilt(sos, sign)

            plt.plot(t_duration, filtd)
            plt.title('After applying '+btype_+' filter')
            plt.tight_layout()
            plt.xlabel('Time (seconds)')
            plt.grid()
        plt.show()

def Python_Scipy_Filters(num_flt:int, flt_func:function, t_duration:np.ndarray, signal_:np.ndarray):
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html
    '''
    flt_func_name = flt_func.__name__+' filter'
    st=get_subplot_num2(4)
    fig2, ax2 = plt.subplots(st[0],st[1], sharex=True, figsize=figsize_st2)
    # ax2.axis([0, 0.5, -2, 2.5])
    plt.suptitle(flt_func_name)
    for i,btype_ in enumerate(flt_type):
        plt.subplot(st[0], st[1], i + 1)
        if (i==0) or (i==1): Wn_ = 20
        else: Wn_ = [10,30]
        match num_flt:
            case 0 | 1:  sos = flt_func(N = 15, Wn = Wn_, btype=btype_, fs=2000, output='sos')
            case 2:
                rp_ = 5
                sos = flt_func(N = 15, rp=rp_, Wn = Wn_, btype=btype_, fs=2000, output='sos')
            case 3:
                rs_ = 40
                sos = flt_func(N=15, rs=rs_, Wn=Wn_, btype=btype_, fs=2000, output='sos')
            case 4:
                rp_ = 5
                rs_ = 40
                sos = flt_func(N=15, rp=rp_, rs=rs_, Wn=Wn_, btype=btype_, fs=2000, output='sos')

        filtd = signal.sosfilt(sos, signal_)

        plt.plot(t_duration, filtd)
        plt.title('After applying '+btype_+' filter')
        plt.tight_layout()
        plt.xlabel('Time (seconds)')
        plt.grid()
    plt.show()

def create_signal()->(np.ndarray, np.ndarray):
    #--(2)----Calculatuions
    # Create the time duration of the signal
    t_duration = np.linspace(0, 0.5, 2000, endpoint=False)
    #  Generate a signal of 20 and 40 Hz frequency
    signal_ = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)

    #--(2)----Graphics
    fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize_st2)
    ax1.plot(t_duration, signal_)
    ax1.set_title('20 and 40 Hz Sinusoid')
    ax1.axis([0, 0.5, -2, 2.5])
    plt.grid()
    plt.show(block=False)

    return t_duration, signal_


def test_Python_Scipy_Filters():
    for s in flt_func_name: print(s);
    print('-----------')
    with plt.ioff():
        t_duration, signal_ = create_signal()
        for i,flt in enumerate(flt_func):
            print(flt.__name__)
            Python_Scipy_Filters(i,flt,t_duration, signal_)

if __name__ == "__main__":
    # Python_Scipy_Butterworth_Filter()
    # Python_Scipy_Bessel_Filter()
    test_Python_Scipy_Filters()

