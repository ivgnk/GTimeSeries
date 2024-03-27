'''
August 29, 2022 Python Scipy Butterworth Filter
https://pythonguides.com/python-scipy-butterworth-filter/
'''

from random import *

from scipy import signal
from scipy.signal import butter, bessel, cheby1, cheby2, ellip
import numpy as np
import matplotlib.pyplot as plt
from sympy.core import function

from psd_filters2 import get_subplot_num2

# Const
figsize_st = (15, 12)
figsize_st2 = (10, 8)
rand_coeff= 2
flt_type = ['lowpass', 'highpass', 'bandpass', 'bandstop']
flt_func = [butter, bessel, cheby1, cheby2, ellip]  # , cheby2
flt_func_name = ['butter', 'bessel', 'cheby1', 'cheby2', 'ellip']  #


def testFull_Python_Scipy_Butterworth_Filter(N_Wn:list, n_signal=0, nnums=2000, rand_coeff = 2.0): # Only LowPass
    # Create the time duration of the signal
    t_duration = np.linspace(0, 0.5, nnums, endpoint=False)
    #  Generate a signal of 20 and 40 Hz frequency
    seed(123); llen = len(t_duration)
    match n_signal:
        case 0: sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)
        case 1: sign = np.array([random()*rand_coeff for i in range(llen)])
        case 2:
                sign1 = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)
                sign2 = np.array([random()*rand_coeff for i in range(llen)])
                sign = sign1+sign2
        case default: sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)

    with plt.ioff():
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize_st2)
        if n_signal==2:
            ax1.plot(t_duration, sign1,label='regular signal',linewidth= 3)
            ax1.plot(t_duration, sign2,label='random signal')
        else:
            ax1.plot(t_duration, sign,label='full signal')
        ax1.set_title('20 and 40 Hz Sinusoid')
        ax1.axis([0, 0.5, -2, 2.5])
        plt.grid()
        plt.legend()
        plt.show(block=False)

        # Create a Butterworth high pass filter of 25 Hz and apply it to the above-created signal using the below code.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        st=get_subplot_num2(len(N_Wn))
        print(st)
        fig2, ax2 = plt.subplots(st[0],st[1], sharex=True, figsize=figsize_st)
        # fig2.suptitle(f'different N & Wn',color='r')
        # ax2.axis([0, 0.5, -2, 2.5])
        for i,dat in enumerate(N_Wn):
            plt.subplot(st[0], st[1], i + 1)
            sos = butter(N = dat[0], Wn = dat[1], btype='low', fs=2000, output='sos')
            filtd = signal.sosfilt(sos, sign)

            plt.plot(t_duration, filtd)
            plt.title(f'N={dat[0]}  Wn={dat[1]} filter')
            plt.tight_layout()
            plt.xlabel('Time (seconds)')
            plt.grid()
        plt.show()

def test1Wn_Python_Scipy_Butterworth_Filter(n_signal=0, nnums=2000, rand_coeff= 2): # Only LowPass
    # Create the time duration of the signal
    t_duration = np.linspace(0, 0.5, nnums, endpoint=False)
    #  Generate a signal of 20 and 40 Hz frequency
    seed(123); llen = len(t_duration)
    match n_signal:
        case 0: sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)
        case 1: sign = np.array([random()*rand_coeff for i in range(llen)])
        case 2: sign = (np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)+
                        np.array([random()*rand_coeff for i in range(llen)]))
        case default: sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)

    with plt.ioff():
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize_st2)
        ax1.plot(t_duration, sign)
        ax1.set_title('20 and 40 Hz Sinusoid')
        ax1.axis([0, 0.5, -2, 2.5])
        plt.grid()
        plt.show(block=False)

        # Create a Butterworth high pass filter of 25 Hz and apply it to the above-created signal using the below code.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        Wn_ = [1, 2, 5, 6, 7, 8, 9, 10, 15, 20,30]
        st=get_subplot_num2(len(Wn_))
        print(st)
        fig2, ax2 = plt.subplots(st[0],st[1], sharex=True, figsize=figsize_st)
        N_ = 5
        fig2.suptitle(f'{N_=}, different Wn',color='r')
        # ax2.axis([0, 0.5, -2, 2.5])
        for i,dat in enumerate(Wn_):
            plt.subplot(st[0], st[1], i + 1)
            sos = butter(N = N_, Wn = dat, btype='low', fs=2000, output='sos')
            filtd = signal.sosfilt(sos, sign)

            plt.plot(t_duration, filtd)
            plt.title('After applying Wn='+str(dat)+' filter')
            plt.tight_layout()
            plt.xlabel('Time (seconds)')
            plt.grid()
        plt.show()

def test1N_Python_Scipy_Butterworth_Filter(n_signal=0, nnums=2000, rand_coeff= 2): # Only LowPass
    # Create the time duration of the signal
    t_duration = np.linspace(0, 0.5, nnums, endpoint=False)
    #  Generate a signal of 20 and 40 Hz frequency
    seed(123); llen = len(t_duration)
    match n_signal:
        case 0: sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)
        case 1: sign = np.array([random()*rand_coeff for i in range(llen)])
        case 2: sign = (np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)+
                        np.array([random()*rand_coeff for i in range(llen)]))
        case default: sign = np.sin(2 * np.pi * 20 * t_duration) + np.sin(2 * np.pi * 40 * t_duration)

    with plt.ioff():
        fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=figsize_st2)
        ax1.plot(t_duration, sign)
        ax1.set_title('20 and 40 Hz Sinusoid')
        ax1.axis([0, 0.5, -2, 2.5])
        plt.grid()
        plt.show(block=False)

        # Create a Butterworth high pass filter of 25 Hz and apply it to the above-created signal using the below code.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        N = [1, 2, 5, 6, 7, 8, 9, 10, 15, 20,30]
        st=get_subplot_num2(len(N))
        print(st)
        fig2, ax2 = plt.subplots(st[0],st[1], sharex=True, figsize=figsize_st)
        Wn_ = 15
        fig2.suptitle(f'{Wn_=}, different N',color='r')
        # ax2.axis([0, 0.5, -2, 2.5])
        for i,dat in enumerate(N):
            plt.subplot(st[0], st[1], i + 1)
            sos = butter(N = dat, Wn = 15, btype='low', fs=2000, output='sos')
            filtd = signal.sosfilt(sos, sign)

            plt.plot(t_duration, filtd)
            plt.title('After applying N ='+str(dat)+' filter')
            plt.tight_layout()
            plt.xlabel('Time (seconds)')
            plt.grid()
        plt.show()


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
    flt_func = [butter, bessel, cheby1, cheby2, ellip]  # , cheby2
    num_flt          0        1      2       3       4
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
            case 0 | 1:  sos = flt_func(N = 15, Wn = Wn_, btype=btype_, fs=2000, output='sos') # butter, bessel
            case 2:
                rp_ = 5
                sos = flt_func(N = 15, rp=rp_, Wn = Wn_, btype=btype_, fs=2000, output='sos')  # cheby1
            case 3:
                rs_ = 40
                sos = flt_func(N=15, rs=rs_, Wn=Wn_, btype=btype_, fs=2000, output='sos')  # cheby2
            case 4:
                rp_ = 5
                rs_ = 40
                sos = flt_func(N=15, rp=rp_, rs=rs_, Wn=Wn_, btype=btype_, fs=2000, output='sos')  # ellip

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
    #  flt_func = [butter, bessel, cheby1, cheby2, ellip]  # , cheby2
    #     i          0        1      2       3       4
    print('-----------')
    with plt.ioff():
        t_duration, signal_ = create_signal()
        for i,flt in enumerate(flt_func):
            print(flt.__name__)
            Python_Scipy_Filters(i,flt,t_duration, signal_)

if __name__ == "__main__":
    N_Wn = [[1,1],[1,2],[1,5],[1,10],[1,15],[1,20],[1,40],[1,60],[1,100]]
    testFull_Python_Scipy_Butterworth_Filter(N_Wn=N_Wn, n_signal = 2, nnums = 400, rand_coeff = 2)
    # test1Wn_Python_Scipy_Butterworth_Filter(2,100,1)  # Best of diff var -> N = 5, Wn = 15
    # test1N_Python_Scipy_Butterworth_Filter(2) # Best of diff var -> N = 5, Wn = 15
    # Python_Scipy_Butterworth_Filter()
    # Python_Scipy_Bessel_Filter()
    # test_Python_Scipy_Filters()

