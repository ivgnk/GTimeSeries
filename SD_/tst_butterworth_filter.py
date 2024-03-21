'''
How can I design a butterworth filter with python specifying that my cutoff freq
https://www.pythonsos.com/numeric-computing/how-can-i-design-a-butterworth-filter-with-python-specifying-that-my-cutoff-freq/?expand_article=1
'''
import inspect
import scipy.signal as signal

# Method 1: Using the scipy.signal.butter function
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

def design_butterworth_filter1(cutoff_freq, order, sampling_rate):
    nyquist_freq = 0.5001 * sampling_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    print(f'{normalized_cutoff_freq=}')
    b, a = signal.butter(N=order, Wn = normalized_cutoff_freq, btype='lowpass', analog=False, output='ba')
    return b, a

def tst_design_butterworth_filter1():
    print('Function ',inspect.currentframe().f_code.co_name)
    cutoff_freq = 1000  # Specify the cutoff frequency in Hz
    order = 4  # Specify the filter order
    sampling_rate = 2000  # Specify the sampling rate in Hz

    b, a = design_butterworth_filter1(cutoff_freq, order, sampling_rate)
    print("Filter coefficients (b):", b)
    print("Filter coefficients (a):", a)

# Method 2: Using the scipy.signal.iirfilter function
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html

def design_butterworth_filter2(cutoff_freq, order,sampling_rate):
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = signal.iirfilter(N=order, Wn = normalized_cutoff_freq, btype='low', analog=False, ftype='butter')
    return b, a

def tst_design_butterworth_filter2():
    print('Function ',inspect.currentframe().f_code.co_name)
    cutoff_freq = 1000  # Specify the cutoff frequency in Hz
    order = 4  # Specify the filter order
    sampling_rate = 2000  # Specify the sampling rate in Hz

    b, a = design_butterworth_filter1(cutoff_freq, order, sampling_rate)
    print("Filter coefficients (b):", b)
    print("Filter coefficients (a):", a)


if __name__ == "__main__":
    # tst_design_butterworth_filter1()
    tst_design_butterworth_filter2()