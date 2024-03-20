'''
https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy.hstack
'''
import numpy as np

b = np.array([1,2,10,100,0,0]); lenb = len(b)
print(b, b[0], b[-1])
a = [-1,-1,-1]; lena = len(a)
c = [1000,1000,1000,1000]; lenc = len(c)
print(f'{lena=}  {lenb=}   {lenc=} \n')
print('list map = ',list(map(len,[a,b,c])),'\n')

b1 = np.hstack((a,b,c))
print(b1, b1[0], b1[-1])
print(b1)
k,l,m = np.hsplit(b1,[lena,lena+lenb])
print(k,l,m)
lm = list(map(len,[k,l,m]))
print('list map = ',lm)
k,l,m = lm
print(k,l,m)
