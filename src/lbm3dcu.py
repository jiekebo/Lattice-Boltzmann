'''
Created on May 7, 2011

@author: jiekebo
'''

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


nx = 10
ny = 10
nz = 10
it = 1

BOUND = np.zeros((nx,ny,nz), dtype=float)
BOUNDi = np.ones(BOUND.shape, dtype=float)

' Create the boundary array'
for i in xrange(nx):
    for j in xrange(ny):
        for k in xrange(nz):
            if ((i-4)**2+(j-5)**2+(k-6)**2) < 6:
                BOUND[i,j,k] = 1.0
                BOUNDi[i,j,k] = 0.0
BOUND [:,0,:] = 1.0
BOUNDi[:,0,:] = 0.0
BOUND [:,:,0] = 1.0
BOUNDi[:,:,0] = 0.0

' And transfer it to the gpu '
pycuda.gpuarray.to_gpu(BOUND)
pycuda.gpuarray.to_gpu(BOUNDi)

mod = SourceModule("""
__global__ void propagateKernel(
                   """)