'''
Created on May 7, 2011

@author: Jacob Salomonsen
'''

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

' Simulation attributes '
nx      = 10
ny      = 10
nz      = 10
it      = 1

' Constants '
omega   = 1.0
density = 1.0
t1      = 1/3.0
t2      = 1/18.0
t3      = 1/36.0

' CUDA specific '
threadsPerBlock = 256
blocksPerGrid   = (19*nx*ny*nz + threadsPerBlock - 1) / threadsPerBlock

' Create the main arrays '
F       = np.zeros((19,nx,ny,nz), dtype=float)
FEQ     = F;
T       = F;
F[:,:,:,:] += density/19.0
FEQ[:,:,:,:] += density/19.0

' Create the scenery '
BOUND   = np.zeros((nx,ny,nz), dtype=float)
BOUNDi  = np.ones(BOUND.shape, dtype=float)
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
' Perhaps try to use gpuarray ? '
#pycuda.gpuarray.to_gpu(BOUND)
#pycuda.gpuarray.to_gpu(BOUNDi)

' Allocate memory on the GPU '
F_gpu = cuda.mem_alloc(F.size * F.dtype.itemsize)
cuda.memcpy_htod(F_gpu, F)

' A preliminary kernel treating block index as z-component ' 
' x and y field is limited to available threads per block (system dependent) '
mod = SourceModule("""
    __global__ void propagateKernel(float *F, float *T) {
        int index = threadIdx.x * threadIdx.y * blockIdx.x;
        
        // Handle boundary cases to wrap around like in original code...
        
        // Nearest neighbours
        int F1  = threadIdx.x * threadIdx.y * (blockIdx.x - 1);
        int F2  = threadIdx.x * threadIdx.y * (blockIdx.x + 1);
        int F3  = threadIdx.x * (threadIdx.y - 1) * blockIdx.x;
        int F4  = threadIdx.x * (threadIdx.y + 1) * blockIdx.x;
        int F5  = (threadIdx.x - 1) * threadIdx.y * blockIdx.x;
        int F6  = (threadIdx.x + 1) * threadIdx.y * blockIdx.x;
        
        // Next-nearest neighbours
        int F7  = threadIdx.x * threadIdx.y * blockIdx.x;
        int F8  = threadIdx.x * threadIdx.y * blockIdx.x;
        int F9  = threadIdx.x * threadIdx.y * blockIdx.x;
        int F10 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F11 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F12 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F13 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F14 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F15 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F16 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F17 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F18 = threadIdx.x * threadIdx.y * blockIdx.x;
        int F19 = threadIdx.x * threadIdx.y * blockIdx.x;
    }
    """)