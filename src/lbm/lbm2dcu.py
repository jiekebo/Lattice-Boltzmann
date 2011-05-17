'''
Created on May 16, 2011

@author: Jacob Salomonsen
'''

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

' Simulation attributes '
nx      = 3
ny      = 3
it      = 1

' Constants '
omega   = 1.0
density = 1.0
t1      = 4/9.0
t2      = 1/9.0
t3      = 1/36.0

' CUDA specific '
threadsPerBlock = 256
blocksPerGrid   = (nx*ny + threadsPerBlock - 1) / threadsPerBlock

' Create the main arrays '
#F       = np.zeros((9,nx,ny), dtype=float)
#FEQ     = F;
#T       = F;
#F[:,:,:] += density/19.0
#FEQ[:,:,:] += density/19.0

' Create the scenery '
BOUND   = np.zeros((nx,ny), dtype=float)
BOUNDi  = np.ones(BOUND.shape, dtype=float)
for i in xrange(nx):
    for j in xrange(ny):
        if ((i-4)**2+(j-5)**2+(5-6)**2) < 6:
            BOUND [i,j] = 1.0
            BOUNDi[i,j] = 0.0
BOUND [:,0] = 1.0
BOUNDi[:,0] = 0.0

#F = np.random.randint(1,9,size=(9,nx,ny))
#F = np.ones((9,nx,ny), dtype=float)
F = np.array(range(9*3*3))
F.shape = (9,3,3)
F = F.astype(np.float32)
T = F


' Allocate memory on the GPU '
F_gpu = cuda.mem_alloc(F.size * F.dtype.itemsize)
T_gpu = cuda.mem_alloc(T.size * T.dtype.itemsize)
cuda.memcpy_htod(F_gpu, F)
cuda.memcpy_htod(T_gpu, T)

' A preliminary kernel treating block index as z-component ' 
' x and y field is limited to available threads per block (system dependent) '
mod = SourceModule("""
    __global__ void propagateKernel(float *F, float *T) {
        int nx = blockDim.x;
        int ny = blockDim.y;
        int blockSize = nx * ny;
        
        // nearest neighbours
        int F1 = (threadIdx.x + 1) % nx + threadIdx.y * nx;
        int F5 = threadIdx.x + ((threadIdx.y + 1) % ny) * nx;
        //int F3 = (threadIdx.x - 1) % nx + threadIdx.y * nx;
        //int F7 = threadIdx.x + ((threadIdx.y - 1) % ny) * nx;
        
        // next-nearest neighbours
        int F2 = (threadIdx.x - 1) % nx + ((threadIdx.y - 1) % ny) * nx;
        int F4 = (threadIdx.x + 1) % nx + ((threadIdx.y - 1) % ny) * nx;
        int F6 = (threadIdx.x - 1) % nx + ((threadIdx.y + 1) % ny) * nx;
        int F8 = (threadIdx.x + 1) % nx + ((threadIdx.y + 1) % ny) * nx;
        
        // self
        int F9 = threadIdx.x + threadIdx.y * nx;
        
        // propagate
        F[F9] = T[F1];
        F[blockSize + F1] = T[blockSize + F9];
        F[2*blockSize + F5] = F[2*blockSize + F9];
        F[3*blockSize + F9] = F[3*blockSize + F5];
        
        /*
        F[F2] = F[F9];
        F[F4] = F[F9];
        F[F6] = F[F9];
        F[F8] = F[F9];*/
    }
    """)

func = mod.get_function("propagateKernel")
func(F_gpu, T_gpu, block=(nx,ny,1))

F_prop = np.empty_like(F)
cuda.memcpy_dtoh(F_prop, F_gpu)
1+1