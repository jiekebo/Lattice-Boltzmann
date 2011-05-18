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

' Allocate memory on the GPU '
F_gpu = cuda.mem_alloc(F.size * F.dtype.itemsize)
cuda.memcpy_htod(F_gpu, F)

' A preliminary kernel treating block index as z-component ' 
' x and y field is limited to available threads per block (system dependent) '
mod = SourceModule("""
    //   F4  F3  F2
    //     \ | /
    //  F5--F9--F1
    //    / | \
    // F6  F7  F8
    
    __global__ void propagateKernel(float *F, float *T) {
        int nx = blockDim.x;
        int ny = blockDim.y;
        int blockSize = nx * ny;
        
        // nearest neighbours
        int F1 = (threadIdx.x==0?nx-1:threadIdx.x-1) + threadIdx.y * nx; // +x
        int F3 = threadIdx.x + (threadIdx.y==0?ny-1:threadIdx.y-1) * nx; // +y
        int F5 = (threadIdx.x==nx-1?0:threadIdx.x+1) + threadIdx.y * nx; // -x
        int F7 = threadIdx.x + (threadIdx.y==ny-1?0:threadIdx.y+1) * nx; // -y
        
        // next-nearest neighbours
        int F2 = (threadIdx.x==0?nx-1:threadIdx.x-1) +
                 (threadIdx.y==0?ny-1:threadIdx.y-1) * nx; //+x+y
                 
        int F4 = (threadIdx.x==nx-1?0:threadIdx.x+1) +
                 (threadIdx.y==0?ny-1:threadIdx.y-1) * nx; //-x+y
        
        int F6 = (threadIdx.x==nx-1?0:threadIdx.x+1) + 
                 (threadIdx.y==ny-1?0:threadIdx.y+1) * nx; //-x-y
                 
        int F8 = (threadIdx.x==0?nx-1:threadIdx.x-1) +
                 (threadIdx.y==ny-1?0:threadIdx.y+1) * nx; //+x-y
        
        // current point
        int F9 = threadIdx.x + threadIdx.y * nx;
        
        // propagate nearest
        F[0*blockSize + F9] = F[0*blockSize + F1];
        F[2*blockSize + F9] = F[2*blockSize + F3];
        F[4*blockSize + F9] = F[4*blockSize + F5];
        F[6*blockSize + F9] = F[6*blockSize + F7];
        
        // propagate next-nearest
        F[1*blockSize + F9] = F[1*blockSize + F2];
        F[3*blockSize + F9] = F[3*blockSize + F4];
        F[5*blockSize + F9] = F[5*blockSize + F6];
        F[7*blockSize + F9] = F[7*blockSize + F8];
    }
    """)

func = mod.get_function("propagateKernel")
func(F_gpu, block=(nx,ny,1))

F_prop = np.empty_like(F)
cuda.memcpy_dtoh(F_prop, F_gpu)
1+1