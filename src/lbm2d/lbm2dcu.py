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
F           = np.zeros((9,nx,ny), dtype=float).astype(np.float32)
F[:,:,:]   += density/9.0
FEQ         = np.copy(F)
DENSITY     = np.zeros((nx,ny), dtype=float).astype(np.float32)
UX          = np.copy(DENSITY)
UY          = np.copy(DENSITY)

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

' A preliminary kernel treating block index as z-component ' 
' x and y field is limited to available threads per block (system dependent) '
mod = SourceModule("""
    //   F4  F3  F2
    //     \ | /
    //  F5--F9--F1
    //    / | \
    // F6  F7  F8
    
    __global__ void propagateKernel(float *F) {
        int nx = blockDim.x;
        int ny = blockDim.y;
        int size = nx * ny;
        
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
        F[0*size + F9] = F[0*size + F1];
        F[2*size + F9] = F[2*size + F3];
        F[4*size + F9] = F[4*size + F5];
        F[6*size + F9] = F[6*size + F7];
        
        // propagate next-nearest
        F[1*size + F9] = F[1*size + F2];
        F[3*size + F9] = F[3*size + F4];
        F[5*size + F9] = F[5*size + F6];
        F[7*size + F9] = F[7*size + F8];
    }
    
    __global__ void bouncebackKernel() {
        /*BOUNCEDBACK=F(TO_REFLECT);*/
    }
    
    __global__ void densityKernel(float *F, float *D, float *UX, float *UY) {
        int size = blockDim.x * blockDim.y;
        int cur = threadIdx.x + threadIdx.y * blockDim.x;
        D[cur] = F[0*size + cur] + 
                 F[1*size + cur] +
                 F[2*size + cur] +
                 F[3*size + cur] +
                 F[4*size + cur] +
                 F[5*size + cur] +
                 F[6*size + cur] +
                 F[7*size + cur] +
                 F[8*size + cur];
        
        UX[cur] = ((F[0*size + cur] + F[1*size + cur] + F[7*size + cur]) -
                   (F[3*size + cur] + F[4*size + cur] + F[5*size + cur])) 
                    / D[cur];
                 
        UY[cur] = ((F[1*size + cur] + F[2*size + cur] + F[3*size + cur]) -
                   (F[5*size + cur] + F[6*size + cur] + F[7*size + cur])) 
                    / D[cur];
    }
    """)

#F = np.random.randint(1,9,size=(9,nx,ny))
#F = np.ones((9,nx,ny), dtype=float)
#F = np.array(range(9*3*3))
#F.shape = (9,3,3)
#F = F.astype(np.float32)

' Allocate memory on the GPU '
F_gpu       = cuda.mem_alloc(F.size * F.dtype.itemsize)
DENSITY_gpu = cuda.mem_alloc(DENSITY.size * DENSITY.dtype.itemsize)
UX_gpu      = cuda.mem_alloc(UX.size * UX.dtype.itemsize)
UY_gpu      = cuda.mem_alloc(UY.size * UY.dtype.itemsize)
cuda.memcpy_htod(DENSITY_gpu, DENSITY)
cuda.memcpy_htod(UX_gpu, UX)
cuda.memcpy_htod(UY_gpu, UY)
cuda.memcpy_htod(F_gpu, F)

#F_prop = np.empty_like(F)
#cuda.memcpy_dtoh(F, F_gpu)

prop = mod.get_function("propagateKernel")
prop(F_gpu, block=(nx,ny,1))
density = mod.get_function("densityKernel")
density(F_gpu, DENSITY_gpu, UX_gpu, UY_gpu, block=(nx,ny,1))

cuda.memcpy_dtoh(F, F_gpu)
cuda.memcpy_dtoh(DENSITY, DENSITY_gpu)
cuda.memcpy_dtoh(UX, UX_gpu)
cuda.memcpy_dtoh(UY, UY_gpu)
1+1