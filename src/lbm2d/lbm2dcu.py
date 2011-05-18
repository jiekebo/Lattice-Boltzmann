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
nx      = 10
ny      = 10
it      = 5

' Constants '
omega   = 1.0
density = 1.0
t1      = 4/9.0
t2      = 1/9.0
t3      = 1/36.0
deltaU  = 1e-7
c_squ   = 1/3.0

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
            BOUND [i,j] = 0.0
            BOUNDi[i,j] = 0.0
BOUND [:,0] = 1.0
BOUNDi[:,0] = 0.0

'''
msize=nx*ny;
CI=0:msize:msize*7;
ON=find(BOUND); %matrix offset of each Occupied Node
TO_REFLECT=[ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4) ...
            ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8)];
REFLECTED= [ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8) ...
            ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4)];
            
avu=1; prevavu=1; ts=0; deltaU=1e-7; numactivenodes=sum(sum(1-BOUND));
'''

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

ts=0
while(ts<it):
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
    
    UX[:,0] = UX[:,0]+deltaU
    '''
    UX(ON)=0;
    UY(ON)=0;
    DENSITY(ON)=0;
    '''
    U_SQU = pow(UX[:,:],2) + pow(UY[:,:],2)
    U_C2=UX+UY
    U_C4=-UX+UY;
    U_C6=-U_C2;
    U_C8=-U_C4;
    
    # Calculate equilibrium distribution: stationary
    FEQ[8,:,:]=t1*DENSITY[:,:]*(1-U_SQU[:,:]/(2*c_squ));
    
    # nearest-neighbours
    FEQ[0,:,:]=t2*DENSITY[:,:]*(1+UX[:,:]/c_squ+0.5*pow((UX[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    FEQ[2,:,:]=t2*DENSITY[:,:]*(1+UY[:,:]/c_squ+0.5*pow((UY[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    FEQ[4,:,:]=t2*DENSITY[:,:]*(1-UX[:,:]/c_squ+0.5*pow((UX[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    FEQ[6,:,:]=t2*DENSITY[:,:]*(1-UY[:,:]/c_squ+0.5*pow((UY[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    
    # next-nearest neighbours
    FEQ[1,:,:]=t3*DENSITY[:,:]*(1+U_C2[:,:]/c_squ+0.5*pow((U_C2[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    FEQ[3,:,:]=t3*DENSITY[:,:]*(1+U_C4[:,:]/c_squ+0.5*pow((U_C4[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    FEQ[5,:,:]=t3*DENSITY[:,:]*(1+U_C6[:,:]/c_squ+0.5*pow((U_C6[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    FEQ[7,:,:]=t3*DENSITY[:,:]*(1+U_C8[:,:]/c_squ+0.5*pow((U_C8[:,:]/c_squ),2)-U_SQU[:,:]/(2*c_squ))
    
    F=omega*FEQ+(1-omega)*F
    
    '''
    F(REFLECTED)=BOUNCEDBACK;
    prevavu=avu;
    avu=sum(sum(UX))/numactivenodes; ts=ts+1;
    '''
    ts += 1


import matplotlib.pyplot as plt
plt.hold(True)
plt.quiver(UX,UY, pivot='middle')
plt.imshow(BOUND)
plt.show()