'''
Created on May 23, 2011

@author: Jacob Salomonsen
'''

import numpy as np

' Simulation attributes '
nx      = 10
ny      = 10
it      = 50

' Constants '
omega   = 1.0
density = 1.0
t1      = 4/9.0
t2      = 1/9.0
t3      = 1/36.0
deltaU  = 1e-7
c_squ   = 1/3.0

' Create the main arrays '
F           = np.zeros((9,nx,ny), dtype=float)
T           = np.zeros((9,nx,ny), dtype=float)
F[:,:,:]   += density/9.0
FEQ         = np.copy(F)
BOUNCEBACK  = np.zeros(F.shape, dtype=float)
DENSITY     = np.zeros((nx,ny), dtype=float)
UX          = np.copy(DENSITY)
UY          = np.copy(DENSITY)

' Create the scenery '
BOUND   = np.zeros((nx,ny), dtype=float)
BOUNDi  = np.ones(BOUND.shape, dtype=float)

#===============================================================================
# for i in xrange(nx):
#    for j in xrange(ny):
#        if ((i-4)**2+(j-5)**2+(5-6)**2) < 6:
#            BOUND [i,j] = 1.0
#            BOUNDi [i,j] = 0.0
# BOUND[:,0] = 1.0
# BOUNDi[:,0] = 0.0
#===============================================================================

BOUND[0,:] = 1.0
BOUNDi[0,:] = 0.0

#BOUND = np.random.randint(2, size=(nx,ny)).astype(np.float32)

ts=0
while(ts<it):
    T[:] = F
    # propagate nearest neigbours
    F[1,0,:]     = T[1,-1,:]
    F[1,1:,:]    = T[1,:-1,:]
    
    F[3,:,0]     = T[3,:,-1]
    F[3,:,1:]    = T[3,:,:-1]
    
    F[5,-1,:]    = T[5,0,:]
    F[5,:-1,:]   = T[5,1:,:]
    
    F[7,:,-1]    = T[7,:,0]
    F[7,:,:-1]   = T[7,:,1:]
    
    # propagate next-nearest neightbours
    F[2,0,0]     = T[2,-1,-1]
    F[2,0,1:]    = T[2,-1,:-1]
    F[2,1:,0]    = T[2,:-1,-1]
    F[2,1:,1:]   = T[2,:-1,:-1]
    
    F[4,-1,0]    = T[4,0,-1]
    F[4,-1,1:]   = T[4,0,:-1]
    F[4,:-1,0]   = T[4,1:,-1]
    F[4,:-1,1:]  = T[4,1:,:-1]
    
    F[6,-1,-1]   = T[6,0,0]
    F[6,-1,:-1]  = T[6,0,1:]
    F[6,:-1,-1]  = T[6,1:,0]
    F[6,:-1,:-1] = T[6,1:,1:]
    
    F[8,0,-1]    = T[8,-1,0]
    F[8,0,:-1]   = T[8,-1,1:]
    F[8,1:,-1]   = T[8,:-1,0]
    F[8,1:,:-1]  = T[8,:-1,1:]
    
    # Densities bouncing back at next timestep
    T[:] = F
    
    T[1:,:,:] *= BOUND[np.newaxis,:,:]
    BOUNCEBACK[1] += T[5,:,:]
    BOUNCEBACK[2] += T[6,:,:]
    BOUNCEBACK[3] += T[7,:,:]
    BOUNCEBACK[4] += T[8,:,:]
    BOUNCEBACK[5] += T[1,:,:]
    BOUNCEBACK[6] += T[2,:,:]
    BOUNCEBACK[7] += T[3,:,:]
    BOUNCEBACK[8] += T[4,:,:]
    
    DENSITY = np.add.reduce(F)
    
    T1 = F[1,:,:]+F[2,:,:]+F[8,:,:]
    T2 = F[4,:,:]+F[5,:,:]+F[6,:,:]
    UX = (T1-T2)/DENSITY
    
    T1 = F[2,:,:]+F[3,:,:]+F[4,:,:]
    T2 = F[6,:,:]+F[7,:,:]+F[8,:,:]
    UY = (T1-T2)/DENSITY
    
    UX[:,0] = UX[:,0]+deltaU # increase inlet pressure
    
    UX[:,:] *= BOUNDi
    UY[:,:] *= BOUNDi
    DENSITY[:,:] *= BOUNDi
    
    U_SQU = UX**2 + UY**2
    U_C2=UX+UY
    U_C4=-UX+UY
    U_C6=-U_C2
    U_C8=-U_C4
    
    # Calculate equilibrium distribution: stationary
    FEQ[0,:,:]=t1*DENSITY*(1-U_SQU  /(2*c_squ))
    
    # nearest-neighbours
    FEQ[1,:,:]=t2*DENSITY*(1+UX/c_squ+0.5*(UX/c_squ)**2-U_SQU/(2*c_squ))
    FEQ[3,:,:]=t2*DENSITY*(1+UY/c_squ+0.5*(UY/c_squ)**2-U_SQU/(2*c_squ))
    FEQ[5,:,:]=t2*DENSITY*(1-UX/c_squ+0.5*(UX/c_squ)**2-U_SQU/(2*c_squ))
    FEQ[7,:,:]=t2*DENSITY*(1-UY/c_squ+0.5*(UY/c_squ)**2-U_SQU/(2*c_squ))
    
    # next-nearest neighbours
    FEQ[2,:,:]=t3*DENSITY*(1+U_C2/c_squ+0.5*(U_C2/c_squ)**2-U_SQU/(2*c_squ))
    FEQ[4,:,:]=t3*DENSITY*(1+U_C4/c_squ+0.5*(U_C4/c_squ)**2-U_SQU/(2*c_squ))
    FEQ[6,:,:]=t3*DENSITY*(1+U_C6/c_squ+0.5*(U_C6/c_squ)**2-U_SQU/(2*c_squ))
    FEQ[8,:,:]=t3*DENSITY*(1+U_C8/c_squ+0.5*(U_C8/c_squ)**2-U_SQU/(2*c_squ))
    
    F=omega*FEQ+(1.0-omega)*F
    
    #Densities bouncing back at next timestep
    F[1:,:,:] *= BOUNDi[np.newaxis,:,:]
    F[1:,:,:] += BOUNCEBACK[1:,:,:]
    
    ts += 1

import matplotlib.pyplot as plt
UX *= -1
plt.hold(True)
plt.quiver(UY,UX, pivot='middle')
plt.imshow(BOUND)
plt.show()