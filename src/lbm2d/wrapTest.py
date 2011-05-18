'''
Created on May 9, 2011

@author: jiekebo
'''

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mod = SourceModule("""
#import <math.h>
__global__ void multiply_them(float *dest, float* index, int *a)
{
  index[threadIdx.x] = threadIdx.x-5.0f;
  dest[threadIdx.x] = floor(1.0f/4.0)+4.0f;
}
""")

multiply_them = mod.get_function("multiply_them")
a = np.zeros(11).astype(np.float32)
dest = np.zeros_like(a).astype(np.float32)
index = np.zeros_like(a).astype(np.float32)
multiply_them(drv.Out(dest), drv.Out(index), drv.In(a), block=(11,1,1), grid=(1,1))

print dest
print index