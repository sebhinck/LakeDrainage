# -*- mode: cython -*-

import numpy as np
cimport numpy as cnp
cimport LakeDrainageModel
import ctypes

cnp.import_array()


cdef extern from "numpy/ndarraytypes.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef class LakeDrainage:
  
  cdef readonly cnp.ndarray topg
  cdef readonly cnp.ndarray thk
  cdef readonly cnp.ndarray ocean_mask
  cdef readonly cnp.ndarray lake_mask
  cdef readonly cnp.ndarray cpp_array
  
  cdef readonly int xDim, yDim
  
  def __init__(self, cnp.ndarray[double, ndim=2, mode="c"] topg, cnp.ndarray[double, ndim=2, mode="c"] thk, cnp.ndarray[int, ndim=2, mode="c"] ocean_mask):
 
    self.topg = topg
    self.thk = thk
    self.ocean_mask = ocean_mask
    self.lake_mask = np.zeros_like(self.topg)

    self.yDim, self.xDim = topg.shape[0], topg.shape[1]
    
    cdef double[:,:] topg_test = self.topg
    cdef int *my_int_ptr;
    cdef int size
    cdef cnp.npy_intp dims[1]
    
    LakeDrainageModel.test(&topg_test[0, 0], self.xDim, self.yDim, my_int_ptr, size)

    dims[0] = size
    
    self.cpp_array = cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_INT, my_int_ptr)
    PyArray_ENABLEFLAGS(self.cpp_array, cnp.NPY_OWNDATA)
    
    
    
