# -*- mode: cython -*-

import numpy as np
cimport numpy as cnp
cimport LakeDrainageModel
import ctypes

cnp.import_array()


cdef extern from "numpy/ndarraytypes.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef class LakeDrainage:

  cdef readonly cnp.ndarray depth  
  cdef readonly cnp.ndarray topg
  cdef readonly cnp.ndarray thk
  cdef readonly cnp.ndarray ocean_mask
  cdef readonly cnp.ndarray lake_mask
  cdef readonly cnp.ndarray area
  cdef readonly cnp.ndarray volume
  
  cdef readonly int xDim, yDim
  
  def __init__(self, cnp.ndarray[double, ndim=2, mode="c"] depth, 
                     cnp.ndarray[double, ndim=2, mode="c"] topg, 
                     cnp.ndarray[double, ndim=2, mode="c"] thk, 
                     cnp.ndarray[int, ndim=2, mode="c"] ocean_mask, 
                     double cell_area):
 
    self.depth = depth
    self.topg = topg
    self.thk = thk
    self.ocean_mask = ocean_mask
    self.lake_mask = np.zeros_like(self.topg, dtype=ctypes.c_int)

    self.yDim, self.xDim = topg.shape[0], topg.shape[1]
    
    cdef double[:,:] c_depth = self.depth
    cdef int[:,:] c_lake_mask = self.lake_mask
    cdef double *area_ptr;
    cdef double *volume_ptr;
    cdef int N_lakes_int
    cdef cnp.npy_intp N_lakes[1]
    
    LakeDrainageModel.runLakePropertiesCC(self.xDim, self.yDim, cell_area, &c_depth[0, 0], &c_lake_mask[0, 0], N_lakes_int, area_ptr, volume_ptr)

    N_lakes[0] = N_lakes_int
    
    self.area = cnp.PyArray_SimpleNewFromData(1, N_lakes, cnp.NPY_DOUBLE, area_ptr)
    PyArray_ENABLEFLAGS(self.area, cnp.NPY_OWNDATA)
    
    self.volume = cnp.PyArray_SimpleNewFromData(1, N_lakes, cnp.NPY_DOUBLE, volume_ptr)
    PyArray_ENABLEFLAGS(self.volume, cnp.NPY_OWNDATA)
    
    
    
    
    
    
