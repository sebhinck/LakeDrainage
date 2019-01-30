# -*- mode: cython -*-

import numpy as np
cimport numpy as cnp
cimport LakeDrainageModel
import ctypes


cdef enum sink:
  UNDEFINED=-6,
  OCEAN=-5,
  NORTH=-4,
  EAST=-3,
  SOUTH=-2,
  WEST=-1

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

  cdef readonly cnp.ndarray usurf
  cdef readonly cnp.ndarray surf_eff
  cdef readonly cnp.ndarray basin_id
  cdef readonly cnp.ndarray drain_dir
  cdef readonly cnp.ndarray drainage_idx

  cdef readonly int xDim, yDim

  def __init__(self, cnp.ndarray[double, ndim=2, mode="c"] depth, 
                     cnp.ndarray[double, ndim=2, mode="c"] topg, 
                     cnp.ndarray[double, ndim=2, mode="c"] thk, 
                     cnp.ndarray[int, ndim=2, mode="c"] ocean_mask, 
                     double cell_area, double rho_i, double rho_w):
 
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


    self.usurf = self.topg + self.thk
    self.surf_eff = self.topg + rho_w / rho_i * self.thk
    self.basin_id = self.lake_mask.copy()

    self.basin_id[self.lake_mask == -1] = sink.UNDEFINED
    self.basin_id[0,:]  = sink.SOUTH
    self.basin_id[-1,:] = sink.NORTH
    self.basin_id[:,0]  = sink.WEST
    self.basin_id[:,-1] = sink.EAST
    self.basin_id[self.ocean_mask == 1] = sink.OCEAN

    self.drain_dir = np.zeros_like(self.topg, dtype=ctypes.c_int)

    cdef double[:,:] c_usurf = self.surf_eff #self.usurf
    cdef int[:,:] c_basin_id = self.basin_id
    cdef int[:,:] c_drain_dir = self.drain_dir

    cdef int *drainage_idx_ptr;
    cdef int N_basins_int
    cdef cnp.npy_intp N_basins[1]

    LakeDrainageModel.findDrainageBasins(self.xDim, self.yDim, &c_usurf[0, 0], &c_basin_id[0, 0], &c_drain_dir[0, 0], N_basins_int, drainage_idx_ptr)

    N_basins[0] = N_basins_int

    self.drainage_idx = cnp.PyArray_SimpleNewFromData(1, N_basins, cnp.NPY_INT, drainage_idx_ptr)
    PyArray_ENABLEFLAGS(self.drainage_idx, cnp.NPY_OWNDATA)

