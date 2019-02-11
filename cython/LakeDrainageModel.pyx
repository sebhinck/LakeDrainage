# -*- mode: cython -*-

import numpy as np
cimport numpy as cnp
cimport LakeDrainageModel
import ctypes


cdef enum sink:
  OCEAN=-7,
  NORTH=-6,
  EAST=-5,
  SOUTH=-4,
  WEST=-3,
  LOOP=-2,
  UNDEFINED=-1

cnp.import_array()


cdef extern from "numpy/ndarraytypes.h":
  void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef class LakeDrainage:

  cdef readonly cnp.ndarray x
  cdef readonly cnp.ndarray y
  cdef readonly cnp.ndarray depth
  cdef readonly cnp.ndarray depth_filtered
  cdef readonly cnp.ndarray topg
  cdef readonly cnp.ndarray topg_filtered
  cdef readonly cnp.ndarray thk
  cdef readonly cnp.ndarray ocean_mask
  cdef readonly cnp.ndarray lake_mask
  cdef readonly cnp.ndarray area
  cdef readonly cnp.ndarray volume
  cdef readonly cnp.ndarray max_depth

  cdef readonly cnp.ndarray usurf
  cdef readonly cnp.ndarray usurf_filtered
  cdef readonly cnp.ndarray surf_eff
  cdef readonly cnp.ndarray basin_id
  cdef readonly cnp.ndarray drain_dir
  cdef readonly cnp.ndarray spillway_idx
  cdef readonly cnp.ndarray drain_basin_id

  cdef readonly int xDim, yDim

  def __init__(self, cnp.ndarray[double, ndim=1, mode="c"] x,
                     cnp.ndarray[double, ndim=1, mode="c"] y,
                     cnp.ndarray[double, ndim=2, mode="c"] depth,
                     cnp.ndarray[double, ndim=2, mode="c"] depth_filtered,
                     cnp.ndarray[double, ndim=2, mode="c"] topg,
                     cnp.ndarray[double, ndim=2, mode="c"] topg_filtered,
                     cnp.ndarray[double, ndim=2, mode="c"] thk,
                     cnp.ndarray[int, ndim=2, mode="c"] ocean_mask,
                     double cell_area, double rho_i, double rho_w,
                     int N_neighbors):
 
    self.x = x
    self.y = y
    self.depth = depth
    self.topg = topg
    self.depth_filtered = depth_filtered
    self.topg_filtered = topg_filtered
    self.thk = thk
    self.ocean_mask = ocean_mask
    self.lake_mask = np.zeros_like(self.topg, dtype=ctypes.c_int)

    self.yDim, self.xDim = topg.shape[0], topg.shape[1]

    cdef double[:,:] c_depth = self.depth
    cdef int[:,:] c_lake_mask = self.lake_mask
    cdef double *area_ptr;
    cdef double *volume_ptr;
    cdef double *max_depth_ptr;
    cdef int N_lakes_int
    cdef cnp.npy_intp N_lakes[1]

    LakeDrainageModel.runLakePropertiesCC(self.xDim,
                                          self.yDim,
                                          cell_area,
                                          &c_depth[0, 0],
                                          &c_lake_mask[0, 0],
                                          N_lakes_int,
                                          area_ptr,
                                          volume_ptr,
                                          max_depth_ptr)

    N_lakes[0] = N_lakes_int

    self.area = cnp.PyArray_SimpleNewFromData(1, N_lakes, cnp.NPY_DOUBLE, area_ptr)
    PyArray_ENABLEFLAGS(self.area, cnp.NPY_OWNDATA)

    self.volume = cnp.PyArray_SimpleNewFromData(1, N_lakes, cnp.NPY_DOUBLE, volume_ptr)
    PyArray_ENABLEFLAGS(self.volume, cnp.NPY_OWNDATA)

    self.max_depth = cnp.PyArray_SimpleNewFromData(1, N_lakes, cnp.NPY_DOUBLE, max_depth_ptr)
    PyArray_ENABLEFLAGS(self.max_depth, cnp.NPY_OWNDATA)

    self.usurf_filtered = self.topg_filtered + self.thk
    self.surf_eff = self.topg_filtered + rho_w / rho_i * self.thk
    self.basin_id = self.lake_mask.copy()

    self.basin_id[self.lake_mask == -1] = sink.UNDEFINED
    self.basin_id[0,:]  = sink.SOUTH
    self.basin_id[-1,:] = sink.NORTH
    self.basin_id[:,0]  = sink.WEST
    self.basin_id[:,-1] = sink.EAST
    self.basin_id[self.ocean_mask == 1] = sink.OCEAN

    self.drain_dir = np.zeros_like(self.topg, dtype=ctypes.c_int)

    cdef double[:,:] c_surf_eff = self.surf_eff #self.usurf
    cdef int[:,:] c_basin_id = self.basin_id
    cdef int[:,:] c_drain_dir = self.drain_dir

    cdef int *spillway_idx_ptr;
    cdef int *drain_basin_id_ptr;
    cdef int N_basins_int = N_lakes_int
    cdef cnp.npy_intp N_basins[1]

    LakeDrainageModel.findDrainageBasins(self.xDim,
                                         self.yDim,
                                         N_neighbors,
                                         &c_surf_eff[0, 0],
                                         &c_basin_id[0, 0],
                                         &c_drain_dir[0, 0],
                                         N_basins_int,
                                         spillway_idx_ptr,
                                         drain_basin_id_ptr)

    N_basins[0] = N_basins_int

    self.spillway_idx = cnp.PyArray_SimpleNewFromData(1, N_basins, cnp.NPY_INT, spillway_idx_ptr)
    PyArray_ENABLEFLAGS(self.spillway_idx, cnp.NPY_OWNDATA)

    self.drain_basin_id = cnp.PyArray_SimpleNewFromData(1, N_basins, cnp.NPY_INT, drain_basin_id_ptr)
    PyArray_ENABLEFLAGS(self.drain_basin_id, cnp.NPY_OWNDATA)

