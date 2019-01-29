# -*- mode: cython -*-

cdef extern from "LakeDrainageModel.hh":
  void runLakePropertiesCC(int xDim, int yDim, double cell_area, double *depth, int *lake_ids, int &N_lakes, double *&area, double *&volume)
  void findDrainageBasins(int xDim, int yDim, double *usurf, int *basin_mask, int &N_basins)
