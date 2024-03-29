# -*- mode: cython -*-

cdef extern from "LakeDrainageModel.hh":
  void runLakePropertiesCC(int xDim, int yDim, double cell_area, double *depth, double *lake_level_map, int *lake_ids, int &N_lakes, double *&area, double *&volume, double *&max_depth, double *&lake_level)
  void findDrainageBasins(int xDim, int yDim, int N_neighbors, double *usurf, int *basin_id, int *drain_dir, int &N_basins, int *&spillway_idx, int *&drain_basin_id)
