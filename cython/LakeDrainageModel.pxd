# -*- mode: cython -*-

cdef extern from "LakeDrainageModel.hh":
  void test(int xDim, int yDim, double cell_area, double *depth, int *lake_ids, int &N_lakes, double *&area, double *&volume)
