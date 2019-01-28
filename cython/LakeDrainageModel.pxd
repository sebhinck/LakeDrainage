# -*- mode: cython -*-

cdef extern from "LakeDrainageModel.hh":
        void test(double *topg, int xDim, int yDim, int*& ptr, int &size)
