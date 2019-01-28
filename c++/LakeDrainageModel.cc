#include "LakeDrainageModel.hh"

void test(double *topg, int xDim, int yDim, int*& ptr, int &size) {

  size = 30;
  ptr = new int[size];
  for (int j=0; j<size; j++) {
    ptr[j] = j+1;
  }
  

  for(int i=0; i<xDim; i++) {
    for(int j=0; j<yDim; j++) {
      topg[(i * yDim+ j)] = 1.;
    }
  }

}
