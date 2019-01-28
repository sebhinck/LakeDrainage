#include "LakeDrainageModel.hh"
#include "LakeProperties_ConnectedComponents.hh"

void test(int xDim, int yDim, double cell_area, double *depth, int *lake_ids, int &N_lakes, double *&area, double *&volume) {

  {
    LakePropertiesCC LPCC = LakePropertiesCC((unsigned int) yDim,
                                             (unsigned int) xDim,
                                             cell_area,
                                             depth,
                                             lake_ids);

    LPCC.run(N_lakes, area, volume);
  }

}
