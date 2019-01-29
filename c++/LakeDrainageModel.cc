#include "LakeDrainageModel.hh"
#include "LakeProperties_ConnectedComponents.hh"
#include "LakeBasins.hh"

void runLakePropertiesCC(int xDim, int yDim, double cell_area, double *depth, int *lake_ids, int &N_lakes, double *&area, double *&volume) {

  LakePropertiesCC LPCC = LakePropertiesCC((unsigned int) yDim,
                                           (unsigned int) xDim,
                                           cell_area,
                                           depth,
                                           lake_ids);

  LPCC.run(N_lakes, area, volume);
}


void findDrainageBasins(int xDim, int yDim, double *usurf, int *basin_id, int *drain_dir, int &N_basins) {

  LakeBasins LB = LakeBasins((unsigned int) yDim,
                             (unsigned int) xDim,
                             N_basins,
                             usurf,
                             basin_id,
                             drain_dir);

  LB.run();
}
