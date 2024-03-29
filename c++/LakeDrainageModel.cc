#include "LakeDrainageModel.hh"
#include "LakeProperties_ConnectedComponents.hh"
#include "LakeBasins.hh"

void runLakePropertiesCC(int xDim, int yDim, double cell_area, double *depth, double *lake_level_map, int *lake_ids, int &N_lakes, double *&area, double *&volume, double *&max_depth, double *&lake_level) {

  LakePropertiesCC LPCC = LakePropertiesCC((unsigned int) yDim,
                                           (unsigned int) xDim,
                                           cell_area,
                                           depth,
                                           lake_level_map,
                                           lake_ids);

  LPCC.run(N_lakes, area, volume, max_depth, lake_level);
}


void findDrainageBasins(int xDim, int yDim, int N_neighbors, double *usurf, int *basin_id, int *drain_dir, int &N_basins, int *&spillway_idx, int *&drain_basin_id) {

  LakeBasins LB = LakeBasins((unsigned int) yDim,
                             (unsigned int) xDim,
                             (unsigned int) N_neighbors,
                             N_basins,
                             usurf,
                             basin_id,
                             drain_dir);

  LB.run(spillway_idx, drain_basin_id);
}
