#ifndef _LAKEDRAIANGEMODEL_H_
#define _LAKEDRAIANGEMODEL_H_

void runLakePropertiesCC(int xDim, int yDim, double cell_area, double *depth, int *lake_ids, int &N_lakes, double *&area, double *&volume);

void findDrainageBasins(int xDim, int yDim, double *usurf, int *basin_mask, int &N_basins);

enum SINK {
    UNDEFINED=-6,
    OCEAN=-5,
    NORTH=-4,
    EAST=-3,
    SOUTH=-2,
    WEST=-1
};


#endif
