#ifndef _LAKEBASINS_H_
#define _LAKEBASINS_H_

class LakeBasins {
public:
  LakeBasins(unsigned int n_rows,
             unsigned int n_cols,
             int &N_basins,
             double *usurf, 
             int *basin_id,
             int *drain_dir);
  virtual ~LakeBasins();
  void run(int *&spillway_idx, int *&drain_basin_id);

  enum SINK {
    UNDEFINED=-6,
    OCEAN=-5,
    NORTH=-4,
    EAST=-3,
    SOUTH=-2,
    WEST=-1
  };

  enum NEIGHBOR {
    SELF=0,
    N=1,
    E=2,
    S=3,
    W=4,
    NE=5,
    SE=6,
    SW=7,
    NW=8
  };

  NEIGHBOR m_directions[8] = {NEIGHBOR::N, NEIGHBOR::NE, NEIGHBOR::E, NEIGHBOR::SE, NEIGHBOR::S, NEIGHBOR::SW, NEIGHBOR::W, NEIGHBOR::NW};

protected:
  unsigned int m_nRows, m_nCols;
  double *m_usurf;
  double m_cell_area;
  int *m_basin_id;
  int *m_drain_dir;
  int &m_N_basins;

private:
  void checkBoundaries(int &x, int &y);
  unsigned int ind2idx(int x, int y);
  unsigned int ind2idx_safe(int x, int y);
  void idx2ind(unsigned int idx, int &x, int &y);

  void neighborOffset(NEIGHBOR n, int &dx, int &dy);
  void neighborInd(NEIGHBOR n, int x, int y, int &xn, int &yn);
  void neighborInd(NEIGHBOR n, unsigned int idx, int &xn, int &yn);
  void neighborInd_safe(NEIGHBOR n, int x, int y, int &xn, int &yn);
  void neighborInd_safe(NEIGHBOR n, unsigned int idx, int &xn, int &yn);
  unsigned int neighborIdx(NEIGHBOR n, int x, int y);
  unsigned int neighborIdx(NEIGHBOR n, unsigned int idx);

  NEIGHBOR findLowestNeighbor(unsigned int idx);
  NEIGHBOR findLowestNeighbor(int x, int y);

  void findBasins();
  int assignBasin(int x, int y);

  void findSpillways(int *spillway_idx, int *drain_basin_id);

};

#endif
