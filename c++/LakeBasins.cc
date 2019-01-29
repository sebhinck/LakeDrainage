#include "LakeBasins.hh"
#include <cmath>

#include<iostream>


LakeBasins::LakeBasins(unsigned int n_rows,
                       unsigned int n_cols,
                       int &N_basins,
                       double *usurf,
                       int *basin_id,
                       int *drain_dir)
  : m_nRows(n_rows), m_nCols(n_cols),
    m_N_basins(N_basins),
    m_usurf(usurf), m_basin_id(basin_id),
    m_drain_dir(drain_dir) {
  for (unsigned int i=0; i<(m_nRows * m_nCols); i++) {
    m_drain_dir[i] = NEIGHBOR::SELF;
  }
}

LakeBasins::~LakeBasins() {
  //Do nothing
}


void LakeBasins::run() {

  findBasins();
  
}



void LakeBasins::findBasins() {
  for (unsigned int y = 0; y < m_nRows; ++y) {
    for (unsigned int x = 0; x < m_nCols; ++x) {
      assignBasin(x, y);
    }
  }
}

int LakeBasins::assignBasin(int x, int y) {
  const unsigned int idx = ind2idx(x, y);
  int self = m_basin_id[idx];

  if (self == SINK::UNDEFINED) {
    NEIGHBOR low_neighbor = findLowestNeighbor(x, y);
    m_drain_dir[idx] = low_neighbor;
    if (low_neighbor != NEIGHBOR::SELF) {
      int xn, yn;
      neighborInd_safe(low_neighbor, x, y, xn, yn);
      self = assignBasin(xn, yn);
    } else {
      //Lowest point...
      //Create new basin...
      //
      self = m_N_basins;
      m_N_basins += 1;
    }
    m_basin_id[idx] = self;
  }

  return self;
}

LakeBasins::NEIGHBOR LakeBasins::findLowestNeighbor(unsigned int idx) {
  int x, y;
  
  idx2ind(idx, x, y);

  return findLowestNeighbor(x, y);
}

LakeBasins::NEIGHBOR LakeBasins::findLowestNeighbor(int x, int y) {
  
  NEIGHBOR low_neighbor = NEIGHBOR::SELF;
  double low_val = m_usurf[ind2idx(x, y)];
  
  for (unsigned int i=0; i<8; i++) {
    const NEIGHBOR n = m_directions[i];
    const unsigned int ind = neighborIdx(n, x, y);
    const double val = m_usurf[ind];
    
    if (val < low_val) {
      low_val = val;
      low_neighbor = n;
    }
  }

  return low_neighbor;
}




void LakeBasins::checkBoundaries(int &x, int &y) {
  if (x < 0) {
    x = 0;
  } else if (x >= (int) m_nCols) {
    x = (m_nCols -1);
  }

  if (y < 0) {
    y = 0;
  } else if (y >= (int) m_nRows) {
    y = (m_nRows -1);
  }
}

unsigned int LakeBasins::ind2idx_safe(int x, int y) {
  checkBoundaries(x, y);
  
  return ind2idx(x, y);
}

unsigned int LakeBasins::ind2idx(int x, int y) {
  return (unsigned int) (y * m_nCols + x);
}

void LakeBasins::idx2ind(unsigned int idx, int &x, int &y) {
  y = (int) idx / (m_nCols);
  x = (int) idx - (m_nCols * y);
}

void LakeBasins::neighborOffset(NEIGHBOR n, int &dx, int &dy) {
  switch(n) {
    case  N: dx= 0; dy= 1;
            break;
    case  E: dx= 1; dy= 0;
            break;
    case  S: dx= 0; dy=-1;
            break;
    case  W: dx=-1; dy= 0;
            break;
    case NE: dx= 1; dy= 1;
            break;
    case SE: dx= 1; dy=-1;
            break;
    case SW: dx=-1; dy=-1;
            break;
    case NW: dx=-1; dy= 1;
            break;
    default: //This shouldn't happen
             dx= 0; dy= 0;
  }
}

void LakeBasins::neighborInd(NEIGHBOR n, int x, int y, int &xn, int &yn) {
  int dx, dy;

  neighborOffset(n, dx, dy);

  xn = x + dx;
  yn = y + dy;
}

void LakeBasins::neighborInd(NEIGHBOR n, unsigned int idx, int &xn, int &yn) {
  int x, y;
  
  idx2ind(idx, x, y);
  
  neighborInd(n, x, y, xn, yn);
}

void LakeBasins::neighborInd_safe(NEIGHBOR n, int x, int y, int &xn, int &yn) {
  neighborInd(n, x, y, xn, yn);
  
  checkBoundaries(xn, yn);
}

void LakeBasins::neighborInd_safe(NEIGHBOR n, unsigned int idx, int &xn, int &yn) {
  int x, y;
  
  idx2ind(idx, x, y);
  
  neighborInd_safe(n, x, y, xn, yn);
}

unsigned int LakeBasins::neighborIdx(NEIGHBOR n, int x, int y) {
  int xn, yn;
  
  neighborInd_safe(n, x, y, xn, yn);
  
  return ind2idx(xn, yn);
}

unsigned int LakeBasins::neighborIdx(NEIGHBOR n, unsigned int idx) {
  int x, y;
  
  idx2ind(idx, x, y);
  
  return neighborIdx(n, x, y);
}

