#include "LakeBasins.hh"
#include <cmath>
#include<limits>
#include<iostream>


LakeBasins::LakeBasins(unsigned int n_rows,
                       unsigned int n_cols,
                       unsigned int N_neighbors,
                       int &N_basins,
                       double *usurf,
                       int *basin_id,
                       int *drain_dir)
  : m_nRows(n_rows), m_nCols(n_cols),
    m_N_neighbors(N_neighbors),
    m_N_basins(N_basins),
    m_usurf(usurf), m_basin_id(basin_id),
    m_drain_dir(drain_dir) {

  if ((m_N_neighbors != 8) and (m_N_neighbors != 4)) {
    std::cout<<"N_neighbors can only be 4 or 8! Using default value 4!\n";
    m_N_neighbors = 4;
  }

  for (unsigned int i=0; i<(m_nRows * m_nCols); i++) {
    m_drain_dir[i] = NEIGHBOR::SELF;
  }
}

LakeBasins::~LakeBasins() {
  //Do nothing
}


void LakeBasins::run(int *&spillway_idx, int *&drain_basin_id) {

  findBasins();

  findSpillways(spillway_idx, drain_basin_id);
}



void LakeBasins::findSpillways(int *&spillway_idx, int *&drain_basin_id) {

  spillway_idx = new int[m_N_basins];
  drain_basin_id = new int[m_N_basins];
  int basin_sink[m_N_basins];
  double spillway_height[m_N_basins],
         spillway_neighbor_height[m_N_basins];

  for (int i=0; i<m_N_basins; i++) {
    spillway_idx[i] = -1;
    drain_basin_id[i] = SINK::UNDEFINED;
    basin_sink[i] = SINK::UNDEFINED;
    spillway_height[i] = std::numeric_limits<double>::max();
    spillway_neighbor_height[i] = std::numeric_limits<double>::max();
  }

  assignSpillway(spillway_idx, drain_basin_id, basin_sink, spillway_height, spillway_neighbor_height, true);

  {
    //Check if basin ends up in a "drainage loop". If so, mark those and treat them as a single basin
    std::vector<int> prev_ids(m_N_basins);
    prev_ids.clear();
    for (int current_id = 0; current_id < m_N_basins; current_id++) {
      int result = traceDrainagePath(current_id, drain_basin_id, basin_sink, prev_ids);
      if (result > SINK::UNDEFINED) {
        drain_basin_id[current_id] = SINK::UNDEFINED;
        spillway_idx[current_id] = -1;
        spillway_height[current_id] = std::numeric_limits<double>::max();
        spillway_neighbor_height[current_id] = std::numeric_limits<double>::max();
      }
    }
  }

  assignSpillway(spillway_idx, drain_basin_id, basin_sink, spillway_height, spillway_neighbor_height, false);

  {
    //Update Lists for basins that are part of a loop
    for (int current_id = 0; current_id < m_N_basins; current_id++) {
      const int current_basin_sink = basin_sink[current_id];
      if (current_basin_sink > SINK::UNDEFINED) {
        spillway_idx[current_id] = spillway_idx[current_basin_sink];
        drain_basin_id[current_id] = drain_basin_id[current_basin_sink];
      }
    }
  }

}

void LakeBasins::assignSpillway(int *spillway_idx, int *drain_basin_id, int *basin_sink, double *spillway_height, double *spillway_neighbor_height, bool update_all) {
  for (unsigned int y = 0; y < m_nRows; ++y) {
    for (unsigned int x = 0; x < m_nCols; ++x) {

      const unsigned int self_idx = ind2idx(x, y);
      int self_basin_id = m_basin_id[self_idx];
      bool reapply = false;

      if ((self_basin_id >= 0) and (basin_sink[self_basin_id] > SINK::UNDEFINED)) {
        //Part of loop -> check for parent basin
        self_basin_id = basin_sink[self_basin_id];
        reapply = true;
      }

      if ((self_basin_id >= 0) and (update_all or reapply)) {
      //(((self_basin_id >= 0) and not reapply) or ((self_basin_id >= 0) )) {
        const double self_height = m_usurf[self_idx];
        double height;
        unsigned int idx;
        for (unsigned int i=0; i<m_N_neighbors; i++) {
          const NEIGHBOR n = m_directions[i];
          const unsigned int n_idx = neighborIdx(n, x, y);
          int n_basin_id = m_basin_id[n_idx];

          if ((n_basin_id >= 0) and (basin_sink[n_basin_id] > SINK::UNDEFINED)) {
            //Part of loop -> check for parent basin
            n_basin_id = basin_sink[n_basin_id];
          }

          if (n_basin_id != self_basin_id) {
            const double n_height = m_usurf[n_idx];

            if (n_height > self_height) {
              height = n_height;
              idx = n_idx;
            } else {
              height = self_height;
              idx = self_idx;
            }

            if (spillway_height[self_basin_id] > height) {
              spillway_height[self_basin_id] = height;
              spillway_idx[self_basin_id] = (int) idx;
              spillway_neighbor_height[self_basin_id] = n_height;
              drain_basin_id[self_basin_id] = n_basin_id;
            } else if ((spillway_height[self_basin_id] == height) and (spillway_neighbor_height[self_basin_id] > n_height)) {
              spillway_idx[self_basin_id] = (int) idx;
              spillway_neighbor_height[self_basin_id] = n_height;
              drain_basin_id[self_basin_id] = n_basin_id;
            }
          }
        } //neighbor loop
      } // if (self_basin_id >= 0)

    } // x
  } // y
}

int LakeBasins::traceDrainagePath(unsigned int current_id, int *drain_basin_id, int *basin_sink, std::vector<int> &prev_ids) {
  int sink = basin_sink[current_id];

  if (sink == SINK::UNDEFINED) {

    std::vector<int>::iterator prev_id = prev_ids.begin();
    while ((prev_id != prev_ids.end()) and ((unsigned int)(*prev_id) != current_id)) {
      ++prev_id;
    }

    if (prev_id != prev_ids.end()) {
      //Found loop
      std::vector<int>::iterator start_loop = prev_id;
      int min_id = current_id;
      for (prev_id = start_loop; prev_id != prev_ids.end(); ++prev_id) {
        if (*prev_id < min_id) {
          min_id = *prev_id;
        }
      }
      //Communicate to all affected basins
      for (prev_id = start_loop; prev_id != prev_ids.end(); ++prev_id) {
        basin_sink[*prev_id] = min_id;
        std::cout<<*prev_id<<", ";
      }
      std::cout<<"\n";
      prev_ids.clear();
      sink = SINK::OTHER;
    } else {
      //No loop found -> Go to next basin
      prev_ids.push_back(current_id);

      sink = drain_basin_id[current_id];
      if (sink > SINK::UNDEFINED) {
        sink = traceDrainagePath((unsigned int)sink, drain_basin_id, basin_sink, prev_ids);
      }
    }

    if (basin_sink[current_id] == SINK::UNDEFINED) {
      //This could have changed if part of a loop -> then already set!
      basin_sink[current_id] = sink;
    }
  }

  return sink;
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
  
  for (unsigned int i=0; i<m_N_neighbors; i++) {
    const NEIGHBOR n = m_directions[i];
    const unsigned int idx = neighborIdx(n, x, y);
    const double val = m_usurf[idx];
    
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

