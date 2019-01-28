#include "LakeProperties_ConnectedComponents.hh"
#include <vector>
#include <cmath>


LakePropertiesCC::LakePropertiesCC(unsigned int n_rows,
                                   unsigned int n_cols,
                                   double cell_area,
                                   double *lake_depth,
                                   int *lake_ids)
  : m_nRows(n_rows), m_nCols(n_cols),
    m_lake_depth(lake_depth), m_cell_area(cell_area),
    m_lake_ids(lake_ids) {
  m_mask_run = new int [m_nRows * m_nCols];
  for (int i=0; i<(m_nRows * m_nCols); i++) {
    m_mask_run[i] = 0;
    m_lake_ids[i]  = -1;
  }
}

LakePropertiesCC::~LakePropertiesCC() {
  delete[] m_mask_run;
}


void LakePropertiesCC::run(int &N_lakes, double *&area, double *&volume) {
  unsigned int max_items = 2 * m_nRows;

  std::vector<unsigned int> parents(max_items), lengths(max_items), rows(max_items), columns(max_items);
  std::vector<double> depths_sum(max_items);

  for(unsigned int i = 0; i < 1; ++i) {
    parents[i]    = 0;
    lengths[i]    = 0;
    rows[i]       = 0;
    columns[i]    = 0;
    depths_sum[i] = 0.0;
  }

  unsigned int run_number = 0;

  for (unsigned int r = 0; r < m_nRows; ++r) {
    for (unsigned int c = 0; c < m_nCols; ++c) {
      if (ForegroundCond(r, c)) {
        checkForegroundPixel(c, r, run_number, rows, columns, parents, lengths, depths_sum);

        if ((run_number + 1) == max_items) {
          max_items += m_nRows;
          parents.resize(max_items);
          lengths.resize(max_items);
          rows.resize(max_items);
          columns.resize(max_items);
          depths_sum.resize(max_items);
        }
      }
    }
  }

  std::vector<unsigned int> N_sum(run_number);
  labelRuns(run_number, parents, lengths, N_sum, depths_sum, N_lakes);
  
  area   = new double[N_lakes];
  volume = new double[N_lakes];
  for (int i=0; i<N_lakes; i++) {
    area[i]   = N_sum[i] * m_cell_area;
    volume[i] = depths_sum[i] * m_cell_area;
  }
  
  //N_lakes contains number of all lakes...
  labelMap(run_number, rows, columns, parents, lengths);
  
}

bool LakePropertiesCC::ForegroundCond(unsigned int r, unsigned int c) {
  return (m_lake_depth[r * m_nCols + c] > 0.0);
}

void LakePropertiesCC::labelMap(unsigned int run_number,
                                std::vector<unsigned int> &rows,
                                std::vector<unsigned int> &columns,
                                std::vector<unsigned int> &parents,
                                std::vector<unsigned int> &lengths) {
  // label Lakes
  for(unsigned int k = 0; k <= run_number; ++k) {
    for(unsigned int n = 0; n < lengths[k]; ++n) {
      m_lake_ids[rows[k] * m_nCols + columns[k] + n] = parents[k];
    }
  }
}


void LakePropertiesCC::run_union(std::vector<unsigned int> &parents, unsigned int run1, unsigned int run2) {
  if((parents[run1] == run2) or (parents[run2] == run1)) {
    return;
  }

  while(parents[run1] != 0) {
      run1 = parents[run1];
  }

  while(parents[run2] != 0) {
      run2 = parents[run2];
  }

  if(run1 > run2) {
      parents[run1] = run2;
  }else if(run1 < run2) {
      parents[run2] = run1;
  }

}

void LakePropertiesCC::checkForegroundPixel(unsigned int c,
                                        unsigned int r,
                                        unsigned int &run_number,
                                        std::vector<unsigned int> &rows,
                                        std::vector<unsigned int> &columns,
                                        std::vector<unsigned int> &parents,
                                        std::vector<unsigned int> &lengths,
                                        std::vector<double> &depths_sum) {
  if((c > 0) && (m_mask_run[r*m_nCols + (c-1)] > 0)) {
    // one to the left is also foreground: continue the run
    lengths[run_number] += 1;
    depths_sum[run_number] += m_lake_depth[r * m_nCols + c];
  } else {
    //one to the left is a background pixel (or this is column 0): start a new run
    unsigned int parent;
    if((r > 0) and (m_mask_run[(r - 1) * m_nCols + c] > 0)) {
      //check the pixel above and set the parent
      parent = (unsigned int)m_mask_run[(r - 1) * m_nCols + c];
    } else {
      parent = 0;
    }

    run_number += 1;

    rows[run_number] = r;
    columns[run_number] = c;
    parents[run_number] = parent;
    lengths[run_number] = 1;
    depths_sum[run_number] = m_lake_depth[r * m_nCols + c];
  }

  if((r > 0) and (m_mask_run[(r - 1) * m_nCols + c] > 0)) {
    run_union(parents, (unsigned int)m_mask_run[(r - 1) * m_nCols + c], run_number);
  }

  m_mask_run[r * m_nCols + c] = run_number;
}


void LakePropertiesCC::labelRuns(unsigned int run_number,
                                 std::vector<unsigned int> &parents,
                                 std::vector<unsigned int> &lengths,
                                 std::vector<unsigned int> &N_sum,
                                 std::vector<double> &depths_sum,
                                 int &N_lakes) {
  unsigned int label = 0;
  for(unsigned int k = 0; k <= run_number; ++k) {
    if(parents[k] == 0) {
      parents[k] = label;
      label += 1;
      N_sum[parents[k]] = lengths[k];
      depths_sum[parents[k]] = depths_sum[k];
    } else {
      parents[k] = parents[parents[k]];
      N_sum[parents[k]] += lengths[k];
      depths_sum[parents[k]] += depths_sum[k];
    }
  }

  N_lakes = (int) label;
}
