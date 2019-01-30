#ifndef _LAKEPROPERTIES_CONNECTEDCOMPONENTS_H_
#define _LAKEPROPERTIES_CONNECTEDCOMPONENTS_H_

#include <vector>

class LakePropertiesCC {
public:
  LakePropertiesCC(unsigned int n_rows,
                   unsigned int n_cols,
                   double cell_area,
                   double *lake_depth,
                   int *lake_ids);
  virtual ~LakePropertiesCC();
  void run(int &N_lakes, double *&area, double *&volume, double *&max_depth_out);

protected:
  unsigned int m_nRows, m_nCols;
  double *m_lake_depth;
  double m_cell_area;
  int *m_mask_run, *m_lake_ids;
  void run_union(std::vector<unsigned int> &parents,
                 unsigned int run1,
                 unsigned int run2);
  void checkForegroundPixel(unsigned int c, unsigned int r, 
                            unsigned int &run_number,
                            std::vector<unsigned int> &rows,
                            std::vector<unsigned int> &columns,
                            std::vector<unsigned int> &parents,
                            std::vector<unsigned int> &lengths,
                            std::vector<double> &depths_sum,
                            std::vector<double> &max_depth);
  virtual void labelRuns(unsigned int run_number,
                         std::vector<unsigned int> &parents,
                         std::vector<unsigned int> &lengths,
                         std::vector<unsigned int> &N_sum,
                         std::vector<double> &depths_sum,
                         std::vector<double> &max_depth,
                         int &N_lakes);
  virtual bool ForegroundCond(unsigned int r,
                              unsigned int c);
  virtual void labelMap(unsigned int run_number,
                        std::vector<unsigned int> &rows,
                        std::vector<unsigned int> &columns,
                        std::vector<unsigned int> &parents,
                        std::vector<unsigned int> &lengths);
};

#endif
