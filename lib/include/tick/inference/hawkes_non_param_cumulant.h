//
// Created by Martin Bompaire on 14/12/2017.
//

#ifndef TICK_HAWKES_NON_PARAM_CUMULANT_H
#define TICK_HAWKES_NON_PARAM_CUMULANT_H

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/optim/model/base/hawkes_list.h"

class HawkesNonParamCumulant : public ModelHawkesList {

  double half_width;
  double sigma;

 public:
  HawkesNonParamCumulant(double half_width);

  SArrayDoublePtr compute_A_and_I_ij_rect(ulong r, ulong i, ulong j, double mean_intensity_j);

  /**
   * Computes K^{kij}
   *
   * @param r
   * @param i
   * @param j
   * @param k
   * @param mean_intensity_i
   * @param mean_intensity_j
   * @param J_ij
   * @return
   */
  double compute_E_ijk_rect(ulong r, ulong i, ulong j, ulong k,
                            double mean_intensity_i, double mean_intensity_j,
                            double J_ij);
};

#endif //TICK_HAWKES_NON_PARAM_CUMULANT_H
