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
  HawkesNonParamCumulant(double half_width, double sigma);

  void compute_A_and_I_ij_rect(ulong i, ulong j, double mean_intensity_j);
};

#endif //TICK_HAWKES_NON_PARAM_CUMULANT_H
