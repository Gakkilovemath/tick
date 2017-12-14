#include "tick/inference/hawkes_non_param_cumulant.h"

HawkesNonParamCumulant::HawkesNonParamCumulant(double half_width, double sigma)
  : half_width(half_width), sigma(sigma) {

}

void HawkesNonParamCumulant::compute_A_and_I_ij_rect(ulong i, ulong j, double mean_intensity_j) {

}