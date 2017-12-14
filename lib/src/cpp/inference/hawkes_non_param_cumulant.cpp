#include "tick/inference/hawkes_non_param_cumulant.h"

HawkesNonParamCumulant::HawkesNonParamCumulant(double half_width, double sigma)
  : half_width(half_width), sigma(sigma) {

}

SArrayDoublePtr HawkesNonParamCumulant::compute_A_and_I_ij_rect(ulong r, ulong i, ulong j,
                                                                double mean_intensity_j) {
  auto timestamps_i = timestamps_list[r][i];
  auto timestamps_j = timestamps_list[r][j];

  ulong n_i = timestamps_i->size();
  ulong n_j = timestamps_j->size();
  double res_C = 0;
  double res_J = 0;
  double width = 2 * half_width;
  double trend_C_j = mean_intensity_j * width;
  double trend_J_j = mean_intensity_j * width * width;

  ulong last_l = 0;
  for (ulong k = 0; k < n_i; ++k) {
    double t_i_k = (*timestamps_i)[k];
    double t_i_k_minus_half_width = t_i_k - half_width;
    double t_i_k_minus_width = t_i_k - width;

    if (t_i_k_minus_half_width < 0) continue;

    // Find next t_j_l that occurs width before t_i_k
    while (last_l < n_j) {
      if ((*timestamps_j)[last_l] <= t_i_k_minus_width) ++last_l;
      else break;
    }

    ulong l = last_l;
    ulong timestamps_in_half_width_interval = 0;

    double sub_res = 0.;

    while (l < n_j) {
      double t_j_l_minus_t_i_k = (*timestamps_j)[l] - t_i_k;
      double abs_t_j_l_minus_t_i_k = abs(t_j_l_minus_t_i_k);

      if (abs_t_j_l_minus_t_i_k < width) {
        double sign = t_j_l_minus_t_i_k < 0? 1. : -1.;
        sub_res += width + sign * t_j_l_minus_t_i_k;

        if (abs_t_j_l_minus_t_i_k < half_width) timestamps_in_half_width_interval++;
      }
      else break;

      l += 1;
    }

    if (l == n_j) continue;
    res_C += timestamps_in_half_width_interval - trend_C_j;
    res_J += sub_res - trend_J_j;
  }

  res_C /= (*end_times)[r];
  res_J /= (*end_times)[r];

  ArrayDouble return_array {res_C, res_J};
  return return_array.as_sarray_ptr();
}