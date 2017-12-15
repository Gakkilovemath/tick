#include "tick/inference/hawkes_non_param_cumulant.h"

HawkesNonParamCumulant::HawkesNonParamCumulant(double half_width)
  : half_width(half_width){

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
    ulong timestamps_in_interval = 0;

    double sub_res = 0.;

    while (l < n_j) {
      double t_j_l_minus_t_i_k = (*timestamps_j)[l] - t_i_k;
      double abs_t_j_l_minus_t_i_k = abs(t_j_l_minus_t_i_k);

      if (abs_t_j_l_minus_t_i_k < width) {
        sub_res += width - abs_t_j_l_minus_t_i_k;

        if (abs_t_j_l_minus_t_i_k < half_width) timestamps_in_interval++;
      } else break;

      l += 1;
    }

    if (l == n_j) continue;
    res_C += timestamps_in_interval - trend_C_j;
    res_J += sub_res - trend_J_j;
  }

  res_C /= (*end_times)[r];
  res_J /= (*end_times)[r];

  ArrayDouble return_array{res_C, res_J};
  return return_array.as_sarray_ptr();
}

double HawkesNonParamCumulant::compute_E_ijk_rect(ulong r, ulong i, ulong j, ulong k,
                                                  double mean_intensity_i, double mean_intensity_j,
                                                  double J_ij) {
  auto timestamps_i = timestamps_list[r][i];
  auto timestamps_j = timestamps_list[r][j];
  auto timestamps_k = timestamps_list[r][k];

  double L_i = mean_intensity_i;
  double L_j = mean_intensity_j;

  double res = 0;
  ulong last_l = 0;
  ulong last_m = 0;
  ulong n_i = timestamps_i->size();
  ulong n_j = timestamps_j->size();
  ulong n_k = timestamps_k->size();

  double trend_i = L_i * 2 * half_width;
  double trend_j = L_j * 2 * half_width;

  for (ulong t = 0; t < n_k; ++t) {
    double tau = (*timestamps_k)[t];

    if (tau - half_width < 0) continue;

    while (last_l < n_i) {
      if ((*timestamps_i)[last_l] <= tau - half_width) last_l += 1;
      else break;
    }
    ulong l = last_l;

    while (l < n_i) {
      if ((*timestamps_i)[l] < tau + half_width) l += 1;
      else break;
    }

    while (last_m < n_j) {
      if ((*timestamps_j)[last_m] <= tau - half_width) last_m += 1;
      else break;
    }
    ulong m = last_m;

    while (m < n_j) {
      if ((*timestamps_j)[m] < tau + half_width) m += 1;
      else break;
    }

    if ((m == n_j) || (l == n_i)) continue;

    res += (l - last_l - trend_i) * (m - last_m - trend_j) - J_ij;
  }
  res /= (*end_times)[r];
  return res;
}
