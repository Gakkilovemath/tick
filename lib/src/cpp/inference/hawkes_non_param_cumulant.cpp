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
        sub_res += width - abs_t_j_l_minus_t_i_k;

        if (abs_t_j_l_minus_t_i_k < half_width) timestamps_in_half_width_interval++;
      } else break;

      l += 1;
    }

    if (l == n_j) continue;
    res_C += timestamps_in_half_width_interval - trend_C_j;
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
  auto realization_i = timestamps_list[r][i];
  auto realization_j = timestamps_list[r][j];
  auto realization_k = timestamps_list[r][k];

  double L_i = mean_intensity_i;
  double L_j = mean_intensity_j;

  double T = (*end_times)[r];

  double res = 0;
  ulong u = 0;
  ulong x = 0;
  ulong n_i = realization_i->size();
  ulong n_j = realization_j->size();
  ulong n_k = realization_k->size();

  double trend_i = L_i * 2 * half_width;
  double trend_j = L_j * 2 * half_width;

  double b = half_width;
  double a = -half_width;

  for (ulong t = 0; t < n_k; ++t) {
    double tau = (*realization_k)[t];

    if (tau + a < 0) continue;

    while (u < n_i) {
      if ((*realization_i)[u] <= tau + a) u += 1;
      else break;
    }
    ulong v = u;

    while (v < n_i) {
      if ((*realization_i)[v] < tau + b) v += 1;
      else break;
    }

    while (x < n_j) {
      if ((*realization_j)[x] <= tau + a) x += 1;
      else break;
    }
    ulong y = x;

    while (y < n_j) {
      if ((*realization_j)[y] < tau + b) y += 1;
      else break;
    }

    if ((y == n_j) || (v == n_i)) continue;

    res += (v - u - trend_i) * (y - x - trend_j) - J_ij;
  }
  res /= T;
  return res;
}
