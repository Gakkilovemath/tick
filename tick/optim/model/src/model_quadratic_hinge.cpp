// License: BSD 3 clause

#include "model_quadratic_hinge.h"

ModelQuadraticHinge::ModelQuadraticHinge(const SBaseArrayDouble2dPtr features,
                                         const SArrayDoublePtr labels,
                                         const bool fit_intercept,
                                         const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {}

const char *ModelQuadraticHinge::get_class_name() const {
  return "ModelQuadraticHinge";
}

double ModelQuadraticHinge::loss_i(const ulong i,
                          const ArrayDouble &coeffs) {
  // Compute x_i^T \beta + b
  const double z = get_inner_prod(i, coeffs);
  const double d = get_label(i) - z;
  return d * d / 2;
}

double ModelQuadraticHinge::grad_i_factor(const ulong i,
                                          const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  return z - get_label(i);
}

void ModelQuadraticHinge::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = features_norm_sq[i] + 1;
      } else {
        lip_consts[i] = features_norm_sq[i];
      }
    }
  }
}
