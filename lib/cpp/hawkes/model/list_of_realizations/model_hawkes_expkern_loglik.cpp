// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_loglik.h"

ModelHawkesFixedExpKernLogLikList::ModelHawkesFixedExpKernLogLikList(
  const double decay, const int max_n_threads) :
  ModelHawkesFixedKernLogLikList(max_n_threads), decay(decay) {}

ulong ModelHawkesFixedExpKernLogLikList::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes;
}
