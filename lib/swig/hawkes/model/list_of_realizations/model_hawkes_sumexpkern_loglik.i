// License: BSD 3 clause


%{
#include "tick/hawkes/model/list_of_realizations/model_hawkes_sumexpkern_loglik.h"
%}


class ModelHawkesFixedSumExpKernLogLikList : public ModelHawkesFixedKernLogLikList {
    
public:
    
  ModelHawkesFixedSumExpKernLogLikList(const ArrayDouble &decays,
                                       const int max_n_threads = 1);

  void set_decays(ArrayDouble &decays);
  SArrayDoublePtr get_decays() const;
};
