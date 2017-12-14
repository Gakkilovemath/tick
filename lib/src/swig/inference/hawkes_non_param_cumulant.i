

%include std_shared_ptr.i
%shared_ptr(HawkesNonParamCumulant);

%{
#include "tick/inference/hawkes_non_param_cumulant.h"
%}


class HawkesNonParamCumulant : public ModelHawkesList {

public:
  HawkesNonParamCumulant(double half_width, double sigma);

  SArrayDoublePtr compute_A_and_I_ij_rect(ulong r, ulong i, ulong j, double mean_intensity_j);
};