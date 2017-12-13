// License: BSD 3 clause

%module hawkes_model

%include defs.i
%include serialization.i

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include base_model_module.i

%shared_ptr(ModelHawkes);

%shared_ptr(ModelHawkesSingle);
%shared_ptr(ModelHawkesFixedExpKernLogLik);
%shared_ptr(ModelHawkesFixedSumExpKernLogLik);
%shared_ptr(ModelHawkesFixedExpKernLeastSq);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSq);

%shared_ptr(ModelHawkesList);
%shared_ptr(ModelHawkesLeastSqList);
%shared_ptr(ModelHawkesFixedKernLogLikList);
%shared_ptr(ModelHawkesFixedExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedExpKernLogLikList);
%shared_ptr(ModelHawkesFixedSumExpKernLogLikList);


%include hawkes_fixed_expkern_leastsq.i
%include hawkes_fixed_sumexpkern_leastsq.i
%include hawkes_fixed_expkern_loglik.i
%include hawkes_fixed_sumexpkern_loglik.i

%include variants/hawkes_list.i
%include variants/hawkes_leastsq_list.i
%include variants/hawkes_fixed_kern_loglik_list.i
%include variants/hawkes_fixed_expkern_leastsq_list.i
%include variants/hawkes_fixed_sumexpkern_leastsq_list.i
%include variants/hawkes_fixed_expkern_loglik_list.i
%include variants/hawkes_fixed_sumexpkern_loglik_list.i