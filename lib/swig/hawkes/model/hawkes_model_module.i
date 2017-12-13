// License: BSD 3 clause

%module hawkes_model

%include defs.i
%include serialization.i
%include <std_shared_ptr.i>

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include base_model_module.i

%shared_ptr(ModelHawkes);
%shared_ptr(ModelHawkesList);
%shared_ptr(ModelHawkesLeastSqList);
%shared_ptr(ModelHawkesFixedKernLogLikList);

%shared_ptr(ModelHawkesFixedExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedExpKernLogLikList);
%shared_ptr(ModelHawkesFixedSumExpKernLogLikList);


%include base/model_hawkes.i
%include base/model_hawkes_list.i
%include base/model_hawkes_leastsq.i
%include base/model_hawkes_loglik.i

%include list_of_realizations/model_hawkes_expkern_leastsq.i
%include list_of_realizations/model_hawkes_sumexpkern_leastsq.i
%include list_of_realizations/model_hawkes_expkern_loglik.i
%include list_of_realizations/model_hawkes_sumexpkern_loglik.i
