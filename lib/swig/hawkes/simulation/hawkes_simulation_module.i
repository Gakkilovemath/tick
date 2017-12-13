// License: BSD 3 clause

%module hawkes_simulation

%include defs.i
%include serialization.i

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include point_process.i
%include poisson.i
%include inhomogeneous_poisson.i
%include hawkes.i
%include hawkes_kernels.i
