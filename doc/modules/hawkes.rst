

.. _hawkes:

==================
:mod:`tick.hawkes`
==================

These classes are called learners. They are meant to be very user friendly
and are most of the time good enough to infer many models.

These classes aim to be scikit-learn compatible and hence implement a `fit`
method.

.. contents::
    :depth: 3
    :backlinks: none


*tick* also provides learners to infer Hawkes processes.

Hawkes processes are point processes defined by the intensities:

.. math::
    \forall i \in [1 \dots D], \quad
    \lambda_i(t) = \mu_i + \sum_{j=1}^D \int \phi_{ij}(t - s) dN_j(s)

where

* :math:`D` is the number of nodes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels
* :math:`dN_j` are the processes differentiates


Parametric Hawkes learners
==========================

One way to infer Hawkes processes is to suppose their kernels have a
parametric shape. Usually people induces an exponential parametrization as it
allows very fast computations. The models associated to these learners are
presented in


As for linear models, `tick.hawkes.HawkesExpKern` and
`tick.hawkes.HawkesSumExpKern` are combination of solver, model and prox.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesExpKern
   HawkesSumExpKern
   HawkesADM4
   HawkesSumGaussians

.. plot:: modules/code_samples/plot_hawkes_sum_exp_kernels.py
    :include-source:

Non-parametric Hawkes learners
==============================

Some other Hawkes learners perform non parametric evaluation of the kernels
and hence don't rely the previous exponential parametrization.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesEM
   HawkesBasisKernels
   HawkesConditionalLaw

These learners might then infer much more exotic kernels

.. plot:: ../examples/plot_hawkes_em.py
    :include-source:


Time function
=============

A class of time function to explicitly define a function on
:math:`[0, \infty)`. It uses several types of interpolation to determine value
between two points.
It is used for the simulation of an inhomogeneous Poisson process and some
Hawkes processes.

**Example**

.. plot:: modules/code_samples/plot_time_function.py
    :include-source:

1.4 Kernels for Hawkes process simulation
=========================================

A Hawkes process is defined through its kernels which are functions defined on
:math:`[0, \infty)`.

.. plot:: modules/code_samples/plot_hawkes_kernels.py
    :include-source:

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesKernel0
   HawkesKernelExp
   HawkesKernelSumExp
   HawkesKernelPowerLaw
   HawkesKernelTimeFunc


4. Point process simulation
===========================

Tick has a particular focus on inference for point processes.
It therefore proposes as well tools for their simulation: for now, inhomogeneous
Poisson processes and Hawkes processes.

4.1 Poisson processes
---------------------

Both homogeneous and inhomogeneous Poisson process might be simulated with tick
thanks to the following classes.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuPoissonProcess
   SimuInhomogeneousPoisson

**Examples**

A Poisson process with constant intensity

.. plot:: modules/code_samples/plot_poisson_constant_intensity.py
    :include-source:

A Poisson process with variable intensity. In this case, the intensity is
defined through a `tick.base.TimeFunction`

.. plot:: ../examples/plot_poisson_inhomogeneous.py
    :include-source:

4.2 Hawkes processes
--------------------

Simulation of Hawkes processes can be done using the following classes. The
main class `tick.simulation.SimuHawkes` might use any type of kernels and
will perform simulation. For some specific cases there are some classes
dedicated to a type of kernel: exponential or sum of exponential kernels.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuHawkes
   SimuHawkesExpKernels
   SimuHawkesSumExpKernels

.. plot:: modules/code_samples/plot_hawkes_1d_simu.py
    :include-source:

.. plot:: modules/code_samples/plot_hawkes_multidim_simu.py
    :include-source:



.. _optim-model-hawkes:

1.5. Hawkes models
------------------

Hawkes processes are point processes defined by the intensities:

.. math::
    \forall i \in [1 \dots D], \quad
    \lambda_i(t) = \mu_i + \sum_{j=1}^D \int \phi_{ij}(t - s) dN_j(s)

where

* :math:`D` is the number of nodes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels
* :math:`dN_j` are the processes differentiates

One way to infer Hawkes processes is to suppose their kernels have a
parametric shape. Usually kernels have an exponential parametrization as it
allows very fast computations.

In *tick*, three exponential models are implemented. They differ by the
parametrization of the kernel (exponential or sum-exponential) or by the loss
function used (least squares or log-likelihood).

===============================================================  ===============================
Model                                                            Class
===============================================================  ===============================
Least-squares for Hawkes model with exponential kernels          :class:`ModelHawkesExpKernLeastSq <tick.optim.model.ModelHawkesExpKernLeastSq>`
Log-likelihood for Hawkes model with exponential kernels         :class:`ModelHawkesExpKernLogLik <tick.optim.model.ModelHawkesExpKernLogLik>`
Least-squares for Hawkes model with sum of exponential kernels   :class:`ModelHawkesSumExpKernLeastSq <tick.optim.model.ModelHawkesSumExpKernLeastSq>`
Log-likelihood for Hawkes model with sum of exponential kernels  :class:`ModelHawkesSumExpKernLogLik <tick.optim.model.ModelHawkesSumExpKernLogLik>`
===============================================================  ===============================

