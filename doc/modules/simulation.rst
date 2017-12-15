
.. _simulation:

==========================================
:mod:`tick.simulation`: simulation toolbox
==========================================

tick provides several classes to simulate datasets.
This is particularly useful to test optimization algorithms, and to
compare the statistical properties of inference methods.

For now, tick gives simulation classes for Generalized Linear
Models, Cox regression, Poisson Processes with any intensity and
Hawkes processes. Utilities for simulation of model coefficients
(with sparsity, etc.) and utilities for features matrix simulation
are provided as well.

.. contents::
    :depth: 2
    :backlinks: none


1. Simulation tools
===================

We gather in this section tools for the simulation of datasets : simulation of
model weights, simulation of a features matrix, kernels for the simulation of
Hawkes processes and time functions for the simulation of Poisson processes.

1.1 Simulation of model weights
-------------------------------

Here are functions for the simulation of model weights.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.weights_sparse_exp
   simulation.weights_sparse_gauss

**Example**

.. plot:: modules/code_samples/simulation/plot_simulation_weights.py
    :include-source:

1.2 Simulation of a features matrix
-----------------------------------

Here are functions for the simulation of a features matrix: each simulated
vector or features is distributed as a centered Gaussian vector with
a particular covariance matrix (uniform symmetrized or toeplitz).

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.features_normal_cov_uniform
   simulation.features_normal_cov_toeplitz

**Example**

.. todo::

    Insert a sample code here

1.3 Time function
-----------------

A class of time function to explicitly define a function on
:math:`[0, \infty)`. It uses several types of interpolation to determine value
between two points.
It is used for the simulation of an inhomogeneous Poisson process and some
Hawkes processes.

**Example**

.. plot:: modules/code_samples/simulation/plot_time_function.py
    :include-source:

1.4 Kernels for Hawkes process simulation
-----------------------------------------

A Hawkes process is defined through its kernels which are functions defined on
:math:`[0, \infty)`.

.. plot:: modules/code_samples/simulation/plot_hawkes_kernels.py
    :include-source:

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.HawkesKernel0
   simulation.HawkesKernelExp
   simulation.HawkesKernelSumExp
   simulation.HawkesKernelPowerLaw
   simulation.HawkesKernelTimeFunc



3. Survival analysis simulation
===============================

3.1 Cox regression simulation (proportional hazards)
----------------------------------------------------

We provide a class for the simulation of a Cox regression model with right-censoring.
This generates data in the form of i.i.d triplets :math:`(x_i, t_i, c_i)`
for :math:`i=1, \ldots, n`, where :math:`x_i \in \mathbb R^d` is a features vector,
:math:`t_i \in \mathbb R_+` is the survival time and :math:`c_i \in \{ 0, 1 \}` is the
indicator of right censoring.
Note that :math:`c_i = 1` means that :math:`t_i` is a failure time
while :math:`c_i = 0` means that :math:`t_i` is a censoring time.

.. todo::

    Describe precisely cox model

For now, the following class is available


===================================  ===================================
Model                                Class
===================================  ===================================
Cox regression with right-censoring  :class:`tick.simulation.SimuCoxReg`
===================================  ===================================

**Examples**

.. plot:: ../examples/plot_simulation_coxreg.py
    :include-source:

3.2 Self-controlled case series (SCCS)
--------------------------------------

We provide a class for the simulation of a SCCS regression model with right-censoring.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.SimuSCCS



4. Point process simulation
===========================

Tick has a particular focus on inference for point processes.
It therefore proposes as well tools for their simulation: for now, inhomogeneous
Poisson processes and Hawkes processes.

4.1 Poisson processes
---------------------

Both homogeneous and inhomogeneous Poisson process might be simulated with tick
thanks to the following classes.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.SimuPoissonProcess
   simulation.SimuInhomogeneousPoisson

**Examples**

A Poisson process with constant intensity

.. plot:: modules/code_samples/simulation/plot_poisson_constant_intensity.py
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

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   simulation.SimuHawkes
   simulation.SimuHawkesExpKernels
   simulation.SimuHawkesSumExpKernels

.. plot:: modules/code_samples/simulation/plot_hawkes_1d_simu.py
    :include-source:

.. plot:: modules/code_samples/simulation/plot_hawkes_multidim_simu.py
    :include-source: