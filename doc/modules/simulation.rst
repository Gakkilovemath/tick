
.. _simulation:

======================
:mod:`tick.simulation`
======================

Simulation toolbox

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

.. currentmodule:: tick.simulation

.. autosummary::
   :toctree: generated/
   :template: function.rst

   weights_sparse_exp
   weights_sparse_gauss

**Example**

.. plot:: modules/code_samples/plot_simulation_weights.py
    :include-source:

1.2 Simulation of a features matrix
-----------------------------------

Here are functions for the simulation of a features matrix: each simulated
vector or features is distributed as a centered Gaussian vector with
a particular covariance matrix (uniform symmetrized or toeplitz).

.. currentmodule:: tick.simulation

.. autosummary::
   :toctree: generated/
   :template: function.rst

   features_normal_cov_uniform
   features_normal_cov_toeplitz

**Example**

.. todo::

    Insert a sample code here
