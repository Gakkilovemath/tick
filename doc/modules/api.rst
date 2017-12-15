:orphan:

.. _api:

=============
API Reference
=============

This is the full class and function references of tick. Please look at
the modules documentation cited below for more examples and use cases,
since direct class and function API is not enough for understanding their uses.


.. _api-linear_model:

:mod:`tick.linear_model`
========================
**User guide:** :ref:`linear_model`

This modules provides tools for inference and simulation of generalized linear models.



Inference
---------

.. currentmodule:: tick.linear_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LinearRegression
   LogisticRegression
   PoissonRegression

Models
------

.. currentmodule:: tick.linear_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelLinReg
   ModelLogReg
   ModelPoisReg
   ModelHinge
   ModelSmoothedHinge
   ModelQuadraticHinge

Simulation
----------

.. currentmodule:: tick.linear_model

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuLinReg
   SimuLogReg
   SimuPoisReg


.. _api-robust:

:mod:`tick.robust`
==================
**User guide:** :ref:`robust`

This module provides tools for robust inference of generalized linear models
and outliers detection.

Inference
---------

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: function.rst

   RobustLinearRegression
   std_mad
   std_iqr

Models
------

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHuber
   ModelModifiedHuber
   ModelAbsoluteRegression
   ModelEpsilonInsensitive
   ModelLinRegWithIntercepts


.. _api-survival:

:mod:`tick.survival`:
====================================================================

**User guide:** :ref:`survival`

This module provides tools for inference and simulation for survival analysis.

Inference
---------

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: function.rst

   CoxRegression
   nelson_aalen
   kaplan_meier

Models
------

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelCoxRegPartialLik
   ModelSCCS

Simulation
----------

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuCoxReg


.. _api-hawkes:

:mod:`tick.hawkes`
==================
**User guide:** :ref:`hawkes`

This module provides tools for inference and simulation of Hawkes processes.

.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesExpKern
   HawkesSumExpKern
   HawkesEM
   HawkesADM4
   HawkesBasisKernels
   HawkesSumGaussians
   HawkesConditionalLaw


.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHawkesExpKernLogLik
   ModelHawkesExpKernLeastSq
   ModelHawkesSumExpKernLogLik
   ModelHawkesSumExpKernLeastSq

Simulation
----------
.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst


   SimuPoissonProcess
   SimuInhomogeneousPoisson
   SimuHawkes
   SimuHawkesExpKernels
   SimuHawkesSumExpKernels
   SimuHawkesMulti

Hawkes kernels
--------------
.. currentmodule:: tick.hawkes

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HawkesKernelExp
   HawkesKernelSumExp
   HawkesKernelPowerLaw
   HawkesKernelTimeFunc



base.TimeFunction


.. _api-prox:

:mod:`tick.prox`: Proximal operators classes
==================================================

This module contains all the proximal operators available in tick.

**User guide:** See the :ref:`prox` section for further details.

.. automodule:: tick.prox
   :no-members:
   :no-inherited-members:

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   prox.ProxZero
   prox.ProxL1
   prox.ProxL1w
   prox.ProxElasticNet
   prox.ProxL2Sq
   prox.ProxL2
   prox.ProxMulti
   prox.ProxNuclear
   prox.ProxPositive
   prox.ProxEquality
   prox.ProxSlope
   prox.ProxTV
   prox.ProxBinarsity
   prox.ProxGroupL1


.. _api-solver:

:mod:`tick.solver`: Solver classes
========================================

This module contains all the solvers available in tick.

**User guide:** See the :ref:`solver` section for further details.

.. automodule:: tick.solver
   :no-members:
   :no-inherited-members:

Batch solvers
-------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   solver.GD
   solver.AGD
   solver.BFGS
   solver.GFB
   solver.SCPG

Stochastic solvers
------------------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   solver.SGD
   solver.AdaGrad
   solver.SVRG
   solver.SAGA
   solver.SDCA

History
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   solver.History


.. _api-plot:

:mod:`tick.plot`: Plotting utilities
====================================

This module contains some utilities functions for plotting

**User guide:** See the :ref:`plot` section for further details.

Functions
---------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_history
   plot.plot_hawkes_kernels
   plot.plot_hawkes_kernel_norms
   plot.plot_basis_kernels
   plot.plot_timefunction
   plot.plot_point_process
   plot.stems


.. _api-preprocessing:

:mod:`tick.preprocessing`: Preprocessing utilities
==================================================

This module contains some utilities functions for preprocessing of data.

**User guide:** See the :ref:`preprocessing` section for further details.

Classes
-------
.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.FeaturesBinarizer
   preprocessing.LongitudinalFeaturesProduct
   preprocessing.LongitudinalFeaturesLagger


.. _api-metrics:

:mod:`tick.metrics`: Metrics utilities
======================================

This module contains some functions to compute some metrics that help evaluate
the performance of learning techniques.

Functions
---------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.support_fdp
   metrics.support_recall


.. _api-simulation:

:mod:`tick.simulation`: basic tools for simulation
==================================================

This module contains basic tools from simulation.
**User guide:** See the :ref:`simulation` section for further details.

Features simulation
-------------------

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.features_normal_cov_uniform
   simulation.features_normal_cov_toeplitz

Weights simulation
------------------

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulation.weights_sparse_exp
   simulation.weights_sparse_gauss


.. _api-datasets:

:mod:`tick.dataset`: Real world dataset
=======================================

**User guide:** See the :ref:`dataset` section for further details.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dataset.fetch_tick_dataset
   dataset.fetch_hawkes_bund_data
