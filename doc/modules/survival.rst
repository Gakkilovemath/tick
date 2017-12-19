

.. _survival:

====================
:mod:`tick.survival`
====================

Survival analysis

1. Inference
============

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: function.rst

   CoxRegression
   nelson_aalen
   kaplan_meier


2. Models
=========

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelCoxRegPartialLik
   ModelSCCS

=================================  ==============================
Model                              Class
=================================  ==============================
Cox regression partial likelihood  :class:`ModelCoxRegPartialLik <tick.optim.model.ModelCoxRegPartialLik>`
Self Control Case Series           :class:`ModelSCCS <tick.optim.model.ModelSCCS>`
=================================  ==============================

3. Simulation
=============

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuCoxReg


Simulation
==========

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
Cox regression with right-censoring  :class:`tick.survival.SimuCoxReg`
===================================  ===================================

**Examples**

.. plot:: ../examples/plot_simulation_coxreg.py
    :include-source:

3.2 Self-controlled case series (SCCS)
--------------------------------------

We provide a class for the simulation of a SCCS regression model with right-censoring.

.. currentmodule:: tick.survival

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimuSCCS


