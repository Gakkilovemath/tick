

.. _robust:

==================
:mod:`tick.robust`
==================

This module provides tools for robust inference of generalized linear models
and outliers detection.

Inference
=========

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: function.rst

   RobustLinearRegression
   std_mad
   std_iqr

Models
======

.. currentmodule:: tick.robust

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHuber
   ModelModifiedHuber
   ModelAbsoluteRegression
   ModelEpsilonInsensitive
   ModelLinRegWithIntercepts


The setting is the same as with generalized linear models, but where we used an
individual intercept :math:`b_i` for each :math:`i=1, \ldots, n`.
Namely we consider a goodness-of-fit of the form

.. math::

    f(w, b) = \frac 1n \sum_{i=1}^n \ell(y_i, b_i + x_i^\top w),

where :math:`w \in \mathbb R^d` is a vector containing the model weights,
:math:`b \in \mathbb R^n` is a vector of individual intercepts and
:math:`\ell : \mathbb R^2 \rightarrow \mathbb R` is a loss function.
Estimation of :math:`b` under a sparse penalization (such as L1 or
Sorted L1, see :ref:`prox classes <prox>`) allows to detect outliers
using this model.


========================================  ==============  ==========  ==========================================
Model                                     Type            Label type  Class
========================================  ==============  ==========  ==========================================
Linear regression with intercepts         Regression      Continuous  :class:`ModelLinRegWithIntercepts <tick.optim.model.ModelLinRegWithIntercepts>`
Huber regression                          Regression      Continuous  :class:`ModelHuber <tick.optim.model.ModelHuber>`
Epsilon-insensitive regression            Regression      Continuous  :class:`ModelEpsilonInsensitive <tick.optim.model.ModelEpsilonInsensitive>`
Absolute regression                       Regression      Continuous  :class:`ModelAbsoluteRegression <tick.optim.model.ModelAbsoluteRegression>`
Modified Huber loss                       Classification  Binary      :class:`ModelModifiedHuber <tick.optim.model.ModelModifiedHuber>`
========================================  ==============  ==========  ==========================================




Modified Huber loss                       Classification  Binary      :class:`ModelModifiedHuber <tick.linear_model.ModelModifiedHuber>`


:class:`ModelHuber <tick.robust.ModelHuber>`
********************************************

The Huber loss for robust regression (less sensitive to
outliers) is given by

.. math::
    \ell(y, y') =
    \begin{cases}
    \frac 12 (y' - y)^2 &\text{ if } |y' - y| \leq \delta \\
    \delta (|y' - y| - \frac 12 \delta) &\text{ if } |y' - y| > \delta
    \end{cases}

for :math:`y, y' \in \mathbb R`, where :math:`\delta > 0` can be tuned
using the ``threshold`` argument.

----------------------------------------


:class:`ModelEpsilonInsensitive <tick.robust.ModelEpsilonInsensitive>`
**********************************************************************

Epsilon-insensitive loss, given by

.. math::
    \ell(y, y') =
    \begin{cases}
    |y' - y| - \epsilon &\text{ if } |y' - y| > \epsilon \\
    0 &\text{ if } |y' - y| \leq \epsilon
    \end{cases}

for :math:`y, y' \in \mathbb R`, where :math:`\epsilon > 0` can be tuned using
the ``threshold`` argument.

----------------------------------------

:class:`ModelAbsoluteRegression <tick.robust.ModelAbsoluteRegression>`
**********************************************************************

The L1 loss given by

.. math::
    \ell(y, y') = |y' - y|

for :math:`y, y' \in \mathbb R`

----------------------------------------


:class:`ModelModifiedHuber <tick.robust.ModelModifiedHuber>`
******************************************************************

The modified Huber loss, used for robust classification (less sensitive to
outliers). The loss is given by

.. math::
    \ell(y, y') =
    \begin{cases}
    - 4 y y' &\text{ if } y y' \leq -1 \\
    (1 - y y')^2 &\text{ if } -1 < y y' < 1 \\
    0 &\text{ if } y y' \geq 1
    \end{cases}

for :math:`y \in \{ -1, 1\}` and :math:`y' \in \mathbb R`

