.. title:: Tick

.. raw:: html

  <div
    class="jumbotron"
    style="padding-top: 10px; padding-bottom: 30px; position: relative"
  >
     <div class="container">
        <h1 style="font-size:40px">tick</h1>
        <p style="font-size:18px">
           tick a machine learning library for Python 3.
           The focus is on statistical learning for time dependent systems,
           such as point processes. Tick features also tools for generalized
           linear models, and a generic optimization toolbox.
        </p>
        <p style="font-size:18px">
           The core of the library is an optimization module providing model
           computational classes, solvers and proximal operators for regularization.

           It comes also with inference and simulation tools intended for end-users.
        </p>
        <a class="btn btn-primary btn-lg" href="auto_examples/index.html"
        role="button">
           Show me »
        </a>
     </div>
     <a href="https://github.com/X-DataInitiative/tick">
       <img style="position: absolute; top: 0; right: 0"
         src="_static/images/fork_me_on_github.png">
     </a>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="auto_examples/index.html">
           <h2>Examples</h2>
        </a>
        <p>
           Examples of how to use tick for inference of a bunch of models, for
           simulations and so much more !
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/hawkes.html">
           <h2>tick.hawkes</h2>
        </a>
        <p>
           A comprehensive module for inference and simulation Hawkes processes.
        </p>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/linear_model.html">
           <h2>tick.linear_model</h2>
        </a>
        <p>
            A scikit-learn compatible module that proposes tools for inference
            and simulation of linear models, including among others linear,
            logistic and Poisson regression. This module propose a large
            number of combinations of models and penalizations from the
            tick.prox module, and uses state-of-the-art
            stochastic optimization solvers proposed in the tick.solver module.
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/robust.html">
           <h2>tick.robust</h2>
        </a>
        <p>
            A scikit-learn compatible module that proposes tools for robust
            inference. It features tools for outliers detection and models such
            as Huber regression, among others.
        </p>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/survival.html">
           <h2>tick.survival</h2>
        </a>
        <p>A module that provides basic tools for survival analysis, such as
            Cox regression.
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/dataset.html">
           <h2>tick.dataset</h2>
        </a>
        <p>Provides easy access to datasets used as benchmarks in tick.
        </p>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/simulation.html">
           <h2>tick.simulation</h2>
        </a>
        <p>Basic tools for simulation, such as simulation of model weights and
            feature matrices.
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/preprocessing.html">
           <h2>tick.preprocessing</h2>
        </a>
        <p>Some tools for preprocessing, such as features binarization (useful
            for the binarsity penalization) and some tools for preprocessing of
            longitudinal features.
        </p>
     </div>
  </div>

  <div class="row" style="margin-bottom:40px">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/solver.html">
           <h2>tick.solver</h2>
        </a>
        <p>
           A module that provides a bunch of state-of-the-art optimization
           algorithms, both batch and stochastic. This is one of the main
           pillars of tick.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/prox.html">
           <h2>Proximal operators</h2>
        </a>
        <p>
           Flexible proximal operators for penalization of models weights. It
           can be used seeminglessly with (almost) any model and any solver.
            This module is used throughout the library.
        </p>
     </div>
 </div>

  <div class="row" style="margin-bottom:40px">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/R.html">
           <h2>Use tick in R</h2>
        </a>
        <p>
           How to use tick from the R software
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/api.html">
           <h2>API reference</h2>
        </a>
        <p>
           The full tick API
        </p>
     </div>
 </div>

  <div class="row" style="margin-bottom:40px">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/dev.html">
           <h2>Development</h2>
        </a>
        <p>
           You would like to contribute? Here you will find many tips.
        </p>
     </div>
  </div>


.. toctree::
    :maxdepth: 2
    :hidden:

    modules/hawkes
    modules/linear_model
    modules/robust
    modules/survival
    modules/simulation
    modules/solver
    modules/prox
    modules/plot
    modules/preprocessing
    modules/dataset
    modules/dev
    modules/R
