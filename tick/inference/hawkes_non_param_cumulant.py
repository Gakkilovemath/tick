import numpy as np
import scipy
from scipy.linalg import qr, sqrtm, norm

from tick.inference.base import LearnerHawkesNoParam
from tick.inference.nphc.cumulants import Cumulants


class NPHC(LearnerHawkesNoParam):
    """
    A class that implements non-parametric estimation described in th paper
    `Uncovering Causality from Multivariate Hawkes Integrated Cumulants` by
    Achab, Bacry, Gaiffas, Mastromatteo and Muzy (2016, Preprint).

    Parameters
    ----------
    half_width : `double`
        kernel support

    C : `float`, default=1e3
        Level of penalization

    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='none'
        The penalization to use. By default no penalization is used.
        Penalty is only applied to adjacency matrix.

    Attributes
    ----------
    L : list of `np.array` shape=(dim,)
        Estimated means

    C : list of `np.array` shape=(dim,dim)
        Estimated covariance

    K_c : list of `np.array` shape=(dim,dim)
        Estimated skewness (sliced)

    R : `np.array` shape=(dim,dim)
        Parameter of interest, linked to the integrals of Hawkes kernels

    Other Parameters
    ----------------
    alpha : `float`, default=`None`
        Ratio between skewness and covariance. The higher it is, the
        more covariance impacts the result which leads to symmetric
        adjacency matrices.
        If None, a default value is computed based on the norm of the
        estimated covariance and skewness cumulants.

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty
    """
    _attrinfos = {
        'cumul': {'writable': False},
        '_solver': {'writable': False},
        '_elastic_net_ratio': {'writable': False},
        'C_pen': {}, '_tf_feed_dict': {}, '_tf_graph': {},
    }

    def __init__(self, half_width, C=1e-3, penalty='none', solver='adam',
                 elastic_net_ratio=0.95,
                 tol=1e-5, verbose=False, max_iter=1000,
                 print_every=100, record_every=10,
                 step=1e-2, alpha=None):
        try:
            import tensorflow as tf
            self._tf_graph = tf.Graph()
        except ImportError:
            raise ImportError('`tensorflow` must be available to use NPHC')

        LearnerHawkesNoParam.__init__(
            self, tol=tol, verbose=verbose, max_iter=max_iter,
            print_every=print_every, record_every=record_every
        )

        self._elastic_net_ratio = None
        self.C_pen = C
        self.penalty = penalty
        self.elastic_net_ratio = elastic_net_ratio
        self.step = step
        self.alpha = alpha

        self.cumul = Cumulants(half_width=half_width)
        self._learner = self.cumul._cumulant
        self._solver = solver
        self._tf_feed_dict = None

        self.history.print_order = ["n_iter", "objective"]

    def _set_data(self, events):
        LearnerHawkesNoParam._set_data(self, events)
        self.cumul.realizations = events

    def _compute_cumulants(self):
        self.cumul.compute_cumulants()

        self.L = self.cumul.L.copy()
        self.C = self.cumul.C.copy()
        self.K_c = self.cumul.K_c.copy()

    def approximate_optimal_alpha(self):
        norm_sq_C = norm(self.C) ** 2
        norm_sq_K_c = norm(self.K_c) ** 2
        return norm_sq_K_c / (norm_sq_K_c + norm_sq_C)

    def objective(self, adjacency=None, R=None):
        """Compute objective value for a given adjacency or variable R

        Parameters
        ----------
        adjacency : `np.ndarray`, shape=(n_nodes, n_nodes), default=None
            Adjacency matrix at which we compute objective.
            If `None`, objective will be computed at `R`

        R : `np.ndarray`, shape=(n_nodes, n_nodes), default=None
            R variable at which objective is computed. Superseded by
            adjacency if adjacency is not `None`

        Returns
        -------
        Value of objective function
        """
        import tensorflow as tf
        cost = self._tf_objective_graph()
        L, C, K_c = self._tf_placeholders()

        if adjacency is not None:
            R = scipy.linalg.inv(np.eye(self.n_nodes) - adjacency)

        with self._tf_graph.as_default():

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self._tf_model_coeffs.assign(R))

                return sess.run(cost,
                                feed_dict={L: self.L, C: self.C, K_c: self.K_c})

    @property
    def _tf_model_coeffs(self):
        import tensorflow as tf

        with self._tf_graph.as_default():
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                return tf.get_variable("R", [self.n_nodes, self.n_nodes],
                                       dtype=tf.float64)

    @property
    def adjacency(self):
        return np.eye(self.n_nodes) - scipy.linalg.inv(self.solution)

    @property
    def baseline(self):
        return scipy.linalg.inv(self.solution).dot(self.L)

    def _tf_placeholders(self):
        import tensorflow as tf

        d = self.n_nodes
        if self._tf_feed_dict is None:
            with self._tf_graph.as_default():
                L = tf.placeholder(tf.float64, d, name='L')
                C = tf.placeholder(tf.float64, (d, d), name='C')
                K_c = tf.placeholder(tf.float64, (d, d), name='K_c')
                self._tf_feed_dict = L, C, K_c

        return self._tf_feed_dict

    def _tf_objective_graph(self):
        import tensorflow as tf
        d = self.n_nodes

        if self.alpha is None:
            alpha = self.approximate_optimal_alpha()
        else:
            alpha = self.alpha

        with self._tf_graph.as_default():
            L, C, K_c = self._tf_placeholders()
            R = self._tf_model_coeffs
            I = tf.constant(np.eye(d), dtype=tf.float64)

            # Construct model
            variable_covariance = \
                tf.matmul(R, tf.matmul(tf.diag(L), R, transpose_b=True))

            variable_skewness = \
                tf.matmul(C, tf.square(R), transpose_b=True) \
                + 2.0 * tf.matmul(R, R * C, transpose_b=True) \
                - 2.0 * tf.matmul(R, tf.matmul(
                    tf.diag(L), tf.square(R), transpose_b=True))

            covariance_divergence = tf.reduce_mean(
                tf.squared_difference(variable_covariance, C))

            skewness_divergence = tf.reduce_mean(
                tf.squared_difference(variable_skewness, K_c))

            cost = (1 - alpha) * skewness_divergence
            cost += alpha * covariance_divergence

            # Add potential regularization
            cost = tf.cast(cost, tf.float64)
            if self.strength_lasso > 0:
                reg_l1 = tf.contrib.layers.l1_regularizer(self.strength_lasso)
                cost += reg_l1((I - tf.matrix_inverse(R)))
            if self.strength_ridge > 0:
                reg_l2 = tf.contrib.layers.l2_regularizer(self.strength_ridge)
                cost += reg_l2((I - tf.matrix_inverse(R)))

            return cost

    def _solve(self, adjacency_start=None):
        """

        Parameters
        ----------
        adjacency_start : `str` or `np.ndarray, shape=(dim + dim * dim,), default=`None`
            Initial guess for the adjacency matrix. Will be used as 
            starting point in optimization.
            If `None`, a default starting point is estimated from the 
            estimated cumulants
            If `"random"`, as with `None`, a starting point is estimated from
            estimated cumulants with a bit a randomness

        max_iter : `int`
            The number of training epochs.

        step : `float`
            The learning rate used by the optimizer.

        solver : {'adam', 'momentum', 'adagrad', 'rmsprop', 'adadelta', 'gd'}, default='adam'
            Solver used to minimize the loss. As the loss is not convex, it
            cannot be optimized with `tick.optim.solver` solvers
        """
        import tensorflow as tf

        self._compute_cumulants()

        if adjacency_start is None or adjacency_start == 'random':
            random = adjacency_start == 'random'
            start_point = self.starting_point(random=random)
        else:
            start_point = scipy.linalg.inv(
                np.eye(self.n_nodes) - adjacency_start)

        cost = self._tf_objective_graph()
        L, C, K_c = self._tf_placeholders()

        # Launch the graph
        with self._tf_graph.as_default():
            solver = self.tf_solver(self.step).minimize(cost)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self._tf_model_coeffs.assign(start_point))
                # Training cycle
                for epoch in range(self.max_iter):

                    # We don't use self.objective here as it would be very slow
                    objective = sess.run(
                        cost, feed_dict={L: self.L, C: self.C, K_c: self.K_c})
                    self._handle_history(epoch, objective=objective)

                    sess.run(solver,
                             feed_dict={L: self.L, C: self.C, K_c: self.K_c})

                print("Optimization Finished!")

                self._set('solution', sess.run(self._tf_model_coeffs))

    def starting_point(self, random=False):
        sqrt_C = sqrtm(self.C)
        sqrt_L = np.sqrt(self.L)
        if random:
            random_matrix = np.random.rand(self.n_nodes, self.n_nodes)
            M, _ = qr(random_matrix)
        else:
            M = np.eye(self.n_nodes)
        initial = np.dot(np.dot(sqrt_C, M), np.diag(1. / sqrt_L))
        return initial


    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        available_solvers = ['momentum', 'adam', 'adagrad', 'rmsprop',
                             'adadelta', 'gd']
        if val.lower() not in available_solvers:
            raise ValueError('solver must be one of {}, recieved {}'
                             .format(available_solvers, val))

        self._set('_solver', val)

    @property
    def tf_solver(self):
        import tensorflow as tf

        if self.solver.lower() == 'momentum':
            return tf.train.MomentumOptimizer
        elif self.solver.lower() == 'adam':
            return tf.train.AdamOptimizer
        elif self.solver.lower() == 'adagrad':
            return tf.train.AdagradOptimizer
        elif self.solver.lower() == 'rmsprop':
            return tf.train.RMSPropOptimizer
        elif self.solver.lower() == 'adadelta':
            return tf.train.AdadeltaOptimizer
        elif self.solver.lower() == 'adadelta':
            return tf.train.GradientDescentOptimizer


    @property
    def elastic_net_ratio(self):
        return self._elastic_net_ratio

    @elastic_net_ratio.setter
    def elastic_net_ratio(self, val):
        if val < 0 or val > 1:
            raise ValueError("`elastic_net_ratio` must be between 0 and 1, "
                             "got %s" % str(val))
        else:
            self._set("_elastic_net_ratio", val)

    @property
    def strength_lasso(self):
        if self.penalty == 'elasticnet':
            return self.elastic_net_ratio / self.C_pen
        elif self.penalty == 'l1':
            return 1. / self.C_pen
        else:
            return 0.

    @property
    def strength_ridge(self):
        if self.penalty == 'elasticnet':
            return (1 - self.elastic_net_ratio) / self.C_pen
        elif self.penalty == 'l2':
            return 1. / self.C_pen
        return 0.

