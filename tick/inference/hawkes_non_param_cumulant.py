import numpy as np
from scipy.linalg import qr, sqrtm, norm

from tick.inference.base import LearnerHawkesNoParam
from tick.inference.nphc.cumulants import Cumulants


def starting_point(cumulants_list,random=False):
    L_list, C_list, K_c_list = cumulants_list
    d = len(L_list[0])
    sqrt_C = sqrtm(np.mean(C_list,axis=0))
    sqrt_L = np.sqrt(np.mean(L_list,axis=0))
    if random:
        M = random_orthogonal_matrix(d)
    else:
        M = np.eye(d)
    initial = np.dot(np.dot(sqrt_C,M),np.diag(1./sqrt_L))
    return initial

def random_orthogonal_matrix(dim):
    M = np.random.rand(dim**2).reshape(dim, dim)
    Q, _ = qr(M)
    return Q


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

    Attributes
    ----------
    L : list of `np.array` shape=(dim,)
        Estimated means

    C : list of `np.array` shape=(dim,dim)
        Estimated covariance

    K_c : list of `np.array` shape=(dim,dim)
        Estimated skewness (sliced)

    L_th : list of `np.array` shape=(dim,)
        Theoric means

    C_th : list of `np.array` shape=(dim,dim)
        Theoric covariance

    K_c_th : list of `np.array` shape=(dim,dim)
        Theoric skewness (sliced)

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
        'C_pen': {},
    }

    def __init__(self, half_width, C=1e-3, penalty='none', solver='adam',
                 elastic_net_ratio=0.95,
                 tol=1e-5, verbose=False, max_iter=1000,
                 print_every=100, record_every=10,
                 step=1e-2,
                 alpha=None, R_true=None, mu_true=None):

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

        self.cumul = Cumulants(half_width=half_width, mu_true=None,
                               R_true=None)
        self._learner = self.cumul._cumulant
        self._solver = solver
        self.R_true = R_true
        self.mu_true = mu_true

    def _set_data(self, events):
        LearnerHawkesNoParam._set_data(self, events)
        self.cumul.realizations = events

    def _compute_cumulants(self):
        self.cumul.compute_cumulants()

        self.L = self.cumul.L.copy()
        self.C = self.cumul.C.copy()
        self.K_c = self.cumul.K_c.copy()
        if self.R_true is not None and self.mu_true is not None:
            self.L_th = self.cumul.L_th
            self.C_th = self.cumul.C_th
            self.K_c_th = self.cumul.K_c_th
        else:
            self.L_th = None
            self.C_th = None
            self.K_c_th = None

    def approximate_optimal_alpha(self):
        norm_sq_C = norm(np.mean([C for C in self.C], axis=0)) ** 2
        norm_sq_K_c = norm(np.mean([K_c for K_c in self.K_c], axis=0)) ** 2
        return norm_sq_K_c / (norm_sq_K_c + norm_sq_C)

    def objective(self, coeffs, loss: float=None):
        raise NotImplementedError()

    def _solve(self, adjacency_start=None):
        """

        Parameters
        ----------
        adjacency_start : np.ndarray, shape=(dim + dim * dim,), default=`None`
            Initial guess for the adjacency matrix. Will be used as 
            starting point in optimization.
            If `None`, a default starting point is estimated from the 
            estimated cumulants

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

        if self.alpha is None:
            alpha = self.approximate_optimal_alpha()
        else:
            alpha = self.alpha

        cumulants_list = [self.L, self.C, self.K_c]
        d = len(self.L[0])
        if adjacency_start is None:
            start_point = starting_point(cumulants_list, random=False)
        else:
            start_point = adjacency_start.copy()

        R0 = tf.constant(start_point.astype(np.float64), shape=[d,d])
        L = tf.placeholder(tf.float64, d, name='L')
        C = tf.placeholder(tf.float64, (d,d), name='C')
        K_c = tf.placeholder(tf.float64, (d,d), name='K_c')

        R = tf.Variable(R0, name='R', dtype=tf.float64)
        I = tf.Variable(initial_value=np.eye(d), dtype=tf.float64)

        # Construct model
        activation_3 = tf.matmul(C,tf.square(R),transpose_b=True) + 2.0*tf.matmul(R,R*C,transpose_b=True) \
                       - 2.0*tf.matmul(R,tf.matmul(tf.diag(L),tf.square(R),transpose_b=True))
        activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))

        cost = (1 - alpha) * tf.reduce_mean(
            tf.squared_difference(activation_3, K_c)) \
               + alpha * tf.reduce_mean(
            tf.squared_difference(activation_2, C))

        reg_l1 = tf.contrib.layers.l1_regularizer(self.strength_lasso)
        reg_l2 = tf.contrib.layers.l2_regularizer(self.strength_ridge)

        if self.strength_ridge * self.strength_lasso > 0:
            cost = tf.cast(cost, tf.float64) + reg_l1((I - tf.matrix_inverse(R))) + reg_l2((I - tf.matrix_inverse(R)))
        elif self.strength_lasso > 0:
            cost = tf.cast(cost, tf.float64) + reg_l1((I - tf.matrix_inverse(R)))
        elif self.strength_ridge > 0:
            cost = tf.cast(cost, tf.float64) + reg_l2((I - tf.matrix_inverse(R)))
        else:
            cost = tf.cast(cost, tf.float64)

        # always use the average cumulants over all realizations
        L_avg = np.mean(self.L, axis=0)
        C_avg = np.mean(self.C, axis=0)
        K_avg = np.mean(self.K_c, axis=0)

        solver = self.tf_solver(self.step).minimize(cost)

        # Initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.max_iter):

                if epoch % self.print_every == 0:
                    avg_cost = np.average([sess.run(cost, feed_dict={L: L_, C: C_, K_c: K_c_})
                                           for (L_, C_, K_c_) in zip(self.L, self.C, self.K_c)])
                    print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))

                sess.run(solver, feed_dict={L: L_avg, C: C_avg, K_c: K_avg})

            print("Optimization Finished!")

            self._set('solution', sess.run(R))


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

