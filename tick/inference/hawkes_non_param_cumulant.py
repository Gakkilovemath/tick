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

    """
    _attrinfos = {
        'cumul': {'writable': False},
        '_solver': {'writable': False},
    }

    def __init__(self, half_width, solver='adam'):
        LearnerHawkesNoParam.__init__(self)
        self.cumul = Cumulants(half_width=half_width, mu_true=None,
                               R_true=None)
        self._learner = self.cumul._cumulant
        self._solver = solver

    def _set_data(self, events):
        LearnerHawkesNoParam._set_data(self, events)
        self.cumul.realizations = events

    def _compute_cumulants(self, R_true=None, mu_true=None):
        self.cumul.compute_cumulants()

        self.L = self.cumul.L.copy()
        self.C = self.cumul.C.copy()
        self.K_c = self.cumul.K_c.copy()
        if R_true is not None and mu_true is not None:
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

    def _solve(self, alpha=None, l_l1=0., l_l2=0., adjacency_start=None,
               max_iter=1000, step=1e6, solver='adam',
               display_step = 100, use_average=False, use_projection=False,
               projection_stable_G=False, positive_baselines=False, l_mu=0.):
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

        if use_projection:
            self.alpha = 0.
        elif alpha is None:
            self.alpha = self.approximate_optimal_alpha()
        else:
            self.alpha = alpha

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

        cost = (1 - self.alpha) * tf.reduce_mean(
            tf.squared_difference(activation_3, K_c)) \
               + self.alpha * tf.reduce_mean(
            tf.squared_difference(activation_2, C))

        reg_l1 = tf.contrib.layers.l1_regularizer(l_l1)
        reg_l2 = tf.contrib.layers.l2_regularizer(l_l2)

        if l_l2 * l_l1 > 0:
            cost = tf.cast(cost, tf.float64) + reg_l1((I - tf.matrix_inverse(R))) + reg_l2((I - tf.matrix_inverse(R)))
        elif l_l1 > 0:
            cost = tf.cast(cost, tf.float64) + reg_l1((I - tf.matrix_inverse(R)))
        elif l_l2 > 0:
            cost = tf.cast(cost, tf.float64) + reg_l2((I - tf.matrix_inverse(R)))
        else:
            cost = tf.cast(cost, tf.float64)

        # always use the average cumulants over all realizations
        if use_average or use_projection or projection_stable_G or positive_baselines:
            L_avg = np.mean(self.L, axis=0)
            C_avg = np.mean(self.C, axis=0)
            K_avg = np.mean(self.K_c, axis=0)
        if use_projection:
            L_avg_sqrt = np.sqrt(L_avg)
            L_avg_sqrt_inv = 1./L_avg_sqrt
            from scipy.linalg import inv, sqrtm
            C_avg_sqrt = sqrtm(C_avg)
            C_avg_sqrt_inv = inv(C_avg_sqrt)
        if projection_stable_G or positive_baselines:
            from scipy.linalg import inv
            C_avg_inv = inv(C_avg)

        if positive_baselines:
            #neg_baselines = - tf.matmul(tf.matmul(np.diag(L_avg),R,transpose_b=True),\
            #                            np.dot(C_avg_inv,L_avg.reshape(d,1)))
            neg_baselines = - tf.matmul(tf.matrix_inverse(R), L_avg.reshape(d,1))
            cost += l_mu * tf.reduce_sum(tf.nn.relu(neg_baselines))

        solver = self.tf_solver(step).minimize(cost)

        # Initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Set logs writer into folder /tmp/tf_cumul
            #summary_writer = tf.train.SummaryWriter('/tmp/tf_cumul', graph=sess.graph)

            # Training cycle
            for epoch in range(max_iter):

                if epoch % display_step == 0:
                    avg_cost = np.average([sess.run(cost, feed_dict={L: L_, C: C_, K_c: K_c_})
                                           for (L_, C_, K_c_) in zip(self.L, self.C, self.K_c)])
                    print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))

                if use_average:
                    sess.run(solver, feed_dict={L: L_avg, C: C_avg, K_c: K_avg})

                elif use_projection:
                    # Fit training using batch data
                    i = np.random.randint(0, len(self.data))
                    sess.run(solver, feed_dict={L: self.L[i], C: self.C[i], K_c: self.K_c[i]})
                    to_be_projected = np.dot(C_avg_sqrt_inv,np.dot(sess.run(R),np.diag(L_avg_sqrt)))
                    U, S, V = np.linalg.svd(to_be_projected)
                    R_projected = np.dot( C_avg_sqrt, np.dot( np.dot(U,V), np.diag(L_avg_sqrt_inv) ) )
                    assign_op = R.assign(R_projected)
                    sess.run(assign_op)
                else:
                    # Fit training using batch data
                    i = np.random.randint(0, len(self.data))
                    sess.run(solver, feed_dict={L: self.L[i], C: self.C[i], K_c: self.K_c[i]})

                if projection_stable_G:
                    to_be_projected = np.eye(d) - np.dot( np.dot(np.diag(L_avg), sess.run(tf.transpose(R))), C_avg_inv)
                    U, S, V = np.linalg.svd(to_be_projected)
                    S[S >= .99] = .99
                    G_projected = np.dot( U, np.dot(np.diag(S), V) )
                    R_projected = np.dot(C_avg, np.dot( np.eye(d) - G_projected.T, np.diag(1./L_avg) ) )
                    assign_op = R.assign(R_projected)
                    sess.run(assign_op)

                # Write logs at every iteration
                #summary_str = sess.run(merged_summary_op, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_c})
                #summary_writer.add_summary(summary_str, epoch)

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

