import numpy as np

import optimusprimal as opt
import darkmappy.logs as lg


class PrimalDual:
    """
    Class which handles all primal dual optimisation paradigms.
    """

    def __init__(
        self,
        data,
        phi,
        psi,
        options={
            "tol": 1e-5,
            "iter": 5000,
            "update_iter": 50,
            "record_iters": False,
            "positivity": False,
            "real": False,
            "nu": 0,
            "constrained": True,
        },
    ):
        """Construct primal dual general class.

        Any additional details should be here.

        Args:

            phi (): Measurement operator (weights for poisson noise)
            psi (): Redundant dictionary (wavelets etc.)
            beta (): (?)
            constrained (bool): Constrained vs unconstrained problem

        Raises:

            ValueError: Data vector contains NaN values.

        """
        if "viewer" not in options:
            self.viewer = lambda *args: None
        else:
            self.viewer = options["viewer"]
        self.options = options
        self.data = data
        self.phi = phi

        if "nu" not in self.options:
            self.nu = opt.linear_operators.power_method(
                phi, np.ones(phi.shape, dtype=complex)
            )[0]
        else:
            self.nu = self.options["nu"]

        if "mu" not in self.options:
            self.mu = opt.linear_operators.power_method(
                psi, np.ones(phi.shape, dtype=complex)
            )[0]
        else:
            self.mu = self.options["mu"]

        self.psi = psi
        self.f = None

        if self.options["real"]:
            self.f = opt.prox_operators.real_prox()

        if self.options["positivity"]:
            self.f = opt.prox_operators.positive_prox()

        self.constrained = self.options["constrained"]
        self.solver = (
            self.l1_constrained_gaussian
            if self.constrained
            else self.l1_unconstrained_gaussian
        )

        # Joint Map Estimation Variables
        self.jm_max_iter = 5
        self.kappa = 1
        self.eta = 1
        self.jm_tol = 1e-3

    def l1_constrained_gaussian(self, warm_start, sigma, beta=1e-2):
        """Solve constrained l1 regularisation problem with Gaussian noise.

        Can be instantiated from warm_start.

        Args:

            data (): Data-set to be optimised over.
            warm_start (): Initial solution of optimisation.
            sigma (): Noise-level present in optimisation.
            beta (): Scaling for l1-norm threshold


        Raises:

            ValueError: Datavector size is 0 (empty set).
            ValueError: Datavector contains NaN values.

        """
        size = len(np.ravel(self.data))
        if size == 0:
            raise ValueError("Data vector is the empty set!")
        if np.any(np.isnan(self.data)):
            raise ValueError("Data vector contains NaN values!")

        epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma
        p = opt.prox_operators.l2_ball(epsilon, self.data, self.phi)
        p.beta = self.nu
        h = opt.prox_operators.l1_norm(
            np.max(np.abs(self.psi.dir_op(self.phi.adj_op(self.data))))
            * beta
            * self.psi.weights,
            self.psi,
        )
        h.beta = self.mu
        return opt.primal_dual.FBPD(
            warm_start, self.options, None, self.f, h, p, viewer=self.viewer
        )

    def l1_unconstrained_gaussian(self, warm_start, sigma, beta):
        """Solve unconstrained l1 regularisation problem with Gaussian noise.

        Can be instantiated from warm_start.

        Args:

            data (): Data-set to be optimised over.
            warm_start (): Initial solution of optimisation.
            sigma (): Noise-level present in optimisation.
            beta (): Regularisation parameter

        Raises:

            ValueError: Datavector size is 0 (empty set).
            ValueError: Datavector contains NaN values.

        """
        if len(np.ravel(self.data)) == 0:
            raise ValueError("Data vector is the empty set!")

        if np.any(np.isnan(self.data)):
            raise ValueError("Data vector contains NaN values!")

        g = opt.grad_operators.l2_norm(sigma, self.data, self.phi)
        g.beta = self.nu / sigma ** 2
        h = (
            None
            if (beta <= 0)
            else opt.prox_operators.l1_norm(beta * self.psi.weights, self.psi)
        )
        h.beta = self.mu
        return opt.primal_dual.FBPD(
            warm_start, self.options, g, self.f, h, viewer=self.viewer
        )

    def l1_unconstrained_gaussian_jm(self, warm_start, sigma, beta):
        """Solve unconstrained l1 regularisation problem with Gaussian noise.

        Can be instantiated from warm_start.

        Args:

            data (): Data-set to be optimised over.
            warm_start (): Initial solution of optimisation.
            sigma (): Noise-level present in optimisation.
            beta (): Regularisation parameter

        Raises:

            ValueError: Datavector size is 0 (empty set).
            ValueError: Datavector contains NaN values.

        """
        if len(np.ravel(self.data)) == 0:
            raise ValueError("Data vector is the empty set!")

        if np.any(np.isnan(self.data)):
            raise ValueError("Data vector contains NaN values!")

        g = opt.grad_operators.l2_norm(sigma, self.data, self.phi)
        g.beta = self.nu / sigma ** 2
        h = (
            None
            if (beta <= 0)
            else opt.prox_operators.l1_norm(beta * self.psi.weights, self.psi)
        )
        h.beta = self.mu
        y = h.dir_op(warm_start) * 0.0
        z = warm_start * 0
        w = warm_start * 0
        sol, diagnostics = opt.primal_dual.FBPD_warm_start(
            warm_start,
            y,
            z,
            w,
            self.options,
            g=g,
            f=self.f,
            h=h,
            p=None,
            r=None,
            viewer=self.viewer,
        )
        for it in range(1, self.jm_max_iter):
            beta_old = beta
            beta = (self.eta + len(y.flatten())) / (
                self.kappa + h.fun(h.dir_op(sol)) / beta
            )
            lg.info_log(
                "[JM] %d out of %d iterations, tol = %f",
                it,
                self.jm_max_iter,
                np.linalg.norm(beta - beta_old) / np.linalg.norm(beta_old),
            )
            lg.info_log("[JM] regularisation is %f", beta)
            if (
                np.linalg.norm(beta - beta_old) < self.jm_tol * np.linalg.norm(beta_old)
                and it > 10
            ):
                lg.info_log("[JM] converged in %d iterations", it)
                break

            h = (
                None
                if (beta <= 0)
                else opt.prox_operators.l1_norm(beta * self.psi.weights, self.psi)
            )
            h.beta = self.mu
            y = diagnostics["y"]
            z = diagnostics["z"]
            w = diagnostics["w"]
            sol, diagnostics = opt.primal_dual.FBPD_warm_start(
                sol,
                y,
                z,
                w,
                self.options,
                g=g,
                f=self.f,
                h=h,
                p=None,
                r=None,
                viewer=self.viewer,
            )

        return sol, diagnostics
