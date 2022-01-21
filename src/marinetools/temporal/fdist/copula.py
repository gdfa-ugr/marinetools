import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from scipy.stats import kendalltau, pearsonr, scoreatpercentile, spearmanr
from sklearn.neighbors import KernelDensity


class Copula:
    """
    This class estimates the copula parameter for the joint random variable. Copulas of Clayton, Frank y Gumbel are available.
    The base file comes from the ambhas 0.4.0 module of Sat Kumar Tomer, http://civil.iisc.ernet.in/~satkumar/,
    satkumartomer@gmail.com'

    Some modifications are:
        1. Conditional copulas included for that families
        2. The package statistics is replaced by sklearn.neighbors.KernelDensity
    """

    def __init__(self, X, Y, family, *args):
        """It is started the class with the variables X and Y

        Args:
            - X (np.ndarray): data of the first variable
            - Y (np.ndarray): data of the second variable
            - family (string): clayton, frank o gumbel

        Note:
            - The size of X and Y should be equal
        """
        # se chequean las dimensiones
        if not ((X.ndim == 1) and (Y.ndim == 1)):
            raise ValueError("The dimension should be one.")

        # se chequean las longitudes de ambos vectores
        if X.size != Y.size:
            raise ValueError("El tamano de ambos vectores debe ser el mismo.")

        # se chequean que la familia es correcta
        copula_family = ["clayton", "frank", "gumbel"]
        if family not in copula_family:
            raise ValueError("Copulas available are clayton, frank or gumbel.")

        self.X = X
        self.Y = Y
        self.family = family

        # si se tienen marginales teoricas
        if args:
            self.F1, self.F2 = args[0][0], args[0][2]
            self.p1, self.p2 = args[0][1], args[0][3]

        # se estiman el coeficiente de correlacion de Kendall
        tau = kendalltau(self.X, self.Y)[0]
        self.tau = tau

        # se estima el R de pearson y R de spearman
        self.pr = pearsonr(self.X, self.Y)
        self.sr = spearmanr(self.X, self.Y)

        # se estima el parametro de la copula
        self._get_parameter()

        # se establece U y V a None
        self.U = None
        self.V = None

    def _get_parameter(self):
        """The parameter theta is estimated"""

        if self.family == "clayton":
            self.theta = 2 * self.tau / (1 - self.tau)

        elif self.family == "frank":
            self.theta = -fmin(self._frank_fun, -5, disp=False)[0]

        elif self.family == "gumbel":
            self.theta = 1 / (1 - self.tau)

    def generate_uv(self, n=1000):
        """Random variables u, v are generated

        Args:
            - n (int): number of elements of the copula. Defaults to 1000.
        """
        # copula de clayton
        if self.family == "clayton":
            U = np.random.uniform(size=n)
            W = np.random.uniform(size=n)

            if self.theta <= -1:
                raise ValueError(
                    "the parameter for clayton copula should be more than -1"
                )
            elif self.theta == 0:
                raise ValueError("The parameter for clayton copula should not be 0")

            if self.theta < sys.float_info.epsilon:
                V = W
            else:
                V = U * (
                    W ** (-self.theta / (1 + self.theta)) - 1 + U ** self.theta
                ) ** (-1 / self.theta)

        # copula de frank
        elif self.family == "frank":
            U = np.random.uniform(size=n)
            W = np.random.uniform(size=n)

            if self.theta == 0:
                raise ValueError("The parameter for frank copula should not be 0")
            if abs(self.theta) > np.log(sys.float_info.max):
                V = (U < 0) + np.sign(self.theta) * U
            elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                V = (
                    -np.log(
                        (np.exp(-self.theta * U) * (1 - W) / W + np.exp(-self.theta))
                        / (1 + np.exp(-self.theta * U) * (1 - W) / W)
                    )
                    / self.theta
                )
            else:
                V = W

        # coupla de gumbel
        elif self.family == "gumbel":
            if self.theta <= 1:
                raise ValueError(
                    "El parametro de la copula de gumbel debe ser mayor que uno."
                )
            if self.theta < 1 + sys.float_info.epsilon:
                U = np.random.uniform(size=n)
                V = np.random.uniform(size=n)
            else:
                u = np.random.uniform(size=n)
                w = np.random.uniform(size=n)
                w1 = np.random.uniform(size=n)
                w2 = np.random.uniform(size=n)

                u = (u - 0.5) * np.pi
                u2 = u + np.pi / 2
                e = -np.log(w)
                t = np.cos(u - u2 / self.theta) / e
                gamma = (
                    (np.sin(u2 / self.theta) / t) ** (1 / self.theta) * t / np.cos(u)
                )
                s1 = (-np.log(w1)) ** (1 / self.theta) / gamma
                s2 = (-np.log(w2)) ** (1 / self.theta) / gamma
                U = np.array(np.exp(-s1))
                V = np.array(np.exp(-s2))

        self.U = U
        self.V = V
        return

    def generate_cond(self):
        """Generates the values of V given U using the conditional copula"""

        if isinstance(self.U, float):
            n = 1
        else:
            n = len(self.U)

        W = np.random.uniform(size=n)

        # copula de clayton
        if self.family == "clayton":
            if self.theta <= -1:
                raise ValueError(
                    "the parameter for clayton copula should be more than -1"
                )
            elif self.theta == 0:
                raise ValueError("The parameter for clayton copula should not be 0")

            if self.theta < sys.float_info.epsilon:
                V = W
            else:
                V = (
                    self.U ** -self.theta
                    * (self.U ** -self.theta + W ** -self.theta - 1)
                    ** (-1 / self.theta)
                ) / (self.U * (self.U ** -self.theta + W ** -self.theta - 1))

        # copula de frank
        elif self.family == "frank":
            if self.theta == 0:
                raise ValueError("The parameter for frank copula should not be 0")

            if abs(self.theta) > np.log(sys.float_info.max):
                V = (self.U < 0) + np.sign(self.theta) * self.U
            elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                tU, tW = (self.theta + 0j) ** self.U, (self.theta + 0j) ** W
                V = (
                    -tU
                    * (tW - 1)
                    / ((1 + (tU - 1) * (tW - 1) / (self.theta - 1)) * (self.theta - 1))
                )
            else:
                V = W

        # copula de gumbel
        elif self.family == "gumbel":
            if self.theta <= 1:
                raise ValueError(
                    "El parametro de la copula gumbel debe ser mayor que 1."
                )
            if self.theta < 1 + sys.float_info.epsilon:
                V = np.random.uniform(size=len(self.U))
            else:
                V = (
                    -np.log(self.U) ** self.theta
                    * ((-np.log(self.U)) ** self.theta + (-np.log(W)) ** self.theta)
                    ** (1 / self.theta)
                    * (
                        np.exp(
                            -(
                                (
                                    (-np.log(self.U)) ** self.theta
                                    + (-np.log(W)) * self.theta
                                )
                                ** (1 / self.theta)
                            )
                        )
                    )
                    / (
                        self.U
                        * np.log(self.U)
                        * ((-np.log(self.U)) ** self.theta + (-np.log(W)) ** self.theta)
                    )
                )

        self.V = np.absolute(V)
        return

    def generate_xy(self, n=0):
        """Generates a random vector of length n for variables x and y when U is not given and
        a vector of conditional V to U otherwise

        Args:
            * n (int, optional): no of elements. Defaults to 0.
        """
        # estimate inverse cdf of x and y

        # if U and V are not already generated
        if self.U is None:
            self.generate_uv(n)
        elif n != 0:
            self.generate_cond()

        if hasattr(self, "p1"):
            X1 = self.F1.ppf(self.U, *self.p1)
            Y1 = self.F2.ppf(self.V, *self.p2)
        else:
            # if not hasattr(self, '_inv_cdf_x'):
            self._inverse_cdf()
            X1 = self._inv_cdf_x(self.U)
            Y1 = self._inv_cdf_y(self.V)
        self.X1 = X1
        self.Y1 = Y1

        # se establece U y V a None
        self.U = None
        self.V = None

        return

    def generate_C(self, u, v):
        """Generates a random vector of length n for variables x and y if U is not given and
        a vector of conditional random variables to U otherwise.

        Args:
            * u (np.array): vector u (cdf of x)
            * v (np.array): vector v (cdf of y)
        """
        uq, vq = np.meshgrid(u, v)

        if self.family == "clayton":
            self.C = np.zeros(np.shape(uq))
            mask = (uq ** -self.theta + vq ** -self.theta - 1) ** -(1 / self.theta) > 0
            self.C[mask] = (uq[mask] ** -self.theta + vq[mask] ** -self.theta - 1) ** -(
                1 / self.theta
            )
        elif self.family == "frank":
            self.C = (
                -1
                / self.theta
                * np.log(
                    1
                    + (np.exp(-self.theta * uq) - 1)
                    * (np.exp(-self.theta ** vq) - 1)
                    / (np.exp(-self.theta) - 1)
                )
            )
        elif self.family == "gumbel":
            self.C = np.exp(
                -(
                    ((-np.log(uq)) ** self.theta + (-np.log(vq)) ** self.theta)
                    ** (1 / self.theta)
                )
            )

        return

    def estimate(self, data=None):
        """Estimates the mean, std, iqr for the generated ensemble

        Args:
            * data (np.array, optional): X variable. Defaults to None.

        Returns:
            * Y1_mean (np.array): mean of the simulated ensemble
            * Y1_std (np.array): std of the simulated ensemble
            * Y1_ll (np.array): lower limit of the simulated ensemble
            * Y1_ul (np.array): upper limit of the simulated ensemble
        """
        nbin = 50
        # se chequea si ya se ha llamado a generate_xy, si no, se hace ahora
        try:
            self.X1
            copula_ens = len(self.X1)
        except:
            copula_ens = 10000
            self.generate_xy(copula_ens)

        if data is None:
            data = self.X

        n_ens = copula_ens / nbin
        ind_sort = self.X1.argsort()
        x_mean = np.zeros((nbin,))
        y_mean = np.zeros((nbin,))
        y_ul = np.zeros((nbin,))
        y_ll = np.zeros((nbin,))
        y_std = np.zeros((nbin,))

        for ii in range(nbin):
            x_mean[ii] = self.X1[ind_sort[n_ens * ii : n_ens * (ii + 1)]].mean()
            y_mean[ii] = self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]].mean()
            y_std[ii] = self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]].std()
            y_ll[ii] = scoreatpercentile(
                self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]], 25
            )
            y_ul[ii] = scoreatpercentile(
                self.Y1[ind_sort[n_ens * ii : n_ens * (ii + 1)]], 75
            )

        foo_mean = interp1d(x_mean, y_mean, bounds_error=False)
        foo_std = interp1d(x_mean, y_std, bounds_error=False)
        foo_ll = interp1d(x_mean, y_ll, bounds_error=False)
        foo_ul = interp1d(x_mean, y_ul, bounds_error=False)

        Y1_mean = foo_mean(data)
        Y1_std = foo_std(data)
        Y1_ll = foo_ll(data)
        Y1_ul = foo_ul(data)

        return Y1_mean, Y1_std, Y1_ll, Y1_ul

    def _inverse_cdf(self):
        """Inverse of the cdf given the joint X and Y por U and V. An Epanechikov kernel is used interpolate data and a linear function
        is used to extrapolate the data outside the range of the data. This module return the interpolator.
        """

        x1 = np.unique(self.X)
        x2 = np.cumsum(
            np.exp(
                KernelDensity(kernel="epanechnikov", bandwidth=0.1)
                .fit(self.X[:, np.newaxis])
                .score_samples(x1[:, np.newaxis])
            )
        )
        x2 = x2 / np.max(x2)

        self._inv_cdf_x = interp1d(
            x2, x1, bounds_error=False, fill_value="extrapolate", kind="linear"
        )

        y1 = np.unique(self.Y)
        y2 = np.cumsum(
            np.exp(
                KernelDensity(kernel="epanechnikov", bandwidth=0.1)
                .fit(self.Y[:, np.newaxis])
                .score_samples(y1[:, np.newaxis])
            )
        )
        y2 = y2 / np.max(y2)

        self._inv_cdf_y = interp1d(
            y2, y1, bounds_error=False, fill_value="extrapolate", kind="linear"
        )
        return

    def _integrand_debye(self, t):
        """Integrands of debye function of first order"""
        return t / (np.exp(t) - 1)

    def _debye(self, alpha):
        """Debye function of first order

        Args:
            * alpha (float): initial guess
        """
        return quad(self._integrand_debye, sys.float_info.epsilon, alpha)[0] / alpha

    def _frank_fun(self, alpha):
        """Optimizes the parameter of the Frank copula

        Args:
            * alpha (float): initial guess
        """
        diff = (1 - self.tau) / 4.0 - (self._debye(-alpha) - 1) / alpha
        return diff ** 2
