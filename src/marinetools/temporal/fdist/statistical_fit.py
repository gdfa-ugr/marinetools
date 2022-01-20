import warnings
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.utils import auxiliar, read
from matplotlib.pyplot import show
from scipy.integrate import quad
from scipy.optimize import differential_evolution, dual_annealing, minimize, shgo

warnings.filterwarnings("ignore")

"""This file is part of MarineTools.

MarineTools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MarineTools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MarineTools.  If not, see <https://www.gnu.org/licenses/>.
"""


def st_analysis(df, param):
    """Fits stationary simple or mixed models

    Args:
        - df (pd.DataFrame): raw time series
        - param (dict): guess parameters for the analysis. More detailed information in 'analysis'.

    Returns:
        * The parameters and the mode of the fit

    """
    par0 = []
    percentiles = np.hstack([0, np.cumsum(param["ws_ps"])])

    df_sort = df[param["var"]].sort_values(ascending=True)
    p = np.linspace(0, 1, len(df_sort))

    if (param["basis_function"]["order"] == 0) and (param["reduction"] == False):
        for i in param["fun"].keys():
            filtro = (p >= percentiles[i]) & (p <= percentiles[i + 1])
            par = param["fun"][i].fit(df_sort[filtro])
            par0 = np.hstack([par0, par]).tolist()
    elif param["piecewise"]:
        percentiles = np.hstack([0, param["ws_ps"], 1])
        df = emp(df, percentiles, param["var"])
        for i in param["fun"].keys():
            filtro = (df[param["var"]] >= df["u" + str(int(i))]) & (
                df[param["var"]] <= df["u" + str(i + 1)]
            )
            par = best_params(df.loc[filtro, param["var"]], 25, param["fun"][i])
            par0 = np.hstack([par0, par]).tolist()

        if not param["fix_percentiles"]:
            par0 = np.hstack([par0, st.norm.ppf(param["ws_ps"])])
    else:
        if param["reduction"]:
            par1 = param["fun"][1].fit(
                df.loc[((p >= percentiles[1]) & (p <= percentiles[2])), param["var"]]
            )

            par2 = -0.05
            par0 = [
                st.norm.ppf(param["ws_ps"][0]),
                st.norm.ppf(param["ws_ps"][-1]),
            ]
            par0 = np.hstack([par1, par2, par0]).tolist()
        else:
            if param["no_fun"] == 1:
                par0 = param["fun"][0].fit(df[param["var"]])
            else:
                percentiles = np.hstack([0, param["ws_ps"], 1])
                df = emp(df, percentiles, param["var"])

                if param["no_fun"] == 2:
                    ibody = df[param["var"]] <= df["u1"]
                    iutail = df[param["var"]] > df["u1"]

                    parb = param["fun"][0].fit(df.loc[ibody, param["var"]])
                    part = param["fun"][1].fit(
                        df.loc[iutail, param["var"]] - df.loc[iutail, "u1"]
                    )
                    par0 = np.hstack([par0, parb, part]).tolist()
                    if not param["fix_percentiles"]:
                        par0 = np.hstack([par0, st.norm.ppf(param["ws_ps"])])

                else:
                    # ----------------------------------------------------------------------
                    # With three PMs it is required that thresholds be related to the data
                    # ----------------------------------------------------------------------
                    iltail = df[param["var"]] < df["u1"]
                    ibody = (df[param["var"]] >= df["u1"]) & (
                        df[param["var"]] <= df["u2"]
                    )
                    iutail = df[param["var"]] > df["u2"]
                    parl = param["fun"][0].fit(
                        df.loc[iltail, "u1"] - df.loc[iltail, param["var"]]
                    )
                    parb = param["fun"][1].fit(df.loc[ibody, param["var"]])
                    part = param["fun"][2].fit(
                        df.loc[iutail, param["var"]] - df.loc[iutail, "u2"]
                    )
                    par0 = np.hstack([par0, parl, parb, part]).tolist()

                    if not param["fix_percentiles"]:
                        par0 = np.hstack([par0, st.norm.ppf(param["ws_ps"])])

    if param[
        "guess"
    ]:  # checked outside because some previous parameters are required later
        print("Initial guess computed: " + str(par0))
        par0 = param["p0"]
        print("Initial guess given: " + str(par0))
    else:
        print("Initial guess computed: " + str(par0))
    mode = np.zeros(param["no_fun"], dtype=int).tolist()

    if (not param["guess"]) & (any(np.abs(par0) > 20)):
        warnings.warn(
            "Parameters of the initial guess are high. The convergence is not ensured."
            + "It is recommended: (i) modify the percentiles, or (ii) increase the"
            + "bounds."
        )

    return df, par0, mode


def best_params(data, bins, distrib, tail=False):
    """Computes the best parameters of a simple probability model attending to the rmse of the pdf

    Args:
        * data (pd.DataFrame): raw time series
        * bins (int): no. of bins for the histogram
        * distrib (string): name of the probability model
        * tail (bool, optional): If it is fit a tail or not. Defaults to False.

    Returns:
        * params (list): the estimated parameters
    """

    dif_, sser = 1e2, 1e3
    nlen = int(len(data) / 200)

    data = data.sort_values(ascending=True).values
    while (dif_ > 1) & (sser > 30) & (0.95 * nlen < len(data)):
        results = fit_(data, bins, distrib)
        sse, params = results[0], results[1:]
        dif_, sser = np.abs(sser - sse), sse

        if tail:
            data = data[int(nlen / 4) :]
        else:
            data = data[nlen:-nlen]
    return params


def fit_(data, bins, model):
    """Fits a simple probability model and computes the sse with the empirical pdf

    Args:
        * data (pd.DataFrame): raw time series
        * bins (int): no. of bins for the histogram
        * model (string): name of the probability model

    Returns:
        * results (np.array): the parameters computed
    """

    y, x = np.histogram(data, bins=bins, density=True)
    xq = np.diff(x)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    yq = np.cumsum(xq * y)

    if model is st.genpareto:
        params = model.fit(data, 0.01, loc=np.min(data))
    else:
        params = model.fit(data)
    cdf = model.cdf(x, loc=params[-2], scale=params[-1], *params[:-2])
    sse = np.sum((yq - cdf) ** 2)
    if np.isnan(sse):
        sse = 1e10

    results = np.zeros(len(params) + 1)
    results[: int(len(params) + 1)] = np.append(sse, params)
    return results


def nonst_analysis(df, param):
    """Makes a non-stationary analysis for several modes

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): guess parameters of the model

    Returns:
        * param (dict): parameters of the model
    """

    par, bic, nllf = nonst_fit(df, param)
    if not any(param["mode"]):
        if param["basis_function"]["order"] > 1:
            if param["bic"]:
                mode = min(bic, key=bic.get)
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
            else:
                mode = min(nllf, key=nllf.get)
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
        elif param["basis_function"]["order"] == 1:
            if param["bic"]:
                mode = list(bic.keys())[0]
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
            else:
                mode = list(nllf.keys())[0]
                param["par"] = list(par[mode])
                param["mode"] = [int(i) for i in mode]
        else:
            mode = np.zeros(param["no_fun"], dtype=int)
            param["par"] = list(par)
            param["mode"] = [int(i) for i in mode]

        all_ = []
        for i in bic.keys():
            all_.append([str(i), bic[i], list(par[i])])

        param["all"] = all_
    else:
        param["par"] = list(par)
        param["all"] = [str(param["mode"]), bic, list(par)]

    return param


def nonst_fit(df, param):
    """Fits a non-stationary probability model

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): the guess parameters of the model

    Returns:
        * par (list): the best parameters
        * bic (float): the Bayesian information criteria
    """

    par, bic = {}, {}
    # ----------------------------------------------------------------------------------
    # Check if a mode is not given. Run the general analysis
    # ----------------------------------------------------------------------------------
    if not any(param["mode"]):

        if param["basis_function"]["order"] >= 1:
            par, nllf, mode = fourier_expansion(df, par, param)

            logndat = np.log(len(df[param["var"]]))
            for i in mode:
                bic[tuple(i)] = 2 * nllf[tuple(i)] + logndat * (len(par[tuple(i)]))
        else:
            par = param["par"]
            bic[0] = 0
    else:
        # ------------------------------------------------------------------------------
        # Check if a mode is given. Optimize a specific mode (and parameters)
        # ------------------------------------------------------------------------------
        nllf = 1e9
        par, nllf, _ = fourier_expansion(df, param["par"], nllf, param)
        bic = 2 * nllf + np.log(len(df[param["var"]])) * (len(par))

    return par, bic, nllf


def fourier_expansion(data, par, param):
    """Prepares the initial guess and estimates the n-order non-stationary parameters

    Args:
        * data (pd.DataFrame): raw data
        * par (dict): the guess parameters of the probability model
        * llf (dict): log-likelihood value of the first order
        * param (dict): the parameters of the analysis

    Returns:
        * par (dict): parameters of the first order
        * llf (dict): log-likelihood value of the first order
        * mode (dict): parameter of the first order
    """
    nllf = {}
    if not any(param["mode"]):
        mode = []
        if param["no_fun"] == 1:
            for i in range(1, param["basis_function"]["order"] + 1):
                mode.append((i,))
        elif param["no_fun"] == 2:
            for i in range(1, param["basis_function"]["order"] + 1):
                mode.append((i, 1))
            for i in range(2, param["basis_function"]["order"] + 1):
                mode.append((param["basis_function"]["order"], i))
        else:
            for i in range(1, param["basis_function"]["order"] + 1):
                mode.append((1, i, 1))
            for i in range(2, param["basis_function"]["order"] + 1):
                mode.append((i, param["basis_function"]["order"], 1))
            for i in range(2, param["basis_function"]["order"] + 1):
                mode.append(
                    (
                        param["basis_function"]["order"],
                        param["basis_function"]["order"],
                        i,
                    )
                )

        ind_ = np.diff(np.vstack([mode[0], mode]), axis=0)

        ind_[ind_ < 0] = 0
        par0, pos, comp_ = {}, [], []
        no_1, no_2 = [0], [0]
        # ------------------------------------------------------------------------------
        # Compute the position after which the parameters will be expanded
        # ------------------------------------------------------------------------------
        for ind_mode, imode in enumerate(mode):
            no_prob_model = np.where(ind_[ind_mode] == 1)[0]

            if not no_prob_model.size == 0:
                pos.append(no_prob_model[0])
            else:
                pos.append(0)

            if not ind_mode:
                nllf[imode] = 1e10
                par[imode] = param["par"]
                comp_.append(None)
            else:
                if no_prob_model == 1:
                    comp_.append(mode[ind_mode - 1])
                elif no_prob_model == 0:
                    comp_.append(mode[no_1[-1]])
                    no_1.append(ind_mode)
                else:
                    comp_.append(mode[no_2[-1]])
                    no_2.append(ind_mode)
                    no_1 = [ind_mode]

        # ------------------------------------------------------------------------------
        # Expand in Fourier Series and fit the parameters
        # ------------------------------------------------------------------------------
        for ind_mode, imode in enumerate(mode):
            par0[imode] = initial_params(
                param, par, pos[ind_mode], imode, comp_[ind_mode]
            )

            print("Mode " + str(imode) + " non-stationary")
            if ind_mode == 0:
                par[imode], nllf[imode] = fit(
                    data, param, par0[imode], imode, nllf[imode]
                )
            else:
                par[imode], nllf[imode] = fit(
                    data, param, par0[imode], imode, nllf[comp_[ind_mode]]
                )
    else:
        mode = [tuple(param["mode"])]
        print("Mode " + str(mode[0]) + " non-stationary")
        par, nllf = fit(data, param, par, mode[0], nllf)

    return par, nllf, mode[1:]


def initial_params(param, par, pos, imode, comp):
    """Prepares the parameters from previous fit to the following

    Args:
        * param (dict): the parameters of the analysis
        * par (dict): the guess parameters of the probability model
        * pos (int): location to place the new terms for the fit
        * imode (list): combination of modes for fitting
        * comp (list): components of the current mode

    Returns:
        * par0 (list): initial guess parameters
    """

    if bool(
        [
            meth
            for meth in ["trigonometric", "modified"]
            if (meth in param["basis_function"]["method"])
        ]
    ):
        add_ = [np.random.rand(1)[0] * 1e-5, np.random.rand(1)[0] * 1e-5]
        npars_ = 2
    else:
        add_ = np.random.rand(1)[0] * 1e-5
        npars_ = 1

    if comp is None:
        comp = imode
        par0 = par[comp]
        for i in range(param["no_tot_param"]):
            par0 = np.insert(par0, i * (npars_ + 1) + 1, add_,)

    elif param["reduction"]:
        if pos == 0:
            loc = npars_ * (imode[0] - 1) + 1
            par0 = np.insert(par[comp], loc, add_,)
            for _ in range(param["no_param"][0] - 1):
                loc = loc + npars_ * (imode[0] - 1) + (npars_ + 1)
                par0 = np.insert(par0, loc, add_,)
        else:
            loc = (
                param["no_param"][0] * (imode[0] * npars_ + 1)
                + npars_ * (imode[1] - 1)
                + 1
            )
            par0 = np.insert(par[comp], loc, add_,)
    else:
        par0 = par[comp]
        loc = 0
        for j in range(pos):
            loc += (npars_ * imode[j] + 1) * param["no_param"][j]

        for i in range(param["no_param"][pos]):
            loc = loc + npars_ * (imode[pos] - 1) + 1
            par0 = np.insert(par0, loc, add_,)
            loc += npars_
    return par0


def matching_lower_bound(par):
    """Matching conditions between two probability models (PMs). Lower refers to the
    low tail-body PMs in the case of fitting three PMs.

    Args:
        par (dict): parameters of the usual dictionary format

    Returns:
        [type]: [description]
    """

    # ----------------------------------------------------------------------------------
    # Obtaining the parameters
    # ----------------------------------------------------------------------------------
    t_expans = params_t_expansion(
        mode, param, df.sort_values(by="n").drop_duplicates(subset=["n"]).loc[:, "n"]
    )
    df_, _ = get_params(
        df.sort_values(by="n").drop_duplicates(subset=["n"]),
        param,
        par,
        mode,
        t_expans,
    )
    # ----------------------------------------------------------------------------------
    # Applying the restrictions along "n"
    # ----------------------------------------------------------------------------------
    if not param["reduction"]:
        # ------------------------------------------------------------------------------
        # Using two PMs, the body PM is the first one and the second PM is used as upper
        # tail model. Using three PMS, the body PM is the center one.
        # ------------------------------------------------------------------------------
        if len(df_) == 2:
            f_body, f_tail = 0, 1
        else:
            f_body, f_tail = 1, 0

        if len(df_) == 2:
            if param["no_param"][f_body] == 2:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
            else:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )

            if param["no_param"][f_tail] == 2:
                ft_u1 = param["fun"][f_tail].pdf(0, df_[f_tail]["s"], df_[f_tail]["l"])
            else:
                ft_u1 = param["fun"][f_tail].pdf(
                    0, df_[f_tail]["s"], df_[f_tail]["l"], df_[f_tail]["e"],
                )
        else:
            if param["no_param"][f_body] == 2:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"], df_[f_body]["s"], df_[f_body]["l"]
                )
            else:
                fc_u1 = param["fun"][f_body].pdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )
                Fc_u1 = param["fun"][f_body].cdf(
                    df_[f_body]["u1"],
                    df_[f_body]["s"],
                    df_[f_body]["l"],
                    df_[f_body]["e"],
                )

            if param["no_param"][f_tail] == 2:
                ft_u1 = param["fun"][f_tail].pdf(0, df_[f_tail]["s"], df_[f_tail]["l"])
            else:
                ft_u1 = param["fun"][f_tail].pdf(
                    0, df_[f_tail]["s"], df_[f_tail]["l"], df_[f_tail]["e"]
                )

    constraints_ = np.sqrt(1 / len(ft_u1) * np.sum((fc_u1 - Fc_u1 * ft_u1) ** 2))

    return constraints_


def matching_upper_bound(par):
    """Matching conditions between two probability models (PMs). Upper refers to the
    low tail-body PMs for fitting three PMs.

    Args:
        par (dict): parameters of the usual dictionary format

    Returns:
        [type]: [description]
    """
    # ----------------------------------------------------------------------------------
    # Obtaining the parameters
    # ----------------------------------------------------------------------------------
    t_expans = params_t_expansion(
        mode, param, df.sort_values(by="n").drop_duplicates(subset=["n"]).loc[:, "n"]
    )
    df_, _ = get_params(
        df.sort_values(by="n").drop_duplicates(subset=["n"]),
        param,
        par,
        mode,
        t_expans,
    )

    # ----------------------------------------------------------------------------------
    # Applying the restrictions along "n"
    # ----------------------------------------------------------------------------------
    if not param["reduction"]:
        # ------------------------------------------------------------------------------
        # Using two PMs, the body PM is the first one and the second PM is used as upper
        # tail model. Using three PMS, the body PM is the center one.
        # ------------------------------------------------------------------------------
        f_body, f_tail = 1, 2

        if param["no_param"][f_body] == 2:
            fc_u2 = param["fun"][f_body].pdf(
                df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"]
            )
            Fc_u2 = param["fun"][f_body].cdf(
                df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"]
            )
        else:
            fc_u2 = param["fun"][f_body].pdf(
                df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"], df_[f_body]["e"]
            )
            Fc_u2 = param["fun"][f_body].cdf(
                df_[f_body]["u2"], df_[f_body]["s"], df_[f_body]["l"], df_[f_body]["e"]
            )

        if param["no_param"][f_tail] == 2:
            ft_u2 = -param["fun"][f_tail].pdf(0, df_[f_tail]["s"], df_[f_tail]["l"])
        else:
            ft_u2 = -param["fun"][f_tail].pdf(
                0, df_[f_tail]["s"], df_[f_tail]["l"], df_[f_tail]["e"]
            )

    constraints_ = np.sqrt(1 / len(ft_u2) * np.sum((ft_u2 * Fc_u2 - fc_u2) ** 2))

    return constraints_


def fit(df_, param_, par0, mode_, ref):
    """Fits the data to the given probability model

    Args:
        * df (pd.DataFrame): raw data
        * param (dict): the parameters of the analysis
        * par0 (list): the guess parameters of the probability model
        * mode (list): components of the current mode
        * ref (int): log-likelihood value of the reference order

    Returns:
        * res['x'] (list): the fit parameters
        * res['fun'] (float): the value of the log-likelihood function
    """
    # ----------------------------------------------------------------------------------
    # Creating boundaries for optimization
    # ----------------------------------------------------------------------------------
    bnds = [[] for _ in range(len(par0))]
    if param_["optimization"]["bounds"] is False:
        bnds = None
    else:
        for i in range(0, len(par0)):
            bnds[i] = (
                par0[i] - param_["optimization"]["bounds"],
                par0[i] + param_["optimization"]["bounds"],
            )

    global df, param, mode, t_expans
    df, param, mode = df_, param_, mode_
    t_expans = params_t_expansion(mode, param, df["n"])

    # ----------------------------------------------------------------------------------
    # For assuring that local minimum of previous optimizations not be accepted during
    # the current optimization mode
    # ----------------------------------------------------------------------------------
    if ref == 1e10:
        ref = 10 * len(df)
    else:
        ref -= np.abs(ref) * 1e-4

    nllf_, j = 1e10, 0
    fixed = par0
    res, nllfv = {}, {}
    if param["constraints"]:
        if param["no_fun"] == 1:
            constraints_ = []
        elif param["no_fun"] == 2:
            constraints_ = [
                {"type": "eq", "fun": lambda x: matching_lower_bound(x)},
            ]
        else:
            constraints_ = [
                {"type": "eq", "fun": lambda x: matching_lower_bound(x)},
                {"type": "eq", "fun": lambda x: matching_upper_bound(x)},
            ]
    else:
        constraints_ = []

    while not ((nllf_ < ref) | (j >= param["optimization"]["giter"])):
        j += 1
        # ------------------------------------------------------------------------------
        # Optimize NLLF using the given algorithm
        # ------------------------------------------------------------------------------
        if param["optimization"]["method"] == "SLSQP":
            res[j] = minimize(
                nllf,
                par0,
                args=(df, mode, param, t_expans),
                bounds=bnds,
                constraints=constraints_,
                method="SLSQP",
                options={
                    "ftol": param["optimization"]["ftol"],
                    "eps": param["optimization"]["eps"],
                    "maxiter": param["optimization"]["maxiter"],
                },
            )
        elif param["optimization"]["method"] == "dual_annealing":
            res[j] = dual_annealing(
                nllf, bnds, x0=par0, args=(df, mode, param, t_expans)
            )
        elif param["optimization"]["method"] == "differential_evolution":
            res[j] = differential_evolution(
                nllf, bnds, args=(df, mode, param, t_expans)
            )
        elif param["optimization"]["method"] == "shgo":
            res[j] = shgo(nllf, par0, args=(df, mode, param, t_expans))

        # ------------------------------------------------------------------------------
        # Check whether the algorithm succesfully run
        # ------------------------------------------------------------------------------
        nllfv[j] = res[j]["fun"]
        if res[j]["success"] and not (
            res[j]["fun"] == 1e10 or res[j]["fun"] == 0.0 or res[j]["fun"] == -0.0
        ):
            nllf_ = res[j]["fun"]
        elif res[j]["message"] == "Iteration limit exceeded":
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        elif res[j]["message"] == "Iteration limit reached":
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        elif res[j]["message"] == "Positive directional derivative for linesearch":
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        elif (res[j]["message"] == "Inequality constraints incompatible") & (
            res[j]["fun"] < ref
        ):
            nllf_ = res[j]["fun"]
            fixed = res[j]["x"]
        else:
            nllf_ = 1e10

        if (
            res[j]["fun"] == 1e10
            or res[j]["fun"] == 0.0
            or res[j]["fun"] == -0.0
            or any(np.isnan(res[j].x))
        ):
            res[j]["message"] = "No valid combination of parameters"
            nllf_ = 1e10

        par0 = fixed + (-1) ** np.random.uniform(1) * (np.random.rand(1) * 1e-4 * j)
        if param_["optimization"]["bounds"] is False:
            bnds = None
        else:
            for i in range(0, len(par0)):
                bnds[i] = (
                    par0[i] - param["optimization"]["bounds"],
                    par0[i] + param["optimization"]["bounds"],
                )

        print_message(res[j], j, mode, ref)

    nllf_min = min(nllfv, key=nllfv.get)
    res = res[nllf_min]

    return res["x"], res["fun"]


def nllf(par, df, imod, param, t_expans):
    """Computes the negative log-likelyhood value

    Args:
        * par (dict): the guess parameters of the probability model
        * df (pd.DataFrame): raw data
        * imod (list): combination of modes for fitting
        * param (dict): the parameters of the analysis
        * t_expans (np.ndarray): the variability of the modes

    Returns:
        * nllf (float): negative log-likelyhood value
    """
    # ----------------------------------------------------------------------------------
    # Obtaining the parameters
    # ----------------------------------------------------------------------------------
    df, esc = get_params(df, param, par, imod, t_expans)

    if np.isnan(par).any():
        nllf = 1e10
    elif param["reduction"]:
        # ------------------------------------------------------------------------------
        # Compute the NLLF reducing some parameters of the probability models
        # ------------------------------------------------------------------------------
        idb = (df[param["var"]] <= df["u2"]) & (df[param["var"]] >= df["u1"])
        idgp1 = df[param["var"]] < df["u1"]
        idgp2 = df[param["var"]] > df["u2"]

        if (
            any(
                df["xi1"][idgp1]
                * (df[param["var"]][idgp1] - df["u1"][idgp1])
                / df["siggp1"][idgp1]
                >= 1
            )
            | any(
                df["xi2"][idgp2]
                * (df[param["var"]][idgp2] - df["u2"][idgp2])
                / df["siggp2"][idgp2]
                <= -1
            )
            | any(df["u1"] <= 0)
            | any(df["u1"] >= df["u2"])
        ):
            nllf = 1e10
        else:
            if param["no_param"][0] == 2:
                lpdf = param["fun"][1].logpdf(
                    df[param["var"]][idb], df["shape"][idb], df["loc"][idb]
                )
            else:
                lpdf = param["fun"][1].logpdf(
                    df[param["var"]][idb],
                    df["shape"][idb],
                    df["loc"][idb],
                    df["scale"][idb],
                )

            lpdf = np.append(
                lpdf,
                np.log(esc[1])
                + param["fun"][0].logpdf(
                    df["u1"][idgp1] - df[param["var"]][idgp1],
                    df["xi1"][idgp1],
                    scale=df["siggp1"][idgp1],
                ),
            )
            lpdf = np.append(
                lpdf,
                np.log(esc[2])
                + param["fun"][2].logpdf(
                    df[param["var"]][idgp2],
                    df["xi2"][idgp2],
                    loc=df["u2"][idgp2],
                    scale=df["siggp2"][idgp2],
                ),
            )
            nllf = -np.sum(lpdf)
    else:
        # ------------------------------------------------------------------------------
        # Compute the NLLF without reducing any parameter of the probability models
        # ------------------------------------------------------------------------------
        nllf, en, lpdf = 0, 0, 0

        if param["constraints"]:
            # if not (param["type"] == "circular"):
            # --------------------------------------------------------------------------
            # Two PMs
            # --------------------------------------------------------------------------
            if len(df) == 2:
                idgp1 = df[0][param["var"]] <= df[0]["u1"]
                idgp2 = df[0][param["var"]] > df[0]["u1"]

                if param["no_param"][0] == 2:
                    lpdf += np.sum(
                        param["fun"][0].logpdf(
                            df[0].loc[idgp1, param["var"]],
                            df[0].loc[idgp1, "s"],
                            df[0].loc[idgp1, "l"],
                        )
                        + np.log(esc[0])
                    )
                else:
                    lpdf += np.sum(
                        param["fun"][0].logpdf(
                            df[0].loc[idgp1, param["var"]],
                            df[0].loc[idgp1, "s"],
                            df[0].loc[idgp1, "l"],
                            df[0].loc[idgp1, "e"],
                        )
                        + np.log(esc[0])
                    )

                if param["no_param"][1] == 2:
                    lpdf += np.sum(
                        param["fun"][1].logpdf(
                            df[1].loc[idgp2, param["var"]] - df[1].loc[idgp2, "u1"],
                            df[1].loc[idgp2, "s"],
                            df[1].loc[idgp2, "l"],
                        )
                        + np.log(esc[1])
                    )
                else:
                    lpdf += np.sum(
                        param["fun"][1].logpdf(
                            df[1].loc[idgp2, param["var"]] - df[1].loc[idgp2, "u1"],
                            df[1].loc[idgp2, "s"],
                            df[1].loc[idgp2, "l"],
                            df[1].loc[idgp2, "e"],
                        )
                        + np.log(esc[1])
                    )
            else:
                # ----------------------------------------------------------------------
                # Three PMs
                # ----------------------------------------------------------------------
                iltail = df[0][param["var"]] < df[0]["u1"]
                ibody = (df[1][param["var"]] >= df[1]["u1"]) & (
                    df[1][param["var"]] <= df[1]["u2"]
                )
                iutail = df[2][param["var"]] > df[2]["u2"]

                if param["no_param"][0] == 2:
                    lpdf = np.hstack(
                        [
                            lpdf,
                            param["fun"][0].logpdf(
                                df[0].loc[iltail, "u1"]
                                - df[0].loc[iltail, param["var"]],
                                df[0].loc[iltail, "s"],
                                df[0].loc[iltail, "l"],
                            )
                            + np.log(esc[0]),
                        ]
                    )
                else:
                    lpdf = np.hstack(
                        [
                            lpdf,
                            param["fun"][0].logpdf(
                                df[0].loc[iltail, "u1"]
                                - df[0].loc[iltail, param["var"]],
                                df[0].loc[iltail, "s"],
                                df[0].loc[iltail, "l"],
                                df[0].loc[iltail, "e"],
                            )
                            + np.log(esc[0]),
                        ]
                    )

                if param["no_param"][1] == 2:
                    lpdf = np.hstack(
                        [
                            lpdf,
                            param["fun"][1].logpdf(
                                df[1].loc[ibody, param["var"]],
                                df[1].loc[ibody, "s"],
                                df[1].loc[ibody, "l"],
                            ),
                        ]
                    )
                else:
                    lpdf = np.hstack(
                        [
                            lpdf,
                            param["fun"][1].logpdf(
                                df[1].loc[ibody, param["var"]],
                                df[1].loc[ibody, "s"],
                                df[1].loc[ibody, "l"],
                                df[1].loc[ibody, "e"],
                            ),
                        ]
                    )

                if param["no_param"][2] == 2:
                    lpdf = np.hstack(
                        [
                            lpdf,
                            param["fun"][2].logpdf(
                                df[1].loc[iutail, param["var"]],
                                -df[1].loc[iutail, "u2"],
                                df[1].loc[iutail, "s"],
                                df[1].loc[iutail, "l"],
                            )
                            + np.log(esc[2]),
                        ]
                    )
                else:
                    lpdf = np.hstack(
                        [
                            lpdf,
                            param["fun"][2].logpdf(
                                df[1].loc[iutail, param["var"]],
                                -df[1].loc[iutail, "u2"],
                                df[1].loc[iutail, "s"],
                                df[1].loc[iutail, "l"],
                                df[1].loc[iutail, "e"],
                            )
                            + np.log(esc[2]),
                        ]
                    )

            if (np.isnan(lpdf).any()) | (np.isinf(lpdf).any()):
                nllf = 1e10
            else:
                if lpdf == 0:
                    nllf = 1e10
                else:
                    nllf = -np.sum(lpdf)

        else:
            if param["no_fun"] == 1:
                if param["no_param"][0] == 2:
                    lpdf = param["fun"][0].logpdf(
                        df[0][param["var"]], df[0]["s"], df[0]["l"]
                    )
                else:
                    lpdf = param["fun"][0].logpdf(
                        df[0][param["var"]], df[0]["s"], df[0]["l"], df[0]["e"]
                    )
            else:
                for i in range(param["no_fun"]):
                    if i == 0:
                        df_ = df[i].loc[df[i][param["var"]] < df[i]["u" + str(i + 1)]]
                    elif i == param["no_fun"] - 1:
                        df_ = df[i].loc[df[i][param["var"]] >= df[i]["u" + str(i)]]
                    else:
                        df_ = df[i].loc[
                            (
                                (df[i][param["var"]] >= df[i]["u" + str(i)])
                                & (df[i][param["var"]] < df[i]["u" + str(i + 1)])
                            )
                        ]

                    # ------------------------------------------------------------------
                    # Solution for piecewise PMs in circular variables
                    # ------------------------------------------------------------------
                    if (
                        (not param["no_fun"] == 1)
                        & (param["circular"])
                        & (param["scipy"][i] == True)
                    ):
                        en = np.log(
                            param["fun"][i].cdf(
                                df_["u" + str(i + 1)], df_["s"], df_["l"]
                            )
                            - param["fun"][i].cdf(df_["u" + str(i)], df_["s"], df_["l"])
                        )

                    nplogesci = np.log(esc[i])

                    if param["no_param"][i] == 2:
                        lpdf = np.hstack(
                            [
                                lpdf,
                                param["fun"][i].logpdf(
                                    df_[param["var"]], df_["s"], df_["l"]
                                )
                                + nplogesci
                                - en,
                            ]
                        )
                    else:
                        lpdf = np.hstack(
                            [
                                lpdf,
                                param["fun"][i].logpdf(
                                    df_[param["var"]], df_["s"], df_["l"], df_["e"]
                                )
                                + nplogesci
                                - en,
                            ]
                        )

            if np.isnan(lpdf).any():
                nllf = 1e10
            else:
                if lpdf.size == 1:
                    nllf = 1e10
                else:
                    nllf += -np.sum(lpdf)

    return nllf


def get_params(df, param, par, imod, t_expans):
    """Gets the parameters of the probability models for fitting

    Args:
        * df (pd.DataFrame): raw data
        * param (dict): the parameters of the analysis
        * par (dict): the guess parameters of the probability model
        * imod (list): combination of modes for fitting
        * t_expans (np.ndarray): the variability of the modes
        * pos (list, optional): location of the different probability models. Defaults to [0, 0, 0].

    Returns:
        * df (pd.DataFrame): the parameters
        * esc (list): weight of the probability models
    """
    mode, esc = imod, {}
    pos = [0, 0, 0]
    if param["reduction"]:
        # --------------------------------------------------------------------------
        # After first order, all parameters of the same function are developed to
        # the next order at the same fit
        # --------------------------------------------------------------------------
        if param["basis_function"]["method"] in [
            "trigonometric",
            "sinusoidal",
            "modified",
        ]:
            pars_fourier = 1
            # Fourier expansion has two parameters per mode
            if bool(
                [
                    meth
                    for meth in ["trigonometric", "modified"]
                    if (meth in param["basis_function"]["method"])
                ]
            ):
                pars_fourier = 2

            df["shape"] = par[0] + np.dot(
                par[1 : mode[0] * pars_fourier + 1],
                t_expans[0 : mode[0] * pars_fourier, :],
            )
            df["loc"] = par[mode[0] * 2 + 1] + np.dot(
                par[mode[0] * pars_fourier + 2 : mode[0] * pars_fourier * 2 + 2],
                t_expans[0 : mode[0] * pars_fourier, :],
            )
            if param["no_param"][0] == 2:
                df["xi2"] = par[mode[0] * pars_fourier * 2 + 2] + np.dot(
                    par[
                        mode[0] * pars_fourier * 2
                        + 3 : mode[0] * pars_fourier * 2
                        + mode[1] * pars_fourier
                        + 3
                    ],
                    t_expans[0 : mode[1] * pars_fourier, :],
                )
            else:
                df["scale"] = par[mode[0] * pars_fourier * 2 + 2] + np.dot(
                    par[
                        mode[0] * pars_fourier * 2 + 3 : mode[0] * pars_fourier * 3 + 3
                    ],
                    t_expans[0 : mode[0] * pars_fourier, :],
                )
                df["xi2"] = par[mode[0] * pars_fourier * 3 + 3] + np.dot(
                    par[
                        mode[0] * pars_fourier * 3
                        + 4 : mode[0] * pars_fourier * 3
                        + mode[1] * pars_fourier
                        + 4
                    ],
                    t_expans[0 : mode[1] * pars_fourier, :],
                )
        else:
            # Polynomial expansion has one parameters per mode
            df["shape"] = params_t_expansion(par[0 : mode[0] * 2 + 1], param, df["n"])
            df["loc"] = params_t_expansion(
                par[mode[0] + 1 : mode[0] * 2 + 2], param, df["n"]
            )

            if param["no_param"][0] == 2:
                df["xi2"] = params_t_expansion(
                    par[mode[0] * 2 + 2 : mode[0] * 2 + mode[1] + 3], param, df["n"]
                )
            else:
                df["scale"] = params_t_expansion(
                    par[mode[0] * 2 + 2 : mode[0] * 3 + 3], param, df["n"]
                )
                df["xi2"] = params_t_expansion(
                    par[mode[0] * 3 + 3 : mode[0] * 3 + mode[1] + 4], param, df["n"]
                )

        esc[1] = st.norm.cdf(par[-2])
        esc[2] = 1 - st.norm.cdf(par[-1])
        if param["no_param"][0] == 2:
            df["u1"] = param["fun"][1].ppf(esc[1], df["shape"], df["loc"])
            df["u2"] = param["fun"][1].ppf(st.norm.cdf(par[-1]), df["shape"], df["loc"])

            df["siggp1"] = esc[1] / param["fun"][1].pdf(
                df["u1"], df["shape"], df["loc"]
            )
            df["xi1"] = -df["siggp1"] / df["u1"]
            df["siggp2"] = esc[2] / param["fun"][1].pdf(
                df["u2"], df["shape"], df["loc"]
            )
        else:
            df["u1"] = param["fun"][1].ppf(esc[1], df["shape"], df["loc"], df["scale"])
            df["u2"] = param["fun"][1].ppf(
                st.norm.cdf(par[-1]), df["shape"], df["loc"], df["scale"]
            )

            df["siggp1"] = esc[1] / param["fun"][1].pdf(
                df["u1"], df["shape"], df["loc"], df["scale"]
            )
            df["xi1"] = -df["siggp1"] / df["u1"]
            df["siggp2"] = esc[2] / param["fun"][1].pdf(
                df["u2"], df["shape"], df["loc"], df["scale"]
            )
    else:
        # ------------------------------------------------------------------------------
        # Obtaining the parameters in a handy form for bi or tri-parametric models
        # ------------------------------------------------------------------------------
        df_, esc = {}, {}
        pars_fourier = 1
        if param["basis_function"]["order"] == 0:
            mode = [0, 0, 0]
        # elif not first:
        if len(imod) == 1:
            mode = [imod[0], imod[0], imod[0]]
        elif len(imod) == 2:
            mode = [imod[0], imod[1], imod[0]]
        else:
            mode = [imod[0], imod[1], imod[2]]

        for i in range(param["no_fun"]):
            if param["basis_function"]["method"] in [
                "trigonometric",
                "sinusoidal",
                "modified",
            ]:
                # Fourier expansion has two parameters per mode
                if bool(
                    [
                        meth
                        for meth in ["trigonometric", "modified"]
                        if (meth in param["basis_function"]["method"])
                    ]
                ):
                    pars_fourier = 2

                df_[i] = df.copy()
                df_[i]["s"] = par[pos[0]]

                df_[i]["s"] = df_[i]["s"] + np.dot(
                    par[pos[0] + 1 : pos[0] + mode[i] * pars_fourier + 1],
                    t_expans[0 : mode[i] * pars_fourier, :],
                )
                df_[i]["l"] = par[pos[0] + mode[i] * pars_fourier + 1]

                df_[i]["l"] = df_[i]["l"] + np.dot(
                    par[
                        pos[0]
                        + mode[i] * pars_fourier
                        + 2 : pos[0]
                        + mode[i] * pars_fourier * 2
                        + 2
                    ],
                    t_expans[0 : mode[i] * pars_fourier, :],
                )

                if param["no_param"][i] == 3:
                    df_[i]["e"] = par[pos[0] + mode[i] * pars_fourier * 2 + 2]

                    df_[i]["e"] = df_[i]["e"] + np.dot(
                        par[
                            pos[0]
                            + mode[i] * pars_fourier * 2
                            + 3 : pos[0]
                            + mode[i] * pars_fourier * 3
                            + 3
                        ],
                        t_expans[0 : mode[i] * pars_fourier, :],
                    )
            else:
                df_[i] = df.copy()
                df_[i]["s"] = params_t_expansion(
                    par[0 : mode[0] * 2 + 1], param, df["n"]
                )

                df_[i]["l"] = params_t_expansion(
                    par[mode[0] + 1 : mode[0] * 2 + 2], param, df["n"]
                )

                if param["no_param"][0] == 3:
                    df_[i]["e"] = params_t_expansion(
                        par[mode[0] * 2 + 2 : mode[0] * 2 + mode[1] + 3],
                        param,
                        df["n"],
                    )

            # --------------------------------------------------------------------------
            # Updating index for the next probability model
            # pos[0]: the index of the first parameter of the i-PM
            # pos[1]: cumulative sum of the number of parameters of the PM
            # pos[2]: i-PM
            # --------------------------------------------------------------------------
            if param["basis_function"]["order"] == 0:
                pos[0] = pos[0] + int(param["no_param"][i])
                pos[1] = pos[1] + param["no_param"][i]
                pos[2] = i + 1

            pos[0] = (
                pos[0]
                + int(param["no_param"][i])
                + pars_fourier * imod[i] * int(param["no_param"][i])
            )
            pos[1] += 1
            pos[2] += 1

        df = df_.copy()

        # ---------------------------------------------------------------------------
        # Obtaining percentiles from parameters.
        # ---------------------------------------------------------------------------
        if param["fix_percentiles"]:
            if param["no_fun"] == 1:
                esc[0] = 1
            elif param["no_fun"] == 2:
                esc[0] = param["ws_ps"][0]
                esc[1] = 1 - param["ws_ps"][0]
            else:
                esc[0] = param["ws_ps"][0]
                esc[1] = param["ws_ps"][1] - param["ws_ps"][0]
                esc[2] = 1 - esc[1] - esc[0]
        else:
            if param["no_fun"] == 1:
                esc[0] = 1
            if param["no_fun"] == 2:
                # Two restrictions
                esc[0] = st.norm.cdf(par[-1])
                esc[1] = 1 - st.norm.cdf(par[-1])
            else:
                # Three restrictions
                esc[0] = st.norm.cdf(par[-2])
                esc[1] = st.norm.cdf(par[-1]) - st.norm.cdf(par[-2])
                esc[2] = 1 - st.norm.cdf(par[-1])

        if ((not param["fix_percentiles"]) | (param["constraints"])) & (
            not param["reduction"]
        ):
            if param["no_fun"] == 2:
                if param["no_param"][0] == 2:
                    df[0]["u1"] = param["fun"][0].ppf(esc[0], df[0]["s"], df[0]["l"])
                    df[1]["u1"] = param["fun"][0].ppf(esc[0], df[0]["s"], df[0]["l"])
                else:
                    df[0]["u1"] = param["fun"][0].ppf(
                        esc[0], df[0]["s"], df[0]["l"], df[0]["e"]
                    )
                    df[1]["u1"] = param["fun"][0].ppf(
                        esc[0], df[0]["s"], df[0]["l"], df[0]["e"]
                    )
            elif param["no_fun"] == 3:
                if param["no_param"][1] == 2:
                    df[0]["u1"] = param["fun"][1].ppf(esc[0], df[1]["s"], df[1]["l"])
                    df[1]["u1"] = param["fun"][1].ppf(esc[0], df[1]["s"], df[1]["l"])
                    df[1]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[1]["s"], df[1]["l"]
                    )
                    df[2]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[1]["s"], df[1]["l"]
                    )
                else:
                    df[0]["u1"] = param["fun"][1].ppf(
                        esc[0], df[1]["s"], df[1]["l"], df[1]["e"]
                    )
                    df[1]["u1"] = param["fun"][1].ppf(
                        esc[0], df[1]["s"], df[1]["l"], df[1]["e"]
                    )
                    df[1]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[1]["s"], df[1]["l"], df[1]["e"]
                    )
                    df[2]["u2"] = param["fun"][1].ppf(
                        1 - esc[2], df[1]["s"], df[1]["l"], df[1]["e"]
                    )

    return df, esc


def ppf(df, param, ppf=True):
    """Computes the inverse of the probability function

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): the parameters of the probability model
        * ppf (boolean, optional): boolean for selecting the method for the computation. Defaults is True (creating a previous mesh of data). False is computed from the probabilty models (sometimes it is not possible).

    Return:
        df (pd.DataFrame): inverse of the cdf
    """

    t_expans = params_t_expansion(param["mode"], param, df["n"])

    if param["reduction"]:
        # ------------------------------------------------------------------------------
        # If reduction is possible
        # ------------------------------------------------------------------------------
        df, esc = get_params(df, param, param["par"], param["mode"], t_expans)
        idu1 = df["prob"] < esc[1]
        df.loc[idu1, param["var"]] = df.loc[idu1, "u1"] - param["fun"][0].ppf(
            (esc[1] - df.loc[idu1, "prob"]) / esc[1],
            df.loc[idu1, "xi1"],
            scale=df.loc[idu1, "siggp1"],
        )

        idu2 = df["prob"] > 1 - esc[2]
        df.loc[idu2, param["var"]] = df.loc[idu2, "u2"] + param["fun"][2].ppf(
            (df.loc[idu2, "prob"] - 1 + esc[2]) / esc[2],
            df.loc[idu2, "xi2"],
            loc=0,
            scale=df.loc[idu2, "siggp2"],
        )
        idb = (df["prob"] <= 1 - esc[2]) & (df["prob"] >= esc[1])
        if param["no_param"][0] == 2:
            df.loc[idb, param["var"]] = param["fun"][1].ppf(
                df.loc[idb, "prob"], df.loc[idb, "shape"], df.loc[idb, "loc"]
            )
        else:
            df.loc[idb, param["var"]] = param["fun"][1].ppf(
                df.loc[idb, "prob"],
                df.loc[idb, "shape"],
                df.loc[idb, "loc"],
                df.loc[idb, "scale"],
            )
    else:
        if param["no_fun"] == 1:
            # --------------------------------------------------------------------------
            # If just one probability models is given
            # --------------------------------------------------------------------------

            df, esc = get_params(df, param, param["par"], param["mode"], t_expans)
            if param["no_param"][0] == 2:
                df[0][param["var"]] = param["fun"][0].ppf(
                    df[0]["prob"], df[0]["s"], df[0]["l"]
                )
            else:
                df[0][param["var"]] = param["fun"][0].ppf(
                    df[0]["prob"], df[0]["s"], df[0]["l"], df[0]["e"]
                )
            df = df[0]
        else:
            # --------------------------------------------------------------------------
            # Wheter more than one probability model are given
            # --------------------------------------------------------------------------
            data = np.linspace(param["minimax"][0], param["minimax"][1], 10000)
            df[param["var"]] = -1

            cdfs = cdf(df, param, ppf=True)

            posi = np.zeros(len(df), dtype=int)
            dfn = np.sort(df["n"].unique())
            # show_message = False
            for i, j in enumerate(df.index):  # Seeking every n (dates)
                posn = np.argmin(np.abs(df["n"][j] - dfn))
                posj = np.argmin(np.abs(df["prob"][j] - cdfs[posn, :].T))
                posi[i] = posj
                if not posj:
                    posi[i] = posi[i - 1]
            #         show_message = True

            # if show_message:
            #     print("The parameter might be not valid for some periods.")

            df.loc[:, param["var"]] = data[posi]

    return df


def cdf(df, param, ppf=False):
    """Computes the cumulative distribution function

    Args:
        * df (pd.DataFrame): raw time series
        * param (dict): the parameters of the probability model
        * ppf (boolean, optional): boolean for selecting the method for the computation. Defaults is False (creating a previous mesh of data). False is computed from the probability models (sometimes it is not possible).

    Return:
        * prob (pd.DataFrame): the non-excedence probability
    """

    t_expans = params_t_expansion(param["mode"], param, df["n"])
    if not any(df.columns == "prob"):
        df["prob"] = 0

    if param["reduction"]:
        df, esc = get_params(df, param, param["par"], param["mode"], t_expans)

        fu1 = df["data"] < df["u1"]
        df.loc[fu1, "prob"] = esc[1] * (
            1
            - df.loc[fu1, "xi1"]
            / df.loc[fu1, "siggp1"]
            * (df.loc[fu1, "data"] - df.loc[fu1, "u1"])
        ) ** (-1.0 / df.loc[fu1, "xi1"])

        fu2 = df["data"] > df["u2"]
        df.loc[fu2, "prob"] = (
            1
            - esc[2]
            + esc[2]
            * (
                1
                - (
                    1
                    + df.loc[fu2, "xi2"]
                    / df.loc[fu2, "siggp2"]
                    * (df.loc[fu2, "data"] - df.loc[fu2, "u2"])
                )
                ** (-1.0 / df.loc[fu2, "xi2"])
            )
        )

        fuc = (df["data"] >= df["u1"]) & (df["data"] <= df["u2"])
        if param["no_param"][0] == 2:
            df.loc[fuc, "prob"] = param["fun"][1].cdf(
                df.loc[fuc, "data"], df.loc[fuc, "shape"], df.loc[fuc, "loc"]
            )
        else:
            df.loc[fuc, "prob"] = param["fun"][1].cdf(
                df.loc[fuc, "data"],
                df.loc[fuc, "shape"],
                df.loc[fuc, "loc"],
                df.loc[fuc, "scale"],
            )
        cdf_ = df["prob"]
    else:
        if param["no_fun"] == 1:
            # One PM
            df, esc = get_params(df, param, param["par"], param["mode"], t_expans)

            if param["no_param"][0] == 2:
                df[0]["prob"] = param["fun"][0].cdf(
                    df[0]["data"], df[0]["s"], df[0]["l"]
                )
            else:
                df[0]["prob"] = param["fun"][0].cdf(
                    df[0]["data"], df[0]["s"], df[0]["l"], df[0]["e"]
                )
            cdf_ = df[0]["prob"]
        else:
            # Si tenemos más de uno
            if ppf:
                data = np.linspace(param["minimax"][0], param["minimax"][1], 10000)
                dfn = np.sort(df["n"].unique())
                t_expans = params_t_expansion(param["mode"], param, dfn)
                aux = pd.DataFrame(-1, index=dfn, columns=["s"])
                aux["n"] = df["n"]
                dff, esc = get_params(
                    aux, param, param["par"], param["mode"], t_expans,
                )
                cdf_ = np.zeros([len(dfn), len(data)])
                if param["constraints"]:
                    if len(dff) == 2:
                        for k, j in enumerate(dfn):
                            fu1 = data <= dff[0].loc[j, "u1"]
                            fu2 = data > dff[0].loc[j, "u1"]

                            if param["no_param"][0] == 2:
                                cdf_[k, fu1] = param["fun"][0].cdf(
                                    data[fu1], dff[0].loc[j, "s"], dff[0].loc[j, "l"]
                                )
                            else:
                                cdf_[k, fu1] = param["fun"][0].cdf(
                                    data[fu1],
                                    dff[0].loc[j, "s"],
                                    dff[0].loc[j, "l"],
                                    dff[0].loc[j, "e"],
                                )

                            if param["no_param"][1] == 2:
                                cdf_[k, fu2] = esc[0] + esc[1] * param["fun"][1].cdf(
                                    data[fu2] - dff[0].loc[j, "u1"],
                                    dff[1].loc[j, "s"],
                                    dff[1].loc[j, "l"],
                                )
                            else:
                                cdf_[k, fu2] = esc[0] + esc[1] * param["fun"][1].cdf(
                                    data[fu2] - dff[0].loc[j, "u1"],
                                    dff[1].loc[j, "s"],
                                    dff[1].loc[j, "l"],
                                    dff[1].loc[j, "e"],
                                )
                    else:
                        for k, j in enumerate(dfn):
                            fu0 = data < dff[0].loc[j, "u1"]
                            fu1 = (data >= dff[1].loc[j, "u1"]) & (
                                data <= dff[1].loc[j, "u2"]
                            )
                            fu2 = data > dff[2].loc[j, "u2"]

                            if param["no_param"][0] == 2:
                                cdf_[k, fu0] = esc[0] * param["fun"][0].cdf(
                                    dff[0].loc[j, "u1"] - data[fu0],
                                    dff[0].loc[j, "s"],
                                    dff[0].loc[j, "l"],
                                )
                            else:
                                cdf_[k, fu0] = esc[0] * param["fun"][0].cdf(
                                    dff[0].loc[j, "u1"] - data[fu0],
                                    dff[0].loc[j, "s"],
                                    dff[0].loc[j, "l"],
                                    dff[0].loc[j, "e"],
                                )

                            if param["no_param"][1] == 2:
                                cdf_[k, fu1] = param["fun"][1].cdf(
                                    data[fu1], dff[1].loc[j, "s"], dff[1].loc[j, "l"]
                                )
                            else:
                                cdf_[k, fu1] = param["fun"][1].cdf(
                                    data[fu1],
                                    dff[1].loc[j, "s"],
                                    dff[1].loc[j, "l"],
                                    dff[1].loc[j, "e"],
                                )

                            if param["no_param"][2] == 2:
                                cdf_[k, fu2] = (
                                    1
                                    - esc[2]
                                    + esc[2]
                                    * param["fun"][2].cdf(
                                        data[fu2] - dff[2].loc[j, "u2"],
                                        dff[2].loc[j, "s"],
                                        dff[2].loc[j, "l"],
                                    )
                                )
                            else:
                                cdf_[k, fu2] = (
                                    1
                                    - esc[2]
                                    + esc[2]
                                    * param["fun"][2].cdf(
                                        data[fu2] - dff[2].loc[j, "u2"],
                                        dff[2].loc[j, "s"],
                                        dff[2].loc[j, "l"],
                                        dff[2].loc[j, "e"],
                                    )
                                )
                else:
                    # For piecewise PMs
                    for k, j in enumerate(dfn):
                        for i in range(param["no_fun"]):
                            esci = esc[i]

                            if param["no_param"][i] == 2:
                                if (param["circular"]) & (
                                    param["fun"][i] == "wrap_norm"
                                ):
                                    cdf_[k, :] += esc[i] * param["fun"][i].cdf(
                                        data, dff[i].loc[j, "s"], dff[i].loc[j, "l"]
                                    )
                                else:
                                    en = param["fun"][i].cdf(
                                        param["minimax"][1],
                                        dff[i].loc[j, "s"],
                                        dff[i].loc[j, "l"],
                                    ) - param["fun"][i].cdf(
                                        param["minimax"][0],
                                        dff[i].loc[j, "s"],
                                        dff[i].loc[j, "l"],
                                    )
                                    cdf_[k, :] += (
                                        esci
                                        * (
                                            param["fun"][i].cdf(
                                                data,
                                                dff[i].loc[j, "s"],
                                                dff[i].loc[j, "l"],
                                            )
                                            - param["fun"][i].cdf(
                                                param["minimax"][0],
                                                dff[i].loc[j, "s"],
                                                dff[i].loc[j, "l"],
                                            )
                                        )
                                        / en
                                    )
                            else:
                                en = param["fun"][i].cdf(
                                    param["minimax"][1],
                                    dff[i].loc[j, "s"],
                                    dff[i].loc[j, "l"],
                                    dff[i].loc[j, "e"],
                                ) - param["fun"][i].cdf(
                                    param["minimax"][0],
                                    dff[i].loc[j, "s"],
                                    dff[i].loc[j, "l"],
                                    dff[i].loc[j, "e"],
                                )
                                cdf_[k, :] += (
                                    esci
                                    * (
                                        param["fun"][i].cdf(
                                            data,
                                            dff[i].loc[j, "s"],
                                            dff[i].loc[j, "l"],
                                            dff[i].loc[j, "e"],
                                        )
                                        - param["fun"][i].cdf(
                                            param["minimax"][0],
                                            dff[i].loc[j, "s"],
                                            dff[i].loc[j, "l"],
                                            dff[i].loc[j, "e"],
                                        )
                                    )
                                    / en
                                )

            else:
                df, esc = get_params(df, param, param["par"], param["mode"], t_expans)
                df[0]["prob"] = 0
                # ----------------------------------------------------------------------
                # Different approach using restrictions
                # ----------------------------------------------------------------------
                if param["constraints"]:
                    # For 2 PMs
                    if len(df) == 2:  # Agregué prob, data y u1
                        fu1 = df[0]["data"] <= df[0]["u1"]
                        fu2 = df[0]["data"] > df[0]["u1"]

                        if param["no_param"][0] == 2:
                            df[0].loc[fu1, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu1, "data"],
                                df[0].loc[fu1, "s"],
                                df[0].loc[fu1, "l"],
                            )
                        else:
                            df[0].loc[fu1, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu1, "data"],
                                df[0].loc[fu1, "s"],
                                df[0].loc[fu1, "l"],
                                df[0].loc[fu1, "e"],
                            )

                        if param["no_param"][1] == 2:
                            df[0].loc[fu2, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                df[1].loc[fu2, "s"],
                                df[1].loc[fu2, "l"],
                            )
                        else:
                            df[0].loc[fu2, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                df[1].loc[fu2, "s"],
                                df[1].loc[fu2, "l"],
                                df[1].loc[fu2, "e"],
                            )
                    else:
                        df[0]["u2"] = df[2]["u2"]
                        # For 3 PMs
                        fu0 = df[0]["data"] < df[0]["u1"]
                        fu1 = (df[0]["data"] >= df[0]["u1"]) & (
                            df[0]["data"] <= df[0]["u2"]
                        )
                        fu2 = df[0]["data"] > df[0]["u2"]

                        if param["no_param"][0] == 2:
                            df[0].loc[fu0, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu0, "u1"] - df[0].loc[fu0, "data"],
                                df[0]["s"],
                                df[0]["l"],
                            )
                        else:
                            df[0].loc[fu0, "prob"] = param["fun"][0].cdf(
                                df[0].loc[fu0, "u1"] - df[0].loc[fu0, "data"],
                                df[0]["s"],
                                df[0]["l"],
                                df[0]["e"],
                            )

                        if param["no_param"][1] == 2:
                            df[0].loc[fu1, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu1, "data"], df[1]["s"], df[1]["l"]
                            )
                        else:
                            df[0].loc[fu1, "prob"] = param["fun"][1].cdf(
                                df[0].loc[fu1, "data"],
                                df[1]["s"],
                                df[1]["l"],
                                df[1]["e"],
                            )

                        if param["no_param"][2] == 2:
                            df[0].loc[fu2, "prob"] = (
                                1
                                - esc[2]
                                + esc[2]
                                * param["fun"][2].cdf(
                                    df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                    df[2]["s"],
                                    df[2]["l"],
                                )
                            )
                        else:
                            df[0].loc[fu2, "prob"] = (
                                1
                                - esc[2]
                                + esc[2]
                                * param["fun"][2].cdf(
                                    df[0].loc[fu2, "data"] - df[0].loc[fu2, "u1"],
                                    df[2]["s"],
                                    df[2]["l"],
                                    df[2]["e"],
                                )
                            )

                else:
                    # ----------------------------------------------------------------------
                    # First approach of the non-stationary analysis
                    # ----------------------------------------------------------------------
                    for i in range(param["no_fun"]):
                        if param["no_param"][i] == 2:
                            en = param["fun"][i].cdf(
                                param["minimax"][1], df[i]["s"], df[i]["l"]
                            ) - param["fun"][i].cdf(
                                param["minimax"][0], df[i]["s"], df[i]["l"]
                            )
                            df[0]["prob"] += (
                                esc[i]
                                * (
                                    param["fun"][i].cdf(
                                        df[i]["data"], df[i]["s"], df[i]["l"]
                                    )
                                    - param["fun"][i].cdf(
                                        param["minimax"][0], df[i]["s"], df[i]["l"]
                                    )
                                )
                                / en
                            )
                        else:
                            en = param["fun"][i].cdf(
                                param["minimax"][1], df[i]["s"], df[i]["l"], df[i]["e"]
                            ) - param["fun"][i].cdf(
                                param["minimax"][0], df[i]["s"], df[i]["l"], df[i]["e"]
                            )
                            df[0]["prob"] += (
                                esc[i]
                                * (
                                    param["fun"][i].cdf(
                                        df[i]["data"],
                                        df[i]["s"],
                                        df[i]["l"],
                                        df[i]["e"],
                                    )
                                    - param["fun"][i].cdf(
                                        param["minimax"][0],
                                        df[i]["s"],
                                        df[i]["l"],
                                        df[i]["e"],
                                    )
                                )
                                / en
                            )

                cdf_ = df[0]["prob"]

    return cdf_


def transform(data, params):
    """Normalized the input data given a normalized method (Box-Cox or Yeo-Johnson)

    Args:
        data ([pd.DataFrame]): input timeseries
        params ([dict]): parameters which can include

    Returns:
        data: normalized input data
        params: a dictionary with lambda of transformation
    """
    from sklearn import preprocessing

    # In some cases, it is usually that pd.Series appears. Transform to pd.DataFrame
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if "lambda" in params["transform"].keys():
        # Use the lambda of a previous transformation if it is given
        powertransform = preprocessing.PowerTransformer(
            params["transform"]["method"], standardize=False
        )
        powertransform.lambdas_ = [params["transform"]["lambda"]]
    else:
        # Compute lambda of transformation
        powertransform = preprocessing.PowerTransformer(
            params["transform"]["method"], standardize=False
        ).fit(data.values.reshape(-1, 1))
        params["transform"]["lambda"] = powertransform.lambdas_[0]

    # Normalized the input data
    data = pd.DataFrame(
        {data.columns[0]: powertransform.transform(data.values.reshape(-1, 1))[:, 0]},
        index=data.index,
    )

    return data, params


def inverse_transform(data, params, ensemble=False):
    """Compute the inverse of the transformation

    Args:
        data ([type]): [description]
        params ([type]): [description]
        ensemble (boolean): True or False

    Returns:
        [type]: [description]
    """
    from sklearn import preprocessing

    if ensemble:
        powertransform = preprocessing.PowerTransformer(
            params["method_ensemble"], standardize=False
        )
        powertransform.lambdas_ = [params["lambda_ensemble"]]
    else:
        powertransform = preprocessing.PowerTransformer(
            params["transform"]["method"], standardize=False
        )
        powertransform.lambdas_ = [params["transform"]["lambda"]]

    data = pd.DataFrame(
        {
            data.columns[0]: powertransform.inverse_transform(
                data.values.reshape(-1, 1)
            )[:, 0]
        },
        index=data.index,
    )
    return data


def numerical_cdf_pdf_at_n(n, param, variable, alpha=0.01):
    """[summary]

    Args:
        ns ([type]): [description]
        param ([type]): [description]
        variable ([type], optional): [description]. Defaults to None.
        alpha (float, optional): [description]. Defaults to 0.01.

    Returns:
        [type]: [description]
    """

    param = auxiliar.str2fun(param, None)
    pemp = np.linspace(0 + alpha / 2, 1 - alpha / 2)

    df = pd.DataFrame(pemp, index=pemp, columns=["prob"])
    df["n"] = np.ones(len(pemp)) * n
    if (param["non_stat_analysis"] == True) | (param["no_fun"] > 1):
        res = ppf(df, param)
    else:
        res = pd.DataFrame(
            param["fun"][0].ppf(df["prob"], *param["par"]),
            index=df.index,
            columns=[variable],
        )

    # Transformed timeserie
    if (not param["transform"]["plot"]) & param["transform"]["make"]:
        if "scale" in param:
            res[param["var"]] = res[param["var"]] * param["scale"]

        res[param["var"]] = res[param["var"]] + param["transform"]["min"]
        res[param["var"]] = inverse_transform(res[[param["var"]]], param)
    elif "scale" in param:
        res[param["var"]] = res[param["var"]] * param["scale"]

    if param["circular"]:
        values_ = (
            np.diff(np.rad2deg(res[param["var"]].values))
            + np.rad2deg(res[param["var"]].values)[:-1]
        )
        index_ = np.diff(res[param["var"]].index)
    else:
        values_ = np.diff(res[param["var"]].index) / np.diff(res[param["var"]].values)
        index_ = np.diff(res[param["var"]].values) + res[param["var"]].values[:-1]
        # index_ = res[
        #     param["var"]
        # ].values
    # values_ = np.append(0, values_)
    res_ = pd.DataFrame(values_, index=index_, columns=["pdf"])
    res_["cdf"] = (res[param["var"]].index[:-1] + res[param["var"]].index[1:]) / 2
    # res_["cdf"] = res[param["var"]].index
    return res_


def ensemble_cdf(data, param, variable, nodes=[4383, 250]):
    """Computes the ensemble of the cumulative distribution function for several models

    Args:
        * data (pd.DataFrame): raw data
        * param (dict): the parameters of the probability model
        * variable (string): name of the variable
        * nodes (list): no of nodes in the time and variable axes. Defaults to [4383, 250].

    Returns:
        * cdfs (pd.DataFrame): the non-stationary non-excedance probability
    """
    cdfs = 0
    data_model = np.tile(data, nodes[0])
    index_ = np.repeat(np.linspace(0, 1 - 1 / nodes[0], nodes[0]), nodes[1])
    dfm = pd.DataFrame(
        np.asarray([index_, data_model]).T, index=index_, columns=["n", "data"]
    )
    indexes = np.where(dfm["data"] == 0)[0]
    for model in param["TS"]["models"]:
        param[variable] = auxiliar.str2fun(param[variable], model)
        cdf_model = cdf(dfm.copy(), param[variable][model])
        cdf_model.iloc[indexes[np.isnan(cdf_model.iloc[indexes]).values]] = 0
        cdf_model.loc[np.isnan(cdf_model)] = 1
        cdfs += cdf_model

    cdfs = cdfs / len(param["TS"]["models"])
    return cdfs


def ensemble_ppf(df, param, variable, nodes=[4383, 250]):
    """Computes the inverse of the cumulative distribution function

    Args:
        * data (pd.DataFrame): raw data
        * param (dict): the parameters of the probability model
        * variable (string): name of the variable
        * nodes (list): no of nodes in the time and variable axes. Defaults to [4383, 250].

    Returns:
        * df (pd.DataFrame): the raw time series
    """

    data = np.linspace(
        param["TS"]["minimax"][variable][0],
        param["TS"]["minimax"][variable][1],
        nodes[1],
    )
    df[variable] = -1

    print("Attempting to read F-file of %s" % (variable))
    try:
        print("F-file of %s readed." % (variable))
        cdfs = read.csv(param["TS"]["F_files"] + variable)
    except:
        if "F_files" in param["TS"].keys():
            cdfs = ensemble_cdf(data, param, variable, nodes)
            print("Writting F-file of %s" % (variable))
            pd.DataFrame(cdfs).to_csv(param["TS"]["F_files"] + variable + ".csv")
        else:
            raise ValueError(
                "F-files are not found at {}.".format(
                    param["TS"]["F_files"] + variable + ".csv"
                )
            )

    dfn = np.sort(cdfs.index.unique())
    print("Estimating pdf of ensemble for %s" % (variable))
    for j in df["n"].index:  # Find the date for every n
        posn = np.argmin(np.abs(df.loc[j, "n"] - dfn))
        posi = np.argmin(np.abs(df["prob"][j] - cdfs.loc[dfn[posn]]))
        df.loc[j, variable] = data[posi]

    return df


def params_t_expansion(mod, param, nper):
    """Computes the oscillatory dependency in time of the parameters

    Args:
        * mod (int): maximum mode of the oscillatory dependency
        * param (dict): the parameters
        * nper (pd.DataFrame): time series of normalize year

    Return:
        * t_expans (np.ndarray): the variability of every mode
    """

    if param["basis_function"]["method"] == "trigonometric":
        t_expans = np.zeros([np.max(mod) * 2, len(nper)])
        for i in range(0, np.max(mod)):
            n = (
                nper
                * np.max(param["basis_function"]["periods"])
                / param["basis_function"]["periods"][i]
            )
            t_expans[2 * i, :] = np.cos(2 * np.pi * n.T)
            t_expans[2 * i + 1, :] = np.sin(2 * np.pi * n.T)
    elif param["basis_function"]["method"] == "modified":
        t_expans = np.zeros([np.max(mod) * 2, len(nper)])
        for i in range(0, np.max(mod)):
            per_ = (
                np.max(param["basis_function"]["periods"])
                / param["basis_function"]["periods"][i]
            )
            n = (
                nper
                * np.max(param["basis_function"]["periods"])
                / param["basis_function"]["periods"][i]
            )

            t_expans[2 * i, :] = np.cos(np.pi * (n.T - 1))
            t_expans[2 * i + 1, :] = np.sin(
                2 * np.pi * n.T - per_ * np.pi - np.pi * nper + 0.5 * np.pi
            )
    elif param["basis_function"]["method"] == "sinusoidal":
        t_expans = np.zeros([np.max(mod), len(nper)])
        for i in range(0, np.max(mod)):
            # n = (
            #     nper
            #     * np.max(param["basis_function"]["periods"])
            #     / param["basis_function"]["periods"][i]
            # )
            n = (i + 1) * (nper + 1)
            #     * np.max(param["basis_function"]["periods"])
            #     / param["basis_function"]["periods"][i]
            # )
            t_expans[i, :] = np.sin(np.pi / 2 * n.T)

    elif param["basis_function"]["method"] == "chebyshev":
        t_expans = np.polynomial.chebyshev.chebval(nper, mod)
    elif param["basis_function"]["method"] == "legendre":
        t_expans = np.polynomial.legendre.legval(nper, mod)
    elif param["basis_function"]["method"] == "laguerre":
        t_expans = np.polynomial.laguerre.lagval(nper, mod)
    elif param["basis_function"]["method"] == "hermite":
        t_expans = np.polynomial.hermite.hermval(nper, mod)
    elif param["basis_function"]["method"] == "ehermite":
        t_expans = np.polynomial.hermite_e.hermeval(nper, mod)
    elif param["basis_function"]["method"] == "polynomial":
        t_expans = np.polynomial.polynomial.polyval(nper, mod)

    return t_expans


def print_message(res, j, mode, ref):
    """Prints the messages during the computation

    Args:
        * res (dict): result from the fitting algorithm
        * j (int): no. of iteration
        * mode (list): the current mode

    Returns:
        None
    """
    print("mode:     " + str(mode))
    print("fun:      " + str(res["fun"]))

    if (ref < 0) & (res["fun"] < 0):
        improve = np.round((ref - res["fun"]) / -ref * 100, decimals=3)
        if improve > 0:
            improve = "Yes (" + str(improve) + " %)"
        else:
            improve = "No or not significant "
    elif (ref > 0) & (res["fun"] > 0):
        improve = np.round((ref - res["fun"]) / ref * 100, decimals=3)
        if improve > 0:
            improve = "Yes (" + str(improve) + " %)"
        else:
            improve = "No or not significant "
    elif (ref > 0) | (res["fun"] < 0):
        improve = np.round((ref - res["fun"]) / ref * 100, decimals=3)
        improve = "Yes (" + str(improve) + " %)"
    else:
        improve = "No or not significant "

    print("improve:  " + improve)
    print("message:  " + str(res["message"]))
    print("feval:    " + str(res["nfev"]))
    print("niter:    " + str(res["nit"]))
    print("giter:    " + str(j))
    print("params:   " + str(res["x"]))
    print(
        "=============================================================================="
    )

    return


def emp(data, pemp, variable, wind_length=1 / 50):
    """Computes the non-stationary empirical percentile of data

    Args:
        * data (pd.DataFrame): raw time series
        * pemp (list): list with the empirical percentiles to be computed
        * variable (string): the name of the variable
        * wind_length (int, optional): window length in term of the normalize time. Defaults to 1/50.

    Returns:
        * data (pd.DataFrame): the data with new columns of percentiles
    """

    thresholds = []

    for i, _ in enumerate(pemp):
        thresholds.append("u" + str(i))
        data["u" + str(i)] = 0

    for i in data["n"].unique():
        data.loc[data["n"] == i, thresholds] = (
            data[variable]
            .loc[(data["n"] >= i - wind_length) & (data["n"] <= i + wind_length)]
            .quantile(q=pemp)
            .values
        )

    return data


class wrap_norm(st.rv_continuous):
    """ Wrapped normal probability model"""

    def __init__(self):
        """Initializate the main parameters and properties"""
        self.loc = 0
        self.scale = 0
        self.x = 0
        self.numargs = 0
        self.no = 100
        self.name = "wrap_norm"

    def pdf(self, x, *args):
        """Compute the probability density

        Args:
            x ([type]): data

        Returns:
            [type]: [description]
        """

        if args:
            self.loc = args[0][0]
            self.scale = args[0][1]

        n_ = np.linspace(-50, 50, self.no)
        if isinstance(x, (float, int)):
            self.len = 1
        else:
            self.len = len(x)
        n_ = np.tile(n_, (self.len, 1)).T

        w = np.tile(np.exp(1j * np.pi * (x - self.loc) / (2 * np.pi)), (self.no, 1))
        q = np.tile(np.exp(-self.scale ** 2 / 2), (self.no, 1))
        f = (w ** 2) ** n_ * q ** (n_ ** 2)

        f = np.sum(f, axis=0) / (2 * np.pi)
        return np.abs(f)

    def logpdf(self, x, *args):
        """Compute the logarithmic probability density

        Args:
            x ([type]): data

        Returns:
            [type]: [description]
        """
        if args:
            self.loc = args[0][0]
            self.scale = args[0][1]

        lpdf = np.log(self.pdf(x))
        return lpdf

    def cdf(self, x, loc, scale):
        """Compute the cumulative distribution

        Args:
            x ([type]): [description]
            loc ([type]): [description]
            scale ([type]): [description]

        Returns:
            [type]: [description]
        """

        self.loc = loc
        self.scale = scale

        if isinstance(x, (float, int)):
            self.len = 1
        else:
            self.len = len(x)

        cdf_ = np.zeros(self.len)

        if self.len == 1:
            cdf_ = quad(self.pdf, 0, x)[0]
        else:
            for ind_, val_ in enumerate(x):
                cdf_[ind_] = quad(self.pdf, 0, val_)[0]
        return cdf_

    def nllf(self, x0):
        """Compute the negative likelyhood function

        Args:
            x0 ([type]): [description]

        Returns:
            [type]: [description]
        """
        nllf_ = np.abs(-np.sum(self.logpdf(self.x, x0)))
        return nllf_

    def fit(self, x):
        """Fit the loc and scale parameters to the data

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.loc = np.mean(x)
        self.scale = np.std(x)
        self.x = x
        x0 = [self.loc, self.scale]

        res = minimize(
            self.nllf, x0, bounds=[(0, 2 * np.pi), (0, 2 * np.pi)], method="SLSQP"
        )
        return res["x"]

    def name():
        """[summary]

        Returns:
            [type]: [description]
        """
        return "wrap_norm"
