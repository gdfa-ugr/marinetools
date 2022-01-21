import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from marinetools.temporal import analysis
from marinetools.temporal.fdist import statistical_fit as stf


def nonstationary_ecdf(
    data: pd.DataFrame,
    variable: str,
    wlen: float = 14 / 365.25,
    equal_windows: bool = False,
    pemp: list = None,
):
    """Computes the empirical percentiles selecting a moving window

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable
        * wlen (float): length of window in days. Defaults to 14 days.
        * pemp (list, optional): given empirical percentiles

    Returns:
        * res (pd.DataFrame): values of the given non-stationary percentiles
        * pemp (list): chosen empirical percentiles
    """

    timestep = 1 / 365.25
    if equal_windows:
        timestep = wlen

    if pemp is None:
        pemp = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        if variable.lower().startswith("d") | (variable == "Wd"):
            pemp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        if (variable == "Hs") | (variable == "Hm0"):
            pemp = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995])

    res = pd.DataFrame(0, index=np.arange(0, 1, timestep), columns=pemp)
    for i in res.index:
        if i >= (1 - wlen):
            final_offset = i + wlen - 1
            mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
                data["n"] <= final_offset
            )
        elif i <= wlen:
            initial_offset = i - wlen
            mask = ((data["n"] >= i - wlen) & (data["n"] <= i + wlen)) | (
                data["n"] >= 1 + initial_offset
            )
        else:
            mask = (data["n"] >= i - wlen) & (data["n"] <= i + wlen)
        res.loc[i, pemp] = data[variable].loc[mask].quantile(q=pemp).values

    return res, pemp


def ecdf(df: pd.DataFrame, variable: str, no_perc: int or bool = False):
    """Computes the empirical cumulative distribution function

    Args:
        * df (pd.DataFrame): raw time series
        * variable (pd.DataFrame): name of the variable
        * no_perc (optional, False|integer): number of empirical percentiles to be interpolated. Defaults all data

    Returns:
        * dfs (pd.DataFrame): sorted values of the time series and the non-excedence probability of every value
    """
    dfs = df[variable].sort_values().to_frame()
    dfs["prob"] = np.arange(1, len(dfs) + 1) / (len(dfs) + 1)
    if not isinstance(no_perc, bool):
        percentiles = np.linspace(1 / no_perc, 1 - (1 / no_perc), no_perc)
        values = np.interp(percentiles, dfs["prob"], dfs[variable])
        dfs = pd.DataFrame(values, columns=[variable], index=percentiles)
    return dfs


def nonstationary_epdf(
    data: pd.DataFrame, variable: str, wlen: float = 14 / 365.25, no_values: int = 14
):
    """Computes the empirical percentiles selecting a moving window

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable
        * wlen (float): length of window in days. Defaults to 14 days.

    Returns:
        * res (pd.DataFrame): values of the given non-stationary percentiles
        * pemp (list): chosen empirical percentiles
    """

    nlen = len(data)
    ndates = np.arange(0, 1, 1 / 365.25)
    values_ = np.linspace(data[variable].min(), data[variable].max(), no_values)
    pdf_ = pd.DataFrame(-1, index=ndates, columns=(values_[:-1] + values_[1:]) / 2)

    columns = pdf_.columns
    for ind_, col_ in enumerate(columns[:-1]):
        for i in pdf_.index:
            if i > (1 - wlen):
                final_offset = i + wlen - 1
                mask = (
                    (data["n"] > i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                ) | (data["n"] <= final_offset)
            elif i < wlen:
                initial_offset = i - wlen
                mask = (
                    (data["n"] >= i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                ) | (data["n"] >= 1 + initial_offset)
            else:
                mask = (
                    (data["n"] >= i - wlen)
                    & (data["n"] <= i + wlen)
                    & (data[variable] > col_)
                    & (data[variable] <= columns[ind_ + 1])
                )
            pdf_.loc[i, col_] = np.sum(mask) / nlen

    return pdf_


def epdf(df: pd.DataFrame, variable: str, no_values: int = 14):
    """Computes the empirical probability distribution function

    Args:
        * df (pd.DataFrame): raw time series
        * variable (pd.DataFrame): name of the variable
        * no_values (optional, False|integer): number of empirical percentiles to be interpolated. Defaults all data

    Returns:
        * dfs (pd.DataFrame): sorted values of the time series and the probability
    """
    dfs = df[variable].sort_values().to_frame()
    dfs["prob"] = np.arange(1, len(dfs) + 1)
    count_ = pd.DataFrame(-1, index=dfs[variable].unique(), columns=["prob"])

    for _, ind_ in enumerate(count_.index):
        count_.loc[ind_] = np.sum(dfs[variable] == ind_)

    values_ = np.linspace(df[variable].min(), df[variable].max(), no_values)
    pdf_ = pd.DataFrame(-1, index=(values_[:-1] + values_[1:]) / 2, columns=["prob"])
    for ind_, index_ in enumerate(pdf_.index):
        # range_ = np.interp(values_, pdf_["prob"], pdf_[variable])
        val_ = np.sum(
            count_.loc[
                ((count_.index < values_[ind_ + 1]) & (count_.index > values_[ind_]))
            ].values
        )
        pdf_.loc[index_, "prob"] = val_
    pdf_.loc[:, "prob"] = pdf_["prob"] / (np.sum(pdf_).values * np.diff(values_))
    return pdf_


def acorr(data, maxlags=24):
    """[summary]

    Args:
        data ([type]): [description]
        maxlags (int, optional): [description]. Defaults to 24.

    Returns:
        [type]: [description]
    """

    lags, c_, _, _ = plt.acorr(data, usevlines=False, maxlags=maxlags, normed=True)
    plt.close()
    lags, c_ = lags[-maxlags:], c_[-maxlags:]
    return lags, c_


def str2fun(param: dict, var_: str = None):
    """Create an object of scipy.stats function given a string with the name

    Args:
        * param (dict): dictionary with parameters
        * var_ (str): name of the variable

    Return:
        * The input dictionary updated
    """

    if var_ is None:
        for key in param["fun"].keys():
            if isinstance(param["fun"][key], str):
                if param["fun"][key] == "wrap_norm":
                    param["fun"][key] = stf.wrap_norm()
                else:
                    param["fun"][key] = getattr(st, param["fun"][key])
    else:
        for key in param[var_]["fun"].keys():
            if isinstance(param[var_]["fun"][key], str):
                if param[var_]["fun"][key] == "wrap_norm":
                    param[var_]["fun"][key] = stf.wrap_norm()
                else:
                    param[var_]["fun"][key] = getattr(st, param[var_]["fun"][key])

    return param


def pre_ensemble_plot(models: list, param: dict, var_: str):
    """Compute the ppf, mean and std of RCP-GCM models given

    Args:
        * models (list): name of models as saved in param
        * param (dict): statistical parameters of all the models
        * var_ (str): name of the variable

    Returns:
        ppfs (pd.DataFrame): ppf, mean and std
    """

    probs = [0.05, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]
    if var_.lower().startswith("d"):
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n = np.linspace(0, 1, 24 * 365.25)

    ppfs = dict()
    for prob in probs:
        df = pd.DataFrame(np.ones(len(n)) * prob, index=n, columns=["prob"])
        df["n"] = n
        ppfs[str(prob)] = df.copy()
        for model in models:
            param[var_][model] = str2fun(param[var_][model], None)
            res = stf.ppf(df.copy(), param[var_][model])
            # Transformed timeserie if required
            if param[var_][model]["transform"]["make"]:
                res[var_] = analysis.inverse_transform(res[[var_]], param[var_][model])
            ppfs[str(prob)].loc[:, model] = res[var_]

        ppfs[str(prob)]["mean"] = ppfs[str(prob)].loc[:, models].mean(axis=1)
        ppfs[str(prob)]["std"] = ppfs[str(prob)].loc[:, models].std(axis=1)

    return ppfs


def mkdir(path: str):
    """Create a folder with path name if not exists

    Args:
        path (str): name of the new folder
    """
    if path != "":
        os.makedirs(path, exist_ok=True)
    return
