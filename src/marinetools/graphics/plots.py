import datetime
from itertools import product

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from marinetools.graphics.utils import handle_axis, labels, show
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.temporal.fdist.copula import Copula
from marinetools.utils import auxiliar
from pandas.plotting import register_matplotlib_converters

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

register_matplotlib_converters()

cmp_g = cmocean.cm.haline_r
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=10)
params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}


def plot_mda(data, cases, variables, title=None, ax=None, fname=None):
    """Plots a 3D scatter figure of the MDA cases over the data cloud of points

    Args:
        * data (pd.DataFrame): time series of the data
        * cases (pd.DataFrame): mda cases
        * variables (list): the name of the variables
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if len(variables) == 1:
        ax = handle_axis(ax)
        ax.plot(data[variables[0]].values, marker=".", alpha=0.05)
        ax.plot(cases[variables[0]].values, color="k", marker="o")
        ax.set_xlabel("time")
        ax.set_ylabel(labels(variables[0]))

    elif len(variables) == 2:
        ax = handle_axis(ax)

    elif len(variables) == 3:
        ax = handle_axis(ax, dim=3)

        ax.scatter(
            data[variables[0]].values,
            data[variables[1]].values,
            data[variables[2]].values,
            marker=".",
            alpha=0.05,
        )
        ax.scatter(
            cases[variables[0]].values,
            cases[variables[1]].values,
            cases[variables[2]].values,
            color="k",
            marker="o",
        )
        ax.set_xlabel(labels(variables[0]))
        ax.set_ylabel(labels(variables[1]))
        ax.set_zlabel(labels(variables[2]))
    else:
        comb = [x[::-1] for x in product(range(0, len(variables)), repeat=2)]
        removed = []
        for i in comb:
            if i[1] <= i[0]:
                removed.append(i)

        _, ax = plt.subplots(len(variables), len(variables), figsize=(20, 20))
        for i in comb:
            if i in removed:
                ax[i[0], i[1]].axis("off")
            else:
                ax[i[0], i[1]].scatter(
                    data[variables[i[1]]].values,
                    data[variables[i[0]]].values,
                    marker=".",
                    alpha=0.05,
                )
                ax[i[0], i[1]].scatter(
                    cases[variables[i[1]]].values,
                    cases[variables[i[0]]].values,
                    color="k",
                    marker=".",
                )

            if i[0] + 1 == i[1]:
                ax[i[0], i[1]].set_ylabel(labels(variables[i[0]]))
                ax[i[0], i[1]].set_xlabel(labels(variables[i[1]]))

    if title is not None:
        ax.set_title(title)

    show(fname)

    return ax


def timeseries(data: pd.DataFrame, variable: str, ax=None, file_name: str = None):
    """Plots the time series of the variable

    Args:
        * data (pd.DataFrame): time series
        * variable (string): variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    ax.plot(data.loc[:, variable])
    try:
        ax.set_ylabel(labels(variable))
    except:
        ax.set_ylabel(variable)

    show(file_name)
    return ax


def storm_timeseries(
    df_sim: pd.DataFrame, df_obs: pd.DataFrame, variables: list, file_name: str = None
):
    """Plots the time series of simulation and observations (from differents RCM) for the choosen variables

    Args:
        * df_sim (pd.DataFrame): simulated time series
        * df_obs (dict): any key of the dict should be a pd.DataFrame of observed time series
        * variables (list): name of the variables
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if not isinstance(df_sim, pd.DataFrame):
        df_sim = df_sim.to_frame()

    _, ax = plt.subplots(
        len(variables), 1, figsize=(12, len(variables) * 2), sharex=True
    )
    # if not isinstance(ax, list):
    #     ax = [ax]

    for i, j in enumerate(variables):
        if isinstance(df_obs, dict):
            for key in df_obs.keys():
                ax[i].plot(
                    df_obs[key][j], ".", ms=2, alpha=0.5, label=key.split("_")[0]
                )
            # ax[i].set_xlim([df_obs[key].index[0], df_obs[key].index[-1]])
        else:
            if not isinstance(df_obs, pd.DataFrame):
                df_obs = df_obs.to_frame()
            ax[i].plot(df_obs[j], label=j)
            # ax[i].set_xlim([df_obs[j].index[0], df_obs[j].index[-1]])
        ax[i].set_ylabel(labels(j))
        ax[i].plot(df_sim[j], ".", ms=2, alpha=0.5, label="Simulation")

        if i == 0:
            ax[i].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0 + 0.1 * len(variables)),
                ncol=4,
            )

    show(file_name)
    return ax


def test_normality(data, params, ax=None, fname=None):
    """[summary]

    Args:
        data ([type]): [description]
        params ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    data, params = stf.transform(data, params)
    ax, fname = handle_axis(ax)

    _, ax = plt.subplots(1, 1, figsize=(8, 4))

    st.probplot(data, dist=st.norm, plot=ax)

    ax.set_title("Normality test")
    ax.set_ylabel("")

    k2, p = st.normaltest(data)
    print("\nChi-squared statistic = %.3f, p = %.3f" % (k2, p))

    alpha = 0.05
    if p > alpha:
        print(
            "\nThe transformed data is Gaussian (fails to reject the null hypothesis)"
        )
    else:
        print(
            "\nThe transformed data does not look Gaussian (reject the null hypothesis)"
        )
    show(fname)

    return ax


def cadency(data, label="", ax=None, legend=False, fname=None):
    """Plots the temporal difference (cadency) of time series

    Args:
        * data (pd.DataFrame): raw time series
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    ax.plot(
        data.index[1:], (data.index[1:] - data.index[:-1]).seconds / 3600, label=label
    )
    ax.set_ylabel("Cadency time (hr)")

    if legend:
        ax.legend()

    show(fname)
    return ax


def plot_spectra(
    data,
    semilog=True,
    title=None,
    fname=None,
    ax=None,
    xlabel="Periods $T$ (years)",
    label="LombScargle",
):
    """Plots the power spectral density

    Args:
        * data (pd.DataFrame): frequencies are in the index while in column should be the magnitude
        * signif (pd.DataFrame): frequencies are in the index while in column should be the magnitude
        * semilog (bool, optional): Type of representation. Defaults to True.
        * title (string, optional): Title of the plot. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * label (str, optional): Type of spectra. Defaults to 'LombScargle'.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if semilog:
        ax.semilogy(data["psd"], label=label)
    else:
        ax.plot(data["psd"], label=label)
    ax.set_xlabel(xlabel)
    if label == "LombScargle":
        ax.set_ylabel("Normalized Lombscargle Periodogram")
    else:
        ax.set_ylabel("Fast Fourier Transform")
    ax.plot(data.loc[data["significant"], "psd"], "bo", label="significant")

    # for index in data.loc[data['significant'], 'psd'].index:
    #     ax.annotate('{:.2f}'.format(index), (index, data.loc[index, 'psd']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.legend(loc="best")
    # ax.set_xlim(left=0.05)
    ax.grid()
    if title is not None:
        ax.set_title(title)
    show(fname)

    return ax


def plot_cdf(
    data: pd.DataFrame,
    var: str,
    ax=None,
    file_name: str = None,
    seaborn: bool = False,
    legend: str = False,
    label: str = None,
):
    """Plots the cumulative density function of data

    Args:
        * data (pd.DataFrame): raw time series
        * var (string): name of the variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * seaborn (bool): True or False if seaborn cdf plot is required
        * legend (bool): True if legend want to be plot
        * label (str): name of the variable

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    if not isinstance(label, str):
        label = labels(var)

    if seaborn:
        import seaborn as sns

        sns.distplot(
            data[var],
            hist_kws={"cumulative": True},
            kde_kws={"cumulative": True},
            ax=ax,
        )
    else:
        emp = auxiliar.ecdf(data, var)
        ax.plot(emp[var], emp["prob"], label=label)

    if legend:
        ax.legend()

    # show(file_name)
    return ax


def boxplot(data: pd.DataFrame, variable: str, ax=None, file_name: str = None):
    """Draws a box-plot of the data

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    import seaborn as sns

    data["date"] = data.index
    data["month"] = data["date"].dt.strftime("%b")
    sns.boxplot(x="month", y=variable, data=data, ax=ax)

    return ax


def qq(
    data,
    prob_model="norm",
    marker=".",
    color="k",
    line="45",
    ax=None,
    label=None,
    fname=None,
):
    """Draws a QQ-plot of data and the distribution provided

    Args:
        * data (pd.DataFrame): raw time series
        * prob_model (str, optional): Name of the scipy.stats probability model. Defaults to 'norm'.
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * label (None or string, optional): variable name. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """
    import statsmodels.api as sm

    ax = sm.qqplot(
        data.values,
        dist=getattr(st, prob_model),
        fit=True,
        line=line,
        marker=marker,
        color=color,
        ax=ax,
        label=label,
    )
    show(fname)

    return ax


def probplot(data, ax=None, fit=True, fname=None):
    """Plots the probability-plot of statsmodels

    Args:
        * data (pd.DataFrame): [description]
        * ax (matplotlib.axis, optional): A given axis. Defaults to None.
        * fit (bool, optional): [description]. Defaults to True.

    Returns:
        * ax (matplotlib.axis): The axis with the plot or None
    """
    import statsmodels.api as sm

    probpl = sm.ProbPlot(data, fit=fit)
    probpl.probplot(ax=ax)
    show(fname)

    return ax


def nonstationary_percentiles(
    data: pd.DataFrame, variable: str, fun: str, pars=None, ax=None, fname=None
):
    """Plots monthly cdf in a normalize sheet

    Args:
        * data (pd.DataFrame): raw time series
        * fun (string): name of the probability model according to scipy.stats
        * variable (string): variable name
        * pars (None or dict, optional): If provided, it stands for the parameters of the cdf function. Defaults to None.
        * ax (matplotlib.axis, optional): axis for the plot or NoneDefaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    data["n"] = (
        (data.index.dayofyear + data.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": data.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    monthly_window = 1 / 12
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    if not pars:
        pars = getattr(st, fun).fit(data[variable])

    t, l = 0, 0
    while t < 1.0:
        aux = data.loc[((data["n"] > t) & (data["n"] < t + monthly_window))][variable]
        # try:
        n = len(aux)
        sorted_ = np.sort(aux)

        x = np.linspace(0, aux.max(), 100)
        cdf = getattr(st, fun).cdf(x, *pars)
        cdfe = np.interp(x, sorted_, np.linspace(0, 1, n))
        plt.plot(st.norm.ppf(cdfe), st.norm.ppf(cdf), label=months[l])
        # except:
        #     pass
        t += monthly_window
        l += 1

    ax.set_xticks(st.norm.ppf([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    ax.set_xticklabels([1, 5, 10, 25, 50, 75, 90, 95, 99])
    ax.set_yticks(st.norm.ppf([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    ax.set_yticklabels([1, 5, 10, 25, 50, 75, 90, 95, 99])
    ax.set_xlabel(r"Empirical percentiles")
    ax.set_ylabel(r"Theoretical percentiles")
    ax.grid(True)
    ax.legend()
    ax.plot([-3, 3], [-3, 3], "k")
    show(fname)
    return


def nonstationary_qq_plot(
    data: pd.DataFrame, var_: str, prob_model: str = "norm", fname=None
):
    """Draws a monthly QQ-plot of data

    Args:
        * data (pd.DataFrame): raw time series
        * prob_model (str, optional): name of the scipy.stats probability model. Defaults to 'norm'.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(10, 8))
    axs = axs.flatten()

    data["n"] = (
        (data.index.dayofyear + data.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": data.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    monthly_window = 1 / 12
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    t, l = 0, 0
    while t < 1.0:
        aux = data.loc[((data["n"] > t) & (data["n"] < t + monthly_window)), var_]
        try:
            qq(aux, prob_model, ax=axs[l], fname="to_axes")
        except:
            pass

        axs[l].text(0.05, 0.85, months[l], transform=axs[l].transAxes, weight="bold")
        t += monthly_window
        l += 1

    show(fname)

    return


def scatter_error_dependencies(
    df_dt: dict, variables: list, label: str, ax=None, file_name: str = None
):
    """Draws the scatter plot of data before and after the computation of the temporal dependency

    Args:
        * df_dt (dict): parameters of the temporal dependency package
        * variables (list): names of the variables
        * ax (matplotlib.axis, optional): axis for the plot or NoneDefaults to None.
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)

    if isinstance(variables, str):
        variables = [variables]

    # _, ax = plt.subplots(1, 1)
    for i, j in enumerate(variables):
        if len(variables) == 1:
            dfy = df_dt["y"][i]
            dfy_ = df_dt["y*"][i]
        else:
            dfy = df_dt["y"][i]
            dfy_ = df_dt["y*"][i]

        ax.plot(
            dfy,
            dfy_,
            ".",
            label=label[i],
        )
    ax.grid()
    ax.set_xlabel(r" Observed ($\Phi^{-1}(F_{\zeta}(\zeta))$")
    ax.set_ylabel(r" Modeled ($\Phi^{-1}(F_{\zeta}(\zeta))$")
    ax.legend(loc=4, title=r"$\zeta$")
    show(file_name)
    return ax


def scatter(df1, df2, variables, names=["Observed", "Modeled"], fname=None, ax=None):
    """Draws the scatter plot of data before and after the computation of the temporal dependency

    Args:
        * df_dt (dict): parameters of the temporal dependency package
        * variables (list): names of the variables
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, axs = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 5))
    if len(variables) != 1:
        axs.flatten()
    else:
        axs = list(axs)

    for index, variable in enumerate(variables):
        axs[index].plot(df1[variable], df2[variable], ".", label=variable)
        axs[index].grid()
        axs[index].set_xlabel(r" " + names[0])
    axs[0].set_ylabel(r" " + names[1])
    axs[0].legend(loc=4, title=r"$\zeta$")
    show(fname)
    return axs


def look_models(data, variable, params, num=10, fname=None):
    """Plots the pdf of data using several models to check the best guess

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable
        * params (dict): with the parameters of every fitted model and the sum of square errors made
        * num (int, optional): number of best model estimations to plot. Defaults to 10.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, ax = plt.subplots(1, 1)
    emp = auxiliar.ecdf(data, variable)
    plt.plot(emp[variable], emp["prob"], label="empirical cdf")

    x = np.linspace(data[variable].min(), data[variable].max(), 1000)

    num = np.min([num, len(params)])
    for i in range(num):
        prob_model = getattr(st, params.iloc[i, 0])
        parameters = params.iloc[i, 2 : prob_model.numargs + 4].values
        try:
            ax.plot(
                x,
                prob_model.cdf(x, *parameters),
                label=params.iloc[i, 0].replace("_", " "),
            )
        except:
            continue
    ax.set_xlabel(labels(variable))
    ax.set_ylabel("prob")
    ax.legend(ncol=2)
    show(fname)

    return ax


def crosscorr(xy, xys, variable, lags=48, fname=None):
    """Plots the cross-correlation of variables xy and xys

    Args:
        * xy (np.ndarray): time series
        * xys (np.ndarray): time series
        * variable (string): name of variable for the plot
        * lags (int): maximum lag time (hours). Defaults to 48 hours.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, ax = plt.subplots(1, 1)
    tiempo = np.arange(-lags, lags, 3 / 24.0)
    i1, i2 = len(xy[0]) / 2.0 - len(tiempo) / 2.0, len(xy[0]) / 2.0 + len(tiempo) / 2.0
    i1s, i2s = (
        len(xys[0]) / 2.0 - len(tiempo) / 2.0,
        len(xys[0]) / 2.0 + len(tiempo) / 2.0,
    )
    ccf = np.correlate(xy[0], xy[1], mode="same")
    ccf = ccf / len(ccf) - np.mean(xy[0]) * np.mean(xy[1])
    ccf = ccf / (np.std(xy[0]) * np.std(xy[1]))
    ccf2 = np.correlate(xys[0], xys[1], mode="same")
    ccf2 = ccf2 / len(ccf2) - np.mean(xys[0]) * np.mean(xys[1])
    ccf2 = ccf2 / (np.std(xys[0]) * np.std(xys[1]))

    plt.figure()
    plt.plot(tiempo, ccf[i1:i2], ".", color="gray")
    plt.plot(tiempo, ccf2[i1s:i2s], "k")
    plt.ylim(ymax=1)

    plt.legend(("Observed", "Simulated"), loc="best")
    plt.xlabel("Time [days]")
    plt.ylabel("Cross-correlation of " + variable)
    plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
    show(fname)

    return ax


def corr(data: pd.DataFrame, lags: int = 24, ax=None, file_name: str = None):
    """Plots the correlation of between time series

    Args:
        * data (pd. DataFrame): timeseries
        * lags (int): maximum lag time (hours). Defaults to 48 hours.
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    ax.acorr(data, usevlines=False, maxlags=lags, normed=True, lw=2)

    ax.set_xlabel(r"Lags (hr)")
    ax.set_ylabel(r"Normalized autocorrelation")
    ax.grid(True)
    ax.legend()
    show(file_name)

    return ax


def joint_plot(data: pd.DataFrame, varx: str, vary: str, ax=None):
    """Plots the joint probability function of data

    Args:
        * data (pd.DataFrame): time series with the variables
        * varx (string): name of main variable
        * vary (string): name of secondary variable
        * ax (matplotlib.axis): axis for the plot or None. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    import seaborn as sns

    sns.jointplot(x=varx, y=vary, data=data, ax=ax)

    return ax


def bivariate_ensemble_pdf(
    df_sim: pd.DataFrame, df_obs: dict, varp: list, file_name: str = None
):
    """Plots together the bivariate probability density function of several time series observations and one simulation

    Args:
        * df_sim (pd.DataFrame): raw time series
        * df_obs (dict): each element is an observed time series
        * varp (list): names of the variables
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))
    row, column = 0, 0
    H, x, y = np.histogram2d(
        df_sim[varp[0]], df_sim[varp[1]], bins=[25, 25], density=True
    )
    x, y = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    y, x = np.meshgrid(y, x)
    levels = np.linspace(np.max(H) / 8, np.max(H), 8)
    ax[row, column].contourf(x, y, H, alpha=0.25, levels=np.append(0, levels))
    CS = ax[row, column].contour(x, y, H, levels=levels)
    ax[row, column].clabel(CS, inline=1, fontsize=10)
    ax[row, column].set_title("SIMULATION")
    ax[row, column].set_ylabel(labels(varp[1]))
    ax[row, column].grid(True)
    ax[0, 1].axis("off")
    column += 2

    for key in df_obs.keys():
        Ho, xo, yo = np.histogram2d(
            df_obs[key][varp[0]], df_obs[key][varp[1]], bins=[25, 25], density=True
        )
        xo, yo = (xo[:-1] + xo[1:]) / 2, (yo[:-1] + yo[1:]) / 2
        yo, xo = np.meshgrid(yo, xo)
        ax[row, column].contourf(xo, yo, Ho, alpha=0.25, levels=np.append(0, levels))
        CS = ax[row, column].contour(x, y, H, levels=levels)
        ax[row, column].clabel(CS, inline=1, fontsize=10)
        ax[row, column].set_title(key.split("_")[0])
        ax[row, column].grid(True)
        if column == 0:
            ax[row, column].set_ylabel(labels(varp[1]))

        if row == 2:
            ax[row, column].set_xlabel(labels(varp[0]))
        column += 1

        if column >= 3:
            column = 0
            row += 1
    show(file_name)
    return


def bivariate_pdf(
    df_sim: pd.DataFrame,
    df_obs: pd.DataFrame,
    variables: list,
    bins: int = None,
    levels: list = None,
    ax=None,
    file_name: str = None,
    logx: str = False,
    logy: str = False,
    contour: bool = False,
):
    """Plots the bivariate distribution function between observed and simulated time series for comparison

    Args:
        * df_sim (pd.DataFrame): simulated data with the main and secondary time series
        * df_obs (pd.DataFrame): observed data with the main and secondary time series
        * variables (list): string with the name of the variables
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax, col_plots=2)

    if bins is None:
        bins = [25, 25]
    if logy:
        df_obs[variables[0]] = np.log(df_obs[variables[0]])
        df_sim[variables[0]] = np.log(df_sim[variables[0]])

    if logx:
        df_obs[variables[1]] = np.log(df_obs[variables[1]])
        df_sim[variables[1]] = np.log(df_sim[variables[1]])

    nn, nx_, ny_ = np.histogram2d(
        df_obs[variables[0]], df_obs[variables[1]], bins=bins, density=True
    )
    ny, nx = np.meshgrid(ny_[:-1] + np.diff(ny_) / 2.0, nx_[:-1] + np.diff(nx_) / 2.0)

    if levels is None:
        levels = np.linspace(np.max(nn) / 12, np.max(nn), 12)

    min_, max_ = np.min(nn), np.max(nn)

    ax[0].imshow(
        np.flipud(nn),
        extent=[np.min(ny), np.max(ny), np.min(nx), np.max(nx)],
        vmin=min_,
        vmax=max_,
        cmap="viridis",
        aspect="auto",
    )

    labelx = labels(variables[1])
    labely = labels(variables[0])
    if logx:
        labelx = "log " + labelx

    if logy:
        labely = "log " + labely

    ax[0].set_xlabel(labelx)
    ax[0].set_ylabel(labely)

    nn_, nx, ny = np.histogram2d(
        df_sim[variables[0]], df_sim[variables[1]], bins=[nx_, ny_], density=True
    )
    ny, nx = np.meshgrid(ny[:-1] + np.diff(ny) / 2.0, nx[:-1] + np.diff(nx) / 2.0)
    cs = ax[1].imshow(
        np.flipud(nn_),
        extent=[np.min(ny), np.max(ny), np.min(nx), np.max(nx)],
        vmin=min_,
        vmax=max_,
        cmap="viridis",
        aspect="auto",
    )
    ax[1].set_xlabel(labelx)
    ax[1].set_yticklabels([])

    show(file_name)

    # R2 = 1 - np.sum((np.ravel(nn) - np.ravel(nn_)) ** 2) / np.sum(
    #     (np.ravel(nn_) - np.mean(np.ravel(nn_))) ** 2
    # )
    # print(R2)

    return ax


def nonstationary_cdf(
    data: pd.DataFrame,
    variable: str,
    param: dict = None,
    daysWindowsLength: int = 14,
    equal_windows: bool = False,
    ax=None,
    log: bool = False,
    file_name: str = None,
    label: str = None,
    legend: bool = True,
    legend_loc: str = "right",
    title: str = None,
    date_axis: bool = False,
    pemp: list = None,
):
    """Plots the time variation of given percentiles of data and theoretical function if provided

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable to be adjusted
        * param (dict, optional): the parameters of the the theoretical model if they are also plotted.
        * daysWindowsLength (int, optional): period of windows length for making the non-stationary empirical distribution function. Defaults to 14 days.
        * equal_windows (bool): use the windows for the ecdf of total data and timestep
        * ax: matplotlib.ax
        * log: logarhitmic scale
        * file_name (string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * label: string with the label
        * legend: plot the legend
        * legend_loc: locate the legend
        * title: draw the title
        * date_axis: create a secondary axis with time
        * pemp: list with percentiles to be plotted

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    T = 1
    if param is not None:
        if param["basis_period"] is not None:
            T = np.max(param["basis_period"])

    data["n"] = np.fmod(
        (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    dt = 366
    n = np.linspace(0, 1, dt)
    xp, pemp = auxiliar.nonstationary_ecdf(
        data,
        variable,
        wlen=daysWindowsLength / (365.25 * T),
        equal_windows=equal_windows,
        pemp=pemp,
    )

    ax = handle_axis(ax)

    ax.set_prop_cycle("color", [plt.cm.winter(i) for i in np.linspace(0, 1, len(pemp))])
    col_per = list()

    if len(xp.index.unique()) > 60:
        marker, ms, markeredgewidth = ".", 8, 1.5
    else:
        marker, ms, markeredgewidth = "+", 4, 1.5

    for j, i in enumerate(pemp):
        if isinstance(param, dict):
            if param["transform"]["plot"]:
                xp[i], _ = stf.transform(xp[[i]], param)
                xp[i] -= param["transform"]["min"]
                if "scale" in param:
                    xp[i] = xp[i] / param["scale"]
        if log:
            p = ax.semilogy(
                xp[i],
                marker=marker,
                ms=ms,
                markeredgewidth=markeredgewidth,
                lw=0,
                label=str(i),
            )
        else:
            p = ax.plot(
                xp[i],
                marker=marker,
                ms=ms,
                markeredgewidth=markeredgewidth,
                lw=0,
                label=str(i),
            )
        col_per.append(p[0].get_color())

    if isinstance(param, dict):
        if param["status"] == "Distribution models fitted succesfully":
            param = auxiliar.str2fun(param, None)

            for i, j in enumerate(pemp):
                df = pd.DataFrame(np.ones(dt) * pemp[i], index=n, columns=["prob"])
                df["n"] = n
                if (param["non_stat_analysis"] == True) | (param["no_fun"] > 1):
                    res = stf.ppf(df, param)
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
                    res[param["var"]] = stf.inverse_transform(
                        res[[param["var"]]], param
                    )
                elif ("scale" in param) & (not param["transform"]["plot"]):
                    res[param["var"]] = res[param["var"]] * param["scale"]

                if log:
                    ax.semilogy(
                        res[param["var"]].index,
                        res[param["var"]].values,
                        color=col_per[i],
                        lw=2,
                        label=str(j),
                    )
                else:
                    if param["circular"]:
                        ax.plot(
                            res[param["var"]].index,
                            np.rad2deg(res[param["var"]].values),
                            color=col_per[i],
                            lw=2,
                            label=str(j),
                        )
                    else:
                        ax.plot(
                            res[param["var"]].index,
                            res[param["var"]].values,
                            color=col_per[i],
                            lw=2,
                            label=str(j),
                        )
        else:
            raise ValueError(
                "Model was not fit successfully. Look at the marginal fit."
            )

    ax.grid()

    if legend:
        # Shrink current axis
        box = ax.get_position()
        if param:
            if legend_loc == "bottom":
                ax.set_position([box.x0, box.y0, box.width, box.height])
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(-0.2, 0.0),
                    ncol=len(pemp),
                    title="Percentiles",
                )
                if param["circular"]:
                    ax.set_yticks([0, 90, 180, 270, 360])
            else:
                ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                # Put a legend to the right of the current axis
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=2,
                    title="Percentiles",
                )
                if param["circular"]:
                    ax.set_yticks([0, 90, 180, 270, 360])
        else:
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, title="Percentiles"
            )

    if isinstance(title, str):
        ax.set_title(title, color="k", fontweight="bold")

    if not label:
        label = labels(variable)

    if log:
        label = "log " + label
    ax.set_ylabel(label)
    ax.set_xlabel("Normalized period")
    if date_axis:
        ax2 = ax.twiny()
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        ax2.set_xticks(
            np.array(
                [
                    0.6 / 13,
                    1.5 / 13,
                    2.5 / 13,
                    3.5 / 13,
                    4.5 / 13,
                    5.5 / 13,
                    6.5 / 13,
                    7.5 / 13,
                    8.5 / 13,
                    9.5 / 13,
                    10.5 / 13,
                    11.5 / 13,
                    12.4 / 13,
                ]
            ),
            minor=False,
        )
        # 16 is a slight approximation since months differ in number of days.
        # ax2.xaxis.set_minor_locator(np.array([1/13, 2/13, 3/13, 4/13, 5/13, 6/13, 7/13, 8/13, 9/13, 10/13, 11/13, 12/13]))
        # ax2.xaxis.set_major_formatter(ticker.NullFormatter())

        # Hide major tick labels
        ax2.set_xticklabels("")

        # Customize minor tick labels
        ax2.set_xticks(
            np.array(
                [
                    1 / 13,
                    2 / 13,
                    3 / 13,
                    4 / 13,
                    5 / 13,
                    6 / 13,
                    7 / 13,
                    8 / 13,
                    9 / 13,
                    10 / 13,
                    11 / 13,
                    12 / 13,
                ]
            ),
            minor=True,
        )
        ax2.set_xticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], minor=True
        )
        ax2.tick_params(
            axis="x",  # changes apply to the x-axis
            which="minor",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True,
        )

        ax2.set_xlabel(r"Normal Year")
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width * 0.6, box.height * 0.9]
        )
    show(file_name)

    return ax


def nonstat_cdf_ensemble(
    data: pd.DataFrame,
    variable: str,
    param: dict = None,
    models: list = None,
    daysWindowsLength: int = 14,
    equal_windows: bool = False,
    ax=None,
    log: bool = False,
    file_name: str = None,
    label: str = None,
    legend: bool = True,
    legend_loc: str = "right",
    title: str = None,
    date_axis: bool = False,
    pemp: list = None,
):
    """Plots the time variation of given percentiles of data and theoretical function if provided

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable to be adjusted
        * param (dict, optional): the parameters of the the theoretical model if they are also plotted.
        * daysWindowsLength (int, optional): period of windows length for making the non-stationary empirical distribution function. Defaults to 14 days.
         * equal_windows (bool): use the windows for the ecdf of total data and timestep
        * ax: matplotlib.ax
        * log: logarhitmic scale
        * file_name (string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * label: string with the label
        * legend: plot the legend
        * legend_loc: locate the legend
        * title: draw the title
        * date_axis: create a secondary axis with time
        * pemp: list with percentiles to be plotted

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    T = 1
    if param is not None:
        if param[variable][models[0]]["basis_period"] is not None:
            T = np.max(param[variable][models[0]]["basis_period"])

    data["n"] = np.fmod(
        (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    dt = 366
    n = np.linspace(0, 1, dt)
    xp, pemp = auxiliar.nonstationary_ecdf(
        data,
        variable,
        wlen=daysWindowsLength / (365.25 * T),
        equal_windows=equal_windows,
        pemp=pemp,
    )

    ax = handle_axis(ax)

    ax.set_prop_cycle("color", [plt.cm.winter(i) for i in np.linspace(0, 1, len(pemp))])
    col_per = list()

    if len(xp.index.unique()) > 60:
        marker, ms, markeredgewidth = ".", 8, 1.5
    else:
        marker, ms, markeredgewidth = "+", 4, 1.5

    for j, i in enumerate(pemp):
        if isinstance(param, dict):
            if param[variable][models[0]]["transform"]["plot"]:
                xp[i], _ = stf.transform(xp[[i]], param[variable][models[0]])
                xp[i] -= param[variable][models[0]]["transform"]["min"]
                if "scale" in param:
                    xp[i] = xp[i] / param[variable][models[0]]["scale"]
        if log:
            p = ax.semilogy(
                xp[i],
                marker=marker,
                ms=ms,
                markeredgewidth=markeredgewidth,
                lw=0,
                label=str(i),
            )
        else:
            p = ax.plot(
                xp[i],
                marker=marker,
                ms=ms,
                markeredgewidth=markeredgewidth,
                lw=0,
                label=str(i),
            )
        col_per.append(p[0].get_color())

    if isinstance(param, dict):
        if (
            param[variable][models[0]]["status"]
            == "Distribution models fitted succesfully"
        ):
            # param = auxiliar.str2fun(param, None)

            for i, j in enumerate(pemp):
                df = pd.DataFrame(np.ones(dt) * pemp[i], index=n, columns=["prob"])
                df["n"] = n
                if (param[variable][models[0]]["non_stat_analysis"] == True) | (
                    param[variable][models[0]]["no_fun"] > 1
                ):
                    res = stf.ensemble_ppf(df, param, "pr", nodes=[4383, 900])
                else:
                    res = pd.DataFrame(
                        param[models[0]]["fun"][0].ppf(
                            df["prob"], *param[models[0]]["par"]
                        ),
                        index=df.index,
                        columns=[variable],
                    )

                # Transformed timeserie
                if (not param[variable][models[0]]["transform"]["plot"]) & param[
                    variable
                ][models[0]]["transform"]["make"]:
                    if "scale" in param:
                        res[param[variable][models[0]]["var"]] = (
                            res[param[variable][models[0]]["var"]]
                            * param[variable][models[0]]["scale"]
                        )

                    res[param[variable][models[0]]["var"]] = (
                        res[param[variable][models[0]]["var"]]
                        + param[variable][models[0]]["transform"]["min"]
                    )
                    res[param[variable][models[0]]["var"]] = stf.inverse_transform(
                        res[[param[variable][models[0]]["var"]]],
                        param[variable][models[0]],
                    )
                elif ("scale" in param) & (
                    not param[variable][models[0]]["transform"]["plot"]
                ):
                    res[param[variable][models[0]]["var"]] = (
                        res[param[variable][models[0]]["var"]]
                        * param[variable][models[0]]["scale"]
                    )

                if log:
                    ax.semilogy(
                        res[param[variable][models[0]]["var"]].index,
                        res[param[variable][models[0]]["var"]].values,
                        color=col_per[i],
                        lw=2,
                        label=str(j),
                    )
                else:
                    if param[variable][models[0]]["circular"]:
                        ax.plot(
                            res[param[variable][models[0]]["var"]].index,
                            np.rad2deg(res[param[variable][models[0]]["var"]].values),
                            color=col_per[i],
                            lw=2,
                            label=str(j),
                        )
                    else:
                        ax.plot(
                            res[param[variable][models[0]]["var"]].index,
                            res[param[variable][models[0]]["var"]].values,
                            color=col_per[i],
                            lw=2,
                            label=str(j),
                        )
        else:
            raise ValueError(
                "Model was not fit successfully. Look at the marginal fit."
            )

    ax.grid()

    if legend:
        # Shrink current axis
        box = ax.get_position()
        if param:
            if legend_loc == "bottom":
                ax.set_position([box.x0, box.y0, box.width, box.height])
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(-0.2, 0.0),
                    ncol=len(pemp),
                    title="Percentiles",
                )
                if param[variable][models[0]]["circular"]:
                    ax.set_yticks([0, 90, 180, 270, 360])
            else:
                ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                # Put a legend to the right of the current axis
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=2,
                    title="Percentiles",
                )
                if param[variable][models[0]]["circular"]:
                    ax.set_yticks([0, 90, 180, 270, 360])
        else:
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, title="Percentiles"
            )

    if isinstance(title, str):
        ax.set_title(title, color="k", fontweight="bold")

    if not label:
        label = labels(variable)
    ax.set_ylabel(label)
    ax.set_xlabel("Normalized period")
    if date_axis:
        ax2 = ax.twiny()
        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        ax2.set_xticks(
            np.array(
                [
                    0.6 / 13,
                    1.5 / 13,
                    2.5 / 13,
                    3.5 / 13,
                    4.5 / 13,
                    5.5 / 13,
                    6.5 / 13,
                    7.5 / 13,
                    8.5 / 13,
                    9.5 / 13,
                    10.5 / 13,
                    11.5 / 13,
                    12.4 / 13,
                ]
            ),
            minor=False,
        )
        # 16 is a slight approximation since months differ in number of days.
        # ax2.xaxis.set_minor_locator(np.array([1/13, 2/13, 3/13, 4/13, 5/13, 6/13, 7/13, 8/13, 9/13, 10/13, 11/13, 12/13]))
        # ax2.xaxis.set_major_formatter(ticker.NullFormatter())

        # Hide major tick labels
        ax2.set_xticklabels("")

        # Customize minor tick labels
        ax2.set_xticks(
            np.array(
                [
                    1 / 13,
                    2 / 13,
                    3 / 13,
                    4 / 13,
                    5 / 13,
                    6 / 13,
                    7 / 13,
                    8 / 13,
                    9 / 13,
                    10 / 13,
                    11 / 13,
                    12 / 13,
                ]
            ),
            minor=True,
        )
        ax2.set_xticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], minor=True
        )
        ax2.tick_params(
            axis="x",  # changes apply to the x-axis
            which="minor",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True,
        )

        ax2.set_xlabel(r"Normal Year")
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width * 0.6, box.height * 0.9]
        )
    show(file_name)

    return ax


def soujourn(data_1, data_2, variable, threshold, ax=None, case="above", fname=None):
    """Plots the distribution function of soujourn above or below a given threshold"""
    # TODO: this package is not available yet.
    from marinetools.temporal.analysis import storm_properties

    if case == "above":
        info = {}
        info["time_step"] = "1H"
        info["min_duration"] = 3
        info["inter_time"] = 3
        info["threshold"] = threshold
        info["interpolation"] = True
        info_1 = storm_properties(data_1, variable, info)
        info_2 = storm_properties(data_2, variable, info)
        var_ = "dur_storm"
    elif case == "below":
        info = {}
        info["time_step"] = "1H"
        info["min_duration"] = 3
        info["inter_time"] = 3
        info["threshold"] = threshold
        info["interpolation"] = True
        info_1 = storm_properties(data_1, variable, info)
        info_2 = storm_properties(data_2, variable, info)
        var_ = "dur_calms"
    else:
        raise ValueError("Case options are above or below. {} given.".format(case))

    ax = handle_axis(ax)
    ax = plot_cdf(
        info_1,
        var_,
        ax=ax,
        file_name="to_axes",
        seaborn=False,
        legend=False,
        label=None,
    )
    ax = plot_cdf(
        info_2, var_, ax=ax, file_name=None, seaborn=False, legend=False, label=None
    )

    show(fname)
    return ax


def nonstationary_cdf_ensemble(
    data: pd.DataFrame, variable: str, ax=None, marker: str = ".", file_name: str = None
):
    """Plots the time variation of given percentiles of data and the theoretical function if provided

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable to be adjusted
        * ax (matplotlib.axis): axis for the plot
        * marker (string): symbol feature of the plot
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot
    """

    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    T = 1
    data["n"] = np.fmod(
        (data.index - datetime.datetime(data.index[0].year, 1, 1, 0))
        .total_seconds()
        .values
        / (T * 365.25 * 24 * 3600),
        1,
    )

    xp, pemp = auxiliar.nonstationary_ecdf(data, variable)
    cols = [
        "blue",
        "orange",
        "green",
        "red",
        "brown",
        "cyan",
        "purple",
        "black",
        "gray",
        "yellow",
    ]
    for j, i in enumerate(pemp):
        ax.plot(xp.loc[:, i], marker, color=cols[j], alpha=0.25)

    ax.grid()

    ax.set_ylabel(labels(variable))
    ax.set_xlabel("Normalized year")
    show(file_name)

    return ax


def pdf_n_i(
    df_obs: pd.DataFrame,
    ns: list,
    param: dict = None,
    variable: str = None,
    nbins: int = 12,
    wlen: float = 14 / 365.25,
    file_name: str = None,
):
    """Compute the pdf at n-times

    Args:
        df_obs (pd.DataFrame): input data
        ns (list): [description]
        param (dict): [description]
        variable (str, optional): [description]. Defaults to None.
        nbins (int): number of bins. Defaults to 12.
        wlen (float): length of the moving windows in days. Defautls 14/365.25
        file_name (str, optional): [description]. Defaults to None.

    Returns:
        matplotlib.ax: the figure
    """

    _, axs = plt.subplots(2, 1, sharex=True)
    axs = axs.flatten()
    colors = ["deepskyblue", "cyan", "darkblue", "royalblue", "b"]

    if param is not None:
        for i, j in enumerate(ns):
            df = stf.numerical_cdf_pdf_at_n(j, param, variable)

            label_ = "F(n: " + str(j) + ")"
            axs[0].plot(df["cdf"], color=colors[i], label=label_)
            axs[1].plot(df["pdf"], color=colors[i], label=label_)

    if df_obs is not None:
        for i, j in enumerate(ns):
            or_ = False
            min_ = j - wlen
            if min_ < 0:
                min_ = 1 - wlen
                or_ = True
            max_ = j + wlen
            if or_:
                mask = (df_obs["n"] <= max_) | (df_obs["n"] >= min_)
            else:
                mask = (df_obs["n"] <= max_) & (df_obs["n"] >= min_)

            label_ = "F(n: " + str(j) + ")"
            x = np.linspace(0, 1, nbins)
            emp = df_obs[variable].loc[mask].quantile(q=x).values

            axs[0].plot(emp, x, ".", color=colors[i], label=label_)

            axs[1].plot(
                emp[1:], np.diff(x) / np.diff(emp), ".", color=colors[i], label=label_
            )

    axs[0].grid()
    axs[1].grid()

    axs[0].legend()
    axs[0].set_ylabel("probability")
    axs[1].set_ylabel("probability")
    axs[1].set_xlabel(labels(variable))

    show(file_name)

    return axs


def wrose(
    wd: np.ndarray,
    ws: np.ndarray,
    legend_title: str = "Wave rose",
    fig_title: str = None,
    var_name: str = "Wave height (m)",
    bins: list = [0, 0.25, 0.5, 1.5, 2.5],
    file_name: str = None,
):
    """Draws a wind or wave rose

    Args:
        * wd (pd.DataFrame): time series with the circular variable
        * ws (pd.DataFrame): time series with the linear variable
        * legend_title (str, optional): set the title of the rose. Defaults to 'Wave rose'.
        * fig_title (str, optional): set the title of the figure. Defaults to None.
        * var_name (str, optional): name of the mean variable. Default 'Wave height (m)'
        * bins: value of segments for variable
        * file_name (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """
    from windrose import WindroseAxes

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        wd,
        ws,
        nsector=16,
        edgecolor="white",
        normed=True,
        cmap=plt.cm.viridis,
        opening=1,
        bins=bins,
    )
    fig.subplots_adjust(top=0.8)

    ax.set_xticklabels(["E", "NE", "N", "NW", "W", "SW", "S", "SE"])

    if isinstance(legend_title, str):
        ax.text(
            0.5,
            1.2,
            legend_title,
            fontsize=12,
            horizontalalignment="center",
            transform=ax.transAxes,
        )

    ax.set_legend(title=var_name, loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
    if isinstance(fig_title, str):
        plt.rc("text", usetex=False)
        ax.text(-0.1, 1, fig_title, transform=ax.transAxes, fontweight="bold")

    show(file_name)
    return ax


def seasonalbox(data, variable, fname=None):
    """Draws a boxplot

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    _, ax = plt.subplots()
    box, median = [], []
    box = [data.loc[data.index.month == i].values[:, 0] for i in range(1, 13)]
    median = [data.loc[data.index.month == i].median() for i in range(1, 13)]

    bp = plt.boxplot(
        box,
        notch=1,
        sym="+",
        patch_artist=True,
        widths=0.3,
        showmeans=False,
        showfliers=False,
    )
    for k in bp["boxes"]:
        k.set(color="brown", linewidth=1, alpha=0.5)

    plt.plot(range(1, 13), median, "r", label="median")
    plt.plot(range(1, 13), np.mean(median) * np.ones(12), color="gray", label="mean")
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    plt.ylabel(labels(variable))

    plt.legend()
    show(fname)

    return ax


def ensemble_acorr(
    lags: list,
    lagsim: list,
    corr_: list,
    corrsim_: list,
    vars_: list,
    ax=None,
    file_name: str = None,
):
    """Plots the correlation bands of RCMs and a given simulation

    Args:
        lags (list): lags of RCM data from correlation functions (x-axis)
        lagsim (list): lags of simulation data from correlation functions (x-axis)
        corr_ (list): correlation of RCM data
        corrsim_ (list): correlation of simulations
        vars_ (list): variables to be plotted
        ax ([matplotlib.ax], optional): Axis where to plot the figure. Defaults to None.
        file_name (str, optional): A file name for saving the figure. Defaults to None.

    Returns:
        ax.matplotlib: the figure
    """

    ax = handle_axis(ax)

    color = ["royalblue", "lightsteelblue", "lightgrey", "darkgoldenrod", "forestgreen"]

    for ind_, var_ in enumerate(vars_):
        if ind_ == 0:
            ax.fill_between(
                lags[var_][0],
                np.min(corr_[var_], axis=0),
                np.max(corr_[var_], axis=0),
                alpha=0.25,
                color=color[ind_],
                label="RCMs band",
            )
        else:
            ax.fill_between(
                lags[var_][0],
                np.min(corr_[var_], axis=0),
                np.max(corr_[var_], axis=0),
                alpha=0.25,
                color=color[ind_],
            )
        ax.plot(
            lagsim[var_], corrsim_[var_], lw=2, color=color[ind_], label=labels(var_)
        )

    ax.set_xlabel("$\mathrm{\mathbf{Lags\quad  (hours)}}$")
    ax.set_ylabel("$\mathrm{\mathbf{Normalized\quad autocorrelation}}$")
    ax.grid(True)
    ax.legend()
    show(file_name)
    return ax


def plot_copula(copula, labels=[], nbins=8, ax=None, fname=None):
    """Plots the copula function

    Args:
        * df (pd.DataFrame): raw time series
        * variables (list): names of the variables
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        The plot
    """
    # _, ax = plt.subplots(figsize=(4, 4))
    ax = handle_axis(ax)
    # plt.subplots_adjust(left=0.2)

    data1, data2 = copula.X, copula.Y

    nlen, nlent = 1000, 1000
    x, y = [], []
    xt = np.linspace(1 / len(data1), 1 - 1 / len(data1), nlent)
    u, v = np.linspace(1 / len(data1), 1 - 1 / len(data1), nlen), np.linspace(
        1 / len(data1), 1 - 1 / len(data1), nlent
    )
    copula.generate_C(u, v)
    for j in xt:
        copula.U = xt
        copula.V = np.ones(nlent) * j
        copula.generate_xy()
        if copula.X1.size == 0:
            x.append(copula.U * 0)
        else:
            x.append(copula.X1)

        if copula.Y1.size == 0:
            y.append(copula.U * 0)
        else:
            y.append(copula.Y1)

    cs = plt.contour(
        np.asarray(x),
        np.asarray(y),
        copula.C,
        nbins,
        linestyles="dashed",
        label="copula",
    )

    Fe, xe, ye = auxiliar.bidimensional_ecdf(data1, data2, nlen)
    cs = plt.contour(xe, ye, Fe, nbins, linestyles="solid", label="empirical")

    ax.clabel(cs, cs.levels, inline=True, fontsize=10)
    plt.text(
        0.6,
        0.8,
        r"$\theta$ = " + str(np.round(copula.theta, decimals=4)),
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.6,
        0.75,
        r"$\tau$ = " + str(np.round(copula.tau, decimals=4)),
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(True)
    ax.legend()
    show(fname)

    return ax


def heatmap(data: np.ndarray, param: dict, type_: str, file_name: str = None):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        * data (np.ndarray): A 2D array of shape (N, M).
        * param (dict): A list or array of length N with the labels for the rows.
        * type_ (str): type of variable to be plotted. B stands for the parameter
            matrix and Q for the covariance matrix
        * file_name: name of the oputput file
    """
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(data)

    # We want to show all ticks ...
    ax.set_xticks(np.arange(np.asarray(data).shape[1]))
    ax.set_yticks(np.arange(np.asarray(data).shape[0]))
    # ... and label them with the respective list entries.

    if type_ == "B":
        column_labels, j = ["mean"], 0
        for i in range(np.asarray(data).shape[1] - 1):
            if not i % len(param["vars"]):
                j += 1
            column_labels.append(
                labels(param["vars"][i % len(data)]) + " (t-" + str(j) + ")"
            )
    else:
        column_labels = labels(param["vars"])

    row_labels = []
    for key_ in param["vars"]:
        row_labels.append(labels(key_))

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(np.asarray(data).shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(np.asarray(data).shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    show(file_name)

    return


def qqplot(
    df1, df2, variable, noperc, ax=None, fname=None, label=None, legend=True, title=None
):
    """[summary]

    Args:
        df1 ([type]): [description]
        df2 ([type]): [description]
        variable ([type]): [description]
        noperc ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
    """
    cdf1 = auxiliar.ecdf(df1, variable, noperc)
    cdf2 = auxiliar.ecdf(df2, variable, noperc)

    ax = handle_axis(ax)

    if not isinstance(label, str):
        label = labels(variable)

    ax.plot(cdf1, cdf2, marker="*", label=label)
    ax.axline([0, 0], [1, 1], color="red", lw=2)
    ax.set_xlabel("Quantiles \n Modeled " + labels(variable))
    ax.set_ylabel("Quantiles \n Observed " + labels(variable))
    ax.grid(True)

    if isinstance(title, str):
        ax.set_title(title, color="black", fontweight="bold")

    if legend:
        ax.legend()

    show(fname)
    return ax


def line_ci(ppfs, var_, keys=["mean", "std"], ax=None, fname=None, title=None):
    """[summary]

    Args:
        ppfs ([type]): [description]
        var_ ([type]): [description]
        keys (list, optional): [description]. Defaults to ['mean', 'std'].
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        title ([type], optional): [description]. Defaults to None.
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.set_prop_cycle(
        "color", [cmocean.cm.matter(i) for i in np.linspace(0, 1, len(ppfs.keys()))]
    )

    for ind_, key in enumerate(ppfs.keys()):
        if var_.lower().startswith("d"):
            ax.plot(np.rad2deg(ppfs[key][keys[0]]))
            ax.fill_between(
                ppfs[key].index,
                np.rad2deg(ppfs[key][keys[0]] - ppfs[key][keys[1]]),
                np.rad2deg(ppfs[key][keys[0]] + ppfs[key][keys[1]]),
                alpha=0.2,
            )
        else:
            ax.plot(ppfs[key][keys[0]])
            ax.fill_between(
                ppfs[key].index,
                ppfs[key][keys[0]] - ppfs[key][keys[1]],
                ppfs[key][keys[0]] + ppfs[key][keys[1]],
                alpha=0.2,
            )
    ax.set_xlabel("Normalized Year", fontweight="bold")
    # ax.set_ylabel(labels(var_))
    ax.grid(True)

    if isinstance(title, str):
        ax.set_title(title, color="black", fontweight="bold")

    show(fname)
    return


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center", size=8)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
