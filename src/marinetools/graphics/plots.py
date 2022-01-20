import datetime
from itertools import product

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from marinetools.graphics.utils import handle_axis, labels, show
from marinetools.temporal.analysis import storm_properties
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


def timeseries(data, variable, ax=None, fname=None):
    """Plots the time series of the variable

    Args:
        * data (pd.DataFrame): time series
        * variable (string): variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    ax.plot(data.loc[:, variable])
    try:
        ax.set_ylabel(labels(variable))
    except:
        ax.set_ylabel(variable)

    show(fname)
    return ax


def plot_cdf(data, var, ax=None, fname=None, seaborn=False, legend=False, label=None):
    """Plots the cumulative density function of data

    Args:
        * data (pd.DataFrame): raw time series
        * var (string): name of the variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

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

    show(fname)

    return ax


def boxplot(data, variable, ax=None, fname=None):
    """Draws a box-plot of the data

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable
        * ax (matplotlib.axis, optional): axis for the plot or None. Defaults to None.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    import seaborn as sns

    data["date"] = data.index
    data["month"] = data["date"].dt.strftime("%b")
    sns.boxplot(x="month", y=variable, data=data, ax=ax)

    return ax


def scatter_error_dependencies(df_dt, variables, fname=None):
    """Draws the scatter plot of data before and after the computation of the temporal dependency

    Args:
        * df_dt (dict): parameters of the temporal dependency package
        * variables (list): names of the variables
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """
    if isinstance(variables, str):
        variables = [variables]

    _, ax = plt.subplots(1, 1)
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
            label=labels(j)
            + " (R$^2_{adj}$ = "
            + str(np.round(df_dt["R2adj"][i], decimals=3))
            + ")",
        )
    ax.grid()
    ax.set_xlabel(r" Observed ($\Phi^{-1}(F_{\zeta}(\zeta))$")
    ax.set_ylabel(r" Modeled ($\Phi^{-1}(F_{\zeta}(\zeta))$")
    ax.legend(loc=4, title=r"$\zeta$")
    show(fname)
    return ax


def corr(data, lags=24, ax=None, fname=None):
    """Plots the correlation of between time series

    Args:
        * x (np.ndarray): time series
        * xs (np.ndarray): time series
        * var (string): name of the variable for the plot
        * lags (int): maximum lag time (hours). Defaults to 48 hours.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """

    ax = handle_axis(ax)
    ax.acorr(data, usevlines=False, maxlags=lags, normed=True, lw=2)

    ax.set_xlabel(r"Lags (hr)")
    ax.set_ylabel(r"Normalized autocorrelation")
    ax.grid(True)
    ax.legend()
    show(fname)

    return ax


def joint_plot(data, varx, vary, ax=None):
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


def bivariate_ensemble_pdf(df_sim, df_obs, varp, fname=None):
    """Plots together the bivariate probability density function of several time series observations and one simulation

    Args:
        * df_sim (pd.DataFrame): raw time series
        * df_obs (dict): each element is an observed time series
        * varp (list): names of the variables
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

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
    show(fname)
    return


def bivariate_pdf(
    df_sim, df_obs, variables, bins=None, levels=None, ax=None, fname=None
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
        bins = []

    nn, nx_, ny_ = np.histogram2d(
        df_obs[variables[0]], df_obs[variables[1]], bins=[25, 25], density=True
    )
    ny, nx = np.meshgrid(ny_[:-1] + np.diff(ny_) / 2.0, nx_[:-1] + np.diff(nx_) / 2.0)

    if levels is None:
        levels = np.linspace(np.max(nn) / 12, np.max(nn), 12)

    # cs = ax.contourf(nx, ny, nn, alpha=0.25, levels=np.append(0, levels))
    min_ = np.min(nn)
    max_ = np.max(nn)

    ax[0].imshow(
        nn,
        extent=[np.min(ny), np.max(ny), np.min(nx), np.max(nx)],
        vmin=min_,
        vmax=max_,
        cmap="viridis",
        aspect="auto",
    )
    ax[0].set_xlabel(labels(variables[1]))
    ax[0].set_ylabel(labels(variables[0]))

    nn_, nx, ny = np.histogram2d(
        df_sim[variables[0]], df_sim[variables[1]], bins=[nx_, ny_], density=True
    )
    ny, nx = np.meshgrid(ny[:-1] + np.diff(ny) / 2.0, nx[:-1] + np.diff(nx) / 2.0)
    # CS = plt.contour(nx, ny, nn, levels=levels)
    # plt.grid(True)
    # plt.clabel(CS, inline=1, fontsize=10)
    cs = ax[1].imshow(
        nn_,
        extent=[np.min(ny), np.max(ny), np.min(nx), np.max(nx)],
        vmin=min_,
        vmax=max_,
        cmap="viridis",
        aspect="auto",
    )
    ax[1].set_xlabel(labels(variables[1]))
    ax[1].set_yticklabels([])

    # cbar = plt.colorbar(cs)
    # cbar.ax.set_ylabel("Mass probability")
    show(fname)

    return


def nonstationary_cdf(
    data,
    variable,
    param=None,
    daysWindowsLength=14,
    equal_windows=False,
    ax=None,
    log=False,
    fname=None,
    legend=True,
    legend_loc="right",
    title=None,
    date_axis=False,
    pemp=None,
):
    """Plots the time variation of given percentiles of data and theoretical function if provided

    Args:
        * data (pd.DataFrame): time series
        * variable (string): name of the variable to be adjusted
        * param (dict, optional): the parameters of the the theoretical model if they are also plotted.
        * daysWindowsLength (int, optional): period of windows length for making the non-stationary empirical distribution function. Defaults to 14 days.
        * fname (string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

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

    ax.set_ylabel(labels(variable))
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
    show(fname)

    return ax


def nonstationary_cdf_ensemble(data, variable, ax, marker=".", fname=None):
    """Plots the time variation of given percentiles of data and the theoretical function if provided

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): name of the variable to be adjusted
        * ax (matplotlib.axis): axis for the plot
        * marker (string): symbol feature of the plot
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

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

    return ax


def pdf_n_i(
    df_obs,
    ns,
    param=None,
    variable=None,
    nbins=12,
    wlen=14 / 365.25,
    fname=None,
    legend=True,
    title=None,
    alpha=0.01,
):
    """[summary]

    Args:
        ns ([type]): [description]
        param ([type]): [description]
        variable ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.
        legend (bool, optional): [description]. Defaults to True.
        title ([type], optional): [description]. Defaults to None.
        alpha (float, optional): [description]. Defaults to 0.01.

    Returns:
        [type]: [description]
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
            # emp = auxiliar.ecdf(df_obs.loc[mask], variable)
            # axs[0].plot(emp[variable], emp["prob"], label=label_)
            x = np.linspace(0, 1, nbins)
            emp = df_obs[variable].loc[mask].quantile(q=x).values

            axs[0].plot(emp, x, ".", color=colors[i], label=label_)

            # emp = auxiliar.epdf(df_obs.loc[mask], variable)
            axs[1].plot(
                emp[1:], np.diff(x) / np.diff(emp), ".", color=colors[i], label=label_
            )

    axs[0].grid()
    axs[1].grid()

    axs[0].legend()
    axs[0].set_ylabel("probability")
    axs[1].set_ylabel("probability")
    axs[1].set_xlabel(labels(variable))

    show(fname)

    return


def wrose(
    wd,
    ws,
    title="Wave rose",
    var_name="Wave height (m)",
    bins=[0, 0.25, 0.5, 1.5, 2.5],
    fname=None,
):
    """Draws a wind or wave rose

    Args:
        * wd (pd.DataFrame): time series with the circular variable
        * ws (pd.DataFrame): time series with the linear variable
        * title (str, optional): set the title of the rose. Defaults to 'Wave rose'.
        * fname (None or string, optional): name of the file to save the plot or None to see plots on the screen. Defaults to None.

    Returns:
        * ax (matplotlib.axis): axis for the plot or None
    """
    from windrose import WindroseAxes

    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        wd,
        ws,
        edgecolor="white",
        normed=True,
        cmap=cmocean.cm.haline_r,
        opening=0.8,
        bins=bins,
    )
    fig.subplots_adjust(top=0.9)

    ax.set_xticklabels(['E', 'NE','N', 'NW', 'W', 'SW', 'S', 'SE'])

    if isinstance(title, str):
        ax.text(
            0.5,
            1.2,
            title,
            fontsize=12,
            horizontalalignment="center",
            transform=ax.transAxes,
        )

    ax.set_legend(title=var_name)
    show(fname)
    return ax


def ensemble_acorr(lags, lagsim, corr_, corrsim_, vars_, ax=None, fname=None):
    """[summary]

    Args:
        lags ([type]): [description]
        lagsim ([type]): [description]
        corr_ ([type]): [description]
        corrsim_ ([type]): [description]
        vars_ ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.
        fname ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
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
    show(fname)
    return ax


def heatmap(data, param, type_, fname=None):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        * data: A 2D numpy array of shape (N, M).
        * param: A list or array of length N with the labels for the rows.
        * fname: name of the oputput file
    """
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(data)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("Coefficients", rotation=-90, va="bottom")

    # We want to show all ticks...
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
    show(fname)

    return


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw
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
    kw = dict(horizontalalignment="center", verticalalignment="center")
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
