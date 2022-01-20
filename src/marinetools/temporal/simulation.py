import warnings
from datetime import datetime, timedelta

import marinetools.temporal.analysis as analysis
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.temporal.fdist.copula import Copula as Copula
from marinetools.utils import auxiliar, read, save


def simulation(
    param,
    tidalConstituents=None,
    dur_storm_calms=None,
    seed=np.random.randint(1e6),
    ensemble=False,
    include=False,
):
    """Simulates time series with the probability model given in param and the temporal dependency given in df_dt

    Args:
        * df_dt (dict): the parameters of the VAR model
        * param (dict): the probability models of every variable
        * tidalConstituents (dict, optional): tidal constituents required to compute
            the tidal elevation and the mean sea level (Códiga, 2011). Defaults is none.
        * dur_storm_calms (pd.DataFrame, optional): time series of storm events
        * seed (float, optional): a value to create a non-random simulation (mainly for debugging actions)
        * ensemble (boolean, optional): True is RCMs parameters are the input data and
            an ensemble analysis want to be performed
        * include () : include some determinist time series given by the user

    Returns:
        - Saved files with simulated time series

    """

    # Reading parameters
    for var_ in param["TD"]["vars"]:
        param[var_] = read.rjson(param[var_]["fname"])

    # Read the multivariate and temporal dependency parameters
    df_dt = read.rjson(param["TD"]["fname"], "td")

    # Check number of simulations
    np.random.seed(seed)
    if param["TS"]["nosim"] <= 0:
        raise ValueError(
            "The number of simulation should be a positive integer. Got {}.".format(
                param["TS"]["nosim"]
            )
        )
    # Start the simulation algorithm
    for nosim in range(param["TS"]["nosim"]):
        logger.debug(
            "Simulation no. %s of %s"
            % (str(nosim + 1).zfill(4), str(param["TS"]["nosim"]).zfill(4))
        )
        # Make storms simulation (Lira et al, 2019)
        if param["TS"]["events"]:
            # Initializa storm duration and calms dictionary
            durs_by_seasons = {
                i: {"storms": [], "calms": []} for i in param["TS"]["season"]
            }
            index = {}

            # Fill storm and calm durations using Copula parameters
            for season in param["TS"]["season"].keys():
                cop = Copula(
                    dur_storm_calms.loc[
                        dur_storm_calms["season"] == season, "dur_storm"
                    ].values[:],
                    dur_storm_calms.loc[
                        dur_storm_calms["season"] == season, "dur_calms"
                    ].values[:],
                    "clayton",
                )
                cop.theta, cop.tau = param["TS"]["season"][season]
                cop.generate_xy(n=10000)
                durs_by_seasons[season]["storms"], durs_by_seasons[season]["calms"] = (
                    cop.X1,
                    cop.Y1,
                )
                index[season] = 0
                # Remove outlayers
                if any(durs_by_seasons[season]["storms"] < 0):
                    indexs = np.where(durs_by_seasons[season]["storms"] > 0)
                    durs_by_seasons[season]["storms"] = durs_by_seasons[season][
                        "storms"
                    ][indexs]
                    durs_by_seasons[season]["calms"] = durs_by_seasons[season]["calms"][
                        indexs
                    ]
                    warnings.warn(
                        "Some storms duration at {} are negative. Removed from events database. Please, check the copula parameters.".format(
                            season
                        )
                    )
                # Remove outlayers
                if any(durs_by_seasons[season]["calms"] < 0):
                    indexs = np.where(durs_by_seasons[season]["calms"] > 0)
                    durs_by_seasons[season]["storms"] = durs_by_seasons[season][
                        "storms"
                    ][indexs]
                    durs_by_seasons[season]["calms"] = durs_by_seasons[season]["calms"][
                        indexs
                    ]
                    warnings.warn(
                        "Some calms duration at {} are negative. Removed from eventsdatabase. Please, check the copula parameters.".format(
                            season
                        )
                    )

            ini, end = (
                datetime.strptime(param["TS"]["start"], "%Y/%m/%d %H:%M:%S"),
                datetime.strptime(param["TS"]["end"], "%Y/%m/%d %H:%M:%S"),
            )
            df = pd.DataFrame()

            calms, storms = [], []
            season = class_seasons(ini, type_=param["TS"]["class_type"])
            calms.append(durs_by_seasons[season]["calms"][0])
            ini = ini + timedelta(hours=durs_by_seasons[season]["calms"][0])

            # Start the simulation
            while ini < end:
                # Look for season and retrieve the durations
                season = class_seasons(ini, type_=param["TS"]["class_type"])
                dstorm, dcalm = (
                    durs_by_seasons[season]["storms"][index[season]],
                    durs_by_seasons[season]["calms"][index[season]],
                )
                calms.append(dcalm)
                storms.append(dstorm)

                # Locate the start/end of the i-storm
                if "D" in param["TS"]["freq"]:
                    timedelta_storm = timedelta(days=dstorm)
                    end_i = ini + timedelta(days=dstorm + dcalm)
                    if param["TS"]["freq"] == "D":
                        factor = 1
                    else:
                        factor = int(param["TS"]["freq"].split("D")[0])
                elif "H" in param["TS"]["freq"]:
                    timedelta_storm = timedelta(hours=dstorm)
                    end_i = ini + timedelta(hours=dstorm + dcalm)
                    if param["TS"]["freq"] == "H":
                        factor = 1
                    else:
                        factor = int(param["TS"]["freq"].split("H")[0])

                # Initialize the normalize simulation for the i-storm
                df_zsim = pd.DataFrame(
                    -1,
                    index=pd.date_range(
                        start=ini, end=ini + timedelta_storm, freq=param["TS"]["freq"]
                    ),
                    columns=param["TS"]["vars"],
                )

                # Generate the i-storm
                zsim = var_simulation(df_dt, int(dstorm / factor) + 1, "normal")
                df_zsim.loc[:, param["TS"]["vars"]] = zsim
                df = df.append(df_zsim)

                index[season] = index[season] + 1
                ini = end_i

            # Create the normalized date
            df["n"] = (
                (df.index.dayofyear + df.index.hour / 24.0 - 1)
                / pd.to_datetime(
                    {"year": df.index.year, "month": 12, "day": 31, "hour": 23}
                ).dt.dayofyear
            ).values

            # Return the value at time t and the parameters associated
            for ind_, var_ in enumerate(param["TS"]["vars"]):
                # Read the NS parameters of the PMs
                if param["TS"]["ensemble"]:
                    for model in param["TS"]["models"]:
                        param[var_] = auxiliar.str2fun(param[var_], model)
                else:
                    param = auxiliar.str2fun(param, var_)

                # Compute the inverse Gaussian for the "var_"
                if var_ == param["TD"]["mvar"]:
                    # If it is the main variable of the analysis, compute the
                    # conditional probability reached the threshold
                    var_ = pd.DataFrame(
                        np.ones(len(df["n"])) * param["TD"]["threshold"],
                        index=df.index,
                        columns=["data"],
                    )
                    var_[var_] = df[var_].values
                    var_["n"] = df["n"].values

                    # Compute the NS-CDF
                    if ensemble:
                        # ensemble asigning equiprobability to all model
                        # TODO: enable weights
                        cdfu = 0
                        for model in param["TS"]["models"]:
                            cdfu_model = stf.cdf(var_, param[var_][model])
                            cdfu += cdfu_model
                        cdfu = cdfu / len(param["TS"]["models"])
                    else:
                        cdfu = stf.cdf(var_, param[ind_])
                    cdfu = pd.DataFrame(cdfu)

                    # Return the values
                    cdfj = (
                        st.norm.cdf(df.loc[:, var_]) * (1 - cdfu["prob"].values[:])
                        + cdfu["prob"].values[:]
                    )
                    dfj = pd.DataFrame(cdfj, index=df.index, columns=["prob"])
                else:
                    # Compute the inverse Gaussian for the "var_"
                    dfj = pd.DataFrame(
                        st.norm.cdf(df.loc[:, var_].values[:]),
                        index=df.index,
                        columns=["prob"],
                    )

                dfj["n"] = df["n"].copy()
                # Compute the ppf of the PM for "var_"
                if ensemble:
                    res = stf.ensemble_ppf(dfj, param, var_)
                else:
                    res = pd.DataFrame(
                        stf.ppf(dfj, param[var_], ppf=True),
                        index=dfj.index,
                        columns=[var_],
                    )
                df[var_] = res[var_].values

                # Compute the inverse of the power transform if it is required
                # if param[var_]["transform"]["make"]:
                #     df[var_] = analysis.inverse_transform(df[[var_]], param[var_])

                # Transformed timeserie
                if param[var_]["transform"]["make"]:
                    if "scale" in param:
                        df[var_] = df[var_] * param[var_]["scale"]

                    df[var_] = df[var_] + param[var_]["transform"]["min"]
                    df[var_] = stf.inverse_transform(df[[var_]], param[var_])
                elif "scale" in param:
                    df[var_] = df[var_] * param[var_]["scale"]

        else:
            # Make the full simulation
            ini, end = (
                datetime.strptime(param["TS"]["start"], "%Y/%m/%d %H:%M:%S"),
                datetime.strptime(param["TS"]["end"], "%Y/%m/%d %H:%M:%S"),
            )
            df = pd.DataFrame(
                index=pd.date_range(start=ini, end=end, freq=param["TS"]["freq"],)
            )

            # Create the normalized date
            df["n"] = (
                (df.index.dayofyear + df.index.hour / 24.0 - 1)
                / pd.to_datetime(
                    {"year": df.index.year, "month": 12, "day": 31, "hour": 23}
                ).dt.dayofyear
            ).values

            # Create the stationary full simulation
            zsim = var_simulation(df_dt, len(df), "normal")

            for ind_, var_ in enumerate(param["TS"]["vars"]):
                # Compute the inverse of the Gaussian PM
                dfj = pd.DataFrame(
                    st.norm.cdf(zsim[:, ind_]), index=df.index, columns=["prob"]
                )
                dfj["n"] = df["n"].copy()

                # Compute the ppf of the PMs for "var_"
                if ensemble:
                    for model in param["TS"]["models"]:
                        param[var_] = auxiliar.str2fun(param[var_], model)
                    df[var_] = stf.ensemble_ppf(dfj, param, var_)

                    # Compute the inverse of the power transform if it is required
                    # if param[var_]["transform"]["make"]:
                    #     res[var_] = analysis.inverse_transform(
                    #         res[[var_]], param[var_], ensemble=True
                    #     )
                    # Transformed timeserie
                    if param[var_]["transform"]["make"]:
                        if "scale" in param:
                            df[var_] = df[var_] * param[var_]["scale"]

                        df[var_] = df[var_] + param[var_]["transform"]["min"]
                        df[var_] = stf.inverse_transform(df[var_], param[var_])
                    elif "scale" in param[var_]:
                        df[var_] = df[var_] * param[var_]["scale"]
                else:
                    param = auxiliar.str2fun(param, var_)
                    df[var_] = pd.DataFrame(
                        stf.ppf(dfj, param[var_], ppf=True),
                        index=dfj.index,
                        columns=[var_],
                    )

                    # # Compute the inverse of the power transform if it is required
                    # if param[var_]["transform"]["make"]:
                    #     res[var_] = analysis.inverse_transform(res[[var_]], param[var_])
                    # Transformed timeserie
                    if param[var_]["transform"]["make"]:
                        if "scale" in param:
                            df[var_] = df[var_] * param[var_]["scale"]

                        df[var_] = df[var_] + param[var_]["transform"]["min"]
                        df[var_] = stf.inverse_transform(df[var_], param[var_])
                    elif "scale" in param[var_]:
                        df[var_] = df[var_] * param[var_]["scale"]

                # df[var_] = res[var_].values

        # Include tidal level if added
        if tidalConstituents is not None:
            # The tidal level is deterministic so just one time per simulations is
            # reconstructed
            if nosim == 0:
                from utide import reconstruct

                time = mdates.date2num(df.index.to_pydatetime())
                tidalLevel = reconstruct(time, tidalConstituents)

            df["ma"] = tidalLevel["h"] - tidalConstituents["mean"]
            if "mm" in df.columns:
                df["eta"] = tidalLevel["h"] + df["mm"]

        # Transform radians to angles (circular variables)
        for var_ in param["TS"]["vars"]:
            if param[var_]["circular"] == True:
                df[var_] = np.rad2deg(df[var_])

        # Include any deterministic timeseries if given
        if "include" in param["TS"]:
            if param["TS"]["include"]:
                df[include.name] = include

        logger.debug(
            "Saving simulation no. %s of %s"
            % (str(nosim + 1).zfill(4), str(param["TS"]["nosim"]).zfill(4))
        )
        # Create the folder for simulations
        auxiliar.mkdir(param["TS"]["folder"])

        # Save simulation file
        save.to_csv(
            df,
            param["TS"]["folder"] + "/simulation_" + str(nosim + 1).zfill(4) + ".zip",
        )

        # If storm analysis, save the storm and calms duration too
        if param["TS"]["events"]:
            dstorm = pd.DataFrame(storms)
            save.to_csv(
                dstorm,
                param["TS"]["folder"]
                + "/durs_storm_"
                + str(nosim + 1).zfill(4)
                + ".zip",
            )

            dcalms = pd.DataFrame(calms)
            save.to_csv(
                dcalms,
                param["TS"]["folder"]
                + "/durs_calms_"
                + str(nosim + 1).zfill(4)
                + ".zip",
            )

        del df

    return


def var_simulation(par, lsim, distribution):
    """Creates new normalized multivariate time series

    Args:
        * par (dict): paramaters of the VAR adjustment
        * lsim (int): length of the new time series
        * distribution (string): name of the probability model acccording to scipy.stats

    Returns:
        * zsim (np.ndarray): a multivariate time series
    """

    dim = par["dim"]
    ord_ = par["id"]
    zsim = np.zeros([dim, lsim])
    if distribution == "normal":  # TODO: some other non-normal multivariate analysis
        y = np.random.multivariate_normal(np.zeros(dim), par["Q"], lsim).T

    # Initialize the simulation matrix
    zsim[:, 0:ord_] = y[:, 0:ord_]
    for i in range(ord_, lsim):
        z = np.fliplr(zsim[:, i - ord_ : i])
        z1 = np.vstack((1, np.reshape(z.T, (ord_ * dim, 1))))
        zsim[:, i] = np.dot(par["B"], z1).T + y[:, i]

    return zsim.T
