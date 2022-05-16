import warnings
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.temporal.fdist.copula import Copula as Copula
from marinetools.utils import auxiliar, save

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


def show_init_message():
    message = (
        "\n"
        + "=============================================================================\n"
        + " Initializing MarineTools.temporal, v.1.0.0\n"
        + "=============================================================================\n"
        + "Copyright (C) 2021 Environmental Fluid Dynamics Group (University of Granada)\n"
        + "=============================================================================\n"
        + "This program is free software; you can redistribute it and/or modify it under\n"
        + "the terms of the GNU General Public License as published by the Free Software\n"
        + "Foundation; either version 3 of the License, or (at your option) any later \n"
        + "version.\n"
        + "This program is distributed in the hope that it will be useful, but WITHOUT \n"
        + "ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS\n"
        + "FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
        + "You should have received a copy of the GNU General Public License along with\n"
        + "this program; if not, write to the Free Software Foundation, Inc., 675 Mass\n"
        + "Ave, Cambridge, MA 02139, USA.\n"
        + "============================================================================="
    )
    return message


def simulation(
    param: dict,
    tidalConstituents: dict = None,
    dur_storm_calms: pd.DataFrame = None,
    seed: int = np.random.randint(1e6),
    include: bool = False,
):
    """Simulates time series with the probability model given in param and the temporal dependency given in df_dt
    TODO: update this docstring

    Args:
        * param (dict): the probability models of every variable
        * tidalConstituents (dict, optional): tidal constituents required to compute
            the tidal elevation and the mean sea level (Códiga, 2011). Defaults is none.
        * dur_storm_calms (pd.DataFrame, optional): time series of storm events
        * seed (int, optional): a value to create a non-random simulation (mainly for debugging actions)
         -  ensemble (boolean, optional): True is RCMs parameters are the input data and
            an ensemble analysis want to be performed
        * include (boolean, optional) : include some determinist time series given by the user

    Example:
        params = {"Hs": {... see marinetools.temporal.analysis.marginalfit},
                  "TD": {... marinetools.temporal.see analysis.dependency},
                  "TS": {}}

    Returns:
        - Saved files with simulated time series

    """
    np.random.seed(seed)

    # Check the parameters
    param = check_parameters(param)

    # Start the simulation algorithm
    for nosim in range(param["TS"]["nosim"]):
        logger.info(
            "Simulation no. %s of %s"
            % (str(nosim + 1).zfill(4), str(param["TS"]["nosim"]).zfill(4))
        )

        # Make storms simulation (Lira et al, 2019)
        if param["TS"]["events"]:
            # Initialize storm duration and calms dictionary
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
                    param["TS"]["family"],
                )
                # cop.theta, cop.tau = param["TS"]["season"][season]
                cop.generate_xy(n=10000)
                (
                    durs_by_seasons[season]["storms"],
                    durs_by_seasons[season]["calms"],
                ) = (
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
                        start=ini,
                        end=ini + timedelta_storm,
                        freq=param["TS"]["freq"],
                    ),
                    columns=param["TS"]["vars"],
                )

                # Generate the i-storm
                zsim = var_simulation(param["TD"], int(dstorm / factor) + 1, "normal")
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
                if param["TS"]["conditional"] & (param["TS"]["mvar"] == var_):
                    # If it is the main variable of the analysis, compute the
                    # conditional probability reached the threshold
                    df_var = pd.DataFrame(
                        np.ones(len(df["n"])) * param["TS"]["threshold"],
                        index=df.index,
                        columns=["data"],
                    )
                    df_var[var_] = df[var_].values
                    df_var["n"] = df["n"].values

                    # Compute the NS-CDF
                    if param["TS"]["ensemble"]:
                        if param["EA"]["weights"] == "equal":
                            cdfu = 0
                            for model in param["TS"]["models"]:
                                cdfu_model = stf.cdf(df_var, param[var_][model])
                                cdfu += cdfu_model
                            cdfu = cdfu / len(param["TS"]["models"])
                        else:
                            cdfu = 0
                            for i, model in enumerate(param["TS"]["models"]):
                                cdfu_model = stf.cdf(df_var, param[var_][model]) * (
                                    param["EA"]["weights"][i]
                                )
                                cdfu += cdfu_model

                    else:
                        cdfu = stf.cdf(df_var, param[ind_])
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
                if param["TS"]["ensemble"]:
                    res = stf.ensemble_ppf(dfj, param, var_, param["TS"]["nodes"])
                else:
                    res = pd.DataFrame(
                        stf.ppf(dfj, param[var_], ppf=True),
                        index=dfj.index,
                        columns=[var_],
                    )
                df[var_] = res[var_].values

                # Compute the inverse of the power transform if it is required
                # Transformed timeserie
                if param["EA"]["make"]:
                    if "scale" in param:
                        df[var_] = df[var_] * param[var_]["scale"]

                    df[var_] = df[var_] + param["EA"]["min_ensemble"]
                    df[var_] = stf.inverse_transform(df[[var_]], param["EA"], True)
                elif "scale" in param:
                    df[var_] = df[var_] * param[var_]["scale"]

        else:
            # Make the full simulation
            ini, end = (
                datetime.strptime(param["TS"]["start"], "%Y/%m/%d %H:%M:%S"),
                datetime.strptime(param["TS"]["end"], "%Y/%m/%d %H:%M:%S"),
            )
            df = pd.DataFrame(
                index=pd.date_range(
                    start=ini,
                    end=end,
                    freq=param["TS"]["freq"],
                )
            )

            # Create the normalized date
            df["n"] = (
                (df.index.dayofyear + df.index.hour / 24.0 - 1)
                / pd.to_datetime(
                    {"year": df.index.year, "month": 12, "day": 31, "hour": 23}
                ).dt.dayofyear
            ).values

            # Create the stationary full simulation
            zsim = var_simulation(param["TD"], len(df), "normal")

            for ind_, var_ in enumerate(param["TD"]["vars"]):
                # Compute the inverse of the Gaussian PM
                dfj = pd.DataFrame(
                    st.norm.cdf(zsim[:, ind_]), index=df.index, columns=["prob"]
                )
                dfj["n"] = df["n"].copy()

                # Compute the ppf of the PMs for "var_"
                if param["TS"]["ensemble"]:
                    for model in param["TS"]["models"]:
                        param[var_] = auxiliar.str2fun(param[var_], model)
                    df[var_] = stf.ensemble_ppf(dfj, param, var_, param["TS"]["nodes"])

                    # Transformed timeserie
                    if param[var_]["transform"]["make"]:
                        if "scale" in param:
                            df[var_] = df[var_] * param[var_]["scale"]

                        if "min" in param:
                            df[var_] = df[var_] + param[var_]["transform"]["min"]
                        df[var_] = stf.inverse_transform(df[[var_]], param[var_])
                    elif "scale" in param[var_]:
                        df[var_] = df[var_] * param[var_]["scale"]
                else:
                    param = auxiliar.str2fun(param, var_)
                    df[var_] = pd.DataFrame(
                        stf.ppf(dfj, param[var_]),
                        index=dfj.index,
                        columns=[var_],
                    )

                    # Transformed timeseries
                    if param[var_]["transform"]["make"]:
                        if "scale" in param[var_]:
                            df[var_] = df[var_] * param[var_]["scale"]

                        df[var_] = df[var_] + param[var_]["transform"]["min"]
                        df[var_] = stf.inverse_transform(df[[var_]], param[var_])
                    elif "scale" in param[var_]:
                        df[var_] = df[var_] * param[var_]["scale"]

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
        # for var_ in param["TD"]["vars"]:
        #    if param[var_]["circular"] == True:
        #        df[var_] = np.rad2deg(df[var_])

        # Include any deterministic timeseries if given
        if "include" in param["TS"]:
            if param["TS"]["include"]:
                df[include.name] = include

        logger.info(
            "Saving simulation no. %s of %s"
            % (str(nosim + 1).zfill(4), str(param["TS"]["nosim"]).zfill(4))
        )
        # Create the folder for simulations
        auxiliar.mkdir(param["TS"]["folder"])

        # Save simulation file
        if param["TS"]["save_z"]:
            save.to_txt(
                zsim,
                param["TS"]["folder"]
                + "/simulation_z_"
                + str(nosim + 1).zfill(4)
                + ".csv",
            )

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


def check_parameters(param):
    """_summary_ TODO

    Args:
        param (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if param["TS"]["nosim"] <= 0:
        raise ValueError(
            "The number of simulation should be a positive integer. Got {}.".format(
                param["TS"]["nosim"]
            )
        )

    # Required for the numerical computation of non-stationary F for ensembles
    if not "nodes" in param["TS"].keys():
        param["TS"]["nodes"] = [4383, 250]

    if not "conditional" in param["TS"].keys():
        param["TS"]["conditional"] = False
        param["TS"]["mvar"] = None
    else:
        if not "mvar" in param["TS"].keys():
            raise ValueError(
                "The main variable (mvar) is required for conditional analysis."
            )

        if not "threshold" in param["TS"].keys():
            raise ValueError(
                "The threshold parameter is required for conditional analysis."
            )

    if not "include" in param["TS"].keys():
        param["TS"]["include"] = False

    if not "F_files" in param["TS"].keys():
        param["TS"]["F_files"] = "nonst_F_ensemble_"

    if not "folder" in param["TS"].keys():
        param["TS"]["folder"] = "simulations"

    if not "events" in param["TS"].keys():
        param["TS"]["events"] = False
    else:
        if param["TS"]["events"]:
            if not "family" in param["TS"].keys():
                raise ValueError("Family is required for events simulation.")

    if not "class_type" in param["TS"].keys():
        param["TS"]["class_type"] = "WSSF"
        param["TS"]["season"] = {"winter": 1, "spring": 2, "fall": 3, "summer": 4}

    if not "ensemble" in param["TS"].keys():
        param["TS"]["ensemble"] = False

    if not "save_z" in param["TS"].keys():
        param["TS"]["save_z"] = False

    return param


def var_simulation(par: dict, lsim: int, distribution: str):
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
        if dim == 1:
            y = np.random.normal(np.zeros(dim), par["Q"], lsim)
            y = y[:, np.newaxis].T
        else:
            y = np.random.multivariate_normal(np.zeros(dim), par["Q"], lsim).T

    # Initialize the simulation matrix
    zsim[:, 0:ord_] = y[:, 0:ord_]
    for i in range(ord_, lsim):
        z = np.fliplr(zsim[:, i - ord_ : i])
        z1 = np.vstack((1, np.reshape(z.T, (ord_ * dim, 1))))
        zsim[:, i] = np.dot(par["B"], z1).T + y[:, i]

    return zsim.T


def class_seasons(date, type_="WSSF"):
    """Obtains the season of any date

    Args:
        * date (datetime): starting date of the event

    Returns:
        * season (string): the season
    """
    if type_ == "WSSF":
        if (
            ((date.month == 12) & (date.day >= 21))
            | (date.month == 1)
            | (date.month == 2)
            | ((date.month == 3) & (date.day < 21))
        ):
            season = "winter"
        elif (
            ((date.month == 3) & (date.day >= 21))
            | (date.month == 4)
            | (date.month == 5)
            | ((date.month == 6) & (date.day < 21))
        ):
            season = "spring"
        elif (
            ((date.month == 6) & (date.day >= 21))
            | (date.month == 7)
            | (date.month == 8)
            | ((date.month == 9) & (date.day < 21))
        ):
            season = "summer"
        else:
            season = "fall"
    elif type_ == "WS":
        if (
            ((date.month == 12) & (date.day >= 21))
            | (date.month == 1)
            | (date.month == 2)
            | (date.month == 3)
            | (date.month == 4)
            | (date.month == 5)
            | ((date.month == 6) & (date.day < 21))
        ):
            season = "WS"
        else:
            season = "SF"
    elif type_ == "SF":
        if (
            ((date.month == 3) & (date.day >= 21))
            | (date.month == 4)
            | (date.month == 5)
            | (date.month == 6)
            | (date.month == 7)
            | (date.month == 8)
            | ((date.month == 9) & (date.day < 21))
        ):
            season = "SS"
        else:
            season = "FW"

    return season
