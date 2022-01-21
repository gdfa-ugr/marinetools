import datetime
import time

import numpy as np
import pandas as pd
import scipy.stats as st
from loguru import logger
from marinetools.temporal.fdist import statistical_fit as stf
from marinetools.utils import auxiliar, read, save

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


def marginalfit(df: pd.DataFrame, parameters: dict):
    """Fits a stationary (or not), simple or mixed probability model to data. Additional
    information can be found in Cobos et al., 2022, 'MarineTools.temporal: A Python
    package to simulate Earth and environmental time series'. Environmental Modelling
    and Software.

    Args:
        * df (pd.DataFrame): the raw time series
        * parameters (dict): the initial guess parameters of the probability models.
            - 'var' key is a  string with the name of the variable,
            - 'type':
            - 'fun' is a list within strings with the name of the probability model,
            - 'non_stat_analysis' stand for stationary (False) or not (True),
            - 'ws_ps': initial guess of percentiles or weights of PMs
            - 'basis_function' is an option to specify the GFS expansion that includes:
                - 'method': a string with an option of the GFS
                - 'noterms': number of terms of GFS
                - 'periods': is a list with periods of oscillation for NS-PMs ,
            - 'transform' stand for normalization that includes:
                - 'make':
                - 'method':
                - 'plot':
            - 'optimize': a dictionary with some initial parameters for the optimization
            method (see scipy.optimize.minimize), some options are:
                - 'method': "SLSQP",
                - 'maxiter': 1e2,
                - 'ftol': 1e-4
                - 'eps': 1e-7
                - 'bounds': 0.5
            - giter: number of global iterations. Repeat the minimization algorithm
            changing the initial guess
            - 'mode': a list with the mode to be computed independently,
            - 'par': initial guess of the parameters for the mode given
            - 'folder_name': string where the folder where the analysis will be saved
            (optional)
            - 'file_name': string where it will be saved the analysis (optional)

    Example:
        * param = {'Hs': {'var': 'Hs',
                        'fun': {0: 'norm'},
                        'type': 'linear' (default) or 'circular',
                        'non_stat_analysis': True (default), False,
                        'basis_functions': None or a list with:
                            {"method": "trigonometric", "sinusoidal", ...
                            "noterms": int,
                            "periods": [1, 2, 4, ...]}
                        'ws_ps': 1 or a list,
                        'transform': None or a list with:
                            {"make": True,
                            "plot": False,
                            "method": "box-cox"}, or "yeo-johnson}
                        'mode': [6] or [2,2] ...
                        'par': a list with initial parameters if mode is given
                        'optimization': {'method': 'SLSQP' (default), 'dual_annealing',
                            'differential_evolution' or 'shgo',
                            'eps', 'ftol', 'maxiter', 'bounds'},
                        'giter': 10,
                        'bic': True or False,
                        'folder_name': 'marginalfit'
                        'file_name': if not given, it is created from input parameters
                        }
                    }

    Returns:
        * dict: the fitting parameters
    """
    # Initial computational time
    start_time = time.time()
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(show_init_message())
    logger.info("Current Time = %s\n" % current_time)

    # Remove nan in the input timeseries
    df = pd.DataFrame(df).dropna()

    # Check if negative and positive values are in the timeseries for fitting purpouses
    if df[df < 0].any().values[0]:
        logger.info(
            "Dataset has negative values. Check that the chosen distribution functions adequately fit negative values."
        )

    # Check that the input dictionary is well defined
    parameters = check_marginal_params(parameters)

    # Transform angles to radian
    if parameters["circular"]:
        df = np.deg2rad(df)

    # Normalized the data using one of the normalization method if it is required
    if parameters["transform"]["make"]:
        df, parameters = stf.transform(df, parameters)
        parameters["transform"]["min"] = df.min().values[0] - 1e-2
        df -= parameters["transform"]["min"]

    # Scale and shift time series for ensuring the use of any PM
    parameters["range"] = float((df.max() - df.min()).values[0])
    if parameters["range"] > 10:
        df = df / (parameters["range"] / 3)
        parameters["scale"] = parameters["range"] / 3

    # Bound the variable with some reference values
    if parameters["circular"]:
        parameters["minimax"] = [0, 2 * np.pi]
    else:
        parameters["minimax"] = [
            float(df[parameters["var"]].min()),
            float(
                np.max(
                    [df[parameters["var"]].max(), df[parameters["var"]].max() * 1.25]
                )
            ),
        ]

    # Calculate the normalize time along the reference oscillating period
    df["n"] = np.fmod(
        (df.index - datetime.datetime(df.index[0].year, 1, 1, 0)).total_seconds().values
        / (parameters["basis_period"][0] * 365.25 * 24 * 3600),
        1,
    )

    logger.info("MARGINAL STATISTICAL FIT")
    logger.info(
        "=============================================================================="
    )
    # Make the full analysis if "mode" is not given or a specify mode wheter "mode" is given
    if not parameters["mode"]:
        # Write the information about the variable, PMs and method
        term = (
            "Stationary fit of "
            + parameters["var"]
            + " to a "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += " - " + str(parameters["fun"][i].name)
        term += " - genpareto " * parameters["reduction"]
        term += " probability model"
        logger.info(term)
        # Make the stationary analysis
        df, parameters["par"], parameters["mode"] = stf.st_analysis(df, parameters)

        # Write the information about the variable, PMs and method
        if parameters["non_stat_analysis"] == True:
            term = (
                "\nNon-stationary fit of "
                + parameters["var"]
                + " with the "
                + str(parameters["fun"][0].name)
            )
            for i in range(1, parameters["no_fun"]):
                term += " - " + str(parameters["fun"][i].name)
            term += " - genpareto " * parameters["reduction"]
            term += " probability model"
            logger.info(term)
            logger.info(
                "with the "
                + parameters["optimization"]["method"]
                + " optimization method."
            )
            logger.info(
                "=============================================================================="
            )
            # Make the non-stationary analysis
            parameters = stf.nonst_analysis(df, parameters)
    else:
        # Write the information about the variable, PMs, method and mode
        term = (
            "Non-stationary fit of "
            + parameters["var"]
            + " with the "
            + str(parameters["fun"][0].name)
        )
        for i in range(1, parameters["no_fun"]):
            term += "-" + str(parameters["fun"][i].name)
        term += " and mode:"
        for mode in parameters["mode"]:
            term += " " + str(mode)
        logger.info(term)
        logger.info(
            "=============================================================================="
        )
        # Make the non-stationary analysis
        parameters = stf.nonst_analysis(df, parameters)

    # Change the object function for its string names
    parameters["fun"] = {i: parameters["fun"][i].name for i in parameters["fun"].keys()}
    parameters["status"] = "Distribution models fitted succesfully"

    # Final computational time
    logger.info("End fitting process")
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    # Save the parameters in the file if "file_name" is given in params
    auxiliar.mkdir(parameters["folder_name"])

    if not "file_name" in parameters.keys():
        filename = parameters["var"] + "_" + str(parameters["fun"][0])
        for i in range(1, parameters["no_fun"]):
            filename += "_" + str(parameters["fun"][i])
        filename += "_genpareto" * parameters["reduction"]

        for i in parameters["ws_ps"]:
            filename += "_" + str(i)

        filename += "_st_" * (not parameters["non_stat_analysis"])
        filename += "_nonst" * parameters["non_stat_analysis"]

        filename += "_" + str(parameters["basis_period"][0])

        filename += "_" + parameters["basis_function"]["method"]
        filename += "_" + str(parameters["basis_function"]["no_terms"])
        filename += "_" + parameters["optimization"]["method"]

        filename = parameters["folder_name"] + filename
        parameters["file_name"] = filename

    save.to_json(parameters, parameters["file_name"])

    # Return the dictionary with the parameters from the analysis
    return parameters


def check_marginal_params(param):
    """Checks the input parameters and includes some required arguments for the computation

    Args:
        * param (dict): the initial guess parameters of the probability models (see the docs of marginalfit)

    Returns:
        * param (dict): checked and updated parameters
    """

    param["no_param"] = {}
    param["scipy"] = {}
    param["reduction"] = False
    param["no_tot_param"] = 0

    logger.info("USER OPTIONS:")
    k = 1

    if not "transform" in param.keys():
        param["transform"] = {}
        param["transform"]["make"] = False
        param["transform"]["plot"] = False
    else:
        if param["transform"]["make"]:
            if not param["transform"]["method"] in ["box-cox", "yeo-johnson"]:
                raise ValueError(
                    "The power transformation methods available are 'yeo-johnson' and 'box-cox', {} given.".format(
                        param["transform"]["method"]
                    )
                )
            else:
                logger.info(
                    str(k)
                    + " - Data is previously normalized ("
                    + param["transform"]["method"]
                    + " method given)".format(str(k))
                )
                k += 1

    # Check if it can be reduced the number of parameters using Solari (2011) analysis
    if len(param["fun"].keys()) == 3:
        if (param["fun"][0] == "genpareto") & (param["fun"][2] == "genpareto"):
            param["reduction"] = True
            logger.info(
                str(k)
                + " - The combination of PMs given enables the reduction"
                + " of parameters during the optimization"
            )
            k += 1

    # Check the number of probability models
    if len(param["fun"].keys()) > 3:
        raise ValueError(
            "No more than three probability models are allowed in this version"
        )

    if param["reduction"]:
        # Particular case where the number of parameters to be optimized is reduced
        param["fun"] = {
            0: getattr(st, param["fun"][0]),
            1: getattr(st, param["fun"][1]),
            2: getattr(st, param["fun"][2]),
        }
        param["no_fun"] = 2
        param["no_param"][0] = int(param["fun"][1].numargs + 2)
        if param["no_param"][0] > 5:
            raise ValueError(
                "Probability models with more than 3 parameters are not allowed in this version"
            )
        param["no_param"][1] = 1
        param["no_tot_param"] = int(param["fun"][1].numargs + 3)
    else:
        param["no_fun"] = len(param["fun"].keys())
        for i in range(param["no_fun"]):
            if isinstance(param["fun"][i], str):
                if param["fun"][i] == "wrap_norm":
                    param["fun"][i] = stf.wrap_norm()
                    param["scipy"][i] = False
                else:
                    param["fun"][i] = getattr(st, param["fun"][i])
                    param["scipy"][i] = True
            param["no_param"][i] = int(param["fun"][i].numargs + 2)
            if param["no_param"][i] > 5:
                raise ValueError(
                    "Probability models with more than 3 parameters are not allowed in this version"
                )
            param["no_tot_param"] += int(param["fun"][i].numargs + 2)

    if not "non_stat_analysis" in param.keys():
        param["basis_period"] = None

    if not "basis_period" in param:
        param["basis_period"] = [1]
    elif param["basis_period"] == None:
        param["order"] = 0
        if param["non_stat_analysis"] == False:
            param["basis_period"] = [1]
    elif isinstance(param["basis_period"], int):
        param["basis_period"] = list(param["basis_period"])

    if not "basis_function" in param.keys():
        raise ValueError("Basis function is required when non_stat_analysis is True.")

    if not "method" in param["basis_function"]:
        raise ValueError("Method is required when non_stat_analysis is True.")
    else:
        if param["basis_function"]["method"] in [
            "trigonometric",
            "sinusoidal",
            "modified",
        ]:
            if ((not "no_terms") & (not "periods")) in param["basis_function"].keys():
                raise ValueError(
                    "Number of terms or periods are required for Fourier Series approximation."
                )
            else:
                if not "periods" in param["basis_function"]:
                    param["basis_function"]["periods"] = list(
                        1 / np.arange(1, param["basis_function"]["no_terms"] + 1)
                    )
                    param["basis_function"]["order"] = param["basis_function"][
                        "no_terms"
                    ]
                else:
                    param["basis_function"]["no_terms"] = len(
                        param["basis_function"]["periods"]
                    )
                    param["basis_function"]["order"] = param["basis_function"][
                        "no_terms"
                    ]
                # param["approximation"]["periods"].sort(reverse=True)
                if not "basis_period" in param:
                    param["basis_period"] = [param["basis_function"]["periods"][0]]
        else:
            if param["basis_function"]["method"] not in [
                "chebyshev",
                "legendre",
                "laguerre",
                "hermite",
                "ehermite",
                "polynomial",
            ]:
                raise ValueError(
                    "Method available are:\
                    trigonometric, modified, sinusoidal, \
                    chebyshev, legendre, laguerre, hermite, ehermite or polynomial.\
                    Given {}.".format(
                        param["basis_function"]["method"]
                    )
                )
            else:
                if not "degree" in param["basis_function"].keys():
                    raise ValueError("The polynomial methods require the degree")
                param["basis_function"]["order"] = param["basis_function"]["degree"]
                if not "basis_period" in param:
                    param["basis_period"] = [param["basis_function"]["periods"][0]]

        logger.info(
            str(k)
            + " - The basis function given is {}.".format(
                param["basis_function"]["method"]
            )
        )
        k += 1

        logger.info(
            str(k)
            + " - The number of terms given is {}.".format(
                param["basis_function"]["order"]
            )
        )
        k += 1

    if not "par" in param.keys():
        param["par"], param["mode"] = {}, {}
    else:
        if not "mode" in param.keys():
            raise ValueError(
                "The evaluation of a mode required the initial parameters 'par'. Give the par."
            )
        else:
            logger.info(
                str(k) + " - Mode of optimization given ({}).".format(param["mode"])
            )
            k += 1

    if not "optimization" in param.keys():
        param["optimization"] = {}
        param["optimization"]["method"] = "SLSQP"
        param["optimization"]["eps"] = 1e-7
        param["optimization"]["maxiter"] = 1e2
        param["optimization"]["ftol"] = 1e-4
    else:
        if param["optimization"] is None:
            param["optimization"] = {}
            param["optimization"]["method"] = "SLSQP"
            param["optimization"]["eps"] = 1e-7
            param["optimization"]["maxiter"] = 1e2
            param["optimization"]["ftol"] = 1e-4
        else:
            param["optimization"]["eps"] = 1e-7
            param["optimization"]["maxiter"] = 1e2
            param["optimization"]["ftol"] = 1e-4

    if not "method" in param["optimization"]:
        param["optimization"]["method"] = "SLSQP"

    if not "giter" in param["optimization"].keys():
        param["optimization"]["giter"] = 10
    else:
        if not isinstance(param["optimization"]["giter"], int):
            raise ValueError("The number of global iterations should be an integer.")
        else:
            logger.info(
                "{} - Global iterations were given by user ({})".format(
                    str(k), str(param["optimization"]["giter"])
                )
            )
            k += 1

    if not "bounds" in param["optimization"].keys():
        param["optimization"]["bounds"] = 0.5
    else:
        if not isinstance(param["optimization"]["bounds"], (float, int, bool)):
            raise ValueError("The bounds should be a float, integer or False.")
        else:
            logger.info(
                "{} - Bounds were given by user (bounds = {})".format(
                    str(k), str(param["optimization"]["bounds"])
                )
            )
            k += 1

    param["constraints"] = True
    if "piecewise" in param:
        if not param["reduction"]:
            if param["piecewise"]:
                param["constraints"] = False
                logger.info(
                    str(k)
                    + " - Piecewise analysis of PMs defined by user. Piecewise is set to True."
                )
                k += 1
        else:
            logger.info(
                str(k)
                + " - Piecewise analysis is not recommended when reduction is applied. Piecewise is set to False."
            )
            param["piecewise"] = False
            k += 1
    else:
        param["piecewise"] = False

    if param["no_fun"] == 1:
        param["constraints"] = False

    if param["reduction"]:
        param["constraints"] = False

    if not "transform" in param.keys():
        param["transform"] = {"make": False, "method": None, "plot": False}

    # if "debug" in param.keys():
    #     if param["debug"]:
    #         logger.add(
    #             "debug_file.log", format="{message}", level="DEBUG", rotation="5 MB"
    #         )
    #         logger.info("{} - Debug mode ON.".format(str(k)))
    #         k += 1

    if param["reduction"]:
        if len(param["ws_ps"]) != 2:
            raise ValueError(
                "Expected two percentiles for the analysis. Got {}.".format(
                    str(len(param["ws_ps"]))
                )
            )
    else:
        if (not "ws_ps" in param) & (param["no_fun"] - 1 == 0):
            param["ws_ps"] = []
        elif (not "ws_ps" in param) & (param["no_fun"] - 1 != 0):
            raise ValueError(
                "Expected {} weight\s for the analysis. However ws_ps option is not given.".format(
                    str(param["no_fun"] - 1)
                )
            )

        if len(param["ws_ps"]) != param["no_fun"] - 1:
            raise ValueError(
                "Expected {} weight\s for the analysis. Got {}.".format(
                    str(param["no_fun"] - 1), str(len(param["ws_ps"]))
                )
            )

    if not "circular" in param.keys():
        logger.info(
            "{} - Circular parameters is not given. Assuming that the variable is not circular.".format(
                str(k)
            )
        )
        k += 1
        param["circular"] = False

    if any(np.asarray(param["ws_ps"]) > 1) or any(np.asarray(param["ws_ps"]) < 0):
        raise ValueError(
            "percentiles cannot be lower than 0 or bigger than one. Got {}.".format(
                str(param["ws_ps"])
            )
        )

    if not "guess" in param.keys():
        param["guess"] = False

    if not "bic" in param.keys():
        param["bic"] = False

    if param["constraints"]:
        if (not param["optimization"]["method"] == "SLSQP") & (param["no_fun"] > 1):
            raise ValueError(
                "Constraints are just available for SLSQP method in this version."
            )

    if "fix_percentiles" in param.keys():
        if param["fix_percentiles"]:
            logger.info(
                "{} - Percentiles are fixed. Fix_percentiles is set to True.".format(
                    str(k)
                )
            )
            k += 1
    else:
        param["fix_percentiles"] = False

    if not "folder_name" in param.keys():
        param["folder_name"] = "marginalfit/"
    else:
        param["folder_name"] += "/marginalfit/"

    if k == 1:
        logger.info("None.")

    logger.info(
        "==============================================================================\n"
    )

    return param


def nanoise(
    data: pd.DataFrame, variable: str, remove: bool = False, filter_: str = None
):
    """Adds noise to time series for better estimations

    Args:
        * data (pd.DataFrame): raw time series
        * variable (string): variable to apply noise
        * remove (bool): if True filtered data is removed
        * filter_ (list): lower limit of values to be filtered. See query pandas DataFrame.

    Returns:
        * df_out (pd.DataFrame): time series with noise
    """

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if isinstance(variable, str):
        variable = [variable]

    # Filtering data
    if filter_ is not None:
        data = data.query(filter_)

    # Remove nans
    df = data.dropna()
    for var_ in variable:
        if not df[var_].empty:
            increments = st.mode(np.diff(np.sort(data[var_].unique())))[0]
            df[[var_]] = df[[var_]] + np.random.rand(len(df[var_]), 1) * increments
        else:
            raise ValueError("Input time series is empty.")

    # Removing data
    if remove:
        df = df.loc[df[variable] == filter_, variable]

    return df


def dependencies(df: pd.DataFrame, param: dict):
    """Computes the temporal dependency using a VAR model (Solari & van Gelder, 2011;
    Solari & Losada, 2011).

    Args:
        - df (pd.DataFrame): raw time series
        - param (dict): parameters of dt. 'mvar' is the main variable,
        'threshold' stands for the threshold of the main variable,
        'vars' is a list with the name of all variables,
        'varord' is the order of the VAR model,
        'events' is True or False standing for storm analysis (Lira-Loarca et al, 2020)
            or Full simulation, 'fname' is output file name.
        - method (string): name of the multivariate method of dependence. Defaults to "VAR".

    Returns:
        - df_dt (dict): parameters of the fitting process
    """
    logger.info(show_init_message())

    logger.info("MULTIVARIATE DEPENDENCY")
    logger.info(
        "=============================================================================="
    )

    # Remove nan in the input timeseries
    df = pd.DataFrame(df).dropna()

    # Reading parameters
    for var_ in param["TD"]["vars"]:
        try:
            param[var_] = read.rjson(param[var_]["fname"])
        except:
            raise ValueError("File {} not found.".format(param[var_]["fname"]))

    # Remove nans
    df.dropna(inplace=True)

    # Check that the input dictionary is well defined
    param["TD"] = check_dependencies_params(param["TD"])

    # Compute: (1) the univariate and temporal analysis is one variable is given,
    #          (2) the multivariate and temporal analysis is more than one is given
    logger.info(
        "Computing the parameters of the stationary {} model up to {} order.".format(
            param["TD"]["method"], param["TD"]["order"]
        )
    )
    logger.info(
        "=============================================================================="
    )

    # Compute the normalize time using the maximum period
    df["n"] = (
        (df.index.dayofyear + df.index.hour / 24.0 - 1)
        / pd.to_datetime(
            {"year": df.index.year, "month": 12, "day": 31, "hour": 23}
        ).dt.dayofyear
    ).values

    # Transform angles into radians
    for var_ in param["TD"]["vars"]:
        if param[var_]["circular"]:
            df[var_] = np.deg2rad(df[var_])

    cdf_ = pd.DataFrame(index=df.index, columns=param["TD"]["vars"])

    for var_ in param["TD"]["vars"]:
        param[var_]["order"] = np.max(param[var_]["mode"])
        param = auxiliar.str2fun(param, var_)

        variable = pd.DataFrame(df[var_].values, index=df.index, columns=["data"])
        variable["n"] = df["n"].values
        # variable[var_] = df[var_].values

        # Transformed timeserie
        if param[var_]["transform"]["make"]:
            variable["data"], _ = transform(variable["data"], param[var_])
            variable["data"] -= param[var_]["transform"]["min"]

        if "scale" in param[var_]:
            variable["data"] = variable["data"] / param[var_]["scale"]

        # Compute the CDF using the estimated parameters
        cdf_[var_] = stf.cdf(variable, param[var_])

        # Remove outlayers
        if any(cdf_[var_] >= 1 - 1e-6):
            print(
                "Warning: Casting with {} cdf-values of {} next to one (F({}) > 1-1e-6).".format(
                    str(np.sum(cdf_[var_] >= 1 - 1e-6)), var_, var_
                )
            )
            cdf_.loc[cdf_[var_] >= 1 - 1e-6, var_] = 1 - 1e-6

        if any(cdf_[var_] <= 1e-6):
            print(
                "Casting with {} cdf-values of {} next to zero (F({}) < 1e-6).".format(
                    str(np.sum(cdf_[var_] <= 1e-6)), var_, var_
                )
            )
            cdf_.loc[cdf_[var_] <= 1e-6, var_] = 1e-6

        # If "events" is True, the conditional analysis over threshold following the
        # steps given in Lira-Loarca et al (2019) is applied
        if (var_ == param["TD"]["mvar"]) & param["TD"]["events"]:
            print(
                "Computing conditioned probability models to the threshold of the main variable"
            )
            cdfj = cdf_[var_].copy()
            variable = pd.DataFrame(
                np.ones(len(df["n"])) * param["TD"]["threshold"],
                index=df.index,
                columns=["data"],
            )
            variable[var_] = df[var_].values
            variable["n"] = df["n"].values
            cdfu = stf.cdf(variable, param[var_])
            cdf_umbral = pd.DataFrame(cdfu)
            cdf_umbral["n"] = variable["n"]
            cdf_[var_] = (cdfj - cdfu) / (1 - cdfu)

    # Remove nans in CDF
    if any(np.sum(np.isnan(cdf_))):
        print(
            "Some Nan ("
            + str(np.sum(np.sum(np.isnan(cdf_))))
            + " values) are founds before the normalization."
        )
        cdf_[np.isnan(cdf_)] = 0.5

    if param["TD"]["method"] == "VAR":
        z = np.zeros(np.shape(cdf_))
        # Normalize the CDF of every variable
        for ind_, var_ in enumerate(cdf_):
            z[:, ind_] = st.norm.ppf(cdf_[var_].values)

        # Fit the parameters of the AR/VAR(p) model
        df_dt = varfit(z.T, param["TD"]["order"])
    else:
        print("No more methods are yet available.")

    # Save to json file
    auxiliar.mkdir("dependency")
    filename = ""
    for var_ in param["TD"]["vars"]:
        filename += var_ + "_"
    filename += str(param["TD"]["order"]) + "_"
    filename += param["TD"]["method"]

    param["TD"]["fname"] = "dependency/" + filename

    save.to_json(df_dt, param["TD"]["fname"], True)

    # Clean memory usage
    del cdf_, param

    return df_dt


def check_dependencies_params(param: dict):
    """Checks the input parameters and includes some required arguments for the computation of multivariate dependencies

    Args:
        * param (dict): the initial guess parameters of the probability models

    Returns:
        * param (dict): checked and updated parameters
    """

    logger.info("USER OPTIONS:")
    k = 1

    if not "method" in param.keys():
        param["method"] = "VAR"
        logger.info(str(k) + " - VAR method used")
        k += 1

    if not "events" in param.keys():
        param["events"] = False

    logger.info(
        "==============================================================================\n"
    )

    return param


def varfit(data: np.ndarray, order: int):
    """Computes the coefficientes of the VAR(p) model and chooses the model with lowest BIC.

    Args:
        * data (np.ndarray): normalize data with its probability model
        * order (int): maximum order (p) of the VAR model

    Returns:
        * par_dt (dict): parameter of the temporal dependency using VAR model
    """

    # Create the list of output parameters
    [dim, t] = np.shape(data)
    t = t - order
    bic, r2adj = np.zeros(order), []

    par_dt = [list() for i in range(order)]
    for p in range(1, order + 1):
        # Create the matrix of input data for p-order
        y = data[:, order:]
        z0 = np.zeros([p * dim, t])
        for i in range(1, p + 1):
            z0[(i - 1) * dim : i * dim, :] = data[:, order - i : -i]
        z = np.vstack((np.ones(t), z0))
        # Estimated the parameters using the ordinary least squared error analysis
        par_dt[p - 1], bic[p - 1], r2a = varfit_OLS(y, z)
        r2adj.append(r2a)

    # Select the minimum BIC and return the parameter associated to it
    id_ = np.argmin(bic)
    par_dt = par_dt[id_]
    par_dt["id"] = int(id_ + 1)
    par_dt["bic"] = [float(bicValue) for bicValue in bic]
    par_dt["R2adj"] = r2adj[par_dt["id"]]
    logger.info(
        "Minimum BIC ("
        + str(par_dt["bic"][par_dt["id"]])
        + ") obtained for p-order "
        + str(par_dt["id"])
        + " and R2adj: "
        + str(par_dt["R2adj"])
    )
    logger.info(
        "=============================================================================="
    )

    if id_ + 1 == order:
        logger.info(
            "Warning! The lower BIC is obtained with the higher order model. Increase the p-order."
        )

    return par_dt


def varfit_OLS(y, z):
    """Estimates the parameters of VAR using the RMSE described in Lutkepohl (ecs. 3.2.1 and 3.2.10)

    Args:
        * y: X Matrix in Lutkepohl
        * z: Z Matrix in Lutkepohl

    Returns:
        * df (dict): matrices B, Q, y U
        * bic (float): Bayesian Information Criteria
        * R2adj (float): correlation factor
    """

    df = dict()
    m1, m2 = np.dot(y, z.T), np.dot(z, z.T)

    # Estimate the parameters
    df["B"] = np.dot(m1, np.linalg.inv(m2))

    nel, df["dim"] = np.shape(df["B"].T)
    df["U"] = y - np.dot(df["B"], z)
    # Estimate de covariance matrix
    df["Q"] = np.cov(df["U"])
    df["y"] = y
    error_ = np.random.multivariate_normal(np.zeros(df["dim"]), df["Q"], z.shape[1]).T
    df["y*"] = np.dot(df["B"], z) + error_

    # Estimate R2 and R2-adjusted parameters
    R2 = 1 - np.sum((y - df["y*"]) ** 2, axis=1) / np.sum((y - np.mean(y)) ** 2, axis=1)
    R2adj = 1 - (1 - R2) * (len(z.T) - 1) / (len(z.T) - nel - 1)

    # rmse = np.sqrt(np.sum((st.norm.cdf(y) - st.norm.cdf(np.dot(df["B"], z))) ** 2, axis=1)/y.shape[1])
    # mae = np.sum(np.abs(st.norm.cdf(y) - st.norm.cdf(np.dot(df["B"], z))), axis=1)/y.shape[1]
    # print(rmse, mae)

    # Compute the LLF
    multivariatePdf = st.multivariate_normal.pdf(
        df["U"].T, mean=np.zeros(df["dim"]), cov=df["Q"]
    )
    mask = multivariatePdf > 0
    if len(multivariatePdf) != len(multivariatePdf[mask]):
        print(
            "Order: "
            + str(int((nel - 1) / (df["dim"]) - 1))
            + ". Casting with {} values equal to zero of the multivariate pdf. Values removed.".format(
                str(np.sum(~mask))
            )
        )
        llf = np.sum(np.log(multivariatePdf[mask]))
    else:
        llf = np.sum(np.log(multivariatePdf))

    # aic = df['dim']*np.log(np.sum(np.abs(y - np.dot(df['B'], z)))) + 2*nel
    # Compute the BIC
    bic = -2 * llf + np.log(np.size(y)) * np.size(np.hstack((df["B"], df["Q"])))

    return df, bic, R2adj.tolist()


def ensemble_dt(models: dict, percentiles="equally"):
    """Compute the ensemble of multivariate and temporal dependency parameters

    Args:
        * models (dict): models where parameters of temporal dependencies are saved
        * percentiles (string or list): "equally" is equally probability is given for RCMs
            and a list with percentiles of every RCMs if not

    Returns:
        [type]: [description]
    """
    # Initialize matrices
    B, Q = [], []
    # Read the parameter of every ensemble model
    for model_ in models.keys:
        df_dt = read.rjson(models[model_], "td")
        B.append(df_dt["B"])
        Q.append(df_dt["Q"])

    nmodels = len(B)
    norders = np.max([np.shape(Bm) for Bm in B], axis=0)

    Bs = np.zeros([norders[0], norders[1], nmodels])

    # Compute the ensemble using percentiles
    for i in range(nmodels):
        if percentiles == "equally":
            Bs[:, : np.shape(B[i])[1], i] = B[i]
        else:
            Bs[:, : np.shape(B[i])[1], i] = B[i] * percentiles[i]

    # Compute the ensemble using percentiles
    if percentiles == "equally":
        B, Q = np.mean(Bs, axis=2), np.mean(Q, axis=0)
    else:
        B, Q = np.sum(Bs, axis=2), np.sum(Q, axis=0)

    # Create a dictionary with parameters of the ensemble
    df_dt_ensemble = dict()
    df_dt_ensemble["B"], df_dt_ensemble["Q"], df_dt_ensemble["id"] = (
        B,
        Q,
        int((norders[1] - 1) / norders[0]),
    )  # ord_

    # Create the fit directory and save the parameters to a json file
    auxiliar.mkdir("fit")
    save.to_json(df_dt_ensemble, "fit/ensemble_df_dt", True)
    return B, Q, int((norders[1] - 1) / norders[0])
