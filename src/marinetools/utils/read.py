import json
import sys
import warnings
from datetime import datetime, timedelta

import glob2
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import dates
from scipy.io import loadmat as ldm


def keys_as_int(obj):
    """[summary]

    Args:
        obj ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(obj, dict):
        try:
            out = {int(k): v for k, v in obj.items()}
        except:
            out = {k: v for k, v in obj.items()}
    return out


def keys_as_nparray(obj):
    """[summary]

    Args:
        obj ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(obj, dict):
        out = {}
        for item0, level0 in obj.items():
            if isinstance(level0, dict):
                out[item0] = {}
                for item1, level1 in level0.items():
                    if isinstance(level1, dict):
                        out[item0][item1] = {}
                        for item2, level2 in level1.items():
                            try:
                                out[item0][item1][item2] = np.asarray(level2)
                            except:
                                out[item0][item1][item2] = level2

                    else:
                        try:
                            out[item0][item1] = np.asarray(level1)
                        except:
                            out[item0][item1] = level1
            else:
                try:
                    out[item0] = np.asarray(level0)
                except:
                    out[item0] = level0

    return out


def rjson(fname, parType=None):
    """Reads data from json files

    Args:
        - fname (string): filename of data
        - parType (None or string): defines the type of data to be read.
        Defaults to read param and "td" from temporal dependency.

    Returns:
        - data (pd.DataFrame): the read data
    """
    if parType == "td":
        params = json.load(open(fname + ".json", "r"), object_hook=keys_as_nparray)
    else:
        params = json.load(open(fname + ".json", "r"), object_hook=keys_as_int)

    return params


def PdE(fname):
    """Reads data from PdE files

    Args:
        - fname (string): filename of data

    Returns:
        - data (pd.DataFrame): the read data
    """
    with open(fname) as file_:
        content = file_.readlines()

    for ind_, line_ in enumerate(content):
        if "LISTADO DE DATOS" in line_:
            skiprows = ind_ + 2

    data = pd.read_table(
        fname,
        delimiter="\s+",
        parse_dates={"date": [0, 1, 2, 3]},
        index_col="date",
        skiprows=skiprows,
        engine="python",
    )

    data.replace(-100, np.nan, inplace=True)
    data.replace(-99.9, np.nan, inplace=True)
    data.replace(-99.99, np.nan, inplace=True)
    data.replace(-9999, np.nan, inplace=True)
    return data


def csv(
    filename,
    ts=False,
    date_parser=None,
    sep=",",
    non_natural_date=False,
    no_data_values=-999,
):
    """Reads a csv file

    Args:
        - filename (string): filename of data
        - ts (boolean, optional): stands for a time series
        (the index is a datetime index) or not
        - date_parser: required for special datetime formats
        - sep: separator
        - non_natural days: some models return 30 days/month which generate problems with real timeseries.

    Retunrs:
        - data (pd.DataFrame): the read data
    """
    if non_natural_date:
        ts = False

    if not ts:
        if "zip" in str(filename):
            data = pd.read_csv(
                str(filename) + ".csv",
                sep=sep,
                index_col=[0],
                compression="zip",
                engine="python",
            )
        else:
            try:
                data = pd.read_csv(
                    str(filename) + ".csv", sep=sep, index_col=[0], engine="python"
                )
            except:
                data = pd.read_csv(str(filename) + ".csv", sep=sep, engine="python")

        if non_natural_date:
            start = pd.to_datetime(data.index[0])
            # days = timedelta(np.arange(len(data)))
            index_ = [
                start + timedelta(nodays)
                for nodays in np.arange(len(data), dtype=np.float64)
            ]
            data.index = index_
    else:
        if "zip" in str(filename):
            data = pd.read_csv(
                str(filename) + ".csv",
                sep=sep,
                parse_dates=[0],
                index_col=[0],
                compression="zip",
                date_parser=date_parser,
            )
        else:
            try:
                data = pd.read_csv(
                    str(filename) + ".csv",
                    sep=sep,
                    parse_dates=["date"],
                    index_col=["date"],
                    date_parser=date_parser,
                )
            except:
                data = pd.read_csv(
                    str(filename) + ".csv",
                    sep=sep,
                    parse_dates=[0],
                    index_col=[0],
                    date_parser=date_parser,
                )
    data = data[data != no_data_values]
    return data


def npy(fname):
    """Reads data from a numpy file

    Args:
        - fname (string): filename of data

    Returns:
        - data (pd.DataFrame): the read data
    """
    try:
        data = np.load(f"{fname}.npy")
    except:
        data = np.load(f"{fname}.npy", allow_pickle=True)
        if not isinstance(data, pd.DataFrame):
            data = {i: data.item().get(i) for i in data.item()}

    return data


def xlsx(fname):
    """Reads xlsx files

    Args:
        - fname (string): filename of data

    Returns:
        - data (pd.DataFrame): the read data
    """
    xlsx = pd.ExcelFile(fname + ".xlsx")
    data = pd.read_excel(xlsx, index_col=0)
    return data


def netcdf(fname, variables=None, latlon=None, depth=None, glob=False):
    """Reads netCDF4 files

    Args:
        - fname (string): filename of data or directory where glob will
        be applied (required command glob True)
        - variables (list): strings with the name of the objective variables

    Returns:
        - df (pd.DataFrame): the read data
    """
    # data = Dataset(fname + '.nc', mode='r', format='NETCDF4')
    if not glob:
        data = xr.open_dataset(fname + ".nc")
    else:
        try:
            data = xr.open_mfdataset(fname + "/*.nc")
        except:
            # try:
            #     files_ = glob2.glob(fname)
            raise ValueError(
                "NetCDF files are not consistents."
                "Some variables are not adequately saved."
                "Glob version cannot be used for this dataset."
            )

    if isinstance(latlon, list):
        try:
            import xrutils as xru
        except:
            raise ValueError("xrutils package is required. Clone it from gdfa.ugr.es")

        if not depth:
            data = xru.nearest(data, latlon[0], latlon[1])
            # data = data.sel(
            #     rlon=lonlat[0], rlat=lonlat[1], method="nearest"
            # ).to_dataframe()
            data = data.to_dataframe()

            try:
                data = data.loc[0]
                data.index = pd.to_datetime(data.index)
            except:
                print("Data has not bounds.")

            nearestLonLat = data.lon.values[0], data.lat.values[0]
        else:
            data = data.sel(
                depth=0.494025,
                longitude=latlon[0],
                latitude=latlon[1],
                method="nearest",
            ).to_dataframe()
        print("Nearest lon-lat point: ", nearestLonLat)
        if variables is not None:
            data = data[[variables]]
        # data.index = data.to_datetimeindex(unsafe=False)
    else:
        # TODO: hacer el nearest en otra funcion
        df = data.to_dataframe()  # TODO: change with more than one variable
        if not df.index.dtype == "<M8[ns]":
            times, goodPoints = [], []
            for index, time in enumerate(data.indexes["time"]):
                try:
                    times.append(time._to_real_datetime())
                    goodPoints.append(index)
                except:
                    continue
            if variables is not None:
                pd.DataFrame(
                    data.to_dataframe()[variables].values[goodPoints],
                    index=times,
                    columns=[variables],
                )
            else:
                pd.DataFrame(data.to_dataframe().values[goodPoints], index=times)
        nearestLonLat = None

    return data


def mat(fname, var_="x", julian=False):
    """[summary]

    Args:
        fname ([type]): [description]
        var_ (str, optional): [description]. Defaults to "x".
        julian (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    data = ldm(fname)
    if not julian:
        date = data[var_][:, 0] + dates.date2num(
            np.datetime64("0000-12-31")
        )  # Added in matplotlib 3.3
        date = [dates.num2date(i - 366, tz=None) for i in date]
    else:
        date = data[var_][:, 0]

    df = pd.DataFrame({"Q": data[var_][:, 1]}, index=date)
    df.index = df.index.tz_localize(None)
    return df
