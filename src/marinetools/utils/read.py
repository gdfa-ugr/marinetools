import json
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import dates
from scipy.io import loadmat as ldm


def keys_as_int(obj: dict):
    """Convert the keys at reading json file into a dictionary of integers

    Args:
        obj (dict):input dictionary

    Returns:
        out: the dictionary
    """
    try:
        out = {int(k): v for k, v in obj.items()}
    except:
        out = {k: v for k, v in obj.items()}
    return out


def keys_as_nparray(obj: dict):
    """Convert the keys at reading json file into a dictionary of np.arrays

    Args:
        obj (dict): input dictionary

    Returns:
        out: the dictionary
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


def rjson(file_name: str, parType: str = None):
    """Reads data from json files

    Args:
        - fname (string): filename of data
        - parType (None or string): defines the type of data to be read.
        Defaults to read param and "td" from temporal dependency.

    Returns:
        - data (pd.DataFrame): the read data
    """
    if parType == "td":
        params = json.load(open(file_name + ".json", "r"), object_hook=keys_as_nparray)
    else:
        params = json.load(open(file_name + ".json", "r"), object_hook=keys_as_int)

    return params


def PdE(file_name: str):
    """Reads data from PdE files

    Args:
        - fname (string): filename of data

    Returns:
        - data (pd.DataFrame): the read data
    """
    with open(file_name) as file_:
        content = file_.readlines()

    for ind_, line_ in enumerate(content):
        if "LISTADO DE DATOS" in line_:
            skiprows = ind_ + 2

    data = pd.read_table(
        file_name,
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
    file_name: str,
    ts: bool = False,
    date_parser=None,
    sep: str = ",",
    encoding: str = "utf-8",
    non_natural_date: bool = False,
    no_data_values: int = -999,
):
    """Reads a csv file

    Args:
        - file_name (string): filename of data
        - ts (boolean, optional): stands for a time series
        (the index is a datetime index) or not
        - date_parser: required for special datetime formats
        - sep: separator
        - encoding: type of encoding
        - non_natural days: some models return 30 days/month which generate problems with real timeseries
        - no_data_values: integer with values that will be considered as nan

    Returns:
        - data (pd.DataFrame): the read data
    """
    if not any(item in str(file_name) for item in ["txt", "csv", "zip"]):
        filename = str(file_name) + ".csv"
    else:
        filename = str(file_name)

    if non_natural_date:
        ts = False

    if not ts:
        if "zip" in filename:
            data = pd.read_csv(
                filename,
                sep=sep,
                index_col=[0],
                compression="zip",
                engine="python",
            )
        else:
            try:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    index_col=[0],
                    engine="python",
                    encoding=encoding,
                )
            except:
                data = pd.read_csv(
                    filename, sep=sep, engine="python", encoding=encoding
                )

        if non_natural_date:
            start = pd.to_datetime(data.index[0])
            # days = timedelta(np.arange(len(data)))
            index_ = [
                start + timedelta(nodays)
                for nodays in np.arange(len(data), dtype=np.float64)
            ]
            data.index = index_
    else:
        if "zip" in filename:
            try:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=[0],
                    index_col=[0],
                    compression="zip",
                    date_parser=date_parser,
                )
            except:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=[0],
                    index_col=[0],
                    date_parser=date_parser,
                )
                logger.info("{}, It is not a zip file.".format(str(filename) + ".csv"))
        else:
            try:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=["date"],
                    index_col=["date"],
                    date_parser=date_parser,
                )
            except:
                data = pd.read_csv(
                    filename,
                    sep=sep,
                    parse_dates=[0],
                    index_col=[0],
                    date_parser=date_parser,
                )
    data = data[data != no_data_values]
    return data


def npy(file_name: str):
    """Reads data from a numpy file

    Args:
        - file_name (string): filename of data

    Returns:
        - data (pd.DataFrame): the read data
    """
    try:
        data = np.load(f"{file_name}.npy")
    except:
        data = np.load(f"{file_name}.npy", allow_pickle=True)
        if not isinstance(data, pd.DataFrame):
            data = {i: data.item().get(i) for i in data.item()}

    return data


def xlsx(file_name: str, sheet_name: str = 0):
    """Reads xlsx files

    Args:
        - file_name (string): filename of data
        - sheet_name (string): name of sheet

    Returns:
        - data (pd.DataFrame): the read data
    """
    xlsx = pd.ExcelFile(file_name + ".xlsx")
    data = pd.read_excel(xlsx, sheet_name=sheet_name, index_col=0)
    return data


def netcdf(
    file_name: str,
    variables: str = None,
    latlon: list = None,
    depth: float = None,
    time_series: bool = True,
    glob: bool = False,
):
    """Reads netCDF4 files

    Args:
        - fname (string): filename of data or directory where glob will
        be applied (required command glob True)
        - variables (list): strings with the name of the objective variables

    Returns:
        - df (pd.DataFrame): the read data
    """

    import xarray as xr

    # data = Dataset(fname + '.nc', mode='r', format='NETCDF4')
    if not glob:
        data = xr.open_dataset(file_name + ".nc")
    else:
        try:
            data = xr.open_mfdataset(file_name)
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
        # df = data.to_dataframe()  # TODO: change with more than one variable
        if time_series:
            if not data.indexes["time"].dtype == "datetime64[ns]":
                times, goodPoints = [], []
                for index, time in enumerate(data.indexes["time"]):
                    try:
                        times.append(time._to_real_datetime())
                        goodPoints.append(index)
                    except:
                        continue
                if variables is not None:
                    data = pd.DataFrame(
                        data.to_dataframe()[variables].values[goodPoints],
                        index=times,
                        columns=[variables],
                    )
                else:
                    pd.DataFrame(data.to_dataframe().values[goodPoints], index=times)
            else:
                data = pd.DataFrame(
                    data.to_dataframe()[variables].values,
                    index=data.indexes["time"],
                    columns=[variables],
                )

    return data


def mat(file_name: str, var_: str = "x", julian: bool = False):
    """[summary]

    Args:
        file_name ([type]): [description]
        var_ (str, optional): [description]. Defaults to "x".
        julian (bool, optional): [description]. Defaults to False.

    Returns:
        df (pd.DataFrame): the readed timeseries
    """

    data = ldm(file_name)
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
