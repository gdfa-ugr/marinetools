import json

import numpy as np
import pandas as pd


def npy2json(params: dict):
    """Convert a dictionary with numpy ndarray into json dictionary and save the file

    Args:
        params (dict): parameters to be transformed into json
    """
    for key in params.keys():
        if isinstance(params[key], np.ndarray):
            params[key] = list(params[key])

    for loc, mode in enumerate(params["mode"]):
        params["mode"][loc] = int(mode)

    if "all" in params.keys():
        for loc, mode in enumerate(params["all"]):
            params["all"][loc] = [str(mode[0]), float(mode[1]), mode[2].tolist()]

    for loc, fun in enumerate(params["fun"]):
        if not isinstance(fun, str):
            params["fun"][loc] = params["fun"][loc].name

    to_json(params, params["fname"])

    return


def to_json(params: dict, file_name: str, npArraySerialization: bool = False):
    """Saves to a json file

    Args:
        - params (dict): data to be saved
        - file_name (string): path of the file
        - npArraySerialization (bool): applied a serialization. True or False.

    Return:
        - None
    """

    with open(f"{str(file_name)}.json", "w") as f:
        if npArraySerialization:
            for key in params.keys():
                if isinstance(params[key], dict):
                    for subkey in params[key].keys():
                        try:
                            params[key][subkey] = params[key][subkey].tolist()
                        except:
                            None
                else:
                    try:
                        params[key] = params[key].tolist()
                    except:
                        None
        json.dump(params, f, ensure_ascii=False, indent=4)

    return


def to_csv(data: pd.DataFrame, file_name: str, compression: str = "infer"):
    """Saves to a csv file

    Args:
        - data (pd.DataFrame): data to be saved
        - file_name (str): path of the file
        - compression (str, opt): define the type of compression if required

    Return:
        - None
    """
    if not ".csv" in file_name:
        file_name = str(file_name) + ".csv"
    else:
        file_name = str(file_name)

    if ".zip" in file_name:
        data.to_csv(file_name, compression="zip")
    else:
        data.to_csv(file_name, compression=compression)

    return


def to_npy(data: np.ndarray, file_name: str):
    """Saves to a numpy file

    Args:
        - params (dict): data to be saved
        - fname (string): path of the file

    Return:
        - None
    """
    np.save(f"{str(file_name)}.npy", data)
    return


def to_xlsx(data: pd.DataFrame, file_name: str):
    """Saves to an excel file

    Args:
        - params (dict): data to be saved
        - file_name (string): path of the file

    Return:
        - None
    """

    wbook, wsheet = cwriter(str(file_name) + ".xlsx")

    # Writting the header
    if data.index.name is not None:
        wsheet.write(0, 0, data.index.name, formats(wbook, "header"))
    else:
        wsheet.write(0, 0, "Index", formats(wbook, "header"))

    for col_num, value in enumerate(data.columns.values):
        wsheet.write(0, col_num + 1, value, formats(wbook, "header"))

    # Adding data
    k = 1
    for i in data.index:
        if k % 2 == 0:
            fmt = "even"
        else:
            fmt = "odd"
        wsheet.write_row(k, 0, np.append(i, data.loc[i, :]), formats(wbook, fmt))
        k += 1

    wbook.close()
    return


def cwriter(file_out: str):
    """Create a new file with the book and sheet for excel

    Args:
        - file_out (string): path of the file

    Returns:
        - wbook (objects): excel book
        - wsheet (objects): excel sheet
    """
    writer = pd.ExcelWriter(file_out, engine="xlsxwriter")
    df = pd.DataFrame([0])
    df.to_excel(writer, index=False, sheet_name="Sheet1", startrow=1, header=False)
    wsheet = writer.sheets["Sheet1"]
    wbook = writer.book
    return wbook, wsheet


def formats(wbook, style):
    """Gives some formats to excel file

    Args:
        - wbook (object): excel book
        - style (string): name of the style

    Returns:
        - Adds the format to the woorkbook
    """
    fmt = {
        "header": {
            "bold": True,
            "text_wrap": True,
            "valign": "center",
            "font_color": "#ffffff",
            "fg_color": "#5983B0",
            "border": 1,
        },
        "even": {
            "bold": False,
            "text_wrap": False,
            "valign": "center",
            "fg_color": "#DEE6EF",
            "border": 1,
        },
        "odd": {
            "bold": False,
            "text_wrap": False,
            "valign": "center",
            "fg_color": "#FFFFFF",
            "border": 1,
        },
    }

    return wbook.add_format(fmt[style])


def as_float_bool(obj: dict):
    """Checks the value of each key of the dict passed
    Args:
        * obj (dict): The object to decode

    Returns:
        * dict: The new dictionary with changes if necessary
    """
    for keys in obj.keys():
        try:
            obj[keys] = float(obj[keys])
            if obj[keys] == np.round(obj[keys]):
                obj[keys] = int(obj[keys])
        except:
            pass

        if obj[keys] == "True":
            obj[keys] = True
        elif obj[keys] == "False":
            obj[keys] = False

    return obj


def to_txt(file_name: str, data: pd.DataFrame, format: str = "%9.3f"):
    """Save data to a txt

    Args:
        * file_name (str): path where save the file
        * data (pd.DataFrame): raw data
        * format: save format
    """
    np.savetxt(str(file_name), data, delimiter="", fmt=format)
    return


def to_netcdf(data: pd.DataFrame, path: str):
    """Save data to netcdf file

    Args:
        data (pd.DataFrame): raw timeseries
        path (str): path where save the file
    """
    data.to_netcdf(str(path) + ".nc")
    return
