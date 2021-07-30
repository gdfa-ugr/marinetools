import json

import numpy as np
import pandas as pd


def npy2json(params):
    """[summary]

    Args:
        params ([type]): [description]
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


def to_json(params, fname, npArraySerialization=False):
    """Saves to a json file

    Args:
        - params (dict): data to be saved
        - fname (string): path of the file

    Return:
        - None
    """

    with open(f"{str(fname)}.json", "w") as f:
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


def to_csv(data, fname, compression="infer"):
    """Saves to a csv file

    Args:
        - data (pd.DataFrame): data to be saved
        - fname (str): path of the file
        - compression (str, opt): define the

    Return:
        - None
    """
    if not isinstance(data, pd.DataFrame):
        data.to_frame().to_csv(str(fname) + ".csv", compression=compression)

    if ".zip" in str(fname):
        data.to_csv(str(fname) + ".csv", compression="zip")
    else:
        data.to_csv(str(fname) + ".csv")
    return


def to_npy(data, fname):
    """Saves to a numpy file

    Args:
        - params (dict): data to be saved
        - fname (string): path of the file

    Return:
        - None
    """
    np.save(f"{str(fname)}.npy", data)
    return


def to_xlsx(data, fname):
    """Saves to an excel file

    Args:
        - params (dict): data to be saved
        - fname (string): path of the file

    Return:
        - None
    """

    wbook, wsheet = cwriter(str(fname) + ".xlsx")
    wsheet = wconfig(wsheet, data)

    # Writting the header
    wsheet.write(0, 0, data.index.name, formats(wbook, "header"))
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


def cwriter(fout):
    """Create a new file with the book and sheet for excel

    Args:
        - fout (string): path of the file

    Returns:
        - wbook (objects): excel book
        - wsheet (objects): excel sheet
    """
    writer = pd.ExcelWriter(fout, engine="xlsxwriter")
    df = pd.DataFrame([0])
    df.to_excel(writer, sheet_name="Sheet1", startrow=1, header=False)
    wsheet = writer.sheets["Sheet1"]
    wbook = writer.book
    return wbook, wsheet


def wconfig(wsheet, data):
    """Configures the cell size and protects it

    Args:
        - wsheet (object): excel sheet
        - data (pd.DataFrame): data

    Returns:
        - wsheet (object): excel sheet
    """
    size = np.max([12, len(data.index.name) * 2])
    wsheet.set_column(cols(0), size)

    for i, j in enumerate(data.columns):
        col = cols(i + 1)
        size = np.max([12, len(j) * 2])
        wsheet.set_column(col, size)

    return wsheet


def cols(col):
    """Associates the name of a variable for a given column

    Args:
        - col (int): number of the column

    Returns:
        - out (string): name of the column
    """
    out = {
        0: "A:A",
        1: "B:B",
        2: "C:C",
        3: "D:D",
        4: "E:E",
        5: "F:F",
        6: "G:G",
        7: "H:H",
        8: "I:I",
        9: "J:J",
        10: "K:K",
        11: "L:L",
        12: "M:M",
        13: "N:N",
        14: "O:O",
        15: "P:P",
        16: "Q:Q",
        17: "R:R",
        18: "S:S",
        19: "T:T",
        20: "U:U",
        21: "V:V",
        22: "W:W",
        23: "X:X",
        24: "Y:Y",
        25: "Z:Z",
    }
    return out[col]


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


def as_float_bool(obj):
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


def to_txt(fname, data, format="%9.3f"):
    """[summary]

    Args:
        fname ([type]): [description]
        data ([type]): [description]
        format:
    """
    np.savetxt(str(fname), data, delimiter="", fmt=format)
    return


def to_netcdf(data, path):
    """[summary]

    Args:
        data ([type]): [description]
        path ([type]): [description]
    """
    data.to_netcdf(str(path) + ".nc")
    return
