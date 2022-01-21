import matplotlib.pyplot as plt


def show(file_name: str = None, res: int = 600):
    """Saves into a file or displays on the screen the result of a plot

    Args:
        * fname (None or string): name of the file to save the plot or None to see plots on the screen. Defaults to None.
        * res (int): resolution in dpi of the saved figure

    Returns:
        * Displays on the screen the figure or saves it into a file
    """

    if not file_name:
        plt.show()
    elif file_name == ("to_axes"):
        pass
    else:
        plt.savefig(f"{file_name}.png", dpi=res, bbox_inches="tight")
        plt.close()
    return


def handle_axis(
    ax,
    row_plots: int = 1,
    col_plots: int = 1,
    dim: int = 2,
    figsize: tuple = (5, 5),
    kwargs: dict = {},
):
    """Creates the matplotlib.axis of the figure if it is required or nothing if it is given

    Args:
        * ax (matplotlib.axis): axis for the plot.
        * row_plots (int, optional): no. of row axis. Defaults to 1.
        * col_plots (int, optional): no. of row axis. Defaults to 1.
        * dim (int, optional): no. of dimensions. Defaults to 2.
        * figsize: as matplotlib
        * kwargs: as matplotlib

    Returns:
        * Same as the arguments
    """

    if not ax:
        if dim == 2:
            _, ax = plt.subplots(row_plots, col_plots, figsize=figsize, **kwargs)
        else:
            fig = plt.figure()
            ax = fig.gca(projection="3d")

    if row_plots + col_plots > 2:
        ax = ax.flatten()

    return ax


def labels(variable):
    """Gives the labels and units for plots

    Args:
        * variable (string): name of the variable

    Returns:
        * units (string): label with the name of the variable and the units
    """

    units = {
        "Hm0": r"$H_{m0}$ (m)",
        "Hs": r"$H_{s}$ (m)",
        "Tm0": r"$T_{m0}$ (s)",
        "Tp": r"$T_p$ (s)",
        "Dmd": r"$\theta_m$ (deg)",
        "DirM": r"$\theta_m$ (deg)",
        "VelV": "[m/s]",
        "Vv": r"$V_v$ (m/s)",
        "Wv": r"$W_v$ (m/s)",
        "Dmv": r"$\theta_v$ (deg)",
        "DirV": r"$\theta_v$ (deg)",
        "Wd": r"$W_d$ (deg)",
        "pr": r"P (mm/day)",
        "storm": r"d (h)",
        "calms": r"$\delta$ (h)",
        "hs": r"$H_{s}$ (m)",
        "tp": r"$T_p$ (s)",
        "dm": r"$\theta_m$ (deg)",
        "dv": r"$\theta_v$ (deg)",
        "vv": r"$V_v$ (m/s)",
        "mm": r"$M_{met}$ (m)",
        "slr": r"$\Delta\eta$ (m)",
        "eta": r"$\eta$ (m)",
        "ma": r"$M_{ast}$ (m)",
        "Qd": r"$Q_d$ (m$^3$/s)",
        "Q": r"Q (m$^3$/s)",
        "dur_storm": r"$d_0$ (hr)",
        "dur_calms": r"$\Delta_0$ (hr)",
        "dur": r"d (hr)",
        "U": r"U (m/s)",
        "V": r"V (m/s)",
        "DirU": r"$\theta_U$ (deg)",
        "x": "x (m)",
        "y": "y (m)",
        "z": "z (m)",
        "t": "t (s)",
        "lat": "Latitud (deg)",
        "lon": "Longitude (deg)",
        "depth": "Depth (m)",
        "None": "None",
    }

    if isinstance(variable, str):
        if not variable in units.keys():
            labels_ = ""
        else:
            labels_ = units[variable]
    elif isinstance(variable, list):
        labels_ = list()
        for var_ in variable:
            if not var_ in units.keys():
                labels_.append("")
            else:
                labels_.append(units[var_])
    else:
        raise ValueError
    return labels_
