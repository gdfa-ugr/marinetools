import numpy as np
from marinetools.utils import auxiliar
from scipy.interpolate import Rbf, griddata
import pandas as pd


def class_storm_seasons(df_vars_ciclos, type_: str = "WSSF"):
    """Splits the data into seasons

    Args:
        * df_vars_ciclos (pd.DataFrame): events information

    Returns:
        * df_vars_ciclos (pd.DataFrame): events information with a new column for the season
    """

    df_vars_ciclos["season"] = None

    if type_ == "WSSF":
        # four groups of three months of winter, spring, summer and fall
        df_vars_ciclos_inv = (
            ((df_vars_ciclos.index.month == 12) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 1)
            | (df_vars_ciclos.index.month == 2)
            | ((df_vars_ciclos.index.month == 3) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos_prim = (
            ((df_vars_ciclos.index.month == 3) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 4)
            | (df_vars_ciclos.index.month == 5)
            | ((df_vars_ciclos.index.month == 6) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos_ver = (
            ((df_vars_ciclos.index.month == 6) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 7)
            | (df_vars_ciclos.index.month == 8)
            | ((df_vars_ciclos.index.month == 9) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos_oto = (
            ((df_vars_ciclos.index.month == 9) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 10)
            | (df_vars_ciclos.index.month == 11)
            | ((df_vars_ciclos.index.month == 12) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos.loc[df_vars_ciclos_inv, "season"] = "winter"
        df_vars_ciclos.loc[df_vars_ciclos_prim, "season"] = "spring"
        df_vars_ciclos.loc[df_vars_ciclos_ver, "season"] = "summer"
        df_vars_ciclos.loc[df_vars_ciclos_oto, "season"] = "fall"
    elif type_ == "WS":
        # two groups of six months of winter-spring and summer-fall
        df_vars_ciclos_WS = (
            ((df_vars_ciclos.index.month == 12) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 1)
            | (df_vars_ciclos.index.month == 2)
            | (df_vars_ciclos.index.month == 3)
            | (df_vars_ciclos.index.month == 4)
            | (df_vars_ciclos.index.month == 5)
            | ((df_vars_ciclos.index.month == 6) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos_SF = (
            ((df_vars_ciclos.index.month == 6) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 7)
            | (df_vars_ciclos.index.month == 8)
            | (df_vars_ciclos.index.month == 9)
            | (df_vars_ciclos.index.month == 10)
            | (df_vars_ciclos.index.month == 11)
            | ((df_vars_ciclos.index.month == 12) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos.loc[df_vars_ciclos_WS, "season"] = "WS"
        df_vars_ciclos.loc[df_vars_ciclos_SF, "season"] = "SF"

    elif type_ == "SF":
        # two groups of six months of spring-summer and fall-winter
        df_vars_ciclos_SS = (
            ((df_vars_ciclos.index.month == 3) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 4)
            | (df_vars_ciclos.index.month == 5)
            | (df_vars_ciclos.index.month == 6)
            | (df_vars_ciclos.index.month == 7)
            | (df_vars_ciclos.index.month == 8)
            | ((df_vars_ciclos.index.month == 9) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos_FW = (
            ((df_vars_ciclos.index.month == 9) & (df_vars_ciclos.index.day >= 21))
            | (df_vars_ciclos.index.month == 10)
            | (df_vars_ciclos.index.month == 11)
            | (df_vars_ciclos.index.month == 12)
            | (df_vars_ciclos.index.month == 1)
            | (df_vars_ciclos.index.month == 2)
            | ((df_vars_ciclos.index.month == 3) & (df_vars_ciclos.index.day < 21))
        )

        df_vars_ciclos.loc[df_vars_ciclos_SS, "season"] = "SS"
        df_vars_ciclos.loc[df_vars_ciclos_FW, "season"] = "FW"

    return df_vars_ciclos


def classification(cases, cases_sha, data, method, notrain):
    """[summary]

    Args:
        cases ([type]): [description]
        cases_sha ([type]): [description]
        data ([type]): [description]
        method ([type]): [description]
        notrain ([type]): [description]

    Returns:
        [type]: [description]
    """

    from sklearn import preprocessing
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=0.025),
        "RBF SVM": SVC(gamma=2, C=1),
        "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1
        ),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }

    clf = classifiers[method]
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(cases_sha.iloc[:notrain, 0].values)
    clf.fit(cases.iloc[:notrain, :].values, encoded)
    # score = clf.score(cases.iloc[notrain:, :].values, cases_sha.iloc[notrain:, 'Hs'].values)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(data)
    else:
        Z = clf.predict_proba(data)[:, 1]

    return Z


def reconstruction(
    cases_deep,
    data_deep,
    cases_shallow,
    index,
    base_vars,
    recons_vars,
    method="rbf-multiquadric",
    smooth=0.5,
    optimize=True,
    num=450,
):
    """[summary]

    Args:
        cases_deep ([type]): [description]
        data_deep ([type]): [description]
        cases_shallow ([type]): [description]
        base_vars ([type]): [description]
        recons_vars ([type]): [description]
        method (str, optional): [description]. Defaults to 'rbf-multiquadric'.
        smooth (float, optional): [description]. Defaults to 0.5.
        optimize (bool, optional): [description]. Defaults to True.
        param (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """

    cases_deepv = cases_deep[base_vars]
    data_deepv = data_deep[base_vars]
    cases_shallowv = cases_shallow[base_vars]

    _, scaler = auxiliar.scaler(cases_shallowv)

    cases_deep_normalize, _ = auxiliar.scaler(cases_deepv, scale=scaler)
    data_deep_normalize, _ = auxiliar.scaler(data_deepv, scale=scaler)
    cases_shallow_normalize, _ = auxiliar.scaler(cases_shallowv, scale=scaler)

    data_reconstructed_norm = pd.DataFrame(-1, index=index, columns=recons_vars)
    data_reconstructed = pd.DataFrame(-1, index=index, columns=recons_vars)

    scalers = {}
    for variable in recons_vars:
        cases_shallow_normalize[variable], scalers[variable] = auxiliar.scaler(
            cases_shallow[[variable]]
        )
        data_reconstructed_norm[variable] = regression(
            cases_deep_normalize[base_vars],
            cases_shallow_normalize[variable],
            data_deep_normalize[base_vars],
            method=method,
            num=num,
            smooth=smooth,
            optimize=optimize,
        )
        data_reconstructed[variable], _ = auxiliar.scaler(
            data_reconstructed_norm[[variable]],
            transform=False,
            scale=scalers[variable],
        )

    return data_reconstructed


def regression(
    X, Y, x, method="rbf-multiquadric", num=450, smooth=0.5, optimize=True, param=1
):
    """[summary]

    Args:
        X (np.ndarray): vector or matrix with raw data (x)
        Y (np.ndarray): vector or matrix with raw data (y)
        x (np.ndarray): data to be reconstructed
        method (str, optional): method or kernel to the regression. Defaults to 'rbf-multiquadric'.

    Raises:
        ValueError: method not implemented

    Returns:
        [type]: [description]
    """

    from sklearn.gaussian_process import (
        GaussianProcessClassifier,
        GaussianProcessRegressor,
    )
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        ExpSineSquared,
        RationalQuadratic,
        WhiteKernel,
    )

    algorithms = [
        "linear",
        "nearest",
        "cubic",
        "rbf-multiquadric",
        "rbf-inverse",
        "rbf-gaussian",
        "rbf-linear",
        "rbf-cubic",
        "rbf-quintic",
        "rbf-thin_plate",
        "gp-rbf",
        "gp-exponential",
        "gp-quadratic",
        "gp-white",
    ]

    kernels = {
        "gp-rbf": 1.0 * RBF(0.5),
        "gp-exponential": ExpSineSquared,
        "gp-quadratic": RationalQuadratic,
        "gp-white": WhiteKernel,
    }

    X, Y, x = X.values, Y.values, x.values

    if any(methods in method for methods in ["linear", "nearest", "cubic"]):
        data_interpolated = griddata(X, Y, x, method=method)
    elif method.startswith("rbf"):
        method = method.split("-")[1]

        if optimize:
            param = auxiliar.optimize_rbf_epsilon(
                X, Y, num, method=method, smooth=smooth
            )

        Z = np.hstack([X, Y.reshape(-1, 1)])
        rbf = Rbf(*Z.T, function=method, smooth=smooth, epsilon=param)
        data_interpolated = rbf(*x.T)
    elif method.startswith("gp"):
        kernel = kernels[method]
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=False
        )
        gp.fit(X, Y)
        data_interpolated = gp.predict(x, return_std=False)
    else:
        raise ValueError(
            "The method {} is not yet implemented. Methods available are {}.".format(
                method, algorithms
            )
        )

    return data_interpolated
