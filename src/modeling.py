"""Training-only transforms and inspectable frozen regression inference."""
from pathlib import Path
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS = ["temp", "atemp", "hum", "windspeed"]
GROUPS = {"hour": ["hr"], "weekday": ["weekday"], "month": ["mnth"],
          "workingday": ["workingday"], "holiday": ["holiday"],
          "weather": ["weathersit"], "temp": ["temp"], "atemp": ["atemp"],
          "humidity": ["hum"], "windspeed": ["windspeed"],
          "joint_temperature": ["temp", "atemp"]}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source(year):
    data = pd.read_csv(ROOT / "data/hour.csv.gz")
    # The fitting stage only receives 2011. Test inference is a separate command.
    return data[data.yr == year - 2011].reset_index(drop=True)


def calendar(data):
    columns = []
    for harmonic in (1, 2):
        angle = data.hr.to_numpy() * (2 * np.pi * harmonic / 24)
        columns += [np.round(np.sin(angle), 12), np.round(np.cos(angle), 12)]
    angle = data.mnth.to_numpy() * (2 * np.pi / 12)
    columns += [np.round(np.sin(angle), 12), np.round(np.cos(angle), 12)]
    columns += [(data.weekday.to_numpy() == code).astype(float) for code in range(7)]
    columns += [(data.weathersit.to_numpy() == code).astype(float) for code in range(1, 5)]
    columns += [data.workingday.to_numpy(), data.holiday.to_numpy()]
    return np.column_stack(columns).astype(float)


def spline_values(values, spec):
    basis = BSpline(np.array(spec["t"]), np.array(spec["c"]), spec["k"], extrapolate=False)
    lo, hi = basis.t[basis.k], basis.t[-basis.k - 1]
    bounded = np.clip(values, lo, hi)
    result = basis(bounded)
    for outside, edge in ((values < lo, lo), (values > hi, hi)):
        result[outside] = basis(edge) + (values[outside] - edge)[:, None] * basis(edge, nu=1)
    return result[:, :-1]


def design(data, transform):
    columns = [calendar(data)]
    for name in CONTINUOUS:
        values = data[name].to_numpy(float)
        if transform["family"] == "additive":
            columns.append(spline_values(values, transform["splines"][name]))
        else:
            columns.append(values[:, None])
    result = np.column_stack(columns)
    if "mean" in transform:
        result = (result - transform["mean"]) / transform["scale"]
    return result


def fit_transform(data, family):
    spec = {"family": family, "calendar_round_decimals": 12, "splines": {}}
    if family == "additive":
        for name in CONTINUOUS:
            values = data[[name]].to_numpy(float)
            fit = SplineTransformer(n_knots=5, degree=3, include_bias=False,
                                    knots="quantile", extrapolation="linear").fit(values)
            spline = fit.bsplines_[0]
            spec["splines"][name] = {"t": spline.t.tolist(), "c": spline.c.tolist(), "k": spline.k}
            np.testing.assert_allclose(spline_values(values[:, 0], spec["splines"][name]),
                                       fit.transform(values), atol=1e-12, rtol=1e-12)
    matrix = design(data, spec)
    if family in ("linear", "additive"):
        spec["mean"] = matrix.mean(axis=0).tolist()
        sd = matrix.std(axis=0)
        spec["scale"] = np.where(sd == 0, 1, sd).tolist()
        matrix = design(data, spec)
    return spec, matrix


def candidates():
    result = []
    for family in ("linear", "additive"):
        for strength in (0.0001, 0.001, 0.01, 0.1, 1):
            result.append({"id": f"{family}_lambda{strength:g}", "family": family, "lambda": strength})
    for depth in (2, 4, 6):
        for leaf in (20, 60):
            result.append({"id": f"cart_depth{depth}_leaf{leaf}", "family": "cart", "depth": depth, "leaf": leaf})
    for leaves in (7, 15):
        for rate in (0.05, 0.1):
            for penalty in (0, 1):
                result.append({"id": f"boost_leaves{leaves}_lr{rate:g}_l2{penalty}",
                               "family": "boost", "leaves": leaves, "rate": rate, "l2": penalty})
    return result


def tree_predict(matrix, tree, single_precision=False):
    x = matrix.astype(np.float32) if single_precision else matrix
    positions = np.zeros(len(x), dtype=int)
    leaf = np.array(tree["leaf"], dtype=bool)
    features = np.array(tree["feature"], dtype=int)
    thresholds = np.array(tree["threshold"])
    left, right = np.array(tree["left"]), np.array(tree["right"])
    rows = np.arange(len(x))
    while True:
        active = ~leaf[positions]
        if not active.any():
            break
        ids = rows[active]
        nodes = positions[active]
        positions[active] = np.where(x[ids, features[nodes]] <= thresholds[nodes], left[nodes], right[nodes])
    return np.array(tree["value"])[positions]


def predict(data, model):
    matrix = design(data, model["transform"])
    family = model["candidate"]["family"]
    if family in ("linear", "additive"):
        return matrix @ np.array(model["coefficient"]) + model["intercept"]
    result = np.full(len(data), model["baseline"], dtype=float)
    for tree in model["trees"]:
        result += tree_predict(matrix, tree, family == "cart")
    return result


def fit(data, candidate):
    assert (data.yr == 0).all(), "No held-year rows may enter fitting"
    started = time.perf_counter()
    family = candidate["family"]
    transform, matrix = fit_transform(data, family)
    target = data.cnt.to_numpy(float)
    if family in ("linear", "additive"):
        estimator = Ridge(alpha=len(data) * candidate["lambda"], solver="svd")
    elif family == "cart":
        estimator = DecisionTreeRegressor(max_depth=candidate["depth"], min_samples_leaf=candidate["leaf"],
                                          criterion="squared_error", random_state=20261028)
    else:
        estimator = HistGradientBoostingRegressor(max_leaf_nodes=candidate["leaves"], learning_rate=candidate["rate"],
            l2_regularization=candidate["l2"], max_iter=120, min_samples_leaf=30, max_bins=64,
            early_stopping=False, random_state=20261028, loss="squared_error")
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(matrix, target)
        reference = estimator.predict(matrix)
    model = {"candidate": candidate, "transform": transform,
             "training_rows": len(data), "training_last_timestamp": data.timestamp.max(),
             "training_row_identity_sha256": hashlib.sha256("\n".join(data.timestamp).encode()).hexdigest(),
             "warnings": [str(w.message) for w in caught]}
    if family in ("linear", "additive"):
        model.update(coefficient=estimator.coef_.tolist(), intercept=float(estimator.intercept_))
        residual = reference - target
        gradient = 2 * (matrix.T @ residual / len(data) + candidate["lambda"] * estimator.coef_)
        model["objective"] = float(np.mean(residual ** 2) + candidate["lambda"] * np.sum(estimator.coef_ ** 2))
        model["max_gradient"] = float(max(abs(gradient).max(), abs(2 * residual.mean())))
        assert model["max_gradient"] <= 1e-9
    elif family == "cart":
        tree = estimator.tree_
        model.update(baseline=0.0, actual_nodes=int(tree.node_count), actual_depth=estimator.get_depth(),
            trees=[{"leaf": (tree.children_left == -1).tolist(), "feature": tree.feature.tolist(),
                    "threshold": tree.threshold.tolist(), "left": tree.children_left.tolist(),
                    "right": tree.children_right.tolist(), "value": tree.value[:, 0, 0].tolist()}])
    else:
        trees = []
        for iteration in estimator._predictors:
            nodes = iteration[0].nodes
            assert not nodes["is_categorical"].any()
            trees.append({"leaf": nodes["is_leaf"].tolist(), "feature": nodes["feature_idx"].tolist(),
                          "threshold": nodes["num_threshold"].tolist(), "left": nodes["left"].tolist(),
                          "right": nodes["right"].tolist(), "value": nodes["value"].tolist()})
        model.update(baseline=float(estimator._baseline_prediction[0, 0]), trees=trees,
                     actual_iterations=int(estimator.n_iter_))
    model["fit_seconds"] = time.perf_counter() - started
    reconstructed = predict(data, model)
    np.testing.assert_allclose(reconstructed, reference, atol=1e-9, rtol=1e-12)
    model["training_inference_max_difference"] = float(np.max(abs(reconstructed - reference)))
    return model
