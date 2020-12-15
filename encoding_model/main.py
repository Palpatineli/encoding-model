"""
Run a encoding model that predict neuronal activity from behavioral predictors.
"""
from typing import Tuple, Optional, List, Sequence, TypeVar
from dataclasses import dataclass
from pkg_resources import Requirement, resource_stream
import numpy as np
from scipy.signal import convolve
from statsmodels import api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from .utils import split_folds

GLM_FAMILY = sm.families.Gaussian(sm.families.links.identity())
T = TypeVar("T", bound="IterData")

@dataclass
class IterData:
    def __iter__(self):
        return (v for k, v in self.__dict__.items() if not k.startswith("__"))

    def copy(self: T) -> T:
        return self.__class__(*iter(self))  # type: ignore

@dataclass
class Predictors(IterData):
    event: List[np.ndarray]
    period: List[np.ndarray]
    trial: List[np.ndarray]
    temporal: List[np.ndarray]

@dataclass
class FlatPredictors(IterData):
    event: np.ndarray
    period: np.ndarray
    trial: np.ndarray
    temporal: np.ndarray

@dataclass
class Grouping(IterData):
    event: List[int]
    period: List[int]
    trial: List[int]
    temporal: List[int]

def load_spline() -> np.ndarray:
    fpb = resource_stream(Requirement.parse("encoding_model"), "encoding_model/data/spline_basis30_int.npy")
    return np.load(fpb)

def convolve_event(event_preds: np.ndarray, grouping: Optional[np.ndarray] = None,
                   spline: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    grouping = np.arange(1, len(event_preds + 1)) if grouping is None else grouping
    spline = load_spline().T if spline is None else spline
    trial_no, trial_len = event_preds.shape[1], event_preds.shape[2]
    result = np.array([[[convolve(trial, basis)[:trial_len] for trial in feature] for basis in spline]
                       for feature in event_preds]).reshape(-1, trial_no, trial_len)
    flat_grouping = np.repeat(grouping, spline.shape[0])
    return result, flat_grouping

def predictor_helper(preds: Predictors, hit: np.ndarray) -> Predictors:
    """
    Args:
        preds: (event, period, trial, temporal), each a list of
            event: int if all trials should have that event at the index,
                   np.ndarray if all|hit trials have event at different indices if positive, depending on length
            period: 2 * k for start|end of k trials
            trial: trial specific parameteres, k for k trials
        hit: bool ndarray for hit trials
    """
    trial_no, trial_samples = hit.shape[0], preds.temporal[0].shape[1]
    feature_no = [len(x) for x in preds]
    output = Predictors(np.zeros((feature_no[0], trial_no, trial_samples), dtype=np.bool_),
                        np.zeros((feature_no[1], trial_no, trial_samples), dtype=np.bool_),
                        np.zeros((feature_no[2], trial_no, trial_samples), dtype=np.float),
                        preds.temporal)
    for row, feature in zip(output.event, preds.event):
        if not isinstance(feature, Sequence):
            row[:, feature] = True
        elif len(feature) == trial_no:
            mask = np.greater_equal(feature, 0)
            row[np.arange(trial_no)[mask], feature[mask]] = True
        else:
            mask = np.greater_equal(feature, 0)
            row[np.flatnonzero(hit)[mask], feature[mask]] = True
    for row, feature in zip(output.period, preds.period):
        if feature.ndim < 2 and feature.shape[1] == 1:
            row[:, feature[0]: feature[1]] = True
        else:
            for trial, (start, end) in zip(row, feature.T):
                if end > 0:
                    trial[start: end] = True
    for row, feature in zip(output.trial, preds.trial):
        if len(feature) == trial_no:
            row[:] = feature[:, np.newaxis]
        else:
            row[hit] = feature[:, np.newaxis]
    return output

def build_poly_predictor(preds: Predictors, y: np.ndarray, grouping: Grouping) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: k * n for k observations and n features
        grouping: n for n flattened features
    """
    flat_preds = FlatPredictors(*(scale(np.asarray(pred).reshape(len(pred), -1).T) for pred in preds))
    y = scale(y.reshape(y.shape[0], -1), 1)
    poly_order = optimal_poly_order(flat_preds, y, max_poly=3)
    flat_preds.temporal = raise_poly(flat_preds.temporal, poly_order)
    try:
        flat_grouping = grouping.copy()
        flat_grouping.temporal = np.repeat(grouping.temporal, poly_order)
    except ValueError as e:
        print("poly order: ", poly_order, " temporal: ", grouping.temporal, flat_grouping.temporal)
        raise e
    return np.hstack(list(flat_preds)), np.hstack(list(flat_grouping))

def validated_glm_model(X: np.ndarray, y: np.ndarray, folds: int = 5) -> np.ndarray:
    """
    Args:
        X: predictor matrix, t x k with k features and t observations (trial x timepoints)
        y: real value, 1D array of t size
        folds: how many folds to validate
    Returns:
        y_hat combined after {folds}-folds break down.
    """
    y_hat = np.zeros(y.shape[0] // folds * folds, dtype=y.dtype)
    for idx_tr, idx_te in split_folds(y.shape[0], folds):
        temp_X = sm.add_constant(X)
        model = sm.GLM(y[idx_tr], temp_X[idx_tr], family=GLM_FAMILY).fit()
        y_hat[idx_te] = model.predict(temp_X[idx_te])
    return y_hat

def scale(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalize features, assuming 2D array t * n with n features and t observations."""
    return (x - x.mean(axis, keepdims=True)) / x.std(axis, keepdims=True)

def raise_poly(X: np.ndarray, poly_array: np.ndarray) -> np.ndarray:
    """Raise each feature to {poly}, assuming that the predictor is 2D matrix observation * feature"""
    result = [X_col ** (exp + 1) for poly, X_col in zip(poly_array, X.T) for exp in range(poly)]
    return np.vstack(result).T

def aov(X, y) -> Tuple[float, int]:
    """from GLM calculate SS and df."""
    SS = ((sm.GLM(y, sm.add_constant(X), family=GLM_FAMILY).fit().predict() - y) ** 2).sum()
    df = X.shape[0] - X.shape[1]
    return SS, df

def get_f_mat(X: np.ndarray, y: np.ndarray, grouping: np.ndarray) -> List[List[Tuple[float, int, int]]]:
    """Get a neuron * feature matrix of F values
    Args:
        X: event, trial and temporal predictors, t * n with t observations and n features
        y: activity of neurons, n * t with n neurons and t observations
        grouping: grouping for all three types of predictors
    Returns:
        a 2D array of (F_values, df1, df2).
            n * m with n neurons and m features, the index of m follows the index in grouping.
    """
    res_mat = list()
    for neuron in y:
        SS_full, df_full = aov(X, neuron)
        res_neuron = list()
        for group_id in np.sort(np.unique(grouping)):
            SS_nested, df_nested = aov(X[:, grouping != group_id], neuron)
            res_neuron.append(((SS_nested - SS_full) / SS_full / ((df_nested - df_full) / df_full),
                               df_nested - df_full, df_full))
        res_mat.append(res_neuron)
    return res_mat

def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ex = x - x.mean()
    ey = y - y.mean()
    return (ex * ey).sum() / np.sqrt((ex * ex).sum() * (ey * ey).sum())

def get_r2_mat_refit(X: np.ndarray, y: np.ndarray, grouping: np.ndarray, folds: int = 5) -> np.ndarray:
    """Calculate r^2 for each neuron and after dropping each of the feature groups.
    Args:
        X: event, trial and temporal predictors, t * n with t observations and n features
        y: activity of neurons, n * t with n neurons and t observations
        grouping: flattened grouping for all predictors
        folds: number of validation folds
    Returns:
        2D array of float for r^2, n * k with n neuron and k feature groups, feature group id follow that in grouping
    """
    r2_dropped = [[_corr(validated_glm_model(X[:, grouping != group_id], neuron, folds), neuron)
                   for group_id in np.unique(grouping)] for neuron in y]
    r2_full = np.array([_corr(validated_glm_model(X, neuron, folds), neuron) for neuron in y])
    return np.hstack([r2_full[:, None] - np.array(r2_dropped), r2_full[:, None]])

def validate_glm_with_drop(X: np.ndarray, y: np.ndarray, grouping: np.ndarray,
                           folds: int = 5) -> List[List[np.ndarray]]:
    """Calculate the contribution of feature groups by using same model and predictor set to zero.
    Args:
        X: t * n predictor with t observations and n flattened features
        y: values to be predicted, n * t with n entities and t observations
    """
    y_hat = list()
    unique_groups = np.sort(np.unique(grouping))
    for neuron in y:
        y_hat_neuron = [np.zeros(neuron.shape[0] // folds * folds, dtype=y.dtype) for _ in range(len(unique_groups))]
        for idx_tr, idx_te in split_folds(neuron.shape[0], folds):
            y_tr = neuron[idx_tr]
            temp_X = sm.add_constant(X)
            model = sm.GLM(y_tr, temp_X[idx_tr, :], family=GLM_FAMILY).fit()
            for idx in range(len(unique_groups)):
                pred = temp_X[idx_te, :].copy()
                if pred.shape[1] > grouping.shape[0]:
                    pred[:, 1:][:, grouping == idx] = 0
                else:
                    pred[:, grouping == idx] = 0
                y_hat_neuron[idx][idx_te] = model.predict(pred)
        y_hat.append(y_hat_neuron)
    return y_hat

def get_r2_mat_norefit(X: np.ndarray, y: np.ndarray, grouping: np.ndarray, folds: int = 5) -> np.ndarray:
    """Calculate r^2 for each neuron and after dropping each of the feature groups.
    Args:
        X: event, trial and temporal predictors, t * n with t observations and n features
        y: activity of neurons, n * t with n neurons and t observations
        grouping: flattened grouping for all predictors
    Returns:
        2D array of float for r^2, n * k with n neuron and k feature groups, feature group id follow that in grouping
    """
    y_hat_dropped = validate_glm_with_drop(X, y, grouping, folds=folds)
    r2_dropped = [[_corr(y_hat_n_f, neuron) for y_hat_n_f in y_hat_n]
                  for y_hat_n, neuron in zip(y_hat_dropped, y)]
    r2_full = np.array([_corr(validated_glm_model(X, neuron, folds=folds), neuron) for neuron in y])
    return np.hstack([r2_full[:, None] - np.array(r2_dropped), r2_full[:, None]])

def optimal_poly_order(preds: FlatPredictors, y: np.ndarray, max_poly: int = 3) -> np.ndarray:
    """Find the polynomial order <= {max_poly} that give most r^2.
    Args:
        X: event and trial predictors, t * n with t observations and n flattened features
        X_temporal: temporal predictors, same, t * n, polynomial only works on them
        y: neuron activity to be predicted. n * t with n neurons and t observations
        max_poly: the maximal order of polynomial
    Returns:
        poly_order, 1D array the size of X_temporal.shape[1]
    """
    sequences = [np.arange(max_poly) for _ in range(preds.temporal.shape[1])]
    idx_combination = np.vstack([x.ravel() for x in np.meshgrid(*sequences)]).T
    r_sq = list()
    temp_preds = preds.copy()
    for idx in idx_combination:
        temp_preds.temporal = raise_poly(preds.temporal, idx + 1)
        X_one = np.hstack(list(preds))
        r_sq.append([_corr(validated_glm_model(X_one, neuron), neuron) for neuron in y])
    return idx_combination[np.mean(r_sq, axis=1).argmax(), :] + 1

def build_predictor(preds: Predictors, grouping: Grouping, hit: np.ndarray,
                    spline: np.ndarray) -> Tuple[Predictors, Grouping]:
    preds = predictor_helper(preds, hit)
    preds.event, grouping.event = convolve_event(preds.event, grouping.event, spline)
    return preds, grouping

def build_model(data: Tuple[Predictors, np.ndarray, Grouping]) -> List[GLMResults]:
    X, y, grouping = data
    flat_X, grouping = build_poly_predictor(X, y, grouping)
    flat_y = scale(y.reshape(y.shape[0], -1), 1)
    models = list()
    for neuron in flat_y:
        models.append(sm.GLM(neuron, sm.add_constant(flat_X), family=GLM_FAMILY).fit_regularized(alpha=0.3))
    return models

def model_r2(models: List[GLMResults], X: np.ndarray, y: np.ndarray, grouping: np.ndarray) -> np.ndarray:
    r2 = list()
    unique_groups = np.sort(np.unique(grouping))
    for neuron, model in zip(y, models):
        neuron_r2 = list()
        if model._results.params.shape[0] > X.shape[1]:
            pred = sm.add_constant(X.copy(), has_constant='add')
            for idx in range(len(unique_groups)):
                temp_pred = pred.copy()
                temp_pred[:, 1:][:, grouping == idx] = 0
                neuron_r2.append(_corr(neuron, model.predict(temp_pred)))
        else:
            for idx in range(len(unique_groups)):
                temp_pred = pred.copy()
                temp_pred[:, grouping == idx] = 0
                neuron_r2.append(_corr(neuron, model.predict(temp_pred)))
        r2.append(neuron_r2)
    return np.array(r2)

def run_encoding(data: Tuple[Predictors, np.ndarray, Grouping]) -> Tuple[np.ndarray, np.ndarray]:
    X, y, grouping = data
    X, grouping = build_poly_predictor(X, y, grouping)
    flat_y = scale(y.reshape(y.shape[0], -1), 1)
    f_mat = get_f_mat(X, flat_y, grouping)
    r2 = get_r2_mat_norefit(X, flat_y, grouping)
    return r2, f_mat
