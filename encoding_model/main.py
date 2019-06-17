"""
Run a encoding model that predict neuronal activity from behavioral predictors.
"""
from typing import Tuple, Optional, List
from collections import namedtuple
from pkg_resources import Requirement, resource_stream
import numpy as np
from scipy.signal import convolve
from statsmodels import api as sm
from .utils import split_folds

GLM_FAMILY = sm.families.Gaussian(sm.families.links.identity())

def load() -> np.ndarray:
    fpb = resource_stream(Requirement.parse("encoding_model"), "encoding_model/data/spline_basis30_int.npy")
    return np.load(fpb)

Grouping = namedtuple('Grouping', 'event trial temporal')

def build_predictor(event_preds: np.ndarray, trial_preds: np.ndarray, spline: np.ndarray,
                    grouping: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build a predictor matrix. preds = predictors, pred = predictor
    Args:
        event_preds: 3D bool array of event predictors, n x k x t for n predictors, k trials and t time points
        trial_preds: 2D float array of trial predictors, n x k for n predictors and k trials
        grouping: Tuple[event, trial], each of which a 1D int array for grouping, where different numbers
            designate different groups
    Returns:
        predictor_mat: n x k x m array with n trials, m observations (timepoints) and k features,
            with features derived from spline_basis * event_features + trial_wise features
        grouping: 1D int array same size as predictor_mat.shape[1], used to exclude features for contribution
            calculation
    """
    if grouping is None:
        grouping = (np.arange(len(event_preds)), np.arange(len(trial_preds)) + len(event_preds))
    spline = load().T
    trial_no, timepoints = event_preds.shape[0], event_preds.shape[2]
    event_mats = [[[convolve(x, base) for base in spline] for x in trial] for trial in event_preds]
    event_mats = np.array(event_mats).reshape(trial_no, -1, timepoints)
    trial_mats = np.tile(trial_preds[:, :, np.newaxis], timepoints)
    flat_grouping = np.hstack([np.repeat(grouping[0]), grouping[1]])
    return np.concatenate([event_mats, trial_mats], axis=1), flat_grouping

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
        model = sm.GLM(y[idx_tr], sm.add_constant(X[idx_tr, :]), family=GLM_FAMILY)
        model.fit()
        y_hat[idx_te] = model.predict(sm.add_constant(X[idx_te, :]))
    return y_hat

def scale(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1)) / x.std(-1)

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
        y_flat = neuron.ravel()
        SS_full, df_full = aov(X, y_flat)
        res_neuron = list()
        for group_id in np.sorted(np.unique(grouping)):
            SS_nested, df_nested = aov(X[:, grouping != group_id], y_flat)
            res_neuron.append(((SS_nested - SS_full) / SS_full / ((df_nested - df_full) / df_full),
                              df_nested - df_full, df_full))
        res_mat.append(res_neuron)
    return res_mat

def get_r2_mat_refit(X: np.ndarray, y: np.ndarray, grouping: np.ndarray,
                     folds: int = 5) -> Tuple[List[List[float]], List[float]]:
    """Calculate r^2 for each neuron and after dropping each of the feature groups.
    Args:
        X: event, trial and temporal predictors, t * n with t observations and n features
        y: activity of neurons, n * t with n neurons and t observations
        grouping: flattened grouping for all predictors
        folds: number of validation folds
    Returns:
        2D array of float for r^2, n * k with n neuron and k feature groups, feature group id follow that in grouping
    """
    r2_dropped = [[np.corr(validated_glm_model(X[:, grouping != group_id], neuron, folds), neuron)
                   for group_id in np.unqiue(grouping)] for neuron in y]
    r2_full = [np.corr(validated_glm_model(X, neuron), neuron, folds) for neuron in y]
    return r2_dropped, r2_full

def validate_glm_with_drop(X: np.ndarray, y: np.ndarray, grouping: np.ndarray,
                           folds: int = 5) -> List[List[np.ndarray]]:
    """Calculate the contribution of feature groups by using same model and predictor set to zero.
    Args:
        X: t * n predictor with t observations and n flattened features
        y: values to be predicted, n * t with n entities and t observations
    """
    preds = [X[:, group_id == grouping].copy() for group_id in np.sorted(np.unique(grouping))]
    y_hat = list()
    for neuron in y:
        y_hat_neuron = [np.zeros(neuron.shape[0] // folds * folds, dtype=y.dtype) for _ in range(len(preds))]
        for idx_tr, idx_te in split_folds(neuron.shape[0], folds):
            y_tr = y[idx_tr]
            for idx, pred in enumerate(preds):
                y_hat_neuron[idx][idx_te] = sm.GLM(y_tr, sm.add_constant(pred[idx_tr, :]), family=GLM_FAMILY).fit()\
                    .predict(sm.add_constant(pred[idx_te, :]))
        y_hat.append(y_hat_neuron)
    return y_hat

def get_r2_mat_norefit(X: np.ndarray, y: np.ndarray, grouping: np.ndarray,
                       folds: int = 5) -> Tuple[List[List[float]], List[float]]:
    """Calculate r^2 for each neuron and after dropping each of the feature groups.
    Args:
        X: event, trial and temporal predictors, t * n with t observations and n features
        y: activity of neurons, n * t with n neurons and t observations
        grouping: flattened grouping for all predictors
    Returns:
        2D array of float for r^2, n * k with n neuron and k feature groups, feature group id follow that in grouping
    """
    y_hat_dropped = validate_glm_with_drop(X, y, grouping, folds=folds)
    r2_dropped = [[np.corr(y_hat_n_f, neuron) for y_hat_n_f in y_hat_n] for y_hat_n, neuron in zip(y_hat_dropped, y)]
    r2_full = [np.corr(validated_glm_model(X, neuron, folds=folds), neuron) for neuron in y]
    return r2_dropped, r2_full

def optimal_poly_order(X: np.ndarray, X_temporal: np.ndarray, y: np.ndarray, max_poly: int = 3) -> np.ndarray:
    """Find the polynomial order <= {max_poly} that give most r^2.
    Args:
        X: event and trial predictors, t * n with t observations and n flattened features
        X_temporal: temporal predictors, same, t * n, polynomial only works on them
        y: neuron activity to be predicted. n * t with n neurons and t observations
        max_poly: the maximal order of polynomial
    Returns:
        poly_order, 1D array the size of X_temporal.shape[1]
    """
    sequences = (np.arange(max_poly) for _ in range(X_temporal.shape[0]))
    idx_combination = np.vstack([x.ravel() for x in np.meshgrid(*sequences)]).T
    r_sq = list()
    for idx in idx_combination:
        X_one = np.hstack([X, raise_poly(X_temporal, idx)])
        r_sq.append([np.corr(validated_glm_model(X_one, neuron), neuron) for neuron in y])
    return idx_combination[np.mean(r_sq, axis=1).argmax(), :]

def run(X: np.ndarray, X_temporal: np.ndarray, y: np.ndarray, grouping: Tuple[np.ndarray, np.ndarray],
        refit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X: combined event and trial predictors, n * k * t with n trials,
            k flattened features and t timepoints.
        X_temporal: temporal predictors, n * k * t with n trials, k features and t timepoints.
        y: neuron activity, n * k * t with n trials, k neurons, and t timepoints
        grouping: [event/trial grouping, temporal group], using the same sequence of labels
        refit: whether refit for R2 calculation of nested models
    Returns:
        contribution: contribution of feature to neurons, in a 2D matrix {feature x neuron}
        F: F statistics for feature to neurons, in a 2D matrix {feature x neuron}
    """
    X, X_temporal = scale(X.reshape(X.shape[0], -1)).T, scale(X_temporal.reshape(X_temporal.shape[0], -1)).T
    y = scale(y.reshape(y.shape[0], -1))
    poly_order = optimal_poly_order(X, X_temporal, y, max_poly=3)
    X_full = np.hsatck([X, raise_poly(X_temporal, poly_order)])
    grouping_full = np.hstack([grouping[0], np.repeat(grouping[1], poly_order)])
    f_mat = get_f_mat(X_full, y, grouping_full)
    if refit:
        r2 = get_r2_mat_refit(X_full, y, grouping_full)
    else:
        r2 = get_r2_mat_norefit(X_full, y, grouping_full)
    return r2, f_mat

def encoding(event_preds, trial_preds, temporal_preds, spike_trial, grouping) -> Tuple[np.ndarray, np.ndarray]:
    X, flat_grouping = build_predictor(event_preds, trial_preds, grouping[0])
    r2, f_mat = run(X, temporal_preds, spike_trial, (flat_grouping, grouping[1]), refit=True)
    return r2, f_mat
