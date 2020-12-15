import numpy as np
from scipy.stats import f
from encoding_model.main import get_f_mat, get_r2_mat_refit, get_r2_mat_norefit, build_predictor
from encoding_model.main import run_encoding, raise_poly
from encoding_model.b_spline import bspline_set

def build():
    a, b, c = np.arange(20) / 20, np.arange(20) / 20, np.arange(20) / 20
    aa, bb, cc = np.meshgrid(a, b, c)
    y0 = (np.random.randn(20, 20, 20) + aa * 10 + bb * 3 + cc * 1).ravel()
    y1 = (np.random.randn(20, 20, 20) + aa * 5 + bb * 10 + cc * 2).ravel()
    X = np.vstack([aa.ravel(), bb.ravel(), cc.ravel()]).T
    y = np.vstack([y0, y1])
    return X, y

def test_f_test():
    X, y = build()
    res = get_f_mat(X, y, grouping=[0, 1, 2])
    df1 = [[x[1] for x in y] for y in res]
    df2 = [[x[2] for x in y] for y in res]
    assert np.allclose(df1, 1)
    assert np.allclose(df2, 7997)
    p_mat = [[f.sf(f_score, df1, df2) for (f_score, df1, df2) in row] for row in res]
    assert np.allclose(p_mat, 0)

def test_r2_test():
    x0 = np.random.randn(1000)
    x1 = x0 * 0.25 + np.random.randn(1000)
    x2 = np.random.randn(1000)
    X = np.vstack([x0, x1, x2]).T
    y = np.random.randn(2, 1000) + np.array([[10, 5, 1], [5, 10, 1]]) @ X.T
    r2_mat, r2_full = get_r2_mat_refit(X, y, [0, 1, 2])
    assert np.all(np.abs(np.subtract(r2_mat, [[0.645, 0.915, 0.993], [0.925, 0.623, 0.994]])) < 0.01)
    assert np.all(np.abs(np.subtract(r2_full, [0.997, 0.997])) < 0.01)
    r2_mat, r2_full = get_r2_mat_norefit(X, y, [0, 1, 2])
    assert np.all(np.abs(np.subtract(r2_mat, [[0.646, 0.916, 0.993], [0.925, 0.624, 0.994]])) < 0.01)
    assert np.all(np.abs(np.subtract(r2_full, [0.997, 0.997])) < 0.01)

def test_build_predictor():
    event = np.zeros((1, 5, 25), dtype=np.bool)
    event[0, np.arange(5), np.random.randint(5, 20, 5)] = True
    trial = np.random.randn(1, 5)
    temporal = np.random.randn(1, 5, 25) + 1
    temporal /= temporal.sum(2, keepdims=True)
    grouping = (np.array([1]), np.array([2]))
    grouping_temp = np.array([3])
    X0, group0 = build_predictor(event, trial, bspline_set(np.arange(9), 2), grouping)
    coef = np.array([[2, 2, 2, 2, 2, 3],
                     [1, 1, 1, 1, 1, 1]])
    coef_t = np.array([[1, 1], [3, 3]]).T
    y = (X0 * (coef)[:, :, None, None]).sum(axis=1)
    y += ((raise_poly(temporal.reshape(temporal.shape[0], -1).T, (2, )) @ coef_t).T).reshape(2, 5, 25)
    y += 0.5 * np.random.randn(2, 5, 25)
    r2, f_mat_2 = run_encoding(X0, temporal, y, (group0, grouping_temp), refit=True)
    assert np.allclose(r2, np.array([[0.0539, 0.825, -0.000769], [0.07, 0.74, 0.013]]), atol=0.01)
