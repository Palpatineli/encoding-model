import numpy as np
from scipy.stats import f
from encoding_model.main import get_f_mat, get_r2_mat_refit, get_r2_mat_norefit

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

# plot
def test_r2_test():
    x0 = np.random.randn(1000)
    x1 = x0 * 0.25 + np.random.randn(1000)
    x2 = np.random.randn(1000)
    X = np.vstack([x0, x1, x2]).T
    y = np.random.randn(2, 1000) + np.array([[10, 1, 1], [1, 10, 1]]) @ X.T
    r2_mat, r2_full = get_r2_mat_refit(X, y, [0, 1, 2])
    assert np.all(np.abs(np.subtract(r2_mat, [[0.285, 0.909, 0.945], [0.859, 0.453, 0.941]])) < 0.01)
    assert np.all(np.abs(np.subtract(r2_full, [0.949, 0.956])) < 0.01)
    r2_mat, r2_full = get_r2_mat_norefit(X, y, [0, 1, 2])
    np.corrcoef(x2, x1)[0, 1]
