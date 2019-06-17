import numpy as np
from encoding_model.main import f_test

def test_f_test():
    a, b, c = np.arange(20) / 20, np.arange(20) / 20, np.arange(20) / 20
    aa, bb, cc = np.meshgrid(a, b, c)
    y = (np.random.randn(20, 20, 20) + aa * 10 + bb * 3 + cc * 1).ravel()
    X = np.vstack([aa.ravel(), bb.ravel(), cc.ravel()])
    f_score, df1, df2 = f_test(X, X[:, (0, 1)], y)
    from scipy.stats import f
    p = f.sf(f_score, df1, df2)
    assert 1E-20 > p > 0
