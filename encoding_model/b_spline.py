import numpy as np

__all__ = ['bspline', 'bspline_set']

def b(x: float, k: int, i: int, t: np.ndarray) -> float:
    """
    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0
    if t[i + k] == t[i]:
        α0 = 0.0
    else:
        α0 = (x - t[i]) / (t[i + k] - t[i])
    if t[i + k + 1] == t[i + 1]:
        α1 = 0.0
    else:
        α1 = (1 - (x - t[i + 1]) / (t[i + k + 1] - t[i + 1]))
    return α0 * b(x, k - 1, i, t) + α1 * b(x, k - 1, i + 1, t)

def bspline(x: float, t: np.ndarray, c: np.ndarray, k: int) -> float:
    n = len(t) - k - 1
    return sum(c[i] * b(x, k, i, t) for i in range(n))

def bspline_set(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Generate the full set
    Args:
        x: 1D array of x points
        k: order
    Returns:
        2D array with each row a basis
    """
    t = np.linspace(x[0], x[-1] + 1E-14, k + 2)
    t_expanded = np.hstack([[t[0]] * (k), t, [t[-1]] * (k)])
    result = list()
    for i in range(2 * k + 1):
        result.append([b(xi, k, i, t_expanded) for xi in x])
    return np.array(result)
