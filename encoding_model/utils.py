from typing import Union, Generator, Tuple
import numpy as np

def convert(array) -> Union[np.ndarray, dict, None, list]:
    """The main conversion function"""
    if isinstance(array, np.ndarray):
        if not array.dtype.fields:
            if len(array) == 0:
                return None
            if len(array) == 1:
                return convert(array[0])
            elif array.dtype == np.dtype('O'):
                return [convert(item) for item in array]
            else:
                return array
        else:
            if len(array) == 0:
                return None
            result = dict()
            for field in array.dtype.fields.keys():
                result[field] = convert(array[field][0])
            return result
    else:
        return array

ValidateSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def _split_folds(X: np.ndarray, y: np.ndarray, folds: int) -> Generator[ValidateSet, None, None]:
    """assumes repeats (observations) are on axis 0"""
    test_size = y.shape[0] // folds
    full_index = np.arange(y.shape[0])
    indices = np.random.permutation(np.arange(test_size * folds)).reshape(folds, test_size)
    for idx_te in indices:
        idx_tr = np.setdiff1d(full_index, idx_te)
        yield idx_te, X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]

def split_folds(size: int, folds: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """assumes repeats (observations) are on axis 0"""
    test_size = size // folds
    full_index = np.arange(size)
    indices = np.random.permutation(test_size * folds).reshape(folds, test_size)
    for idx_te in indices:
        idx_tr = np.setdiff1d(full_index, idx_te)
        yield idx_tr, idx_te
