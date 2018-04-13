import numpy as np


def is_standardish(array, globally=False, epsilon=1e-7):
    if globally:
        return np.allclose(array.mean(), 0., atol=epsilon) and np.allclose(array.std(), 1., atol=epsilon )
    return np.allclose(array.mean(axis=0), 0., atol=epsilon) and np.allclose(array.std(axis=0), 1., atol=epsilon)


def is_normalish(array, epsilon=1e-7):
    return np.allclose(np.linalg.norm(array, axis=1), 1., atol=epsilon)
