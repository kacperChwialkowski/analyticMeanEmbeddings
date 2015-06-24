from warnings import warn
from numpy import mean, transpose, cov, cos, sin, shape, exp, newaxis, concatenate
from numpy.linalg import linalg, LinAlgError, solve
from scipy.stats import chi2

__author__ = 'kcx'


def mahalanobis_distance(difference, num_random_features):
    num_samples, _ = shape(difference)
    sigma = cov(transpose(difference))

    try:
        linalg.inv(sigma)
    except LinAlgError:
        warn('covariance matrix is singular. Pvalue returned is 1.1')
        raise

    mu = mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        stat = num_samples * mu.dot(solve(sigma, transpose(mu)))

    return chi2.sf(stat, num_random_features)


def smooth(data):
    w = linalg.norm(data, axis=1)
    w = exp(-w ** 2 / 2)
    return w[:, newaxis]


def smooth_cf(data, w, random_frequencies):
    n, _ = data.shape
    _, d = random_frequencies.shape
    mat = data.dot(random_frequencies)
    arr = concatenate((sin(mat) * w, cos(mat) * w), 1)
    n1, d1 = arr.shape
    assert n1 == n and d1 == 2 * d and w.shape == (n, 1)
    return arr


def smooth_difference(random_frequencies, X, Y):
    x_smooth = smooth(X)
    y_smooth = smooth(Y)
    characteristic_function_x = smooth_cf(X, x_smooth, random_frequencies)
    characteristic_function_y = smooth_cf(Y, y_smooth, random_frequencies)
    return characteristic_function_x - characteristic_function_y
