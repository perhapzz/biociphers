"""
path_outliers_correct.py

Provides function correct_path_outliers to apply weighted Lp distance-based
correction to large step sizes in path. This generalizes the old
correct_path_outliers which was based on L^infty distances between points
weighted by a heuristic based on IQR and multiple of the median. This previous
method was overly sensitive to noisy high-dimensional features. Old
functionality is approximated by old_correct_path_outliers to automatically
calculate weights but is not recommended at this time.

Uses function `log_concern` to print events that are of potential concern. This
will allow quick identification of paths that we may want to investigate
graphically to ensure good behavior of input paths.

Defines:
+ correct_path_outliers(path, **optional_params)
+ old_correct_path_outliers(path, **outlier_params)
+ outlier_iqr_rule(dpath, max_step_size_iqr)
+ outlier_median_rule(dpath, max_step_size_multiple)
+ determine_index_points(dpath, outlier_threshold, step_size_fill_max)
+ log_concern(*args, file, **kwargs)

"""

# standard imports
from __future__ import print_function
import sys  # for logging potential concerns
import numpy as np  # for matrix operations
from scipy.spatial.distance import cdist  # for fast pairwise distance
# local imports
from eig.paths.path_utils import reinterpolate_path  # put reinterpolating outliers
from eig.paths.path_utils import index_points_add  # determine where to reinterpolate


def correct_path_outliers(path, p=1, feature_weights=None,
                          multiply_npoints=None, add_npoints=None,
                          new_npoints=None):
    """ Returns path where large steps have been filled in linearly

    Returns path where large steps have been filled in linearly. Large steps
    are determined by a Lp distance (a.k.a. Minkowski distance) (p=1 by
    default) and steps are added to obtain a desired path size.

    Parameters
    ----------
    path: np.ndarray(shape=(npoints, *feature_shape))
        Path over features
    p: int (p >= 1)
        Parameter for distance function
    feature_weights: Optional[np.ndarray(shape=*feature_shape)]
        Scale by which each feature contributes to the distance function.
        Default is None, which corresponds to equal weights.
    multiply_npoints, add_npoints, new_npoints: Optional[int]
        Exactly one of these must be specified to change from default, which is
        equivalent to multiply_npoints=2. This determines the size of the
        resulting paths (new_npoints). The value new_npoints can be specified
        directly, or it can be calculated by either
        new_npoints=n_points+add_npoints or
        new_npoints=n_points*multiply_npoints.

    Returns
    -------
    np.ndarray(shape=(new_npoints, *feature_shape))
        The new path where large steps have been filled in linearly
    """
    # determine the new number of points
    # first check if we have an appropriate number of inputs
    narg_npoints = sum(
        bool(x) for x in (multiply_npoints, add_npoints, new_npoints)
    )
    if narg_npoints == 0:
        multiply_npoints = 2  # DEFAULT VALUE if none specified
    elif narg_npoints > 1:
        raise ValueError("At most one npoints arguments may be specified")
    # get the resulting number of points
    if multiply_npoints:
        new_npoints = int(len(path) * multiply_npoints)
    if add_npoints:
        new_npoints = int(len(path) + add_npoints)

    # get the steps in the path
    dpath = path[1:] - path[:-1]
    # if we have feature_weights, scale accordingly
    if feature_weights:
        dpath = dpath * feature_weights[np.newaxis, :]

    # calculate distances between points
    distances = cdist(
        np.zeros(shape=[1] + list(dpath[0].shape)),  # distance to zero
        dpath,
        metric="minkowski", p=p
    )[0]

    # determine the indices of points at which to add by minmax distance
    index_points = index_points_add(distances, new_npoints - len(path))

    # reinterpolate the input path
    new_path = reinterpolate_path(path, index_points)

    # return the resulting path
    return new_path


def old_correct_path_outliers(
        path, max_step_size_iqr=1.5, max_step_size_multiple=10.,
        step_size_fill_max=40., path_length_max_multiple=10.):
    """ Returns path where large steps have been filled in linearly

    Parameters
    ----------
    path: np.ndarray(shape=(n, *feature_shape))
        Path over features
    max_step_size_iqr: float
        Step sizes must be no greater than this many times
        interquartile range above the 75th percentile. Must be
        greater than zero. Used by `outlier_iqr_rule`.
    max_step_size_multiple: float
        Step sizes must be no greater than this many times larger
        than the median per feature. Must be greater than 1.
        Used by `outlier_median_rule`.
    step_size_fill_max: int
        If step size is an outlier, maximum number of steps allowed to be
        used to fill in the gap is this. Must be greater than 1 (something
        changes). Used by `determine_index_points`.
    path_length_max_multiple: float
        Resulting path must be no larger than this many times the original
        size (new_n <= path_length_max_multiple * n). Returns error on runtime
        if this happens.

    Returns
    -------
    np.ndarray(shape=(new_n, *feature_shape))
        The new path where large steps have been filled in linearly
    """
    # check that the parameters make sense
    assert path_length_max_multiple > 1.

    # compute the step sizes per feature (name dpath ~ differential calculus)
    dpath = np.abs(path[1:] - path[:-1])

    # obtain the outlier cutoffs according to our rules
    max_iqr_rule = outlier_iqr_rule(dpath, max_step_size_iqr)
    max_median_rule = outlier_median_rule(dpath, max_step_size_multiple)
    # we combine these cutoffs by the feature-wise *minimum* of these rules
    outlier_threshold = np.minimum(max_iqr_rule, max_median_rule)
    # raise maximum if non-positive (really only if zero -- no divide by zero)
    outlier_threshold[outlier_threshold <= 0] = 1e-3

    # determine the number of points that will be needed for each index
    index_points = determine_index_points(dpath, outlier_threshold,
                                          step_size_fill_max)

    # determine if we return original path, raise error, or construct new path
    # check length of path (+1 comes from going from steps to endpoints)
    total_points = 1 + index_points.sum()
    if total_points <= len(path):
        # No outliers, we should just return the original path
        return path
    log_concern("Outliers detected.")

    # get new path
    # determine desired length of new path
    total_points = 1 + index_points.sum()
    if total_points > path_length_max_multiple * len(path):
        multiple = total_points / len(path)  # assume float division (Python3)
        err_msg = [
            "Proposed corrected path is too many times longer than original.",
            "({} vs error threshold {})".format(multiple,
                                                path_length_max_multiple)
        ]
        log_concern(" ".join(err_msg))
        raise RuntimeError(" ".join(err_msg))
    new_path = reinterpolate_path(path, index_points)

    # return the new path
    return new_path


def outlier_iqr_rule(dpath, max_step_size_iqr, zero_threshold=1e-10):
    """ Determines the maximum step size per feature according to the IQR rule

    Determines the maximum step size per feature according to the IQR
    (interquartile range) rule. For a given feature, this is: 3rd quartile +
    max_step_size_iqr * IQR. This corresponds to the common rule that
    outliers are points that are 1.5 times IQR above the third quartile (for
    max_step_size_iqr equal to 1.5).

    Helper function for correct_path_outliers()

    Parameters
    ----------
    dpath: np.ndarray(shape=(n, *feature_shape))
        Step sizes over n steps for all the features
    max_step_size_iqr: float
        How many times the IQR above the 3rd quartile we are willing to
        tolerate
    zero_threshold: Optional[float]
        When difference between max and quartile3 is less than zero_threshold,
        ignore it because effectively zero change

    Returns
    -------
    np.ndarray(shape=feature_shape)
        Maximum step size per feature
    """
    # check that the provided value makes some semblance of sense
    assert max_step_size_iqr > 0.
    # get first and third quartiles
    quartile1, quartile3 = np.percentile(dpath, (25, 75), axis=0)
    # get the IQR
    interquartile_range = quartile3 - quartile1
    # get the cutoff for an outlier by this rule
    max_iqr_rule = quartile3 + max_step_size_iqr * interquartile_range

    # apply zero_threshold
    feature_max = np.max(dpath, axis=0)
    # when is the difference negligible?
    ndx_zt = feature_max - quartile3 < zero_threshold
    # set outlier threshold above the max so that it won't be used
    max_iqr_rule[ndx_zt] = feature_max[ndx_zt] + zero_threshold

    # return this value
    return max_iqr_rule


def outlier_median_rule(dpath, max_step_size_multiple):
    """ Determines the maximum step size per feature according to median rule

    Determines maximum step size per feature according to median rule. This
    sets a threshold for being called an outlier to max_step_size_multiple
    times the median value per feature.

    Helper function for correct_path_outliers()

    Parameters
    ----------
    dpath: np.ndarray(shape=(n, *feature_shape))
        Step sizes over n steps for all the features
    max_step_size_multiple: float
        How many times the median we are willing to tolerate

    Returns
    -------
    np.ndarray(shape=feature_shape)
        Maximum step size per feature
    """
    # check that the provided parameter makes some semblance of sense
    assert max_step_size_multiple > 1.
    # get median value
    median = np.percentile(dpath, 50, axis=0)
    # get the cutoff for an outlier by this rule
    max_median_rule = max_step_size_multiple * median
    # return this value
    return max_median_rule


def determine_index_points(dpath, outlier_threshold, step_size_fill_max):
    """ Determines number of steps in new path per index of old path

    Parameters
    ----------
    dpath: np.ndarray(shape=(n, *feature_shape))
        Step sizes over n steps for all the features
    outlier_threshold: np.ndarray(shape=feature_shape)
        Threshold per feature that under which we are willing to accept
    step_size_fill_max: int
        We do not permit the number of steps per index in new path to be
        greater than this value (to prevent very large values)

    Returns
    -------
    np.ndarray(shape=(n,))
        The number of steps in the new path per step/index in the old path
    """
    # assert that the parameter makes sense
    assert step_size_fill_max > 1
    # Obtain the ratio of each step size compared to outlier threshold
    dpath_outlier_ratio = dpath / outlier_threshold[np.newaxis, :]

    # Determine number of points that will be needed for each index
    # per feature
    index_points_features = np.ceil(dpath_outlier_ratio).astype(int)
    # overall is maximum of the per feature values
    index_points = index_points_features.max(axis=tuple(range(1, dpath.ndim)))

    # they all must be positive (should only happen if degenerate points)
    index_points[index_points <= 0] = 1

    # apply the step_size_fill_max threshold, log concerns if met
    max_points = np.max(index_points)  # max points needed over all steps
    if max_points > step_size_fill_max:
        concern_msg = [
            "Outliers detected many times (more than {} times) threshold",
            "Largest is {} times above threshold"
            "There will be steps above threshold"
        ]
        log_concern("\n".join(concern_msg).format(step_size_fill_max,
                                                  max_points))
        index_points[index_points > step_size_fill_max] = step_size_fill_max

    # return these index_points
    return index_points


def log_concern(*args, **kwargs):
    """ Logs concerns raised by functions in this module

    Currently pass-through to print function using sys.stderr.
    """
    print(*args, file=sys.stderr, **kwargs)
    return
