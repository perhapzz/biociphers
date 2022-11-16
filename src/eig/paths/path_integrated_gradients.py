"""
path_integrated_gradients.py

Functions to perform path integrated gradients for arbitrary paths. Based on
theory described in "Axiomatic Attribution for Deep Networks" by M Sundarajan
et al (https://arxiv.org/abs/1703.01365).

Currently, integration is performed using a trapezoidal rule. Choice because it
is not more expensive than Riemann sums (as in external implementations of this
method) but has an order better convergence for most functions
Chosen over Simpson's rule because convergence is comparable or better for
larger classes of functions (less smooth then twice continuously differentiable
functions, i.e. Holder continuous/functions of bounded variation) (see
Cruz-Uribe et al 2002). Would be very straightforward to switch to Simpson's
rule if we add additional dependency on SciPy.

Defines:
+ function path_integrated_gradients(gradient_function, path, **outlier_params)
+ function paths_integrated_gradients(gradient_function, paths)
+ function run_path_integrated_gradients(gradient_function, paths,
                                         **outlier_params)

"""

# standard imports
import numpy as np  # matrix operations
# local imports
from path_utils import reinterpolate_paths


def single_path_integrated_gradients(gradient_function, path, **outlier_params):
    """ Computes integral of gradients corresponding to the provided path

    Parameters
    ----------
    gradient_function: Callable[[np.ndarray(shape=(m, *feature_shape))],
                                np.ndarray(shape=(m, *feature_shape))]
        Function that takes in an array of points in some feature space (e.g.
        shape `(m, *feature_shape)`) and returns an equal dimensional array
        corresponding to the gradients of some target with respect to those
        features for each point.
    path: np.ndarray(shape=(n, *feature_shape))
        Path over features
    outlier_params: Dict[str, Any]
        These are ignored because we have moved outlier detection out of this
        module

    Returns
    -------
    np.ndarray(shape=feature_shape)
        Array corresponding to integral for each feature.

    Notes
    -----
    Uses trapezoidal rule to perform the integration (see Wikipedia). Done
    instead of Simpson's rule, which has better convergence by typical
    analysis because typical analysis assumes twice-continuous
    differentiability, whereas these functions may not even be continuous
    (especially for ReLUs). Analysis for some classes of 'rougher' (but still
    continuous) functions have trapezoidal rule with better convergence
    properties (Cruz-Uribe 2002).
    """
    # put it through vectorized version with additional axis; remove at end
    return multiple_paths_integrated_gradients(gradient_function, path[np.newaxis])[0]


def multiple_paths_integrated_gradients(gradient_function, paths):
    """ Computes integral of gradients corresponding to provided paths

    Parameters
    ----------
    gradient_function: Callable[[np.ndarray(shape=(m, *feature_shape))],
                                np.ndarray(shape=(m, *feature_shape))]
        Function that takes in an array of points in some feature space (e.g.
        shape `(m, *feature_shape)`) and returns an equal dimensional array
        corresponding to the gradients of some target with respect to those
        features for each point.
    paths: np.ndarray(shape=(num_paths, n, *feature_shape))
        Array of paths over features with equal length

    Returns
    -------
    np.ndarray(shape=(num_paths, *feature_shape))
        Array corresponding to integral for each path and feature

    Notes
    -----
    Uses trapezoidal rule to perform the integration (see Wikipedia). Done
    instead of Simpson's rule, which has better convergence by typical
    analysis because typical analysis assumes twice-continuous
    differentiability, whereas these functions may not even be continuous
    (especially for ReLUs). Analysis for some classes of 'rougher' (but still
    continuous) functions have trapezoidal rule with better convergence
    properties (Cruz-Uribe 2002).
    """
    # compute gradient for these points. This requires flattening paths
    paths_flattened = paths.reshape(
        [paths.shape[0] * paths.shape[1]] + list(paths.shape[2:])
    )
    print("path.shape: ", paths.shape, paths_flattened.shape)
    # compute gradients on flattened path
    gradients_flattened = gradient_function(paths_flattened)
    # unflatten gradients
    gradients = gradients_flattened.reshape(paths.shape)
    print("gradients.shape: ", gradients.shape, gradients_flattened.shape)

    attributions = np.trapz(gradients, paths, axis=1)
    # return the attributions
    return attributions


def paths_integrated_gradients_error(gradient_function, paths,
                                     n_interpolate=2):
    """ Estimates error relative to reinterpolated path

    Returns absolute and relative differences between attributions evaluated
    using original path vs reinterpolated path. These evaluations are
    estimating the same integrand; however, the reduced step size should reduce
    error if the function being integrated is nice enough. Returns an estimate
    of relative error (percent error).

    Parameters
    ----------
    gradient_function: Callable[[np.ndarray(shape=(m, *feature_shape))],
                                np.ndarray(shape=(m, *feature_shape))]
        Function that takes in an array of points in some feature space (e.g.
        shape `(m, *feature_shape)`) and returns an equal dimensional array
        corresponding to the gradients of some target with respect to those
        features for each point.
    paths: np.ndarray(shape=(num_paths, n, *feature_shape))
        Array of paths over features with equal length
    n_interpolate: int
        Number of points to interpolate steps by

    Returns
    -------
    np.ndarray(shape=(num_paths, *feature_shape))
    """
    # compute attributions for original paths
    attributions_coarse = multiple_paths_integrated_gradients(gradient_function, paths)
    # compute attributions for reinterpolated paths
    # reinterpolate all the points
    index_points_array = (
        n_interpolate * np.ones(shape=(paths.shape[0], paths.shape[1] - 1),
                                dtype=int)
    )
    rpaths = reinterpolate_paths(paths, index_points_array)
    # compute attributions for reinterpolated paths
    attributions_fine = multiple_paths_integrated_gradients(gradient_function, rpaths)

    # absolute error estimate
    est_abs_err = np.abs(attributions_coarse - attributions_fine)
    # relative error estimate (absolute error / average estimate)
    # we use maximum of attributions as average instead of mean to reduce
    # issues with dividing by zero
    est_rel_err = 2 * est_abs_err / \
        (1e-10 + np.maximum(np.abs(attributions_coarse),
                            np.abs(attributions_fine)))

    # return estimated relative error
    return est_rel_err


def run_path_integrated_gradients(gradient_function, paths, **outlier_params):
    """ Return attributions for all the paths in the paths variable.

    Parameters
    ----------
    gradient_function: Callable[[np.ndarray(shape=(m, *feature_shape))],
                                np.ndarray(shape=(m, *feature_shape))]
        Function that takes in an array of points in some feature space (e.g.
        shape `(m, *feature_shape)`) and returns an equal dimensional array
        corresponding to the gradients of some target with respect to those
        features for each point.
    paths: Dict[Any, np.ndarray]
        A dictionary for paths for all samples from all baselines with keys as
        sample_id + "_" + baseline_id
    outlier_params: Dict[str, Any]
        ignored

    Returns
    -------
    Dict[Any, np.ndarray]
        Dictionary for attributions, with keys corresponding to paths, for
        attributions for paths
    """
    all_attributions = {}
    # Paths for all samples and baselines
    for i in paths.keys():
        sample_baseline_path = paths[i]
        attributions = single_path_integrated_gradients(gradient_function,
                                                        sample_baseline_path,
                                                        **outlier_params)
        all_attributions[i] = attributions
    return all_attributions
