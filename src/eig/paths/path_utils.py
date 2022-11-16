"""
path_utils.py

Provides utility functions for manipulating paths over tensor features provided
as arrays with shape (npoints, *feature_shape), where feature_shape corresponds
to the shape of individual tensors along the path.

Defines:
+ function interpolate_linear(src, dst, npoints, endpoint)
+ reinterpolate_path(path, index_points)
+ index_points_add(distances, add_points)

"""

import numpy as np
import heapq  # heap data structure


def interpolate_linear(src, dst, npoints, endpoint, interpolate_axis=0):
    """ Creates linear interpolating path between src and dst

    Creates linear interpolating path between src and dst. This is implemented
    separately from LinearPathGenerator to generalize on including the endpoint
    or not in the path. This is useful for adjusting paths linearly when step
    sizes between points become too large.

    Parameters
    ----------
    src, dst: np.ndarray
        The source(s) and destination(s) for interpolation. Must have same
        shape
    npoints: int
        The number of interpolated points to return (src always included, dst
        is included if endpoint=True)
    endpoint: bool
        Specifies whether the endpoint (dst) should be included or not
    interpolate_axis: Optional[int]
        Index of interpolation axis in resulting shape. Default is 0.

    Returns
    -------
    np.ndarray
        The resulting linear interpolation between src and dst. Has shape of
        src but with axis added in the `interpolate_axis` position, with
        shape[interpolate_axis] = npoints
    """
    # path for given interpolant value (alpha in [0, 1]) is src+alpha*(dst-src)
    alpha = np.linspace(0, 1, npoints, endpoint=endpoint)  # flattened
    # we need to reshape for proper broadcasting in NumPy
    # alpha = alpha.reshape((npoints, *[1 for dim in src.shape]))
    alpha = alpha.reshape(
        [npoints] + [1 for dim in src.shape[interpolate_axis:]]
    )
    # compute the path now
    path = np.expand_dims(src, axis=interpolate_axis) \
        + alpha * np.expand_dims(dst - src, axis=interpolate_axis)
    # return the path
    return path


def reinterpolate_path(path, index_points):
    """ Creates new path interpolating old path where index_points > 1

    Parameters
    ----------
    path: np.ndarray(shape=(n, *feature_shape))
        Path over features
    index_points: np.ndarray(shape=(n-1, ))
        The number of steps in the new path per step/index in the old path

    Returns
    -------
    np.ndarray(shape=(new_n, *feature_shape))
        The new path where large steps have been filled in linearly
    """

    # get the desired length of the new path (+1 from final point of path)
    total_points = 1 + index_points.sum()

    # initialize our new path
    # new_path = np.zeros(shape=(total_points, *path.shape[1:]))
    new_path = np.zeros(shape=[total_points] + list(path.shape[1:]))
    # add the destination point to the end
    new_path[-1] = path[-1]

    # loop over old indices, add new indices as indicated
    new_ndx = 0  # keep track of new index
    for old_ndx, add_points in enumerate(index_points):
        if add_points > 1:
            # get interpolating path between endpoints of this step
            interpolating_path = interpolate_linear(path[old_ndx],
                                                    path[old_ndx + 1],
                                                    add_points,
                                                    endpoint=False)
            # put in appropriate location in new path
            new_path[new_ndx:(new_ndx + add_points)] = interpolating_path
        else:  # add_points == 1 (only 1 point to add)
            new_path[new_ndx] = path[old_ndx]
        # either way, update new_ndx
        new_ndx += add_points
    # done looping through, should have new path. Return it.
    return new_path


def reinterpolate_paths(paths, index_points_array):
    """ Creates new path interpolating old paths where index_points > 1

    Creates new path interpolating old paths where index_points > 1. Requires
    index_points for all points to sum to the same final number of points

    Parameters
    ----------
    paths: np.ndarray(shape=(num_paths, n, *feature_shape))
        Paths over features
    index_points: np.ndarray(shape=(num_paths, n-1, ))
        The number of steps in the new path per step/index in the old path

    Returns
    -------
    np.ndarray(shape=(num_paths, new_n, *feature_shape))
        The new paths where large steps have been filled in linearly
    """
    # get desired length of the new path
    total_points = 1 + index_points_array.sum(axis=1)
    # check lengths
    if not np.all(total_points[0] == total_points):
        raise ValueError("Index points array must have equal row sums")

    # initialize returned path
    new_paths = np.zeros(
        shape=[len(paths), total_points[0]] + list(paths.shape[2:])
    )
    # loop and reinterpolate paths (not vectorized unfortunately)
    for ndx, (path, index_points) in enumerate(zip(paths, index_points_array)):
        new_paths[ndx] = reinterpolate_path(path, index_points)

    # return new paths
    return new_paths


def index_points_add(distances, add_points):
    """ Determines index points for reinterpolate_path by minmax distance

    Returns index_points for reinterpolate_path given provided "distances" in a
    path to minimize the maximum distance between points by adding points in
    between

    Parameters
    ----------
    distances: np.ndarray(shape=(m,), dtype=float)
    index_points: int

    Returns
    -------
    index_points: np.ndarray(shape=(m,), dtype=int)
    """
    # initialize index_points to one
    index_points = np.ones(shape=distances.shape, dtype=int)
    # initialize a heap with negative distances
    distance_heap = [(-d, ndx) for ndx, d in enumerate(distances)]
    heapq.heapify(distance_heap)
    # add the points according to the maximum distance at each time
    for _ in range(add_points):
        # the first element of distance_heap corresponds to where we will add
        update_ndx = distance_heap[0][1]
        index_points[update_ndx] += 1
        # update_item is the new distance (negative) and the same index
        update_item = (
            -distances[update_ndx] / index_points[update_ndx],
            update_ndx
        )
        # pop this first item and replace it with the new one
        heapq.heapreplace(distance_heap, update_item)
    # we are done adding all the points...
    return index_points
