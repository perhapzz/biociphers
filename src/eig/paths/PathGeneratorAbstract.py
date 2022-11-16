"""
PathGeneratorAbstract.py

Abstract (unimplemented) class for generating paths between points that
provides interface for concrete subclasses that provide paths between a
source/reference point and a destination point for which relative attributions
for features are deisred. This is in the context of path integrated gradients
as described in "Axiomatic Attribution for Deep Networks" by M Sundarajan et al
(https://arxiv.org/abs/1703.01365).

Defines:
+ class AbstractPathGenerator(object)

"""

import numpy as np


class AbstractPathGenerator(object):
    """
    Abstract class for generating path from one point to another.

    Methods
    -------
    get_path(src, dst, npoints):
        Obtain path over npoints points starting at src and ending at dst.
    get_paths(src_array, dst_array, npoints):
        Obtain paths over npoints points for each of the src/dst pairs.
    """
    def get_path(self, src, dst, npoints):
        """ Unimplemented function to obtain discrete path between src, dst.

        Parameters
        ----------
        src: np.ndarray
            The source of the resultant path
        dst: np.ndarray
            The destination of the resultant path. Must have same shape as src.
        npoints: int
            The number of points in the resultant path. Must be at least 2.

        Returns
        -------
        np.ndarray(shape=(npoints, *src.shape))
            The path from src to dst

        Notes
        -----
        This is not implemented for AbstractPathGenerator. Subclasses should
        keep the same function signature
        """
        err_msg = "This is an abstract class; use its subclasses."
        raise NotImplementedError(err_msg)

    def get_paths(self, src_array, dst_array, npoints):
        """ Non-vectorized function to obtain discrete paths between src, dst

        Parameters
        ----------
        src_array, dst_array: np.ndarray(shape=(npaths, *feature_shape))
            The sources and destinations of resulting paths
        npoints: int
            The number of points in the resulting paths. Must be at least 2.

        Returns
        -------
        np.ndarray(shape=(npaths, npoints, *feature_shape))
            The paths from sources to destinations

        Notes
        -----
        This should be overriden by concrete subclasses to provide faster
        functionality if possible
        """
        # initialize return array
        paths = np.zeros(
            shape=[len(src_array), npoints] + list(src_array.shape[1:])
        )

        # loop over all source/destination combinations and get path
        for path_ndx, (src, dst) in enumerate(zip(src_array, dst_array)):
            paths[path_ndx] = self.get_path(src, dst, npoints)

        # return the resulting paths array
        return paths
