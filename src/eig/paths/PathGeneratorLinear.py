"""
PathGeneratorLinear.py

Concrete subclass of AbstractPathGenerator that generates linear paths between
points in their original feature space. In the context of "Axiomatic
Attribution for Deep Networks" by M Sundarajan et al
(https://arxiv.org/abs/1703.01365), this corresponds to the canonical linear
path for integrated gradients as a special case of path integrated gradients.

Defines:
+ class LinearPathGenerator(AbstractPathGenerator)

"""

import numpy as np  # for numerical operations
from eig.paths.PathGeneratorAbstract import AbstractPathGenerator  # parent class
from eig.paths.path_utils import interpolate_linear  # general linear interpolation


class LinearPathGenerator(AbstractPathGenerator):
    """
    Concrete realization of AbstractPathGenerator that creates linear paths.

    Methods
    -------
    get_path(src, dst, npoints):
        Obtains linear path between src and dst.
    """
    def get_path(self, src, dst, npoints):
        """ Obtain the linear path between src and dst with equal step sizes

        Parameters
        ----------
        src: np.ndarray
            The source of the resultant path
        dst: np.ndarray
            The destination of the resultant path
        npoints: int
            The number of points in the resultant path. This must be at least
            2.

        Returns
        -------
        np.ndarray(shape=(npoints, *src.shape))
            The resulting path from src to dst, linear in original features.
        """
        # use get_paths
        return self.get_paths(src[np.newaxis], dst[np.newaxis], npoints)[0]

    def get_paths(self, src_array, dst_array, npoints):
        """ Obtain linear paths between source and destination points

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
        """
        # check provided input
        if npoints < 2:
            raise ValueError("npoints must be greater than or equal to 2")
        if src_array.shape != dst_array.shape:
            raise ValueError("src and dst must have same shapes")
        # use interpolate_linear to create the path, including endpoint
        paths = interpolate_linear(src_array, dst_array, npoints,
                                   endpoint=True, interpolate_axis=1)
        # return the resulting paths
        return paths
