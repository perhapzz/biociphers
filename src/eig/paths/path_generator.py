"""
path_generator.py

Defines function interface to obtain function to generate paths.

Defines:
+ function get_path_generator(path_key, *args, **kwargs)
+ function get_paths_generator(path_keys, *args, **kwargs)

"""

from eig.paths.PathGeneratorLinear import LinearPathGenerator
from eig.paths.PathGeneratorLatentLinear import LatentLinearPathGenerator
from eig.paths.PathGeneratorNeighbors import NeighborsPathGenerator


# Update this dictionary if we define new path generators
DEFINED_PATH_GENERATORS = {
    "linear": LinearPathGenerator,
    "latent_linear": LatentLinearPathGenerator,
    "neighbors": NeighborsPathGenerator,
}


def get_path_generator(path_key, *args, **kwargs):
    """ Return function to get path between points using provided parameters.

    Returns function to obtain path between points. Dependent on global
    constant dictionary DEFINED_PATH_GENERATORS to match strings to concrete
    subclasses of AbstractPathGenerator.

    Parameters
    ----------
    path_key: str
        Key for obtaining concrete subclass from DEFINED_PATH_GENERATORS.
    *args, **kwargs:
        Positional/named arguments for initialization of the concrete
        subclass obtained by path_key.

    Returns
    -------
    Callable[[array_like, array_like, int], array_like]
        The member function get_path(src, dst, npoints: int) from
        instantiation of the path generator. This function obtains the linear
        path between src and dst with equal step
    """
    # obtain class we are using
    path_class = DEFINED_PATH_GENERATORS[path_key]
    # obtain instantiation of this class
    path_object = path_class(*args, **kwargs)
    # return member function
    return path_object.get_path


def get_paths_generator(path_key, *args, **kwargs):
    """ Return function to get paths between points using provided parameters.

    Returns function to obtain paths between points. Dependent on global
    constant dictionary DEFINED_PATH_GENERATORS to match strings to concrete
    subclasses of AbstractPathGenerator.

    Parameters
    ----------
    path_key: str
        Key for obtaining concrete subclass from DEFINED_PATH_GENERATORS.
    *args, **kwargs:
        Positional/named arguments for initialization of the concrete
        subclass obtained by path_key.

    Returns
    -------
    Callable[[array_like, array_like, int], array_like]
        The member function get_path(src, dst, npoints: int) from
        instantiation of the path generator. This function obtains the linear
        path between src and dst with equal step
    """
    # obtain class we are using
    path_class = DEFINED_PATH_GENERATORS[path_key]
    # obtain instantiation of this class
    path_object = path_class(*args, **kwargs)
    # return member function
    return path_object.get_paths
