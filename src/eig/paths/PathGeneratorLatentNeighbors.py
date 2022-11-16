"""
PathGeneratorNeighbors.py

Concrete subclass of AbstractPathGenerator that generates nonlinear paths
between two points in some feature space. The nonlinear path is determined by
taking the shortest weighted path over the k-nearest neighbors graph composed
of a training set of features and the source and destination points. By default
the feature space is the original feature space, but an encoder may be passed
in to construct the nearest-neighbor graph in latent space.

It is worth noting that it is possible that the k-nearest neighbors graph has
multiple connected components. In this case, it is not possible to guarantee
paths between points. In this case, a runtime error will be raised by the
constructor.

Defines:
+ class LatentNeighborsPathGenerator(AbstractPathGenerator)

"""

# matrix operations
import numpy as np

# nearest neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric

# sparse graph functions
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

# paths
from eig.paths.PathGeneratorAbstract import AbstractPathGenerator
from eig.paths.PathGeneratorLinear import LinearPathGenerator

# path utilities
from eig.paths.path_utils import reinterpolate_path, index_points_add

# constants
DEFAULT_NUM_NEIGHBORS = 5  # default number of neighbors to use in the graph
EST_MIN_NPOINTS_TRIES = 10  # number of seed points to give safe use bound


class LatentNeighborsPathGenerator(AbstractPathGenerator):
    """
    Conrete realization of AbstractPathGenerator that creates a path between
    nearest neighbors to get from source point to destination point.

    Attributes
    ----------
    num_neighbors: int
        the number of neighbors used to create the neighbor graph
    distance_metric: DistanceMetric
        instance of DistanceMetric exposing distance function used
    est_min_npoints: int
        bound on npoints that will guarantee no runtime error on get_path
        so long as npoints >= est_min_npoints
    _data: np.ndarray(shape=(n, *feature_shape))
        original input data for generating paths for
    _encoder: Callable[[np.ndarray(shape=(m, *feature_shape))], np.ndarray]
        encoder function to map to latent space that nearest-neighbors search
        is performed on
    _neighbors: NearestNeighbors
        object "trained" on initializing dataset to identify nearest neighbors
        of input points
    _data_graph: csr_matrix
        sparse, weighted adjacency matrix of distances between nearest
        neighbors. Output from self._neighbors.kneighbors_graph. Note that this
        is not symmetric because the property of being nearest neighbors is not
        reflexive, so graph algorithms used need to handle this appropriately
        as we want to treat the graph as undirected (csgraph.dijkstra can do
        this)

    Methods
    -------
    get_path(src, dst, npoints):
        Obtains path between src and dst. Raises runtime error if number of
        points along graph is greater than npoints

    _estimate_min_npoints():
        Obtains bound on npoints
    """
    def __init__(self, data, encoder=None, decoder=None, encoder_data=None, decoder_data=None,
                 num_neighbors=DEFAULT_NUM_NEIGHBORS, metric="minkowski", p=2, metric_params=None):
        """
        Parameters
        ----------
        data: np.ndarray(shape=(n, *feature_shape))
            Points in feature space to interpolate between nearest neighbors
        encoder: Optional[Callable[[np.ndarray], np.ndarray]]
            Optional encoder to map to latent space with more meaningful
            distances for nearest neighbors. Default is identity function
        decoder: Optional[Callable[[np.ndarray], np.ndarray]]
            Optional decoder to map to original feature space with more meaningful
            distances for nearest neighbors. Default is identity function
        num_neighbors: int
            Number of nearest neighbors to connect points with
        metric, p, metric_params:
            Parameters for NearestNeighbors and DistanceMetric specifying the
            distance to use

        Notes
        -----
        Raises RuntimeError if resulting nearest neighbors graph is not
        connected.
        """
        # save the data points

        self._data = data
        if encoder_data is not None:
            all_encoder_data = [data, encoder_data]
        else:
            all_encoder_data = [data]

        assert encoder is not None and decoder is not None, "please specify encoder/decoder"
        self._encoder = encoder
        self._decoder = decoder

        # get the nearest neighbors information
        # set up the NearestNeighbors object
        self._neighbors = NearestNeighbors(n_neighbors=num_neighbors,
                                           metric=metric, p=p,
                                           metric_params=metric_params)
        # use it to 'fit'/remember latent projection of data, but flattened
        latent_data = self._encoder(all_encoder_data)
        self._neighbors.fit(
            latent_data.reshape(
                (len(latent_data), np.prod(latent_data.shape[1:]))
            )
        )

        self._data = latent_data
        # use it to cache the majority of the adjacency matrix
        self._data_graph = self._neighbors.kneighbors_graph(mode="distance")
        # make sure that this graph is connected
        n_components = connected_components(self._data_graph, directed=False,
                                            return_labels=False)
        if n_components > 1:
            # raise RuntimeError("Nearest neighbors graph not connected")
            print("Nearest neighbors graph not connected", n_components)

        # define self.distance_metric
        if not metric_params:
            use_metric_params = dict()
        else:
            use_metric_params = metric_params.copy()
        if metric.find("minkowski") >= 0:  # only provide p for minkowski
            use_metric_params["p"] = p
        self.distance_metric = DistanceMetric.get_metric(metric,
                                                         **use_metric_params)

        # define self._get_linear_path
        self._get_linear_path = LinearPathGenerator().get_path
        # calculate self._est_min_npoints
        self._est_min_npoints = self._estimate_min_npoints(self._data_graph)
        return

    def get_path_decode(self, src_array, dst_array, decoder_val, npoints):
        """ Obtain nonlinear paths between points using encoder/decoder

        Uses encoder/decoder to make a nonlinear path in original feature
        space corresponding to linear path in latent space corresponding to
        encoder/decoder.

        Parameters
        ----------
        src_array, dst_array: np.ndarray(shape=(npaths, *feature_shape))
            The sources and destinations of resulting paths
        npoints: int
            The number of points in the resultant path. This must be at least 4
            because it requires both the original and autoencoded endpoints.

        Returns
        -------
        np.ndarray(shape=(npoints, *src.shape))
            The resulting paths from sources to destinations, linear in latent
            features
        """

        if isinstance(src_array, list) and isinstance(dst_array, list):
            src = src_array[0][np.newaxis, :]
            dst = dst_array[0][np.newaxis, :]
            if src_array[1] is not None:
                src_encoder = [src, src_array[1]]
            else:
                src_encoder = [src]
            if dst_array[1] is not None:
                dst_encoder = [dst, dst_array[1]]
            else:
                dst_encoder = [dst]
        else:
            src_encoder = src_array
            dst_encoder = dst_array
            src = src_array
            dst = dst_array
        # check provided input
        if npoints < 4:
            raise ValueError("npoints must be greater than or equal to 4")
        if src.shape != dst.shape:
            print(src, dst)
            raise ValueError("src and dst must have same shapes")

        src_array = self._encoder(src_encoder)
        dst_array = self._encoder(dst_encoder)
        endpoints = np.array(
            [src_array.flatten(), dst_array.flatten()]
        )

        # determine whether we should just use a linear path
        dist_direct = self.distance_metric.pairwise(endpoints)[0, 1]
        dist_graph_min = self._neighbors.kneighbors(
            endpoints, n_neighbors=1
        )[0].min()
        if dist_direct < dist_graph_min:
            # if the two points are closer together than either of them is to
            # the original data, just use linear path
            return self._get_linear_path(src_array, dst_array, npoints)
        # if we are here, we will use the shortest distance path

        # determine adjacency of src, dst to originally provided data
        adj_endpoints = self._neighbors.kneighbors_graph(endpoints,
                                                         mode="distance")

        # combine with self._data_graph
        # we append the csr_matrix structures appropriately
        adj_data = np.concatenate(
            (self._data_graph.data, adj_endpoints.data)
        )
        adj_indices = np.concatenate(
            (self._data_graph.indices, adj_endpoints.indices)
        )
        adj_indptr = np.concatenate(
            (self._data_graph.indptr,
             self._data_graph.indptr[-1] + adj_endpoints.indptr[1:])
        )
        # construct adjacency matrix
        num_vertices = len(adj_indptr) - 1
        adj = csr_matrix((adj_data, adj_indices, adj_indptr),
                         shape=(num_vertices, num_vertices))
        src_ndx = num_vertices - 2
        dst_ndx = num_vertices - 1

        # find the shortest path from src to dst
        predecessors = dijkstra(adj, indices=[src_ndx],
                                return_predecessors=True, directed=False)[1]
        # number of steps between src and dst according to this path
        num_steps = self._num_steps(adj, src_ndx, predecessors)[0, dst_ndx]
        # if this requires a path longer than requested it is a runtime error
        if num_steps >= npoints:
            raise RuntimeError("Number of points requested insufficient")

        # therefore construct the minimum path between points
        min_path = np.zeros(shape=[num_steps + 1] + list(src_array.shape[1:]))
        min_path[0] = src_array  # source
        min_path[-1] = dst_array  # destination
        cur_ndx = dst_ndx
        # for all the path indices we haven't assigned yet
        for path_ndx in range(num_steps - 1, 0, -1):
            # update current index to its predecessor
            cur_ndx = predecessors[0, cur_ndx]
            # update path with this one
            min_path[path_ndx] = self._data[cur_ndx]

        # now we need to expand this to requested number of points
        # get distances between points according to our metric
        dmin_path = min_path[1:] - min_path[:-1]
        dmin_path = np.array([ii.flatten() for ii in dmin_path])
        distances = self.distance_metric.pairwise(
            np.zeros(shape=min_path[0].shape)[np.newaxis],
            dmin_path
        )[0]
        # get number of points for reinterpolation
        index_points = index_points_add(distances, npoints - len(min_path))
        # reinterpolate to get final path
        path = reinterpolate_path(min_path, index_points)
        path_list = [path]
        if decoder_val is not None:
            path_list.extend(decoder_val)

        decoded_path = self._decoder(
            path_list
        )
        # return the final path
        return decoded_path

    @property
    def num_neighbors(self):
        """ Number of neighbors used to create graph structure
        """
        return self._neighbors.n_neighbors

    @property
    def est_min_npoints(self):
        """ Safe minimum value for npoints in get_path
        """
        return self._est_min_npoints

    @classmethod
    def _estimate_min_npoints(cls, graph, n_tries=EST_MIN_NPOINTS_TRIES):
        """ Computes bound on 3 + maximum unweighted length between points

        Parameters
        ----------
        graph: csr_matrix(shape=(n, n))
            Sparse weighted adjacency matrix between points
        n_tries: int (n_tries < n)
            Number of points to start from to get lowest bound

        Returns
        -------
        int
            Bound on npoints

        Notes
        -----
        Uses Dijkstra's algorithm for n_tries randomly sampled starting points.
        Triangle inequality says that the maximum distance between any two
        points in the graph is less than or equal to 2 times maximum distance
        from arbitrarily chosen point to rest of points. So get minimum value
        of this distance over some number of tries of arbitrary points and add
        1 to go from steps to points on the graph, 2 for source/destination.

        Assumes graph is connected.
        """
        # prevent too many tries
        n_tries = min(graph.shape[0], n_tries)
        # get indices of starting points
        start_points = np.random.choice(graph.shape[0], size=n_tries,
                                        replace=False)

        # get shortest paths between these points and rest of points
        # note we have to do this weighted and then compute number of edges
        # because there can be a difference
        # we only care about the predecessors (second element of tuple)
        predecessors = dijkstra(graph, directed=False, indices=start_points,
                                return_predecessors=True)[1]
        # get number of steps for these
        num_steps = cls._num_steps(graph, start_points, predecessors)
        # now compute minimum of maximum number of steps
        minmax_steps = np.min(np.max(num_steps, axis=1))
        # return 3 plus two times that
        return 3 + 2 * minmax_steps

    @staticmethod
    def _num_steps(graph, source, predecessors):
        """ Returns number of steps from source indices to rest of graph

        Parameters
        ----------
        graph: csr_matrix(shape=(n, n))
            Sparse weighted adjacency matrix between points
        source: array_like(shape=(m,))
            Indices of starting points
        predecessors: np.ndarray(shape=(m, n))
            Predecessors for the path from a source to destination

        Returns
        -------
        np.ndarray(shape=(m, n), dtype=int)
            Number of steps on graph corresponding to predecessors
        """
        # initialize return value
        num_steps = np.zeros(shape=predecessors.shape, dtype=int)
        # get current predecessors
        current_predecessors = predecessors.copy()
        # identify indices of points that can step back
        ndx_step = current_predecessors >= 0
        # while we have any points that can step back
        while np.any(ndx_step):
            # update number of steps
            num_steps[current_predecessors >= 0] += 1
            # step back
            row_predecessors = np.where(ndx_step)[0]
            col_predecessors = current_predecessors[ndx_step]
            current_predecessors[ndx_step] = predecessors[row_predecessors,
                                                          col_predecessors]
            # update indices of points that can step back
            ndx_step = current_predecessors >= 0
        # return resulting number of steps
        return num_steps