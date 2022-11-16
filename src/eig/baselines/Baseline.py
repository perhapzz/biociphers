"""
Baseline.py

Enhanced Integrated Gradients (EIG) allows users to use different type of baselines when computing attributions.
This class computes different types of baselines for EIG (zero, k-means, median, close and random).

1. zero: This baseline is an all zero data point. Computing zero baseline does not require a baseline class.

2. k-means: This baseline computes x clusters in the baseline class. Cluster centroids are returned as baseline points.

3. median: This baseline returns points close to the median of the baseline class.

4. close: This baseline returns points from the baseline that are closest to the sample for which attributions are being
            generated.

5. random: This baseline returns random points from the baseline class.

One or more baseline points can be returned for: k-means, median, close and random.
This class returns baseline points in the original feature space, see BaselineLatent to generate baseline points in
latent spaces.

Defines:
+ class Baseline

"""

import numpy as np
from sklearn.cluster import KMeans

BASELINES_ZERO = "zero"
BASELINES_K_MEANS = "k-means"
BASELINES_MEDIAN = "median"
BASELINES_RANDOM = "random"
MAX_POINTS_FOR_K_MEANS = 20000


class Baseline(object):
    """
    Class for baselines.

    Attributes
    ----------
    _zeros: np.zeros(shape=(number_of_baselines, number of features))
        numpy array with all zero baseline
    _k_means: np.array(shape=(number_of_baselines, number of features))
        numpy array with cluster centers as baselines
    _median: np.array(shape=(number_of_baselines, number of features))
        numpy array with number_of_baseline points closest according to L1 to median as baselines
    _random: np.array(shape=(number_of_baselines, number of features))
        numpy array with randomly chosen number_of_baseline points as baselines

    Methods
    -------
   _get_zero_baselines(data: dict, no_baselines: int) -> np.array
        Obtains zero baseline as np.array
    _get_k_means_baselines(data: dict, no_baselines: int) -> np.array
        Obtains k_means_baseline as np.array
    _get_median_baselines(data: dict, no_baselines: int) -> np.array
        Obtains median baselines as np.array
    _get_random_baselines(data: dict, no_baselines: int) -> np.array
        Obtains random baselines as np.array
    get_baseline(data: dict, no_baselines: int, baseline_type: str) -> np.array
        Based on the baseline_type, return baseline array with no_of_baselines points
    """

    def __init__(self):
        self._zeros = []
        self._k_means = []
        self._median = []
        self._random = []

    def _get_zero_baselines(self, data, no_baselines):
        """
        Compute all zero baselines.
        :param data: np.array(shape=(number of events, number of features))
        :param no_baselines: int
        :return: np.array(shape=(no_baselines, number of features)):
            zero baseline
        """
        data_shape = list(data.shape)
        data_shape[0] = no_baselines
        data_shape = tuple(data_shape)
        self._zeros = np.zeros(data_shape)

    def _get_k_means_baselines(self, data, no_baselines):
        """
        Compute k-means baselines.
        :param data: np.array(shape=(number of events, number of features))
        :param no_baselines: int
        :return: np.array(shape=(no_baselines, number of features)):
            k-means baselines
        """
        data_model = data
        data_shape = data_model.shape
        data_flattened = False
        if len(data_shape) > 2:
            data_flattened = True
            data_points_flatten = np.array([np.array(ii).flatten() for ii in data_model])
        else:
            data_points_flatten = data_model
        # To avoid computation intensive operation, we limit max number of points for kmeans
        if len(data_model) > MAX_POINTS_FOR_K_MEANS:
            print("Too many data points for k-means, should be less than {}".format(MAX_POINTS_FOR_K_MEANS))
            exit(1)
        # Run k-means with number of clusters as the number of points we are running.
        k_means = KMeans(n_clusters=no_baselines, random_state=0).fit(data_points_flatten)

        if data_flattened:
            cluster_centers = k_means.cluster_centers_
            median_shape = list(data_shape)
            median_shape[0] = no_baselines
            median_shape = tuple(median_shape)
            self._k_means = cluster_centers.reshape(median_shape)
        else:
            self._k_means = k_means.cluster_centers_

    def _get_median_baselines(self, data, no_baselines):
        """
        Compute median baselines.
        :param data: np.array(shape=(number of events, number of features))
        :param no_baselines: int
        :return: np.array(shape=(no_baselines, number of features)):
            median baselines
        """
        data_model = data
        data_shape = data_model.shape
        data_flattened = False
        if len(data_shape) > 2:
            data_flattened = True
            data_points_flatten = np.array([np.array(ii).flatten() for ii in data_model])
        else:
            data_points_flatten = data_model
        # Find median points
        data_points_median = np.tile(np.median(data_points_flatten, axis=0), (data_points_flatten.shape[0], 1))
        values = np.apply_along_axis(lambda row: np.linalg.norm(row, ord=1), 1,
                                     data_points_flatten - data_points_median)
        median_points = np.flip(np.argsort(values), axis=0)
        # Select points close to the median
        if data_flattened:
            median_points = median_points[1:no_baselines + 1]
            median_vals = data_points_flatten[median_points]
            median_shape = list(data_shape)
            median_shape[0] = no_baselines
            median_shape = tuple(median_shape)
            self._median = median_vals.reshape(median_shape)
        else:
            median_points = median_points[1:no_baselines + 1]
            self._median = data_points_flatten[median_points, :]

    def _get_random_baselines(self, data, no_baselines):
        """
        Compute random baselines.
        :param data: np.array(shape=(number of events, number of features))
        :param no_baselines: int
        :return: np.array(shape=(no_baselines, number of features)):
            random baselines
        """
        data_points = data
        # Randomly choose baseline points
        idx = np.random.choice(np.arange(0, len(data_points)), size=no_baselines)
        self._random = data_points[idx]

    def get_baseline(self, data, no_baselines, baseline_type):
        """
        Based on the baseline_type, compute and return baselines.
        :param data: np.array(shape=(number of events, number of features))
        :param no_baselines: int
        :param baseline_type: str, which baseline class (zeros, k-means, median, random)
        :return: np.array(shape=(no_baselines, number of features)):
            return baselines based on baseline_type
        """
        if baseline_type == BASELINES_ZERO:
            self._get_zero_baselines(data, no_baselines)
            return self._zeros
        if baseline_type == BASELINES_K_MEANS:
            self._get_k_means_baselines(data, no_baselines)
            return self._k_means
        if baseline_type == BASELINES_MEDIAN:
            self._get_median_baselines(data, no_baselines)
            return self._median
        if baseline_type == BASELINES_RANDOM:
            self._get_random_baselines(data, no_baselines)
            return self._random

    @staticmethod
    def get_closest_baselines(data, no_baselines, samples):
        """
        Compute closest points of opposite class as baselines.
        :param data: np.array(shape=(number of events, number of features))
        :param no_baselines: int
        :param samples: np.array(shape=(number of events, number of features))
        :return: np.array(shape=(no_baselines, number of features)): median baselines, list: baseline_ids
        """
        all_baselines = []
        all_baseline_ids = []
        data_points = data

        # if len(data_points) > MAX_POINTS_FOR_K_MEANS:
        #    print("Too many data points")
        #    exit(1)

        data_shape = data_points.shape
        data_flattened = False
        if len(data_shape) > 2:
            data_flattened = True
            data_points_flatten = np.array([np.array(ii).flatten() for ii in data_points])
            samples_flatten = np.array([np.array(ii).flatten() for ii in samples])
        else:
            data_points_flatten = data_points
            samples_flatten = samples

        cnt = 0
        for ii in samples_flatten:
            if cnt % 50 == 0:
                print(cnt)
            # Find points that are in the baseline class but close to the sample
            baseline = np.tile(ii, (data_points_flatten.shape[0], 1))
            values = np.apply_along_axis(lambda row: np.linalg.norm(row, ord=1), 1, data_points_flatten - baseline)
            close_points = np.argsort(values)
            # Select those as baseline
            close_points = close_points[1:no_baselines + 1]
            points = data_points_flatten[close_points, :]
            if data_flattened:
                close_shape = list(data_shape)
                close_shape[0] = no_baselines
                close_shape = tuple(close_shape)
                points = points.reshape(close_shape)
            all_baselines.extend(points)
            all_baseline_ids.append(cnt)
            cnt += 1
        all_baselines = np.array(all_baselines)
        all_baseline_ids = np.array(all_baseline_ids)
        # print(all_baselines.shape)
        return all_baselines, all_baseline_ids
