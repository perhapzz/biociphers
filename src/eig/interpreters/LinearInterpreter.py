"""
LinearInterpreter.py

Class for Linear Interpreter (O-L-IG: Original feature space Linear Integrated Gradients). This class gives attributes
for features using linear path in the original feature space. The baselines are computed in the original feature space,
if you want baselines to be computed in the latent space of the autoencoder, use LinearInterpreterLatentBL.

Defines:
+ class LinearInterpreter
"""
import numpy as np
import tensorflow as tf
from eig.interpreters.Interpreter import Interpreter
from eig.baselines.Baseline import Baseline
from eig.paths.PathGeneratorLinear import LinearPathGenerator

BASELINES_ZERO = "zero"
BASELINES_K_MEANS = "k-means"
BASELINES_MEDIAN = "median"
BASELINES_RANDOM = "random"
BASELINES_CLOSE = "close"
ALL_BASELINES = [BASELINES_ZERO, BASELINES_K_MEANS, BASELINES_MEDIAN,
                 BASELINES_RANDOM, BASELINES_CLOSE]


class LinearInterpreter(Interpreter):
    """
    This is the Interpreter for the linear path in the original feature space and baseline computed in original feature
    space.
    """

    def __init__(self, baselines_data, samples_data, nn_model=None, baseline_type=BASELINES_MEDIAN,
                 no_baselines=3, no_points=250):
        """
        Initialize variables for the linear interpreter.
        :param baselines_data: list, [np.array(), data from where baselines are to be chosen.]
        :param samples_data: list, [np.array(), sample data]
        :param other_data: list, if the nn model has additional placeholders, they can be put here.
        :param nn_model: DeepNetworkModel, Object of the DeepNetworkModel class
        :param baseline_type: str, type of baseline (zero, encoded-zero, k-means, median, random, closest)
        :param no_baselines: int, number of baselines per sample
        :param no_points: int, number of data points to be computed between the baseline and the sample.
        """

        assert baseline_type in ALL_BASELINES, "baseline type not available, choose from available options {}".format(
            ALL_BASELINES)
        assert nn_model is not None, "Provide deep learning model for interpretation. "

        # Initialize linear path generator object
        ll = LinearPathGenerator()

        samples = samples_data[0]
        baseline_data = baselines_data[0]

        # compute paths for sample and baseline data
        baseline_obj = Baseline()
        # If closest baseline, we need both sample and baseline data to compute the baselines
        if baseline_type == BASELINES_CLOSE:
            base_points = baseline_obj.get_closest_baselines(baseline_data, no_baselines, samples)
            baselines, baseline_ids = base_points
        # If not closest, only baseline data suffices to compute the baselines
        else:
            base_points = baseline_obj.get_baseline(baseline_data,
                                                    no_baselines,
                                                    baseline_type)
            baselines = np.repeat(base_points, len(samples), axis=0)

        # If more than one baselines per sample, repeat the sample array no_baselines times.
        samples = np.repeat(samples, no_baselines, axis=0)

        # Compute paths using the baselines and samples, no_points gives number of data points to compute between
        # baseline and sample.
        paths = ll.get_paths(baselines, samples, no_points)
        self.all_paths = paths

        # Initialize the deep learning model on which the interpreter has to run
        self.model = nn_model
        # Initialize number of baselines per sample.
        self.no_baselines = no_baselines

    def attributions(self):
        """
        Compute attributions using the model and data initialized before.
        :return: attributions for all the data points.
        """
        # Compute gradients for the model using data for paths and other optional data if needed.
        gradients = self.model.gradients(self.all_paths)
        attributions = [np.trapz(ii, jj, axis=0) for ii, jj in zip(gradients, self.all_paths)]
        attributions = np.array(attributions)

        # Average attributions if number of baselines for each sample > 1.
        if self.no_baselines > 1:
            attributions_shape = list(attributions.shape)
            num_samples = attributions_shape[0] / self.no_baselines
            new_shape = [0] * (len(attributions_shape) + 1)
            new_shape[0] = num_samples
            new_shape[1] = self.no_baselines
            for i in range(2, len(new_shape)):
                new_shape[i] = attributions_shape[i - 1]
            new_shape = tuple(np.array(new_shape, dtype=int))
            attributions = attributions.reshape(new_shape)
            attributions = np.median(attributions, axis=1)
        return attributions
