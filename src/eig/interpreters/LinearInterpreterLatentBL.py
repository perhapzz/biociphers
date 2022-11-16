"""
LinearInterpreterLatentBL.py

Class for Linear Interpreter (O-L-IG: Original feature space Linear Integrated Gradients). This class gives attributes
for features using linear path in the original feature space. The baseline is computed in the latent space of the
autoencoder.

Defines:
+ class LinearInterpreterLatentBL

"""
import numpy as np
from eig.interpreters.Interpreter import Interpreter
from eig.baselines.Baseline import Baseline
from eig.baselines.BaselineLatent import BaselineLatent
from eig.paths.PathGeneratorLinear import LinearPathGenerator

BASELINES_ENCODER_ZERO = "encoded_zero"
BASELINES_K_MEANS = "k-means"
BASELINES_MEDIAN = "median"
BASELINES_RANDOM = "random"
BASELINES_CLOSE = "close"
ALL_BASELINES = [BASELINES_ENCODER_ZERO, BASELINES_K_MEANS, BASELINES_MEDIAN,
                 BASELINES_RANDOM, BASELINES_CLOSE]


class LinearInterpreterLatentBL(Interpreter):
    """
    This is the Interpreter for the linear path in the original feature space.
    """

    def __init__(self, baselines_data, samples_data, other_data=None, nn_model=None, baseline_type=BASELINES_MEDIAN,
                 no_baselines=3, no_points=250, autoencoder=None):
        """
        Initialize variables for the linear interpreter.
        :param baselines_data: list, [baseline_data: np.array(), data from where baselines are to be chosen.
                                encoder_data: np.array(), any additional data to be fed to the encoder
                                decoder_data: np.array(), any additional data to be fed to the decoder]
        :param samples_data: list, [sample data: np.array(), sample data.
                                encoder_data: np.array(), any additional data to be fed to the encoder
                                decoder_data: np.array(), any additional data to be fed to the decoder]
        :param other_data: list, if the nn model has additional placeholders, they can be put here.
        :param nn_model: DeepNetworkModel, Object of the DeepNetworkModel class
        :param baseline_type: str, type of baseline (zero, encoded-zero, k-means, median, random, closest)
        :param no_baselines: int, number of baselines per sample
        :param no_points: int, number of data points to be computed between the baseline and the sample.
        :param autoencoder: LatentModel object, if baselines are needed in the latent space
        """

        assert baseline_type in ALL_BASELINES, "baseline type not available, choose from available options {}".format(
            ALL_BASELINES)
        assert nn_model is not None, "Provide deep learning model for interpretation. "

        if autoencoder is None:
            print("please provide an autoencoder to compute baseline in hidden space. ")
            exit(1)

        # Initialize linear path generator object
        ll = LinearPathGenerator()

        # compute paths for sample and baseline data
        self.all_paths = []
        baseline_obj = BaselineLatent()
        # If closest baseline, we need both sample and baseline data to compute the baselines
        if baseline_type == BASELINES_CLOSE:
            base_points = baseline_obj.get_closest_baselines(baselines_data, no_baselines, samples_data,
                                                             autoencoder=autoencoder)
            baselines, baseline_ids = base_points
        # If not closest, only baseline data suffices to compute the baselines
        else:
            base_points = baseline_obj.get_baseline(baselines_data,
                                                    no_baselines,
                                                    baseline_type,
                                                    autoencoder=autoencoder)
            baselines = np.repeat(base_points, len(samples_data[0]), axis=0)

        # If more than one baselines per sample, repeat the sample array no_baselines times.
        samples = np.repeat(samples_data[0], no_baselines, axis=0)

        # Compute paths using the baselines and samples, no_points gives number of data points to compute between
        # baseline and sample.
        paths = ll.get_paths(baselines, samples, no_points)

        self.all_paths.append(paths)

        # Initialize the deep learning model on which the interpreter has to run
        self.model = nn_model
        # If more than one input placeholders are needed, provide them here.
        self.other_data = other_data
        # Initialize number of baselines per sample.
        self.no_baselines = no_baselines

    def attributions(self):
        """
        Compute attributions using the model and data initialized before.
        :return: attributions for all the data points.
        """
        # Compute gradients for the model using data for paths and other optional data if needed.
        gradients = self.model.gradients(self.all_paths, self.other_data)
        all_attributions = []
        for gradient, path in zip(gradients, self.all_paths):
            attributions = [np.trapz(ii, jj, axis=0) for ii, jj in zip(gradient, path)]
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
                attributions = np.mean(attributions, axis=1)

            all_attributions.append(attributions)

        return all_attributions

    def attributions_subgroups(self, subgroups):
        """
        Get attributions for a subgroups of features.
        :param subgroups: list, groups of features that belong to a group.
            two columns array where the first column specifies feature index and the second
            column indicates the feature subgroup
        :return: attributions for feature subgroups
        """
        all_attributions = self.attributions()
        all_attribution_subgroups = []
        # For all samples, group together the attributions for all features that belong in a group.
        for attributions in all_attributions:
            attributions_subgroup = np.array(
                [np.abs(
                    np.sum(attributions[:, ii], axis=1)
                )
                    for ii in subgroups]).transpose()
            all_attribution_subgroups.append(attributions_subgroup)

        return all_attribution_subgroups
