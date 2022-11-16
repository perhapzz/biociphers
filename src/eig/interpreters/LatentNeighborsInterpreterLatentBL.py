"""
LatentNeighborsInterpreterLatentBL.py

Class for Latent neighbor Interpreter (H-N-IG: Hidden feature space neighbors Integrated Gradients). This class gives
attributes for features using neighbors path in the latent feature space produced by the autoencoder. Baselines are
assumed to be computed in the latent space of the autoencoder.

Defines:
+ class LatentNeighborsInterpreterLatentBL
"""
import numpy as np
from eig.interpreters.Interpreter import Interpreter
from eig.baselines.BaselineLatent import BaselineLatent
from eig.paths.PathGeneratorLatentNeighbors import LatentNeighborsPathGenerator

BASELINES_ZERO = "zero"
BASELINES_ENCODER_ZERO = "encoded_zero"
BASELINES_K_MEANS = "k-means"
BASELINES_MEDIAN = "median"
BASELINES_RANDOM = "random"
BASELINES_CLOSE = "close"
ALL_BASELINES = [BASELINES_ZERO, BASELINES_ENCODER_ZERO, BASELINES_K_MEANS, BASELINES_MEDIAN,
                 BASELINES_RANDOM, BASELINES_CLOSE]


class LatentNeighborsInterpreterLatentBL(Interpreter):
    """
    This is the Interpreter for the neighbors path in the latent space. Baseline points are computed in the latent
    space.
    """

    def __init__(self, baselines_data, samples_data, encoder_data=None, decoder_data=None, other_data_nn=None,
                 nn_model=None, autoencoder=None, baseline_type=BASELINES_MEDIAN, no_baselines=3, no_points=250,
                 neighbor_obj=None):
        """
        Initialize variables for the neighbors interpreter.
        :param baselines_data: np.array(), data from where baselines are to be chosen.
        :param samples_data: np.array(), sample data
        :param encoder_data: list, if encoder has additional placeholder data, put them here.
        :param decoder_data: list, if decoder has additional placeholder data, put them here.
        :param other_data_nn: list, if the model has additional placeholders, they can put put here.
        :param nn_model: DeepNetworkModel, Object of the DeepNetworkModel class
        :param baseline_type: str, type of baseline (zero, encoded-zero, k-means, median, random, closest)
        :param no_baselines: int, number of baselines per sample
        :param no_points: int, number of data points to be computed between the baseline and the sample.
        :param neighbor_obj: NeighborsPathGenerator obj, Neighbors object with populated nearest neighbors graph
        """

        assert baseline_type in ALL_BASELINES, "baseline type not available, choose from available options {}".format(
            ALL_BASELINES)
        assert nn_model is not None, "Provide deep learning model for interpretation. "
        assert autoencoder is not None, "Provide autoencoder model for generating latent path. "

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
            base_points = baseline_obj.get_baseline(baselines_data, no_baselines, baseline_type,
                                                    autoencoder=autoencoder)
            baselines = np.repeat(base_points, len(samples_data[0]), axis=0)

        # If more than one baselines per sample, repeat the sample array no_baselines times.
        samples = np.repeat(samples_data[0], no_baselines, axis=0)

        # Get paths function for the latent neighbors interpreter.
        paths = []
        for i in range(len(samples)):
            baseline = baselines[i]
            sample = samples[i]
            if encoder_data is not None and decoder_data is not None:
                baselines_data_encoder, samples_data_encoder = encoder_data
                baselines_data_decoder, samples_data_decoder = decoder_data
                decoder_val = samples_data_decoder
                all_baselines = [baseline, baselines_data_encoder, baselines_data_decoder]
                all_samples = [sample, samples_data_encoder, samples_data_decoder]
            elif encoder_data is not None:
                baselines_data_encoder, samples_data_encoder = encoder_data
                decoder_val = None
                all_baselines = [baseline, baselines_data_encoder, None]
                all_samples = [sample, samples_data_encoder, None]
            elif decoder_data is not None:
                baselines_data_decoder, samples_data_decoder = decoder_data
                decoder_val = samples_data_decoder
                all_baselines = [baseline, None, baselines_data_decoder]
                all_samples = [sample, None, samples_data_decoder]
            else:
                decoder_val = None
                all_baselines = [baseline, None, None]
                all_samples = [sample, None, None]
            path = neighbor_obj.get_path_decode(all_baselines, all_samples, decoder_val, no_points)
            paths.append(path)
        paths = np.array(paths)
        self.all_paths.append(paths)
        # Initialize the deep learning model on which the interpreter has to run
        self.model = nn_model
        # If more than one input placeholders are needed, provide them here.
        self.other_data = other_data_nn
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

    @staticmethod
    def initialize_neighbors_path(neighbors_data, encoder, decoder, encoder_data=None, decoder_data=None):
        """
        initalize the neighbors graph.
        :param neighbors_data: np.array(), Data to populate the neighbors path.
        :param encoder: LatentModel.encoder function to encode from original feature space to latent space.
        :param decoder: LatentModel.decoder function to decode from latent space to original feature space.
        :param encoder_data: list, if encoder has additional placeholder data, put them here.
        :param decoder_data: list, if decoder has additional placeholder data, put them here.
        :return: function to compute neighbors path
        """
        print("Populate nearest neighbors graph...")
        print("neighbors_data: ", neighbors_data.shape)
        neighbor_obj = LatentNeighborsPathGenerator(data=neighbors_data,
                                                    encoder=encoder,
                                                    decoder=decoder,
                                                    encoder_data=encoder_data,
                                                    decoder_data=decoder_data)
        print("Done.")
        return neighbor_obj
