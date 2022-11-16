"""
LatentModel.py

Class for autoencoder/variational-autoencoder model. Provides access to encoder and decoder of the
autoencoder/variational-autoencoder. encoder function allows encoding from original feature space to
latent space and decoder function allows decoding from latent space to original feature space.

Defines:
+ class LatentModel
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants


class LatentModel:
    """
        Class for Autoencoder/variational-autoencoder for latent modelling with EIG.

    """
    def __init__(self, session_encoder, session_decoder, tensor_ops):
        """
        initialize the tensorflow sessions for the encoder and the decoder and additionally the input and output
        operations for the encoder and the decoder.
        :param session_encoder: tensorflow session for the encoder model.
        :param session_decoder: tensorflow session for the decoder model.
        :param tensor_ops: list, tensorflow operations and placeholders for providing input placeholder and output tensor.
        """
        assert len(tensor_ops) == 4, "# of ops incorrect, need input_op, decoder_input_op, encoder_op and decoder_op."
        self.input_op, self.encoder_op, self.decoder_input_op, self.decoder_op = tensor_ops

        self.sess_enc = session_encoder
        self.sess_dec = session_decoder

    def encoder(self, data_list):
        """
        Encode data given using the encoder loaded in session for the encoder model.
        :param data_list: list, one or more data arrays for the input placeholders for the encoder.
        :return: data embedded in the last latent space of the encoder.
        """
        # Input features placeholder
        if isinstance(self.input_op, list) and isinstance(data_list, list):
            assert len(self.input_op) == len(data_list)
            input_dict = dict(zip(self.input_op, data_list))
        else:
            input_dict = {self.input_op: data_list}

        # Run network to compute latent features and reconstructed output for the autoencoder
        embedding = self.sess_enc.run(self.encoder_op, feed_dict=input_dict)
        return embedding

    def decoder(self, data_list):
        """
        Decode data given using the decoder loaded in session for the decoder model.
        :param data_list: list, one or more data arrays for the input placeholders for the decoder.
        :return: Decoded data in the original feature space.
        """
        # Input features placeholder
        if isinstance(self.decoder_input_op, list) and isinstance(data_list, list):
            assert len(self.decoder_input_op) == len(data_list)
            encoder_dict = dict(zip(self.decoder_input_op, data_list))

        else:
            encoder_dict = {self.decoder_input_op: data_list}

        # Run network to compute latent features and reconstructed output for the autoencoder
        reconstructed_output = self.sess_dec.run(self.decoder_op, feed_dict=encoder_dict)
        return reconstructed_output
