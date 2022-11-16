"""
DeepNetworkModel.py

DeepNetworkModel class that defines the gradient function to compute gradients from different paths for EIG.
Defines:
+ class DeepNetworkModel(object)
"""
import numpy as np
import tensorflow as tf


class DeepNetworkModel(object):
    """
    This is the DeepNetworkModel class that defines the gradient function to compute gradients for a tensorflow
    deep learning model.
    """
    def __init__(self, model, output_index, batch_size):
        """
        Initialize the tensorflow session and the tensor inputs/outputs.
        :param session: tensorflow session that contains the tensorflow graph for the model
        :param tensor_ops: The input and output tensors for the tensorflow model

        """
        self.model = model
        self.output_index = output_index
        self.batch_size = batch_size

    def predict_output(self, x):
        """
        """
        preds = self.model(x)
        preds = preds[:, self.output_index]

        return preds

    def gradients_input(self, x):
        """
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = self.predict_output(x)

        grads = tape.gradient(preds, x)
        return grads

    def gradients(self, paths_inputs):
        """
        Compute gradients for the points in the paths_input.
        """
        samples, num_steps, features = paths_inputs.shape
        #print("samples, num_steps, features: ", samples, num_steps, features)
        paths_input_flat = paths_inputs.reshape((samples*num_steps, features))
        #print("paths_input_flat: ", paths_input_flat.shape, self.batch_size)
        paths_input_flat = tf.convert_to_tensor(paths_input_flat)
        paths_tf = tf.data.Dataset.from_tensor_slices(paths_input_flat).batch(self.batch_size)
        reshaped_gradients = []
        cnt = 0
        for path in paths_tf:
            gradients_all = self.gradients_input(path)
            reshaped_gradients.append(gradients_all)
            #print(cnt, gradients_all.shape)
            cnt += 1

        reshaped_gradients = tf.concat(reshaped_gradients, 0)
        reshaped_gradients = tf.reshape(reshaped_gradients, (samples, num_steps, features))
        #print("reshaped_gradients: ", reshaped_gradients[0], reshaped_gradients.shape)
        return reshaped_gradients
