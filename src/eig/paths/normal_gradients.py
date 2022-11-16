"""
normal_gradients.py

This file contains function to compute gradients given samples.

Defines:
+ function normal_gradients(gradient_function, samples)

"""


def normal_gradients(gradient_function, samples):
    """
    Given the samples and the gradient function, compute the gradient.
    :param gradient_function: Gradient function to use
    :param samples: Input samples
    :return:
    """
    gradients = gradient_function(samples)
    attributions = samples*gradients
    # return the attributions
    return attributions
