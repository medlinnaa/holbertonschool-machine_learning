#!/usr/bin/env python3
"""Module that calculates L2 regularized cost"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the original cost (without regularization)
        model: Keras model with L2 regularization

    Returns:
        tensor containing the total cost for each layer
    """
    # get a list of L2 losses(one per layer)
    l2_losses = model.losses

    # Add base cost to each L2 loss
    total_costs = [cost + loss for loss in l2_losses]

    # return as tensor
    return tf.stack(total_costs)
