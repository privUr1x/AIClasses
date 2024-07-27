#!/usr/bin/python3

"""
Module representing a set of commonly used loss functions.
"""

from typing import Optional, Union
from math import log


def mean_squared_error(
    tags: Union[int, float], pred: Union[int, float]
) -> float:
    """
    Calculates the mean squared error between true labels and predictions.

    Args:
        tags (Union[int, float]): True labels.
        pred (Union[int, float]): Predicted values.

    Returns:
        float: Mean squared error between tags and pred.
    """
    return sum([(t - p) ** 2 for t, p in zip(tags, pred)]) / len(tags)


def binary_cross_entropy(y_true: Union[int, float], y_pred: Union[int, float]) -> float:
    """
    Calculates binary cross-entropy between true labels and predictions.

    Args:
        y_true (float): True value (binary label).
        y_pred (float): Predicted value (estimated probability).

    Returns:
        float: Binary cross-entropy between y_true and y_pred.
    """
    epsilon = 1e-15  # To avoid log(0), add a small epsilon value
    y_pred = max(epsilon, min(y_pred, 1 - epsilon))
    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))


def categorical_cross_entropy(y_true: list, y_pred: list) -> float:
    """
    Calculates categorical cross-entropy between true labels and predictions.

    Args:
        y_true (list): List of true values encoded as one-hot.
        y_pred (list): List of predicted probabilities for each class.

    Returns:
        float: Categorical cross-entropy between y_true and y_pred.
    """
    loss = 0.0
    for true, pred in zip(y_true, y_pred):
        loss -= true * log(pred)
    return loss


def huber_loss(
    y_true: Union[int, float],
    y_pred: Union[int, float],
    delta: Optional[Union[int, float]] = 1.0,
) -> float:
    """
    Calculates the Huber loss between true labels and predictions.

    Args:
        y_true (float): True value.
        y_pred (float): Predicted value.
        delta (float, optional): Threshold parameter. Defaults to 1.0.

    Returns:
        float: Huber loss between y_true and y_pred.
    """
    error = abs(y_true - y_pred)
    if error <= delta:
        return 0.5 * error**2
    else:
        return delta * (error - 0.5 * delta)


def kl_divergence(p: list, q: list) -> float:
    """
    Calculates the Kullback-Leibler divergence between two probability distributions p and q.

    Args:
        p (list): Probability distribution p.
        q (list): Probability distribution q.

    Returns:
        float: Kullback-Leibler divergence between p and q.
    """
    divergence = 0.0
    for i in range(len(p)):
        if p[i] != 0:
            divergence += p[i] * log(p[i] / q[i])
    return divergence


loss_map: dict = {
    "mse": mean_squared_error,
    "cross-entropy": categorical_cross_entropy,
    "binary-cross-entropy": binary_cross_entropy,
    "huber": huber_loss,
    "kl-divergence": kl_divergence,
}

