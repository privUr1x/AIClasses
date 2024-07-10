#!/usr/bin/python3

"""
Module representing a set of commonly used activation functions.
"""

from typing import Optional, Union, List
from easyAI.clsstools.Verifiers import verify_type
from math import log, exp

def step(x: Union[int, float], threshold: Union[int, float] = 0) -> int:
    """
    Heaviside step function.

    Args:
        x (Union[int, float, npnum]): Input value.
        threshold (Union[int, float]): Threshold value.

    Returns:
        int: 1 if x >= threshold, else 0.
    """
    x = verify_type(x, (int, float))
    threshold = verify_type(threshold, (int, float))
    return 1 if x >= threshold else 0


def sigmoid(x: Union[int, float]) -> float:
    """
    Sigmoid function.

    Args:
        x (Union[int, float, npnum]): Input value.

    Returns:
        float: Sigmoid of x.
    """
    x = verify_type(x, (int, float))
    return 1 / (1 + exp(-x))


def relu(x: Union[int, float]) -> float:
    """
    Rectified Linear Unit function.

    Args:
        x (Union[int, float, npnum]): Input value.

    Returns:
        float: ReLU of x.
    """
    x = verify_type(x, (int, float))
    return max(0, x)


def leaky_relu(x: Union[int, float]) -> float:
    """
    Leaky Rectified Linear Unit function.

    Args:
        x (Union[int, float, npnum]): Input value.

    Returns:
        float: Leaky ReLU of x.
    """
    x = verify_type(x, (int, float))
    return x if x >= 0 else x / 10


def tanh(x: Union[int, float]) -> float:
    """
    Hyperbolic Tangent function.

    Args:
        x (Union[int, float, npnum]): Input value.

    Returns:
        float: Tanh of x.
    """
    x = verify_type(x, (int, float))
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def softmax(x: List[float], n: Optional[int] = None) -> List[float]:
    """
    Softmax function.

    Args:
        x (Union[List[float], ndarray]): Input vector.
        n (int, optional): Index for a specific value. Defaults to None.

    Returns:
        List[float]: Softmax distribution.
    """
    x = verify_type(x, list)
    e_x = [exp(i) for i in x]
    sum_e_x = sum(e_x)
    distr = [i / sum_e_x for i in e_x]

    if n is not None:
        verify_type(n, int)
        return [distr[n]]

    return distr


def prelu(x: Union[int, float], lp: Union[int, float]) -> float:
    """
    Parametric Rectified Linear Unit function.

    Args:
        x (Union[int, float, npnum]): Input value.
        lp (Union[int, float, npnum]): Learned parameter.

    Returns:
        float: PReLU of x.
    """
    x = verify_type(x, (int, float))
    lp = verify_type(lp, (int, float))
    return max(lp * x, x)


def elu(x: Union[int, float], alpha: Union[int, float]) -> float:
    """
    Exponential Linear Unit function.

    Args:
        x (Union[int, float, npnum]): Input value.
        alpha (Union[int, float, npnum]): Scaling parameter.

    Returns:
        float: ELU of x.
    """
    x = verify_type(x, (int, float))
    alpha = verify_type(alpha, (int, float))
    return x if x > 0 else alpha * (exp(x) - 1)


def softplus(x: Union[int, float]) -> float:
    """
    Softplus function.

    Args:
        x (Union[int, float, npnum]): Input value.

    Returns:
        float: Softplus of x.
    """
    x = verify_type(x, (int, float))
    return log(1 + exp(x))

activation_map: dict = {
    "step": step, 
    "sigmoid": sigmoid,
    "relu": relu, 
    "leaky_relu": leaky_relu, 
    "tanh": tanh, 
    "softmax": softmax, 
    "prelu": prelu, 
    "elu": elu, 
    "softplus": softplus
}

