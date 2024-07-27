#!/usr/bin/python3


"""
Module representing a set of commonly used loss functions.
"""

from typing import Optional, Union

# from easyAI.clsstools.Verifiers import verify_type, verify_components_type
from math import log


def mean_squared_error(
    tags: Union[int, float], pred: Union[int, float]
) -> float:
    """
    Calcula el error cuadrático medio entre las etiquetas verdaderas y las predicciones.

    Args:

    Returns:
    """
    return sum([(t - p) ** 2 for t, p in zip(tags, pred)]) / len(tags)


def binary_cross_entropy(y_true: Union[int, float], y_pred: Union[int, float]) -> float:
    """
    Calcula la entropía cruzada binaria entre las etiquetas verdaderas y las predicciones.

    Args:
        y_true (float): Valor verdadero (etiqueta binaria).
        y_pred (float): Valor predicho (probabilidad estimada).

    Returns:
        float: Entropía cruzada binaria entre y_true e y_pred.
    """
    epsilon = 1e-15  # para evitar log(0), se añade un pequeño valor epsilon
    y_pred = max(epsilon, min(y_pred, 1 - epsilon))
    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))


def categorical_cross_entropy(y_true: list, y_pred: list) -> float:
    """
    Calcula la entropía cruzada categórica entre las etiquetas verdaderas y las predicciones.

    Args:
        y_true (list): Lista de valores verdaderos codificados como one-hot.
        y_pred (list): Lista de probabilidades predichas para cada clase.

    Returns:
        float: Entropía cruzada categórica entre y_true e y_pred.
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
    Calcula la función de pérdida de Huber entre las etiquetas verdaderas y las predicciones.

    Args:
        y_true (float): Valor verdadero.
        y_pred (float): Valor predicho.
        delta (float, optional): Parámetro de umbral. Por defecto es 1.0.

    Returns:
        float: Pérdida de Huber entre y_true e y_pred.
    """
    error = abs(y_true - y_pred)
    if error <= delta:
        return 0.5 * error**2
    else:
        return delta * (error - 0.5 * delta)


def kl_divergence(p: list, q: list) -> float:
    """
    Calcula la divergencia de Kullback-Leibler entre dos distribuciones de probabilidad p y q.

    Args:
        p (list): Distribución de probabilidad p.
        q (list): Distribución de probabilidad q.

    Returns:
        float: Divergencia de Kullback-Leibler entre p y q.
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
