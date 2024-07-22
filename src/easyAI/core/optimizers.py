from typing import Any, List, Union
from abc import ABC, abstractmethod

"""Module representing optimizers used for training models."""


class Optimizer(ABC):
    """Class representing the abstraction to an optimizer."""

    def __init__(self, learning_rate: Union[int, float]):
        self._learning_rate: Union[int, float] = learning_rate

    def __call__(self) -> None: ...

class SGD(Optimizer):
    """Class representing Stochastic Gradient Descent."""

    def __init__(
        self,
        learning_rate: Union[int, float],
    ):
        """
        Inicializa el optimizador de descenso de gradiente estocástico.

        :param learning_rate: Tasa de aprendizaje para las actualizaciones de los parámetros.
        :param epochs: Número de épocas de entrenamiento.
        """
        super().__init__(learning_rate=learning_rate)

    def fit(
        self,
        X: list[Union[int, float]],
        Y: list[Union[int, float]],
        epochs,
        verbose: bool,
    ):
        """
        Ajusta los parámetros del modelo utilizando descenso de gradiente estocástico.

        :param X: Lista de listas con características de entrenamiento.
        :param y: Lista con valores reales.
        :param model: Instancia del modelo que contiene los métodos predict y update_parameters.
        """
        m = len(y)  # Número de ejemplos de entrenamiento

        for epoch in range(self.epochs):
            for i in range(m):
                # Seleccionar un solo ejemplo
                xi = X[i]
                yi = y[i]

                # Predicción del modelo para el ejemplo actual
                predictions = model.predict(xi)

                # Calcular el error
                error = predictions - yi

                # Calcular el gradiente (derivada del error con respecto a los parámetros)
                gradients = model.compute_gradients(xi, error)

                # Actualizar los parámetros del modelo
                model.update_parameters(self.learning_rate, gradients)

            # Opcional: imprimir el progreso cada ciertos epochs
            if (epoch + 1) % 10 == 0:
                loss = self.calculate_loss(X, y, model)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def calculate_loss(self, X, y, model):
        """
        Calcula la pérdida MSE del modelo sobre todo el conjunto de datos.

        :param X: Lista de listas con características de entrenamiento.
        :param y: Lista con valores reales.
        :param model: Instancia del modelo.
        :return: Pérdida MSE.
        """
        total_loss = 0.0
        m = len(y)
        for i in range(m):
            predictions = model.predict(X[i])
            error = predictions - y[i]
            total_loss += error**2
        return total_loss / (2 * m)


class Adam(Optimizer):
    """Class representing Adaptative Moment Estimation."""

    def __init__(self, learning_rate: Union[int, float]):
        super().__init__(learning_rate)


class PLR(Optimizer):
    """Class representing Perceptron Learning Rule"""

    def __init__(self, learning_rate: Union[int, float]):
        super().__init__(learning_rate)

    def __call__(
        self,
        P,  # The perceptron class itself
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        *,
        verbose: bool,
        epochs: int
    ) -> Any:
        for epoch in range(epochs):
            for i in range(len(y)):
                # Narrowing down y for X
                eX = X[i * P.n : (i + 1) * P.n]
                ey = y[i]

                z = P.__call__(eX)[0]

                # Updating parameters
                if z != ey:
                    for n in P.output:  # [n0, n1, ..., nn]
                        for i in range(n.n):  # [w-1, w1, ..., wn]
                            n.w[i] += self._learning_rate * (ey - z) * eX[i]
                        n.b += P._lr * (ey - z)

                if verbose:
                    print(f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}")

        return None


optimizers_map: dict = {
    "sgd": SGD,
    "adam": Adam,
    "plr": PLR,
}
