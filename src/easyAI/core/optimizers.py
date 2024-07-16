from typing import List, Union

"""Module representing optimizers used for training models."""


def stochastc_grdnt_descnt(Model, X, y, verbose: bool = False) -> None:
    pass


def adam(Model, X, y, verbose: bool = False) -> None:
    pass

def perceptron_learning_rule(
    P, # The perceptron class itself
    X: List[Union[int, float]],
    y: List[Union[int, float]],
    verbose: bool = False,
):
    for epoch in range(len(y)):
        # Narrowing down y for X
        eX = X[epoch * P._n : (epoch + 1) * P._n]
        ey = y[epoch]

        z = P.__call__(eX)[0]

        # Updating parameters
        if z != ey:
            for n in P.output:  # [n0, n1, ..., nn]
                for i in range(n.n):  # [w-1, w1, ..., wn]
                    n.w[i] += P._lr * (ey - z) * eX[i]
                n.b += P._lr * (ey - z)

        if verbose:
            print(f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}")

    return None


optimizers_map: dict = {
    "sgd": stochastc_grdnt_descnt,
    "adam": adam,
    "plr": perceptron_learning_rule,
}
