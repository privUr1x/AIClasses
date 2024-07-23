from typing import Any

# Default config for a personalized nn
config: dict[str, Any] = {
    'epochs': 100,
    'learning_rate': 0.01,
    'batch_size': 32,
    'optimizer': 'sgd',
    'seed': 42,
    'activation': 'sigmoid',
    'loss': 'mse',
}

...
