# easyAI

## Description

`easyAI` is a Python library designed to simplify the use of deep learning models, alongside integrated tools for data processing, loading, and visualization. This personal project aims to maintain a simple yet useful structure, avoiding the use of analytical sub-libraries like scikit-learn, and providing a friendly API for implementing artificial intelligence models.

> [!NOTE]
> Still under development!

## Installation

To install `easyAI`, first clone the repository and then install the dependencies:

```bash
git clone <REPOSITORY_URL>
cd easyAI
pip install -r requirements.txt
```

### Usage

Below is an example of using `easyAI` to train a deep learning model on a simple XOR logic problem:

```Python
from easyAI.core import Model
from easyAI.arquitectures import Perceptron
from easyAI.datasets import load_xor_data

# Load the data
train_data, train_labels = load_xor_data()

# Define the model architecture
model = Model(Perceptron(input_size=2, output_size=1))

# Train the model
model.train(train_data, train_labels, epochs=1000, learning_rate=0.01)

# Evaluate the model
accuracy = model.evaluate(train_data, train_labels)
print(f'Accuracy: {accuracy}')
````

## Contributing

Contributions are welcome. If you would like to contribute, please follow these steps:

Fork the repository.
Create a branch (`git checkout -b feature-new`).
Make your changes and commit them (`git commit -m 'Add new feature'`).
Push to the branch (`git push origin feature-new`).
Open a Pull Request.

Thanks.
