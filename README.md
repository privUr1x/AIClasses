![nn (Personalizado)](https://github.com/user-attachments/assets/413e1717-ef16-4a50-b758-0415478685a2)
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
from easyAI.arquitectures import MLP
from easyAI.layers import Dense
from easyAI.datasets import load, mnist

# Load the data
train_data, train_labels = load(mnist)

# Define the model architecture
model = MLP([
    Dense(128),
    Dense(40),
    Dense(1)
])

# Train the model
history = model.fit(train_data, train_labels, epochs=1000, learning_rate=0.01)

# Evaluate the model
accuracy = history.accuracy
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
