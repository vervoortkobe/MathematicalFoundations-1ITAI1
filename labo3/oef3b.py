# Voer deze cel eerst uit om alle benodigde libraries te importeren

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

np.set_printoptions(precision=3, suppress=True)

# Laad MNIST
print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

class Layer:
    def __init__(self, input_size, output_size, activation='none'):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)
        self.activation = activation

    def relu(self, X):
        return np.maximum(0, X)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def forward(self, X):
       res = X @ self.W + self.b
       if self.activation == 'relu':
           return self.relu(res)
       elif self.activation == 'sigmoid':
           return self.sigmoid(res)
       else:
           return res

    def __repr__(self):
        return f"Layer(input_size={self.W.shape[0]}, output_size={self.W.shape[1]})"

# Test je implementatie:
# Verschillende lagen
linear_layer  = Layer(784, 128, 'none')
relu_layer    = Layer(784, 128, 'relu')
sigmoid_layer = Layer(784, 128, 'sigmoid')

X = np.random.randn(4, 784)
print("Linear: ", linear_layer.forward(X))
print("ReLU:   ", relu_layer.forward(X))
print("Sigmoid:", sigmoid_layer.forward(X))
print(linear_layer)  # Layer(input_size=3, output_size=2, activation='none')
print(f"Output shape: {linear_layer.forward(X[:5])}")  # 5 MNIST afbeeldingen
print(relu_layer)    # Layer(input_size=3, output_size=2, activation='relu'
print(f"Output shape: {relu_layer.forward(X[:5])}")  # 5 MNIST afbeeldingen
print(sigmoid_layer) # Layer(input_size=3, output_size=2, activation='sigmoid'
print(f"Output shape: {sigmoid_layer.forward(X[:5])}")  # 5 MNIST afbeeldingen
