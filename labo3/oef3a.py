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
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

    def forward(self, X):
        return X @ self.W + self.b

    def __repr__(self):
        return f"Layer(input_size={self.W.shape[0]}, output_size={self.W.shape[1]})"

# Test je implementatie:
layer = Layer(784, 128)
print(layer)
output = layer.forward(X[:5])  # 5 MNIST afbeeldingen
print(f"Output shape: {output.shape}")
