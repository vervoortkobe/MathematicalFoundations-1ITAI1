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
        self.activation = activation
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'softmax':
            return self.softmax(z)
        else:
            return z

    def __repr__(self):
        return f"Layer(input_size={self.W.shape[0]}, output_size={self.W.shape[1]})"

network = [
    Layer(784, 256, activation='relu'),
    Layer(256, 128, activation='relu'),
    Layer(128, 10, activation='softmax')
]

import time

# Neem eerste layer uit je netwerk (784 -> 256)
layer = network[0]  # ReLU laag
X_subset = X[:1000]  # 1000 afbeeldingen

print("=== EEN VOOR EEN (for loop) ===")
start = time.time()
for i in range(1000):
    single_output = layer.forward(X_subset[i:i+1])  # (1,784) -> (1,256)
end_single = time.time()
single_time = end_single - start
print(f"Tijd: {single_time:.4f} seconden")

print("\n=== ALS EEN BATCH VAN 100 ===")
batch_size = 100
n_batches = len(X_subset) // batch_size  # 10 batches
start = time.time()
for i in range(n_batches):
    batch_output = layer.forward(X_subset[i*batch_size:(i+1)*batch_size])
end_batch = time.time()
batch_time = end_batch - start
print(f"Tijd: {batch_time:.4f} seconden")

print("\n=== VERGELIJKING ===")
print(f"Enkele afbeeldingen: {single_time:.4f}s ({1000*1000/len(X_subset):.0f}x langzamer)")
print(f"Batch van 100:       {batch_time:.4f}s")
print(f"Versnelling:          {single_time/batch_time:.1f}x sneller!")

