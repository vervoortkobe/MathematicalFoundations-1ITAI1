# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 1d: 2x5 matrix met willekeurige gehele getallen

random_matrix = np.random.randint(0, 101, size=(2, 5))
print("2x5 matrix met willekeurige getallen:")
print(random_matrix)
print(f"Shape: {random_matrix.shape}")
print(f"Gemiddelde: {random_matrix.mean():.2f}")
