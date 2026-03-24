# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 1b: 3x3 matrix met nullen

matrix_nullen = np.zeros((3, 3))
print("3x3 matrix met nullen:")
print(matrix_nullen)
print(f"Shape: {matrix_nullen.shape}")
