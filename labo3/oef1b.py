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

v = np.arange(12)
print(f"Originele vector: {v}\n")

# 1. Een 3×4 matrix
print(np.reshape(v, (3, 4)))

# 2. Een 4×3 matrix
print(np.reshape(v, (4, 3)))

# 3. Een 2×6 matrix
print(np.reshape(v, (2, 6)))

# 4. Een 2×2×3 tensor (3D)
print(np.reshape(v, (2, 2, 3)))
