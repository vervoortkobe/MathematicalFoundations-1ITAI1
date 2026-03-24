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

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. A @ B
# [[19 22]
#  [43 50]]
res1 = A @ B
print(res1)
# 2. B @ A
# [[23 34]
#  [31 46]]
res2 = B @ A
print(res2)
# 3. Verifieer dat A @ B ≠ B @ A
print(res1 == res2)
