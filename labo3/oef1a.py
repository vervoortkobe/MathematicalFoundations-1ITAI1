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

# 1. Een 3×4 matrix gevuld met nullen
print(np.zeros((3, 4)))

# 2. Een 2×5 matrix gevuld met enen
print(np.ones((2, 5)))

# 3. Een 4×4 identiteitsmatrix
print(np.identity(4))

# 4. Een 3×3 diagonaalmatrix met waarden [2, 5, 7] op de diagonaal
print(np.diag([2, 5, 7]))

# 5. Een 3×4 matrix met random waarden uit een normale verdeling
print(np.random.randn(3, 4))
