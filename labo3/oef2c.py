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

# Verifieer de associativiteit van matrixvermenigvuldiging: (A @ B) @ C = A @ (B @ C).
# Maak drie willekeurige matrices A (2×3), B (3×4) en C (4×2) en toon dat beide zijden gelijk zijn.
A = np.random.randn(2, 3)
B = np.random.randn(3, 4)
C = np.random.randn(4, 2)

res1 = (A @ B) @ C
res2 = A @ (B @ C)

print(res1)
print(res2)
print("Shapes:", left.shape, right.shape)
print("Zijn ze gelijk?", np.allclose(left, right))
