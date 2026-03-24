# Imports en data laden
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

def cosine_similarity(u, v):
    """Bereken de cosine similarity tussen twee vectoren."""
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Opdracht 2b
img_matrix = X[0].reshape(28, 28)

# Row-major (standaard)
row_major = img_matrix.flatten()

# Column-major
col_major = img_matrix.flatten(order='F')

fig, axes = plt.subplots(3, 1, figsize=(12, 6))

axes[0].imshow(img_matrix, cmap='gray')
axes[0].set_title(f'Originele matrix (label: {y[0]})')
axes[0].axis('off')

axes[1].imshow(row_major.reshape(1, -1), cmap='gray', aspect='auto')
axes[1].set_title('Row-major: rijen achter elkaar')
axes[1].set_yticks([])

axes[2].imshow(col_major.reshape(1, -1), cmap='gray', aspect='auto')
axes[2].set_title('Column-major: kolommen achter elkaar')
axes[2].set_yticks([])

plt.tight_layout()
plt.show()

print("Bij row-major zie je horizontale strepen van het cijfer.")
print("Bij column-major zie je verticale strepen van het cijfer.")