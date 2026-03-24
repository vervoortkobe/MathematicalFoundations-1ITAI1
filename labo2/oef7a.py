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

# Opdracht 7a
print("Opdracht 7a: Gemiddelde afbeeldingen\n")

gemiddelde_cijfers = np.array([X[y == i].mean(axis=0) for i in range(10)])

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(gemiddelde_cijfers[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Cijfer {i}')
    ax.axis('off')

plt.suptitle('Gemiddelde afbeelding per cijfer', fontsize=14)
plt.tight_layout()
plt.show()