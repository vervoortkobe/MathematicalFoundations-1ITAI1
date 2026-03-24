# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 3c: Controle met andere indices

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

indices = [100, 500, 1000, 5000]

for ax, idx in zip(axes, indices):
    afb = X[idx].reshape(28, 28)
    ax.imshow(afb, cmap='gray')
    ax.set_title(f'Index {idx}, label: {y[idx]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

print("De labels komen overeen met de getoonde cijfers.")
