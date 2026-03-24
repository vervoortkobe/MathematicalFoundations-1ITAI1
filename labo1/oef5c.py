# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 5c: Gemiddelde acht berekenen en vergelijken

achten = X[y == 8]
gemiddelde_acht = np.mean(achten, axis=0).reshape(28, 28)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

im1 = axes[0].imshow(gemiddelde_drie, cmap='gray')
axes[0].set_title('Gemiddelde 3', fontsize=14)
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(gemiddelde_acht, cmap='gray')
axes[1].set_title('Gemiddelde 8', fontsize=14)
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
