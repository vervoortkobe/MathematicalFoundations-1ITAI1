# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 3b: Visualisatie met label

label = y[100]

plt.figure(figsize=(5, 5))
plt.imshow(afbeelding_matrix, cmap='gray')
plt.title(f'Afbeelding op index 100, label: {label}', fontsize=14)
plt.axis('off')
plt.show()
