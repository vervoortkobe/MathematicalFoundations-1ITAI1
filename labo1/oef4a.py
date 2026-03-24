# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 4: Grid met één voorbeeld per cijfer

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Één voorbeeld van elk cijfer (0-9)', fontsize=16)

for cijfer, ax in enumerate(axes.flat):
    # Vind de index van het eerste voorbeeld met dit label
    index = np.where(y == cijfer)[0][0]
    
    # Haal de afbeelding op en reshape
    afbeelding = X[index].reshape(28, 28)
    
    # Toon de afbeelding
    ax.imshow(afbeelding, cmap='gray')
    ax.set_title(f'Cijfer: {cijfer}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()
