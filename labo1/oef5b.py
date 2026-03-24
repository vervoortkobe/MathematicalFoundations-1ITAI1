# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 5b: Gemiddelde drie berekenen en visualiseren

gemiddelde_drie = np.mean(drieen, axis=0).reshape(28, 28)

plt.figure(figsize=(5, 5))
plt.imshow(gemiddelde_drie, cmap='gray')
plt.title('Gemiddelde afbeelding van het cijfer 3', fontsize=14)
plt.colorbar(label='Gemiddelde pixelwaarde')
plt.axis('off')
plt.show()
