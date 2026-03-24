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

# Berekening
n_samples = 1000
batch_size = 32
n_full_batches = n_samples // batch_size  # 31
remainder = n_samples % batch_size       # 8

print(f"Aantal volledige batches: {n_full_batches}")
print(f"Overige afbeeldingen: {remainder}")
print(f"Totaal: {n_full_batches * batch_size + remainder} = 1000")

"""
Opties voor de 8 overgebleven afbeeldingen:
1. Negeer ze (gebruik alleen 31×32 = 992 afbeeldingen)
2. Kleinere laatste batch (32, 32, ..., 32, 8)
3. Padding (voeg 24 nul-afbeeldingen toe om batch van 32 te maken)
4. Herhaal (herhaal eerste 24 van de 8 om batch vol te maken)
Meest voorkomende aanpak (optie 1):
"""

X_train = X[:1000]
batches = []
for i in range(0, len(X_train)-batch_size+1, batch_size):
    batches.append(X_train[i:i+batch_size])
print(f"Aantal batches: {len(batches)}")  # 31
