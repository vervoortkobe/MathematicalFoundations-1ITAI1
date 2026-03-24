# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 2a: Aantal voorbeelden per cijfer

unieke_labels, aantallen = np.unique(y, return_counts=True)

print("Aantal voorbeelden per cijfer:")
print()
for label, aantal in zip(unieke_labels, aantallen):
    print(f"  Cijfer {label}: {aantal:,} voorbeelden")

print(f"\nTotaal: {aantallen.sum():,} voorbeelden")
