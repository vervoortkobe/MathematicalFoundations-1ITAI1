# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 4a: Print middelste deel van de pixelwaarden

afbeelding = X[100].reshape(28, 28)
midden = afbeelding[9:19, 9:19]

print("Middelste 10x10 pixels van de afbeelding op index 100:")
print()

for rij in midden:
    print(' '.join(f'{int(pixel):3d}' for pixel in rij))
