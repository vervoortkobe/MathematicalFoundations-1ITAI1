# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 2b: Gemiddelde pixelwaarde

gem_pixel = X.mean()
print(f"Gemiddelde pixelwaarde over alle afbeeldingen: {gem_pixel:.2f}")
print()
print("Dit getal is veel dichter bij 0 dan bij 255.")
print("Dit betekent dat de meeste pixels zwart zijn (achtergrond).")
print("De cijfers zelf vormen slechts een klein deel van elke afbeelding.")
