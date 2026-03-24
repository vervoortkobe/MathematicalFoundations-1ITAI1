# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 4c: Aantal niet-zwarte pixels

afbeelding = X[100]

niet_zwart = np.sum(afbeelding > 0)
totaal = len(afbeelding)
percentage = (niet_zwart / totaal) * 100

print(f"Aantal niet-zwarte pixels: {niet_zwart}")
print(f"Totaal aantal pixels: {totaal}")
print(f"Percentage niet-zwart: {percentage:.1f}%")
print()
print("Het cijfer beslaat dus slechts een klein deel van de totale afbeelding.")
