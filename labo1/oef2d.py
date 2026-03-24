# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 2d: Uitleg shape

print(f"Shape van X: {X.shape}")
print()
print("Het eerste getal (70000) is het aantal afbeeldingen in de dataset.")
print("Het tweede getal (784) is het aantal pixels per afbeelding.")
print("784 = 28 × 28, de afmetingen van elke afbeelding.")
