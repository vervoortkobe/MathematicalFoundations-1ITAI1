# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 2c: Minimum en maximum pixelwaarden

print(f"Minimum pixelwaarde: {X.min()}")
print(f"Maximum pixelwaarde: {X.max()}")
print()
print("Dit komt overeen met wat we verwachten voor grijswaarden.")
print("0 = volledig zwart, 255 = volledig wit.")
