# Imports en data laden

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Geladen: {len(X)} afbeeldingen")

# Opdracht 3a: Reshape van vector naar matrix

afbeelding_vector = X[100]
afbeelding_matrix = afbeelding_vector.reshape(28, 28)

print(f"Shape van vector: {afbeelding_vector.shape}")
print(f"Shape van matrix: {afbeelding_matrix.shape}")
