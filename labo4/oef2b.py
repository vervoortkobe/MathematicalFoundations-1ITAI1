import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# x + 2y = 3
# 2x + 4y = 6
A = [[1, 2], [2, 4]]
b = [3, 6]

x_oplossing = np.linalg.solve(A, b)

print(f"Oplossing: x = {x_oplossing}")
print(f"  x = {x_oplossing[0]}")
print(f"  y = {x_oplossing[1]}")
print()

# Verificatie
print("Verificatie (A @ x moet gelijk zijn aan b):")
print(f"  A @ x = {A @ x_oplossing}")
print(f"  b     = {b}")
print(f"  Gelijk? {np.allclose(A @ x_oplossing, b)}")

# Leg uit waarom dit stelsel oneindig veel oplossingen heeft.
"""
numpy.linalg.LinAlgError: Singular matrix
Dit gebeurt omdat dit stelsel oneindig veel oplossingen heeft; het is namelijk een homogeen stelsel. 
Een homogeen stelsel heeft de eigenschap dat alle oplossingen een parameter hebben, waardoor er oneindig veel mogelijke oplossingen zijn.

De tweede vergelijking is het dubbel van de eerste vergelijking en dus zijn ze eigenlijk gelijk.
Hierdoor zal je determinant gelijk zijn aan 0 en zul je oneindig veel oplossingen hebben.
Dit komt doordat je vergelijkingen op elkaar liggen.
"""
