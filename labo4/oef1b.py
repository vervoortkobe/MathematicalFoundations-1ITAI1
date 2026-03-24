import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Probleem 1: appels (a) en peren (p)
# 3a + 2p = 5
# 1a + 4p = 6
# Ax = b
A = [[3, 2], [1, 4]]
b = [5, 6]

# Oplossen met NumPy
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

print()

# Probleem 2: snelheid (s) en wind (w)
# (s + w) * 2 = 300  →  2s + 2w = 300
# (s - w) * 3 = 300  →  3s - 3w = 300
# Ax = b
A = [[2, 1], [3, -1]]
b = [300, 300]

# Oplossen met NumPy
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

