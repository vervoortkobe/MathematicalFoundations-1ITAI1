import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Gebruik de inverse om het stelsel Ax = b op te lossen via x = A⁻¹b:
A = [[4, 7],
     [2, 6]]
b = [5, 3]

# Bereken de inverse van A.
A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print()
print("Inverse A⁻¹:")
print(A_inv)
print()

# Bereken de oplossing via x = A⁻¹b.
x_oplossing = A_inv @ b

print(f"Oplossing: x = {x_oplossing}")
print(f"  x = {x_oplossing[0]}")
print(f"  y = {x_oplossing[1]}")
print()

# Vergelijk met de oplossing via np.linalg.solve().
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

"""
We zien dat de oplossingen van x = A⁻¹b en np.linalg.solve(A, b) hetzelfde zijn.
"""