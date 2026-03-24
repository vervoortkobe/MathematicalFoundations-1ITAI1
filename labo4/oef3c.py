import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Bereken de inverse van een 3×3 matrix met NumPy en verifieer:
B = [[1, 2, 3],
     [0, 1, 4],
     [5, 6, 0]]

B_inv = np.linalg.inv(B)

print("Matrix B:")
print(B)
print()
print("Inverse B⁻¹:")
print(B_inv)
print()

# Verificatie: B × B⁻¹ = I
print("Verificatie: B @ B⁻¹ =")
print(B @ B_inv)
print()
print("Verificatie: B⁻¹ @ B =")
print(B_inv @ B)
print()
print(f"Beide zijn de identiteitsmatrix? {np.allclose(B @ B_inv, np.eye(3)) and np.allclose(B_inv @ B, np.eye(3))}")