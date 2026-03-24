import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

A = [[4, 7],
     [2, 6]]

# Formule: A⁻¹ = (1/det(A)) × [[d, -b], [-c, a]] voor A = [[a, b], [c, d]]
# (1/det(A)) * [[6, -7], [-2, 4]]
# det(A) = ad - bc = 24-14 = 10
# A^-1 = 1/10 * [[6, -7], [-2, 4]] = [[0.6, 0.7], [-0.2, 0.4]]
print("Inverse A⁻¹:")
print([[0.6, 0.7], [-0.2, 0.4]])
print()

A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print()
print("Inverse A⁻¹:")
print(A_inv)
print()

# Verificatie: A × A⁻¹ = I
print("Verificatie: A @ A⁻¹ =")
print(A @ A_inv)
print()
print("Verificatie: A⁻¹ @ A =")
print(A_inv @ A)
print()
print(f"Beide zijn de identiteitsmatrix? {np.allclose(A @ A_inv, np.eye(2)) and np.allclose(A_inv @ A, np.eye(2))}")
