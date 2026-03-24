import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Bakker verkoopt 3 soorten brood.
# Tarwebrood kost €2, roggebrood €3 en meergranenbrood €4.
# maandag: 10t + 5r + 8m = €67
# dinsdag: 8t + 10r + 6m = €70
# woensdag: 12t + 8r + 10m = €88
# Ax = b
A = [[10, 5, 8], [8, 10, 6], [12, 8, 10]]
b = [67, 70, 88]

# Oplossen met NumPy
x_oplossing = np.linalg.solve(A, b)

print(f"Oplossing: x = {x_oplossing}")
print(f"  x = {x_oplossing[0]}")
print(f"  y = {x_oplossing[1]}")
print(f"  z = {x_oplossing[2]}")
print()

# Verificatie
print("Verificatie (A @ x moet gelijk zijn aan b):")
print(f"  A @ x = {A @ x_oplossing}")
print(f"  b     = {b}")
print(f"  Gelijk? {np.allclose(A @ x_oplossing, b)}")

