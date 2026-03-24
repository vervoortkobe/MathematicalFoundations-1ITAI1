import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Gegeven de volgende data, implementeer lineaire regressie via de normal equations:
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

# Stappen:
# 1. Maak de design matrix X met een kolom van enen en een kolom met x_data
n_points = len(x_data)
X_design = np.column_stack([np.ones(n_points), x_data])

print("Design matrix X (eerste 5 rijen):")
print(X_design[:5])
print(f"Shape: {X_design.shape}")
print()

# 2. Bereken XᵀX en Xᵀy
XtX = X_design.T @ X_design
Xty = X_design.T @ y_data

print("XᵀX:")
print(XtX)
print()

# 3. Los op voor w = [intercept, slope]
w = np.linalg.solve(XtX, Xty)
print(f"Oplossing: w = {w}")
print(f"  Intercept (b): {w[0]:.4f}")
print(f"  Slope (a): {w[1]:.4f}")
print()

# 4. Visualiseer de data en de fit
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', label='Data punten')
x_fit = np.linspace(0, 6, 100)
y_fit = w[0] + w[1] * x_fit
plt.plot(x_fit, y_fit, color='red', label='Lineaire fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
