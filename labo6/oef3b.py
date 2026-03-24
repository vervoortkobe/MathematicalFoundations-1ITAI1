import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Breid gradient descent uit naar 2D. Minimaliseer f(x, y) = (x - 2)^2 + (y - 3)^2 (minimum bij (2, 3)).
def gradient_descent_2d(f, grad_f, xy_init, learning_rate, n_iterations):
    """Voer gradient descent uit voor een 2D functie.
    
    Args:
        f: functie f(x, y)
        grad_f: gradiënt functie die [∂f/∂x, ∂f/∂y] teruggeeft
        xy_init: startpunt [x, y]
        learning_rate: stapgrootte
        n_iterations: aantal iteraties
    
    Returns:
        xy_final: gevonden minimum [x, y]
        history: lijst van alle [x, y] waarden
    """
    xy_final = np.array(xy_init)
    history = [xy_final]
    for _ in range(n_iterations):
        grad = grad_f(xy_final[0], xy_final[1])
        xy_final = xy_final - learning_rate * grad
        history.append(xy_final)
    return xy_final, history

# Functie en gradiënt
def f_2d(x, y):
    return (x - 2)**2 + (y - 3)**2

def grad_f_2d(x, y):
    return np.array([2*(x - 2), 2*(y - 3)])

# Test
xy_final, history = gradient_descent_2d(f_2d, grad_f_2d, xy_init=[-1, 6], 
                                         learning_rate=0.1, n_iterations=50)
print(f"Gevonden minimum: ({xy_final[0]:.4f}, {xy_final[1]:.4f})")
print(f"Verwacht minimum: (2, 3)")

# Visualiseer het pad op een contourplot van de functie.
path = np.array(history)
x_vals = np.linspace(-2, 6, 200)
y_vals = np.linspace(-1, 7, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = (X - 2)**2 + (Y - 3)**2

plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, levels=20, alpha=0.6, cmap="viridis")
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(path[:, 0], path[:, 1], "o-", color="red", markersize=4, label="Gradient descent path")
plt.plot(2, 3, "go", markersize=8, label="True minimum (2,3)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent Path on Contour Plot\n$f(x,y) = (x-2)^2 + (y-3)^2$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.show()
