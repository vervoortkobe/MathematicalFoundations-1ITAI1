import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Maak een contourplot van f(x, y) = x² + y² (een paraboloïde) en teken gradiëntvectoren als pijlen. De gradiënt is ∇f = [2x, 2y].
# Definieer het bereik van x en y
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Bereken de waarde van f(x, y)
Z = X**2 + Y**2

# Maak een contourplot
fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z)
ax.clabel(contour, inline=1, fontsize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Paraboloide')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Teken gradiëntvectoren
grad_x = 2 * X
grad_y = 2 * Y
ax.quiver(X, Y, grad_x, grad_y, angles='xy', scale_units='xy', scale=1, color='black')

plt.show()