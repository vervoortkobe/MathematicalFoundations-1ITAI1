import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Plot de functie f(x) = x³ - 3x + 1 en zijn afgeleide f’(x) = 3x² - 3 op hetzelfde figuur voor x ∈ [-3, 3].

f = lambda x: x**3 - 3*x + 1
f_prime = lambda x: 3*x**2 - 3
x = np.linspace(-3, 3, 400)
y = f(x)
y_prime = f_prime(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, y_prime)
ax.axhline(y=0, color='r', linestyle='-', linewidth=1)
ax.legend(['f(x)', "f'(x)", 'f\'(0)'])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('f(x) = x^3 - 3x + 1 en f\'(x) = 3x^2 - 3')
plt.show()

# Waar is f’(x) = 0? Wat betekent dit voor f(x)?
"""
f'(x) = 0 wanneer 3x² - 3 = 0, wat betekent dat x² = 1, dus x = ±1.
Dit betekent dat f(x) een lokaal maximum heeft bij x = -1 en een lokaal minimum bij x = 1.
"""