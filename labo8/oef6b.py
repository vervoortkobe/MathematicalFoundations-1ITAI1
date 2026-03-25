import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Onderzoek het effect van “temperatuur” op softmax. De softmax met temperatuur T is:
# softmax(x/T)
# Plot de softmax output voor dezelfde logits met T = 0.5, 1, 2, 5. Wat observeer je?

def softmax_with_temperature(x, temperature):
    """Softmax-functie met temperatuurparameter."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

# Test logits
logits = np.array([1.0, 2.0, 3.0])

# Temperaturen om te testen
temperatures = [0.5, 1, 2, 5]

# Plot de softmax output voor verschillende temperaturen
plt.figure(figsize=(10, 6))

for T in temperatures:
    softmax_output = softmax_with_temperature(logits, T)
    plt.plot(logits, softmax_output, 'o-', label=f'T = {T}')

plt.title('Effect van temperatuur op softmax')
plt.xlabel('Logits')
plt.ylabel('Softmax output')
plt.xticks(logits)
plt.legend()
plt.grid(True)
plt.show()