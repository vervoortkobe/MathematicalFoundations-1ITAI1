import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Demonstreer de centrale limietstelling met een exponentiële verdeling (die scheef is, niet normaal).
# 1. Trek 10000 samples uit Exp(lambda = 1) en plot het histogram
# 2. Trek 10000 keer het gemiddelde van n = 2 exponentiële samples, plot histogram
# 3. Herhaal voor n = 5, 10, 30
# 4. Observeer hoe de verdeling steeds normaler wordt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

# Parameters
n_samples = 10000
lambda_exp = 1
scale = 1 / lambda_exp

# Maak een figuur met subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Enkele exponentiële verdeling
samples_single = np.random.exponential(scale, n_samples)
axes[0, 0].hist(samples_single, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_title(f'Exponentiële verdeling (λ = {lambda_exp})')
axes[0, 0].set_xlabel('Waarde')
axes[0, 0].set_ylabel('Dichtheid')

# Plot 2: Gemiddelde van n = 2 exponentiële variabelen
samples_2 = np.random.exponential(scale, (n_samples, 2))
means_2 = np.mean(samples_2, axis=1)
axes[0, 1].hist(means_2, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].set_title(f'Gemiddelde van n = 2')
axes[0, 1].set_xlabel('Gemiddelde waarde')
axes[0, 1].set_ylabel('Dichtheid')

# Plot 3: Gemiddelde van n = 5 exponentiële variabelen
samples_5 = np.random.exponential(scale, (n_samples, 5))
means_5 = np.mean(samples_5, axis=1)
axes[1, 0].hist(means_5, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[1, 0].set_title(f'Gemiddelde van n = 5')
axes[1, 0].set_xlabel('Gemiddelde waarde')
axes[1, 0].set_ylabel('Dichtheid')

# Plot 4: Gemiddelde van n = 30 exponentiële variabelen
samples_30 = np.random.exponential(scale, (n_samples, 30))
means_30 = np.mean(samples_30, axis=1)
axes[1, 1].hist(means_30, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[1, 1].set_title(f'Gemiddelde van n = 30')
axes[1, 1].set_xlabel('Gemiddelde waarde')
axes[1, 1].set_ylabel('Dichtheid')

plt.suptitle('Centrale Limietstelling: Gemiddelden van exponentiële variabelen', fontsize=16)
plt.tight_layout()
plt.show()