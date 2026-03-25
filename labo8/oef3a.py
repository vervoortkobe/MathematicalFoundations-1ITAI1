import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# IQ-scores zijn normaal verdeeld met μ = 100 en σ = 15.
# Welk percentage van de bevolking heeft een IQ boven 130?
# Welk percentage heeft een IQ tussen 85 en 115?
# Wat is de IQ-score waarboven slechts 1% van de bevolking scoort?
# Als je 1000 mensen willekeurig kiest, hoeveel verwacht je met IQ > 145?

# Parameters
mu = 100  # Gemiddelde IQ-score
sigma = 15  # Standaarddeviatie

# a. Percentage van de bevolking met een IQ boven 130
percentage_boven_130 = 1 - stats.norm.cdf(130, mu, sigma)

# b. Percentage van de bevolking met een IQ tussen 85 en 115
percentage_tussen_85_en_115 = stats.norm.cdf(115, mu, sigma) - stats.norm.cdf(85, mu, sigma)

# c. IQ-score waarboven slechts 1% van de bevolking scoort
iq_boven_1_procent = stats.norm.ppf(0.99, mu, sigma)

# d. Verwacht aantal mensen met IQ > 145 in een groep van 1000
kans_boven_145 = 1 - stats.norm.cdf(145, mu, sigma)
verwacht_aantal_boven_145 = kans_boven_145 * 1000

print(f"Percentage boven 130: ", percentage_boven_130)
print(f"Percentage tussen 85 en 115: ", percentage_tussen_85_en_115)
print(f"IQ boven 1 percent: ", iq_boven_1_procent)
print(f"Verwacht aantal boven 145: ", verwacht_aantal_boven_145)