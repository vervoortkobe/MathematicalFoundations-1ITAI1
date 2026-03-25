import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Je gooit twee dobbelstenen. Bereken de kans op:
# 1. Totaal van 7
# 6/36 = 1/6
# 2. Totaal van 12
# 1/36
# 3. Totaal groter dan 9
# 6/36 = 1/6
# 4. Minstens één 6
# 1 - 25/36 = 11/36
# Verifieer je antwoorden met een simulatie.

# Aantal simulaties
n_simulations = 100000

# Simuleer het gooien van twee dobbelstenen
dice1 = np.random.randint(1, 7, n_simulations)
dice2 = np.random.randint(1, 7, n_simulations)
totals = dice1 + dice2

# Bereken de kansen
kans_totaal_7 = np.sum(totals == 7) / n_simulations
kans_totaal_12 = np.sum(totals == 12) / n_simulations
kans_totaal_groter_dan_9 = np.sum(totals > 9) / n_simulations
kans_minstens_een_6 = np.sum((dice1 == 6) | (dice2 == 6)) / n_simulations

print(f"Kans totaal 7: {kans_totaal_7}")
print(f"Kans totaal 12: {kans_totaal_12}")
print(f"Kans totaal groter dan 9: {kans_totaal_groter_dan_9}")
print(f"Kans minstens een 6: {kans_minstens_een_6}")