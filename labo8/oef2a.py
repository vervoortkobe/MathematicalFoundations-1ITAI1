import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Een fabriek produceert chips met een defectrate van 2%. Je inspecteert een batch van 100 chips.
# Wat is de kans op exact 2 defecte chips?
# Wat is de kans op minder dan 5 defecte chips?
# Wat is de kans op meer dan 5 defecte chips?
# Gebruik de binomiaalverdeling.

# Parameters
n = 100  # Aantal chips
p = 0.02  # Defectrate

# Kansen berekenen
kans_exact_2_defect = stats.binom.pmf(2, n, p)
kans_minder_dan_5_defect = stats.binom.cdf(4, n, p)
kans_meer_dan_5_defect = 1 - stats.binom.cdf(5, n, p)

print(f"Kans exact 2 defect: ", kans_exact_2_defect)
print(f"Kans minder dan 5 defect: ", kans_minder_dan_5_defect)
print(f"Kans meer dan 5 defect: ", kans_meer_dan_5_defect)