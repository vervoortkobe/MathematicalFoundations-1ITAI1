import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Een spamfilter heeft de volgende eigenschappen:
# P(spam) = 0.3 (30% van alle emails is spam)
# P("gratis"|spam) = 0.8 (80% van spam bevat “gratis”)
# P("gratis"|niet spam) = 0.1 (10% van normale emails bevat “gratis”)
# Bereken met Bayes:
# 1. P(spam|"gratis")
# 2. P(niet spam|"gratis")

# Gegevens
P_spam = 0.3
P_niet_spam = 1 - P_spam
P_gratis_gegeven_spam = 0.8
P_gratis_gegeven_niet_spam = 0.1

# Bereken P("gratis")
P_gratis = (P_gratis_gegeven_spam * P_spam) + (P_gratis_gegeven_niet_spam * P_niet_spam)

# Bereken P(spam | "gratis")
P_spam_gegeven_gratis = (P_gratis_gegeven_spam * P_spam) / P_gratis

# Bereken P(niet spam | "gratis")
P_niet_spam_gegeven_gratis = 1 - P_spam_gegeven_gratis

# Resultaten
print(f"P(spam | 'gratis') = {P_spam_gegeven_gratis:.4f} ({P_spam_gegeven_gratis*100:.2f}%)")
print(f"P(niet spam | 'gratis') = {P_niet_spam_gegeven_gratis:.4f} ({P_niet_spam_gegeven_gratis*100:.2f}%)")