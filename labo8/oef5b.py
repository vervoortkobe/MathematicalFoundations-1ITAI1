import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Verifieer je antwoord met een simulatie: genereer 10000 emails en tel.

# Parameters
P_spam = 0.3
P_niet_spam = 1 - P_spam
P_gratis_gegeven_spam = 0.8
P_gratis_gegeven_niet_spam = 0.1

# Natuurlijke frequenties
n_emails = 10000
n_spam = int(n_emails * P_spam)
n_niet_spam = n_emails - n_spam

n_spam_gratis = int(n_spam * P_gratis_gegeven_spam)
n_niet_spam_gratis = int(n_niet_spam * P_gratis_gegeven_niet_spam)

n_totaal_gratis = n_spam_gratis + n_niet_spam_gratis

print(f"Van {n_emails} e-mails:")
print(f"  {n_spam} zijn spam, {n_niet_spam} zijn geen spam")
print(f"  {n_spam_gratis} spam e-mails bevatten 'gratis'")
print(f"  {n_niet_spam_gratis} niet-spam e-mails bevatten 'gratis'")
print(f"  Totaal e-mails met 'gratis': {n_totaal_gratis}")
print(f"  Kans spam gegeven 'gratis': {n_spam_gratis}/{n_totaal_gratis} = {n_spam_gratis/n_totaal_gratis:.4f}")

# Visualisatie
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Links: verdeling van de e-mails
labels_emails = ['Spam', 'Geen Spam']
sizes_emails = [n_spam, n_niet_spam]
colors_emails = ['#e74c3c', '#2ecc71']
axes[0].pie(sizes_emails, labels=labels_emails, colors=colors_emails, autopct='%1.1f%%',
           textprops={'fontsize': 12}, startangle=90)
axes[0].set_title('Verdeling van e-mails', fontsize=13)

# Rechts: samenstelling van e-mails met 'gratis'
labels_gratis = [f'Spam\n({n_spam_gratis})', f'Geen Spam\n({n_niet_spam_gratis})']
sizes_gratis = [n_spam_gratis, n_niet_spam_gratis]
colors_gratis = ['#e74c3c', '#f39c12']
axes[1].pie(sizes_gratis, labels=labels_gratis, colors=colors_gratis, autopct='%1.1f%%',
           textprops={'fontsize': 12}, startangle=90)
axes[1].set_title(f'E-mails met "gratis" ({n_totaal_gratis} e-mails)', fontsize=13)

plt.suptitle('Spamfilter: E-mails met "gratis"', fontsize=14)
plt.tight_layout()
plt.show()

# Simulatie
np.random.seed(42)
emails = np.random.choice(['spam', 'niet_spam'], size=n_emails, p=[P_spam, P_niet_spam])
spam_with_gratis = np.random.binomial(1, P_gratis_gegeven_spam, size=n_spam)
niet_spam_with_gratis = np.random.binomial(1, P_gratis_gegeven_niet_spam, size=n_niet_spam)

simulated_gratis = sum(spam_with_gratis) + sum(niet_spam_with_gratis)
simulated_spam_gratis = sum(spam_with_gratis)

print(f"\nSimulatie:")
print(f"  Totaal e-mails met 'gratis': {simulated_gratis}")
print(f"  Spam e-mails met 'gratis': {simulated_spam_gratis}")
print(f"  Kans spam gegeven 'gratis': {simulated_spam_gratis}/{simulated_gratis} = {simulated_spam_gratis/simulated_gratis:.4f}")