import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    moment,
    binom,
    poisson,
    uniform,
    expon,
    norm,
    skew,
    kurtosis,
)


df = pd.read_csv("/Users/mmadando/Downloads/Netflix_stock_data.csv")
print("Columns in dataset:", df.columns)


data = pd.to_numeric(df["Close"], errors="coerce").values
data = data[~np.isnan(data)]

mean = np.mean(data)
print("The average Close price is:", mean)
variance = np.var(data)
print()
third_moment = moment(data, moment=3)
fourth_moment = moment(data, moment=4)
data_skew = skew(data)
data_kurt = kurtosis(data)

print("\nMOMENTS")
print("Mean:", mean)
print("Variance:", variance)
print("3rd Moment:", third_moment)
print("4th Moment:", fourth_moment)
print("Skewness:", data_skew)
print("Excess Kurtosis:", data_kurt)

# --------------------------------------
# 2. Binomial Distribution
mean_close = np.mean(data)
p = np.mean(data > mean_close)
n = 10
k = 3
binom_prob = binom.pmf(k, n, p)
print("\nBINOMIAL DISTRIBUTION (based on Close > mean)")
print(f"P(X=3) for Binomial(n=10, p={p:.3f}): {binom_prob}")

# --------------------------------------
# 3. Poisson Distribution
lam = np.mean(data)
k = int(np.round(lam))
poisson_prob = poisson.pmf(k, lam)
print("\nPOISSON DISTRIBUTION (lambda = mean Close)")
print(f"P(X={k}) for Poisson(lambda={lam:.2f}): {poisson_prob}")

# --------------------------------------
# 4. Uniform Distribution
a = np.min(data)
b = np.max(data)
x_uniform = np.median(data)
uniform_density = uniform.pdf(x_uniform, loc=a, scale=b - a)
print("\nUNIFORM DISTRIBUTION (min/max of Close)")
print(f"PDF at median Close {x_uniform:.2f}: {uniform_density}")

# --------------------------------------
# 5. Exponential Distribution
scale = np.mean(data)
x_expon = scale
expon_density = expon.pdf(x_expon, scale=scale)
print("\nEXPONENTIAL DISTRIBUTION (scale = mean Close)")
print(f"PDF at x=mean Close ({x_expon:.2f}): {expon_density}")

# --------------------------------------
# 6. Normal Distribution
mu = np.mean(data)
sigma = np.std(data)
x_norm = mu
norm_density = norm.pdf(x_norm, loc=mu, scale=sigma)
print("\nNORMAL DISTRIBUTION (mean and std of Close)")
print(f"PDF at x=mean Close ({x_norm:.2f}): {norm_density}")

# --------------------------------------
# 7. Central Limit Theorem demonstration
sample_means = []
for _ in range(1000):
    sample = np.random.choice(data, size=30, replace=True)
    sample_means.append(np.mean(sample))

plt.figure()
plt.hist(sample_means, bins=30, density=True)
plt.title("Central Limit Theorem: Means of Stock Close Samples")
plt.xlabel("Sample Mean")
plt.ylabel("Density")

# Correlation between Close and Open
open_data = pd.to_numeric(df["Open"], errors="coerce").values
mask = ~np.isnan(open_data) & ~np.isnan(data)
correlation = np.corrcoef(open_data[mask], data[mask])[0, 1]

print("\nCORRELATION (Open vs Close)")
print("Correlation between Open and Close:", correlation)

# --------------------------------------
# 9. Covariance
covariance = np.cov(open_data[mask], data[mask])[0, 1]
print("\nCOVARIANCE (Open vs Close)")
print("Covariance between Open and Close:", covariance)

# --------------------------------------
# 10. Random Walk of daily price changes
price_changes = np.diff(data)
walk = np.cumsum(price_changes)

plt.figure()
plt.plot(walk)
plt.title("Random Walk of Daily Price Changes")
plt.xlabel("Days")
plt.ylabel("Cumulative Change")

# --------------------------------------
# 11. Markov Process simulation
states = ["Up", "Down"]
transition_matrix = [[0.7, 0.3], [0.4, 0.6]]
current_state = 0  # Start in 'Up' state
n_days = 10
sequence = []

for _ in range(n_days):
    sequence.append(states[current_state])
    current_state = np.random.choice([0, 1], p=transition_matrix[current_state])

print("\nMARKOV PROCESS (Price Movement)")
print("Price Movement Sequence over 10 days:", sequence)

# --------------------------------------
plt.show()
