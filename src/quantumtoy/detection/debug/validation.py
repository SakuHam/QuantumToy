import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# -----------------------------
# Load data
# -----------------------------

with open("output_flux_summary.json") as f:
    summary = json.load(f)

with open("output_pseudo_clicks.jsonl") as f:
    clicks = [json.loads(l) for l in f]

y_click = np.array([c["y"] for c in clicks])

y_coords = np.array(summary["y_coords"])
flux = np.array(summary["flux_y_accum"])


# normalize flux
flux = flux / np.sum(flux)


# -----------------------------
# histogram from clicks
# -----------------------------

hist, bins = np.histogram(
    y_click,
    bins=len(y_coords),
    range=(y_coords.min(), y_coords.max()),
    density=True
)

hist = hist / np.sum(hist)


# -----------------------------
# KL divergence
# -----------------------------

kl = np.sum(hist * np.log((hist + 1e-12) / (flux + 1e-12)))

print("KL divergence =", kl)


# -----------------------------
# Kolmogorov–Smirnov
# -----------------------------

# build flux CDF
flux_cdf = np.cumsum(flux)
flux_cdf /= flux_cdf[-1]

def flux_cdf_interp(x):
    return np.interp(x, y_coords, flux_cdf)

ks = stats.kstest(y_click, flux_cdf_interp)

print("KS statistic =", ks.statistic)
print("KS p-value =", ks.pvalue)


# -----------------------------
# chi-square
# -----------------------------

expected = flux * len(y_click)

observed, _ = np.histogram(
    y_click,
    bins=len(y_coords),
    range=(y_coords.min(), y_coords.max())
)

chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-12))

print("Chi2 =", chi2)


# -----------------------------
# visual check
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(y_coords, flux, label="Flux P(y)")
plt.hist(y_click, bins=80, density=True, alpha=0.4, label="Pseudo-clicks")

plt.legend()
plt.grid()
plt.show()