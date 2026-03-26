import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("/Users/pepedesintas/PycharmProjects/ResNet50/results/cbis_ddsm_resnet50_frozen/metrics/test_predictions.csv")

y_true = df["y_true"].values
y_prob = df["y_prob"].values

thresholds = np.arange(0.1, 0.9, 0.05)

# =========================
# STORAGE
# =========================

results = []

print(f"{'TH':<6} {'FN':<6} {'TP':<6} {'FP':<6} {'TN':<6} {'SENS':<8} {'SPEC':<8}")

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"{t:<6.2f} {fn:<6} {tp:<6} {fp:<6} {tn:<6} {sensitivity:<8.3f} {specificity:<8.3f}")

    results.append({
        "threshold": t,
        "fn": fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity
    })

# Convertimos a DataFrame
results_df = pd.DataFrame(results)

# =========================
# PLOT
# =========================

plt.figure(figsize=(9, 6))

plt.plot(results_df["threshold"], results_df["sensitivity"], marker="o", label="Sensitivity")
plt.plot(results_df["threshold"], results_df["specificity"], marker="s", label="Specificity")

# Línea vertical en 0.5 (baseline)
plt.axvline(0.5, linestyle="--", linewidth=1.5, label="Threshold = 0.5")

# Línea vertical en 0.3 (recomendado)
plt.axvline(0.3, linestyle="--", linewidth=1.5, label="Threshold = 0.3")

plt.title("Sensitivity-Specificity trade-off vs threshold")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.xlim(0.1, 0.85)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

# Guardar
plt.savefig("threshold_analysis_cbis.png", bbox_inches="tight", dpi=300)

plt.show()