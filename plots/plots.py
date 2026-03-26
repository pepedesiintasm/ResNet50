import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix


# =========================================================
# CONFIGURATION
# =========================================================

RESULTS_DIR = Path("/Users/pepedesintas/PycharmProjects/ResNet50/results")
OUTPUT_DIR = RESULTS_DIR / "plots"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================
# UTILITIES
# =========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_read_csv(path, index_col=None):
    if Path(path).exists():
        return pd.read_csv(path, index_col=index_col)
    return None


def save_figure(fig, out_path):
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def discover_experiments(results_dir):
    experiments = []
    for item in sorted(results_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("plots"):
            continue

        metrics_dir = item / "metrics"
        if metrics_dir.exists():
            experiments.append({
                "name": item.name,
                "root": item,
                "metrics_dir": metrics_dir
            })
    return experiments


def prettify_name(name):
    return name.replace("_", " ").upper()


def infer_dataset_and_mode(experiment_name):
    name = experiment_name.lower()

    if "busi" in name:
        dataset = "BUSI"
    elif "cbis" in name:
        dataset = "CBIS-DDSM"
    elif "mias" in name or "stage1" in name or "stage2" in name or "stage3" in name:
        dataset = "MIAS"
    else:
        dataset = "UNKNOWN"

    if "frozen" in name:
        mode = "Frozen"
    elif "finetuning" in name or "fine_tuning" in name:
        mode = "Fine-tuning"
    else:
        mode = "Unknown"

    return dataset, mode


# =========================================================
# INDIVIDUAL PLOTS
# =========================================================

def plot_loss(history_df, experiment_name, out_dir):
    if history_df is None or history_df.empty:
        return
    if "loss" not in history_df.columns or "val_loss" not in history_df.columns:
        return

    epochs = history_df["epoch"] if "epoch" in history_df.columns else np.arange(1, len(history_df) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, history_df["loss"], marker="o", markersize=4, label="Train loss")
    ax.plot(epochs, history_df["val_loss"], marker="s", markersize=4, label="Validation loss")

    ax.set_title(f"{prettify_name(experiment_name)} - Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)

    save_figure(fig, out_dir / "loss.png")


def plot_auc(history_df, experiment_name, out_dir):
    if history_df is None or history_df.empty:
        return
    if "auc" not in history_df.columns or "val_auc" not in history_df.columns:
        return

    epochs = history_df["epoch"] if "epoch" in history_df.columns else np.arange(1, len(history_df) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, history_df["auc"], marker="o", markersize=4, label="Train AUC")
    ax.plot(epochs, history_df["val_auc"], marker="s", markersize=4, label="Validation AUC")

    ax.set_title(f"{prettify_name(experiment_name)} - AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True)

    save_figure(fig, out_dir / "auc.png")


def plot_roc_curve(roc_df, experiment_name, out_dir, auc_value=None):
    if roc_df is None or roc_df.empty:
        return
    if not {"fpr", "tpr"}.issubset(roc_df.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    label = f"ROC curve (AUC = {auc_value:.3f})" if auc_value is not None else "ROC curve"

    ax.plot(roc_df["fpr"], roc_df["tpr"], label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random classifier")

    ax.set_title(f"{prettify_name(experiment_name)} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=True)

    save_figure(fig, out_dir / "roc_curve.png")


def plot_confusion_matrix(cm_df, experiment_name, out_dir):
    if cm_df is None or cm_df.empty:
        return

    cm = cm_df.values.astype(int)
    if cm.shape != (2, 2):
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")

    ax.set_title(f"{prettify_name(experiment_name)} - Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    threshold = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, out_dir / "confusion_matrix.png")


def plot_threshold_analysis(pred_df, experiment_name, out_dir):
    if pred_df is None or pred_df.empty:
        return
    if not {"y_true", "y_prob"}.issubset(pred_df.columns):
        return

    y_true = pred_df["y_true"].astype(int).values
    y_prob = pred_df["y_prob"].astype(float).values

    thresholds = np.linspace(0.0, 1.0, 101)
    sensitivity_list = []
    specificity_list = []
    f1_list = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_list.append(f1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, sensitivity_list, label="Sensitivity / Recall")
    ax.plot(thresholds, specificity_list, label="Specificity")
    ax.plot(thresholds, f1_list, label="F1-score")
    ax.axvline(0.5, linestyle="--", linewidth=1.5, label="Threshold = 0.5")

    ax.set_title(f"{prettify_name(experiment_name)} - Threshold Analysis")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=True)

    save_figure(fig, out_dir / "threshold_analysis.png")


# =========================================================
# GLOBAL COMPARISONS
# =========================================================

def build_summary(experiments):
    rows = []

    for exp in experiments:
        final_df = safe_read_csv(exp["metrics_dir"] / "final_metrics.csv")
        if final_df is None or final_df.empty:
            continue

        row = final_df.iloc[0].to_dict()
        dataset, mode = infer_dataset_and_mode(exp["name"])
        row["experiment_name"] = exp["name"]
        row["dataset_group"] = dataset
        row["mode_group"] = mode
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def plot_grouped_metric(summary_df, metric_col, title, ylabel, out_path):
    if summary_df.empty or metric_col not in summary_df.columns:
        return

    df = summary_df.copy()
    df = df[df["mode_group"].isin(["Frozen", "Fine-tuning"])]

    datasets = sorted(df["dataset_group"].unique())

    frozen_vals = []
    ft_vals = []

    for dataset in datasets:
        df_dataset = df[df["dataset_group"] == dataset]

        frozen = df_dataset[df_dataset["mode_group"] == "Frozen"]
        finetune = df_dataset[df_dataset["mode_group"] == "Fine-tuning"]

        frozen_vals.append(float(frozen.iloc[0][metric_col]) if not frozen.empty else np.nan)
        ft_vals.append(float(finetune.iloc[0][metric_col]) if not finetune.empty else np.nan)

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, frozen_vals, width, label="Frozen")
    bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-tuning")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    h + 0.015,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

    save_figure(fig, out_path)


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    experiments = discover_experiments(RESULTS_DIR)
    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiments.")

    for exp in experiments:
        print(f"Generating main plots for: {exp['name']}")

        exp_out_dir = OUTPUT_DIR / exp["name"]
        ensure_dir(exp_out_dir)

        history_df = safe_read_csv(exp["metrics_dir"] / "training_history.csv")
        roc_df = safe_read_csv(exp["metrics_dir"] / "roc_curve.csv")
        pred_df = safe_read_csv(exp["metrics_dir"] / "test_predictions.csv")
        cm_df = safe_read_csv(exp["metrics_dir"] / "confusion_matrix.csv", index_col=0)
        final_df = safe_read_csv(exp["metrics_dir"] / "final_metrics.csv")

        auc_value = None
        if final_df is not None and not final_df.empty and "roc_auc" in final_df.columns:
            auc_value = float(final_df.iloc[0]["roc_auc"])

        plot_loss(history_df, exp["name"], exp_out_dir)
        plot_auc(history_df, exp["name"], exp_out_dir)
        plot_roc_curve(roc_df, exp["name"], exp_out_dir, auc_value)
        plot_confusion_matrix(cm_df, exp["name"], exp_out_dir)
        plot_threshold_analysis(pred_df, exp["name"], exp_out_dir)

    summary_df = build_summary(experiments)
    if not summary_df.empty:
        summary_df.to_csv(OUTPUT_DIR / "summary_metrics_all_experiments.csv", index=False)

        comparison_dir = OUTPUT_DIR / "comparisons"
        ensure_dir(comparison_dir)

        plot_grouped_metric(
            summary_df, "roc_auc",
            "ROC-AUC Comparison by Dataset",
            "ROC-AUC",
            comparison_dir / "roc_auc_comparison.png"
        )

        plot_grouped_metric(
            summary_df, "recall_sensitivity",
            "Sensitivity Comparison by Dataset",
            "Sensitivity",
            comparison_dir / "sensitivity_comparison.png"
        )

        plot_grouped_metric(
            summary_df, "specificity",
            "Specificity Comparison by Dataset",
            "Specificity",
            comparison_dir / "specificity_comparison.png"
        )

        plot_grouped_metric(
            summary_df, "f1_score",
            "F1-score Comparison by Dataset",
            "F1-score",
            comparison_dir / "f1_comparison.png"
        )

    print("\nMain plots generated successfully.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()