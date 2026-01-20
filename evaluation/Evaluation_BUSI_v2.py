import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc as sk_auc,
    classification_report
)

# ======================
# CONFIG
# ======================
DATA_DIR   = "/Users/pepedesintas/Desktop/TFG/BUSI_processed"  # <-- cambia
MODEL_PATH = "../models/resnet50_busi_final.keras"             # <-- cambia
OUT_DIR    = "../results_busi_eval"                            # donde guardar

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16

# Binario
LABEL_MODE = "binary"
THRESHOLD  = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# LOAD DATASETS
# ======================
def load_dataset(split):
    return tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        label_mode=LABEL_MODE,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

train_ds = load_dataset("train")
val_ds   = load_dataset("valid")
test_ds  = load_dataset("test")

class_names = train_ds.class_names
print("Clases:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ======================
# LOAD MODEL (NO TRAINING)
# ======================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modelo cargado:", MODEL_PATH)

# ======================
# HELPERS
# ======================
def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

def evaluate_basic(model, ds):
    """
    Devuelve dict con loss/accuracy/auc si existen.
    Para que funcione con compile=False, compilamos "light" solo para evaluate.
    """
    # Intentamos inferir si el modelo es binario y compilar con AUC
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model.evaluate(ds, verbose=0, return_dict=True)

def get_probs_and_labels(model, ds):
    y_prob = model.predict(ds, verbose=0).ravel()
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    return y_true, y_prob

def save_confusion_matrix_png(cm, labels, out_png, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def save_roc_png(fpr, tpr, roc_auc, out_png, title):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def evaluate_split(split_name, ds):
    print(f"\n==============================")
    print(f"📊 Evaluación en {split_name.upper()}")
    print(f"==============================")

    # Métricas base (loss/acc/auc)
    base = evaluate_basic(model, ds)
    for k, v in base.items():
        print(f"{k}: {v:.4f}")

    # Predicciones
    y_true, y_prob = get_probs_and_labels(model, ds)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:\n", cm)
    print(f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)
    print(f"ROC AUC (sklearn): {roc_auc:.4f}")

    # Classification report
    rep_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(rep_txt)

    # ---------- SAVE FILES ----------
    # 1) Predicciones
    pred_csv = os.path.join(OUT_DIR, f"{split_name}_predictions.csv")
    write_csv(
        pred_csv,
        header=["y_true", "y_prob", "y_pred"],
        rows=list(zip(y_true.tolist(), y_prob.tolist(), y_pred.tolist()))
    )

    # 2) Confusion matrix PNG + CSV
    cm_png = os.path.join(OUT_DIR, f"{split_name}_confusion_matrix.png")
    save_confusion_matrix_png(cm, class_names, cm_png, f"Confusion Matrix ({split_name})")

    cm_csv = os.path.join(OUT_DIR, f"{split_name}_confusion_matrix.csv")
    write_csv(
        cm_csv,
        header=["", f"pred_{class_names[0]}", f"pred_{class_names[1]}"],
        rows=[
            (f"true_{class_names[0]}", cm[0, 0], cm[0, 1]),
            (f"true_{class_names[1]}", cm[1, 0], cm[1, 1]),
        ]
    )

    # 3) ROC PNG + CSV (puntos)
    roc_png = os.path.join(OUT_DIR, f"{split_name}_roc.png")
    save_roc_png(fpr, tpr, roc_auc, roc_png, f"ROC Curve ({split_name})")

    roc_csv = os.path.join(OUT_DIR, f"{split_name}_roc_points.csv")
    write_csv(
        roc_csv,
        header=["fpr", "tpr", "threshold"],
        rows=list(zip(fpr.tolist(), tpr.tolist(), thresholds.tolist()))
    )

    # 4) Classification report TXT + CSV
    rep_txt_path = os.path.join(OUT_DIR, f"{split_name}_classification_report.txt")
    with open(rep_txt_path, "w", encoding="utf-8") as f:
        f.write(rep_txt)

    rep_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    rep_csv = os.path.join(OUT_DIR, f"{split_name}_classification_report.csv")
    rows = []
    for label, vals in rep_dict.items():
        if isinstance(vals, dict) and "precision" in vals:
            rows.append((label, vals["precision"], vals["recall"], vals["f1-score"], vals["support"]))
    write_csv(rep_csv, header=["label", "precision", "recall", "f1", "support"], rows=rows)

    # 5) Summary split
    summary_csv = os.path.join(OUT_DIR, f"{split_name}_summary.csv")
    write_csv(
        summary_csv,
        header=["metric", "value"],
        rows=[
            ("loss", base.get("loss", "")),
            ("accuracy", base.get("accuracy", "")),
            ("auc_tf", base.get("auc", "")),
            ("auc_sklearn", roc_auc),
            ("tp", tp),
            ("tn", tn),
            ("fp", fp),
            ("fn", fn),
            ("threshold", THRESHOLD),
        ]
    )

    return {
        "loss": float(base.get("loss", np.nan)),
        "accuracy": float(base.get("accuracy", np.nan)),
        "auc_tf": float(base.get("auc", np.nan)),
        "auc_sklearn": float(roc_auc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

# ======================
# RUN EVALUATION
# ======================
res_train = evaluate_split("train", train_ds)
res_val   = evaluate_split("valid", val_ds)
res_test  = evaluate_split("test", test_ds)

# ======================
# FINAL SUMMARY (TFG-FRIENDLY)
# ======================
print("\n==============================")
print("📌 RESUMEN FINAL")
print("==============================")
for name, res in [("Train", res_train), ("Validation", res_val), ("Test", res_test)]:
    print(f"\n{name}:")
    print(f"  Loss: {res['loss']:.4f}")
    print(f"  Accuracy: {res['accuracy']:.4f}")
    print(f"  AUC (TF): {res['auc_tf']:.4f}")
    print(f"  AUC (sklearn): {res['auc_sklearn']:.4f}")
    print(f"  FP: {res['fp']} | FN: {res['fn']}")

print("\n✅ Todo guardado en:", OUT_DIR)
