import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)

# ======================
# CONFIG
# ======================
DATA_DIR = "/Users/pepedesintas/Desktop/TFG/CBIS_DDSM/processed"
MODEL_PATH = "../models/resnet50_cbis_final.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ======================
# LOAD DATASETS
# ======================
def load_dataset(split):
    return tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

train_ds = load_dataset("train")
val_ds   = load_dataset("valid")
test_ds  = load_dataset("test")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado")

# ======================
# EVALUATION FUNCTION
# ======================
def evaluate_split(model, dataset, split_name):
    print(f"\n==============================")
    print(f"📊 Evaluación en {split_name.upper()}")
    print(f"==============================")

    # Métricas básicas
    loss, acc, auc_score = model.evaluate(dataset, verbose=0)
    print(f"Loss:      {loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc_score:.4f}")

    # Predicciones
    y_prob = model.predict(dataset).ravel()
    y_true = np.concatenate([y.numpy() for _, y in dataset])

    # Umbral
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    # ======================
    # CONFUSION MATRIX
    # ======================
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP (Benign → Malignant): {fp}")
    print(f"FN (Malignant → Benign): {fn}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.show()

    # ======================
    # ROC CURVE
    # ======================
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({split_name})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # ======================
    # CLASSIFICATION REPORT
    # ======================
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"]
    ))

    return {
        "loss": loss,
        "accuracy": acc,
        "auc": auc_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

# ======================
# RUN EVALUATION
# ======================
results_train = evaluate_split(model, train_ds, "train")
results_val   = evaluate_split(model, val_ds, "validation")
results_test  = evaluate_split(model, test_ds, "test")

# ======================
# SUMMARY (TFG FRIENDLY)
# ======================
print("\n==============================")
print("📌 RESUMEN FINAL")
print("==============================")

for name, res in zip(
    ["Train", "Validation", "Test"],
    [results_train, results_val, results_test]
):
    print(f"\n{name}:")
    print(f"  Loss: {res['loss']:.4f}")
    print(f"  Accuracy: {res['accuracy']:.4f}")
    print(f"  AUC: {res['auc']:.4f}")
    print(f"  FP: {res['fp']} | FN: {res['fn']}")
