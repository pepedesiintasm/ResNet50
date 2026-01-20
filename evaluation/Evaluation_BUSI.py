import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
DATA_DIR   = "/Users/pepedesintas/Desktop/TFG/BUSI_processed"   # <-- cambia
OUT_DIR    = "../results_busi"                                  # carpeta para PNG/CSV/TXT
MODEL_DIR  = "../models"                                                  # carpeta modelos
MODEL_NAME = "resnet50_busi_best.h5"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 20
SEED       = 42

# Si es binario benign/malignant:
LABEL_MODE = "binary"   # "binary" (o "int" si 3 clases)
NUM_CLASSES = 1         # 1 binario; si 3 clases pon 3
# ======================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibilidad
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ======================
# DATASETS
# ======================
def load_ds(split, shuffle=False):
    return tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        label_mode=LABEL_MODE,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None
    )

train_ds = load_ds("train", shuffle=True)
val_ds   = load_ds("valid", shuffle=False)
test_ds  = load_ds("test", shuffle=False)

class_names = train_ds.class_names
print("Clases:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ======================
# DATA AUGMENTATION
# ======================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
], name="data_augmentation")

# ======================
# MODEL
# ======================
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

# Fine-tuning: congela primeras capas
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

if NUM_CLASSES == 1:
    outputs = layers.Dense(1, activation="sigmoid")(x)
else:
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

if NUM_CLASSES == 1:
    loss_fn = "binary_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
else:
    loss_fn = "sparse_categorical_crossentropy"
    metrics = ["accuracy"]  # AUC multiclase requiere configuración extra

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=loss_fn,
    metrics=metrics
)

model.summary()

# ======================
# CALLBACK: log test metrics each epoch
# ======================
class TestMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        results = self.model.evaluate(self.test_data, verbose=0, return_dict=True)

        # Guardar con prefijo test_
        for k, v in results.items():
            key = f"test_{k}"
            logs[key] = v
            self.history.setdefault(key, []).append(v)

test_cb = TestMetricsCallback(test_ds)

# ======================
# CALLBACKS
# ======================
best_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.h5")
final_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.h5")

monitor_metric = "val_auc" if (NUM_CLASSES == 1) else "val_accuracy"
monitor_mode   = "max"

callbacks = [
    EarlyStopping(
        monitor=monitor_metric,
        patience=6,
        mode=monitor_mode,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        best_path,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_best_only=True
    ),
    test_cb
]

# ======================
# TRAINING
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Guardar modelo final
model.save(final_path)
print("✅ Modelo guardado:", final_path)

# ======================
# SAVE HISTORY CSV
# ======================
hist_dict = history.history.copy()
# Asegura que existan en hist_dict las keys de test_cb (por si keras no las incluye)
for k, v in test_cb.history.items():
    hist_dict[k] = v

# Convertir a CSV sin pandas (para no depender)
csv_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_history.csv")
keys = list(hist_dict.keys())
num_rows = len(hist_dict[keys[0]]) if keys else 0

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("epoch," + ",".join(keys) + "\n")
    for i in range(num_rows):
        row = [str(i + 1)]
        for k in keys:
            row.append(str(hist_dict[k][i]) if i < len(hist_dict[k]) else "")
        f.write(",".join(row) + "\n")

print("✅ History CSV:", csv_path)

# Guardar también JSON por comodidad
json_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_history.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(hist_dict, f, indent=2)
print("✅ History JSON:", json_path)

# ======================
# PLOTS (PNG): Loss curve + Learning curves
# ======================
def save_plot(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("✅ PNG:", out_path)

def plot_metric(metric_name, title, filename):
    fig = plt.figure(figsize=(7, 5))

    # Train
    if metric_name in hist_dict:
        plt.plot(hist_dict[metric_name], label=f"train_{metric_name}")

    # Val (Keras usa 'val_<metric>')
    val_key = f"val_{metric_name}"
    if val_key in hist_dict:
        plt.plot(hist_dict[val_key], label=val_key)

    # Test (nuestro callback usa 'test_<metric>')
    test_key = f"test_{metric_name}"
    if test_key in hist_dict:
        plt.plot(hist_dict[test_key], label=test_key)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()

    save_plot(fig, os.path.join(OUT_DIR, filename))

# 1) Loss curve (train/val/test)
plot_metric("loss", "Loss curve (train / val / test)", f"{MODEL_NAME}_loss_curve.png")

# 2) Learning curve (Accuracy)
plot_metric("accuracy", "Learning curve - Accuracy (train / val / test)", f"{MODEL_NAME}_learning_curve_accuracy.png")

# 3) Learning curve (AUC) si binario
if NUM_CLASSES == 1:
    plot_metric("auc", "Learning curve - AUC (train / val / test)", f"{MODEL_NAME}_learning_curve_auc.png")

# ======================
# FINAL EVALUATION ON TEST + EXTRA METRICS
# ======================
print("\n🧪 Evaluación final en test:")
test_eval = model.evaluate(test_ds, return_dict=True)
print(test_eval)

# Predicciones
y_prob = model.predict(test_ds).ravel()
y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)

# Umbral binario
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# ---- Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

fig = plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(values_format="d")
plt.title("Confusion Matrix (test)")
save_plot(fig, os.path.join(OUT_DIR, f"{MODEL_NAME}_confusion_matrix_test.png"))

# ---- ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = sk_auc(fpr, tpr)

fig = plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (test)")
plt.legend(loc="lower right")
plt.grid(True)
save_plot(fig, os.path.join(OUT_DIR, f"{MODEL_NAME}_roc_test.png"))

# ---- Classification report (TXT + CSV)
report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
txt_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_classification_report_test.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(report_txt)
print("✅ Report TXT:", txt_path)

# CSV simple del report
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_csv_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_classification_report_test.csv")
with open(report_csv_path, "w", encoding="utf-8") as f:
    f.write("label,precision,recall,f1-score,support\n")
    for label, vals in report_dict.items():
        if isinstance(vals, dict) and "precision" in vals:
            f.write(f"{label},{vals['precision']},{vals['recall']},{vals['f1-score']},{vals['support']}\n")
print("✅ Report CSV:", report_csv_path)

# ---- Resumen métricas finales (CSV)
summary_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_test_summary.csv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    for k, v in test_eval.items():
        f.write(f"{k},{v}\n")
    f.write(f"tn,{tn}\nfp,{fp}\nfn,{fn}\ntp,{tp}\n")
    f.write(f"roc_auc,{roc_auc}\n")

print("✅ Summary CSV:", summary_path)
print("\n✅ Todo listo en:", OUT_DIR)
