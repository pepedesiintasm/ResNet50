import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve
)

# =========================
# CONFIGURATION
# =========================

# DATA_DIR = "/Users/pepedesintas/Desktop/TFG/all-mias/outputData"
DATA_DIR = "/Users/pepedesintas/Desktop/TFG/DDBB/all-mias/stage1_normal_vs_lesion"
# DATA_DIR = "/Users/pepedesintas/Desktop/TFG/all-mias/stage2_benign_vs_malignant"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Entrenamiento en 2 fases
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Cuántas capas finales de ResNet50 se van a descongelar
FINE_TUNE_AT = 140

# Nombre del experimento
EXPERIMENT_NAME = "resnet50_stage1_finetuning"

# Carpetas de salida
BASE_OUTPUT_DIR = "/Users/pepedesintas/PycharmProjects/ResNet50/results"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME)
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")


# =========================
# DATASET LOADING
# =========================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)


# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    # layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    # layers.RandomContrast(0.2),
    # layers.RandomBrightness(0.2),
])


# =========================
# MODEL
# =========================

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

# FASE 1: congelado
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# Compilación inicial
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# =========================
# CALLBACKS
# =========================

callbacks_phase1 = [
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]


# =========================
# PHASE 1: TRAIN HEAD ONLY
# =========================

print("\n-> PHASE 1: training classification head with frozen ResNet50\n")

start_train_time = time.time()

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_phase1
)


# =========================
# PHASE 2: FINE-TUNING
# =========================

print("\n-> PHASE 2: fine-tuning ResNet50\n")

base_model.trainable = True

# Congelamos las capas iniciales y dejamos entrenables las últimas
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True

# Muy importante: BatchNormalization suele dejarse congelada en fine-tuning
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompilar con learning rate más pequeño
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=len(history_phase1.history["loss"]),
    callbacks=callbacks_phase2
)

end_train_time = time.time()
training_time_sec = end_train_time - start_train_time


# =========================
# MERGE HISTORY
# =========================

history_dict = {}

for key in history_phase1.history.keys():
    history_dict[key] = history_phase1.history[key] + history_phase2.history[key]

history_df = pd.DataFrame(history_dict)
history_df["epoch"] = np.arange(1, len(history_df) + 1)
history_df.to_csv(os.path.join(METRICS_DIR, "training_history.csv"), index=False)

print("\n-> Training finished")
print(f"Training time: {training_time_sec:.2f} seconds")


# =========================
# TEST EVALUATION (Keras)
# =========================

print("\n-> Evaluating on test set (Keras metrics):\n")
test_results = model.evaluate(test_ds, verbose=1)

keras_metrics = dict(zip(model.metrics_names, test_results))
print("Keras test metrics:", keras_metrics)


# =========================
# GET TRUE LABELS + PREDICTIONS
# =========================

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy().ravel())

y_true = np.array(y_true).astype(int)

y_prob = model.predict(test_ds).ravel()
y_pred = (y_prob >= 0.5).astype(int)


# =========================
# EXTRA METRICS (sklearn)
# =========================

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
balanced_acc = balanced_accuracy_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
sensitivity = recall

fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
precisions_curve, recalls_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)


# =========================
# SAVE PREDICTIONS
# =========================

predictions_df = pd.DataFrame({
    "y_true": y_true,
    "y_prob": y_prob,
    "y_pred": y_pred
})
predictions_df.to_csv(os.path.join(METRICS_DIR, "test_predictions.csv"), index=False)


# =========================
# SAVE CONFUSION MATRIX
# =========================

cm_df = pd.DataFrame(
    cm,
    index=["real_0", "real_1"],
    columns=["pred_0", "pred_1"]
)
cm_df.to_csv(os.path.join(METRICS_DIR, "confusion_matrix.csv"))


# =========================
# SAVE ROC CURVE
# =========================

roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": np.append(roc_thresholds, np.nan)[:len(fpr)]
})
roc_df.to_csv(os.path.join(METRICS_DIR, "roc_curve.csv"), index=False)


# =========================
# SAVE PR CURVE
# =========================

pr_df = pd.DataFrame({
    "precision": precisions_curve[:-1],
    "recall": recalls_curve[:-1],
    "threshold": pr_thresholds
})
pr_df.to_csv(os.path.join(METRICS_DIR, "pr_curve.csv"), index=False)


# =========================
# SAVE FINAL METRICS
# =========================

final_metrics = {
    "experiment_name": EXPERIMENT_NAME,
    "data_dir": DATA_DIR,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "initial_epochs": INITIAL_EPOCHS,
    "fine_tune_epochs": FINE_TUNE_EPOCHS,
    "total_epochs_configured": TOTAL_EPOCHS,
    "epochs_trained": len(history_df),
    "fine_tune_at_layer": FINE_TUNE_AT,
    "training_time_sec": training_time_sec,

    "test_loss_keras": float(keras_metrics.get("loss", 0.0)),
    "test_accuracy_keras": float(keras_metrics.get("accuracy", 0.0)),
    "test_auc_keras": float(keras_metrics.get("auc", 0.0)),
    "test_precision_keras": float(keras_metrics.get("precision", 0.0)),
    "test_recall_keras": float(keras_metrics.get("recall", 0.0)),

    "accuracy": float(acc),
    "precision": float(precision),
    "recall_sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "balanced_accuracy": float(balanced_acc),

    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

with open(os.path.join(METRICS_DIR, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

final_metrics_df = pd.DataFrame([final_metrics])
final_metrics_df.to_csv(os.path.join(METRICS_DIR, "final_metrics.csv"), index=False)


# =========================
# PRINT SUMMARY
# =========================

print("\n-> FINAL TEST METRICS")
for k, v in final_metrics.items():
    print(f"{k}: {v}")

print("\nFiles saved in:", OUTPUT_DIR)