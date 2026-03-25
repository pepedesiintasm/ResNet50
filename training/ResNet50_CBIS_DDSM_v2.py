import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================
# CONFIG
# ======================
DATA_DIR = "/Users/pepedesintas/Desktop/TFG/DDBB/CBIS_DDSM/processed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42

# ======================
# DATASETS
# ======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ======================
# DATA AUGMENTATION (CONTROLADA)
# ======================
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.1),
])

# ======================
# MODEL
# ======================
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

# 🔒 FASE 1: congelar TODO
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# ======================
# COMPILE (FASE 1)
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ======================
# CALLBACKS
# ======================
callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "../models/resnet50_cbis_improved.h5",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

# ======================
# TRAINING – FASE 1
# ======================
print("\n🔹 FASE 1: entrenando cabeza")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# ======================
# FASE 2: fine-tuning SUAVE
# ======================
print("\n🔹 FASE 2: fine-tuning")

for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

# ======================
# EVALUATION
# ======================
print("\n🧪 Evaluación en test:")
model.evaluate(test_ds)

model.save("../models/resnet50_cbis_finalV2.h5")
print("✅ Modelo CBIS-DDSM mejorado guardado")
