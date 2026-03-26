import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================
# CONFIGURATION
# ======================
DATA_DIR = "/Users/pepedesintas/Desktop/TFG/DDBB/CBIS_DDSM/processed"
IMG_SIZE = (224, 224)
# Probar BATCH de 32 (si no peta la RAM) -> menos steps = menos tiempo
BATCH_SIZE = 16
EPOCHS = 10
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

class_names = train_ds.class_names
print("Clases:", class_names)  # ['benign', 'malignant']

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ======================
# DATA AUGMENTATION
# ======================
# Se puede reducir para CBIS, es más grande que MIAS, no necesita tanta DA
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
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

# Congelar primeras capas
base_model.trainable = True #para que vaya más rápido podemos congelar y solo entrenar cabeza
for layer in base_model.layers[:100]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ======================
# CALLBACKS
# ======================
callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=6,
        mode="max",
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "../models/resnet50_cbis_best.h5",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
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

# ======================
# EVALUATION
# ======================
print("\n🧪 Evaluación en test:")
model.evaluate(test_ds)

model.save("../models/resnet50_cbis_final.h5")
print("✅ Modelo CBIS-DDSM guardado")
