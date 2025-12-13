import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

#DATA_DIR = "/Users/pepedesintas/Desktop/TFG/all-mias/outputData"
DATA_DIR = "/Users/pepedesintas/Desktop/TFG/all-mias/stage1_normal_vs_lesion"
#DATA_DIR = "/Users/pepedesintas/Desktop/TFG/all-mias/stage2_benign_vs_malignant"


IMG_SIZE = (224, 224) # we use this size, bc ResNet50 uses this standard size to train
# permits the use of pretrained weights of ImageNet
# lower computational cost
BATCH_SIZE = 16 # quantity of images processed at a time
EPOCHS = 20 # Not too much, as we are using transfer learning and it's already trained in ImageNet
# If we train too much, we can produce overfitting -> Mias is small




# We read images created in folders, and we assign them in order the name of the folder
# it converts them in tensors (data save form in various dimensions)
# Tensors allow the GPU/CPU to process all that data in parallel and very quickly, as it has many dimensions (heigh, channels, nº images...)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True # for a random train (more generalization)
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

# we save data in memory, so in every epoch we don't have to go back and search for the data again
# .prefetch() -> prepares next batch while CNN is training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)


# Artificially increases the dataset
# it Helps the model generalize better
# Reduces overfitting
#TODO we have augmented the augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    #layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    #layers.RandomContrast(0.2),
    #layers.RandomBrightness(0.2),
])
# in mammograms it is not always advisable to use horizontal flips and rotations, since they can alter the anatomical meaning.


inputs = layers.Input(shape=(224,224,3)) # defines the way CNN receives the images (3 -> channel RGB)
x = data_augmentation(inputs)
x = preprocess_input(x) #normalizes using standard ResNet50 things


# Convolutional part of the ResNet50 (all layers)
base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
#Freezes the weights of ResNet50
#They are not modified during training
#Prevents overfitting (ideal with a small dataset)
base_model.trainable = False

# Reduces the feature maps to a vector → very efficient.
x = layers.GlobalAveragePooling2D()(base_model.output)
# Dense layer to learn more abstract patterns.
#x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
# Randomly turns off 40% of neurons → prevents overfitting.
# Reduced to 15% to do not lose info
#x = layers.Dropout(0.15)(x)
x = layers.Dropout(0.3)(x)
# Final output → probability of abnormality (binary).
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy", # loss function (binary-> 2 classes: normal, abnormal)
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

callbacks = [
    # Stops training if the AUC does not improve for 5 epochs → prevents overfitting.
    EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
    #saves the best model according to validation
    ModelCheckpoint("/Users/pepedesintas/PycharmProjects/ResNet50/models/resnet50_mias_prueba.h5", monitor="val_auc", mode="max", save_best_only=True)
]

# -> TRAINING (the real learning)
# only our final layers are trained
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("-> Evaluating on test set:\n")
model.evaluate(test_ds)
