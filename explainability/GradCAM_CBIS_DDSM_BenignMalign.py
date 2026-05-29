import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# =====================
# CONFIG
# =====================
MODEL_PATH = "/Users/pepedesintas/PycharmProjects/ResNet50/models/resnet50_cbis_final.h5"

# pon aquí UNA imagen concreta (benign o malignant)
#IMAGE_PATH = "/Users/pepedesintas/Desktop/TFG/CBIS_DDSM/processed/test/benign/1.3.6.1.4.1.9590.100.1.2.339524904810678180527824073070691579819_1-257.jpg"
IMAGE_PATH = "/Users/pepedesintas/Desktop/cbis.jpg"

IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "conv5_block3_out"  # ResNet50 estándar

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# LOAD & PREPROCESS IMAGE
# =====================
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, IMG_SIZE)

img_array = np.expand_dims(img_resized, axis=0).astype("float32")
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# =====================
# GRAD-CAM
# =====================
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

# Global average pooling of gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) + 1e-8

# =====================
# VISUALIZATION
# =====================
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# =====================
# PLOT
# =====================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(superimposed)
plt.axis("off")

plt.tight_layout()
plt.show()

# =====================
# PREDICTION INFO
# =====================
pred = predictions.numpy()[0][0]
print(f"Predicción (prob. malignidad): {pred:.3f}")
