import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

MODEL_PATH = "/Users/pepedesintas/PycharmProjects/ResNet50_MiniMIAS/resnet50_mias.h5"
IMAGE_PATH = "/Users/pepedesintas/Desktop/TFG/all-mias/outputData/test/abnormal/mdb001.pgm"
IMG_SIZE = (224, 224)

# ---- LOAD MODEL ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- LOAD IMAGE ----
def load_and_prepare(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, img.numpy()

# ---- GRAD CAM ----
def make_gradcam_heatmap(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ---- RUN ----
img_tensor, orig_img = load_and_prepare(IMAGE_PATH)
img_tensor = tf.expand_dims(img_tensor, axis=0)

heatmap = make_gradcam_heatmap(img_tensor, model)

# ---- SUPERIMPOSE ----
heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed = heatmap_color * 0.4 + orig_img

# ---- DISPLAY ----
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(orig_img.astype("uint8"))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM")
plt.imshow(superimposed.astype("uint8"))
plt.axis("off")

plt.show()