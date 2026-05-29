import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "/Users/pepedesintas/Desktop/TFG/Results/ResNet50/ResNet50_SeparateDDBB/CBIS/cbis_ddsm_resnet50_finetuning/model/best_model.keras"
IMAGE_PATH = "/Users/pepedesintas/Desktop/2.jpg"
OUTPUT_DIR = "/Users/pepedesintas/Desktop/TFG/GradCAM"

IMG_SIZE = 224
THRESHOLD = 0.5

LAST_CONV_LAYER = "conv5_block3_out"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nÚltimas capas del modelo:")
for layer in model.layers[-15:]:
    print(layer.name)

last_conv_layer = model.get_layer(LAST_CONV_LAYER)

# =========================================================
# LOAD IMAGE
# =========================================================

print("\nCargando imagen...")

img_bgr = cv2.imread(IMAGE_PATH)

if img_bgr is None:
    raise ValueError(f"No se pudo cargar la imagen: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
original = img_rgb.copy()

img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# =========================================================
# PREDICTION
# =========================================================

pred = float(model.predict(img_array, verbose=0)[0][0])
label = "Malignant" if pred >= THRESHOLD else "Benign"

print(f"\nPredicción: {label}")
print(f"Probabilidad maligna: {pred:.4f}")

# =========================================================
# GRADCAM
# =========================================================

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0)
heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
heatmap = heatmap.numpy()

# =========================================================
# IMPROVE HEATMAP VISUALIZATION
# =========================================================

heatmap = cv2.resize(
    heatmap,
    (original.shape[1], original.shape[0])
)

heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=12, sigmaY=12)

p_low, p_high = np.percentile(heatmap, [10, 99])
heatmap = np.clip((heatmap - p_low) / (p_high - p_low + 1e-8), 0, 1)

heatmap = np.power(heatmap, 1.4)

heatmap_uint8 = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_TURBO
)

heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

alpha_map = 0.10 + 0.35 * heatmap[..., np.newaxis]

overlay = (
    (1 - alpha_map) * original.astype(np.float32)
    + alpha_map * heatmap_color.astype(np.float32)
)

overlay = np.uint8(np.clip(overlay, 0, 255))

# =========================================================
# SAVE
# =========================================================

base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

cv2.imwrite(
    os.path.join(OUTPUT_DIR, f"{base_name}_resnet50_overlay_tfg.png"),
    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, f"{base_name}_resnet50_heatmap_tfg.png"),
    heatmap_uint8
)

# =========================================================
# PLOT FOR TFG
# =========================================================

plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original image", fontsize=12)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title(
    f"Grad-CAM visualization\nPrediction: {label} | P(malignant) = {pred:.3f}",
    fontsize=12
)
plt.axis("off")

plt.tight_layout(pad=1.5)

plt.savefig(
    os.path.join(OUTPUT_DIR, f"{base_name}_resnet50_gradcam_tfg.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()

print("\nGrad-CAM ResNet50 generado correctamente.")
print(f"Resultados guardados en: {OUTPUT_DIR}")