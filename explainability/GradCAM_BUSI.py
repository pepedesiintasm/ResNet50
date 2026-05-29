import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kimage

MODEL_PATH = "/Users/pepedesintas/Desktop/TFG/Results/ResNet50/ResNet50_SeparateDDBB/BUSI/busi_resnet50_finetuning/model/best_model.keras"

IMAGE_PATH = "/Users/pepedesintas/Desktop/busi5.png"

OUTPUT_DIR = "/Users/pepedesintas/Desktop/TFG/GradCAM"
IMG_SIZE   = (224, 224)
LAST_CONV_LAYER = "conv5_block3_out"

os.makedirs(OUT_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_img_rgb(path):
    img = kimage.load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    arr = kimage.img_to_array(img)  # (H,W,3) float32 0..255
    return arr

def make_gradcam_heatmap(img_array):
    x = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x, training=False)  # <-- importante
        score = pred[:, 0]  # binario (1,1)

    grads = tape.gradient(score, conv_out)                 # (1,h,w,c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (c,)

    conv_out = conv_out[0]                                 # (h,w,c)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(pred.numpy().ravel()[0])

def save_imgs(img_array, heatmap, prefix, alpha=0.35):
    heatmap_resized = tf.image.resize(heatmap[..., None], IMG_SIZE).numpy().squeeze()

    # opcional: resaltar solo lo más “importante”
    # heatmap_resized = np.clip((heatmap_resized - 0.5) / 0.5, 0, 1)

    # original
    fig = plt.figure(figsize=(5,5))
    plt.imshow(img_array.astype("uint8"))
    plt.axis("off")
    plt.title("Original")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{prefix}_original2.png"), dpi=200)
    plt.close(fig)

    # heatmap
    fig = plt.figure(figsize=(5,5))
    plt.imshow(heatmap_resized)
    plt.axis("off")
    plt.title("Grad-CAM heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{prefix}_heatmap2.png"), dpi=200)
    plt.close(fig)

    # overlay
    fig = plt.figure(figsize=(5,5))
    plt.imshow(img_array.astype("uint8"))
    plt.imshow(heatmap_resized, alpha=alpha)
    plt.axis("off")
    plt.title("Grad-CAM overlay")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{prefix}_overlay2.png"), dpi=200)
    plt.close(fig)

img_array = load_img_rgb(IMG_PATH)
heatmap, prob = make_gradcam_heatmap(img_array)

base = os.path.splitext(os.path.basename(IMG_PATH))[0]
save_imgs(img_array, heatmap, base)

print("Prob:", prob)
print("Guardado en:", OUT_DIR)
