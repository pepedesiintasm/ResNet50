import os
import random
import shutil
from pathlib import Path

# ======================
# CONFIG
# ======================
RAW_DIR = "/Users/pepedesintas/Desktop/TFG/Dataset_BUSI_with_GT"
OUT_DIR = "/Users/pepedesintas/Desktop/TFG/BUSI_processed"
SEED = 42
SPLIT = (0.7, 0.15, 0.15)  # train/valid/test

CLASSES = ["benign", "malignant"]  # quitamos "normal" pq lo queremos binario

random.seed(SEED)

def is_mask_file(p: Path) -> bool:
    name = p.name.lower()
    return "mask" in name  # BUSI usa *_mask.png normalmente

def list_images(class_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in class_dir.iterdir() if p.suffix.lower() in exts and not is_mask_file(p)]
    return sorted(files)

def split_list(items, split=SPLIT):
    n = len(items)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    test = items[n_train+n_val:]
    return train, val, test

# Crear carpetas
for split_name in ["train", "valid", "test"]:
    for c in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, split_name, c), exist_ok=True)

# Copiar
for c in CLASSES:
    class_path = Path(RAW_DIR) / c
    files = list_images(class_path)
    random.shuffle(files)

    train, val, test = split_list(files)

    for p in train:
        shutil.copy2(p, os.path.join(OUT_DIR, "train", c, p.name))
    for p in val:
        shutil.copy2(p, os.path.join(OUT_DIR, "valid", c, p.name))
    for p in test:
        shutil.copy2(p, os.path.join(OUT_DIR, "test", c, p.name))

    print(f"{c}: total={len(files)} train={len(train)} val={len(val)} test={len(test)}")

print("✅ BUSI preparado en:", OUT_DIR)
