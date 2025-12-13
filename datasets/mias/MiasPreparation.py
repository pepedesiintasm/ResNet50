import os
import random
import cv2

# =====================
# CONFIG
# =====================

BASE_DIR = "/Users/pepedesintas/Desktop/TFG/all-mias"
ANNOT_FILE = os.path.join(BASE_DIR, "mias_classification.txt")
IMG_DIR = BASE_DIR

OUT_STAGE1 = os.path.join(BASE_DIR, "stage1_normal_vs_lesion")
OUT_STAGE2 = os.path.join(BASE_DIR, "stage2_benign_vs_malignant")

IMG_SIZE = 224
ROI_SCALE = 2.5
IMG_HEIGHT = 1024

TRAIN = 0.7
VAL = 0.15

# =====================
# HELPERS
# =====================

def find_img(img_id):
    for ext in [".pgm", ".png", ".jpg"]:
        p = os.path.join(IMG_DIR, img_id + ext)
        if os.path.exists(p):
            return p
    return None


def extract_roi(img, x, y, r):
    y = IMG_HEIGHT - y
    half = int(ROI_SCALE * r)
    roi = img[max(0,y-half):min(img.shape[0],y+half),
              max(0,x-half):min(img.shape[1],x+half)]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE))


def split(data):
    n = len(data)
    t1 = int(TRAIN * n)
    t2 = int((TRAIN + VAL) * n)
    return data[:t1], data[t1:t2], data[t2:]


def mkdirs(base, classes):
    for s in ["train", "valid", "test"]:
        for c in classes:
            os.makedirs(os.path.join(base, s, c), exist_ok=True)

# =====================
# MAIN
# =====================

normals = []
lesions = []
benign = []
malignant = []

with open(ANNOT_FILE) as f:
    for line in f:
        p = line.strip().split()

        if len(p) < 3:
            continue

        img_id = p[0]
        img_path = find_img(img_id)
        if img_path is None:
            continue

        # -------- NORMAL --------
        if p[2] == "NORM":
            normals.append(img_path)
            continue

        # -------- LESION --------
        lesions.append(img_path)

        # sin ROI usable
        if len(p) < 7:
            continue

        sev = p[3]  # B o M
        try:
            x, y, r = int(p[4]), int(p[5]), int(p[6])
        except:
            continue

        if sev == "B":
            benign.append((img_path, x, y, r))
        elif sev == "M":
            malignant.append((img_path, x, y, r))

# =====================
# SPLITS
# =====================

random.shuffle(normals)
random.shuffle(lesions)
random.shuffle(benign)
random.shuffle(malignant)

n_tr, n_va, n_te = split(normals)
l_tr, l_va, l_te = split(lesions)
b_tr, b_va, b_te = split(benign)
m_tr, m_va, m_te = split(malignant)

# =====================
# STAGE 1: NORMAL vs LESION
# =====================

mkdirs(OUT_STAGE1, ["normal", "lesion"])

def save_full(data, split, cls):
    for i, path in enumerate(data):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        out = os.path.join(OUT_STAGE1, split, cls, f"{cls}_{i}.png")
        cv2.imwrite(out, img)

save_full(n_tr, "train", "normal")
save_full(l_tr, "train", "lesion")
save_full(n_va, "valid", "normal")
save_full(l_va, "valid", "lesion")
save_full(n_te, "test",  "normal")
save_full(l_te, "test",  "lesion")

# =====================
# STAGE 2: BENIGN vs MALIGNANT
# =====================

mkdirs(OUT_STAGE2, ["benign", "malignant"])

def save_roi(data, split, cls):
    for i, (path,x,y,r) in enumerate(data):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        roi = extract_roi(img, x, y, r)
        if roi is None:
            continue
        out = os.path.join(OUT_STAGE2, split, cls, f"{cls}_{i}.png")
        cv2.imwrite(out, roi)

save_roi(b_tr, "train", "benign")
save_roi(m_tr, "train", "malignant")
save_roi(b_va, "valid", "benign")
save_roi(m_va, "valid", "malignant")
save_roi(b_te, "test",  "benign")
save_roi(m_te, "test",  "malignant")

print("✅ Dataset MIAS en dos etapas creado")
