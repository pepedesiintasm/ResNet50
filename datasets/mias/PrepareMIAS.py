import os
import random
import cv2 # leery procesar imagenes

BASE_DIR = "/Users/pepedesintas/Desktop/TFG/DDBB/all-mias"
ANNOT_FILE = os.path.join(BASE_DIR, "mias_classification.txt") # ruta del archivo de anotaciones
IMG_DIR = BASE_DIR

OUT_STAGE1 = os.path.join(BASE_DIR, "stage1_normal_vs_lesion")
OUT_STAGE2 = os.path.join(BASE_DIR, "stage2_benign_vs_malignant")

IMG_SIZE = 224 # todas las imagenes/ROIs se redimensionan a 224x224 pixeles, tamaño de entrada de resnet y densenet
ROI_SCALE = 2.5 # cuanto contexto alrededor de la lesion se recorta, es decir, recortamos mas tejido q solo la region
IMG_HEIGHT = 1024 # altura de las imagnes originales para corregir coordenadas

TRAIN = 0.7
VAL = 0.15

# para encontrar imagenes
def find_img(img_id):
    for ext in [".pgm", ".png", ".jpg"]:
        p = os.path.join(IMG_DIR, img_id + ext)
        if os.path.exists(p):
            return p
    return None


# funcion para extraer la ROI
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


# MAIN
#Listas para guardar imagenes
normals = []
lesions = []
benign = []
malignant = []

# solo nos centraremos en las benignas y malignas
with open(ANNOT_FILE) as f:
    for line in f: # lee el archivo por lineas y lo divide en partes por cad alinea (id imagen, tipo tejido, tipo anomalia...)
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

        sev = p[3]  # Extraemos severidad ,maligno benigno
        try:
            x, y, r = int(p[4]), int(p[5]), int(p[6]) # extraemos coordenadas y radio para luego sacar ROI
        except:
            continue

        if sev == "B":
            benign.append((img_path, x, y, r)) # guardamos en lista segun severidad
        elif sev == "M":
            malignant.append((img_path, x, y, r))

# SPLITS
random.shuffle(normals)
random.shuffle(lesions)
random.shuffle(benign)
random.shuffle(malignant)

n_tr, n_va, n_te = split(normals)
l_tr, l_va, l_te = split(lesions)
b_tr, b_va, b_te = split(benign)
m_tr, m_va, m_te = split(malignant)

# STAGE 1: NORMAL vs LESION
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

# STAGE 2: BENIGN vs MALIGNANT
mkdirs(OUT_STAGE2, ["benign", "malignant"])

def save_roi(data, split, cls):
    for i, (path,x,y,r) in enumerate(data):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # carga mamografia completa en escala de grises
        roi = extract_roi(img, x, y, r) # saca la ROI
        if roi is None:
            continue
        out = os.path.join(OUT_STAGE2, split, cls, f"{cls}_{i}.png")
        cv2.imwrite(out, roi) # guarda la ROI en el directorio final

save_roi(b_tr, "train", "benign")
save_roi(m_tr, "train", "malignant")
save_roi(b_va, "valid", "benign")
save_roi(m_va, "valid", "malignant")
save_roi(b_te, "test",  "benign")
save_roi(m_te, "test",  "malignant")

print(" Dataset MIAS en dos etapas creado")
