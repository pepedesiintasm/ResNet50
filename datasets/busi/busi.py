import os #Nos sirve para trabajar con rutas, carpetas y archivos
import random
import shutil # para copiar, mover o borrar archivos
from pathlib import Path # para manejar rutas de archivos completas

# CONFIGURATION:
RAW_DIR = "/Users/pepedesintas/Desktop/TFG/Dataset_BUSI_with_GT" # busi original descargado de kaggle
OUT_DIR = "/Users/pepedesintas/Desktop/TFG/BUSI_processed"
SEED = 42 # esto es una semilla aleatoria. Sirve porq si ejecutas el script otra vez, cuando se haga el shuffle, sea el mismo. Como que parte de esa semilla
SPLIT = (0.7, 0.15, 0.15)  # train/valid/test

CLASSES = ["benign", "malignant"]  # quitamos "normal" pq lo queremos binario

random.seed(SEED) # fijamos el comportmiento aleatorio con esa semilla

def is_mask_file(p: Path) -> bool:
    name = p.name.lower()
    return "mask" in name  # Este metodo nos sirve para eliminar las mascaras (este trabajo es de clasificacacion binaria, no de segmentación, por eso se quitan)

def list_images(class_dir: Path): #listamos las imagenes que tenemis en cada directorio de RAW para ver el formato
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in class_dir.iterdir() if p.suffix.lower() in exts and not is_mask_file(p)]
    return sorted(files)

def split_list(items, split=SPLIT):
    n = len(items) # items = lista de imagenes
    n_train = int(n * split[0]) # calcula cuantas imagenes van a training
    n_val = int(n * split[1]) # calcula cuantas imagenes van a validation
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    test = items[n_train+n_val:]
    return train, val, test


for split_name in ["train", "valid", "test"]:
    for c in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, split_name, c), exist_ok=True) # creamos las 3 carpetas, en el directorio de salida


for c in CLASSES: # bucle por cada clase
    class_path = Path(RAW_DIR) / c # construye la ruta de la carpeta original de esa clase c = "benign"
    files = list_images(class_path) # guardamos la lsita que nos ha salido en la ruta creada, excluyendo las máscaras
    random.shuffle(files) # mezclamos las imágenes de forma aleatoria antes de dividirlas en splits

    train, val, test = split_list(files)

    for p in train:
        shutil.copy2(p, os.path.join(OUT_DIR, "train", c, p.name)) # shutil, para copiar las imagenes de la carpeta original a la de output
    for p in val:
        shutil.copy2(p, os.path.join(OUT_DIR, "valid", c, p.name))
    for p in test:
        shutil.copy2(p, os.path.join(OUT_DIR, "test", c, p.name))

    print(f"{c}: total={len(files)} train={len(train)} val={len(val)} test={len(test)}")

print("--> BUSI preparado en:", OUT_DIR)
