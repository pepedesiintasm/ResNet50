import os # manipular archivos, rutas, driectorios
import shutil # copiar archivos
import pandas as pd # para leer y manipular tablas
from sklearn.model_selection import train_test_split


BASE_DIR = "/Users/pepedesintas/Desktop/TFG/CBIS_DDSM"
CSV_DIR  = os.path.join(BASE_DIR, "csv")
JPEG_DIR = os.path.join(BASE_DIR, "jpeg") # aquí están las carpetas UID con las imagenes jpeg
OUT_DIR  = os.path.join(BASE_DIR, "processed")

os.makedirs(OUT_DIR, exist_ok=True) # crea la carpeta final processed


# Problema principal de CBIS - las rutas de los .csv contienen UID y necesitamos encontrar donde esta ese UID para localizar la carpeta con ese UID y extraer la imagen
# Devuelve TODOS los segmentos UID (1.3.6...) presentes en la ruta del CSV.
def extract_uid_candidates(path: str):
    if not isinstance(path, str): # recibe una ruta en formato texto
        return []
    parts = path.replace("\\", "/").split("/") # normalizacion para encontrar la ruta que toca
    uids = [p for p in parts if p.startswith("1.3.6.1.4.1.9590")] # todos empiezan con esta nomenclatura
    return uids  # devuelve todos los posibles UIDs q a encontrado en la ruta

# recibe una lista con todos los UIDs encontrados y comprueba que ralmente existe esa carpeta
def pick_existing_uid(candidates):
    for uid in reversed(candidates):  # primero intenta el último
        if os.path.isdir(os.path.join(JPEG_DIR, uid)):
            return uid
    return None


# funcion para leer archivos csv
def load_cases(csv_path):
    print(f"   -> Processing {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path) # lee el csv y lo guarda en un data frame

    # Columnas: en CBIS a veces la columna que queremos es "image file path" y a veces "cropped image file path" depende del csv
    possible_cols = ["cropped image file path", "image file path"]
    img_col = None
    for c in possible_cols:
        if c in df.columns:
            img_col = c
            break
    if img_col is None:
        raise RuntimeError(f"No encuentro columna de path en {csv_path}. Columnas: {list(df.columns)}")

    if "pathology" not in df.columns:
        raise RuntimeError(f"No encuentro columna 'pathology' en {csv_path}. Columnas: {list(df.columns)}")

    df = df[[img_col, "pathology"]].dropna() # aquí reducimo el data frame en dos columnas, la ruta de la imagen y la patologia

    # crea columna Label para benign/malignant (ojo: BENIGN_WITHOUT_CALLBACK cuenta como benign)
    df["label"] = df["pathology"].astype(str).str.upper().apply(
        lambda x: "malignant" if "MALIGNANT" in x else "benign"
    )

    # sacar UID correcto (el que exista en JPEG_DIR). Necesitamos sencontrar la carpeta de imagenes correspondiente a cada fila del csv
    df["uid_candidates"] = df[img_col].apply(extract_uid_candidates)
    df["uid"] = df["uid_candidates"].apply(pick_existing_uid)
    df = df[df["uid"].notnull()].copy()

    return df[["uid", "label"]] # Nos devuelve el UID y la patologia - lo único q necesitamos para despues copiar las imagenes en el OUTPUTDIR



# SCRIPT FUNCIONAL:
# Cargamos todos los .csv para extraer unicamente  las filas UID y LABEL (patologia) con la función anterior
print("-> Reading CSVs (mass)...")
mass_train = load_cases(os.path.join(CSV_DIR, "mass_case_description_train_set.csv"))
mass_test  = load_cases(os.path.join(CSV_DIR, "mass_case_description_test_set.csv"))
print("-> Reading CSVs (calc)...")
calc_train = load_cases(os.path.join(CSV_DIR, "calc_case_description_train_set.csv"))
calc_test  = load_cases(os.path.join(CSV_DIR, "calc_case_description_test_set.csv"))

df = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True) # metemos all df en un df para luego hacer el split que queremos

print("-> Distribution:")
print(df["label"].value_counts()) # muestra los casos que hay de cada clase

if len(df) == 0:
    raise RuntimeError("❌ No se ha podido asociar ninguna fila del CSV con carpetas UID en jpeg/.")

# debug
sample_uid = df.iloc[0]["uid"]
print("\n UID ejemplo usado:", sample_uid)
print(" ¿Existe en jpeg?:", os.path.isdir(os.path.join(JPEG_DIR, sample_uid)))
print(" Ejemplo de JPG dentro:", os.listdir(os.path.join(JPEG_DIR, sample_uid))[:5])


# SPLITS (por UID, no por imagen)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

# copiamos las imagenes en los nuevo directorios
def copy_split(split_df, split_name):
    copied = 0
    missing = 0

    for _, row in split_df.iterrows():
        uid_dir = os.path.join(JPEG_DIR, row["uid"])
        if not os.path.isdir(uid_dir):
            missing += 1
            continue

        dst_dir = os.path.join(OUT_DIR, split_name, row["label"])
        os.makedirs(dst_dir, exist_ok=True)

        for f in os.listdir(uid_dir):
            if f.lower().endswith(".jpg"):
                src = os.path.join(uid_dir, f)
                dst = os.path.join(dst_dir, f"{row['uid']}_{f}")
                shutil.copy(src, dst)
                copied += 1

    print(f"   -> {split_name}: copied {copied} JPGs (missing_uid_dirs={missing})")


print("\n-> Copying images...")
copy_split(train_df, "train")
copy_split(val_df, "valid")
copy_split(test_df, "test")

print("\n CBIS-DDSM preparado correctamente")
print(" Dataset final en:", OUT_DIR)
