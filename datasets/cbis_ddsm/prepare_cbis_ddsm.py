import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/pepedesintas/Desktop/TFG/CBIS_DDSM"
CSV_DIR = os.path.join(BASE_DIR, "csv")
JPEG_DIR = os.path.join(BASE_DIR, "jpeg")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_csv(csv_file):
    df = pd.read_csv(csv_file)

    df = df[["cropped image file path", "pathology"]]
    df = df.dropna()

    # Normalizar etiquetas
    df["label"] = df["pathology"].apply(
        lambda x: "benign" if "BENIGN" in x else "malignant"
    )

    df["image_path"] = df["cropped image file path"].apply(
        lambda x: os.path.join(BASE_DIR, x.replace("CBIS-DDSM/", ""))
    )

    return df[["image_path", "label"]]


print("📄 Leyendo CSVs...")
train_df = process_csv(os.path.join(CSV_DIR, "mass_case_description_train_set.csv"))
test_df  = process_csv(os.path.join(CSV_DIR, "mass_case_description_test_set.csv"))

full_df = pd.concat([train_df, test_df], ignore_index=True)

print("📊 Distribución:")
print(full_df["label"].value_counts())

train_df, temp_df = train_test_split(
    full_df,
    test_size=0.3,
    stratify=full_df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

def copy_images(df, split):
    for _, row in df.iterrows():
        dst_dir = os.path.join(OUTPUT_DIR, split, row["label"])
        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(row["image_path"]):
            shutil.copy(row["image_path"], dst_dir)
        else:
            print(f"[WARN] No existe {row['image_path']}")

print("📂 Copiando imágenes...")
copy_images(train_df, "train")
copy_images(val_df, "valid")
copy_images(test_df, "test")

print("✅ CBIS-DDSM preparado")
