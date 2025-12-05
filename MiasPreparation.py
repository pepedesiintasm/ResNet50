import os
import shutil
import random

IMAGE_DIRECTORY = "/Users/pepedesintas/Desktop/TFG/all-mias/" # directory where i have all the images
ANNOTATION_FILE = "/Users/pepedesintas/Desktop/TFG/all-mias/mias_classification.txt" # here we have the info related with the classifications and columns of the dataset
OUTPUT_DIRECTORY = "/Users/pepedesintas/Desktop/TFG/all-mias/outputData"

# ratios of split
TRAINING_DATA = 0.7
VALIDATION_DATA = 0.15
TESTING_DATA = 0.15

def mias_class_annotation(annotation_file):
    normal = []
    abnormal = []

    with open(annotation_file, "r") as f: # "r" means open the file in read mode
        for line in f.readlines():
            parts = line.strip().split() # Divides each lines with spaces
            # Ex: "mdb001 G CIRC B 535 425 197".split()
            # ["mdb001", "G", "CIRC", "B", "535", "425", "197"]
            # parts[0]	mdb001
            # parts[1]	G
            # ....

            if len(parts) < 3: # If there is a line malformed (not our case, just for prove)
                continue

            img_id = parts[0]
            abnormality_class = parts[2]

            filename = img_id + ".pgm"

            if abnormality_class == "NORM":
                normal.append(filename)
            else:
                abnormal.append(filename)

    return normal, abnormal

# Creates automatically all folders divided in ratios of split
def create_ratios_directories():
    for split in ["train", "valid", "test"]:
        for cls in ["normal", "abnormal"]:
            path = os.path.join(OUTPUT_DIRECTORY, split, cls)
            os.makedirs(path, exist_ok=True)


def shuffle(normal, abnormal):
    random.shuffle(normal)
    random.shuffle(abnormal)
    return normal, abnormal

def split_list(list):
    n = len(list)
    train_end = int(TRAINING_DATA * n)
    valid_end = int((VALIDATION_DATA + TRAINING_DATA) * n)
    train = list[:train_end]
    valid = list[train_end:valid_end]
    test = list[valid_end:]
    return train, valid, test

def copy_split_images(normal_splits, abnormal_splits):
    normal_train, normal_val, normal_test = normal_splits
    abnormal_train, abnormal_val, abnormal_test = abnormal_splits

    for split_name, normal_list, abnormal_list in [
        ("train", normal_train, abnormal_train),
        ("valid", normal_val, abnormal_val),
        ("test", normal_test, abnormal_test)
    ]:
        # Copiar normales
        for fname in normal_list:
            shutil.copy(os.path.join(IMAGE_DIRECTORY, fname), os.path.join(OUTPUT_DIRECTORY, split_name, "normal", fname))

        # Copiar anormales
        for fname in abnormal_list:
            shutil.copy(os.path.join(IMAGE_DIRECTORY, fname), os.path.join(OUTPUT_DIRECTORY, split_name, "abnormal", fname))

if __name__ == "__main__":
    print("Leyendo anotaciones...")
    normal_imgs, abnormal_imgs = mias_class_annotation(ANNOTATION_FILE)

    print("Mezclando listas...")
    normal_imgs, abnormal_imgs = shuffle(normal_imgs, abnormal_imgs)

    print("Dividiendo listas...")
    normal_train, normal_val, normal_test = split_list(normal_imgs)
    abnormal_train, abnormal_val, abnormal_test = split_list(abnormal_imgs)

    print("Creando carpetas...")
    create_ratios_directories()

    print("Copiando imágenes...")
    copy_split_images(normal_splits=(normal_train, normal_val, normal_test), abnormal_splits=(abnormal_train, abnormal_val, abnormal_test))

    print("¡Dataset listo!")