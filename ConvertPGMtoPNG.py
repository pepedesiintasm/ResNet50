import os
from PIL import Image

DIR = "/Users/pepedesintas/Desktop/TFG/all-mias/outputData"

def convert_pgm_to_png(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            pgm_path = os.path.join(subdir, file)
            png_path = pgm_path.replace(".pgm", ".png")

            try:
                im = Image.open(pgm_path)
                im.save(png_path)
                print(f"Converted: {pgm_path} -> {png_path}")
            except Exception as e:
                print(f"Error converting images! -> {e}")

def delete_pgm_files(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pgm"):
                os.remove(os.path.join(subdir, file))
                print(f"Deleted: ", file)

if __name__ == "__main__":
    convert_pgm_to_png(DIR)
    delete_pgm_files(DIR)