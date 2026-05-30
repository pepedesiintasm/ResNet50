import sys
import platform
import importlib.metadata


packages = [
    "numpy",
    "tensorflow",
    "matplotlib",
    "opencv-python",
]

print("Python version:")
print(sys.version)
print()

print("Python executable:")
print(sys.executable)
print()

print("Platform:")
print(platform.platform())
print()

print("Library versions:")

for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: not installed")

# Extra check for OpenCV import name
try:
    import cv2
    print(f"cv2 import version: {cv2.__version__}")
except ImportError:
    print("cv2: not importable")

# Extra check for TensorFlow / Keras
try:
    import tensorflow as tf
    print(f"TensorFlow import version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
except Exception as e:
    print(f"TensorFlow/Keras check failed: {e}")