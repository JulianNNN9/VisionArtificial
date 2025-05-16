import os
import cv2
import numpy as np
from glob import glob

# Definir transformaciones
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def apply_gaussian_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)

def flip_horizontal(image):
    return cv2.flip(image, 1)

def apply_erosion(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_dilation(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def change_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# Directorio actual
path = os.getcwd()
image_extensions = ['*.jpg', '*.jpeg', '*.png']
image_paths = []

# Obtener todas las imágenes
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(path, ext)))

# Crear aumentaciones
for image_path in image_paths:
    filename = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)

    if image is None:
        continue

    # Aplicar transformaciones
    transformations = {
        '_rotated_15': rotate_image(image, 15),
        '_rotated_-15': rotate_image(image, -15),
        '_flipped': flip_horizontal(image),
        '_blurred': apply_gaussian_blur(image),
        '_eroded': apply_erosion(image),
        '_dilated': apply_dilation(image),
        '_brighter': change_brightness(image, 40),
        '_darker': change_brightness(image, -40),
    }

    # Guardar imágenes transformadas
    for suffix, transformed_image in transformations.items():
        output_filename = f"{filename}{suffix}.jpg"
        output_path = os.path.join(path, output_filename)
        cv2.imwrite(output_path, transformed_image)

print("Aumentación de imágenes completada.")
