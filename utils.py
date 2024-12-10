import cv2
import numpy as np

def save_image(path, img):
    max_value = np.max(img)
    min_value = np.min(img)

    img = (img - min_value) / (max_value - min_value) * 256

    print(img.shape)
    cv2.imwrite(path, img.astype(np.uint8))
