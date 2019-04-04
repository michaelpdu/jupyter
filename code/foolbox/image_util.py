import numpy as np
from PIL import Image

def open_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    return image

def save_image(image, image_path):
    im = image.astype(np.uint8)
    Image.fromarray(im).save(image_path)

def vgg_preprocessing(images):
    return images - [123.68, 116.78, 103.94]