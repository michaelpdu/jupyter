import numpy as np
from PIL import Image

def open_image(image_path, height=224, width=224):
    image = Image.open(image_path)
    image = image.resize((height, width))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    return image

def save_image(image, image_path):
    im = image.astype(np.uint8)
    Image.fromarray(im).save(image_path)

def vgg_preprocessing(images):
    return images - [123.68, 116.78, 103.94]

def inception_preprocessing(image):
    return image / 255.0
    # return (image / 255.0) * 2.0 - 1.0

def inception_postprocessing(image):
    return (image * 255.0).astype(np.uint8)
    # return (((image + 1.0) * 0.5) * 255.0).astype(np.uint8)