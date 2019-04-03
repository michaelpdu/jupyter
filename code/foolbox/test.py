import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.nets import vgg
from foolbox.models import TensorFlowModel
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack
import matplotlib.pyplot as plt


def open_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    return image

def vgg_preprocessing(images):
    return images - [123.68, 116.78, 103.94]

def main(image_path, ckpt_path):
    images = tf.placeholder(tf.float32, (None, 224, 224, 3))
    preprocessed = vgg_preprocessing(images)
    logits, _ = vgg.vgg_19(preprocessed, is_training=False)
    restorer = tf.train.Saver(tf.trainable_variables())
    image = open_image(image_path)

    with tf.Session() as session:
        restorer.restore(session, ckpt_path)
        model = TensorFlowModel(images, logits, (0, 255))
        label = np.argmax(model.predictions(image))
        print('label:', label)

        target_class = 22
        criterion = TargetClassProbability(target_class, p=0.99)
        attack = LBFGSAttack(model, criterion)

        adversarial = attack(image, label=label)

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.imshow(adversarial)
        plt.subplot(1, 3, 3)
        plt.imshow(adversarial - image)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])