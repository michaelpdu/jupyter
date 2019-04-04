import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from foolbox.models import TensorFlowModel
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack
from foolbox.attacks import FGSM
from foolbox.attacks import MomentumIterativeAttack
from foolbox.attacks import SinglePixelAttack
from foolbox.attacks import LocalSearchAttack
import matplotlib.pyplot as plt
from image_util import *


def main(image_path, ckpt_path, predict_status=False):
    images = tf.placeholder(tf.float32, (None, 224, 224, 3))
    preprocessed = vgg_preprocessing(images)
    logits, _ = vgg.vgg_19(preprocessed, is_training=False)
    restorer = tf.train.Saver(tf.trainable_variables())

    image = open_image(image_path)
    p, ext = os.path.splitext(image_path)
    adv_path = p+'-adv'+ext
    pert_path = p+'-pert'+ext

    with tf.Session() as session:
        restorer.restore(session, ckpt_path)
        model = TensorFlowModel(images, logits, (0, 255))
        label = np.argmax(model.predictions(image))
        print('label:', label)
        if predict_status:
            return

        # target_class = 22
        # criterion = TargetClassProbability(target_class, p=0.99)

        # attack = LBFGSAttack(model, criterion)

        # attack = FGSM(model, criterion)
        attack = FGSM(model)

        # attack = MomentumIterativeAttack(model, criterion)
        # attack = MomentumIterativeAttack(model)

        # attack = SinglePixelAttack(model)
        # attack = LocalSearchAttack(model)

        adversarial = attack(image, label=label)
        new_label = np.argmax(model.predictions(adversarial))
        print('new label:', new_label)

        image = image.astype(np.uint8)
        adversarial = adversarial.astype(np.uint8)
        pert = adversarial-image

        save_image(adversarial, adv_path)
        save_image(pert, pert_path)

        # show images
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.imshow(adversarial)
        plt.subplot(1, 3, 3)
        plt.imshow(pert)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true', help='predict input image')
    parser.add_argument('-a', action='store_true', help='attack input image')
    parser.add_argument('-i', '--image', type=str, required=True, help='input image path')
    parser.add_argument('-m', '--model', type=str, required=True, help='model file')
    args = parser.parse_args()

    if args.p:
        main(args.image, args.model, True)
    elif args.a:
        main(args.image, args.model, False)
    else:
        print('Unknown command!')
