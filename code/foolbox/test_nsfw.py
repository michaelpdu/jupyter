import os
import sys
import argparse
import numpy as np
from tensorflow.contrib.slim.nets import inception
from foolbox.models import TensorFlowModel
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack
from foolbox.attacks import FGSM
from foolbox.attacks import MomentumIterativeAttack
from foolbox.attacks import SinglePixelAttack
from foolbox.attacks import LocalSearchAttack
import matplotlib.pyplot as plt
from image_util import *
import tensorflow as tf
from tensorflow.python.platform import gfile

categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

def predict_dir(model, dir_path):
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext == '.png' or ext == '.jpg':
                image_path = os.path.join(root, name)
                image = inception_preprocessing(open_image(image_path, 299, 299))
                value = model.predictions(image)
                label = np.argmax(value)
                print(name, ': ', value, ': ', label)

def predict(pb_path, image_path):
    with tf.Session() as session:
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            session.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        session.run(tf.global_variables_initializer())
        images = session.graph.get_tensor_by_name('input_1:0')
        logits = session.graph.get_tensor_by_name('dense_3/BiasAdd:0')
        output = session.graph.get_tensor_by_name('dense_3/Softmax:0')
        # for n in tf.get_default_graph().get_operations():
        #     print(n.values())
        model = TensorFlowModel(images, output, (0, 255))

        if os.path.isdir(image_path):
            predict_dir(model, image_path)
        else:
            image = open_image(image_path, 299, 299)
            image = inception_preprocessing(image)
            values = model.predictions(image)
            print('values:', values)
            label = np.argmax(values)
            print('label:', categories[label])

def attack(pb_path, image_path):
    with tf.Session() as session:
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            session.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        session.run(tf.global_variables_initializer())
        images = session.graph.get_tensor_by_name('input_1:0')
        logits = session.graph.get_tensor_by_name('dense_3/BiasAdd:0')
        output = session.graph.get_tensor_by_name('dense_3/Softmax:0')
        # for n in tf.get_default_graph().get_operations():
        #     print(n.values())
        model = TensorFlowModel(images, logits, (-1, 1))

        image = inception_preprocessing(open_image(image_path, 299, 299))
        p, ext = os.path.splitext(image_path)
        adv_path = p+'-adv'+ext
        pert_path = p+'-pert'+ext

        values = model.predictions(image)
        label = np.argmax(values)
        print('label:', categories[label])

        print('attacking...')
        target_class = 2
        criterion = TargetClassProbability(target_class, p=0.99)

        # attack = LBFGSAttack(model, criterion)

        # attack = FGSM(model, criterion)
        attack = FGSM(model)

        # attack = MomentumIterativeAttack(model, criterion)
        # attack = MomentumIterativeAttack(model)

        # attack = SinglePixelAttack(model)
        # attack = LocalSearchAttack(model)

        adversarial = attack(image, label=label)
        new_label = np.argmax(model.predictions(adversarial))
        print('new label:', categories[new_label])

        image = inception_postprocessing(image)
        adversarial = inception_postprocessing(adversarial)
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
        predict(args.model, args.image)
    elif args.a:
        attack(args.model, args.image)
    else:
        print('Unknown command!')
