import json

import os

import numpy as np

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

K.set_image_dim_ordering('th')

FILES_PATH = 'http://files.fast.ai/models/'
CLASS_FILE = 'imagenet_class_index.json'


def conv_block(layers, model, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def fully_connected_block(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))


def vgg_preprocess(x):
    x = x - VGG_MEAN
    return x[:, ::-1]


def vgg16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224)))

    conv_block(2, model, 64)
    conv_block(2, model, 128)
    conv_block(3, model, 256)
    conv_block(3, model, 512)
    conv_block(3, model, 512)

    model.add(Flatten())
    fully_connected_block(model)
    fully_connected_block(model)
    model.add(Dense(1000, activation='softmax'))
    return model


def load_vgg_weights(model, files_path, cache_subdir):
    fpath = get_file('vgg16.h5', os.path.join(files_path, 'vgg16.h5'), cache_subdir=cache_subdir)
    model.load_weights(fpath)


def get_classes_from_file(class_file, files_path, cache_subdir):
    # Keras' get_file() is a handy function that downloads files, and caches them for re-use later
    fpath = get_file(class_file,
                     os.path.join(files_path, class_file),
                     cache_subdir=cache_subdir)
    with open(fpath) as fileobj:
        class_dict = json.load(fileobj)
        classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        return classes


BATCH_SIZE = 4


def get_batches(data_path,
                dirname,
                gen=image.ImageDataGenerator(),
                shuffle=True,
                batch_size=BATCH_SIZE,
                class_mode='categorical'):
    return gen.flow_from_directory(os.path.join(data_path, dirname),
                                   target_size=(224, 224),
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)


def predict_batches(model, images, classes):
    predictions = model.predict(images)
    indexes = np.argmax(predictions, axis=1)

    for i, idx in enumerate(indexes):
        yield (idx, predictions[i, idx], classes[idx])


def predict_all_batches_from_sets(model, classes, data_path):
    batches = get_batches(data_path, 'train')
    val_batches = get_batches(data_path, 'valid')

    for images, labels in batches:
        for index, prediction, klass in predict_batches(model, images, classes):
            print("{:.4f}/{}".format(prediction, klass))


def main():
    classes = get_classes_from_file(CLASS_FILE, FILES_PATH, 'models')
    model = vgg16()
    load_vgg_weights(model, FILES_PATH, 'weights')

    predict_all_batches_from_sets(model, classes, 'data/dogscats/sample/')


if __name__ == "__main__":
    main()
