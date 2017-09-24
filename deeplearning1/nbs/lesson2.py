"""Use the existing VGG16 classifications to feed into a linear model.

In this example, what we'll be doing is using the second last layer of
the VGG16 model and training a new linear layer to just give us the
probability that an image is a cat or a dog, thus reducing the dimensions
of the image down from 4096 to 2.

The way this is done is to add a new Dense layer. This is a linear layer
with a fixed number of inputs and outputs. We tell it to use stochastic
gradient descent in fitting a line with a "rectified linear" activation
function between the layers (eg, max(0, x)).
"""

import argparse

import itertools

import json

import os

import operator

import sys

import numpy as np

import bcolz

from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop

from keras.preprocessing import image

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from vgg16 import Vgg16

from tqdm import tqdm

K.set_image_dim_ordering('th')

FILES_PATH = 'http://files.fast.ai/models/'
CLASS_FILE = 'imagenet_class_index.json'

BATCH_SIZE = 1


def get_classes_from_file(class_file, files_path, cache_subdir):
    # Keras' get_file() is a handy function that downloads files, and caches them for re-use later
    fpath = get_file(class_file,
                     os.path.join(files_path, class_file),
                     cache_subdir=cache_subdir)
    with open(fpath) as fileobj:
        class_dict = json.load(fileobj)
        classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        return classes


def extract_features(batches):
    """Extract some features from the given batches."""
    pass


def batches_to_np_arrays(batches, batch_size=BATCH_SIZE, limit=None):
    iterations = batches.nb_sample // batch_size
    iterations = min(iterations, limit or iterations)
    for i in tqdm(range(iterations),
                  total=iterations,
                  desc="Generating array images"):
        images, labels = batches.next()
        yield (images, labels)


def get_batches(source,
                gen=image.ImageDataGenerator(),
                shuffle=True,
                batch_size=BATCH_SIZE,
                class_mode='categorical'):
    return gen.flow_from_directory(source,
                                   target_size=(224, 224),
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)


def load_data(generator, cache_key):
    try:
        return bcolz.open(cache_key)[:]
    except OSError:
        data = generator()

        # Now save it under the cache key
        c = bcolz.carray(data, rootdir=cache_key, mode='w')
        c.flush()

        return data


def onehot_encode(array):
    return np.array(OneHotEncoder().fit_transform(array.reshape(-1, 1)).todense())


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("images",
                        metavar="PATH",
                        type=str)
    parser.add_argument("--weights",
                        metavar="WEIGHTS",
                        type=str)
    parser.add_argument("--train",
                        action="store_true")
    parser.add_argument("--load",
                        action="store_true")
    arguments = parser.parse_args(argv or sys.argv[1:])

    vgg = Vgg16()
    model = vgg.model
    linear_model = Sequential([
        Dense(
            2,  # 2 columns in output
            activation='softmax',  # Rectified linear
            input_shape=(1000, )  # 1000 inputs
        )
    ])
    linear_model.compile(optimizer=RMSprop(lr=0.1),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    print("Loading training data...")
    training_batches = get_batches(os.path.join(arguments.images,
                                                'train'),
                                   batch_size=1,
                                   shuffle=False)
    training_encoded_labels = onehot_encode(training_batches.classes)
    training_data = load_data(lambda: np.concatenate(list(map(operator.itemgetter(0),
                                                              batches_to_np_arrays(training_batches)))),
                              'cats-dogs-encoded-train-images.bc')

    print("Loading validation data...")
    validation_batches = get_batches(os.path.join(arguments.images,
                                                  'valid'),
                                     batch_size=1,
                                     shuffle=True)
    validation_encoded_labels = onehot_encode(validation_batches.classes)
    validation_data = load_data(lambda: np.concatenate(list(map(operator.itemgetter(0),
                                                                batches_to_np_arrays(validation_batches)))),
                                'cats-dogs-encoded-valid-images.bc')

    if arguments.load:
        model.load_weights(arguments.weights)

    if arguments.train:
        # We're going to create a new layer for the VGG model now, so shuffle the training
        # data around a little
        gen = image.ImageDataGenerator()
        lastlayer_training_batches = gen.flow(training_data, training_encoded_labels, batch_size=4, shuffle=True)
        lastlayer_validation_batches = gen.flow(validation_data, validation_encoded_labels, batch_size=4, shuffle=False)

        model.pop()
        for layer in model.layers:
            layer.trainable = False

        optimizer = RMSprop(lr=0.1)

        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(lastlayer_training_batches,
                            samples_per_epoch=lastlayer_training_batches.n,
                            nb_epoch=2,
                            validation_data=lastlayer_validation_batches,
                            nb_val_samples=lastlayer_validation_batches.n)


        # Decrease the learning weight a little
        K.set_value(optimizer.lr, 0.01)

        # Now find the first dense layer and mark it and all subsequent layers
        # as trainable
        first_dense = [index for index, layer in enumerate(model.layers) if type(layer) is Dense][0]
        for layer in model.layers[first_dense:]:
            layer.trainable = True

        model.fit_generator(lastlayer_training_batches,
                            samples_per_epoch=lastlayer_training_batches.n,
                            nb_epoch=3,
                            validation_data=lastlayer_validation_batches,
                            nb_val_samples=lastlayer_validation_batches.n)

        if arguments.weights:
            model.save_weights(arguments.weights)

    predictions = model.predict_classes(validation_data, batch_size=1)
    print(confusion_matrix(validation_batches.classes, predictions))

if __name__ == "__main__":
    main(sys.argv[1:])
