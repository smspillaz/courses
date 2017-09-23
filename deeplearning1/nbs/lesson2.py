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
    training_data = load_data(lambda: np.concatenate(list(itertools.imap(operator.itemgetter(0),
                                                                         batches_to_np_arrays(training_batches)))),
                              'cats-dogs-encoded-train-images.bc')

    print("Loading validation data...")
    validation_batches = get_batches(os.path.join(arguments.images,
                                                  'valid'),
                                     batch_size=1,
                                     shuffle=True)
    validation_encoded_labels = onehot_encode(validation_batches.classes)
    validation_data = load_data(lambda: np.concatenate(list(batches_to_np_arrays(itertools.imap(operator.itemgetter(0),
                                                                                                validation_batches)))),
                                'cats-dogs-encoded-valid-images.bc')

    # Now generate the features for each
    print("Generating features...")
    training_features = load_data(lambda: model.predict(training_data),
                                  'cats-dogs-training-features.bc')
    validation_features = load_data(lambda: model.predict(validation_data),
                                    'cats-dogs-validation-features.bc')

    # Okay, now that we have the features, we can use stochastic
    # gradient descent to try and fit our linear model
    linear_model.fit(training_features,
                     training_encoded_labels,
                     nb_epoch=3,
                     batch_size=4,
                     validation_data=(validation_features,
                                      validation_encoded_labels))

    # Now that we've fitted our model, lets make some predictions
    for images, labels in get_batches(os.path.join(arguments.images, 'train')):
        vgg_prediction = model.predict(images)
        print(labels, linear_model.predict_classes(vgg_prediction))

if __name__ == "__main__":
    main(sys.argv[1:])
