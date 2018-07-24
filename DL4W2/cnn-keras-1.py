import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D(padding=(3, 3), data_format='channels_last')(X_input)

    # CONV (32 filters, 7x7 each, stride 1) -> BatchNorm -> ReLU
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    # Dropout regularization
    X = Dropout(0.2)(X)

    # CONV (128 filter, 3x3, stride 1) -> BN -> ReLU
    # X = Conv2D(64, (5,5), strides = (1,1), name = 'conv1')(X)
    # X = BatchNormalization(axis = 3, name = 'bn1')(X)
    # X = Activation('relu')(X)

    # MAXPOOL
    # X = MaxPooling2D((2,2), name = 'max_pool1')(X)

    # Dropout regularization
    # X = Dropout(0.25)(X)

    # CONV (64 filter, 3x3, stride 1) -> BN -> ReLU
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    # Dropout regularization
    X = Dropout(0.25)(X)

    # Flatten X + FC
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    ### END CODE HERE ###

    return model


if __name__ == "__main__":
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    # create model
    happyModel = HappyModel(X_train.shape[1:])

    # compile model
    happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train model
    happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)

    # evaluation model
    preds = happyModel.evaluate(x=X_test, y=Y_test)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    # show model summary
    happyModel.summary()

    # save model representation to file
    plot_model(happyModel, to_file='HappyModel.png')


