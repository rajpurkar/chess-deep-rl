import time
import os
import argparse
from data import Dataset
import numpy as np
np.random.seed(20)

FOLDER_TO_SAVE = "./saved/"
NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 10016  # tune for feedback/speed balance
VERBOSE_LEVEL = 1


def common_network(**kwargs):
    from keras.layers.convolutional import Convolution2D
    from keras.layers import Input, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    defaults = {
        "board": 8,
        "board_depth": 12,
        "layers": 6,
        "num_filters": 64,
        "one_convolve": False
    }
    params = defaults
    params.update(kwargs)

    conv_input = Input(shape=(
        params["board_depth"],
        params["board"],
        params["board"]))

    conv_start = conv_input
    for i in range(0, params["layers"]):
        # use filter_width_K if it is there, otherwise use 3
        filter_key = "filter_width_%d" % i
        filter_width = params.get(filter_key, 3)
        num_filters = params["num_filters"]
        if i == params["layers"] - 1 and params["one_convolve"] is True:
            filter_width = 1
            num_filters = 1
        conv_start = Convolution2D(
            nb_filter=num_filters,
            nb_row=filter_width,
            nb_col=filter_width,
            init='he_normal',
            border_mode='same')(conv_start)
        # conv_start = Activation('relu')(conv_start)
        conv_start = BatchNormalization()(conv_start)
        conv_start = PReLU()(conv_start)
    flattened = Flatten()(conv_start)
    return conv_input, flattened


def value_network(**kwargs):
    from keras.models import Model
    from keras.layers import Dense
    """ Use a variation of the ROCAlphaGo Value Network. """
    conv_input, flattened = common_network(**kwargs)
    dense_1 = Dense(500, activation="relu")(flattened)
    output = Dense(1, activation="tanh")(dense_1)
    model = Model(conv_input, output)
    model.compile('adamax', 'mse')
    return model


def policy_network(**kwargs):
    from keras.models import Model
    from keras.layers import Dense
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU

    params = {
        "dense_layers": 2,
    }
    params.update(kwargs)

    conv_input, flattened = common_network(**kwargs)
    for i in range(params["dense_layers"]):
        dense_mess = Dense(64, init="he_normal")(flattened)
        dense_mess = BatchNormalization()(dense_mess)
        dense_mess = PReLU()(dense_mess)
        # dense_mess = Dropout(0.5)(dense_mess)
    output = Dense(64, activation="softmax")(dense_mess)
    model = Model(conv_input, output)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model


def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{val_loss:.2f}.hdf5"


def train(net_type):
    from keras.callbacks import ModelCheckpoint
    d = Dataset('data/large-ccrl.pgn')
    start_time = str(int(time.time()))
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(net_type, start_time),
        verbose=2,
        save_best_only=True)
    if net_type == "value":
        model = value_network()
        generator_str = 'random_black_state'
        generator_fn = d.random_black_state
    elif net_type == 'policy':
        model = policy_network()
        generator_str = 'white_state_action_sl'
        generator_fn = d.white_state_action_sl
    d_test = Dataset('data/small_test.pgn')
    (X_test, y_test) = d_test.load(generator_str, refresh=False)
    model.fit_generator(
        generator_fn(),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        nb_epoch=NUMBER_EPOCHS,
        callbacks=[checkpointer],
        validation_data=(X_test, y_test),
        verbose=VERBOSE_LEVEL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_type", help="Either value or policy")
    args = parser.parse_args()
    train(args.network_type)
