import time
import os
import argparse
from data import Dataset
import numpy as np
np.random.seed(20)

FOLDER_TO_SAVE = "./saved/"
NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 40064  # tune for feedback/speed balance
VERBOSE_LEVEL = 1


def common_network(**kwargs):
    from keras.layers.convolutional import Convolution2D
    from keras.layers import Input, Flatten, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    defaults = {
        "board_side_length": 8,
        "layers": 4,
        "num_filters": 6,
        "dropout": 0
    }
    params = defaults
    params.update(kwargs)

    conv_input = Input(shape=(
        params["board_num_channels"],
        params["board_side_length"],
        params["board_side_length"]))

    conv_mess = conv_input
    for i in range(0, params["layers"]):
        # use filter_width_K if it is there, otherwise use 3
        filter_key = "filter_width_%d" % i
        filter_width = params.get(filter_key, 3)
        num_filters = params["num_filters"]
        conv_mess = Convolution2D(
            nb_filter=num_filters,
            nb_row=filter_width,
            nb_col=filter_width,
            init='he_normal',
            border_mode='same')(conv_mess)
        conv_mess = BatchNormalization()(conv_mess)
        conv_mess = PReLU()(conv_mess)
        if params["dropout"] > 0:
            conv_mess = Dropout(params["dropout"])(conv_mess)
    flattened = Flatten()(conv_mess)
    return conv_input, flattened


def policy_network(**kwargs):
    from keras.models import Model
    from keras.layers import Dense, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU

    params = {
        "dense_layers": 1,
        "dense_hidden": 64,
        "output_size": 64,
        "dropout": 0
    }
    params.update(kwargs)

    conv_input, flattened = common_network(**kwargs)
    dense_mess = flattened
    for i in range(params["dense_layers"]):
        dense_mess = Dense(params["dense_hidden"], init="he_normal")(dense_mess)
        dense_mess = BatchNormalization()(dense_mess)
        dense_mess = PReLU()(dense_mess)
        if params["dropout"] > 0:
            dense_mess = Dropout(params["dropout"])(dense_mess)

    # output for the first board
    output1 = Dense(params["output_size"], activation="softmax")(dense_mess)

    # output for the second board
    output2 = Dense(params["output_size"], activation="softmax")(dense_mess)

    model = Model(conv_input, [output1, output2])
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model


def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{val_loss:.2f}.hdf5"


def train(net_type):
    d = Dataset('data/large-ccrl.pgn')
    start_time = str(int(time.time()))
    generator_fn = d.white_state_action_sl
    d_test = Dataset('data/small_test.pgn')
    X_val, y_from_val, y_to_val = d_test.load('white_state_action_sl', refresh=True)
    print(X_val[0].shape)
    model = policy_network(board_num_channels=X_val[0].shape[0])
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving('policy', start_time),
        verbose=2,
        save_best_only=True)
    model.fit_generator(
        generator_fn(featurized=True),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        nb_epoch=NUMBER_EPOCHS,
        callbacks=[checkpointer],
        validation_data=(X_val, [y_from_val, y_to_val]),
        verbose=VERBOSE_LEVEL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_type", help="Either value or policy")
    args = parser.parse_args()
    train(args.network_type)
