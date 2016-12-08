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


def build_network(**kwargs):
    from keras.models import Model
    from keras.layers.convolutional import Convolution2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.layers import Dense, Dropout, Activation, \
        Reshape, Flatten, Input, merge

    defaults = {
        "board_side_length": 8,
        "conv_layers": 4,
        "num_filters": 32,
        "dropout": 0,
        "dense_layers": 2,
        "dense_hidden": 64,
        "output_size": 64
    }
    params = defaults
    params.update(kwargs)

    conv_input = Input(shape=(
        params["board_num_channels"],
        params["board_side_length"],
        params["board_side_length"]))

    def conv_wrap(conv_out):
        # use filter_width_K if it is there, otherwise use 3
        filter_key = "filter_width_%d" % i
        filter_width = params.get(filter_key, 3)
        num_filters = params["num_filters"]
        conv_out = Convolution2D(
            nb_filter=num_filters,
            nb_row=filter_width,
            nb_col=filter_width,
            init='he_normal',
            border_mode='same')(conv_out)
        conv_out = BatchNormalization()(conv_out)
        conv_out = PReLU()(conv_out)
        if params["dropout"] > 0:
            conv_out = Dropout(params["dropout"])(conv_out)
        return conv_out

    def dense_wrap(dense_out):
        dense_out = Dense(params["dense_hidden"],
                          init="he_normal")(dense_out)
        dense_out = BatchNormalization()(dense_out)
        dense_out = PReLU()(dense_out)
        if params["dropout"] > 0:
            dense_out = Dropout(params["dropout"])(dense_out)
        return dense_out

    conv_out = conv_input
    for i in range(0, params["conv_layers"]):
        conv_out = conv_wrap(conv_out)

    flattened = Flatten()(conv_out)
    dense_out = flattened
    for i in range(params["dense_layers"]):
        dense_out = dense_wrap(dense_out)

    # output for the first board
    output_pre_activation = Dense(
        params["output_size"])(dense_out)
    output_from = Activation('softmax')(output_pre_activation)

    # output for the second board
    output_reshaped = Reshape((1, 8, 8))(output_from)
    conv_merged = merge(
        [output_reshaped, conv_input],
        mode='concat',
        concat_axis=1)

    conv_out_2 = conv_merged
    for i in range(0, params["conv_layers"]):
        conv_out_2 = conv_wrap(conv_out_2)
    flattened2 = Flatten()(conv_out_2)
    dense_out = flattened2
    for i in range(params["dense_layers"]):
        dense_out = dense_wrap(dense_out)
    output_to = Dense(params["output_size"], activation="softmax")(dense_out)

    model = Model(conv_input, [output_from, output_to])
    model.compile('adamax', 'categorical_crossentropy', metrics=['accuracy'])
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
    featurized = True
    X_val, y_from_val, y_to_val = d_test.load(
        'white_state_action_sl',
        featurized=featurized,
        refresh=False)
    model = build_network(board_num_channels=X_val[0].shape[0])
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving('policy', start_time),
        verbose=2,
        save_best_only=True)
    model.fit_generator(
        generator_fn(featurized=featurized),
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
