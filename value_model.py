import time
import os
import argparse
from data import Dataset
import numpy as np
np.random.seed(20)

FOLDER_TO_SAVE = "./saved/"
NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 12800  # tune for feedback/speed balance
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
        "dropout": 0.5,
        "dense_layers": 2,
        "dense_hidden": 64,
        "output_size": 1,
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
    value = Dense(params["output_size"], activation="tanh", name='value')(dense_out)

    model = Model(conv_input, value)

    model.compile('adamax', 'mse', metrics=['mean_squared_error', 'mean_absolute_error'])
    return model


def get_folder_name(start_time):
    folder_name = FOLDER_TO_SAVE + 'value/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def get_filename_for_saving(start_time):
    saved_filename = get_folder_name(start_time) + "/{epoch:02d}-{val_loss:.2f}.hdf5"
    return saved_filename


def plot_model(model, start_time):
    from keras.utils.visualize_util import plot
    plot(
        model,
        to_file=get_folder_name(start_time) + '/model.png',
        show_shapes=True,
        show_layer_names=False)


def train():
    featurized = True

    d = Dataset('data/large-ccrl_train.pgn')
    generator_fn = d.state_value

    d_test = Dataset('data/large-ccrl_test.pgn')
    X_val, y_val = d_test.load('state_value',
        featurized = featurized,
        refresh    = False)

    model = build_network(board_num_channels=X_val[0].shape[0])
    start_time = str(int(time.time()))
    try:
        plot_model(model, start_time)
    except:
        print("Skipping plot")
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath       = get_filename_for_saving(start_time),
        verbose        = 2,
        save_best_only = True)

    model.fit_generator(generator_fn(featurized=featurized),
        samples_per_epoch = SAMPLES_PER_EPOCH,
        nb_epoch          = NUMBER_EPOCHS,
        callbacks         = [checkpointer],
        validation_data   = (X_val, y_val),
        verbose           = VERBOSE_LEVEL)

if __name__ == '__main__':
    train()
