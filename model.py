from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import time
import os
# from tqdm import tqdm
from data import Dataset

FOLDER_TO_SAVE = "./saved/"
NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 10000  # tune for feedback/speed balance
VERBOSE_LEVEL = 1


def common_network(**kwargs):
    defaults = {
        "board": 8,
        "board_depth": 12,
        "layers": 6,
        "num_filters": 100
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
        filter_width = params.get(filter_key, 2 + i*2)
        conv_start = Convolution2D(
            nb_filter=params["num_filters"],
            nb_row=filter_width,
            nb_col=filter_width,
            init='uniform',
            activation='relu',
            border_mode='same')(conv_start)

    # the last layer maps each <filters_per_layer> feature to a number
    one_channel_conv = Convolution2D(
        nb_filter=1,
        nb_row=1,
        nb_col=1,
        init='uniform',
        border_mode='same',
        bias=True)(conv_input)

    flattened = Flatten()(one_channel_conv)
    return conv_input, flattened


def value_network(**kwargs):
    """ Use a variation of the ROCAlphaGo Value Network. """
    conv_input, flattened = common_network(**kwargs) 
    output = Dense(1, init='uniform', activation="tanh")(flattened)
    model = Model(conv_input, output)
    model.compile('adam', 'mse')
    return model


def six_piece_policy_network(**kwargs):
    conv_input, flattened = common_network(**kwargs) 
    dense_1 = Dense(1000, activation="relu")(flattened)
    dense_2 = Dense(500, activation="relu")(dense_1)
    dense_3 = Dense(250, activation="relu")(dense_2)
    output = Dense(6, activation="softmax")(dense_3)
    model = Model(conv_input, output)
    model.compile('adamax', 'categorical_crossentropy', metrics=['accuracy'])
    return model


def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{val_loss:.2f}.hdf5"


def train(net_type):
    d = Dataset('data/medium.pgn')
    start_time = str(int(time.time()))
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(net_type, start_time),
        verbose=VERBOSE_LEVEL,
        save_best_only=True)
    if net_type == "value":
        model = value_network()
        generator_str = 'random_black_state'
        generator_fn = d.random_black_state
    elif net_type == 'six_piece_policy':
        model = six_piece_policy_network()
        generator_str = 'white_state_action_sl'
        generator_fn = d.white_state_action_sl
    d_test = Dataset('data/medium_test.pgn')
    (X_test, y_test) = d_test.load(generator_str)
    model.fit_generator(
        generator_fn(),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        nb_epoch=NUMBER_EPOCHS,
        callbacks=[checkpointer],
        validation_data=(X_test, y_test),
        verbose=VERBOSE_LEVEL)

if __name__ == '__main__':
    train('six_piece_policy')
