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
SAMPLES_PER_EPOCH = 1000  # tune for feedback/speed balance
VERBOSE_LEVEL = 1

def common_network(**kwargs):
    defaults = {
        "board": 8,
        "board_depth": 12,
        "layers": 0,
        "num_filters": 50
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
    output = Dense(6, activation="softmax")(flattened)
    model = Model(conv_input, output)
    model.compile('adamax', 'categorical_crossentropy')
    return model


def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{val_loss:.2f}.hdf5"


def train(net_type):
    d = Dataset('data/medium.pgn')
    d_test = Dataset('data/medium_test.pgn')
    d_test.pickle()
    start_time = str(int(time.time()))
    (X_test, y_test) = d_test.unpickle()
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(net_type, start_time),
        verbose=VERBOSE_LEVEL,
        save_best_only=True)
    if net_type == "value":
        model = value_network()
        model.fit_generator(
            d.random_black_state(),
            samples_per_epoch=SAMPLES_PER_EPOCH,
            nb_epoch=NUMBER_EPOCHS,
            callbacks=[checkpointer],
            validation_data=(X_test, y_test),
            verbose=VERBOSE_LEVEL)
    elif net_type == 'six_piece_policy':
        model = six_piece_policy_network()
        model.fit_generator(
            d.white_state_action_sl(),
            samples_per_epoch=SAMPLES_PER_EPOCH,
            nb_epoch=NUMBER_EPOCHS,
            callbacks=[checkpointer],
            validation_data=(X_test, y_test),
            verbose=VERBOSE_LEVEL)

if __name__ == '__main__':
    train('value')
