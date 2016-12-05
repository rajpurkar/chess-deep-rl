from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import time
# from tqdm import tqdm
from data import Dataset

FOLDER_TO_SAVE = "./saved/"
NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 1000  # tune for feedback/speed balance
VERBOSE_LEVEL = 2


def value_network(**kwargs):
    """ Use a variation of the ROCAlphaGo Value Network. """
    defaults = {
        "board": 8,
        "board_depth": 12,
        "layers": 5,
        "num_filters": 40
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
    densed = Dense(256, init='uniform', activation='relu')(flattened)
    output = Dense(1, init='uniform', activation="tanh")(densed)
    model = Model(conv_input, output)
    model.compile('adamax', 'mse')
    return model


def get_filename_for_saving(start_time):
    return FOLDER_TO_SAVE + str(start_time) + \
        "-{epoch:02d}-{val_loss:.2f}.hdf5"


def train():
    d = Dataset('data/medium.pgn')
    d_test = Dataset('data/medium_test.pgn')
    d_test.pickle()
    start_time = int(time.time())
    (X_test, y_test) = d_test.unpickle()
    model = value_network()
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(start_time),
        verbose=VERBOSE_LEVEL,
        save_best_only=True)
    model.fit_generator(
        d.random_black_state(),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        nb_epoch=NUMBER_EPOCHS,
        callbacks=[checkpointer],
        validation_data=(X_test, y_test),
        verbose=VERBOSE_LEVEL)

if __name__ == '__main__':
    train()
