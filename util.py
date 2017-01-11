import numpy as np
import os
import time
from data import Dataset

np.random.seed(20)

FOLDER_TO_SAVE = "./saved/"
NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 12800  # tune for feedback/speed balance
VERBOSE_LEVEL = 1

def get_folder_name(start_time, net_type):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def get_filename_for_saving(start_time, net_type):
    saved_filename = get_folder_name(start_time, net_type) + "/{epoch:02d}-{val_loss:.2f}.hdf5"
    return saved_filename


def plot_model(model, start_time, net_type):
    from keras.utils.visualize_util import plot
    plot(
        model,
        to_file=get_folder_name(start_time, net_type) + '/model.png',
        show_shapes=True,
        show_layer_names=False)


def conv_wrap(params, conv_out, i):
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.layers.convolutional import Convolution2D
    from keras.layers import Dropout
    
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

def dense_wrap(params, dense_out, i):
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.layers import Dense, Dropout

    dense_out = Dense(params["dense_hidden"],
                      init="he_normal")(dense_out)
    dense_out = BatchNormalization()(dense_out)
    dense_out = PReLU()(dense_out)
    if params["dropout"] > 0:
        dense_out = Dropout(params["dropout"])(dense_out)
    return dense_out


def train(net_type, generator_fn_str, dataset_file, build_net_fn, featurized=True):
    d = Dataset(dataset_file + 'train.pgn')
    generator_fn = getattr(d, generator_fn_str)
    d_test = Dataset(dataset_file + 'test.pgn')

    X_val, y_val = d_test.load(generator_fn.__name__,
        featurized = featurized,
        refresh    = False,
        board = net_type)

    board_num_channels = X_val[0].shape[1] if net_type == 'to' else X_val[0].shape[0]
    model = build_net_fn(board_num_channels=board_num_channels, net_type=net_type)
    start_time = str(int(time.time()))
    try:
        plot_model(model, start_time, net_type)
    except:
        print("Skipping plot")
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath       = get_filename_for_saving(start_time, net_type),
        verbose        = 2,
        save_best_only = True)

    model.fit_generator(generator_fn(featurized=featurized, board=net_type),
        samples_per_epoch = SAMPLES_PER_EPOCH,
        nb_epoch          = NUMBER_EPOCHS,
        callbacks         = [checkpointer],
        validation_data   = (X_val, y_val),
        verbose           = VERBOSE_LEVEL)
