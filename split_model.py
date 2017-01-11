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
        "dropout": 0.3,
        "dense_layers": 2,
        "dense_hidden": 64,
        "output_size": 64,
    }
    params = defaults
    params.update(kwargs)

    board_input = Input(shape=(
        params["board_num_channels"],
        params["board_side_length"],
        params["board_side_length"]))

    if params["net_type"] == 'to':
        from_input = Input(shape=(1,
        params["board_side_length"],
        params["board_side_length"]))
        merged_board_from = merge(
            [board_input, from_input],
            mode='concat',
            concat_axis=1)
        pre_conv = merged_board_from
    else:
        pre_conv = board_input

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

    conv_out = pre_conv
    for i in range(0, params["conv_layers"]):
        conv_out = conv_wrap(conv_out)

    flattened = Flatten()(conv_out)

    dense_out = flattened
    for i in range(params["dense_layers"]):
        dense_out = dense_wrap(dense_out)
    output = Dense(params["output_size"], activation="softmax")(dense_out)

    if params["net_type"] == 'from':
        model = Model(board_input, output)
    else:
        model = Model([board_input, from_input], output)
    model.compile('adamax', 'categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model


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


def train(net_type):
    assert(net_type == 'from' or net_type == 'to')
    d = Dataset('data/medium_train.pgn')
    start_time = str(int(time.time()))
    generator_fn = d.state_action_sl
    print(generator_fn.__name__)
    d_test = Dataset('data/medium_test.pgn')
    featurized = True
    X_val, y_val = d_test.load(
        generator_fn.__name__,
        featurized=featurized,
        refresh=False,
        board=net_type)

    model = build_network(board_num_channels=X_val[0].shape[0], net_type=net_type)
    try:
        plot_model(model, start_time, net_type)
    except:
        print("Skipping plot")
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath=get_filename_for_saving(start_time, net_type),
        verbose=VERBOSE_LEVEL,
        save_best_only=True)
    model.fit_generator(
        generator_fn(featurized=featurized),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        nb_epoch=NUMBER_EPOCHS,
        callbacks=[checkpointer],
        validation_data=(X_val, y_val),
        verbose=VERBOSE_LEVEL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_type", help="Either from or to")
    args = parser.parse_args()
    train(args.network_type)
