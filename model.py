from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Flatten, Dense
# from tqdm import tqdm
from data import Dataset


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

    model = Sequential()
    model.add(convolutional.Convolution2D(
        input_shape=(params["board_depth"], params["board"], params["board"]),
        nb_filter=params["num_filters"], nb_row=3, nb_col=3,
        init='uniform', activation='relu', border_mode='same')
    )
    for i in range(2, params["layers"] + 1):
        # use filter_width_K if it is there, otherwise use 3
        filter_key = "filter_width_%d" % i
        filter_width = params.get(filter_key, 3)
        model.add(convolutional.Convolution2D(
            nb_filter=params["num_filters"],
            nb_row=filter_width,
            nb_col=filter_width,
            init='uniform',
            activation='relu',
            border_mode='same'))

    # the last layer maps each <filters_per_layer> feature to a number
    model.add(convolutional.Convolution2D(
        nb_filter=1,
        nb_row=1,
        nb_col=1,
        init='uniform',
        border_mode='same', bias=True)
    )
    model.add(Flatten())
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation="tanh"))
    model.compile('adam', 'mse')
    return model

if __name__ == '__main__':
    d = Dataset('data/medium.pgn')
    d_test = Dataset('data/medium_test.pgn', test_set=True)
    model = value_network()
    checkpointer = ModelCheckpoint(filepath="./saved/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)
    model.fit_generator(d.random_black_state(), 1000, 70,  callbacks=[checkpointer], validation_data=d_test.random_black_state(), nb_val_samples=100)
        # validation_data=test_data, nb_worker=8, pickle_safe=True)
