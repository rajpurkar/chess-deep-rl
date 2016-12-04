from data import Dataset
import numpy as np
from tqdm import tqdm


def value_network(params):
    from keras.models import Sequential
    from keras.layers import convolutional
    from keras.layers.core import Flatten, Dense
    model = Sequential()
    model.add(convolutional.Convolution2D(
        input_shape=(params["board_depth"], params["board"], params["board"]),
        nb_filter=10,
        nb_row=3,
        nb_col=3,
        init='uniform',
        activation='relu',
        border_mode='same')
    )
    model.add(convolutional.Convolution2D(
        nb_filter=1,
        nb_row=1,
        nb_col=1,
        init='uniform',
        border_mode='same', bias=True)
    )
    model.add(Flatten())
    model.add(Dense(1, init='uniform', activation="tanh"))
    model.compile('adam', 'mse')
    return model

if __name__ == '__main__':
    d = Dataset('data/medium.pgn')
    model = value_network({"board_depth": 12, "board": 8})
    model.fit_generator(d.random_black_state(), 100, 70)
    model.fit_generator(d.random_black_state(), 100, 70)
