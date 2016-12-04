

def value_network(params):
    from keras.models import Sequential
    from keras.layers import convolutional
    from keras.layers.core import Flatten
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
    model.compile('adam', 'mse')
    return model

if __name__ == '__main__':
    d = Dataset()
    while True:
        (s, r, moves_remaining) = d.random_black_state()
    # X, y = get_data()
    model = value_network({"board_depth": 1, "board": 8}) 
    # model.fit(X, y)
