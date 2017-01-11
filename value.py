import util

def build_network(**kwargs):
    from keras.models import Model
    from keras.layers import Dense, Reshape, Flatten, Input, merge

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

    conv_out = conv_input
    for i in range(0, params["conv_layers"]):
        conv_out = util.conv_wrap(params, conv_out, i)

    flattened = Flatten()(conv_out)

    dense_out = flattened
    for i in range(params["dense_layers"]):
        dense_out = util.dense_wrap(params, dense_out, i)
    value = Dense(params["output_size"], activation="tanh", name='value')(dense_out)

    model = Model(conv_input, value)

    model.compile('adamax', 'mse', metrics=['mean_squared_error', 'mean_absolute_error'])
    return model

if __name__ == '__main__':
    util.train('value', 'state_value', 'data/large-ccrl_', build_network, featurized=True)
