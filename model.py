import m as util

def build_network(**kwargs):
    from keras.models import Model
    from keras.layers import Dense, Activation, Reshape, Flatten, Input, merge

    defaults = {
        "board_side_length": 8,
        "conv_layers": 4,
        "num_filters": 32,
        "dropout": 0.3,
        "dense_layers": 2,
        "dense_hidden": 64,
        "output_size": 64,
        "conditioned_architecture": True
    }
    params = defaults
    params.update(kwargs)

    conv_input = Input(shape=(
        params["board_num_channels"],
        params["board_side_length"],
        params["board_side_length"]))

    conv_out = conv_input
    for i in range(0, params["conv_layers"]):
        conv_out = util.conv_wrap(conv_out)

    flattened = Flatten()(conv_out)

    dense_out = flattened
    for i in range(params["dense_layers"]):
        dense_out = util.dense_wrap(dense_out)
    output_from = Dense(params["output_size"], activation="softmax", name='from_board')(dense_out)

    if params["conditioned_architecture"] is True:
        output_reshaped = Reshape((1, 8, 8))(output_from)
        conv_merged = merge(
            [output_reshaped, conv_input],
            mode='concat',
            concat_axis=1)

        conv_out_2 = conv_merged
        for i in range(0, params["conv_layers"]):
            conv_out_2 = util.conv_wrap(conv_out_2)
        flattened2 = Flatten()(conv_out_2)
        dense_out_2 = flattened2
    else:
        dense_out_2 = flattened

    for i in range(params["dense_layers"]):
        dense_out_2 = util.dense_wrap(dense_out_2)
    output_to = Dense(params["output_size"], activation="softmax", name="to_board")(dense_out_2)

    model = Model(conv_input, [output_from, output_to])

    model.compile('adamax', 'categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    return model

if __name__ == '__main__':
    util.train('policy', 'state_action_sl', 'data/large-ccrl_', build_network, featurized=True)
