import argparse
import util

def split_net(**kwargs):
    from keras.models import Model
    from keras.layers import Dense, Reshape, Flatten, Input, merge

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

    conv_out = pre_conv
    for i in range(0, params["conv_layers"]):
        conv_out = util.conv_wrap(params, conv_out, i)

    flattened = Flatten()(conv_out)

    dense_out = flattened
    for i in range(params["dense_layers"]):
        dense_out = util.dense_wrap(params, dense_out, i)
    output = Dense(params["output_size"], activation="softmax")(dense_out)

    if params["net_type"] == 'from':
        model = Model(board_input, output)
    else:
        model = Model([board_input, from_input], output)
    model.compile('adamax', 'categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_type", help="Either from or to")
    args = parser.parse_args()
    net_type = args.network_type
    assert(net_type == 'from' or net_type == 'to')
    util.train(net_type, 'state_action_sl', 'data/large-ccrl_', build_network, featurized=True)
