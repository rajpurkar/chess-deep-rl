import time
import os
import argparse
from data import Dataset
import numpy as np
import m
np.random.seed(20)

NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 12800  # tune for feedback/speed balance
VERBOSE_LEVEL = 1

def inside(net_type, generator_fn_str, dataset_file, build_net_fn, featurized=True):
    d = Dataset(dataset_file + 'train.pgn')
    generator_fn = getattr(d, generator_fn_str)
    d_test = Dataset(dataset_file + 'test.pgn')

    X_val, y_val = d_test.load(generator_fn.__name__,
        featurized = featurized,
        refresh    = False)

    model = build_net_fn(board_num_channels=X_val[0].shape[0])
    start_time = str(int(time.time()))
    try:
        m.plot_model(model, start_time, net_type)
    except:
        print("Skipping plot")
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(
        filepath       = m.get_filename_for_saving(start_time, net_type),
        verbose        = 2,
        save_best_only = True)

    model.fit_generator(generator_fn(featurized=featurized),
        samples_per_epoch = SAMPLES_PER_EPOCH,
        nb_epoch          = NUMBER_EPOCHS,
        callbacks         = [checkpointer],
        validation_data   = (X_val, y_val),
        verbose           = VERBOSE_LEVEL)

if __name__ == '__main__':
    inside('value', 'state_value', 'data/large-ccrl_', m.build_network, featurized=True)
