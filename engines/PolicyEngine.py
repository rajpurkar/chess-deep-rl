#!/usr/bin/env python3
from ChessEngine import ChessEngine
from keras.models import load_model
import numpy as np
import sys
sys.path.append('.')
import data

class PolicyEngine(ChessEngine):
    def __init__(self, keras_model_hdf5):
        super().__init__()
        self.model = load_model(keras_model_hdf5)

    def search(self, boards=None):
        if boards is not None:
            states = []
            for board in boards.values():
                states.append(data.state_from_board(board))
            batch_size = len(states)
            X = np.array(states)
            y = self.model.predict(X, batch_size=batch_size, verbose=1)
            # TODO: find moves from y
            return X, y, moves

if __name__ == "__main__":
    engine = ValueBaselineEngine("./saved/1480896779-06-0.40.hdf5")
    engine.run()
