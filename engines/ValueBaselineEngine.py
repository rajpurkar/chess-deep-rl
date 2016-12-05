#!/usr/bin/env python3
from ChessEngine import ChessEngine
from keras.models import load_model
import numpy as np
import sys
sys.path.append('.')
import data

class ValueBaselineEngine(ChessEngine):
    def __init__(self, keras_model_h5):
        super().__init__()
        self.model = load_model(keras_model_h5)

    def search(self):
        moves = []
        scores = []
        states = []
        for move in self.board.generate_legal_moves():
            # Play move and convert board to state
            test_board = self.board.copy()
            test_board.push(move)
            state = data.state_from_board(test_board)
            states.append(state)
            moves.append(move)
            scores = self.model.predict(np.array(states)).flatten().tolist()
        if (len(scores) == 0):
            self.moves = None
            return
        idx = np.argmin(scores)
        self.moves = [moves[idx]]


if __name__ == "__main__":
    engine = ValueBaselineEngine("./saved/1480896779-06-0.40.hdf5")
    engine.run()
