#!/usr/bin/env python3
from ChessEngine import ChessEngine
from keras.models import load_model
import numpy as np
import sys
sys.path.append('.')
import data

class ValueEngine(ChessEngine):
    def __init__(self, keras_model_h5, black=False):
        super().__init__()
        self.model = load_model(keras_model_h5)
        self.is_black = black

    def search(self):
        moves = []
        scores = []
        states = []
        for move in self.board.generate_legal_moves():
            # Play move and convert board to state
            test_board = self.board.copy()
            test_board.push(move)
            state = data.state_from_board(test_board, black=self.is_black)
            states.append(state)
            moves.append(move)
            scores = self.model.predict(np.array(states)).flatten().tolist()
        if (len(scores) == 0):
            self.moves = None
            return
        idx = np.argmax(scores)
        self.moves = [moves[idx]]


if __name__ == "__main__":
    is_black = False
    engine = ValueEngine("./saved/value_network_253.hdf5", black=is_black)
    engine.run()
