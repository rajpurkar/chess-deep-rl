#!/usr/bin/env python3
from ChessEngine import ChessEngine
from keras.models import load_model
import numpy as np
import data

class ValueBaselineEngine(ChessEngine):
    def __init__(self, keras_model_h5):
        super().__init__()

        self.model = load_model(keras_model_h5)
    
    def search(self):
        moves = []
        scores = []
        for move in self.board.generate_legal_moves()
            # Play move and convert board to state
            self.board.push(move)
            state = data.state_from_board(self.board)
            self.board.pop()

            # Predict state score
            score = self.model.predict(state)
            moves.append(move)
            scores.append(score)

        idx = np.argmin(scores)
        self.moves = [moves[idx]]

if __name__ == "__main__":
    engine = ValueBaselineEngine("value_baseline_model.h5")
    engine.run()
