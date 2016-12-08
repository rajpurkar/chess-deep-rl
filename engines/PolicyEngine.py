#!/usr/bin/env python3
from ChessEngine import ChessEngine
from keras.models import load_model
import numpy as np
import sys
sys.path.append('.')
import data

class PolicyEngine(ChessEngine):
    def __init__(self, model_hdf5=None):
        super().__init__()
        if model_hdf5 is not None:
            self.model = load_model(model_hdf5)

    def search(self, boards=None):
        if boards is not None:
            # Create X batch
            states = []
            for board in boards.values():
                states.append(data.state_from_board(board))
            batch_size = len(states)
            X = np.array(states)

            # Predict batch
            y_hat_from, y_hat_to = self.model.predict(X, batch_size=batch_size, verbose=1)

            # Extract best legal move
            moves = []
            y_from = []
            y_to = []
            for i in range(y_hat_from.shape[0]):
                # Multiply probabilities
                p = np.outer(y_hat_from[i,:], y_hat_to[i,:]).reshape((-1,))

                # Find max probability action
                for idx in argsort(p):
                    from_square, to_square = np.unravel_index(idx, p.shape)
                    move = data.move_from_action(from_square, to_square)
                    if board.is_legal(move):
                        moves.append(move)
                        a_from, a_to = data.action_from_move(move)
                        y_from.append(a_from)
                        y_to.append(a_to)
                        break

            y_from = np.array(y_from)
            y_to = np.array(y_to)
            return X, [y_from, y_to], moves

if __name__ == "__main__":
    engine = ValueBaselineEngine("./saved/1480896779-06-0.40.hdf5")
    engine.run()
