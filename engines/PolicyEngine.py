#!/usr/bin/env python3
from engines.ChessEngine import ChessEngine
from keras.models import load_model
import numpy as np
import sys
sys.path.append('.')
import data
import random

class PolicyEngine(ChessEngine):
    def __init__(self, model_hdf5=None, epsilon=0.1):
        super().__init__()
        if model_hdf5 is not None:
            self.model = load_model(model_hdf5)
            self.epsilon = epsilon

    def search(self, boards=None):
        if boards is not None:
            # Create X batch
            states = []
            boards_list = []
            for board in boards.values():
                states.append(data.state_from_board(board, featurized=True))
                boards_list.append(board)
            batch_size = len(states)
            X = np.array(states)

            # Predict batch
            y_hat_from, y_hat_to = self.model.predict(X, batch_size=batch_size, verbose=0)

            # Extract best legal move
            moves = []
            y_from = []
            y_to = []
            for i in range(y_hat_from.shape[0]):
                board = boards_list[i]
                if random.random() < self.epsilon:
                    move = random.choice(list(board.generate_legal_moves()))
                    a_from, a_to = data.action_from_move(move)
                    moves.append(move)
                    y_from.append(a_from)
                    y_to.append(a_to)
                    continue

                    # TODO random item from list

                # Multiply probabilities
                p = np.outer(y_hat_from[i,:], y_hat_to[i,:])
                p_shape = p.shape
                p = p.reshape((-1,))

                # Find max probability action
                appended = False
                for idx in reversed(np.argsort(p).tolist()):
                    from_square, to_square = np.unravel_index(idx, p_shape)
                    move = data.move_from_action(from_square, to_square)
                    if board.is_legal(move):
                        appended = True
                        moves.append(move)
                        a_from, a_to = data.action_from_move(move)
                        y_from.append(a_from)
                        y_to.append(a_to)
                        break

                if not appended:
                    move = random.choice(list(board.generate_legal_moves()))
                    a_from, a_to = data.action_from_move(move)
                    moves.append(move)
                    y_from.append(a_from)
                    y_to.append(a_to)

            y_from = np.array(y_from)
            y_to = np.array(y_to)
            return X, [y_from, y_to], moves

if __name__ == "__main__":
    engine = ValueBaselineEngine("./saved/1480896779-06-0.40.hdf5")
    engine.run()
