#!/usr/bin/env python3
from keras.models import load_model
import numpy as np
import sys
sys.path.append('.')
from engines.ChessEngine import ChessEngine
import data
import random
NUM_TRIES = 10

class PolicyEngine(ChessEngine):
    def __init__(self, model_hdf5=None, black=False):
        super().__init__()
        if model_hdf5 is not None:
            self.model = load_model(model_hdf5)
            self.is_black = black

    def search(self, boards=None):
        if boards is None:
            boards = [self.board]

        # Create X batch
        batch_size = len(boards)
        states = [data.state_from_board(board, featurized=True, black=self.is_black) for board in boards]
        X = np.array(states)

        # Predict batch
        y_hat_from, y_hat_to = self.model.predict(X, batch_size=batch_size, verbose=0)

        moves = []
        y_from = []
        y_to = []
        num_random = 0
        for i, board in enumerate(boards):
            # Multiply probabilities
            p = np.outer(y_hat_from[i], y_hat_to[i])
            p_shape = p.shape
            p = p.reshape((-1,))

            # Find max probability action
            move = None
            num_non_nan = np.count_nonzero(~np.isnan(p))
            if num_non_nan == 0:
                #  print("WARNING: Model predictions are all NaN", file=sys.stderr)
                raise Exception("WARNING: Model predictions are all NaN")
            idx_random = np.random.choice(p.shape[0], min(NUM_TRIES, np.count_nonzero(p), num_non_nan), replace=False, p=p)
            for idx in idx_random:
                from_square, to_square = np.unravel_index(idx, p_shape)
                move_attempt = data.move_from_action(from_square, to_square, black=self.is_black)
                if board.is_legal(move_attempt):
                    move = move_attempt
                    break
            if move is None:
                num_random += 1
                move = random.choice(list(board.generate_legal_moves()))
            moves.append(move)
            a_from, a_to = data.action_from_move(move, black=self.is_black)
            y_from.append(a_from)
            y_to.append(a_to)
        
        # Return moves for UCI
        if moves:
            self.moves = [moves[0]]
        else:
            self.moves = None

        print("Random moves: %d out of %d" % (num_random, batch_size), "black" if self.is_black else "white")
        y_from = np.array(y_from)
        y_to = np.array(y_to)
        return X, [y_from, y_to], moves

if __name__ == "__main__":
    engine = PolicyEngine("./saved/black_model.hdf5")
    engine.run()
