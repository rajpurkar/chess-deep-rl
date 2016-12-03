import sys
import random
import chess
import chess.pgn
import numpy as np

NUM_PIECES = len(chess.PIECE_TYPES)
NUM_COLORS = len(chess.COLORS)
NUM_SQUARES = len(chess.SQUARE_NAMES)
NUM_COLS = 8

class Dataset:
    def __init__(self, filename):
        self.filename = filename

    def random_states(self):
        """
        Returns (state, reward) tuple from white's perspective
        - state: np.array [12 pieces x 64 squares]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - square order: a1 b1 c1 ... h8
        - action: index in array [7 pieces x 64 squares x 64 squares]
            - action_array[ind2sub(action)]: move piece at square j to square k and promote to piece type i
            - promotion piece order: None p n b r q k
        - result: {-1, 0, 1} (lose, draw, win)
        """
        with open(self.filename) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                num_moves = int(game.headers["PlyCount"])
                # Choose a random white-turn state
                idx_move = random.randint(1, int(num_moves / 2)) * 2

                # Play moves up to idx_move
                board = game.board()
                node = game.root()
                for i in range(idx_move):
                    board.push(node.variations[0].move)
                    node = node.variations[0]
                move = node.variations[0].move
                promotion = move.promotion
                if promotion is None:
                    promotion = 0
                action = promotion * NUM_SQUARES * NUM_SQUARES + move.from_square * NUM_SQUARES + move.to_square

                # headers["Result"]: {"0-1", "1-0", "1/2-1/2"}
                # result: {-1, 0, 1}
                # Parse result from header
                white_score = game.headers["Result"].split("-")[0].split("/")
                if len(white_score) == 1:
                    result = 2 * int(white_score[0]) - 1
                else:
                    result = 0

                state = np.zeros((NUM_COLORS*NUM_PIECES, NUM_SQUARES))
                for piece_type in chess.PIECE_TYPES:
                    for color in chess.COLORS:
                        pieces = bin(board.pieces(piece_type, color))
                        for i, piece in enumerate(reversed(pieces)):
                            if piece == 'b':
                                break
                            elif piece == '1':
                                state[(1-color)*NUM_PIECES + piece_type - 1, i] = 1

                yield state, action, result
