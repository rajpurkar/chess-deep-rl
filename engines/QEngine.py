#!/usr/bin/env python3
from ChessEngine import ChessEngine
import chess
import sys
sys.path.append('.')
import data

class QEngine(ChessEngine):
    def __init__(self, picklefile):
        super().__init__()
        with open(picklefile, "rb") as f:
            self.Q = pickle.load(Q, f)

    def search(self):
        s = data.state_from_board(board, hashable=True)
        try:
            a = Q[s]
            from_square = a // NUM_SQUARES
            to_square = a % NUM_SQUARES
            move = chess.Move(from_square, to_square)
        except:
            moves = list(self.board.generate_legal_moves())
            move = random.choice(moves)
        self.moves = [move]

if __name__ == "__main__":
    engine = QEngine("engines/sarsa_Q_-_.pickle")
    engine.run()
