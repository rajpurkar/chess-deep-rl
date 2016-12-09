#!/usr/bin/env python3
import sys
sys.path.append('.')
from ChessEngine import ChessEngine
import data
import chess
import pickle
import random

class QEngine(ChessEngine):
    def __init__(self, picklefile):
        super().__init__()
        with open(picklefile, "rb") as f:
            self.Q = pickle.load(f)

    def search(self):
        s = data.state_from_board(self.board, hashable=True)
        try:
            a = self.Q[s]
            from_square = a // data.NUM_SQUARES
            to_square = a % data.NUM_SQUARES
            move = chess.Move(from_square, to_square)
            if not self.board.is_legal(move):
                raise Exception()
        except:
            moves = list(self.board.generate_legal_moves())
            if not moves:
                self.moves = None
                return
            move = random.choice(moves)
        self.moves = [move]

if __name__ == "__main__":
    engine = QEngine("engines/sarsa_Q_CCRL.pickle")
    engine.run()
