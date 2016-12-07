from data import Dataset

# d = Dataset("data/small.pgn")
# d = Dataset("data/STS1.epd")
d = Dataset("data/medium.pgn")
# d = Dataset("data/CCRL-4040.[671444].pgn")
# d = Dataset("data/ficsgamesdb_1999-2015_standard2000_nomovetimes.pgn")

# for state, reward in d.random_black_state():
#     print(state.shape, reward)

# X, y = d.load("white_state_action_sl")

# X, y = d.strategic_test_suite()
# print(X.shape, y.shape)

for state, action in d.white_state_action_sl():
    print(state.shape, action.shape)

# import chess
# import chess.pgn
# import data
# with open("data/small.pgn") as f:
#     game = chess.pgn.read_game(f)
#     board = game.board()
#     piece_tuples = data.state_from_board(board, featurized=True)
