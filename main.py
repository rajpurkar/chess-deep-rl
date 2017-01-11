from data import Dataset
import data
import chess
import numpy as np
import traceback

d = Dataset("data/small.pgn")
# d = Dataset("data/STS1.epd")
# d = Dataset("data/medium.pgn")
# d = Dataset("data/CCRL-4040.[671444].pgn")
# d = Dataset("data/ficsgamesdb_1999-2015_standard2000_nomovetimes.pgn")
d_test = Dataset('data/strategic_test_suite/STS1.epd')
X_val, y_val = d_test.load_sts(featurized=True, board_type="to")
print(X_val[0].shape, X_val[1].shape, y_val.shape)

#  for s, a_from in d.state_action_sl(loop=False, board="to"):
#      print(s[1].shape, np.array(a_from).shape)
exit()
# for state, reward in d.random_black_state():
#     print(state.shape, reward)

# X, y = d.load("white_state_action_sl")

# X, y = d.strategic_test_suite()
# print(X.shape, y.shape)

# for i in range(63):
#     move = chess.Move(i,i+9)
#     print(move)
#     a_from, a_to = data.action_from_move(move, black=True)
#     a_from, a_to = (np.where(a_from==1)[0], np.where(a_to==1)[0])
#     a_from_0, a_to_0 = data.action_from_move(move, black=False)
#     a_from_0, a_to_0 = (np.where(a_from_0==1)[0], np.where(a_to_0==1)[0])
#     m = data.move_from_action(a_from, a_to, black=True)
#     m_0 = data.move_from_action(a_from_0, a_to_0, black=False)
#     # m = data.move_from_action(i, i+9, black=True)
#     # m_0 = data.move_from_action(i, i+9, black=False)
#     print(m, m_0)

# board = chess.Board()
# phi = data.state_from_board(board, hashable=True)
# phi_black = data.state_from_board(board, hashable=True, black=True)

# phi = data.featurized_state_from_board(board)
# X, y = d.load("black_phi_action_sl")

try:
    raise Exception("sdfasdf")
except Exception as e:
    with open("policy_engine_error.log","w") as f:
        f.write(traceback.format_exc())

# i = 0
# for s, [a_from, a_to] in d.state_action_sl(loop=False):
#     print("===")
#     for i in range(31):
#         i_from, i_to = (np.where(a_from[i,:]==1)[0], np.where(a_to[i,:]==1)[0])
#         print(data.move_from_action(i_from, i_to))
#     break

# print("board")
# print(state[0,:,:,:].sum(axis=0))
# print("from")
# print(action1[0,:].reshape((8,8)))
# print("to")
# print(action2[0,:].reshape((8,8)))

# import chess
# import chess.pgn
# import data
# with open("data/small.pgn") as f:
#     game = chess.pgn.read_game(f)
#     board = game.board()
#     piece_tuples = data.state_from_board(board, featurized=True)
