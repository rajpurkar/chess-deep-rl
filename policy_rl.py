from engines.PolicyEngine import PolicyEngine
import chess
import numpy as np

NUM_GAMES_PER_BATCH = 12
NUMBER_EPOCHS = 1  # some large number
VERBOSE_LEVEL = 1


def custom_result(board):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return "0-1"
        else:
            return "1-0"

    if (board.is_insufficient_material() or
            not any(board.generate_legal_moves())):
        return "1/2-1/2"

    return "*"


def get_result(board):
    result = custom_result(board)
    if result is not "*":
        white_score = result.split("-")[0]
        if len(white_score) > 1:
            return 0
        elif white_score is "0":
            return -1
        elif white_score is "1":
            return 1
    return None


def play(white_engine, black_engine):
    """
    Play NUM_GAMES_PER_BATCH games and return X, y
    - X: np.array [n states (total from all games) x state dim]
    - y: np.array [n actions (total from all games) x action dim]
    - r: np.array [n results (total from all games) x 1]
    """
    boards = {i: chess.Board() for i in range(NUM_GAMES_PER_BATCH)}
    num_white_moves = [0] * NUM_GAMES_PER_BATCH
    num_black_moves = [0] * NUM_GAMES_PER_BATCH

    white_states = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    white_actions_from = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    white_actions_to = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    black_states = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    black_actions_from = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    black_actions_to = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    white_scores = [None] * NUM_GAMES_PER_BATCH
    black_scores = [None] * NUM_GAMES_PER_BATCH

    scores = [0, 0, 0]

    while True:
        # Filter out finished games
        boards_next = {}
        for idx, board in boards.items():
            result = get_result(board)
            if result is not None:
                white_scores[idx] = [result] * num_white_moves[idx]
                black_scores[idx] = [-result] * num_black_moves[idx]
                if result > 0:
                    scores[0] += 1
                elif result < 0:
                    scores[1] += 1
                else:
                    scores[2] += 1
                continue
            boards_next[idx] = board
        boards = boards_next
        if not boards:
            break

        # Play white
        X, [y_from, y_to], moves = white_engine.search(boards)
        for i, (idx, board) in enumerate(boards.items()):
            if not board.is_legal(moves[i]):
                print(moves[i])
                print(board)
                return
            board.push(moves[i])
            white_states[idx].append(np.expand_dims(X[i, :], axis=0))
            white_actions_from[idx].append(np.expand_dims(y_from[i, :], axis=0))
            white_actions_to[idx].append(np.expand_dims(y_to[i, :], axis=0))
            num_white_moves[idx] += 1

        # Filter out finished games
        boards_next = {}
        for idx, board in boards.items():
            result = get_result(board)
            if result is not None:
                white_scores[idx] = [result] * num_white_moves[idx]
                black_scores[idx] = [-result] * num_black_moves[idx]
                if result > 0:
                    scores[0] += 1
                elif result < 0:
                    scores[1] += 1
                else:
                    scores[2] += 1
                continue
            boards_next[idx] = board
        boards = boards_next
        if not boards:
            break

        # Play black
        X, [y_from, y_to], moves = black_engine.search(boards)
        for i, (idx, board) in enumerate(boards.items()):
            if not board.is_legal(moves[i]):
                print(moves[i])
                print(board)
                return
            board.push(moves[i])
            black_states[idx].append(np.expand_dims(X[i, :], axis=0))
            black_actions_from[idx].append(np.expand_dims(y_from[i, :], axis=0))
            black_actions_to[idx].append(np.expand_dims(y_to[i, :], axis=0))
            num_black_moves[idx] += 1

        #print(board)
    # Flatten lists
    white_states = [a for game in white_states for a in game]
    black_states = [a for game in black_states for a in game]
    white_actions_from = [a for game in white_actions_from for a in game]
    white_actions_to = [a for game in white_actions_to for a in game]
    black_actions_from = [a for game in black_actions_from for a in game]
    black_actions_to = [a for game in black_actions_to for a in game]
    white_scores = [score for game in white_scores for score in game]
    black_scores = [score for game in black_scores for score in game]

    # Shuffle lists
    white_idx = list(np.random.permutation(len(white_states)))
    white_states = np.array([white_states[i] for i in white_idx])
    white_actions_from = np.array([white_actions_from[i] for i in white_idx])
    white_actions_to = np.array([white_actions_to[i] for i in white_idx])
    white_scores = np.array([white_scores[i] for i in white_idx])
    black_idx = list(np.random.permutation(len(black_states)))
    black_states = np.array([black_states[i] for i in black_idx])
    black_actions_from = np.array([black_actions_from[i] for i in black_idx])
    black_actions_to = np.array([black_actions_to[i] for i in black_idx])
    black_scores = np.array([black_scores[i] for i in black_idx])

    return
    (
        white_states,
        [white_actions_from, white_actions_to],
        white_scores),
    (
        black_states,
        [black_actions_from, black_actions_to],
        black_scores), scores


"""
def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{val_loss:.2f}.hdf5"

import time
import os

def train(engine, X, y, r):
    from keras.callbacks import ModelCheckpoint

    start_time = str(int(time.time()))
    checkpointer = ModelCheckpoint(filepath=get_filename_for_saving('policy_rl', start_time), verbose=2, save_best_only=True)
    engine.model.fit(X, y, sample_weight=r, batch_size=BATCH_SIZE, nb_epoch=NUMBER_EPOCHS, callbacks=[checkpointer], verbose=VERBOSE_LEVEL)
"""

if __name__ == "__main__":
    print("Initializing engines")
    white_model_hdf5 = "saved/white_model.hdf5"
    black_model_hdf5 = "saved/black_model.hdf5"

    white_engine = PolicyEngine(white_model_hdf5)
    black_engine = PolicyEngine(black_model_hdf5)

    print("Begin play")
    while True:
        white_sar, black_sar, scores = play(white_engine, black_engine)
        print(scores)
        break
        # train(white_engine, white_sar[0], white_sar[1], white_sar[2])
        # train(black_engine, black_sar[0], black_sar[1], black_sar[2])
