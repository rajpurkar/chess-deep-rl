from engines.PolicyEngine import PolicyEngine
import chess
import numpy as np
import os

FOLDER_TO_SAVE = "./saved/"
NUM_GAMES_PER_BATCH = 10#128
NUMBER_EPOCHS = 1  # some large number
VERBOSE_LEVEL = 1
MAX_TURNS_PER_GAME = 100

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
            return -0.01
        elif white_score is "0":
            return -1
        elif white_score is "1":
            return 1
    return None

def filter_finished_games(boards, scores, white_scores, black_scores, num_white_moves, num_black_moves):
    # Filter out finished games
    boards_next = {}
    for idx, board in boards.items():
        result = get_result(board)
        if result is not None:
            white_scores[idx] = [result] * num_white_moves[idx]
            black_scores[idx] = [-result] * num_black_moves[idx]
            if result >= 1:
                scores[0] += 1
            elif result <= -1:
                scores[1] += 1
            else:
                scores[2] += 1
            continue
        boards_next[idx] = board
    return boards_next

def play_engine_move(engine, boards, states, actions_from, actions_to, num_moves):
    X, [y_from, y_to], moves = engine.search(boards)
    for i, (idx, board) in enumerate(boards.items()):
        if not board.is_legal(moves[i]):
            print(moves[i])
            print(board)
            raise("Move not legal")
        board.push(moves[i])
        states[idx].append(np.expand_dims(X[i, :], axis=0))
        actions_from[idx].append(np.expand_dims(y_from[i, :], axis=0))
        actions_to[idx].append(np.expand_dims(y_to[i, :], axis=0))
        num_moves[idx] += 1

    return board

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
    white_scores = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    black_scores = [[] for _ in range(NUM_GAMES_PER_BATCH)]

    # [white, black, draw]
    scores = [0, 0, 0]

    i = 0
    while True:
        if i == MAX_TURNS_PER_GAME:
            for idx, board in boards.items():
                white_scores[idx] = [0] * num_white_moves[idx]
                black_scores[idx] = [0] * num_black_moves[idx]
                scores[2] += 1
            break

        boards = filter_finished_games(boards, scores, white_scores, black_scores, num_white_moves, num_black_moves)
        if not boards:
            break

        # Play white
        play_engine_move(white_engine, boards, white_states, white_actions_from, white_actions_to, num_white_moves)

        # Filter out finished games
        boards = filter_finished_games(boards, scores, white_scores, black_scores, num_white_moves, num_black_moves)
        if not boards:
            break

        # Play black
        board_to_print = play_engine_move(black_engine, boards, black_states, black_actions_from, black_actions_to, num_black_moves)

        i += 1
        #  os.system("clear")
        #  print(board_to_print)
        #  print("White: %d   Black: %d   Draw: %d" % (scores[0], scores[1], scores[2]))

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
    white_states = np.concatenate([white_states[i] for i in white_idx])
    white_actions_from = np.concatenate([white_actions_from[i] for i in white_idx])
    white_actions_to = np.concatenate([white_actions_to[i] for i in white_idx])
    white_scores = np.array([white_scores[i] for i in white_idx])
    black_idx = list(np.random.permutation(len(black_states)))
    black_states = np.concatenate([black_states[i] for i in black_idx])
    black_actions_from = np.concatenate([black_actions_from[i] for i in black_idx])
    black_actions_to = np.concatenate([black_actions_to[i] for i in black_idx])
    black_scores = np.array([black_scores[i] for i in black_idx])

    print("White: %d   Black: %d   Draw: %d" % (scores[0], scores[1], scores[2]))

    return (white_states, [white_actions_from, white_actions_to], white_scores), \
           (black_states, [black_actions_from, black_actions_to], black_scores)


def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{loss:.2f}.hdf5"


import time
import os

def train(engine, X, y, r, engine_type):
    from keras.callbacks import ModelCheckpoint

    start_time = str(int(time.time()))
    #  checkpointer = ModelCheckpoint(filepath=get_filename_for_saving(engine_type + '_policy_rl', start_time), verbose=2, save_best_only=True)
    #  engine.model.fit(X, y, sample_weight=[r,r], nb_epoch=NUMBER_EPOCHS, callbacks=[checkpointer], verbose=VERBOSE_LEVEL)
    engine.model.fit(X, y, sample_weight=[r,r], nb_epoch=NUMBER_EPOCHS, verbose=VERBOSE_LEVEL)


if __name__ == "__main__":
    print("Initializing engines")
    white_model_hdf5 = "saved/white_model.hdf5"
    black_model_hdf5 = "saved/black_model.hdf5"

    white_engine = PolicyEngine(white_model_hdf5)
    black_engine = PolicyEngine(black_model_hdf5)

    print("Begin play")
    while True:
        white_sar, black_sar = play(white_engine, black_engine)
        train(white_engine, white_sar[0], white_sar[1], white_sar[2], "white")
        train(black_engine, black_sar[0], black_sar[1], black_sar[2], "black")
