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
    """
    Custom implementation of chess.Board.result() without repetition draws
    """
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return "0-1"
        else:
            return "1-0"
    if board.is_insufficient_material() \
       or not any(board.generate_legal_moves()):
        return "1/2-1/2"
    return "*"

def get_result(board):
    """
    Parse result string from chess.Board.result()
    """
    #  result = board.result()
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

class SelfPlayController:

    def __init__(self, white_engine, black_engine):
        self.white_engine = white_engine
        self.black_engine = black_engine

    def filter_finished_games(self):
        # Filter out finished games
        boards_next = {}
        for idx, board in self.boards.items():
            result = get_result(board)
            if result is not None:
                self.white_scores[idx] = [result] * self.num_white_moves[idx]
                self.black_scores[idx] = [-result] * self.num_black_moves[idx]
                if result >= 1:
                    self.scores[0] += 1
                elif result <= -1:
                    self.scores[1] += 1
                else:
                    self.scores[2] += 1
                continue
            boards_next[idx] = board

        self.boards = boards_next
        return len(self.boards)

    def play_engine_move(self, engine, states, actions_from, actions_to, num_moves):
        X, [y_from, y_to], moves = engine.search(self.boards)
        for i, (idx, board) in enumerate(self.boards.items()):
            if not board.is_legal(moves[i]):
                print(moves[i])
                print(board)
                raise("Move not legal")
            board.push(moves[i])
            states[idx].append(np.expand_dims(X[i, :], axis=0))
            actions_from[idx].append(np.expand_dims(y_from[i, :], axis=0))
            actions_to[idx].append(np.expand_dims(y_to[i, :], axis=0))
            num_moves[idx] += 1

    def play(self):
        """
        Play NUM_GAMES_PER_BATCH games and return X, y
        - X: np.array [n states (total from all games) x state dim]
        - y: np.array [n actions (total from all games) x action dim]
        - r: np.array [n results (total from all games) x 1]
        """

        self.boards = {i: chess.Board() for i in range(NUM_GAMES_PER_BATCH)}
        self.num_white_moves = [0] * NUM_GAMES_PER_BATCH
        self.num_black_moves = [0] * NUM_GAMES_PER_BATCH

        white_states = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        white_actions_from = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        white_actions_to = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        black_states = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        black_actions_from = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        black_actions_to = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        self.white_scores = [[] for _ in range(NUM_GAMES_PER_BATCH)]
        self.black_scores = [[] for _ in range(NUM_GAMES_PER_BATCH)]

        # [white, black, draw]
        self.scores = [0, 0, 0]

        i = 0
        while True:
            if i == MAX_TURNS_PER_GAME:
                for idx, board in self.boards.items():
                    self.white_scores[idx] = [0] * self.num_white_moves[idx]
                    self.black_scores[idx] = [0] * self.num_black_moves[idx]
                    self.scores[2] += 1
                break

            # Filter out finished games
            if self.filter_finished_games() == 0:
                break

            # Play white
            self.play_engine_move(self.white_engine, white_states, white_actions_from, white_actions_to, self.num_white_moves)

            # Filter out finished games
            if self.filter_finished_games() == 0:
                break

            # Play black
            self.play_engine_move(self.black_engine, black_states, black_actions_from, black_actions_to, self.num_black_moves)

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
        self.white_scores = [score for game in self.white_scores for score in game]
        self.black_scores = [score for game in self.black_scores for score in game]

        # Shuffle lists
        white_idx = list(np.random.permutation(len(white_states)))
        white_states = np.concatenate([white_states[i] for i in white_idx])
        white_actions_from = np.concatenate([white_actions_from[i] for i in white_idx])
        white_actions_to = np.concatenate([white_actions_to[i] for i in white_idx])
        black_idx = list(np.random.permutation(len(black_states)))
        black_states = np.concatenate([black_states[i] for i in black_idx])
        black_actions_from = np.concatenate([black_actions_from[i] for i in black_idx])
        black_actions_to = np.concatenate([black_actions_to[i] for i in black_idx])
        self.white_scores = np.array([self.white_scores[i] for i in white_idx])
        self.black_scores = np.array([self.black_scores[i] for i in black_idx])

        print("White: %d   Black: %d   Draw: %d" % (self.scores[0], self.scores[1], self.scores[2]))

        return (white_states, [white_actions_from, white_actions_to], self.white_scores), \
               (black_states, [black_actions_from, black_actions_to], self.black_scores)


def get_filename_for_saving(net_type, start_time):
    folder_name = FOLDER_TO_SAVE + net_type + '/' + start_time
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name + "/{epoch:02d}-{loss:.2f}.hdf5"


import time
import os

# TODO: compile model with this loss
def log_loss(y_true, y_pred):
    import keras.backend as K
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.

    Taken from https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/training/reinforcement_policy_trainer.py
    '''
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))

def train(engine, X, y, r, engine_type):
    from keras.callbacks import ModelCheckpoint

    start_time = str(int(time.time()))
    #  checkpointer = ModelCheckpoint(filepath=get_filename_for_saving(engine_type + '_policy_rl', start_time), verbose=2, save_best_only=True)
    #  engine.model.fit(X, y, sample_weight=[r,r], nb_epoch=NUMBER_EPOCHS, callbacks=[checkpointer], verbose=VERBOSE_LEVEL)
    learning_rate = engine.model.optimizer.lr.get_value()
    sample_weight = learning_rate * r
    engine.model.fit(X, y, sample_weight=[sample_weight,sample_weight], nb_epoch=NUMBER_EPOCHS, verbose=VERBOSE_LEVEL)


if __name__ == "__main__":
    print("Initializing engines")
    white_model_hdf5 = "saved/white_model.hdf5"
    black_model_hdf5 = "saved/black_model.hdf5"

    white_engine = PolicyEngine(white_model_hdf5)
    black_engine = PolicyEngine(black_model_hdf5)
    controller = SelfPlayController(white_engine, black_engine)

    print("Begin play")
    while True:
        white_sar, black_sar = controller.play()
        train(white_engine, white_sar[0], white_sar[1], white_sar[2], "white")
        train(black_engine, black_sar[0], black_sar[1], black_sar[2], "black")
