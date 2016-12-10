from engines.PolicyEngine import PolicyEngine
import chess
import numpy as np
import os

FOLDER_TO_SAVE = "./saved/"

NUMBER_EPOCHS = 10000  # some large number
SAMPLES_PER_EPOCH = 10016  # tune for feedback/speed balance
VERBOSE_LEVEL = 1

NUM_PARALLELL_GAMES = 128
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

    def play_engine_move(self, engine, states, actions_from, actions_to):
        X, [y_from, y_to], moves = engine.search(self.boards)
        for i, board in enumerate(self.boards):
            if not board.is_legal(moves[i]):
                print(moves[i])
                print(board)
                raise Exception("Move not legal")

            board.push(moves[i])
            states[i].append(X[i])
            actions_from[i].append(y_from[i])
            actions_to[i].append(y_to[i])

    def collect_game_results(self):
        # Check game results
        for i, board in enumerate(self.boards):
            result = get_result(board)
            if result is not None or len(self.white_states[i]) > MAX_TURNS_PER_GAME:
                # Add game to finished pool
                if result >= 1:
                    self.finished_win_states        += self.white_states[i]
                    self.finished_win_actions_from  += self.white_actions_from[i]
                    self.finished_win_actions_to    += self.white_actions_to[i]
                    self.finished_lose_states       += self.black_states[i]
                    self.finished_lose_actions_from += self.black_actions_from[i]
                    self.finished_lose_actions_to   += self.black_actions_to[i]
                elif result <= -1:
                    self.finished_win_states        += self.black_states[i]
                    self.finished_win_actions_from  += self.black_actions_from[i]
                    self.finished_win_actions_to    += self.black_actions_to[i]
                    self.finished_lose_states       += self.white_states[i]
                    self.finished_lose_actions_from += self.white_actions_from[i]
                    self.finished_lose_actions_to   += self.white_actions_to[i]

                # Reset board
                self.boards[i] = chess.Board()
                self.white_states[i]       = []
                self.white_actions_from[i] = []
                self.white_actions_to[i]   = []
                self.black_states[i]       = []
                self.black_actions_from[i] = []
                self.black_actions_to[i]   = []

                # Play a single white move if it's black's turn for the other boards
                if self.black_turn:
                    X, [y_from, y_to], moves = self.white_engine.search([self.boards[i]])
                    self.white_states[i].append(X[0])
                    self.white_actions_from[i].append(y_from[0])
                    self.white_actions_to[i].append(y_to[0])

                # Update scoreboard
                if result >= 1:
                    self.scoreboard[0] += 1
                elif result <= -1:
                    self.scoreboard[1] += 1
                elif result == 0:
                    self.scoreboard[2] += 1
                else:
                    self.scoreboard[3] += 1

    def play_generator(self):
        self.boards = [chess.Board() for i in range(NUM_PARALLELL_GAMES)]

        self.white_states       = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.white_actions_from = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.white_actions_to   = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.black_states       = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.black_actions_from = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.black_actions_to   = [[] for _ in range(NUM_PARALLELL_GAMES)]

        self.finished_win_states        = []
        self.finished_win_actions_from  = []
        self.finished_win_actions_to    = []
        self.finished_lose_states       = []
        self.finished_lose_actions_from = []
        self.finished_lose_actions_to   = []

        # [white, black, draw]
        self.scoreboard = [0, 0, 0, 0]

        self.black_turn = False

        while True:
            # Play white move in all games
            self.play_engine_move(self.white_engine, self.white_states, self.white_actions_from, self.white_actions_to)
            self.black_turn = not self.black_turn
            self.collect_game_results()

            # Play black move in all games
            self.play_engine_move(self.black_engine, self.black_states, self.black_actions_from, self.black_actions_to)
            self.black_turn = not self.black_turn
            self.collect_game_results()

            os.system("clear")
            print(self.boards[i])
            print("White: %d   Black: %d   Draw: %d   Endless: %d" % (scores[0], scores[1], scores[2], scores[3]))

            # Yield won games
            while len(finished_win_states) > BATCH_SIZE:
                idx_random = list(np.random.permutation(len(finished_win_states)))
                win_states_shuffle       = [self.finished_win_states[idx] for idx in idx_random]
                win_actions_from_shuffle = [self.finished_win_actions_from[idx] for idx in idx_random]
                win_actions_to_shuffle   = [self.finished_win_actions_to[idx] for idx in idx_random]
                X_win      = np.array(win_states_shuffle[:BATCH_SIZE])
                y_from_win = np.array(win_actions_from_shuffle[:BATCH_SIZE])
                y_to_win   = np.array(win_actions_to_shuffle[:BATCH_SIZE])
                self.finished_win_states       = win_states_shuffle[BATCH_SIZE:]
                self.finished_win_actions_from = win_actions_from_shuffle[BATCH_SIZE:]
                self.finished_win_actions_to   = win_actions_to_shuffle[BATCH_SIZE:]

                learning_rate = abs(engine.model.optimizer.lr.get_value())
                engine.model.optimizer.lr.set_value(learning_rate)

                yield X_win, [y_from_win, y_to_win]

            # Yield lost games
            while len(finished_lose_states) > BATCH_SIZE:
                idx_random = list(np.random.permutation(len(finished_win_states)))
                lose_states_shuffle       = [self.finished_lose_states[idx] for idx in idx_random]
                lose_actions_from_shuffle = [self.finished_lose_actions_from[idx] for idx in idx_random]
                lose_actions_to_shuffle   = [self.finished_lose_actions_to[idx] for idx in idx_random]
                X_lose      = np.array(lose_states_shuffle[:BATCH_SIZE])
                y_from_lose = np.array(lose_actions_from_shuffle[:BATCH_SIZE])
                y_to_lose   = np.array(lose_actions_to_shuffle[:BATCH_SIZE])
                self.finished_lose_states       = lose_states_shuffle[BATCH_SIZE:]
                self.finished_lose_actions_from = lose_actions_from_shuffle[BATCH_SIZE:]
                self.finished_lose_actions_to   = lose_actions_to_shuffle[BATCH_SIZE:]

                learning_rate = abs(engine.model.optimizer.lr.get_value())
                engine.model.optimizer.lr.set_value(-learning_rate)

                yield X_lose, [y_from_lose, y_to_lose]

    def play(self):
        """
        Play NUM_PARALLELL_GAMES games and return X, y
        - X: np.array [n states (total from all games) x state dim]
        - y: np.array [n actions (total from all games) x action dim]
        - r: np.array [n results (total from all games) x 1]
        """

        self.boards = {i: chess.Board() for i in range(NUM_PARALLELL_GAMES)}
        self.num_white_moves = [0] * NUM_PARALLELL_GAMES
        self.num_black_moves = [0] * NUM_PARALLELL_GAMES

        white_states = [[] for _ in range(NUM_PARALLELL_GAMES)]
        white_actions_from = [[] for _ in range(NUM_PARALLELL_GAMES)]
        white_actions_to = [[] for _ in range(NUM_PARALLELL_GAMES)]
        black_states = [[] for _ in range(NUM_PARALLELL_GAMES)]
        black_actions_from = [[] for _ in range(NUM_PARALLELL_GAMES)]
        black_actions_to = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.white_scores = [[] for _ in range(NUM_PARALLELL_GAMES)]
        self.black_scores = [[] for _ in range(NUM_PARALLELL_GAMES)]

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

def train(controller):
    from keras.callbacks import ModelCheckpoint

    start_time = str(int(time.time()))

    checkpointer = ModelCheckpoint(
        filepath       = get_filename_for_saving(engine_type + '_policy_rl', start_time),
        verbose        = 2,
        save_best_only = True)

    engine.model.fit_generator(
        controller.play_generator(),
        samples_per_epoch = SAMPLES_PER_EPOCH
        nb_epoch          = NUMBER_EPOCHS,
        # callbacks         = [checkpointer],
        verbose           = VERBOSE_LEVEL)

if __name__ == "__main__":
    white_model_hdf5 = "saved/sl_model.hdf5"
    black_model_hdf5 = "saved/sl_model.hdf5"
    while True:
        print("Initializing engines")
        white_engine = PolicyEngine(white_model_hdf5)
        black_engine = PolicyEngine(black_model_hdf5, black=True)
        controller = SelfPlayController(white_engine, black_engine)

        train(controller)
