from engines.PolicyEngine import PolicyEngine
import chess
import numpy as np
import os
import time
import random

FOLDER_TO_SAVE = "./saved/policy_rl/"

BATCH_SIZE          = 32
SAMPLES_PER_EPOCH   = 1*BATCH_SIZE
NUMBER_EPOCHS       = 10
VERBOSE_LEVEL       = 1

NUM_PARALLELL_GAMES = 128
MAX_TURNS_PER_GAME  = 100

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
    # result = board.result()
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
                if result is not None:
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
                if result is None:
                    self.scoreboard[3] += 1
                elif result >= 1:
                    self.scoreboard[0] += 1
                elif result <= -1:
                    self.scoreboard[1] += 1
                else:
                    self.scoreboard[2] += 1

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

            # os.system("clear")
            print(self.boards[0])
            print("White: %d   Black: %d   Draw: %d   Endless: %d" % tuple(self.scoreboard))

            # Yield won games
            while len(self.finished_win_states) > BATCH_SIZE:
                idx_random = list(np.random.permutation(len(self.finished_win_states)))
                win_states_shuffle       = [self.finished_win_states[idx] for idx in idx_random]
                win_actions_from_shuffle = [self.finished_win_actions_from[idx] for idx in idx_random]
                win_actions_to_shuffle   = [self.finished_win_actions_to[idx] for idx in idx_random]
                X_win      = np.array(win_states_shuffle[:BATCH_SIZE])
                y_from_win = np.array(win_actions_from_shuffle[:BATCH_SIZE])
                y_to_win   = np.array(win_actions_to_shuffle[:BATCH_SIZE])
                self.finished_win_states       = win_states_shuffle[BATCH_SIZE:]
                self.finished_win_actions_from = win_actions_from_shuffle[BATCH_SIZE:]
                self.finished_win_actions_to   = win_actions_to_shuffle[BATCH_SIZE:]

                learning_rate = abs(self.white_engine.model.optimizer.lr.get_value())
                self.white_engine.model.optimizer.lr.set_value(learning_rate)

                yield X_win, [y_from_win, y_to_win]

            # Yield lost games
            while len(self.finished_lose_states) > BATCH_SIZE:
                idx_random = list(np.random.permutation(len(self.finished_win_states)))
                lose_states_shuffle       = [self.finished_lose_states[idx] for idx in idx_random]
                lose_actions_from_shuffle = [self.finished_lose_actions_from[idx] for idx in idx_random]
                lose_actions_to_shuffle   = [self.finished_lose_actions_to[idx] for idx in idx_random]
                X_lose      = np.array(lose_states_shuffle[:BATCH_SIZE])
                y_from_lose = np.array(lose_actions_from_shuffle[:BATCH_SIZE])
                y_to_lose   = np.array(lose_actions_to_shuffle[:BATCH_SIZE])
                self.finished_lose_states       = lose_states_shuffle[BATCH_SIZE:]
                self.finished_lose_actions_from = lose_actions_from_shuffle[BATCH_SIZE:]
                self.finished_lose_actions_to   = lose_actions_to_shuffle[BATCH_SIZE:]

                learning_rate = abs(self.white_engine.model.optimizer.lr.get_value())
                self.white_engine.model.optimizer.lr.set_value(-learning_rate)

                yield X_lose, [y_from_lose, y_to_lose]

def get_filename_for_saving(start_time):
    folder_name = FOLDER_TO_SAVE
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # return folder_name + "/{epoch:02d}-{loss:.2f}.hdf5"
    return folder_name + "/" + start_time + ".hdf5"

def train(controller, engine):
    from keras.callbacks import ModelCheckpoint

    start_time = str(int(time.time()))
    filename = get_filename_for_saving(start_time),

    checkpointer = ModelCheckpoint(
        filepath       = filename,
        verbose        = 2,
        save_best_only = True)

    engine.model.fit_generator(
        controller.play_generator(),
        samples_per_epoch = SAMPLES_PER_EPOCH,
        nb_epoch          = NUMBER_EPOCHS,
        callbacks         = [checkpointer],
        verbose           = VERBOSE_LEVEL)

    return filename

if __name__ == "__main__":
    print("Initializing engines")
    white_model_hdf5 = "saved/sl_model.hdf5"
    white_engine = PolicyEngine(white_model_hdf5)

    black_model_pool = [white_model_hdf5]
    while True:
        black_model_hdf5 = random.choice(black_model_pool)
        black_engine = PolicyEngine(black_model_hdf5, black=True)

        controller = SelfPlayController(white_engine, black_engine)

        saved_model = train(controller, white_engine)
        black_model_pool.append(saved_model)
