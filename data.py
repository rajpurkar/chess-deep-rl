import sys
import random
import chess
import chess.pgn
import numpy as np

NUM_PIECES = len(chess.PIECE_TYPES)
NUM_COLORS = len(chess.COLORS)
NUM_SQUARES = len(chess.SQUARE_NAMES)
NUM_COLS = 8
NUM_ROWS = 8

GAMMA = 0.99

def state_from_board(board, hashable=False):
    if not hashable:
        state = np.zeros((NUM_COLORS * NUM_PIECES, NUM_ROWS, NUM_COLS))
        for piece_type in chess.PIECE_TYPES:
            for color in chess.COLORS:
                pieces = bin(board.pieces(piece_type, color))
                for i, piece in enumerate(reversed(pieces)):
                    if piece == 'b':
                        break
                    elif piece == '1':
                        row = i // NUM_ROWS
                        col = i % NUM_ROWS
                        state[(1-color)*NUM_PIECES + piece_type - 1, row, col] = 1
    else:
        state = [0] * NUM_SQUARES
        for piece_type in chess.PIECE_TYPES:
            for color in chess.COLORS:
                pieces = bin(board.pieces(piece_type, color))
                for i, piece in enumerate(reversed(pieces)):
                    if piece == 'b':
                        break
                    elif piece == '1':
                        state[i] = (1-color)*NUM_PIECES + piece_type - 1
                        # row = i // NUM_ROWS
                        # col = i % NUM_ROWS
                        # state[row, col] = (1-color)*NUM_PIECES + piece_type - 1
        state = str(state)
    return state

def action_from_board(board, move):
    if type(move) is chess.Move:
        # piece - 1: pawn, 2: knight, 3: bishop, 4: rook, 5: queen, 6: king
        piece = board.piece_at(move.from_square).piece_type
        # square order - a1 b1 ... h1 a2 ... h2 ... h8
        return (piece, move.from_square, move.to_square)

class Dataset:
    def __init__(self, filename, loop=False):
        self.filename = filename
        self.loop = loop
        self.idx_game = 0
        self.num_games = 0
        # self.idx_moves = []

    def load(self, generator):
        assert(type(generator) == str)
        try:
            X_y = self.unpickle(generator)
        except:
            X_y = self.pickle(generator)
        return X_y

    def pickle(self, generator):
        X = []
        Y = []
        for x, y in getattr(self, generator)():
            X.append(np.squeeze(x))
            Y.append(np.squeeze(y))
        X = np.array(X)
        Y = np.array(Y)
        np.save(self.filename + "." + generator + ".X.npy", X)
        np.save(self.filename + "." + generator + ".y.npy", Y)
        return X, Y

    def unpickle(self, generator):
        X = np.load(self.filename + "." + generator + ".X.npy")
        y = np.load(self.filename + "." + generator + ".y.npy")
        return X, y

    def white_sarsa(self):
        with open(self.filename) as pgn:
            game = chess.pgn.read_game(pgn)
            idx_move = 0
            num_moves = int(game.headers["PlyCount"])
            board = game.board()
            node = game.root()
            s = state_from_board(board, hashable=True)
            s_prime = s
            while True:
                if idx_move >= num_moves or num_moves <= 4:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        # EOF
                        break

                    # Make sure game was played all the way through
                    last_node = game.root()
                    while last_node.variations:
                        last_node = last_node.variations[0]
                    if "forfeit" in last_node.comment:
                        continue

                    # Setup game and make sure it has enough moves
                    idx_move = 0
                    num_moves = int(game.headers["PlyCount"])
                    board = game.board()
                    node = game.root()
                    continue

                new_game = (idx_move == 0)

                try:
                    # Play white
                    s = s_prime
                    move = node.variations[0].move
                    board.push(move)
                    a = move.from_square * NUM_SQUARES + move.to_square
                    idx_move += 1

                    # Play black
                    node = node.variations[0]
                    if node.variations:
                        move = node.variations[0].move
                        board.push(move)
                        node = node.variations[0]
                        idx_move += 1

                    s_prime = state_from_board(board, hashable=True)

                    a_prime = None
                    if node.variations:
                        move = node.variations[0].move
                        a_prime = move.from_square * NUM_SQUARES + move.to_square

                    r = 0
                    if idx_move >= num_moves:
                        # Parse result from header
                        white_score = game.headers["Result"].split("-")[0].split("/")
                        if len(white_score) == 1:
                            r = 2 * int(white_score[0]) - 1
                except:
                    print("ERROR: ", s, a, r, s_prime, a_prime, game, idx_move, num_moves)
                    idx_move = num_moves
                    continue

                yield s, a, r, s_prime, a_prime, new_game

    def white_state_action_sl(self):
        """
        Returns (state, action) tuple from white's perspective
        - state: np.array [12 pieces x 64 squares]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - square order: a1 b1 c1 ... h8
        - action: np.array [6 pieces x 1] representing piece type
            - piece type: p n b r q k
        """
        with open(self.filename) as pgn:
            game = chess.pgn.read_game(pgn)
            idx_move = 0
            num_moves = int(game.headers["PlyCount"])
            board = game.board()
            node = game.root()

            while True:
                if idx_move >= num_moves or num_moves <= 4:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        # EOF
                        break

                    # Make sure game was played all the way through
                    last_node = game.root()
                    while last_node.variations:
                        last_node = last_node.variations[0]
                    if "forfeit" in last_node.comment:
                        continue

                    # Setup game and make sure it has enough moves
                    idx_move = 0
                    num_moves = int(game.headers["PlyCount"])
                    board = game.board()
                    node = game.root()
                    continue

                try:
                    s = state_from_board(board).reshape((1, NUM_COLORS * NUM_PIECES, NUM_ROWS, NUM_COLS))
                    move = node.variations[0].move
                    (piece_type, from_square, to_square) = action_from_board(board, move)
                    a = np.zeros((1,NUM_PIECES))
                    a[0, piece_type - 1] = 1

                    # Play white
                    board.push(move)
                    idx_move += 1

                    # Play black
                    node = node.variations[0]
                    if node.variations:
                        move = node.variations[0].move
                        board.push(move)
                        idx_move += 1

                        if node.variations:
                            node = node.variations[0]

                except Exception as e:
                    print(e, file=sys.stderr)
                    print("ERROR: ", s, a, r, s_prime, a_prime, game, idx_move, num_moves, file=sys.stderr)
                    idx_move = num_moves
                    continue

                yield s, a

    def random_white_state(self):
        """
        Returns (state, action, reward) tuple from white's perspective
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
                if num_moves < 2:
                    continue

                # Make sure game was played all the way through
                last_node = game.root()
                while last_node.variations:
                    last_node = last_node.variations[0]
                if "forfeit" in last_node.comment:
                    continue

                # Choose a random white-turn state
                idx_move = random.randint(1, num_moves // 2) * 2
                moves_remaining = num_moves - idx_move

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

                state = state_from_board(board)

                yield state, action, result

    def random_black_state(self):
        """
        Returns (state, reward) tuple at black's turn from white's perspective
        - state: np.array [12 pieces x 8 rows x 8 cols]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - row order: a b c ... h
            - col order: 1 2 3 ... 8
        - reward: GAMMA^moves_remaining * {-1, 0, 1} (lose, draw, win)
        """
        with open(self.filename) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    if not self.loop:
                        break
                    pgn.seek(0)
                    self.num_games = self.idx_game
                    self.idx_game = 0
                    continue

                num_moves = int(game.headers["PlyCount"])
                if num_moves < 2:
                    continue

                # Make sure game was played all the way through
                last_node = game.root()
                while last_node.variations:
                    last_node = last_node.variations[0]
                if "forfeit" in last_node.comment:
                    continue

                # Choose a random black-turn state
                # if self.test_set:
                #     if self.num_games:
                #         idx_move = self.idx_moves[self.idx_game]
                #     else:
                #         idx_move = random.randint(1, num_moves // 2) * 2 - 1
                #         self.idx_moves.append(idx_move)
                # else:
                #     idx_move = random.randint(1, num_moves // 2) * 2 - 1
                idx_move = random.randint(1, num_moves // 2) * 2 - 1
                moves_remaining = (num_moves - idx_move) // 2

                # Play moves up to idx_move
                board = game.board()
                node = game.root()
                for i in range(idx_move):
                    board.push(node.variations[0].move)
                    node = node.variations[0]

                # headers["Result"]: {"0-1", "1-0", "1/2-1/2"}
                # result: {-1, 0, 1}
                # Parse result from header
                white_score = game.headers["Result"].split("-")[0].split("/")
                if len(white_score) == 1:
                    result = 2 * int(white_score[0]) - 1
                else:
                    result = 0

                state = state_from_board(board).reshape((1, NUM_COLORS * NUM_PIECES, NUM_ROWS, NUM_COLS))
                reward = np.array([(GAMMA ** moves_remaining) * result])

                self.idx_game += 1
                yield state, reward

    def strategic_test_suite(self):
        """
        Returns (state, action) tuple from white's perspective
        - state: np.array [12 pieces x 64 squares]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - square order: a1 b1 c1 ... h8
        - action: np.array [6 pieces x 1] representing piece type
            - piece type: p n b r q k
        """
        with open(self.filename) as epd:
            game = chess.Game()
            board = game.board()
            for line in epd:
                print(line)
                board.set_epd(line)
                print(game)
                return game

            return None
            # game = chess.pgn.read_game(pgn)
            # idx_move = 0
            # num_moves = int(game.headers["PlyCount"])
            # board = game.board()
            # node = game.root()

            # while True:
            #     if idx_move >= num_moves or num_moves <= 4:
            #         game = chess.pgn.read_game(pgn)
            #         if game is None:
            #             # EOF
            #             break

            #         # Make sure game was played all the way through
            #         last_node = game.root()
            #         while last_node.variations:
            #             last_node = last_node.variations[0]
            #         if "forfeit" in last_node.comment:
            #             continue

            #         # Setup game and make sure it has enough moves
            #         idx_move = 0
            #         num_moves = int(game.headers["PlyCount"])
            #         board = game.board()
            #         node = game.root()
            #         continue

            #     try:
            #         s = state_from_board(board).reshape((1, NUM_COLORS * NUM_PIECES, NUM_ROWS, NUM_COLS))
            #         move = node.variations[0].move
            #         (piece_type, from_square, to_square) = action_from_board(board, move)
            #         a = np.zeros((1,NUM_PIECES))
            #         a[0, piece_type - 1] = 1

            #         # Play white
            #         board.push(move)
            #         idx_move += 1

            #         # Play black
            #         node = node.variations[0]
            #         if node.variations:
            #             move = node.variations[0].move
            #             board.push(move)
            #             idx_move += 1

            #             if node.variations:
            #                 node = node.variations[0]

            #     except Exception as e:
            #         print(e, file=sys.stderr)
            #         print("ERROR: ", s, a, r, s_prime, a_prime, game, idx_move, num_moves, file=sys.stderr)
            #         idx_move = num_moves
            #         continue

            #     yield s, a

    # def load_games(self):
    #     with open(self.filename) as pgn:
    #         self.games = []
    #         while True:
    #             game = chess.pgn.read_game(pgn)
    #             if game is None:
    #                 break

    #             num_moves = int(game.headers["PlyCount"])
    #             if num_moves < 2:
    #                     continue

    #             self.games.append(game)
    #         self.num_games = len(self.games)
    #         self.idx_game = 0

    # def pickle_games(self, filename):
    #     with open("filename", "w") as f:
    #         p = pickle.Pickler(f)
    #         p.dump(self.games)

    # def unpickle_games(self, filename):
    #     with open("filename", "w") as f:
    #         up = pickle.Unpickler(f)
    #         self.games = up.load()

    # def set_batch_size(self, batch_size):
    #     self.batch_size = batch_size
    #     self.num_batches = self.num_games // self.batch_size

    # def random_black_state_batch(self):
    #     for idx_batch in range(self.num_batches):
    #         X = []
    #         y = []
    #         for i in np.random.permutation(self.batch_size):
    #             game = self.games[self.batch_size * idx_batch + i]
    #             state, reward = self.fetch_black_state(game)
    #             X.append(state)
    #             y.append(reward)
    #         X = np.array(X)
    #         y = np.array(y)
    #         yield X, y

    # def fetch_black_state(self, game):
    #     num_moves = int(game.headers["PlyCount"])
    #     if num_moves < 2:
    #         return
    #     # Choose a random black-turn state
    #     idx_move = random.randint(1, int(num_moves / 2)) * 2 - 1
    #     moves_remaining = num_moves - idx_move

    #     # Play moves up to idx_move
    #     board = game.board()
    #     node = game.root()
    #     for i in range(idx_move):
    #         board.push(node.variations[0].move)
    #         node = node.variations[0]

    #     # headers["Result"]: {"0-1", "1-0", "1/2-1/2"}
    #     # result: {-1, 0, 1}
    #     # Parse result from header
    #     white_score = game.headers["Result"].split("-")[0].split("/")
    #     if len(white_score) == 1:
    #         result = 2 * int(white_score[0]) - 1
    #     else:
    #         result = 0

    #     state = np.zeros((1, NUM_COLORS * NUM_PIECES, NUM_ROWS, NUM_COLS))
    #     for piece_type in chess.PIECE_TYPES:
    #         for color in chess.COLORS:
    #             pieces = bin(board.pieces(piece_type, color))
    #             for i, piece in enumerate(reversed(pieces)):
    #                 if piece == 'b':
    #                     break
    #                 elif piece == '1':
    #                     row = i // NUM_ROWS
    #                     col = i % NUM_ROWS
    #                     state[0, (1-color)*NUM_PIECES + piece_type - 1, row, col] = 1

    #     return state, np.array([(GAMMA ** moves_remaining) * result])

    # def random_black_state(self):
    #     """
    #     Returns (state, reward, moves_remaining) tuple at black's turn from white's perspective
    #     - state: np.array [12 pieces x 8 rows x 8 cols]
    #         - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
    #         - row order: a b c ... h
    #         - col order: 1 2 3 ... 8
    #     - result: {-1, 0, 1} (lose, draw, win)
    #     - moves_remaining: number of moves left to the end of the game
    #     """
    #     with open(self.filename) as pgn:
    #         while True:
    #             if self.games:
    #                 if self.idx_game >= self.num_games:
    #                     break

    #                 game = self.games[self.idx_game]
    #                 self.idx_game += 1
    #             else:
    #                 game = chess.pgn.read_game(pgn)
    #                 if game is None:
    #                     break

    #             yield self.fetch_black_state(game)
