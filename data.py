import sys
import random
import chess
import chess.pgn
import numpy as np
import itertools
from tqdm import tqdm
import re

NUM_PIECES = len(chess.PIECE_TYPES)
NUM_COLORS = len(chess.COLORS)
NUM_SQUARES = len(chess.SQUARE_NAMES)
NUM_COLS = 8
NUM_ROWS = 8
BATCH_SIZE = 32
POOL_SIZE = 32

GAMMA = 0.99

def np_flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

def flip_state(state):
    num_axes = len(state.shape)
    # state = np_flip(state, num_axes-1)
    state = np_flip(state, num_axes-2)
    # Swap first six rows with last six rows
    state[[0,1,2,3,4,5,6,7,8,9,10,11]] = state[[6,7,8,9,10,11,0,1,2,3,4,5]]
    if state.shape[0] > 12:
        idx_start_layer = 12
        num_quad_layers = (state.shape[0] - idx_start_layer) // 4
        for i in range(num_quad_layers):
            idx = 4*i + idx_start_layer
            state[[idx+0,idx+3]] = state[[idx+3,idx+0]]
            state[[idx+1,idx+2]] = state[[idx+2,idx+1]]
        idx_start_layer += 4*num_quad_layers
        num_pair_layers = (state.shape[0] - idx_start_layer) // 2
        for i in range(num_pair_layers):
            idx = 2*i + idx_start_layer
            state[[idx+0,idx+1]] = state[[idx+1,idx+0]]
    return state

def state_from_board(board, hashable=False, featurized=False, black=False):
    if featurized:
        phi = featurized_state_from_board(board)
        if black:
            phi = flip_state(phi)
        return phi

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
        if black:
            state = flip_state(state)
    else:
        # TODO: black state
        state = [0] * NUM_SQUARES
        for piece_type in chess.PIECE_TYPES:
            for color in chess.COLORS:
                pieces = bin(board.pieces(piece_type, color))
                for i, piece in enumerate(reversed(pieces)):
                    if piece == 'b':
                        break
                    elif piece == '1':
                        if black:
                            row = NUM_ROWS-1 - (i // NUM_ROWS)
                            col = i % NUM_ROWS
                            state[row*NUM_ROWS + col] = color*NUM_PIECES + piece_type
                        else:
                            state[i] = (1-color)*NUM_PIECES + piece_type
        state = tuple(state)
    return state

def featurized_state_from_board(board):
    def bitmap_to_array(bitmap):
        bitmap = bin(bitmap)[2:].zfill(NUM_SQUARES)
        return np.array([int(i) for i in reversed(bitmap)]).reshape(NUM_ROWS, NUM_COLS)

    def get(arr, idx):
        try:
            return arr[idx]
        except:
            return 0

    def add_to(arr_dst, arr_src, idx):
        for i, val in enumerate(idx):
            if val < 0 or val >= arr_dst.shape[i]:
                return False
        try:
            arr_dst[idx] += arr_src[idx]
            return arr_src[idx] != 0.0
        except:
            return False

    state = state_from_board(board)
    pieces = board.pawns | board.knights | board.bishops | board.rooks | board.queens | board.kings
    white_pieces = bitmap_to_array(pieces & board.occupied_co[chess.WHITE])
    black_pieces = bitmap_to_array(pieces & board.occupied_co[chess.BLACK])
    free_spaces = bitmap_to_array(~np.uint64(pieces))
    white_king = bitmap_to_array(board.kings & board.occupied_co[chess.WHITE])

    WHITE, BLACK, OTHER_WHITE, OTHER_BLACK = (0, 1, 2, 3)
    def apply_mask(mask):
        non_mask = pieces # & (~mask)
        white_mask = bitmap_to_array(mask & board.occupied_co[chess.WHITE])
        black_mask = bitmap_to_array(mask & board.occupied_co[chess.BLACK])
        white_non_mask = bitmap_to_array(non_mask & board.occupied_co[chess.WHITE])
        black_non_mask = bitmap_to_array(non_mask & board.occupied_co[chess.BLACK])
        return white_mask, black_mask, white_non_mask, black_non_mask

    # Knights
    knights = apply_mask(board.knights)
    phi_knights = (np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)))
    idx_knight = tuple([[(row,col)] for row in (2,-2) for col in (1,-1)] + \
                       [[(row,col)] for row in (1,-1) for col in (2,-2)])

    # Rooks
    rooks = apply_mask(board.rooks)
    phi_rooks = (np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)))
    idx_rook = ([(row,0) for row in range(1,NUM_ROWS)], \
                [(0,col) for col in range(1,NUM_COLS)], \
                [(-row,0) for row in range(1,NUM_ROWS)], \
                [(0,-col) for col in range(1,NUM_COLS)])

    # Bishops
    bishops = apply_mask(board.bishops)
    phi_bishops = (np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)))
    idx_bishop = ([(row,row) for row in range(1,NUM_ROWS)], \
                  [(-row,row) for row in range(1,NUM_ROWS)], \
                  [(-row,-row) for row in range(1,NUM_ROWS)], \
                  [(row,-row) for row in range(1,NUM_ROWS)])

    # Queens
    queens = apply_mask(board.queens)
    phi_queens = (np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)), np.zeros((NUM_ROWS, NUM_COLS)))
    idx_queen = idx_rook + idx_bishop

    piece_tuples = [(knights, phi_knights, idx_knight), \
                    (rooks, phi_rooks, idx_rook), \
                    (bishops, phi_bishops, idx_bishop), \
                    (queens, phi_queens, idx_queen)]

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            for pieces, phi_pieces, idx_piece in piece_tuples:
                for direction in idx_piece:
                    if pieces[WHITE][row,col]:
                        for idx in direction:
                            if get(black_pieces, (row+idx[0], col+idx[1])):
                                break
                            if add_to(phi_pieces[0], pieces[OTHER_WHITE], (row+idx[0], col+idx[1])):
                                break
                        for idx in direction:
                            if get(white_pieces, (row+idx[0], col+idx[1])):
                                break
                            if add_to(phi_pieces[1], pieces[OTHER_BLACK], (row+idx[0], col+idx[1])):
                                break
                    if pieces[BLACK][row,col]:
                        for idx in direction:
                            if get(black_pieces, (row+idx[0], col+idx[1])):
                                break
                            if add_to(phi_pieces[2], pieces[OTHER_WHITE], (row+idx[0], col+idx[1])):
                                break
                        for idx in direction:
                            if get(white_pieces, (row+idx[0], col+idx[1])):
                                break
                            if add_to(phi_pieces[3], pieces[OTHER_BLACK], (row+idx[0], col+idx[1])):
                                break

    # phi = np.array([*phi_knights, *phi_rooks, *phi_bishops, *phi_queens, free_spaces])
    phi = np.array([*phi_rooks, *phi_bishops, *phi_queens, white_pieces, black_pieces, free_spaces])
    return np.append(state, phi, axis=0)

def flip_color_square_idx(from_square, to_square):
    row = NUM_ROWS-1 - (from_square // NUM_ROWS)
    col = from_square % NUM_ROWS
    from_square = row*NUM_ROWS + col
    row = NUM_ROWS-1 - (to_square // NUM_ROWS)
    col = to_square % NUM_ROWS
    to_square = row*NUM_ROWS + col
    return from_square, to_square

def action_from_move(move, black=False):
    from_square = move.from_square
    to_square = move.to_square
    if black:
        from_square, to_square = flip_color_square_idx(from_square, to_square)
    a_from = np.zeros((NUM_SQUARES,))
    a_to = np.zeros((NUM_SQUARES,))
    a_from[from_square] = 1
    a_to[to_square] = 1
    return (a_from, a_to)

def move_from_action(from_square, to_square, black=False):
    if black:
        from_square, to_square = flip_color_square_idx(from_square, to_square)
    return chess.Move(from_square, to_square)

class Dataset:
    def __init__(self, filename, loop=False):
        self.filename = filename
        self.loop = loop
        self.idx_game = 0
        self.num_games = 0
        # self.idx_moves = []

    def load(self, generator, featurized=True, refresh=False):
        assert(type(generator) == str)

        if refresh:
            return self.pickle(generator, featurized=featurized)

        try:
            X_y = self.unpickle(generator, featurized=featurized)
        except:
            X_y = self.pickle(generator, featurized=featurized)
        return X_y

    def pickle(self, generator, featurized):
        X = []
        Y1 = []
        Y2 = []
        print("Pickling data:")
        for x, y in tqdm(getattr(self, generator)(featurized=True, loop=False)):
            X.append(x)
            if type(y) is list:
                Y1.append(y[0])
                Y2.append(y[1])
            else:
                Y1.append(y)

        X = np.concatenate(X)
        np.save(self.filename + "." + generator + "-" + str(featurized) + "-X.npy", X)

        Y1 = np.concatenate(Y1)
        np.save(self.filename + "." + generator + "-" + str(featurized) + "-y.npy", Y1)
        if not Y2:
            return X, Y1

        Y2 = np.concatenate(Y2)
        np.save(self.filename + "." + generator + "-" + str(featurized) + "-y2.npy", Y2)
        return X, Y1, Y2

    def unpickle(self, generator, featurized):
        X = np.load(self.filename + "." + generator + "-" + str(featurized) + "-X.npy")
        Y1 = np.load(self.filename + "." + generator + "-" + str(featurized) + "-y.npy")
        try:
            Y2 = np.load(self.filename + "." + generator + "-" + str(featurized) + "-y2.npy")
        except:
            return X, Y1
        return X, Y1, Y2

    def white_sarsa(self):
        return self.sarsa(black=False)

    def black_sarsa(self):
        return self.sarsa(black=True)

    def sarsa(self, black=False):
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

                    if black:
                        move = node.variations[0].move
                        board.push(move)
                        node = node.variations[0]
                        idx_move += 1

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

    def phi_action_sl(self, loop=False):
        return self.state_action_sl(loop=loop, featurized=True)

    def state_action_sl(self, loop=True, featurized=False):
        """
        Returns (state, action) tuple from white's perspective - flips black's perspective to match
        - state: np.array [12 pieces x 64 squares]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - square order: a1 b1 c1 ... h8
        - action: [np.array [1 x 64 squares], np.array [1 x 64 squares]] representing [from_board, to_board]
        """
        idx_batch = 0
        with open(self.filename) as pgn:
            S = []
            A_from = []
            A_to = []
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    if not loop:
                        break
                    print("\n******************************************")
                    print("********** LOOPING OVER DATASET **********")
                    print("******************************************\n")
                    pgn.seek(0)
                    continue

                num_moves = int(game.headers["PlyCount"])
                board = game.board()
                node = game.root()

                if num_moves <= 4:
                    continue

                # Make sure game was played all the way through
                last_node = game.root()
                while last_node.variations:
                    last_node = last_node.variations[0]
                if "forfeit" in last_node.comment:
                    continue

                black_turn = False
                while node.variations:
                    # Play white
                    s = state_from_board(board, featurized=featurized, black=black_turn)
                    move = node.variations[0].move
                    a_from, a_to = action_from_move(move, black=black_turn)
                    board.push(move)
                    black_turn = not black_turn

                    node = node.variations[0]

                    S.append(s)
                    A_from.append(a_from)
                    A_to.append(a_to)
                    idx_batch += 1

                    if idx_batch >= BATCH_SIZE and len(S) >= POOL_SIZE:
                        # Shuffle moves in game
                        idx = list(np.random.permutation(len(S)))
                        S_shuffle = [S[i] for i in idx]
                        A_from_shuffle = [A_from[i] for i in idx]
                        A_to_shuffle = [A_to[i] for i in idx]
                        S = S_shuffle[BATCH_SIZE:]
                        A_from = A_from_shuffle[BATCH_SIZE:]
                        A_to = A_to_shuffle[BATCH_SIZE:]
                        idx_batch = 0
                        yield np.array(S_shuffle[:BATCH_SIZE]), [np.array(A_from_shuffle[:BATCH_SIZE]), np.array(A_to_shuffle[:BATCH_SIZE])]

    def white_phi_action_sl(self, loop=False):
        return self.white_state_action_sl(loop=loop, featurized=True)

    def white_state_action_sl(self, loop=True, featurized=False):
        """
        Returns (state, action) tuple from white's perspective
        - state: np.array [12 pieces x 64 squares]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - square order: a1 b1 c1 ... h8
        - action: [np.array [1 x 64 squares], np.array [1 x 64 squares]] representing [from_board, to_board]
        """
        BATCH_SIZE = 32
        idx_batch = 0
        with open(self.filename) as pgn:
            S = []
            A_from = []
            A_to = []
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    if not loop:
                        break
                    print("\n******************************************")
                    print("********** LOOPING OVER DATASET **********")
                    print("******************************************\n")
                    pgn.seek(0)
                    continue

                num_moves = int(game.headers["PlyCount"])
                board = game.board()
                node = game.root()

                if num_moves <= 4:
                    continue

                # Make sure game was played all the way through
                last_node = game.root()
                while last_node.variations:
                    last_node = last_node.variations[0]
                if "forfeit" in last_node.comment:
                    continue

                while node.variations:
                    s = state_from_board(board, featurized=featurized)
                    move = node.variations[0].move
                    a_from, a_to = action_from_move(move)

                    # Play white
                    board.push(move)

                    # Play black
                    node = node.variations[0]
                    if node.variations:
                        move = node.variations[0].move
                        board.push(move)

                        if node.variations:
                            node = node.variations[0]

                    S.append(s)
                    A_from.append(a_from)
                    A_to.append(a_to)
                    idx_batch += 1

                    if idx_batch == BATCH_SIZE:
                        # Shuffle moves in game
                        idx = list(np.random.permutation(len(S)))
                        S_shuffle = [S[i] for i in idx]
                        A_from_shuffle = [A_from[i] for i in idx]
                        A_to_shuffle = [A_to[i] for i in idx]
                        S = []
                        A_from = []
                        A_to = []
                        idx_batch = 0
                        yield np.array(S_shuffle), [np.array(A_from_shuffle), np.array(A_to_shuffle)]

    def black_phi_action_sl(self, loop=False):
        return self.white_state_action_sl(loop=loop, featurized=True)

    def black_state_action_sl(self, loop=True, featurized=False):
        """
        Returns (state, action) tuple from black's perspective
        - state: np.array [12 pieces x 64 squares]
            - piece order:  wp wn wb wr wq wk bp bn bb br bq bk
            - square order: a1 b1 c1 ... h8
        - action: [np.array [1 x 64 squares], np.array [1 x 64 squares]] representing [from_board, to_board]
        """
        BATCH_SIZE = 32
        idx_batch = 0
        with open(self.filename) as pgn:
            S = []
            A_from = []
            A_to = []
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    if not loop:
                        break
                    print("\n******************************************")
                    print("********** LOOPING OVER DATASET **********")
                    print("******************************************\n")
                    pgn.seek(0)
                    continue

                num_moves = int(game.headers["PlyCount"])
                board = game.board()
                node = game.root()

                if num_moves <= 5:
                    continue

                # Make sure game was played all the way through
                last_node = game.root()
                while last_node.variations:
                    last_node = last_node.variations[0]
                if "forfeit" in last_node.comment:
                    continue

                # Play white
                move = node.variations[0].move
                board.push(move)
                node = node.variations[0]

                while node.variations:
                    s = state_from_board(board, featurized=featurized)
                    move = node.variations[0].move
                    a_from, a_to = action_from_move(move)

                    # Play black
                    board.push(move)

                    # Play white
                    node = node.variations[0]
                    if node.variations:
                        move = node.variations[0].move
                        board.push(move)

                        if node.variations:
                            node = node.variations[0]

                    S.append(s)
                    A_from.append(a_from)
                    A_to.append(a_to)
                    idx_batch += 1

                    if idx_batch == BATCH_SIZE:
                        # Shuffle moves in game
                        idx = list(np.random.permutation(len(S)))
                        S_shuffle = [S[i] for i in idx]
                        A_from_shuffle = [A_from[i] for i in idx]
                        A_to_shuffle = [A_to[i] for i in idx]
                        S = []
                        A_from = []
                        A_to = []
                        idx_batch = 0
                        yield np.array(S_shuffle), [np.array(A_from_shuffle), np.array(A_to_shuffle)]

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
            S = []
            A = []
            for line in epd:
                # Setup board
                board = chess.Board()
                board.set_epd(line)

                # Parse test id
                tokens = line.split(";")
                matches = re.match('id "(.*)"$', tokens[1].strip())
                if matches is not None:
                    id_test = matches.group(1)

                # Parse possible moves and keep track of best one
                scores = {}
                max_score = 0
                max_move = None
                matches = re.match('c0 "(.*)"$', tokens[2].strip())
                if matches is not None:
                    tokens = matches.group(1).split(",")
                    for token in tokens:
                        pair = token.strip().split("=")
                        move = board.parse_san(pair[0])
                        uci = board.uci(move)
                        score = int(pair[1])
                        scores[uci] = score
                        if score > max_score:
                            max_score = score
                            max_move = move

                # Convert to state and action representation
                s = state_from_board(board)
                (piece_type, from_square, to_square) = action_from_move(max_move)
                a = np.zeros((NUM_PIECES,))
                a[piece_type-1] = 1

                S.append(s)
                A.append(a)

            S = np.array(S)
            A = np.array(A)
            return S, A
