from engines.PolicyEngine import PolicyEngine
import chess
import data

white_model_hdf5 = ""
black_model_hdf5 = ""
NUM_GAMES_PER_BATCH = 100

def get_result(board):
    result = board.result()
    if result is not "*":
        white_score = result.split("-")
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
    white_actions = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    black_states = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    black_actions = [[] for _ in range(NUM_GAMES_PER_BATCH)]
    white_scores = [None] * NUM_GAMES_PER_BATCH
    black_scores = [None] * NUM_GAMES_PER_BATCH

    while True:
        # Filter out finished games
        boards_next = {}
        for idx, board in boards.items():
            result = get_result(board)
            if result is not None:
                white_scores[idx] = [result] * num_white_moves[idx]
                black_scores[idx] = [-result] * num_black_moves[idx]
                continue
            boards_next[idx] = board
        boards = boards_next
        if not boards:
            break

        # Play white
        X, y, moves = white_engine.search(boards)
        for i, (idx, board) in enumerate(boards.items()):
            board.push(moves[i])
            white_states[idx].append(X[i,:].reshape(1, X.shape[1]))
            white_actions[idx].append(y[i,:].reshape(1, y.shape[1]))
            num_white_moves[idx] += 1

        # Filter out finished games
        boards_next = {}
        for idx, board in boards.items():
            result = get_result(board)
            if result is not None:
                white_scores[idx] = [result] * num_white_moves[idx]
                black_scores[idx] = [-result] * num_black_moves[idx]
                continue
            boards_next[idx] = board
        boards = boards_next
        if not boards:
            break

        # Play black
        X, y, moves = black_engine.search(boards)
        for i, (idx, board) in enumerate(boards.items()):
            board.push(moves[i])
            black_states[idx].append(X[i,:].reshape(1, X.shape[1]))
            black_actions[idx].append(y[i,:].reshape(1, y.shape[1]))
            num_black_moves[idx] += 1

    # Flatten lists
    white_states = [a for game in white_states for a in game]
    black_states = [a for game in black_states for a in game]
    white_actions = [a for game in white_actions for a in game]
    black_actions = [a for game in black_actions for a in game]
    white_scores = [score for game in white_scores for score in game]
    black_scores = [score for game in black_scores for score in game]

    # Shuffle lists
    idx = list(np.random.permutation(len(actions)))
    white_states = np.array([white_states[i] for i in idx])
    black_states = np.array([black_states[i] for i in idx])
    white_actions = np.array([white_actions[i] for i in idx])
    black_actions = np.array([black_actions[i] for i in idx])
    white_scores = np.array([white_scores[i] for i in idx])
    black_scores = np.array([black_scores[i] for i in idx])

    return (white_states, white_actions, white_scores), (black_states, black_actions, black_scores)

def train(engine, X, y, r):
    # TODO
    pass

if __name__ == "__main__":
    white_engine = PolicyEngine(white_model_hdf5)
    black_engine = PolicyEngine(black_model_hdf5)

    while True:
        white_sar, black_sar = play(white_engine, black_engine)
        train(white_engine, white_sar[0], white_sar[1], white_sar[2])
        train(black_engine, black_sar[0], black_sar[1], black_sar[2])
