import chess
import chess.uci

# Load engines
engine_w = chess.uci.popen_engine("engines/ChessEngine.py")
engine_b = chess.uci.popen_engine("engines/stockfish")

# Initialize engines
command_w = engine_w.uci(async_callback=True)
command_b = engine_b.uci(async_callback=True)
command_w.result()
command_b.result()

# Create new game
command_w = engine_w.ucinewgame(async_callback=True)
command_b = engine_b.ucinewgame(async_callback=True)
command_w.result()
command_b.result()
board = chess.Board()

while True:
    # Play white
    command_w = engine_w.position(board, async_callback=True)
    command_w.result()
    command_w = engine_w.go(movetime=20, async_callback=True)
    move = command_w.result().bestmove
    if move is None:
        print("\n\nBlack wins!")
        break
    board.push(move)
    print(move, end=" ", flush=True)

    # Play black
    command_b = engine_b.position(board, async_callback=True)
    command_b.result()
    command_b = engine_b.go(movetime=20, async_callback=True)
    move = command_b.result().bestmove
    if move is None:
        print("\n\nWhite wins!")
        break
    board.push(move)
    print(move, end=" ", flush=True)

print(board)
