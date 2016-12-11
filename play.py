import chess
import chess.uci
import os
import argparse

def main(path_engine_w, path_engine_b, move_time, num_games, verbose):
    # Load engines
    engine_w = chess.uci.popen_engine(path_engine_w)
    engine_b = chess.uci.popen_engine(path_engine_b)
    print("Loaded engines")

    # Initialize engines
    command_w = engine_w.uci(async_callback=True)
    command_b = engine_b.uci(async_callback=True)
    command_w.result()
    command_b.result()
    print("Initialized engines")

    total_num_moves = []
    for _ in range(num_games):
        # Create new game
        command_w = engine_w.ucinewgame(async_callback=True)
        command_b = engine_b.ucinewgame(async_callback=True)
        command_w.result()
        command_b.result()

        board = chess.Board()
        num_moves = 0
        while True:
            # Play white
            command_w = engine_w.position(board, async_callback=True)
            command_w.result()
            command_w = engine_w.go(movetime=move_time, async_callback=True)
            move = command_w.result().bestmove
            if move is None or board.result() != "*":
                if board.result() == "1/2-1/2":
                    print("\nDraw")
                    break
                print("\nBlack wins!")
                break
            board.push(move)

            # Play black
            command_b = engine_b.position(board, async_callback=True)
            command_b.result()
            command_b = engine_b.go(movetime=move_time, async_callback=True)
            move = command_b.result().bestmove
            if move is None or board.result() != "*":
                if board.result() == "1/2-1/2":
                    print("\nDraw")
                    break
                print("\nWhite wins!")
                break

            num_moves += 1
            board.push(move)

            if verbose:
                os.system("clear")
                print(board)

        # print(board)
        print("Number of moves: %d" % num_moves)
        total_num_moves.append(num_moves)

    print("Average number of moves: ", sum(total_num_moves)/len(total_num_moves))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("white_engine", default="engines/ChessEngine.py")
    parser.add_argument("black_engine", default="engines/ChessEngine.py")
    parser.add_argument("-t", type=int, default=20, help="Time to move in milliseconds. Default: 20ms")
    parser.add_argument("-n", type=int, default=1, help="Number of games to play. Default: 1")
    parser.add_argument("-v", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    main(args.white_engine, args.black_engine, args.t, args.n, args.v)
