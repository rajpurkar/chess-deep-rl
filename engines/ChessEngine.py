#!/usr/bin/env python3
import chess
import sys
import os
import traceback
from concurrent import futures

class ChessEngine:
    def __init__(self):
        self.engine_name = "Dummy Chess Engine"
        self.author = "P. Rajpurkar & T. Migimatsu"
        self.board = chess.Board()
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=2)

    ###################
    # Virtual methods #
    ###################
    def search(self):
        """
        Find two best moves
        """
        moves_generator = self.board.generate_legal_moves()
        self.moves = []
        for i, move in enumerate(moves_generator):
            if i == 2:
                break
            self.moves.append(move)

    def ponder(self):
        """
        Consider moves during opponent's turn
        """
        pass

    def setoption(self, options):
        """
        Setup engine options
        """
        pass

    def stop(self):
        """
        Stop searching/pondering and submit moves
        """
        pass

    def __str__(self):
        return str(self.board)

    ################
    # Base methods #
    ################
    def uci(self):
        print("id name", self.engine_name)
        print("id author", self.author)
        print("uciok")

    def isready(self):
        print("readyok")

    def ucinewgame(self):
        self.board.reset_board()

    def position(self, input_tokens):
        # Play through last two moves on internal board, regardless of given position
        try:
            if input_tokens[0] == "startpos" and input_tokens[1] == "moves":
                moves = input_tokens[2:]
            elif input_tokens[0] == "fen" and "moves" in input_tokens:
                # TODO: fen broken
                moves = input_tokens[input_tokens.index("moves")+1:]
            else:
                return
        except:
            return

        if len(moves) == 1:
            self.board.push_uci(moves[0])
        else:
            self.board.push_uci(moves[-2])
            self.board.push_uci(moves[-1])

    def go(self, input_tokens=None):
        # Parse search options
        self.search_options = {}
        if input_tokens[0] == "searchmoves":
            # Restrict search to this moves only
            self.search_options["searchmoves"] = input_tokens[1:]
        elif input_tokens[0] == "ponder":
            # Start searching in pondering move. Do not exit the search in ponder
            # mode, even if it’s mate! This means that the last move sent in in the
            # position string is the ponder move.The engine can do what it wants to
            # do, but after a “ponderhit” command it should execute the suggested
            # move to ponder on.
            ponder()
        elif input_tokens[0] in ["movetime", "infinite", "wtime", "btime", \
                                 "winc", "binc", "movestogo", "depth", "nodes", "mate"]:
            # wtime: White has x msec left on the clock
            # btime: Black has x msec left on the clock
            # winc: White increment per move in mseconds if x > 0
            # binc: Black increment per move in mseconds if x > 0
            # movestogo: Here are x moves to the next time control, this will
            #     only be sent if x > 0, if you don’t get this and get the wtime and
            #     btime it’s sudden death
            # depth: Search x plies only
            # nodes: Search x nodes only
            # mate: Search for a mate in x moves
            # movetime: Search exactly x mseconds
            self.search_options[input_tokens[0]] = int(input_tokens[1])
        elif input_tokens[0] == "infinite":
            # Search until the “stop” command. Do not exit the search without being
            # told so in this mode!
            self.search_options["infinite"] = True

        self.search()
        self.send_move()

    def send_move(self):
        # Send two best moves
        if not self.moves:
            print("bestmove (none)")
        elif len(self.moves) == 1:
            print("bestmove", self.moves[0])
        else:
            print("bestmove", self.moves[0], "ponder", self.moves[1])

    def run(self):
        while True:
            try:
                input_msg = input()
                self.thread_pool.submit(self.handle_msg, input_msg)
            except:
                self.exit()
                return

    def handle_msg(self, input_msg):
        try:
            if input_msg.startswith("setoption"):
                self.setoption(input_msg.split(' ')[1:])
            elif input_msg.startswith("position"):
                self.position(input_msg.split(' ')[1:])
            elif input_msg == "go":
                self.go()
            elif input_msg.startswith("go"):
                self.go(input_msg.split(' ')[1:])
            elif input_msg == "stop":
                self.stop()
                self.send_move()
            elif input_msg == "uci":
                self.uci()
            elif input_msg == "ucinewgame":
                self.ucinewgame()
            elif input_msg == "isready":
                self.isready()
            elif input_msg == "print":
                print(self)
            elif input_msg == "quit":
                self.exit()
        except:
            print("\n*** Exception from ChessEngine.thread_pool *** {\n", file=sys.stderr)
            print(traceback.print_exc())
            print("\n}\n", file=sys.stderr)
            os._exit(1)

        sys.stdout.flush()

    def exit(self):
        self.stop()
        self.thread_pool.shutdown()

if __name__ == "__main__":
    engine = ChessEngine()
    engine.run()
