import numpy as np
import data
import pickle
from tqdm import tqdm
import time
from os.path import basename
import argparse

dim_S = data.NUM_SQUARES * data.NUM_PIECES * data.NUM_COLORS
dim_A = dim_S * dim_S


def sarsa_lambda(data, color_to_play, LAMBDA=0.99, GAMMA=0.99, ALPHA=1/np.sqrt(dim_S)):
    Q = {}
    N = {}
    gen = data.black_sarsa if color_to_play == 'black' else data.white_sarsa
    i = 0
    for s, a, r, s_prime, a_prime, new_game in tqdm(gen()):
        if new_game:
            N = {}
        if s not in N:
            N[s] = {}
            if s not in Q:
                Q[s] = {}
        if a not in N[s]:
            N[s][a] = 0
            if a not in Q[s]:
                Q[s][a] = 0
        Q_prime = 0
        if s_prime in Q and a_prime in Q[s_prime]:
            Q_prime = Q[s_prime][a_prime]

        N[s][a] += 1
        delta = r + GAMMA * (Q_prime - Q[s][a])
        for s_hat in N:
            for a_hat in N[s_hat]:
                Q[s_hat][a_hat] += ALPHA * delta * N[s_hat][a_hat]
                N[s_hat][a_hat] *= GAMMA * LAMBDA

        if i % 500000 == 0:
            try:
                with open(picklefile, "wb") as f:
                    pickle.dump(Q, f)
            except:
                print("Failed to save pickle at iteration: %d" % i)
                pass
        i += 1
    return Q, N

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "color_to_play",
        help="Either black or white",
        choices=['black', 'white'])
    parser.add_argument("filename", help="file to learn Q from")

    args = parser.parse_args()
    datafile = args.filename
    filename = basename(args.filename)
    picklefile = "saved/" + args.color_to_play + \
        "_sarsa_Q_%s-%s.pickle" % (filename, time.time())
    try:
        d = data.Dataset(datafile)
        Q, N = sarsa_lambda(d, args.color_to_play)
    except Exception as e:
        print(e)

    try:
        with open(picklefile, "wb") as f:
            pickle.dump(Q, f)
    except:
        print("Failed to save pickle")
