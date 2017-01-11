import os
from tqdm import tqdm
import argparse

size_test = 200

def split(filename):
    name, ext = os.path.splitext(filename)
    train_filename = name + "_train" + ext
    test_filename = name + "_test" + ext

    with open(filename) as f:
        with open(train_filename, "w") as f_train:
            with open(test_filename, "w") as f_test:
                idx_game = 0
                for line in tqdm(f):
                    if not line.strip():
                        idx_game += 1

                        if idx_game % 2 == 0:
                            idx_game_train = 0
                            for line in f:
                                if not line.strip():
                                    idx_game_train += 1
                                f_train.write(line)
                                if idx_game_train == 1000:
                                    break

                    f_test.write(line)
                    if idx_game == size_test * 2:
                        f_test.write("\n")
                        break

            for line in f:
                f_train.write(line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file to split")
    args = parser.parse_args()
    split(args.filename)
