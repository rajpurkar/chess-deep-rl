import os

filename = "small.pgn"
size_test = 500

name, ext = os.path.splitext(filename)
train_filename = name + "_train" + ext
test_filename = name + "_test" + ext

with open(filename) as f:
    with open(test_filename, "w") as f_test:
        idx_game = 0
        for line in f:
            if not line.strip():
                idx_game += 1
            f_test.write(line)
            if idx_game == size_test * 2:
                f_test.write("\n")
                break

    with open(train_filename, "w") as f_train:
        for line in f:
            f_train.write(line)
