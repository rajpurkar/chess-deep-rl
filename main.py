from data import Dataset

d = Dataset("data/small.pgn")
# d = Dataset("data/medium.pgn")
# d = Dataset("data/CCRL-4040.[671444].pgn")
# d = Dataset("data/ficsgamesdb_1999-2015_standard2000_nomovetimes.pgn")
for state, action, result in d.random_states():
    print(state.shape, action, result)
