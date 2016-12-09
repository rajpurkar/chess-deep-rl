import numpy as np
from data import Dataset
from tqdm import tqdm


def get_joint_accuracy(y_from, y_to, y_from_true, y_to_true):
    num_top = 3
    score = np.zeros((y_to.shape[0],))
    for i in tqdm(range(y_to.shape[0])):
        from_square_true = np.where(y_from_true[i] == 1)[0]
        to_square_true = np.where(y_to_true[i] == 1)[0]
        p = np.outer(y_from[i], y_to[i])
        p_shape = p.shape
        p = p.reshape((-1,))

        for j, idx in enumerate(np.argsort(p)[::-1]):
            if j >= num_top:
                break
            from_square, to_square = np.unravel_index(idx, p_shape)
            if from_square == from_square_true and to_square == to_square_true:
                score[i] = 1
                break
    print("Joint move accuracy: %f" % (score.sum() / score.shape[0]))

if __name__ == '__main__':
    d_test = Dataset('data/small_test.pgn')
    X_val, y_from_val, y_to_val = d_test.load(
        'white_state_action_sl',
        featurized=True,
        refresh=False)
    from keras.models import load_model
    model = load_model("./saved/policy/1481219504/94-4.61.hdf5")
    y_from, y_to = model.predict(X_val, verbose=1)
    get_joint_accuracy(y_from, y_to, y_from_val, y_to_val)
