from keras.models import load_model
import numpy as np

if __name__ == '__main__':
    model = load_model("./saved/1480896779-06-0.40.hdf5")
    with open("log","w") as f:
        predictions = model.predict(np.zeros((2, 12, 8, 8))).flatten()
        f.write(str(predictions))
