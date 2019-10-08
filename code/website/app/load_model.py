from keras.models import load_model
from tqdm import tqdm
import numpy as np

MODEL_PATH = "./models/xyzNews-classifier.h5"
EMBEDDING_PATH = "word_embedding/glove.840B.300d.txt"


def get_embeddings():
    embeddings_index = {}
    print("\nReading in word embeddings. This may take a couple minutes.")
    with open(EMBEDDING_PATH, encoding="utf8") as embed:
        try:
            for line in tqdm(embed):
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
        except KeyboardInterrupt:
            print("\n\nInterrupted. Closing the file and stopping... ")
    return embeddings_index


def load_nn_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    model._make_predict_function()
    return model


EMBEDDING_INDEX = get_embeddings()
MODEL = load_nn_model()