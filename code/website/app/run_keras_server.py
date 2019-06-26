from keras.models import load_model
from keras.utils import to_categorical
import flask
import numpy as np
from tqdm import tqdm
import os

app = flask.Flask(__name__)

MODEL_PATH = "../models/xyzNews-classifier.h5"
EMBEDDING_PATH = "word_embedding/glove.840B.300d.txt"
EMBEDDINGS_INDEX = {}
MODEL = None


def get_embeddings():
    print("\nReading in word embeddings. This may take a minute.")
    with open(EMBEDDING_PATH, encoding="utf8") as embed:
        for line in tqdm(embed):
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            EMBEDDINGS_INDEX[word] = coefs


def prepare_article(text, article_length=500):
    empty_emb = np.zeros(300)  # each word is represented by a length 300 vector
    text = text.split()[:article_length]  # each article is length 500

    # look for word embedding, return zero array otherwise.
    embeds = [EMBEDDINGS_INDEX.get(x, empty_emb) for x in text]
    embeds += [empty_emb] * (article_length - len(embeds))
    return np.array(embeds)


def load_data_and_model():
    global MODEL
    get_embeddings()
    MODEL = load_model(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        if flask.request.form.get("article"):
            txt = flask.request.form.get("article")
            print(f"\n\nThe input text is: {txt}")

            txt = prepare_article(txt)
            pred = MODEL.predict(txt, batch_size=1)
            print(f"\n\nThe prediction is {pred}")

            data["predictions"] = pred
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(
        (
            "Loading Keras model and Flask starting server..."
            "please wait until server has fully started"
        )
    )
    load_data_and_model()
    app.run()
