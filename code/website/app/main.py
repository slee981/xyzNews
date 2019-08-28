from keras.models import load_model
from keras.utils import to_categorical
import flask
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

from inputForm import InputForm

app = flask.Flask(__name__)
app.static_folder = os.path.join(os.getcwd(), 'static')
app.static_url_path = os.path.join(os.getcwd(), 'static')

MODEL_PATH = "./models/xyzNews-classifier.h5"
EMBEDDING_PATH = "word_embedding/glove.840B.300d.txt"
EMBEDDINGS_INDEX = {}
MODEL = None
graph = tf.get_default_graph()


def get_embeddings():
    print("\nReading in word embeddings. This may take a couple minutes.")
    with open(EMBEDDING_PATH, encoding="utf8") as embed:
        try:
            for line in tqdm(embed):
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                EMBEDDINGS_INDEX[word] = coefs
        except KeyboardInterrupt:
            print("\n\nInterrupted. Closing the file and stopping... ")


def prepare_article(text, article_length=500):
    empty_emb = np.zeros(300)  # each word is represented by a length 300 vector
    text = text.split()[:article_length]  # each article is length 500

    # look for word embedding, return zero array otherwise.
    embeds = [np.asarray(EMBEDDINGS_INDEX.get(x, empty_emb)) for x in text]
    embeds += [empty_emb] * (article_length - len(embeds))
    return np.array(embeds).reshape(1, 500, 300)


def load_data_and_model():
    global MODEL
    get_embeddings()
    print("Loading model...")
    MODEL = load_model(MODEL_PATH)
    MODEL._make_predict_function()


@app.route("/", methods=["GET", "POST"])
def home():
    # init form and predictions
    form = InputForm(flask.request.form)
    prediction = None

    # ensure an article was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.form.get("article"):
            txt = flask.request.form.get("article")
            txt = prepare_article(txt)

            global graph
            with graph.as_default():
                pred = MODEL.predict(txt, batch_size=1)

            # model is trained to represent:
            # 0 - PBS
            # 1 - Vox
            # 2 - Fox
            preds = pred[0].tolist()
            values = [round(i, 3) for i in preds]
            prediction = {"pbs": preds[0], "vox": preds[1], "fox": preds[2]}
            success=True

            return flask.render_template("index.html", form=form, success=success, prediction=prediction, values=values)

    # return the data dictionary as a JSON response
    # return flask.jsonify(data)
    return flask.render_template("index.html", form=form)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned
    data = {"success": False}

    # ensure an article was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.form.get("article"):
            txt = flask.request.form.get("article")
            txt = prepare_article(txt)

            global graph
            with graph.as_default():
                pred = MODEL.predict(txt, batch_size=1)

            # model is trained to represent:
            # 0 - PBS
            # 1 - Vox
            # 2 - Fox
            values = pred[0].tolist()
            labels = ['PBS News', 'Vox News', 'Fox News']
            prediction = {"pbs": values[0], "vox": values[1], "fox": values[2]}
            data["prediction"] = prediction
            data["success"] = True

    # return the data dictionary as a JSON response
    # return flask.jsonify(data)
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(
        (
            "Loading Keras model and Flask starting server... "
            "please wait until server has fully started"
        )
    )
    load_data_and_model()
    app.run()