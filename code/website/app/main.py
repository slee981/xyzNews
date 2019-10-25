#!/usr/bin/env python3

############################################################
# Imports
############################################################

from keras.utils import to_categorical
import flask
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

from inputForm import InputForm
from load_model import MODEL, EMBEDDING_INDEX

print('Model is not empty: {}'.format(MODEL == None))
print('Embeddings have {} entries.'.format(len(EMBEDDING_INDEX)))

############################################################
# Storage
############################################################

app = flask.Flask(__name__)
app.static_folder = os.path.join(os.getcwd(), 'static')
app.static_url_path = os.path.join(os.getcwd(), 'static')

graph = tf.get_default_graph()

############################################################
# Functions
############################################################


def prepare_article(text, article_length=500):
    global EMBEDDING_INDEX
    empty_emb = np.zeros(300)  # each word is represented by a length 300 vector
    text = text.split()[:article_length]  # each article is length 500

    # look for word embedding, return zero array otherwise.
    embeds = [np.asarray(EMBEDDING_INDEX.get(x, empty_emb)) for x in text]
    embeds += [empty_emb] * (article_length - len(embeds))
    return np.array(embeds).reshape(1, 500, 300)


@app.route("/", methods=["GET", "POST"])
def home():
    global MODEL
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

            return flask.render_template("index.html",\
                                         form=form, success=success, prediction=prediction, values=values)

    # return the home page
    return flask.render_template("index.html", form=form)


@app.route("/predict", methods=["POST"])
def predict():
    global MODEL
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

############################################################
# Main
############################################################

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(
        (
            "Loading Keras model and Flask starting server... "
            "please wait until server has fully started"
        )
    )
    app.run()
