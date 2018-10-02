"""
app.py

(C) 2017 by Abhishek Babuji <abhishekb2209@gmail.com>

Deploys application to the web
"""

import json
import pickle
import pandas as pd
from flask import Flask, render_template, request
from vectorspace import VectorSpace

with open('/Users/abhishekbabuji/Desktop/spooky_author_model.pkl', 'rb') as fid:
    pkl_model_loaded = pickle.load(fid)

app = Flask(__name__, static_url_path='')


@app.route('/')
def input_form():
    """
    Renders the index.html page

    :return: render_template('/index.html')
    """
    return render_template('/index.html')


@app.route('/api', methods=['POST'])
def predict():
    """
    Parses in the input from the form

    :return: render_template("/index.html", results=parse(pd.Series([text_input])))
    """

    text_input = request.form['passage_input']
    return render_template("/index.html", results=parse(pd.Series([text_input])))


def parse(input_passage):
    """
    Fits the best classifier and best weighting factor to the input text
    and makes a prediction using the sklearn pickled model

    :return: render_template("/index.html", results=parse(pd.Series([text_input])))
    """
    reduction_type = 'lemmatize'
    trans_input_passage = VectorSpace(input_passage, reduction=reduction_type).apply_reduction()

    if pkl_model_loaded.predict(trans_input_passage)[0] == "EAP":

        return json.dumps("Edgar Allan Poe")

    elif pkl_model_loaded.predict(trans_input_passage)[0] == "HPL":
        return json.dumps("HP Lovecraft")

    else:
        return json.dumps("Mary Shelley")


if __name__ == '__main__':
    app.run(port=9000, debug=True)
