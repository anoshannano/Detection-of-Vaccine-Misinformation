from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact/')
def contact():
    return render_template('contact.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    Naive, Test_Y, score, Tfidf_vect = pickle.load(open('modal.sav', 'rb'))

    if request.method == 'POST':
        message = request.form['message']
        vect = Tfidf_vect.transform([message])
        my_prediction = Naive.predict(vect)
        scoreModel = "{:.2f}".format(score * 100)
    return render_template('result.html', prediction=my_prediction, score=scoreModel)


if __name__ == '__main__':
    app.run(debug=True)