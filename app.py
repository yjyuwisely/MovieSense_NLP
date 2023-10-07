from flask import Flask, render_template, request
from main_sentiment_analysis import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    show_prediction = False
    text = ""
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_sentiment(text)
        show_prediction = True
    return render_template('index.html', prediction=prediction, user_input=text, show_prediction=show_prediction)

""" def index():
    user_input = ''  # default to empty string
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        prediction = predict_sentiment(user_input)
    return render_template('index.html', prediction=prediction, user_input=user_input) """

""" @app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_sentiment(text)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None) """

if __name__ == '__main__':
    app.run(debug=True)