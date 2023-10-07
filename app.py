from flask import Flask, render_template, request
from main_sentiment_analysis import predict_sentiment, generate_summary

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    summary = None
    user_input = None

    if request.method == 'POST':
        text = request.form['text']
        user_input = text
        if "Predict" in request.form:
            prediction = predict_sentiment(text)
        elif "Summary" in request.form:
            summary = generate_summary(text)

    return render_template('index.html', prediction=prediction, summary=summary, user_input=user_input)

""" @app.route('/', methods=['GET', 'POST'])
def index():
    show_prediction = False
    text = ""
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_sentiment(text)
        show_prediction = True
    return render_template('index.html', prediction=prediction, user_input=text, show_prediction=show_prediction) """

if __name__ == '__main__':
    app.run(debug=True)