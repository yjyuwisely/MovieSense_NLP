from flask import Flask, render_template, request
from main_sentiment_analysis import predict_sentiment, generate_summary, translate_to_french

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    summary = None
    translation = None
    user_input = ""   # Set default value to empty string instead of None

    if request.method == 'POST':
        text = request.form['text']
        user_input = text
        
        if "Analysis" in request.form:
            prediction = predict_sentiment(text)
            summary = generate_summary(text)
            translation = translate_to_french(text)

    return render_template('index.html', prediction=prediction, summary=summary, user_input=user_input, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)