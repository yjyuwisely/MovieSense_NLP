from flask import Flask, render_template, request
from nlp.sentiment_analysis_bert import predict_sentiment
from nlp.summarization_bart import generate_summary
from nlp.translation_mbart import translate_to_french
from nlp.text_generation_gpt2 import generate_text

# Specify template and static folders
app = Flask(__name__, template_folder='../templates', static_folder='../static')
# app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    summary = None
    translation = None
    generation = None
    user_input = ""   # Set default value to empty string instead of None

    if request.method == 'POST':
        text = request.form['text']
        user_input = text
        
        if "Analysis" in request.form:
            prediction = predict_sentiment(text)
            summary = generate_summary(text)
            translation = translate_to_french(text)
        elif "Generator" in request.form:
            generation = generate_text(text)  # Assuming you have a function called generate_text
    # Ensure summary is not None before calling replace
    if summary:
        # Remove any existing "Summarize: " or "summarize: " prefix, regardless of case
        summary = summary.replace("Summarize: ", "").replace("summarize: ", "")
    return render_template('index.html', prediction=prediction, summary=summary, user_input=generation if generation else user_input, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)