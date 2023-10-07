<h1>MovieSense: Sentiment Analysis, Summarisation, and Translation</h1>
<p align="justify">Welcome to <b>MovieSense</b>, an advanced Natural Language Processing project that provides sentiment analysis, summarization, and translation services for movie reviews.<p align="justify">
<br>
  
<h2>Overview:</h2>
<p align="justify">
Designed with an AI and NLP focus, this project aims to assist in understanding the sentiments behind movie reviews, provide a concise summary for lengthy reviews, and translate them from English to French.</p>
<br>

<h2>Features:</h2>
<p align="justify">
1. <b>Sentiment Analysis:</b> This feature leverages the Naive Bayes Classifier to discern whether reviews are positive or negative, utilising data from Rotten Tomatoes.<br>
2. <b>Summarisation:</b> The application employs the BART model from Facebook to generate concise summaries for extended reviews.<br>
3. <b>Translation:</b> Reviews can be translated into French to cater to a diverse audience.<br></p>
<br>
  
## Technologies Used:

- **Languages:** Python
- **NLP Techniques & Models:**
  - Sentiment Analysis: Naive Bayes Classifier
  - Summarisation: BART (from Hugging Face's Transformers)
  - Translation: Helsinki-NLP's model (from Hugging Face's Transformers)
- **Frameworks/Libraries:**
  - NLTK
  - Transformers (Hugging Face)
  - Flask (for Backend development)
- **Frontend:** HTML, CSS, JavaScript.
<br>

<h2>Installation:</h2>
<p align="justify" style="display:none">
  
1. Clone the repository: <br>
  `git clone https://github.com/yjyuwisely/Sentiment_Analysis.git`<br>
2. Navigate to the project directory: <br>
  `cd Sentiment_Analysis`<br>
3. Install the required packages: <br>
  `pip install -r requirements.txt`<br>
4. Run the main script: <br>
  `python app.py`<br>
5. Run the main script: <br>
  `Open a web browser and go to http://127.0.0.1:5000/to use MovieSense.`<br>
</p>
<br>

<h2>Optional: Setting up a virtual environment</h2>

While the project runs without a virtual environment, it's recommended to use one for isolation:
1. stall <b>virtualenv</b> if you haven't: <br>
  `git clone https://github.com/yjyuwisely/Sentiment_Analysis.git`<br>
2. Create a virtual environment: <br>
  `virtualenv .env`<br>
3. Activate the virtual environment: <br>
  - On macOS and Linux:<br>
  `source .env/bin/activate`<br>
  - On Windows:<br>
  `.\.env\Scripts\activate`<br>
4. When you're done working on the project, you can deactivate the virtual environment: <br>
  `deactivate`<br>
<br>

<h2>Usage:</h2>
1. <b>Sentiment Analysis:</b> Reviews are categorised into 'positive' or 'negative' using the Naive Bayes Classifier and represented with emojis.<br>
2. <b>Summarisation:</b> Input the desired text to receive a concise summary.<br>
3. <b>Translation:</b> Translate reviews into French.<br>
<br>

<h2>Future Scope:</h2>
<li>Extend translation support to other languages.</li>
<li>Integration with online platforms or databases for automated review analysis.</li>
<li>Enhanced prediction capabilities using Deep Learning.</li>
<li>User accounts to save and manage past reviews.</li>
<br>

<h2>Page Screenshot</h2>
Positive Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5udNA%2Fbtsxp3doUAk%2FU38yky0rcDo3KPc6yCGtLk%2Fimg.png">
Negative Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCpxxG%2FbtsxvIFFLyI%2FRthmnpzRTiaanXaDxgCEjK%2Fimg.png">
