<h1>MovieSense: Sentiment Analysis, Summarisation, and Translation</h1>
<p align="justify">Welcome to <b>MovieSense</b>, an advanced Natural Language Processing project that provides sentiment analysis, summarization, and translation services for movie reviews.<p align="justify">
<br>
  
<h2>Overview</h2>
<p align="justify">
Designed with an AI and NLP focus, this project aims to assist in understanding the sentiments behind movie reviews, provide a concise summary for lengthy reviews, and translate them from English to French.</p>
<br>

## Features
1. <b>Sentiment Analysis:</b> This feature leverages the Naive Bayes Classifier to discern whether reviews are positive or negative, utilising data from Rotten Tomatoes.<br>
2. <b>Summarisation:</b> The application employs the BART model from Facebook to generate concise summaries for extended reviews.<br>
3. <b>Translation:</b> Reviews can be translated into French to cater to a diverse audience.<br>
<br>
  
## Technologies Used

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

<h2>Installation</h2>
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

## Optional: Setting up a virtual environment

While the project runs without a virtual environment, it's recommended to use one for isolation:

1. Install **virtualenv** if you haven't:  
   `pip install virtualenv`
2. Clone the repository:  
   `git clone https://github.com/yjyuwisely/Sentiment_Analysis.git`
3. Create a virtual environment:  
   `virtualenv .env`
4. Activate the virtual environment:
   - On macOS and Linux:  
     `source .env/bin/activate`
   - On Windows:  
     `.\.env\Scripts\activate`
5. When you're done working on the project, you can deactivate the virtual environment:  
   `deactivate`
<br>

## Usage
1. <b>Sentiment Analysis:</b> Reviews are categorised into 'positive' or 'negative' using the Naive Bayes Classifier and represented with emojis.<br>
2. <b>Summarisation:</b> Input the desired text to receive a concise summary.<br>
3. <b>Translation:</b> Translate reviews into French.<br>
<br>

## Future Scope
- Extend translation support to other languages.
- Integration with online platforms or databases for automated review analysis.
- Enhanced prediction capabilities using Deep Learning.
- User accounts to save and manage past reviews.
<br>

## References
1. [Natural Language Processing with Transformers, Revised Edition](https://www.amazon.com/-/ko/dp/1098136799/ref=sr_1_1?qid=1696744546&refinements=p_27%3ALewis+Tunstall&s=books&sr=1-1&text=Lewis+Tunstall)<br>
2. [Artificial Intelligence with Python: Your complete guide to building intelligent apps using Python 3.x, 2nd Edition](https://www.amazon.com/-/ko/dp/183921953X/ref=sr_1_1?crid=SVEK8NYGJHHH&keywords=Artificial+Intelligence+with+Python%3A+Your+complete+guide+to+building+intelligent+apps+using+Python&qid=1696744519&s=books&sprefix=%2Cstripbooks-intl-ship%2C334&sr=1-1)<br>
3. [Python Deep Learning Projects: 9 projects demystifying neural network and deep learning models for building intelligent systems](https://www.amazon.com/dp/B07FNY2BZR?ref_=ast_author_dp)<br>
<br>

<h2>Page Screenshot</h2>
Positive Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5udNA%2Fbtsxp3doUAk%2FU38yky0rcDo3KPc6yCGtLk%2Fimg.png">
Negative Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCpxxG%2FbtsxvIFFLyI%2FRthmnpzRTiaanXaDxgCEjK%2Fimg.png">
