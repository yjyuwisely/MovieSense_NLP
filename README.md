<h1>MovieSense: Sentiment Analysis, Text Generation, Summarization, and Translation</h1>
<p align="justify">Welcome to <b>MovieSense</b>, an advanced Natural Language Processing project that provides sentiment analysis, text generation, summarization, and translation services for movie reviews.<p align="justify">
<br>
  
## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
   - [Optional: Setting Up a Virtual Environment](#optional-setting-up-a-virtual-environment)
5. [Usage](#usage)
6. [Future Scope](#future-scope)
7. [References and Further Readings](#references-and-further-readings)
8. [Page Screenshot](#page-screenshot)
<br>

## Overview
<p align="justify">
Designed with an AI and NLP focus, this project aims to assist in understanding the sentiments behind movie reviews, provide a concise summary for lengthy reviews, generate contextually relevant reviews based on specific prompts, and translate them from English to French.</p>
<br>

## Features
1. <b>Sentiment Analysis:</b> Utilizes a pre-trained BERT model (`distilbert-base-uncased-finetuned-sst-2-english`) 
to classify movie reviews as positive or negative, based on data from Rotten Tomatoes.<br>
2. <b>Text Generation:</b> Implements a pre-trained GPT-2 model for generating movie reviews based on user prompts. 
To improve the contextual relevance and specificity of the generated text, 
a Retrieval-Augmented Generation (RAG) approach is integrated. 
This enhancement allows the model to retrieve relevant information from a knowledge base, 
such as a database of existing movie reviews, before generating new content.
3. <b>Summarization:</b> Generates concise summaries of extended reviews using Facebook's BART model.<br>
4. <b>Translation:</b> Supports multilingual audiences by translating reviews into French with the mBART model.<br>
<br>
  
## Technologies Used

- **Languages:** Python
- **NLP Techniques & Models:**
  - **Sentiment Analysis:** BERT (`distilbert-base-uncased-finetuned-sst-2-english`, from Hugging Face's Transformers)
  - **Text Generation:** GPT-2 (from Hugging Face's Transformers), currently using the base pre-trained model to generate movie reviews based on specific prompts. 
  - **Summarization:** BART (from Hugging Face's Transformers)
  - **Translation:** mBART model (from Hugging Face's Transformers)
- **Frameworks/Libraries:**
  - NLTK
  - Transformers (Hugging Face)
  - Flask (for Backend development)
- **Frontend:** HTML, CSS, JavaScript.
<br>

## Installation
  
1. Clone the repository: <br>
  `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`<br>
2. Navigate to the project directory: <br>
  `cd MovieSense_NLP/src`<br>
3. Install the required packages: <br>
  `pip install -r ../requirements.txt`<br>
4. Run the main script: <br>
  `python app.py`<br>
5. Run the main script: <br>
  `Open a web browser and go to http://127.0.0.1:5000/to use MovieSense.`<br>
</p>
<br>

### Optional: Setting up a virtual environment

While the project runs without a virtual environment, it's recommended to use one for isolation:

1. Install `virtualenv` if you haven't: 
   `pip install virtualenv`
2. Clone the repository:
   `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`
3. Navigate to the project directory:
   `cd MovieSense_NLP`
4. Create a virtual environment:
   `virtualenv .env`
5. Activate the virtual environment:
   - On macOS and Linux:  
     `source .env/bin/activate`
   - On Windows:  
     `.\.env\Scripts\activate`

6. Follow steps 2 to 5 in the [Installation section](#installation) to navigate to `src`, install the required packages, and run the application.

7. When you're done working on the project, you can deactivate the virtual environment:  
   `deactivate`
<br>


## Usage
1. <b>Sentiment Analysis:</b> Reviews are categorised into 'positive' or 'negative' using the DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) and represented with emojis.<br>
2. <b>Text Generation:</b> Enter a prompt (e.g., "Write a positive review about [Movie Name]") to generate a contextually relevant movie review. 
The RAG model retrieves relevant information from a database of existing movie reviews to ensure that the generated text is contextually accurate and aligned with the desired sentiment.
3. <b>Summarization:</b> Input the desired text to receive a concise summary.<br>
4. <b>Translation:</b> Translate reviews into French.<br>
<br>

## Future Scope
- Extend translation support to other languages.
- Integration with online platforms or databases for automated review analysis.
- Enhanced prediction capabilities using Deep Learning.
- User accounts to save and manage past reviews.
<br>

## References and Further Readings
1. Artasanchez, A., & Joshi, P. (2019). *Artificial Intelligence with Python: Your complete guide to building intelligent apps using Python 3.x* (2nd ed.). Packt Publishing.<br>
[Available on Amazon](https://www.amazon.com/Artificial-Intelligence-Python-complete-intelligent/dp/183921953X)<br>
2. Lamons, M., Kumar, R., & Nagaraja, A. (2018) *Python Deep Learning Projects: 9 projects demystifying neural network and deep learning models for building intelligent systems* (1st ed.). Packt Publishing.<br>
[Available on Amazon](https://www.amazon.com/Python-Deep-Learning-Projects-demystifying/dp/1788997093)<br>
3. Seo, J. (2024). *Developing AI services based on LLM with LangChain*. Gilbut.<br>
[Available on GitHub](https://github.com/gilbutITbook/080413)<br>
4. Tunstall, L., von Werra, L., & Wolf, T. (2022) *Natural Language Processing with Transformers: Building Language Applications with Hugging Face* (1st ed.). O'Reilly Media.<br>
[Available on Amazon](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)<br>
<br>

## Page Screenshot
Positive Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5udNA%2Fbtsxp3doUAk%2FU38yky0rcDo3KPc6yCGtLk%2Fimg.png">
Negative Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCpxxG%2FbtsxvIFFLyI%2FRthmnpzRTiaanXaDxgCEjK%2Fimg.png">