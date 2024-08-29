<h1>MovieSense: Sentiment Analysis, Translation, Summarization, and Text Generation</h1>
<p align="justify">Welcome to <b>MovieSense</b>, an advanced Natural Language Processing (NLP) project 
designed to analyze and enhance movie reviews using state-of-the-art AI techniques. 
This project offers a comprehensive suite of services, including sentiment analysis, translation, summarization, 
and text generation, specifically tailored for movie review data.
<p align="justify">
<br>
  
## Table of Contents

Navigate through this `README.md` to learn more about the project, its features, setup, and future plans.

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Models and Methods Used](#models-and-methods-used)
5. [Installation](#installation)
   - [Optional: Setting Up a Virtual Environment](#optional-setting-up-a-virtual-environment)
6. [Usage](#usage)
7. [Future Scope](#future-scope)
8. [References and Further Readings](#references-and-further-readings)
9. [Page Screenshot](#page-screenshot)
<br>


## Overview
<p align="justify"> <b>MovieSense</b> is an advanced NLP-powered application designed for in-depth analysis 
and enhancement of movie reviews using cutting-edge AI techniques. 
Leveraging state-of-the-art NLP models like BERT, mBART, BART, and GPT-2, 
<b>MovieSense</b> provides a robust toolset for sentiment analysis, translation, summarization, and text generation. 
Users can classify reviews as positive or negative, translate reviews from English to French, 
generate concise summaries, and create new, contextually relevant reviews based on specific prompts. <br><br>
Tailored for movie enthusiasts, critics, and NLP researchers, <b>MovieSense</b> aims to enrich the movie review experience 
and offers valuable insights through AI-driven methodologies. 
Future plans include expanding translation support, integrating with online platforms for automated analysis, 
and exploring more advanced models for enhanced performance.
</p>
<br>


## Features

1. **Sentiment Analysis**
   - Utilizes a pre-trained BERT model (`distilbert-base-uncased-finetuned-sst-2-english`) to classify movie reviews as positive or negative.
   - Based on data from Rotten Tomatoes.
   - Demonstrates the ability to work with transformer-based models for text classification tasks.

2. **Translation**
   - Supports multilingual audiences by translating reviews into French using the mBART model from Hugging Face Transformers.
   - Showcases the ability to handle multilingual NLP tasks using sequence-to-sequence transformer models.

3. **Summarization**
   - Generates concise summaries of extended reviews using Facebook's BART model.
   - Demonstrates the capability to create coherent summaries that capture the essence of the original content.

4. **Text Generation**
   - Implements a pre-trained GPT-2 model for generating movie reviews based on user prompts.
   - Uses a Retrieval-Augmented Generation (RAG) approach to improve contextual relevance and specificity.
   - Retrieves relevant information from a knowledge base, such as a database of existing movie reviews, before generating new content.
<br>  

## Technologies Used

- **Languages:** Python
- **NLP Techniques & Models:**
  - **Sentiment Analysis:** BERT (`distilbert-base-uncased-finetuned-sst-2-english`, from Hugging Face's Transformers)
  - **Translation:** mBART model (from Hugging Face's Transformers)
  - **Summarization:** BART (from Hugging Face's Transformers)
  - **Text Generation:** GPT-2 (from Hugging Face's Transformers), 
  currently using the base pre-trained model to generate movie reviews based on specific prompts. 
- **Frameworks/Libraries:**
  - **NLTK**: Used for text processing tasks and basic NLP utilities.
  - **Transformers (Hugging Face)**: Core library for NLP models such as BERT, GPT-2, BART, and mBART.
  - **Flask**: Backend development framework used to build and deploy web services for the project.
- **Frontend:** HTML, CSS, JavaScript for user interface development.
<br>


## Models and Methods Used

The project utilizes several state-of-the-art NLP models for different tasks. 
Below is a summary of the models used, along with their respective release years:

1. **Sentiment Analysis**: 
   - **BERT** (`distilbert-base-uncased-finetuned-sst-2-english`), **2018**  
     BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model released by Google in 2018, 
   known for its effectiveness in natural language understanding tasks.

2. **Translation**:
   - **mBART** (`facebook/mbart-large-50`), **2020**  
     mBART (Multilingual BART) is a transformer-based sequence-to-sequence model by Facebook AI, released in 2020. 
   It supports multiple languages for translation and other tasks.

3. **Summarization**:
   - **BART** (`facebook/bart-large-cnn`), **2019**  
     BART (Bidirectional and Auto-Regressive Transformers) is a denoising autoencoder model also released by Facebook AI in 2019. 
   It is widely used for text summarization tasks.

4. **Text Generation**:
   - **GPT-2** (base pre-trained model, `gpt2`), **2019**  
     GPT-2 (Generative Pre-trained Transformer 2) is an autoregressive language model developed by OpenAI and released in 2019. 
   It is known for generating coherent and contextually relevant text.

   - **RAG** (Retrieval-Augmented Generation), **2020**  
     RAG is an approach developed by Facebook AI in 2020 that combines retrieval-based and generative methods 
   to improve the quality of generated text by leveraging relevant information from external sources.
<br>


## Archived Models

The following models were previously used in this project but have since been archived in favor of more advanced approaches:

1. **Sentiment Analysis**:
   - **Naive Bayes**, implemented using traditional machine learning techniques for sentiment classification. This method has been archived in favor of BERT for more accurate and context-aware sentiment analysis.

2. **Translation**:
   - **Helsinki-NLP**, a collection of translation models from the OPUS-MT project. These models were initially used for translation but have been replaced by mBART to support more languages and improve translation quality.

The archived implementations can be found in the `archive` directory of the project for reference purposes.
<br>
<br>


## Installation
  
1. Clone the repository: <br>
  `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`
2. Navigate to the project directory: <br>
  `cd MovieSense_NLP/src`
3. Install the required packages: <br>
  `pip install -r ../requirements.txt`
4. Run the main script: <br>
  `python app.py`
5. Run the main script: <br>
  `Open a web browser and go to http://127.0.0.1:5000/to use MovieSense.`
<br>


### Optional: Setting up a virtual environment

While the project runs without a virtual environment, it's recommended to use one for isolation:

1. Install `virtualenv` if you haven't: <br>
   `pip install virtualenv`
2. Clone the repository:<br>
   `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`
3. Navigate to the project directory:<br>
   `cd MovieSense_NLP`
4. Create a virtual environment:<br>
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
1. <b>Sentiment Analysis:</b> Reviews are categorised into 'positive' or 'negative' 
using the DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) and represented with emojis.<br>
2. <b>Translation:</b> Translate reviews into French.<br>
3. <b>Summarization:</b> Input the desired text to receive a concise summary.<br>
4. <b>Text Generation:</b> Enter a prompt (e.g., "Write a positive review about [Movie Name]") to generate a contextually relevant movie review. 
The RAG model retrieves relevant information from a database of existing movie reviews to ensure 
that the generated text is contextually accurate and aligned with the desired sentiment.
<br>


## Future Scope
- Extend translation support to other languages.
- Integration with online platforms or databases for automated review analysis.
- User accounts to save and manage past reviews.
- Experimenting with newer models (e.g., GPT-3, T5)
- Optimizing the current model implementations
- Enhanced prediction capabilities using Deep Learning.
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