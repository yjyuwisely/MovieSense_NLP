<h1>MovieSense: Sentiment Analysis, Translation, Summarization, and Text Generation</h1>
<p align="justify">Welcome to <b>MovieSense</b>, an advanced Natural Language Processing (NLP) project 
designed to analyze and enhance movie reviews using state-of-the-art AI techniques. 
This project offers a comprehensive suite of services, including sentiment analysis, translation, summarization, 
and text generation, specifically tailored for movie review data.
</p>
<br>
  
## Table of Contents

Navigate through this `README.md` to learn more about the project, its features, setup, and future plans.

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Models and Methods Used](#models-and-methods-used)
   - [Archived Models](#archived-models)
5. [Your Contribution](#your-contribution) 
6. [Installation](#installation)
   - [Optional: Setting Up a Virtual Environment](#optional-setting-up-a-virtual-environment)
7. [Usage](#usage)
8. [Results and Performance](#results-and-performance)
9. [Future Scope](#future-scope)
10. [References and Further Readings](#references-and-further-readings)
11. [Page Screenshot](#page-screenshot)
<br>

## Overview
<p align="justify">
<b>MovieSense</b> is an advanced NLP-powered application designed for in-depth analysis and enhancement of movie reviews 
using cutting-edge AI techniques. By leveraging state-of-the-art NLP models like BERT, mBART, BART, and GPT-2, 
<b>MovieSense</b> provides a robust toolset for sentiment analysis, translation, summarization, and text generation. 
Users can classify reviews as positive or negative, translate reviews from English to French, 
generate concise summaries, and create new, contextually relevant reviews based on specific prompts.<br><br>
These NLP tasks were chosen because they address key aspects of understanding and interacting with movie reviews: 
sentiment analysis helps capture the audience's emotional response, translation expands accessibility 
for non-English-speaking users, summarization condenses lengthy reviews for quick insights, 
and text generation offers a creative way to explore potential reviews.<br><br>
While this project focuses on English to French translation to leverage well-established datasets and models, 
my personal interest in translation and improving NLP models, 
particularly to create more natural translations between languages like English and Korean, 
inspired the inclusion of these tasks. 
This serves as a foundation for future work in developing more nuanced translation systems for other language pairs.<br><br>
Tailored for movie enthusiasts, critics, and NLP researchers, 
<b>MovieSense</b> aims to enrich the movie review experience and offers valuable insights through AI-driven methodologies. 
Future plans include expanding translation support, integrating with online platforms for automated analysis, 
and exploring more advanced models for enhanced performance. 
Additionally, the techniques demonstrated in <b>MovieSense</b> have applications in broader fields 
like marketing, customer feedback analysis, and content moderation, 
highlighting the adaptability and potential impact of this project.
</p>
<br>

## Features
1. **Sentiment Analysis**
   - Classifies movie reviews as positive or negative using a pre-trained BERT model (`distilbert-base-uncased-finetuned-sst-2-english`), trained on Rotten Tomatoes data.
2. **Translation**
   - Translates reviews from English to French using the mBART model, demonstrating the handling of multilingual NLP tasks.
3. **Summarization**
   - Generates concise summaries of reviews with Facebook's BART model, effectively capturing the core content.
4. **Text Generation**
   - Produces movie reviews based on user prompts with GPT-2, enhanced by Retrieval-Augmented Generation (RAG) for contextual accuracy.
<br>  

## Technologies Used
- **Languages:** Python
- **NLP Techniques & Models:**
  - **Sentiment Analysis:** BERT
  - **Translation:** mBART 
  - **Summarization:** BART 
  - **Text Generation:** GPT-2, currently using the base pre-trained model to generate movie reviews based on specific prompts.
- **Frameworks/Libraries:**
  - **NLTK**: Used for text processing tasks and basic NLP utilities.
  - **Transformers**: Core library for NLP models such as BERT, GPT-2, BART, and mBART.
  - **Flask**: Backend development framework used to build and deploy web services for the project.
- **Frontend:** HTML, CSS, JavaScript for user interface development.
<br>

## Models and Methods Used
The project utilizes several state-of-the-art NLP models, each tailored for specific tasks related to analyzing and generating movie reviews. These models were selected based on their effectiveness, performance, and suitability for the required NLP tasks. Below is a summary of the models and methods used, including their release years and core functionalities:
<br><br>

1. **BERT (Bidirectional Encoder Representations from Transformers)**, **2018**  

   - **Model Used**: `distilbert-base-uncased-finetuned-sst-2-english`  
   
   - **Purpose**: Sentiment analysis to classify movie reviews as positive or negative.  
   
   - **Details**: BERT is a transformer-based model developed by Google, known for its ability 
   to understand the context of words in a sentence. DistilBERT is a smaller, faster, and lighter version of BERT, 
   fine-tuned on the SST-2 dataset to improve performance for sentiment classification tasks.
   <br><br>
2. **mBART (Multilingual BART)**, **2020**  
   
   - **Model Used**: `facebook/mbart-large-50-many-to-many-mmt`  
   
   - **Purpose**: Translation of movie reviews into French to support multilingual audiences.  
  
   - **Details**: mBART is a sequence-to-sequence transformer model that extends BART's capabilities to multiple languages. 
   The `facebook/mbart-large-50-many-to-many-mmt` variant is specifically designed for many-to-many translation tasks, 
   enhancing the accessibility of movie reviews for non-English-speaking audiences by effectively translating content 
   from English to French and other languages.
   <br><br>
3. **BART (Bidirectional and Auto-Regressive Transformers)**, **2019**  

   - **Model Used**: `facebook/bart-large-cnn`  
   
   - **Purpose**: Summarization of extended movie reviews.  
   
   - **Details**: BART is a transformer-based model developed by Facebook AI, 
   widely used for text generation and summarization tasks. 
   It combines a bidirectional encoder with an autoregressive decoder to produce high-quality summaries 
   that capture the essence of longer texts.
   <br><br>
4. **GPT-2 (Generative Pre-trained Transformer 2)**, **2019**  

   - **Model Used**: `gpt2`  
   
   - **Purpose**: Text generation based on user prompts.  
   
   - **Details**: GPT-2 is an autoregressive language model developed by OpenAI, capable of 
   generating coherent and contextually relevant text. The project employs a Retrieval-Augmented Generation (RAG) approach, 
   which enhances text relevance by retrieving information from a knowledge base before generation.
   <br><br>
5. **RAG (Retrieval-Augmented Generation)**, **2020**  

   - **Model Used**: `facebook/rag-token-nq` or similar variant  
   
   - **Purpose**: Enhancing text generation by retrieving relevant information from a knowledge base.  
   
   - **Details**: RAG is an approach developed by Facebook AI in 2020 that combines retrieval-based and generative methods 
   to improve the quality of generated text. It leverages relevant information from external sources to ensure 
   that the generated content is more accurate and contextually appropriate.
<br>

### Archived Models
The following models were previously used in this project but have since been archived in favor of more advanced approaches. Below is a summary of the archived models and their roles in the project:
<br><br>

1. **Naive Bayes**
   
   - **Purpose**: Used for Sentiment Analysis to classify movie reviews as positive or negative.  
   
   - **Details**: Implemented using traditional machine learning techniques for sentiment classification. 
   This method has been archived in favor of BERT, which provides more accurate and context-aware sentiment analysis due to its transformer-based architecture.
   <br><br>
2. **Helsinki-NLP**
   
   - **Purpose**: Used for Translation of movie reviews into French to support multilingual audiences.  
   
   - **Details**: A collection of translation models from the OPUS-MT project. 
   These models were initially used for translation but have been replaced by mBART, 
   which supports more languages and provides better translation quality 
   due to its sequence-to-sequence transformer architecture.

The archived implementations can be found in the `archive` directory of the project for reference purposes.
<br>
<br>

## Your Contribution
As the sole developer of <b>MovieSense</b>, I was responsible for the entire project lifecycle, including:

- **Project Design and Architecture**: Conceptualized the overall structure of the NLP pipeline to handle multiple tasks such as sentiment analysis, translation, summarization, and text generation.

- **Model Selection and Implementation**: Chose appropriate models (e.g., BERT, mBART, BART, GPT-2) based on task requirements and implemented them using Hugging Face Transformers.

- **Data Preparation and Evaluation**: Loaded, labeled, and shuffled movie review datasets for sentiment analysis. Implemented evaluation metrics to assess model performance and ensure robust results.

- **Model Implementation and Optimization**: Implemented pre-trained models for specific tasks like sentiment analysis and summarization, leveraging retrieval-augmented generation (RAG) to improve contextual relevance and accuracy.

- **Evaluation and Metrics**: Evaluated model performance using metrics such as accuracy, BLEU, and ROUGE, and iteratively improved models based on these results.

- **Integration and Deployment**: Developed the backend using Flask and integrated the NLP models into a cohesive application. Designed a user-friendly frontend using HTML, CSS, and JavaScript for deployment.

- **Documentation**: Created comprehensive documentation, including this `README.md`, detailing the project's features, models, installation, usage, and future scope.

This end-to-end approach demonstrates my ability to design, develop, and deploy complex NLP systems while continuously optimizing them for better performance.
<br>
<br>

## Installation
Follow these steps to set up **MovieSense** on your local machine:

1. **Clone the repository**: Clone the project repository from GitHub to your local machine.<br>
  `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`

2. **Navigate to the project directory**: Move into the source directory where the application code resides.<br>
  `cd MovieSense_NLP/src`

3. **Install the required packages**: Install all necessary Python packages and dependencies listed in `requirements.txt`.<br>
  `pip install -r ../requirements.txt`

4. **Run the main script**: Start the Flask web application by running the following command:<br>
  `python app.py`
   - **Expected Output**: You should see output in the terminal indicating that the Flask server is running, 
   e.g., Running on `http://127.0.0.1:5000/`.

5. **Access the application**: Open a web browser and navigate to the following URL to use **MovieSense**:<br>
  `http://127.0.0.1:5000/`
   - **Expected Outcome**: You should see the **MovieSense** interface where you can use features 
   like sentiment analysis, translation, summarization, and text generation.
<br>

### Optional: Setting up a virtual environment
While the project can run without a virtual environment, it's highly recommended to use one for package isolation and to avoid conflicts with other Python projects.

1. **Install `virtualenv`**: If you don't have `virtualenv` installed, run the following command:<br>
   `pip install virtualenv`

2. **Clone the repository**: If you haven't already, clone the repository from GitHub:<br>
   `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`

3. **Navigate to the project directory**: Move into the project directory where the virtual environment will be set up.<br>
   `cd MovieSense_NLP`

4. **Create a virtual environment**: Create a new virtual environment named `.env`<br>
   `virtualenv .env`

5. **Activate the virtual environment**: Activate the environment depending on your operating system:
   - On **macOS** and **Linux**:  
     `source .env/bin/activate`
   - On **Windows**:  
     `.\.env\Scripts\activate`

6. **Install dependencies and run the application**: Now, navigate to the `src`
directory and follow steps 3 to 5 from the [Installation section](#installation) to 
install the required packages, and run the application.

7. **Deactivate the virtual environment**: When you are done working on the project, deactivate the virtual environment using:<br>
   `deactivate`
<br>

## Usage
1. <b>Sentiment Analysis:</b> Reviews are categorised into 'positive' or 'negative' 
using the DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) and represented with emojis.<br>

2. <b>Translation:</b> Translate reviews into French.<br>

3. <b>Summarization:</b> Input the desired text to receive a concise summary.<br>

4. <b>Text Generation:</b> Enter a prompt (e.g., "Write a positive review about [Movie Name]") 
to generate a contextually relevant movie review. 
The RAG model retrieves relevant information from a database of existing movie reviews to ensure 
that the generated text is contextually accurate and aligned with the desired sentiment.
<br>

## Results and Performance
This section will be updated soon to include the following:
- Evaluation metrics (e.g., accuracy, BLEU score, ROUGE score) for sentiment analysis, translation, summarization, and text generation.
- Sample outputs and comparisons with baseline models.
- Discussion on the performance and potential improvements.

Stay tuned for updates!
<br><br>

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
3. Seo, J. (2024). *Developing AI services based on LLM with LangChain*. Gilbut. (Written in Korean) <br>
[Available on GitHub](https://github.com/gilbutITbook/080413)<br>
4. Tunstall, L., von Werra, L., & Wolf, T. (2022) *Natural Language Processing with Transformers: Building Language Applications with Hugging Face* (1st ed.). O'Reilly Media.<br>
[Available on Amazon](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)<br>
<br>

## Page Screenshot
Positive Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5udNA%2Fbtsxp3doUAk%2FU38yky0rcDo3KPc6yCGtLk%2Fimg.png">
Negative Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCpxxG%2FbtsxvIFFLyI%2FRthmnpzRTiaanXaDxgCEjK%2Fimg.png">