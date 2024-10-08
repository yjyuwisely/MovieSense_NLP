﻿<h1>MovieSense: Sentiment Analysis, Translation, Summarization, and Text Generation</h1>
<p align="justify">Welcome to <b>MovieSense</b>, an advanced Natural Language Processing (NLP) project 
designed to analyze and enhance movie reviews using state-of-the-art AI techniques. 
This project offers a comprehensive suite of services, including sentiment analysis, translation, summarization, 
and text generation, specifically tailored for movie review data.
</p>
<br>
  

## Table of Contents
Navigate through this `README.md` to learn more about the project, its features, setup, and future plans.

### 1. Overview and Features
- 1.1 [Overview](#11-overview)  
- 1.2 [Background and Motivation](#12-background-and-motivation)  
- 1.3 [Project Composition Diagram](#13-project-composition-diagram)  
- 1.4 [Features and Advantages](#14-features-and-advantages)

### 2. Technologies and Models
- 2.1 [Project Development Environment](#21-project-development-environment)  
- 2.2 [Models and Methods Used](#22-models-and-methods-used)  
  - 2.2.1 [Archived Models](#221-archived-models)  

### 3. Problem Solving and Outcomes
- 3.1 [Issues and Solutions](#31-issues-and-solutions)  
- 3.2 [Expected Outcomes and Application Areas](#32-expected-outcomes-and-application-areas)  

### 4. Contributions and Execution
- 4.1 [Contributions](#41-contributions)  
- 4.2 [Project Timeline](#42-project-timeline)  

### 5. Installation
- 5.1 [Installation](#51-installation)  
  - 5.1.1 [Optional: Setting Up a Virtual Environment](#511-optional-setting-up-a-virtual-environment)  

### 6. Usage
- 6.1 [Usage Instructions](#61-usage-instructions)  
  - 6.1.1 [Analysis Button](#611-analysis-button)  
  - 6.1.2 [Generator Button](#612-generator-button)  
  - 6.1.3 [Reset Button](#613-reset-button)  
- 6.2 [Configuration and Output Settings](#62-configuration-and-output-settings)  
  - 6.2.1 [Setting the Output Location for Text Generation](#621-setting-the-output-location-for-text-generation)
  - 6.2.2 [Additional Tips](#622-additional-tips)  
  - 6.2.3 [Limitations](#623-limitations) 

### 7. Performance and Future Work
- 7.1 [Results and Performance](#71-results-and-performance)  
- 7.2 [Discussion](#72-discussion)  
- 7.3 [Future Scope](#73-future-scope)  
  - 7.3.1 [Model Enhancements](#731-model-enhancements)  
  - 7.3.2 [Performance Optimization](#732-performance-optimization)  
  - 7.3.3 [Evaluation and Metrics](#733-evaluation-and-metrics)  
  - 7.3.4 [System and Feature Enhancements](#734-system-and-feature-enhancements)   

### 8. Additional Resources
- 8.1 [Page Screenshots](#81-page-screenshots)  
- 8.2 [References and Further Readings](#82-references-and-further-readings)  

<br>


# 1. Overview and Features
## 1.1 Overview
<!--1. Starting with a general overview provides readers with a quick understanding of what the project is about. -->
<p align="justify">
<b>MovieSense</b> is an advanced NLP-powered application designed for in-depth analysis and enhancement of movie reviews 
using cutting-edge AI techniques. By leveraging state-of-the-art NLP models like BERT, mBART, BART, and 
GPT-3.5 Turbo enhanced with Retrieval-Augmented Generation (RAG), 
<b>MovieSense</b> provides a robust toolset for sentiment analysis, translation, summarization, and text generation. 
Users can classify reviews as positive or negative, translate reviews from English to French, 
generate concise summaries, and create new, contextually relevant reviews based on specific prompts.
</p>
<br>

## 1.2 Background and Motivation

These NLP tasks were chosen because they address key aspects of understanding and interacting with movie reviews: 
sentiment analysis helps capture the audience's emotional response, translation expands accessibility 
for non-English-speaking users, summarization condenses lengthy reviews for quick insights, 
and text generation offers a creative way to explore potential reviews.<br><br>
While this project focuses on English to French translation to leverage well-established datasets and models, 
my personal interest in translation and improving NLP models, 
particularly to create more natural translations between languages like English and Korean, 
inspired the inclusion of these tasks.
<br><br>

## 1.3 Project Composition Diagram

The following diagram provides an overview of the architecture and components of the **MovieSense** project:

![Project Composition Diagram](src/images/diagram_bold2.png)

*The diagram illustrates the flow of data through various components such as the NLP pipeline, models 
used for sentiment analysis, translation, summarization, and text generation, and the backend/frontend integration.*
<br><br>

## 1.4 Features and Advantages
<!-- 2. Highlighting the unique aspects and benefits of your project early on helps readers see its value and 
why they should be interested. -->
This serves as a foundation for future work in developing more nuanced translation systems for other language pairs.
Tailored for movie enthusiasts, critics, and NLP researchers, 
<b>MovieSense</b> aims to enrich the movie review experience and offers valuable insights through AI-driven methodologies.
Building on the project’s architectural components, the following features showcase how **MovieSense** leverages state-of-the-art NLP techniques to enhance user experience.
<br>
1. **Sentiment Analysis**
   - Classifies movie reviews as positive or negative using a pre-trained **BERT** model (**`distilbert-base-uncased-finetuned-sst-2-english`**), trained on Rotten Tomatoes data.

2. **Translation**
   - Translates reviews from English to French using the **mBART** model, demonstrating the handling of multilingual NLP tasks.

3. **Summarization**
   - Generates concise summaries of reviews with Facebook's **BART** model, effectively capturing the core content.

4. **Text Generation**
   - Produces movie reviews based on user prompts using the pre-trained **GPT-2** model and **GPT-3.5-turbo** enhanced by **Retrieval-Augmented Generation (RAG)** for contextual accuracy.
<br>

### Key Advantages

- **One-Click Multi-View**: Users can view three different features (Sentiment Analysis, Summarization, and Translation) 
with a single click, simplifying the interaction and enhancing user experience.

- **Intuitive Emojis**: Utilizes intuitive emojis to represent sentiment and features, 
making the interface more user-friendly and visually engaging.<br>
<br>

Additionally, the techniques demonstrated in **MovieSense** have applications in broader fields like marketing, 
customer feedback analysis, and content moderation, highlighting the adaptability and potential impact of this project.
<br><br>

# 2. Technologies and Models
 
## 2.1 Project Development Environment

The following tools, technologies, and configurations were used to develop the **MovieSense** project:

- **Programming Language**:  
  - Python 3.11.9

- **Frameworks and Libraries**:  
  - **Backend**: Flask for creating the web server.  
  - **Frontend**: HTML, CSS, JavaScript for user interface development and Bootstrap for styling.
  - **NLP Libraries**:  
    - **Transformers**: Core library from Hugging Face used for NLP models, including **BERT**, **mBART**, **BART**, **GPT-2**, **GPT-3.5-turbo**, and **RAG**.  
    - **NLTK**: Used for text processing tasks and basic NLP utilities such as tokenization, stemming, and stop-word removal.
    - **Sentence-Transformers**: Used for generating sentence embeddings with models like **all-MiniLM-L6-v2** to facilitate semantic search and enhance the retrieval process in the RAG pipeline.

- **Development Tools**:  
  - Visual Studio Code

- **Version Control**:  
  - Git and GitHub

- **Dependency Management**:  
  - Python: `pip` for installing packages listed in `requirements.txt`.

- **Operating System**:  
  - Windows 11

- **Hardware Requirements**:  
  - 8GB RAM minimum, GPU recommended for training deep learning models.

- **Cloud Services and External APIs**:  
  - OpenAI API for accessing GPT-3.5-turbo.  
  - Hugging Face Model Hub for pre-trained models.

- **Database System**:  
  - Chroma Vector Database for similarity search.

- **Environment Setup**:  
  - Install dependencies: `pip install -r requirements.txt`.  
  - Create a `.env` file to store the OpenAI API key.

These components constitute the development environment required to run and contribute to the **MovieSense** project.
<br><br>

## 2.2 Models and Methods Used
<!-- 4. This section dives deeper into the specifics of the models and methods, providing more technical readers with detailed insights. -->
The project utilizes several state-of-the-art NLP models and methods, each tailored to specific tasks for analyzing and generating movie reviews. These models were selected for their effectiveness, performance, and suitability for the required NLP tasks. Below is a summary of the models and methods used:
<br><br>
1. **BERT (Bidirectional Encoder Representations from Transformers), 2018**  

   - **Model Used**: `distilbert-base-uncased-finetuned-sst-2-english`  
   
   - **Purpose**: Sentiment analysis to classify movie reviews as positive or negative.  
   
   - **Details**: A smaller, faster version of BERT fine-tuned on the SST-2 dataset, 
   designed to understand the context of words for accurate sentiment classification.
   <br><br>

2. **mBART (Multilingual BART), 2020**  
   
   - **Model Used**: `facebook/mbart-large-50-many-to-many-mmt`  
   
   - **Purpose**: Translation of movie reviews into French to support multilingual audiences.  
   
   - **Details**: A sequence-to-sequence transformer model tailored for many-to-many translation tasks, 
   enhancing accessibility by translating content from English to multiple languages.
   <br><br>

3. **BART (Bidirectional and Auto-Regressive Transformers), 2019**  
   
   - **Model Used**: `facebook/bart-large-cnn`  
   
   - **Purpose**: Summarization of extended movie reviews.  
   
   - **Details**: Combines a bidirectional encoder with an autoregressive decoder to produce high-quality summaries 
   that capture the essence of longer texts.
   <br><br>

4. **GPT-2 (Generative Pre-trained Transformer 2), 2019**  
   
   - **Model Used**: `gpt2` (Hugging Face Transformers)  
   
   - **Purpose**: Baseline text generation based on user prompts.  
   
   - **Details**: Utilized as a baseline model for generating movie reviews. The pre-trained model 
   from Hugging Face Transformers serves as a comparison against more advanced text generation techniques.
   <br><br>

5. **Retrieval-Augmented Generation (RAG) with GPT-3.5-turbo, 2023**  

   - **Model Used**: `gpt-3.5-turbo` (Accessed via OpenAI API) with retrieval methods  

   - **Purpose**: To generate contextually relevant text by retrieving and using relevant information 
   from a knowledge base before generating the output.  

   - **Details**: This approach combines the GPT-3.5-turbo language model, accessed through the OpenAI API, 
   with a retrieval mechanism that uses `all-MiniLM-L6-v2` embeddings and the `Chroma` vector database. 
   When a user provides a prompt, the system first retrieves the most relevant documents using similarity search in `Chroma`. 
   The retrieved documents are then used to guide the GPT-3.5-turbo model 
   to produce more accurate and context-aware text outputs.  
   
   **Note**: The OpenAI API key is required to access the GPT-3.5-turbo model, and usage of the API may incur costs depending on your OpenAI subscription plan.
   <br><br>

6. **Sentence Embedding Model, 2020**  
   
   - **Model Used**: `all-MiniLM-L6-v2` (Sentence Transformers)  
   
   - **Purpose**: To create dense vector embeddings for similarity search in the RAG approach.  
   
   - **Details**: Generates embeddings for efficient similarity searches, supporting the RAG model by providing relevant context for text generation.
   <br><br>

7. **Vector Database and Similarity Search, 2023**  
   
   - **Tool Used**: `Chroma` (Vector Database)  
   
   - **Purpose**: To store document embeddings and perform fast similarity searches for the RAG approach.  
   
   - **Details**: Facilitates the retrieval of relevant documents, enhancing the contextual accuracy of text generated by the RAG model.
<br><br>

### 2.2.1 Archived Models
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

# 3. Problem Solving and Outcomes

## 3.1 Issues and Solutions
<!-- 5. Discussing the challenges faced and how you solved them adds depth to your project's story 
and demonstrates problem-solving skills.-->

### Issue: Suboptimal Performance in Text Generation

- **Problem**: The text generation using GPT-2 did not produce contextually relevant results 
for longer movie reviews. The model struggled to maintain coherence over extended outputs, 
leading to repetitive or irrelevant content.

  - **Examples of GPT-2 Output**:
    - **Generated Positive Review**: `"Review a movie. Review a movie. Review a movie. Review a movie. Review a movie. Review a movie. Review a movie. Review a movie."`
    - **Generated Negative Review**: `"I'm not sure if I'm going to be able to do that. I'm not sure if I'm going to be able to do that. I'm not sure if I'm going to be able to do that. I'm not..."`

- **Solution**: Integrated **GPT-3.5 Turbo** with **Retrieval-Augmented Generation (RAG)** 
to improve performance and contextual relevance. 
By using RAG, the model could retrieve relevant information from a knowledge base 
before generating text, significantly enhancing the coherence and quality of the outputs.

- **Outcome**: Improved text generation quality, with outputs that were more contextually relevant to the input prompts.

A screenshot of the text generation output using **GPT-3.5 Turbo** 
with **Retrieval-Augmented Generation (RAG)** can be seen in the [Page Screenshots](#81-page-screenshots), 
demonstrating the model's ability to produce contextually relevant and coherent movie reviews 
based on specific input prompts.

<br>

## 3.2 Expected Outcomes and Application Areas
<!-- 6. Explaining the potential impact and real-world applications shows the relevance and utility of your work. -->
**Expected Outcomes**:
- **Enhanced Review Insights**: By providing sentiment analysis and summarization, **MovieSense** enables users to quickly grasp the essence and emotional tone of movie reviews.
- **Increased Accessibility**: Translation features broaden the reach of reviews to non-English-speaking audiences, expanding the user base and improving the inclusivity of the platform.
- **Creative Exploration**: Text generation capabilities allow users to experiment with new review content, fostering creative engagement with movie reviews.

**Applications**:
- **Marketing**: Analyzing customer feedback and reviews to tailor marketing strategies and improve product offerings.
- **Customer Feedback Analysis**: Understanding customer sentiments and opinions from reviews to enhance service quality and customer satisfaction.
- **Content Moderation**: Automatically summarizing and filtering user-generated content for relevance and appropriateness.
- **Broader NLP Research**: Applying the techniques demonstrated in **MovieSense** to other domains and languages for more generalized NLP applications.
<br>

# 4. Contributions and Execution
## 4.1 Contributions
<!-- 7. Clearly states your role and contributions, which is especially important for job applications. -->
As the sole developer of <b>MovieSense</b>, I was responsible for the entire project lifecycle, including:

1. **Project Design and Architecture**: 
   - Conceptualized the overall structure of the NLP pipeline 
to handle multiple tasks such as sentiment analysis, translation, summarization, and text generation.

2. **Model Selection, Implementation, and Optimization**: 
   - Selected appropriate pre-trained models 
   (e.g., DistilBERT, mBART, BART, GPT-2) based on task requirements and integrated them into the project using the Hugging Face Transformers library. 
   Utilized techniques such as Retrieval-Augmented Generation (RAG) to enhance the contextual relevance and accuracy of text generation. 
   For specific details, refer to the [Models and Methods Used](#22-models-and-methods-used) section.

   - Achieved an improvement in context relevance and coherence, 
   as demonstrated by the BLEU score increase from **0.00 to 0.09** for positive reviews 
   and from **0.00 to 0.02** for negative reviews. 

3. **Innovative Use of Retrieval-Augmented Generation (RAG)**: 
   - Integrated the **RAG** approach to improve text generation by retrieving relevant information 
   from a knowledge base, resulting in a noticeable enhancement in the contextual accuracy of generated content. This technique led to more meaningful and coherent outputs, overcoming the limitations of the baseline GPT-2 model.
   
4. **Data Preparation and Evaluation**: 
   - Loaded, labeled, and preprocessed movie review datasets for sentiment analysis, 
   and implemented robust evaluation metrics to assess model performance, including BLEU scores 
   to quantify improvements in text generation.

5. **Integration and Deployment**: 
   - Developed the backend using Flask and integrated the NLP models into a cohesive application. 
   
   - Designed a user-friendly frontend using HTML, CSS, and JavaScript for deployment.

6. **Documentation**: 
   - Created comprehensive documentation, including this `README.md`, detailing 
   the project's features, models, installation, usage, and future scope.

This comprehensive approach not only demonstrates my ability to design, develop, and deploy complex NLP systems
but also highlights my commitment to continuous optimization and innovation to achieve better performance.
<br><br>

## 4.2 Project Timeline
<!-- 8. Provides context for the development process and how the project evolved over time. -->
- **October 6, 2023**  
  - Created the initial web template for the **MovieSense** interface.

- **October 7-8, 2023**  
  - Implemented translation from English to French.
  - Added Bootstrap styling for improved UI/UX and made final styling adjustments.
  - Developed initial Sentiment Analysis and Summarization features using Transformer models.
  - Introduced a Reset button to clear inputs and adjusted page styles.

- **Mid-October 2023**  
  - Trained models using a new dataset to improve performance.

- **August 22-24, 2024**  
  - Updated **requirements.txt** for dependency management.
  - Enhanced Sentiment Analysis by replacing Naive Bayes with a pre-trained BERT model.
  - Switched the translation method to use the mBART model.

- **August 27, 2024**  
  - Refactored text generation to use `max_new_tokens` for better output control and excluded the prompt message.
  - Integrated GPT-2 model for initial text generation.

- **August 29-31, 2024**  
  - Refactored project structure: Separated `main_movie_sense.py` into individual modules for Sentiment Analysis, Text Generation, Summarization, and Translation.
  - Implemented RAG-based text generation using LangChain for improved contextual relevance.

- **September 1, 2024**  
  - Added `.env` file for OpenAI API key configuration and updated `.gitignore` to exclude it.
  - Updated **requirements.txt** to include necessary dependencies.

- **September 2, 2024**  
  - Completed the evaluation metrics for text generation using BLEU scores.
  - Enhanced `README.md` for clarity and completeness.
<br>

# 5. Installation
## 5.1 Installation
<!-- 9. After understanding the project, readers might want to try it out. Having the installation instructions here makes sense. -->
Follow these steps to set up **MovieSense** on your local machine:

1. **Clone the repository**: Clone the project repository from GitHub to your local machine.<br>
  `git clone https://github.com/yjyuwisely/MovieSense_NLP.git`

2. **Navigate to the project directory**: Move into the source directory where the application code resides.<br>
  `cd MovieSense_NLP/src`

3. **Create an `.env` file for environment variables**: Create a `.env` file in the `src` directory to store your OpenAI API key  . 
   - Open a terminal or text editor and create a new file named `.env`:<br>
   `touch .env`

   - Add your OpenAI API key to the `.env` file in the following format:<br>
   `OPENAI_API_KEY=your_openai_api_key_here`<br>
   Replace `your_openai_api_key_here` with your actual OpenAI API key.

4. **Install the required packages**: Install all necessary Python packages and dependencies listed in `requirements.txt`.<br>
  `pip install -r ../requirements.txt`

5. **Run the main script**: Start the Flask web application by running the following command:<br>
  `python app.py`
   - **Expected Output**: You should see output in the terminal indicating that the Flask server is running, 
   e.g., Running on `http://127.0.0.1:5000/`.

6. **Access the application**: Open a web browser and navigate to the following URL to use **MovieSense**:<br>
  `http://127.0.0.1:5000/`
   - **Expected Outcome**: You should see the **MovieSense** interface where you can use features 
   like sentiment analysis, translation, summarization, and text generation.
<br>

### 5.1.1 Optional: Setting up a virtual environment
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

# 6. Usage
## 6.1 Usage Instructions
<!-- 10. Explains how to use the project after it is set up, logically following the installation steps. -->

Follow the instructions below to use the features of **MovieSense** on the web interface:

### 6.1.1 **Analysis Button**  
   - **Description**: The **"Analysis"** button performs three tasks simultaneously:  
     - **Sentiment Analysis**: Classifies the input movie review as 'positive' or 'negative' using the DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`). The result is represented with an emoji (😊 for positive, 😞 for negative).
     - **Translation**: Translates the input movie review from English to French using the mBART model (`facebook/mbart-large-50-many-to-many-mmt`).
     - **Summary**: Generates a concise summary of the input movie review using the BART model (`facebook/bart-large-cnn`).

   - **How to Use**:  
     - Enter a movie review in the input text box.
     - Click the **"Analysis"** button. The system will:
       - Display the predicted sentiment with an emoji.
       - Provide the French translation below the input box.
       - Show the summarized version of the review on the right side of the webpage.
<br>
       
### 6.1.2 **Generator Button**  
   - **Description**: The **"Generator"** button generates a new movie review based on a specific sentiment using the Retrieval-Augmented Generation (RAG) approach with the GPT-3.5-turbo model.
   - **How to Use**:  
     - Click the **"Generator"** button. You will be prompted to select either "Write a positive review about a movie" or "Write a negative review about a movie."
     - The generated movie review, based on your chosen sentiment, will appear in the input text box.
<br>
     
### 6.1.3 **Reset Button**  
   - **Description**: The **"Reset"** button clears all input fields, allowing you to start a new analysis or generate a new review.
   - **How to Use**:  
     - Click the **"Reset"** button to clear the input text box and reset the page to its default state.'     

<br>

## 6.2 Configuration and Output Settings

### 6.2.1 Setting the Output Location for Text Generation

By default, the text generation using the GPT-3.5-turbo model outputs the result in the terminal. If you want the output to be displayed directly in the text box on the webpage:

- **To Display Output in the Web Interface**:  
  Modify the code in the `text_generation_rag_lanchain.py` file by commenting out the terminal-based output section and uncommenting the `generate_text` function. This change will enable the output to be displayed on the web interface.

- **To Display Output in the Terminal**:  
  Ensure the terminal-based output section in `text_generation_rag_lanchain.py` is active, and comment out the `generate_text` function.

For more details, refer to the [`text_generation_rag_lanchain.py`](src/nlp/text_generation_rag_lanchain.py) file in the repository.

<br>

### 6.2.2 Additional Tips

- **Ensure Proper Configuration**: Make sure you have set up your `.env` file with the OpenAI API key before using the text generation feature.
- **Internet Connection Required**: An internet connection is necessary to access the OpenAI API for text generation.

<br>

### 6.2.3 Limitations

- **Text Generation Feature**: The text generation feature using the OpenAI API requires a paid subscription to access the GPT-3.5-turbo model. Users will need their own API key with an active subscription to enable this feature.

<br><br>

<!-- 12. Presents the results and evaluation metrics after readers know how the project works. -->
<!-- This section will be updated soon to include the following:
- Evaluation metrics (e.g., accuracy, BLEU score, ROUGE score) for sentiment analysis, translation, summarization, and text generation.
- Sample outputs and comparisons with baseline models.
- Discussion on the performance and potential improvements. -->

## 7. Performance and Future Work

### 7.1 Results and Performance
To evaluate the effectiveness of the text generation models, the **BLEU** score is used. **BLEU** measures the precision of n-grams between the generated text and a reference text, assessing the fluency and accuracy of the generated movie reviews. This metric is particularly useful for evaluating the relevance and coherence of text generated by language models.

We evaluated two pre-trained models for text generation on movie reviews:

1. **Baseline GPT-2 Model**: A pre-trained GPT-2 model used without fine-tuning.
2. **RAG-Enhanced GPT-3.5-turbo**: A Retrieval-Augmented Generation approach combining document retrieval with GPT-3.5-turbo, used without fine-tuning.

#### BLEU Score Results:
- **Baseline GPT-2 Model**: 
  - **Positive Review**: 0.00  
  - **Negative Review**: 0.00  

- **RAG-Enhanced GPT-3.5-turbo**:
  - **Positive Review**: 0.09  
  - **Negative Review**: 0.02  
<br>
 
### 7.2 Discussion

The BLEU score comparison indicates that the RAG-enhanced GPT-3.5-turbo model outperforms the baseline GPT-2 model, even without fine-tuning. While the BLEU scores are relatively low, which is expected in creative text generation tasks due to the diversity of possible outputs, the RAG-enhanced GPT-3.5-turbo still shows better performance in generating more contextually relevant and coherent text compared to GPT-2.

The baseline GPT-2 model produced highly repetitive and non-informative output, resulting in BLEU scores of 0.00 for both positive and negative reviews. This suggests that GPT-2, in its pre-trained state, struggles with generating meaningful content for movie reviews without additional training or fine-tuning.

The RAG-enhanced GPT-3.5-turbo model, with BLEU scores of 0.09 for positive reviews and 0.02 for negative reviews, demonstrates some level of contextual relevance and coherence, although there is still room for improvement.
<br><br>

### 7.3 Future Scope 
<!-- 13. Outlines future improvements and extensions, which is a natural progression after discussing current results.-->
<!-- The next steps for your project, inspires confidence in its ongoing development, and can attract contributions or collaborations. -->
<!-- Potential Enhancements, New Features or Functionalities, Further Research, 
Long-Term Goals, Opportunities for Collaboration -->

Based on the current results, several improvements and extensions are planned:
<br>
#### 7.3.1 Model Enhancements

1. **Fine-Tuning and Advanced Models**: Fine-tune GPT-2 and GPT-3.5-turbo on domain-specific datasets. Experiment with other models such as GPT-4, T5, or local alternatives (e.g., T5, DistilGPT-2) to enhance text generation while reducing API costs.

2. **Advanced Learning Techniques**: 
   
   - **Transfer Learning**: Use larger pre-trained models for improved accuracy in specific tasks.
   
   - **Reinforcement Learning from Human Feedback (RLHF)**: A method where the model learns and refines its outputs based on feedback from human evaluators, helping to align generated text more closely with human preferences and expectations.
   
   - **Knowledge Distillation**: Create smaller, efficient models that retain the performance of larger models, allowing for faster inference.
   
   - **Attention Mechanisms and Transformer Variants**: Utilize models like **Longformer** and **BigBird** to effectively handle longer sequences of text.
   
   - **Self-Supervised Learning (SSL)**: A technique where the model is trained on large amounts of unlabeled data, learning patterns and representations in an unsupervised manner, which can then be fine-tuned for specific tasks using a smaller labeled dataset.
<br><br>

#### 7.3.2 Performance Optimization

1. **Optimize Current Model Implementations for Better Performance**:

   - Fine-Tuning pre-trained models on domain-specific datasets to improve performance for tasks like sentiment analysis, translation, and summarization.

   - Model Quantization to reduce model size and speed up inference, even when using pre-trained models.

   - Pruning to remove unnecessary parameters and decrease computational load, particularly useful during fine-tuning.

   - Efficient Batching and Data Loading to enhance data processing speed, applicable to both training and inference.

   - Using Mixed-Precision Training to improve training speed and reduce memory usage when fine-tuning models.

   - Hyperparameter Tuning to find the optimal settings for enhanced model performance, even for pre-trained models.

   - Caching and Pre-computation to minimize redundant calculations during inference, improving responsiveness.
<br><br>
#### 7.3.3 Evaluation and Metrics

1. **Using Additional Metrics**: Consider using other evaluation metrics like **ROUGE** or **METEOR**, 
or even conducting human evaluations to better assess the quality of generated text.
<br><br>
#### 7.3.4 System and Feature Enhancements

1. **Enhancing Retrieval Methods**: Improve the document retrieval mechanism in the RAG approach 
by incorporating **FAISS (Facebook AI Similarity Search)** 
to ensure faster and more efficient similarity searches. 
This could provide more relevant context to the model, 
helping to generate more accurate and context-aware outputs.

2. **Extend Translation Support**: Add support for more languages.

3. **Integrate with Online Platforms**: Connect with online platforms or databases for automated review analysis.

4. **Add User Accounts**: Implement user accounts to save and manage past reviews.

<br>

# 8. Additional Resources
<!-- Positive Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5udNA%2Fbtsxp3doUAk%2FU38yky0rcDo3KPc6yCGtLk%2Fimg.png">
Negative Sentiment Example:
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCpxxG%2FbtsxvIFFLyI%2FRthmnpzRTiaanXaDxgCEjK%2Fimg.png"> -->

## 8.1 Page Screenshots
<!-- 11. Placing the screenshots at the end serves as a visual summary and reinforces everything explained before.-->
Below are some screenshots of the **MovieSense** interface showcasing its key functionalities:

### Initial Web Page
![Initial Web Page](src/images/initial.png)<br>
*This screenshot displays the initial state of the **MovieSense** web interface 
before any input is provided. It shows the layout of the input text box and 
the main buttons (Analysis, Generator, Reset) available for user interaction.*
<br><br>

### Positive Sentiment Example
![Positive Sentiment Example](src/images/positive.png)<br>
*This screenshot demonstrates a movie review analyzed by the **MovieSense** tool, showing a positive sentiment prediction along with the translated French version and a summarized output.*
<br><br>

### Negative Sentiment Example
![Negative Sentiment Example](src/images/negative.png)<br>
*This screenshot demonstrates a movie review analyzed by the **MovieSense** tool, showing a negative sentiment prediction along with the translated French version and a summarized output.*
<br><br>

### Text Generation Output Example

Below are examples of text generated by the GPT-3.5-turbo model in the terminal:

#### Positive Review Output
![Positive Review Output](src/images/terminal-positive-output1.png)<br>
![Positive Review Output](src/images/terminal-positive-output2.png)
<br>
*These screenshots show the output of a positive movie review generated using the GPT-3.5-turbo model via the terminal. Since I do not currently have a paid subscription to the OpenAI API, the generated text is displayed in the terminal rather than within the web interface.*

#### Negative Review Output
![Negative Review Output](src/images/terminal-negative-output1.png)<br>
![Negative Review Output](src/images/terminal-negative-output2.png)
<br>
*These screenshots show the output of a negative movie review generated using the GPT-3.5-turbo model via the terminal. The terminal output illustrates how the model handles generating text for different sentiments.*

<br>

## 8.2 References and Further Readings
<!-- 14. Offers additional resources for readers who want to learn more or verify the information. -->
1. Artasanchez, A., & Joshi, P. (2019). *Artificial Intelligence with Python: Your complete guide to building intelligent apps using Python 3.x* (2nd ed.). Packt Publishing.<br>
[Available on Amazon](https://www.amazon.com/Artificial-Intelligence-Python-complete-intelligent/dp/183921953X)<br>
2. Lamons, M., Kumar, R., & Nagaraja, A. (2018) *Python Deep Learning Projects: 9 projects demystifying neural network and deep learning models for building intelligent systems* (1st ed.). Packt Publishing.<br>
[Available on Amazon](https://www.amazon.com/Python-Deep-Learning-Projects-demystifying/dp/1788997093)<br>
3. Seo, J. (2024). *Developing AI services based on LLM with LangChain*. Gilbut. (Written in Korean) <br>
[Available on GitHub](https://github.com/gilbutITbook/080413)<br>
4. Tunstall, L., von Werra, L., & Wolf, T. (2022) *Natural Language Processing with Transformers: Building Language Applications with Hugging Face* (1st ed.). O'Reilly Media.<br>
[Available on Amazon](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)<br>