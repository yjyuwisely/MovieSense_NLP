"""
Sentiment Analysis using BERT:

- Utilizes a pre-trained BERT model for sentiment analysis, 
  applied to the 'sentiment_data' dataset derived from Rotten Tomato reviews.

- This implementation replaces the previous Naive Bayes method, 
  leveraging BERT's deep learning capabilities for more accurate sentiment classification.

- The 'sentiment_data' directory contains reviews categorized as 'positive' and 'negative' 
  stored in corresponding subfolders.
"""

import os
import random
from transformers import Trainer, TrainingArguments, pipeline

# Load individual reviews from the given file
def load_reviews_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

# Path to sentiment data
path_to_sentiment_data = 'sentiment_data'

# Paths to positive and negative data files
positive_file_path = os.path.join(path_to_sentiment_data, 'positive', 'positive.txt')
negative_file_path = os.path.join(path_to_sentiment_data, 'negative', 'negative.txt')

# Load and label the data
positive_reviews = [(review, 1) for review in load_reviews_from_file(positive_file_path)]  # Label 1 for positive
negative_reviews = [(review, 0) for review in load_reviews_from_file(negative_file_path)]  # Label 0 for negative

# Combine the datasets
combined_dataset = positive_reviews + negative_reviews

# Shuffle the combined dataset
random.shuffle(combined_dataset)

# Split into texts and labels
texts, labels = zip(*combined_dataset)

# Use the pretrained model for sentiment
classifier = pipeline("sentiment-analysis")

# Check the model being used
model = classifier.model
config = model.config
# Print the model's name or path
print("Model Name:", classifier.model.name_or_path)  # Expected: distilbert-base-uncased-finetuned-sst-2-english
# Print the model's architecture
print("Model Architecture:", config.architectures)  # Expected: ['DistilBertForSequenceClassification']
# Print the number of labels the model can predict
print("Number of Labels:", config.num_labels)  # Expected: 2

def predict_sentiment(sentence):
    result = classifier(sentence)[0]
    label = result['label']

    # Debugging: Print the full prediction result
    print(f"Full prediction result: {result}")  # Example output: {'label': 'POSITIVE', 'score': 0.9998805522918701}
    # Debugging: Print the predicted label
    print(f"Predicted label: {label}")  # Example output: POSITIVE

    # Return the corresponding emoji based on the sentiment prediction
    if label == "NEGATIVE":  
        return "😞"  # Emoji for negative sentiment
    elif label == "POSITIVE":  
        return "😊"  # Emoji for positive sentiment

# Test cases
print('1: ' + predict_sentiment("I absolutely loved this movie, it was fantastic!"))  # Expected: 😊
print('2: ' + predict_sentiment("This was the worst movie I have ever seen."))  # Expected: 😞
result = classifier("I loved this movie, it was fantastic!")
print(f"3: {result}")  # Expected output: {'label': 'POSITIVE', 'score': 0.9998804330825806}


"""
Summarization using BART: 
- Utilizes the BART model for generating concise summaries of longer text passages.
"""
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


"""
Translation to French using mBART:
- Utilizes the mBART model for translating text from English to French.
"""
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_to_french(text):
    tokenizer.src_lang = "en_XX"
    encoded_text = translation_tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_text,
        forced_bos_token_id=translation_tokenizer.lang_code_to_id["fr_XX"]
    )
    outputs = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs[0]