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

# Corrected path to sentiment data folder
path_to_sentiment_data = os.path.join('..', 'data', 'sentiment_data')
# # Path to sentiment data
# path_to_sentiment_data = 'sentiment_data'

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