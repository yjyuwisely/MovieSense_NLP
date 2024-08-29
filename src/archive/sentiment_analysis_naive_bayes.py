# Deprecated sentiment analysis using Naive Bayes
"""
Sentiment Analysis using the Naive Bayes Classifier:

- Trains on the 'sentiment_data' dataset derived from Rotten Tomato reviews.
- This implementation is a modification of the original which used the NLTK movie reviews dataset.
- The 'sentiment_data' directory contains reviews categorized as 'positive' and 'negative' 
  stored in corresponding subfolders.
"""

import os
from nltk.classify import NaiveBayesClassifier 

# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])

# Load individual reviews from the given file
def load_reviews_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()
    
# Path to sentiment data
path_to_sentiment_data = 'sentiment_data' 

# Paths to positive and negative data files
positive_file_path = os.path.join(path_to_sentiment_data, 'positive', 'positive.txt')
negative_file_path = os.path.join(path_to_sentiment_data, 'negative', 'negative.txt')

# Extract features and labels for each individual review
features_pos = [(extract_features(review.split()), 'Positive') for review in load_reviews_from_file(positive_file_path)]
features_neg = [(extract_features(review.split()), 'Negative') for review in load_reviews_from_file(negative_file_path)]
     
# Define the train (80%) and test split (20%)
threshold = 0.8
num_pos = int(threshold * len(features_pos))
num_neg = int(threshold * len(features_neg))
     
# Create training datasets
features_train = features_pos[:num_pos] + features_neg[:num_neg]
     
# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(features_train)

def predict_sentiment(sentence):
    # Compute the probabilities for each class
    probabilities = classifier.prob_classify(extract_features(sentence.split()))
    
    # Pick the maximum value
    # Return the corresponding emoji based on the sentiment prediction
    if probabilities.max() == "Positive":
        return "😊"  # Emoji for positive sentiment
    elif probabilities.max() == "Negative":
        return "😞"  # Emoji for negative sentiment
    # predicted_sentiment = probabilities.max()
    # return predicted_sentiment