# sentiment_analysis.py
import nltk
from nltk.corpus import movie_reviews  
from nltk.classify import NaiveBayesClassifier 

# Make sure nltk resources are downloaded
nltk.download('movie_reviews', quiet=True)

# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])

# Load the reviews from the corpus 
fileids_pos = movie_reviews.fileids('pos')
fileids_neg = movie_reviews.fileids('neg')
     
# Extract the features from the reviews and label 
features_pos = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in fileids_pos]
features_neg = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in fileids_neg]
     
# Define the train and test split (80% and 20%)
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
    predicted_sentiment = probabilities.max()

    return predicted_sentiment