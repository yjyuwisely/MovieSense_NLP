"""
Sentiment Analysis on the NLTK movie reviews dataset using the Naive Bayes Classifier.
"""
import nltk #Natural Language Toolkit
nltk.download('movie_reviews')  

from nltk.corpus import movie_reviews 
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
 
# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])
 
if __name__=='__main__': #main function 
    # Load the reviews from the corpus 
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')
     
    # Extract the features from the reviews and label 
    features_pos = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Negative') for f in fileids_neg]
     
    # Define the train and test split (80% and 20%)
    threshold = 0.8
    num_pos = int(threshold * len(features_pos))
    num_neg = int(threshold * len(features_neg))
     
    # Create training and training datasets
    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]  

    # Print the number of datapoints used
    print('\nNumber of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))
     
    # Train a Naive Bayes classifier and calculate accuracy 
    classifier = NaiveBayesClassifier.train(features_train)
    print('\nAccuracy of the classifier:', nltk_accuracy(
            classifier, features_test))

    #The top 15 words that shows important information.
    N = 15
    print('\nTop ' + str(N) + ' most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i+1) + '. ' + item[0])
        if i == N - 1:
            break

    # Define and Test input movie reviews
    input_reviews = [
        'The costumes in this movie were great', 
        'I think the story was terrible and the characters were very weak',
        'People say that the director of the movie is amazing', 
        'This is such an idiotic movie. I will not recommend it to anyone.' 
    ]

    #Predict results
    print("\nMovie review predictions:")
    for review in input_reviews:
        print("\nReview:", review)

        # Compute the probabilities for each class
        probabilities = classifier.prob_classify(extract_features(review.split()))

        # Pick the maximum value
        predicted_sentiment = probabilities.max()

        # Print outputs (Positive or Negative)
        print("Predicted sentiment:", predicted_sentiment)
        print("Probability:", round(probabilities.prob(predicted_sentiment), 2))
