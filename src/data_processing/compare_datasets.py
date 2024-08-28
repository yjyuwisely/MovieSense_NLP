"""
Check the equivalence between the NLTK movie_reviews dataset and the sentiment_data from Rotten Tomato.
"""
from nltk.corpus import movie_reviews

# The first five sentences from sentiment_data's negative 
sentences = [
    "simplistic silly and tedious",
    "its so laddish and juvenile only teenage boys could possibly find it funny",
    "exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable",
    "garbus discards the potential for pathological study exhuming instead the skewed melodrama of the circumstantial situation",
    "a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification"
]

# Collect all the words from negative movie reviews from the nltk dataset
negative_reviews = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('neg')]

# Check if provided sentences appear in the dataset
for sentence in sentences:
    if sentence in negative_reviews:
        print(f"'{sentence}' is in the nltk movie_reviews dataset.")
    else:
        print(f"'{sentence}' is NOT in the nltk movie_reviews dataset.")

# Result: They are not the same.
# 'simplistic silly and tedious' is NOT in the nltk movie_reviews dataset.
# 'its so laddish and juvenile only teenage boys could possibly find it funny' is NOT in the nltk movie_reviews dataset.
# 'exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable' is NOT in the nltk movie_reviews dataset.
# 'garbus discards the potential for pathological study exhuming instead the skewed melodrama of the circumstantial situation' is NOT in the nltk movie_reviews dataset.
# 'a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification' is NOT in the nltk movie_reviews dataset.