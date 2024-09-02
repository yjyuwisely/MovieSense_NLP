from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Smoothing function for BLEU score calculation
smoothing_function = SmoothingFunction().method4

# Tokenize the reference texts
reference_positive_review = [
    word_tokenize("The film was an exhilarating journey from beginning to end. Not only was the plot engaging, but the characters were also crafted with such depth and nuance that you couldn't help but root for them. The cinematography painted a visual tapestry that was nothing short of breathtaking, drawing the audience into each scene. The soundtrack, with its sublime melodies, further elevated the movie-watching experience. All in all, it was a cinematic gem that I believe will be remembered for years to come.")
]

reference_negative_review = [
    word_tokenize("I walked into the theater with high expectations, but sadly, the film didn't meet any of them. "
                  "The story was disjointed, hopping from one plot point to another without any coherence. "
                  "The characters, rather than being relatable, came off as caricatures, lacking any real depth. "
                  "The dialogue felt forced and unnatural, making many scenes cringe-worthy. "
                  "To top it off, the editing was sloppy, with several continuity errors that were hard to ignore. "
                  "It was a disappointing experience to say the least.")
]

# Generated texts from GPT-2
generated_positive_review_gpt2 = "Review a movie. Review a movie. Review a movie. Review a movie. Review a movie. Review a movie. Review a movie. Review a movie.".split()
generated_negative_review_gpt2 = "I'm not sure if I'm going to be able to do that. I'm not sure if I'm going to be able to do that. I'm not sure if I'm going to be able to do that. I'm not".split()

# Generated texts from GPT-3.5-turbo
# Combine all generated sentences into a single text
generated_positive_review_gpt3 = (
    "An incredible film that captivates from start to finish. "
    "The storyline is compelling, and the character development is superb. "
    "The cinematography is stunning, painting a visual tapestry that draws the audience into every scene. "
    "The soundtrack adds another layer of beauty to the experience, perfectly complementing the narrative. "
    "It's a masterpiece that will be cherished for generations, showcasing the director's unique vision and the actors' remarkable performances. "
    "A must-watch for anyone who appreciates cinematic art."
)

generated_negative_review_gpt3 = (
    "The movie is a loud, ugly, irritating experience where none of its satirical salvos hit a discernible target. "
    "It's nothing more than a paint-by-numbers picture — predictable and lacking any originality, and it is difficult to consider it a work of art. "
    "While it features a brilliant performance by Nicholson, the film itself is an agonizing bore except for the rare moments when the fantastic Kathy Bates turns up. "
    "Unfortunately, even Liotta's character is put in an impossible spot, with his deceptions ultimately undoing both his character and the believability of the entire scenario. "
    "A long-winded and stagy session of romantic contrivances that fails to gel into the shrewd feminist fairy tale it aspired to be, the film is simply lame and unnecessary. "
    "It's two-fifths of a satisfying movie experience at best, leaving the audience with only so much baked cardboard to chew through. "
    "In summary, it is a film that tries the patience of even the most dedicated movie critic."
)

# Tokenize generated reviews
generated_positive_tokens_gpt3 = word_tokenize(generated_positive_review_gpt3)
generated_negative_tokens_gpt3 = word_tokenize(generated_negative_review_gpt3)

# Calculate BLEU scores for GPT-2 generated texts
bleu_positive_gpt2 = sentence_bleu(reference_positive_review, generated_positive_review_gpt2, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
bleu_negative_gpt2 = sentence_bleu(reference_negative_review, generated_negative_review_gpt2, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

# Calculate BLEU scores for GPT-3.5-turbo generated texts
bleu_positive_gpt3 = sentence_bleu(reference_positive_review, generated_positive_tokens_gpt3, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
bleu_negative_gpt3 = sentence_bleu(reference_negative_review, generated_negative_tokens_gpt3, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

# Print BLEU scores
print(f"BLEU score for GPT-2 positive review: {bleu_positive_gpt2:.2f}")
print(f"BLEU score for GPT-2 negative review: {bleu_negative_gpt2:.2f}")
print(f"BLEU score for GPT-3.5-turbo generated positive review: {bleu_positive_gpt3:.2f}")
print(f"BLEU score for GPT-3.5-turbo generated negative review: {bleu_negative_gpt3:.2f}")