"""
Summarization using BART: 
- Utilizes the BART model for generating concise summaries of longer text passages.
"""
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
summary_model = BartForConditionalGeneration.from_pretrained(model_name)
summary_tokenizer = BartTokenizer.from_pretrained(model_name)

def generate_summary(text):
    inputs = summary_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=100, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)