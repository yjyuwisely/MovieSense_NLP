# Deprecated translation using Helsinki-NLP
"""
Translation to French using Helsinki-NLP:
- Utilizes the Helsinki-NLP model for translating text from English to French.
"""
from transformers import pipeline
def translate_to_french(text): 
    translator = pipeline("translation_en_to_fr", model="helsinki-nlp/opus-mt-en-fr")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
    return outputs[0]['translation_text']