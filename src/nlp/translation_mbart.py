"""
Translation to French using mBART:
- Utilizes the mBART model for translating text from English to French.
"""
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_to_french(text):
    translation_tokenizer.src_lang = "en_XX"
    encoded_text = translation_tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_text,
        forced_bos_token_id=translation_tokenizer.lang_code_to_id["fr_XX"]
    )
    outputs = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs[0]