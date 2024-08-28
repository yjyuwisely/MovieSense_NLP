"""
Text Generation using GPT-2

This function generates text based on a user-provided prompt using the GPT-2 model.
- Input: A string prompt (e.g., "Write a positive movie review").
- Output: A generated text that follows the prompt, typically in the same style or context.
- The function uses the Hugging Face Transformers library to load the pre-trained GPT-2 model and tokenizer.

This is integrated into the MovieSense_NLP project to allow users to generate custom movie reviews or related content.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
generator_model = GPT2LMHeadModel.from_pretrained(model_name)
generator_tokenizer  = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(text, max_new_tokens=50):
    inputs = generator_tokenizer.encode(text, return_tensors="pt")
    outputs = generator_model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
    generated_text = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    if generated_text.startswith(text):
        generated_text = generated_text[len(text):].strip()
    
    return generated_text

    # Keep the prompt and combine it with the generated text
    # full_output = text + "\n" + generated_text.strip()
    
    # return full_output