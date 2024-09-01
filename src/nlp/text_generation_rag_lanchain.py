'''
Text Generation using Retrieval-Augmented Generation (RAG)

This script:
- Implements a text generation pipeline using the RAG model.
- Retrieves relevant documents based on a query.
- Generates context-aware text (e.g., movie reviews) using a language model guided by the retrieved information.

Key Components:
- Libraries: langchain, openai, unstructured, sentence-transformers, chromadb
- Language Model: GPT-3.5-turbo
- Embedding Model: all-MiniLM-L6-v2
- Vector Database: Chroma for similarity search and storage of embeddings

Note:
- This script requires an OpenAI API key to access the GPT-3.5-turbo model for text generation.
- The API key should be stored in a .env file in the root directory of the project as `OPENAI_API_KEY`.
- Usage of the OpenAI API may incur costs depending on the usage tier.
'''

# Loading Documents
from langchain_community.document_loaders import TextLoader

# 1. Load positive and negative movie reviews
import os
# Correct path to the sentiment data folder
# path_to_sentiment_data = os.path.abspath(os.path.join('..', '..', 'data', 'sentiment_data'))
# path_to_sentiment_data = os.path.join('..', '..', 'data', 'sentiment_data')

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for the sentiment data
path_to_sentiment_data = os.path.join(current_dir, '..', '..', 'data', 'sentiment_data')

# Correct paths to positive and negative data files
positive_file_path = os.path.join(path_to_sentiment_data, 'positive', 'positive.txt')
negative_file_path = os.path.join(path_to_sentiment_data, 'negative', 'negative.txt')

# Load positive and negative movie reviews
positive_reviews = TextLoader(positive_file_path).load()
negative_reviews = TextLoader(negative_file_path).load()

# Combine positive and negative documents into a single list
documents = positive_reviews + negative_reviews

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# 2. Split the combined documents into smaller chunks
docs = split_docs(documents)

# Creating Embeddings and a Vector Store
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Create embeddings for documents
# Use HuggingFaceEmbeddings for SentenceTransformer models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

# Store embeddings in a vector database
db = Chroma.from_documents(docs, embeddings)

from dotenv import load_dotenv
import os

# Loads the environment variables from the .env file
load_dotenv()  

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

from langchain.chat_models import ChatOpenAI

# 4. Initialize the language model
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)

# Combining with Language Model for Generation
from langchain.chains.question_answering import load_qa_chain

# 5. Create a chain for question-answering with the language model
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# The following block of code demonstrates how to retrieve relevant documents based on a query
# and generate text using the RAG approach with GPT-3.5-turbo. This code is currently active 
# and outputs the generated text in the terminal:

# Retrieving Relevant Documents
# Example prompt
query = "Write a positive review about a movie."
#query = "Write a negative review about a movie."

# 6. Perform similarity search to find the most relevant documents
print("Starting similarity search...")
matching_docs = db.similarity_search(query)
print("Similarity search completed. Generating answer...")

# 7. Generate a review using the retrieved documents
answer = chain.run(input_documents=matching_docs, question=query)
print("Answer generated.")
#answer
print(answer)

# However, if you comment out the above block of code and uncomment the `generate_text` function below, 
# users will be able to see the generated output directly in the text box on the webpage instead of the terminal.
# This is useful for integrating the text generation feature into the web interface of the MovieSense application.

# Uncomment the following function to enable text generation in the web interface:

# def generate_text(query):
#     """
#     Generates text based on a query using the RAG model.
    
#     Args:
#         query (str): The prompt or query for generating text.
    
#     Returns:
#         str: The generated text.
#     """
#     # Perform similarity search to find the most relevant documents
#     print("Starting similarity search...")
#     matching_docs = db.similarity_search(query)
#     print("Similarity search completed. Generating answer...")

#     # Generate a review using the retrieved documents
#     answer = chain.run(input_documents=matching_docs, question=query)
#     print("Answer generated.")

#     return answer

# Make sure to adjust your code accordingly based on how you want to display the generated text.
# For web-based output, the function `generate_text` should be active, while the terminal-based 
# output should be commented out.