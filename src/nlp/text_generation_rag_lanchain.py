'''
Text Generation using Retrieval-Augmented Generation (RAG)

This script implements a text generation pipeline using the RAG model.
It retrieves relevant documents based on a query and generates text
using a language model guided by the retrieved information.

Key Components:
- Libraries: langchain, openai, unstructured, sentence-transformers, chromadb
- Language Model: GPT-3.5-turbo
- Embedding Model: all-MiniLM-L6-v2
- Vector Database: Chroma for similarity search and storage of embeddings
- Task: Generate context-aware text (e.g., movie reviews) based on retrieved documents
'''

#from xml.dom.minidom import Document

# Loading Documents
#from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

# #documents = TextLoader("e:/data/AI.txt").load()

# # 1. Load positive and negative movie reviews
# positive_reviews = TextLoader("data/sentiment_data/positive/positive.txt").load()
# negative_reviews = TextLoader("data/sentiment_data/negative/negative.txt").load()

import os
# Correct path to the sentiment data folder
path_to_sentiment_data = os.path.join('..', '..', 'data', 'sentiment_data')

# Correct paths to positive and negative data files
positive_file_path = os.path.join(path_to_sentiment_data, 'positive', 'positive.txt')
negative_file_path = os.path.join(path_to_sentiment_data, 'negative', 'negative.txt')

# Load positive and negative movie reviews
positive_reviews = TextLoader(positive_file_path).load()
negative_reviews = TextLoader(negative_file_path).load()

# Combine positive and negative documents into a single list
documents = positive_reviews + negative_reviews

# print("Positive file path:", positive_file_path)
# print("Negative file path:", negative_file_path)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# 2. Split the combined documents into smaller chunks
docs = split_docs(documents)

# Creating Embeddings and a Vector Store
#from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_community.embeddings import SentenceTransformerEmbeddings

# # 3. Create embeddings for documents
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_huggingface import HuggingFaceEmbeddings

#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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
# openai_api_key = os.getenv("OPENAI_API_KEY")

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Use this code in your terminal: set OPENAI_API_KEY "your-api-key"
# Set your OpenAI API key
#os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

from langchain.chat_models import ChatOpenAI

# 4. Initialize the language model
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)

# Combining with Language Model for Generation
from langchain.chains.question_answering import load_qa_chain

# 5. Create a chain for question-answering with the language model
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Retrieving Relevant Documents
# Example prompt
query = "Write a positive review about a movie."

# 6. Perform similarity search to find the most relevant documents
print("Starting similarity search...")
matching_docs = db.similarity_search(query)
print("Similarity search completed. Generating answer...")

# 7. Generate a review using the retrieved documents
answer = chain.run(input_documents=matching_docs, question=query)
print("Answer generated.")
#answer
print(answer)