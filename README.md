# E-Commerce Chatbot using NLTK

This project implements an E-Commerce Chatbot using Natural Language Processing (NLP) techniques and the NLTK (Natural Language Toolkit) library in Python. The chatbot is designed to assist users in navigating an e-commerce website, providing product information, recommendations, and answering customer queries.

## Project Overview

The project follows the following processing steps:

1. Text Pre-Processing with NLTK:
   - Conversion of text to uppercase or lowercase for consistency.
   - Tokenization of text into sentences and words using NLTK's Punkt tokenizer.

2. TF-IDF (Term Frequency-Inverse Document Frequency) Approach:
   - Calculation of TF (Term Frequency) to determine how frequently a word appears in a document.
   - Calculation of IDF (Inverse Document Frequency) to measure the rarity of a word across documents.
   - TF-IDF scores are obtained by combining TF and IDF.

3. Cosine Similarity:
   - Calculation of cosine similarity to measure the similarity between two documents.
   - Cosine similarity is used to find the similarity between user input and predefined patterns.

## Files

- `Intents.json`: Data file containing predefined patterns and responses for the chatbot.
- `Words.pkl`: Pickle file containing the vocabulary list of words used in the chatbot.
- `Classes.pkl`: Pickle file containing the list of categories or intents.
- `Chatbot_model.h5`: Trained model file with weights and information about the model.

## Acknowledgments

This project is based on the NLTK library and follows the concepts of NLP, TF-IDF, and Cosine Similarity. It serves as an example of building a chatbot for e-commerce applications using Python and NLP techniques.

