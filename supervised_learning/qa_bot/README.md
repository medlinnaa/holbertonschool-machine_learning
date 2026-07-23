# QA Bot

> Implementation of an interactive Question-Answering Bot using TensorFlow Hub and pre-trained NLP models.


## 📖Description

This project implements an interactive question-answering assistant that processes natural language queries and extracts precise information from a provided corpus of reference documents. 

Rather than relying on simple keyword matching, the algorithm scans a dataset of Zendesk articles to identify the most relevant document for a given question. It then isolates the exact snippet of text that serves as the answer, providing a continuous command-line interface for user interaction until gracefully terminated.


## 🧠Concepts

* Extractive Question Answering: Leveraging a fine-tuned BERT model (`bert-uncased-tf2-qa`) to predict the start and end logits of an answer within a reference text.
* Semantic Search: Utilizing the Universal Sentence Encoder to map sentences into high-dimensional vectors and computing cosine similarity to find contextual matches.
* Tokenization: Processing strings with `BertTokenizer` to generate input IDs, attention masks, and token type IDs necessary for deep learning text models.
* Interactive Loops: Building persistent, case-insensitive loops to handle dynamic user inputs and exit conditions seamlessly.


## ⚙️Requirements

* Python 3.9
* Ubuntu 20.04 LTS
* TensorFlow 2.15
* TensorFlow Hub 0.15.0
* Transformers 4.44.2
* NumPy 1.25.2
