# Natural Language Processing - Evaluation Metrics

> Implementation of fundamental Natural Language Processing evaluation metrics, including N-gram and Cumulative BLEU scores.


## 📖Description

This project implements core evaluation metrics from scratch using Python and NumPy to assess the performance of Natural Language Processing (NLP) models. 

By utilizing these algorithms, the project evaluates the quality of model-generated text (such as machine translations or text summarizations) by comparing it against human-written reference texts. Instead of relying on external libraries like NLTK, this project calculates precise Bilingual Evaluation Understudy (BLEU) scores natively. It systematically analyzes generated sentences to quantify translation quality by accounting for exact word matches, continuous phrase overlaps, and necessary length penalties.


## 🧠Concepts

* Unigram Precision: Evaluating the accuracy of individual predicted words against a set of reference texts while clipping counts to prevent artificial score inflation.
* N-gram Overlap: Extracting and comparing continuous sequences of words (n-grams) to capture local grammatical correctness, word order, and phrase structure.
* Brevity Penalty: Applying a mathematical penalty to prevent models from "cheating" the precision metric by outputting overly short, highly conservative sentences.
* Cumulative Evaluation: Calculating the weighted geometric mean of multiple n-gram precisions (from unigrams up to n-grams) to provide a robust, holistic assessment of overall text quality.


## ⚙️Requirements

* Python 3.9
* NumPy 1.25.2
* Ubuntu 20.04
