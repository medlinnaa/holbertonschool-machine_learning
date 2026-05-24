# Decision Tree & Random Forest

> Implementation of a Decision Tree and Random Forest classifier from scratch using Python and NumPy.


## 📖Description

This project implements a complete machine learning classification pipeline entirely from scratch, bypassing high-level ML libraries like scikit-learn. The primary goal is to demystify the "black-box" nature of predictive models by manually constructing their underlying mathematical and programmatic foundations. By building everything from the foundational data structures up to a complete ensemble model, this project serves as a practical exploration of how algorithms evaluate data, partition feature spaces, and aggregate multiple models to make robust predictions.


## 🧠Concepts

* **Tree Traversal & Metrics:** Utilizing recursive algorithms to calculate maximum depth and differentiate internal nodes from terminal leaves.
* **Tree Visualization:** Generating clear string representations of decision boundaries and hierarchy using custom prefix formatting.
* **Feature Bounding & Indicators:** Computing dynamic upper and lower boundaries to build vectorized `lambda` indicator functions, allowing for rapid dataset evaluation without step-by-step traversal.
* **Splitting Criteria:** Calculating Gini Impurity across possible feature thresholds to identify optimal decision boundaries and maximize information gain.
* **Ensemble Learning:** Designing a Random Forest architecture that trains multiple trees with unique random seeds to reduce variance, ultimately predicting outcomes through statistical mode (majority voting).


## ⚙️Requirements

* Python 3.9
* NumPy 1.25.2
* Ubuntu 20.04
