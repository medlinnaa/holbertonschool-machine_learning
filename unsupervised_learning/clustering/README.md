# Clustering

> Implementation of Unsupervised Learning clustering algorithms from scratch and using scikit-learn.


## 📖Description

This project explores unsupervised machine learning techniques designed to uncover hidden structures within unlabeled datasets. By analyzing the inherent geometry and statistical distribution of data points, these algorithms automatically group similar items together. The pipeline progresses from hard-clustering approaches—where each point is strictly assigned to a single distinct group—to soft-clustering probabilistic models, and finally explores bottom-up hierarchical tree structures. The primary goal is to provide a robust toolkit for data segmentation, anomaly detection, and optimal model selection without relying on predefined labels.


## 🧠Concepts

* **K-Means**: A geometric algorithm that partitions data by iteratively minimizing intra-cluster variance and updating centroid coordinates based on Euclidean distances until convergence.
* **Gaussian Mixture Models (GMM)**: A probabilistic approach utilizing the Expectation-Maximization (EM) algorithm to represent clusters as multivariate normal distributions, calculating the fractional likelihood of points belonging to overlapping spaces.
* **Bayesian Information Criterion (BIC)**: An evaluation metric used to prevent overfitting by programmatically balancing model complexity (the number of parameters) against performance (the log-likelihood).
* **Agglomerative Clustering**: A hierarchical method that merges points from the bottom up utilizing Ward's linkage, relying on cophenetic distances to construct visual dendrograms for dynamic cluster boundary extraction.


## ⚙️Requirements

* Ubuntu 20.04 LTS
* Python 3.9
* NumPy (version 1.25.2)
* SciPy (version 1.11.4)
* Scikit-Learn (version 1.5.0)
* Matplotlib (for visualizations)
