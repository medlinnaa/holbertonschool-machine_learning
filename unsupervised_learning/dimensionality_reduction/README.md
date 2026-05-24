# Dimensionality Reduction

> Implementation of Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) from scratch using NumPy.


## 📖Description

This project implements fundamental dimensionality reduction algorithms entirely from scratch. High-dimensional datasets, such as images with thousands of pixels, are often too complex to visualize or process efficiently. 

To solve this, the pipeline compresses massive datasets into readable 2D or 3D spaces. Instead of simply dropping variables, these algorithms analyze the mathematical relationships between data points to preserve their inherent structure. Whether identifying the axes of greatest variance or calculating probabilistic similarities, the goal is to ensure that data points that are related in the complex, high-dimensional space remain clustered together in the simplified visualization.


## 🧠Concepts

* **Principal Component Analysis (PCA):** Utilizing Singular Value Decomposition (SVD) on mean-centered datasets to extract orthogonal feature vectors that maximize variance preservation.
* **Probability Affinities:** Calculating symmetric pairwise Euclidean distances to construct high-dimensional ($P$) and low-dimensional ($Q$) similarity matrices using highly vectorized operations.
* **Shannon Entropy & Perplexity:** Employing binary search algorithms to dynamically scale the standard deviation (inversely represented by $\beta$) of the local Gaussian distribution for each individual data point.
* **Kullback-Leibler (KL) Divergence:** Formulating a numerically stable cost function to evaluate the structural discrepancy between the high-dimensional and low-dimensional data spaces.
* **Gradient Descent Optimization:** Iteratively updating the low-dimensional map ($Y$) utilizing vectorized gradients, momentum tracking, and early exaggeration to isolate data clusters and avoid local minima.


## ⚙️Requirements

* Python 3.9
* NumPy 1.25.2
* Matplotlib (for final plot visualization)
* Ubuntu 20.04 LTS
* pycodestyle 2.11.1
