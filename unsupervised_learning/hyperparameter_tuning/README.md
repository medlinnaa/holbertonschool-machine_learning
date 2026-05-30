# Hyperparameter Tuning

> Implementation of Gaussian Processes and Bayesian Optimization from scratch using NumPy.


## 📖Description

This project features a complete hyperparameter optimization pipeline built entirely from scratch. 

Rather than relying on brute-force grid searches or random parameter guessing, this algorithm intelligently searches for the global optimum of an expensive, black-box function. It maintains a probabilistic surrogate model of the objective function and continuously updates its beliefs as new data points are sampled. By mathematically balancing the search between unknown regions and known high-performing regions, it minimizes the total number of iterations required to find the optimal solution.


## 🧠Concepts

* Gaussian Processes: Constructing noiseless 1D surrogate models to predict the mean and variance of unsampled points across a continuous space.
* Covariance Kernels: Utilizing the Radial Basis Function (RBF) to calculate the spatial correlation between different input features without explicit distance loops.
* Acquisition Functions: Implementing Expected Improvement (EI) to mathematically evaluate the next best sample location by balancing exploration (areas of high uncertainty) and exploitation (areas with known optimal values).
* Predictive Updating: Recalculating matrix inverses and standard deviations to dynamically update the model's confidence intervals after each new observation.


## ⚙️Requirements

* Python 3.9
* NumPy 1.25.2
* SciPy (for `scipy.stats.norm`)
* Ubuntu 20.04 LTS
* pycodestyle 2.11.1
