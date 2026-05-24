# Probability Distributions

> Implementation of foundational statistical distributions (Binomial, Exponential, Normal, and Poisson) from scratch in Python.


## 📖Description

This project provides a custom, lightweight Python library for modeling core probability distributions. 

Instead of relying on heavy external mathematical libraries like SciPy or NumPy, this implementation calculates probabilities and statistical parameters directly using pure Python. The suite allows users to initialize statistical models using either raw datasets or explicit parameters. It serves as both a functional toolkit for basic statistical analysis and an educational breakdown of the underlying mathematics driving predictive modeling and probability theory.


## 🧠Concepts

* **Parameter Estimation:** Dynamically calculating vital metrics—such as mean, variance, and standard deviation—directly from raw data lists to initialize distribution parameters.
* **Binomial Distribution:** Modeling discrete probabilities for a fixed number of independent trials. It includes custom factorial calculations to evaluate the Probability Mass Function (PMF) and aggregates these for the Cumulative Distribution Function (CDF).
* **Exponential Distribution:** Modeling the time between events in a Poisson point process, utilizing hardcoded approximations of Euler's number to determine the Probability Density Function (PDF) and CDF.
* **Normal Distribution:** Representing the classic bell curve, featuring a custom Z-score calculator and a mathematical approximation of the error function (erf) to compute the CDF without external dependencies.
* **Poisson Distribution:** Evaluating the probability of a given number of events occurring in a fixed interval, computing the PMF and iteratively accumulating it to construct the CDF.


## ⚙️Requirements

* Python 3.9 (or higher)
* No external libraries (Strictly standard Python arithmetic operations)
* Ubuntu 20.04 (or standard OS environment)
