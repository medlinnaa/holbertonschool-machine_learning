# Bayesian Probability

> An implementation of Bayes' Theorem for statistical inference using Python and NumPy.


## 📖Description

This project provides a programmatic pipeline for computing Bayesian probabilities from scratch. By breaking down Bayes' theorem into individual functional components, the codebase calculates how probabilities are updated as new evidence is introduced. 

The scripts utilize `numpy` to handle 1D arrays representing multiple hypothetical probabilities simultaneously. Through robust input validation and vectorized operations, the tools compute the components necessary to perform statistical inference on binomial trial data—processing the number of successes ($x$) out of a given number of trials ($n$) against various prior beliefs. 


## 🧠Concepts

The underlying mathematical principles applied in this project are the fundamental components of Bayesian statistics:

* **Likelihood:** The probability of observing the specific data given a certain hypothesis. It is calculated using the Binomial distribution probability mass function:
  $$P(X | \theta) = \frac{n!}{x!(n - x)!} \theta^x (1 - \theta)^{n-x}$$
* **Intersection (Joint Probability):** The probability of both the hypothesis and the observed data occurring together. It is the product of the likelihood and the prior probability:
  $$P(X \cap \theta) = P(X | \theta) \times P(\theta)$$
* **Marginal Probability (Evidence):** The overall probability of observing the data under all possible, mutually exclusive hypotheses. It acts as a normalizing constant:
  $$P(X) = \sum P(X | \theta_i) \times P(\theta_i)$$
* **Posterior Probability:** The updated probability of the hypothesis after the data has been observed. It is derived by dividing the intersection by the marginal probability (Bayes' rule):
  $$P(\theta | X) = \frac{P(X \cap \theta)}{P(X)}$$


## ⚙️Requirements

* Python 3.x
* NumPy 1.15+
* Ubuntu 20.04 LTS
* All functions include comprehensive type and value checking to ensure valid probability ranges $[0, 1]$ and valid integer trial inputs.
