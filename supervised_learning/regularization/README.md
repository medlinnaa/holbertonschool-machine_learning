# Neural Network Regularization Techniques

This project explores **regularization methods** in neural networks to prevent overfitting and improve generalization, using **NumPy** and **TensorFlow**. The tasks demonstrate both theoretical concepts and practical implementations for building more robust models.

---

## Implemented Features

### 1. L2 Regularization
L2 regularization adds a penalty to large weights, helping the model avoid overfitting by keeping weights small.  
- **Functions:**  
  - `l2_reg_cost(cost, model)` — computes the cost including L2 penalty.  
  - `l2_reg_create_layer(prev, n, activation, lambtha)` — creates a neural network layer with L2 regularization.

### 2. Dropout
Dropout randomly disables neurons during training, reducing reliance on any single neuron and improving model generalization.  
- **Functions:**  
  - `dropout_forward_prop(X, weights, L, keep_prob)` — forward propagation with dropout masks.  
  - `dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L)` — updates weights with gradient descent under dropout.  
  - `dropout_create_layer(prev, n, activation, keep_prob, training=True)` — creates a neural network layer with dropout applied.

### 3. Early Stopping
Early stopping halts training when the validation loss does not improve beyond a defined threshold for a set patience period, preventing unnecessary training and overfitting.  
- **Function:**  
  - `early_stopping(cost, opt_cost, threshold, patience, count)` — determines if training should stop early based on validation cost trends.

---

## Project Overview

This repository is organized to showcase a step-by-step exploration of neural network regularization. Each task focuses on a specific technique:

1. **L2 Regularization** — penalizes large weights to control model complexity.  
2. **Dropout** — introduces stochasticity during training for better generalization.  
3. **Early Stopping** — stops training at the optimal point to prevent overfitting.  

The project combines **manual NumPy implementations** with **TensorFlow layer constructions**, demonstrating both foundational and practical aspects of neural network regularization.

---

## Notes

- Random seeds are set for reproducibility across experiments.  
- The project covers both theory and hands-on coding for regularization in supervised learning models.  
- All layers except the final one typically use the `tanh` activation; the final layer uses `softmax` for classification.
