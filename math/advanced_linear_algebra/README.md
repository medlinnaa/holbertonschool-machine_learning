# Advanced Linear Algebra

> Algorithmic implementation of matrix transformations, inversions, and definiteness classification.


## 📖Description

This project builds a sequential toolkit for complex matrix operations in Python. To deeply understand the mechanics of linear algebra, the majority of the functions in this repository are written entirely from scratch using nested lists, deliberately avoiding heavy dependencies like NumPy for the foundational tasks.

The codebase is structured so that each calculation serves as a building block for the next. The recursive determinant algorithm is utilized to compute the minor matrix, which then feeds into the cofactor matrix, ultimately allowing for the calculation of the adjugate and the full inverse of a matrix. The final module introduces `numpy` strictly to evaluate the eigenvalues of symmetric matrices to classify their definiteness.


## 🧠Concepts

* **Laplace Expansion:** A recursive mathematical method used to calculate the determinant of a square matrix by breaking it down into smaller sub-matrices.
* **Minors and Cofactors:** The minor is the determinant of the sub-matrix left after removing the $i$-th row and $j$-th column. The cofactor applies a positional sign change using the formula $(-1)^{i+j}$.
* **Adjugate Matrix:** The transpose of the cofactor matrix.
* **Matrix Inversion:** A matrix multiplied by its inverse yields the identity matrix. It is calculated by scaling the adjugate matrix by the reciprocal of the original determinant:
  $$A^{-1} = \frac{1}{|A|} \text{adj}(A)$$
* **Definiteness:** A classification of a symmetric matrix based on the signs of its eigenvalues (the roots of its characteristic equation). For example, if all eigenvalues are strictly greater than zero, the matrix is positive definite.


## ⚙️Requirements

* Python 3.x
* NumPy (for definiteness evaluation)
* Ubuntu 20.04 LTS
* Matrices are represented as 2D lists (lists of lists) for tasks 0-5, and as `numpy.ndarray` objects for task 6.
