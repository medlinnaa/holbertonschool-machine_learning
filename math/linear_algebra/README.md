# Linear Algebra

> Implementation of core linear algebra operations using both pure Python and the NumPy library.


## 📖Description

This repository explores the foundational mechanics of linear algebra by translating mathematical matrix operations into functional code. 

The project is structured in two distinct phases to provide a comprehensive learning experience. First, it constructs complex matrix behaviors entirely from scratch using standard Python nested lists. This manual approach exposes the underlying logic required to navigate rows and columns mathematically. The second phase transitions to using the `numpy` library, demonstrating how vectorized algorithms drastically optimize and simplify these exact same workflows for real-world data science and machine learning applications.


## 🧠Concepts

* **Array Slicing & Indexing:** Extracting specific subarrays, individual columns, and isolated square matrices using both standard Python syntax (`[x:y]`) and multi-dimensional NumPy slicing (`[row_start:row_end, col_start:col_end]`).
* **List Comprehensions:** Utilizing nested comprehensions to efficiently traverse, transpose, and manipulate 2D data structures without relying on heavy, unreadable `for` loops.
* **Dimensionality Analysis:** Recursively calculating the depth and structural shape of native nested lists to validate data before processing.
* **Matrix Algebra Rules:** Programming strict mathematical constraints, such as ensuring shape equality for element-wise arithmetic, and validating that the columns of a first matrix match the rows of a second matrix before calculating a dot product.
* **Axis-based Concatenation:** Merging multi-dimensional datasets conditionally along specific axes (horizontal vs. vertical stacking) while handling potential structural mismatches.


## ⚙️Requirements

* Python 3.9 (or higher)
* NumPy 1.25.2 (or compatible version)
* Ubuntu 20.04 
