#!/usr/bin/env python3
"""4-frequency.py: Plot histogram of student grades for Project A."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Display frequency histogram of student grades."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 101, 10)  # bins every 10 units

    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.show()
