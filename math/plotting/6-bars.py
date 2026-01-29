#!/usr/bin/env python3
"""6-bars.py: Plot a stacked bar graph of fruit consumption."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar graph with the following requirements:
    - Data: Randomly generated 4x3 matrix (4 fruits, 3 people).
    - People (X-axis): Farrah, Fred, Felicia.
    - Fruits (Stacked): Apples (red), Bananas (yellow),
                        Oranges (#ff8000), Peaches (#ffe5b4).
    - Dimensions: Bar width 0.5, Y-axis 0-80 (ticks every 10).
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    plt.figure(figsize=(6.4, 4.8))

    bottom_heights = np.zeros(len(people))

    for row in range(len(fruit)):
        plt.bar(
            people,
            fruit[row],
            width=0.5,
            bottom=bottom_heights,
            color=colors[row],
            label=fruit_names[row]
        )
        bottom_heights += fruit[row]

    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    plt.show()
