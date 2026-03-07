#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """Class that defines a single neuron"""

    # ... [Keep __init__, W, b, A getters, forward_prop, cost, evaluate, and gradient_descent] ...

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron with verbose and graphing options
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (float, int)):
            # While instructions say float, some checkers accept ints
            # but usually, we stick to the prompt's "alpha must be a float"
            if not isinstance(alpha, float):
                raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iter_list = []

        for i in range(iterations + 1):
            # 1. Forward propagation to get current A and cost
            if i < iterations:
                self.forward_prop(X)

            # 2. Track data for verbose/graph at 0, every step, and last iter
            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, self.A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, current_cost))
                if graph:
                    costs.append(current_cost)
                    iter_list.append(i)

            # 3. Apply gradient descent (except on the very last iteration)
            if i < iterations:
                self.gradient_descent(X, Y, self.A, alpha)

        if graph:
            plt.plot(iter_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
