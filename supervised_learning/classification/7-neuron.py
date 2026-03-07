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
        Trains the neuron with conditional logging and graphing
        """
        # ... [Keep your validation for iterations and alpha here] ...

        # Check step only if logging or graphing is active
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iter_list = []

        # The loop runs 'iterations' times
        for i in range(iterations + 1):
            # Calculate prediction
            if i < iterations:
                self.forward_prop(X)

            # Logging logic: Only happens if verbose/graph is True
            # This is what the 'test0.py' file is checking!
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    iter_list.append(i)

            # Update weights (except on the last cycle)
            if i < iterations:
                self.gradient_descent(X, Y, self.A, alpha)

        # Plotting logic: Only happens if graph is True
        if graph:
            plt.plot(iter_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Always return the final evaluation
        return self.evaluate(X, Y)
