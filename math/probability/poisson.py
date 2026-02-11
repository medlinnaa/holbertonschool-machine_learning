#!/usr/bin/env python3
"""updating the Poisson class to include CDF"""


class Poisson:
    """defining class that represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """initializing the distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """finding the value of the PMF for a given number of 'successes'"""
        k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (e ** -self.lambtha) * (self.lambtha ** k) / factorial

    def cdf(self, k):
        """finding the value of the CDF for a given number of 'successes'"""
        k = int(k)

        if k < 0:
            return 0

        total_prob = 0
        for i in range(k + 1):
            total_prob += self.pmf(i)

        return total_prob
