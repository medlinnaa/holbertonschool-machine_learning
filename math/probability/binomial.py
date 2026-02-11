#!/usr/bin/env python3
"""update Binomial distribution class with PMF"""


class Binomial:
    """represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """class constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_initial = 1 - (variance / mean)
            self.n = int(round(mean / p_initial))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """finding the value of the PMF for a given number of successes"""
        # Ensure k is an integer
        k = int(k)

        # k must be within the range [0, n]
        if k < 0 or k > self.n:
            return 0

        # Calculate factorial for binomial coefficient: n! / (k!(n-k)!)
        def factorial(num):
            res = 1
            for i in range(1, num + 1):
                res *= i
            return res

        n_fact = factorial(self.n)
        k_fact = factorial(k)
        nk_fact = factorial(self.n - k)

        # Binomial coefficient (n choose k)
        combination = n_fact / (k_fact * nk_fact)

        # PMF = (nCk) * (p^k) * ((1-p)^(n-k))
        pmf_val = combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_val
