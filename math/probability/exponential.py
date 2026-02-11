#!/usr/bin/env python3
"""module to represent an exponential distribution"""


class Exponential:
    """the class that represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """initializing the distribution"""
        # Option1:no data provided
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        # Option2:data list provided
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(len(data) / sum(data))
