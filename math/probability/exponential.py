#!/usr/bin/env python3
"""updating the exponential class to include PDF"""


class Exponential:
    """class that represents an exponential distribution"""

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
            # lambtha is the inverse of the mean
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """calculating the value of the PDF 
        for a given time period"""
        if x < 0:
            return 0

        e = 2.7182818285

        pdf_val = self.lambtha * (e ** (-self.lambtha * x))

        return pdf_val
