#!/usr/bin/env python3
"""update to Normal class to include PDF"""


class Normal:
    """class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """initializing the distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            sum_diff_sq = 0
            for x in data:
                sum_diff_sq += (x - self.mean) ** 2
            variance = sum_diff_sq / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """calculating the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculating the x-value of a given z-score"""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """calculating the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))

        z = self.z_score(x)
        exponent = -0.5 * (z ** 2)

        result = coefficient * (e ** exponent)

        return result
