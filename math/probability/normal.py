#!/usr/bin/env python3
"""module to represent a normal distribution"""


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
