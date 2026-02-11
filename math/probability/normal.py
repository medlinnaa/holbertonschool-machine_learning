#!/usr/bin/env python3
"""changing  Normal class to include CDF"""


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
        """finding the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def cdf(self, x):
        """finding the value of the CDF for a given x-value"""
        pi = 3.1415926536

        val = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = (2 / (pi ** 0.5)) * (
            val -
            (val ** 3) / 3 +
            (val ** 5) / 10 -
            (val ** 7) / 42 +
            (val ** 9) / 216
        )

        return 0.5 * (1 + erf)
