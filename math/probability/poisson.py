#!/usr/bin/env python3
"""a module to represent a Poisson distribution"""


class Poisson:
    """here is the class that contains a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """initializing the distribution"""

        # option1: data is not provided, use the lambtha parameter
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        
        # option2: data is provided, estimate lambtha from the list
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            self.lambtha = float(sum(data) / len(data))
