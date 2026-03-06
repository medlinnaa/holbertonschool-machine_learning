import numpy as np


class Neuron:
    """Class that defines a single neuron"""

    def __init__(self, nx):
        """Initializes the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W"""
        return self.__W

    @property
    def b(self):
        """Getter for b"""
        return self.__b

    @property
    def A(self):
        """Getter for A"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X: numpy.ndarray with shape (nx, m)
        Returns: private attribute __A
        """
        # 1. Calculate Z (the weighted sum)
        Z = np.matmul(self.__W, X) + self.__b

        # 2. Apply Sigmoid Activation Function
        # Sigmoid formula: 1 / (1 + e^-z)
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: numpy.ndarray with shape (1, m) (correct labels)
        A: numpy.ndarray with shape (1, m) (activated outputs)
        Returns: the cost
        """
        m = Y.shape[1]

        # Logistic Regression Cost formula
        # We use 1.0000001 - A to avoid log(0) which is undefined
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        # The cost is the average (mean) of all individual losses
        cost = (1 / m) * np.sum(loss)

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        X: numpy.ndarray with shape (nx, m) (input data)
        Y: numpy.ndarray with shape (1, m) (correct labels)
        Returns: the prediction (A as integers) and the cost
        """
        # 1. Get the probabilities using the method we already wrote
        A = self.forward_prop(X)

        # 2. Get the cost using the method we already wrote
        cost = self.cost(Y, A)

        # 3. Convert probabilities to 0 or 1
        # np.where(condition, value_if_true, value_if_false)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X: numpy.ndarray (nx, m) containing input data
        Y: numpy.ndarray (1, m) containing correct labels
        A: numpy.ndarray (1, m) containing the activated output
        alpha: the learning rate
        Updates the private attributes __W and __b
        """
        m = Y.shape[1]

        # 1. Calculate the error (difference between prediction and reality)
        dz = A - Y

        # 2. Calculate the gradients (slopes)
        # We use np.matmul for the matrix multiplication of dz and X transposed
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        # 3. Update the private attributes
        # We move in the opposite direction of the gradient to reduce cost
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
