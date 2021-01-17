from typing import Callable
import numpy as np

class Function:
    """
    Class that implement some activate functions (method with name 'function')
    and their derivations (name of method - 'function_d')

    Attributes
    ----------
    function_name: str
        name of the function
    function: Callable[[float], float]
        function that muches the passed name of function
    function_d: Callable[[float], float]
        derivative of the class function

    """


    def __init__(self, function_name):
        self.function_name = function_name
        if function_name == "sigmoid":
            self.function = np.vectorize(self.sigmoid)
            self.function_d = np.vectorize(self.sigmoid_d)
        elif function_name == "tanh":
            self.function = np.vectorize(self.tanh)
            self.function_d = np.vectorize(self.tanh_d)
        elif function_name == "relu":
            self.function = np.vectorize(self.relu)
            self.function_d = np.vectorize(self.relu_d)
        elif function_name == "softplus":
            self.function = np.vectorize(self.softplus)
            self.function_d = np.vectorize(self.softplus_d)
        else:
            raise NameError(f"Function '{function_name}' isn't supported. Supported functions: 'sigmoid', 'tanh', 'relu', 'softplus'")


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_d(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    def tanh_d(self, x):
        return 1 - self.tahn(x)


    def relu(self, x):
        return max(0, x)


    def relu_d(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        return None


    def softplus(self, x):
        return np.log(1 + np.exp(x))


    def softplus_d(self, x):
        return 1 / (1 + np.exp(x))
