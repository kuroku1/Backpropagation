from typing import Callable
import numpy as np

class ActivationFunction:
    """
    Class that implements some activate functions (method with name 'calculate')
    and their derivations (name of method - 'calculate_d')

    Attributes
    ----------
    function_name: str
        name of the function
    calculate: Callable[[float], float]
        function that muches the passed name of function
    calculate_d: Callable[[float], float]
        derivative of the class function
    """


    def __init__(self, function_name):
        self.function_name = function_name
        if function_name == "sigmoid":
            self.calculate = self.sigmoid
            self.calculate_d = self.sigmoid_d
        elif function_name == "tanh":
            self.calculate = self.tanh
            self.calculate_d = self.tanh_d
        elif function_name == "relu":
            self.calculate = self.relu
            self.calculate_d = self.relu_d
        elif function_name == "softplus":
            self.calculate = self.softplus
            self.calculate_d = self.softplus_d
        elif function_name == "linear":
            self.calculate = self.linear
            # ???
            self.calculate_d = self.linear_d
        else:
            raise NameError(f"Function '{function_name}' isn't supported. Supported functions: 'sigmoid', 'tanh', 'relu', 'softplus', 'linear'")


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


    def linear(self, x):
        return x


    def linear_d(self, x):
        return 1


class CostFunction:
    """
    Class that implements some cost functions (method with name 'calculate')
    and their derivations (name of method - 'calculate_d')

    Attributes
    ----------
    function_name: str
        name of the function
    calculate: Callable[Dict[str, float], float]
        calculates passed function between predicted and target values of some key
    calculate_d: Callable[Dict[str, float], float]
        calculates derivative value of the class function between predicted and target values of some key
    calculate_error: Callable[List[float], float]
        calulates error between predicted and target values
    """

    def __init__(self, function_name):
        self.function_name = function_name
        if function_name == "sqared error":
            self.calculate = self.sqared_eroor
            self.calculate_d = self.sqared_eroor_d
            self.calculate_error = self.mean_sqared_error
        else:
            raise NameError(f"Function '{function_name}' isn't supported. Supported functions: 'sqared error'")


    def sqared_eroor(self, predicted, target):
        return {key: (predicted[key] - target[key])**2 for key in predicted}


    def sqared_eroor_d(self, predicted, target):
        return {key: 2*(predicted[key] - target[key]) for key in predicted}


    def mean_sqared_error(self, predicted, target):
        return np.sum( (np.array(list(predicted)) - np.array(list(target)))**2 )
