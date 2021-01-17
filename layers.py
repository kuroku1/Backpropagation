from typing import Callable, Optional, Any, List
from functions import Function
from abc import ABC, abstractmethod
# import inspect
import numpy as np



class Layer(ABC):
    """
    An abstract class that implements functionality of the perceptron layer

    Attributes
    ----------
    name: str
        name of the class
    n_nodes: int
        number of nodes in layer
    """


    def __init__(self, name, n_nodes) -> None:
        self.name = name
        self.n_nodes = n_nodes


    @abstractmethod
    def get_output(self, inputs) -> List[float]:
        pass



class HiddenLayer(Layer):
    """Class for storing and processing hidden layer data

    Methods
    -------

    Attributes
    ----------
    name: str
        name of the layer
    n_nodes: int
        number of the layer nodes

    activation: function
        activation function of the layer
    __weights: list[float]
        the layer weights
    __bias: float
        the layer bias
    """


    def __init__(self, name, n_nodes, activation, parent) -> None:
        super().__init__(name=name, n_nodes=n_nodes)
        self.activation = Function(activation)
        self.parent = parent

        self.__weights = np.ones((self.n_nodes, self.parent.n_nodes)).tolist()
        self.__bias = np.zeros(self.n_nodes).tolist()


    def get_output(self, inputs) -> List[float]:
        """
        That method gets inputs (vector with numbers, calculated on parent
        layer), computes the layer calculations and return output vector
        """
        outputs = []

        # 1. multiplying the input vector by the weight matrix
        # 2. adding the bias verctor
        for i in range(self.n_nodes):
            outputs.append( sum( np.multiply(inputs, self.__weights[i]),
                                    self.__bias[i] )
                          )
        # 3. applying activation function
        return (self.activation.function(np.array(outputs))).tolist()



class InputLayer(Layer):
    """
    Input layer of the perceptron


    Attributes
    ----------
    name: "x"
        name of the class(always "x")
    n_nodes: int
        the layer nodes (number of the perceptron inputs)
    """


    def __init__(self, n_nodes) -> None:
        super().__init__(name="x", n_nodes=n_nodes)


    def get_output(self, inputs):
        """
        That method just returns the inputs, because input layer of perceptron
        has no any calculations"""
        return inputs
