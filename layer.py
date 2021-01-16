from typing import Callable, Optional, Any, List

import inspect
import numpy as np



class Layer:
    """Class for storing and processing layer data

    Methods
    -------


    Attributes
    ----------
    name: str, None
        name of the layer
    activation: Callable[[float], float]
        activation function of layer
    n_nodes: int, None
        number of the layer nodes
    __weights: list[float]
        layer weights
    __bias: float
        bias
    """


    def __init__(self, name, activation, n_nodes, prev_layer_nodes_number=1,
                    **kwargs) -> None:
        """
        Method for initialization layer

        If parameter is_input_layer == True, the layer will be used as input
        layer i.e. the layer will be first layer of network and willn't have
        activation function, weights and bias. Otherwise, the layer will have
        all the class parameters
        """

        if "is_input_layer" in kwargs.keys() and kwargs["is_input_layer"] == True:
            self.name = "x"
            self.activation = None
            self.n_nodes = n_nodes
            self.__weights = None
            self.__bias = None
        else:
            self.name = name
            self.activation = activation
            self.n_nodes = n_nodes
            self.__weights = np.ones((self.n_nodes, prev_layer_nodes_number))
            self.__bias = np.zeros(self.n_nodes)


    def get_output(self, inputs) -> List[float]:
        outputs = []

        if self.activation != None:
            for i in range(self.n_nodes):
                outputs.append( sum(np.array(inputs) * self.__weights[i])
                                + self.__bias[i] )
            outputs = (self.activation(np.array(outputs))).tolist()
        else:
            outputs = inputs

        return outputs


    def get_function_str(self) -> str:
        lines = inspect.getsource(self.activation_func)
        return lines
