from typing import Callable, Optional, Any, List
import numpy as np

# my modules
from .functions import ActivationFunction
from .nodes import Node

# for creating abstract class and abstract method
from abc import ABC, abstractmethod




class Layer(ABC):
    """
    An abstract class that implements functionality of the perceptron layer

    Attributes
    ----------
    name: str
        name of the layer
    size: int
        number of the layer nodes

    next_layer: Layer
        the next layer
    prev_layer: Layer
        the previous layer

    nodes: list[Node]
        all nodes of the layer
    """


    def __init__(self, name, nodes_number, node_names, prev_layer,
                    layer_weights, layer_bias) -> None:
        self.name = name
        self.size = nodes_number
        self.next_layer = None
        self.prev_layer = prev_layer

        self.nodes = {}
        for node_name in node_names:
            self.nodes[node_name] = Node(name=node_name,
                                            weights=layer_weights[node_name],
                                            bias=layer_bias[node_name])



class HiddenLayer(Layer):
    """
    Class which implements hidden layer of the perceptron

    Methods
    -------

    Attributes
    ----------
    name: str
        name of the layer
    size: int
        number of the layer nodes

    next_layer: Layer
        next layer
    prev_layer: Layer
        previous layer

    activation: function
        activation function of the layer
    nodes: list[Node]
        all nodes of the layer
    """


    def __init__(self, name, nodes_number, activation, prev_layer) -> None:
        self.activation = ActivationFunction(activation)

        node_names = []
        layer_weights = {}
        layer_bias = {}
        for i in range(nodes_number):
            node_names.append(f"{name}{i+1}")
            weights = dict(zip( prev_layer.nodes,
                                np.random.random(size=prev_layer.size) ))
            layer_weights[node_names[-1]] = weights
            layer_bias[node_names[-1]] = np.random.random()

        super().__init__(name=name, nodes_number=nodes_number,
                            prev_layer=prev_layer, node_names=node_names,
                            layer_weights=layer_weights, layer_bias=layer_bias)


    def get_calculations(self, input_values) -> List[float]:
        """
        That method gets input vector (dictionary with numbers, calculated on previous
        layer in "prev layer node: value" format), computes the layer
        calculations and returns the output dictionary in the same format
        """
        calculations = {self.name + '_ws': { },
                        self.name + '_a': {} }

        for node in self.nodes.values():
            weighted_sum = [input_values[input_node_name]
                            * node.weights[input_node_name] + node.bias
                            for input_node_name in input_values.keys()]
            calculations[self.name + '_ws'][node.name] = sum(weighted_sum)
            calculations[self.name + '_a'][node.name] = \
            float( self.activation.calculate( calculations[self.name + '_ws'][node.name]) )
        return calculations


class OutputLayer(HiddenLayer):
    """
    Class which implements output layer of the perceptron

    Methods
    -------

    Attributes
    ----------
    name: str
        name of the layer
    size: int
        number of the layer nodes

    next_layer: None, because it is the last layer of perceptron
        next layer
    prev_layer: Layer
        previous layer

    activation: function
        activation function of the layer
    nodes: list[Node]
        all nodes of the layer
    """

    def __init__(self, name, nodes_number, activation, prev_layer):
        super().__init__(name=name, nodes_number=nodes_number,
                            activation=activation, prev_layer=prev_layer)


class InputLayer(Layer):
    """
    Input layer of the perceptron

    Attributes
    ----------
    name: str
        name of the layer
    size: int
        number of the layer nodes (number of the perceptron inputs)

    next_layer: Layer
        next layers
    prev_layer: None
        previous layer (always None, because input is the first layer)

    nodes: list[Node]
        all nodes of the layer
    """


    def __init__(self, name, nodes_number) -> None:
        node_names = []
        layer_weights = {}
        layer_bias = {}
        for i in range(nodes_number):
            node_names.append(f"{name}{i+1}")
            layer_weights[node_names[-1]] = None
            layer_bias[node_names[-1]] = None

        super().__init__(name=name, nodes_number=nodes_number,
                            prev_layer=None, node_names=node_names,
                            layer_weights=layer_weights, layer_bias=layer_bias)
