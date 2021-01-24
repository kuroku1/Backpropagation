# for working with graphs
import networkx as nx
import numpy as np

from typing import Callable, Optional, List, Dict
from .layers import Layer, InputLayer, HiddenLayer, OutputLayer
from .validators import validate_name, validate_type, validate_layers
from .nodes import Node
from .functions import CostFunction



class MultilayerPerceptron:
    """
    Class that implements the functionality of a multilayered perceptron.

    Attributes
    ----------
    depth: int
        the number of perceptron all perceptron layers

    Methods
    -------

    """

    def __init__(self, cost_function: str = "sqared error") -> None:
        # input layer initialization
        self.__first_layer = None
        self.__last_layer = None
        self.depth = 0
        self.cost = CostFunction(cost_function)


    def structure(self) -> Dict[str, Dict[str, Node]]:
        structure = {}
        for layer in self.__get_layers():
            layer_nodes = {}
            for node in layer.nodes.values():
                layer_nodes[node.name] = {'weights': node.weights,
                                            'bias': node.bias}
            structure[layer.name] = layer_nodes
        return structure


    #### ADDING LAYERS ####
    def add_input_layer(self, name: str, nodes_number: int) -> None:
        validate_name(layer_name=name, layers=self.__get_layers())
        validate_type(layer_type="input", layers=self.__get_layers())

        self.__first_layer = InputLayer(name=name, nodes_number=nodes_number)
        self.__last_layer = self.__first_layer
        self.depth += 1


    def add_hidden_layer(self, name: str, nodes_number: int,
                            activation: str = "output") -> None:
        validate_name(layer_name=name, layers=self.__get_layers())
        validate_type(layer_type="hidden", layers=self.__get_layers())

        layer = HiddenLayer(name=name, nodes_number=nodes_number,
                                activation=activation,
                                prev_layer=self.__last_layer)
        self.__last_layer.next_layer = layer
        self.__last_layer = layer
        self.depth += 1


    def add_output_layer(self, name: str, nodes_number: int,
                            activation: str) -> None:
        validate_name(layer_name=name, layers=self.__get_layers())
        validate_type(layer_type="output", layers=self.__get_layers())

        layer = OutputLayer(name=name, nodes_number=nodes_number,
                            activation=activation,
                            prev_layer=self.__last_layer)
        self.__last_layer.next_layer = layer
        self.__last_layer = layer
        self.depth += 1


    def __get_layers(self) -> List[Layer]:
        layer = self.__first_layer
        layers = []
        for _ in range(self.depth):
            layers.append(layer)
            layer = layer.next_layer
        return layers


    def get_calculations(self, input_values):
        calculations = {}
        layer = self.__first_layer
        calculations = {layer.name + '_a': dict(zip( list(layer.nodes),
                                                    input_values )) }
        layer = layer.next_layer
        while layer != None:
            calculations.update( layer.get_calculations(
                                calculations[layer.prev_layer.name + '_a']) )
            layer = layer.next_layer
        return calculations


    def back_propagation(self, input_values, output_values, learning_rate,
                            batch_size):
        output = dict(zip( self.__last_layer.nodes, output_values ))
        calculations = self.get_calculations(input_values)

        changes = self.cost.calculate_d(
                                calculations[self.__last_layer.name + '_a'],
                                output)
        layer = self.__last_layer
        while type(layer) != InputLayer:
            prev_changes = {node_name: 0 for node_name in layer.prev_layer.nodes}
            for node in layer.nodes.values():
                changes[node.name] *= layer.activation.calculate_d(
                            calculations[layer.name+'_ws'][node.name])

                # changes in layer weights and bias
                node.bias -= learning_rate * changes[node.name] / batch_size
                for key in node.weights:
                    node.weights[key] -= learning_rate * changes[node.name] \
                                * calculations[layer.prev_layer.name+'_a'][key] \
                                / batch_size

                for prev_node in node.weights:
                    prev_changes[prev_node] += node.weights[prev_node] * changes[node.name]

            changes = prev_changes
            layer = layer.prev_layer
        return changes


    def fit(self, X, y, train_size=0.8, epochs=1000, batch_size=1,
            learning_rate=0.01):
        if len(X) != len(y):
            raise ValueError('Sizes are not equal')

        data_size = len(y)
        epochs_data = {'train_error': [], 'test_error': []}
        for i in range(epochs):
            train_indexes = np.random.randint(data_size,
                                                size=int(train_size * data_size))
            test_indexes = [i for i in range(data_size) if i not in train_indexes]
            train_error, test_error = 0, 0
            batch_size = min(batch_size, len(train_indexes))

            for index in train_indexes:
                self.back_propagation(X[index], y[index],
                                        batch_size=batch_size,
                                        learning_rate=learning_rate)
                train_error += self.cost.mean_sqared_error(
                                                self.predict(X[index]),
                                                y[index] )

            for index in test_indexes:
                test_error += self.cost.mean_sqared_error(
                                                self.predict(X[index]),
                                                y[index] )
            epochs_data['train_error'].append(train_error / len(train_indexes))
            epochs_data['test_error'].append(test_error / len(test_indexes))
        return epochs_data


    def predict(self, data):
        return list(self.get_calculations(data)[self.__last_layer.name+'_a'].values())


    #### GRAPH CREATING METHODS ####
    def get_full_graph(self) -> nx.DiGraph:
        """Method for creating graph which shows all connections between layers"""

        graph = nx.DiGraph()
        layer = self.__first_layer
        for _ in range(self.depth-1):
            for node_name in layer.nodes.keys():
                for next_node_name in layer.next_layer.nodes.keys():
                    graph.add_edge(node_name, next_node_name)
            layer = layer.next_layer

        return graph


    def get_simple_graph(self):
        """Method for creating a graph with abbreviated layer names"""

        graph = nx.DiGraph()
        layer = self.__first_layer
        for _ in range(self.depth-1):
            graph.add_edge(layer.name, layer.next_layer.name)
            layer = layer.next_layer
        return graph
