# for working with graphs
import networkx as nx

from typing import Callable, Optional, List
from layers import Layer, HiddenLayer, InputLayer




class MultilayerPerceptron:
    """
    Class that implements the functionality of a multilayered perceptron.

    Attributes
    ----------
    __last_layer: HiddenLayer
        list of layers in perceptron
    n_inputs: int
        the number of perceptron inputs
    depth: int
        the number of perceptron hidden layers
    """

    def __init__(self, n_inputs) -> None:
        # input layer initialization
        self.__last_layer = InputLayer(n_inputs)
        self.n_inputs = n_inputs
        self.depth = 0


    def add_layer(self, name: str, activation: str,
                    n_nodes: int) -> None:
        """Adds a layer to the end of the perceptron layers

        Raises
        ------
        NameError
            if perceptron already have a layer with passed name
        """

        layer_names = [layer.name for layer in self.get_layers()]
        if name not in layer_names:
            # creating layer if passed name is unique
            layer = HiddenLayer(name=name, activation=activation,
                            n_nodes=n_nodes, parent=self.__last_layer)
            self.__last_layer = layer

            self.depth += 1
        else:
            raise NameError(f"Layer with name '{name}' already exists.")


    def get_layers(self) -> List[Layer]:
        """
        Method which return perceptron structure (all layers from first to
        last in list)
        """

        layers = [self.__last_layer]
        for _ in range(self.depth):
            layers.append(layers[-1].parent)
        return layers[::-1]


    def run(self, inputs):
        output = []
        for layer in self.get_layers():
            output = layer.get_output(inputs)
            inputs = output
        return output



    def get_full_graph(self) -> nx.DiGraph:
        """Method for creating graph which shows all connections between layers"""

        graph = nx.DiGraph()
        for curr_layer, next_layer in zip(self.get_layers()[:-1],
                                            self.get_layers()[1:]):
            for i in range(curr_layer.n_nodes):
                for j in range(next_layer.n_nodes):
                    graph.add_edge("$%s_{%i}}$" % (curr_layer.name, i),
                                    "$%s_{%i}}$" % (next_layer.name, j),
                                    label="gege")
        return graph


    def get_simple_graph(self):
        """Method for creating a graph with abbreviated layer names"""

        graph = nx.DiGraph()
        for curr_layer, next_layer in zip(self.get_layers()[:-1],
                                            self.get_layers()[1:]):
            graph.add_edge(f"${curr_layer.name}$", f"${next_layer.name}$")
        return graph
