# for working with graphs
import networkx as nx

from typing import Callable, Optional, List
from layer import Layer




class MultilayerPerceptron:
    """
    Class that implements the functionality of a multilayered perceptron.

    Attributes
    ----------
    __layers: List[Layer]
        list of layers in newral perceptron
    depth: int
        the number of perceptron layers
    """

    def __init__(self, n_inputs: int) -> None:
        # input layer initialization
        self.__layers = [Layer(name = "x", activation=None, n_nodes=n_inputs,
                                is_input_layer=True)]
        self.depth = 0


    def add_layer(self, name: str, activation: Callable[[float], float],
                    n_nodes: int) -> None:
        """Adds a layer to the end of the perceptron layers

        Raises
        ------
        NameError
            if perceptron already have a layer with passed name
        """

        if name not in [layer.name for layer in self.__layers]:
            self.__layers.append( Layer(name=name, activation=activation, n_nodes=n_nodes,
                                        prev_layer_nodes_number=self.__layers[-1].n_nodes) )
            self.depth += 1
        else:
            raise NameError(f"Layer with name '{name}' already exists.")


    def get_layers(self) -> List:
        return self.__layers.copy()


    def run(self, inputs):
        output = []
        for layer in self.__layers:
            output = layer.get_output(inputs)
            inputs = output



    def get_full_graph(self) -> nx.DiGraph:
        """Method for creating graph which shows all connections between layers"""

        graph = nx.DiGraph()
        for curr_layer, next_layer in zip(self.__layers[:-1],
                                            self.__layers[1:]):
            for i in range(curr_layer.n_nodes):
                for j in range(next_layer.n_nodes):
                    graph.add_edge("$%s_{%i}}$" % (curr_layer.name, i), "$%s_{%i}}$" % (next_layer.name, j))
        return graph


    def get_simple_graph(self):
        """Method for creating a graph with abbreviated layer names"""

        graph = nx.DiGraph()
        for curr_layer, next_layer in zip(self.__layers[:-1],
                                            self.__layers[1:]):
            graph.add_edge(f"${curr_layer.name}$", f"${next_layer.name}$")
        return graph
