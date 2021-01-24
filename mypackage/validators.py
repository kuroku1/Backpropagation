from typing import List
from .layers import Layer, InputLayer, HiddenLayer, OutputLayer




class StructureError(Exception):
    pass


def validate_data(X, y):
    if type(X) == list:
        X = np.array(X)
    if type(y) == list:
        y = np.array(y)
    pass


##### VALIDATORS #####
def validate_name(layer_name: str, layers: List[Layer]) -> None:
    """
    Method that checks if a layer with passed name exists
    """
    layer_names = [layer.name for layer in layers]
    if layer_name in layer_names:
        raise NameError(f"Layer with name '{layer_name}' already exists.")


def validate_type(layer_type: type, layers: List[Layer]) -> None:
    """
    Method that validates layer type
    """
    if layer_type not in ["input", "hidden", "output"]:
        raise StructureError(f"Cannot add a '{layer_type}' layer type. Type can be 'input', 'hidden' or 'output'")

    if layer_type in ["hidden", "output"] and len(layers) == 0:
        raise StructureError(f"Model doesn't have an input layer. Please, add it")

    if layer_type == "output" and len(layers) == 1:
        raise StructureError(f"Model doesn't have at least one hidden layer. Please, add it")

    for layer in layers:
        if type(layer) == InputLayer and layer_type == "input":
            raise StructureError(f"Model already have an input layer")
        elif type(layer) == OutputLayer and layer_type in ["hidden", "output"]:
            raise StructureError(f"Model already have an output layer")


def validate_layers(layers: List[Layer]):
    """
    Method that checks the correctness of the perceptron structure
    """
    layer_types = [type(layer) for layer in layers]
    if InputLayer not in layer_types:
        raise StructureError("Model should have at least one input layer")
    elif HiddenLayer not in layer_types:
        raise StructureError("Model should have at least one hidden layer")
    elif OutputLayer not in layer_types:
        raise StructureError("Model should have at least one output layer")
