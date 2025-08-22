"""
activations
-----------
"""
import torch.nn as nn


def get_activation(name: str):
    """
    A simple method to return neural network activations (Pytorch modules) from their name (string)

    Args: 
        name (str): The activation function to return 

    Returns: 
        (Torch.nn.Module): The activation function as a torch module
    """
    name = name.lower()
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "none": nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]
