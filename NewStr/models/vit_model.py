# models/vit_model.py

import timm
import torch.nn as nn
# models/vit_model.py

import timm
import torch.nn as nn

# models/vit_model.py

import timm
import torch.nn as nn
'''
def get_modified_vit_model(num_classes, model_name='vit_base_patch16_224', num_frozen_layers=0):
    """
    Loads a pre-trained ViT model, modifies it for a new number of classes,
    and freezes the attention heads in the first 'num_frozen_layers' layers.

    Args:
        num_classes (int): The number of classes for the classification task.
        model_name (str): Name of the ViT model to load.
        num_frozen_layers (int): Number of layers to freeze attention heads.

    Returns:
        torch.nn.Module: The modified ViT model.
    """
    # Load the ViT model
    print(num_frozen_layers);
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    # Freeze attention heads in the first 'num_frozen_layers' layers
    for idx, block in enumerate(model.blocks):
        if idx < num_frozen_layers:
            for param in block.parameters():
                param.requires_grad = False

    return model

'''
def get_modified_vit_model(num_classes, model_name='vit_base_patch16_224'):
    """
    Loads a pre-trained ViT model, modifies it for a new number of classes,
    and freezes the attention heads in layers 8, 9, 10, and 11.

    Args:
        num_classes (int): The number of classes for the classification task.
        model_name (str): Name of the ViT model to load.

    Returns:
        torch.nn.Module: The modified ViT model.
    """
    # Load the ViT model
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    # Freeze attention heads in layers 8, 9, 10, and 11
    layers_to_freeze = [1,2,3,4,5]
    for idx, block in enumerate(model.blocks):
        if idx in layers_to_freeze:
            for param in block.parameters():
                param.requires_grad = False

    return model
'''

def get_modified_vit_model(num_classes, model_name='vit_base_patch16_224'):
    """
    Loads a pre-trained Vision Transformer (ViT) model using timm and modifies it for a new number of classes.

    Args:
        num_classes (int): The number of classes for the classification task.
        model_name (str): The name of the ViT model to load. Defaults to 'vit_base_patch16_224'.

    Returns:
        torch.nn.Module: The modified ViT model.
    """
    # Load the ViT model using timm
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes  # Replaces the classifier head
    )

    return model

'''