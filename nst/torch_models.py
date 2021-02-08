# -*- coding: utf-8 -*-

import collections
from typing import Sequence

import torch
import torchvision

from .torch_utils import NstModuleWrapper, ImageNormalization


################################################################################

_VGG19_LAYER_NAMES = {torch.nn.Conv2d: "conv2d_{}",
                      torch.nn.ReLU: "relu_{}",
                      torch.nn.MaxPool2d: "pool_{}",
                      torch.nn.BatchNorm2d: "bn_{}"}


################################################################################

def make_vgg19_nst() -> NstModuleWrapper:
    vgg19 = torchvision.models.vgg19(pretrained=True)

    vgg19_style_layers = [0, 5, 10, 19, 28]
    vgg19_content_layers = [21]
    last_layer = max(max(vgg19_style_layers), max(vgg19_content_layers))

    style_layers = []
    content_layers = []
    model = torch.nn.Sequential(collections.OrderedDict(
        [("normalize", ImageNormalization())]))

    conv_idx = 0
    for i, layer in enumerate(vgg19.features.children()):
        if isinstance(layer, torch.nn.Conv2d):
            conv_idx += 1

        name = _VGG19_LAYER_NAMES[type(layer)].format(conv_idx)
        model.add_module(name, layer)

        if i in vgg19_content_layers:
            content_layers.append(layer)
        if i in vgg19_style_layers:
            style_layers.append(layer)

        if i >= last_layer:  # The model should not be any longer
            break

    # Freeze model
    for param in model.parameters():
        param.requires_grad_(False)

    return NstModuleWrapper(model=model,
                            style_layers=style_layers,
                            content_layers=content_layers)
