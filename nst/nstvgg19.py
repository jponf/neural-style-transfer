# -*- coding: utf-8 -*-

from typing import Sequence

import torch
import torchvision


###############################################################################

class Vgg19Nst:

    def __init__(self,
                 style_layers: Sequence[int] = (0, 5, 10, 19, 28),
                 content_layers: Sequence[int] = (21,)):
        self.style_layers = style_layers
        self.content_layers = content_layers

        vgg19features = torchvision.models.vgg19(pretrained=True)
