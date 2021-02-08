# -*- coding: utf-8 -*-

from typing import Sequence, Union

import torch


###############################################################################

class NstModuleWrapper:

    def __init__(self, model: torch.nn.Module,
                 style_layers: Sequence[torch.nn.Module],
                 content_layers: Sequence[torch.nn.Module]):
        if len(style_layers) < 1:
            raise ValueError("At least one style layer must be specified")
        if len(content_layers) < 1:
            raise ValueError("At least one content layer must be specified")

        self._model = model
        self._style_layers = style_layers
        self._content_layers = content_layers

        self._style_out = [None] * len(self._style_layers)
        self._content_out = [None] * len(self._content_layers)

        self._style_target = [None] * len(self._style_layers)
        self._content_target = [None] * len(self._content_layers)

        # Register forward hooks
        for i, layer in enumerate(self._style_layers):
            layer.register_forward_hook(self._get_style_activation_hook(i))

        for i, layer in enumerate(self._content_layers):
            layer.register_forward_hook(self._get_content_activation_hook(i))

    def call(self, image: torch.Tensor) -> torch.Tensor:
        """Calls the internal `model` with the given image.

        Parameters
        ----------
        image : torch.Tensor
            An image tensor with dimensions (batch, channels, height, width)
            or (channels, height, width).
        """
        if len(image.shape) == 3:
            image = torch.unsqueeze(image, 0)
        return self._model(image)

    def compute_style_loss(self) -> torch.Tensor:
        """Computes the style loss of the last run of the model with
        respect to the previously set target.
        """
        loss = 0.0
        for value, target in zip(self._style_out, self._style_target):
            gram = compute_gram_matrix(value)
            loss += torch.nn.functional.mse_loss(gram, target)
        return loss / len(self._style_out)

    def compute_content_loss(self) -> torch.Tensor:
        """Computes the content loss of the last run of the model with
        respect to the previously set target.
        """
        loss = 0.0
        for value, target in zip(self._content_out, self._content_target):
            loss += torch.nn.functional.mse_loss(value, target)
        return loss / len(self._style_out)

    def set_style_image(self, image: torch.Tensor):
        """Processes the style image and saves its features for the style
        transfer process.

        Parameters
        ----------
        image : torch.Tensor
            An image tensor with dimensions (batch, channels, height, width)
            or (channels, height, width).
        """
        self.call(image)
        for i, features in enumerate(self._style_out):
            gram = compute_gram_matrix(features)
            self._style_target[i] = gram.detach()

    def set_content_image(self, image: torch.Tensor):
        """Processes the content image and saves its features for the style
        transfer process.

        Parameters
        ----------
        image : torch.Tensor
            An image tensor with dimensions (batch, channels, height, width)
            or (channels, height, width).
        """
        self.call(image)
        for i, features in enumerate(self._content_out):
            self._content_target[i] = features.detach()

    def to(self, device):
        self._model.to(device)

        # Move style to device
        for i, tensor in enumerate(self._style_target):
            if tensor is not None:
                self._style_target[i] = tensor.to(device)
        for i, tensor in enumerate(self._style_out):
            if tensor is not None:
                self._style_out[i] = tensor.to(device)

        # Move style to device
        for i, tensor in enumerate(self._content_target):
            if tensor is not None:
                self._content_target[i] = tensor.to(device)
        for i, tensor in enumerate(self._content_out):
            if tensor is not None:
                self._content_out[i] = tensor.to(device)


    def _get_style_activation_hook(self, layer_idx):
        def hook(model, input, output):
            self._style_out[layer_idx] = output
        return hook

    def _get_content_activation_hook(self, layer_idx):
        def hook(model, input, output):
            self._content_out[layer_idx] = output
        return hook



class ImageNormalization(torch.nn.Module):
    """This module should be used to normalize an image before feeding it
    to the rest of the network.

    Parameters
    ----------
    mean : Sequence[float]
        Image channels mean.
    std : Sequence[float]
        Image channels standard deviation.
    """

    def __init__(self,
                 mean: Sequence[float] = (0.485, 0.456, 0.406),
                 std: Sequence[float] = (0.229, 0.224, 0.225)):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.FloatTensor(mean).view(-1, 1, 1)
        std = torch.FloatTensor(std).view(-1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, img):
        return (img - self.mean) / self.std


################################################################################

def compute_gram_matrix(inputs):
    """Computes the gram matrix of the given inputs.

    Parameters
    ----------
    inputs : torch.Tensor
        An image tensor with dimensions (batch, channels, height, width).
    """
    batch_size, n_channels, height, width = inputs.size()

    features = inputs.view(batch_size * n_channels,
                           height * width)  # resise F_XL into \hat F_XL

    gram = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return gram.div(batch_size * n_channels * height * width)