# -*- coding: utf-8 -*-

from typing import Sequence, Union
import torch


###############################################################################

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
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.std = torch.FloatTensor(std).view(-1, 1, 1)

    def to(self, device, *args, **kwargs):
      self.mean = self.mean.to(device)
      self.std = self.std.to(device)
      super().to(device, *args, **kwargs)

    def forward(self, img):
        return (img - self.mean) / self.std