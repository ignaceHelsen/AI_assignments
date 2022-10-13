from typing import Tuple

import torch
import torch.nn as nn


def calc_vae_loss(distance: nn.Module, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  target, options: dict) -> torch.Tensor:
    """
    Calculate the vae loss
    TODO: complete this function.
    :param options: Options
    :param distance: The distance to use for reconstruction loss
    :param input: tuple consisting of (decoded image, latent_vector, mu, log_var)
    :param target: target image (torch.Tensor)
    :rtype vae_loss: torch.Tensor
    """
    decoded_image, latent_vector, mu, log_var = input

    return distance(decoded_image, target) + options["lambda"] * (- 1/2 * (1+log_var-mu*mu - torch.exp(log_var)).sum()) # loss beta-VAE
