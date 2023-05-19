"""
Constraints on the weights of a Pytorch neural networks.
"""
import torch

_eps = 1e-7


def weight_clipper(x, eps=_eps):
    """
    Set the input tensor values such that the minimum value is eps.
    :param x: torch.Tensor
    :param eps: float
    """
    torch.clamp_(x, min=eps)


def unit_norm_clipper(x, norm=1):
    """
    Set the values of the tensor such that the norm is unitary.
    :param x: torch.Tensor
    :param norm: int
    """
    x.div_(torch.norm(x, norm))


def zero_one_clipper(x):
    """
    Set the elements of the tensor in a range between 0 and 1.
    :param x: torch.Tensor
    """
    x.sub_(torch.min(x)).div_(torch.max(x) - torch.min(x))


def constraint_range(x, constraint_low=0, constraint_high=1):
    """
    Set the values of the tensor x in a range between [constraint_low, constraint_high]
    :param x: torch.Tensor
    :param constraint_low: int
    :param constraint_high: int
    """
    x.sub_(torch.min(x)).div_(torch.max(x) - torch.min(x))
    x.mul_(constraint_low + (constraint_high-constraint_low))


def sum_to_one(x):
    """
    Set the values of the tensor such that the sum is 1.
    :param x: torch.Tensor
    """
    x.div_(x.sum().clamp(_eps))
