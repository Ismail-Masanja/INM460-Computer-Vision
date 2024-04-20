import torch


__all__ = ['AddGaussianNoise']


class AddGaussianNoise(object):
    """ Adds Gaussian noise in an Image Tensor """

    def __init__(self, mean=0., std=1.):
        self.std = torch.Tensor(std).view(-1, 1, 1)
        self.mean = torch.Tensor(mean).view(-1, 1, 1)

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean