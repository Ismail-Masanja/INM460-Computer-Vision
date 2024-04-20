import torch
from collections import namedtuple


__all__ = ['calc_channel_mean_std']


# Define the named tuple structure
ChannelStats = namedtuple('ChannelStats', ['mean', 'std'])


def calc_channel_mean_std(dataloader):
    channel_sum, channel_sum_squared, frequency = 0, 0, 0

    # Loop over the whole dataset
    for images, _ in dataloader:
        # Make sure the images tensor is on the CPU and in float format
        images = images.float()

        # Accumulate sum and sum of squares per channel
        channel_sum += torch.mean(images, dim=[0, 2, 3])
        channel_sum_squared += torch.mean(images ** 2, dim=[0, 2, 3])

        frequency += 1

    # Calculate the mean and std per channel
    mean = channel_sum / frequency
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channel_sum_squared / frequency - mean ** 2) ** 0.5

    # Pack the results in a named tuple before returning
    stats = ChannelStats(mean=mean, std=std)

    return stats
