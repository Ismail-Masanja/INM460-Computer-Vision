import random
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


__all__ = ['show_rand_sample']


def show_rand_sample(data_loader, *, title, rows=4, cols=5, labels_map=None, fig_show=True):
    """ Shows a Random sample of the dataloader.
    
        REF: Ismail Masanja, N3063/INM702: Mathematics and Programming for AI.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(title)

    images, labels, *_ = next(iter(data_loader))
    images_copy = images.clone()
    labels_copy = labels.clone()

    plots = list()
    plot = namedtuple('SinglePlot', 'image label')

    # Flatten axes array for easier navigation
    axes = axes.flatten()

    for ax in axes:
        # Select a random index in the batch
        idx = random.randint(0, len(images_copy) - 1)
        img = images_copy[idx].numpy()
        # Rearrange dimensions from CxHxW to HxWxC
        img = np.transpose(img, (1, 2, 0))
        img = (img * 0.5) + 0.5  # Unnormalize
        label = labels_copy[idx].item()

        if labels_map:
            label = labels_map.get(label, str(label))

        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

        plots.append(plot(img, label))

    if fig_show:
        plt.show()

    return fig, axes, plots
