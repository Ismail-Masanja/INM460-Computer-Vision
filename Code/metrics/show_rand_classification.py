import random
import torch
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['random_classification']


def random_classification(model,
                          data_loader,
                          device,
                          *, rows=5, cols=4,
                          labels_map=None,
                          title=None,
                          fig_show=True):
    """  Takes a random sample of the dataloader and uses the model to predict the outcome.
        the outcome is compared to the True label. A figure is plotted to show the results

        REF: Ismail Masanja, N3063/INM702: Mathematics and Programming for AI. Heavy Modifications
            done to fit this project.
    """

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if title is None:
        title = f'Random Sample Classification Report \n {model.__class__.__name__}'
    fig.suptitle(title)

    # Fetch a random batch from the DataLoader
    images, labels = next(iter(data_loader))
    labels = labels.detach()
    images = images.to(device).detach()

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        predictions = model(images)

    pred_labels = predictions.argmax(dim=1).cpu().numpy()

    # Flatten axes array for easier navigation
    axes = axes.flatten()

    for ax in axes:
        # Select a random index in the batch
        idx = random.randint(0, len(images) - 1)
        # Convert tensor to numpy array
        img = images[idx].detach().cpu().numpy()
        # Rearrange dimensions from CxHxW to HxWxC
        img = np.transpose(img, (1, 2, 0))
        img = (img * 0.5) + 0.5  # Unnormalize
        label_true = labels[idx].item()
        label_pred = pred_labels[idx]

        if labels_map:
            label_true = labels_map[label_true]
            label_pred = labels_map[label_pred]
            label = f'True: {label_true}\nPred: {label_pred}'
        color = 'green' if label_true == label_pred else 'red'
        _ = ax.set_title(label, color=color, fontsize=9)

        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

    if fig_show:
        plt.show()

    return fig, axes
