import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter


__all__ = ['plot_distribution']


def plot_distribution(dataloader, *, title, labels_map=None, fig_show=True):
    """  Plots the frequency distribution of the dataloader

        REF: Ismail Masanja N3063/INM702: Mathematics and Programming for AI
            Adapted to fit this project.
    """
    
    label_counts = Counter()

    for _, labels in dataloader:
        labels = labels.numpy()
        if labels_map:
            labels = [labels_map[label] for label in labels]
        label_counts.update(labels)

    # Plotting
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel('Number of Occurrences')
    ax.set_xlabel('Labels')
    bars = ax.bar(label_counts.keys(), label_counts.values())

    # Remove the top spine
    ax.spines['top'].set_visible(False)

    # Annotate each bar with its height value
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    if fig_show:
        plt.show()

    return label_counts, fig, ax
