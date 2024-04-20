from collections import Counter


__all__ = ['substitute_labels']


def substitute_labels(counter, labels_map):
    # Invert the labels map to get a mapping from descriptive labels to numeric labels
    inverted_labels_map = {v: k for k, v in labels_map.items()}

    # Create a new counter with numeric labels
    new_counter = Counter({inverted_labels_map[label]: count for label, count in counter.items(
    ) if label in inverted_labels_map})

    return new_counter
