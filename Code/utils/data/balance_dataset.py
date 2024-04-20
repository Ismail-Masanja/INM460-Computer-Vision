import itertools
import random
from collections import defaultdict
from torch.utils.data import Dataset


__all__ = ['BalanceDataset']


class BalanceDataset(Dataset):
    """ Balances the dataset by Oversampling technique. Uses a precalculated index map to 
        original dataset to achieve this.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.index_map = self._gen_index_map()

    def _gen_index_map(self):
        all_labels = [self.dataset[idx][1] for idx in range(len(self.dataset))]
        
        # a dict where tha value is a list.
        index_locations_by_label = defaultdict(list)

        for idx, label in enumerate(all_labels):
            index_locations_by_label[label].append(idx)

        # Find the maximum list length
        max_length = max(len(v) for v in index_locations_by_label.values())

        for label, indices in index_locations_by_label.items():
            cycled_indices = itertools.cycle(
                indices)  # Create a cycling iterator
            index_locations_by_label[label] = list(
                itertools.islice(cycled_indices, max_length))

        combined_list = list(itertools.chain.from_iterable(
            index_locations_by_label.values()))
        random.shuffle(combined_list)

        return combined_list

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        image, label = self.dataset[self.index_map[idx]]

        return image, label
