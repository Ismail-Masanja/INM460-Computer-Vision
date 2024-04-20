from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


__all__ = ['CachedDataset']


class CachedDataset(Dataset):
    """ Caches data into memory to reduce data loading times especially when loading from 
        Google Drive or a remote location.
    """

    def __init__(self, data_dir, transform=None, cache_size_mb=1024):
        self.data_dir = data_dir
        self.transform = transform
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')

        # Ensuring a consistent order
        self.image_filenames = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith('.jpeg')])
        self.cache_size_mb = cache_size_mb
        self.cache = {}
        # Index to keep track of where the current cache started filling
        self.cache_start_idx = 0

        # Initial cache loading
        self.reload_cache()

    def reload_cache(self, start_idx=0):
        """Reloads cache starting from a specific index."""
        self.cache = {}
        cache_size_bytes = self.cache_size_mb * 1024 * 1024
        current_cache_size = 0
        self.cache_start_idx = start_idx

        for i, filename in enumerate(self.image_filenames[start_idx:], start=start_idx):
            if current_cache_size >= cache_size_bytes:
                break  # Stop if cache is full

            img_path = os.path.join(self.images_dir, filename)
            label_path = os.path.join(
                self.labels_dir, filename.replace('.jpeg', '.txt'))

            image = Image.open(img_path).convert('RGB')
            with open(label_path, 'r') as f:
                label = int(f.read().strip())

            img_size_bytes = os.path.getsize(img_path)
            if current_cache_size + img_size_bytes <= cache_size_bytes:
                self.cache[i] = (image, label)
                current_cache_size += img_size_bytes
            else:
                # If adding this image exceeds the cache, stop adding further
                break

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if idx not in self.cache:
            if idx >= len(self.image_filenames):
                # If idx is beyond the dataset size, start from the beginning
                idx = idx % len(self.image_filenames)
            self.reload_cache(start_idx=idx)

        if idx not in self.cache:
            # If still not in cache after reload, there's a logical error
            raise RuntimeError(
                'Item not loaded into cache. Check cache size and item sizes.')

        image, label = self.cache[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
