from torch.utils.data import Dataset
from PIL import Image
import os


__all__ = ['CustomDataset']


class CustomDataset(Dataset):
    """ Loads Dataset from data folder """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')

        self.image_filenames = [f for f in os.listdir(
            self.images_dir) if f.endswith('.jpeg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(img_name)

        label_name = os.path.join(
            self.labels_dir, self.image_filenames[idx].replace('.jpeg', '.txt'))
        with open(label_name, 'r') as f:
            label = int(f.read().strip())

        if self.transform:
            image = self.transform(image)

        return image, label
