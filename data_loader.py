from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, sample_dir):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.sample_dir = sample_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        self.num_images = len(self.train_dataset) if self.mode == 'train' else len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        if self.mode == 'train':
            lines = [line.rstrip() for line in open(self.attr_path, 'r')]
            all_attr_names = lines[1].split()
            for i, attr_name in enumerate(all_attr_names):
                self.attr2idx[attr_name] = i
                self.idx2attr[i] = attr_name
            lines = lines[2:]
            random.seed(1234)
            random.shuffle(lines)
            for i, line in enumerate(lines):
                split = line.split()
                filename = split[0]
                values = split[1:]

                label = []
                for attr_name in self.selected_attrs:
                    idx = self.attr2idx[attr_name]
                    label.append(values[idx] == '1')
                self.train_dataset.append([filename, label])
            print(self.train_dataset[0])
        elif self.mode == 'test':
            filenames = os.listdir(self.sample_dir)
            default_label = [False for i in range(6)]
            for filename in filenames:
                self.test_dataset.append([filename, default_label])
            print(self.test_dataset[0])
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename)) if self.mode == 'train' else Image.open(os.path.join(self.sample_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', sample_dir = '', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, sample_dir)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
