# dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class CBISDDSM_Dataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            label_file (str): Path to the csv/txt file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Assuming label_file is CSV: filename,label
        with open(label_file, 'r') as f:
            lines = f.readlines()
            self.labels = []
            for line in lines:
                parts = line.strip().split(',')
                self.labels.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
