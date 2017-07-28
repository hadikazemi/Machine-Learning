import torch
import torch.utils.data as data
import os
import glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batchSize = 10
shuffle = True
nThreads = 4


# Define the corresponding class for your dataset
class my_dataset(data.Dataset):
    def initialize(self, root_path):
        assert os.path.isdir(root_path), '%s is not a valid directory' % root_path

        # List all JPEG images
        files_path = os.path.join(root_path, '*.jpg')
        self.images = glob.glob(files_path)
        self.size = len(self.images)

        # Define a transform to be applied to each image
        # Here we resize the image to 224X224 and convert it to Tensor
        self.transform = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor()])

    def __getitem__(self, index):
        # Loads a single data point from the dataset
        # Supporting integer indexing in range from 0 to the size of the dataset (exclusive)

        path = self.images[index % self.size]
        label = int(((path.split('/')[-1]).split('.')[0])=='cat')   # Ectract label from the filename
        img = Image.open(path).convert('RGB')                       # Load the image and convert to RGB
        img = self.transform(img)                                   # Apply the defined transform

        return {'img': img, 'label': label}

    def __len__(self):
        # Provides the size of the dataset

        return self.size


path = 'Cat vs Dog/train'

# Create your dataset and initialize it
dataset = my_dataset()
dataset.initialize(path)

# Create a dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=nThreads)

# Loop over dataset
for i, data in enumerate(dataloader):
    img = data['img']
    label = data['label']

    label = label.numpy()
    img = img.numpy()
    img = np.rollaxis(img, 1, 4)

    for j in range(10):
        plt.subplot(2, 5, j+1)
        plt.imshow(img[j])
        plt.title('{}'.format('cat' if label[j]==1 else 'dog'))
    plt.show()