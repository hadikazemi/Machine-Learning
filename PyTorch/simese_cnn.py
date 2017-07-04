from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
batch_size = 500   # Number of samples in each batch
epoch_num = 10      # Number of epochs to train the network
lr = 0.0005        # Learning rate

x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels
print('# of training data: ', x_train.shape[0])

def next_pair(batch_number, is_train=True):
    # A function to read the next pair of MNIST images and labels
    # Args:
    #   batch_number: indicates which batch to load.
    #   is_train: indicates whether load the batch from train or test dataset
    # Returns:
    #   batch_img_1: a pytorch Variable of size [batch_size, 1, 28, 28].
    #   batch_img_2: a pytorch Variable of size [batch_size, 1, 28, 28].
    #   batch_label_1: a pytorch Variable of size [batch_size, ].
    #   batch_label_2: a pytorch Variable of size [batch_size, ].
    #   batch_label_c: a pytorch Variable of size [batch_size, ].

    if is_train:
        dataset_x = x_train
        dataset_y = y_train
    else:
        dataset_x = x_test
        dataset_y = y_test

    batch_x = dataset_x[batch_number*batch_size:min([(batch_number+1)*batch_size, dataset_x.shape[0]]), :]
    batch_y = dataset_y[batch_number*batch_size:min([(batch_number+1)*batch_size, dataset_y.shape[0]])]

    batch_img_1 = np.zeros((2 * batch_size, 784))   # image set 1
    batch_img_2 = np.zeros((2 * batch_size, 784))   # image set 2
    batch_label_1 = np.zeros((2 * batch_size, ))    # labels for image set 1
    batch_label_2 = np.zeros((2 * batch_size,))     # labels for image set 2
    batch_label_c = np.zeros((2 * batch_size,))     # contrastive label: 0 if genuine pair, 1 if impostor pair
    for i in range(batch_size):
        # find and add a genuine sample
        l = batch_y[i]
        ind_g = np.squeeze(np.argwhere(dataset_y == l))
        batch_img_1[2*i] = batch_x[i]
        batch_img_2[2*i] = dataset_x[random.sample(ind_g, 1)]
        batch_label_1[2*i] = l
        batch_label_2[2*i] = l
        batch_label_c[2*i] = 0

        # find and add an impostor sample
        ind_d = np.squeeze(np.argwhere(dataset_y != l))
        i_d = random.sample(ind_d, 1)
        batch_img_1[2*i+1] = batch_x[i]
        batch_img_2[2*i+1] = dataset_x[i_d]
        batch_label_1[2*i+1] = l
        batch_label_2[2*i+1] = dataset_y[i_d]
        batch_label_c[2*i+1] = 1

    # reshape the sample to a batch of images in pytorch order (batch, channels, height, width)
    batch_img_1 = batch_img_1.reshape((-1, 1, 28, 28))
    batch_img_2 = batch_img_2.reshape((-1, 1, 28, 28))

    batch_label_1 = torch.from_numpy(batch_label_1).long()  # convert the numpy array into torch tensor
    batch_label_1 = Variable(batch_label_1).cuda()          # create a torch variable and transfer it into GPU

    batch_label_2 = torch.from_numpy(batch_label_2).long()  # convert the numpy array into torch tensor
    batch_label_2 = Variable(batch_label_2).cuda()          # create a torch variable and transfer it into GPU

    batch_label_c = batch_label_c.reshape((-1, 1))
    batch_label_c = torch.from_numpy(batch_label_c).float()  # convert the numpy array into torch tensor
    batch_label_c = Variable(batch_label_c).cuda()           # create a torch variable and transfer it into GPU

    batch_img_1 = torch.from_numpy(batch_img_1).float()     # convert the numpy array into torch tensor
    batch_img_1 = Variable(batch_img_1).cuda()              # create a torch variable and transfer it into GPU

    batch_img_2 = torch.from_numpy(batch_img_2).float()  # convert the numpy array into torch tensor
    batch_img_2 = Variable(batch_img_2).cuda()           # create a torch variable and transfer it into GPU

    return batch_img_1, batch_img_2, batch_label_1, batch_label_2, batch_label_c


class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 64, 7, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64)
        return x


# define the CNN and move the network into GPU
cnn = CNN(10)
cnn.cuda()

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

# define the loss (criterion) and create an optimizer
optimizer = optim.Adam(cnn.parameters(), lr=lr)

loss_log = []
for ep in range(epoch_num):  # epochs loop
    for batch_n in range(batch_per_ep):  # batches loop
        batch_img_1, batch_img_2, batch_label_1, batch_label_2, batch_label_c = next_pair(batch_n)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        features_1 = cnn(batch_img_1)
        features_2 = cnn(batch_img_2)

        euclidean_distance = F.pairwise_distance(features_1, features_2)
        loss_contrastive = torch.mean((1 - batch_label_c) * torch.pow(euclidean_distance, 2) +
                                      batch_label_c * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))

        if batch_n%10 == 0:
            loss_log.append(loss_contrastive.data[0])
            print('\nepoch: {} - batch: {}/{}'.format(ep, batch_n, batch_per_ep))
            print('Contrastive loss: ', loss_contrastive.data[0])

        # Backward pass and updates
        loss_contrastive.backward()                     # calculate the gradients (backpropagation)
        optimizer.step()                    # update the weights

torch.save(cnn.state_dict(), './siamese.pth')

import matplotlib.pyplot as plt
plt.plot(loss_log)
plt.title('Contrastive Loss')
plt.show()


