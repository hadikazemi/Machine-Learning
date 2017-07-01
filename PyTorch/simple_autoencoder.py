from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.examples.tutorials.mnist import input_data
from skimage import transform
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate


def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a pytorch Variable of size [batch_size, 1, 32, 32].

    # reshape the sample to a batch of images in pytorch order (batch, channels, height, width)
    imgs = imgs.reshape((-1, 1, 28, 28))

    # resize the images to (32, 32)
    resized_imgs = np.zeros((imgs.shape[0], 1, 32, 32))
    for i in range(imgs.shape[0]):
        resized_imgs[i, 0, ...] = transform.resize(imgs[i, 0,...], (32, 32))

    resized_imgs = torch.from_numpy(resized_imgs).float()       # convert the numpy array into torch tensor
    resized_imgs = Variable(resized_imgs).cuda()                # create a torch variable and transfer it into GPU
    return resized_imgs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 8, 5, stride=4, padding=2)

        # deconv layers: (in_channel size, out_channel size, kernel_size, stride, padding, output_padding)
        self.deconv1 = nn.ConvTranspose2d(8, 16, 5, stride=4, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 32, 5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        # the autoencoder has 3 con layers and 3 deconv layers (transposed conv). All layers but the last have ReLu
        # activation function
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        return x

# define the autoencoder and move the network into GPU
ae = Net()
ae.cuda()

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

# define the loss (criterion) and create an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=lr)

for ep in range(epoch_num):  # epochs loop
    for batch_n in range(batch_per_ep):  # batches loop
        print('epoch: {} - batch: {}/{} \n'.format(ep, batch_n, batch_per_ep))
        batch_img, batch_label = mnist.train.next_batch(batch_size)
        input = resize_batch(batch_img)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = ae(input)
        loss = criterion(output, input)     # calculate the loss
        print('loss: ', loss.data[0])

        # Backward pass and updates
        loss.backward()                     # calculate the gradients (backpropagation)
        optimizer.step()                    # update the weights


# ------ test the trained network ----- #

# read a batch of test data containing 50 samples
batch_img, batch_label = mnist.test.next_batch(50)
input = resize_batch(batch_img)

# pass the test samples to the network and get the reconstructed samples (output of autoencoder)
recon_img = ae(input)

# transfer the outputs and the inputs from GPU to CPU and convert into numpy array
recon_img = recon_img.data.cpu().numpy()
batch_img = input.data.cpu().numpy()

# roll the second axis so the samples follow the matplotlib order (batch, height, width, channels)
# (batch, channels, height, width) --> (batch, height, width, channels)
recon_img = np.moveaxis(recon_img, 1, -1)
batch_img = np.moveaxis(batch_img, 1, -1)

# plot the reconstructed images and their ground truths (inputs)
plt.figure(1)
plt.title('Reconstructed Images')
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(recon_img[i, ..., 0], cmap='gray')
plt.figure(2)
plt.title('Input Images')
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(batch_img[i, ..., 0], cmap='gray')
plt.show()
