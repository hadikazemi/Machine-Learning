from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.examples.tutorials.mnist import input_data
import torch.optim as optim

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
batch_size = 500  # Number of samples in each batch
epoch_num = 15    # Number of epochs to train the network
lr = 0.001        # Learning rate


def next_batch(train=True):
    # A function to read the next batch of MNIST images and labels
    # Args:
    #   train: a boolean array, if True it will return the next train batch, otherwise the next test batch
    # Returns:
    #   batch_img: a pytorch Variable of size [batch_size, 748].
    #   batch_label: a pytorch Variable of size [batch_size, ].

    if train:
        batch_img, batch_label = mnist.train.next_batch(batch_size)
    else:
        batch_img, batch_label = mnist.test.next_batch(batch_size)

    batch_label = torch.from_numpy(batch_label).long()  # convert the numpy array into torch tensor
    batch_label = Variable(batch_label).cuda()          # create a torch variable and transfer it into GPU

    batch_img = torch.from_numpy(batch_img).float()     # convert the numpy array into torch tensor
    batch_img = Variable(batch_img).cuda()              # create a torch variable and transfer it into GPU
    return batch_img, batch_label


class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_features, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_classes)

    def forward(self, x, training=True):
        # a neural network with 2 hidden layers
        # x -> FC -> relu -> dropout -> FC -> relu -> dropout -> FC -> output
        x = F.relu(self.layer1(x))
        x = F.dropout(x, 0.5, training=training)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, 0.5, training=training)
        x = self.layer3(x)
        return x

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = F.softmax(self.forward(x, training=False))
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs
        #   y: the true labels associated with x
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu().data[0]

# define the neural network (multilayer perceptron) and move the network into GPU
net = MLP(784, 10)
net.cuda()

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

# define the loss (criterion) and create an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

for ep in range(epoch_num):  # epochs loop
    for batch_n in range(batch_per_ep):  # batches loop
        features, labels = next_batch()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = net(features)
        loss = criterion(output, labels)

        # Backward pass and updates
        loss.backward()                 # calculate the gradients (backpropagation)
        optimizer.step()                # update the weights

        if not batch_n%10:
            print('epoch: {} - batch: {}/{} \n'.format(ep, batch_n, batch_per_ep))
            print('loss: ', loss.data[0])

# test the accuracy on a batch of test data
features, labels = next_batch(train=False)
print('\n \n Test accuracy: ', net.accuracy(features, labels))