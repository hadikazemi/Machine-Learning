import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import os

lr = 0.0002
test_num = 9
num_epochs = 200
batch_size = 50

# Define a set of transforms to be applied on the training images
# Convert it to torch Tensor
transform = transforms.Compose([
        transforms.ToTensor()
])

# Download and Load the CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_batches = len(train_loader)     # Number of batches per Epoch

# Define the Generator, a simple CNN with 1 fully connected and 4 convolution layers
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 4*4*256),
            nn.LeakyReLU()
        )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(-1, 256, 4, 4)
        x = self.cnn(x)
        return x

# Define the Discriminator, a simple CNN with 3 convolution and 2 fully connected layers
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 4*4*256)
        x = self.fc(x)
        return x

discriminator = Discriminator().cuda()
generator = Generator().cuda()

# The classification loss of Discriminator, binary classification, 1 -> real sample, 0 -> fake sample
criterion = nn.BCELoss()

# Define optimizers
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)

# Draw 9 samples from the input distribution as a fixed test set
# Can follow how the generator output evolves
test_z = Variable(torch.randn(test_num, 100).cuda())

for ep in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):

        images = Variable(images.cuda())
        labels_real = Variable(torch.ones(batch_size).cuda())       # Labels for real images - all ones
        labels_fake = Variable(torch.zeros(batch_size).cuda())      # Labels for fake images - all ones

        # Train the discriminator, it tries to discriminate between real and fake (generated) samples
        discriminator.zero_grad()
        outputs = discriminator(images)
        loss_real = criterion(outputs, labels_real)

        z = Variable(torch.randn(batch_size, 100).cuda())
        images_fake = generator(z)
        outputs = discriminator(images_fake.detach())
        loss_fake = criterion(outputs, labels_fake)

        loss_d = loss_real + loss_fake              # Calculate the total loss
        loss_d.backward()                           # Backpropagation
        optimizer_d.step()                          # Update the weights

        # Train the generator, it tries to fool the discriminator
        # Draw samples from the input distribution and pass to generator
        z = Variable(torch.randn(batch_size, 100).cuda())
        images_fake = generator(z)

        # Pass the genrated images to discriminator
        outputs = discriminator(images_fake)

        generator.zero_grad()
        loss_g = criterion(outputs, labels_real)    # Calculate the loss
        loss_g.backward()                           # Backpropagation
        optimizer_g.step()                          # Update the weights

        if not n%10:
            print('epoch: {} - loss_d: {} - loss_g: {}'.format(ep, loss_d.data[0], loss_g.data[0]))

    # Save the test results after each Epoch
    print('Testing ...')
    images_test = generator(test_z)
    images_test = images_test.data.cpu().numpy()
    images_test = np.moveaxis(images_test, 1, -1)
    if not os.path.isdir('results'):
        os.makedirs('results')
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images_test[i])
        plt.savefig('results/cfar10-gan-e{}.png'.format(ep))
    print('A new test image saved !')
