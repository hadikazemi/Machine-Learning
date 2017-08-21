import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import os

lr = 0.0002
test_num = 50
num_epochs = 200
batch_size = 50
z_dim = 40

idx = [i for i in range(10) for _ in range(5)]
print idx
for i in range(50):
    print (i // 5) + (i % 5) * 10 + 1

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
            nn.Linear(z_dim+10, 4*4*256),
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

    def forward(self, z, c):
        x = torch.cat((z, c), 1)
        x = self.model(x)
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


# Define the conditional distribution Q(c|X), a simple CNN with 3 convolution and 2 fully connected layers
class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(),
            nn.Linear(4*4*64, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 4*4*64)
        x = self.fc(x)
        return x


def draw_c(size):
    # Draw categorical latent code 'c'
    c = np.random.multinomial(1, 10*[0.1], size=size)
    c = Variable(torch.from_numpy(c.astype('float32')).cuda())
    return c

discriminator = Discriminator().cuda()
generator = Generator().cuda()
q = Q().cuda()

# The classification loss of Discriminator, binary classification, 1 -> real sample, 0 -> fake sample
criterion = nn.BCELoss()

# Define optimizers
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_q = torch.optim.Adam(list(generator.parameters()) + list(q.parameters()), lr=lr)

# Draw 50 samples from the input distribution as a fixed test set (5 sample per 'c' latent code)
# Can follow how the generator output evolves
test_z = Variable(torch.randn(test_num, z_dim).cuda())

for ep in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):

        images = Variable(images.cuda())
        labels_real = Variable(torch.ones(batch_size).cuda())       # Labels for real images - all ones
        labels_fake = Variable(torch.zeros(batch_size).cuda())      # Labels for fake images - all ones

        # Train the discriminator, it tries to discriminate between real and fake (generated) samples
        discriminator.zero_grad()
        outputs = discriminator(images)
        loss_real = criterion(outputs, labels_real)

        z = Variable(torch.randn(batch_size, z_dim).cuda())
        c = draw_c(batch_size)
        images_fake = generator(z, c)
        outputs = discriminator(images_fake.detach())
        loss_fake = criterion(outputs, labels_fake)

        loss_d = loss_real + loss_fake              # Calculate the total loss
        loss_d.backward()                           # Backpropagation
        optimizer_d.step()                          # Update the weights

        # Train the generator, it tries to fool the discriminator
        # Draw samples from the input distribution and pass to generator
        z = Variable(torch.randn(batch_size, z_dim).cuda())
        images_fake = generator(z, c)

        # Pass the genrated images to discriminator
        outputs = discriminator(images_fake)

        generator.zero_grad()
        loss_g = criterion(outputs, labels_real)    # Calculate the loss
        loss_g.backward()                           # Backpropagation
        optimizer_g.step()                          # Update the weights

        # Train the conditional distribution Q(c|X)
        # We maximize mutual information between c and G(z,c)
        q.zero_grad()
        images_fake = generator(z, c)   # Generate a fake image
        Q_c_given_x = q(images_fake)    # Gives the latent code 'c' given the generated image
        cross_entropy = torch.mean(-torch.sum(c * torch.log(Q_c_given_x + 1e-8), dim=1))
        entropy = torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1))                    # Entropy of the prior, H(c)
        loss_q = cross_entropy + entropy
        loss_q.backward()
        optimizer_q.step()

        if not n%10:
            print('epoch: {} - loss_d: {} - loss_g: {} - loss_q: {}'.format(ep, loss_d.data[0], loss_g.data[0], loss_q.data[0]))

    # Save the test results after each Epoch
    print('Testing ...')
    idx = [i for i in range(10) for _ in range(5)]
    c = np.zeros([50, 10])
    c[range(50), idx] = 1
    c = Variable(torch.from_numpy(c.astype('float32')).cuda())
    images_test = generator(test_z, c).data.cpu().numpy()

    images_test = np.moveaxis(images_test, 1, -1)
    if not os.path.isdir('results'):
        os.makedirs('results')

    plt.figure(num=None, figsize=(6, 8), dpi=100)
    for i in range(50):
        plt.subplot(5, 10, (i//5)+(i%5)*10+1)
        plt.axis('off')
        plt.imshow(images_test[i])
        plt.savefig('results/cfar10-info-gan-e{}.png'.format(ep))
    plt.clf()
    plt.close()
    print('A new test image saved !')
