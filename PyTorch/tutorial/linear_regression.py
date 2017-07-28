from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.datasets import load_boston
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

lr = 0.01        # Learning rate
epoch_num = 10000

boston = load_boston()

features = torch.from_numpy(np.array(boston.data)).float()  # convert the numpy array into torch tensor
features = Variable(features).cuda()                        # create a torch variable and transfer it into GPU

labels = torch.from_numpy(np.array(boston.target)).float()  # convert the numpy array into torch tensor
labels = Variable(labels).cuda()                            # create a torch variable and transfer it into GPU

linear_regression = nn.Linear(13, 1)
linear_regression.cuda()

# define the loss (criterion) and create an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(linear_regression.parameters(), lr=lr)

for ep in range(epoch_num):  # epochs loop
    # Reset gradients
    linear_regression.zero_grad()

    # Forward pass
    output = linear_regression(features)
    loss = criterion(output, labels)        # calculate the loss
    if not ep%500:
        print('Epoch: {} - loss: {}'.format(ep, loss.data[0]))

    # Backward pass and updates
    loss.backward()                         # calculate the gradients (backpropagation)
    optimizer.step()                        # update the weights

output = output.data.cpu().numpy()
labels = np.array(boston.target)

fig, ax = plt.subplots()
ax.scatter(labels, output)
ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

