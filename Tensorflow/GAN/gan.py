import matplotlib.pyplot as plt
import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np
import os
import tarfile
import urllib

lr = 0.0002
num_epochs = 200
batch_size = 50
data_path = 'data/cifar-10-batches-py/{}'

# Download weights
if not os.path.isdir('data'):
    os.makedirs('data')
if not os.path.isfile('data/cifar-10-batches-py/data_batch_1'):
    print('Downloading the data ...')
    urllib.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "data/cifar-10-python.tar.gz")
    with tarfile.open('data/cifar-10-python.tar.gz', "r:gz") as tar:
        tar.extractall('data/')
    os.remove('data/cifar-10-python.tar.gz')
    print('Download is complete !')

# unpickle data files
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# Load all the data files and stack them together
dataset_x = unpickle(data_path.format('data_batch_1'))['data']
for i in range(2,6):
    dataset_x = np.vstack((dataset_x, unpickle(data_path.format('data_batch_{}'.format(i)))['data']))
# calculate the number of batches per epoch
batch_per_ep = dataset_x.shape[0]//batch_size


def next_batch(batch_number):
    # a function to load a bacth of images
    batch_x = dataset_x[(batch_number) * batch_size:min([((batch_number) + 1) * batch_size, dataset_x.shape[0]]), :]

    # reshape the sample to a batch of images
    batch_img = batch_x.reshape((-1, 3, 32, 32))/255.0
    batch_img = batch_img.transpose([0, 2, 3, 1])
    return batch_img


def leaky_relu(x, alpha=0.1):
    # Leaky Relu activation function
    m_x = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= alpha * m_x
    return x

# Define the Generator, a simple CNN with 1 fully connected and 4 convolution layers
def generator(inputs, reuse=False):
    with tf.variable_scope('generator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net = lays.fully_connected(inputs, 4*4*256, scope='fc1')
        net = tf.reshape(net, (batch_size, 4, 4, 256))
        net = lays.conv2d_transpose(net, 128, 3, stride=2, scope='conv1', padding='SAME', activation_fn=leaky_relu)
        net = lays.conv2d_transpose(net, 64, 3, stride=2, scope='conv2', padding='SAME', activation_fn=leaky_relu)
        net = lays.conv2d_transpose(net, 64, 3, stride=2, scope='conv3', padding='SAME', activation_fn=leaky_relu)
        net = lays.conv2d(net, 3, 3, scope='conv4', padding='SAME', activation_fn=tf.nn.tanh)
        return net


# Define the Discriminator, a simple CNN with 3 convolution and 2 fully connected layers
def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net = lays.conv2d_transpose(inputs, 64, 3, stride=1, scope='conv1', padding='SAME', activation_fn=leaky_relu)
        net = lays.max_pool2d(net, 2, 2, 'SAME', scope='max1')
        net = lays.conv2d_transpose(net, 128, 3, stride=1, scope='conv2', padding='SAME', activation_fn=leaky_relu)
        net = lays.max_pool2d(net, 2, 2, 'SAME', scope='max2')
        net = lays.conv2d_transpose(net, 256, 3, stride=1, scope='conv3', padding='SAME', activation_fn=leaky_relu)
        net = lays.max_pool2d(net, 2, 2, 'SAME', scope='max3')
        net = tf.reshape(net, (batch_size, 4 * 4 * 256))
        net = lays.fully_connected(net, 128, scope='fc1', activation_fn=leaky_relu)
        net = lays.dropout(net, 0.5)
        net = lays.fully_connected(net, 1, scope='fc2', activation_fn=None)
        return net


images = tf.placeholder(tf.float32, (batch_size, 32, 32, 3))    # input images
z_in = tf.placeholder(tf.float32, (batch_size, 100))            # input noises

# Train the discriminator, it tries to discriminate between real and fake (generated) samples
outputs_real = discriminator(images)
loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs_real), logits=outputs_real))

images_fake = generator(z_in)
outputs_fake = discriminator(images_fake, reuse=True)
loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(outputs_fake), logits=outputs_fake))

loss_d = loss_real + loss_fake  # Calculate the total loss
discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
discrim_train = tf.train.AdamOptimizer(lr).minimize(loss_d, var_list=discrim_tvars)

# Train the generator, it tries to fool the discriminator
with tf.control_dependencies([discrim_train]):
    outputs = discriminator(images_fake, reuse=True)
    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs), logits=outputs))
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    gen_train = tf.train.AdamOptimizer(lr).minimize(loss_g, var_list=gen_tvars)

# Draw samples from the input distribution as a fixed test set
# Can follow how the generator output evolves
test_z = np.random.normal(size=(batch_size, 100))

# initialize the network
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(num_epochs):
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img = next_batch(batch_n)
            train_z = np.random.normal(size=(batch_size, 100))

            _, gl, dl = sess.run([gen_train, loss_g, loss_d], feed_dict={images: batch_img, z_in: train_z})

            if not batch_n%10:
                print('epoch: {} - loss_d: {} - loss_g: {}'.format(ep, dl, gl))

        # Save the test results after each Epoch
        print('Testing ...')
        images_test = sess.run([images_fake], feed_dict={z_in: test_z})[0]

        if not os.path.isdir('results'):
            os.makedirs('results')
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(images_test[i])
            plt.savefig('results/cfar10-gan-e{}.png'.format(ep))
        print('A new test image saved !')

