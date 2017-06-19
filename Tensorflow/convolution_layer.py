import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from skimage import exposure
from math import ceil


def convolution2d(conv_input, conv_kernel, bias, strides=(1, 1), padding='same'):
    # This function which takes an input (Tensor) and a kernel (Tensor)
    # and returns the convolution of them
    # Args:
    #   conv_input: a numpy array of size [input_height, input_width, input # of channels].
    #   conv_kernel: a numpy array of size [kernel_height, kernel_width, input # of channels, output # of channels]
    #                represents the kernel of the Convolutional Layer's filter.
    #   bias: a numpy array of size [output # of channels], represents the bias of the Convolutional Layer's filter.
    #   strides: a tuple of (convolution vertical stride, convolution horizontal stride).
    #   padding: type of the padding scheme: 'same' or 'valid'.
    # Returns:
    #   a numpy array (convolution output).

    assert len(conv_kernel.shape) == 4, "The size of kernel should be (kernel_height, kernel_width, input # of channels, output # of channels)"
    assert len(conv_input.shape) == 3, "The size of input should be (input_height, input_width, input # of channels)"
    assert conv_kernel.shape[2] == conv_input.shape[2], "the input and the kernel should have the same depth."

    input_w, input_h = conv_input.shape[1], conv_input.shape[0]      # input_width and input_height
    kernel_w, kernel_h = conv_kernel.shape[1], conv_kernel.shape[0]  # kernel_width and kernel_height
    output_depth = conv_kernel.shape[3]

    if padding == 'same':
        output_height = int(ceil(float(input_h) / float(strides[0])))
        output_width = int(ceil(float(input_w) / float(strides[1])))

        # Calculate the number of zeros which are needed to add as padding
        pad_along_height = max((output_height - 1) * strides[0] + kernel_h - input_h, 0)
        pad_along_width = max((output_width - 1) * strides[1] + kernel_w - input_w, 0)
        pad_top = pad_along_height // 2             # amount of zero padding on the top
        pad_bottom = pad_along_height - pad_top     # amount of zero padding on the bottom
        pad_left = pad_along_width // 2             # amount of zero padding on the left
        pad_right = pad_along_width - pad_left      # amount of zero padding on the right

        output = np.zeros((output_height, output_width, output_depth))  # convolution output

        # Add zero padding to the input image
        image_padded = np.zeros((conv_input.shape[0] + pad_along_height,
                                 conv_input.shape[1] + pad_along_width, conv_input.shape[2]))
        image_padded[pad_top:-pad_bottom, pad_left:-pad_right, :] = conv_input

        for ch in range(output_depth):
            for x in range(output_width):  # Loop over every pixel of the output
                for y in range(output_height):
                    # element-wise multiplication of the kernel and the image
                    output[y, x, ch] = (conv_kernel[..., ch] * image_padded[y * strides[0]:y * strides[0] + kernel_h,
                                    x * strides[1]:x * strides[1] + kernel_w, :]).sum() + bias[ch]

    elif padding == 'valid':
        output_height = int(ceil(float(input_h - kernel_h + 1) / float(strides[0])))
        output_width = int(ceil(float(input_w - kernel_w + 1) / float(strides[1])))

        output = np.zeros((output_height, output_width, output_depth))  # convolution output

        for ch in range(output_depth):
            for x in range(output_width):  # Loop over every pixel of the output
                for y in range(output_height):
                    # element-wise multiplication of the kernel and the image
                    output[y, x, ch] = (conv_kernel[..., ch] * conv_input[y * strides[0]:y * strides[0] + kernel_h,
                                    x * strides[1]:x * strides[1] + kernel_w, :]).sum() + bias[ch]

    return output

# load the image
img = misc.imread('images/statue.jpg', mode='RGB')

# The edge detection kernel
kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])[..., None]
kernel1 = np.repeat(kernel1, 3, axis=2)

kernel2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])[..., None]/9.0
kernel2 = np.repeat(kernel2, 3, axis=2)

kernel = np.zeros_like(kernel1, dtype=np.float)[..., None]
kernel = np.repeat(kernel, 2, axis=3)
kernel[..., 0] = kernel1
kernel[..., 1] = kernel2

# Convolve image and kernel
image_edges = convolution2d(img, kernel, bias=[1, 0], padding='valid')

# Adjust the contrast and plot the first channel of the output
image_sharpen_equalized = exposure.equalize_adapthist(image_edges[..., 0] / np.max(np.abs(image_edges[..., 0])),
                                                      clip_limit=0.03)
plt.figure(1)

# Plot the first channel of the output
plt.subplot(221)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')

# Plot the second channel of the output
plt.subplot(222)
plt.imshow(image_edges[..., 1], cmap=plt.cm.gray)
plt.axis('off')

# Plot the input
plt.subplot(223)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
