Image Filtering
---------------

A comprehensive tutorial towards 2D convolution and image filtering (The
first step to understand Convolutional Neural Networks (CNNs))

Introduction
------------

Convolution is one of the most important operations in signal and image
processing. It could operate in 1D (e.g. speech processing), 2D (e.g.
image processing) or 3D (video processing). In this post, we discuss
convolution in 2D spatial which is mostly used in image processing for
feature extraction and is also the core block of Convolutional Neural
Networks (CNNs). Generally, we can consider an image as a matrix whose
elements are numbers between 0 and 255. The size of this matrix is
(image height) x (image width) x (\# of image channels). A grayscale
image has 1 channel where a color image has 3 channels (for an RGB). In
this tutorial we are going to work on a grayscale image shown in Figure
1 and apply different convolution kernels on it.

[![](../../../_images/topics/computer_vision/basics/convolution/image.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/image.jpg)

**Figure 1:** The original grayscale image


If we zoom on the very top-left corner of the image, we can see the
pixels of the image. You can see the pixels on the top-left corner of
the image (first five rows and five columns) and their corresponding
values in Figure 2.

[![](../../../_images/topics/computer_vision/basics/convolution/7.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/7.jpg)

**Figure 2:** The first 5 columns and rows of the image in Figure 1


You can load and plot the image as a **Numpy** array using **skimage**
library in python:


**Note:** *skimage* load grayscale images in \[0-1\] scale instead of
\[0-255\].

```python
from skimage import io, viewer
img = io.imread('image.jpg', as_grey=True)  # load the image as grayscale
print 'image matrix size: ', img.shape      # print the size of image
print '\n First 5 columns and rows of the image matrix: \n', img[:5,:5]*255 
viewer.ImageViewer(img).show()              # plot the image
```

```shell
Image matrix size:  (897, 1168)

 First 5 columns and rows of the image matrix: 
[[ 105.  102.  100.   97.   96.]
 [ 103.   99.  103.  101.  102.]
 [ 101.   98.  104.  102.  100.]
 [  99.  101.  106.  104.   99.]
 [ 104.  104.  104.  100.   98.]]
```

Convolution
-----------

Each convolution operation has a kernel which could be a any matrix
smaller than the original image in height and width. Each kernel is
useful for a spesific task, such as sharpening, blurring, edge
detection, and more. Let's start with the sharpening kernel which is
defined as:

\$Kernel = \\begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 5 & -1 \\\\ 0 & -1 & 0
\\end{bmatrix}\$

You can find a list of most common kernels
[here](https://en.wikipedia.org/wiki/Kernel_(image_processing)). As
previously mentioned, each kernel has a specific task to do and the
sharpen kernel accentuate edges but with the cost of adding noise to
those area of the image which colors are changing gradually. The output
of image convolution is calculated as follows:

-   Put the center of the kernel at every pixel of the image (element of
    the image matrix). Then each element of the kernel will stand on top
    of an element of the image matrix.
-   Multiply each element of the kernel with its corresponding element
    of the image matrix (the one which is overlapped with it)
-   Sum up all product outputs and put the result at the same position
    in the output matrix as the center of kernel in image matrix.
-   For the pixels on the border of image matrix, some elements of the
    kernel might stands out of the image matrix and therefore does not
    have any corresponding element from the image matrix. In this case,
    you can eliminate the convolution operation for these position which
    end up an output matrix smaller than the input (image matrix) or we
    can apply padding to the input matrix (based on the size of the
    kernel we might need one or more pixels padding, in our example we
    just need 1 pixel padding):

As you can see in Figure 5, the output of convolution might violate the
input range of \[0-255\]. Even though the python packages would take
care of it by considering the maximum value of the image as the pure
white (correspond to 255 in \[0-255\] scale) and the minimum value as
the pure black (correspond to 0 in \[0-255\] scale), the values of the
convolution output (filtered image) specially along the edges of the
image (which are calculated based on the added zero padding) can cause a
low contrast filtered image. In this post, to overcome this loss of
contrast issue, we use [Histogram Equalization]{.inner_shadow}
technique. However, you might be able to end up with a better contrast
neglecting the zero padding. The following python code convolves an
image with the sharpen kernel and plots the result:


```python
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pylab

def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    output = np.zeros_like(image)  # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output

img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

# Adjust the contrast of the image by applying Histogram Equalization 
image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
plt.imshow(image_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Convolve the sharpen kernel and the image
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
image_sharpen = convolve2d(img,kernel)
print '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255

# Plot the filtered image
plt.imshow(image_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
```

```shell
 First 5 columns and rows of the image_sharpen matrix: 
[[ 320.  206.  198.  188.  182.]
 [ 210.   89.  111.  101.  112.]
 [ 205.   85.  111.  101.   94.]
 [ 189.   98.  117.  113.   91.]
 [ 217.  108.  112.   95.   85.]]
```

and you can see the filtered image after applying **sharpen** filter in
Figure 6 and the filtered image after Histogram Equalization in Figure
7.

[![](../../../_images/topics/computer_vision/basics/convolution/sharpen.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/sharpen.jpg)
**Figure 6:** Sharpened image


[![](../../../_images/topics/computer_vision/basics/convolution/sharpen_eq.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/sharpen_eq.jpg)
**Figure 7:** Sharpened image after Histogram Equalization

So far, we have been using our own convolution function which was not
written to be efficient. Hopefully, you can easily find well written
functions for 1D, 2D, and 3D convolutions in most of the python packages
which are related to machine learning and image processing. Here is our
previous code but using [Scipy]{.inner_shadow} or
[OpenCV]{.inner_shadow} built-in functions.

```python
import numpy as np
import scipy
from skimage import io, color
from skimage import exposure
import matplotlib.pyplot as plt

img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

# you can use 'valid' instead of 'same', then it will not add zero padding
image_sharpen = scipy.signal.convolve2d(img, kernel, 'same')
print '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure

img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

image_sharpen = cv2.filter2D(img,-1,kernel)
print '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
```

More Filters
------------

There are many other filters which are really useful in image processing
and computer vision. One of the most important one is edge detection.
Edge detection aims to identify pixels of an image at which the
brightness changes drastically. Let's apply one of the simplest edge
detection filters to our image and see the result. Here is the kernel:

\$Kernel = \\begin{bmatrix} -1 & -1 & -1 \\\\ -1 & 8 & -1 \\\\ -1 & -1 &
-1 \\end{bmatrix}\$

and here is the python code:

```python
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure

img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# we use 'valid' which means we do not add zero padding to our image
edges = scipy.signal.convolve2d(img, kernel, 'valid')
print '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
```

and here is what you will see when you run the code:


[![](../../../_images/topics/computer_vision/basics/convolution/edges.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/edges.jpg)
**Figure 7:** Filtered image

What about if we apply the edge detection kernel to the output of
sharpen filter? Let's have a look at it:

```python
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure

img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

# apply sharpen filter to the original image
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
image_sharpen = scipy.signal.convolve2d(img, sharpen_kernel, 'valid')

edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edges = scipy.signal.convolve2d(image_sharpen, edge_kernel, 'valid')

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)

plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
```

[![](../../../_images/topics/computer_vision/basics/convolution/edge2.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/edge2.jpg)
<div class="desc">
**Figure 8:** Filtered image

As we mentioned before, sharpen filter bolds the edges but with the cost
of adding noise to the image. You can clearly see these effects
comparing Figure 8 and Figure 7. Now it's time to apply a filter to the
noisy image and reduce the noise. Blur filter could be a smart choise:

\$Kernel = \\dfrac{1}{9}\\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & 1 & 1 \\\\ 1
& 1 & 1 \\end{bmatrix}\$

```python
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure

img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

# apply sharpen filter to the original image
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
image_sharpen = scipy.signal.convolve2d(img, sharpen_kernel, 'valid')

# apply edge detection filter to the sharpen image
edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
edges = scipy.signal.convolve2d(image_sharpen, edge_kernel, 'valid')

# apply blur filter to the edge detection filtered image
blur_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0;
denoised = scipy.signal.convolve2d(edges, blur_kernel, 'valid')

# Adjust the contrast of the filtered image by applying Histogram Equalization
denoised_equalized = exposure.equalize_adapthist(denoised/np.max(np.abs(denoised)), clip_limit=0.03)

plt.imshow(denoised_equalized, cmap=plt.cm.gray)    # plot the denoised_clipped
plt.axis('off')
plt.show()
```

[![](../../../_images/topics/computer_vision/basics/convolution/blur.jpg){width="600"
height="400"}](../../../_images/topics/computer_vision/basics/convolution/blur.jpg)
**Figure 9:** Denoised Image

\
[Go Top](#post_top)\
\
