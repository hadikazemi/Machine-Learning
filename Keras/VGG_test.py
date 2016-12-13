#
# Test a VGG16 model
# Assumed that all images and their labels are saved in a single hdf5 file using the code provided in Tools folder
# namely "hdf5_builder.py"
#
# =============================================================================

#
from keras.optimizers import SGD
from keras_tools import build_vgg_model, image_loader
from keras.utils.visualize_util import plot
import tables
import numpy as np

train_path = 'train.hdf5'     # path to the hdf5 containing the training data (hdf5_builder.py)
test_path = 'test.hdf5'       # path to the hdf5 containing the test data (hdf5_builder.py)
visualize = True              # True if you want to plot the model into an image file
model_img_path = 'vgg.png'    # where to plot the model when visualize = True
batch_size = 40               # batch size
load_path="vgg16-e10.h5"      # path to a h5 file to load the trained model
nb_class = 129                # Number of classes 

# Create a VGG16 model and load the weights
# weight_path=None means load the imagenet weights for all conv layers and FC layers (if possible)
model = build_vgg_model(weight_path=load_path, nb_class=nb_class, nb_fc=[512, 512])

# Stochastic Gradient Decent optimizer
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

# set the model's optimizer, loss function and metrics
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

# Plot the model into an image file
if visualize:
    plot(model, to_file=model_img_path, show_shapes=True)

# open the train hdf5 file to load the mean 
hdf5_file = tables.open_file(train_path, mode='r')
mm = hdf5_file.root.mean[0]
mm = mm[np.newaxis, ...]
hdf5_file.close()

test_acc = 0       # training accuracy in current epoch 
sample_num = 0      # total number of training sample

# load test data in an asynchronous fashion  
for X_train, Y_train in image_loader(batch_size, test_path, nb_class, is_training=False):
    X_train -= mm       # remove the mean
    scores = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0], verbose=0, sample_weight=None)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    sample_num += Y_train.shape[0]
    test_acc += scores[1] * 100 * Y_train.shape[0]
    
print 'test accuracy = {}'.format(test_acc/float(sample_num))
    
