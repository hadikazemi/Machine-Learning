#
# Train two Siamese VGG16 models which are coupled and we are trying to minimize the Euclidean distance between the
# feature vectors in the coupled layer.
# Assumed that all images and their labels are saved in a single hdf5 file using the code provided in Tools folder
# namely "hdf5_builder_coupled.py"
#
# =============================================================================

#
from keras.optimizers import SGD
from keras_tools import build_siamese_vgg_model, coupled_image_loader, coupled_loss
from keras.utils.visualize_util import plot

hdf5_path = 'train.hdf5'      # path to the hdf5 containing the training data (hdf5_builder.py)
visualize = True              # True if you want to plot the model into an image file
model_img_path = 'vgg.png'    # where to plot the model when visualize = True
nb_epoch = 20                 # number of epochs
batch_size = 40               # batch size
save_path="vgg16-e{}.h5"      # path to a h5 file to save the trained model after each epoch
nb_class = 129                # Number of classes 
learning_rate_l = 1           # Learning rate of the left branch
learning_rate_r = 1           # Learning rate of the right branch
learning_rate_c = 0.1         # Learning rate of the coupled layer

# Create two Siamese VGG16 models and load the weights (coupled on their flatten layers)
# weight_path=None means load the imagenet weights for all conv layers and FC layers (if possible)
model = build_siamese_vgg_model(weight_path=None, nb_class=nb_class, nb_fc=[512, 512])

# Stochastic Gradient Decent optimizer
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

# set the model's optimizer, loss function and metrics
model.compile(optimizer=sgd,
                   loss={"predictions_l": 'categorical_crossentropy', "predictions_r": 'categorical_crossentropy', 'coupled_loss': coupled_loss},
                   loss_weights={"predictions_l": learning_rate_l, "predictions_r": learning_rate_r, 'coupled_loss': learning_rate_c}, metrics = ['accuracy'])

# Plot the model into an image file
if visualize:
    plot(model, to_file=model_img_path, show_shapes=True)

# loop over all epochs
for e in range(nb_epoch):
    print("epoch %d" % e)
    
    # load training data in an asynchronous fashion  
    for X_train, Y_train in coupled_image_loader(batch_size, hdf5_path, nb_class):
        history = model.fit(X_train, Y_train, batch_size=Y_train[0].shape[0], nb_epoch=1, verbose=1, validation_split=0.0, validation_data=None)
    
    # save model weights at the end of each epoch
    model.save_weights(save_path.format(e))
    print "Model is saved in {}".format(save_path.format(e))
