# Keras

**VGG_train.py**: Fine-tune a VGG16 model

**VGG_coupled_train.py**: Train two VGG16 models which are coupled and we are trying to minimize the Euclidean distance between the feature vectors in the coupled layer.

**VGG_contrastive_train.py**: Train two VGG16 models which are coupled and we are trying to minimize the contrastive loss between the feature vectors in the coupled layer.

**VGG_siamese_train.py**: Train two Siamese VGG16 models which are coupled and we are trying to minimize the Euclidean distance between the feature vectors in the coupled layer.

**VGG_test.py**: Test a VGG16 model

**Note:** For all files it is assumed that all images and their labels are saved in a single hdf5 file using the code provided in Tools folder namely "hdf5_builder.py"
