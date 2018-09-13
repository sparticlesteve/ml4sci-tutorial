"""
This module contains model and training code for the HEP-CNN-lite
RPV classifier.
"""

# Compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# System
import os

# Externals
import h5py
from keras import layers, models, optimizers


def load_file(filename, n_samples):
    """Load one file from the dataset"""
    with h5py.File(filename, 'r') as f:
        data_group = f['all_events']
        data = data_group['hist'][:n_samples][:,:,:,None]
        labels = data_group['y'][:n_samples]
        weights = data_group['weight'][:n_samples]
    return data, labels, weights

def load_dataset(path, n_train=412416, n_valid=137471, n_test=137471):
    """Load the entire dataset"""
    train_file = os.path.join(path, 'train.h5')
    valid_file = os.path.join(path, 'val.h5')
    test_file = os.path.join(path, 'test.h5')
    train_input, train_labels, train_weights = load_file(train_file, n_train)
    valid_input, valid_labels, valid_weights = load_file(valid_file, n_valid)
    test_input, test_labels, test_weights = load_file(test_file, n_test)
    return ((train_input, train_labels, train_weights),
            (valid_input, valid_labels, valid_weights),
            (test_input, test_labels, test_weights))

def build_model(input_shape,
                conv_sizes=[8, 16, 32], fc_sizes=[64],
                dropout=0.5,
                optimizer='Adam', learning_rate=0.001,
                use_horovod=False):
    """Construct the Keras model"""

    # Define the inputs
    inputs = layers.Input(shape=input_shape)
    h = inputs

    # Convolutional layers
    conv_args = dict(kernel_size=(3, 3), activation='relu', padding='same')
    for conv_size in conv_sizes:
        h = layers.Conv2D(conv_size, **conv_args)(h)
        h = layers.MaxPooling2D(pool_size=(2, 2))(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Flatten()(h)

    # Fully connected  layers
    for fc_size in fc_sizes:
        h = layers.Dense(fc_size, activation='relu')(h)
        h = layers.Dropout(dropout)(h)

    # Ouptut layer
    outputs = layers.Dense(1, activation='sigmoid')(h)

    # Construct the optimizer
    opt_dict = dict(Adam=optimizers.Adam,
                    Nadam=optimizers.Nadam,
                    Adadelta=optimizers.Adadelta)
    opt = opt_dict[optimizer](lr=learning_rate)
    if use_horovod:
        import horovod.keras as hvd
        opt = hvd.DistributedOptimizer(opt)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs, name='RPVClassifier')
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
