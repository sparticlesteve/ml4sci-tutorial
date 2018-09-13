"""
Distributed data-parallel training of the HEP-CNN-lite RPV Classifier.
"""

# Compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# System
import os
import socket
import argparse
import logging

# Externals
import yaml
import keras
import horovod.keras as hvd

# Locals
from hepcnn import load_dataset, build_model
from utils import configure_session

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('runTraining.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='config.yaml')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def main():
    """Main function"""

    # Initialize horovod
    hvd.init()

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    logging.info('MPI rank %i, local rank %i, host %s' %
                 (hvd.rank(), hvd.local_rank(), socket.gethostname()))

    # Load configuration file
    with open(args.config) as f:
        config = yaml.load(f)
    logging.info('Configuration: %s' % config)

    # Load the data files
    train_data, valid_data, test_data = load_dataset(**config['data_config'])
    train_input, train_labels, train_weights = train_data
    valid_input, valid_labels, valid_weights = valid_data
    test_input, test_labels, test_weights = test_data
    logging.info('train shape: %s Mean label %s' % (train_input.shape, train_labels.mean()))
    logging.info('valid shape: %s Mean label %s' % (valid_input.shape, valid_labels.mean()))
    logging.info('test shape:  %s Mean label %s' % (test_input.shape, test_labels.mean()))

    # Configure the session (e.g. thread settings)
    keras.backend.set_session(
        configure_session(**config['session_config'])
    )

    # Scale the learning rate
    model_config = config['model_config']
    if model_config.pop('scale_learning_rate'):
        model_config['learning_rate'] = model_config['learning_rate'] * hvd.size()

    # Build the model
    logging.info(config)
    model = build_model(train_input.shape[1:],
                        use_horovod=True,
                        **model_config)
    if hvd.rank() == 0:
        model.summary()

    # Training hooks
    callbacks = []

    # Horovod model synchronization during initialization
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    # Model checkpointing
    if hvd.rank() == 0:
        checkpoint_file = os.path.expandvars(config['checkpoint_file'])
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_file))

    # Batch size
    training_config = config['training_config']
    bsize = training_config['batch_size']
    per_node = training_config.pop('batch_size_per_node')
    training_config['batch_size'] = bsize if per_node else (bsize // hvd.size())

    # Run the training
    logging.info('Final training config: %s' % training_config)
    history = model.fit(x=train_input, y=train_labels,
                        validation_data=(valid_input, valid_labels),
                        callbacks=callbacks, verbose=2,
                        **training_config)

    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(test_input, test_labels, verbose=2)
    logging.info('Test loss:     %g' % test_loss)
    logging.info('Test accuracy: %g' % test_acc)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
