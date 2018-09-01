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
#import keras

# Locals
from hepcnn import load_dataset, build_model

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
    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load configuration file
    with open(args.config) as f:
        config = yaml.load(f)
    print(config)

    # Load the data files
    train_data, valid_data, test_data = load_dataset(**config['data_config'])
    train_input, train_labels, train_weights = train_data
    valid_input, valid_labels, valid_weights = valid_data
    test_input, test_labels, test_weights = test_data
    print('train shape:', train_input.shape, 'Mean label:', train_labels.mean())
    print('valid shape:', valid_input.shape, 'Mean label:', valid_labels.mean())
    print('test shape: ', test_input.shape, 'Mean label:', test_labels.mean())

    # Build the model
    model = build_model(train_input.shape[1:], **config['model_config'])
    model.summary()

    # TODO: Add the checkpointing

    # Run the training
    history = model.fit(x=train_input, y=train_labels,
                        validation_data=(valid_input, valid_labels),
                        **config['training_config'])

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
