"""
Utility functions for sessions.
"""

# System
import os

# Externals
import tensorflow as tf

def configure_session(n_inter_threads=2, n_intra_threads=32):
    """Configure TensorFlow session with thread settings."""
    config = tf.ConfigProto(
        inter_op_parallelism_threads=n_inter_threads,
        intra_op_parallelism_threads=n_intra_threads
    )
    os.environ['OMP_NUM_THREADS'] = str(n_intra_threads)
    return tf.Session(config=config)
