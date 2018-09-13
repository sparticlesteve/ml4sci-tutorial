# ML4Sci / Data Day Keras hands-on tutorial code.
Keras tutorial code for the ML4Sci workshop at LBNL.
This is a Keras version of the code in
https://github.com/MustafaMustafa/data-day-2018-DL-Scaling

It should serve as a handy reference for distributed training with horovod
and Keras on the Cori system at NERSC.

## To run

The provided batchScript.sh should provide all you need.
To run on 4 Cori Haswell nodes, submit it with:

`
sbatch -N 4 ./batchScript.sh
`

Here are some additional details you can use to customize.

The configuration options are set in config.yaml. This includes layer sizes
for the network, number of samples for training, validation, testing,
batch size, learning rate, etc.

An example environment is setup in setup.sh. This uses the 1.9.0 TensorFlow
installation on Cori which may have everything you need.
