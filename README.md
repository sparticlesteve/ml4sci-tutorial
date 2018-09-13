# ML4Sci / Data Day Keras hands-on tutorial code.
Keras tutorial code for the ML4Sci workshop at LBNL.
This is a Keras version of the code in
https://github.com/MustafaMustafa/data-day-2018-DL-Scaling

It should serve as a handy reference for distributed training with horovod
and Keras on the Cori system at NERSC.

## To run

The provided batchScript.sh should provide all you need.
To run on 4 Cori Haswell nodes, submit it with:
```
    sbatch -N 4 ./batchScript.sh
```
