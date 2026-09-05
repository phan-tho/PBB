# PAC-Bayes with Backprop (PBB)

## Matched MNIST / CIFAR benchmarks (Kaggle)

The fork adds `python -m pbb.benchmark` for the original PBB MNIST CNN and
standard WRN-28-4 on CIFAR-10/100, with a 50/50 learned-prior split,
Gaussian parameter posteriors, and two-GPU DataParallel support.
See **[BENCHMARKS.md](BENCHMARKS.md)** for copy/paste Kaggle commands,
presets, certificate accounting, outputs, and local validation.

Use `requirements-benchmark.txt` with Kaggle's existing PyTorch installation.
The original `running_example.py` and historical conda `requirements.txt`
below remain for reference; they are not the matched benchmark entry point.

This repository includes the code associated to the paper: 

"Tighter risk certificates for neural networks", M. Perez-Ortiz, O. Rivasplata, J. Shawe-Taylor and C. Szepesvari, 2020. 

There are a few examples of how to run the code in the running_example.py script. 

Additional functionalities will be added in subsequent versions.

To create a conda environment to run these experiments in, simply run:

``` bash
$ conda create --name <envname> --file requirements.txt
```

If you have any questions don't hesitate to contact me!
