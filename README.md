# cifar-10
Test on the CIFAR-10 dataset to have the best accuracy with the least number of parameters and/or operations. Done in the context of the Efficient Deep Learning course from IMT Atlantique

# How to run
Install the source code as a package running
```bash
pip install -e .
```
`script.sh` contain some examples of command line to run in order to train a model.

## Train a model from scra ch
TO BE COMPLETED

## Prune a model
TO BE COMPLETED

## Test a model using post training 8-bit quantization
TO BE COMPLETED

# Known Issues
The `train.py` args could be better written using FACTORY. Kwargs could be used for the trainer class to make it more clean and more modulable

Post training 8-bit quantization only works with CPU but that's PyTorch problem not me.
