# PyTorch Dynamic Batching Benchmark

This repository represents a benchmark for dyanamic batching algorithm [1] implemented in PyTorch. The implementation for the dynamic batching algorithm can be found at https://github.com/Devin-Taylor/pytorch-tools.

The benchmark implements the SPINN model for entailment classification using the SNLI dataset. The aim of the benchmark is to investigate speed-up in inference time when using dynamic batching as opposed to manual batching in PyTorch.

## Usage

pipenv was used as the package manager so simply run:

> pipenv install

For using the library run:

> pipenv run python benchmark.py --gpu [True/False] --dynamic [True/False]

## References

[1] Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, and Peter Norvig. Deep learning with dynamic computation graphs. arXiv preprint arXiv:1702.02181, 2017.