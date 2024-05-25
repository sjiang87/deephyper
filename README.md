<!-- <p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![DOI](https://zenodo.org/badge/156403341.svg)](https://zenodo.org/badge/latestdoi/156403341)
![GitHub tag (latest by date)](https://img.shields.io/github/tag-date/deephyper/deephyper.svg?label=version)
[![Documentation Status](https://readthedocs.org/projects/deephyper/badge/?version=latest)](https://deephyper.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/deephyper.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deephyper.svg?label=Pypi%20downloads)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb) -->
<!-- [![Build Status](https://travis-ci.com/deephyper/deephyper.svg?branch=develop)](https://travis-ci.com/deephyper/deephyper) -->


## Install Instructions

```console
$ git clone git@github.com:sjiang87/deephyper.git
$ cd deephyper
$ conda create --name agu python=3.8.16
$ conda activate agu
$ pip install -e ".[default,analytics]"
```


## Quickstart

Train `AutoGNNUQ` neural architecture search
```console
$ conda activate agu
$ python train.py --ROOT_DIR "./autognnuq/" --DATA_DIR "/scratch/gpfs/sj0161/autognnuq/data/" --SPLIT_TYPE "523" --seed 0 --dataset "delaney" --batch_size 128 --learning_rate 0.001 --epoch 30 --simple 1 --max_eval 1000
```

Train `AutoGNNUQ` for post-training
```console
$ conda activate agu
$ python post_training.py --ROOT_DIR "/scratch/gpfs/sj0161/autognnuq/" --POST_DIR "./autognnuq/" --DATA_DIR "/scratch/gpfs/sj0161/autognnuq/data/" --SPLIT_TYPE "523" --seed 0 --dataset "delaney" --batch_size 128 --learning_rate 0.001 --epoch 1000 --mode "normal"
```

Train MC dropout
```console
$ conda activate agu
$ python mc_dropout.py --ROOT_DIR "/scratch/gpfs/sj0161/autognnuq/" --POST_DIR "./autognnuq/" --DATA_DIR "/scratch/gpfs/sj0161/autognnuq/data/" --SPLIT_TYPE "523" --seed 0 --dataset "delaney" --batch_size 128 --learning_rate 0.001 --epoch 1000
```

OOD PC9
```console
$ conda activate agu
$ python ood_pc9.py --ROOT_DIR "/scratch/gpfs/sj0161/autognnuq/" --POST_DIR "./autognnuq/" --DATA_DIR "/scratch/gpfs/sj0161/autognnuq/data/" --SPLIT_TYPE "811" --seed 0
```

Generate all figures from `AutoGNNUQ` result files.
```console
$ conda activate agu
$ python gen_fig.py --ROOT_DIR "./autognnuq/" --RESULT_DIR "./autognnuq/result/" --PLOT_DIR "./autognnuq/fig/" --DATA_DIR "./autognnuq/data/"
```

Generate result pickle csv files from raw `AutoGNNUQ` outputs.
```console
$ conda activate agu
$ python gen_result.py --ROOT_DIR "./autognnuq/" --RESULT_DIR "./autognnuq/result/" --PLOT_DIR "./autognnuq/fig/" --DATA_DIR "./autognnuq/data/"
```

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb)

The black-box function named `run` is defined by taking an input job named `job` which contains the different variables to optimize `job.parameters`. Then the run-function is bound to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `CBO` is created and executed to find the values of config which **MAXIMIZE** the return value of `run(job)`. -->


<!-- ## How do I learn more?

* Documentation: <https://deephyper.readthedocs.io>

* GitHub repository: <https://github.com/deephyper/deephyper>

* Blog: <https://deephyper.github.io>

## Contributions

Find the list of contributors on the [DeepHyper Authors](https://deephyper.github.io/aboutus) page of the Documentation.

## Citing DeepHyper

If you wish to cite the Software, please use the following:

```
@misc{deephyper_software,
    title = {"DeepHyper: A Python Package for Scalable Neural Architecture and Hyperparameter Search"},
    author = {Balaprakash, Prasanna and Egele, Romain and Salim, Misha and Maulik, Romit and Vishwanath, Venkat and Wild, Stefan and others},
    organization = {DeepHyper Team},
    year = 2018,
    url = {https://github.com/deephyper/deephyper}
} 
```

Find all our publications on the [Research & Publication](https://deephyper.github.io/papers) page of the Documentation. -->

