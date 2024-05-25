
# Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search


## Install Instructions

The `setup.py` file contains all the necessary packages to reproduce the results. The resulting Conda environment takes approximately 8 GB of space. Please ensure that sufficient disk space is available for installation.

```console
$ git clone git@github.com:sjiang87/deephyper.git
$ cd deephyper
$ conda create --name agu python=3.8.16
$ conda activate agu
$ # or source activate agu
$ pip install -e ".[default,analytics]"
```

## Download Data and Results
Please select a disk location and data storage preference based on your needs. When running the program, please update the following directory:
- `--ROOT_DIR`: This is where search and post-training results are saved. The total results can take about 20 GB.
- `--DATA_DIR`: This is where data CSV and pickle files are saved. The total results can take about 5 GB.
- `--RESULT_DIR`: This directory is used to store the analysis results in CSV and pickle formats. The total results can take aboutd 1 GB.
- `--PLOT_DIR`: This directory is used to store figures. The total results can take over 200 MB.

Please download [data](https://drive.google.com/file/d/1O6uH1t2VVyzQJNfRXDGWtzWcxicOjSV5/view?usp=sharing) and [results](https://drive.google.com/file/d/1ujHNHOxwot7uYi9ZdWDDctaZ3hNKmVKU/view?usp=sharing) from Google Drive. An alternative option is to download using `gdown`.

```console
$ conda activate agu
$ pip install gdown
$ cd autognnuq
$ gdown "1O6uH1t2VVyzQJNfRXDGWtzWcxicOjSV5"
$ gdown "1ujHNHOxwot7uYi9ZdWDDctaZ3hNKmVKU"
$ gdown "1SRJZwWUhrVBK5s6ZBpJ2Z54doB-E0BAl"
$ gdown "1Mz87Ovgt9aFrQW6D0Gnk8YDnR9iI1nPl"
$ gdown "1QQlBp0whn-KfyC8pKrz9tUQ9Ny_mp7tS"
$ tar -xzvf data.tar.gz
$ tar -xzvf result.tar.gz
$ tar -xzvf NAS.tar.gz
$ tar -xzvf NAS_simple.tar.gz
$ tar -xzvf post_result.tar.gz
```

```console
$ gdown "1QQlBp0whn-KfyC8pKrz9tUQ9Ny_mp7tS"
$ tar -xzvf post_model.tar.gz
```

Your deephyper folder show look like this
```
deephyper/
├── train.py
├── post_training.py
├── mc_dropout.py
├── ood_pc9.py
├── deephyper/
│ ├── ...
│ └── gnn_uq/
├── autognnuq/
│ ├── data/
│ ├── result/
│ ├── fig/
│ ├── NEW_POST_RESULT/
│ ├── NEW_POST_MODEL/
│ └── ... other_NAS_and_post_training_folders/
```


## Quickstart

Train `AutoGNNUQ` neural architecture search, run one instance
```console
$ python train.py --ROOT_DIR "./autognnuq/" --DATA_DIR "./autognnuq/data/" --SPLIT_TYPE "523" --seed 0 --dataset "delaney" --batch_size 128 --learning_rate 0.001 --epoch 30 --simple 1 --max_eval 1000
```


Train `AutoGNNUQ` for post-training
```console
$ conda activate agu
$ python post_training.py --ROOT_DIR "./autognnuq/" --POST_DIR "./autognnuq/" --DATA_DIR "./autognnuq/data/" --SPLIT_TYPE "523" --seed 0 --dataset "delaney" --batch_size 128 --learning_rate 0.001 --epoch 1000 --mode "normal"
```

Train MC dropout
```console
$ conda activate agu
$ python mc_dropout.py --ROOT_DIR "./autognnuq/" --POST_DIR "./autognnuq/" --DATA_DIR "./autognnuq/data/" --SPLIT_TYPE "523" --seed 0 --dataset "delaney" --batch_size 128 --learning_rate 0.001 --epoch 1000
```

OOD PC9
```console
$ conda activate agu
$ python ood_pc9.py --ROOT_DIR "./autognnuq/" --POST_DIR "./autognnuq/" --DATA_DIR "./autognnuq/data/" --SPLIT_TYPE "811" --seed 0
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



SLURM Script Example for Neural Architecture Search
```bash
#!/bin/bash
#SBATCH --job-name=nas
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

module load python/3.8  # Adjust the module as per your environment
module load cuda/11.2  # Adjust the module as per your environment
conda activate agu
# or source activate agu

ROOT_DIR="./autognnuq/"
DATA_DIR="./autognnuq/data/"
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCH=30
SIMPLE=1
MAX_EVAL=1000

for SEED in {0..7}; do
  for DATASET in "delaney" "freesolv" "lipo" "qm7" "qm9"; do
    if [ "$DATASET" == "qm9" ]; then
      SPLIT_TYPE="811"
    else
      SPLIT_TYPE="523"
    fi
    
    srun --gres=gpu:4 --cpus-per-task=4 -n1 -N1 \
      python train.py --ROOT_DIR "$ROOT_DIR" --DATA_DIR "$DATA_DIR" --SPLIT_TYPE "$SPLIT_TYPE" \
      --seed $SEED --dataset "$DATASET" --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --epoch $EPOCH --simple $SIMPLE --max_eval $MAX_EVAL &
  done
done

wait

```



<!-- The black-box function named `run` is defined by taking an input job named `job` which contains the different variables to optimize `job.parameters`. Then the run-function is bound to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `CBO` is created and executed to find the values of config which **MAXIMIZE** the return value of `run(job)`. -->


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

