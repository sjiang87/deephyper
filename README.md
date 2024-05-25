
# Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search

<br />
<img src="./autognnuq/fig/toc.png" />
<br />



## Install Instructions

The `setup.py` file contains all the necessary packages to reproduce the results. The resulting Conda environment takes approximately 8 GB of space. Please ensure that sufficient disk space is available for installation.

```console
$ git clone https://github.com/sjiang87/deephyper.git
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
- `--RESULT_DIR`: This directory is used to store the analysis results in CSV and pickle formats. The total results can take about 1 GB.
- `--PLOT_DIR`: This directory is used to store figures. The total results can take about 200 MB.


Here is the structure of the `deephyper` folder:
```
deephyper/
├── train.py
├── post_training.py
├── mc_dropout.py
├── ood_pc9.py
├── deephyper/
│   ├── ...
│   └── gnn_uq/
├── autognnuq/
│   ├── data/
│   ├── result/
│   ├── fig/
│   ├── NEW_POST_RESULT/
│   ├── NEW_POST_MODEL/
│   └── ... other_NAS_and_post_training_folders/
└── ...
```

Please download [data](https://drive.google.com/file/d/1O6uH1t2VVyzQJNfRXDGWtzWcxicOjSV5/view?usp=sharing), [results](https://drive.google.com/file/d/1ujHNHOxwot7uYi9ZdWDDctaZ3hNKmVKU/view?usp=sharing), [AutoGNNUQ NAS](https://drive.google.com/file/d/1SRJZwWUhrVBK5s6ZBpJ2Z54doB-E0BAl/view?usp=sharing), [AutoGNNUQ-Simple NAS](https://drive.google.com/file/d/1Mz87Ovgt9aFrQW6D0Gnk8YDnR9iI1nPl/view?usp=drive_link) and [post-training results](https://drive.google.com/file/d/1QQlBp0whn-KfyC8pKrz9tUQ9Ny_mp7tS/view?usp=drive_link) from Google Drive. An alternative option is to download using `gdown`.

Here are the necessary downloads to reproduce all results and figures.
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

### Reproduce Results and Figures
Generate result pickle and CSV files from the raw `AutoGNNUQ` outputs.
```console
$ python gen_result.py \
  --ROOT_DIR "./autognnuq/" \
  --RESULT_DIR "./autognnuq/result/" \
  --PLOT_DIR "./autognnuq/fig/" \
  --DATA_DIR "./autognnuq/data/"
```

Generate all figures from the `AutoGNNUQ` result files.
```console
$ python gen_fig.py \
  --ROOT_DIR "./autognnuq/" \
  --RESULT_DIR "./autognnuq/result/" \
  --PLOT_DIR "./autognnuq/fig/" \
  --DATA_DIR "./autognnuq/data/"
```

### Additional Post-Training Weights
Post-training weights can be downloaded [here](https://drive.google.com/file/d/16TnIW2HIS6PtfIsnbDDoE0WGAkddcPT8/view?usp=sharing) (16 GB).

```console
$ gdown "16TnIW2HIS6PtfIsnbDDoE0WGAkddcPT8"
$ tar -xzvf post_model.tar.gz
```

## Train `AutoGNNUQ` From Scratch

The `--dataset` options include: delaney (ESOL), lipo, freesolv, qm7, and qm9. For datasets other than qm9, the `--SPLIT_TYPE` is "523". For the qm9 dataset, the SPLIT_TYPE is "811".

```console
$ python train.py --ROOT_DIR "./autognnuq/" \
  --DATA_DIR "./autognnuq/data/" \
  --SPLIT_TYPE "523" \
  --seed 0 \
  --dataset "delaney" \
  --batch_size 128 \
  --learning_rate 0.001 \
  --epoch 30 \
  --simple 1 \
  --max_eval 1000
```
Full code `bash train.sh`

SLURM Script Example for Neural Architecture Search:
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
module load cuda/11.2   # Adjust the module as per your environment
conda activate agu      # or source activate agu

ROOT_DIR="./autognnuq/"
DATA_DIR="./autognnuq/data/"
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCH=30
MAX_EVAL=1000

for SEED in {0..7}; do
  for DATASET in "delaney" "freesolv" "lipo" "qm7" "qm9"; do
    if [ "$DATASET" == "qm9" ]; then
      SPLIT_TYPE="811"
    else
      SPLIT_TYPE="523"
    fi
    for SIMPLE in 0 1; do
        srun --gres=gpu:4 --cpus-per-task=4 -n1 -N1 \
        python train.py --ROOT_DIR "$ROOT_DIR" --DATA_DIR "$DATA_DIR" --SPLIT_TYPE "$SPLIT_TYPE" \
        --seed $SEED --dataset "$DATASET" --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
        --epoch $EPOCH --simple $SIMPLE --max_eval $MAX_EVAL &
    done
  done
done

wait
```


## Post-training with NAS Already Done

Train `AutoGNNUQ` for post-training by running one instance with the `--mode` parameter set to one of the following options: "normal" for `AutoGNNUQ`, "simple" for `AutoGNNUQ-Simple`, or "random" for `random ensemble`.
```console
$ python post_training.py --ROOT_DIR "./autognnuq/" \
  --POST_DIR "./autognnuq/" \
  --DATA_DIR "./autognnuq/data/" \
  --SPLIT_TYPE "523" \
  --seed 0 \
  --dataset "delaney" \
  --batch_size 128 \
  --learning_rate 0.001 \
  --epoch 1000 \
  --mode "normal"
```
Full code `bash post_training.sh`


Train MC dropout
```console
$ python mc_dropout.py \
  --ROOT_DIR "./autognnuq/" \
  --POST_DIR "./autognnuq/" \
  --DATA_DIR "./autognnuq/data/" \
  --SPLIT_TYPE "523" \
  --seed 0 \
  --dataset "delaney" \
  --batch_size 128 \
  --learning_rate 0.001 \
  --epoch 1000
```
Full code `bash mc_dropout.sh`

PC9 Out-of-distribution analysis
```console
$ python ood_pc9.py \
  --ROOT_DIR "./autognnuq/" \
  --POST_DIR "./autognnuq/" \
  --DATA_DIR "./autognnuq/data/" \
  --SPLIT_TYPE "811" \
  --seed 0
```
Full code `bash ood_pc9.sh`

