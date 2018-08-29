Directory structure 
===================
```
benchmarks
    directory for problems
experiments
    directory for saving the running the experiments and storing the results
search
    directory for source files
```
Install instructions
====================

With anaconda do the following:

```
conda create -n dl-hps python=3
source activate dl-hps
conda install h5py
conda install scikit-learn
conda install pandas
conda install mpi4py
conda install -c conda-forge keras
conda install -c conda-forge scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -e.
conda install -c conda-forge xgboost 
```

Usage (with Balsam)
=====================


Run once 
----------
```    
    source activate dl-hps   # balsam is installed here too (commands like “balsam ls” must work)

    cd directory_containing_dl-hps
    mv dl-hps dl_hps         # important: change to underscore (import system relies on this)

    cd dl_hps/search
    balsam app --name search --description "run async_search" --executable async-search.py
```

From a qsub bash script (or in this case, an interactive session)
----------------------------------------------------------------------
```
    qsub -A datascience -n 8 -t 60 -q debug-cache-quad -I 

    source ~/.bash_profile    # this should set LD_library_path correctly for mpi4py and make conda available (see balsam quickstart guide)
    source activate dl-hps   # balsam is installed here too (commands like “balsam ls” should work)

    balsam job --name test --workflow b1_addition --app search --wall-minutes 20 --num-nodes 1 --ranks-per-node 1 --args '--max_evals=20'

    balsam launcher --consume --max-ranks-per-node 4   
    # will auto-recognize the nodes and allow only 4 addition_rnn.py tasks to run simultaneously on a node
```

To restart:
----------------------------------------------------------------------
If async-search.py stops for any reason, it will create checkpoint files in the
search working directory.  If you simply restart the balsam launcher, it will
resume any timed-out jobs which have the state "RUN_TIMEOUT".  The async-search
will automatically resume where it left off by finding the checkpoint files in
its working directory.

Alternatively, async-search may have completed, but you wish to extend the
optimization with more iterations.  In this case, you can create a new
async-search job and specify the argument "--restart-from" with the full path
to the previous run's working directory.

```
    # To simply re-start timed-out jobs:
    balsam launcher --consume --max-ranks 4

    # To create a new job extending a previously finished optimization
    balsam job --name test --workflow b1_addition --app search --wall-minutes 20 --num-nodes 1 --ranks-per-node 1 --args '--max_evals=20 --restart-from    /path/to/previous/search/directory'
    balsam launcher --consume --max-ranks-per-node 4   
```

How to install deephyper with the integrated acquistion function
---------------------------------------------------------------------

Create a new conda environment

---
conda create -n dl-hps python=3.6
---

Then install Balsam within this new environment


---
source activate dl-hps
git clone git@xgitlab.cels.anl.gov:turam/hpc-edge-service.git
cd hpc-edge-service
git checkout develop
pip install -e .
---

Once Balsam is installed, install the following dependencies within this conda environment

---
conda install h5py
conda install scikit-learn
conda install pandas
conda install mpi4py
conda install -c conda-forge keras
conda install -c conda-forge xgboost
conda install -c astropy emcee
cd scikit-optimize
pip install -e.
conda install -c conda-forge xgboost 
---

Note: Deephyper with integrated acquistion function uses a local version for scikit-optimize. So, please make sure you've installed the local version of scikit-opt using pip install -e.

How to run deephyper with the integrated acquistion function on theta
---------------------------------------------------------------------
Once you have succesfully created the conda environment, we can create job script for running deephyper.

First load the conda module on theta

---
module load miniconda-3.6/conda-4.4.10
---

Then go to scripts folder of deephyper. You should see a file runjob.py. 

Suppose you want to run the mnistmlp benchmark using the gaussian process as the surrogate model and use the integrated acquisition function on 8 nodes of debug-cache-quad queue.
---
python runjob.py theta_postgres gp mnistmlp.mnist_mlp  EI -q debug-cache-quad -n 8 -t 60 --use-int-acq True
---

In the above command mnistmlp.mnist_mlp is the full name of the benchmark. The flags -q, -n, -t are for name of the queue, number of nodes and total wall time. The flag --use-int-acq is
determine whether you want to use the integrated acquistion function or not. By default the integrated acquisition function is set to False.

The first time you run this command, it won't any create any run script. Rather it will ask to you to edit a runjob.conf. We will see how to edit the runjob.conf in the next section.
Once you have edited the runjob.conf correctly, rerun the above command. You will see the following something like this on your screen

---
CREATED JOB IN /gpfs/mira-home/bob/deephyper/scripts/runs/mnistmlp.mnist_mlp.8.gp.EI.pg.sh
Dry run -- change DISABLE_SUBMIT in runjob.conf to enable auto-submission
---
Now just submit the script to the queue.

---
qsub -A datascience -n 8 -t 60 -q debug-cache-quad /gpfs/mira-home/bob/deephyper/scripts/runs/mnistmlp.mnist_mlp.8.gp.EI.pg.sh
---

---