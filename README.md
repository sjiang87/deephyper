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

```
     conda create -n dl-hps python=3.6
```

Then install Balsam within this new environment


```
    source activate dl-hps
    git clone git@xgitlab.cels.anl.gov:turam/hpc-edge-service.git
    cd hpc-edge-service
    git checkout develop
    pip install -e .
```

Once Balsam is installed, install the following dependencies within this conda environment

```
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
```

Note: Deephyper with integrated acquistion function uses a local version for scikit-optimize. So, please make sure you've installed the local version of scikit-opt using pip install -e.

How to run deephyper with the integrated acquistion function on theta
---------------------------------------------------------------------
Note: These instructions are valid for theta.

Once you have succesfully created the conda environment, the following steps decribe how to a create job script for running deephyper.

First load the conda module on theta

```
   module load miniconda-3.6/conda-4.4.10
```

Then go to scripts folder of deephyper. You should see a file runjob.py. 

Suppose you want to run the mnistmlp benchmark using the gaussian process as the surrogate model and use the integrated acquisition function on 8 nodes of debug-cache-quad queue.
```
   python runjob.py theta_postgres gp mnistmlp.mnist_mlp  EI -q debug-cache-quad -n 8 -t 60 --use-int-acq True
```

In the above command mnistmlp.mnist_mlp is the full name of the benchmark. The flags -q, -n, -t are for the name of the queue, number of nodes and total wall time. The flag --use-int-acq is
determines whether you want to use the integrated acquistion function or not. By default the --use-int-acq is set to False.

The first time you run this command, it won't any create any run script. Rather it will ask to you to edit a runjob.conf. We will see how to edit the runjob.conf in the next section.
Once you have edited the runjob.conf correctly, rerun the above command. You will see something like this on your screen

```
   CREATED JOB IN /gpfs/mira-home/bob/deephyper/scripts/runs/mnistmlp.mnist_mlp.8.gp.EI.pg.sh
   Dry run -- change DISABLE_SUBMIT in runjob.conf to enable auto-submission
```
Now just submit the script to the queue.

```
    qsub -A datascience -n 8 -t 60 -q debug-cache-quad /gpfs/mira-home/bob/deephyper/scripts/runs/mnistmlp.mnist_mlp.8.gp.EI.pg.sh
```

How to edit the runjob.conf file
----------------------------------------------------------------------

The first time you run the runjob.py, it will create a runjob.conf file. You will edit its contents accordingly. This is how a runjob.conf file looks like

```
{
    "DEEPHYPER_ENV_NAME":   "dl-hps",
    "DEEPHYPER_TOP":        "/home/bob/deephyper",
    "DATABASE_TOP":          "/projects/datascience/bob/deephyper/database",
    "BALSAM_PATH":           "/home/bob/hpc-edge-service/balsam",
    "STAGE_IN_DIR":          "",
    "DISABLE_SUBMIT":        true
}

```

The variable "DEEPHYPER_ENV_NAME" should be set the conda environment where deephyper is installed. The variable "DEEPHYPER_TOP" should have location of where deephyper is installed.
The variable "DATABASE_TOP" should have the location of the folder where Balsam will creates its database. This folder can anywhere you like but keep in mind all results and data of the deephyper 
will stored in this folder. The variable "BALSAM_PATH" should set to the location where deephyper is installed. The variable "STAGE_IN_DIR" should be set to where want to stage your dataset for benchmarks.
This variable can be left empty. The "DISABLE_SUBMIT" variable if set to True will not allow the runjob.py script to automatically qsub the script you generate. You have to manually submit 
the script to the queque using qsub command.


How to use the EIps and PIps acquistion functions in deephyper
----------------------------------------------------------------------
You can use the EI per second (EIps) and PI per second (PIps) acquistion function with deephyper. To use the EIps and PIps acquistion function, first make sure benchmark contains correct
time in information. For example take tmnmistmlp benchmark in deephyper. Go to the benchmark folder and to the folder tmnistmlp folder. Inside this folder, you will find the file
mnist_mlp.py. If you look at the bottom of the main function of this file, you will see the following

```
print('OUTPUT:', -score[1], end_time - start_time )
```

Here end_time - start_time is basically the total running time of the code. If you want to use EIps and PIps acquistion functions, please make you have similar print statement at the bottom
of your file. It should print the string "OUTPUT" followed by the value of the objective function (in this case the test accuracy) and the time to compute the function. The evaluator in deephyper
will read this string from the Balsam log and send it to the optimizer, so it is very important that the print statement in your benchmark has the exactly the same format otherwise it will
crash.

To generate the script for EIps and PIps do exactly as mentioned in the previous sections. Run the runjob.py file from scripts folder in deephyper.

Suppose you want to run the tmnistmlp benchmark using the gaussian process as the surrogate model and use the EIps acquistion function on 8 nodes of debug-cache-quad queue. You should run
run the following command

```
   python runjob.py theta_postgres gp tmnistmlp.mnist_mlp  EIps -q debug-cache-quad -n 8 -t 60 --use-int-acq False
```
or
```
   python runjob.py theta_postgres gp tmnistmlp.mnist_mlp  EIps -q debug-cache-quad -n 8 -t 60 
```

NOTE: The integrated acquistion function can't be used with EIps or PIps. Deephyper will fail if try to do so.

In the above command tmnistmlp.mnist_mlp is the full name of the benchmark. The flags -q, -n, -t are for the name of the queue, number of nodes and total wall time. The EIps is acquistion
function.

Tuning the integrated acquistion function
===================================================
The main logic for computing the integrated acquistion function is inside the file acqisition.py of inside the skopt. It is implemented by the int_gaussian_acquisition() function. The int_gaussian_acquisition() intializes two objects. The first object logprob  belongs to the class GPMCMC and second
object sampleremCees is an object of the class ensembleSampler. The class GPMCMC computes the log_probability of the distribution and actual sampling is done by the class  ensembleSampler.

Class GPMCMC
------------------------
The class GPMCMC is use for computing the log probality and it has all the information about the kernel of the Gaussian process. To create an object of Class GPMCMC
```
logprob = GPMCMC(model=model,Xi=model.X_train_,Yi=model.y_train_,length_scale =1*np.ones(model.X_train_.shape[1]))
```
Here model is the base estimator used by scikit-opt, model.X_train and model.Y_train is data already seen by the base estimator and length_scale is length scale used by the Gaussian process.

Class ensembleSampler
--------------------------
The class ensembleSampler has two functions, init and get_samples(). To create object of class  ensembleSampler first create a object logprob of class GPMCMC like shown in the previous section and then
```
sampleremCees = ensemblesampler(logprob, ndim=model.X_train_.shape[1], scale= np.std(model.y_train_+1e-4))

```

Here ndim is equal to the hyperparmeters to you are optimizing and scale is intial value of the scale you will use in the sampler for the scale of the kernel of the Gaussian process.

To generate samples from the sampler
```
sampleemCees = sampleremCees.get_samples()
```

What to do if the integrated acquistion is not performing well
-----------------------------------------------------------------
There are few things you can try out to tune the performance of integrated acquistion function

1. Change the number of walkers
-------------------------------
The get_samples() of ensembleSampler sets the number of walkers
```
nwalkers = 400
```

Currently the number of walkers is set to 400, you can change the value to any number you like. The sapmling can improve with a higher number of workers but more walkers also means more time for sampling.


2. Change the intialization in the sampler
----------------------------------------------
The following lines in the get_samples() function of  ensembleSampler sets the intial values of the
parameters for each walker in the sampler

```
pos_min = np.concatenate((np.array([self.initial_scale, self.initial_noise]), np.zeros(self.ndim)))
pos_max = np.concatenate((np.array([self.initial_scale, self.initial_noise]), 2.0*np.ones(self.ndim)))
psize = pos_max - pos_min
pos = [pos_min + psize*np.random.rand(self.ndim+2) for i in range(nwalkers)]
``` 
The array pos contains the intials values of the parameters for each walker. Currently it is choosenrandomly with uniform distribution and ranges are being determined by pos_min and pos_max. Tuning the the intialization can improve the performance of the sampler.

3. Change the prior distribution
-----------------------------------------
The function lnprior() inside __call__() of class GPMCMC sets the prior distribution

```
def lnprior(theta):
    l = theta[2:]
    s2_f = theta[0]
    s2_n = theta[1]
    if 0 < s2_f  and 0 < s2_n  and (l > 0).all() and (l < 2.0).all():
        return np.log(np.log(1 + (0.1/s2_n)**2)) -0.5*(np.log(np.sqrt(s2_f))/1.0)**2 
    return -np.inf
```
Currently it is set to uniform distribution for length scale parameters, Gaussian for the amplitude parameter and horse-shoe prior for the noise scale. Sometimes changing the prior can significantly change the performance of the integrated acquistion function.

4. Change the number of samples used to do the integration
---------------------------------------------------------------
Currently, we're using only 500 samples for the integration. The quality of results can improve if more samples are being used.

What to do if the sampling is too slow?
----------------------------------------------------------------
You can try the following to speed up the sampling. Depending on the benchmarks and number of hyperparmters sampling time can be something between 60s to 110s currently. The sampling depends on the number of walkers being used. More walkers will result in higher sampling.

1. Change the number of processes in the multiprocessing pool
----------------------------------------------------------------
The emcee library uses multiprocessing pool for parallelizing the sampler. Currently, we are using 8processes in the pool
```
 with Pool(8) as pool:  
      sampler = emcee.EnsembleSampler(nwalkers, self.ndim+2, lg_prob, pool=pool, args=[self.lnprob])
      pos, prob, state = sampler.run_mcmc(pos, 200)
      sampler.reset()
      sampler.run_mcmc(pos, 300)
```
One can change the number of processes in the pool to any number they like ( for example to change to 10 just use Pool(10) instead of Pool(8)). Keep in my too many processes in the multiprocessing pool can slow down the sampling.

2. Change the number of walkers
------------------------------------
Fewer walkers will reduce the sampling but it may also adversely affect the performace and quality of sampling.

