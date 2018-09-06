import mpi4py.rc
mpi4py.rc.initialize = False
import numpy as np
import warnings
import emcee
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky, cho_solve, solve_triangular
from multiprocessing import Pool
import time
import logging

def lg_prob(theta, lnprob):
    return lnprob(theta)

class GPMCMC:
    def __init__(self,model, Xi, Yi, length_scale):
        self.length_scale= length_scale
        self.Xi = Xi
        self.Yi = Yi
        self.alpha = model.alpha
        self.model = model
        self.length_scale = length_scale
        self.f = np.zeros(len(Yi))
        self.n = len(Yi)
        self.kernel = self.model.kernel_

    def __call__(self, theta):
        def lnprior(theta):
          l = theta[2:]
          s2_f = theta[0]
          s2_n = theta[1]
          if 0 < s2_f  and 0 < s2_n  and (l > 0).all() and (l < 2.0).all():
            return np.log(np.log(1 + (0.1/s2_n)**2)) -0.5*(np.log(np.sqrt(s2_f))/1.0)**2 
          return -np.inf

        def lnlike(theta):
          l = theta[2:]
          sigma_f = theta[0]
          sigman  = theta[1]
          self.kernel.k1.k1.constant_value = sigma_f
          self.kernel.k1.k2.length_scale = l
          self.kernel.k2.noise_level = sigman
        
          K = self.kernel(self.Xi)

          K[np.diag_indices_from(K)] += self.alpha
        
          L = cholesky(K, lower=True)  # Line 2
        

          # Support multi-dimensional output of self.y_train_
          y_train = self.Yi
          if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

          alpha = cho_solve((L, True), y_train)  # Line 3

          # Compute log-likelihood (compare line 7)
          log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
          log_likelihood_dims -= np.log(np.diag(L)).sum()
          log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
          log_likelihood = log_likelihood_dims.sum(-1)
          return log_likelihood

  
        lp = lnprior(theta)
        if not np.isfinite(lp):
          return -np.inf
        return lp + lnlike(theta)

class ensemblesampler:
    def __init__(self,lnprob, ndim=2, scale=10, noise=1e-3):
        self.lnprob = lnprob
        self.ndim = ndim
        self.initial_scale = scale
        self.initial_noise = noise

    def get_samples(self):
        nwalkers = 400

        pos_min = np.concatenate((np.array([self.initial_scale, self.initial_noise]), np.zeros(self.ndim)))
        pos_max = np.concatenate((np.array([self.initial_scale, self.initial_noise]), 2.0*np.ones(self.ndim)))
        

        psize = pos_max - pos_min
        pos = [pos_min + psize*np.random.rand(self.ndim+2) for i in range(nwalkers)]
        with Pool(8) as pool:
          print("burning_started")  
          sampler = emcee.EnsembleSampler(nwalkers, self.ndim+2, lg_prob, pool=pool, args=[self.lnprob])
          print("Loading sampler finished")
          pos, prob, state = sampler.run_mcmc(pos, 200)
          sampler.reset()
          print("burning_finished")
          sampler.run_mcmc(pos, 300)
        samples = sampler.flatchain[-500:,:]
        return samples
def int_gaussian_acquisition(X,model,y_opt=None,acq_func="EI",acq_func_kwargs=None,return_grad= False):
    logprob = GPMCMC(model=model,Xi=model.X_train_,Yi=model.y_train_,length_scale =1*np.ones(model.X_train_.shape[1]))
    sampleremCees = ensemblesampler(logprob, ndim=model.X_train_.shape[1], scale= np.std(model.y_train_+1e-4))
    start_time = time.time()
    print("Started sampling")
    sampleemCees = sampleremCees.get_samples()
    end_time = time.time()
    print("Sampling time", end_time-start_time)
    acqu_values = np.zeros((X.shape[0],500))

    acq_grad = []
    
    traceemCees = sampleemCees
    
    for i in range(500):
        sigmaf = traceemCees[i][0]
        sigman = traceemCees[i][1]
        ls = traceemCees[i][2:]
        
        model.kernel_.k1.k1.constant_value = sigmaf
        model.kernel_.k1.k2.length_scale = ls
        model.kernel_.k2.noise_level = sigman
        
        K = model.kernel_(model.X_train_)
        K[np.diag_indices_from(K)] += model.alpha
        model.L_ = cholesky(K, lower=True)
        model.alpha_ = cho_solve((model.L_, True), model.y_train_)
        model.kernel_.k2.noise_level = 0.0
        
        if return_grad:
          acqu_values[:,i],grad = _gaussian_acquisition(
                       X=X, model= model, y_opt= y_opt,
                       acq_func=acq_func,
                       acq_func_kwargs=acq_func_kwargs,return_grad=return_grad)
          acq_grad.append(grad)
        else:  
          acqu_values[:,i] = _gaussian_acquisition(
                       X=X, model= model, y_opt= y_opt,
                       acq_func=acq_func,
                       acq_func_kwargs=acq_func_kwargs,return_grad=return_grad)
    
    
    if return_grad:
       return np.mean(acqu_values, axis = 1), np.mean(acq_grad, axis = 0)
    else:
      return  np.mean(acqu_values, axis = 1)          

def gaussian_acquisition_1D(X, model, y_opt=None, acq_func="LCB",
                            acq_func_kwargs=None, return_grad=True):
    """
    A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    return _gaussian_acquisition(np.expand_dims(X, axis=0),
                                 model, y_opt, acq_func=acq_func,
                                 acq_func_kwargs=acq_func_kwargs,
                                 return_grad=return_grad)


def _gaussian_acquisition(X, model, y_opt=None, acq_func="LCB",
                          return_grad=False, acq_func_kwargs=None):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X is {}-dimensional, however,"
                         " it must be 2-dimensional.".format(X.ndim))

    if acq_func_kwargs is None:
        acq_func_kwargs = dict()
    xi = acq_func_kwargs.get("xi", 0.01)
    kappa = acq_func_kwargs.get("kappa", 1.96)

    # Evaluate acquisition function
    per_second = acq_func.endswith("ps")
    if per_second:
        model, time_model = model.estimators_

    if acq_func == "LCB":
        func_and_grad = gaussian_lcb(X, model, kappa, return_grad)
        if return_grad:
            acq_vals, acq_grad = func_and_grad
        else:
            acq_vals = func_and_grad

    elif acq_func in ["EI", "PI", "EIps", "PIps"]:
        if acq_func in ["EI", "EIps"]:
            func_and_grad = gaussian_ei(X, model, y_opt, xi, return_grad)
        else:
            func_and_grad = gaussian_pi(X, model, y_opt, xi, return_grad)

        if return_grad:
            acq_vals = -func_and_grad[0]
            acq_grad = -func_and_grad[1]
        else:
            acq_vals = -func_and_grad

        if acq_func in ["EIps", "PIps"]:

            if return_grad:
                mu, std, mu_grad, std_grad = time_model.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=True)
            else:
                mu, std = time_model.predict(X, return_std=True)

            # acq = acq / E(t)
            inv_t = np.exp(-mu + 0.5*std**2)
            acq_vals *= inv_t

            # grad = d(acq_func) * inv_t + (acq_vals *d(inv_t))
            # inv_t = exp(g)
            # d(inv_t) = inv_t * grad(g)
            # d(inv_t) = inv_t * (-mu_grad + std * std_grad)
            if return_grad:
                acq_grad *= inv_t
                acq_grad += acq_vals * (-mu_grad + std*std_grad)

    else:
        raise ValueError("Acquisition function not implemented.")

    if return_grad:
        return acq_vals, acq_grad
    return acq_vals


def gaussian_lcb(X, model, kappa=1.96, return_grad=False):
    """
    Use the lower confidence bound to estimate the acquisition
    values.

    The trade-off between exploitation and exploration is left to
    be controlled by the user through the parameter ``kappa``.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `kappa`: [float, default 1.96 or 'inf']:
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        If set to 'inf', the acquisition function will only use the variance
        which is useful in a pure exploration setting.
        Useless if ``method`` is set to "LCB".

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.

    * `grad`: [array-like, shape=(n_samples, n_features)]:
        Gradient at X.
    """
    # Compute posterior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

            if kappa == "inf":
                return -std, -std_grad
            return mu - kappa * std, mu_grad - kappa * std_grad

        else:
            mu, std = model.predict(X, return_std=True)
            if kappa == "inf":
                return -std
            return mu - kappa * std


def gaussian_pi(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    """
    Use the probability of improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)`form a gaussian with a
    certain mean and standard deviation approximated by the model.

    The PI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 1``, if ``f(x) < y_opt`` and ``u(f(x)) = 0``,
    if``f(x) > y_opt``.

    This means that the PI condition does not care about how "better" the
    predictions are than the previous values, since it gives an equal reward
    to all of them.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `y_opt` [float, default 0]:
        Previous minimum value which we would like to improve upon.

    * `xi`: [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)
        else:
            mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    values[mask] = norm.cdf(scaled)

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std**2

        return values, improve_grad * norm.pdf(scaled)

    return values


def gaussian_ei(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    """
    Use the expected improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)`form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    * `X` [array-like, shape=(n_samples, n_features)]:
        Values where the acquisition function should be computed.

    * `model` [sklearn estimator that implements predict with ``return_std``]:
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    * `y_opt` [float, default 0]:
        Previous minimum value which we would like to improve upon.

    * `xi`: [float, default=0.01]:
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    * `return_grad`: [boolean, optional]:
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    * `values`: [array-like, shape=(X.shape[0],)]:
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

        else:
            mu, std = model.predict(X, return_std=True)

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std ** 2
        cdf_grad = improve_grad * pdf
        pdf_grad = -improve * cdf_grad
        exploit_grad = -mu_grad * cdf - pdf_grad
        explore_grad = std_grad * pdf + pdf_grad

        grad = exploit_grad + explore_grad
        return values, grad

    return values
