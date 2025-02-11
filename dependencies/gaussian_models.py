# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:51:19 2025

@author: oconn
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def loss(hyps,D,kernel):

    jitter = 1e-6

    xtrain = D[0]
    ytrain = D[1]

    N = len(ytrain)

    hyps = np.exp(hyps)

    sn2 = hyps[-1]

    kxx, _ , _ = kernel(xtrain,xtrain,hyps).compute_kernel()
    kxx += (sn2 * np.eye(N))

    L = np.linalg.cholesky(kxx + jitter*np.eye(N))
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,ytrain)) #cho_solve((L, True), ytrain) 

    nlml = ((0.5 * ytrain.T @ alpha) + np.sum(np.log(np.diag(L))) + (0.5 * N * np.log(2*np.pi)))
    part_1 = (-0.5 * (ytrain.T @ alpha))
    part_2 = -0.5* np.sum(np.log(np.diag(L)))
    part_3 = (-0.5 * N * np.log(2*np.pi))
    nlml_1 = part_1 + part_2 + part_3    
    return nlml_1

def loss_pos(hyps,D,kernel):

    jitter = 1e-6

    xtrain = D[0]
    ytrain = D[1]

    N = len(ytrain)

    hyps = np.exp(hyps)

    sn2 = hyps[-1]

    kxx, _ , _ = kernel(xtrain,xtrain,hyps).compute_kernel()
    kxx += (sn2 * np.eye(N))

    L = np.linalg.cholesky(kxx + jitter*np.eye(N))
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,ytrain)) #cho_solve((L, True), ytrain) 

    nlml = ((0.5 * ytrain.T @ alpha) + np.sum(np.log(np.diag(L))) + (0.5 * N * np.log(2*np.pi)))
    part_1 = (-0.5 * (ytrain.T @ alpha))
    part_2 = -0.5* np.sum(np.log(np.diag(L)))
    part_3 = (-0.5 * N * np.log(2*np.pi))
    nlml_1 = part_1 + part_2 + part_3    
    return nlml
    
def train(hyps,D,kernel,bounds,opt_type):
    if opt_type =="min":
        opt = minimize(loss, hyps, args =(D, kernel),method = 'L-BFGS-B',bounds=bounds)
    else:
        opt= minimize(loss_pos, hyps, args =(D, kernel),method = 'L-BFGS-B',bounds=bounds)
    
    hyps_opt = np.exp(opt.x)
    nlml = opt.fun
    
    return hyps_opt, nlml
    
def predict(hyps,D,xtest,kernel):

    xtrain = D[0]
    ytrain = D[1]
    N = len(ytrain)

    jitter = 1e-6
    sn2 = hyps[-1]

    kxx, kxt, ktt = kernel(xtrain,xtest,hyps).compute_kernel()
    kxx += sn2*np.eye(N)
    
    L = np.linalg.cholesky(kxx + jitter*np.eye(N))
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,ytrain))

    v = np.linalg.solve(L,kxt)

    mu_pred = kxt.T @ alpha
    cov_pred = ktt - v.T @ v


    return mu_pred, cov_pred    

   
    #return model_predictions

def expected_improvement_min(mu, sigma, y_min,xi):
    # xi = 0.05
    gamma = (mu-(y_min+xi))/sigma
    EI = (y_min-mu+xi)*norm.cdf(gamma) + sigma*norm.pdf(gamma)
    return EI

def upper_confidence_bound_min(mu, sigma,beta):
    #beta is negative in this definition
    # beta = -3
    UCB = mu - beta*sigma
    return UCB

def probability_of_improvement_min(mu,sigma, y_min,xi):
    # xi = 0.05
    gamma = (mu-(y_min+xi))/sigma
    pi = norm.cdf(gamma)
    return pi


def upper_confidence_bound_max(mu, sigma,beta):
    UCB = mu + beta*sigma
    return UCB

def probability_of_improvement_max(mu,sigma, y_max,xi):
    gamma = (mu-(y_max-xi))/sigma
    pi = norm.cdf(gamma)
    return pi


def expected_improvement_max(mu, sigma, y_max,xi):
    gamma = (mu-(y_max-xi))/sigma
    EI = (mu - y_max - xi)*norm.cdf(gamma) + sigma*norm.pdf(gamma)
    return EI
