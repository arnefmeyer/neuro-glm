# neuro-glm

A Python package to fit Poisson Generalized Linear Models to neural spike trains 
based on multiple covariates, e.g., the animal's position and sensory input.

GLM weight penalties (L2, smooth-L1, 1D/2D roughness) can be applied to the different 
covariates, and hyperparameters can be found using exhaustive or Bayesian grid search.

**Example code and data will be made available soon.**


# Installation

1. Clone or download the repository

2. Locate the terminal to the code folder and run `pip install -e .`


# Reference

_The Hybrid Drive: a chronic implant device combining tetrode arrays with silicon probes for layer-resolved ensemble electrophysiology in freely moving mice_  
Guardamagna M, Eichler R, Pedrosa R, Aarts A, Meyer AF\*, Battaglia FP\*  
bioRxiv, 2021.
([preprint](https://www.biorxiv.org/content/10.1101/2021.08.20.457090v2))


