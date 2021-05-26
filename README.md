# Neural Approximate Sufficient Statistics

-------------------------------------------------------------------------

Official codes for paper "Neural Approximate Sufficient Statistics for Implict Models", ICLR 21 spotlight
ICLR forum: https://openreview.net/forum?id=SRDuJssQud
Arxiv: https://arxiv.org/abs/2010.10079
To learn sufficient statistics we need not to estimate density or even density ratio!

-------------------------------------------------------------------------


## Package Dependencies
* Python 3
* Pytorch
* Numpy
* Matplotlib


## Summary statistics 
at /nn
* Mean as statistics (MSN.py)
* Infomax statistics (ISN.py)


## Likelihood-free algorithms
at /algorithms
* Sequential Monte Carlo ABC (SMC_ABC.py)
* Sequential Monte Carlo ABC with s.s (SMC2_ABC.py)
* Sequential Neural Likelihood (SNL_ABC.py)
* Sequential Neural Likelihood with s.s (SNL2_ABC.py)

## Inference problems
at /problems
* Ising Model
* Gaussian copula Model
* Ornstein-Uhlenbeck process


