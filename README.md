# CLUB
Code for ICML2020 paper - CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information

This repository contains source code to implement the CLUB mutual information (MI) upper bound estimator.

Here is the link to our ICML2020 paper:
* [CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information](https://arxiv.org/abs/2006.12013)

## Dependencies: 
This code is written in python. The dependencies are:
* Python 3
* Pytorch (recent version) for simulation study
* Tensorflow for real-world experiments

## Mutual Information Estimation

We evaluate the estimation ability of CLUB and other baselines under the simulation setups. Samples from Gaussian and Cubic distributions are generated with the true MI values pre-known. Then we compare different MI estimators on estimating MI values based on the given samples. The code in this section is written with Pytorch. 

The implementation of our CLUB estimator, along with other baselines ([NWJ](https://media.gradebuddy.com/documents/2949555/12a1c544-de73-4e01-9d24-2f7c347e9a20.pdf), [MINE](http://proceedings.mlr.press/v80/belghazi18a), [InfoNCE](https://arxiv.org/pdf/1807.03748.pdf), [VUB](https://arxiv.org/abs/1612.00410), [L1Out](https://arxiv.org/pdf/1905.06922.pdf)), is in `mi_estimators.py`. VUB and L1Out are implemented in variational forms proposed in [our paper](https://arxiv.org/abs/2006.12013).  Follow the steps in `simulation.ipynb` to demonstrate the MI estimation performance of different methods.

## Mutual Information Minimization

We will update the source for real-world mutual information minimization tasks soon.

## Citation 
Please cite our ICML 2020 paper if you found the code useful.

```latex
@misc{@article{cheng2020club,
  title={CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information},
  author={Cheng, Pengyu and Hao, Weituo and Dai, Shuyang and Liu, Jiachang and Gan, Zhe and Carin, Lawrence},
  journal={arXiv preprint arXiv:2006.12013},
  year={2020}
}
}
```
