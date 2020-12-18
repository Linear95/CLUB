# CLUB
This repository contains source code to our ICML2020 paper: 

* [CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information](https://arxiv.org/abs/2006.12013)

CLUB is a sample-based estimator to mutual information (MI), which can not only provide reliable upper bound MI estimation, but also effectively minimize correlation in deep models as a learning critic.

## Mutual Information Estimation

We provide toy simulations in `mi_estimation.ipynb` to show the estimation performance of CLUB and other MI estimators. The code in this section is written with [Pytorch](https://pytorch.org/) (latest version). 

The implementation of our CLUB estimator, along with other baselines ([NWJ](https://media.gradebuddy.com/documents/2949555/12a1c544-de73-4e01-9d24-2f7c347e9a20.pdf), [MINE](http://proceedings.mlr.press/v80/belghazi18a), [InfoNCE](https://arxiv.org/pdf/1807.03748.pdf), [VUB](https://arxiv.org/abs/1612.00410), [L1Out](https://arxiv.org/pdf/1905.06922.pdf)), is in `mi_estimators.py`. VUB and L1Out are implemented in the variational forms proposed in [our paper](https://arxiv.org/abs/2006.12013).  

Follow the steps in `mi_estimation.ipynb` to demonstrate the MI estimation performance of different MI estimators.

## Mutual Information Minimization

We test the MI minimization performance of our CLUB estimator on two real-world tasks: Information Bottleneck (IB) and Domain Adaptation (DA). We provide the instructions to reproduce the results of IB and DA in the folder [MI_IB](https://github.com/Linear95/CLUB/tree/master/MI_IB) and [MI_DA](https://github.com/Linear95/CLUB/tree/master/MI_DA) respectively. 

Besides, we provide another toy example in `mi_minimization.ipynb` to visualize how our MI minimization algorthm works under multivariate Gaussian setups.

## Citation 
Please cite our ICML 2020 paper if you found the code useful.

```latex
@article{cheng2020club,
  title={CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information},
  author={Cheng, Pengyu and Hao, Weituo and Dai, Shuyang and Liu, Jiachang and Gan, Zhe and Carin, Lawrence},
  journal={arXiv preprint arXiv:2006.12013},
  year={2020}
}
```
