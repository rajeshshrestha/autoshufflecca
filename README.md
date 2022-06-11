# DASCCA: Deep AutoShuffle Canonical Correlation Analysis

This is an implementation of our proposed novel model DASCCA in python using pytorch. DASCCA is motivated from the following papers on DCCA and autoshuffling:

Andrew, Galen, et al. "[Deep canonical correlation analysis.](https://proceedings.mlr.press/v28/andrew13.html)" International conference on machine learning. PMLR, 2013.

Lyu, Jiancheng, et al. "[Autoshufflenet: Learning permutation matrices via an exact lipschitz continuous penalty in deep convolutional neural networks.](https://dl.acm.org/doi/abs/10.1145/3394486.3403103)" Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.

This is adapted from the implementation by [Michaelvll](https://github.com/Michaelvll/DeepCCA).

## Create an environement
```bash
conda env create -f environment.yml
```

## Download required [MNIST view 1 data](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz), [MNIST view 2 noisy data](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz) and [CIFAR10 data](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

```bash
sh download_data.sh
```

## Run the code
```bash
python main.py [--use-cifar] [--split-image][--no-shuffle]

where,
--use-cifar: 
    if passed then, the code will use the CIFAR10 data otherwise use MNIST dataset
--split-image:
    if passed then, the code will form the second view by splitting the image into two halves(left and right) otherwise noisy augmentation is used as the second view.
--no-shuffle:
    if passed the code will train the vanilla DCCA and if not passed, then our proposed model (DASCCA) will be trained
```

The metrics are printed in the console and the graphs along with the figures are logged using tensorboard in the directory ```./runs```

## Run Tensorboard
```
tensorboard --logdir runs
```