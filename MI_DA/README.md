# MI Minimization in Domain Adaptation

This folder includes the code to reproduce the domain adaptation results in our [paper](https://arxiv.org/abs/2006.12013).


## Dependencies
* python 2.7
* tensorflow 1.15.0
* scikit-learn 
* opencv

## Download Dataset
Our code can conduct domain adaptation experiments on six datasets: MNIST([mnist](http://yann.lecun.com/exdb/mnist/)), SVHN([svhn](http://ufldl.stanford.edu/housenumbers/)), MNIST-M([mnistm](https://github.com/pumpikano/tf-dann/blob/master/create_mnistm.py)), USPS([usps](https://github.com/mingyuliutw/CoGAN/tree/master/cogan_pytorch/data/uspssample)), CIFAR-10(cifar), and STL(stl). Follow the instruction from the link of each dataset to download the data.

## Run the Code

To run the code, specify the source and the target datasets with the following command:
```bash
python main_DANN.py --data_path /path/to/data_folder/ --save_path /path/to/save_dir/ --source mnist --target svhn
```