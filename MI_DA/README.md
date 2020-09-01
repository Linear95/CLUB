# MI Minimization in Domain Adaptation

This folder includes the code to reproduce the domain adaptation results in our [paper](https://arxiv.org/abs/2006.12013).


## Dependencies:
* Python 2.7
* scikit-learn 0.20.3
* matplotlib 2.2.3
* opencv 3.4.2
* Tensorflow 1.13.1


## Run the Code:

To run the code,
```bash
python main_DANN.py --data_path /work/pc174/club_da_data/ --save_path /work/pc174/tmp/ --source mnist --target svhn
```