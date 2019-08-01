# Deep Supervised Discrete Hashing

论文[Supervised Online Hashing via Hadamard Codebook Learning](https://dl.acm.org/citation.cfm?id=3240519)

## Requirements
1. pytorch 1.1
2. loguru

## Dataset
1. [CIFAR-10-vgg](http://cs-people.bu.edu/hekun/data/mihash/CIFAR10_VGG16_fc7.mat)

## Usage
`python run.py --data-path <dataset-path> --num-hadamard 32 --lr 2e-4 --topk -1 `

日志记录在`logs`文件夹内

```
usage: run.py [-h] [--dataset DATASET] [--data-path DATA_PATH]
              [--num-hadamard NUM_HADAMARD] [--code-length CODE_LENGTH]
              [--num-query NUM_QUERY] [--num-train NUM_TRAIN] [--topk TOPK]
              [--lr LR] [--gpu GPU]

HCOH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset used to train (default: cifar10-vgg)
  --data-path DATA_PATH
                        Path of dataset
  --num-hadamard NUM_HADAMARD
                        Number of hadamard codebook columns.
  --code-length CODE_LENGTH
                        Binary hash code length (default: 12)
  --num-query NUM_QUERY
                        Number of query dataset. (default: 1000)
  --num-train NUM_TRAIN
                        Number of training dataset. (default: 20000)
  --topk TOPK           Compute map of top k (default: 5000)
  --lr LR               Learning rate(default: 2e-4)
  --gpu GPU             >0, using gpu id; -1, cpu (default: -1)

```

# Experiments
复现过程中有两个坑。
1. W,W_prime需要进行归一化，否则W @ data过大，造成tanh只会输出-1,+1，梯度为0，没法更新。
2. 论文3.4节给出的P的计算公式为P=(1-tanh(x))*tanh(x)，作者实际放出的代码里是P=1-tanh(x)*tanh(x)，前者没法更新或者效果很差，
我没细推公式，可能是作者论文里打错公式了。

cifar10: 1000 query images, 20000 training images, 60000 database images.

每个code_length跑10次，计算map平均值。

8-bit | 16-bit | 32-bit | 64-bit | 128-bit  
:-:   | :-:    | :-:    | :-:    | :-:
0.561 | 0.656  | 0.728  | 0.736  | 0.745
