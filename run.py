import HCOH
import data.dataloader as dataloader
from data.transform import normalization, encode_onehot

import argparse
import torch
from loguru import logger


def run_hcoh(args):
    """Run HCOH algorithm

    Parameters
        args: parser
        Configuration

    Returns
        None
    """
    # Load dataset
    train_data, train_targets, query_data, query_targets, database_data, database_targets = dataloader.load_data(args)

    # Preprocess dataset
    # Normalization
    train_data = normalization(train_data)
    query_data = normalization(query_data)
    database_data = normalization(database_data)

    # One-hot
    query_targets = encode_onehot(query_targets, 10)
    database_targets = encode_onehot(database_targets, 10)

    # Convert to Tensor
    train_data = torch.from_numpy(train_data).float().to(args.device)
    query_data = torch.from_numpy(query_data).float().to(args.device)
    database_data = torch.from_numpy(database_data).float().to(args.device)
    train_targets = torch.from_numpy(train_targets).squeeze().to(args.device)
    query_targets = torch.from_numpy(query_targets).to(args.device)
    database_targets = torch.from_numpy(database_targets).to(args.device)

    # HCOH algorithm
    for code_length in [8, 16, 32, 64, 128]:
        args.code_length = code_length
        mAP = 0.0
        precision = 0.0
        for i in range(10):
            m, p = HCOH.hcoh(
                train_data,
                train_targets,
                query_data,
                query_targets,
                database_data,
                database_targets,
                args.code_length,
                args.lr,
                args.num_hadamard,
                args.device,
                args.topk,
            )
            mAP += m
            precision += p
        logger.info('[code_length:{}][map:{:.3f}][precision:{:.3f}]'.format(code_length, mAP / 10, precision / 10))


def load_parse():
    """Load configuration

    Parameters
        None

    Returns
        args: parser
        Configuration
    """
    parser = argparse.ArgumentParser(description='HCOH_PyTorch')
    parser.add_argument('--dataset', default='cifar10-vgg', type=str,
                        help='Dataset used to train (default: cifar10-vgg)')
    parser.add_argument('--data-path', type=str,
                        help='Path of dataset')
    parser.add_argument('--num-hadamard', type=int,
                        help='Number of hadamard codebook columns.')
    parser.add_argument('--code-length', default=12, type=int,
                        help='Binary hash code length (default: 12)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query dataset. (default: 1000)')
    parser.add_argument('--num-train', default=20000, type=int,
                        help='Number of training dataset. (default: 20000)')
    parser.add_argument('--topk', default=5000, type=int,
                        help='Compute map of top k (default: 5000)')
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Learning rate(default: 2e-4)')
    parser.add_argument('--gpu', default=-1, type=int,
                        help='>0, using gpu id; -1, cpu (default: -1)')

    return parser.parse_args()


if __name__ == "__main__":
    args = load_parse()
    logger.add('logs/file_{time}.log')

    if args.gpu == -1:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    run_hcoh(args)
