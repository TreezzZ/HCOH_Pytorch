import numpy as np
import scipy.io as sio


def load_data_vgg(opt):
    """
    加载对cifar10使用vgg提取的数据

    Parameters
        opt: Parser
        参数

    Returns
        train_data, query_data, data: numpy.ndarray
        数据

        train_targets, query_targets, targets: numpy.ndarray
        标签
    """
    mat_data = sio.loadmat(opt.data_path)

    train_data = mat_data['trainCNN']
    train_targets = mat_data['trainLabels'].astype(np.int)
    test_data = mat_data['testCNN']
    test_targets = mat_data['testLabels'].astype(np.int)

    data = np.concatenate((train_data, test_data), axis=0)
    targets = np.concatenate((train_targets, test_targets))

    # Split dataset
    rand_perm = np.random.permutation(data.shape[0])
    query_index = rand_perm[:opt.num_query]
    train_index = rand_perm[opt.num_query: opt.num_query + opt.num_train]

    query_data = data[query_index, :]
    query_targets = targets[query_index, :]

    train_data = data[train_index, :]
    train_targets = targets[train_index, :]

    return train_data, train_targets, query_data, query_targets, data, targets
