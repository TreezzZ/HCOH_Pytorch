import data.cifar10 as cifar10


def load_data(opt):
    """加载数据

    Parameters
        opt: Parser
        参数

    Returns
        DataLoader
        数据加载器
    """
    if opt.dataset == 'cifar10-vgg':
        return cifar10.load_data_vgg(opt)
