import torch

from scipy.linalg import hadamard

from utils.evaluate import evaluate


def hcoh(train_data,
         train_targets,
         query_data,
         query_targets,
         database_data,
         database_targets,
         code_length,
         lr,
         num_hadamard,
         device,
         topk,
         ):
    """HCOH algorithm

    Parameters
        train_data: Tensor
        Training data

        train_targets: Tensor
        Training targets

        query_data: Tensor
        Query data

        query_targets: Tensor
        Query targets

        Database_data: Tensor
        Database data

        Database_targets: Tensor
        Database targets

        code_length: int
        Hash code length

        lr: float
        Learning rate

        num_hadamard: int
        Number of hadamard codebook columns

        device: str
        Using cpu or gpu

        topk: int
        Compute mAP using top k retrieval result

    Returns
        meanAP: float
        mean Average precision
    """
    # Construct hadamard codebook
    hadamard_codebook = torch.from_numpy(hadamard(num_hadamard)).float().to(device)
    hadamard_codebook = hadamard_codebook[torch.randperm(num_hadamard), :]

    # Initialize
    num_train, num_features = train_data.shape
    W = torch.randn(num_features, code_length).to(device)

    # Matrix normalazation
    W = W / torch.diag(torch.sqrt(W.t() @ W)).t().expand(num_features, code_length)
    if code_length == num_hadamard:
        W_prime = torch.eye(num_hadamard).to(device)
    else:
        W_prime = torch.randn(num_hadamard, code_length).to(device)
        W_prime = W_prime / torch.diag(torch.sqrt(W_prime.t() @ W_prime)).t().expand(num_hadamard, code_length)

    # Train
    for i in range(train_data.shape[0]):
        data = train_data[i, :].reshape(1, -1)
        lsh_x = (hadamard_codebook[train_targets[i], :].view(1, -1) @ W_prime).sign()
        tanh_x = torch.tanh(data @ W)
        dW = data.t() @ ((tanh_x - lsh_x) * (1 - tanh_x * tanh_x))

        W = W - lr * dW

    # Evaluate
    # Generate hash code
    query_code = generate_code(query_data, W)
    database_code = generate_code(database_data, W)

    # Compute map
    mAP, precision = evaluate(
        query_code,
        database_code,
        query_targets,
        database_targets,
        device,
        topk,
    )

    return mAP, precision


def generate_code(data, W):
    """
    Generate hash code

    Parameters
        data: Tensor
        Data

        W, b: Tensor
        Parameters

    Returns
        code: Tensor
        Hash code
    """
    B = (data @ W).sign()

    return B
