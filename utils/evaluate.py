from utils.calc_hamming_dist import calc_hamming_dist

import torch


def evaluate(query_code,
             database_code,
             query_targets,
             database_targets,
             device,
             topk=None,
             ):
    """
    Calculate precision

    Args:
        query_code (torch.Tensor): Query images hash code.
        database_code (torch.Tensor): Database images hash code.
        query_targets (torch.Tensor): Query images targets.
        database_targets(torch.Tensor): Database images targets.
        device (str): Using gpu or cpu.
        topk (int): Calculate preicision using top k data.

    Returns
        precision (float): Precision.
        meanAP (float): Mean Average Precision
    """
    num_query = query_targets.shape[0]
    num_database = database_targets.shape[0]
    mean_AP = 0.0
    precision = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ database_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = calc_hamming_dist(query_code[i, :], database_code)

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        if topk == -1:
            precision += retrieval_cnt / num_database
        else:
            precision += retrieval_cnt / topk

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean().item()

    mean_AP = mean_AP / num_query
    precision = precision / num_query
    return mean_AP, precision


