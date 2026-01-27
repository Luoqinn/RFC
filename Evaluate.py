import torch

def evaluate_k_torch(topk_idx, label, k):
    '''
    pre@k, recall@k, f1@k, ndcg@k, map@k
    topk_idx: (num_users, k), torch.LongTensor
    label: (num_users, num_items), torch.Tensor (0/1 values)
    '''
    device = topk_idx.device

    discount = 1 / torch.log2(torch.arange(k, device=device) + 2)
    label_counts = label.sum(dim=1)
    valid_mask = label_counts > 0

    if not valid_mask.any():
        return (torch.tensor(torch.nan, device=device),) * 5

    topk_idx = topk_idx[valid_mask]  # (num_valid, k)
    label = label[valid_mask]  # (num_valid, num_items)
    label_counts = label_counts[valid_mask]  # (num_valid,)

    topk_label = torch.gather(label, 1, topk_idx).float()
    true_positive = topk_label.sum(dim=1)  # (num_valid,)

    recall = true_positive / label_counts

    # NDCG
    dcg = (topk_label * discount).sum(dim=1)
    idcg_mask = torch.arange(k, device=device).expand(len(label_counts), -1) < label_counts.unsqueeze(1)
    idcg = (discount.unsqueeze(0) * idcg_mask).sum(dim=1)
    ndcg = dcg / idcg

    cumulative_hits = torch.cumsum(topk_label, dim=1)
    positions = torch.arange(1, k + 1, device=device, dtype=torch.float32).unsqueeze(0)
    precisions = cumulative_hits / positions

    relevant_precisions = precisions * topk_label
    retrieved_counts = topk_label.sum(dim=1)

    ap = relevant_precisions.sum(dim=1) / (retrieved_counts + 1e-12)
    ap[retrieved_counts == 0] = 0.0
    map_score = ap.mean()


    return (
        recall.mean().item(),
        ndcg.mean().item(),
        map_score.item()
    )

