import torch

def compute_nme(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Normalized Mean Error in normalized coordinates ([0,1] x [0,1]).
    preds/targets: [B, K, 2]
    Returns scalar mean over batch.
    """
    # Euclidean per landmark
    diff = preds - targets
    d = torch.sqrt((diff ** 2).sum(dim=-1))  # [B, K]
    # normalize by bounding box diagonal == sqrt(1^2 + 1^2) = sqrt(2) in normalized coords
    nme = d.mean() / (2 ** 0.5)
    return float(nme.detach().cpu().item())
