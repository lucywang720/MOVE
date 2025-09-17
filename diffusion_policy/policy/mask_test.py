import torch
import torch.nn.functional as F
import random

def random_mask_sequences(pred, target):
    """
    Randomly mask sequences with varying lengths
    
    Args:
    - pred (torch.Tensor): Predicted sequences of shape (bs, 132, action_dim)
    - target (torch.Tensor): Target sequences of shape (bs, 132, action_dim)
    
    Returns:
    - masked_pred (torch.Tensor): Masked predictions
    - masked_target (torch.Tensor): Masked targets
    - loss_mask (torch.Tensor): Mask indicating which elements to use in loss computation
    """
    batch_size, seq_len, action_dim = pred.shape
    
    # Possible mask lengths
    mask_lengths = [34, 66, 98, 132]
    
    # Initialize loss mask
    loss_mask = torch.zeros_like(pred, dtype=torch.bool)
    
    for b in range(batch_size):
        # Randomly choose a mask length
        chosen_length = random.choice(mask_lengths)
        
        # Create mask for this batch element
        loss_mask[b, :chosen_length, :] = 1

        print(loss_mask[b], sum(loss_mask[b]))
    
    # Apply mask
    masked_pred = pred * loss_mask
    masked_target = target * loss_mask
    
    return masked_pred, masked_target, loss_mask


mp, mt, lm = random_mask_sequences(torch.randn(256,132,2), torch.randn(256,132,2))

print(mp.shape, mt.shape, lm.shape)