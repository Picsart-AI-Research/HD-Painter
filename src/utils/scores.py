import torch 
from torch import nn
import torch.nn.functional as F

def l1(_crossattn_similarity, mask, token_idx = [1,2]):
    similarity = torch.cat(_crossattn_similarity,1)[1]
    similarity = similarity.mean(0).permute(2,0,1)
    # similarity = similarity.softmax(dim = 0)
    
    return (similarity[token_idx] * mask.cuda()).sum()

def bce(_crossattn_similarity, mask, token_idx = [1,2]):
    similarity = torch.cat(_crossattn_similarity,1)[1]
    similarity = similarity.mean(0).permute(2,0,1)
    # similarity = similarity.softmax(dim = 0)
    
    return -sum([
        F.binary_cross_entropy_with_logits(x - 1.0, mask.cuda())
        for x in similarity[token_idx]
    ]) 

def softmax(_crossattn_similarity, mask, token_idx = [1,2]):
    similarity = torch.cat(_crossattn_similarity,1)[1]
    similarity = similarity.mean(0).permute(2,0,1)

    similarity = similarity[1:].softmax(dim = 0) # Comute the softmax to obtain probability values
    token_idx = [x - 1 for x in token_idx]

    score = similarity[token_idx].sum(dim = 0) # Sum up all relevant tokens to get pixel-wise probability of belonging to the correct class
    score = torch.log(score) # Obtain log-probabilities per-pixel
    return (score * mask.cuda()).sum() # Sum up log-probabilities (equivalent to multiplying P-values) for all pixels inside of the mask