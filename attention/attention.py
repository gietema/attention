import torch
import torch.nn.functional as F
import math


def standard_attention(query, key, value):
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
    attn_scores = F.softmax(scores, dim=-1)
    return torch.bmm(attn_scores, value)
