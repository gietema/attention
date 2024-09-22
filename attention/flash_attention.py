import torch
import math


def compute_attention_block(current_attn_output, query_block_i, key_block_j, value_block_j, current_softmax_denom, current_max_attn_score, scale):
    # Step 9: Compute block_attn_logits (S_ij)
    block_attn_logits = query_block_i @ key_block_j.transpose(1, 2) * scale
    # Step 10: compute block_max_attn_score (m_ij_tilde), block_attn_exp_scores (P_ij_tilde), block_attn_exp_sum (l_ij_tilde)
    block_max_attn_score = torch.max(block_attn_logits, dim=-1).values
    block_attn_exp_scores = torch.exp(block_attn_logits - block_max_attn_score.unsqueeze(-1))
    block_attn_exp_sum = torch.sum(block_attn_exp_scores, dim=-1)
    # Step 11: Compute m_i_new and l_i_new
    updated_max_attn_score = torch.maximum(current_max_attn_score, block_max_attn_score)
    updated_softmax_denom = (
        torch.exp(current_max_attn_score - updated_max_attn_score) * current_softmax_denom
        + 
        torch.exp(block_max_attn_score - updated_max_attn_score) * block_attn_exp_sum
    )
    # Step 12: Update
    past_block_adjustment = current_softmax_denom.unsqueeze(-1) * torch.exp(current_max_attn_score - updated_max_attn_score).unsqueeze(-1) * current_attn_output
    new_block_contribution = torch.exp(block_max_attn_score - updated_max_attn_score).unsqueeze(-1) * (block_attn_exp_scores @ value_block_j)
    updated_attn_output = (past_block_adjustment + new_block_contribution) / updated_softmax_denom.unsqueeze(-1)
    return updated_attn_output, updated_softmax_denom, updated_max_attn_score

def flash_attention_forward(query, key, value, sram_size: int = 65536):
    """
    sram_size: int = 65536 to get block size 256
    """
    batch_size, N, d = query.shape
    # Step 1: set block sizes
    Bc = int(sram_size / (4 * d))
    Br = int(min(Bc, d))
    print(f"{N=}, {Br=}, {Bc=}, {d=}")
    
    scale = 1.0 / math.sqrt(d)

    # Step 2: Initialize attn_output, softmax_denom, and max_attn_score (in FA in HBM) 
    initial_attn_output = torch.zeros_like(query)
    initial_softmax_denom = torch.zeros(batch_size, N, device=query.device)
    initial_max_attn_score = torch.ones(batch_size, N, device=query.device) * float('-inf')

    # Step 3: Divide Q, K and V into Tr and Tc blocks
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    query_blocks = torch.split(query, Br, dim=1)
    key_blocks = torch.split(key, Bc, dim=1)
    value_blocks = torch.split(value, Bc, dim=1)
    attn_output_blocks = list(torch.split(initial_attn_output, Br, dim=1))
    softmax_denom_blocks = list(torch.split(initial_softmax_denom, Br, dim=1))
    max_attn_score_blocks = list(torch.split(initial_max_attn_score, Br, dim=1))

    # Step 5: for 1 <= j <= Tc, do
    for j in range(Tc):
        # Step 6: Simulate loading blocks from HBM to on-chip SRAM
        key_block_j = key_blocks[j]
        value_block_j = value_blocks[j]
        # Step 7: For 1 <= i <= Tr, do
        for i in range(Tr):
            attn_output_blocks[i], softmax_denom_blocks[i], max_attn_score_blocks[i] = compute_attention_block(
                 current_attn_output=attn_output_blocks[i],
                 query_block_i=query_blocks[i],
                 key_block_j=key_block_j,
                 value_block_j=value_block_j,
                 current_softmax_denom=softmax_denom_blocks[i],
                 current_max_attn_score=max_attn_score_blocks[i],
                 scale=scale
            )
    
    final_attn_output = torch.cat(attn_output_blocks, dim=1)
    # final_softmax_denom = torch.cat(softmax_denom_blocks, dim=1)
    # final_max_attn_score = torch.cat(max_attn_score_blocks, dim=1)
    
    return final_attn_output