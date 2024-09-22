from attention.attention import standard_attention
from attention.flash_attention import flash_attention_forward
import torch

def test_flash_attention():
    torch.manual_seed(0)  # for reproducibility
    batch_size, seq_len, head_dim = 2, 1024, 64
    q = torch.randn(batch_size, seq_len, head_dim)
    k = torch.randn(batch_size, seq_len, head_dim)
    v = torch.randn(batch_size, seq_len, head_dim)

    flash_output = flash_attention_forward(q, k, v)
    standard_output = standard_attention(q, k, v)

    torch.testing.assert_close(standard_output, flash_output)