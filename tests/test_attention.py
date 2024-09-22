from attention.attention import standard_attention
import torch


def test_attention():
    torch.manual_seed(0)  # for reproducibility
    batch_size, seq_len, head_dim = 2, 1024, 64
    q = torch.randn(batch_size, seq_len, head_dim)
    k = torch.randn(batch_size, seq_len, head_dim)
    v = torch.randn(batch_size, seq_len, head_dim)

    flash_output = standard_attention(q, k, v)
    standard_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(standard_output, flash_output)