"""Swap DINOv2's hand-written attention for PyTorch's fused scaled_dot_product_attention.

Same maths, but it never materialises the full N x N attention matrix, which is what
makes batch 8 at 518x518 fit on a 6 GB GPU. Call after the model is imported.
"""
import torch.nn.functional as F


def use_memory_efficient_attention():
    from depth_anything_v2.dinov2_layers import attention as A

    def forward(self, x, attn_bias=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))

    A.Attention.forward = forward
    A.MemEffAttention.forward = forward
