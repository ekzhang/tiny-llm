# ruff: noqa: F401
import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,  # E
        num_heads: int,  # H_q
        num_kv_heads: int,  # H
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        head_dim = wq.shape[0] // num_heads  # D
        assert wq.shape == wo.T.shape == (num_heads * head_dim, hidden_size)
        assert wk.shape == wv.shape == (num_kv_heads * head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        assert bq.shape == (num_heads * head_dim,)
        assert bk.shape == bv.shape == (num_kv_heads * head_dim,)
        self.bq = bq
        self.bk = bk
        self.bv = bv

        self.rope = RoPE(head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, -1)
        k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, -1)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, -1)
        q = self.rope(q)
        k = self.rope(k)
        q = q.swapaxes(-3, -2).astype(mx.float32)
        k = k.swapaxes(-3, -2).astype(mx.float32)
        v = v.swapaxes(-3, -2).astype(mx.float32)
        y = scaled_dot_product_attention_grouped(q, k, v, mask=mask)
        y = y.astype(x.dtype).swapaxes(-3, -2).reshape(B, L, -1)
        return linear(y, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        intermediate = silu(linear(x, self.w_gate)) * linear(x, self.w_up)
        return linear(intermediate, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
