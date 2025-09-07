import mlx.core as mx
from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: mx.array,  # [N.., L_query, D]
    key: mx.array,  # [N.., L_key, D]
    value: mx.array,  # [N.., L_key, D]
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:  # [N.., L_query, D]
    if scale is None:
        scale = mx.rsqrt(query.shape[-1])

    # [N.., L_query, L_key]
    attn_logits = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        attn_logits += mask

    attn_weights = softmax(attn_logits, axis=-1)  # [N.., L_query, L_key]
    return mx.matmul(attn_weights, value)  # [N.., L_query, D]


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,  # E
        num_heads: int,  # H
        wq: mx.array,  # [H x D, E]
        wk: mx.array,  # [H x D, E]
        wv: mx.array,  # [H x D, E]
        wo: mx.array,  # [E, H x D]
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,  # [N.., L, E]
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        *batch_dims, l_dim, e_dim = query.shape

        # [N.., L, H, D]
        nlhd_shape = (*batch_dims, l_dim, self.num_heads, -1)

        q = linear(query, self.wq).reshape(nlhd_shape)
        k = linear(key, self.wk).reshape(nlhd_shape)
        v = linear(value, self.wv).reshape(nlhd_shape)

        # [N.., H, L, D]
        q = q.swapaxes(-3, -2)
        k = k.swapaxes(-3, -2)
        v = v.swapaxes(-3, -2)
        attn = scaled_dot_product_attention_simple(q, k, v, None, mask)

        return linear(attn.swapaxes(-3, -2).reshape((*batch_dims, l_dim, -1)), self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    assert S >= L, f"causal mask requires S >= L but got L={L}, S={S}"
    return mx.triu(mx.full((L, S), -mx.inf, dtype=dtype), S - L + 1)


def scaled_dot_product_attention_grouped(
    query: mx.array,  # [N.., H_q, L_q, D]
    key: mx.array,  # [N.., H, L_k, D]
    value: mx.array,  # [N.., H, L_k, D]
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    *B, H_q, L_q, D = query.shape
    H, L_k, D = key.shape[-3:]

    if scale is None:
        scale = mx.rsqrt(D)

    assert H_q % H == 0, f"query heads {H_q} must be multiple of k/v heads {H}"
    G = H_q // H

    query = query.reshape(*B, H, G, L_q, D)
    key = key.reshape(*B, H, 1, L_k, D)
    value = value.reshape(*B, H, 1, L_k, D)

    # [N.., H, G, L_q, L_k]
    attn_logits = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            mask = causal_mask(L_q, L_k, attn_logits.dtype)
        attn_logits += mx.broadcast_to(mask, attn_logits.shape)
    attn_weights = softmax(attn_logits, axis=-1)

    return mx.matmul(attn_weights, value).reshape(*B, H_q, L_q, D)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
