# ruff: noqa: F401
import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


# "Note that another refactor of this week's code is that all modules now take
# QuantizedWeights instead of mx.array for some weights."
precision = mx.float16


def dequantize_astype(x):
    return dequantize_linear(x).astype(precision)


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        wq_ = dequantize_astype(wq)
        wk_ = dequantize_astype(wk)
        wv_ = dequantize_astype(wv)
        wo_ = dequantize_astype(wo)

        head_dim = wq_.shape[0] // num_heads  # D
        assert wq_.shape == wo_.T.shape == (num_heads * head_dim, hidden_size)
        assert wk_.shape == wv_.shape == (num_kv_heads * head_dim, hidden_size)
        self.wq = wq_
        self.wk = wk_
        self.wv = wv_
        self.wo = wo_

        assert bq.shape == (num_heads * head_dim,)
        assert bk.shape == bv.shape == (num_kv_heads * head_dim,)
        self.bq = bq
        self.bk = bk
        self.bv = bv

        self.rope = RoPE(head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,  # [B, L', E]
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L_, E = x.shape
        q = linear(x, self.wq, self.bq).reshape(B, L_, self.num_heads, -1)
        k = linear(x, self.wk, self.bk).reshape(B, L_, self.num_kv_heads, -1)
        v = linear(x, self.wv, self.bv).reshape(B, L_, self.num_kv_heads, -1)
        q = self.rope(q, offset=[slice(o, o + L_) for o in offsets])
        k = self.rope(k, offset=[slice(o, o + L_) for o in offsets])

        # now k, v shape: [B, L, H, D]
        #        q shape: [B, L', H, D]
        k, v = cache.update_and_fetch(k, v)

        q = q.swapaxes(-3, -2).astype(mx.float32)
        k = k.swapaxes(-3, -2).astype(mx.float32)
        v = v.swapaxes(-3, -2).astype(mx.float32)

        y = scaled_dot_product_attention_grouped(q, k, v, mask=mask)
        y = y.astype(x.dtype).swapaxes(-3, -2).reshape(B, L_, -1)

        return linear(y, self.wo)  # [B, L', E]


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = dequantize_astype(w_gate)
        self.w_up = dequantize_astype(w_up)
        self.w_down = dequantize_astype(w_down)

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
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.gqa = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, rms_norm_eps
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        assert not use_flash_attention, "Flash attention is not implemented"

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x = x + self.gqa(
            self.input_layernorm(x),
            offsets=[offset for _ in range(x.shape[0])],
            cache=cache,
            mask=mask,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        hidden_size = mlx_model.args.hidden_size
        num_hidden_layers = mlx_model.args.num_hidden_layers
        intermediate_size = mlx_model.args.intermediate_size
        num_attention_heads = mlx_model.args.num_attention_heads
        rms_norm_eps = mlx_model.args.rms_norm_eps
        vocab_size = mlx_model.args.vocab_size
        num_key_value_heads = mlx_model.args.num_key_value_heads
        max_position_embeddings = mlx_model.args.max_position_embeddings
        rope_theta = mlx_model.args.rope_theta

        self.embed_tokens = Embedding(
            vocab_size=vocab_size,
            embedding_dim=hidden_size,
            weight=dequantize_astype(mlx_model.model.embed_tokens),
        )
        self.layers: list[Qwen2TransformerBlock] = []
        for i in range(num_hidden_layers):
            mlx_layer = mlx_model.model.layers[i]
            block = Qwen2TransformerBlock(
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                wq=QuantizedWeights.from_mlx_layer(mlx_layer.self_attn.q_proj),
                wk=QuantizedWeights.from_mlx_layer(mlx_layer.self_attn.k_proj),
                wv=QuantizedWeights.from_mlx_layer(mlx_layer.self_attn.v_proj),
                wo=QuantizedWeights.from_mlx_layer(mlx_layer.self_attn.o_proj),
                bq=mlx_layer.self_attn.q_proj.bias.astype(precision),
                bk=mlx_layer.self_attn.k_proj.bias.astype(precision),
                bv=mlx_layer.self_attn.v_proj.bias.astype(precision),
                w_gate=QuantizedWeights.from_mlx_layer(mlx_layer.mlp.gate_proj),
                w_up=QuantizedWeights.from_mlx_layer(mlx_layer.mlp.up_proj),
                w_down=QuantizedWeights.from_mlx_layer(mlx_layer.mlp.down_proj),
                w_input_layernorm=mlx_layer.input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_layer.post_attention_layernorm.weight.astype(
                    precision
                ),
                max_seq_len=max_position_embeddings,
                theta=rope_theta,
            )
            self.layers.append(block)

        # final layernorm
        self.norm = RMSNorm(
            dim=hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=rms_norm_eps,
        )

        self.num_hidden_layers = num_hidden_layers

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        x = self.embed_tokens(inputs)  # [N.., E]
        for i, layer in enumerate(self.layers):
            x = layer(x, offset=offset, cache=cache[i], mask="causal")
        x = self.norm(x)
        x = self.embed_tokens.as_linear(x)  # [N.., vocab_size]
        return x
