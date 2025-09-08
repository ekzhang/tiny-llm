import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [N.., D]
        mean = mx.rsqrt(
            x.astype(mx.float32).square().mean(axis=-1, keepdims=True) + self.eps
        )
        return x * mean.astype(x.dtype) * self.weight
