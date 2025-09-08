import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        assert weight.shape == (vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        # [N..] -> [N.., embedding_dim]
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        # [N.., embedding_dim] -> [N.., vocab_size]
        return x @ self.weight.T
