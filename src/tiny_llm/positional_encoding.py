import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,  # D
        seq_len: int,  # L
        base: int = 10000,
        traditional: bool = False,
    ):
        # [D // 2]
        theta = mx.power(base, -mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        # [L, D // 2]
        freqs = mx.arange(seq_len)[:, mx.newaxis] * theta
        self.sin_freqs = mx.sin(freqs)
        self.cos_freqs = mx.cos(freqs)
        self.traditional = traditional

    def __call__(
        self,
        x: mx.array,  # [N, L, H, D]
        offset: list[slice] | slice | None = None,
    ) -> mx.array:
        N, L, H, D = x.shape

        seq_idx: slice | mx.array  # [L] or [N, L]
        if offset is None:
            seq_idx = slice(0, L)
        elif isinstance(offset, slice):
            assert offset.stop - offset.start == L, f"offset must be of length {L}"
            seq_idx = offset
        elif isinstance(offset, list):
            for o in offset:
                assert o.stop - o.start == L, f"offset must be of length {L}"
            seq_idx = mx.array([list(range(i.start, i.stop)) for i in offset])

        cos_basis = self.cos_freqs[seq_idx].reshape(-1, L, 1, D // 2)
        sin_basis = self.sin_freqs[seq_idx].reshape(-1, L, 1, D // 2)

        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)
            x0, x1 = x[..., 0], x[..., 1]
        else:
            x = x.reshape(N, L, H, 2, D // 2)
            x0, x1 = x[..., 0, :], x[..., 1, :]

        y0 = x0 * cos_basis - x1 * sin_basis
        y1 = x0 * sin_basis + x1 * cos_basis
        return mx.stack([y0, y1], axis=-1 if self.traditional else -2).reshape(
            N, L, H, D
        )
