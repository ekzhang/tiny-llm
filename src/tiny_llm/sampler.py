import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        if top_k:
            mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
            logprobs = mx.put_along_axis(
                logprobs, mask_idx, mx.array(-mx.inf, logprobs.dtype), axis=-1
            )

        if top_p:
            probs = mx.exp(logprobs)
            sorted_idx = mx.argsort(logprobs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)

            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
            # Rearrange cumulative probs back to original order
            inverse_indices = mx.put_along_axis(
                mx.zeros_like(sorted_idx),
                sorted_idx,
                mx.arange(sorted_idx.shape[-1], dtype=sorted_idx.dtype),
                axis=-1,
            )
            cumulative_probs = mx.take_along_axis(
                cumulative_probs, inverse_indices, axis=-1
            )
            logprobs = mx.where(cumulative_probs > 1 - top_p, logprobs, -mx.inf)

        return mx.random.categorical(logprobs / temp)

    return sample
