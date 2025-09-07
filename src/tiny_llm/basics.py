import mlx.core as mx


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,  # [N.., I]
    w: mx.array,  # [O, I]
    bias: mx.array | None = None,  # [O]
) -> mx.array:  # [N.., O]
    y = mx.matmul(x, w.T)
    if bias is not None:
        y += bias
    return y


def silu(x: mx.array) -> mx.array:
    pass
