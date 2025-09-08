import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):  # [N.., S] -> [N.., vocab_size]
        output_logits = model(y)
        return output_logits[..., -1, :]

    if sampler is None:

        def sampler(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1)

    tokens: list[int] = list(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    while True:
        logits = _step(model, mx.array([tokens], dtype=mx.int32))
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        next_token = sampler(logprobs)[0]
        tokens.append(int(next_token))
        if next_token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(int(next_token))
        print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass
