import torch
from flash_attn import flash_attn_func


def flash_attn_with_sink_func(
    q,
    k,
    v,
    sink: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    out, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    origin_dtype = out.dtype

    # (batch_size, seqlen, nheads, 1)
    lse = lse.transpose(-2, -1).unsqueeze(dim=-1)
    sink = sink.reshape(1, 1, -1, 1)

    multiplier = 1 / (torch.exp(sink - lse) + 1)
    out = (out * multiplier).to(origin_dtype)

    return out
