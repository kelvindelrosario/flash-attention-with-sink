import torch
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import _flash_attn_backward


class FlashAttentionWithSink(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sink: torch.Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        ctx.save_for_backward(q, k, v, sink, alibi_slopes)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_attn_probs = return_attn_probs

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

        ctx.raw_output = out.clone()
        ctx.lse = lse.clone()

        lse = lse.transpose(-2, -1).unsqueeze(dim=-1)
        sink = sink.reshape(1, 1, -1, 1)

        multiplier = 1 / (torch.exp(sink - lse) + 1)
        out = (out * multiplier).to(origin_dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, sink, alibi_slopes = ctx.saved_tensors
        raw_output = ctx.raw_output
        lse = ctx.lse

        lse = lse.transpose(-2, -1).unsqueeze(dim=-1)
        sink_reshaped = sink.reshape(1, 1, -1, 1)
        multiplier = 1 / (torch.exp(sink_reshaped - lse) + 1)

        grad_sink = torch.sum(
            grad_output * raw_output * multiplier * (1 - multiplier), dim=(0, 1, 3)
        )

        grad_raw_output = (grad_output * multiplier).to(q.dtype)

        # Use flash attention backward function
        # Function signature: _flash_attn_backward(dout, q, k, v, out, softmax_lse,
        # dq, dk, dv, dropout_p, softmax_scale, causal, window_size_left,
        # window_size_right, softcap, alibi_slopes, deterministic, rng_state)
        # Create output tensors
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        _flash_attn_backward(
            grad_raw_output,  # dout: adjusted gradient
            q,  # q
            k,  # k
            v,  # v
            ctx.raw_output,  # out: original output
            lse,  # softmax_lse
            grad_q,  # dq: output parameter
            grad_k,  # dk: output parameter
            grad_v,  # dv: output parameter
            ctx.dropout_p,  # dropout_p
            ctx.softmax_scale,  # softmax_scale
            ctx.causal,  # causal
            ctx.window_size[0],  # window_size_left
            ctx.window_size[1],  # window_size_right
            ctx.softcap,  # softcap
            alibi_slopes,  # alibi_slopes
            ctx.deterministic,  # deterministic
        )

        return (
            grad_q,
            grad_k,
            grad_v,
            grad_sink,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    return FlashAttentionWithSink.apply(
        q,
        k,
        v,
        sink,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
