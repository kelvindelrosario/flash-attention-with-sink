import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
from functools import cache
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


def flash_attention_with_sink_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    params = get_default_args(_flash_attn_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
    )
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    outputs = _flash_attn_forward(**params)
    # out: (batch_size, seqlen, nheads, headdim).
    # softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen)
    if len(outputs) == 8:
        out, _, _, _, _, lse, _, _ = outputs
    else:
        assert len(outputs) == 4
        out, lse, _, _ = outputs

    # (batch_size, seqlen, nheads, 1)
    lse = lse.transpose(-2, -1).unsqueeze(dim=-1)
    sink = sink.reshape(1, 1, -1, 1)

    multiplier = 1 / (torch.exp(sink - lse) + 1)
    out = (out * multiplier).to(torch.bfloat16)

    return out
