import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from flash_attn_with_sink import flash_attn_with_sink_func


def get_default_config():
    """Get default configuration for tests"""
    return {
        "batch": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "seq_len": 512,
        "device": "cuda",
        "dtype": torch.bfloat16,
        "dropout": 0.0,
    }


def create_test_tensors(
    batch,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    seq_len,
    device,
    dtype,
    requires_grad=False,
):
    """Create test tensors for attention computation"""
    query = torch.randn(
        (batch, num_attention_heads, seq_len, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        (batch, num_key_value_heads, seq_len, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    value = torch.randn(
        (batch, num_key_value_heads, seq_len, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    sink = torch.randn(
        (num_attention_heads,), dtype=dtype, device=device, requires_grad=requires_grad
    )
    return query, key, value, sink


def create_attention_mask(batch, num_attention_heads, seq_len, device, dtype):
    """Create causal attention mask"""
    attention_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    attention_mask = (
        attention_mask.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch, num_attention_heads, -1, -1)
    )
    return attention_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    print(num_key_value_heads, n_rep)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# from https://github.com/huggingface/transformers/blob/369c99d0cea403b77bd0aef818527106453fd9fc/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L227
def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sink: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    num_key_value_groups: int = 8,
    **kwargs,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = sink.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def test_gradients(
    batch=1,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=64,
    seq_len=512,
    device="cuda",
    dtype=torch.bfloat16,
    dropout=0.0,
    **kwargs,
):
    """Test backward gradients for q, k, v, and sink"""
    print("\n" + "=" * 50)
    print("TESTING GRADIENTS")
    print("=" * 50)

    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim**-0.5

    torch.cuda.set_device(0)

    # Create tensors for eager attention (requires_grad=True)
    query_eager, key_eager, value_eager, sink_eager = create_test_tensors(
        batch,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        seq_len,
        device,
        dtype,
        requires_grad=True,
    )

    # Create tensors for flash attention (same data but requires_grad=True)
    query_flash = query_eager.clone().detach().requires_grad_(True)
    key_flash = key_eager.clone().detach().requires_grad_(True)
    value_flash = value_eager.clone().detach().requires_grad_(True)
    sink_flash = sink_eager.clone().detach().requires_grad_(True)

    # Create causal attention mask
    attention_mask = create_attention_mask(
        batch, num_attention_heads, seq_len, device, dtype
    )

    print("Running eager attention forward...")
    eager_output, _ = eager_attention_forward(
        query_eager,
        key_eager,
        value_eager,
        sink_eager,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        num_key_value_groups=num_key_value_groups,
    )

    print("Running flash attention forward...")
    # Reshape tensors for flash attention (batch, seq_len, num_heads, head_dim)
    q_flash_reshaped = query_flash.transpose(
        1, 2
    )  # (batch, seq_len, num_heads, head_dim)
    k_flash_reshaped = key_flash.transpose(
        1, 2
    )  # (batch, seq_len, num_kv_heads, head_dim)
    v_flash_reshaped = value_flash.transpose(
        1, 2
    )  # (batch, seq_len, num_kv_heads, head_dim)

    flash_output = flash_attn_with_sink_func(
        q_flash_reshaped,
        k_flash_reshaped,
        v_flash_reshaped,
        sink_flash,
        softmax_scale=scaling,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    )

    # Compare forward outputs
    print(f"\nForward pass comparison:")
    print(f"Eager output shape: {eager_output.shape}, dtype: {eager_output.dtype}")
    print(f"Flash output shape: {flash_output.shape}, dtype: {flash_output.dtype}")
    print(
        f"Max absolute difference: {torch.max(torch.abs(eager_output - flash_output))}"
    )

    # Backward pass for eager attention
    print("\nRunning backward pass for eager attention...")
    loss_eager = eager_output.sum()
    loss_eager.backward()

    # Store eager gradients
    query_grad_eager = query_eager.grad.clone()
    key_grad_eager = key_eager.grad.clone()
    value_grad_eager = value_eager.grad.clone()
    sink_grad_eager = sink_eager.grad.clone()

    # Backward pass for flash attention
    print("Running backward pass for flash attention...")
    loss_flash = flash_output.sum()
    loss_flash.backward()

    # Store flash gradients
    query_grad_flash = query_flash.grad.clone()
    key_grad_flash = key_flash.grad.clone()
    value_grad_flash = value_flash.grad.clone()
    sink_grad_flash = sink_flash.grad.clone()

    # Compare gradients
    print("\n" + "=" * 50)
    print("GRADIENT COMPARISON")
    print("=" * 50)

    print(f"\nQuery gradients:")
    print(f"  Eager grad shape: {query_grad_eager.shape}")
    print(f"  Flash grad shape: {query_grad_flash.shape}")
    print(
        f"  Max absolute difference: {torch.max(torch.abs(query_grad_eager - query_grad_flash))}"
    )
    print(
        f"  Mean absolute difference: {torch.mean(torch.abs(query_grad_eager - query_grad_flash))}"
    )
    print(
        f"  Relative error: {torch.mean(torch.abs(query_grad_eager - query_grad_flash) / (torch.abs(query_grad_eager) + 1e-8))}"
    )

    print(f"\nKey gradients:")
    print(f"  Eager grad shape: {key_grad_eager.shape}")
    print(f"  Flash grad shape: {key_grad_flash.shape}")
    print(
        f"  Max absolute difference: {torch.max(torch.abs(key_grad_eager - key_grad_flash))}"
    )
    print(
        f"  Mean absolute difference: {torch.mean(torch.abs(key_grad_eager - key_grad_flash))}"
    )
    print(
        f"  Relative error: {torch.mean(torch.abs(key_grad_eager - key_grad_flash) / (torch.abs(key_grad_eager) + 1e-8))}"
    )

    print(f"\nValue gradients:")
    print(f"  Eager grad shape: {value_grad_eager.shape}")
    print(f"  Flash grad shape: {value_grad_flash.shape}")
    print(
        f"  Max absolute difference: {torch.max(torch.abs(value_grad_eager - value_grad_flash))}"
    )
    print(
        f"  Mean absolute difference: {torch.mean(torch.abs(value_grad_eager - value_grad_flash))}"
    )
    print(
        f"  Relative error: {torch.mean(torch.abs(value_grad_eager - value_grad_flash) / (torch.abs(value_grad_eager) + 1e-8))}"
    )

    print(f"\nSink gradients:")
    print(f"  Eager grad shape: {sink_grad_eager.shape}")
    print(f"  Flash grad shape: {sink_grad_flash.shape}")
    print(
        f"  Max absolute difference: {torch.max(torch.abs(sink_grad_eager - sink_grad_flash))}"
    )
    print(
        f"  Mean absolute difference: {torch.mean(torch.abs(sink_grad_eager - sink_grad_flash))}"
    )
    print(
        f"  Relative error: {torch.mean(torch.abs(sink_grad_eager - sink_grad_flash) / (torch.abs(sink_grad_eager) + 1e-8))}"
    )

    # Print some sample gradient values
    print(f"\nSample gradient values:")
    print(f"  Query eager grad sample (first 3x3): \n{query_grad_eager[0, 0, :3, :3]}")
    print(f"  Query flash grad sample (first 3x3): \n{query_grad_flash[0, 0, :3, :3]}")
    print(f"  Key eager grad sample (first 3x3): \n{key_grad_eager[0, 0, :3, :3]}")
    print(f"  Key flash grad sample (first 3x3): \n{key_grad_flash[0, 0, :3, :3]}")
    print(f"  Value eager grad sample (first 3x3): \n{value_grad_eager[0, 0, :3, :3]}")
    print(f"  Value flash grad sample (first 3x3): \n{value_grad_flash[0, 0, :3, :3]}")
    # Print sink gradients
    print(f"  Sink eager grad sample (first 10): {sink_grad_eager[:10]}")
    print(f"  Sink flash grad sample (first 10): {sink_grad_flash[:10]}")


def test_forward(
    batch=1,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=64,
    seq_len=512,
    device="cuda",
    dtype=torch.bfloat16,
    dropout=0.0,
    **kwargs,
):
    """Test forward pass comparison between eager and flash attention"""
    print("=" * 50)
    print("FORWARD PASS TEST")
    print("=" * 50)

    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim**-0.5

    torch.cuda.set_device(0)

    # Create test tensors
    query, key, value, sink = create_test_tensors(
        batch,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        seq_len,
        device,
        dtype,
        requires_grad=False,
    )

    # Create causal attention mask
    attention_mask = create_attention_mask(
        batch, num_attention_heads, seq_len, device, dtype
    )

    print("Running eager attention forward...")
    eager_output, eager_weights = eager_attention_forward(
        query,
        key,
        value,
        sink,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        num_key_value_groups=num_key_value_groups,
    )

    print("Running flash attention forward...")
    # Reshape tensors for flash attention (batch, seq_len, num_heads, head_dim)
    q_flash = query.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
    k_flash = key.transpose(1, 2)  # (batch, seq_len, num_kv_heads, head_dim)
    v_flash = value.transpose(1, 2)  # (batch, seq_len, num_kv_heads, head_dim)

    flash_output = flash_attn_with_sink_func(
        q_flash,
        k_flash,
        v_flash,
        sink,
        softmax_scale=scaling,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    )

    # Compare outputs
    print(f"Eager output shape: {eager_output.shape}, dtype: {eager_output.dtype}")
    print(f"Flash output shape: {flash_output.shape}, dtype: {flash_output.dtype}")

    print(
        f"Max absolute difference: {torch.max(torch.abs(eager_output - flash_output))}"
    )
    print(
        f"Mean absolute difference: {torch.mean(torch.abs(eager_output - flash_output))}"
    )
    print(
        f"Relative error: {torch.mean(torch.abs(eager_output - flash_output) / (torch.abs(eager_output) + 1e-8))}"
    )

    print("\nEager output sample (first 5x5 elements):")
    print(eager_output[0, 0, :5, :5])
    print("\nFlash output sample (first 5x5 elements):")
    print(flash_output[0, 0, :5, :5])


if __name__ == "__main__":
    # Get default configuration
    config = get_default_config()

    # Run forward pass tests
    test_forward(**config)

    # Run gradient tests
    test_gradients(**config)
