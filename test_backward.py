import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from flash_attn_with_sink import flash_attn_with_sink_func


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    print(num_key_value_heads, n_rep)
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


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
    print(query.shape, key_states.shape, value_states.shape, sink.shape)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = sink.reshape(1, -1, 1, 1).expand(
        query.shape[0], -1, query.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; 
    # it prevents overflow in BF16/FP16 when training with bsz>1 we clamp max values.
    combined_logits = combined_logits - combined_logits.max(
        dim=-1, keepdim=True
    ).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


if __name__ == "__main__":
    batch = 1
    num_attention_heads = 64
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads
    head_dim = 64
    seq_len = 512
    scaling = head_dim**-0.5

    torch.cuda.set_device(0)
    query = torch.randn(
        (batch, num_attention_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    key = torch.randn(
        (batch, num_key_value_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    value = torch.randn(
        (batch, num_key_value_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    sink = torch.randn(
        (num_attention_heads,), 
        dtype=torch.bfloat16, 
        device="cuda",
        requires_grad=True,
    )

    # Create causal attention mask
    # The mask should be of shape (batch, num_heads, seq_len, seq_len)
    # For causal attention, we mask out future positions 
    # (set them to large negative value)
    attention_mask = torch.triu(
        torch.full(
            (seq_len, seq_len), 
            float("-inf"), 
            device="cuda", 
            dtype=torch.bfloat16
        ),
        diagonal=1,
    )
    attention_mask = (
        attention_mask.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch, num_attention_heads, -1, -1)
    )

    print("Running eager attention forward...")
    eager_output, eager_weights = eager_attention_forward(
        query,
        key,
        value,
        sink,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=0.0,
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

    # Test backward pass
    print("\n" + "="*50)
    print("Testing backward pass...")
    
    # Reset gradients (handle None case)
    if query.grad is not None:
        query.grad.zero_()
    if key.grad is not None:
        key.grad.zero_()
    if value.grad is not None:
        value.grad.zero_()
    if sink.grad is not None:
        sink.grad.zero_()
    
    # Compute loss for eager attention
    target = torch.randn_like(eager_output, device="cuda")
    eager_loss = F.mse_loss(eager_output, target)
    
    print(f"Eager loss: {eager_loss.item():.6f}")
    
    # Backward pass for eager attention
    eager_loss.backward()
    
    # Save eager gradients
    eager_query_grad = query.grad.clone()
    eager_key_grad = key.grad.clone()
    eager_value_grad = value.grad.clone()
    eager_sink_grad = sink.grad.clone()
    
    print("\nEager gradient information:")
    print(f"Query gradient norm: {eager_query_grad.norm().item():.6f}")
    print(f"Key gradient norm: {eager_key_grad.norm().item():.6f}")
    print(f"Value gradient norm: {eager_value_grad.norm().item():.6f}")
    print(f"Sink gradient norm: {eager_sink_grad.norm().item():.6f}")
    
    # Reset gradients for flash attention (handle None case)
    if query.grad is not None:
        query.grad.zero_()
    if key.grad is not None:
        key.grad.zero_()
    if value.grad is not None:
        value.grad.zero_()
    if sink.grad is not None:
        sink.grad.zero_()
    
    # Compute loss for flash attention
    flash_loss = F.mse_loss(flash_output, target)
    
    print(f"\nFlash loss: {flash_loss.item():.6f}")
    
    # Backward pass for flash attention
    flash_loss.backward()
    
    # Save flash gradients
    flash_query_grad = query.grad.clone()
    flash_key_grad = key.grad.clone()
    flash_value_grad = value.grad.clone()
    flash_sink_grad = sink.grad.clone()
    
    print("\nFlash gradient information:")
    print(f"Query gradient norm: {flash_query_grad.norm().item():.6f}")
    print(f"Key gradient norm: {flash_key_grad.norm().item():.6f}")
    print(f"Value gradient norm: {flash_value_grad.norm().item():.6f}")
    print(f"Sink gradient norm: {flash_sink_grad.norm().item():.6f}")
    
    # Compare gradients
    print("\n" + "="*50)
    print("Comparing gradients...")
    
    # Calculate gradient differences
    query_grad_diff = torch.abs(eager_query_grad - flash_query_grad).max().item()
    key_grad_diff = torch.abs(eager_key_grad - flash_key_grad).max().item()
    value_grad_diff = torch.abs(eager_value_grad - flash_value_grad).max().item()
    sink_grad_diff = torch.abs(eager_sink_grad - flash_sink_grad).max().item()
    
    print(f"Query gradient max difference: {query_grad_diff:.2e}")
    print(f"Key gradient max difference: {key_grad_diff:.2e}")
    print(f"Value gradient max difference: {value_grad_diff:.2e}")
    print(f"Sink gradient max difference: {sink_grad_diff:.2e}")
    
    # Check if gradients are close (within tolerance)
    tolerance = 1e-3  # Adjust tolerance as needed
    query_grad_close = query_grad_diff < tolerance
    key_grad_close = key_grad_diff < tolerance
    value_grad_close = value_grad_diff < tolerance
    sink_grad_close = sink_grad_diff < tolerance
    
    print(f"\nGradient comparison (tolerance: {tolerance}):")
    print(f"Query gradients close: {'✅' if query_grad_close else '❌'}")
    print(f"Key gradients close: {'✅' if key_grad_close else '❌'}")
    print(f"Value gradients close: {'✅' if value_grad_close else '❌'}")
    print(f"Sink gradients close: {'✅' if sink_grad_close else '❌'}")
    
    # Check if gradients are non-zero
    query_grad_zero = eager_query_grad.norm().item() < 1e-8
    key_grad_zero = eager_key_grad.norm().item() < 1e-8
    value_grad_zero = eager_value_grad.norm().item() < 1e-8
    sink_grad_zero = eager_sink_grad.norm().item() < 1e-8
    
    print(f"\nGradient non-zero check:")
    print(f"Query gradient non-zero: {'✅' if not query_grad_zero else '❌'}")
    print(f"Key gradient non-zero: {'✅' if not key_grad_zero else '❌'}")
    print(f"Value gradient non-zero: {'✅' if not value_grad_zero else '❌'}")
    print(f"Sink gradient non-zero: {'✅' if not sink_grad_zero else '❌'}")
    
    all_grads_close = (query_grad_close and key_grad_close and 
                      value_grad_close and sink_grad_close)
    all_grads_nonzero = not (query_grad_zero or key_grad_zero or 
                            value_grad_zero or sink_grad_zero)
    
    print(f"\nOverall result:")
    print(f"  All gradients close: {'✅' if all_grads_close else '❌'}")
    print(f"  All gradients non-zero: {'✅' if all_grads_nonzero else '❌'}")
    
    if all_grads_close and all_grads_nonzero:
        print("\n🎉 Backward test passed! Gradients match and are non-zero.")
    else:
        print("\n❌ Backward test failed!")
        if not all_grads_close:
            print("  - Some gradients don't match between eager and flash attention")
        if not all_grads_nonzero:
            print("  - Some gradients are zero")
