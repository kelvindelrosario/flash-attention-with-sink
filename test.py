import torch
from flash_attn_with_sink import flash_attn_with_sink_func
from naive_attn_with_sink import eager_attention_forward


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
    )
    key = torch.randn(
        (batch, num_key_value_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn(
        (batch, num_key_value_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    sink = torch.randn((num_attention_heads,), dtype=torch.bfloat16, device="cuda")

    # Create causal attention mask
    # The mask should be of shape (batch, num_heads, seq_len, seq_len)
    # For causal attention, we mask out future positions (set them to large negative value)
    attention_mask = torch.triu(
        torch.full(
            (seq_len, seq_len), float("-inf"), device="cuda", dtype=torch.bfloat16
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
