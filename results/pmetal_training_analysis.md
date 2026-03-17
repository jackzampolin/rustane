# PMetal Training Analysis: Can You Pretrain with PMetal?

> Analysis of [Epistates/pmetal](https://github.com/Epistates/pmetal) v0.3.7 for pretraining 110M and 600M parameter models on Apple Silicon.

## TL;DR

PMetal is a **fine-tuning framework**, not a pretraining framework. It can technically do "full fine-tuning" (all parameters trainable), but lacks the infrastructure for efficient pretraining from scratch. Adapting it for pretraining is possible but would require significant work and would be slower than purpose-built alternatives.

---

## What PMetal Actually Is

PMetal ("Powdered Metal") is a 17-crate Rust ML SDK for Apple Silicon:

- **Primary focus**: LoRA/QLoRA fine-tuning of pre-trained HuggingFace models
- **Backend**: MLX (Apple's ML framework) via `mlx-rs` bindings, plus custom Metal shaders
- **Gradient computation**: MLX autograd (`nn::value_and_grad()`), not custom backward passes
- **Models supported**: Llama, Qwen, DeepSeek, Mistral, Gemma, Phi, etc.
- **Training methods**: SFT, LoRA, QLoRA, DPO, SimPO, ORPO, KTO, GRPO, DAPO, distillation

### What "Full FT" Means in PMetal

The README claims "Full FT" support for most models. This means **all LoRA/model parameters are trainable** (no frozen base), but:

- The training loop still expects a pre-trained model from HuggingFace
- Checkpoints save `lora_weights.safetensors` (adapter format)
- No random weight initialization path
- No pretraining data pipeline (expects chat/instruction format, not raw text)
- No pretraining-specific learning rate schedules or warmup strategies

---

## Architecture Comparison: PMetal vs Rustane

| Feature | PMetal | Rustane |
|---------|--------|---------|
| **Training type** | Fine-tuning (LoRA/full) | Pretraining from scratch |
| **Backward pass** | MLX autograd | Hand-written CPU/ANE backward |
| **Optimizer** | AdamW (MLX), Metal fused opt | Metal AdamW (custom kernel) |
| **Forward compute** | MLX + custom Metal shaders | ANE kernels + CPU |
| **Precision** | f32 via MLX | f32 CPU, f16 ANE |
| **Memory model** | MLX lazy evaluation | Pre-allocated workspaces |
| **Weight init** | Load from safetensors | Random init (LCG) |
| **Data format** | Chat/instruction JSONL | Raw uint16 mmap'd tokens |
| **Gradient checkpointing** | Placeholder (not active) | N/A (fits in memory) |
| **Sequence packing** | Yes (2-5x throughput) | No |

---

## Benchmark Estimates: 110M and 600M Models

### Model Configurations

**110M model** (comparable to Orion paper):
```
dim=768, hidden=2048, heads=12, kv_heads=12, layers=12, vocab=32000, seq=512
Parameters: ~110M
```

**600M model** (comparable to Qwen3-0.6B):
```
dim=1024, hidden=3072, heads=16, kv_heads=8, layers=28, vocab=151936, seq=512
Parameters: ~508M (Qwen3 actual), or ~600M with larger vocab/hidden
```

### Memory Requirements (f32 training)

| Component | 110M | 600M |
|-----------|------|------|
| Weights (f32) | 440 MB | 2.0 GB |
| Gradients (f32) | 440 MB | 2.0 GB |
| Optimizer state (AdamW, 2x) | 880 MB | 4.0 GB |
| Activations (seq=512, batch=1) | ~200 MB | ~800 MB |
| **Total (LoRA r=16)** | **~1.5 GB** | **~4 GB** |
| **Total (full param)** | **~2.0 GB** | **~9 GB** |
| **Total (pretraining, full+workspace)** | **~2.5 GB** | **~12 GB** |

All Apple Silicon Macs with 16GB+ RAM can handle 110M. 600M requires 16GB+ for LoRA, 24GB+ for full pretraining.

### Throughput Estimates on M4 Max

PMetal's observed throughput (from their docs):

| Mode | Throughput | Notes |
|------|-----------|-------|
| LoRA fine-tuning (Qwen3-0.6B, batch=4, seq=512) | 1700-1800 tok/s | Deferred eval, fused path |
| LoRA fine-tuning (basic) | 500-600 tok/s | Per-step eval |
| mlx-lm reference (JIT compiled) | 2200-2300 tok/s | Full graph fusion |

**Key insight**: These numbers are for **LoRA fine-tuning**, which only trains ~0.1-1% of parameters. Full pretraining trains 100% of parameters.

#### Extrapolated Pretraining Throughput

For full-parameter pretraining, the bottleneck shifts from LoRA adapter math to full matmul backward passes through every layer.

**110M model (pretraining from scratch, M4 Max):**

| Metric | Estimate | Reasoning |
|--------|----------|-----------|
| Forward FLOPS | ~1.3 GFLOP/token | 6 * 2 * N_params (standard estimate) |
| Backward FLOPS | ~2.6 GFLOP/token | ~2x forward |
| Total FLOPS/token | ~3.9 GFLOP/token | Forward + backward |
| Achievable TFLOPS (MLX) | ~8-10 TFLOPS | ~55-65% MFU on M4 Max GPU |
| **Throughput** | **~2000-2500 tok/s** | Smaller model, compute-bound on GPU |
| **ms/step (seq=512, batch=4, accum=4)** | **~3300-4100 ms** | 8192 tokens/step |
| **Time for 1000 steps** | **~55-68 min** | Comparable to Orion paper (22 min at 110M) |
| **Time for 10K steps** | **~9-11 hours** | |

**600M model (pretraining from scratch, M4 Max):**

| Metric | Estimate | Reasoning |
|--------|----------|-----------|
| Forward FLOPS | ~7.2 GFLOP/token | 6 * 2 * 600M |
| Backward FLOPS | ~14.4 GFLOP/token | ~2x forward |
| Total FLOPS/token | ~21.6 GFLOP/token | Forward + backward |
| Achievable TFLOPS (MLX) | ~10-12 TFLOPS | Better MFU at larger size |
| **Throughput** | **~460-555 tok/s** | Memory-bandwidth starts to matter |
| **ms/step (seq=512, batch=1, accum=10)** | **~9200-11100 ms** | 5120 tokens/step |
| **Time for 1000 steps** | **~2.6-3.1 hours** | |
| **Time for 10K steps** | **~26-31 hours** | |
| **Time for 100K steps** | **~10-13 days** | Minimum for meaningful pretraining |

### Comparison: Rustane on ANE vs PMetal on MLX/GPU

For the 48.8M model (rustane's current gpt_karpathy config):

| | Rustane (current) | PMetal (estimated) |
|---|---|---|
| ms/step | ~1240 ms | ~2500-3000 ms (estimated) |
| Compute path | ANE forward + CPU backward | MLX GPU (autograd) |
| Optimization | Pre-allocated workspaces, async dispatch | MLX lazy eval, deferred |

Rustane's approach (custom ANE kernels + hand-written backward) should be **~2-3x faster** than PMetal for pretraining at the same model size, because:
1. Pre-allocated workspaces eliminate allocation overhead
2. ANE hardware is specialized for the 1x1 conv matmul pattern
3. Hand-written backward avoids autograd overhead
4. Direct Metal Adam avoids MLX synchronization

---

## What Would It Take to Pretrain with PMetal?

### Required Modifications

1. **Random weight initialization** — PMetal always loads from safetensors. You'd need to add a `ModelWeights::random()` path with proper init scales (Xavier, He, etc.).

2. **Raw text data pipeline** — PMetal's `DataLoader` expects chat/instruction format. Pretraining needs raw tokenized text with simple next-token prediction. You'd need to add a `PretrainingDataset` variant.

3. **Training loop changes** — The SFT training loop would mostly work, but:
   - Remove LoRA wrapping (train base model directly)
   - Add proper pretraining LR schedule (longer warmup, cosine decay over 100K+ steps)
   - Add validation BPB computation
   - Save full model checkpoints (not just LoRA adapters)

4. **Memory optimization** — Full pretraining at 600M needs gradient checkpointing to actually work. PMetal has the infrastructure but it's a placeholder.

5. **Performance gap** — Without MLX JIT compilation working properly (their known limitation), you're stuck at 75-80% of optimal MLX throughput. For pretraining where every ms/step matters over millions of steps, this is painful.

### Difficulty Assessment

| Task | Effort | Impact |
|------|--------|--------|
| Random init + save full checkpoints | Low (1-2 days) | Required |
| Raw text data pipeline | Medium (2-3 days) | Required |
| Remove LoRA, train all params | Low (1 day) | Required |
| Pretraining LR schedule + val_bpb | Low (1 day) | Required |
| Gradient checkpointing (real impl) | High (1-2 weeks) | Required for 600M |
| Fix MLX JIT for full throughput | Very High (upstream blocker) | Nice to have |

**Total estimate**: ~1-2 weeks to get basic pretraining working, 3-4 weeks for production quality.

---

## Recommendation

### For 110M pretraining:
- **PMetal**: Possible after ~1 week of modifications. ~55-68 min for 1K steps. Feasible but not optimized.
- **Rustane**: Already designed for this. Faster per-step, but currently only has 48.8M config. Scaling to 110M would require expanding the model config.
- **MLX Python (mlx-lm/autoresearch-mlx)**: Best option for quick iteration. Full JIT, mature pretraining loops, ~22 min for 1K steps per Orion paper.

### For 600M pretraining:
- **PMetal**: Would require significant work (gradient checkpointing, memory optimization). Estimated ~10-13 days for 100K steps on M4 Max.
- **Rustane**: Would need major expansion (more layers, GQA, larger configs, memory management). Currently ANE kernels are tuned for 48.8M.
- **MLX Python**: Most practical. mlx-lm has working gradient checkpointing, JIT compilation, and has been validated at this scale.

### Bottom line:
PMetal is excellent for what it's designed for (fine-tuning), but using it for pretraining from scratch is forcing a square peg into a round hole. The amount of modification needed is close to writing a pretraining loop from scratch — at which point you might as well use MLX Python directly or extend Rustane.

---

## Key PMetal Files for Reference

| File | Purpose |
|------|---------|
| `pmetal-trainer/src/training_loop.rs` | Main SFT training loop, throughput benchmarks |
| `pmetal-trainer/src/lora_trainer.rs` | LoRA-specific trainer (most used path) |
| `pmetal-trainer/src/adamw_groups.rs` | Multi-group optimizer with LoRA+ |
| `pmetal-trainer/src/mlx_metal_optimizer.rs` | Fused Metal optimizer kernel |
| `pmetal-metal/src/kernels/` | Custom Metal shaders (attention, cross-entropy, etc.) |
| `pmetal-models/src/architectures/qwen3.rs` | Qwen3-0.6B architecture (your 600M target) |
| `pmetal-core/src/config.rs` | TrainingConfig, LoraConfig structs |
| `docs/hardware-support.md` | Per-chip kernel tuning matrix |

---

*Analysis performed 2026-03-17. Based on PMetal v0.3.7 and Rustane master branch.*
