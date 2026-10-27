# Micro-Offload: CPU-GPU Optimizer State Offloading for Memory-Constrained Language Model Pre-Training

## Abstract
Training modern Transformer-based language models on resource-constrained hardware (such as single-GPU environments or restricted MIG instances) is fundamentally bottlenecked by device memory rather than raw compute. Specifically, Adam optimizer states (momentum and variance vectors) and FP32 master weights consume up to 12 bytes per parameter—occupying nearly 3x the memory footprint of the model parameters themselves in mixed-precision training. 

This project introduces **Micro-Offload**, a lightweight, modular systems framework implemented in PyTorch that decouples optimizer execution from the accelerator. By offloading optimizer memory states and weight updates to host CPU RAM, Micro-Offload dramatically expands the maximum trainable model size and batch capacity on a single 20GB MIG GPU without requiring full distributed infrastructure.

---

## 1. Introduction & Systems Problem
In standard mixed-precision training (FP16/BF16 forward/backward passes with FP32 master weights), the memory breakdown for an optimizer like AdamW per parameter consists of:
* Model Parameters (FP16/BF16): 2 bytes
* Gradients (FP16/BF16): 2 bytes
* Master Weights (FP32): 4 bytes
* Momentum Buffer ($m$, FP32): 4 bytes
* Variance Buffer ($v$, FP32): 4 bytes
* **Total Optimizer State Footprint:** 12 bytes per parameter.

For a 1-billion parameter model, optimizer states alone demand **12 GB of VRAM**, instantly triggering Out-Of-Memory (OOM) errors on a restricted 20GB MIG instance once activations and batch overhead are introduced. 

### Core Objectives
1. **Memory Efficiency:** Reclaim ~60% of GPU VRAM by migrating optimizer state tensors and updates entirely to host CPU memory.
2. **Throughput Optimization:** Mitigate PCIe communication bottlenecks using non-blocking CUDA streams and pinned host memory buffers.
3. **Perplexity Scalability:** Leverage the newly freed VRAM headroom to scale up model depth/width, achieving lower test perplexity within the course’s fixed compute constraints.

---

## 2. Related Work & Existing Solutions

Our design builds upon foundational heterogeneous memory management paradigms established in large-scale distributed training:

* **ZeRO-Offload (Rajbhandari et al., 2021 / Microsoft DeepSpeed):** Demonstrated that optimizer states and computations can be successfully offloaded to host CPU memory in data-parallel setups, enabling massive model scaling on limited hardware. However, DeepSpeed is engineered for multi-GPU clusters and introduces heavy configuration overhead.
* **PyTorch CPU-Offload & Torchtune Utilities:** Modern lightweight packages provide rudimentary offloading wrappers (`torchao`, `torchtune`), but lack fine-grained integration tailored for custom architectural pre-training loops.
* **Stall-Free Pipeline Research:** Highlights the performance cost of PCIe serialization where GPUs sit idle waiting for CPU weight updates, motivating our implementation of asynchronous overlapping.

---

## 3. Architecture & Proposed Strategy

### How Micro-Offload Works
```
  [ GPU: Forward & Backward Pass ] 
                 │
                 ▼  (Offload Gradients via Pinned Memory)
          [ Host CPU RAM ]
                 │
                 ▼  (Execute AdamW Step on CPU)
          [ Update Master Weights on CPU ]
                 │
                 ▼  (Copy FP16/BF16 Weights Back Asynchronously)
  [ GPU: Next Training Step Ready ]
```

1. **Pinned Host Memory (Page-Locked RAM):** We allocate CPU-side optimizer buffers using `torch.pin_memory()` to maximize PCIe bandwidth throughput during tensor migration.
2. **Custom `OffloadedAdamW` Optimizer Hook:** We override PyTorch's base optimizer step. Instead of updating parameters inside GPU VRAM, gradients are copied to the CPU, and the element-wise Adam update rules (computing first/second moments and weight decay) execute on the host CPU.
3. **Asynchronous CUDA Streams:** To hide transfer latencies, parameter synchronization back to the GPU is scheduled on non-blocking streams, overlapping communication with the subsequent iteration's data loading where possible.

---

## 4. Evaluation Plan & Key System Metrics

We will conduct rigorous comparative benchmarking against a standard PyTorch baseline (`torch.optim.AdamW` fully resident in GPU VRAM):

| Metric | Baseline (Standard GPU Adam) | Micro-Offload (Our System) | Target Improvement |
| :--- | :--- | :--- | :--- |
| **Max Trainable Model Size** | $\approx$ 350M parameters (OOM past this) | $\ge$ 1B parameters | **>2.8× scale increase** |
| **Peak GPU VRAM Usage** | ~19.2 GB (Near OOM limit) | ~8.4 GB | **~55% reduction** |
| **Throughput (Tokens/sec)** | Baseline speed ($1.0\times$) | Expected $0.75\times - 0.85\times$ | Bounded PCIe overhead |
| **Final Test Perplexity** | Constrained by smaller baseline model | Optimized via larger model capacity | Lower perplexity score |

### Ablation Studies
* **PCIe Bandwidth Impact:** Measure step-time latency with standard CPU RAM vs. pinned (page-locked) host memory.
* **Batch Size Scaling:** Determine maximum batch sizes attainable under varying sequence lengths before encountering memory saturation.

---

## 5. References
1. Rajbhandari, S., Ruwase, O., Rasley, J., Amenbakhsh, S., & He, Y. (2021). *ZeRO-Offload: Democratizing Billion-Scale Model Training*. USENIX ATC.
2. DeepSpeed Documentation. *ZeRO-Offload & ZeRO-3 Architecture*. Microsoft Research. Available online: https://www.deepspeed.ai/tutorials/zero-offload/
3. PyTorch Foundation. *ZenFlow: Stall-Free Offloading Engine for LLM Training*. PyTorch Engineering Blog, 2025.