# Chapter 17 — Efficient Inference & Compression

> **Prerequisites:** Ch. 11, 12, 14.
> **Why this matters:** a model is trained once and served billions of times. Inference is where almost all of the lifetime compute is spent, and where most ML engineering jobs actually live.

---

## 17.1 The two phases of inference

### The one-line idea

Generating text has two completely different computational profiles — a compute-bound phase that processes the prompt in parallel, and a memory-bound phase that emits one token at a time — and almost every optimization targets the second.

### The analogy

Cooking a meal versus serving it one spoonful at a time. Preparation (prefill) uses the whole kitchen at once and is limited by how fast you can chop. Serving (decode) is limited by how fast you can walk to the table and back — the work per trip is trivial, the *trip* is the cost.

### Arithmetic intensity

▸ $$\text{arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes moved}}$$

A GPU is compute-bound above its ridge point and memory-bound below it. For an H100 SXM at bf16 the dense ridge point is $\frac{495\ \text{TFLOP/s}}{3.35\ \text{TB/s}}\approx 148$ FLOP/byte (≈295 if you quote the with-sparsity peak — see Ch. 14 §14.7 on which to use).

| Phase | Work | Intensity | Bound by |
|---|---|---|---|
| **Prefill** | $T$ tokens through the whole model at once | high (matmuls) | **compute** |
| **Decode** | 1 token, but every weight must be read | $\approx 2B$ FLOP/byte | **memory bandwidth** |

▸ **At batch size 1, decode reads every parameter to produce one token.** A 70B bf16 model is 140 GB; at 3.35 TB/s that's a hard floor of 42 ms/token = **24 tokens/s, no matter how fast the GPU computes.** The arithmetic is trivial and the bandwidth is everything.

▸ **The direct consequence: batching is nearly free during decode.** Reading the weights once and applying them to 64 sequences costs almost the same wall-clock as applying them to 1. This single fact drives the entire design of modern serving systems.

---

## 17.2 Serving systems

**Static batching** wastes enormous capacity: the batch is held until the longest sequence finishes.

▸ **Continuous (in-flight) batching:** as soon as one sequence emits EOS, evict it and admit a waiting request. Typically **10–20× throughput improvement** over static batching. This is the single highest-leverage serving optimization.

▸ **PagedAttention (vLLM):** allocate the KV cache in fixed-size **blocks** with a page table, exactly like virtual memory. Solves the fragmentation problem — you no longer need to reserve `max_length` per sequence, so memory utilization goes from ~30% to >90%. Also makes **prefix sharing** trivial: a shared system prompt occupies one copy of the blocks across all requests, with copy-on-write for divergence.

**Chunked prefill:** split long prompts into chunks and interleave them with decode steps, so a long prefill doesn't stall everyone else's token stream. Balances TTFT (time-to-first-token) against ITL (inter-token latency).

**Disaggregated serving:** run prefill and decode on *separate* hardware pools, since one is compute-bound and the other memory-bound. Increasingly standard at scale.

**The metrics to know:** TTFT, ITL/TPOT, end-to-end latency, throughput (tokens/s per GPU), goodput (requests meeting an SLO).

---

## 17.3 Speculative decoding

### The one-line idea

Have a small fast model guess several tokens, then have the big model check them all in a single forward pass — which costs the same as generating one token, because decode is memory-bound.

### The analogy

An assistant drafts a paragraph; the expert reads it and accepts everything up to the first thing they'd have written differently. Reading is one pass regardless of length; writing is one word at a time. So you get several words for the price of reading once.

### The algorithm

1. Draft model $q$ generates $\gamma$ tokens autoregressively.
2. Target model $p$ scores all $\gamma+1$ positions **in one forward pass** (parallel, because the tokens are already known).
3. Accept token $i$ with probability $\min\left(1,\frac{p(x_i)}{q(x_i)}\right)$.
4. On the first rejection, resample from the residual distribution
▸ $$p'(x) = \frac{\max(0,\ p(x)-q(x))}{\sum_{x'}\max(0,\ p(x')-q(x'))}$$
5. Continue.

▸ **The guarantee:** the accepted sequence is distributed **exactly** as if sampled from $p$. This is a modified-rejection-sampling argument, and it means speculative decoding is *lossless* — it is a pure latency optimization with no quality trade-off. Being able to state that is the point of the technique.

### The speedup

With per-token acceptance rate $\alpha$, the expected number of tokens accepted per round is
▸ $$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

**Numbers:** $\alpha=0.8$, $\gamma=4$: $\frac{1-0.8^5}{0.2} = \frac{0.672}{0.2}=3.36$ tokens per target forward pass. Net speedup after draft cost is typically **2–3×**.

**Variants:** **Medusa** (extra prediction heads on the target model — no separate draft model); **EAGLE** (draft in feature space, higher acceptance); **Lookahead decoding** (Jacobi iteration, no draft model); **prompt lookup** (copy $n$-grams from the prompt — near-free and excellent for summarization and code editing).

---

## 17.4 Quantization

### The one-line idea

Store weights and activations in fewer bits. Since inference is memory-bandwidth-bound, halving the bits nearly halves the latency.

### Basic uniform quantization

▸ $$x_q = \mathrm{round}\left(\frac{x}{s}\right)+z,\qquad \hat x = s(x_q - z)$$

with scale $s = \frac{\max-\min}{2^b-1}$ (asymmetric) or $s=\frac{\max|x|}{2^{b-1}-1}$ (symmetric).

**Granularity** matters more than the formula: per-tensor (coarse, fast) → per-channel → per-group (e.g. 128 weights share a scale). **Group size 128 is the modern default** and recovers most of the accuracy.

### The activation-outlier problem

▸ Weights quantize easily. **Activations do not**, because transformers develop *systematic outlier channels* — a handful of dimensions with magnitudes 10–100× the rest, appearing consistently in the same channels across all tokens. A per-tensor scale set by those outliers crushes everything else to a few levels.

**The fixes:**
- **LLM.int8():** decompose — keep outlier channels in fp16, quantize the rest to int8. Exact where it matters.
- **SmoothQuant:** migrate the difficulty from activations to weights with a per-channel rescaling $X\mathrm{diag}(s)^{-1}\cdot\mathrm{diag}(s)W$, which leaves the product unchanged but makes both factors easier to quantize.
- **AWQ (Activation-aware Weight Quantization):** protect the ~1% of weight channels with the largest *activation* magnitudes by scaling them up before quantization. Fast, calibration-light, very widely used.
- **GPTQ:** quantize weights column by column, using the inverse Hessian ($H = 2XX^\top$, from a calibration set) to compensate remaining columns for the error already introduced. Solves a layerwise reconstruction problem $\min_{\widehat W}\|WX-\widehat WX\|_F^2$. Excellent at 4-bit and even 3-bit.

### QAT and the straight-through estimator

Quantization-aware training simulates quantization in the forward pass. But $\mathrm{round}(\cdot)$ has zero gradient almost everywhere.

▸ **Straight-through estimator (STE):** pretend the rounding is the identity in the backward pass.
$$\frac{\partial \hat x}{\partial x} := 1\ \ \text{(within the clipping range; 0 outside)}$$

Biased, and works extremely well. **The STE is a general tool** — it appears again in VQ-VAE (Ch. 19), binary networks, and any discrete bottleneck.

### What actually degrades

| Bits | Typical quality | Note |
|---|---|---|
| bf16 | baseline | |
| int8 (W8A8) | ~lossless | with SmoothQuant |
| **int4 weights (W4A16)** | **~1% perplexity** | GPTQ/AWQ, group 128 — the standard deployment point |
| int3 | noticeable | needs care |
| int2 / ternary | large loss | needs QAT or special architectures |

▸ **The scaling insight:** at a *fixed total bit budget*, a larger model quantized to 4 bits generally beats a smaller model at 16 bits. Quantization is not just compression; it is a better point on the quality-per-byte frontier.

**KV-cache quantization** is separately valuable — Ch. 12 §12.7 showed the cache often exceeds the weights.

---

## 17.5 Pruning

**Unstructured (magnitude) pruning:** zero the smallest-magnitude weights. Can reach 90%+ sparsity with retraining — but **gives no speedup on standard hardware**, because sparse matmuls with irregular patterns are slower than dense ones.

**Structured pruning:** remove whole heads, channels, or layers. Real speedup; larger quality cost.

**Semi-structured (2:4):** exactly 2 of every 4 consecutive weights are zero. ▸ Supported natively by NVIDIA sparse tensor cores for **~2× matmul throughput**. The practical sweet spot.

**SparseGPT / Wanda:** one-shot pruning with no retraining. Wanda's criterion is elegantly simple — prune by $|W_{ij}|\cdot\|X_j\|_2$, i.e. weight magnitude times the input activation norm. Reaches 50% sparsity on large LLMs with minimal loss and no gradient computation at all.

▸ **The empirical regularity:** larger models are more prunable. Redundancy grows with scale, which is consistent with the lottery-ticket and superposition pictures (Ch. 31, 32).

---

## 17.6 Knowledge distillation

### The one-line idea

Train a small model to match a large model's full output *distribution*, not just its argmax — because the relative probabilities of the wrong answers carry most of the information.

### The analogy

A student who is told "the answer is B" learns one bit. A student told "it's B, but C was tempting and D was absurd" learns the shape of the problem. Hinton called the latter **dark knowledge**.

### The loss

▸ $$\mathcal{L} = \alpha\,\tau^2\,\mathrm{KL}\!\left(p_T^{(\tau)}\,\|\,p_S^{(\tau)}\right) + (1-\alpha)\,\mathrm{CE}(y, p_S)$$

with $p^{(\tau)}=\mathrm{softmax}(z/\tau)$, $\tau\approx2$–$10$.

**Why the $\tau^2$?** The gradient of the soft-target term scales as $1/\tau^2$ (differentiate $\mathrm{softmax}(z/\tau)$ w.r.t. $z$ and note the extra $1/\tau$ in the chain, appearing twice through the KL). Multiplying by $\tau^2$ keeps the relative weight of the two terms constant as you vary $\tau$. **This is a favourite interview follow-up.**

**Why high $\tau$ helps:** at $\tau=1$ a confident teacher's distribution is nearly one-hot and carries no more information than the label. Raising $\tau$ exposes the ratios among the small probabilities.

### Variants

- **Feature/hint distillation:** match intermediate activations (FitNets), attention maps, or relations between examples.
- **Sequence-level KD:** for generative models, train the student on the teacher's *generated sequences* — usually more effective than token-level KL, because it fixes exposure bias too.
- **On-policy / GKD:** compute the KL on the *student's own* samples, which fixes the train/inference distribution mismatch.
- **Self-distillation:** teacher and student are the same size; still improves accuracy, which is evidence that the effect is regularization rather than compression.
- **Reverse KL** $\mathrm{KL}(p_S\|p_T)$ is mode-seeking (Ch. 1 §1.4.1) and often preferable for generation, where you want the student to be sharp rather than to smear over everything the teacher considers possible.

---

## 17.7 Parameter-efficient fine-tuning

### LoRA

### The one-line idea

Freeze the pretrained weights and learn a low-rank correction, because the *update* needed for a downstream task turns out to have far lower rank than the weight matrix itself.

▸ $$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$

**Initialization:** $A\sim\mathcal{N}(0,\sigma^2)$, $B=0$. So $\Delta W = 0$ at the start and training begins exactly at the pretrained model — the same zero-init-residual idea as Ch. 6, Ch. 8, and AdaLN-Zero.

**Parameter count:** $r(d+k)$ instead of $dk$. For $d=k=4096$, $r=8$: $65{,}536$ vs $16.8$M — a **256× reduction**.

▸ **Why it works:** the intrinsic dimension of a fine-tuning task is small; Aghajanyan et al. showed you can fine-tune to within 90% of full performance in a few hundred random dimensions. LoRA exploits this directly.

**Practical guidance:**
- Apply to **all** linear layers (Q, K, V, O, and the FFN), not just Q and V. This matters more than the rank.
- $r=8$–64 is typical; higher $r$ helps for tasks requiring new knowledge rather than new style.
- $\alpha/r$ is the effective scale; $\alpha=2r$ is a common default. Note that with this parameterization, the LR needed varies with $r$ — **rsLoRA** uses $\alpha/\sqrt r$ to fix this.
- **Zero inference cost:** merge $W_0 + \frac\alpha r BA$ into a single matrix after training.
- Many adapters can be hot-swapped on one base model — the basis of multi-tenant serving.

**QLoRA:** base model in **NF4** (a 4-bit format whose levels are the quantiles of a normal distribution — information-theoretically optimal for normally distributed weights), LoRA adapters in bf16, plus double quantization of the scales and paged optimizers. Enables fine-tuning a 65B model on a single 48 GB GPU.

**DoRA:** decompose $W$ into magnitude and direction, apply LoRA only to the direction. Closes much of the remaining gap to full fine-tuning.

### The other PEFT methods

| Method | Mechanism |
|---|---|
| Adapters (Houlsby) | small bottleneck MLPs inserted in each block; adds inference latency |
| Prefix / P-tuning | learn virtual key/value vectors prepended to every layer's attention |
| Prompt tuning | learn soft embeddings prepended to the input only; matches full FT only at very large scale |
| **BitFit** | train only the biases (~0.1% of parameters); surprisingly competitive on small tasks |
| **IA³** | learn per-channel rescaling vectors for K, V, and FFN activations |

▸ **When full fine-tuning still wins:** learning genuinely new knowledge or a new domain/language. PEFT excels at style, format, and task adaptation.

---

## 17.8 Model merging

Combine multiple fine-tuned models **without any retraining**.

**Task arithmetic.** Define a task vector $\tau_i = \theta_i^{\text{FT}} - \theta^{\text{pre}}$. Then
▸ $$\theta_{\text{merged}} = \theta^{\text{pre}} + \sum_i\lambda_i\tau_i$$

Remarkably, this works: adding task vectors composes capabilities, and **negating** one ($-\tau$) reliably *removes* a behaviour (used for detoxification and unlearning). That such simple vector arithmetic works in a 10-billion-dimensional non-convex space is a genuinely surprising empirical fact, and it's evidence for the linear-mode-connectivity picture in Ch. 31.

**TIES-Merging:** trim small-magnitude entries, elect a sign per parameter by summed magnitude, and average only the entries agreeing with the elected sign. Resolves the sign-conflict interference that plagues naive averaging.

**DARE:** randomly drop a large fraction of task-vector entries and rescale the rest by $1/(1-p)$ — most of a task vector is redundant.

**Model soups:** average many models fine-tuned with different hyperparameters from the same initialization. Reliably beats the best individual model with zero extra inference cost.

**SLERP:** spherical interpolation between two models, preserving norm.

▸ **The precondition for all of these:** the models must share an initialization and stay in the same loss basin. Merging models trained from different random seeds fails badly unless you first align their neurons by permutation (Ch. 31 §31.6).

---

## Check for Understanding

**Decode is memory-bandwidth-bound, which means every parameter is read to produce one token — so batching is almost free, quantization buys near-linear speedup, speculative decoding gets multiple tokens per weight-read at provably zero quality cost, and the KV cache rather than the weights is usually what limits how many users you can serve.**

---

**Next:** [Chapter 18 — Retrieval, RAG, Tools & Agents](18-retrieval-rag-agents.md)
