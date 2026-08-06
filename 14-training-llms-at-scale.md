# Chapter 14 — Training LLMs at Scale

> **Prerequisites:** Ch. 11, 13.
> **Scope:** everything between "I have an architecture" and "I have a trained model." This is where most real-world ML engineering time actually goes.

---

## 14.1 Data

### The one-line idea

At scale, data quality and composition determine model quality more than architecture does — and almost all of the work is deduplication and filtering, not collection.

### The pipeline

**1. Sourcing.** Common Crawl (raw web), refined derivatives (C4, RefinedWeb, FineWeb, DCLM), code (GitHub, The Stack), books, papers (arXiv, PubMed), Wikipedia, Q&A forums, and increasingly **synthetic data**.

**2. Language identification and extraction.** HTML → text (trafilatura, resiliparse). Boilerplate removal matters enormously.

**3. Quality filtering.**
- *Heuristic*: mean word length, symbol-to-word ratio, fraction of lines ending in punctuation, stopword presence, repeated-line fraction. (Gopher's rules are the canonical set.)
- *Model-based*: a classifier trained to distinguish a high-quality reference corpus from raw web, applied as a filter or a sampling weight. **This is now the dominant approach** (DCLM, FineWeb-Edu) and is worth more than any heuristic set.
- *Perplexity filtering* against a reference model — but beware: it selects for text resembling the reference model's training data, which narrows diversity.

**4. Deduplication.** The highest-value single step.

▸ **Exact duplicates** are found by hashing. **Near-duplicates** require MinHash + LSH:
- Represent a document by its set of $n$-grams (typically $n=5$ words).
- **Jaccard similarity**: $J(A,B)=\frac{|A\cap B|}{|A\cup B|}$.
- **MinHash:** for a random permutation $\pi$ of the universe, $\Pr[\min\pi(A)=\min\pi(B)] = J(A,B)$ **exactly**. Use $k$ hash functions to get a $k$-dim signature; the fraction of matching entries estimates $J$ with standard error $\approx\sqrt{J(1-J)/k}$.
- **LSH banding:** split the $k$ signature entries into $b$ bands of $r$ rows ($k=br$). Two documents become candidates if any band matches exactly. Probability of becoming a candidate:
▸ $$P(\text{candidate}) = 1-(1-J^r)^b$$
This is an S-curve with threshold near $J^*\approx(1/b)^{1/r}$. With $k=128$, $b=16$, $r=8$: $J^*=16^{-1/8}\approx0.71$ — tune $b,r$ to place the cutoff where you want it.

**Why dedup matters so much:** duplicated text is memorized rather than generalized; it inflates evaluation via contamination; and Lee et al. showed dedup lets models reach the same loss with substantially less compute while emitting ~10× less verbatim training data.

**5. Decontamination.** Remove documents overlapping evaluation sets ($n$-gram matching, $n=13$ is a common choice).

**6. PII removal and safety filtering.**

**7. Mixture weights.** Domain proportions are a first-class hyperparameter. Methods: manual (LLaMA's mix), **DoReMi** (train a small proxy model with group-DRO to *learn* the weights), or online adjustment. Multi-epoching: up to ~4 epochs on high-quality data is roughly as good as fresh data; beyond that returns collapse (Muennighoff et al.).

**8. Curriculum.** Increasingly standard: general web early, then upweight code/math/high-quality/long-context data in a final "mid-training" or annealing phase. The WSD schedule (Ch. 5) is designed around this — the sharp decay phase is where the highest-quality data goes.

---

## 14.2 Numerical precision

### The formats

| Format | Bits (S/E/M) | Max | Min normal | Relative precision |
|---|---|---|---|---|
| fp32 | 1/8/23 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $\sim10^{-7}$ |
| **fp16** | 1/5/10 | $65{,}504$ | $6.1\times10^{-5}$ | $\sim10^{-3}$ |
| **bf16** | 1/8/7 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $\sim10^{-2}$ |
| fp8 (E4M3) | 1/4/3 | 448 | | $\sim10^{-1}$ |
| fp8 (E5M2) | 1/5/2 | 57344 | | |

▸ **bf16 has fp32's exponent range with fewer mantissa bits.** That trade is exactly right for deep learning: gradients span many orders of magnitude (range matters) but need few significant digits (precision doesn't). **bf16 needs no loss scaling; fp16 does.**

### Loss scaling (fp16 only)

fp16's minimum normal is $6\times10^{-5}$; gradients routinely fall below it and flush to zero. Fix:
1. Multiply the loss by $S$ (e.g. $2^{16}$) before backward.
2. Gradients are scaled by $S$, landing in representable range.
3. Unscale by $1/S$ before the optimizer step.
4. **Dynamic scaling:** if any gradient is inf/NaN, skip the step and halve $S$; after $N$ successful steps, double $S$.

### Mixed precision, properly

▸ Keep **fp32 master weights**. Compute forward/backward in bf16. Update the fp32 master copy, then cast down.

**Why:** the update $\eta\cdot\hat m/\sqrt{\hat v}$ can be $10^{-7}$ relative to a weight of $10^{-2}$ — a ratio of $10^{-5}$, well below bf16's $10^{-2}$ resolution. **Adding it to a bf16 weight is a no-op; the update is silently discarded.** This is a real and subtle failure mode, and "why do we keep fp32 master weights" is a good interview question.

**Keep in fp32 regardless:** loss computation and softmax, normalization statistics, and optimizer moments (or use stochastic rounding for bf16 moments).

**FP8** (Hopper/Blackwell): per-tensor or per-block scaling factors, E4M3 for forward, E5M2 for gradients. ~2× throughput over bf16; requires careful scaling management.

---

## 14.3 The memory budget

For $N$ parameters with AdamW in mixed precision:

| Item | Bytes |
|---|---|
| bf16 weights | $2N$ |
| bf16 gradients | $2N$ |
| fp32 master weights | $4N$ |
| fp32 Adam $m$ | $4N$ |
| fp32 Adam $v$ | $4N$ |
| **Total (states)** | $\mathbf{16N}$ |

▸ **A 7B model needs 112 GB of optimizer/weight state alone** — more than a single 80 GB GPU, before a single activation. This is why distributed training is not optional above ~1.5B parameters.

**Activation memory** per layer per token $\approx \mathcal{O}(sd)$ with $s\approx 10$–20 depending on what is stored; times $B\times T\times L$.

### Gradient checkpointing (activation recomputation)

Store only a subset of activations; recompute the rest during backward.

▸ Checkpointing every $\sqrt L$ layers gives memory $O(\sqrt L)$ instead of $O(L)$, at the cost of one extra forward pass ($\approx+33\%$ compute, since forward:backward is 1:2).

**Selective checkpointing** is better: recompute only the cheap-but-memory-hungry ops (norms, activations, dropout) and keep the expensive matmul outputs. Gets most of the memory saving for ~5% compute.

---

## 14.4 Parallelism

### Data parallelism (DP)

Replicate the model, split the batch, **all-reduce** the gradients.
- Communication: $2N$ bytes per step (ring all-reduce is bandwidth-optimal).
- Scales until the batch per device is too small, or the global batch exceeds the critical batch size (Ch. 15).

### ZeRO / FSDP — shard the states

Standard DP replicates all $16N$ bytes on every device. ZeRO shards them:

| Stage | Shards | Memory per device |
|---|---|---|
| ZeRO-1 | optimizer states | $4N + \frac{12N}{P}$ |
| ZeRO-2 | + gradients | $2N+\frac{14N}{P}$ |
| **ZeRO-3 / FSDP** | + parameters | $\frac{16N}{P}$ |

▸ ZeRO-3 gives **linear memory scaling in the number of devices** at the cost of an all-gather of each layer's parameters just before it is used (and again in backward). With overlapping of communication and compute, the throughput cost is typically 10–20%.

### Tensor parallelism (TP)

Split individual matrices across devices *within* a layer.
- **Column-parallel** then **row-parallel** for the FFN: $Y = \mathrm{GeLU}(XA)$, split $A$ by columns; $Z=YB$, split $B$ by rows. One all-reduce per block instead of two.
- Attention: split by heads — each device owns a subset of heads.

▸ **Requires very high bandwidth (NVLink).** Keep TP **within a node** (typically TP ≤ 8); across nodes it is disastrous.

### Pipeline parallelism (PP)

Split layers across devices; micro-batch to keep everyone busy.

▸ **Bubble fraction** $= \frac{P-1}{m + P - 1}$ for $P$ stages and $m$ micro-batches. With $P=8$, $m=8$: 47% idle. With $m=64$: 10%. **Use many micro-batches.** Interleaved (virtual-stage) schedules and zero-bubble variants reduce it further.

### Sequence / context parallelism

Split the sequence dimension. **Ring Attention** passes K/V blocks around a ring so each device attends over the full sequence while holding only a slice — this is what makes million-token training feasible.

### Expert parallelism (MoE)
Distribute experts across devices; routing becomes an all-to-all.

### 3D/4D parallelism — the composition

▸ Typical frontier configuration: **TP within a node (8), PP across a few nodes, DP/FSDP across everything else, plus context parallelism for long sequences.** Order the dimensions so that the highest-communication one (TP) uses the fastest links.

### Measuring efficiency

▸ $$\mathrm{MFU} = \frac{6ND / t}{\text{peak FLOP/s}\times\text{devices}}$$

Model FLOPs Utilization. **40–55% is good** for large transformers; below 30% means something is wrong (usually communication, small batch, or bad kernel choice). Report MFU, not "GPU utilization," which only measures whether kernels are running, not whether they're doing useful work.

---

## 14.5 Batch size

**Critical batch size** (McCandlish et al.): the gradient-noise scale
▸ $$B_{\text{crit}} \approx \frac{\mathrm{tr}(\Sigma)}{\|\nabla\mathcal{L}\|^2}$$

Below $B_{\text{crit}}$, doubling the batch roughly halves the steps needed (near-perfect scaling). Above it, you buy almost nothing.

▸ $B_{\text{crit}}$ **grows during training** — as the loss falls, $\|\nabla\mathcal{L}\|$ shrinks faster than the noise, so the ratio grows. Hence **batch-size ramping**: start small (sample-efficient), grow large (compute-efficient). Frontier runs commonly ramp from ~1M to ~60M tokens per batch.

Pair with the LR rules from Ch. 4 §4.6: linear scaling for SGD, square-root scaling for Adam is the safer default.

---

## 14.6 Instabilities and how to fix them

Large runs fail in characteristic ways. Knowing the taxonomy is genuinely valuable.

**Loss spikes.** Sudden jumps, sometimes recovering, sometimes not.
- *Causes:* attention-logit growth; a bad data batch; fp16 overflow; too-high LR interacting with a sharpness increase.
- *Fixes:* **skip the batch** and continue (standard practice — PaLM restarted from a checkpoint ~100 steps back and skipped ~250 batches); lower LR; gradient clipping; the structural fixes below.

▸ **Attention-logit explosion.** $q^\top k$ grows without bound, softmax saturates, gradients vanish, then a large step destabilizes everything. **Fix: QK-normalization** — RMSNorm on $Q$ and $K$ before the dot product. This has become standard and is one of the highest-value stability interventions known.

▸ **Output-logit divergence.** The final logits grow, driving overconfidence and fp16 overflow. **Fix: z-loss**, an auxiliary penalty
$$\mathcal{L}_z = \alpha\left(\log\sum_j e^{z_j}\right)^2,\qquad \alpha\approx10^{-4}$$
which pulls the log-partition function toward 0 and keeps logits bounded without constraining their differences. Used by PaLM and many since.

**Divergence after an LR warmup ends** — usually the peak LR is too high for the current sharpness (Ch. 5 §5.5).

**Silent degradation:**
- Dead/underflowing gradients in fp16.
- A tokenizer/data bug producing repeated content.
- Expert collapse in MoE (check routing entropy).
- Weight-norm blowup from missing weight decay.

**The monitoring set worth having by default:** loss, grad norm (global and per-layer), LR, weight norm per layer, update-to-weight ratio, attention-logit max, router entropy (MoE), throughput and MFU, and per-domain validation loss.

---

## 14.7 A realistic budget

Training a 7B model on 2T tokens:

$$C = 6ND = 6\times7\times10^9\times2\times10^{12} = 8.4\times10^{22}\ \text{FLOPs}$$

On 512 H100 SXM at **dense** bf16 peak ($\approx4.95\times10^{14}$ FLOP/s — note vendor sheets quote $\approx9.9\times10^{14}$, but that figure assumes 2:4 structured sparsity and must not be used here), at 40% MFU:
$$\text{effective} = 512\times4.95\times10^{14}\times0.40 = 1.01\times10^{17}\ \text{FLOP/s}$$
$$t = \frac{8.4\times10^{22}}{1.01\times10^{17}} = 8.3\times10^{5}\ \text{s} \approx \mathbf{9.6\ days}$$

▸ **Always state which peak you used.** Dense-vs-sparse peak is a factor of 2, and quoting the sparsity number against a dense workload silently halves your reported MFU.

▸ This calculation — $C=6ND$, divide by (devices × peak × MFU) — is the single most useful back-of-envelope in the field, and it comes up constantly in interviews. Practice it until it's automatic.

---

## Check for Understanding

**Training at scale is a memory and communication problem, not a mathematics problem: the $16N$ bytes of optimizer state force sharding, the $O(T^2)$ attention forces FlashAttention and context parallelism, bf16 works only because gradients need range rather than precision, and the quality of the result is determined mostly by how aggressively you deduplicated and filtered the data.**

---

**Next:** [Chapter 15 — Scaling Laws & Emergence](15-scaling-laws-and-emergence.md)
