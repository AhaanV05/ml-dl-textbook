# Chapter 12 — Positional Information & Long Context

> **Prerequisites:** Ch. 11.

---

## 12.1 Why position needs encoding at all

### The one-line idea

Attention is a weighted average over a *set*, so without extra information a transformer literally cannot tell "dog bites man" from "man bites dog."

### The proof

Let $P$ be a permutation matrix. Then $\mathrm{Attention}(PX) = P\,\mathrm{Attention}(X)$ — self-attention is **permutation equivariant**. The FFN is applied position-wise, so it is too. Therefore the whole transformer is, and a token's output depends only on the *multiset* of tokens present.

▸ Position must be injected explicitly. There are two families: **absolute** (add position information to the token representation) and **relative** (make the attention score depend on $i-j$).

---

## 12.2 Absolute positional encodings

### Sinusoidal (Vaswani et al.)

▸ $$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Added to the token embedding.

**Why sinusoids?** Because a shift is a **linear** transformation of the encoding. For a fixed offset $k$, using the angle-addition identities:

$$\begin{pmatrix}\sin(\omega_i(pos+k))\\\cos(\omega_i(pos+k))\end{pmatrix} = \begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\-\sin\omega_ik&\cos\omega_ik\end{pmatrix}\begin{pmatrix}\sin(\omega_i\,pos)\\\cos(\omega_i\,pos)\end{pmatrix}$$

▸ $PE_{pos+k} = M_k\,PE_{pos}$ with $M_k$ **independent of $pos$** — a rotation by a fixed angle in each 2-D subspace. So relative position is linearly recoverable, and the encoding extrapolates to positions never seen.

**Wavelengths** range from $2\pi$ (fastest, $i=0$) to $10000\cdot2\pi$ (slowest). It is a **binary-like positional code in continuous form**: fast dimensions give fine local resolution, slow dimensions give coarse global position.

**Weakness:** it's *added* to the embedding, so position and content compete for the same $d$ dimensions, and the model must learn to disentangle them. Extrapolation works in theory but is poor in practice.

### Learned absolute

$PE\in\mathbb{R}^{T_{\max}\times d}$, trained. Used by BERT, GPT-2, ViT. Slightly better in-distribution.

▸ **Fatal flaw: hard length limit.** Position $T_{\max}+1$ has no embedding, and untrained positions are garbage. This is the direct cause of GPT-2's 1024-token wall.

---

## 12.3 Relative positional encodings

### Shaw et al. / T5 bias

Add a learned scalar to each attention score based on $i-j$:
$$e_{ij} = \frac{(x_iW_Q)(x_jW_K)^\top}{\sqrt{d_k}} + b_{i-j}$$

T5 **buckets** relative distances logarithmically (exact for small offsets, coarse for large) and shares $b$ across layers but not heads. Extrapolates gracefully; costs an extra $T\times T$ addition.

### ALiBi

▸ $$e_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} - m_h\,(i-j)$$

A **linear penalty** with a head-specific slope $m_h$, fixed as a geometric sequence: for $h$ heads, $m_h = 2^{-8h'/h}$ for $h'=1..h$.

**Intuition:** each head gets a different "attention span." Small slope = long-range head; large slope = local head. No learned parameters, and it extrapolates remarkably well (train at 1k, evaluate at 16k). Used by BLOOM, MPT.

**Limitation:** the monotone decay is a strong prior. It hurts on tasks requiring precise retrieval from far away.

---

## 12.4 RoPE — Rotary Position Embedding

**The dominant scheme in 2026.** Used by LLaMA, Mistral, Qwen, Gemma, DeepSeek, GPT-NeoX, PaLM.

### The one-line idea

Instead of adding position, **rotate** the query and key vectors by an angle proportional to their position — so their dot product automatically depends only on the *difference* in positions.

### The analogy

Two clock hands. If you rotate the hour hand of clock A by $m$ degrees and clock B by $n$ degrees, the angle *between* them depends only on $m-n$. Absolute rotations cancel in the comparison; relative rotation survives. RoPE does this in $d/2$ independent 2-D planes at once, each plane spinning at a different rate.

### The construction

Split the $d$-dimensional vector into $d/2$ consecutive pairs. For pair $i$ at position $m$, rotate by angle $m\theta_i$ where $\theta_i = 10000^{-2i/d}$:

▸ $$R_m^{(i)} = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}$$

Apply to queries and keys: $\tilde q_m = R_mq_m$, $\tilde k_n = R_nk_n$.

### The key property, derived

$$\tilde q_m^\top\tilde k_n = (R_mq)^\top(R_nk) = q^\top R_m^\top R_nk = q^\top R_{n-m}k$$

using $R_m^\top R_n = R_{n-m}$ (rotations form a group; $R_m^{-1}=R_m^\top=R_{-m}$).

▸ $$\boxed{\ \langle\mathrm{RoPE}(q,m),\ \mathrm{RoPE}(k,n)\rangle = g(q,k,\,n-m)\ }$$

**The attention score depends only on relative position, achieved through purely absolute operations.** That is the elegance: you rotate each vector once, independently, and relativity falls out of the inner product for free — no $T\times T$ bias matrix, no extra memory, fully compatible with the KV cache (cached keys are already rotated at their own absolute position and stay valid).

### Efficient implementation

Never build the rotation matrix. Using the "rotate-half" formulation:
```
q_rot = q * cos(mθ) + rotate_half(q) * sin(mθ)
rotate_half(q) = concat(-q[d/2:], q[:d/2])
```
Two elementwise multiplies and an add. $O(d)$, not $O(d^2)$.

### Properties

- **Long-range decay:** by a Cauchy–Schwarz-style bound, the expected score attenuates as positions separate — a soft locality prior, without ALiBi's hard monotone penalty.
- **Only applied to $Q$ and $K$**, never to $V$. Values carry content, not position.
- **The base $\theta$ (10000) is the key knob for context extension.**

### Context extension

The problem: a model trained at $T=4096$ has never seen rotation angles beyond $4096\theta_i$, and behaves badly past it.

| Method | Mechanism | Notes |
|---|---|---|
| **Position Interpolation (PI)** | scale positions: $m\to m\cdot\frac{T_{\text{train}}}{T_{\text{new}}}$ | all angles stay in-distribution, but *crowds* nearby positions and blurs local resolution. Needs a little fine-tuning. |
| **NTK-aware scaling** | increase the base: $\theta_{\text{base}}: 10000\to10000\cdot s^{d/(d-2)}$ | stretches low-frequency dims more, leaves high-frequency (local) dims alone. Often works with **no** fine-tuning. |
| **YaRN** | per-dimension: interpolate low-frequency dims, extrapolate high-frequency ones, plus a temperature correction on attention | current best; 10× extension with ~0.1% of original training tokens |
| **Train with a large base** | e.g. $\theta_{\text{base}}=500{,}000$ from scratch | LLaMA-3's approach; simplest if you control pretraining |

▸ **The unifying insight (the "NTK" argument):** RoPE dimensions form a spectrum of frequencies. High-frequency dimensions complete many full rotations within the training length, so they are well-trained and *can* extrapolate; low-frequency dimensions complete less than one rotation and have never seen large angles, so they *cannot*. The right fix therefore treats dimensions differently by frequency — which is exactly what YaRN does and what naive PI does not.

---

## 12.5 Efficient attention

### The cost

$O(T^2)$ time and, naively, $O(T^2)$ memory per head. At $T=100$k with $h=32$: the attention matrix alone is $32\times10^{10}$ entries = **640 GB in bf16 per layer**. Impossible without restructuring.

### FlashAttention — the one that actually mattered

▸ **The key realization: attention is memory-bound, not compute-bound.** GPU HBM bandwidth (~2–3 TB/s) is ~10× slower than SRAM. Standard attention writes the $T\times T$ matrix to HBM and reads it back — three round trips over an enormous array.

**The solution: tiling + online softmax + recomputation.** Never materialize the full matrix. Process $K,V$ in blocks, keeping a running output and running softmax statistics in SRAM.

**Online softmax** (the mathematical core). To compute a softmax-weighted sum in one pass, maintain running max $m$, running sum $\ell$, and running output $O$. On seeing a new block with max $m^{\text{new}}$:

▸ $$m' = \max(m, m^{\text{new}}),\qquad \ell' = e^{m-m'}\ell + e^{m^{\text{new}}-m'}\ell^{\text{new}},\qquad O' = \frac{e^{m-m'}\ell\,O + e^{m^{\text{new}}-m'}\ell^{\text{new}}O^{\text{new}}}{\ell'}$$

The rescaling factors correct the previously-accumulated result for the new global maximum. Numerically stable and exactly equal to standard softmax.

**Backward pass:** recompute the attention block from $Q,K,V$ instead of storing it — extra FLOPs, far fewer memory accesses, net large win.

▸ **Result:** memory drops from $O(T^2)$ to $O(T)$; speed improves 2–4×; the output is **numerically exact**, not an approximation. FlashAttention-2 improved work partitioning (~2× more); FlashAttention-3 exploits Hopper asynchrony and FP8.

**This is the single most important systems result in modern deep learning**, and it is a great example of the general lesson: *count memory movements, not FLOPs*.

### Approximate methods (largely superseded, still asked about)

| Method | Idea | Complexity |
|---|---|---|
| **Sparse Transformer** | strided + local patterns | $O(T\sqrt T)$ |
| **Longformer** | sliding window + global tokens | $O(Tw)$ |
| **BigBird** | window + global + random; proven Turing-complete | $O(T)$ |
| **Linformer** | project $K,V$ to length $k$ | $O(Tk)$ |
| **Performer** | random-feature kernel approximation of softmax | $O(T)$ |
| **Reformer** | LSH bucketing of similar queries | $O(T\log T)$ |

▸ **Why they mostly lost:** FlashAttention made exact attention fast enough that approximation's quality cost stopped being worth it. **Sliding-window attention survives** (Mistral, Gemma-2) because it composes: a window of $w$ over $L$ layers gives an effective receptive field of $L\cdot w$, and interleaving local with occasional global layers gets most of the benefit at a fraction of the cost.

### Linear attention

Replace $\mathrm{softmax}(QK^\top)V$ with $\phi(Q)(\phi(K)^\top V)$ for a feature map $\phi$. By associativity, compute $\phi(K)^\top V$ first ($d\times d$), giving **$O(Td^2)$ instead of $O(T^2d)$.**

▸ In the causal case this is exactly a **linear RNN** with state $S_t = S_{t-1} + \phi(k_t)v_t^\top$, so inference is $O(1)$ memory per token — no KV cache at all. The cost is quality: linear attention cannot do sharp, selective retrieval, because a fixed $d\times d$ state must summarize everything.

---

## 12.6 State-space models and the recurrence revival

### The idea

A continuous linear system $h'(t)=Ah(t)+Bx(t)$, $y=Ch(t)$, discretized to $h_t = \bar Ah_{t-1}+\bar Bx_t$, $y_t = Ch_t$.

**The two-mode trick:** because the recurrence is linear,
- **Training:** unroll into a convolution $y = \bar K * x$ with kernel $\bar K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B,\dots)$, computable in $O(T\log T)$ by FFT — fully parallel.
- **Inference:** run the recurrence, $O(1)$ state per token.

▸ **This solves the exact problem that killed RNNs in Ch. 9** (no training parallelism) while keeping their exact advantage (constant-size state).

**S4** makes long-range memory work by initializing $A$ with **HiPPO** structure — a matrix derived from optimal online polynomial approximation of the input history, so the state provably compresses the past in a principled basis.

**Mamba (S6)** makes $B$, $C$, and the discretization step $\Delta$ **input-dependent** — "selective" — so the model can choose to remember or forget based on content. This breaks linear-time-invariance and rules out the FFT trick, so Mamba uses a **hardware-aware parallel associative scan** instead, with the state kept in SRAM.

### The honest comparison

| | Transformer | Mamba/SSM |
|---|---|---|
| Training | $O(T^2)$, perfectly parallel | $O(T)$, parallel via scan |
| Inference memory | $O(T)$ KV cache | $O(1)$ state |
| Selective recall from long context | **excellent** | weaker |
| In-context learning / induction | strong | weaker in pure form |

▸ **The empirical resolution is hybrids.** Jamba, Zamba, Samba and similar interleave a few full-attention layers among many Mamba layers, getting near-transformer recall at a fraction of the KV cache. This is the current practical answer, and a fair thing to say when asked "will transformers be replaced?": *not replaced, hybridized.*

---

## 12.7 KV-cache reduction

At inference, past keys and values are cached to avoid recomputation. Size:

▸ $$\text{KV cache bytes} = 2\times L\times T\times h_{kv}\times d_{\text{head}}\times \text{bytes/elem}$$

**Numbers.** A 70B-class model *with full MHA* — $L=80$, $h_{kv}=64$, $d_{\text{head}}=128$, $T=4096$, bf16:
$$2\times80\times4096\times64\times128\times2 = 10.7\ \text{GB per sequence.}$$
For a batch of 32, that is 343 GB — far more than the 140 GB of weights. **The KV cache, not the model, is usually the binding memory constraint in serving.**

▸ *This is exactly why no shipped 70B model uses MHA.* LLaMA-2-70B has the geometry above but uses **GQA with $h_{kv}=8$**, cutting the cache 8× to 1.34 GB per sequence. The MHA figure is what GQA was invented to avoid — keep both numbers in mind, since the contrast is the argument.

### The fixes

**MQA (Multi-Query Attention):** all query heads share **one** K/V head. $h_{kv}=1$ ⇒ 64× reduction. Noticeable quality loss.

**GQA (Grouped-Query Attention):** $g$ K/V heads shared among $h$ query heads. $h_{kv}=8$ with $h=64$ ⇒ 8× reduction at near-MHA quality. ▸ **The current default** (LLaMA-2/3 70B, Mistral, most production models). It is a clean interpolation: $g=h$ is MHA, $g=1$ is MQA.

**MLA (Multi-head Latent Attention, DeepSeek):** compress K and V jointly into a low-rank latent $c_t = W_{DKV}x_t$ and cache only $c_t$, reconstructing K/V on the fly. ~90%+ cache reduction with quality *better* than GQA at matched cache size.

**Quantized KV cache:** store in int8 or int4. 2–4× reduction, small quality cost, composes with the above.

**Eviction/streaming:** StreamingLLM keeps the first few "sink" tokens plus a recent window — enables unbounded streaming, at the cost of genuinely forgetting the middle. H2O keeps "heavy hitter" tokens by accumulated attention mass.

---

## 12.8 Does long context work?

**"Needle in a haystack"** tests retrieval of a planted fact. Most modern models pass it — but it is a weak test.

▸ **"Lost in the middle"** (Liu et al.): accuracy is high for information at the start and end of the context and **sags in the middle**, tracing a U-shape. Attributable partly to positional-encoding decay and partly to training-data position statistics.

Harder benchmarks (RULER, multi-hop over long context, aggregation queries) show that **effective context is typically much shorter than advertised context** — a model claiming 1M tokens may degrade materially past 100k on tasks requiring integration rather than lookup.

**Practical rule:** long context and retrieval are complements, not substitutes. Retrieval reduces the haystack; long context handles what's left. (Ch. 18.)

---

## Check for Understanding

**Attention is permutation-equivariant so position must be injected, and the winning scheme (RoPE) rotates queries and keys by angles proportional to their absolute positions so that their dot product depends only on the difference — while the practical limits on context come not from the $O(T^2)$ FLOPs, which FlashAttention made cheap by never materializing the attention matrix, but from the KV cache, which grows linearly with length and is what GQA, MLA, and state-space models exist to shrink.**

---

**Next:** [Chapter 13 — GPT: Autoregressive Language Modelling](13-gpt-autoregressive-language-models.md)
