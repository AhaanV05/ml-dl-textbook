# Chapter 11 — Attention & the Transformer

> **Prerequisites:** Ch. 6, Ch. 7 (LayerNorm), Ch. 9 (§9.6), Ch. 10.
> **This is the most important chapter in the book.** Nearly every model deployed in 2026 is a transformer.

---

## 11.1 Attention from first principles

### The one-line idea

Every position emits a question, every position advertises what it has, and each position builds its next representation as a weighted average of what everyone has — weighted by how well the questions match the advertisements.

### The analogy

A conference poster session. You (a **query**) walk the hall with a specific question. Every poster has a title (**key**) and content (**value**). You glance at all the titles, decide how relevant each is, and then absorb content from each poster in proportion to that relevance — mostly from one or two, a little from several others. Then you update your understanding and walk the hall again with a *new* question. That second pass is the second layer.

### Building it up

Suppose we want position $i$'s new representation to be a mixture of all positions' information:

$$y_i = \sum_j \alpha_{ij}v_j,\qquad \sum_j\alpha_{ij}=1,\ \alpha_{ij}\ge0$$

We need $\alpha_{ij}$ to depend on how relevant $j$ is *to $i$*. Use a dot product as the relevance score and softmax to normalize. Give each position three different learned projections of its embedding so that "what I'm looking for" (query), "what I contain" (key), and "what I pass on" (value) can be different things:

$$Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V$$

▸ $$\boxed{\ \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\ }$$

Shapes: $X\in\mathbb{R}^{T\times d}$, $W_Q,W_K\in\mathbb{R}^{d\times d_k}$, $W_V\in\mathbb{R}^{d\times d_v}$, scores $\in\mathbb{R}^{T\times T}$, output $\in\mathbb{R}^{T\times d_v}$.

### Why three projections and not one?

If $Q=K$, the attention matrix is symmetric and every token attends maximally to itself (its own dot product is the largest). Separating $Q$ and $K$ allows **asymmetric** relations: "adjective looks for the noun it modifies" is not the same relation as "noun looks for its adjective."

If $V=K$, the thing used for *matching* is forced to equal the thing that gets *transmitted*. Separating them means a token can be found by one property and contribute a different one. (In circuit terms, Ch. 32: the QK circuit decides *where* to read, the OV circuit decides *what* to write.)

---

## 11.2 The $\sqrt{d_k}$ scaling — derive it, it's asked constantly

Assume $q,k\in\mathbb{R}^{d_k}$ have i.i.d. entries with mean 0 and variance 1. Then

$$\mathbb{E}[q^\top k] = \sum_{i=1}^{d_k}\mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$
$$\mathrm{Var}(q^\top k) = \sum_{i=1}^{d_k}\mathrm{Var}(q_ik_i) = \sum_{i=1}^{d_k}\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$$

▸ So the raw scores have standard deviation $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ restores unit variance.

### Why unit variance matters — the gradient argument

Recall (Ch. 1 §1.3.4) that for $p=\mathrm{softmax}(z)$,
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j)$$

If one score dominates, $p\to$ one-hot, and every entry of that Jacobian $\to0$. **The softmax saturates and its gradient vanishes.**

**Numbers.** $d_k=64$, unscaled scores have SD 8. The gap between the largest and second-largest of $T$ such scores is several units, so $e^{\Delta}$ with $\Delta\approx10$ gives a softmax that is $>0.9999$ on one entry. Gradient $\approx p(1-p)\approx10^{-4}$. With scaling, SD is 1, gaps are $O(1)$, and the softmax stays in its responsive range.

▸ **General principle worth extracting:** any time you feed a sum of $n$ terms into a saturating nonlinearity, check whether you need to divide by $\sqrt n$. This is the same variance-propagation argument as He initialization (Ch. 6 §6.4).

**Related failure at scale — logit explosion.** During training, $\|q\|$ and $\|k\|$ can grow, re-saturating attention even with the $\sqrt{d_k}$ factor. Fix: **QK-normalization** (apply LayerNorm/RMSNorm to $Q$ and $K$ before the dot product), now standard in large models (Ch. 14 §14.6).

---

## 11.3 Multi-head attention

### The one-line idea

Run $h$ independent attention operations in parallel on lower-dimensional projections, so different heads can attend to different kinds of relationships simultaneously.

### The analogy

Reading a contract with a team: one person tracks defined terms, one tracks dates, one tracks dollar amounts, one tracks obligations. Each reads the whole document with a different question in mind. Then you combine their notes. A single reader with one question would miss most of it.

### The math

▸ $$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O,\qquad \mathrm{head}_i = \mathrm{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

with $d_k = d_v = d_{\text{model}}/h$, so the total parameter count and FLOP count match single-head attention at full width.

**Standard config:** $d_{\text{model}}=512$, $h=8$, $d_k=64$. Or $d=4096$, $h=32$, $d_k=128$.

### Why a single wide head is worse

A single softmax produces **one** distribution per query. It must commit to one weighting. With $h$ heads you get $h$ distributions, so a token can simultaneously copy from its syntactic head, its coreferent antecedent, and the previous token.

There is also a rank argument: $\mathrm{softmax}(QK^\top/\sqrt{d_k})$ is a $T\times T$ matrix built from a rank-$d_k$ score matrix. Multiple heads give a *sum* of $h$ such low-rank-driven operations, which is strictly more expressive than one at the same total width.

**Diminishing returns:** Michel et al. (2019) showed most heads can be pruned at inference with little loss — often 1 head per layer suffices for many layers. Heads are redundant, but the redundancy appears to help *optimization*.

### What heads actually learn (empirically, in trained models)

- **Positional heads:** attend to $i-1$ or $i+1$.
- **Syntactic heads:** verb→subject, noun→determiner (measurably aligned with dependency parses).
- **Coreference heads:** pronoun→antecedent.
- **Duplicate-token / previous-token heads:** the components of induction circuits (Ch. 32).
- **"Attention sink" behaviour:** many heads dump most of their mass on the first token (or a BOS token) when they have nothing to do. This is a *no-op mechanism* — softmax must sum to 1, so a head with nothing to retrieve parks its mass somewhere harmless. Knowing this explains a real deployment issue: **evicting the first tokens from a KV cache destroys the model** (hence StreamingLLM keeps them).

---

## 11.4 Masking

### Causal (decoder) masking

▸ $$\mathrm{scores}_{ij} \leftarrow \begin{cases}\mathrm{scores}_{ij} & j\le i\\ -\infty & j>i\end{cases}$$

$-\infty$ (in practice $-10^{9}$ or the dtype minimum) makes $e^{\text{score}}=0$ after softmax. This enforces autoregression: position $i$ cannot see the future.

▸ **The crucial consequence:** with causal masking you can compute the loss at *every* position in one forward pass, so a length-$T$ sequence yields $T$ training signals. This is why GPT-style pretraining is so sample-efficient per FLOP, and why it beat BERT-style masked LM (which only supervises 15% of positions) for generative scaling.

### Padding masking

Mask out padded positions so they contribute nothing. Must be applied in attention *and* excluded from the loss and from any mean-pooling.

### Prefix / block-causal masking

Bidirectional within a prompt, causal within the completion. Used by prefix-LMs (UL2, some multimodal models) and for packed-sequence training (block-diagonal masks so packed documents can't attend across boundaries).

---

## 11.5 The transformer block

### Pre-LN (the modern standard)

```
x = x + Attn(LN(x))          # communication: tokens exchange information
x = x + FFN(LN(x))           # computation: each token thinks independently
```

▸ **This two-phase decomposition is the cleanest mental model of a transformer.** Attention moves information *between* positions; the FFN transforms information *within* a position. Nothing else happens.

### The feed-forward network

▸ $$\mathrm{FFN}(x) = W_2\,\phi(W_1x + b_1) + b_2,\qquad W_1\in\mathbb{R}^{d_{\text{ff}}\times d},\ d_{\text{ff}}=4d$$

**Why $4\times$?** Empirical, and remarkably stable across four generations of models. Narrower underperforms; wider gives little.

▸ **The FFN holds about $\frac{2}{3}$ of all non-embedding parameters** ($8d^2$ of $12d^2$ per layer — see §11.7). Any claim that "transformers are attention" is wrong on a parameter-count basis. Attention is the *routing*; the FFN is the *storage*.

**The key–value memory interpretation** (Geva et al., 2021): write $W_1$'s rows as keys $k_i$ and $W_2$'s columns as values $v_i$. Then
$$\mathrm{FFN}(x) = \sum_{i=1}^{d_{\text{ff}}}\phi(k_i^\top x)\,v_i$$
Each hidden unit is a pattern detector that, when it fires, **adds a fixed vector to the residual stream**. Empirically these correspond to interpretable patterns (a specific $n$-gram, a topic, a syntactic construction), and the values shift the output distribution toward related tokens. **The FFN is an associative memory with $d_{\text{ff}}$ slots per layer.** This is the current best account of where factual knowledge lives in an LLM, and it is what model-editing methods (ROME, MEMIT) exploit.

**Modern variant — SwiGLU** (Ch. 6 §6.5): $W_3(\mathrm{SiLU}(W_1x)\odot W_2x)$ with $d_{\text{ff}}=\frac83 d$ to hold parameters constant. ~1% perplexity gain.

### The residual stream

▸ Because every sub-layer *adds* to $x$, the residual stream is a **shared communication bus of dimension $d$** that all $2L$ sub-layers read from and write to. Consequences (developed fully in Ch. 32):

- Layers communicate by writing into *subspaces* of the stream; a layer can be read many layers later.
- The stream's dimension $d$ is a hard **bandwidth limit** — with far more features than $d$, the model must store them in superposition.
- The norm of the residual stream grows roughly monotonically with depth, since each layer adds.
- **Logit lens:** applying the final unembedding to an intermediate residual stream gives a readable (if noisy) distribution, showing the prediction refine layer by layer.

### Post-LN vs pre-LN
Covered in Ch. 7 §7.3. Summary: post-LN needs warmup and is fragile beyond ~12 layers without care; pre-LN is the default; DeepNorm/sandwich-LN recover post-LN's slight quality edge at depth.

---

## 11.6 Encoder, decoder, and encoder–decoder

**Encoder block:** bidirectional self-attention + FFN. Used for understanding (BERT, ViT, retrievers).

**Decoder block:** causal self-attention + FFN. Used for generation (GPT, LLaMA). Decoder-only is the dominant paradigm.

**Encoder–decoder:** the decoder additionally has a **cross-attention** sub-layer where $Q$ comes from the decoder and $K,V$ come from the encoder output:
```
x = x + CausalSelfAttn(LN(x))
x = x + CrossAttn(LN(x), enc_out, enc_out)
x = x + FFN(LN(x))
```
Used by T5, BART, Whisper, and most translation systems.

▸ **Why did decoder-only win for LLMs?** (A standard interview question.)
1. **Training efficiency:** every token is a prediction target.
2. **Simplicity:** one stack, one objective, trivially scalable.
3. **In-context learning** emerges naturally from a single sequence containing both instruction and content.
4. Encoder–decoder splits parameters into two pools; at fixed budget, one pool that does both is empirically better for open-ended generation.

Encoder–decoder remains better when there is a fixed, repeatedly-attended source (translation, speech, some multimodal settings), because the source is encoded once bidirectionally.

---

## 11.7 The complete accounting

### Parameters per layer (excluding biases and norms)

| Component | Parameters |
|---|---|
| $W_Q,W_K,W_V,W_O$ | $4d^2$ |
| FFN ($W_1,W_2$, $d_{\text{ff}}=4d$) | $8d^2$ |
| **Total per layer** | $\mathbf{12d^2}$ |

Plus $|V|d$ for embeddings (and again for the output head if untied).

▸ $$N_{\text{non-embed}} = 12Ld^2$$

**Check against GPT-3:** $L=96$, $d=12288$. $12\times96\times12288^2 = 1.74\times10^{11} = 174$B. Reported: 175B. ✓

### FLOPs

Per token, forward pass:
- Attention projections: $2\cdot4d^2 = 8d^2$
- Attention scores + weighted values: $2\cdot 2Td = 4Td$
- FFN: $2\cdot 8d^2 = 16d^2$

▸ $$\text{FLOPs/token/layer} \approx 24d^2 + 4Td$$

Total forward $\approx 2N$ FLOPs per token (where $N$ = parameter count), backward $\approx 4N$:

▸ $$\boxed{\ C \approx 6ND\ }$$

$C$ = total training FLOPs, $N$ = parameters, $D$ = training tokens. **Memorize this.** It underpins all of Chapter 15.

**When does attention dominate?** Attention's share is $\frac{4Td}{24d^2+4Td}$, which passes 50% at $T = 6d$. For $d=4096$: $T=24{,}576$. ▸ **Below ~25k context, a large transformer's cost is dominated by the FFN and projections, not by the $O(T^2)$ attention.** This surprises people and it is the correct answer to "isn't attention the bottleneck?"

### Memory
- Parameters: $N$ (× 2 bytes bf16)
- Gradients: $N$
- Adam states: $2N$ (fp32: $8N$ bytes)
- Activations: the dominant term; $O(BTLd)$ plus $O(BhT^2L)$ for materialized attention matrices — which FlashAttention eliminates (Ch. 12).

---

## 11.8 Mixture of Experts

### The one-line idea

Replace one big FFN with $E$ smaller ones and a router that sends each token to only $k$ of them, so parameter count grows without FLOPs growing.

### The analogy

A hospital. Rather than every patient seeing one enormous generalist who knows everything, a triage nurse routes each patient to 2 of 64 specialists. Total institutional knowledge is huge; the cost per patient is two consultations.

### The math

▸ $$y = \sum_{i\in\mathrm{TopK}(g(x))} g_i(x)\,E_i(x),\qquad g(x)=\mathrm{softmax}(W_gx)$$

**Total vs active parameters:** Mixtral 8×7B has 47B total, 13B active per token. That is the whole point: **quality tracks total parameters, cost tracks active parameters.**

### Load balancing

Left alone, the router collapses onto a few experts (rich-get-richer: a good expert gets more tokens, improves, gets more tokens). Fix with an auxiliary loss:

▸ $$\mathcal{L}_{\text{aux}} = \alpha E\sum_{i=1}^{E} f_i\,P_i$$

where $f_i$ = fraction of tokens routed to expert $i$ and $P_i$ = mean router probability for expert $i$. Minimized when both are uniform ($1/E$). Typically $\alpha=0.01$.

**Other mechanisms:** expert capacity limits with token dropping; **Expert Choice** routing (experts pick their top-$k$ tokens, guaranteeing perfect balance); auxiliary-loss-free load balancing via learned per-expert bias terms (DeepSeek-V3).

**Costs:** high memory (all experts must be resident), all-to-all communication in distributed training, training instability, and worse fine-tuning stability. **Benefit:** roughly 4–7× the effective capacity per unit of inference FLOPs.

---

## 11.9 What transformers cannot do

Worth knowing precisely, because it comes up:

- **A fixed-depth transformer is not Turing-complete** on a single forward pass. It's in a complexity class around $\mathsf{TC}^0$ (constant-depth threshold circuits) under standard assumptions — so it provably cannot solve certain problems (e.g. some state-tracking and composition tasks) in one pass at fixed depth, regardless of width.
- **Chain-of-thought changes this**: generating intermediate tokens gives the model an external serial scratchpad, lifting the effective complexity class. ▸ **This is the rigorous reason chain-of-thought works — it isn't "thinking," it's recurrence purchased with tokens.**
- **Length generalization** is poor: models trained on length $T$ often fail at $2T$, largely a positional-encoding problem (Ch. 12).
- **Exact copying and counting** are surprisingly hard without the right positional scheme.

---

## Check for Understanding

**A transformer alternates two operations — attention, which lets positions exchange information via a learned soft dictionary lookup, and a position-wise FFN, which is an associative memory holding two-thirds of the parameters — and both write into a shared residual stream whose dimension is a hard bandwidth limit on how much the model can represent at once.**

---

**Next:** [Chapter 12 — Positional Information & Long Context](12-positional-encoding-long-context.md)
