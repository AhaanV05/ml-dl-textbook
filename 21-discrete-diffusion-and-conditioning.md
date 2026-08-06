# Chapter 21 — Discrete Diffusion & Conditional Generation

> **Prerequisites:** Ch. 20, Ch. 11.
> **Scope:** two topics that belong together because discrete diffusion is almost always used conditionally, and because the DiT architecture is the standard backbone for both.

---

## Part A — Discrete diffusion

## 21.1 Why continuous diffusion doesn't transfer

Text, molecular graphs, and code are **categorical**. "Add Gaussian noise to a token ID" is meaningless — token 47 is not between 46 and 48.

Two responses:
1. **Embed and diffuse in continuous space** (Diffusion-LM, Bit Diffusion). Works, but the rounding step at the end is lossy and the model spends capacity modelling embedding geometry that doesn't matter.
2. **Define diffusion directly on the categorical simplex.** This is D3PM, and it is the cleaner answer.

---

## 21.2 D3PM

### The one-line idea

Replace "add Gaussian noise" with "randomly resample tokens according to a transition matrix," and everything from Chapter 20 goes through with matrix multiplication where the Gaussians were.

### The analogy

A game of telephone with a specific corruption rule. At each round, every word independently has some chance of being replaced — either by a random word (uniform kernel) or by the word "[BLANK]" (absorbing kernel). After enough rounds the message is unrecognizable. The model learns to undo one round of the game.

### The forward process

Let $x_t \in \{1,\dots,K\}$ be a token, represented as a one-hot row vector. The forward step is a categorical distribution defined by a transition matrix $Q_t \in \mathbb{R}^{K\times K}$ with $[Q_t]_{ij} = q(x_t=j\mid x_{t-1}=i)$, rows summing to 1:

▸ $$q(x_t\mid x_{t-1}) = \mathrm{Cat}\big(x_t;\ p = x_{t-1}Q_t\big)$$

### The closed form

Because the process is a Markov chain, the $t$-step transition is just the matrix product:

▸ $$\bar Q_t = Q_1Q_2\cdots Q_t,\qquad q(x_t\mid x_0) = \mathrm{Cat}\big(x_t;\ p = x_0\bar Q_t\big)$$

**This is the exact analogue of $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$** and serves the same essential purpose: one-shot corruption to any level, so training doesn't require simulating the chain.

### The exact posterior

By Bayes, with $\odot$ elementwise:

▸ $$q(x_{t-1}\mid x_t,x_0) = \mathrm{Cat}\!\left(x_{t-1};\ p = \frac{\big(x_tQ_t^\top\big)\odot\big(x_0\bar Q_{t-1}\big)}{x_0\bar Q_t x_t^\top}\right)$$

The denominator $x_0\bar Q_tx_t^\top$ is the scalar $q(x_t\mid x_0)$ — the normalizer. Everything is a $K$-vector operation, so the KL terms in the ELBO are **exact, closed-form categorical KLs.** No approximation anywhere.

### The transition kernels

**Uniform / multinomial:**
▸ $$Q_t = (1-\beta_t)I + \frac{\beta_t}{K}\mathbf{1}\mathbf{1}^\top$$
Each token stays with probability $1-\beta_t+\beta_t/K$, or jumps to a uniformly random token. Stationary distribution: uniform. **Analogue of the Gaussian kernel.**

**Absorbing / masking:**
▸ $$Q_t = (1-\beta_t)I + \beta_t\,\mathbf{1}\,e_{[\text{MASK}]}^\top$$
Each token either stays or is permanently replaced by `[MASK]`. Stationary distribution: all mask.

▸ **The absorbing kernel is the one that won.** Reasons: (i) the model always knows *which* positions are corrupted, so it never wastes capacity deciding whether a token is real; (ii) it connects directly to BERT-style masked modelling, so the objective is familiar and well-conditioned; (iii) the posterior simplifies dramatically — an unmasked token stays unmasked with probability 1, so only masked positions need prediction.

**Structured kernels:** if the state space has metric structure (discretized pixel intensities, ordinal categories), use a **discretized Gaussian band** $Q_{ij}\propto\exp(-|i-j|^2/\sigma^2)$ so corruption respects locality. For molecules, kernels can be built from a **token similarity matrix** (e.g. chemically similar atom types more likely to interchange), or from the data's marginal distribution ($Q_t = (1-\beta_t)I+\beta_t\mathbf{1}\tilde p^\top$ with $\tilde p$ the empirical unigram), which makes the noise process match the data's own statistics.

### The loss

The ELBO is the same structure as Chapter 20:

$$L_{\text{vb}} = \mathbb{E}_q\Big[\underbrace{\mathrm{KL}(q(x_T|x_0)\|p(x_T))}_{L_T} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Big]$$

**The parameterization that makes it work:** rather than predicting $p_\theta(x_{t-1}\mid x_t)$ directly, the network predicts a distribution over the **clean data** $\tilde p_\theta(x_0\mid x_t)$, and the reverse step is obtained by plugging that into the known posterior:

▸ $$p_\theta(x_{t-1}\mid x_t) \ \propto\ \sum_{\tilde x_0} q(x_{t-1}\mid x_t,\tilde x_0)\,\tilde p_\theta(\tilde x_0\mid x_t)$$

This is exactly the discrete analogue of $\epsilon$-prediction: **let the network solve the easy, well-posed problem (what was the original token?) and let the known posterior handle the rest.** It also means sampling can skip steps trivially.

### The auxiliary $x_0$ loss

D3PM's full objective adds a direct cross-entropy term on the clean-data prediction:

▸ $$\boxed{\ \mathcal{L} = \mathcal{L}_{\text{vb}} + \lambda\ \mathbb{E}_{x_0,t,x_t}\big[-\log\tilde p_\theta(x_0\mid x_t)\big]\ }$$

typically $\lambda = 0.001$–$0.01$.

▸ **Why it's there:** the VLB's per-timestep KL terms are small and noisy at large $t$ (where the posterior is nearly the stationary distribution regardless of $x_0$), giving a weak training signal. The direct cross-entropy on $x_0$ is a dense, well-scaled auxiliary target at every timestep. It plays the same role as $\mathcal{L}_{\text{simple}}$'s reweighting in continuous diffusion — trading likelihood optimality for sample quality and trainability.

**This auxiliary term is the quantity most implementations log as the primary training/validation metric**, since it is interpretable directly as a token-level cross-entropy (compare to $\log K$; see Ch. 1 §1.4.2). It is what Case Study A's `val_realCE` measures.

▸ **The evaluation subtlety that follows** (and the reason Case Study A appears throughout Part I): this metric is a **conditional expectation over $t$**, and its value varies enormously across $t$ — near $\log K$ at large $t$, near zero at small $t$. Sampling $t$ freshly per batch therefore injects a between-$t$ variance term into the estimate that divides only by the *number of batches*, not the number of examples (Ch. 3 §3.6). **Always report per-$t$-bucket cross-entropy alongside the aggregate**, and use a fixed stratified $t$ grid for validation. Aggregate CE can be flat while every bucket improves, purely from a shift in which $t$ values happened to be drawn.

---

## 21.3 The modern discrete-diffusion landscape

**SEDD (Score Entropy Discrete Diffusion).** Generalizes the score to discrete spaces as the **concrete score** — the ratio $\frac{p(y)}{p(x)}$ between states — and trains it with a denoising score entropy loss. Achieves perplexities competitive with autoregressive transformers of the same size, which was the first time discrete diffusion was genuinely competitive on language.

**MDLM / MD4 (Masked Diffusion Language Models).** Show that with the absorbing kernel, a well-chosen parameterization, and careful attention to the continuous-time limit, the objective simplifies to a **weighted average of masked-language-modelling losses**:
$$\mathcal{L} \propto \mathbb{E}_{t}\left[\frac{1}{t}\,\mathbb{E}\big[-\log p_\theta(x_0^{\text{masked}}\mid x_t)\big]\right]$$
▸ **Discrete diffusion with an absorbing kernel is BERT with a random, continuously-varying mask rate and a principled weighting.** That is the cleanest possible statement of what these models are, and it explains both why they train stably and why they inherit BERT-like inductive biases.

**Why anyone wants this over autoregressive modelling:**
- **Parallel generation.** All positions decode simultaneously; a full sequence in 10–50 steps rather than $T$.
- **Bidirectional context** — every position sees every other at every step.
- **Controllable infilling** — conditioning on arbitrary known positions is trivial (just don't mask them), whereas autoregressive infilling requires special training.
- Natural fit for **non-sequential data**: molecular graphs, protein sequences with structural constraints, layouts.

**Why autoregressive still wins on general language:** likelihood is still somewhat worse at matched compute, KV caching has no clean analogue (each step recomputes everything), and the parallel-decoding independence assumption within a step hurts coherence.

---

## Part B — Conditional generation

## 21.4 The conditioning mechanisms

### The one-line idea

There are four ways to tell a network "generate *this* kind of thing," and they differ in how much interaction they allow between the condition and the content.

| Mechanism | Form | Best for |
|---|---|---|
| **Concatenation** | append $c$ to the input | spatially aligned conditions (segmentation maps, depth, inpainting masks) |
| **Cross-attention** | $Q$ from $x$, $K,V$ from $c$ | variable-length, token-structured conditions (text) |
| **FiLM / AdaGN** | $\gamma(c)\odot h + \beta(c)$ | global scalar or vector conditions (class, timestep) |
| **AdaLN-Zero** | FiLM on LayerNorm, gated, zero-init | **the DiT standard** |

### FiLM

▸ $$\mathrm{FiLM}(h\mid c) = \gamma(c)\odot h + \beta(c)$$

A cheap, powerful mechanism: a global condition modulates the *scale and shift* of every feature. It is how timestep information enters almost every diffusion model.

### AdaLN and AdaLN-Zero

Adaptive LayerNorm makes the norm's gain and bias functions of the condition:
$$\mathrm{AdaLN}(h\mid c) = \gamma(c)\odot\frac{h-\mu}{\sigma}+\beta(c)$$

**AdaLN-Zero** adds a third output — a per-block gate $\alpha(c)$ applied to the residual branch — and **initializes the MLP producing $\alpha$ to output zero**:

▸ $$x \leftarrow x + \alpha(c)\odot\mathrm{Attn}\big(\mathrm{AdaLN}(x\mid c)\big)$$

with $\alpha=0$ at initialization, so **every block starts as the exact identity.**

▸ **Why this matters so much:** it is the same zero-init-residual principle as Fixup, ReZero, and `zero_init_residual` in ResNets (Ch. 6 §6.4, Ch. 8 §8.2). The network begins as a shallow, perfectly-conditioned function and grows depth as training proceeds. In the DiT ablations, AdaLN-Zero beat in-context conditioning, cross-attention, and plain AdaLN **by a wide margin at every model size** — it was the single most important architectural finding in that paper. If you are debugging early instability in a conditional diffusion transformer, verify this zero-init is intact; it is easy to break during refactoring.

---

## 21.5 The DiT architecture

**Diffusion Transformer** — replaces the U-Net backbone with a plain transformer.

**The forward path:**
1. **Patchify.** A latent of shape $I\times I\times C$ becomes $T=(I/p)^2$ tokens of dimension $d$ via a linear projection of $p\times p$ patches. $p\in\{2,4,8\}$; smaller $p$ means more tokens, more FLOPs, better quality.
2. **Positional embeddings** — frozen sinusoidal (2-D) or RoPE.
3. **Condition embedding.** Timestep $t$ → sinusoidal → MLP. Class label → embedding table. Summed into a single conditioning vector $c$. (For text, a cross-attention sub-layer is added instead, as in PixArt/SD3.)
4. **$L$ DiT blocks:**
   ```
   shift, scale, gate  = MLP(c).chunk(6)      # 6 outputs: 3 for attn, 3 for ffn
   x = x + gate_1 * Attn(modulate(LN(x), shift_1, scale_1))
   x = x + gate_2 * FFN(modulate(LN(x), shift_2, scale_2))
   ```
   with the modulation MLP zero-initialized (AdaLN-Zero).
5. **Final layer:** AdaLN → linear → unpatchify to predict $\epsilon$ (and optionally $\Sigma$).

### The scaling result

▸ **DiT's central finding: sample quality (FID) improves monotonically and predictably with model FLOPs**, across patch sizes and model widths — the same clean scaling behaviour as language models, which U-Nets did not exhibit. That is why every major image and video model since 2023 (SD3, Flux, Sora, and successors) uses a transformer backbone.

**MM-DiT** (SD3) extends this with **separate weights for text and image token streams** that attend jointly — a two-stream transformer. It substantially improves text rendering and prompt adherence over a single shared stream.

### Conditioning strength diagnostics

Worth building into any conditional generative project:
- **Condition ablation.** Generate with $c$ and with $\varnothing$ and measure the divergence between output distributions. If they are similar, the model is ignoring the condition.
- **Condition swap.** Generate with a *mismatched* condition; the outputs should change substantially.
- **CFG sensitivity curve.** Sweep $w$ and plot fidelity vs diversity. A flat curve means the conditional and unconditional branches have not differentiated — usually a sign of too-high condition dropout, too-weak conditioning, or a broken AdaLN-Zero.
- **Per-condition validation loss.** Aggregate loss can hide the fact that the model is excellent on common conditions and useless on rare ones.

---

## 21.6 Guidance in the discrete setting

Classifier-free guidance was derived for continuous scores (Ch. 20 §20.8). In the discrete case it is applied to **logits**:

▸ $$\ell_{\text{guided}} = \ell_\varnothing + w\,(\ell_c - \ell_\varnothing)$$

then softmax. This is extrapolation in logit space, which is equivalent to a *product-of-experts* reweighting of the categorical distribution:
$$p_{\text{guided}}(x)\ \propto\ p_\varnothing(x)\left(\frac{p_c(x)}{p_\varnothing(x)}\right)^{w}$$

Same trade-off as continuous CFG: fidelity up, diversity down, and at large $w$ the distribution collapses onto its mode.

**Practical notes for discrete guidance:**
- Guidance is applied at *every* denoising step, so effects compound; $w$ values are typically smaller than in image diffusion.
- With absorbing kernels, guidance only affects masked positions.
- **Condition dropout rate** (for training the $\varnothing$ branch) is typically 10%. Too low and the unconditional branch is undertrained; too high and the conditional branch suffers.

---

## 21.7 Practical guidance for discrete diffusion projects

1. **Prefer the absorbing/masking kernel** unless there is a specific reason for uniform or structured noise.
2. **Predict $x_0$**, not the reverse transition directly.
3. **Include the auxiliary $x_0$ cross-entropy** with $\lambda\approx0.01$; it is what makes training stable and the metric interpretable.
4. **Log per-$t$-bucket cross-entropy**, always. The aggregate is nearly uninterpretable on its own.
5. **Fix the validation $t$ grid** (stratified) and the validation subset. This is free and typically reduces metric noise by an order of magnitude (Ch. 3 §3.6).
6. **Maintain an EMA of the weights** ($\gamma\approx0.9999$) and evaluate the EMA.
7. **Compare CE to $\log K$**, and estimate the irreducible floor at each $t$ if possible, so you know how much headroom actually exists.
8. **Use AdaLN-Zero** for conditioning, and verify the zero-init survived refactoring.
9. **Evaluate generation, not just likelihood.** For molecules: validity, uniqueness, novelty, distributional property matches (a KL or Wasserstein distance on logP, QED, molecular weight), and scaffold diversity. Cross-entropy can improve while sample quality does not.
10. **Read the slope, not the minimum**, when judging progress (Ch. 3 §3.6).

---

## Check for Understanding

**Discrete diffusion replaces Gaussian noise with a categorical transition matrix, keeping every structural feature of Chapter 20 — a closed-form cumulative corruption $\bar Q_t$, an exact posterior, a clean-data parameterization, and an auxiliary $x_0$ cross-entropy that carries most of the training signal — and with an absorbing kernel it reduces to BERT with a continuously varying mask rate; conditioning enters through AdaLN-Zero, whose zero-initialized gate makes every block start as the identity, which is the same trick that made deep residual networks trainable in the first place.**

---

**Next:** [Chapter 22 — Classical Supervised Learning](22-classical-supervised-learning.md)
