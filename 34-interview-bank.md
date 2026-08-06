# Chapter 34 — The Interview Bank

> **How to use this:** cover the answer, attempt it out loud, then compare. An answer you can *recognize* is not an answer you can *give*. Questions marked **★** are the ones that most reliably separate strong candidates; §34.11 collects them.
> Section references point to where the full treatment lives.

---

## 34.1 Mathematics & statistics

**1. What does the SVD tell you about a matrix?**
$A=U\Sigma V^\top$ decomposes any linear map into rotate → scale → rotate. Singular values give the scaling in each direction; the number of nonzero ones is the rank; $\sigma_1$ is the operator norm; $\sigma_1/\sigma_r$ is the condition number. Truncating gives the optimal rank-$k$ approximation in Frobenius and spectral norm (Eckart–Young). (§1.1)

**2. Why is the condition number important in optimization?**
Gradient descent's convergence rate for a quadratic is $\left(\frac{\kappa-1}{\kappa+1}\right)^t$. Large $\kappa$ means the loss surface is a narrow ravine: the step size is bounded by the steepest direction while progress is limited by the flattest. Momentum improves the dependence to $\sqrt\kappa$. (§4.3–4.5)

**3. What is the difference between forward-mode and reverse-mode autodiff, and why does DL use reverse?**
Forward computes a Jacobian-vector product per input dimension; reverse computes a vector-Jacobian product per output dimension. Cost scales with $\min(n_{\text{in}}, n_{\text{out}})$ respectively. A loss has one output and millions of parameters, so reverse mode gets all gradients in one backward pass. (§1.3)

**4. Explain KL divergence, and the difference between forward and reverse KL.** ★
$\mathrm{KL}(p\|q)=\mathbb{E}_p[\log p/q]$ — the extra bits from coding $p$ with a codebook built for $q$. Non-negative, asymmetric, not a metric. **Forward** $\mathrm{KL}(p\|q)$ is mode-covering: $q$ must be positive wherever $p$ is, so it smears. **Reverse** $\mathrm{KL}(q\|p)$ is mode-seeking: $q$ can ignore modes but must not put mass where $p$ has none. Maximum likelihood minimizes forward KL (hence blurry VAEs); variational inference minimizes reverse (hence underestimated posterior variance). (§1.4)

**5. Derive the ELBO.**
$\log p(x) = \log\int p(x,z)dz = \log\mathbb{E}_q\left[\frac{p(x,z)}{q(z)}\right]\ge\mathbb{E}_q\left[\log\frac{p(x,z)}{q(z)}\right]$ by Jensen. Expanding, the gap is exactly $\mathrm{KL}(q(z)\|p(z|x))$. (§1.4)

**6. What is the standard error of the mean, and why does it matter?**
$\sigma/\sqrt n$. Halving error bars needs $4\times$ the data. It governs whether any measured improvement is real. (§1.3, §3.6)

**7. Central Limit Theorem — statement and a caveat.**
The standardized sample mean of i.i.d. finite-variance variables converges to $\mathcal{N}(0,1)$. Caveats: requires finite variance (fails for Cauchy), convergence is slow in the tails, and it says nothing about the maximum — which is why record statistics (§3.6) need a different tool.

**8. Bayes' theorem, and what a prior does.**
$p(\theta|x)\propto p(x|\theta)p(\theta)$. The prior is a regularizer: $\ell_2$ is a Gaussian prior, $\ell_1$ a Laplace prior, and MAP estimation is penalized maximum likelihood. (§7.5)

**9. What is mutual information and where does it appear in DL?**
$I(X;Y)=H(X)-H(X|Y)=\mathrm{KL}(p(x,y)\|p(x)p(y))$. Appears in InfoNCE bounds (§25.3), the information bottleneck, feature selection, and the epistemic-uncertainty decomposition (§33.3).

**10. Explain the Johnson–Lindenstrauss lemma and one place it matters.**
$n$ points can be embedded into $O(\log n/\epsilon^2)$ dimensions preserving pairwise distances to $1\pm\epsilon$. Underpins random projection, LSH-based deduplication, and — conceptually — superposition, since it implies exponentially many nearly-orthogonal directions exist. (§1.2, §32.2)

---

## 34.2 Learning theory & evaluation

**11. Derive the bias–variance decomposition.**
$\mathbb{E}[(y-\hat f)^2] = \big(\mathbb{E}[\hat f]-f\big)^2 + \mathbb{E}\big[(\hat f-\mathbb{E}[\hat f])^2\big] + \sigma^2$. Add and subtract $\mathbb{E}[\hat f]$; the cross term vanishes. (§2.2)

**12. Why are VC-dimension bounds useless for deep networks?** ★
A network with $p$ parameters at 32-bit precision gives $\log|\mathcal{H}|\le22p$; for $p=10^7$, $n=10^5$ the bound exceeds 1 on a loss bounded by 1. It's vacuous. Deeper reason: uniform convergence bounds what the class *could* do; deep nets can fit random labels (Zhang et al.), so any capacity-only bound must be vacuous. The explanation must come from the algorithm's implicit bias. (§2.4, §2.7, §31.2)

**13. Explain the bootstrap. When does it fail?**
Resample with replacement $B$ times, recompute the statistic, use the spread as the sampling distribution. Fails for extremes (max/min — the bootstrap distribution is atomic), for dependent data without block resampling, and for heavy tails with infinite variance. (§3.1)

**14. Why does $k$-fold CV variance not shrink like $1/k$?**
The folds share training data, so the fold estimates are correlated: $\mathrm{Var}=\frac{\sigma^2}{k}+\frac{k-1}{k}\rho\sigma^2$. The correlation term doesn't vanish. (§3.3)

**15. What is nested cross-validation and why do you need it?**
Outer loop estimates performance; inner loop selects hyperparameters. Tuning and reporting on the same CV gives optimistically biased numbers, with bias growing in the number of configurations tried. (§3.3)

**16. You track the minimum of a noisy metric across 40 epochs. Why is the reported best optimistically biased?** ★
The minimum of $n$ noisy draws is below the true mean by roughly $\sigma\Phi^{-1}\!\big(1-\frac{1}{n+1}\big)$ — about $2\sigma$ for $n\approx43$. You are reporting a favourable noise draw, not a better model. Fix: re-evaluate the selected checkpoint on a fresh, larger set. (§3.6)

**17. A model shows no new best for 13 epochs, then two records back to back. What can you conclude?** ★
Under a flat-performance null, $P(\text{no record in epochs }23..35)=\prod_{k=23}^{35}(1-1/k)=22/35=0.63$ — the dry spell is the *most likely* outcome and carries no information. Back-to-back records at 36 and 37 have probability $\frac{1}{36\cdot37}=7.5\times10^{-4}$ — that's the evidence. **Read the cluster, not the gap.** (§3.6)

**18. Precision, recall, F1, ROC-AUC, PR-AUC — when do you use which?**
Precision = TP/(TP+FP); recall = TP/(TP+FN). ROC-AUC is invariant to class balance, which is a *problem* when positives are rare — it looks good while precision is terrible. PR-AUC is sensitive to the positive rate and is the right choice under heavy imbalance. F1 is their harmonic mean; use $F_\beta$ when the costs differ. (§22.6)

**19. What's a permutation test?**
Shuffle labels many times, recompute the statistic, and locate the observed value in that null distribution. Assumption-free, exact in principle, and the right tool when you don't trust a parametric null. (§3.4)

**20. What is Monte Carlo standard error and why report it?**
The uncertainty from your own sampling procedure, $\sigma/\sqrt{B}$. Without it, a "0.3% improvement" from 100 samples is indistinguishable from noise. (§3.5)

---

## 34.3 Optimization

**21. Adam vs SGD — what actually differs, and when do you use each?** ★
Adam keeps per-parameter first and second moment estimates and normalizes the step by $\sqrt{\hat v}$, making it invariant to per-parameter gradient scale. SGD+momentum generalizes better on vision (its implicit bias is $\ell_2$/max-margin, §31.2); Adam is essential for transformers, where gradient scales vary enormously across layers and embedding rows are sparsely updated. (§5.3, §5.6)

**22. Derive Adam's bias correction.**
$m_t=(1-\beta_1)\sum_{i}\beta_1^{t-i}g_i$. With $m_0=0$ and stationary $g$, $\mathbb{E}[m_t]=(1-\beta_1^t)\mathbb{E}[g]$. Divide by $(1-\beta_1^t)$. Without it, early steps are biased toward zero — and for $\beta_2=0.999$ the $v$ bias persists for ~1000 steps, which would make early steps enormous. (§5.3)

**23. What's the difference between Adam+L2 and AdamW?** ★
L2 adds $\lambda\theta$ to the gradient, which then gets divided by $\sqrt{\hat v}$ — so parameters with large historical gradients get *less* decay. AdamW applies $\theta\leftarrow\theta-\eta\lambda\theta$ directly, decoupled from the adaptive scaling. That restores the intended uniform shrinkage, and it's why AdamW is standard. (§5.2)

**24. Why do transformers need learning-rate warmup?**
Early gradients are unrepresentative, and Adam's $\hat v$ estimate is high-variance in the first ~$1/(1-\beta_2)$ steps, so the effective step can be enormous. Post-LN transformers also have $O(\sqrt L)$ gradient amplification at initialization. Warmup lets the second-moment estimate stabilize. RAdam derives the correction analytically. (§5.4)

**25. What is the Edge of Stability?**
GD on neural networks doesn't stay in the regime $\eta<2/\lambda_{\max}$; instead $\lambda_{\max}$ *rises* until it hits $2/\eta$ and then hovers there, with the loss decreasing non-monotonically. So sharpness is determined by the learning rate, not fixed by the problem. (§5.5)

**26. Explain gradient clipping and which variant to use.**
Rescale $g\leftarrow g\cdot\min(1, c/\|g\|)$ (norm clipping, preserves direction) rather than clipping elementwise (which changes direction). Essential for RNNs and large transformers where rare batches produce huge gradients. (§4.8)

**27. How should learning rate scale with batch size?**
Linear scaling ($\eta\propto B$) for SGD, from the gradient-noise argument; square-root scaling ($\eta\propto\sqrt B$) is safer for Adam. Both need warmup at large $B$. (§4.6)

**28. Why does SGD generalize better than full-batch GD?**
Gradient noise acts as a temperature $\propto\eta/B$ in an SDE whose stationary distribution weights basins by *volume*. Wide (flat) basins have exponentially more volume in high dimensions, and flat minima generalize better. Predicts — correctly — that large batch at fixed LR generalizes worse. (§4.6, §31.4)

**29. What is natural gradient and how does K-FAC approximate it?**
Natural gradient is $F^{-1}g$ with $F$ the Fisher information — steepest descent in distribution space rather than parameter space, hence invariant to reparameterization. K-FAC approximates $F$ per layer as a Kronecker product $A\otimes G$ of input and gradient covariances, making the inverse cheap. (§4.7)

**30. What's the point of SAM?**
Minimize $\max_{\|\epsilon\|\le\rho}\mathcal{L}(\theta+\epsilon)$ — the worst loss in a neighbourhood, so the solution is flat. Implemented as two forward-backward passes: ascend to the worst nearby point, take the gradient there, apply it at the original point. ~2× cost, ~1% ImageNet gain. (§5.7)

---

## 34.4 Neural networks & training

**31. Derive backpropagation for one layer.**
With $z=Wa_{\text{prev}}+b$, $a=\sigma(z)$, $\delta=\frac{\partial\mathcal{L}}{\partial z}$: $\delta^{(l)}=\big(W^{(l+1)\top}\delta^{(l+1)}\big)\odot\sigma'(z^{(l)})$, $\frac{\partial\mathcal{L}}{\partial W^{(l)}}=\delta^{(l)}a^{(l-1)\top}$, $\frac{\partial\mathcal{L}}{\partial b^{(l)}}=\delta^{(l)}$. (§6.3)

**32. Derive He initialization.** ★
For $z=Wa$ with $n$ inputs, $\mathrm{Var}(z)=n\,\mathrm{Var}(W)\mathrm{Var}(a)$. ReLU zeroes half the units, halving the variance: $\mathrm{Var}(a)=\frac12\mathrm{Var}(z_{\text{prev}})$. To keep variance constant, $n\,\mathrm{Var}(W)\cdot\frac12=1$, so $\mathrm{Var}(W)=2/n$. Xavier uses $2/(n_{\text{in}}+n_{\text{out}})$ for symmetric activations. (§6.4)

**33. Why does the vanishing gradient problem occur, quantitatively?**
The gradient through $L$ layers multiplies $L$ Jacobians. With sigmoid, $\max\sigma'=0.25$, so 10 layers gives $\le0.25^{10}\approx10^{-6}$. ReLU (derivative 1 on the positive side) and residual connections (an additive identity path) fix it. (§6.4, §8.2)

**34. Why do ResNets work? Give three explanations.**
(a) Gradient highway — $\frac{\partial x_L}{\partial x_\ell}=1+\sum\frac{\partial F}{\partial x_\ell}$, so the leading 1 guarantees flow. (b) Ensemble of $2^L$ shallow paths — deleting a block barely hurts. (c) Loss-landscape smoothing. The first is strongest. Note the original problem was *degradation* (higher training error at depth), i.e. optimization, not overfitting. (§8.2)

**35. BatchNorm vs LayerNorm — why do transformers use LN?** ★
BN normalizes over the batch per channel, so it creates a train/eval discrepancy, breaks with small batches, leaks information across samples, and interacts badly with variable sequence lengths and padding. LN normalizes per sample over features — batch-independent, identical in train and eval, length-agnostic. (§7.2–7.3)

**36. What is BatchNorm actually doing, if not fixing internal covariate shift?**
Smoothing the loss landscape (reducing the Lipschitz constants of the loss and its gradient), plus making the layer scale-invariant so the weight norm becomes an inverse effective learning rate. Santurkar et al. disproved the covariate-shift story directly. (§7.1, §7.4)

**37. Why should you not apply weight decay to LayerNorm gains and biases?** ★
Scale-invariance is what makes weight decay a learning-rate control (§7.4). Norm gains and biases are *not* scale-invariant — they directly set the layer's output magnitude — so decaying them shrinks the function for no benefit.

**38. Explain the coupling between learning rate and weight decay in a normalized network.** ★
With normalization, the function is invariant to $\|W\|$, the gradient is orthogonal to $W$ (so $\|W\|$ grows under SGD), and the angular update scales as $\eta/\|W\|^2$. Weight decay keeps $\|W\|$ small so the effective LR stays high. At equilibrium $\eta_{\text{eff}}\propto\sqrt{\eta\lambda}$ — so only the product matters. (§7.4)

**39. Three ways to understand dropout.**
(a) Ensembling $2^n$ weight-sharing subnetworks, approximated at test time by the geometric mean. (b) For linear regression, exactly equivalent to data-dependent $\ell_2$. (c) Approximate variational inference — hence MC dropout. Note dropout is absent from LLM pretraining, because with one pass over a huge corpus there is no overfitting to prevent. (§7.5)

**40. Why is early stopping equivalent to $\ell_2$?**
For a quadratic with eigenvalues $\lambda_i$, GD from 0 gives $\theta_i(t)=(1-(1-\eta\lambda_i)^t)\theta_i^*$; matching ridge's $\frac{\lambda_i}{\lambda_i+\lambda}$ gives $\lambda\approx1/(\eta t)$. Training longer = weaker regularization. (§7.5)

**41. Why GELU/SwiGLU over ReLU?**
GELU $=x\Phi(x)$ is smooth and non-monotone, giving nonzero gradient for small negatives (no dead units) and better empirical results. SwiGLU adds a multiplicative gate: $W_3(\mathrm{SiLU}(W_1x)\odot W_2x)$, with $d_{\text{ff}}=\frac83d$ to hold parameters constant. ~1% perplexity gain. (§6.5)

**42. Your training loss is NaN. Walk me through debugging.**
Check for: LR too high (lower it, add warmup); missing gradient clipping; log(0) or division by zero in a custom loss; fp16 overflow (switch to bf16 or check loss scaling); a corrupted batch (log the batch index and inspect); exploding attention logits (add QK-norm); bad initialization. Then bisect: overfit a single batch first — if that fails, the bug is in the model or loss, not the data pipeline. (§6.7, §14.6)

**43. Your model gets 99% train and 60% test accuracy. What do you do?**
That's overfitting: more data or augmentation first, then regularization (weight decay, dropout, early stopping), then reduce capacity. But check first for a *leak* in the other direction — verify the split is grouped correctly and that no preprocessing was fitted on the full dataset.

**44. Your model gets 60% on both train and test.**
Underfitting or a bug. Check that the model can overfit 10 examples; if not, there's a bug (label misalignment, wrong loss reduction, frozen parameters, LR ~0). If it can, increase capacity, train longer, raise the LR, or improve features.

---

## 34.5 Transformers & LLMs

**45. Why divide by $\sqrt{d_k}$ in attention?** ★
With unit-variance $q,k$, $\mathrm{Var}(q^\top k)=d_k$, so scores have SD $\sqrt{d_k}$. Large-variance scores saturate the softmax, whose Jacobian $p_i(\delta_{ij}-p_j)\to0$ — gradients vanish. Dividing restores unit variance. (§11.2)

**46. Why multiple heads instead of one wide one?**
One softmax gives one attention distribution per query; the model must commit to one weighting. $h$ heads let a token simultaneously attend to its syntactic head, its antecedent, and the previous token. Also a rank argument: a sum of $h$ low-rank-driven operations is more expressive than one. (§11.3)

**47. Why separate Q, K, and V projections?**
$Q=K$ forces a symmetric attention matrix and maximal self-attention. Separating them allows asymmetric relations. $V=K$ would force the matching key to equal the transmitted content; separating lets a token be *found* by one property and *contribute* another. (§11.1)

**48. Give the parameter count and FLOP formula for a transformer.** ★
$4d^2$ (attention) $+8d^2$ (FFN) $=12d^2$ per layer, so $N\approx12Ld^2$ non-embedding. Training FLOPs $C\approx6ND$. Check: GPT-3, $L=96$, $d=12288$ → 174B. (§11.7)

**49. Is attention the compute bottleneck?**
Not usually. Attention's share is $\frac{4Td}{24d^2+4Td}$, hitting 50% at $T=6d$ — about 25k tokens for $d=4096$. Below that, the FFN and projections dominate. The real bottleneck at inference is the **KV cache**, which is memory. (§11.7, §12.7)

**50. Explain RoPE and why it's better than learned absolute embeddings.** ★
Rotate $q$ and $k$ by angle $m\theta_i$ in $d/2$ 2-D planes. Then $\langle R_mq, R_nk\rangle = q^\top R_{n-m}k$ — the score depends only on relative position, achieved with purely absolute operations. No $T\times T$ bias matrix, KV-cache compatible, and extendable by rescaling the base. Learned absolute embeddings have a hard length limit and don't extrapolate. (§12.4)

**51. How would you extend a model's context from 4k to 32k?**
Options: Position Interpolation (scale positions down — crowds local resolution, needs fine-tuning); NTK-aware base scaling (increase $\theta_{\text{base}}$; often works without fine-tuning); **YaRN** (per-dimension: interpolate low frequencies, extrapolate high, plus attention temperature — current best). Then fine-tune on long documents, and verify with RULER, not just needle-in-a-haystack. (§12.4, §12.8)

**52. What does FlashAttention do?** ★
It recognizes attention is *memory-bound*, not compute-bound. It tiles Q, K, V into SRAM-sized blocks and uses an **online softmax** (running max and sum with rescaling) to compute the exact result without ever materializing the $T\times T$ matrix, recomputing it in the backward pass instead of storing it. Memory $O(T^2)\to O(T)$, 2–4× faster, **numerically exact**.

**53. Compute the KV cache size for a 70B model at 4k context.**
$2\times L\times T\times h_{kv}\times d_{\text{head}}\times$ bytes. With $L=80$, $d_{\text{head}}=128$, bf16, and **full MHA** ($h_{kv}=64$): $2\cdot80\cdot4096\cdot64\cdot128\cdot2=10.7$ GB **per sequence** — 343 GB at batch 32, more than the weights. With **GQA** ($h_{kv}=8$, which is what LLaMA-2-70B actually ships) it's 1.34 GB. Quoting both, and explaining that the second is why the first never gets deployed, is the complete answer. (§12.7)

**54. MHA vs MQA vs GQA vs MLA.**
MHA: $h$ K/V heads. MQA: 1 shared K/V head — maximal saving, noticeable quality loss. GQA: $g$ groups (typically 8) — 8× saving at near-MHA quality, the current default. MLA: compress K/V into a low-rank latent and cache only that — >90% reduction with quality better than GQA at matched cache. (§12.7)

**55. Why did decoder-only beat encoder–decoder for LLMs?**
Training-signal density (every token is a target vs BERT's 15%), architectural simplicity, natural in-context learning, and the fact that splitting parameters into two pools is wasteful at a fixed budget for open-ended generation. Encoder–decoder still wins with a fixed repeatedly-attended source (translation, ASR). (§11.6, §13.6)

**56. Walk through BPE.**
Start with characters (or the 256 bytes), count adjacent pairs, merge the most frequent, record the merge, repeat to the target vocabulary. Encoding applies the merges in learned order. Byte-level BPE means no `<UNK>` ever. WordPiece merges by $\frac{c(ab)}{c(a)c(b)}$ (a PMI-like criterion) rather than raw frequency; Unigram runs backwards, pruning from a large candidate set by likelihood loss. (§10.4)

**57. Why are LLMs bad at counting letters in a word?** ★
Tokenization. "strawberry" is ~3 tokens, not 10 characters — the model can't *see* the letters, only a memorized association. Same root cause as poor arithmetic (inconsistent digit grouping), rhyming, and reversal. (§10.4)

**58. Derive word2vec's negative sampling objective.**
Replace the $|V|$-way softmax with binary logistic classification: $-\log\sigma(u_o^\top v_c)-\sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\log\sigma(-u_{w_i}^\top v_c)$, with $P_n\propto U(w)^{3/4}$. Levy & Goldberg showed this implicitly factorizes the shifted PMI matrix. (§10.3)

**59. What is in-context learning and why does it happen?** ★
Task performance from prompt examples with no weight update. Mechanistically explained by **induction heads**: a previous-token head writes token $t-1$'s identity into position $t$; a later head queries with the current token to find earlier occurrences and copies what followed. Requires ≥2 layers. Evidence: the ability and the heads appear in the same narrow training window, and ablating the heads destroys the ability. Note that randomizing the demonstration *labels* barely hurts — the demonstrations mainly convey format and label space. (§13.3)

**60. Nucleus sampling vs top-$k$ — why did top-$p$ win?**
Top-$k$ uses a fixed count, but the number of plausible next tokens varies enormously by context. Top-$p$ takes the smallest set summing to $p$, adapting to the distribution's entropy. (§13.4)

**61. Why is beam search wrong for open-ended generation?**
Higher sequence probability correlates with worse human-judged quality past a point — the likelihood trap. The highest-probability continuation of most prompts is degenerate repetition. Beam search is right for translation and summarization, where the output is highly constrained by the input. (§13.4)

**62. What are the pitfalls of perplexity?**
Tokenizer-dependent (compare bits-per-byte instead), data-dependent, invalidated by contamination, and only loosely related to usefulness after post-training — RLHF typically *raises* perplexity while improving preference scores. (§13.5)

**63. Derive the Chinchilla scaling result.** ★
Fit $L=E+AN^{-\alpha}+BD^{-\beta}$, minimize subject to $C=6ND$. Substituting $D=C/6N$ and setting $dL/dN=0$ gives $N\propto C^{\beta/(\alpha+\beta)}$, $D\propto C^{\alpha/(\alpha+\beta)}$. With $\alpha=0.34,\beta=0.28$: exponents 0.452 and 0.548 — roughly equal scaling, ~20 tokens per parameter. (§15.2)

**64. Why is LLaMA-3-8B trained on 15T tokens when Chinchilla says 160B?**
Chinchilla optimizes *training* compute only. Inference cost is linear in $N$ and paid forever, so accounting for deployment shifts the optimum strongly toward smaller models trained far longer. A model too large to serve is worthless. (§15.2)

**65. Is emergence real?** ★
Partly. Schaeffer et al. showed that discontinuous metrics manufacture discontinuities: if per-token accuracy $p$ improves smoothly and the metric is exact match over $k$ tokens, the measured score $p^k$ looks like a phase transition. Continuous metrics on the same outputs show smooth improvement. But some *mechanistic* transitions (induction-head formation) do appear genuinely sharp. Underlying capability: smooth. User-visible usefulness: can jump. (§15.4)

**66. What is $\mu$P and why does it matter?**
A parameterization whose per-layer init and LR scalings keep activations and updates $\Theta(1)$ at any width, so optimal hyperparameters become width-independent and transfer from a small proxy to a large model. It also keeps the model in the feature-learning rather than lazy/NTK regime. (§15.3, §30.2)

**67. Explain Mixture of Experts and load balancing.**
Replace the FFN with $E$ experts and a top-$k$ router; total parameters grow while active FLOPs don't. Left alone the router collapses (rich-get-richer), so add $\mathcal{L}_{\text{aux}}=\alpha E\sum_i f_iP_i$, minimized at uniform routing. Alternatives: expert-choice routing, capacity limits, bias-based balancing. (§11.8)

**68. Why do transformers need chain of thought for hard problems?** ★
A fixed-depth transformer has bounded serial computation per forward pass (roughly $\mathsf{TC}^0$). Emitting intermediate tokens provides an external serial scratchpad — each token is another forward pass conditioned on the last. **CoT buys depth with tokens.** (§11.9, §16.7)

---

## 34.6 Post-training, RAG, inference

**69. Derive DPO.** ★★
(1) The KL-regularized RLHF objective's optimum is $\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)e^{r(x,y)/\beta}$ — rewrite the objective as $-\beta[\mathrm{KL}(\pi\|\pi^*)-\log Z]$ and note KL is minimized at 0. (2) Invert: $r=\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z(x)$. (3) Substitute into Bradley–Terry, which depends only on the *difference* of rewards, so $\beta\log Z(x)$ cancels. Result: $\mathcal{L}=-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$. (§16.5)

**70. Where is DPO weaker than PPO?**
DPO is offline: its constraint binds only on responses in the dataset, so it can push probability mass onto unseen outputs — the observed pathology is that both $y_w$ and $y_l$ decrease in likelihood. Fixes: iterative/online DPO, plus an SFT term on $y_w$. Well-tuned online PPO still edges it out on hard tasks. (§16.5)

**71. Why is the KL penalty in RLHF essential?**
It keeps the policy inside the region where the reward model is valid, preserves fluency and pretrained knowledge, and prevents collapse onto a few reward-maximizing degenerate outputs. Without it the policy exploits the reward model. (§16.4)

**72. What is GRPO and why does it work?**
Replace the value network with the mean reward over a group of $G$ completions for the same prompt. Valid because any action-independent baseline leaves the policy gradient unbiased — the group mean is a Monte-Carlo estimate of $V(x)$. Trades a second large model for more sampling. (§16.6, §27.4)

**73. What is RLVR and why did it change reasoning models?**
Reinforcement learning from *verifiable* rewards — check the math answer or run the unit tests instead of using a learned reward model. It eliminates reward hacking at the source, because you can't fool a compiler. Limited to verifiable domains, and test-exploitation becomes the new failure mode. (§16.6)

**74. Explain speculative decoding and why it's lossless.** ★
A small draft model proposes $\gamma$ tokens; the target scores all of them in one forward pass (cheap, since decode is memory-bound). Accept token $i$ with probability $\min(1, p_i/q_i)$; on rejection, resample from the normalized residual $\max(0,p-q)$. That modified rejection sampling makes the output distribution **exactly** $p$. Expected tokens per round $=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$; typical 2–3× speedup. (§17.3)

**75. Why is LLM decoding memory-bandwidth-bound?** ★
At batch 1, producing one token requires reading every parameter. A 70B bf16 model is 140 GB; at 3.35 TB/s that's 42 ms/token = 24 tok/s regardless of compute. Consequence: batching is nearly free, and quantization buys near-linear speedup. (§17.1)

**76. Why is quantizing activations harder than weights?**
Transformers develop systematic outlier channels with magnitudes 10–100× the rest, consistently in the same dimensions. A per-tensor scale set by them crushes everything else. Fixes: LLM.int8() (keep outliers in fp16), SmoothQuant (migrate difficulty to weights), AWQ (protect high-activation channels). (§17.4)

**77. Explain LoRA. What rank and which layers?**
$W'=W_0+\frac{\alpha}{r}BA$ with $B=0$ at init, so training starts exactly at the pretrained model. Apply to **all** linear layers (not just Q,V) — that matters more than the rank. $r=8$–64; higher for new knowledge, lower for style. Merges into $W$ at inference for zero added latency. (§17.7)

**78. Why the $\tau^2$ in distillation loss?**
The gradient of the soft-target term scales as $1/\tau^2$, so multiplying by $\tau^2$ keeps the relative weight of soft and hard targets constant as $\tau$ varies. High $\tau$ matters because a confident teacher at $\tau=1$ carries no more information than the label — the "dark knowledge" is in the ratios among small probabilities. (§17.6)

**79. Design a RAG system and tell me its failure modes.**
Chunk structurally (index small, expand to parent), embed with an instruction-prefixed model, hybrid BM25 + dense retrieval fused by **RRF** ($\sum 1/(60+\text{rank})$), cross-encoder rerank top-100 → top-10, place the best material at the start and end of the context, instruct explicit citation and "I don't know." Failures: document never retrieved (chunking/vocabulary — add BM25), retrieved but ignored (buried in the middle), model overriding context with parametric knowledge, multi-hop (needs decomposition), aggregation queries (wrong tool — use SQL). **Evaluate retrieval and generation separately.** (§18.3–18.4)

**80. Long context or RAG?**
Both. Retrieval reduces the haystack; long context reasons over what's left. Feeding 1M tokens when 5k suffice costs 200× more for a quality *decrease* ("lost in the middle"). (§18.5)

**81. Why do agents fail?** ★
Compounding error: $p^n$. At 95% per-step reliability, 20 steps gives 36%. Improving per-step reliability from 95% to 99% takes you to 82% — worth more than any architectural change. Plus context rot, looping, and prompt injection from tool outputs. (§18.7)

---

## 34.7 Generative models

**82. Compare VAE, GAN, flow, and diffusion.**
VAE: bounded likelihood, fast, blurry, stable. GAN: no likelihood, fast, sharp, mode-collapse-prone, unstable. Flow: exact likelihood, fast, architecturally constrained (invertibility, no dimensionality reduction). Diffusion: bounded likelihood, high quality, mode-covering, stable, slow — and the slowness was fixable by distillation while GAN instability was not. (§19.8)

**83. Why are VAE samples blurry?** ★
(a) A Gaussian likelihood makes reconstruction an MSE, and the MSE-optimal prediction under uncertainty is the *mean* — an average of plausible images. (b) Maximum likelihood minimizes forward KL, which is mode-covering. (c) The ELBO is a loose bound. (§19.3)

**84. Explain the reparameterization trick and why it's necessary.**
Write $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$, so the expectation is over a fixed distribution and the gradient passes inside. The alternative (REINFORCE/score-function) is unbiased but has orders of magnitude higher variance. (§19.3)

**85. What is posterior collapse and how do you fix it?**
If the decoder is powerful enough to model $x$ alone, the optimum sets $q(z|x)=p(z)$, KL$=0$, and the latent is unused. Fixes: KL annealing, free bits (don't penalize KL below a floor per dimension), weaker decoder. (§19.3)

**86. Derive the GAN's optimal discriminator and what the generator then minimizes.** ★
Pointwise maximization of $a\log y+b\log(1-y)$ gives $D^*=\frac{p_{\text{data}}}{p_{\text{data}}+p_g}$. Substituting back yields $-\log 4+2\,\mathrm{JSD}(p_{\text{data}}\|p_g)$, minimized when $p_g=p_{\text{data}}$. **But** when the supports are disjoint — essentially always in high dimensions — JSD is constant at $\log 2$ and its gradient is zero. That's the fundamental instability, and it's why WGAN replaces JSD with the Earth-Mover distance. (§19.4)

**87. Explain the VQ-VAE straight-through estimator.**
The nearest-neighbour codebook lookup has no gradient, so copy the decoder's gradient directly to the encoder as if quantization were the identity. Biased, and it works. The same trick appears in quantization-aware training and any discrete bottleneck. (§19.3, §17.4)

**88. Derive the diffusion forward-process closed form.** ★
By induction: $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon''$; substituting $x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}}\epsilon'$ and combining the two independent Gaussians gives variance $\alpha_t(1-\bar\alpha_{t-1})+1-\alpha_t=1-\bar\alpha_t$. So $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$. **This is what makes training one-shot** — you jump to any noise level without simulating the chain. (§20.2)

**89. What is the actual diffusion training loss?**
$\mathcal{L}_{\text{simple}}=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2$ — plain MSE denoising. It's the VLB with the weighting dropped, which is a worse likelihood objective and a better perceptual one. (§20.4)

**90. Why predict $\epsilon$ rather than $x_0$? What is $v$-prediction?**
At large $t$, $x_t$ is nearly noise so $\epsilon$ is well-scaled while $x_0$ is nearly unconstrained (high-variance target). At small $t$, the reverse. $v=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}x_0$ interpolates smoothly and is preferred for distillation and high resolution. (§20.4)

**91. Show that diffusion and score matching are the same.** ★
$\nabla_{x_t}\log q(x_t|x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$. So $s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$. DDPM and NCSN were developed independently and are the same algorithm. Tweedie's formula adds the third equivalence: the optimal denoiser, the score, and the posterior mean are one object. (§20.6)

**92. Explain classifier-free guidance.** ★
Train one network with the condition randomly dropped ~10% of the time, then extrapolate at sampling: $\tilde\epsilon=\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)$. Derivable from classifier guidance by substituting the implicit classifier $\nabla\log p(y|x)=\nabla\log p(x|y)-\nabla\log p(x)$. $w>1$ pushes beyond the conditional distribution: higher prompt adherence and fidelity, **lower diversity**, and saturation artifacts at large $w$. (§20.8)

**93. What is DDIM and why does it allow 50 steps instead of 1000?**
A non-Markovian family with the same marginals and the same trained network; $\sigma_t=0$ gives a deterministic first-order discretization of the probability-flow ODE, so you can skip timesteps and apply standard ODE solvers. (§20.7)

**94. What is flow matching and why is it replacing DDPM?** ★
Define $x_t=(1-t)x_{\text{noise}}+t\,x_{\text{data}}$ and regress the velocity $v_\theta(x_t,t)$ onto $x_{\text{data}}-x_{\text{noise}}$; sample by integrating the ODE. Straight paths mean far less discretization error, so 10–20 steps without distillation, and there's no noise schedule to tune. Conceptually the same family as diffusion, with a better-chosen path. (§20.10)

**95. Why latent diffusion?**
Run diffusion in a pretrained autoencoder's latent space: $512^2\times3\to64^2\times4$ is a 48× reduction in elements. The autoencoder handles imperceptible high-frequency detail; the diffusion model handles semantics. The autoencoder is also a hard ceiling — lost fine text and small faces cannot be recovered. (§20.11)

**96. How does discrete diffusion work?**
Replace Gaussian noise with a categorical transition matrix $Q_t$; the cumulative $\bar Q_t=\prod Q_s$ plays the role of $\bar\alpha_t$, and the posterior $q(x_{t-1}|x_t,x_0)$ is exact and closed-form. The absorbing/masking kernel dominates, and with it the objective reduces to a weighted average of masked-LM losses — **BERT with a continuously varying mask rate.** (§21.2–21.3)

**97. What is AdaLN-Zero and why does it matter?** ★
Adaptive LayerNorm plus a per-block gate $\alpha(c)$ on the residual branch, with $\alpha$'s producing MLP zero-initialized — so every block starts as the exact identity. Same principle as zero-init residual, Fixup, and ReZero. It beat cross-attention and in-context conditioning by a wide margin at every scale in the DiT ablations. (§21.4)

---

## 34.8 Classical ML

**98. Why does ridge shrink some directions more than others?** ★
In the SVD basis, $\hat y=\sum_j u_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}u_j^\top y$. Directions with small $\sigma_j$ — poorly determined by the data, hence noisily estimated — are shrunk most. Exactly the right behaviour. (§22.2)

**99. Why does LASSO produce exact zeros and ridge doesn't?**
The subgradient of $\lambda|\beta_j|$ at 0 is the interval $[-\lambda,\lambda]$, so $\beta_j=0$ is optimal whenever $|x_j^\top r|\le\lambda$. Ridge's penalty gradient $2\lambda\beta_j\to0$ as $\beta_j\to0$, so it never pins. Geometrically: the $\ell_1$ ball has corners on the axes. (§22.3)

**100. When do you use elastic net?**
Correlated predictors. LASSO arbitrarily selects one of a correlated group and zeros the rest, which is unstable across resamples; the ridge component induces a grouping effect. (§22.3)

**101. Derive the SVM dual and explain support vectors.** ★
Lagrangian, stationarity gives $w=\sum_i\alpha_iy_ix_i$ and $\sum_i\alpha_iy_i=0$; substituting gives $\max_\alpha\sum\alpha_i-\frac12\sum\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle$. Data enters only through inner products (→ kernels), and KKT complementary slackness forces $\alpha_i=0$ for every point off the margin — exact sparsity. (§22.5)

**102. State the representer theorem and why it matters.**
For $\min_f\sum_iL(y_i,f(x_i))+\Omega(\|f\|_\mathcal{H})$ with $\Omega$ strictly increasing, the minimizer is $f^*=\sum_i\alpha_ik(x_i,\cdot)$. Proof: decompose $f=f_\parallel+f_\perp$; the reproducing property means $f_\perp$ doesn't affect the loss but strictly increases the norm. It's why an infinite-dimensional optimization reduces to $n$ coefficients. (§22.5)

**103. Why does the RBF kernel's infinite dimensionality not cause overfitting?**
Capacity is controlled by norm, not dimension. The margin/norm-based bounds of Chapter 2 §2.5 don't involve $d$. (§22.5)

**104. Why not use misclassification error as a tree splitting criterion?** ★
It's piecewise linear in $p$, so a split that increases purity without changing the argmax gives zero gain. Gini and entropy are strictly concave, so any purity-increasing split registers. Concavity is what makes greedy search work. (§23.1)

**105. Why do random forests use random feature subsets?** ★
Ensemble variance is $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$. The second term vanishes with more trees; the first does not. So the only way to improve a large ensemble is to reduce the correlation $\rho$ — and random features stop every tree from choosing the same dominant split at the root. (§23.2)

**106. Bagging vs boosting.**
Bagging averages many low-bias, high-variance deep trees to reduce **variance**. Boosting sums many high-bias, low-variance shallow trees to reduce **bias**. Opposite terms of the same decomposition, which is why the tree depths differ. (§23.3)

**107. Show AdaBoost minimizes exponential loss.** ★
Forward stagewise: $\min_{\alpha,G}\sum_iw_i e^{-\alpha y_iG(x_i)}$ where $w_i=e^{-y_if_{m-1}(x_i)}$ — AdaBoost's weight, derived rather than posited. Splitting by correct/incorrect gives $(e^\alpha-e^{-\alpha})\mathrm{err}+e^{-\alpha}$; minimizing over $G$ is minimizing weighted error, and $d/d\alpha=0$ gives $\alpha=\frac12\log\frac{1-\mathrm{err}}{\mathrm{err}}$. (§23.3)

**108. Derive XGBoost's leaf weight and split gain.** ★★
Second-order expansion, grouped by leaf: $\tilde{\mathcal{L}}=\sum_j[G_jw_j+\frac12(H_j+\lambda)w_j^2]+\gamma T$. Each leaf is an independent quadratic, so $w_j^*=-\frac{G_j}{H_j+\lambda}$ and $\tilde{\mathcal{L}}^*=-\frac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T$. Hence $\mathrm{Gain}=\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma$: don't split if negative. (§23.5)

**109. What does LightGBM do differently?**
Histogram binning (255 bins, $O(\#\text{bins})$ split search), leaf-wise rather than level-wise growth (control with `num_leaves`), GOSS (keep large-gradient instances, subsample the rest with a reweighting that keeps the gain estimate unbiased), and EFB (bundle mutually-exclusive sparse features). (§23.6)

**110. How does CatBoost avoid target leakage?** ★
Ordered target statistics: encode a categorical value using only the examples *earlier* in a random permutation. Same idea as out-of-fold target encoding, applied online. Ordered boosting extends it to the residuals themselves. (§23.6)

**111. Derive EM for a GMM and show it monotonically increases the likelihood.** ★
E-step: $\gamma_{ik}=\frac{\pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_i|\mu_j,\Sigma_j)}$. M-step: $\pi_k=N_k/n$, $\mu_k=\frac1{N_k}\sum_i\gamma_{ik}x_i$, $\Sigma_k=\frac1{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$. Monotonicity: the E-step raises the Jensen bound to *equality* with the log-likelihood; the M-step then increases the bound; the log-likelihood is $\ge$ the bound, so it increased too. (§24.4)

**112. What's the relationship between k-means and GMM?**
k-means is GMM with isotropic equal-weight shared covariance $\sigma^2I$ in the limit $\sigma^2\to0$, where soft responsibilities become hard assignments. (§24.4)

**113. Why is the GMM likelihood unbounded?**
A component collapsing onto a single point has $|\Sigma_k|\to0$ and likelihood $\to\infty$. Fix with a covariance floor $\epsilon I$ (a MAP prior) or by restarting collapsed components. (§24.4)

**114. Give three derivations of PCA.**
Maximum projected variance (top eigenvector of $\Sigma$), minimum reconstruction error (equivalent, since $\|x\|^2=\|Px\|^2+\|x-Px\|^2$), and the SVD of the centred data matrix. Always compute via SVD, never by forming $X^\top X$. (§24.1)

**115. Why does t-SNE use a Student-$t$ in the low-dimensional space?** ★
The crowding problem: ball volume grows as $r^d$, so a point in 50-D has far more roughly-equidistant neighbours than can be placed at similar distance in 2-D. A Gaussian would over-penalize moderate distances and collapse everything; the heavy tail lets moderately-dissimilar points sit far apart cheaply. (§24.3)

**116. What can you not read off a t-SNE plot?**
Cluster sizes (density is equalized), distances between clusters, and — at low perplexity — the existence of clusters at all (it will cluster pure noise). Also: never cluster on t-SNE coordinates and report the clusters as findings. (§24.3)

**117. How do you handle class imbalance?**
In order: use the right metric (PR-AUC, not accuracy or ROC-AUC); class weights; **threshold tuning on validation** — often the entire solution and frequently skipped; then resampling/SMOTE; focal loss for extreme dense-prediction imbalance. **Always resample inside the CV loop, never before splitting.** (§22.6)

**118. Explain the curse of dimensionality concretely.**
To capture a fraction $r$ of the volume in $d$ dimensions, a hypercube neighbourhood needs edge length $r^{1/d}$: for $d=10$, $r=0.01$, that's 63% of every feature's range. "Local" isn't local. And $\frac{d_{\max}-d_{\min}}{d_{\min}}\to0$, so all points become equidistant. (§22.6)

---

## 34.9 Reinforcement learning

**119. Prove the Bellman operator is a contraction.** ★
$\|\mathcal{T}V-\mathcal{T}W\|_\infty\le\max_{s,a}\gamma\sum_{s'}P(s'|s,a)|V(s')-W(s')|\le\gamma\|V-W\|_\infty$, using $|\max_af-\max_ag|\le\max_a|f-g|$ and $\sum P=1$. Banach then gives a unique fixed point and geometric convergence at rate $\gamma^k$. (§26.4)

**120. Why does $\gamma$ exist?**
Convergence of the infinite sum, equivalence to a per-step termination probability, and variance reduction. Effective horizon $\frac{1}{1-\gamma}$. Higher $\gamma$ is more farsighted and provably slower to converge. (§26.2)

**121. SARSA vs Q-learning — explain with Cliff Walking.** ★
SARSA's target uses the action actually taken next, so it learns the value of the $\epsilon$-greedy policy *including* exploration and picks a safe path away from the cliff. Q-learning's $\max$ learns the greedy policy's value and takes the optimal cliff-edge path — better asymptotic policy, worse online return because it occasionally explores off the edge. (§26.6)

**122. What is maximization bias and how is it fixed?**
$\mathbb{E}[\max_a\hat Q]\ge\max_a\mathbb{E}[\hat Q]$ by Jensen — the max of noisy estimates is too large, and Q-learning uses the same values to select and evaluate. Double Q-learning decouples: one estimator picks the action, the other scores it. Double DQN applies this with the online and target networks. (§26.6, §27.3)

**123. What is the deadly triad?** ★
Function approximation + bootstrapping + off-policy training. Any two are safe; all three can diverge (Baird's counterexample). Every stabilizer in deep RL — target networks, replay, trust regions, conservative updates — is a partial countermeasure. (§26.7)

**124. Why is a target network necessary in DQN?**
Without it the regression target depends on the parameters being updated, so raising $Q(s,a)$ raises the target for $Q(s',a')$ — a positive feedback loop. Freezing the target turns RL back into a sequence of ordinary supervised regressions. (§27.2)

**125. Derive the policy gradient theorem.** ★
Log-derivative trick: $\nabla_\theta\mathbb{E}[R]=\mathbb{E}[R\nabla_\theta\log p_\theta(\tau)]$. Expanding $\log p_\theta(\tau)$, the transition dynamics and initial distribution don't depend on $\theta$ and vanish, leaving $\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\Psi_t]$. **The environment dropping out is what makes it model-free.** (§27.4)

**126. Prove a baseline doesn't bias the policy gradient.**
$\mathbb{E}_a[\nabla\log\pi_\theta(a|s)b(s)]=b(s)\nabla\sum_a\pi_\theta(a|s)=b(s)\nabla 1=0$. Any state-dependent baseline is free; the variance-minimizing one is near $V^\pi(s)$, giving the advantage. (§27.4)

**127. Derive GAE and explain $\lambda$ vs $\gamma$.**
$\hat A^{\mathrm{GAE}}_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ — an exponentially weighted average of all $k$-step advantage estimators. $\lambda=0$ gives $\delta_t$ (max bias, min variance); $\lambda=1$ gives Monte Carlo (unbiased, max variance). **$\gamma$ defines the objective; $\lambda$ is purely an estimator knob.** (§27.6)

**128. Explain PPO's clipped objective, including the sign asymmetry.** ★
$\mathcal{L}=\mathbb{E}[\min(\rho\hat A, \mathrm{clip}(\rho,1\pm\epsilon)\hat A)]$. For $\hat A>0$ the $\min$ caps the reward at $\rho=1+\epsilon$, removing incentive to move further. For $\hat A<0$ the $\min$ picks the more negative term, so there's no ceiling on pushing a bad action down until $\rho<1-\epsilon$. The $\min$ makes it a pessimistic lower bound on the surrogate — which is what makes multiple epochs on the same batch safe. (§27.7)

**129. What is SAC's objective and how does it relate to RLHF?** ★
Maximize reward plus policy entropy. The optimal policy is Boltzmann: $\pi^*\propto\exp(Q/\alpha)$ — **exactly the KL-regularized optimum from DPO's derivation with a uniform reference policy.** SAC's entropy bonus and RLHF's KL-to-reference are the same mathematical object. (§27.8, §16.5)

**130. What is the core problem in offline RL?**
Distributional shift on **actions**: the policy proposes actions absent from the data, $Q$ is unconstrained there, and the $\max$ systematically overestimates them — with no environment feedback to correct it. More gradient steps make it worse. Fixes: policy constraints (TD3+BC), conservative values (CQL), in-sample learning (IQL), or sequence modelling (Decision Transformer, which loses trajectory stitching). (§27.10)

**131. State the reward-shaping theorem.**
Adding $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ leaves the optimal policy unchanged for any $\Phi$. Any non-potential-based shaping can change the optimum, usually badly. (§26.9)

**132. Why is deep RL hard to reproduce?**
Results vary enormously across seeds; single-curve reporting is meaningless. Always report ≥5 seeds with a distribution. This is Chapter 3's lesson, violated more in RL than anywhere else. (§27.11)

---

## 34.10 Vision, graphs, and the science of DL

**133. Two 3×3 convs vs one 5×5?**
Same receptive field, $18C^2$ vs $25C^2$ parameters, and two nonlinearities instead of one. The whole VGG argument. (§8.6)

**134. What does a 1×1 convolution do?**
Channel dimensionality reduction, cross-channel mixing, and cheap added nonlinearity. It's a per-position dense layer. (§8.1)

**135. Why is the effective receptive field smaller than the theoretical one?**
Paths from the centre to the output vastly outnumber paths from the edge, so influence is a random-walk convolution — approximately Gaussian with radius $\propto\sqrt L$ rather than $L$. Assume roughly the square root of the formula. (§8.1)

**136. What causes checkerboard artifacts?**
Transposed convolution with a stride that doesn't divide the kernel size, giving uneven overlap. Fix: nearest/bilinear upsample followed by a normal conv. (§8.1)

**137. Why did ViT lose on ImageNet-1k and win on JFT-300M?** ★
A CNN's locality/translation prior is correct but restrictive. When data is scarce, a correct prior substitutes for data; when data is abundant, it becomes a ceiling. Probing shows large-data ViTs *rediscover* convolution-like early layers. (§28.1)

**138. Explain CLIP's objective and one known weakness.**
Symmetric InfoNCE over a batch: an $N$-way "which caption belongs to this image" classification in both directions, with a learned temperature. Weakness: it behaves substantially like a bag of concepts — "a horse riding an astronaut" embeds near "an astronaut riding a horse" (ARO benchmark). Also weak at counting, spatial relations, and negation. (§28.2)

**139. Why SigLIP over CLIP?**
The softmax needs a global normalization over the batch, forcing an expensive all-gather. A pairwise sigmoid loss is decomposable per pair, so it works at small batch sizes and scales without the communication cost. (§28.2)

**140. What limits GNN depth?** ★
Over-smoothing: repeated neighbourhood averaging is diffusion; $\hat A^L$ converges to a rank-one projection, so all node representations collapse, exponentially in $L$. Over-squashing pushes the other way: an exponentially large receptive field compressed into a fixed vector, bounded by the graph's spectral gap. Hence 2–4 layers, and hence graph transformers. (§29.5)

**141. State the WL expressivity bound.** ★
Message-passing GNNs are at most as powerful as the 1-WL test, because a layer computes a function of (node feature, multiset of neighbour features) — structurally a WL refinement round. Consequence: they cannot count triangles, detect cycles, or distinguish two 3-regular graphs of the same size. GIN reaches the bound by using **sum** aggregation (injective on multisets — mean and max are not) plus an MLP. (§29.4)

**142. Why does equivariance help?**
It's a hard constraint, so the model never spends capacity learning that physics is rotation-invariant, and it generalizes perfectly to unseen orientations. ~10× data efficiency on molecular property prediction. (§29.8)

**143. Explain double descent.** ★
Test error peaks at the interpolation threshold ($p\approx n$) because the design matrix is generically near-singular there (Marchenko–Pastur), so the minimum-norm interpolant's norm blows up. Past it, the solution space grows and the minimum-norm element gets smaller and smoother. Model-wise, epoch-wise, and sample-wise variants all exist — and more data can hurt. Explicit regularization removes the peak. (§30.1)

**144. What is the NTK and what does it fail to explain?** ★
At infinite width, the tangent kernel $\Theta(x,x')=\langle\nabla_\theta f(x),\nabla_\theta f(x')\rangle$ is deterministic and constant during training, so the network is exactly kernel regression with a fixed feature map. Explains convergence to global minima and spectral bias (smooth functions fit first). Fails because features never change — so it cannot explain transfer learning or representation learning, and finite networks beat their NTK. Real networks are deliberately kept in the feature-learning regime, which is what $\mu$P preserves. (§30.2)

**145. Explain grokking.** ★
Delayed generalization: 100% train accuracy at step $10^3$, test accuracy jumps at $10^5$. A memorizing circuit is found first; a generalizing one has smaller weight norm. Once training loss is ~0, the CE gradient vanishes and **weight decay becomes the dominant force**, slowly drifting the model along the zero-loss manifold to the minimum-norm (generalizing) solution. For modular addition the circuit was fully reverse-engineered as a discrete Fourier transform. Continuous progress measures show smooth formation — the discontinuity is in the metric. (§30.3)

**146. What is neural collapse?** ★
In the terminal phase of training: within-class variability →0 (NC1), class means form a simplex ETF with pairwise cosine $-\frac{1}{C-1}$ (NC2), classifier weights align with class means (NC3), and classification reduces to nearest class centre (NC4). It's the global optimum of the unconstrained-features problem. NC1 destroys within-class information, which is a quantitative account of why over-training a classifier gives worse transfer features.  (§31.1)

**147. What is the implicit bias of gradient descent?** ★
On separable data with logistic loss, $w(t)/\|w(t)\|$ converges to the max-margin SVM direction — at rate $O(1/\log t)$, which is why training long past zero error still helps. On least squares from zero init it converges to the minimum $\ell_2$-norm interpolant. Adam's coordinate-wise normalization gives a different ($\ell_\infty$-geometry / $\ell_1$-margin) bias, which is the real explanation for the Adam generalization gap. (§31.2)

**148. What's wrong with "flat minima generalize better"?**
Sharpness isn't reparameterization-invariant: $(W_1,W_2)\to(\alpha W_1,\alpha^{-1}W_2)$ leaves a ReLU network's function unchanged while scaling Hessian eigenvalues by $\alpha^{\pm2}$ (Dinh et al.). So any minimum can be made arbitrarily sharp. Responses: scale-invariant sharpness measures, the observation that such reparameterizations aren't visited in practice, and SAM's empirical success. (§31.3)

**149. State the Lottery Ticket Hypothesis and the correction to it.** ★
A dense random network contains a sparse subnetwork that, trained *from the original initialization*, matches full accuracy. Found by iterative magnitude pruning with weight **resetting** — reinitializing the same mask randomly fails, so the ticket is (mask, init). The correction: at ImageNet scale you must **rewind** to $\theta_k$ for small $k$ rather than $\theta_0$ — the ticket forms in the first few hundred steps, at the point of linear mode connectivity. (§31.4)

**150. Why does model merging work?** ★
Models fine-tuned from a shared base stay in the same loss basin, so task vectors $\tau=\theta_{\text{FT}}-\theta_{\text{pre}}$ add meaningfully (and negate to remove behaviours). Models from *different* seeds are in the same basin only up to **permutation symmetry** of hidden units — align the units first (Git Re-Basin) and the interpolation barrier largely disappears. (§17.8, §31.5)

**151. Explain superposition.** ★
A model represents more features than it has dimensions by using nearly-orthogonal directions, tolerating interference because features are sparse and rarely co-active — and a ReLU suppresses the small crosstalk. JL guarantees exponentially many such directions exist. Consequence: individual neurons are polysemantic, so "what does neuron 1432 do" is the wrong question. In the toy model, sparsity drives a phase transition from orthogonal representation to antipodal pairs to tetrahedra and other sphere packings. (§32.2)

**152. What is a sparse autoencoder and what are its problems?**
An overcomplete ($m\approx 8$–256$\times d$) autoencoder with an $\ell_1$ penalty, trained to reconstruct activations from a sparse code. Problems: $\ell_1$ shrinkage biases all magnitudes toward zero (fixed by TopK/JumpReLU), dead features, **feature splitting with no canonical granularity**, and circular evaluation. Strongest evidence for their validity is causal: clamping a feature steers behaviour. (§32.3)

**153. What are QK and OV circuits?**
An attention head factorizes into $W_{QK}=W_Q^\top W_K$, which decides *where* to read, and $W_{OV}=W_OW_V$, which decides *what* to write. Independently analyzable low-rank operations in the residual-stream basis. (§32.4)

**154. Why is activation patching better than a saliency map?**
It's causal. Patching an activation from a clean run into a corrupted one and measuring the output change tests whether the component carries the relevant information. Denoising finds sufficient components, noising finds necessary ones, and path patching isolates specific edges. Note self-repair means ablation *understates* importance. (§32.5)

**155. Explain why deep generative models assign higher likelihood to OOD data.** ★
Likelihood in high dimensions is dominated by low-level statistics, not semantics: SVHN is smoother than CIFAR-10, so its pixels are easier to predict, so a CIFAR-trained Glow gives it higher likelihood. **Likelihood is not typicality** — a high-likelihood point can lie far outside the typical set. Fixes use likelihood ratios against a background model. (§33.6)

**156. Why are modern networks miscalibrated, and how do you fix it?** ★
Capacity plus reduced regularization: the model drives training NLL toward zero, pushing probabilities to 1 long after accuracy saturates. Fix with **temperature scaling** — one scalar fitted on validation NLL, which cannot change accuracy and often reduces ECE 10×. Caveat: it calibrates in-distribution only. (§33.1–33.2)

**157. Explain conformal prediction and prove the coverage guarantee.** ★★
Split off a calibration set, compute nonconformity scores, take the $\lceil(n+1)(1-\alpha)\rceil/n$ quantile $\hat q$, and predict $\{y: s(x,y)\le\hat q\}$. Proof: under exchangeability the new score's rank among $n+1$ scores is uniform, so $P(s_{\text{new}}\le s_{(\lceil(n+1)(1-\alpha)\rceil)})\ge1-\alpha$. Requires nothing of the model. **But coverage is marginal, not conditional** — 90% overall can hide 40% on a subgroup; use group-conditional calibration. (§33.5)

**158. Aleatoric vs epistemic uncertainty, and why do ensembles beat MC dropout?**
Aleatoric is irreducible data noise; epistemic is model uncertainty, reducible with data, and equals the *disagreement* among plausible models. Ensembles beat MC dropout and variational methods because different initializations land in genuinely different loss basins, giving functional diversity; single-mode methods only explore one basin's neighbourhood. (§33.3–33.4)

**159. Why do adversarial examples exist?** ★
Locally, $w^\top\delta$ with $\delta=\epsilon\,\mathrm{sign}(w)$ gives a change of $\epsilon\|w\|_1$, which grows with dimension — many tiny coordinated changes sum to a large logit change. Deeper: Ilyas et al. showed they arise from **non-robust features that are genuinely predictive** — a dataset labelled only by non-robust features yields good clean test accuracy. Vulnerability is a property of the data the model faithfully learned, not a bug.  (§33.8)

**160. How do you evaluate an adversarial defence honestly?**
AutoAttack plus adaptive attacks designed against your specific defence. Watch for gradient masking: black-box beating white-box, unbounded attacks failing to reach 0%, or one-step beating iterative. Most published defences were broken this way. (§33.8)

**161. How do you fix a spurious correlation?**
Ideally break it with data or counterfactual augmentation. Otherwise GroupDRO (needs group labels and strong regularization) or JTT (upweight the first model's errors — no group labels). But first try the cheapest fix: **retrain only the last layer on a small group-balanced set** — Kirichenko et al. showed the representation usually already contains the core feature and only the classifier relies on the shortcut. (§33.9)

---

## 34.11 The ten answers that most separate candidates

If you internalize only ten things, make it these.

1. **Why VC bounds are vacuous for deep nets, and what replaces them.** The bound is on what the class *could* do; deep nets fit random labels, so any capacity-only bound must be vacuous. The explanation is the optimizer's implicit bias. (Q12, §2.7, §31.2)

2. **The $\eta\lambda$ coupling in normalized networks.** Normalization makes the layer scale-invariant, the gradient orthogonal to $W$, and the effective LR $\eta/\|W\|^2$. Weight decay is a learning-rate control, and only the product matters. (Q38, §7.4)

3. **The full DPO derivation.** Gibbs optimum → invert → $Z(x)$ cancels in the Bradley–Terry difference. (Q69, §16.5)

4. **$C=6ND$ and Chinchilla, derived rather than recited** — including why nobody follows it (inference cost). (Q48, Q63, Q64)

5. **Why decode is memory-bound**, and the three consequences: batching is nearly free, quantization is near-linear, and the KV cache is the real constraint. (Q75, Q49, Q53)

6. **Speculative decoding is lossless**, with the modified-rejection-sampling reason. (Q74)

7. **XGBoost's gain formula, derived.** Second-order expansion → $w^*=-G/(H+\lambda)$ → gain with a $\gamma$ per-leaf cost, which is principled pruning. (Q108)

8. **The variance formula $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$** and the fact that it *is* the design rationale for random forests. (Q105)

9. **Why the emergence debate has two correct sides.** Underlying capability improves smoothly; discontinuous metrics manufacture cliffs; mechanistic transitions are nonetheless real. (Q65, §15.4, §30.3)

10. **Superposition and what it implies about interpretability.** Sparse features in nearly-orthogonal directions → polysemantic neurons → the neuron basis is the wrong basis → sparse autoencoders, validated causally. (Q151)

---

## 34.12 The questions to ask them

Interviews are two-way, and these signal seniority:

- How do you decide a metric improvement is real? Do you report confidence intervals?
- What's your evaluation set, and how do you guard against contamination?
- Where does the team sit between research and production, and how does work move between them?
- What was the last experiment that failed, and what did it change?
- How much of the work is data versus modelling?
- How do you handle reproducibility — seeds, environment, data versioning?

---

## Check for Understanding

**The questions that separate candidates are almost never "what is X" — they are "derive X," "why is X true," and "when does X fail." If you can produce the derivation, name the failure mode, and give a number, the definition takes care of itself.**

---

**Next:** [Verification Notes](VERIFICATION.md)
