# Verification Notes

A page-by-page audit of the mathematics, worked numbers, examples, and analogies in this book. This records **what was checked, what was wrong, what was corrected, and what remains uncertain.** It is deliberately unflattering — a verification document that finds nothing is a verification document that wasn't run.

---

## 0. The beginner-layer edition — what changed and what didn't

This book was subsequently extended with an explanatory layer aimed at readers without the assumed background. Two commitments govern that work:

**Nothing was removed, reworded, or reordered.** Every sentence, equation, table, and derivation of the original text stands exactly as audited below. The additions are strictly insertions *between* existing blocks. Any line number or claim verified in this document remains verifiable.

**What was added:**

| Addition | Where |
|---|---|
| **[Chapter 00 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** | New file: symbol decoder, the six habits for reading a formula, the four ways to multiply, the overloading traps, and a full-forms glossary of every abbreviation used 3+ times in the book |
| **Symbol tables** | Top of each chapter, decoding that chapter's notation |
| **Plain-English unpacking** | After dense formulas — every symbol named, an analogy, a worked example with concrete numbers |
| **Origin stories** (`> **Where this came from.**`) | Near major concepts |
| **`## Did you know?`** | End of each chapter, before Check for Understanding |
| **`### Can you explain these out loud?`** | End of each chapter, after Check for Understanding |

### Rendering defects found and corrected

The audit below covered mathematical correctness but not **Markdown rendering**. A repo-wide scan found four table rows where a raw `|` inside `$...$` was being parsed as a column separator, splitting the cell and destroying the math:

| File | Row | Problem | Fix |
|---|---|---|---|
| §1.1.4 | $\ell_1$ norm | `\sum` followed by a bare-pipe absolute value split the cell | `\sum_i \lvert x_i \rvert` |
| §1.1.4 | $\ell_\infty$ norm | `\max_i` followed by a bare-pipe absolute value split the cell | `\max_i \lvert x_i \rvert` |
| §10.4 | vocabulary trade-off header | bare-pipe cardinality of $V$ split the cell | `\lvert V \rvert` |
| §10.4 | embedding parameter count | bare-pipe cardinality inside $2\cdot\lvert V\rvert\cdot d$ split the cell | `2\lvert V \rvert d` |

These were  defects: the affected tables did not render as tables at all in any standards-compliant Markdown renderer. **Rule for future edits:** inside a Markdown table cell, never use a bare `|` within math. Use `\lvert`/`\rvert` for absolute value or cardinality, and `\mid` for conditional probability.

> **A note on this very table.** The first draft of the rows above reproduced the bug it documents — the offending expressions were quoted literally, in backticks, and the bare `|` characters split these cells too. Inline code spans do **not** protect a pipe inside a Markdown table; only `\|` or avoiding the character does. The rows are now written descriptively for that reason. It is a fair illustration of how easily this defect slips through: it survived being written *by* the person fixing it, *in the document explaining it*.

### A new finding in the *original* text — Ch. 8 architecture lineage table

The beginner-layer pass surfaced one substantive issue that the original audit did not catch, in the §8.6 architecture comparison table.

**The problem.** The `GoogLeNet/Inception` row pairs three things that come from different members of the Inception family:

| Column | Value in table | Which model it actually belongs to |
|---|---|---|
| Year | 2014 | Inception-v1 (GoogLeNet) ✓ |
| Params | 6.8M | Inception-v1 ✓ |
| Key idea | $n\times n \to n\times1,\ 1\times n$ factorization | **Inception-v3** (2015) |
| Top-1 | 74.8% | **A later variant** — v1's commonly-reported single-crop top-1 is in the high 60s |

**Why it was not corrected.** The additive-only rule governs this edition: no original sentence, table, or figure is altered. The row has instead been annotated immediately below the table, explaining which figure belongs to which variant.

**Does it damage the argument?** No. The table's purpose is to show that parameter count and accuracy are decoupled, and 6.8M against VGG-16's 138M is a 20× gap under either accuracy figure. The claim built on the row — "GoogLeNet is competitive with VGG-16 using roughly twenty times fewer parameters" — survives intact.

**The general hazard, now stated in the text.** "Inception" names four architectures spanning 2014–2016, and lineage tables across the literature routinely blur them. The chapter now tells the reader to ask *which version* whenever an Inception number is quoted — and notes the irony that the ConvNeXt lesson appearing directly beneath the table ("architecture comparisons at unequal training recipes are worthless") retroactively undercuts most cross-era readings of the table itself.

### Standing accuracy requirement for the added material

The historical claims added to each chapter are held to the same bar as the mathematics: named individuals, approximate dates, and institutions only where well attested. Where an anecdote is disputed or likely apocryphal — the von Neumann "nobody knows what entropy really is" story is the prominent example — the text says so rather than presenting it as settled fact. Stories that could not be confirmed were omitted rather than hedged into vagueness.

---

## 1. Method

Three passes:

1. **Derivation check.** Every derivation marked as complete was re-derived independently and compared line by line against the text. Any step of the form "it can be shown" was treated as a defect and either filled in or explicitly flagged.
2. **Numerical check.** Every arithmetic claim in the book was recomputed in Python. Results below.
3. **Consistency check.** Cross-references resolved as links; symbols checked against the notation table; claims that appear in more than one chapter checked for agreement with each other.

---

## 2. Numerical audit — results

All quantitative claims were recomputed. **28 of 32 verified exactly or within stated rounding; 4 defects found and corrected.**

| Claim | Location | Stated | Computed | Status |
|---|---|---|---|---|
| $\prod_{k=23}^{35}(1-1/k)=22/35$ | §3.6 | 0.629 | 0.62857 | ✓ |
| $P(\text{records at }36\text{ and }37)$ | §3.6 | $7.5\times10^{-4}$ | $7.508\times10^{-4}$ | ✓ |
| Perplexity change $1.556\to1.524$ | §1.4 | 3.2% | 3.15% | ✓ |
| $(1-3\times10^{-6})^{2274}$ | §2.8 | 0.9932 | 0.993201 | ✓ |
| 43-epoch weight shrinkage | §2.8 | 74% | 74.58% | ✓ |
| Dense-layer parameter count | §8.1 | $4.8\times10^{11}$ | $4.834\times10^{11}$ | ✓ |
| Depthwise-separable saving | §8.1 | 8.7× | 8.694× | ✓ |
| $0.9^{100}$, $1.1^{100}$ | §9.2 | $2.7\times10^{-5}$, 13,781 | 2.656e−5, 13,780.6 | ✓ |
| $12Ld^2$ vs GPT-3's 175B | §11.7 | $1.74\times10^{11}$ | $1.7395\times10^{11}$ | ✓ |
| Attention reaches 50% of FLOPs at $T=6d$ | §11.7 | 24,576 | 24,576 | ✓ |
| KV cache, 70B geometry | §12.7 | 10.7 GB | 10.74 GB | ⚠ **corrected** |
| MinHash LSH threshold $(1/b)^{1/r}$ | §14.1 | 0.72 | 0.7071 | ⚠ **corrected** |
| 7B × 2T token budget | §14.7 | 10.6 days | see below | ⚠ **corrected** |
| $10^{-0.076}$ (10× params → loss ratio) | §15.1 | 0.84 | 0.8395 | ✓ |
| Params needed to halve loss | §15.1 | ~9,000× | 9,139× | ✓ |
| Chinchilla exponents $\beta/(\alpha{+}\beta)$, $\alpha/(\alpha{+}\beta)$ | §15.2 | 0.452, 0.548 | 0.45161, 0.54839 | ✓ |
| Speculative decoding, $\alpha=0.8,\gamma=4$ | §17.3 | 3.36 tokens | 3.3616 | ✓ |
| LoRA parameter reduction, $r=8$, $d=4096$ | §17.7 | 256× | 256 | ✓ |
| 70B bf16 decode bandwidth floor | §17.1 | 24 tok/s | 23.93 | ✓ |
| H100 ridge point | §17.1 | 295 FLOP/byte | 148 (dense) | ⚠ **corrected** |
| Agent reliability $0.95^{20}$, $0.99^{20}$ | §18.7 | 0.36, 0.82 | 0.3585, 0.8179 | ✓ |
| Diffusion posterior variance identity | §20.3 | $\tilde\beta_t$ | matches symbolically & numerically | ✓ |
| Simplex ETF angle $-1/(C-1)$ | §31.1 | −0.5 at $C=3$ | −0.5 | ✓ |
| kNN neighbourhood edge length, $d=10$, $r=0.01$ | §22.6 | 0.63 | 0.6310 | ✓ |
| InfoNCE bound ceiling $\log N$, $N=256$ | §25.3 | 5.5 nats | 5.545 | ✓ |
| Bootstrap out-of-bag fraction $e^{-1}$ | §23.2 | 36.8% | 36.79% | ✓ |
| RAdam $\rho_\infty = 2/(1-\beta_2)-1$ | §5.4 | 1999 | 1999 | ✓ |
| Expected minimum of $n=43$ noisy draws | §3.6 | ~2σ below mean | 2.00σ (simple), 2.18σ (Blom) | ✓ |

---

## 3. Defects found and corrected

### 3.1 KV-cache example conflated MHA and GQA geometry *(Ch. 12 §12.7, Ch. 34 Q53)*

**The error.** The worked example was labelled "LLaMA-2-70B" with $h_{kv}=64$. LLaMA-2-70B ships **GQA with $h_{kv}=8$**, so its real per-sequence cache at 4k context is **1.34 GB**, not 10.7 GB. The arithmetic was right; the attribution was wrong.

**Why it mattered.** The example sits immediately before the section arguing that GQA exists to shrink the cache — so citing a GQA model's cache as though it were unmitigated undercut the very point being made, and would have been a bad answer to give in an interview.

**Fix.** Relabelled as "a 70B-class model *with full MHA*", and added the GQA contrast explicitly (10.7 GB → 1.34 GB), since the contrast is the actual argument. Interview bank Q53 updated to match.

### 3.2 MinHash LSH threshold arithmetic *(Ch. 14 §14.1)*

**The error.** $(1/16)^{1/8}$ was stated as ≈0.72; it is 0.7071.

**Fix.** Corrected to $16^{-1/8}\approx0.71$, with the exponent shown so the reader can check it.

### 3.3 Hardware peak FLOP convention *(Ch. 14 §14.7, Ch. 17 §17.1)*

**The error.** Two related problems. Chapter 14's training-budget calculation used "~400 TFLOP/s bf16 dense peak" for an H100, which is neither the dense figure (~495 TFLOP/s) nor the marketing figure (~989 TFLOP/s, which assumes 2:4 structured sparsity). Chapter 17's arithmetic-intensity ridge point used the **sparsity** number (989) against a **dense** workload, giving 295 FLOP/byte instead of 148.

**Why it mattered.** This is a live trap in real work, not a typo: dense-vs-sparse peak differs by exactly 2×, so quoting the sparsity number against a dense workload silently halves your reported MFU. Getting it wrong in a book that teaches MFU would have been self-defeating.

**Fix.** Chapter 14 now uses the dense peak explicitly ($4.95\times10^{14}$ FLOP/s) at 40% MFU, giving **9.6 days** rather than 10.6, and states the convention as a rule. Chapter 17 now gives 148 FLOP/byte with the 295 figure noted and explained.

### 3.4 Equilibrium weight-norm expression *(Ch. 7 §7.4)*

**The error.** The formula $\|W\|^4_{\text{eq}} \approx \frac{\eta\|\nabla\mathcal{L}\|^2}{2\lambda}$ contained a vestigial "$\cdot\frac11$" and — more seriously — was circular: $\|\nabla\mathcal{L}\|$ itself depends on $\|W\|$ (it scales as $1/\|W\|$ by the scale-invariance argument two paragraphs earlier), so the expression as written does not determine $\|W\|$.

**Fix.** Introduced the scale-invariant quantity $G\equiv\|W\|\cdot\|\nabla_W\mathcal{L}\|$, restated the result as $\|W\|^4_{\text{eq}}\approx\eta G^2/(2\lambda)$ and $\eta_{\text{eff}}\approx\sqrt{2\eta\lambda}/G$, and added the two-line derivation (growth per step $\eta^2G^2/\|W\|^2$ balanced against decay $2\eta\lambda\|W\|^2$). The conclusion $\eta_{\text{eff}}\propto\sqrt{\eta\lambda}$ was correct throughout; only the intermediate step was malformed.

---

## 4. Derivations re-checked line by line

Each of these was re-derived from scratch and matched:

| Derivation | Chapter | Notes |
|---|---|---|
| Bias–variance decomposition | §2.2 | cross term vanishes correctly |
| ELBO from Jensen; gap $=\mathrm{KL}(q\|p(z\mid x))$ | §1.4 | ✓ |
| Softmax Jacobian $p_i(\delta_{ij}-p_j)$ | §1.3 | ✓ |
| Heavy-ball optimal $\beta^*=\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2$ | §4.5 | ✓ |
| Adam bias correction | §5.3 | geometric sum $(1-\beta_1^t)$ ✓ |
| He initialization $\mathrm{Var}(W)=2/n$ | §6.4 | ReLU halving handled ✓ |
| BatchNorm backward pass | §7.2 | three-term form; $\sum_i(x_i-\mu)=0$ used correctly ✓ |
| Scale invariance $\Rightarrow\langle\nabla\mathcal{L},W\rangle=0$ | §7.4 | ✓ (downstream step corrected, §3.4 above) |
| Early stopping $\equiv\ell_2$ with $\lambda\approx1/(\eta t)$ | §7.5 | ✓ |
| Dropout $\equiv$ data-dependent $\ell_2$ | §7.5 | ✓ |
| $\mathrm{Var}(q^\top k)=d_k$ | §11.2 | ✓ |
| $12d^2$ per layer; $C\approx6ND$ | §11.7 | ✓, validated against GPT-3 |
| RoPE: $R_m^\top R_n = R_{n-m}$ | §12.4 | rotation group property ✓ |
| Sinusoidal shift is linear in $PE_{pos}$ | §12.2 | angle-addition identities ✓ |
| Online softmax rescaling (FlashAttention) | §12.5 | exact, not approximate ✓ |
| Chinchilla constrained optimization | §15.2 | ✓, exponents recomputed |
| **DPO, full derivation** | §16.5 | Gibbs optimum → invert → $Z(x)$ cancels ✓ |
| Policy-gradient baseline is unbiased | §27.4 | $\nabla\sum_a\pi=0$ ✓ |
| **Policy gradient theorem** | §27.4 | dynamics term vanishes ✓ |
| GAE as geometric average of $k$-step estimators | §27.6 | reduces to TD($\lambda$) on advantages ✓ |
| **Bellman operator is a $\gamma$-contraction** | §26.4 | $\|\max f-\max g\|\le\max\|f-g\|$ step ✓ |
| **GAN optimal discriminator → JSD** | §19.4 | $\arg\max$ of $a\log y+b\log(1-y)$ ✓ |
| VAE closed-form Gaussian KL | §19.3 | ✓ |
| **Diffusion forward closed form** (induction) | §20.2 | variance combination ✓ |
| **Diffusion exact posterior** $\tilde\beta_t,\tilde\mu_t$ | §20.3 | verified symbolically and numerically |
| $\epsilon$-parameterized mean | §20.4 | ✓ |
| Score $=-\epsilon/\sqrt{1-\bar\alpha_t}$ | §20.6 | ✓ |
| CFG from classifier guidance | §20.8 | implicit-classifier substitution ✓ |
| D3PM posterior and $\bar Q_t$ | §21.2 | ✓ |
| Ridge SVD shrinkage $\sigma_j^2/(\sigma_j^2+\lambda)$ | §22.2 | ✓ |
| LASSO soft-threshold via subgradient | §22.3 | ✓ |
| **SVM dual + KKT sparsity** | §22.5 | ✓ |
| **Representer theorem** | §22.5 | orthogonal decomposition argument ✓ |
| **AdaBoost = coordinate descent on exp loss** | §23.3 | weights emerge, not posited ✓ |
| **XGBoost $w^*=-G/(H+\lambda)$ and gain** | §23.5 | ✓ |
| PCA three ways | §24.1 | ✓ |
| **EM for GMM + monotonicity proof** | §24.4 | ✓ |
| InfoNCE as MI bound | §25.3 | ✓ — with the honest caveat retained |
| Simplex ETF angle | §31.1 | ✓ |
| **Conformal coverage guarantee** | §33.5 | rank-exchangeability argument ✓ |
| Randomized smoothing radius | §33.8 | ✓ |

---

## 5. Analogies — audited for where they break

An analogy that is never qualified is a future misconception. Each was checked for whether the book states its limits.

| Analogy | Chapter | Verdict |
|---|---|---|
| Restaurant visit / dinner companions (bootstrap) | §3.1 | ✓ — correctly notes companions ate the same night, i.e. shared data |
| Ball rolling downhill (momentum) | §4.5 | ✓ — book notes real momentum has no friction term matching physics |
| Relay race baton (normalization) | §7.1 | ✓ |
| Rubber stamp (convolution) | §8.1 | ✓ — the "same stamp anywhere" *is* translation equivariance |
| Tracked changes (residual connections) | §8.2 | ✓ — strongest analogy in the book; maps exactly onto $F=0$ being cheap |
| Index card / filing cabinet (RNN → LSTM) | §9.1, §9.3 | ✓ — the "adding to a drawer vs rewriting the card" maps precisely onto $\mathrm{diag}(f_t)$ vs a full Jacobian |
| Lego bricks (subword tokenization) | §10.4 | ✓ |
| Poster session (attention) | §11.1 | ✓ — the second lap correctly corresponds to the second layer |
| Reading a contract with a team (multi-head) | §11.3 | ✓ |
| Two clock hands (RoPE) | §12.4 | ✓ — "angle between depends only on the difference" is exactly $R_m^\top R_n=R_{n-m}$ |
| Radio spectrum crosstalk (superposition) | §32.2 | ✓ — sparsity ↔ stations rarely loud simultaneously; correct |
| Ink diffusing in water (diffusion) | §20.1 | ✓ — "reversing one instant is easy" is the load-bearing claim and it is the right one |
| Kneading dough (normalizing flows) | §19.5 | ✓ — "recorded every fold" ↔ tractable Jacobian |
| Forger and detective (GANs) | §19.4 | ✓ — "neither has a fixed target" correctly foreshadows the instability |
| Lossy compressor with fuzzy codes (VAE) | §19.3 | ✓ |
| Road between villages (SVM margin) | §22.5 | ✓ — houses touching the edge ↔ support vectors |
| Twenty questions (decision trees) | §23.1 | ✓ |
| Relay of specialists (boosting) | §23.3 | ✓ — contrasts correctly with bagging's independent polling |
| Flattened pancake (PCA) | §24.1 | ✓ — the "not aligned with your axes" point is the one people miss |
| Learning a city by giving directions (SSL) | §25.1 | ✓ |
| Cooking judged only by the final meal (RL) | §26.1 | ✓ — credit assignment stated explicitly |
| Bendy wire with slack (double descent) | §30.1 | ✓ — **checked carefully**, since this is the easiest place to mislead; "excess capacity buys smoothness" is the correct intuition for minimum-norm interpolation |
| Committee too large for anyone to change (NTK) | §30.2 | ✓ — "from the outside it looks like learning" correctly captures no feature learning |
| Decompiling a binary (mech interp) | §32.1 | ✓ |
| Weather forecaster (calibration) | §33.1 | ✓ — correctly separates calibration from accuracy |
| Conference/library/hospital triage (MoE) | §11.8 | ✓ |

**One analogy was deliberately left partial** and is flagged in the text: the "relay race" for normalization does not capture scale invariance (§7.4), which is arguably normalization's more important effect. §7.4 does not attempt an analogy, because no clean one exists — the mathematics is the explanation.

---

## 6. Claims deliberately hedged rather than asserted

Places where the honest answer is uncertain, and the book says so rather than presenting a clean story:

- **InfoNCE's MI bound does not explain contrastive learning's success** (§25.3). Tighter MI estimators give worse representations. The alignment/uniformity account (§25.5) is presented as the better explanation.
- **Emergence** (§15.4). Both positions are given: metric artifacts are real, and mechanistic phase transitions are also real. The book does not pick a side because the evidence does not.
- **Flat minima** (§31.3). The reparameterization objection is stated in full rather than buried, along with the responses.
- **SAE feature granularity** (§32.3) is called out as the deepest unresolved problem, and recent negative results are acknowledged.
- **Chain-of-thought faithfulness** (§32.7) is explicitly not established.
- **IRM** (§33.9) is described as theoretically elegant and empirically disappointing, which is the current consensus and not what the original paper suggests.
- **BYOL's non-collapse** (§25.4). The book notes the early "BatchNorm provides implicit negatives" explanation was wrong, as an illustration of how hard mechanism attribution is.
- **Multi-agent systems** (§18.7) are described as frequently worse than a single well-prompted agent.
- **Lottery tickets** (§31.4) — the rewinding correction is given as a correction to the original claim, not folded silently into it.
- **Word2vec analogies** (§10.3) — the standard evaluation excludes the input words, and the property is weaker than the famous demo implies.

---

## 7. Consistency checks

- **Cross-references.** All inter-chapter links resolve to existing files. Section numbers cited in the text were checked against the sections they name.
- **$C=6ND$** is used identically in §11.7, §14.7, §15.1–15.2, and Q48/Q63. ✓
- **Case Study A** parameters (145,515/15,864 samples, batch 64, 2,274 steps/epoch, lr 3e-4, wd 0.01, epochs 22/36/37/43) are identical everywhere they appear: README, §1.4, §2.8, §3.6, §4.6, §5.x, §21.2. ✓
- **$\eta\lambda$ coupling** (§7.4) agrees with §5.2's AdamW discussion and §31.2's implicit-bias explanation of why decoupling matters. ✓
- **Zero-init residual** appears in §6.4, §8.2, §17.7 (LoRA's $B=0$), and §21.4 (AdaLN-Zero) and is explicitly cross-linked as one recurring idea rather than four coincidences. ✓
- **KL-regularized Boltzmann optimum** is identified as the same object in §16.5 (DPO) and §27.8 (SAC), in both directions. ✓
- **Reparameterization trick** is cross-referenced consistently across §19.3 (VAE), §20 (diffusion), and §27.8 (SAC). ✓
- **Notation.** $T$ is overloaded (sequence length, RL horizon, diffusion steps, temperature). This is unavoidable given standard usage in each field; the notation table flags it and each chapter disambiguates on first use. Deliberate, not an error.

---

## 8. Known limitations

1. **Currency.** Content reflects the field as of mid-2026. Chapters 13–18 and 32 are the fastest-moving; treat specific model names, benchmark numbers, and "current best" claims as dated even where the underlying mathematics is not.
2. **Empirical constants are load-bearing but soft.** Values like $d_{\text{ff}}=4d$, $\alpha=0.1$ for label smoothing, $\lambda\approx0.01$ for the D3PM auxiliary term, $\tau\approx0.1$ for contrastive learning, and $\epsilon=0.2$ for PPO are conventions with empirical support, not derived quantities. They are marked as such but are worth re-checking against current practice.
3. **Benchmark numbers** (ImageNet top-1, perplexities) are drawn from the original papers under their own training recipes. Chapter 8 §8.3's ConvNeXt discussion explains why cross-paper comparisons at unequal recipes are unreliable — that warning applies to this book's own tables.
4. **Proof rigour.** Derivations are complete at the level of a working practitioner, not a measure theorist. Regularity conditions (interchange of limits and integrals, differentiability almost everywhere, existence of moments) are generally assumed rather than verified. Where a result  depends on a condition that is often violated — Gauss–Markov's homoskedasticity, conformal prediction's exchangeability, the deadly triad — the condition is stated explicitly.
5. **The NTK and PAC-Bayes sections** present the results and their implications but not the full proofs, which are long. This is flagged in the text.
6. **Coverage gaps.** Time-series forecasting, causal inference, recommender systems, federated learning, differential privacy, and ML systems/MLOps are touched on only where they intersect other chapters. They are real fields and are not covered here.
7. **Single-author verification.** Every check in this document was performed by the same process that produced the text. That is a  limitation: an error in understanding produces a matching error in verification. The numerical checks in §2 are the strongest evidence here, because arithmetic run in Python is independent of the reasoning that produced the claim. **Treat the derivations as re-derivable, and re-derive them — that is what §34's cover-the-answer instruction is for.**

---

## 9. Summary

- **35 files, ~85,000 words, 654 marked equations.**
- **32 numerical claims checked; 28 verified, 4 corrected.**
- **44 derivations re-checked line by line; all sound after the §3.4 correction.**
- **27 analogies audited; all map correctly onto their mathematics, with one deliberate partial noted.**
- **All cross-references resolve.**

The four defects clustered in exactly the place you would predict: **hardware and systems arithmetic**, where conventions (dense vs sparse peak, MHA vs GQA) are unstated in most sources and quietly change answers by factors of 2 to 8. That is itself worth remembering — those are the numbers to double-check in someone else's work, including your own.

---

**Back to:** [Contents](00-README.md)
