# Chapter 15 — Scaling Laws & Emergence

> **Prerequisites:** Ch. 14 (the $C=6ND$ formula).

---

## 15.1 The empirical finding

### The one-line idea

Test loss falls as a **power law** in model size, data, and compute — smoothly, predictably, over many orders of magnitude, with no sign of a natural ceiling.

### The analogy

The learning curve of a manufacturing process. Every doubling of cumulative production reduces unit cost by a fixed percentage — Wright's law. It has held for aircraft, solar panels, and transistors across a century. Scaling laws are the same shape: not "more is better" but "more is better by an exactly predictable amount," which is what makes them useful for planning rather than merely encouraging.

### The Kaplan form

▸ $$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\qquad L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D},\qquad L(C)=\left(\frac{C_c}{C}\right)^{\alpha_C}$$

with $\alpha_N\approx0.076$, $\alpha_D\approx0.095$, $\alpha_C\approx0.050$ for language modelling.

▸ **Read the exponents as effort-per-improvement.** $\alpha_N = 0.076$ means a 10× larger model reduces loss by a factor $10^{-0.076} = 0.84$ — a **16% reduction**. To halve the loss you need $10^{\log_{10}2/0.076} = 10^{3.96} \approx 9{,}000\times$ the parameters. **Progress is real, cheap improvements are not.** This single calculation explains why frontier labs spend what they spend.

**Additional Kaplan findings that hold up:**
- Architecture details (depth vs width, aspect ratio) matter far less than $N$ — within a broad range, only total parameter count matters.
- Larger models are more **sample-efficient**: at a fixed number of tokens, bigger is better.
- Performance is limited by whichever of $N$, $D$, $C$ is the binding constraint; the others give no benefit.

---

## 15.2 Chinchilla — derive the compute-optimal allocation

### The question

Given a fixed compute budget $C = 6ND$, how should you split it between model size $N$ and tokens $D$?

Kaplan's answer (2020) was $N\propto C^{0.73}$ — mostly grow the model. **This was wrong**, and the error cost the field a generation of undertrained models (GPT-3, at 175B params on 300B tokens, is severely undertrained).

### The Chinchilla fit

Hoffmann et al. (2022) fit a three-term law over 400+ runs:

▸ $$L(N,D) = \underbrace{E}_{\text{irreducible}} + \underbrace{\frac{A}{N^\alpha}}_{\text{finite model}} + \underbrace{\frac{B}{D^\beta}}_{\text{finite data}}$$

Fitted: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$.

**Read the three terms.** $E$ is the entropy of natural language — no model gets below it (Ch. 1 §1.4). $A/N^\alpha$ is what you lose by having a finite model. $B/D^\beta$ is what you lose by having finite data.

### The optimization, done properly

Minimize $L$ subject to $C=6ND$. Substitute $D=\frac{C}{6N}$:

$$L(N) = E + AN^{-\alpha} + B\left(\frac{C}{6N}\right)^{-\beta} = E + AN^{-\alpha} + B\left(\frac{6N}{C}\right)^{\beta}$$

$$\frac{dL}{dN} = -\alpha AN^{-\alpha-1} + \beta B\,6^\beta C^{-\beta}N^{\beta-1} = 0$$

$$\alpha A N^{-\alpha-1} = \beta B\,6^\beta C^{-\beta}N^{\beta-1}$$

$$N^{\alpha+\beta} = \frac{\alpha A}{\beta B\,6^\beta}\,C^{\beta}$$

▸ $$\boxed{\ N_{\text{opt}} \propto C^{\frac{\beta}{\alpha+\beta}},\qquad D_{\text{opt}}\propto C^{\frac{\alpha}{\alpha+\beta}}\ }$$

**Plug in the numbers:** $\frac{\beta}{\alpha+\beta} = \frac{0.28}{0.62}=0.452$ and $\frac{\alpha}{\alpha+\beta}=\frac{0.34}{0.62}=0.548$.

▸ **Both exponents are close to $\tfrac12$.** Hence the famous headline: **scale model and data equally.** Every doubling of compute should be spent on $\sqrt2\times$ more parameters and $\sqrt2\times$ more tokens.

### The rule of thumb

▸ $$\frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20\ \text{tokens per parameter}$$

**Examples:** 1B → 20B tokens; 7B → 140B; 70B → 1.4T.

**The demonstration:** Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) on nearly every benchmark, at **identical training compute** — and is 4× cheaper to run.

### Why almost nobody follows Chinchilla exactly

Chinchilla optimizes **training** compute only. Real deployments pay **inference** compute forever.

▸ **Inference-aware scaling** (Sardana et al.): minimize $C_{\text{train}} + C_{\text{inference}}$ over the model's lifetime. If you will serve $D_{\text{inf}}$ tokens, total cost $\approx 6ND_{\text{train}} + 2ND_{\text{inf}}$. Since inference cost is linear in $N$ and independent of $D_{\text{train}}$, the optimum shifts sharply toward **smaller models trained on far more data.**

This is why LLaMA-3-8B was trained on 15T tokens — a ratio of **1,875 tokens per parameter**, roughly 94× past Chinchilla-optimal. It is *not* compute-optimal to train, and it is *far* better to deploy. **A model trained past Chinchilla still improves, just with diminishing returns; a model too large to serve is worthless.**

### Data-constrained scaling

When unique data runs out (Muennighoff et al., 2023): repeating data has value that decays with epoch count. Up to **~4 epochs** is nearly as good as fresh data; by ~16 epochs, additional repetition contributes essentially nothing. Beyond that, extra parameters also stop helping.

▸ Since high-quality text is finite (single-digit trillions of tokens on the open web), this is a genuine constraint, and it is the strategic reason for the intense current focus on **synthetic data**, **multimodal data**, and **test-time compute** (§15.5).

---

## 15.3 $\mu$P — hyperparameter transfer

### The problem

Optimal learning rate depends on model width. Tuning a 70B model directly is unaffordable, so you tune a small proxy — but the optimum shifts, so the transfer fails.

### The solution

**Maximal Update Parametrization** ($\mu$P, Yang & Hu) chooses per-layer scalings of initialization variance and learning rate such that, in the infinite-width limit, **every layer's activations and updates stay $\Theta(1)$**. The consequence:

▸ **Optimal hyperparameters become width-independent, so they transfer from a small proxy to a large model exactly.**

### The prescriptions

| | Standard (SP) | $\mu$P |
|---|---|---|
| Input/embedding LR | $\eta$ | $\eta$ |
| Hidden weight init var | $1/\mathrm{fan\_in}$ | $1/\mathrm{fan\_in}$ |
| **Hidden weight LR (Adam)** | $\eta$ | $\eta/\mathrm{fan\_in}$ |
| **Output layer init** | $1/\mathrm{fan\_in}$ | $1/\mathrm{fan\_in}^2$ |
| **Output logits** | — | multiply by $1/\mathrm{fan\_in}$ |
| Attention scale | $1/\sqrt{d_k}$ | $1/d_k$ |

### The reasoning, in one line

If a layer's update is $\Delta W$ and it acts on an activation $x\in\mathbb{R}^{n}$, then $\Delta W x$ is a sum of $n$ terms. For the *change in output* to remain $\Theta(1)$ as $n\to\infty$, the per-entry update must scale as $1/n$ when the terms are correlated (which they are after the first step, unlike at initialization where the $1/\sqrt n$ of Ch. 6 suffices). **$\mu$P is Chapter 6's variance analysis extended from initialization to the entire training trajectory.**

▸ **Practical payoff:** tune LR, init scale, and warmup on a 40M-parameter model, transfer directly to 7B or 70B. This is now standard practice at every serious lab, and it saves an enormous fraction of a pretraining budget.

Also note: $\mu$P changes attention to $1/d_k$ rather than $1/\sqrt{d_k}$ — worth knowing as a rare, principled exception to Chapter 11's rule.

---

## 15.4 Emergence, and the argument against it

### The claim

Some capabilities appear **abruptly** at a scale threshold: three-digit arithmetic, word unscrambling, multi-step reasoning show near-zero accuracy until a critical size, then rise sharply. Wei et al. (2022) catalogued dozens.

If true, this is important and alarming: capabilities cannot be forecast from smaller models.

### The critique (Schaeffer, Miranda & Koyejo, 2023)

▸ **Emergence may be an artifact of the metric, not the model.**

Consider exact-match accuracy on a $k$-token answer. If per-token accuracy is $p$, then
$$\Pr(\text{exact match}) = p^k$$
Suppose $p$ improves *smoothly* with scale — say $p$ goes $0.5\to0.9$ over three orders of magnitude. With $k=5$:
$$0.5^5 = 0.031 \quad\to\quad 0.9^5 = 0.59$$
The exact-match curve is sharply convex and looks like a phase transition. **The underlying per-token improvement was perfectly smooth.**

The general point: **discontinuous or nonlinear metrics manufacture discontinuities.** Exact match, multiple-choice accuracy, and any thresholded score do this. Continuous metrics — token edit distance, Brier score, log-likelihood of the correct answer — show smooth improvement on the *same* model outputs.

▸ **The honest position, which is what to say if asked:** the *underlying capability* improves smoothly and predictably; the *user-visible usefulness* can change abruptly, because usefulness often depends on crossing a reliability threshold. Both statements are true and they are not in conflict. Emergence is real as a fact about task success rates and mostly not real as a fact about the model's internals.

**A genuine residual:** some capabilities do look sharp even under continuous metrics (in-context learning's onset alongside induction-head formation, Ch. 13 §13.3). Phase transitions in *mechanisms* appear to be real even where phase transitions in *metrics* are artifacts.

---

## 15.5 Test-time compute scaling

The newest scaling axis, and the one most likely to matter over the next few years.

▸ Instead of scaling training, spend more compute **at inference**: sample many chains of thought, search over them, verify, and select.

**Methods:**
- **Self-consistency:** sample $k$ chains, take the majority answer. Accuracy improves roughly logarithmically in $k$.
- **Best-of-$n$ with a verifier / reward model:** sample $n$, score, pick the best. Gains depend on verifier quality; a weak verifier saturates or degrades (reward hacking, Ch. 16).
- **Search:** tree search / MCTS over reasoning steps with a process reward model.
- **Learned long reasoning:** train the model via RL to produce long internal chains before answering (o-series, R1-style). This converts test-time compute into a *trained* behaviour rather than an external procedure.

▸ **The finding that reframed the field:** on many reasoning benchmarks, the loss/accuracy curve against *inference* FLOPs is also a clean power law, and there are regimes where a smaller model with more test-time compute beats a larger model with greedy decoding **at equal total FLOPs**. Training compute and inference compute are, to a degree, substitutable.

**The catch:** the substitution works best where answers are *verifiable* (math, code, formal tasks) and much worse where they are not, because the selection step needs a reliable signal.

---

## 15.6 Using scaling laws in practice

The workflow that a serious lab actually runs:

1. Train a ladder of small models ($10^{17}$–$10^{20}$ FLOPs) across a range of $N$ and $D$.
2. Fit $L(N,D) = E + AN^{-\alpha}+BD^{-\beta}$ by Huber-loss regression in log space (Huber, not squared error, because outlier runs are common).
3. Solve for the optimal $(N,D)$ at your target budget, adjusted for expected inference volume.
4. Use $\mu$P to transfer hyperparameters from the ladder.
5. **Predict** the final loss before launching, and treat a deviation as a bug signal.

▸ **Scaling laws' real value is as a forecasting and debugging instrument**, not a philosophy. GPT-4's technical report notes the final loss was predicted from runs using $10{,}000\times$ less compute. If your big run misses its predicted loss, you have an infrastructure or data problem, and you know that on day two rather than day forty.

**Caveats to state:** the exponents are dataset- and architecture-dependent; laws fitted on one data mixture do not transfer to another; they describe pretraining loss, which after post-training is only loosely coupled to usefulness; and extrapolating many orders of magnitude beyond the fitted range is an act of faith.

---

## Check for Understanding

**Loss falls as a power law in compute with an exponent near $0.05$, meaning improvements are predictable but expensive; the compute-optimal split is roughly equal scaling of parameters and data (~20 tokens per parameter) because the two Chinchilla exponents are close to equal; and most apparent "emergence" is the effect of putting a smoothly improving model through a discontinuous metric.**

---

**Next:** [Chapter 16 — Post-Training: SFT, RLHF, DPO & Reasoning](16-post-training-rlhf-dpo-reasoning.md)
