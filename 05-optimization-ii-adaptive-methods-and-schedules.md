# Chapter 5 — Optimization II: Adaptive Methods & Learning-Rate Schedules

> **Prerequisites:** Ch. 4.
> **This chapter directly answers the two questions from the Case Study A log:** what "adaptive" actually means in AdamW, and what every learning-rate decay technique is.

---

## 5.1 The idea behind adaptivity

### The one-line idea

Different parameters have wildly different gradient scales. Give each parameter its own step size, inferred from its own gradient history.

### The analogy

You're managing a team where one person's estimates are always in dollars and another's in millions of dollars. Rather than shouting the same correction at both, you normalize each person's feedback by how much they typically vary. Someone who always reports huge numbers gets scaled down; someone who whispers gets scaled up.

### The problem being solved

In Ch. 4 we saw everything depends on $\kappa = \lambda_{\max}/\lambda_{\min}$. In deep nets, curvature varies by orders of magnitude *across coordinates*: embedding rows for rare tokens get gradients thousands of times smaller than the final layer bias. A single global $\eta$ is either too big for one and too small for the other.

▸ Adaptive methods approximate a **diagonal preconditioner**: $\theta_{t+1} = \theta_t - \eta D_t^{-1}g_t$ with $D_t$ diagonal. This is a cheap ($O(p)$) stand-in for $H^{-1}$, capturing per-coordinate scale but not cross-coordinate rotation.

---

## 5.2 The lineage: AdaGrad → RMSProp → Adam → AdamW

### AdaGrad (2011)

$$G_t = G_{t-1} + g_t^2 \quad\text{(elementwise)},\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\odot g_t$$

**Property:** parameters with historically large gradients get small steps. Excellent for **sparse features** — a rare token's embedding has $G$ near zero, so it gets a large step when it finally appears. Provably good regret for online convex optimization.

▸ **Fatal flaw:** $G_t$ is a monotonically increasing sum. The effective LR $\eta/\sqrt{G_t}$ decays to zero *no matter what*, on a schedule dictated by accumulated history rather than by need. In deep learning this kills training prematurely.

### RMSProp (Hinton, unpublished, 2012)

Fix the flaw: replace the sum with an **exponential moving average**.

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t}+\epsilon}\odot g_t$$

Now $v_t$ is a *running estimate of $\mathbb{E}[g^2]$* over a window of $\frac{1}{1-\beta_2}$ steps, and it can go down as well as up. This is a real second-moment *estimator*, not an accumulator.

### Adam (Kingma & Ba, 2015)

Add momentum on the first moment too.

▸ $$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t &&\text{(1st moment: mean of gradient)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 &&\text{(2nd moment: uncentered variance)}\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t} &&\text{(bias correction)}\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

#### Deriving the bias correction (do this once)

Unroll $m_t$ with $m_0=0$:
$$m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$$

Assume $g_i$ are drawn with stationary mean $\mathbb{E}[g]$:
$$\mathbb{E}[m_t] = \mathbb{E}[g](1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g](1-\beta_1)\frac{1-\beta_1^t}{1-\beta_1} = \mathbb{E}[g]\left(1-\beta_1^t\right)$$

▸ So $m_t$ **underestimates** the true mean by the factor $(1-\beta_1^t)$, purely because it started at zero. Dividing by $(1-\beta_1^t)$ makes it unbiased. Same argument for $v_t$ with $\beta_2$. ∎

**Why it matters most for $v$:** at $t=1$, $1-\beta_2^1 = 0.001$. Without correction, $v_1 = 0.001g_1^2$, so $\sqrt{v_1} = 0.032|g_1|$, and the update would be $\eta g_1/(0.032|g_1|) = 31\eta$ — a **31× too-large first step**. Bias correction turns that into exactly $\eta\cdot\mathrm{sign}(g_1)$. Without it, Adam blows up in the first few dozen steps.

#### What the update actually looks like

▸ $$\frac{\hat m_t}{\sqrt{\hat v_t}} \approx \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}} = \frac{\text{mean}}{\text{RMS}} \in [-1, 1]$$

**Adam's per-coordinate step size is bounded by roughly $\eta$**, regardless of gradient magnitude. In the limit of a perfectly consistent gradient (no noise), $\hat m/\sqrt{\hat v}\to \mathrm{sign}(g)$ and Adam becomes **signSGD with step $\eta$**. In the limit of pure noise with zero mean, the ratio is $\approx \sqrt{1-\beta_1}\cdot$something small, and the step shrinks.

▸ **The single most useful mental model: Adam takes a step of size $\eta$ in the direction of the gradient's *sign*, damped by how inconsistent the gradient has been.** It is a signal-to-noise ratio detector, per coordinate.

This tells you immediately why $\eta=3\times10^{-4}$ is the "magic" Adam LR: it's the *actual displacement per parameter per step*, in raw parameter units. For weights of typical magnitude $\sim0.02$ (He init at width 1024 gives $\sigma=\sqrt{2/1024}=0.044$), a step of $3\times10^{-4}$ is a **0.7% relative change per step.** Over 2,274 steps in one epoch, if steps were coherent, a weight could travel $0.68$ — far more than its own magnitude. They aren't coherent, so it doesn't; but that's the scale you're working at.

#### The $\epsilon$ placement matters

- PyTorch: $\dfrac{\hat m}{\sqrt{\hat v}+\epsilon}$
- Original paper's alternative: $\dfrac{\hat m}{\sqrt{\hat v + \epsilon}}$

They differ when $\hat v \ll \epsilon^2$. In fp16/bf16 training, $\epsilon = 10^{-8}$ can *underflow*, so people use $10^{-6}$ or even $10^{-4}$. Note that increasing $\epsilon$ makes Adam more like SGD (large $\epsilon$ ⇒ the denominator is constant ⇒ plain momentum). **$\epsilon$ is secretly an interpolation knob between Adam and SGD.**

#### $\beta_2$ and the memory horizon — *this settles the Case Study A question*

The EMA with decay $\beta_2$ has effective memory $\frac{1}{1-\beta_2}$ steps.

▸ For $\beta_2=0.999$: **1,000 steps.**
▸ In Case Study A at 2,274 steps/epoch: **0.44 epochs.**

The optimizer's entire representation of the loss landscape spans less than half an epoch of data. It has no state variable that persists over 13 epochs. It has no state variable indexed by epoch at all. And it never sees `best`, `val_realCE`, or any validation quantity — those live in Python outside the optimizer's `state` dict, and the optimizer's `step()` receives only `p.grad`.

▸ **Conclusion, stated precisely: there is no channel through which a validation dry spell could influence AdamW's behaviour, and no state with a long enough horizon to encode one even if there were.**

(For very large batch / long training, $\beta_2=0.95$ is often better — shorter memory tracks a fast-changing landscape. For small batches, $\beta_2=0.999$ or higher smooths more noise. Your $B=64$ justifies 0.999.)

### AdamW — decoupled weight decay (Loshchilov & Hutter, 2019)

This distinction is subtle, frequently misunderstood, and directly relevant to your `weight_decay=0.01`.

**L2 regularization** adds $\frac\lambda2\|\theta\|^2$ to the loss, so the gradient becomes $g_t + \lambda\theta_t$. In Adam, that modified gradient goes *through the preconditioner*:

$$\theta_{t+1} = \theta_t - \eta\frac{\widehat{m_t(g+\lambda\theta)}}{\sqrt{\widehat{v_t(g+\lambda\theta)}}+\epsilon}$$

▸ **The bug:** the decay term gets divided by $\sqrt{\hat v}$. Parameters with large gradients (large $\hat v$) receive **less** decay; parameters with small gradients receive **more**. That is exactly backwards from what you want, and it means the strength of your regularization depends on gradient magnitudes you don't control.

**AdamW** decouples it:

▸ $$\boxed{\ \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\right)\ }$$

Now the shrinkage is a clean multiplicative $\theta \leftarrow (1-\eta\lambda)\theta$ applied uniformly.

**Numbers for Case Study A.** $\eta\lambda = 3\times10^{-4}\times0.01 = 3\times10^{-6}$ per step.
- Per epoch (2,274 steps): $(1-3\times10^{-6})^{2274} = 0.99320$ → **0.68% norm decay per epoch** with zero gradient.
- Over 43 epochs: $0.9932^{43} = 0.745$ → weights would shrink to 74.5% of initial norm.

This is real but gentle. Note it's coupled to $\eta$ in PyTorch's AdamW (decay $=\eta\lambda\theta$), so **if you ever add an LR schedule, your weight decay decays with it.** Some implementations decouple fully ($\lambda\theta$ without the $\eta$); know which you have.

**Practical:** exclude biases, LayerNorm/RMSNorm gains, and (usually) embeddings from weight decay. Decaying a LayerNorm gain toward zero is actively harmful — it shrinks the layer's output scale for no regularization benefit (see the scale-invariance discussion in Ch. 7).

---

## 5.3 Every learning-rate schedule, with formulas

### Why schedules exist, in one sentence

From Ch. 4 §4.6: SGD with constant $\eta$ equilibrates in a noise ball of radius $\propto \eta\sigma^2/(\mu B)$. Decaying $\eta$ shrinks the ball. **A schedule is a cooling schedule** — you anneal the temperature $\eta/B$.

### The catalogue

| Schedule | Formula | Notes |
|---|---|---|
| **Constant** | $\eta_t=\eta_0$ | your current setup. Never converges; diffuses. |
| **Step** | $\eta_t = \eta_0\gamma^{\lfloor t/s\rfloor}$ | classic ResNet recipe: $\times0.1$ at epochs 30/60/90. Loss shows characteristic cliffs. |
| **Multi-step** | drops at specified milestones | same, hand-placed |
| **Exponential** | $\eta_t=\eta_0 e^{-kt}$ or $\eta_0\gamma^t$ | smooth version of step |
| **Polynomial / linear** | $\eta_t=\eta_0(1-t/T)^p$ | $p=1$ is linear-to-zero; the standard for BERT-family and, per Chinchilla-era practice, very strong |
| **Cosine** | $\eta_t = \eta_{\min} + \frac{\eta_0-\eta_{\min}}{2}\left(1+\cos\frac{\pi t}{T}\right)$ | the modern default; slow start, fast middle, slow finish |
| **Cosine w/ warm restarts (SGDR)** | cosine over cycles $T_i$, $T_{i+1}=T_i\cdot T_{\text{mult}}$ | each restart escapes a basin; gives free ensembles |
| **1/t (Robbins–Monro)** | $\eta_t=\eta_0/(1+kt)$ | theoretically optimal for strongly convex; too aggressive in practice |
| **Inverse-sqrt** | $\eta_t = \eta_0/\sqrt{\max(t,t_w)}$ | the original Transformer schedule |
| **One-cycle** | LR up to $\eta_{\max}$ then down below $\eta_0$; momentum moves inversely | Smith's "super-convergence" |
| **ReduceLROnPlateau** | if no improvement for `patience` epochs: $\eta\leftarrow\gamma\eta$ | *reactive*, not scheduled |
| **WSD (warmup–stable–decay)** | warmup, long constant plateau, short sharp decay at the end | lets you extend training without recommitting to $T$; now common for LLMs |

### The Transformer schedule, decoded

$$\eta_t = d_{\text{model}}^{-0.5}\cdot\min\left(t^{-0.5},\ t\cdot t_{\text{warmup}}^{-1.5}\right)$$

The two branches meet at $t=t_{\text{warmup}}$. Before: linear warmup. After: inverse-sqrt decay. The $d_{\text{model}}^{-0.5}$ prefactor is an early, crude version of $\mu$P (Ch. 21) — LR should shrink with width.

### On ReduceLROnPlateau specifically

Your project notes correctly observe this isn't in your setup, and correctly note that if it *were*, a dry spell would **lower** the LR rather than waste anything. Let me sharpen why that's the right instinct — and add the caveat.

The mechanism: you're in a noise ball of radius $\propto\eta$. If the loss has stopped improving, either (a) you're at the basin floor and the noise ball is now the limiting factor — shrinking $\eta$ genuinely helps, or (b) you're on a plateau and shrinking $\eta$ makes it worse. ReduceLROnPlateau bets on (a).

▸ **The caveat, which follows directly from Ch. 3:** ReduceLROnPlateau's trigger is `no improvement in best for patience epochs`, and we showed that under pure noise a 13-epoch gap occurs 63% of the time. **A plateau scheduler fed a noisy metric will fire on noise.** With your $\mathrm{SE}\approx0.15$ and `patience=10`, it would have cut your LR at least once during a period when the model was in fact improving steadily. If you ever add one:
- feed it a **smoothed** metric (EMA over 5–10 epochs), and
- set `threshold` to something meaningful relative to your measured noise (e.g. `threshold=0.05`, `threshold_mode='abs'`), and
- set `cooldown` so it doesn't fire twice on the same noise excursion.

Honestly, for your situation **cosine decay to $0.1\eta_0$ over your planned epoch budget** is a better choice than plateau-based reduction, because it doesn't depend on a noisy signal at all. And keeping the recipe identical across the four arms is the right scientific call — a scheduler that fires on noise would fire at *different times* in different arms and destroy the comparison.

### Choosing $\eta_0$: the LR range test

Run a short training with $\eta$ increasing exponentially from $10^{-8}$ to $10$. Plot loss vs $\log\eta$. You'll see: flat, then descending, then a minimum, then divergence.

▸ **Pick $\eta_0$ about one order of magnitude below the divergence point**, or at the steepest descent point. Costs a few hundred steps. Almost nobody does it, and almost everyone should.

---

## 5.4 Warmup, and why it exists

Three explanations, all partially true:

**1. Adam's variance at small $t$.** $\hat v_t$ is estimated from few samples. Its relative variance is large early, so $1/\sqrt{\hat v_t}$ has a heavy tail and occasionally produces enormous steps. **RAdam** makes this precise: define the "length of the approximated SMA"
$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1-\beta_2^t},\qquad \rho_\infty = \frac{2}{1-\beta_2}-1$$
and apply the rectification factor
▸ $$r_t = \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_\infty}{(\rho_\infty-4)(\rho_\infty-2)\rho_t}}$$
only when $\rho_t>4$; otherwise fall back to SGD-with-momentum. This is warmup *derived* rather than tuned. For $\beta_2=0.999$, $\rho_\infty=1999$ and $\rho_t>4$ first happens around $t\approx5$ — but $r_t$ stays well below 1 for several thousand steps, which is why warmup periods of ~2,000–10,000 steps are standard.

**2. Curvature is highest at initialization.** $\lambda_{\max}(H)$ typically spikes early. Since stability needs $\eta<2/\lambda_{\max}$, you need a small $\eta$ early. (See §5.5.)

**3. Large-batch loss landscapes.** With large $B$ the gradient is nearly deterministic, so a too-large first step commits hard to a bad direction. Warmup lets the model "find its feet."

**Practical:** linear warmup over 1–5% of total steps, from 0 (or $\eta_0/100$) to $\eta_0$. For Case Study A, ~2,000 steps ≈ 1 epoch would be typical. **Case Study A has no warmup and hasn't diverged, which suggests $3\times10^{-4}$ is comfortably below your stability threshold** — possibly meaning you could train faster with a higher peak LR plus warmup.

---

## 5.5 Edge of Stability — the phenomenon that broke the classical picture

### The classical claim

From Ch. 4 §4.4: GD on a quadratic is stable iff $\eta < 2/\lambda_{\max}$. So training should require $\lambda_{\max} < 2/\eta$, and if $\lambda_{\max}$ exceeds that, training diverges.

### What actually happens (Cohen et al., 2021)

Train a real network with full-batch GD at fixed $\eta$ and track $\lambda_{\max}(H)$ over time:

1. **Progressive sharpening:** $\lambda_{\max}$ *increases* steadily during training. The optimizer walks toward sharper regions.
2. It rises until it hits $2/\eta$.
3. Then it **stops** and hovers just above $2/\eta$, oscillating.
4. **The loss keeps decreasing anyway**, non-monotonically, in a sawtooth.

▸ $$\lambda_{\max}(H) \approx \frac{2}{\eta}\quad\text{for most of training}$$

This is called the **Edge of Stability (EoS)**.

### Why it's not a contradiction

The quadratic analysis is local and assumes fixed $H$. In reality: when $\eta\lambda_{\max}$ slightly exceeds 2, the top eigendirection oscillates with growing amplitude — but the *third-order* term means that as the oscillation grows, the model moves to a region of *lower* $\lambda_{\max}$. This is a self-stabilizing negative feedback loop. The system finds an equilibrium at the boundary.

### The consequences you should internalize

▸ **The learning rate directly sets the sharpness of the solution you find.** $\lambda_{\max}\approx 2/\eta$ means a *smaller* LR gives you a *sharper* minimum. Combined with "flat minima generalize better" (Ch. 19), this is a mechanistic reason why larger learning rates often generalize better — the LR is an *implicit sharpness regularizer*, not just a speed knob.

For Case Study A: $\eta = 3\times10^{-4}$ implies an EoS sharpness of $\lambda_{\max}\approx 6{,}667$. (Adam complicates this — the relevant quantity is the sharpness of the *preconditioned* landscape, and the Adaptive-EoS threshold is roughly $\lambda_{\max}(\text{diag}(\hat v)^{-1/2}H) \approx 38/\eta$ empirically — but the qualitative story survives.)

▸ **Loss going up for a few steps is not a bug.** At EoS, training is *supposed* to be non-monotone. If you see small sawteeth in your training loss at constant LR, that's the mechanism working, not instability.

**Practical implication:** if your model isn't at EoS, your LR is probably too low and you're leaving both speed and generalization on the table. You can measure this: power-iterate $Hv$ (Ch. 1 §1.2.4) for 20 iterations every few hundred steps and plot $\lambda_{\max}\eta$. If it's well below 2, raise $\eta$.

---

## 5.6 Adam vs SGD: the generalization question

**The empirical fact:** on image classification, well-tuned SGD+momentum often beats Adam on *test* accuracy despite Adam winning on *training* loss. On language and generative modelling (including diffusion), Adam wins decisively and SGD often fails to train at all.

**Why Adam is essential for transformers:**
- Gradient scales differ enormously across layers/parameter types (embeddings vs attention vs LayerNorm gains). Diagonal preconditioning fixes this; SGD can't.
- Heavy-tailed gradient noise. Adam's normalization is robust to outlier gradients; SGD's isn't.
- Token frequency is Zipfian — rare-token embeddings need AdaGrad-like treatment.

**Why Adam can generalize worse (when it does):**

▸ **The implicit bias argument.** On separable data with logistic loss, GD converges in direction to the **max-$\ell_2$-margin** solution (Ch. 19). Adam/signSGD converges toward a **max-$\ell_\infty$-margin**-like solution. These are different inductive biases, and $\ell_2$ margin happens to match the geometry of vision problems better.

▸ **The sharpness argument.** Adam's effective per-coordinate LR is larger in low-curvature directions, so it happily settles in sharper minima that SGD's noise would escape.

**Middle grounds:** decoupled weight decay (AdamW) closes much of the gap; **SAM** (Sharpness-Aware Minimization) closes more:
$$\min_\theta \max_{\|\delta\|\le\rho}\mathcal{L}(\theta+\delta) \quad\Rightarrow\quad \tilde g = \nabla\mathcal{L}\left(\theta + \rho\frac{\nabla\mathcal{L}(\theta)}{\|\nabla\mathcal{L}(\theta)\|}\right)$$
Two forward-backward passes per step (2× cost) to explicitly optimize for flatness. Consistently improves generalization by 0.5–2% on vision; mixed results on generative modelling.

---

## 5.7 The modern optimizers worth knowing

**Lion** (evolved by search, 2023):
$$c_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad \theta_t = \theta_{t-1} - \eta\big(\mathrm{sign}(c_t) + \lambda\theta_{t-1}\big),\quad m_t = \beta_2 m_{t-1}+(1-\beta_2)g_t$$
Only **one** state buffer instead of Adam's two — 50% optimizer-memory saving. Pure sign update. Needs $\eta$ about $3$–$10\times$ smaller than Adam and $\lambda$ correspondingly larger. Competitive on vision and language.

**Shampoo / SOAP:** full-matrix preconditioners per tensor mode. $P_L = (\sum_t G_tG_t^\top)^{-1/4}$, $P_R = (\sum_t G_t^\top G_t)^{-1/4}$, update $= P_L G P_R$. Captures cross-coordinate structure Adam can't. 1.3–2× wall-clock speedup at scale; expensive matrix roots (amortized over many steps).

**Muon** (2024): orthogonalize the momentum matrix via Newton–Schulz iteration, i.e. take $\mathrm{orth}(M) = UV^\top$ from $M=U\Sigma V^\top$. The update has all singular values equal to 1, which is a spectral-norm-controlled step. Currently gives the best measured speedups on transformer pretraining among cheap methods. Used for 2-D weight matrices only; keep AdamW for embeddings, biases, norms.

---

## 5.8 What I would change in Case Study A

Concretely, given everything above:

1. ▸ **Add an EMA of the weights** ($\gamma=0.9999$). Highest expected return, near-zero risk, standard for diffusion. Evaluate the EMA weights, not the raw ones.
2. ▸ **Run an LR range test.** You may be well below EoS at $3\times10^{-4}$ with $B=64$; $1\times10^{-3}$ with 2,000 steps of warmup is a plausible faster recipe.
3. ▸ **If you add a schedule, use cosine to $0.1\eta_0$**, not plateau-based. It doesn't depend on your noisy metric, and it stays identical across all four arms.
4. ▸ **Exclude LayerNorm gains and biases from weight decay** if you haven't. One line, small consistent gain.
5. **Log $\|g\|$, $\|\theta\|$, and the per-layer update-to-weight ratio $\eta\|\Delta\theta_\ell\|/\|\theta_\ell\|$.** That last one should sit around $10^{-3}$; if a layer is at $10^{-1}$ or $10^{-5}$ you've found a bug or a scaling problem.

Keeping the recipe identical across the four arms is correct and I'd keep doing it — change the recipe for *all* arms or none.

---

## Check for Understanding

**Adam is a per-coordinate signal-to-noise detector that takes steps of size $\eta$ in the sign direction of the gradient, its entire memory spans $1/(1-\beta_2)$ steps and contains nothing about validation, and a learning-rate schedule is a cooling schedule for the temperature $\eta/B$ that determines both how tightly you converge and how sharp a minimum you converge to.**

---

**Next:** [Chapter 06 — Neural Networks & Backpropagation](06-neural-networks-and-backpropagation.md)
