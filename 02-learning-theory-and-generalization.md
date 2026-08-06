# Chapter 2 — Learning Theory & Generalization

> **Prerequisites:** Ch. 1 (probability, expectation, variance).
> **Payoff:** this is the chapter that makes double descent (Ch. 18) shocking rather than confusing. You cannot appreciate why deep learning broke the theory without knowing what the theory said.

---

## 2.1 The setup: empirical risk minimization

### The one-line idea

You want low error on data you've never seen. You can only measure error on data you have. ERM is the decision to optimize the thing you can measure and hope it transfers.

### The analogy

You're a chef hiring based on a tasting. The tasting score is *empirical risk*. How good the food will be over a year of service is *true risk*. A chef who memorized one dish perfectly will ace the tasting and fail the job. That's overfitting, and everything in this chapter is about bounding the gap between the tasting and the job.

### Formalism

Data $(x,y) \sim \mathcal{D}$, unknown. Hypothesis class $\mathcal{H}$ (e.g., all networks of a given architecture). Loss $\ell(h(x),y)$.

$$\underbrace{R(h) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(h(x),y)]}_{\text{true / population risk}} \qquad \underbrace{\hat R_n(h) = \frac1n\sum_{i=1}^n \ell(h(x_i),y_i)}_{\text{empirical risk}}$$

▸ ERM: $\hat h = \arg\min_{h\in\mathcal{H}} \hat R_n(h)$.

The **generalization gap** is $R(\hat h) - \hat R_n(\hat h)$. All of classical learning theory is the project of bounding this.

### The error decomposition

$$R(\hat h) - R^* = \underbrace{\big(R(\hat h) - \inf_{h\in\mathcal{H}} R(h)\big)}_{\text{estimation error}} + \underbrace{\big(\inf_{h\in\mathcal{H}} R(h) - R^*\big)}_{\text{approximation error}}$$

- **Approximation error** shrinks as $\mathcal{H}$ grows. (Bigger models can express more.)
- **Estimation error** *classically* grows as $\mathcal{H}$ grows. (Bigger models overfit more.)

The classical story says these trade off, and there's a sweet spot. Chapter 18 shows that the second claim is false for modern models. Hold that thought.

---

## 2.2 The bias–variance decomposition, derived

### The claim

For squared loss and a *fixed* test point $x$, with randomness over the draw of the training set $S$:

▸ $$\mathbb{E}_S\big[(y - \hat f_S(x))^2\big] = \underbrace{\big(\bar f(x) - f^*(x)\big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat f_S(x) - \bar f(x))^2\big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

where $\bar f(x) = \mathbb{E}_S[\hat f_S(x)]$ is the average prediction over training sets, $f^*$ is the true regression function, and $y = f^*(x)+\varepsilon$ with $\mathbb{E}[\varepsilon]=0$, $\mathrm{Var}(\varepsilon)=\sigma^2$.

### The derivation (do this once, it's short)

Write $\hat f = \hat f_S(x)$, $f^* = f^*(x)$.

$$
\mathbb{E}_{S,\varepsilon}\big[(y-\hat f)^2\big] = \mathbb{E}\big[(f^* + \varepsilon - \hat f)^2\big]
$$

Expand, using $\mathbb{E}[\varepsilon]=0$ and $\varepsilon \perp \hat f$:

$$= \mathbb{E}[(f^*-\hat f)^2] + 2\underbrace{\mathbb{E}[\varepsilon]}_{0}\mathbb{E}[f^*-\hat f] + \mathbb{E}[\varepsilon^2] = \mathbb{E}_S[(f^*-\hat f)^2] + \sigma^2$$

Now insert $\pm\bar f$ into the first term:

$$\mathbb{E}_S[(f^* - \bar f + \bar f - \hat f)^2] = (f^*-\bar f)^2 + 2(f^*-\bar f)\underbrace{\mathbb{E}_S[\bar f - \hat f]}_{=0} + \mathbb{E}_S[(\bar f - \hat f)^2]$$

$$= \mathrm{Bias}^2 + \mathrm{Variance}. \qquad \blacksquare$$

### Reading it

- **Bias**: are you systematically wrong even with infinite data from this hypothesis class? A linear model fitting a sine wave has bias.
- **Variance**: how much does your fitted function jiggle when you resample the training set? A degree-20 polynomial on 20 points has enormous variance.
- **Noise $\sigma^2$**: irreducible. Also called the Bayes error. **In a diffusion model this is $H(p_{\text{data}}\mid t)$** — the part of the data that is genuinely ambiguous at noise level $t$. No model reaches below it.

### Important caveats people get wrong

1. **This decomposition is specific to squared loss.** For 0-1 loss and cross-entropy there are analogous decompositions but they don't split as cleanly, and the "variance" term can be *helpful* (averaging over an ensemble can push a point across a decision boundary in the right direction).
2. **The classical U-shaped curve is a claim about how bias and variance move with capacity, not a theorem.** The decomposition itself is exact and always true. The U-shape is empirical folklore that turns out to be a special case (Ch. 18).
3. Variance is measured over *training set resampling*, which is precisely what the bootstrap estimates (Ch. 3). That's not a coincidence — bagging works by driving the variance term down.

### A number

Consider fitting polynomials of degree $d$ to $n=20$ noisy points from $f^*(x)=\sin(2\pi x)$, $\sigma=0.3$.

| $d$ | Bias² | Variance | Total (+ $\sigma^2 = 0.09$) |
|---|---|---|---|
| 1 | 0.31 | 0.01 | 0.41 |
| 3 | 0.02 | 0.03 | 0.14 |
| 9 | 0.001 | 0.19 | 0.28 |
| 19 | 0.000 | 4.8 | 4.9 |

At $d=19$ (= $n-1$, exact interpolation) the variance explodes. **This is the interpolation threshold.** Classical theory stops the story here. Chapter 18 continues it past $d = 20$, where something surprising happens.

---

## 2.3 Concentration: why finite samples say anything at all

### Hoeffding's inequality

For independent $X_i \in [a,b]$ with mean $\mu$:

▸ $$\Pr\left(\left|\tfrac1n\sum X_i - \mu\right| \ge \epsilon\right) \le 2\exp\!\left(\frac{-2n\epsilon^2}{(b-a)^2}\right)$$

**Reading:** the probability of being wrong by $\epsilon$ decays *exponentially in $n$* and in $\epsilon^2$. Invert it: with probability $1-\delta$,

$$\left|\hat R_n(h) - R(h)\right| \le (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}$$

**Number:** for a bounded loss in $[0,1]$, $n=1024$ samples, $\delta=0.05$: the bound is $\sqrt{\log(40)/2048} = \sqrt{3.69/2048} = 0.042$. So a single evaluation on 1,024 samples pins the true loss to within $\pm 0.042$ **at 95% confidence, worst case.**

▸ **This is directly relevant to you.** Your validation uses ~1,024 molecules. Hoeffding's worst-case bound is $\pm0.042$; the improvement you measured was $1.556 \to 1.524 = 0.032$. The improvement is *smaller than the worst-case error bar on a single measurement.* Chapter 3 does the sharper, variance-based version, but the order of magnitude is the point: **this is an experiment operating at the edge of its own measurement precision.**

### Union bound and the first real generalization bound

For a *finite* hypothesis class $|\mathcal{H}| = M$, apply Hoeffding to each $h$ and union-bound:

$$\Pr\left(\exists h: |\hat R_n(h)-R(h)| \ge \epsilon\right) \le 2M e^{-2n\epsilon^2}$$

Set the RHS to $\delta$ and solve:

▸ $$R(h) \le \hat R_n(h) + \sqrt{\frac{\log M + \log(2/\delta)}{2n}} \quad \forall h\in\mathcal{H},\ \text{w.p. } 1-\delta$$

**Read the shape:** generalization gap $\sim \sqrt{\frac{\text{complexity}}{n}}$. Every classical bound has this form. $\log M$ is "bits needed to describe the hypothesis" — a **description-length** measure. This connects directly to PAC-Bayes and MDL below.

**Number:** a network with $p$ parameters at 32-bit precision has $M \le 2^{32p}$, so $\log M \le 32p\log 2 = 22p$. For $p = 10^7$ and $n=10^5$: bound $= \sqrt{2.2\times10^8/2\times10^5} = \sqrt{1100} = 33$. The bound says "your error is at most your training error plus 33," on a loss bounded by 1. **The bound is vacuous.** This is not a small technical problem — it is *the* problem with classical theory applied to deep nets.

---

## 2.4 VC dimension

### The one-line idea

VC dimension measures how many points a hypothesis class can label *arbitrarily* — a capacity measure that doesn't care about how many parameters you have, only about how expressive you actually are.

### The analogy

Think of a hypothesis class as a set of cookie cutters. VC dimension is the largest number of scattered crumbs such that, no matter which subset you want inside the cutter and which outside, some cutter in your set can do it.

### Definition

$\mathcal{H}$ **shatters** a set of $m$ points if for all $2^m$ labelings, some $h\in\mathcal{H}$ realizes it. $\mathrm{VC}(\mathcal{H})$ = largest $m$ shatterable.

**Examples:**
- Thresholds on $\mathbb{R}$ ($h(x) = \mathbb{1}[x>a]$): VC = 1.
- Intervals on $\mathbb{R}$: VC = 2.
- Linear classifiers in $\mathbb{R}^d$ (with bias): VC = $d+1$.
- Axis-aligned rectangles in $\mathbb{R}^2$: VC = 4.
- $h(x)=\mathbb{1}[\sin(\omega x)>0]$, one parameter $\omega$: **VC = $\infty$.** ⇒ *parameter count is not capacity.* Remember this when someone says "more parameters = more overfitting."

### The VC bound

With probability $\ge 1-\delta$, for all $h$:

▸ $$R(h) \le \hat R_n(h) + \sqrt{\frac{8\big(d_{\mathrm{VC}}\log\frac{2en}{d_{\mathrm{VC}}} + \log\frac{4}{\delta}\big)}{n}}$$

**Sauer–Shelah lemma** is the technical engine: a class with VC dimension $d$ can realize at most $\sum_{i=0}^{d}\binom{n}{i} \le (en/d)^d$ labelings of $n$ points — *polynomial*, not exponential. That polynomial growth is what saves the union bound.

### VC dimension of neural networks

For a ReLU network with $W$ weights and $L$ layers:

$$\mathrm{VC} = O(WL\log W)$$

**Number:** a modest network with $W=10^7$, $L=12$: $\mathrm{VC} \approx 10^7 \cdot 12 \cdot 16 = 1.9\times10^9$. With $n=10^5$ training points, the bound gives $\sqrt{1.9\times10^9/10^5} \approx 138$. Vacuous by two orders of magnitude.

▸ **The Zhang et al. (2017) experiment.** Take CIFAR-10, replace all labels with random noise, train a standard ResNet. It reaches **zero training error**. So $\mathcal{H}$ shatters 50,000 points; VC dimension is at least 50,000; uniform-convergence bounds are useless. And yet *the same architecture trained on real labels generalizes fine.* Conclusion: generalization in deep learning is **not** a property of the hypothesis class. It's a property of the class *plus the optimizer* — which is why Chapter 19 (implicit bias) exists.

---

## 2.5 Rademacher complexity

### The one-line idea

Instead of counting hypotheses, directly measure: how well can my model class fit *pure noise*? That number is the complexity.

### Definition

Given samples $S = \{z_1,\dots,z_n\}$ and Rademacher variables $\sigma_i \in \{\pm1\}$ uniform:

▸ $$\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma\left[\sup_{f\in\mathcal{F}}\frac1n\sum_{i=1}^n \sigma_i f(z_i)\right]$$

This is a *correlation with random labels*, maximized over the class. If $\mathcal{F}$ can fit any noise, $\hat{\mathfrak{R}} \to 1$. If $\mathcal{F}$ is a single constant function, $\hat{\mathfrak{R}} = 0$.

### The bound

▸ $$R(f) \le \hat R_n(f) + 2\mathfrak{R}_n(\mathcal{F}) + 3\sqrt{\frac{\log(2/\delta)}{2n}}$$

**Why it's better than VC:** it's *data-dependent* and *scale-sensitive*. VC dimension of linear classifiers in $\mathbb{R}^d$ is $d+1$ regardless of margins; Rademacher complexity of *norm-bounded* linear predictors is

$$\mathfrak{R}_n(\{x\mapsto \langle w,x\rangle : \|w\|\le B\}) \le \frac{B\max_i\|x_i\|}{\sqrt n}$$

— **independent of dimension $d$.** This is why SVMs generalize in infinite-dimensional feature spaces. Norm, not dimension, is the capacity that matters.

### Contraction and layer-wise bounds for networks

Talagrand's contraction lemma: if $\phi$ is $L$-Lipschitz, $\mathfrak{R}(\phi\circ\mathcal{F}) \le L\,\mathfrak{R}(\mathcal{F})$. ReLU is 1-Lipschitz, so it's free.

Chaining through $L$ layers with spectral norm bounds gives things like

$$\mathfrak{R}_n \lesssim \frac{\prod_{\ell=1}^L \|W_\ell\|_2 \cdot \left(\sum_\ell \frac{\|W_\ell\|_{2,1}^{2/3}}{\|W_\ell\|_2^{2/3}}\right)^{3/2}}{\sqrt n}$$

(Bartlett–Foster–Telgarsky). This is *norm-based* and can be non-vacuous for small networks. For large ones it's still typically vacuous, because $\prod\|W_\ell\|_2$ is astronomically large in practice.

---

## 2.6 PAC-Bayes — the bound that actually works

### The one-line idea

Don't bound the risk of a single hypothesis. Bound the *average* risk of a distribution over hypotheses, and pay a KL-divergence price for how far that distribution moved from your prior.

### The analogy

A prior is a codebook you agreed on before seeing data. The posterior is where you actually ended up. The KL term is the number of extra bits you need to transmit to tell someone where you ended up, given the codebook. **The fewer extra bits, the more you can trust the result** — because a hypothesis you could have described cheaply in advance is less likely to be a coincidence.

### The McAllester bound

For a prior $P$ over $\mathcal{H}$ chosen *before* seeing data, and any posterior $Q$ (which may depend on data), with probability $\ge 1-\delta$:

▸ $$\mathbb{E}_{h\sim Q}[R(h)] \le \mathbb{E}_{h\sim Q}[\hat R_n(h)] + \sqrt{\frac{\mathrm{KL}(Q\|P) + \log\frac{2\sqrt n}{\delta}}{2n}}$$

### Why this one succeeds where VC fails

- $\mathrm{KL}(Q\|P)$ measures **how much information the training data injected into the parameters**, not how many parameters exist. A 100M-parameter network that barely moved from initialization has small KL.
- It's naturally connected to **flat minima**: if the loss is flat around $\hat\theta$, you can use a wide $Q$ (large-variance Gaussian around $\hat\theta$) without hurting $\mathbb{E}_Q[\hat R_n]$. A wide $Q$ has *small* $\mathrm{KL}$ to a wide prior. **So flatness literally buys you a tighter generalization bound.** This is the rigorous version of "flat minima generalize better" (Ch. 19).

Formally: let $Q = \mathcal{N}(\hat\theta, \sigma^2 I)$, $P = \mathcal{N}(0,\sigma_0^2 I)$. Then
$$\mathrm{KL}(Q\|P) = \frac{\|\hat\theta\|^2}{2\sigma_0^2} + \frac{p}{2}\left(\frac{\sigma^2}{\sigma_0^2} - 1 - \log\frac{\sigma^2}{\sigma_0^2}\right)$$
The larger $\sigma$ you can tolerate (i.e. the flatter the minimum), the closer $\sigma \to \sigma_0$ and the smaller the second term. And the whole bound only depends on $\|\hat\theta\|^2$ — **weight norm**, not parameter count.

▸ **Dziugaite & Roy (2017)** optimized this bound directly and obtained the first **non-vacuous** generalization bound for a real neural network on MNIST (bound ≈ 0.16 error, actual ≈ 0.03). Not tight, but finite — a genuine milestone.

### Connection to MDL / Occam

$\mathrm{KL}(Q\|P)$ has units of nats = information. The bound says:

> generalization gap $\lesssim \sqrt{\dfrac{\text{bits to describe your model given a prior}}{n}}$

This is Occam's razor made quantitative, and it's the same $\log M$ from §2.3 in continuous clothing.

---

## 2.7 Uniform convergence may be unable to explain deep learning

Nagarajan & Kolter (2019) constructed settings where the learned classifier generalizes well, but **any** uniform-convergence bound (over any hypothesis set containing the learned predictors) must be vacuous. The mechanism: SGD's solutions have a complicated data-dependent boundary that hugs the training points; there exists a set of "adversarial" resampled datasets on which the same solution misclassifies badly, which uniform convergence must account for.

**The takeaway that should reshape your intuition:**

▸ Generalization in deep learning is **not** explained by the hypothesis class. It is explained by the *combination* of (architecture, optimizer, initialization, data). The optimizer's implicit bias is a first-class citizen, not an implementation detail. That's why Ch. 19 spends so long on it.

---

## 2.8 The classical picture in one figure (in words)

```
Test error
   ^
   |  \                              CLASSICAL VIEW
   |   \                          (what you were taught)
   |    \        _____
   |     \      /
   |      \    /
   |       \__/  <- "sweet spot"
   |
   +---------------------------------> model capacity
      underfit   optimal    overfit
```

Everything on this page is correct and rigorously derived. And the picture is **incomplete**. Extend the x-axis far enough right and the curve comes back down. That's Chapter 18.

---

## 2.9 Practical translation

What actually controls generalization in a deep model you're training today, ranked by effect size:

1. **Amount and quality of data.** Dominates everything.
2. **Data augmentation / the effective size of the training distribution.**
3. **Optimizer + LR + batch size** (implicit regularization; $\eta/B$ is a temperature — Ch. 19).
4. **Weight decay** (directly shrinks $\|\theta\|$, which is the quantity in the PAC-Bayes bound).
5. **Architecture priors** (convolution = translation equivariance; attention = permutation equivariance + learned structure).
6. **Explicit regularizers** (dropout, label smoothing).
7. **Parameter count.** Weakest and most non-monotone of the lot.

Case Study A uses `weight_decay=0.01` with AdamW at `lr=3e-4`. In AdamW the decay is decoupled, so the per-step shrinkage is exactly $\theta \leftarrow (1-\eta\lambda)\theta = (1 - 3\times10^{-6})\theta$. Over 2,274 steps/epoch that's a factor $(1-3\times10^{-6})^{2274} = 0.9932$ per epoch of pure shrinkage — about **0.68% of the weight norm removed per epoch**, in the absence of gradients. Over 43 epochs, if gradients were zero, weights would shrink to $0.9932^{43} = 74\%$ of their initial norm. That's a real, non-negligible regularization pressure — and, per PAC-Bayes, it is *directly* the quantity your generalization bound depends on.

---

## Check for Understanding

**Classical theory bounds the generalization gap by the complexity of what your model *could* have done, which for deep networks is so large the bounds say nothing — so the real explanation must live in what the optimizer *actually* does, which is why implicit bias, flatness, and description length turn out to matter more than parameter count.**

---

**Next:** [Chapter 03 — Resampling & the Statistics of Noisy Evaluation](03-resampling-and-noisy-evaluation.md)
