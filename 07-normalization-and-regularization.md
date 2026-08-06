# Chapter 7 — Normalization & Regularization

> **Prerequisites:** Ch. 4 (condition number), Ch. 6 (variance propagation).

---

## 7.1 Why normalization exists

### The one-line idea

Rescale activations so that every layer receives inputs with a controlled mean and variance, which keeps the loss surface well-conditioned and makes the network's behaviour insensitive to the scale of its own weights.

### The analogy

A relay race where each runner hands off a baton. Without normalization, one runner might sprint and the next might crawl, so the baton arrives at wildly varying speed and each subsequent runner has to guess how hard to push. Normalization is a checkpoint after every leg that resets the baton to a standard speed — now every runner can train on the same expectation.

### The original claim and its refutation

BatchNorm was introduced (Ioffe & Szegedy, 2015) to reduce **internal covariate shift** — the changing distribution of layer inputs as earlier layers update. Santurkar et al. (2018) largely **disproved** this: injecting deliberate distribution shift *after* BatchNorm doesn't hurt training, and networks without covariate shift don't train faster.

▸ The better explanation is **smoothing**: normalization reduces the Lipschitz constant of both the loss and its gradient, making the landscape more predictable so larger steps are safe. Formally, with BN the effective gradient magnitude is bounded relative to the un-normalized case, and $\lambda_{\max}(H)$ drops — which is exactly the $\kappa$ from Chapter 4.

The second, and possibly more important, mechanism is **scale invariance**, §7.4.

---

## 7.2 BatchNorm

### The math

For a mini-batch of activations $\{x_i\}_{i=1}^{B}$ at a given feature/channel:

$$\mu_{\mathcal{B}} = \frac1B\sum_i x_i,\qquad \sigma^2_{\mathcal{B}} = \frac1B\sum_i (x_i-\mu_\mathcal{B})^2$$
▸ $$\hat x_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}},\qquad y_i = \gamma\hat x_i + \beta$$

$\gamma,\beta$ are learned per channel, so the network can undo the normalization if it wants — which matters, because forcing zero mean into a ReLU or sigmoid would restrict it to a near-linear regime.

**Placement:** conv → BN → activation. The bias in the conv is redundant (BN's $\beta$ subsumes it); set `bias=False`.

### The backward pass, derived

This is a classic interview question. The subtlety is that $\mu$ and $\sigma^2$ *depend on every $x_i$*, so gradients flow through three paths.

Let $\bar y_i = \partial\mathcal{L}/\partial y_i$. Then $\partial\mathcal{L}/\partial\hat x_i = \bar y_i\gamma$, and:

$$\frac{\partial\mathcal{L}}{\partial\sigma^2} = \sum_i \frac{\partial\mathcal{L}}{\partial\hat x_i}\cdot(x_i-\mu)\cdot\left(-\tfrac12\right)(\sigma^2+\epsilon)^{-3/2}$$
$$\frac{\partial\mathcal{L}}{\partial\mu} = \sum_i \frac{\partial\mathcal{L}}{\partial\hat x_i}\cdot\frac{-1}{\sqrt{\sigma^2+\epsilon}} + \frac{\partial\mathcal{L}}{\partial\sigma^2}\cdot\frac{-2\sum_i(x_i-\mu)}{B}$$

Collecting (the second term of $\partial\mathcal{L}/\partial\mu$ vanishes since $\sum_i(x_i-\mu)=0$):

▸ $$\frac{\partial\mathcal{L}}{\partial x_i} = \frac{1}{B\sqrt{\sigma^2+\epsilon}}\left(B\,\frac{\partial\mathcal{L}}{\partial\hat x_i} - \sum_j\frac{\partial\mathcal{L}}{\partial\hat x_j} - \hat x_i\sum_j \frac{\partial\mathcal{L}}{\partial\hat x_j}\hat x_j\right)$$

Read the three terms: the raw gradient, minus its batch mean, minus its projection onto $\hat x$. **BatchNorm's backward pass removes the component of the gradient that would change the mean or the scale** — those directions are already handled by $\beta$ and $\gamma$. That is why BN makes the layer's weight *magnitude* irrelevant (§7.4).

Also note: $\bar\gamma = \sum_i \bar y_i\hat x_i$, $\bar\beta = \sum_i \bar y_i$.

### The train/eval discrepancy — the biggest practical gotcha

**Training:** uses batch statistics $\mu_\mathcal{B},\sigma^2_\mathcal{B}$, so a sample's output *depends on the other samples in its batch*. This is a form of noise, and it's a real regularizer.

**Inference:** uses running estimates accumulated during training:
$$\mu_{\text{run}} \leftarrow (1-m)\mu_{\text{run}} + m\,\mu_\mathcal{B}$$
with momentum $m$ (PyTorch default 0.1, i.e. horizon $\approx10$ batches).

▸ **Failure modes this creates:**
1. **Small batches.** $B\le 8$: the batch statistics are wildly noisy ($\mathrm{Var}(\hat\sigma^2)\propto 1/B$), training degrades sharply. At $B=2$ BN is nearly useless.
2. **Train/test mismatch.** If the running stats haven't converged (short training, or an LR schedule that moves activations late), eval performance collapses while training loss looks fine. Classic symptom: **model performs well in `train()` mode and badly in `eval()` mode.**
3. **Distribution shift at test time** breaks the frozen statistics. (Test-time BN adaptation — recomputing stats on the test batch — often recovers most of the loss.)
4. **Any per-sample dependency is broken by BN**: it leaks information across the batch, which is a genuine problem for contrastive learning (fixed by ShuffleBN in MoCo) and for RL.
5. **Sequence models.** Sequence length varies and padding pollutes the statistics. This is the main reason transformers use LayerNorm.

### Ghost BatchNorm

Compute BN statistics over *sub-batches* of e.g. 32 even when the real batch is 1024. Preserves the regularizing noise that large-batch training destroys, and recovers part of the large-batch generalization gap.

---

## 7.3 The normalization family

All of them compute $\hat x = (x-\mu)/\sqrt{\sigma^2+\epsilon}$; they differ **only in which axes $\mu,\sigma$ are computed over**. For an activation tensor of shape $(B, C, H, W)$:

| Method | Normalize over | Independent of batch? | Typical use |
|---|---|---|---|
| **BatchNorm** | $(B,H,W)$ per channel | ✗ | CNNs, large batch |
| **LayerNorm** | $(C,H,W)$ per sample | ✓ | Transformers, RNNs |
| **InstanceNorm** | $(H,W)$ per sample per channel | ✓ | style transfer |
| **GroupNorm** | $(C/G, H, W)$ per sample per group | ✓ | detection/segmentation, small batch |
| **RMSNorm** | scale only, no mean | ✓ | modern LLMs |

**GroupNorm** interpolates: $G=1$ is LayerNorm, $G=C$ is InstanceNorm. $G=32$ is standard and matches BN accuracy at $B\ge16$ while *beating* it badly at $B=2$.

### LayerNorm

▸ $$\mathrm{LN}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu=\frac1d\sum_j x_j,\ \ \sigma^2=\frac1d\sum_j(x_j-\mu)^2$$

Per-token, per-sample. No batch dependence, identical in train and eval, works with any sequence length. **This is why it won for transformers.**

**Geometric reading:** LN projects $x$ onto the hyperplane $\sum_j x_j = 0$ (mean removal) and then onto a sphere of radius $\sqrt d$ (variance normalization). So LN maps $\mathbb{R}^d \to$ a $(d-2)$-sphere. Every LayerNorm output has norm exactly $\sqrt d$ before the $\gamma$ scaling. This is worth remembering when reasoning about the residual stream (Ch. 32).

### RMSNorm

▸ $$\mathrm{RMSNorm}(x) = \gamma\odot\frac{x}{\sqrt{\frac1d\sum_j x_j^2 + \epsilon}}$$

Drops the mean subtraction and $\beta$. Rationale (Zhang & Sennrich, 2019): the re-*scaling* is what matters; the re-*centering* contributes little. Saves ~7–10% of layer-norm compute and one reduction pass.

**Used by LLaMA, Mistral, Gemma, and most current LLMs.** Empirically matched quality.

### Where to put it: pre-LN vs post-LN

**Post-LN (original Transformer):** $x_{\ell+1} = \mathrm{LN}(x_\ell + F(x_\ell))$
**Pre-LN (everything modern):** $x_{\ell+1} = x_\ell + F(\mathrm{LN}(x_\ell))$

▸ In pre-LN there is a **clean identity path from input to output with no normalization on it**, so the gradient reaches layer 1 undiminished. In post-LN, every residual addition is followed by a normalization that rescales the gradient, and the product over $L$ layers grows like $O(\sqrt L)$ at initialization — which is why post-LN transformers *require* warmup and pre-LN often doesn't.

**The trade-off:** pre-LN is far more stable but slightly underperforms post-LN at matched size, because the residual stream's magnitude grows with depth and later layers contribute proportionally less. Fixes: **DeepNorm** (post-LN with an $\alpha$-scaled residual and $\beta$-scaled init, allowing 1000-layer transformers), and **sandwich/peri-LN** (normalize both before and after the block).

---

## 7.4 Scale invariance — the deep reason normalization changes optimization

Consider a linear layer immediately followed by BN or LN. Scaling the weights $W\to cW$ scales the pre-activations by $c$, and then normalization divides it right back out:

▸ $$\mathrm{Norm}((cW)x) = \mathrm{Norm}(Wx)$$

**The function is invariant to the weight norm.** Two consequences follow, both important:

**1. The gradient is orthogonal to the weights.** Differentiating $\mathcal{L}(cW)=\mathcal{L}(W)$ w.r.t. $c$ at $c=1$ gives

$$\langle \nabla_W\mathcal{L},\ W\rangle = 0$$

So SGD never changes $\|W\|$ *directly* — only the second-order effect of taking a step perpendicular to $W$ increases its norm:
$$\|W_{t+1}\|^2 = \|W_t\|^2 + \eta^2\|\nabla\mathcal{L}\|^2$$

**The weight norm grows monotonically under plain SGD.**

**2. The gradient scales as $1/\|W\|$.** By homogeneity, $\nabla_{cW}\mathcal{L} = \frac1c\nabla_W\mathcal{L}$. Therefore the *angular* update — the only thing that changes the function — has size

▸ $$\Delta\angle \approx \frac{\eta\|\nabla\mathcal{L}\|}{\|W\|} \propto \frac{\eta}{\|W\|^2}$$

▸ **The effective learning rate is $\eta_{\text{eff}} = \eta/\|W\|^2$.** Weight norm growth is therefore *automatic learning-rate decay*, and **weight decay's real job in a normalized network is not to regularize but to keep $\|W\|$ small so the effective LR stays high.** This resolves a long-standing puzzle: why does weight decay help even when it provably cannot reduce the function class (since the function is scale-invariant)? Because it controls the effective step size. The equilibrium norm where growth and decay balance is
$$\|W\|^4_{\text{eq}} \approx \frac{\eta\,G^2}{2\lambda},\qquad G \equiv \|W\|\cdot\|\nabla_W\mathcal{L}\|\ \ (\text{scale-invariant, so } G \text{ does not depend on } \|W\|)$$
$$\Rightarrow\quad \eta_{\text{eff}} = \frac{\eta}{\|W\|^2} \approx \frac{\sqrt{2\eta\lambda}}{G}\ \propto\ \sqrt{\eta\lambda}$$

*(Derivation: $\|W\|^2$ grows by $\eta^2\|\nabla\mathcal{L}\|^2 = \eta^2G^2/\|W\|^2$ per step and decays by a factor $(1-\eta\lambda)^2 \approx 1-2\eta\lambda$. Setting growth equal to decay gives $2\eta\lambda\|W\|^2 = \eta^2G^2/\|W\|^2$.)*

▸ **So in a normalized network, $\eta$ and $\lambda$ are not independent knobs — only the product $\eta\lambda$ matters** (approximately). This is a genuinely useful, non-obvious, and frequently-asked fact.

**Corollary:** never apply weight decay to LayerNorm/RMSNorm gains or to biases. They are *not* scale-invariant (the gain directly sets the layer's output magnitude), so decaying them shrinks the network's actual output for no benefit.

---

## 7.5 Regularization: the catalogue

### $\ell_2$ / weight decay

Adds $\frac\lambda2\|\theta\|^2$. In the eigenbasis of the Hessian at a minimum, the solution shrinks componentwise:
▸ $$\hat\theta_i^{\text{ridge}} = \frac{\lambda_i}{\lambda_i+\lambda}\hat\theta_i^{\text{OLS}}$$
**Directions with small curvature (poorly determined by the data) are shrunk most.** That is exactly the right behaviour, and it's the cleanest one-line justification of ridge regression. (Full derivation in Ch. 22.)

Bayesian reading: $\ell_2$ = Gaussian prior $\mathcal{N}(0,1/\lambda)$ on weights, MAP estimation.

### $\ell_1$

Adds $\lambda\|\theta\|_1$. Produces **exact zeros** because the subgradient at $0$ is the interval $[-\lambda,\lambda]$ — any coordinate whose data-gradient magnitude is below $\lambda$ is pinned at exactly zero. Geometric reading: the $\ell_1$ ball has corners on the axes, and a randomly-oriented level set is likeliest to first touch a corner.

Bayesian reading: Laplace prior.

### Dropout

▸ Training: $\tilde h = h\odot m/(1-p)$, $m_i\sim\mathrm{Bernoulli}(1-p)$. Inference: identity.

(The $1/(1-p)$ is "inverted dropout" — it keeps $\mathbb{E}[\tilde h]=h$ so no rescaling is needed at test time.)

**Three ways to understand it:**

1. **Ensembling.** A network with $n$ droppable units defines $2^n$ sub-networks sharing weights. Training samples one per step; inference approximates the *geometric mean* of all of them. For a single-layer linear-softmax model this approximation is exact.

2. **Explicit regularization.** For linear regression, dropout on inputs is *exactly* equivalent to $\ell_2$ regularization on a rescaled problem:
$$\mathbb{E}_m\|y - w^\top(m\odot x)/(1-p)\|^2 = \|y-w^\top x\|^2 + \frac{p}{1-p}\sum_j w_j^2\,\mathbb{E}[x_j^2]$$
▸ i.e. **dropout $\equiv$ data-dependent $\ell_2$**, penalizing weights on high-variance features more.

3. **Approximate Bayesian inference** (Gal & Ghahramani). Dropout training is variational inference in a deep Gaussian process. Consequence: **keeping dropout on at test time and averaging $T$ stochastic forward passes gives a usable posterior predictive** — "MC dropout" (Ch. 33).

**Variants:** DropConnect (drop weights not activations), Spatial/2D dropout (drop entire channels — necessary in CNNs, since adjacent pixels are correlated so per-element dropout barely removes information), DropPath / **stochastic depth** (drop whole residual blocks: $x_{\ell+1} = x_\ell + b_\ell F(x_\ell)$, $b_\ell\sim\mathrm{Bern}(1-p_\ell)$ with $p_\ell$ increasing linearly with depth — standard in ViT and ConvNeXt), attention dropout.

▸ **Note:** dropout has largely *disappeared* from large-scale LLM pretraining ($p=0$ is standard) because with a single pass over an enormous dataset there is no overfitting to prevent, and dropout costs capacity. It remains standard in fine-tuning and in small-data regimes. Knowing *why* it was abandoned is a better answer than knowing the formula.

### Label smoothing

▸ $$y^{\text{LS}} = (1-\alpha)y_{\text{one-hot}} + \frac{\alpha}{K}\mathbf{1},\qquad \alpha\approx0.1$$

Prevents the logit gap from diverging: the optimal logit gap becomes finite, $\log\frac{(1-\alpha)(K-1)}{\alpha} + \text{const}$, instead of $\infty$.

**Effects:** better calibration (Ch. 33), better generalization, **tighter clustering of penultimate-layer features** — Müller et al. showed label smoothing induces equidistant class clusters, which is a deliberate acceleration of *neural collapse* (Ch. 31).

**Cost:** it destroys information useful for knowledge distillation. A teacher trained with label smoothing gives worse students, because the relative "dark knowledge" among wrong classes is erased.

### Data augmentation and mixup

Augmentation is the highest-value regularizer in vision and audio, and it works by enlarging the effective support of $p_{\text{data}}$ along directions you know are label-preserving. It is an **encoding of invariances**, i.e. a prior.

**Mixup:** $\tilde x = \lambda x_i + (1-\lambda)x_j$, $\tilde y = \lambda y_i+(1-\lambda)y_j$, $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$.
Forces linear behaviour between examples. Improves calibration and adversarial robustness. **CutMix** pastes a patch instead of blending, preserving local statistics; usually better for classification.

### Early stopping

▸ For a quadratic loss with Hessian eigenvalues $\lambda_i$, gradient descent for $t$ steps at LR $\eta$ from $\theta_0=0$ gives
$$\theta_i(t) = \left(1-(1-\eta\lambda_i)^t\right)\theta_i^*$$
Compare to ridge: $\theta_i^{\text{ridge}} = \frac{\lambda_i}{\lambda_i+\lambda}\theta^*_i$. Matching the two gives $\lambda \approx \frac{1}{\eta t}$.

▸ **Early stopping is $\ell_2$ regularization with $\lambda = 1/(\eta t)$.** Training longer = weaker regularization. This is exact for quadratics and a good heuristic in general, and it's the cleanest available answer to "why does early stopping work?"

### Others worth naming

- **Gradient penalty / spectral normalization** — control the Lipschitz constant directly (GANs, Ch. 19).
- **Noise injection** — Gaussian noise on inputs is equivalent to Tikhonov regularization to second order.
- **Multi-task learning / auxiliary losses** — regularization by shared representation.
- **Batch size and LR** — implicit, and usually stronger than anything on this list (Ch. 4 §4.6).

---

## 7.6 How to choose

| Situation | Use |
|---|---|
| CNN, batch ≥ 32 | BatchNorm |
| CNN, batch < 16 (detection, 3D, video) | GroupNorm |
| Transformer | Pre-LN RMSNorm |
| Very deep transformer (>100 layers) | DeepNorm or per-layer scaled init |
| Small dataset | augmentation ≫ dropout > weight decay |
| LLM pretraining | weight decay only; dropout 0 |
| Fine-tuning | dropout 0.1, low LR, early stop |
| Need calibration | label smoothing or temperature scaling (Ch. 33) |

---

## Check for Understanding

**Normalization's most important effect is not fixing covariate shift but making the layer scale-invariant, which turns the weight norm into an inverse effective learning rate — so in a normalized network, weight decay is a learning-rate control and only the product $\eta\lambda$ matters.**

---

**Next:** [Chapter 08 — Convolutions, ResNets & Vision Architectures](08-convolutions-resnets-vision.md)
