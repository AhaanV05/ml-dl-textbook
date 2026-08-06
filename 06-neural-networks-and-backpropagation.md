# Chapter 6 — Neural Networks & Backpropagation

> **Prerequisites:** Ch. 1 (§1.2 matrix calculus).
> **Goal:** derive backprop from nothing, derive He/Xavier initialization from variance propagation, and be able to say *numerically* why a 50-layer plain network fails to train.

---

## 6.1 What a neural network is

### The one-line idea

A neural network is a composition of affine maps and elementwise nonlinearities, and the nonlinearity is the entire reason it isn't just one big matrix.

### The analogy

Think of a factory line. Each station takes a batch of parts (a vector), mixes and reweighs them (matrix multiply), then applies a simple pass/fail gate to each item (activation). The mixing lets stations combine information; the gate is what stops the whole line from collapsing into a single mixing step.

### The math

$$h^{(0)} = x,\qquad z^{(\ell)} = W^{(\ell)}h^{(\ell-1)} + b^{(\ell)},\qquad h^{(\ell)} = \sigma(z^{(\ell)}),\qquad \hat y = h^{(L)}$$

▸ **Without $\sigma$**, $\hat y = W^{(L)}\cdots W^{(1)}x = \tilde Wx$: an $L$-layer linear network computes exactly the same function class as a 1-layer one. (It optimizes differently — deep linear networks have interesting implicit bias — but it *expresses* nothing more.)

### Universal approximation, and why it's less impressive than it sounds

**Theorem (Cybenko '89, Hornik '91):** a one-hidden-layer network with a non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy, given enough width.

**Why it doesn't explain deep learning:**
- It's an *existence* result. It says nothing about whether SGD finds those weights.
- The required width can be exponential in the input dimension.
- It gives no reason to prefer depth.

**The depth results are more informative.** Telgarsky (2016): there are functions computable by a network of depth $O(k^2)$ and width $O(1)$ that require width $\Omega(2^k)$ at depth $O(k)$. Depth buys **exponential** expressivity for certain compositional functions. A ReLU network with $L$ layers of width $d$ can carve input space into $O((d/n_{\text{in}})^{(L-1)n_{\text{in}}}d^{n_{\text{in}}})$ linear regions — exponential in depth, polynomial in width.

▸ **The honest summary:** depth is efficient for functions that are *themselves* compositional — which real-world data (edges → textures → parts → objects; atoms → functional groups → scaffolds → molecules) tends to be. It's a match between architecture and the structure of reality, not a universal advantage.

---

## 6.2 Backpropagation, derived properly

### The one-line idea

Compute the derivative of the loss with respect to every parameter by applying the chain rule once, backwards, reusing every intermediate result.

### The analogy

Blame assignment in an organization. The CEO knows the company lost $1M. Rather than each employee independently tracing their impact on the bottom line (forward mode: expensive, $p$ separate traces), the blame flows *down* the hierarchy: each manager receives their share of blame and splits it among reports in proportion to their contribution. One pass, everyone gets their number.

### Setup and the key quantity

Define the **error signal** at layer $\ell$:

▸ $$\delta^{(\ell)} := \frac{\partial\mathcal{L}}{\partial z^{(\ell)}} \in \mathbb{R}^{n_\ell}$$

This is the thing that propagates. Once you have $\delta^{(\ell)}$, the parameter gradients are immediate.

### Derivation, index notation

**Base case** (output layer). For softmax + cross-entropy (Ch. 1 §1.3.4):
$$\delta^{(L)}_j = p_j - \mathbb{1}[j=y]$$

For MSE with linear output: $\delta^{(L)} = 2(\hat y - y)$.

**Recursion.** The loss depends on $z^{(\ell)}$ only through $h^{(\ell)} = \sigma(z^{(\ell)})$, which feeds $z^{(\ell+1)} = W^{(\ell+1)}h^{(\ell)}+b^{(\ell+1)}$. Chain rule:

$$
\delta^{(\ell)}_j = \frac{\partial\mathcal{L}}{\partial z^{(\ell)}_j} = \sum_k \frac{\partial\mathcal{L}}{\partial z^{(\ell+1)}_k}\cdot\frac{\partial z^{(\ell+1)}_k}{\partial h^{(\ell)}_j}\cdot\frac{\partial h^{(\ell)}_j}{\partial z^{(\ell)}_j}
= \sum_k \delta^{(\ell+1)}_k W^{(\ell+1)}_{kj}\,\sigma'(z^{(\ell)}_j)
$$

▸ In matrix form:
$$\boxed{\ \delta^{(\ell)} = \left(W^{(\ell+1)\top}\delta^{(\ell+1)}\right)\odot\sigma'(z^{(\ell)})\ }$$

**Parameter gradients.** Since $z^{(\ell)}_j = \sum_i W^{(\ell)}_{ji}h^{(\ell-1)}_i + b^{(\ell)}_j$:

$$\frac{\partial\mathcal{L}}{\partial W^{(\ell)}_{ji}} = \delta^{(\ell)}_j h^{(\ell-1)}_i,\qquad \frac{\partial\mathcal{L}}{\partial b^{(\ell)}_j} = \delta^{(\ell)}_j$$

▸ $$\boxed{\ \nabla_{W^{(\ell)}}\mathcal{L} = \delta^{(\ell)}h^{(\ell-1)\top},\qquad \nabla_{b^{(\ell)}}\mathcal{L}=\delta^{(\ell)}\ }$$

That's the entirety of backpropagation. Three boxed equations.

### The batched version

With a batch of $B$ examples, $H^{(\ell)}\in\mathbb{R}^{n_\ell\times B}$, $\Delta^{(\ell)}\in\mathbb{R}^{n_\ell\times B}$:

$$\Delta^{(\ell)} = \left(W^{(\ell+1)\top}\Delta^{(\ell+1)}\right)\odot\sigma'(Z^{(\ell)}),\qquad \nabla_{W^{(\ell)}}\mathcal{L} = \frac1B\Delta^{(\ell)}H^{(\ell-1)\top}$$

The outer product becomes a matmul, summing over the batch. Note the $1/B$: **your gradient is a mean, not a sum**, which is why loss scales are batch-size-independent but gradient *noise* isn't.

### Computational cost

Forward through layer $\ell$: $O(n_\ell n_{\ell-1}B)$ FLOPs (× 2 for multiply-add).
Backward: **two** matmuls of the same size ($\delta$ propagation and weight gradient).

▸ **Backward ≈ 2× forward.** Total training step ≈ 3× a forward pass. This is where the ubiquitous $C \approx 6ND$ FLOP formula comes from (Ch. 21): 2 FLOPs per parameter per token forward, 4 backward, 6 total.

**Memory** is the real constraint: you must store every $h^{(\ell)}$ (or $z^{(\ell)}$) from the forward pass to compute the backward pass. Activation memory $\approx B\sum_\ell n_\ell$ — which for a transformer at long sequence length dwarfs the parameters. Gradient checkpointing (Ch. 21) trades recompute for memory.

### Reverse-mode AD in general

Backprop is reverse-mode automatic differentiation applied to a layered graph. The general algorithm:
1. Build a DAG of primitive operations during the forward pass (the "tape").
2. Each primitive registers a **VJP rule**: given $\bar y$, return $\bar x = (\partial y/\partial x)^\top\bar y$.
3. Traverse the DAG in reverse topological order, accumulating $\bar{\cdot}$ at each node (**summing** contributions when a node has multiple consumers — this is why residual connections add gradients from both paths).

Some VJP rules to have memorized:

| Forward | VJP |
|---|---|
| $y=Wx$ | $\bar W = \bar yx^\top$, $\bar x = W^\top\bar y$ |
| $y=x_1+x_2$ | $\bar x_1=\bar x_2=\bar y$ (gradient **copies**) |
| $y=x_1\odot x_2$ | $\bar x_1 = \bar y\odot x_2$ |
| $y=\sigma(x)$ | $\bar x = \bar y\odot\sigma'(x)$ |
| $y=\mathrm{softmax}(x)$ | $\bar x = y\odot(\bar y - \langle\bar y,y\rangle)$ |
| $y = x/\|x\|$ | $\bar x = (\bar y - y\langle y,\bar y\rangle)/\|x\|$ |

Note the addition rule: **a residual connection is a gradient highway** because $+$ passes $\bar y$ through unchanged. That's the whole trick of ResNets (Ch. 8).

---

## 6.3 Vanishing and exploding gradients, quantified

Unroll the recursion:

$$\delta^{(1)} = \left(\prod_{\ell=2}^{L}W^{(\ell)\top}D^{(\ell)}\right)\delta^{(L)},\qquad D^{(\ell)} = \mathrm{diag}(\sigma'(z^{(\ell)}))$$

▸ The gradient at layer 1 is a product of $L-1$ matrices. Products of matrices behave like products of scalars in log-space: if each contributes an average multiplicative factor $\gamma$, the total is $\gamma^{L}$.

- $\gamma<1$ ⇒ **vanishing**: $\gamma=0.8$, $L=50$ ⇒ $0.8^{50}=1.4\times10^{-5}$. Early layers receive essentially no signal.
- $\gamma>1$ ⇒ **exploding**: $\gamma=1.2$, $L=50$ ⇒ $1.2^{50}=9100$. NaNs.
- $\gamma=1$ is a knife-edge. **Everything about initialization and normalization is the project of engineering $\gamma\approx1$.**

**The sigmoid disaster.** $\sigma(z)=1/(1+e^{-z})$ has $\sigma'(z)=\sigma(1-\sigma)\le 1/4$. So even with perfectly scaled weights, $\gamma \le 0.25$ from the activation alone:
$$0.25^{10} = 9.5\times10^{-7}$$
▸ **A 10-layer sigmoid network cannot be trained by gradient descent.** This single number is why deep learning didn't work before ReLU, and why the 2006–2012 era needed layerwise pretraining.

---

## 6.4 Initialization, derived

### The principle

Choose the initial weight distribution so that **the variance of activations is preserved going forward and the variance of gradients is preserved going backward.**

### Forward pass analysis

Let $z_j = \sum_{i=1}^{n_{\text{in}}}W_{ji}h_i$ with $W_{ji}$ i.i.d. mean-0 variance $\sigma_W^2$, independent of $h$, and $h_i$ i.i.d. with variance $\sigma_h^2$ and mean 0.

$$\mathrm{Var}(z_j) = \sum_{i=1}^{n_{\text{in}}}\mathrm{Var}(W_{ji}h_i) = n_{\text{in}}\,\sigma_W^2\,\sigma_h^2$$

(Using $\mathrm{Var}(XY)=\mathbb{E}[X^2]\mathbb{E}[Y^2]-\mathbb{E}[X]^2\mathbb{E}[Y]^2 = \sigma_W^2\sigma_h^2$ for independent zero-mean $X,Y$.)

▸ To preserve variance ($\mathrm{Var}(z)=\sigma_h^2$) we need $\boxed{\sigma_W^2 = 1/n_{\text{in}}}$.

### Backward pass analysis

$\delta^{(\ell)} = W^{(\ell+1)\top}\delta^{(\ell+1)}\odot\sigma'$. The same argument with the transpose gives

▸ $$\sigma_W^2 = 1/n_{\text{out}}$$

### Xavier/Glorot: compromise

Can't satisfy both unless $n_{\text{in}}=n_{\text{out}}$. Take the harmonic-mean-flavoured compromise:

▸ $$\sigma_W^2 = \frac{2}{n_{\text{in}}+n_{\text{out}}}$$

Uniform version: $W\sim\mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right]$ (because $\mathrm{Var}(\mathcal{U}[-a,a])=a^2/3$).

**Assumes $\sigma'\approx1$ near zero** — true for tanh, false for ReLU.

### He/Kaiming: the ReLU correction

ReLU zeroes half its inputs. For $z$ symmetric about 0:
$$\mathbb{E}[\mathrm{ReLU}(z)^2] = \mathbb{E}[z^2\mathbb{1}(z>0)] = \tfrac12\mathbb{E}[z^2]$$

▸ **ReLU halves the variance.** Compensate by doubling:

$$\boxed{\ \sigma_W^2 = \frac{2}{n_{\text{in}}}\ }\qquad W\sim\mathcal{N}\left(0,\tfrac{2}{n_{\text{in}}}\right)$$

**The empirical check that made this famous:** He et al. showed a 30-layer plain ReLU network with Xavier init **does not converge at all**, while the same network with He init trains fine. The difference is a factor of $\sqrt2$ per layer: $(1/\sqrt2)^{30} = 3\times10^{-5}$ — the activations die out by layer 30. **One factor of two, compounded 30 times, is the difference between working and not working.**

**Numbers.** Width 1024, He init: $\sigma_W = \sqrt{2/1024} = 0.0442$. A typical weight is $\pm0.044$. Your AdamW step of $3\times10^{-4}$ is thus $0.7\%$ of a typical weight per step. Useful calibration.

### Other schemes you'll encounter

| Scheme | $\sigma_W^2$ | Use |
|---|---|---|
| LeCun | $1/n_{\text{in}}$ | SELU, self-normalizing nets |
| Xavier | $2/(n_{\text{in}}+n_{\text{out}})$ | tanh, sigmoid |
| He | $2/n_{\text{in}}$ | ReLU family |
| Orthogonal | $Q$ from QR of Gaussian, scaled by gain | RNNs; preserves norm *exactly*, not just in expectation |
| **Zero-init on residual branch** | last layer of each block $=0$ | **critical** — see below |

▸ **Zero-init of residual branches** (Fixup, ReZero, and the $\gamma$ in **AdaLN-Zero** used by DiT — Ch. 13): initialize the final projection of each residual block to zero, so the block initially computes the identity. The network starts as a shallow, well-conditioned function and *grows* depth as training proceeds. This removes the need for warmup in some settings and is the single most important initialization trick in modern generative architectures. **If you are training a DiT, you are already using this in AdaLN-Zero — and if you modified the architecture and broke it, that would show up as early instability.**

### Scaling by depth

For residual networks, the variance grows *additively* with depth: $\mathrm{Var}(h^{(L)}) = \mathrm{Var}(h^{(0)}) + \sum_\ell\mathrm{Var}(F_\ell)$, so it grows like $L$. Common fixes: scale residual branch outputs by $1/\sqrt{L}$ (used in GPT-2's init: final projections scaled by $1/\sqrt{2L}$), or use zero-init as above.

---

## 6.5 Activation functions

| Name | $f(z)$ | $f'(z)$ | Notes |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $f(1-f)\le\frac14$ | saturating, non-zero-centered. Avoid except as a gate. |
| Tanh | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1-f^2\le1$ | zero-centered sigmoid. Fine in RNNs. |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | fast, sparse, no saturation for $z>0$. **Dying ReLU.** |
| Leaky ReLU | $\max(\alpha z,z)$, $\alpha=0.01$ | $\alpha$ or 1 | fixes dying ReLU |
| PReLU | learnable $\alpha$ | | slight gain, extra params |
| ELU | $z$ or $\alpha(e^z-1)$ | | smooth, negative saturation, mean closer to 0 |
| GELU | $z\Phi(z)$ | $\Phi(z)+z\phi(z)$ | **transformer default** |
| SiLU/Swish | $z\sigma(z)$ | $\sigma(z)(1+z(1-\sigma(z)))$ | ≈GELU, cheaper |
| Mish | $z\tanh(\mathrm{softplus}(z))$ | | marginal gains, slower |
| GLU family | $ (Wx)\odot\sigma(Vx)$ | | **SwiGLU** is the LLM standard |

### GELU, properly

▸ $$\mathrm{GELU}(z) = z\cdot\Phi(z) = z\cdot\tfrac12\left[1+\mathrm{erf}\!\left(\tfrac{z}{\sqrt2}\right)\right]$$

**Interpretation:** it's a *stochastic regularizer made deterministic*. Consider multiplying $z$ by a Bernoulli mask with probability $\Phi(z)$ (so large inputs are kept more often — "adaptive dropout"). GELU is the expectation of that: $\mathbb{E}[z\cdot m] = z\Phi(z)$.

Tanh approximation (what most code uses):
$$\mathrm{GELU}(z)\approx 0.5z\left(1+\tanh\left[\sqrt{2/\pi}\left(z+0.044715z^3\right)\right]\right)$$

**Why it beats ReLU in transformers:** it's smooth (better-behaved second derivative for Adam's preconditioner), and it's non-monotone near zero, which lets it represent slightly richer local behaviour. The gain is small but consistent (~0.5% perplexity).

### SwiGLU

$$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(W_1x)\big)\odot(W_2x),\quad \text{then } W_3(\cdot)$$

Three matrices instead of two, so to keep parameter count fixed you shrink the hidden dim by $2/3$: $d_{\text{ff}} = \frac{8}{3}d_{\text{model}}$ instead of $4d_{\text{model}}$. **Consistently ~1% better perplexity at matched params.** Used in LLaMA, PaLM, Mixtral. Noam Shazeer's paper famously concludes with "we attribute their success to divine benevolence" — which is honest, since nobody has a clean theory for why gating helps.

### Dying ReLU, quantified

If a unit's pre-activation is negative for every training example, its gradient is exactly zero forever — it's dead. With He init and no bias, roughly 50% of units are inactive *per example*, which is fine (that's sparsity). Permanent death happens when a large gradient step pushes the bias very negative. **Rate:** with a too-high LR, 20–40% of ReLU units can die in the first few hundred steps.

Fixes: leaky/ELU/GELU, lower LR, proper init, or normalization layers (which recenter pre-activations every step and make permanent death nearly impossible).

---

## 6.6 Debugging a network with numbers

The diagnostics that find 90% of bugs, in order:

1. **Overfit 1 batch.** Take 8 examples, train until loss $\approx0$. If you can't, you have a bug — not a hyperparameter problem. Do this before *anything* else.
2. **Check initial loss.** For $K$-way classification it should be $\log K$. If a D3PM starts at CE $\ne \log K$ at high $t$, your logits are miscalibrated at init or your loss is wrong.
3. **Activation statistics per layer.** Mean should be ~0 (or ~0.5·std for ReLU), std should be ~constant across depth. If std shrinks 10× per layer, your init is wrong.
4. **Gradient norms per layer.** Should be within ~1 order of magnitude of each other. A 4-order-of-magnitude spread means vanishing gradients.
5. **Update-to-weight ratio** $\frac{\|\eta\Delta\theta_\ell\|}{\|\theta_\ell\|}$. Target $\approx10^{-3}$. Above $10^{-2}$: LR too high. Below $10^{-4}$: that layer is frozen.
6. **Gradient check** (for custom ops): $\frac{\mathcal{L}(\theta+\epsilon e_i)-\mathcal{L}(\theta-\epsilon e_i)}{2\epsilon}$ vs analytic, $\epsilon=10^{-4}$ in **float64**. Relative error should be $<10^{-6}$.

---

## Check for Understanding

**Backpropagation is one recursion, $\delta^{(\ell)} = (W^{(\ell+1)\top}\delta^{(\ell+1)})\odot\sigma'(z^{(\ell)})$, applied backwards; everything difficult about training deep networks is that this recursion multiplies $L$ matrices together, so initialization, normalization, and residual connections all exist to keep that product's scale near one.**

---

**Next:** [Chapter 07 — Normalization & Regularization](07-normalization-and-regularization.md)
