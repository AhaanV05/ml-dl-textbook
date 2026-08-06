# Chapter 30 — Double Descent, Grokking & the NTK

> **Prerequisites:** Ch. 2, Ch. 4, Ch. 22 (§22.2 ridge/SVD).
> **This is where the textbook story of machine learning breaks and the modern one begins.**

---

## 30.1 Double descent

### The one-line idea

The classical U-shaped test-error curve is only the *left half* of the real picture: past the point where the model exactly interpolates the training data, test error starts falling again — and often ends up lower than the classical optimum.

### The analogy

Fitting a curve through noisy points with a bendy wire. With too few bends the wire can't reach the points (underfitting). With *just barely* enough bends, the wire is forced to contort violently to hit every point — it is stretched to its limit and whips wildly between them. That is the interpolation threshold, and it is the worst place to be. With far *more* bends than needed, the wire has slack: it can pass through every point while still lying almost straight. **Excess capacity buys smoothness.**

### The picture

```
Test
error │      ╭─╮  ← the interpolation peak (n ≈ p)
      │  ╭───╯ ╰──╮
      │ ╱          ╰────────────  ← the modern regime
      │╱
      └──────────────────────────→  model capacity
       classical U    n=p     overparameterized
```

▸ **Three variants, all real:**
- **Model-wise:** peak at parameters ≈ samples.
- **Epoch-wise:** test error rises then falls again *during training at fixed model size*, which is why "early stopping is always right" is false.
- **Sample-wise:** ▸ **more data can make test error worse**, if it moves you from the overparameterized regime toward the interpolation threshold. Genuinely counterintuitive, and empirically verified.

### Why the peak exists — the linear analysis

Consider minimum-$\ell_2$-norm interpolation ("ridgeless regression"): with $n$ samples and $p$ features,

$$\hat\beta = X^+y = X^\top(XX^\top)^{-1}y \quad (p>n)$$

The estimator's variance involves $(X^\top X)^{-1}$ or its pseudo-inverse, whose scale is set by the **smallest nonzero singular value** of $X$.

▸ **At $p=n$, $X$ is square and generically nearly singular** — by random-matrix theory (the Marchenko–Pastur law), the smallest singular value of an $n\times n$ random matrix concentrates near **zero**. So $\|\hat\beta\|\to\infty$ and the variance blows up. **That is the pole, and it is a property of the linear algebra, not of neural networks.**

- $p<n$: the system is overdetermined, $X^\top X$ is well-conditioned, ordinary least squares behaves classically.
- $p=n$: exactly determined, no slack, the fit is forced through every point by a nearly-singular inverse.
- $p>n$: **underdetermined — infinitely many exact solutions exist, and we get to pick.** The minimum-norm choice is smooth, and as $p$ grows the solution space grows so the minimum-norm element gets *smaller* in norm. Variance falls.

▸ **The key conceptual shift: in the overparameterized regime, what matters is not "how many solutions fit the data" but "which one the optimizer picks."** Capacity stops being the controlling variable and *implicit bias* takes over (Ch. 31).

### Effective rather than raw parameter count

Raw parameter count is the wrong $x$-axis. Better measures place the peak more reliably:
- Effective degrees of freedom $\sum_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}$ (Ch. 22 §22.2).
- Weight-norm-based capacity measures.
- Nakkiran et al.'s **effective model complexity**: the largest $n$ for which the training procedure achieves ≈0 training error.

▸ **Explicit regularization mitigates or removes the peak.** With optimal ridge $\lambda$, the double-descent curve becomes monotone. This is important: **double descent is a phenomenon of *unregularized* interpolation.** It says less about the impossibility of classical theory than about what happens when you turn regularization off.

### Benign overfitting

The theory (Bartlett, Long, Lugosi, Tsigler) explains *when* interpolating noise is harmless: when the data covariance spectrum has a few large eigenvalues carrying signal and a long tail of many small ones. Then the noise is absorbed by the tail directions — spread thinly across thousands of low-variance components, each contributing negligibly to predictions — while the signal is fit in the top directions.

▸ **The high-dimensional tail acts as a sink for noise.** This requires a specific spectral structure; overparameterization alone is not sufficient. Being able to state this condition, rather than just saying "big models don't overfit," is the mark of understanding the result.

---

## 30.2 The Neural Tangent Kernel

### The one-line idea

An infinitely wide network trained by gradient descent doesn't really change its features at all — it behaves exactly like linear regression in a fixed feature space determined at initialization, so its entire training dynamics can be solved in closed form.

### The analogy

A committee so large that no individual member's opinion shifts much. The committee's collective output changes a lot, but only because millions of members each nudged infinitesimally. From the outside it looks like learning; from the inside, nobody learned anything new — the aggregation weights just got re-tuned.

### The derivation

First-order Taylor expansion around initialization $\theta_0$:
$$f(x;\theta)\approx f(x;\theta_0) + \nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0)$$

**This is linear in $\theta$** with fixed features $\phi(x)=\nabla_\theta f(x;\theta_0)$. Define the kernel:

▸ $$\Theta(x,x') = \big\langle\nabla_\theta f(x;\theta_0),\ \nabla_\theta f(x';\theta_0)\big\rangle$$

Under gradient flow with squared loss, the *function* evolves as:

▸ $$\frac{df(x)}{dt} = -\eta\sum_i\Theta(x,x_i)\,\big(f(x_i)-y_i\big)$$

▸ **The parameters have vanished. The dynamics are entirely determined by the kernel**, and they are linear ODEs with a closed-form solution: the residual decays as $e^{-\eta\Theta t}$ along each eigendirection of the kernel matrix.

### The two theorems

**(1) At infinite width, $\Theta$ converges to a deterministic limit** depending only on the architecture, depth, and activation — not on the random initialization.

**(2) At infinite width, $\Theta$ stays constant during training.** Each weight moves by $O(1/\sqrt{\text{width}})$, so the Jacobian doesn't change, so the kernel doesn't change.

▸ **Therefore: an infinitely wide network trained by gradient descent is exactly kernel regression with the NTK.** A closed-form solution for a neural network's entire training trajectory. That is the result, and it is remarkable.

### What it explains

- **Why gradient descent finds a global minimum:** the loss becomes convex in the linearized model, and a positive-definite $\Theta$ guarantees convergence.
- **The convergence rate:** the residual along eigendirection $j$ decays at rate $\eta\lambda_j$. ▸ **Since the NTK's large eigenvalues correspond to low-frequency, smooth functions, networks fit smooth structure first and high-frequency detail last** — the *spectral bias*. This is a genuine, quantitative explanation of a real phenomenon, and it also explains why random labels take much longer to fit than real ones.
- **Wide networks are easier to optimize** — the linearization is more accurate.

### What it fails to explain, and why that matters more

▸ **Finite networks beat their NTK, typically by several percent on ImageNet.** And the NTK cannot explain transfer learning, because it has **no feature learning at all**: the features $\phi(x)=\nabla_\theta f(x;\theta_0)$ are fixed at initialization. A model whose features never change cannot learn a representation, and representation learning is the thing deep learning is actually for.

### Lazy vs rich (feature-learning) regimes

Whether a network is in the NTK regime depends on the **parameterization and initialization scale**, not merely on width.

| | Lazy / NTK | Rich / feature-learning |
|---|---|---|
| Weight movement | $O(1/\sqrt{\text{width}})$ | $\Theta(1)$ |
| Kernel | frozen | evolves |
| Features | fixed at init | learned |
| Theory | closed form | hard |
| Real networks | small-LR, large-init limit | **where the good ones live** |

▸ **This is exactly the distinction $\mu$P was designed around** (Ch. 15 §15.3): standard parameterization drifts toward the lazy regime as width grows, while $\mu$P keeps the network in the feature-learning regime at any width. **That is *why* $\mu$P transfers hyperparameters — it preserves the regime.** Connecting these two chapters is a strong signal in an interview.

**The honest summary:** the NTK is the correct theory of a limit that real networks are deliberately kept out of. It is valuable as the only fully-solved case, as a source of quantitative predictions like spectral bias, and as the reference point against which feature learning is defined.

---

## 30.3 Grokking

### The phenomenon

Train a small transformer on modular arithmetic (e.g. $a\circ b = (a+b)\bmod 97$) with a fraction of the pairs held out:

- **Step ~$10^3$:** training accuracy 100%. Test accuracy ~random. The model has memorized.
- **Steps $10^3$ to $10^5$:** nothing visibly happens. Training loss is at zero; test accuracy stays flat.
- **Step ~$10^5$:** test accuracy jumps to 100% over a short window.

▸ **Generalization occurs long after the training loss has been zero, with no visible signal in between.** This is a direct refutation of the intuition that training loss tracks learning.

### The mechanism

The best-supported account has several converging strands:

**1. Two competing solutions.** A **memorizing** circuit (a lookup table) and a **generalizing** circuit (an actual algorithm). Memorization is found first because it is reachable greedily. The generalizing solution is harder to find but, crucially, **has smaller weight norm** — an algorithm is more parameter-efficient than a table.

**2. Weight decay drives the transition.** ▸ **Grokking largely disappears without weight decay** (or with weight norm otherwise controlled). Once training loss is ~0, the cross-entropy gradient is nearly zero and **weight decay becomes the dominant force**, slowly moving the model along the zero-loss manifold toward the minimum-norm solution — which is the generalizing one. The delay is the time this drift takes.

**3. The circuit is identifiable.** For modular addition, the network learns a **discrete Fourier transform**: embeddings become $\big(\cos(\omega_k a),\sin(\omega_k a)\big)$ for a handful of frequencies $\omega_k$, attention and MLP layers compute products realizing the trigonometric identity
$$\cos(\omega(a+b)) = \cos\omega a\cos\omega b - \sin\omega a\sin\omega b$$
and the unembedding reads off the argmax over $c$ of $\cos(\omega(a+b-c))$. ▸ **The network implements modular addition by rotating on a circle.** Nanda et al. reverse-engineered this completely — one of the cleanest full mechanistic explanations of a trained network in existence (Ch. 32).

**4. Progress measures.** The generalizing circuit forms *gradually* and continuously; only the *test accuracy metric* is discontinuous. Defining a continuous progress measure — restricted loss (ablate all but the key Fourier frequencies) and excluded loss (ablate only them) — shows smooth formation throughout the plateau.

▸ **This is exactly the emergence critique from Chapter 15 §15.4, at the level of a single small model where we can verify it completely.** The underlying change is smooth; the metric is not. Grokking is the strongest available evidence for that view, because here we can see the mechanism directly rather than inferring it.

### Where it happens

Algorithmic tasks with a clean underlying rule and enough (but not too much) data. ▸ **Below a critical data fraction, the model never groks** — memorization is a valid solution and no pressure moves it off. Above a larger fraction, generalization is immediate — memorization is no longer easier. **Grokking lives in the window between**, and that window is why it took so long to notice.

**Does it happen at scale?** Probably in a distributed form: circuits form at different times during training, and phase transitions in specific capabilities (induction heads, Ch. 13 §13.3) look structurally like grokking. Delayed generalization on subtasks has been observed in language models.

---

## 30.4 The synthesis

▸ **What these three phenomena have in common:** all three are cases where *capacity* is the wrong variable and *the optimization trajectory* is the right one.

- Double descent: in the interpolating regime, which of the infinitely many zero-error solutions you get is what determines test error.
- NTK: the training dynamics are what's solvable; the parameters are not the interesting object.
- Grokking: two solutions both achieve zero training error, and the slow drift between them is invisible to the loss.

**Classical theory bounds what the model class *could* do. Modern deep learning is governed by what the optimizer *does*.** That is the through-line into Chapter 31, which studies the optimizer's preferences directly.

---

## Check for Understanding

**Test error peaks exactly at the interpolation threshold because the design matrix is nearly singular there and the minimum-norm solution blows up, then falls again because extra capacity gives the optimizer slack to choose a smooth interpolant — and the same shift in perspective explains grokking, where two zero-training-loss solutions compete and weight decay slowly drifts the model from the memorizing one to the generalizing one long after the loss has stopped moving.**

---

**Next:** [Chapter 31 — Neural Collapse, Implicit Bias & Lottery Tickets](31-neural-collapse-implicit-bias-lottery-tickets.md)
