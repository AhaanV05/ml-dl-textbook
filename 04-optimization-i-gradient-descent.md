# Chapter 4 — Optimization I: Gradient Descent

> **Prerequisites:** Ch. 1 (matrix calculus, eigendecomposition).
> **Goal:** by the end you should be able to say, for any optimizer, *what quantity it is estimating and what its convergence rate depends on* — and to explain why the condition number is the villain in every story.

---

## 4.1 The problem

### The one-line idea

Minimize a function you can only probe locally, using only its slope.

### The analogy

You're on a foggy hillside and want to reach the bottom. You can feel the slope under your feet but see nothing. Gradient descent is: take a step downhill, repeat. Everything interesting is about **how big a step** and **whether to trust the slope you just felt** — because the ground is bumpy and the slope under your feet may not point toward the valley.

### Setup

$$\theta_{t+1} = \theta_t - \eta\,\nabla\mathcal{L}(\theta_t)$$

Three regimes matter:

| Regime | What we assume | What we can prove |
|---|---|---|
| Convex + smooth | $L$-Lipschitz gradient | $\mathcal{L}(\theta_T)-\mathcal{L}^* = O(1/T)$ |
| Strongly convex + smooth | also $\mu$-strongly convex | linear rate $O((1-\mu/L)^T)$ |
| Nonconvex + smooth | just smoothness | $\min_t\|\nabla\mathcal{L}(\theta_t)\|^2 = O(1/T)$ (stationary point only) |

Deep learning is the third row, which is why theory gives so little and practice gives so much.

---

## 4.2 The definitions that make the proofs work

**$L$-smooth** (gradient is $L$-Lipschitz):
$$\|\nabla\mathcal{L}(x)-\nabla\mathcal{L}(y)\| \le L\|x-y\| \iff \lambda_{\max}(\nabla^2\mathcal{L}) \le L$$

The key consequence — the **descent lemma**:
▸ $$\mathcal{L}(y) \le \mathcal{L}(x) + \langle\nabla\mathcal{L}(x), y-x\rangle + \frac{L}{2}\|y-x\|^2$$

*Read it as:* the function is below a quadratic bowl that touches it at $x$. So if you minimize the bowl, you're guaranteed to decrease the function.

**$\mu$-strongly convex:**
$$\mathcal{L}(y) \ge \mathcal{L}(x) + \langle\nabla\mathcal{L}(x),y-x\rangle + \frac{\mu}{2}\|y-x\|^2 \iff \lambda_{\min}(\nabla^2\mathcal{L}) \ge \mu > 0$$

**Condition number:** $\kappa = L/\mu$. This is the ratio of the steepest to the shallowest curvature. **It is the single number that determines how hard your optimization problem is.**

---

## 4.3 The convergence proof for smooth convex GD

Plug $y = \theta_{t+1} = \theta_t - \eta g_t$ into the descent lemma:

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta\|g_t\|^2 + \frac{L\eta^2}{2}\|g_t\|^2 = \mathcal{L}(\theta_t) - \eta\left(1 - \frac{L\eta}{2}\right)\|g_t\|^2$$

▸ **Stability condition:** we need $1 - L\eta/2 > 0$, i.e. $\eta < 2/L$. With $\eta = 1/L$ we get the clean

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \frac{1}{2L}\|g_t\|^2$$

**Every step decreases the loss by an amount proportional to the squared gradient norm, divided by the sharpness.** That's the whole mechanism of gradient descent in one line.

Summing over $T$ steps (nonconvex case):
$$\frac{1}{2L}\sum_{t=0}^{T-1}\|g_t\|^2 \le \mathcal{L}(\theta_0) - \mathcal{L}^* \implies \min_{t<T}\|g_t\|^2 \le \frac{2L(\mathcal{L}(\theta_0)-\mathcal{L}^*)}{T}$$

So gradient norm shrinks as $O(1/\sqrt T)$ in the nonconvex case. **This is all the guarantee deep learning has.** Not a global minimum, not even a local minimum — just "you'll eventually pass near a point with small gradient."

**With convexity added**, a few more lines give $\mathcal{L}(\theta_T) - \mathcal{L}^* \le \frac{\|\theta_0-\theta^*\|^2}{2\eta T}$, the $O(1/T)$ rate.

**With strong convexity:**
▸ $$\|\theta_T - \theta^*\|^2 \le \left(1-\frac{\mu}{L}\right)^T\|\theta_0-\theta^*\|^2 = \left(1-\frac{1}{\kappa}\right)^T\|\theta_0-\theta^*\|^2$$

To reach error $\epsilon$: $T = O(\kappa\log(1/\epsilon))$ iterations.

**Number:** if $\kappa = 10^4$ (typical for a poorly-conditioned deep network), you need $\sim 10^4$ iterations per digit of accuracy. If $\kappa = 10$, you need $\sim10$. **This is why conditioning is everything, and why normalization layers (Ch. 7) are worth more than most architectural cleverness.**

---

## 4.4 The quadratic model: seeing $\kappa$ do its damage

Take $\mathcal{L}(\theta) = \frac12\theta^\top H\theta$ with $H = Q\Lambda Q^\top$. In the eigenbasis ($u = Q^\top\theta$), GD decouples completely:

$$u^{(i)}_{t+1} = (1-\eta\lambda_i)\,u^{(i)}_t \implies u^{(i)}_t = (1-\eta\lambda_i)^t u^{(i)}_0$$

▸ Each eigendirection contracts independently by factor $|1-\eta\lambda_i|$.

- **Convergence requires** $|1-\eta\lambda_i|<1$ for all $i$, i.e. $\eta < 2/\lambda_{\max}$. **This is the stability threshold** — remember it for Edge of Stability in Ch. 5.
- The **slowest** direction is $\lambda_{\min}$: contraction $1-\eta\lambda_{\min} = 1-\lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa$ at the largest stable LR.
- The **optimal** LR balances the two extremes: $\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}}$, giving contraction $\frac{\kappa-1}{\kappa+1}$.

**The picture:** a long narrow valley. GD zig-zags across the narrow direction (which is near its stability limit) while crawling along the long direction. You've seen this plot; now you know it's $|1-\eta\lambda_i|$ close to $-1$ in one direction and close to $+1$ in the other.

**Numbers.** $\lambda_{\max}=100$, $\lambda_{\min}=0.01$, $\kappa=10^4$. Max stable $\eta = 0.02$. Contraction in the slow direction: $1 - 0.02\times0.01 = 0.9998$. To reduce that component by $e^{-1}$ takes $5{,}000$ steps. To reduce it by $10^{-3}$ takes ~35,000 steps. **That's a long training run.**

---

## 4.5 Momentum

### The one-line idea

Accumulate a running average of past gradients so that consistent directions build up speed and oscillating directions cancel out.

### The analogy

A ball rolling down the valley instead of a hiker taking discrete steps. The ball's inertia carries it along the valley floor and averages out the side-to-side sloshing.

### Heavy-ball (Polyak, 1964)

▸ $$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

(PyTorch's convention. The "$\ (1-\beta)g_t$" variant just rescales $\eta$.)

**What it does, quantitatively.** For constant $g$, the velocity converges to $v_\infty = g/(1-\beta)$.

▸ $$\text{effective step} = \frac{\eta}{1-\beta}\qquad\Rightarrow\qquad \beta=0.9 \text{ multiplies your step size by } 10.$$

This is why you must lower the LR when you turn on momentum, and why "momentum 0.9 vs 0.99" is a bigger change than it looks (10× vs 100×).

The **memory horizon** is $1/(1-\beta)$ steps: $\beta=0.9 \to 10$ steps, $\beta=0.99\to 100$.

**Analysis on the quadratic.** In eigendirection $i$, the coupled recursion has characteristic polynomial
$$z^2 - (1+\beta-\eta\lambda_i)z + \beta = 0$$
Optimal tuning $\beta^* = \left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2$, $\eta^* = \frac{4}{(\sqrt{L}+\sqrt\mu)^2}$ gives contraction rate

▸ $$1 - \frac{1}{\sqrt\kappa}\quad\text{instead of}\quad 1-\frac{1}{\kappa}$$

**This is the acceleration.** For $\kappa=10^4$: plain GD needs $10^4$ iterations per digit, momentum needs $10^2$. **A 100× speedup for one extra buffer.** That is the best deal in optimization.

### Nesterov accelerated gradient

Evaluate the gradient at the *look-ahead* point:
$$v_{t+1} = \beta v_t + \nabla\mathcal{L}(\theta_t - \eta\beta v_t),\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}$$

**Intuition:** heavy-ball looks at where it is and then jumps; Nesterov jumps first and then corrects. If momentum is about to overshoot, Nesterov feels the upward slope *before* committing, and brakes.

Nesterov achieves the optimal $O(1/T^2)$ rate for smooth convex problems (vs $O(1/T)$ for GD), and this is provably optimal for any first-order method (Nemirovski–Yudin lower bound). Heavy-ball achieves $\sqrt\kappa$ acceleration only for quadratics; Nesterov achieves it in general.

**Practical note:** the difference between heavy-ball and Nesterov in deep learning is small (a few percent). The difference between momentum and no momentum is enormous.

---

## 4.6 Stochastic gradient descent

### The one-line idea

Estimate the gradient from a minibatch instead of the full dataset — accept noise in exchange for doing $n/B$ times more steps per epoch.

### The analogy

Rather than polling every voter in the country before making each decision, poll 64 people. Your direction is noisier, but you can make 2,274 decisions in the time it took to make one.

### Setup

$$g_t = \frac1B\sum_{i\in\mathcal{B}_t}\nabla\ell_i(\theta_t),\qquad \mathbb{E}[g_t]=\nabla\mathcal{L}(\theta_t),\qquad \mathrm{Cov}(g_t) = \frac{\Sigma(\theta_t)}{B}$$

**Unbiased but noisy, with variance $\propto 1/B$.**

### The convergence rate and why constant LR plateaus

Repeat the descent-lemma argument with $\mathbb{E}\|g_t\|^2 = \|\nabla\mathcal{L}\|^2 + \frac{\mathrm{tr}\Sigma}{B}$:

$$\mathbb{E}[\mathcal{L}(\theta_{t+1})] \le \mathcal{L}(\theta_t) - \eta\|\nabla\mathcal{L}\|^2 + \frac{L\eta^2}{2}\left(\|\nabla\mathcal{L}\|^2 + \frac{\mathrm{tr}\Sigma}{B}\right)$$

▸ The last term **does not vanish as $\nabla\mathcal{L}\to0$**. Near a minimum, SGD with fixed $\eta$ doesn't converge — it settles into a noise ball of radius

$$\mathbb{E}\|\theta-\theta^*\|^2 \approx \frac{\eta\,\mathrm{tr}\Sigma}{2\mu B}$$

**This is why LR decay exists.** Not because "the optimizer gets tired," but because the *stationary noise floor is proportional to $\eta/B$*, and the only way to get below it is to shrink $\eta$ or grow $B$.

For convergence you need the **Robbins–Monro conditions**:
▸ $$\sum_t \eta_t = \infty \quad\text{(can still travel far)},\qquad \sum_t \eta_t^2 < \infty \quad\text{(noise is summable)}$$
$\eta_t = \eta_0/t$ satisfies both. $\eta_t = \eta_0/\sqrt t$ satisfies the first but not the second — it converges in the weaker averaged sense with rate $O(1/\sqrt T)$, which is optimal for nonsmooth stochastic convex problems.

**Constant $\eta$ satisfies neither**, which is exactly your setup. So your model is *not* converging to a point; it is diffusing in a noise ball around a region of low loss. That is a completely standard and often *good* place to be — but it means your parameters at epoch 43 are meaningfully different from those at epoch 42, even if the loss is the same, and it means an **EMA of the weights** (§4.8) will beat any single checkpoint.

### SGD as an SDE — the temperature

For small $\eta$, SGD is well approximated by the stochastic differential equation

▸ $$d\theta = -\nabla\mathcal{L}(\theta)\,dt + \sqrt{\frac{\eta}{B}\Sigma(\theta)}\,dW$$

The stationary distribution (for isotropic $\Sigma = \sigma^2 I$) is approximately Gibbs:
$$p(\theta) \propto \exp\left(-\frac{2\mathcal{L}(\theta)}{T}\right),\qquad \boxed{T = \frac{\eta\sigma^2}{B}}$$

▸ **The learning-rate-to-batch-size ratio $\eta/B$ is a temperature.** High temperature ⇒ the walker escapes narrow basins and settles in wide ones. This is the mechanistic core of "SGD prefers flat minima" (Ch. 19), and it immediately explains:

- **The linear scaling rule** (Goyal et al.): if you multiply $B$ by $k$, multiply $\eta$ by $k$ to hold temperature fixed. Works up to a critical batch size.
- **Large-batch generalization gap:** big $B$ at fixed $\eta$ = low temperature = sharp minima = worse generalization.
- Why "just use a bigger batch, it's faster" often silently costs you accuracy.

**Numbers for Case Study A:** $\eta=3\times10^{-4}$, $B=64$, so $\eta/B = 4.7\times10^{-6}$. If you moved to $B=256$ for speed, holding accuracy would want $\eta \approx 1.2\times10^{-3}$ (linear rule) or $\eta\approx6\times10^{-4}$ (square-root rule, often better for Adam). Changing batch size without changing LR is changing two things at once.

### Variance reduction (why it doesn't help in deep learning)

SVRG maintains a snapshot $\tilde\theta$ and uses $g_t = \nabla\ell_i(\theta_t) - \nabla\ell_i(\tilde\theta) + \nabla\mathcal{L}(\tilde\theta)$, which is unbiased with variance $\to0$ as $\theta_t\to\tilde\theta$. It gives linear convergence for strongly convex finite sums.

▸ **It reliably fails to help deep networks.** Reasons: (1) data augmentation makes the "finite sum" not finite; (2) the snapshot goes stale within a few hundred steps because $\theta$ moves fast; (3) most importantly, **the noise is doing useful work** (it's the temperature above) — removing it removes the implicit regularization. A rare case where a theoretically superior algorithm is practically worse *because* the theory optimized the wrong objective.

---

## 4.7 Second-order and quasi-Newton methods

**Newton's method:** $\theta_{t+1} = \theta_t - H^{-1}g_t$. Converges quadratically near the optimum and is **affine invariant** — $\kappa$ disappears entirely. Cost: $O(p^3)$ to invert. Dead on arrival for $p=10^8$.

**Natural gradient:** replace $H$ with the Fisher information $F = \mathbb{E}[\nabla\log p_\theta \nabla\log p_\theta^\top]$. This makes the step invariant to *reparameterization of the model's distribution* rather than of the parameters — you're doing steepest descent in KL-divergence geometry, not Euclidean geometry.

▸ $$\Delta\theta^* = \arg\min_{\Delta\theta} \mathcal{L}(\theta+\Delta\theta)\ \text{ s.t. }\ \mathrm{KL}(p_\theta\|p_{\theta+\Delta\theta})\le\epsilon \implies \Delta\theta \propto -F^{-1}g$$

This is the mathematical foundation of TRPO (Ch. 17).

**K-FAC** approximates $F$ as a Kronecker product per layer: $F_\ell \approx A_{\ell-1}\otimes G_\ell$ where $A$ is the input covariance and $G$ the output-gradient covariance. Inversion becomes $(A\otimes G)^{-1} = A^{-1}\otimes G^{-1}$, cost $O(d^3)$ per layer instead of $O(d^6)$.

**Shampoo / Muon** are the modern descendants — preconditioners built from $\left(\sum g g^\top\right)^{-1/4}$ per tensor mode. Muon in particular orthogonalizes the momentum matrix (via Newton–Schulz iteration for the "sign of a matrix"), which is a spectral-norm-controlled step. These are currently the strongest challengers to AdamW at scale (Ch. 5).

**L-BFGS** builds an implicit inverse-Hessian from the last $m$ (gradient, step) pairs. Excellent for deterministic full-batch problems (it's what `scipy.optimize` uses). **Poor with minibatch noise** — the curvature pairs are corrupted, and the line search it needs is impractical.

---

## 4.8 Two practical things that are worth more than they look

### Gradient clipping

▸ $$g \leftarrow g\cdot\min\left(1, \frac{c}{\|g\|}\right)$$

Clip by **global norm** across all parameters, not per-tensor (per-tensor clipping changes the *direction* of the update, global clipping only its length).

**Why it works:** the smoothness constant $L$ isn't uniform. Deep nets have regions ("cliffs") where $L$ is enormous, and a normal-size step there overshoots catastrophically. Clipping enforces a trust region: **it is an adaptive step size for the local Lipschitz constant.** Theoretically justified under $(L_0,L_1)$-smoothness, where $\|\nabla^2\mathcal{L}\| \le L_0 + L_1\|\nabla\mathcal{L}\|$ — which empirically holds for transformers.

Typical $c$: 1.0 for transformers. Log $\|g\|$ during training; if it's routinely 100× the clip value you have a problem elsewhere.

### Weight averaging (EMA / SWA)

▸ $$\bar\theta_t = \gamma\bar\theta_{t-1} + (1-\gamma)\theta_t,\qquad \gamma \in [0.999, 0.9999]$$

Because SGD with constant LR *diffuses in a noise ball* (§4.6), the individual iterates are all slightly off-center. Averaging them lands you closer to the center of the basin, which is both lower-loss and flatter.

▸ **For diffusion models specifically, EMA weights are not optional — they are standard practice and typically worth more than any architectural change you're considering.** DDPM, Stable Diffusion, DiT, and essentially every strong diffusion model reports EMA weights. If Case Study A doesn't maintain an EMA, adding one is very likely the highest-return change available to you, and it also solves the checkpoint-selection-bias problem from Ch. 3 (you stop needing to pick a lucky epoch).

With $\gamma=0.9999$ and 2,274 steps/epoch, the EMA horizon is $1/(1-\gamma)=10{,}000$ steps $=4.4$ epochs. That is a sensible setting for Case Study A.

---

## 4.9 Summary table

| Method | Update | Rate (strongly cvx) | Memory | Notes |
|---|---|---|---|---|
| GD | $-\eta g$ | $(1-1/\kappa)^T$ | $0$ | baseline |
| Heavy-ball | $-\eta v$, $v=\beta v+g$ | $(1-1/\sqrt\kappa)^T$ | $p$ | quadratics only |
| Nesterov | look-ahead gradient | $(1-1/\sqrt\kappa)^T$ | $p$ | general, optimal |
| SGD | minibatch $g$ | noise floor $\eta\sigma^2/B$ | $0$ | needs decay |
| Newton | $-H^{-1}g$ | quadratic | $p^2$ | $O(p^3)$ solve |
| Natural grad | $-F^{-1}g$ | — | $p^2$ | KL geometry |
| K-FAC | Kronecker $F^{-1}$ | — | $\sim2pd$ | practical 2nd order |
| Adam(W) | Ch. 5 | — | $2p$ | the default |

---

## Check for Understanding

**Gradient descent's speed is governed by the condition number, momentum buys you a square root of it for the price of one extra buffer, and SGD's minibatch noise is not a defect to be minimized but a temperature $\eta/B$ that determines which basin you end up in.**

---

**Next:** [Chapter 05 — Optimization II: Adaptive Methods & Schedules](05-optimization-ii-adaptive-methods-and-schedules.md)
