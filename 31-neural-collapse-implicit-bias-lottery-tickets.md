# Chapter 31 — Neural Collapse, Implicit Bias & Lottery Tickets

> **Prerequisites:** Ch. 2, Ch. 4, Ch. 30.

---

## 31.1 Neural collapse

### The phenomenon

Keep training a classifier *past* the point where training error hits zero — into the "terminal phase of training" — and the last-layer geometry converges to a rigid, universal structure. This happens across architectures, datasets, and optimizers.

### The four properties

Let $h_{i,c}$ be the penultimate-layer feature of the $i$-th example of class $c$, $\mu_c$ the class mean, $\mu_G$ the global mean, and $w_c$ the classifier weight vector for class $c$.

▸ **NC1 — Variability collapse.** Within-class scatter $\to0$: $\Sigma_W\to0$, so every example of a class maps to *the same point*.

▸ **NC2 — Simplex ETF.** The centred class means $\tilde\mu_c=\mu_c-\mu_G$ become equinorm and equiangular, forming a **simplex equiangular tight frame**:
$$\frac{\langle\tilde\mu_c,\tilde\mu_{c'}\rangle}{\|\tilde\mu_c\|\|\tilde\mu_{c'}\|} \to \begin{cases}1 & c=c'\\[2pt] -\dfrac{1}{C-1} & c\ne c'\end{cases}$$

▸ **NC3 — Self-duality.** The classifier weights align with the class means: $W\propto \tilde M^\top$. The classifier and the features become the same object.

▸ **NC4 — Simplification to nearest class centre.** The network's decision reduces to $\arg\min_c\|h-\mu_c\|$ — a nearest-centroid classifier.

### Why $-\frac{1}{C-1}$

▸ It is the **maximally separated** configuration for $C$ points on a sphere in $\ge C-1$ dimensions. Proof: if $\sum_c\tilde\mu_c=0$ (which centring enforces) and all have equal norm, then
$$0 = \Big\|\sum_c\tilde\mu_c\Big\|^2 = C\|\tilde\mu\|^2 + C(C-1)\|\tilde\mu\|^2\cos\vartheta \implies \cos\vartheta = -\frac{1}{C-1}$$
∎ For $C=3$ this is $-1/2$, i.e. 120° apart — an equilateral triangle. For large $C$, nearly orthogonal.

### Why it happens

The **unconstrained features model**: treat the penultimate features as free variables (justified by the network's expressivity) and minimize cross-entropy plus weight decay over both features and classifier. The global minimizer of that problem is provably the simplex ETF with self-dual weights.

▸ **So neural collapse is the global optimum of the last two layers' problem** — the network is not doing anything mysterious; it is finding the geometry that maximizes margin under a norm constraint.

### Why it matters practically

- **Explains why the last layer can be frozen at a fixed ETF** with no loss of accuracy, saving parameters at large $C$.
- **Explains label smoothing's effect** — it accelerates collapse (Ch. 7 §7.5).
- ▸ **NC1 is a form of information destruction.** Collapsing all within-class variability discards exactly the information needed for transfer, fine-grained distinctions, and OOD detection. **This is a quantitative account of why training a classifier too long produces worse features for downstream use**, and it is an argument for using self-supervised features (Ch. 25) instead.
- Under class imbalance, collapse becomes **minority collapse**: minority class means converge toward each other and become indistinguishable — a precise characterization of what imbalance does to representations.

---

## 31.2 Implicit bias — the central question

▸ **The question that organizes all of modern generalization theory:** an overparameterized network has infinitely many parameter settings achieving zero training loss, and they have wildly different test errors. Gradient descent finds a good one. **Why?**

The answer cannot be capacity (Ch. 2 showed those bounds are vacuous; Ch. 30 showed capacity is the wrong axis). It must be a property of the **optimizer**.

### The linear case, solved

**Separable logistic regression** (Soudry, Hoffer & Srebro, 2018). The loss $\sum_i\log(1+e^{-y_i w^\top x_i})$ has no finite minimizer — $\|w\|\to\infty$. But the *direction* converges:

▸ $$\frac{w(t)}{\|w(t)\|}\ \longrightarrow\ \frac{\hat w_{\text{SVM}}}{\|\hat w_{\text{SVM}}\|}$$

**Gradient descent on logistic loss converges in direction to the max-margin (hard-margin SVM) solution.** No regularization required, no early stopping.

▸ **The rate is $O(1/\log t)$ — extraordinarily slow.** Consequences worth stating:
1. **The loss keeps improving long after training error is zero**, and the margin keeps growing. This is what "training past convergence" is doing.
2. **The margin at any finite time is far from optimal.** So the implicit bias is real but weak, and explicit regularization still helps.
3. It gives a principled reason to train longer than the loss curve suggests.

**Other cases:** gradient descent on **least squares** from $\theta_0=0$ converges to the **minimum $\ell_2$-norm** interpolant (it stays in the row space of $X$, which is the smallest-norm solution). Deep *linear* networks bias toward low-rank / low nuclear norm. Matrix factorization with small initialization biases toward low rank.

### Nonlinear networks

Fully rigorous results are limited to special cases, but the established picture:
- **Homogeneous networks** (ReLU, no bias) converge in direction to a KKT point of a **margin-maximization** problem in parameter space (Lyu & Li, 2020).
- **Small initialization is essential.** Large initialization keeps you in the lazy/NTK regime (Ch. 30 §30.2) and destroys the sparsity/low-rank bias. **The initialization scale is a regularization hyperparameter**, which is not how most people think of it.
- **Simplicity bias:** networks fit low-frequency, low-complexity functions first (the spectral bias of Ch. 30 §30.2). This is a blessing for generalization and a curse for spurious correlations — the model latches onto the *simplest* predictive feature, which is often background texture rather than the object.

### Adam has a different bias — and it's worse

▸ SGD's bias is toward minimum $\ell_2$ norm / max $\ell_2$ margin. **Adam's coordinate-wise normalization changes the geometry**: its implicit bias is closer to $\ell_\infty$-geometry, i.e. toward maximum $\ell_1$-margin.

**This is the explanation for the persistent Adam-generalization gap** (Ch. 5 §5.6): Adam converges faster but often to a worse-generalizing solution on vision tasks, while being essential for transformers. It is a genuine difference in what the optimizer prefers, not merely a tuning artifact.

▸ **This is also the correct reason to prefer AdamW over Adam+L2** (Ch. 5 §5.2). With Adam's preconditioner, an $\ell_2$ term added to the loss gets divided by $\sqrt{\hat v}$, so the effective decay differs per coordinate and no longer corresponds to any clean norm penalty. Decoupling restores it.

---

## 31.3 Flat minima

### The hypothesis

Flat minima generalize better than sharp ones, because a flat minimum's loss is insensitive to parameter perturbation — and the train-to-test shift acts like a perturbation.

**Measures of sharpness:**
- $\lambda_{\max}(\nabla^2\mathcal{L})$ — the top Hessian eigenvalue (Ch. 5 §5.5's Edge of Stability quantity).
- $\mathrm{tr}(H)$ — average curvature.
- $\max_{\|\epsilon\|\le\rho}\mathcal{L}(\theta+\epsilon)-\mathcal{L}(\theta)$ — worst-case within a ball; this is SAM's objective.
- The PAC-Bayes / MDL argument: a flat minimum needs fewer bits to specify to a given loss tolerance, and Chapter 2 §2.6 showed description length bounds generalization.

### The objection you must know

▸ **Dinh et al. (2017): sharpness is not reparameterization-invariant.** For a ReLU network, the map $(W_1,W_2)\to(\alpha W_1,\alpha^{-1}W_2)$ leaves the function *exactly* unchanged, but scales the Hessian eigenvalues by $\alpha^{\pm2}$. **So any minimum can be made arbitrarily sharp or flat without changing the function at all.**

Therefore naive sharpness cannot, by itself, explain generalization.

**The responses:** use scale-invariant measures (relative flatness, normalized by weight norm); note that the reparameterizations in question are pathological and not visited by actual training; and observe that **sharpness-aware minimization empirically works** (Ch. 5 §5.7, ~+1% on ImageNet), which is evidence the underlying intuition captures something real even if the naive measure is flawed.

### Why SGD finds flat minima

▸ **The SDE temperature argument** (Ch. 4 §4.6): SGD approximates a stochastic differential equation with temperature $T\propto\eta/B$. In a Gibbs-like stationary distribution $\propto e^{-\mathcal{L}/T}$, the probability of occupying a basin scales with its **volume**, not merely its depth. Wide basins have exponentially more volume in high dimensions. **So SGD's noise selects for flat minima automatically**, and this predicts — correctly — that large batch sizes (low temperature) find sharper minima and generalize worse, unless the learning rate is raised to compensate.

This is one of the more satisfying explanations in the field: a single parameter $\eta/B$ predicts a real, reproducible generalization effect.

---

## 31.4 The Lottery Ticket Hypothesis

### The claim

▸ **A randomly-initialized dense network contains a sparse subnetwork ("winning ticket") that — when trained in isolation *from the original initialization* — matches the full network's accuracy in the same number of steps.**

### Finding one: iterative magnitude pruning

```
1. Initialize θ₀; save it.
2. Train to convergence.
3. Prune the p% smallest-magnitude weights → mask m.
4. RESET surviving weights to their values in θ₀.        ← the essential step
5. Repeat from 2 with the mask applied.
```

Typically 20% pruned per round, 15–30 rounds, reaching 90–99% sparsity.

▸ **Step 4 is the entire hypothesis.** Reinitializing the same mask with *fresh* random values fails badly. So the ticket is the pair (**mask, initialization**), not the mask alone — the structure alone is not sufficient.

### Rewinding — the necessary correction

At ImageNet scale, resetting to $\theta_0$ **does not work**. The fix (Frankle et al., 2020): reset to $\theta_k$ for a small $k$ (0.1–7% of training) rather than to $\theta_0$.

▸ **"Weight rewinding" implies the ticket is not present at initialization but forms in the first few hundred steps.** Frankle called this the point of **linear mode connectivity** — the moment after which the network's trajectory becomes stable to SGD noise. Two runs branched before it end up in different basins; two runs branched after it end up in the same one. **The unstated qualification of the original LTH is that "initialization" means "shortly after initialization" for any realistic model.**

### What tickets tell us

- **Transfer:** tickets found on one dataset transfer to related ones, and tickets from larger datasets transfer better — so they encode something about the *task family*, not one dataset.
- **The strong LTH** (Ramanujan et al., 2020): a sufficiently overparameterized random network contains a subnetwork that achieves good accuracy **with no training at all** — only masking. Proved formally (Malach et al.): a random network of polynomially larger width contains, with high probability, a subnetwork approximating any target network of the smaller width. ▸ **Pruning alone is a form of learning**, and this is the most surprising result in the area.
- **Not a practical speedup.** IMP requires training the dense network many times over, so it costs far *more* than dense training. The value is scientific, plus the connection to one-shot pruning methods (Ch. 17 §17.5) that *are* practical.

### The honest caveats

- Results are noisier and weaker at scale than the early papers suggested.
- Unstructured sparsity gives no hardware speedup (Ch. 17 §17.5).
- Simple magnitude pruning of a *trained* network, followed by fine-tuning, is competitive with IMP and far cheaper. **If you need a small model, this is what to do; LTH is what to understand.**

---

## 31.5 Mode connectivity and permutation symmetry

### Mode connectivity

▸ Two independently-trained networks are typically connected by a **path of near-constant low loss** through parameter space — not a straight line, but a simple curve (a quadratic Bézier or polygonal chain) found by optimization.

**Implication:** the loss landscape's minima are not isolated valleys. They are connected through high-dimensional manifolds — the landscape is far better connected than the 2-D pictures everyone draws.

### Linear mode connectivity and the permutation story

Two networks trained from the *same* initialization with different data order are usually **linearly** connected (the straight line between them has low loss). Two networks from *different* initializations are not — the barrier is large.

▸ **The resolution (Entezari et al.; Ainsworth et al., "Git Re-Basin"): permutation symmetry.** A neural network's hidden units can be permuted arbitrarily without changing the function, so each functional solution corresponds to $\prod_\ell(n_\ell!)$ parameter settings. Two independently-trained networks may be in the *same* basin but expressed under different permutations.

**Align the units first** (by matching activations or weights, solved as a linear assignment problem), and the barrier largely disappears:

▸ $$\mathcal{L}\big(\tfrac12\theta_A + \tfrac12\pi(\theta_B)\big)\ \approx\ \mathcal{L}(\theta_A)$$

**The conjecture this supports: modulo permutation symmetry, SGD solutions may all lie in a single basin.** That is a strong claim about the geometry of the loss landscape, and it is well-supported empirically for small and medium networks.

### The practical payoffs

- ▸ **Model merging works because of this** (Ch. 17 §17.8). Merging models fine-tuned from a *shared* base works because they never left the basin. Merging models from *different* seeds requires permutation alignment first.
- **Model soups** and Fisher-weighted averaging rest on the same fact.
- **Federated learning** benefits from aligning client models before averaging.
- **SWA/EMA** (Ch. 4 §4.9) work because the region being averaged over is a connected flat basin, so the average is a valid — and flatter — point in it.

---

## 31.6 The synthesis

▸ **The picture that emerges across Chapters 30 and 31:**

1. Overparameterized networks have vast sets of zero-training-error solutions.
2. **Which one you get is determined by the optimizer's implicit bias, not by capacity.** Gradient descent prefers minimum-norm, maximum-margin, low-rank, flat, simple solutions.
3. These preferences correlate with generalization — which is why the bounds of Chapter 2, which ignore the algorithm, are vacuous.
4. The solution space is *far* more connected than intuition suggests, once permutation symmetry is quotiented out.
5. **The optimizer is part of the model.** Changing SGD to Adam changes the hypothesis actually selected, not merely the speed of getting there.

**"Deep learning generalizes because SGD is a good regularizer" is the one-sentence version, and it is roughly right.**

---

## Check for Understanding

**Because overparameterized networks have infinitely many zero-loss solutions, generalization is determined by which one the optimizer selects — gradient descent provably converges to the max-margin direction on separable data (at a $1/\log t$ rate, which is why training past zero loss helps), Adam's different geometry gives it a different and often worse bias, SGD's noise temperature $\eta/B$ selects high-volume flat basins, and the lottery-ticket and mode-connectivity results together suggest that the solution set is one basin viewed through the permutation symmetry of hidden units.**

---

**Next:** [Chapter 32 — Mechanistic Interpretability & Superposition](32-mechanistic-interpretability-superposition.md)
