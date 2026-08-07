# Chapter 31 — Neural Collapse, Implicit Bias & Lottery Tickets

> **Prerequisites:** Ch. 2, Ch. 4, Ch. 30.

> **New to the notation?** If symbols like $\in$, $\sum$, $\langle\cdot,\cdot\rangle$, $\propto$, $\to$, or $\arg\min$ are unfamiliar — or if you have ever wondered why $\sigma$ seems to mean four different things — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $h_{i,c}$ | "h-i-c" | The **feature vector** of example $i$ from class $c$, taken from the second-to-last layer |
| $\mu_c$ | "mu-c" | The **class mean** — average feature vector of class $c$ |
| $\mu_G$ | "mu-G" | The **global mean** — average over everything |
| $\tilde\mu_c$ | "mu-c tilde" | The **centred** class mean, $\mu_c - \mu_G$ |
| $\Sigma_W$ | "Sigma-W" | **Within-class scatter** — how spread out one class's features are |
| $w_c$ | "w-c" | The classifier's **weight vector** for class $c$ |
| $C$ | "C" | The **number of classes** |
| $\langle a,b\rangle$ | "inner product" | Dot product — a similarity/alignment score |
| $\vartheta$ | "theta" | The **angle** between two class means |
| $\hat w_{\text{SVM}}$ | "w-hat SVM" | The **max-margin** solution a support vector machine would find |
| $\frac{w(t)}{\|w(t)\|}$ | "w over norm w" | The **direction** of the weights, with length divided out |
| $\theta_0$ | "theta-zero" | The **initialization** — weights before any training |
| $O(1/\log t)$ | "order one over log t" | A **very** slow rate — see below |

### Abbreviations used in this chapter

Full glossary in [Chapter 0 §0.13](00-notation-and-math-primer.md).

| Short | Full form |
|---|---|
| ETF | Equiangular Tight Frame |
| GD / SGD | (Stochastic) Gradient Descent |
| IMP | Iterative Magnitude Pruning |
| KKT | Karush–Kuhn–Tucker (optimality conditions) |
| LTH | Lottery Ticket Hypothesis |
| NC1–NC4 | Neural Collapse properties 1–4 |
| NTK | Neural Tangent Kernel |
| OOD | Out-Of-Distribution |
| SDE | Stochastic Differential Equation |
| SVM | Support Vector Machine |

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

#### Neural collapse in plain English

**What is being observed.** Train a classifier until it gets every training example right. Then *keep going*. Something strange and highly regular happens to the geometry of the second-to-last layer.

The four properties, stated without notation:

| Property | Plain statement |
|---|---|
| **NC1** | Every example of a class ends up at **the same point**. Individual differences are erased. |
| **NC2** | Those class points arrange themselves into the **most spread-out configuration possible** on a sphere |
| **NC3** | The classifier's weights become **copies of the class points** themselves |
| **NC4** | The network's decision reduces to "**which class point is nearest?**" |

▸ **Together: a deep network trained to convergence turns into a nearest-centroid classifier**, with the centroids arranged as symmetrically as geometry allows. All that architecture, and the endgame is *"find the closest prototype."*

> **Analogy.** A parade forming up. Early on, everyone mills around loosely near their group. As the drill sergeant keeps calling, each group compresses into a single tight cluster (NC1), and the clusters space themselves as far apart as the parade ground permits (NC2). Ask "which group is that person in?" and you simply find the nearest cluster (NC4).

**Why $-\frac{1}{C-1}$ is the magic number.** It's the cosine of the angle between class means, and it says: *"be as far apart as possible while summing to zero."*

Work the small cases:

| $C$ | $-\frac{1}{C-1}$ | Angle | Shape |
|---|---|---|---|
| 2 | $-1$ | 180° | Opposite ends of a line |
| 3 | $-0.5$ | 120° | Equilateral triangle |
| 4 | $-0.33$ | 109.5° | Tetrahedron |
| 1000 | $-0.001$ | ≈90° | Essentially orthogonal |

▸ **A "simplex" is just the generalization of triangle/tetrahedron to any dimension**, and an "equiangular tight frame" means all pairs are the same angle apart, spread as evenly as possible. **At large $C$ the angle approaches 90°** — which connects straight back to Chapter 1 §1.1.5: in high dimensions, near-orthogonal is the natural resting state, and there is room for many such directions.

**Why the negative sign?** Because the centred means must sum to zero (that's what centring does). Vectors summing to zero must, on average, point away from one another — you cannot have three vectors that all agree and still cancel.

▸ **The  important practical consequence is NC1, and it is bad news.** Collapsing all within-class variation to a point means **deliberately destroying information**: the difference between a husky and a malamute is erased once both are just "dog." That information is exactly what you need for transfer learning, fine-grained tasks, and detecting out-of-distribution inputs. **Training a classifier to convergence produces a great classifier and mediocre features** — which is a precise, quantitative argument for using self-supervised features (Chapter 25) when you want a representation rather than a decision.

#### Examples and non-examples: neural collapse

**✅  neural collapse**

| Situation | Why it qualifies |
|---|---|
| ResNet on CIFAR-10, trained 300 epochs past zero training error | Terminal phase, balanced classes |
| Any architecture, any dataset, once training error is 0 and you continue | The phenomenon is remarkably universal |

**❌ Near-misses — look like collapse, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| **Posterior collapse** in a VAE (Ch. 19) | Different phenomenon entirely — the latent is ignored | Unrelated, unfortunately similar name |
| **Mode collapse** in a GAN (Ch. 19) | Generator drops modes; nothing to do with class geometry | Unrelated, also similarly named |
| **Representational collapse** in SSL (Ch. 25) | Model outputs a constant for *all* inputs, a training failure | Neural collapse keeps classes **distinct** — it's a success state |
| Model stops improving on validation | Ordinary overfitting or convergence | Not a geometric statement |
| Features clustering by class early in training | Clustering, but not yet equinorm/equiangular | Partial NC1 only |

▸ **The boundary:** neural collapse is what happens when training goes *right* — it's the global optimum of the last-layer problem, not a bug. The three similarly-named "collapses" in this book are all **failures**; this one is a success with an unfortunate side effect. **Keeping these four straight is worth doing deliberately**, because the shared vocabulary causes real confusion.

> **Common misconception.** *"Neural collapse means the network has degenerated or overfit."* It is the **provable global minimizer** of cross-entropy plus weight decay in the unconstrained-features model. The network isn't malfunctioning; it's finding the max-margin geometry the objective asked for. The cost — destroyed within-class information — is a side effect of the objective, not a training failure.

> **Where this came from.** Neural collapse was identified and named by **Vardan Papyan, X. Y. Han, and David Donoho** in a 2020 paper in the *Proceedings of the National Academy of Sciences*. What made it notable was the universality: they observed the same four properties across many architectures and datasets, in a regime — training long past zero error — that conventional wisdom said was pointless. Donoho is a major figure in compressed sensing and high-dimensional statistics, and the framing shows it: the paper treats a deep network as an object whose *geometry* can be measured, rather than as a black box to be benchmarked.

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

#### What "implicit bias" means, and why it's the central question

**The puzzle, stated plainly.** A modern network has far more parameters than training examples. That means there are **infinitely many** weight settings that classify the training data perfectly. Most of them are terrible on new data. Gradient descent reliably finds a good one.

▸ **Nothing in the loss function asked for this.** The loss only says "get the training data right," and every one of those infinite solutions does. So the preference for good solutions is not coming from the objective — **it is coming from the optimizer itself.** That hidden preference is the "implicit bias," and identifying it is the central open problem of generalization theory.

> **Analogy.** You ask a contractor to build a house meeting a list of code requirements. Thousands of designs satisfy the code, and they differ enormously in quality. Two contractors given the identical spec produce different houses because each has habits the spec never mentioned. **Gradient descent is a contractor with good habits, and we are trying to write down what those habits are.**

**Reading the max-margin result.** For separable logistic regression, the weights grow without bound — $\|w\|\to\infty$ — because the loss can always be reduced a little more by becoming more confident. So the weight *vector* doesn't converge. But its **direction** does, and it converges to the max-margin solution: the boundary that leaves the widest possible gap between the classes.

▸ **That is a striking result.** Nobody added a margin term. Nobody regularized. Plain gradient descent on plain logistic loss ends up solving the same problem a support vector machine solves explicitly. **The optimizer smuggled in a preference for wide margins for free.**

**Why $O(1/\log t)$ is the important detail.** This is a  brutal rate. Put numbers on it:

| Steps $t$ | $1/\log t$ |
|---|---|
| $10^3$ | $0.145$ |
| $10^6$ | $0.072$ |
| $10^{12}$ | $0.036$ |

**A billion-fold increase in training halves the gap.** Compare with a typical $O(1/t)$ rate, where a billion-fold increase improves things a billion-fold.

▸ **Three consequences, all practical:**
1. The loss keeps improving long after training accuracy hits 100% — **that is not wasted computation**, it is margin still growing.
2. The margin you actually reach in finite time is **far from optimal**, so the implicit bias is real but weak — which is precisely why explicit regularization (weight decay, augmentation) still earns its keep.
3. It is a principled reason to train longer than the loss curve seems to justify.

> **Common misconception.** *"Once training accuracy hits 100%, further training is pointless or actively harmful."* The margin continues to improve, the geometry continues to organize (that's neural collapse), and test accuracy often continues to rise. Chapter 30's grokking is the extreme version — generalization appearing *long* after training loss flatlines. **Training accuracy reaching 100% tells you almost nothing about whether learning has finished.**

> **Common misconception.** *"Initialization just needs to be small enough to avoid exploding gradients."* The **scale** of initialization determines which regime you train in: large initialization keeps you in the lazy/NTK regime where features barely move, and destroys the low-rank and sparsity biases; small initialization enables  feature learning. **The initialization scale is a regularization hyperparameter**, which is not how most practitioners think of it.

#### Examples and non-examples: implicit bias

**✅  implicit bias**

| Example | The unrequested preference |
|---|---|
| GD on separable logistic loss → max-margin direction | Widest gap, never asked for |
| GD on least squares from $\theta_0 = 0$ → minimum $\ell_2$-norm interpolant | Smallest weights among perfect fits |
| Deep *linear* nets → low-rank / low nuclear norm | Simplicity in rank |
| Simplicity bias → low-frequency functions fit first | Smooth before jagged |

**❌ Near-misses — not implicit bias**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Weight decay shrinking weights | You wrote it into the loss | **Explicit** regularization |
| Dropout improving generalization | Deliberately added | Explicit stochastic regularization |
| Early stopping | A decision you made | Explicit (though it *approximates* $\ell_2$, Ch. 7) |
| Data augmentation | You chose the invariances | Explicit, via the data |
| A smaller model generalizing better | Capacity control by design | Explicit architectural choice |

▸ **The boundary:** implicit bias is a preference the optimizer exhibits that **nobody wrote down**. If you can point to the line of code or the term in the loss that caused it, it's explicit. The interesting and unsettling part is that the implicit effects are often *stronger* than the explicit ones — and we have rigorous characterizations only for linear and near-linear cases.

> **Where this came from.** The max-margin result for logistic regression is due to **Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro** in 2018 — a paper that crystallized a question the field had been circling since Zhang and colleagues showed in 2016 that networks can perfectly fit *random labels*. That earlier result was devastating for classical theory: if a network can memorize noise, no capacity-based bound can explain why it generalizes on real data. The implicit-bias programme is the field's answer — **stop asking what the model *can* represent, and start asking what the optimizer actually *finds*.** Chapter 2 walks through why the capacity bounds fail; this chapter is the constructive follow-up.

### Adam has a different bias — and it's worse

▸ SGD's bias is toward minimum $\ell_2$ norm / max $\ell_2$ margin. **Adam's coordinate-wise normalization changes the geometry**: its implicit bias is closer to $\ell_\infty$-geometry, i.e. toward maximum $\ell_1$-margin.

**This is the explanation for the persistent Adam-generalization gap** (Ch. 5 §5.6): Adam converges faster but often to a worse-generalizing solution on vision tasks, while being essential for transformers. It is a  difference in what the optimizer prefers, not merely a tuning artifact.

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

#### The Lottery Ticket Hypothesis, decoded

**The name explains the idea.** Buying many lottery tickets improves your odds not because any individual ticket is good, but because you bought a lot of them. The hypothesis says a big network is a **big pile of tickets**: most of its random subnetworks are useless, but with enough of them, a few start out lucky — positioned so that training works well.

▸ **This reframes what overparameterization is for.** The usual story is that extra parameters add capacity. The lottery story says they add **draws**: a wide network isn't one model with many knobs, it's an ensemble of enormously many candidate subnetworks, and training amplifies whichever ones began in a favourable spot.

**Why step 4 is the whole hypothesis.** The procedure resets surviving weights to their **original random values** $\theta_0$, not to fresh random values. That distinction is everything:

| What you reset to | Result |
|---|---|
| The original $\theta_0$ values | Works — matches full network accuracy |
| **Fresh** random values, same mask | **Fails badly** |

▸ So the winning ticket is the **pair (mask, initialization)**, not the mask alone. The specific starting numbers matter, not just which connections exist. **The structure is not sufficient** — and that is what makes the claim surprising rather than obvious.

**Why rewinding matters, and what it quietly concedes.** At ImageNet scale, resetting to $\theta_0$ stops working; you must reset to $\theta_k$ after a small amount of training. That is a real qualification: **the ticket isn't present at initialization, it forms during the first few hundred steps.** Before that point two runs diverge into different basins; after it they stay in the same one. For any realistic model, "the winning initialization" means "shortly after initialization."

**The strong version, and why it's startling.** A large enough random network contains a subnetwork that performs well **with no training at all** — found purely by choosing which weights to keep.

▸ **Pruning alone is a form of learning.** Selecting a subnetwork carries enough information to specify a working model. Every bit of "which weights to keep" is a bit of learned information, so masking is not merely a compression step — it is training by a different mechanism.

#### Examples and non-examples: lottery tickets

**✅  lottery-ticket findings**

| Claim | Status |
|---|---|
| Sparse subnetwork + original init matches dense accuracy | Reproduced on small/medium scale |
| Tickets transfer across related datasets | Supported |
| Random networks contain good untrained subnetworks | Proved formally |

**❌ Near-misses — what LTH is *not***

| Common claim | Why it's wrong |
|---|---|
| "LTH gives you faster training" | IMP trains the dense net **many times over** — strictly more expensive |
| "Just prune to 95% and get a 20× speedup" | Unstructured sparsity gives **no hardware speedup** without structured patterns |
| "The mask is the ticket" | Fresh random values with the same mask fail — init is half the ticket |
| "Tickets exist at initialization" | At scale they form after a few hundred steps (rewinding) |
| "This is how you should compress a model" | Prune-a-trained-net + fine-tune is competitive and far cheaper |

▸ **The boundary:** the Lottery Ticket Hypothesis is a **scientific claim about why overparameterization helps**, not an engineering recipe. It tells you something true about the nature of training and almost nothing about how to ship a smaller model. Chapter 17's one-shot pruning is the practical descendant.

> **Common misconception.** *"Winning tickets prove most of the network is useless."* The dense network was **necessary to find the ticket** — you cannot identify it without training the full model, repeatedly. The extra parameters did work; they just did it by providing enough draws, rather than by all being present in the final model. Removing them up front doesn't work, which is the whole point.

> **Where this came from.** **Jonathan Frankle and Michael Carbin** introduced the Lottery Ticket Hypothesis at ICLR 2019, where it won a best paper award. It set off a large replication effort — much of which found the effect weaker and noisier at scale than the original results suggested, prompting Frankle and colleagues' own 2020 rewinding correction. The episode is a credit to the field: the authors published the limitation themselves rather than leaving it to critics. The **strong** version, that untrained random networks contain good subnetworks, came from Vivek Ramanujan and colleagues in 2020, with a formal proof by Eran Malach and coauthors showing a random network of polynomially larger width contains, with high probability, a subnetwork approximating any smaller target network.

---

## Did you know?

- **A trained deep network's last layer becomes a nearest-centroid classifier.** After all that architecture, neural collapse (NC4) shows the final decision reduces to "which class prototype is closest?" — the simplest classifier there is.

- **The magic angle in neural collapse is 120° for three classes** and approaches 90° as the number of classes grows. It is the most spread-out arrangement possible for points that must sum to zero — an equilateral triangle for $C=3$, a tetrahedron for $C=4$.

- **This book contains four different "collapses" and they are unrelated.** Neural collapse (a success state), posterior collapse (a VAE failure), mode collapse (a GAN failure), and representational collapse (a self-supervised learning failure). The shared vocabulary causes  confusion.

- **Training a classifier to convergence makes it worse as a feature extractor.** NC1 erases within-class variation — precisely the information needed for transfer, fine-grained distinction, and out-of-distribution detection. Your classifier gets better while your representation gets worse.

- **Plain gradient descent secretly solves the support vector machine problem.** On separable logistic regression it converges in direction to the max-margin solution, with no regularization and no early stopping. Nobody asked for a margin; the optimizer supplies one.

- **That convergence rate is $O(1/\log t)$** — increasing training a *billion-fold* only halves the gap to optimal. It is one of the slowest meaningful rates in the field, and it explains why explicit regularization still helps despite the implicit bias being real.

- **The 2016 finding that networks can perfectly memorize random labels broke classical learning theory.** If a model can fit pure noise, no capacity-based bound can explain why it generalizes on real data. The entire implicit-bias research programme is the field's response.

- **Initialization scale is a regularization hyperparameter.** Large initialization traps you in the lazy/neural-tangent-kernel regime where features barely move; small initialization enables real feature learning. Almost nobody thinks of initialization this way.

- **The Lottery Ticket paper won a best paper award, and its authors later published its most important limitation themselves.** Resetting to the true initialization fails at ImageNet scale; you must rewind to a few hundred steps in. The ticket forms shortly *after* initialization, not at it.

- **A large enough random, untrained network already contains a subnetwork that works.** You can get a functioning model by choosing which weights to delete and never adjusting a single one. Pruning, it turns out, is itself a form of learning.

- **The Lottery Ticket Hypothesis costs more compute than ordinary training, not less.** Iterative magnitude pruning requires training the dense network many times over. Its value is scientific insight, not efficiency — a distinction frequently lost in summaries.

- **Two independently trained networks are usually connected by a low-loss path** through parameter space — not a straight line, but a simple curve. The loss landscape's separate "valleys" are, in a real sense, one connected basin once you account for permutation symmetry.

---

## Check for Understanding

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

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What are the four neural collapse properties**, stated without any notation? (The parade-ground analogy.)
2. **Why is the angle between class means $-1/(C-1)$?** What shape is that for 3 classes?
3. **Why does neural collapse make a classifier *worse* as a feature extractor**, even as it gets better at classifying?
4. **Distinguish the four "collapses" in this book** — neural, posterior, mode, and representational. Which one is a success?
5. **State the implicit bias puzzle**: why is it strange that gradient descent finds a solution that generalizes? (The contractor analogy.)
6. **Why does plain gradient descent end up solving the support vector machine problem** without being asked?
7. **Why does an $O(1/\log t)$ rate mean explicit regularization still matters**, even though the implicit bias is real?
8. **Why is training past 100% training accuracy not wasted computation?**
9. **Why is initialization *scale* a regularization choice** and not merely a numerical-stability one?
10. **Why is a winning ticket the pair (mask, initialization) rather than the mask alone?** What happens with fresh random values?
11. **What did "rewinding" quietly concede** about the original Lottery Ticket Hypothesis?
12. **Why is the Lottery Ticket Hypothesis not a way to train faster?** What is it actually good for?

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 32 — Mechanistic Interpretability & Superposition](32-mechanistic-interpretability-superposition.md)
