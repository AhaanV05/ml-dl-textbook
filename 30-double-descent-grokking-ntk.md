# Chapter 30 — Double Descent, Grokking & the NTK

> **Prerequisites:** Ch. 2, Ch. 4, Ch. 22 (§22.2 ridge/SVD).
> **This is where the textbook story of machine learning breaks and the modern one begins.**

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $n$ | "n" | How many **training examples** you have |
| $p$ | "p" | How many **parameters** (or features) the model has |
| $p = n$ | "p equals n" | The **interpolation threshold** — exactly as many knobs as data points |
| $X$ | "the design matrix" | The data laid out as a grid: one row per example, one column per feature |
| $\hat\beta$ | "beta-hat" | The fitted coefficients. The hat means "estimated" (§0.6) |
| $X^+$ | "X-plus" / "X-dagger" | The **pseudo-inverse** — the closest thing to an inverse a non-square matrix has |
| $\sigma_j$ | "sigma-j" | A **singular value** of $X$ (§1.1.3) — here, *not* a standard deviation |
| $\lambda$ | "lambda" | Ridge regularization strength, or an eigenvalue. Both appear; context decides |
| $\Theta(x,x')$ | "the neural tangent kernel of x and x-prime" | How much training on example $x'$ moves the prediction at $x$ |
| $\nabla_\theta f(x;\theta_0)$ | "grad-theta f at theta-nought" | How the network's output at $x$ responds to each parameter, **measured at initialization** |
| $\phi(x)$ | "phi of x" | The feature map — here it equals the gradient above, and it never changes |
| $\theta_0$ | "theta-nought" | The parameters at initialization, before any training |
| $\dfrac{df(x)}{dt}$ | "d f by d t" | **Gradient flow** — gradient descent with infinitesimally small steps, so time is continuous |
| $e^{-\eta\Theta t}$ | "e to the minus eta Theta t" | Exponential decay of the error, one rate per kernel eigendirection |
| $a\circ b = (a+b)\bmod 97$ | "a plus b, mod 97" | Add, then keep only the remainder after dividing by 97 |
| $\omega_k$ | "omega-k" | A **frequency** the network discovers in its embeddings |

### ⚠ The one notation trap in this chapter

**$\Theta$ means two completely unrelated things here, and they appear within a page of each other.**

| Written | Means | How to tell |
|---|---|---|
| $\Theta(x,x')$, $\Theta$ | The **neural tangent kernel** | Takes arguments that are *data points*, or stands alone as a matrix |
| $\Theta(1)$, $\Theta(\cdot)$ | **Big-Theta** asymptotic notation: "grows exactly like" (§0.10) | Appears in a table of *rates*, next to $O(\cdot)$ |

▸ In the lazy-versus-rich table below, the row "Weight movement: $O(1/\sqrt{\text{width}})$ versus $\Theta(1)$" uses the **asymptotic** sense — it means "shrinks with width" versus "stays a constant." Nothing to do with the kernel. This collision is unfortunate and universal in the literature; there is no way to read around it except to know it is there.

### Full forms of every abbreviation in this chapter

| Short | Full form |
|---|---|
| NTK | Neural Tangent Kernel |
| NNGP | Neural Network Gaussian Process |
| ODE | Ordinary Differential Equation |
| SVD | Singular Value Decomposition |
| OLS | Ordinary Least Squares |
| MSE | Mean Squared Error |
| SGD | Stochastic Gradient Descent |
| LR | Learning Rate |
| EMC | Effective Model Complexity |
| DFT | **D**iscrete **F**ourier **T**ransform (in Chapter 29 the same letters meant density functional theory) |
| $\mu$P | **m**aximal **u**pdate **P**arameterization |
| MLP | Multi-Layer Perceptron |
| ReLU | Rectified Linear Unit |
| PNAS | Proceedings of the National Academy of Sciences (where several of these results were published) |

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
- **Sample-wise:** ▸ **more data can make test error worse**, if it moves you from the overparameterized regime toward the interpolation threshold.  counterintuitive, and empirically verified.

#### The three variants, decoded — and why they are one phenomenon

First, the vocabulary that the rest of the chapter assumes.

- **Interpolate** — hit every training point *exactly*. Training error is literally zero, not small. In classical statistics this was a synonym for "you have ruined the model."
- **Overparameterized** — more parameters $p$ than training examples $n$. Modern language models sit at $p/n$ ratios that would have been considered absurd in 1995.
- **The interpolation threshold** — the capacity at which the model becomes *just barely* able to interpolate. Not before, not comfortably after. **This exact point is where everything goes wrong.**

▸ **The unifying statement: the peak occurs wherever the model is exactly stretched to its limit, and all three variants are different ways of walking past that point.**

| Variant | What you vary | What you hold fixed | The move that crosses the threshold |
|---|---|---|---|
| **Model-wise** | model size $p$ | data $n$, epochs | Grow the model until $p \approx n$ |
| **Epoch-wise** | training time | model size, data | Train longer — **effective** capacity grows during training |
| **Sample-wise** | data $n$ | model size $p$ | *Add* data until $n \approx p$ |

**Epoch-wise, unpacked.** A model's *usable* capacity is not fixed at the start; it grows as training explores more of parameter space. Early in training a large network behaves like a small one — Chapter 30's spectral bias says it fits smooth structure first — so it effectively creeps rightward along the same $x$-axis as it trains. **When it passes through "just barely able to fit the training set," test error rises, then falls again as it moves comfortably past.**

▸ **This is why "early stopping is always right" is false**, and it is not a small correction. The standard practice of stopping at the first upturn in validation error would, on such a run, stop you precisely at the worst point and never let you reach the better solution beyond it.

**Sample-wise, unpacked — the  disturbing one.** Fix the model at $p = 10{,}000$ parameters. You have $n = 500$ examples and the model is comfortably overparameterized ($p/n = 20$). You collect more data. At $n = 10{,}000$ you are at the threshold, and **your test error is worse than it was with 500 examples.** Push on to $n = 50{,}000$ and it improves again.

> **Analogy for all three.** A bookshelf and a pile of books. With few books they sit loosely — you can arrange them however you like. With **exactly** as many books as the shelf holds, they are jammed in edge to edge, every one under pressure, and the whole row is one nudge from buckling. With far more shelf than books, everything relaxes again. **The trouble is not too many books or too few — it is the precise ratio at which the system has no slack.**

**Why the whole classical story was built on the left half.** Until roughly 2015, essentially no one trained models past the interpolation threshold on purpose, because the classical theory said the region beyond it was catastrophe. The U-curve was not wrong — it is the accurate description of everything to the left of the peak — it was **incomplete**, in the way a map of the coastline is incomplete about what lies out at sea. Nobody had sailed there.

#### Examples and non-examples: is that bump really double descent?

A rise-then-fall in a test-error curve has several possible causes, and only one of them is this. Misdiagnosing it wastes weeks, because the prescribed responses are opposite: double descent says *keep going*, and most of the alternatives say *stop and fix something*.

**✅  double descent**

| Case | What was varied | The signature |
|---|---|---|
| ResNet-18 on CIFAR-10, width from 1 to 64 channels, label noise 15% | Model size | Test error peaks where the model first reaches zero *training* error, then falls below the earlier minimum |
| Ridgeless linear regression, $p$ swept from $0.1n$ to $10n$ | Feature count | A pole at $p=n$ predicted exactly by Marchenko–Pastur |
| A fixed transformer trained for $10^6$ steps with no weight decay | Training time | Test loss rises around the epoch where training loss first hits zero, then falls again |
| A fixed model, $n$ swept upward past $p$ | Dataset size | **More data, worse test error**, then better again |
| Random-feature regression with $p$ random features | Feature count | The cleanest lab version — reproducible in twenty lines of NumPy |

**❌ Near-misses — bumps that are not double descent**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Test error rises while training error also rises | Double descent's peak sits at or past the point of **zero training error** | An optimization failure — learning rate too high, or a bad initialization |
| Validation error rises then falls over 3 epochs | The peak is a property of capacity, not of a warmup schedule | A learning-rate warmup or schedule artifact |
| Test error wobbles by $\pm0.3\%$ across model sizes | The peak is a large, systematic, reproducible spike | Evaluation noise. Ch. 3: check your error bars before believing a shape |
| Test error rises as you add parameters, and never comes back down | The second descent is the defining half of the phenomenon | Ordinary overfitting, usually with too little data or no regularization |
| Test error rises when you add a new data source | Nothing to do with $n$ versus $p$ | Distribution shift — the new data is not from the same distribution |
| A bump that vanishes when you turn up weight decay | With optimal ridge the curve is monotone by construction | **Under-regularization** — and this is the first hypothesis you should test |

▸ **The boundary:** double descent is the peak that occurs **exactly where the model becomes just barely able to interpolate the training data**, and it is followed by a descent to a *lower* error than the classical optimum. No zero training error, no interpolation threshold, no double descent. Anything without the second descent is just overfitting with extra steps.

> **Common misconception.** *"Double descent proves the bias–variance tradeoff is wrong."* It proves the classical curve was **incomplete**, which is a much weaker and much more useful claim. Everything to the left of the peak still behaves exactly as Geman, Bienenstock and Doursat described, and — the detail most people skip — **with a well-chosen ridge penalty the peak disappears entirely and the curve is monotone.** So double descent is a phenomenon of *unregularized interpolation*, not a refutation of regularized statistics. The misconception is tempting because "the textbook curve is wrong" is a far better story than "the textbook curve doesn't cover a regime the textbook told you never to enter."

> **Common misconception.** *"Bigger models always overfit less, so scale is a free lunch."* The right-hand descent is real and it is not unconditional. The benign-overfitting theory in §30.1 gives the actual requirement — a data covariance spectrum with a few large eigenvalues and a long thin tail — and you can construct a hundred-million-parameter model whose data has a short, flat spectrum that overfits precisely as classical theory predicts. **"Big models don't overfit" is false as stated.** It is tempting because in the domains people actually work in (images, text), the spectral condition happens to hold, so the shortcut has been reliable enough to feel like a law.

> **Where the U-curve came from, and the irony in it.** The **bias–variance dilemma** was made canonical for this field by **Stuart Geman, Elie Bienenstock and René Doursat** in a long, influential 1992 paper in *Neural Computation* titled *Neural Networks and the Bias/Variance Dilemma*. Their argument, laid out carefully and at the time entirely persuasive, was that neural networks with many parameters would be doomed by variance, and that useful learning would therefore require substantial built-in structure and prior knowledge rather than raw capacity. **A paper about neural networks became the standard citation for the principle that modern neural networks violate.** The double-descent picture was named and popularized by **Mikhail Belkin, Daniel Hsu, Siyuan Ma and Soumik Mandal** in *Reconciling modern machine learning practice and the bias–variance trade-off* (PNAS, 2019), and demonstrated at scale for deep networks by **Preetum Nakkiran** and colleagues at OpenAI in *Deep Double Descent* (2020). But the peak itself was not new: physicists studying the **statistical mechanics of learning** in the late 1980s and 1990s — a literature associated with names including Manfred Opper, Sompolinsky, Tishby and Seung — had derived learning curves with exactly this non-monotonic shape for simple models. **It was published, correct, and then effectively forgotten for twenty-five years, because it described a regime nobody had any reason to train in.**

### Why the peak exists — the linear analysis

Consider minimum-$\ell_2$-norm interpolation ("ridgeless regression"): with $n$ samples and $p$ features,

$$\hat\beta = X^+y = X^\top(XX^\top)^{-1}y \quad (p>n)$$

The estimator's variance involves $(X^\top X)^{-1}$ or its pseudo-inverse, whose scale is set by the **smallest nonzero singular value** of $X$.

▸ **At $p=n$, $X$ is square and generically nearly singular** — by random-matrix theory (the Marchenko–Pastur law), the smallest singular value of an $n\times n$ random matrix concentrates near **zero**. So $\|\hat\beta\|\to\infty$ and the variance blows up. **That is the pole, and it is a property of the linear algebra, not of neural networks.**

- $p<n$: the system is overdetermined, $X^\top X$ is well-conditioned, ordinary least squares behaves classically.
- $p=n$: exactly determined, no slack, the fit is forced through every point by a nearly-singular inverse.
- $p>n$: **underdetermined — infinitely many exact solutions exist, and we get to pick.** The minimum-norm choice is smooth, and as $p$ grows the solution space grows so the minimum-norm element gets *smaller* in norm. Variance falls.

▸ **The key conceptual shift: in the overparameterized regime, what matters is not "how many solutions fit the data" but "which one the optimizer picks."** Capacity stops being the controlling variable and *implicit bias* takes over (Ch. 31).

#### Reading the minimum-norm interpolation argument

This is the load-bearing derivation of the chapter and it uses nothing beyond Chapter 1. Take it slowly.

**What "ridgeless regression" means.** Ordinary ridge regression minimizes $\|y - X\beta\|^2 + \lambda\|\beta\|^2$: fit the data, and don't let the coefficients get large. **Ridgeless** means $\lambda = 0$ — the second term is switched off entirely. No regularization at all. Remember this: **double descent is a story about what happens when you remove the safety rail.**

**Reading the formula.** $\hat\beta = X^+y = X^\top(XX^\top)^{-1}y$

| Piece | Read aloud | Shape | Job |
|---|---|---|---|
| $X$ | "the design matrix" | $n\times p$ | Your data: $n$ examples, $p$ features |
| $y$ | "y" | $n$ | The targets |
| $X^+$ | "X pseudo-inverse" | $p\times n$ | The best stand-in for $X^{-1}$ when $X$ isn't square |
| $XX^\top$ | "X X transpose" | $n\times n$ | Note the shape: it is $n\times n$, **not** $p\times p$ — this form is the one that works when $p>n$ |

**The pseudo-inverse, in one sentence.** A square, well-behaved matrix has an inverse that undoes it. A non-square one does not, so $X^+$ does the next best thing: **among all $\beta$ that fit the data exactly, it returns the one with the smallest $\|\beta\|_2$.** That choice — "smallest norm" — is not a neutral technical default. It is the entire reason overparameterization works, and it is the ancestor of every implicit-bias result in Chapter 31.

**A two-parameter example you can do in your head.** One data point: $x = (1,1)$, $y = 2$. Fit $\beta_1 x_1 + \beta_2 x_2 = 2$. There are **infinitely many** exact solutions — every point on the line $\beta_1 + \beta_2 = 2$:

| Solution $\beta$ | Fits the data? | $\|\beta\|_2$ |
|---|---|---|
| $(2, 0)$ | ✓ | $2.00$ |
| $(1, 1)$ | ✓ | $1.41$ ← **minimum-norm** |
| $(10, -8)$ | ✓ | $12.81$ |
| $(500,-498)$ | ✓ | $705.7$ |

▸ **Every row fits the training data perfectly, and they will behave wildly differently on a new point.** The last one is a hair-trigger: change $x_1$ by $0.01$ and the prediction moves by $5$. The pseudo-inverse picks row two — the placid one. **"Which of the infinitely many perfect fits do you take?" is the only question that matters here, and the answer is a property of the algorithm, not of the model class.**

**Now why the norm *shrinks* as $p$ grows.** Same setup, but with $p$ features all equal to 1 and a target of 1. The minimum-norm solution spreads the job evenly: $\beta_j = 1/p$ for each $j$, so

$$\|\hat\beta\|_2 = \sqrt{p\cdot\left(\tfrac1p\right)^2} = \frac{1}{\sqrt p}$$

| $p$ | $\|\hat\beta\|_2$ |
|---|---|
| 1 | $1.000$ |
| 100 | $0.100$ |
| 10,000 | $0.010$ |

▸ **More parameters means each one has to do less.** The solution set grows, and the smallest element of a bigger set is smaller. This is the entire right-hand branch of the double-descent curve in one line: **the descent after the peak is the minimum-norm solution getting quieter as the space of exact fits gets roomier.**

**Now the peak itself, and where the singular values come in.** The variance of the estimator is governed by $1/\sigma_{\min}^2$ — the *smallest* singular value of $X$. Small singular value means the matrix nearly squashes some direction to nothing (§1.1.3), and inverting it means dividing by nearly nothing.

> **Analogy.** A set of bathroom scales that reads to the nearest kilogram. Weigh an elephant and the reading is fine. Weigh a feather by putting it on the scales and reading the difference — you divide a rounding error by an almost-zero quantity, and the answer is nonsense. **Inverting a near-singular matrix is dividing by a feather.**

**The Marchenko–Pastur law says exactly how bad it gets.** For a random $n\times p$ matrix with $\gamma = p/n$, the eigenvalues of $\frac1n X^\top X$ fall in the window $\big[(1-\sqrt\gamma)^2,\ (1+\sqrt\gamma)^2\big]$. Watch the lower edge:

| $p/n$ | Smallest eigenvalue $(1-\sqrt{p/n})^2$ | Variance blow-up $\propto 1/\lambda_{\min}$ |
|---|---|---|
| $0.5$ | $0.086$ | $\times 12$ |
| $0.9$ | $0.0026$ | $\times 380$ |
| $0.99$ | $0.000025$ | $\times 40{,}000$ |
| $1.0$ | $0$ | $\infty$ |

▸ **That last row is the pole.** It is not an artifact of neural networks, of ReLU, of SGD, or of any modelling choice — **it is what happens to random matrices when they become square.** A square random matrix is generically almost-singular, and the entire interpolation peak is that fact, seen through a fitting procedure.

**The three regimes, said out loud:**

- **$p < n$ — overdetermined.** More equations than unknowns; no exact fit exists; least squares finds the best compromise. Classical statistics, well-behaved.
- **$p = n$ — exactly determined.** Exactly one solution exists, and you are forced to take it, however violent. **No slack, no choice.**
- **$p > n$ — underdetermined.** Infinitely many exact solutions exist. **And now you get to choose**, which changes the nature of the problem completely.

#### Examples and non-examples: interpolation

The word does a lot of work in this chapter, and it does not mean what it means in classical statistics texts.

**✅  interpolation**

| Example | Training error | Why it counts |
|---|---|---|
| A 1-nearest-neighbour classifier | Exactly 0 | Every training point is its own nearest neighbour. The oldest interpolating method there is |
| Minimum-norm ridgeless regression with $p>n$ | Exactly 0 | The pseudo-inverse returns an exact solution |
| A ResNet trained to zero training loss on CIFAR-10 with **randomized labels** | Exactly 0 | Zhang et al.'s experiment: 50,000 arbitrary labels memorized perfectly |
| A wide network trained past the point where training accuracy hits 100% | Exactly 0 | The regime every large model is trained in today |
| A degree-$n$ polynomial through $n+1$ points | Exactly 0 | The classical textbook cautionary tale, which is also a  example |

**❌ Near-misses — often called interpolation, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Training loss of $0.003$ | Not zero. The distinction matters because the *gradient from the data* is what has to vanish before implicit bias takes over | A very good fit, still being pushed by the data |
| 100% training **accuracy** with nonzero cross-entropy | The argmax is right everywhere and the loss still has a gradient — logits keep growing | Accuracy-interpolation, not loss-interpolation. This gap is exactly what drives margin maximization (Ch. 31) |
| A model that memorizes 99% of the training set | One missed point means the constraint set is not the zero-loss manifold | Nearly interpolating, and the theory's guarantees are about the exact case |
| Overfitting | Overfitting is a *statement about test error*; interpolation is a statement about **training** error | Two independent facts — benign overfitting is the case where you interpolate and generalize anyway |
| A model with more parameters than data | Capacity to interpolate is not the same as having done so | Overparameterized. It becomes an interpolator only after you train it there |

▸ **The boundary:** interpolation means **training error exactly zero**, which is a claim about the training set and says nothing whatever about the test set. The single most useful reframe in this chapter is that *interpolation and overfitting are different words for different things*, and classical practice conflated them because in the regimes it studied they always coincided.

> **Common misconception.** *"An interpolating model has memorized and therefore cannot generalize."* Consider 1-nearest-neighbour: it interpolates by construction, has been known since the 1950s, and is a consistent estimator under mild conditions — it does generalize. The intuition that fitting every point is fatal comes from the *polynomial* case, where the interpolant is forced to oscillate violently between points. Change the interpolant to a smooth minimum-norm one and the oscillation goes away. **What kills you is not passing through every point; it is what your model does *between* the points**, and that is decided by which interpolant your algorithm selects.

**What would break — a concrete numerical demonstration of the pole.** Take $n = 100$ samples and sweep $p$. The minimum-norm solution's size behaves like this:

| $p$ | Regime | Typical $\lVert\hat\beta\rVert$ | Test error |
|---|---|---|---|
| $10$ | Underparameterized | small | good |
| $95$ | Approaching the threshold | growing fast | degrading |
| $100$ | **Exactly at it** | enormous — the matrix is near-singular | **worst point on the curve** |
| $150$ | Just past | falling | recovering |
| $10{,}000$ | Deeply overparameterized | smallest of all | **best point on the curve** |

▸ **The failure at $p=n$ is division by a nearly-zero singular value, and nothing else.** Set $\lambda = 10^{-6}$ instead of exactly $0$ and the pole is capped: you are no longer inverting a feather, you are inverting a feather plus a microgram. **The peak is a numerical event with a statistical shadow**, which is why the tiniest amount of ridge regularization removes it.

> **Where random matrix theory came from.** **Vladimir Marchenko and Leonid Pastur** published their law in 1967, in the Soviet Union, as a result in pure mathematics about the limiting spectrum of large random covariance matrices. The field they were working in had been founded by **Eugene Wigner** in the 1950s for an entirely different purpose: Wigner wanted to predict the spacing of energy levels in heavy atomic nuclei, which are far too complicated to solve exactly, and his radical proposal was to **replace the true Hamiltonian with a random matrix and study the statistics instead.** It worked, and it produced the semicircle law. **The mathematics that explains why your model's test error spikes at $p=n$ was invented to describe uranium nuclei**, which is a good indication of how far the pipeline from physics to machine learning still runs.

### Effective rather than raw parameter count

Raw parameter count is the wrong $x$-axis. Better measures place the peak more reliably:
- Effective degrees of freedom $\sum_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}$ (Ch. 22 §22.2).
- Weight-norm-based capacity measures.
- Nakkiran et al.'s **effective model complexity**: the largest $n$ for which the training procedure achieves ≈0 training error.

▸ **Explicit regularization mitigates or removes the peak.** With optimal ridge $\lambda$, the double-descent curve becomes monotone. This is important: **double descent is a phenomenon of *unregularized* interpolation.** It says less about the impossibility of classical theory than about what happens when you turn regularization off.

#### Why counting parameters is the wrong axis

**Raw parameter count assumes every parameter is doing an equal share of work, and that is essentially never true.** A network can have ten million weights of which, functionally, a few thousand matter. Plotting test error against the raw count is like measuring a company's output by counting employees.

**Reading the effective-degrees-of-freedom formula.**

$$\mathrm{df}(\lambda) = \sum_j \frac{\sigma_j^2}{\sigma_j^2+\lambda}$$

Look at one term. It is a fraction between 0 and 1 that answers: *"is this direction actually being used?"*

| Situation | $\dfrac{\sigma_j^2}{\sigma_j^2+\lambda}$ | Reads as |
|---|---|---|
| $\sigma_j^2 \gg \lambda$ (strong direction) | $\approx 1$ | "This direction counts as one full parameter" |
| $\sigma_j^2 \approx \lambda$ | $= 0.5$ | "Half a parameter" |
| $\sigma_j^2 \ll \lambda$ (weak direction) | $\approx 0$ | "Regularization has switched this one off; it costs nothing" |

Put numbers in with $\lambda = 1$ and singular values $\sigma^2 = (100, 10, 1, 0.1, 0.01)$:

$$\frac{100}{101} + \frac{10}{11} + \frac{1}{2} + \frac{0.1}{1.1} + \frac{0.01}{1.01} = 0.99 + 0.91 + 0.50 + 0.09 + 0.01 = 2.50$$

▸ **Five parameters, but only two and a half of them are switched on.** As $\lambda\to0$ every term goes to 1 and you recover the raw count; as $\lambda\to\infty$ every term goes to 0 and the model is a constant. **The regularizer is a dimmer switch on capacity, and this sum is the reading on the dial.** It is exactly the same style of quantity as stable rank in §1.1.3 and perplexity in §1.4.2 — a smooth, honest count that replaces a brittle integer one.

**Effective model complexity, decoded.** Nakkiran and colleagues defined it operationally rather than analytically: **EMC is the largest training-set size $n$ for which this exact training procedure still drives training error to about zero.** Note what is being measured — not the architecture, but *the architecture plus the optimizer plus the schedule plus the number of epochs, as one inseparable object.*

▸ **That definition is the chapter's thesis in disguise.** If capacity depended only on the model, you could measure it once. Defining it by what the training procedure achieves is an admission that **the optimizer is part of the model** — the line that Chapter 31 §31.6 arrives at from the other direction.

And it immediately explains epoch-wise double descent: train longer and the same architecture can fit larger training sets, so its EMC rises *during* the run. **The model walks along the $x$-axis of the double-descent plot while you watch, and crosses the peak partway through.**

**Now the caveat, which deserves more attention than it usually gets.** With a well-chosen ridge $\lambda$, the peak *disappears* and the curve is monotone. So:

- Double descent is **not** evidence that classical statistics was wrong. It is evidence about a regime — unregularized exact interpolation — that classical practice never entered.
- ▸ **The practical reading: if you observe a test-error bump as you scale a model, your first hypothesis should be under-regularization, not a deep truth about overparameterization.** Turn up weight decay and see if the bump goes away.
- The reason the phenomenon matters anyway is that modern deep learning  does run close to the ridgeless regime — weight decay of $0.1$ is not "optimal ridge," and a great deal of what makes large models work is implicit rather than explicit regularization.

### Benign overfitting

The theory (Bartlett, Long, Lugosi, Tsigler) explains *when* interpolating noise is harmless: when the data covariance spectrum has a few large eigenvalues carrying signal and a long tail of many small ones. Then the noise is absorbed by the tail directions — spread thinly across thousands of low-variance components, each contributing negligibly to predictions — while the signal is fit in the top directions.

▸ **The high-dimensional tail acts as a sink for noise.** This requires a specific spectral structure; overparameterization alone is not sufficient. Being able to state this condition, rather than just saying "big models don't overfit," is the mark of understanding the result.

#### Benign overfitting, decoded

**The puzzle it resolves.** Classical statistics says fitting noise is fatal: memorize the random errors in your training set and you will carry them into your predictions. Modern models fit the noise *exactly* — training loss zero, on data that certainly contains label errors — and generalize fine anyway. **Benign overfitting is the theory of when that is allowed.**

**"Data covariance spectrum," decoded.** The covariance matrix $\Sigma$ of your inputs has eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$, each measuring how much the data varies along one direction (§1.1.2). The *spectrum* is that sorted list. The theory says the shape of the list decides everything:

| Direction type | Eigenvalue | What lives there |
|---|---|---|
| Top few | large | **Signal** — the real structure |
| Long tail | many, each tiny | **Capacity to absorb noise** |

**The mechanism, with numbers.** Suppose a training point carries a label error of $+1.0$. The interpolating model must account for that whole unit of error somewhere.

- **Bad case — few directions available.** The error is loaded onto 3 directions, so each carries about $0.33$. A new test point that happens to have any weight in those directions receives a large, wrong contribution. Disaster.
- **Good case — a long tail.** The error is spread across 10,000 low-variance directions, roughly $0.0001$ each. A test point overlaps each of them a little, and the contributions have random signs, so they **cancel** rather than accumulate. The noise has been buried.

> **Analogy.** A drop of ink. In a shot glass, it ruins the drink. In a swimming pool, it is undetectable. The ink was not removed — **it was diluted below the level at which it matters.** The long tail of the spectrum is the swimming pool.

▸ **Now the condition, stated precisely enough to be useful: you need enough low-variance directions to dilute the noise, *and* they must be low-variance enough that they contribute almost nothing to the actual prediction.** A tail that is too strong is not a sink, it is a second source of signal, and the noise it carries comes right back out. Formally the theory expresses this as a requirement on the **effective rank** of the tail — the same effective-count idea as §1.1.3, appearing for the third time in this chapter.

▸ **And the negative result matters as much as the positive one: overparameterization by itself does not buy you this.** You can build a model with a hundred million parameters whose data covariance has a short, flat spectrum, and it will overfit exactly as classically predicted. **"Big models don't overfit" is false as stated; "models whose data has a heavy-tailed covariance spectrum can interpolate noise harmlessly" is the true version**, and being able to say the second one is the difference between having read about the result and having understood it.

> **Where benign overfitting came from.** **Peter Bartlett, Philip Long, Gábor Lugosi and Alexander Tsigler** gave the characterization for linear regression in *Benign overfitting in linear regression* (PNAS, 2020). Bartlett is a notable figure here because he spent the 1990s proving the classical margin-based generalization bounds for neural networks that the modern picture had to be reconciled with — this is a researcher publishing the theory of why his own earlier framework needed extending. The empirical provocation that set all of this in motion was **Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht and Oriol Vinyals'** *Understanding Deep Learning Requires Rethinking Generalization* (ICLR 2017 best paper). Their experiment was almost aggressively simple: **take CIFAR-10, replace every label with a random one, and train.** The network reached zero training error. It had memorized 50,000 arbitrary labels perfectly — so no capacity-based bound could possibly explain why the *same* architecture, trained on *real* labels, generalizes. The paper offered no new method and no fix; it just demonstrated that the standard explanation could not be right, and the field has been rebuilding ever since.

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

#### The NTK derivation, decoded

Three moves, each simple. The result is startling.

**Move 1 — a Taylor expansion.** $f(x;\theta)\approx f(x;\theta_0) + \nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0)$

This is the first-order Taylor approximation from calculus: *"the value here, plus the slope, times how far you moved."* Reading each piece:

| Piece | Read aloud | Job |
|---|---|---|
| $f(x;\theta_0)$ | "f of x at theta-nought" | The network's output **before any training**, from random weights |
| $\nabla_\theta f(x;\theta_0)$ | "grad-theta f at theta-nought" | A vector with **one entry per parameter**: how much output $f(x)$ moves per unit change in that parameter |
| $\theta - \theta_0$ | "theta minus theta-nought" | How far training has moved the weights from where they started |

▸ **Note the shape of $\nabla_\theta f$: it has $p$ entries, one per parameter.** For a network with $10^8$ parameters, this "feature vector" for a single input is a hundred million numbers long. It is enormous, and it is *fixed at initialization* — that combination is what the whole theory hangs on.

**Move 2 — notice what kind of function this is.** Read the expansion again with $\phi(x) \equiv \nabla_\theta f(x;\theta_0)$ and $w \equiv \theta-\theta_0$:

$$f(x) \approx \text{const} + \phi(x)^\top w$$

**That is linear regression.** Fixed features $\phi(x)$, learned weights $w$. The network's nonlinearity has not vanished — it lives inside $\phi$, which is a wildly nonlinear function of $x$ — but the *learning problem* is now linear, and linear problems are solvable.

> **Analogy.** A pipe organ. Every pipe is a fixed, complicated resonating object, and you cannot change what a pipe sounds like. All you can do is decide how far to open each stop. The music is enormously expressive; the *control problem* is just a set of sliders. **NTK theory says a very wide neural network is an organ: the pipes are cast at initialization and training only adjusts the stops.**

**Move 3 — define the kernel.**

$$\Theta(x,x') = \big\langle\nabla_\theta f(x;\theta_0),\ \nabla_\theta f(x';\theta_0)\big\rangle = \phi(x)^\top\phi(x')$$

A dot product of two gradient vectors — so, by §0.8, **a measure of alignment**. In words: *"do these two inputs want the same parameters changed?"*

- $\Theta(x,x')$ **large** → training on $x'$ nudges the parameters in a way that also moves the prediction at $x$. The two examples reinforce each other.
- $\Theta(x,x') \approx 0$ → they use disjoint parts of the network. Learning one teaches you nothing about the other.

▸ **This is a similarity measure the network defines for itself, from its own architecture, before it has seen a single label.** It is not cosine similarity of inputs, or of activations — it is similarity of *how they would be learned*.

**Now the dynamics equation.**

$$\frac{df(x)}{dt} = -\eta\sum_i\Theta(x,x_i)\,\big(f(x_i)-y_i\big)$$

Read it right to left, as a sentence:

1. $\big(f(x_i)-y_i\big)$ — **"how wrong am I on training example $i$?"** The residual.
2. $\Theta(x,x_i)$ — **"how much does example $i$ influence the prediction at $x$?"** The influence weight.
3. $\sum_i$ — add up the influence from every training example.
4. $-\eta$ — move against the error, at the learning rate.

**In one sentence: "my prediction at $x$ changes by the sum, over all training points, of how wrong I am there times how connected $x$ is to there."**

▸ **The remarkable part is what is absent: $\theta$.** The parameters do not appear on either side. You began with a hundred-million-dimensional nonlinear optimization problem and ended with a linear differential equation in *function space*, whose only ingredient is a fixed $n\times n$ matrix. This is why the result is celebrated: **it is the only case in deep learning where the entire training trajectory has a closed-form answer**, at every time $t$, for every input.

**Reading $e^{-\eta\Theta t}$.** Eigendecompose the $n\times n$ kernel matrix $\Theta$ (it is symmetric and positive semi-definite, so §1.1.2 applies). Along eigendirection $j$ the residual obeys $r_j(t) = r_j(0)\,e^{-\eta\lambda_j t}$ — plain exponential decay, one rate per direction.

| Eigenvalue $\lambda_j$ | Decay rate | Learned |
|---|---|---|
| large | fast | early in training |
| small | slow | late in training, or never |

▸ **So "what does the network learn first?" has a precise answer: whatever lies along the kernel's largest eigenvalues.** If $\lambda_1/\lambda_{50} = 100$, then direction 50 needs a hundred times as many steps to reach the same level of fit. That single inequality is the entire quantitative content of spectral bias, which is the next section's subject.

### The two theorems

**(1) At infinite width, $\Theta$ converges to a deterministic limit** depending only on the architecture, depth, and activation — not on the random initialization.

**(2) At infinite width, $\Theta$ stays constant during training.** Each weight moves by $O(1/\sqrt{\text{width}})$, so the Jacobian doesn't change, so the kernel doesn't change.

▸ **Therefore: an infinitely wide network trained by gradient descent is exactly kernel regression with the NTK.** A closed-form solution for a neural network's entire training trajectory. That is the result, and it is remarkable.

#### The two theorems, decoded

The whole edifice rests on these, and both are statements about what happens when width $\to\infty$.

**Theorem 1 — the randomness averages away.** Initialize a network randomly and $\Theta$ is a random object; it depends on which numbers came out of the random number generator. Theorem 1 says: **as width grows, that randomness vanishes and $\Theta$ converges to a fixed, deterministic function of the architecture alone.**

The mechanism is the law of large numbers. $\Theta(x,x')$ is a sum of contributions from every parameter, and a wide layer has many parameters, so the sum concentrates around its mean — the same $1/\sqrt{n}$ averaging as the standard error in §1.3.1.

▸ **The consequence is worth stating plainly: at infinite width, two networks initialized with different random seeds compute the *same kernel*.** The seed stops mattering. And this is testable — it is why practitioners observe that very wide networks have far less seed-to-seed variance than narrow ones.

**Theorem 2 — the features never move.** Each individual weight travels only $O(1/\sqrt{\text{width}})$ during the whole of training.

| Width | How far each weight moves |
|---|---|
| 100 | $\sim 0.10$ |
| 10,000 | $\sim 0.01$ |
| 1,000,000 | $\sim 0.001$ |

**Why the movement shrinks.** The network's output is a sum over width-many units. To change the output by a fixed amount, you can either move a few weights a lot or move all of them a tiny bit — and gradient descent, taking the shortest path in parameter space, does the second. **Widen the network and the work gets spread more thinly.** The function changes by $\Theta(1)$; each parameter changes by nearly nothing.

And if the parameters barely move, the gradient $\nabla_\theta f(x;\theta_0)$ — which is a *function of the parameters* — barely changes either. So $\phi$ is frozen, so $\Theta$ is frozen. **That is the whole argument.**

> **Analogy.** A stadium crowd doing a wave. The wave travels — a large, coordinated, visible change. **No individual stood up more than once.** Add fifty thousand more people and each person's contribution to the visible effect becomes smaller still, while the wave itself is unchanged. Theorem 2 says an infinitely wide network learns like a stadium: the function moves a lot, and nobody in it moved at all.

▸ **And here is the sting, which the next section develops: if the features never move, the network never learns a representation.** It learns *coefficients* on features that were random from the start. The organ pipes were cast at the foundry and training only worked the stops. Everything deep learning is celebrated for — transfer, pretraining, learned embeddings — requires the pipes themselves to change shape.

#### Examples and non-examples: when NTK theory actually describes your network

**✅ Settings where the linearization is a good description**

| Example | Why the kernel stays roughly frozen |
|---|---|
| A 2-layer MLP of width 100,000, trained a few hundred steps with a small learning rate | Each weight moves $\sim 1/\sqrt{10^5} \approx 0.003$ |
| A network initialized with large weights and trained briefly | Large init means the function starts big; relatively little movement is needed |
| The last-layer-only fine-tune of a frozen backbone | The features are *literally* frozen — this is kernel regression by construction |
| Linear probing of a pretrained model | Same argument; the kernel is whatever the backbone gives you |
| A network's first few steps, at any width | The Taylor expansion is accurate near $\theta_0$ regardless of width |

**❌ Near-misses — settings where the NTK's predictions do not hold**

| Looks applicable | Why it fails | Symptom |
|---|---|---|
| A trained ResNet-50 on ImageNet | Features move enormously; the whole point of the model is that they do | The finite network beats its own NTK by several percent |
| Any model you intend to transfer or fine-tune | Transfer requires learned features, which the NTK has none of | The theory predicts transfer is impossible; it plainly isn't |
| A network trained with a large learning rate | Large steps leave the neighbourhood where the Taylor expansion is valid | Edge-of-stability behaviour that the linear theory does not contain |
| "It is wide, so it is lazy" | The regime is set by **parameterization and init scale**, not width alone | A $\mu$P-parameterized network stays rich at any width |
| A network trained with small initialization | Small init means the weights must move a great deal relative to their size | Rich regime — features rebuilt, theory intractable |

▸ **The boundary:** you are in the NTK regime when **the features $\nabla_\theta f(x;\theta_0)$ do not meaningfully change during training.** That is a measurable property, not an architectural one: compute the kernel at initialization and again at the end and look at how much it moved. The number you get is, in a real sense, how much representation learning happened.

> **Common misconception.** *"The NTK shows neural networks are 'just' kernel machines."* It shows that a network *in a particular limit* is exactly a kernel machine — and that limit is one that practitioners deliberately steer away from, because networks in it cannot learn features. Finite networks beat their own NTK by several percent on ImageNet, and that gap is not a rounding error; **it is a measurement of the thing the theory is missing.** The misconception is tempting because "X is just Y" is a satisfying reduction and the mathematics really is exact. The correct sentence is narrower and more interesting: *the NTK is the correct theory of the regime in which deep learning does not work very well.*

> **Common misconception.** *"Wider is always lazier, so if I want feature learning I should keep my network narrow."* Width tips the balance only under **standard parameterization**. Under $\mu$P the network stays in the feature-learning regime at any width — which is precisely why $\mu$P lets you tune hyperparameters on a small model and apply them to a large one. The lesson is not "avoid width"; it is **"width and regime are separable, and $\mu$P is how you separate them."**

> **Where the NTK came from.** **Arthur Jacot, Franck Gabriel and Clément Hongler** at EPFL published *Neural Tangent Kernel: Convergence and Generalization in Neural Networks* at NeurIPS 2018; Jacot was a doctoral student at the time. The idea has a clear ancestor. In his 1994–1996 doctoral work at Toronto, supervised by **Geoffrey Hinton**, **Radford Neal** proved that a single-hidden-layer neural network with random weights converges, as the layer becomes infinitely wide, to a **Gaussian process** — a fully-specified, closed-form probabilistic object. That result (extended to deep networks in 2018 by Jaehoon Lee and colleagues, and known as the **NNGP**, neural network Gaussian process) describes the network *at initialization*. The NTK is the corresponding statement about the network *while training*. **Two decades separate "an infinitely wide network is a Gaussian process at birth" from "an infinitely wide network is a kernel machine for life,"** and both came out of asking what happens in a limit that no one intended to build.

### What it explains

- **Why gradient descent finds a global minimum:** the loss becomes convex in the linearized model, and a positive-definite $\Theta$ guarantees convergence.
- **The convergence rate:** the residual along eigendirection $j$ decays at rate $\eta\lambda_j$. ▸ **Since the NTK's large eigenvalues correspond to low-frequency, smooth functions, networks fit smooth structure first and high-frequency detail last** — the *spectral bias*. This is a , quantitative explanation of a real phenomenon, and it also explains why random labels take much longer to fit than real ones.
- **Wide networks are easier to optimize** — the linearization is more accurate.

#### Spectral bias, decoded — the NTK's best prediction

**The claim: networks learn smooth things first and detailed things last.** This is not a vague impression; the NTK turns it into an equation.

**Why "large eigenvalue = smooth function."** The NTK's eigenfunctions, ordered by eigenvalue, run from slowly-varying to rapidly-oscillating — the same ordering as the Laplacian eigenvectors in Chapter 29, and for the same underlying reason: **smooth functions are the ones that many nearby inputs agree on, so they accumulate large kernel mass.** Combine that with $r_j(t) = r_j(0)e^{-\eta\lambda_j t}$ and you get a schedule.

> **Analogy.** Developing a photograph, or a low-resolution image sharpening into focus. The broad shapes appear first, then edges, then texture, then grain. **A neural network trains in the same order, and the NTK tells you the exact rate for each level of detail.** The reason it looks like a progressive JPEG is that, mathematically, it is one: a decomposition into frequency components fitted at different speeds.

**Why this explains the random-label observation.** A real dataset is *smooth* in the relevant sense — similar images have similar labels — so it lies mostly along the kernel's large-eigenvalue directions and is fitted fast. **Random labels are maximally jagged**: two nearly-identical images have unrelated targets, so the target function has essentially no mass on the smooth directions and all of it on the slow ones.

▸ **So the prediction is quantitative, not hand-wavy: a network fits random labels using the directions with the smallest $\lambda_j$, which decay slowest, so it takes far longer.** Zhang et al. measured exactly this — random labels are learnable but take several times as many epochs — and the NTK predicts it from the spectrum alone. **This is the single best piece of evidence that the theory captures something real about finite networks**, because the prediction was quantitative and made about a phenomenon nobody was trying to explain.

**Two more consequences worth carrying.**

- **A blessing.** Spectral bias is a *free regularizer*. A model that struggles to represent jagged functions will struggle to memorize noise, which is most of why early stopping works at all.
- **A curse.** "Simplest predictive feature first" means the model latches onto whatever is easiest, and background texture is often easier than the object. Chapter 31 §31.2 develops this as **simplicity bias**, and it is the mechanism behind a large fraction of spurious-correlation failures.

▸ **Why gradient descent finds a global minimum, in one line:** once the model is linear in $w$ with a positive-definite kernel, the squared loss is a **convex** bowl in $w$ — one basin, no local minima, no saddles. The famous difficulty of non-convex optimization simply evaporates in this limit. That is a satisfying explanation, and it is also a warning: **the limit removed the very thing that makes the real problem hard.**

#### Examples and non-examples: spectral bias in the wild

**✅ Observations that spectral bias  explains**

| Observation | The mechanism |
|---|---|
| Random labels take several times as many epochs to fit than real ones | Random targets live on the smallest-$\lambda$ directions, which decay slowest |
| Generated images look blurry before they look sharp | Low frequencies have large kernel eigenvalues and are fitted first |
| A network fitting a $\sin(50x)$ target needs far more steps than $\sin(x)$ | The high-frequency target sits on slow directions |
| Early stopping works as a regularizer | Stopping early truncates the slow, high-frequency directions — which is where noise lives |
| A model learns "grass ⇒ cow" before it learns cow anatomy | Background texture is a *smoother* function of the pixels than shape |

**❌ Near-misses — things spectral bias is often blamed for**

| Blamed on it | Why it isn't | What it actually is |
|---|---|---|
| Underfitting a small dataset | Spectral bias is about *order*, not about a ceiling | Insufficient capacity, or too few steps |
| The model ignoring a rare class | Class frequency, not function frequency | Class imbalance |
| High-frequency adversarial vulnerability | Networks are *too* sensitive to high frequencies at test time, not too slow to learn them | A different phenomenon — Ch. 33's subject |
| Blurry outputs from an autoencoder trained with MSE | The pixel-wise squared error itself rewards the conditional mean | A loss-function property (Ch. 19), which the spectral story then reinforces |
| "The model prefers simple explanations" as a general law | Spectral bias is a specific statement about the NTK's eigenspectrum, in a specific limit | Simplicity bias, a broader and less well-characterized claim (Ch. 31 §31.2) |

▸ **The boundary:** spectral bias is the quantitative claim that **the residual along kernel eigendirection $j$ decays at rate $\eta\lambda_j$** — so what is learned first is decided by the kernel's spectrum, and nothing else. Anything that is not an ordering-in-time claim about function frequency is a different phenomenon that happens to share the word "simple."

> **Common misconception.** *"Spectral bias means the network can't represent high-frequency functions."* It can represent them fine; it *reaches* them slowly. Given enough steps a network fits random labels on CIFAR-10 exactly, and random labels are the most high-frequency target there is. The distinction is between **expressivity** (what is representable) and **optimization order** (what arrives when), and conflating them is one of the most common errors in reading deep learning theory. The misconception is tempting because "learns it last" and "can't learn it" produce identical observations on any training run you actually stop.

> **Where spectral bias came from.** It was identified independently by two groups at almost the same moment. **Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio and Aaron Courville** published *On the Spectral Bias of Neural Networks* in 2019, coining the name. Working separately, **Zhi-Qin John Xu** and colleagues described the same effect as the **Frequency Principle** ("F-Principle") — networks fit low frequencies before high ones — in papers from the same period. Two names for one phenomenon, arrived at from different directions, in a book that has already recorded the SVD being discovered twice and backpropagation four times.

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

#### Lazy versus rich, decoded — and why $\mu$P exists

**The two names are literal descriptions.**

- **Lazy** — the network barely bothers to change its internals. It solves the problem by re-weighting features it already had. Minimum effort, and the theory is exactly solvable.
- **Rich** — the network rebuilds its internal representation. The features at the end bear no resemblance to the features at the start. Far more powerful, and mathematically intractable.

**Now the row that surprises people: the regime is not decided by width.** You might reasonably expect "wide network ⇒ lazy," but the honest statement is that it depends on **how the network is parameterized and how large the initialization is**, and width only tips the balance under one particular parameterization.

**The intuition for initialization scale.** Ask: *how much does the function have to change, relative to how large the weights already are?*

| Initialization | The function starts | To fit the data, weights must | Regime |
|---|---|---|---|
| **Large** | already producing big outputs | move a little, relatively speaking | **Lazy** — features roughly preserved |
| **Small** | producing near-zero outputs | move a lot, relatively speaking | **Rich** — features rebuilt |

> **Analogy.** Renovating a house. If it is already a mansion, making it liveable means moving furniture — the structure is fine. If it is a bare plot, you have to build walls. **Same final house, entirely different amount of structural change**, and the difference was set by what you started with.

▸ **So the initialization scale is a regularization hyperparameter**, which is not how most people think of it. It is usually treated as a numerical hygiene setting — "keep the activations from exploding" — but it also silently selects whether your network is capable of learning features at all.

**Now the $\mu$P connection, which is the payoff.**

Under **standard parameterization**, the scaling of the weights and learning rates is such that as width grows the network drifts toward the lazy regime. Concretely: the optimal learning rate shifts with width, and a network wide enough starts behaving like a kernel machine. **This is a problem, because it means a hyperparameter tuned at small width is wrong at large width — and the error is not random, it is a systematic drift toward a worse regime.**

**$\mu$P (maximal update parameterization)** chooses the per-layer scalings so that, in the infinite-width limit, **every layer's features still move by $\Theta(1)$.** The network stays rich at any width.

▸ **And that is *why* $\mu$P transfers hyperparameters (Ch. 15 §15.3).** The usual explanation — "$\mu$P makes the optimal learning rate width-independent" — is true but is the symptom. The cause is that $\mu$P **preserves the regime**: a 100-million-parameter model and a 100-billion-parameter model under $\mu$P are doing the same *kind* of learning, so a setting tuned on one applies to the other. Under standard parameterization they are not, and the transfer fails for a reason no amount of careful tuning at small scale can fix.

**Connecting these two chapters correctly is worth practising as a sentence:** *"The NTK describes the lazy regime; real networks need the rich regime; standard parameterization drifts lazy as you widen; $\mu$P is the fix that keeps you rich, which is why it lets you tune small and train big."*

**The honest summary, expanded.** Why study a theory of a limit we avoid?

- It is **the only fully-solved case.** Everything else in deep learning theory is bounds and conjecture; here we have the exact trajectory.
- It **makes correct quantitative predictions** anyway — spectral bias, wide networks optimizing more easily, low seed variance at width.
- It **defines the baseline.** "Feature learning" is a slippery notion until you have a precise model of a network that does none, and can measure the gap. **The several-percent that finite networks beat their NTK by is the numerical value of feature learning**, and having a number for it is worth a great deal.

> **Analogy.** The frictionless plane in mechanics. No real surface is frictionless, and the model is still the right first thing to teach — because it is exactly solvable, because it predicts a great deal correctly, and because *friction is defined as the discrepancy*. The NTK is deep learning's frictionless plane, and feature learning is its friction.

> **Where "lazy training" got its name.** The term was introduced by **Lénaïc Chizat, Edouard Oyallon and Francis Bach** in *On Lazy Training in Differentiable Programming* (NeurIPS 2019). Their contribution was to show that laziness is **not a property of neural networks at all** — it is a consequence of how a differentiable model is *scaled*. Take almost any differentiable parametric model, multiply its output by a large constant, and gradient descent will barely move the parameters while the function changes a great deal: you have made it lazy, whatever it is. That reframing was clarifying, because it separated "the network is very wide" from "the network is in the kernel regime," which the early NTK literature had tended to run together. The complementary construction — a parameterization that *stays* in the feature-learning regime as width grows — came from **Greg Yang and Edward Hu's** Tensor Programs work at Microsoft Research, which produced $\mu$P and the zero-shot hyperparameter transfer result used in Chapter 15 §15.3.

#### Examples and non-examples: lazy versus rich, diagnosed

**✅ Evidence a run is in the rich, feature-learning regime**

| Evidence | Why it indicates rich |
|---|---|
| The final-layer features are useful for a *different* task | Transfer requires the features to have moved |
| The kernel measured at the end differs substantially from the kernel at init | That is the definition |
| Interpretable circuits form during training (induction heads, Fourier features) | A frozen kernel cannot grow new structure |
| The finite network beats its own analytically-computed NTK | The gap is the feature learning |
| Different seeds produce recognizably different learned features | Rich dynamics amplify small differences |

**❌ Near-misses — signs that look like feature learning and aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| The loss went down a lot | The NTK also drives loss to zero — that is Theorem 2's whole content | Fitting, which is available in both regimes |
| The activations changed a great deal | Activations move because the coefficients moved; the *features* $\nabla_\theta f$ may not have | A downstream consequence, not a diagnostic |
| The weights changed a great deal in absolute terms | What matters is the change **relative to the initialization scale** | Measure $\lVert\theta-\theta_0\rVert / \lVert\theta_0\rVert$, not the numerator alone |
| The model is very deep | Depth does not decide the regime any more than width does | An architecture choice; the regime is a parameterization choice |
| Linear probing of the frozen backbone works well | You have measured how good the *fixed* features are — the kernel view exactly | Kernel regression on a very good kernel |

▸ **The boundary:** rich versus lazy is decided by whether $\nabla_\theta f(x;\theta_0)$ — the feature map — is still the feature map at the end of training. **The one honest measurement is the kernel at init versus the kernel at convergence**; everything else is a proxy.

---

## 30.3 Grokking

### The phenomenon

Train a small transformer on modular arithmetic (e.g. $a\circ b = (a+b)\bmod 97$) with a fraction of the pairs held out:

- **Step ~$10^3$:** training accuracy 100%. Test accuracy ~random. The model has memorized.
- **Steps $10^3$ to $10^5$:** nothing visibly happens. Training loss is at zero; test accuracy stays flat.
- **Step ~$10^5$:** test accuracy jumps to 100% over a short window.

▸ **Generalization occurs long after the training loss has been zero, with no visible signal in between.** This is a direct refutation of the intuition that training loss tracks learning.

#### The grokking setup, decoded

**What "modular arithmetic" actually means.** $(a+b) \bmod 97$ says: add the two numbers, divide by 97, and keep only the remainder.

| $a$ | $b$ | $a+b$ | $(a+b)\bmod 97$ |
|---|---|---|---|
| 3 | 5 | 8 | **8** |
| 90 | 20 | 110 | **13** |
| 96 | 1 | 97 | **0** |

> **Analogy.** A clock. Four hours after ten o'clock is two o'clock, not fourteen. Clock arithmetic is modular arithmetic with modulus 12 — **the number line has been bent into a circle**, and once you have seen that, the finding in strand 3 below stops being mysterious and becomes almost inevitable.

**Why 97?** It is prime, which makes the arithmetic well behaved (every nonzero element has a multiplicative inverse) and rules out the shortcut structures a composite modulus would offer. It is also small enough that the full problem is $97 \times 97 = 9{,}409$ pairs — small enough to enumerate, large enough that memorizing is real work.

**Reading the timeline properly.** The steps are on a log scale, and that matters:

| Phase | Steps | Training accuracy | Test accuracy |
|---|---|---|---|
| Memorization | $\sim 10^3$ | **100%** | ~1% (chance is $1/97$) |
| The plateau | $10^3 \to 10^5$ | 100% | ~1%, flat |
| Grokking | $\sim 10^5$ | 100% | **100%**, over a short window |

▸ **The plateau is 100 times longer than everything that came before it.** Put that in practical terms: at the moment every dashboard says the run is finished — training loss zero, test accuracy flat for tens of thousands of steps — you would have to keep going for **a hundred times as long as you have already trained** before anything happened. No sane engineer does this. **That, and not any subtlety in the mechanism, is why the phenomenon went unnoticed for so long.**

▸ **And note precisely what is being refuted.** Not "training loss is a bad metric" — it is a fine metric for its job. The refuted belief is that *training loss going flat means learning has stopped.* Something is still changing, continuously, for two orders of magnitude longer. **The loss is at zero and the model is still moving**, which sounds impossible until you notice that "zero loss" is a whole surface in parameter space, not a point.

#### Examples and non-examples: what counts as grokking

The word has drifted since 2022 and is now applied to almost any late improvement. The original phenomenon is narrower and much stranger.

**✅  grokking**

| Case | The three required ingredients |
|---|---|
| A 1-layer transformer on $(a+b)\bmod 97$, 30–50% of pairs held out, weight decay on | Training accuracy 100% **long before** any test improvement; a plateau spanning orders of magnitude; an abrupt jump to near-perfect test accuracy |
| The same setup on modular multiplication, or on composition in a permutation group | Same signature; the learned circuit differs but the shape of the curve does not |
| Sparse parity: predict the XOR of $k$ hidden bits out of $d$ | Memorization first, algorithm much later, transition sharp in accuracy and smooth in a progress measure |

**❌ Near-misses — late improvements that are not grokking**

| Looks like it | Which ingredient is missing | What it actually is |
|---|---|---|
| Test accuracy improves slowly over a long run | No plateau, no abrupt transition | Ordinary training. Learning is meant to take time |
| A jump in test accuracy when the learning-rate schedule drops | The cause is an external event, not an internal drift | A schedule artifact — the same jump appears in the training loss |
| Training loss is $0.02$ and falling, and test accuracy rises later | Training loss is not **zero**, so the data is still pushing | Normal fitting. Grokking requires the data's gradient to have essentially vanished |
| The model improves after you add data or unfreeze layers | You changed the problem | An intervention, not a drift |
| "Emergent abilities" appearing at a scale threshold | Different axis entirely — model size, not training time, and no plateau in a single run | The emergence debate of Ch. 15 §15.4, which grokking illuminates but is not |
| A run with **no weight decay** that eventually generalizes | Grokking largely disappears without weight decay | Something else; find out what |

▸ **The boundary:** grokking requires all three of — **(1)** training loss at essentially zero, so the data has stopped pushing; **(2)** a plateau in test performance lasting orders of magnitude longer than the memorization phase; **(3)** an abrupt transition driven by an ongoing force, in practice weight decay. Drop any one and you have a different phenomenon that happens to be slow.

> **Common misconception.** *"Grokking shows that if you just train long enough, models eventually understand."* It shows that **when a lower-norm generalizing solution exists and a regularizer is slowly pulling toward it**, the model can reach it long after the loss has flattened. Every clause is load-bearing. Turn off weight decay and the model sits in the memorizing solution indefinitely; drop below the critical data fraction and there is no pressure toward the algorithm at all; and on a task with no clean underlying rule there is no algorithmic solution to drift to. The misconception is tempting because "keep training" is actionable advice and the alternative — "check whether a better solution exists and whether anything is pushing you toward it" — is not a knob you can turn.

> **Common misconception.** *"Zero training loss means nothing can change any more."* Zero loss is a **surface** in parameter space — the set of all parameter settings that fit the data exactly — not a single point. The model can slide freely along that surface without the loss registering anything, and weight decay is a constant force doing exactly that. This is the single most useful mental correction in the section: **flat loss curve, moving model.** The belief is tempting because in the underparameterized world classical statistics studied, zero training loss  was a unique point or did not exist at all.

### The mechanism

The best-supported account has several converging strands:

**1. Two competing solutions.** A **memorizing** circuit (a lookup table) and a **generalizing** circuit (an actual algorithm). Memorization is found first because it is reachable greedily. The generalizing solution is harder to find but, crucially, **has smaller weight norm** — an algorithm is more parameter-efficient than a table.

**2. Weight decay drives the transition.** ▸ **Grokking largely disappears without weight decay** (or with weight norm otherwise controlled). Once training loss is ~0, the cross-entropy gradient is nearly zero and **weight decay becomes the dominant force**, slowly moving the model along the zero-loss manifold toward the minimum-norm solution — which is the generalizing one. The delay is the time this drift takes.

**3. The circuit is identifiable.** For modular addition, the network learns a **discrete Fourier transform**: embeddings become $\big(\cos(\omega_k a),\sin(\omega_k a)\big)$ for a handful of frequencies $\omega_k$, attention and MLP layers compute products realizing the trigonometric identity
$$\cos(\omega(a+b)) = \cos\omega a\cos\omega b - \sin\omega a\sin\omega b$$
and the unembedding reads off the argmax over $c$ of $\cos(\omega(a+b-c))$. ▸ **The network implements modular addition by rotating on a circle.** Nanda et al. reverse-engineered this completely — one of the cleanest full mechanistic explanations of a trained network in existence (Ch. 32).

**4. Progress measures.** The generalizing circuit forms *gradually* and continuously; only the *test accuracy metric* is discontinuous. Defining a continuous progress measure — restricted loss (ablate all but the key Fourier frequencies) and excluded loss (ablate only them) — shows smooth formation throughout the plateau.

▸ **This is exactly the emergence critique from Chapter 15 §15.4, at the level of a single small model where we can verify it completely.** The underlying change is smooth; the metric is not. Grokking is the strongest available evidence for that view, because here we can see the mechanism directly rather than inferring it.

#### The four strands, decoded

**Strand 1 — two solutions, and why one is bigger.**

Both circuits achieve zero training loss. The difference is how much *machinery* each requires.

| Solution | What it is | Roughly what it must store |
|---|---|---|
| **Memorization** | A lookup table | one entry per training pair — thousands of independent numbers |
| **Generalization** | An algorithm | a handful of frequencies and the weights to combine them |

> **Analogy.** Two students preparing for a multiplication test. One memorizes the times table. The other learns multiplication. **Both score 100% on the practice sheet.** Only one of them can handle $23\times47$ if it wasn't on the sheet. And crucially — the memorizer's head is *fuller*. The algorithm is the smaller object.

▸ **"Smaller weight norm" is the precise version of that.** A lookup table needs many large, independent weights, each carving out one specific input. An algorithm reuses the same small structure everywhere. **So the generalizing solution sits lower on the $\|\theta\|^2$ landscape than the memorizing one** — and that single fact is what makes the transition happen at all.

**Strand 2 — why weight decay is the engine.**

Follow the forces once training loss hits zero:

1. Cross-entropy loss is ~0, so $\nabla_\theta \mathcal{L}_{\text{CE}} \approx 0$. **The data has stopped pushing.**
2. Weight decay contributes a gradient of $\lambda\theta$ — a constant pull toward the origin — and it **never stops**, because it does not care about the loss.
3. So the model slides along the **zero-loss manifold**: the set of all parameter settings that fit the training data perfectly. It cannot leave (the data would object) but it can move freely *within* it.
4. It drifts downhill in norm until it reaches the smallest-norm point on that surface — **the generalizing circuit.**

> **Analogy.** A marble in a long, perfectly level trough. Gravity along the trough is zero; the marble will sit wherever you put it. Now tilt the whole trough by one degree. Nothing dramatic happens — but over a long enough time the marble ends up at the far end. **Weight decay is the one-degree tilt, the plateau is the marble rolling, and grokking is it arriving.**

▸ **The prediction this makes is sharp and it holds: turn weight decay off and grokking largely disappears.** No tilt, no drift, no transition — the model sits in the memorizing solution forever. **A phenomenon that looks like a mysterious emergent insight is caused by a regularization term you can switch off in one line.** That is about as satisfying as explanations in this field get.

**Strand 3 — the circuit, and why a circle.**

The network discovers that modular addition **is rotation**. Follow the construction:

1. **Embeddings become points on a circle.** Number $a$ is mapped to $\big(\cos(\omega_k a),\ \sin(\omega_k a)\big)$ for a few frequencies $\omega_k$. That is the standard way to place an integer on a circle — the same construction as a transformer's sinusoidal position encoding (Ch. 12), arrived at independently by gradient descent.
2. **Attention and the MLP compute products of those coordinates**, which by the trigonometric identity
$$\cos(\omega(a+b)) = \cos\omega a\cos\omega b - \sin\omega a\sin\omega b$$
gives you the *sum's* angle from the two inputs' angles. **Adding numbers has become adding angles.**
3. **The unembedding scores each candidate answer $c$** by $\cos(\omega(a+b-c))$, which is maximal exactly when $a+b-c$ is a multiple of the full turn — that is, when $c \equiv a+b$.

▸ **And the reason a circle is the right answer is the clock analogy from above, made literal: modular arithmetic *is* arithmetic on a circle, so a machine that represents numbers as angles gets the wraparound for free.** The network was not taught this. It was given input-output pairs and it found the geometry.

**Why this matters beyond the toy problem.** Nanda and colleagues did not infer this from behaviour; they **read it out of the weights** — identified the frequencies, ablated them individually, and confirmed the causal role of each component. That is a complete mechanistic account of a trained network, of a kind that remains rare (Ch. 32).

**Strand 4 — progress measures, and the emergence lesson.**

The generalizing circuit is being built *the entire time*, continuously and gradually. What is discontinuous is **test accuracy**, which is a threshold: it reports nothing at all until the circuit is good enough to win the argmax, then reports everything at once.

Two continuous measures make the smooth build visible:

| Measure | How it is computed | What it shows |
|---|---|---|
| **Restricted loss** | Delete everything *except* the key Fourier frequencies, then measure loss | Falls smoothly throughout the plateau — the circuit is forming |
| **Excluded loss** | Delete *only* those frequencies, then measure loss | Rises smoothly — the model increasingly depends on them |

> **Analogy.** Water heating on a stove. The temperature climbs steadily from 20 °C to 100 °C — smooth, continuous, measurable at every moment. But if your only instrument is "is it boiling?", you observe nothing, nothing, nothing, then suddenly *boiling*. **The discontinuity is in the instrument, not the kettle.**

▸ **This is why grokking is the strongest evidence for the emergence critique of Chapter 15.** For large language models, the claim that "emergent abilities" are metric artifacts is an inference — the models are too big and too opaque to check directly. Here, on a tiny transformer doing arithmetic mod 97, **it is not an inference. You can point at the circuit, watch it form smoothly, and identify exactly which threshold made it look sudden.**

### Where it happens

Algorithmic tasks with a clean underlying rule and enough (but not too much) data. ▸ **Below a critical data fraction, the model never groks** — memorization is a valid solution and no pressure moves it off. Above a larger fraction, generalization is immediate — memorization is no longer easier. **Grokking lives in the window between**, and that window is why it took so long to notice.

**Does it happen at scale?** Probably in a distributed form: circuits form at different times during training, and phase transitions in specific capabilities (induction heads, Ch. 13 §13.3) look structurally like grokking. Delayed generalization on subtasks has been observed in language models.

#### The data window, decoded

Grokking needs a Goldilocks amount of data, and the reason is a competition between two costs.

| Data fraction | Which solution is easier? | What you observe |
|---|---|---|
| **Too little** | Memorization, permanently — there is not enough signal to pin down the algorithm, and memorizing a small table is cheap | Never groks. Trains forever at 100% train, 1% test |
| **The window** | Memorization first (greedy), algorithm eventually (lower norm) | **Grokking** — the delayed transition |
| **Plenty** | The algorithm, immediately — the table has become expensive and the pattern is obvious | Generalizes right away. Nothing to see |

▸ **Read the middle row as a race.** Memorization wins the sprint because it is reachable greedily: every training example independently pushes its own entry into place. The algorithm wins the marathon because it costs less to hold. **Grokking is what a delayed overtake looks like when your only instrument is the finish line.**

▸ **And this is why it took so long to find.** Below the window there is nothing to see; above it there is nothing to see; and inside it you have to train a hundred times longer than any reasonable person would. **The phenomenon was hiding in a narrow band of a hyperparameter that nobody had a reason to sweep.**

**Does it happen at scale — how to think about the answer.** At scale, thousands of circuits are being learned at once and each has its own window and its own transition time. Averaged over all of them, the aggregate loss curve is smooth: **the individual jumps are real, and they are washed out by summation.** That is entirely consistent with individual capabilities appearing abruptly, which is what induction-head formation looks like (Ch. 13 §13.3) — a sharp, identifiable phase change in a specific mechanism, visible in a narrow probe and invisible in the loss.

> **Where grokking came from.** The phenomenon was reported by **Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin and Vedant Misra** at OpenAI in *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (2022, first circulated at an ICLR workshop in 2021). The name comes from **Robert A. Heinlein's** 1961 science-fiction novel *Stranger in a Strange Land*, in which "grok" is a Martian verb meaning to understand something so completely that you merge with it — the literal Martian sense is "to drink." The word passed into hacker slang in the 1970s and 1980s, which is how it reached a machine learning paper. It is an unusually apt name: the model does not gradually improve at the task, it abruptly *gets* it. The full mechanistic explanation came from **Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith and Jacob Steinhardt** in *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR 2023), which reverse-engineered the Fourier circuit completely and introduced the restricted/excluded loss measures. **It is one of very few cases in deep learning where a surprising empirical phenomenon was fully explained down to the level of individual weights within two years of being reported.**

---

## 30.4 The synthesis

▸ **What these three phenomena have in common:** all three are cases where *capacity* is the wrong variable and *the optimization trajectory* is the right one.

- Double descent: in the interpolating regime, which of the infinitely many zero-error solutions you get is what determines test error.
- NTK: the training dynamics are what's solvable; the parameters are not the interesting object.
- Grokking: two solutions both achieve zero training error, and the slow drift between them is invisible to the loss.

**Classical theory bounds what the model class *could* do. Modern deep learning is governed by what the optimizer *does*.** That is the through-line into Chapter 31, which studies the optimizer's preferences directly.

#### The synthesis, decoded

**The old question:** *"Is my model big enough, and is it too big?"* Capacity was the dial, and the whole job was setting it correctly.

**The new question:** *"Among all the settings that fit my data, which one does my training procedure land on?"* Capacity determines the *size of the menu*. The optimizer determines *what you eat*.

Line up the three phenomena and they are one sentence three times:

| Phenomenon | The set of zero-error solutions | What decides which you get |
|---|---|---|
| **Double descent** | Empty ($p<n$), a single point ($p=n$), infinite ($p>n$) | Minimum-norm selection by the pseudo-inverse |
| **NTK** | Solvable in closed form once linearized | The kernel's eigenspectrum sets what is learned and when |
| **Grokking** | Contains both a lookup table and an algorithm | Weight decay, drifting slowly toward the smaller-norm one |

> **Analogy.** Hiring. The classical question was "how many candidates should we interview?" — set the pool size correctly and good outcomes follow. The modern realization is that with a large enough pool, **thousands of candidates are perfectly qualified on paper, and everything now depends on how you choose among them.** Widening the pool further does not help; changing the selection criterion does. **Deep learning stopped being about the size of the pool around the time models started fitting their training sets exactly.**

▸ **The one-sentence version to be able to say out loud:** *"Once a model can fit the training data perfectly, capacity has stopped being the interesting variable — the training procedure is now the thing choosing your hypothesis, and studying it is what Chapter 31 does."*

**And the practical residue, which is not nothing.** This is not purely a theoretical reorientation:

- If test error rises as you scale, suspect **under-regularization** before you suspect a fundamental limit.
- If a curve has flattened, "train ten times longer" is occasionally the correct move, and the classical instinct to early-stop can be exactly wrong.
- If you are choosing an optimizer, you are choosing a **prior over solutions**, not just a convergence speed — which is the subject of Chapter 31 §31.2.

#### Examples and non-examples: capacity questions versus trajectory questions

The whole chapter is one reclassification. Here is the sorting exercise, because getting a question into the right bucket determines which knob you reach for.

**✅  capacity questions** — answered by counting what the model *could* represent

| Question | Why it is about capacity |
|---|---|
| "Can a 2-layer MLP represent XOR?" | An expressivity fact, decided by the architecture alone |
| "Can a message-passing GNN count triangles?" (Ch. 29) | An information bound — the architecture never receives what it would need |
| "Is my model large enough to fit 50,000 arbitrary labels?" | A statement about the size of the hypothesis class |
| "How many bits can this network store?" | A counting argument |

**✅  trajectory questions** — answered by asking what the optimizer *does*

| Question | Why it is about the trajectory |
|---|---|
| "Why does my model generalize when it can fit random labels?" | Both solutions are available; something chose the good one |
| "Why does test error peak exactly at $p=n$?" | The minimum-norm *selection rule* blows up there |
| "What does my network learn first?" | The kernel's eigenspectrum sets the order |
| "Why did generalization appear 100,000 steps after the loss hit zero?" | A slow drift along the zero-loss surface |
| "Why does $\mu$P let me tune small and train big?" | It preserves the *regime* of the dynamics |

**❌ Near-misses — questions that get answered with the wrong tool**

| Question asked | Wrong framing | Right framing |
|---|---|---|
| "My model overfits — should I shrink it?" | Capacity: reduce $p$ | Trajectory: you may be at the interpolation threshold. Try **more** capacity, or more regularization |
| "Validation error went up — should I stop?" | Capacity: the model has run out of road | Trajectory: epoch-wise double descent means stopping here may be exactly wrong |
| "How many parameters do I need?" | Capacity: count them | Trajectory: effective model complexity is architecture **plus optimizer plus schedule**, measured, not counted |
| "It fit random labels, so the bound is vacuous — theory is useless" | Capacity: no bound can work | Trajectory: bounds on the *algorithm's output* rather than the *hypothesis class* are exactly what Ch. 31 pursues |

▸ **The boundary:** ask *"would the answer change if I kept the model identical and changed only the optimizer, the schedule, or the regularizer?"* If yes, it is a trajectory question, and counting parameters will not answer it. **Once a model can fit its training data exactly, almost every interesting question has moved into the second column.**

> **Common misconception.** *"Implicit regularization means SGD secretly adds a penalty term to the loss."* In some restricted settings that picture is provably right, and in general it is not — there is no single scalar penalty whose explicit minimization reproduces what SGD does. "Implicit bias" is the more honest phrase: the algorithm **prefers** certain solutions among the many that fit, and characterizing that preference is an open research programme rather than a solved reduction. The misconception is tempting because the min-norm case works out so cleanly that it invites generalization — and Chapter 31 is largely the story of how far that generalization can be pushed.

---

## Did you know?

- **"Grok" is a Martian verb from a 1961 science-fiction novel.** Robert A. Heinlein coined it in *Stranger in a Strange Land*, where it literally means "to drink" and figuratively means to understand something so thoroughly that you merge with it. It entered hacker slang in the 1970s and reached a machine learning paper fifty years after the book.

- **Double descent was published, correct, and then forgotten for about twenty-five years.** Physicists working on the statistical mechanics of learning derived learning curves with the interpolation peak in the late 1980s and 1990s. The result went unused because nobody had a reason to train past the peak.

- **The standard citation for the bias–variance dilemma is a paper about neural networks.** Geman, Bienenstock and Doursat's 1992 *Neural Networks and the Bias/Variance Dilemma* argued carefully that high-capacity networks would be defeated by variance and that strong built-in priors would be necessary. It became the canonical statement of the principle modern networks violate.

- **The mathematics behind the interpolation peak was invented for atomic nuclei.** Random matrix theory began with Eugene Wigner in the 1950s, who proposed modelling the energy levels of heavy nuclei by replacing an intractable Hamiltonian with a random matrix. Marchenko and Pastur's 1967 law — the one that puts the smallest singular value at zero when $p=n$ — is a descendant.

- **The NTK's ancestor is a 1990s thesis supervised by Geoffrey Hinton.** Radford Neal showed that an infinitely wide neural network with random weights is a Gaussian process. That describes the network at birth; the NTK, twenty-odd years later, describes it for life.

- **$\Theta$ means two unrelated things within one page of this chapter.** $\Theta(x,x')$ is the neural tangent kernel; $\Theta(1)$ is big-Theta asymptotic notation. The collision is universal in the literature and there is no convention that resolves it.

- **Networks fit randomly-shuffled labels perfectly.** Zhang et al. took CIFAR-10, replaced every label with a random one, and reached zero training error. The paper offered no method and no fix — it simply demonstrated that no capacity-based bound could explain generalization, and the field has been rebuilding since.

- **More data can make your model worse.** Sample-wise double descent is real and reproducible: adding examples can walk a fixed-size model *toward* the interpolation threshold, and test error rises until you pass through it. It is the most counterintuitive result in this chapter and it is not controversial.

- **Spectral bias was discovered independently twice, in the same period.** One group named it "spectral bias"; another described the identical effect as the "Frequency Principle." This book has now recorded the SVD found twice, backpropagation four times, mode connectivity twice, and this.

- **You can compute the infinite-width limit exactly, on a laptop.** Because the NTK is a deterministic function of the architecture, there are libraries that evaluate an infinitely wide network's predictions in closed form. **The infinite network is often cheaper to evaluate than the finite one** — you never instantiate any weights, you just build an $n\times n$ kernel matrix.

- **Grokking essentially disappears if you switch off weight decay.** A phenomenon that looks like a mysterious flash of insight is driven by a regularization term. Once the training loss is zero the data has stopped pushing, and weight decay — which never stops pushing — slowly slides the model to the smaller-norm solution.

- **The network solves modular arithmetic by putting numbers on a clock face.** It learns to embed each integer as $(\cos\omega a, \sin\omega a)$ and to add by composing angles, using the cosine addition formula. Nobody told it to; it reinvented the trigonometric identity from input–output pairs alone.

- **The modular-arithmetic modulus 97 is prime for a reason.** A prime modulus makes every nonzero element invertible and removes the sub-structure a composite modulus would offer, so the only clean solution available is the actual algorithm.

---

## Check for Understanding

**Test error peaks exactly at the interpolation threshold because the design matrix is nearly singular there and the minimum-norm solution blows up, then falls again because extra capacity gives the optimizer slack to choose a smooth interpolant — and the same shift in perspective explains grokking, where two zero-training-loss solutions compete and weight decay slowly drifts the model from the memorizing one to the generalizing one long after the loss has stopped moving.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is the worst place to be the point where the model can *just barely* fit the data?** (Use the bendy wire, or the packed bookshelf. The answer is about slack, not about size.)
2. **What does "minimum-norm interpolation" mean,** and why is "which of the infinitely many perfect fits do you take?" a more useful question than "how many parameters do I have?"
3. **Why does adding more parameters make the solution *smaller*?** (One data point, $p$ features, each doing $1/p$ of the work.)
4. **Explain sample-wise double descent to someone who thinks more data is always better.** Where exactly does the extra data hurt, and why does it stop hurting?
5. **Why is "double descent disproves the bias–variance tradeoff" the wrong summary?** What single change makes the peak disappear entirely?
6. **What is the difference between interpolating and overfitting?** Name a method from the 1950s that interpolates by construction and generalizes fine.
7. **State the benign-overfitting condition without using the word "overparameterized."** (It is about the shape of the data's covariance spectrum, and it is a condition, not a guarantee.)
8. **What does the neural tangent kernel measure,** in terms of two examples wanting the same parameters changed?
9. **Why does an infinitely wide network barely move its weights,** and what does the stadium wave have to do with it?
10. **Why is "networks are just kernel machines" the wrong takeaway from the NTK,** and what number measures how wrong it is?
11. **Explain spectral bias without the word "eigenvalue,"** and use it to say why random labels take longer to fit than real ones.
12. **What is the difference between "the model can't represent this" and "the model learns this last"?** Why do they look identical on any run you actually stop?
13. **How can a model still be changing when its training loss has been exactly zero for 50,000 steps?** (The answer is one word about the geometry of the zero-loss set.)
14. **What force drives grokking, and what happens if you switch it off?**
15. **Why does the network solve modular arithmetic by putting numbers on a clock face?** Why is that the natural answer rather than a clever one?
16. **Why is grokking the best available evidence that "emergent abilities" can be metric artifacts?** What can you check here that you cannot check on a large language model?
17. **Explain the difference between the lazy and rich regimes,** and say what $\mu$P has to do with it, in one sentence.

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 31 — Neural Collapse, Implicit Bias & Lottery Tickets](31-neural-collapse-implicit-bias-lottery-tickets.md)
