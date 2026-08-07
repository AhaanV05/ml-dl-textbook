# Chapter 7 — Normalization & Regularization

> **Prerequisites:** Ch. 4 (condition number), Ch. 6 (variance propagation).

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, $\odot$, $\langle a,b\rangle$, or $\|W\|$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. Two warnings specific to this chapter: **$\gamma$ and $\beta$ here are a normalization layer's learned scale and shift, not Adam's momentum coefficients**, and **$\lambda$ without a subscript is weight-decay strength while $\lambda_i$ with one is a Hessian eigenvalue.** Both collisions appear on the same page in §7.5.

### Symbols introduced in this chapter

Skim once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\mathcal{B}$ | "script B" | The **mini-batch** — the group of examples processed together |
| $\mu_\mathcal{B},\ \sigma^2_\mathcal{B}$ | "batch mu, batch sigma-squared" | The mean and variance computed **over that batch** |
| $\hat x$ | "x-hat" | The **standardized** activation — here "hat" means normalized, not "predicted" |
| $\gamma,\ \beta$ | "gamma, beta" | Learned per-channel **scale** and **shift** applied after normalizing |
| $\epsilon$ | "epsilon" | A tiny constant ($\sim10^{-5}$) that prevents division by zero |
| $(B,C,H,W)$ | "batch, channels, height, width" | The shape of a convolutional activation tensor |
| $\odot$ | "elementwise times" (Hadamard) | Multiply matching entries, keep them separate |
| $\eta$ | "eta" | Learning rate |
| $\lambda$ | "lambda" | **Weight-decay strength** — but $\lambda_i$ with an index is a Hessian eigenvalue |
| $\eta_{\text{eff}}$ | "eta effective" | The step size the *function* actually experiences, as opposed to the one you set |
| $\|W\|$ | "norm of W" | The length of the weight matrix treated as one long vector |
| $\langle a, b\rangle$ | "inner product of a and b" | Dot product; **zero means perpendicular** |
| $\kappa,\ \lambda_{\max}(H)$ | "kappa", "lambda-max of H" | Condition number; the sharpest curvature of the loss surface |
| $m$ | "m" | **Two jobs here:** the running-average momentum in §7.2, and the dropout mask in §7.5 |
| $p$ | "p" | Dropout probability — the chance a unit is **dropped**, not kept |
| $\mathrm{Bernoulli}(q)$ | "Bernoulli of q" | A coin flip that lands 1 with probability $q$ |
| $\mathrm{Beta}(\alpha,\alpha)$ | "Beta alpha alpha" | A distribution over the interval $[0,1]$ |
| $\mathbf{1}$ | "the all-ones vector" | Every entry equal to 1 |
| $\Delta\angle$ | "delta angle" | How far the weight vector's **direction** moved |
| $\propto$ | "is proportional to" | Equal up to a constant we don't care about |

**Full forms for every abbreviation used in this chapter:**

| Short | Full form |
|---|---|
| BN | Batch Normalization |
| LN | Layer Normalization |
| GN / IN | Group Normalization / Instance Normalization |
| RMSNorm | Root Mean Square Normalization |
| EMA | Exponential Moving Average |
| SGD | Stochastic Gradient Descent |
| LR | Learning Rate |
| CNN | Convolutional Neural Network |
| ViT | Vision Transformer |
| OLS | Ordinary Least Squares |
| MAP | Maximum A Posteriori |
| MC dropout | Monte Carlo dropout |
| LASSO | Least Absolute Shrinkage and Selection Operator |
| MoCo | Momentum Contrast (a contrastive-learning method) |
| ShuffleBN | Shuffling BatchNorm (MoCo's fix for BN's cross-sample leakage) |
| GAN | Generative Adversarial Network |
| RL | Reinforcement Learning |
| DeepNorm | (not an acronym — a post-LN variant with scaled residuals) |
| DropPath | (not an acronym — dropout applied to whole residual branches) |

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

#### "Internal covariate shift," and what replaced it

**First the phrase itself.** *Covariate* is a statistician's word for an input variable. *Covariate shift* is the situation where the distribution of inputs changes between training and deployment — you trained on daytime photographs and are now being shown night-time ones. **Internal** covariate shift extends the idea to the *inside* of a network: layer 7's inputs are layer 6's outputs, and layer 6's weights are changing every step, so layer 7 is being retrained on a moving target.

▸ **It is a  appealing story, and it appears to be mostly wrong.** The decisive experiment was to *deliberately inject* distribution shift immediately after each BatchNorm layer — adding random, time-varying noise so that later layers really did face a moving target — and observe that training was essentially unharmed. If eliminating covariate shift were the mechanism, deliberately reintroducing it should have destroyed the benefit. It did not.

**What "smoothing" means instead.** Recall from Chapter 1 §1.1.4 that a **Lipschitz constant** is a speed limit: if a function has Lipschitz constant $K$, nudging its input by $\epsilon$ moves its output by at most $K\epsilon$. Normalization lowers that constant for both the loss *and* its gradient.

| Quantity | What a smaller Lipschitz constant means |
|---|---|
| The loss $\mathcal{L}$ | The loss cannot change wildly for a small parameter step |
| The gradient $\nabla\mathcal{L}$ | The **direction** you computed stays valid over a longer distance |

> **Analogy.** You are walking downhill in fog and can only sense the slope where you stand. If the terrain is jagged, the direction that is downhill *here* may be uphill one metre away — so you must take tiny steps. If the terrain is smooth and rolling, the slope you measure remains roughly correct for a hundred metres, so you can stride. **Normalization does not tell you a better direction; it makes the direction you already have trustworthy for longer.** That is what buys you a larger learning rate, and a larger learning rate is what buys you speed.

**Connecting to $\kappa$ from Chapter 4.** The condition number $\kappa = \lambda_{\max}(H)/\lambda_{\min}(H)$ is the ratio of the sharpest curvature to the flattest — how much more elongated the loss valley is in one direction than another. Gradient descent's convergence rate degrades directly with $\kappa$, and the maximum stable learning rate is set by $\lambda_{\max}$ alone. So **dropping $\lambda_{\max}$ raises the ceiling on $\eta$**, which is the concrete, measurable version of "the landscape is more predictable." Put a number on it: if normalization halves $\lambda_{\max}$, you can double the learning rate and take the same risk you were taking before.

> **Where this came from.** BatchNorm was introduced by **Sergey Ioffe and Christian Szegedy** at Google in 2015. It worked immediately and dramatically — networks that had needed careful initialization and small learning rates suddenly trained fast and robustly — and the paper became one of the most-cited in the history of machine learning, with tens of thousands of citations. The explanation it offered, internal covariate shift, was largely taken on faith for three years, because the technique so obviously worked that few people interrogated the reason. **Santurkar, Tsipras, Ilyas, and Mądry** at MIT published the careful ablations in 2018, and the field quietly changed its mind about *why* while changing nothing about *what* it did. This is a recurring shape of story in deep learning, and worth internalizing as a general caution: **a method's empirical success is not evidence for the explanation attached to it.** The two are separable, and here they were separated by three years and thousands of citations.

---

## 7.2 BatchNorm

### The math

For a mini-batch of activations $\{x_i\}_{i=1}^{B}$ at a given feature/channel:

$$\mu_{\mathcal{B}} = \frac1B\sum_i x_i,\qquad \sigma^2_{\mathcal{B}} = \frac1B\sum_i (x_i-\mu_\mathcal{B})^2$$
▸ $$\hat x_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}},\qquad y_i = \gamma\hat x_i + \beta$$

$\gamma,\beta$ are learned per channel, so the network can undo the normalization if it wants — which matters, because forcing zero mean into a ReLU or sigmoid would restrict it to a near-linear regime.

**Placement:** conv → BN → activation. The bias in the conv is redundant (BN's $\beta$ subsumes it); set `bias=False`.

#### Reading the BatchNorm equations in plain English

Four short formulas, and they say: **measure this batch, subtract the middle, divide by the spread, then let the network scale and shift it back however it likes.**

- $\mu_\mathcal{B} = \frac1B\sum_i x_i$ — *"the average of this feature across the examples in the batch."* The subscript $\mathcal{B}$ is a reminder that it is computed over the batch, not over the dataset.
- $\sigma^2_\mathcal{B} = \frac1B\sum_i(x_i - \mu_\mathcal{B})^2$ — *"the average squared distance from that mean."* Squaring is what makes it a size rather than a direction; without it the deviations would cancel to exactly zero.
- $\hat x_i = (x_i - \mu_\mathcal{B})/\sqrt{\sigma^2_\mathcal{B}+\epsilon}$ — *"how many standard deviations above the batch mean is this value?"* This is the **z-score** from elementary statistics, and after this step the batch has mean exactly 0 and variance exactly 1, by construction.
- $y_i = \gamma\hat x_i + \beta$ — *"now scale it by a learned number and shift it by another learned number."*

**Put numbers in.** One channel, batch of four activations $x = (2,\ 4,\ 4,\ 10)$:

| Step | Computation | Result |
|---|---|---|
| Mean | $(2+4+4+10)/4$ | $\mu = 5$ |
| Deviations | $(2{-}5,\ 4{-}5,\ 4{-}5,\ 10{-}5)$ | $(-3,\ -1,\ -1,\ 5)$ |
| Variance | $(9+1+1+25)/4$ | $\sigma^2 = 9$, so $\sigma = 3$ |
| Standardize | divide each deviation by 3 | $\hat x = (-1,\ -\tfrac13,\ -\tfrac13,\ \tfrac53)$ |
| Scale/shift, $\gamma=2,\beta=1$ | $2\hat x + 1$ | $y = (-1,\ \tfrac13,\ \tfrac13,\ \tfrac{13}{3})$ |

Check: $\hat x$ has mean $(-1 - \tfrac13 - \tfrac13 + \tfrac53)/4 = 0$ ✓ and variance $(1 + \tfrac19 + \tfrac19 + \tfrac{25}9)/4 = 1$ ✓.

> **Analogy.** Grading on a curve. The raw scores were 2, 4, 4, and 10; nobody can interpret those without knowing the exam. Convert to "standard deviations above the class mean" and every exam becomes comparable. **BatchNorm curves every layer's output against its own batch, so downstream layers always receive scores on the same scale regardless of what the layer upstream has been doing to its weights.**

**Why $\epsilon$ exists.** If every value in the batch is identical, $\sigma^2 = 0$ and you would divide by zero. Adding $\epsilon \approx 10^{-5}$ inside the square root makes that impossible. It also caps the amplification of a nearly-constant channel: without it, a channel with $\sigma = 10^{-8}$ would be multiplied by $10^8$, turning floating-point noise into a signal. **It is a numerical guard, not a modelling choice — but it is not free, since it means the output variance is slightly under 1.**

**Why $\gamma$ and $\beta$ are not self-defeating.** It looks circular: normalize, then immediately allow the network to un-normalize. The point is that **the network now *chooses* the scale rather than inheriting it.** Consider a sigmoid: its useful, non-saturated region is roughly $z \in [-2, 2]$. Forcing every input into mean 0, variance 1 traps the sigmoid in its near-linear middle, which throws away the nonlinearity. With $\gamma$ learnable, the layer can set $\gamma = 4$ if it wants saturation and $\gamma = 0.5$ if it wants linearity. ▸ **Normalization removes the scale as an accident of optimization and re-introduces it as a parameter.** That swap — from emergent to explicit — is the mechanism, and the same idea recurs throughout the chapter.

**Why the convolution's bias becomes redundant.** BatchNorm subtracts the batch mean. Any constant added before that subtraction is subtracted straight back out: if $x_i \to x_i + b$, then $\mu \to \mu + b$ and $x_i - \mu$ is unchanged. So the bias has **exactly zero effect on the output**, while still consuming memory, receiving gradients, and drifting aimlessly. $\beta$ does the same job on the other side of the normalization, where it actually survives. Setting `bias=False` is not a micro-optimization; it removes a parameter that provably does nothing.

### The backward pass, derived

This is a classic interview question. The subtlety is that $\mu$ and $\sigma^2$ *depend on every $x_i$*, so gradients flow through three paths.

Let $\bar y_i = \partial\mathcal{L}/\partial y_i$. Then $\partial\mathcal{L}/\partial\hat x_i = \bar y_i\gamma$, and:

$$\frac{\partial\mathcal{L}}{\partial\sigma^2} = \sum_i \frac{\partial\mathcal{L}}{\partial\hat x_i}\cdot(x_i-\mu)\cdot\left(-\tfrac12\right)(\sigma^2+\epsilon)^{-3/2}$$
$$\frac{\partial\mathcal{L}}{\partial\mu} = \sum_i \frac{\partial\mathcal{L}}{\partial\hat x_i}\cdot\frac{-1}{\sqrt{\sigma^2+\epsilon}} + \frac{\partial\mathcal{L}}{\partial\sigma^2}\cdot\frac{-2\sum_i(x_i-\mu)}{B}$$

Collecting (the second term of $\partial\mathcal{L}/\partial\mu$ vanishes since $\sum_i(x_i-\mu)=0$):

▸ $$\frac{\partial\mathcal{L}}{\partial x_i} = \frac{1}{B\sqrt{\sigma^2+\epsilon}}\left(B\,\frac{\partial\mathcal{L}}{\partial\hat x_i} - \sum_j\frac{\partial\mathcal{L}}{\partial\hat x_j} - \hat x_i\sum_j \frac{\partial\mathcal{L}}{\partial\hat x_j}\hat x_j\right)$$

Read the three terms: the raw gradient, minus its batch mean, minus its projection onto $\hat x$. **BatchNorm's backward pass removes the component of the gradient that would change the mean or the scale** — those directions are already handled by $\beta$ and $\gamma$. That is why BN makes the layer's weight *magnitude* irrelevant (§7.4).

Also note: $\bar\gamma = \sum_i \bar y_i\hat x_i$, $\bar\beta = \sum_i \bar y_i$.

#### The BatchNorm backward pass, decoded

**Why this is harder than any other backward pass in the book.** Every other layer is *per-example*: example 3's output depends only on example 3's input. BatchNorm is not. Because $\mu$ and $\sigma^2$ are computed from the whole batch, **example 3's output depends on examples 1, 2, and 4 as well** — so when you differentiate, changing $x_3$ moves $\mu$, which moves *everyone's* $\hat x$. Hence three paths, not one:

| Path | Route | In words |
|---|---|---|
| Direct | $x_i \to \hat x_i$ | "I changed, so my own normalized value changed" |
| Via the mean | $x_i \to \mu \to \hat x_j$ for all $j$ | "I changed, so the batch average moved, so everyone shifted" |
| Via the variance | $x_i \to \sigma^2 \to \hat x_j$ for all $j$ | "I changed, so the batch spread moved, so everyone rescaled" |

The messy intermediate formulas are just those three paths written out and then added, exactly as the "multiple paths add" rule from Chapter 6 §6.2 requires.

**Now the collected result, read as three terms.** Write $g_i = \partial\mathcal{L}/\partial\hat x_i$ for the gradient arriving at the normalized value. Divide the boxed formula through by $B$ and it says:

$$\frac{\partial\mathcal{L}}{\partial x_i} = \frac{1}{\sqrt{\sigma^2+\epsilon}}\Big(\underbrace{g_i}_{\text{the raw gradient}} - \underbrace{\bar g}_{\text{its batch mean}} - \underbrace{\hat x_i\,\tfrac1B\textstyle\sum_j g_j\hat x_j}_{\text{its projection onto }\hat x}\Big)$$

▸ **In one sentence: BatchNorm hands back the incoming gradient with two directions surgically deleted — the direction that would shift the batch mean, and the direction that would rescale the batch.** Those two moves are already available to the network through $\beta$ and $\gamma$, so allowing $x$ to make them too would be duplicated effort pulling in the same direction.

**Verify it, which takes two lines and is worth doing.** Sum the returned gradient over the batch: the first two terms cancel exactly (the mean of $g_i$ minus the mean of $g_i$), and the third vanishes because $\sum_i \hat x_i = 0$. So

$$\sum_i \frac{\partial\mathcal{L}}{\partial x_i} = 0 \qquad\text{and, by the same kind of cancellation,}\qquad \sum_i \hat x_i\frac{\partial\mathcal{L}}{\partial x_i} = 0$$

**Two exact constraints, always, on every batch.** The gradient flowing back through BatchNorm is orthogonal to the all-ones vector and orthogonal to $\hat x$. Out of $B$ available degrees of freedom, two are removed by construction, leaving $B-2$.

> **Analogy.** A committee that has already delegated two decisions — "what the overall budget is" ($\gamma$) and "what the baseline is" ($\beta$) — to two named officers. When a proposal comes back to the floor, the chair strips out any part of it that merely re-argues the budget or the baseline, and forwards only the part about *relative allocation*. **The gradient is not diminished; it is projected onto the subspace of questions this layer is still allowed to answer.**

▸ **This is the cleanest available explanation of §7.4 in advance:** if the gradient is always orthogonal to the "make everything bigger" direction, then gradient descent literally cannot change the layer's overall scale — which is precisely why the weight *magnitude* stops being a property of the function and becomes a property of the optimizer.

**The parameter gradients, decoded.** $\bar\gamma = \sum_i \bar y_i\hat x_i$ is a dot product: *"how much does the loss want the output to move in the direction of $\hat x$ itself?"* — i.e. **should this channel be louder or quieter overall.** And $\bar\beta = \sum_i \bar y_i$ is simply the total incoming gradient: *"should this channel be shifted up or down."* Both are sums over the batch because both parameters are shared across it.

### The train/eval discrepancy — the biggest practical gotcha

**Training:** uses batch statistics $\mu_\mathcal{B},\sigma^2_\mathcal{B}$, so a sample's output *depends on the other samples in its batch*. This is a form of noise, and it's a real regularizer.

**Inference:** uses running estimates accumulated during training:
$$\mu_{\text{run}} \leftarrow (1-m)\mu_{\text{run}} + m\,\mu_\mathcal{B}$$
with momentum $m$ (PyTorch default 0.1, i.e. horizon $\approx10$ batches).

▸ **Failure modes this creates:**
1. **Small batches.** $B\le 8$: the batch statistics are wildly noisy ($\mathrm{Var}(\hat\sigma^2)\propto 1/B$), training degrades sharply. At $B=2$ BN is nearly useless.
2. **Train/test mismatch.** If the running stats haven't converged (short training, or an LR schedule that moves activations late), eval performance collapses while training loss looks fine. Classic symptom: **model performs well in `train()` mode and badly in `eval()` mode.**
3. **Distribution shift at test time** breaks the frozen statistics. (Test-time BN adaptation — recomputing stats on the test batch — often recovers most of the loss.)
4. **Any per-sample dependency is broken by BN**: it leaks information across the batch, which is a  problem for contrastive learning (fixed by ShuffleBN in MoCo) and for RL.
5. **Sequence models.** Sequence length varies and padding pollutes the statistics. This is the main reason transformers use LayerNorm.

### Ghost BatchNorm

Compute BN statistics over *sub-batches* of e.g. 32 even when the real batch is 1024. Preserves the regularizing noise that large-batch training destroys, and recovers part of the large-batch generalization gap.

#### Why the train/eval split causes so much pain

**The structural problem, stated once.** During training, BatchNorm's output for a given image *depends on which other images happened to be in its batch*. At inference you often have one input and no batch, and in any case you need a deterministic answer. So the two modes compute  different functions, and every failure below is a consequence of that single fact.

**Reading the running-statistics update.** $\mu_{\text{run}} \leftarrow (1-m)\mu_{\text{run}} + m\,\mu_\mathcal{B}$ is an **EMA (exponential moving average)**: *"keep $90\%$ of what you believed and mix in $10\%$ of what you just measured."* The $\leftarrow$ is assignment, not equality (Chapter 0 §0.11) — it is a line of code. The effective horizon is $1/m$: with PyTorch's default $m = 0.1$, roughly the last 10 batches dominate, and a measurement from 50 batches ago retains weight $0.9^{50} = 0.005$, i.e. half a percent. **The running statistics have a memory of about ten batches, and everything before that has been forgotten.**

**Failure 1, with numbers.** Estimating a variance from $B$ samples has a relative standard error of about $\sqrt{2/B}$:

| $B$ | Typical error in $\hat\sigma^2$ |
|---|---|
| 2 | $\approx100\%$ |
| 8 | $\approx50\%$ |
| 32 | $\approx25\%$ |
| 256 | $\approx9\%$ |

▸ At $B = 2$ your estimate of the spread is, on average, wrong by about as much as the spread itself. You are not normalizing; you are dividing by a random number. **This is why BatchNorm degrades sharply below $B \approx 16$, and it is a fact about statistics, not about deep learning** — no implementation trick can extract a reliable second moment from two samples.

**Failure 2, the one that wastes whole afternoons.** The symptom is unmistakable once you have seen it: **training loss looks excellent, `eval()` performance is terrible, and switching the model back to `train()` mode makes the problem disappear.** The cause is that the running statistics do not match the statistics the network is actually producing — either training was too short for the EMA to converge, or the learning-rate schedule moved the activations substantially in the final phase, and the ten-batch memory could not keep up. ▸ **If eval is broken but train is fine, suspect the normalization statistics before you suspect the model.** It is the single most common instance of a model that is correct and an evaluation that is not.

**Failure 4, in a sentence:** BatchNorm makes a network's prediction for one example depend on other examples in the same batch. For most supervised learning nobody notices. For **contrastive learning** it is a leak — the model can identify which samples share a batch and exploit that instead of learning representations, which is what ShuffleBN in MoCo exists to prevent. For **reinforcement learning**, where batches are correlated slices of a trajectory rather than independent samples, the statistics are biased in a way that quietly poisons the estimates.

**Failure 5, and why transformers use LayerNorm.** Sequences have different lengths, so batches are padded. Padding tokens are not data, but they are numbers, and they land in the same tensor positions the normalization sums over. Your batch mean is then partly an average over meaningless zeros, and — worse — **it varies with how many pad tokens happened to be in the batch**, which is a property of your data loader rather than your model. LayerNorm, which normalizes each token independently, never touches another token's values and is completely immune.

**Ghost BatchNorm, decoded.** Split a batch of 1024 into 32 groups of 32 and normalize each group with its own statistics. The observation behind it: BatchNorm's batch-dependence is a **noise source**, and noise is a regularizer. A large batch gives precise statistics, which is good for numerical stability and bad for generalization, because it removes that noise. Ghost BatchNorm deliberately keeps the statistics noisy. ▸ **You get large-batch throughput with small-batch regularization** — an unusually clean case of choosing which property of a hyperparameter you want, rather than accepting the bundle.

#### Examples and non-examples: what BatchNorm actually guarantees

BatchNorm is the most over-credited layer in deep learning. It does a small, precisely-specified thing, and almost everything else attributed to it is either a downstream consequence or simply false. Here is the line.

**✅  examples — things BatchNorm really does deliver**

| Example | Why it qualifies |
|---|---|
| Channel 7's pre-$\gamma$ values across a batch of 256 have sample mean $0$ and sample variance $1$ | This is the formula executed literally: subtract $\mu_\mathcal{B}$, divide by $\sqrt{\sigma_\mathcal{B}^2+\epsilon}$ |
| Doubling every weight in the conv layer feeding the BN changes nothing about the output | $\mathrm{BN}(2Wx) = \mathrm{BN}(Wx)$ — the scale cancels in the ratio (§7.4) |
| An image's logits shift slightly depending on which 255 other images shared its batch | The statistics are computed *over the batch*, so cross-sample coupling is structural, not incidental |
| Training with $\eta = 1.0$ where the un-normalized net diverged at $\eta = 0.05$ | Better conditioning of the loss surface — the documented, reproducible benefit |
| Removing the layer's bias term with no loss of expressivity | $\mu_\mathcal{B}$ subtraction erases any constant added upstream; $\beta$ replaces it |

**❌ Near-misses — credited to BatchNorm, but not what it does**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "The activations become Gaussian" | Standardizing fixes the **first two moments only**. Feed in the bimodal set $\{-10,-10,10,10\}$ and you get $\{-1,-1,1,1\}$ — still bimodal, still not remotely bell-shaped | Mean-and-variance standardization. Skew, kurtosis, and modality survive untouched |
| "The layer's output has mean 0 and variance 1" | $\gamma$ and $\beta$ are applied *after* normalizing and are **learned**. Output mean is $\beta$, output sd is $\lvert\gamma\rvert$ | The *normalized intermediate* has mean 0, variance 1. The output has whatever the network decided it wanted |
| "It removes internal covariate shift" | The original claim; Santurkar et al. measured it and found BN networks can have *more* distributional drift, yet still train better (§7.1) | A conditioning improvement, whose mechanism the original paper mis-diagnosed |
| "It normalizes each example" | Each *channel* is normalized *across* examples. One example alone has no BN statistics at all | A per-channel, across-batch operation. The per-example version is LayerNorm |
| Dividing an input tensor by 255 before the first layer | A fixed, data-independent rescaling with no learned parameters and no batch coupling | **Input preprocessing.** Useful, but it is not a normalization *layer* |
| `F.normalize(x, dim=-1)` (projecting to the unit sphere) | Divides by $\|x\|$ without subtracting a mean and without a learned gain | $\ell_2$ **normalization** — closer to RMSNorm without $\gamma$ |

▸ **The boundary:** BatchNorm guarantees exactly two numbers about a set of activations — their mean and their variance, computed across the batch, before the affine step. **Any claim about the *shape* of the distribution, about a single example in isolation, or about the layer's final output is outside what the formula promises.**

> **Common misconception.** *"BatchNorm makes the activations Gaussian."* It fixes the mean and the variance. It has no access to, and no effect on, any higher moment. Standardize $\{-10, -10, 10, 10\}$ and you get $\{-1, -1, 1, 1\}$: mean 0, variance 1, and a perfectly bimodal distribution with zero density at the mean. Standardize a heavy-tailed set and you get a heavy-tailed set with unit variance. The belief is tempting for two reasons. First, the word "normalize" shares a root with "normal distribution," and they are unrelated here — "normalize" means "put on a standard scale," not "make normally distributed." Second, the *pictures* in the original paper show activation distributions that do look roughly bell-shaped — but that is a property of what sums of many terms tend to look like (the central limit theorem acting on the pre-activation), and it would be true with or without the BN layer.

> **Common misconception.** *"BatchNorm is just a normalization — it computes the same thing whether the model is in `train()` or `eval()` mode."* It computes two  different functions. In `train()` it uses $\mu_\mathcal{B}, \sigma_\mathcal{B}^2$ from the current batch, so the output for one image depends on the other images beside it. In `eval()` it uses frozen running estimates, so the output depends on nothing but that image. Same weights, same input, two different numbers out. The belief is tempting because every *other* common layer — conv, linear, ReLU, softmax, embedding —  is mode-independent, so "the model in eval mode" feels like a bookkeeping flag rather than a change of function. It is not: **forgetting `model.eval()` and forgetting `model.train()` are two distinct bugs with opposite symptoms**, and BatchNorm and dropout are the only common layers that produce either.

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

#### The normalization family, decoded — it is one formula and one choice

**Take the sentence at the top of this section seriously: all five methods compute the identical formula.** They differ in exactly one respect — *which numbers get averaged together to produce the mean and the variance.* Once you see that, the table is a table of one decision.

**Understanding $(B, C, H, W)$ physically.** Take a batch of 32 photographs, each processed into 64 feature channels of resolution $28\times28$. Then $B = 32$, $C = 64$, $H = W = 28$, and the tensor holds $32 \times 64 \times 28 \times 28 = 1{,}605{,}632$ numbers. A **channel** is one learned feature detector — "vertical edges," say — evaluated at every spatial position of every image.

Now each method picks a different group of numbers to standardize together:

| Method | Groups together | How many groups | Numbers per group |
|---|---|---|---|
| **BatchNorm** | one channel, all images, all pixels | 64 | 25,088 |
| **LayerNorm** | one image, all channels, all pixels | 32 | 50,176 |
| **InstanceNorm** | one image, one channel, all pixels | 2,048 | 784 |
| **GroupNorm** ($G{=}32$) | one image, 2 channels, all pixels | 1,024 | 1,568 |

▸ **The single most consequential column is the second one: does the group span the batch dimension?** BatchNorm's does; nothing else's does. Every practical difference between these methods — the small-batch failure, the train/eval gap, the padding problem, the contrastive-learning leak — follows from that one structural fact and nothing else.

> **Analogy.** You are grading students and must convert raw scores to a curve. **BatchNorm** curves each *question* across all students: "how did everyone do on question 7?" You need many students for that to be meaningful. **LayerNorm** curves each *student* across all their questions: "how did this student do relative to their own average?" — which works even if there is only one student in the room. **InstanceNorm** curves each student on each *section* separately. **GroupNorm** curves each student on each *group of related sections*. The formula is identical in all four; only the reference population changes, and the reference population is the entire design.

**Why GroupNorm is the safe default when batches are small.** Setting $G=1$ groups all channels together, which is LayerNorm; setting $G=C$ gives each channel its own group, which is InstanceNorm. $G=32$ sits between: enough channels per group for a stable estimate, few enough that unrelated features are not forced onto a shared scale. Because no group spans the batch, the statistics are just as reliable at $B=2$ as at $B=256$ — which is why detection, segmentation, video, and 3D work, where a single example can fill a GPU, use GroupNorm as a matter of course.

> **Where this came from.** Each member of this family was invented for a specific problem that BatchNorm could not handle, and only afterwards understood as a variation on one theme. **LayerNorm** (Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton, 2016) was designed for **recurrent networks**, where the sequence length varies per example so batch statistics are ill-defined. The transformer did not exist when it was published — it arrived the following year — so the technique that now normalizes essentially every large language model on earth was built for an architecture it would help displace. **InstanceNorm** (Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky, 2016) came from **artistic style transfer**, where the observation was that an image's overall contrast is precisely the thing you want to discard before restyling it. **GroupNorm** (Yuxin Wu and Kaiming He, 2018) came from **object detection**, where high-resolution inputs force batch sizes of one or two. Three unrelated problems, three papers, one formula with a different index set.

#### Examples and non-examples: is that a normalization layer?

The family shares a definition: **compute statistics over some index set of the activation tensor, standardize by them, then apply a learned affine correction.** Plenty of operations look like members and aren't.

**✅  members of the family**

| Example | Which axes it averages over | Why it qualifies |
|---|---|---|
| `nn.BatchNorm2d(64)` on a $(32,64,28,28)$ tensor | $(B,H,W)$, one statistic pair per channel | Standardizes by statistics computed *from the activations themselves* |
| `nn.LayerNorm(768)` on a transformer token | The 768 features of that one token | Same formula, index set is one sample |
| `nn.GroupNorm(32, 64)` | 2 channels $\times\, H \times W$, per sample | Same formula, index set is a channel group |
| RMSNorm in LLaMA | The 768 features, second moment only | Same formula with the centring step dropped |
| InstanceNorm in a style-transfer net | $H\times W$, per sample per channel | Same formula, index set is one feature map |

**❌ Near-misses — rescale things, but are not normalization layers**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Dividing pixels by 255 | Constant divisor, chosen by you, not measured from the data at run time | **Input scaling / preprocessing** |
| Subtracting the ImageNet channel means $(0.485,0.456,0.406)$ | Statistics computed *once, offline, over the whole dataset* — frozen constants, not per-forward-pass quantities | **Dataset standardization** |
| `F.normalize(x, p=2, dim=-1)` before a cosine similarity | No mean subtracted, no learned $\gamma,\beta$, and the divisor is a norm rather than a standard deviation | **$\ell_2$ projection onto the unit sphere** |
| Softmax | Makes the output sum to 1, but is a nonlinear reweighting, not a shift-and-scale | A **normalized exponential** — normalizing a *distribution*, not a scale |
| Weight normalization ($W = g\,v/\|v\|$) | Operates on the **parameters**, never looks at any activation | A **reparameterization** of the weights |
| Gradient clipping to norm 1.0 | Rescales gradients in the backward pass; forward activations are untouched | An **optimizer-side** stabilizer (Ch. 9) |
| A learned scalar multiplier $\alpha\,x$ (as in ReZero) | There is no measured statistic anywhere — nothing is divided by a spread | A **learned gain**, and a very useful one |

▸ **The boundary:** a normalization *layer* measures a statistic from the activations flowing through it *on this forward pass* and divides by it. If the divisor was fixed before training started, or comes from the weights rather than the activations, it is not a member of this family — and, crucially, it will not give you the scale invariance that §7.4 is about.

> **Common misconception.** *"Once you add normalization, initialization and learning rate stop mattering."* Normalization widens the range of learning rates that work — it does not make the choice free. Pre-LN transformers still need warmup at scale; post-LN transformers still diverge without it; residual branches still need small or zero-initialized final layers (§8.2). What normalization actually buys is *insensitivity to the scale of the weights feeding it*, which is a narrower guarantee than "insensitivity to how you set things up." The belief is tempting because the first thing anyone notices about BatchNorm is that a learning rate which previously exploded now trains fine — so it feels like the constraint was removed rather than relaxed. **The constraint moved:** §7.4 shows that with normalization in place, the quantity that actually governs training is the *product* $\eta\lambda$, and getting that wrong is just as fatal as getting $\eta$ wrong was before.

### LayerNorm

▸ $$\mathrm{LN}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu=\frac1d\sum_j x_j,\ \ \sigma^2=\frac1d\sum_j(x_j-\mu)^2$$

Per-token, per-sample. No batch dependence, identical in train and eval, works with any sequence length. **This is why it won for transformers.**

**Geometric reading:** LN projects $x$ onto the hyperplane $\sum_j x_j = 0$ (mean removal) and then onto a sphere of radius $\sqrt d$ (variance normalization). So LN maps $\mathbb{R}^d \to$ a $(d-2)$-sphere. Every LayerNorm output has norm exactly $\sqrt d$ before the $\gamma$ scaling. This is worth remembering when reasoning about the residual stream (Ch. 32).

#### LayerNorm's geometry, decoded

**What the formula says.** $\mu = \frac1d\sum_j x_j$ averages over the **feature dimension** — across the 768 numbers describing *this one token*, not across tokens and not across the batch. Everything else is identical to BatchNorm. **One index changed, and every practical property changed with it.**

**Now the geometric reading, one move at a time.**

*Move 1 — subtract the mean.* The set of vectors whose entries sum to zero is a **hyperplane**: a flat slab through the origin, one dimension lower than the space it sits in ($d-1$ dimensions inside $\mathbb{R}^d$). Subtracting the mean is the projection onto it. Concretely, $(1, 5, 3)$ has mean 3 and becomes $(-2, 2, 0)$, which sums to zero.

*Move 2 — divide by the standard deviation.* After centring, $\sigma^2 = \frac1d\sum_j \hat x_j^2 = \frac1d\|\hat x\|^2$, so dividing by $\sigma$ forces $\|\hat x\|^2 = d$, i.e. $\|\hat x\| = \sqrt d$ **exactly, for every input, always.** The second move pins the length.

▸ **So LayerNorm takes all of $\mathbb{R}^d$ and squeezes it onto a single sphere of radius $\sqrt d$ living inside a $(d-1)$-dimensional hyperplane — a surface of dimension $d-2$.** Two degrees of freedom are destroyed: one by fixing the mean, one by fixing the length. This is the *same* pair of constraints that appeared in BatchNorm's backward pass, viewed from the forward direction rather than the backward one.

**Put numbers on it.** For a transformer with $d = 768$: every LayerNorm output has norm exactly $\sqrt{768} = 27.7$ before $\gamma$ is applied. **Not approximately — exactly, on every token, on every layer, forever.** If you instrument a network and find a post-LN activation with a different norm, you have a bug, and this is one of the sharpest assertions you can make about a running model.

> **Analogy.** Every arriving message is rewritten to be exactly one page long and to have exactly zero net sentiment. What survives is **only the relative emphasis between the words** — the direction, not the volume and not the baseline. This is why LayerNorm is described as making the network care about *direction in feature space*: it has removed, by construction, the only two things that are not direction.

▸ **Why this matters for the residual stream (Ch. 32).** Since every normalized vector has the same length, a token's meaning is carried purely by *where it points* in the 768-dimensional space, not by how loud it is. Combined with the Johnson–Lindenstrauss result from Chapter 1 §1.1.5 — that a $d$-dimensional space contains exponentially many near-orthogonal directions — this is the geometric setup that makes superposition possible. **Normalization is not merely a training aid; it establishes the coordinate system in which the network's representations live.**

### RMSNorm

▸ $$\mathrm{RMSNorm}(x) = \gamma\odot\frac{x}{\sqrt{\frac1d\sum_j x_j^2 + \epsilon}}$$

Drops the mean subtraction and $\beta$. Rationale (Zhang & Sennrich, 2019): the re-*scaling* is what matters; the re-*centering* contributes little. Saves ~7–10% of layer-norm compute and one reduction pass.

**Used by LLaMA, Mistral, Gemma, and most current LLMs.** Empirically matched quality.

#### RMSNorm, decoded

**Read the name.** RMS is **root mean square**: square every entry, take the mean, take the square root. It is a measure of typical magnitude that ignores sign — the same quantity an electrical engineer uses for AC voltage, and for the same reason (a sine wave's average is zero, but it certainly delivers power).

$$\mathrm{RMS}(x) = \sqrt{\tfrac1d\textstyle\sum_j x_j^2}$$

Compare with LayerNorm's denominator, $\sqrt{\frac1d\sum_j (x_j - \mu)^2}$. **The only difference is the $-\mu$.** RMSNorm measures distance from *zero*; LayerNorm measures distance from *the batch of features' own mean*. If the mean happens to be near zero — which, in a trained network with zero-mean-ish activations, it usually is — the two are nearly identical numbers.

**Put it in numbers.** Take $x = (3, -1, 4, -2)$. Mean $= 1$. LayerNorm's denominator: deviations $(2,-2,3,-3)$, mean square $= (4+4+9+9)/4 = 6.5$, so $\sigma = 2.55$. RMSNorm's denominator: $(9+1+16+4)/4 = 7.5$, so $\mathrm{RMS} = 2.74$. **Different by 7%, and the network's learned $\gamma$ absorbs a systematic 7% without noticing.**

**What is actually saved.** Two things, and neither is arithmetic:

1. **One reduction pass.** Computing a variance the careful way needs the mean first, so it is two sweeps over the data (or a fused one-pass formula with worse numerics). RMS needs one sweep. On a GPU, a "reduction" — collapsing a vector to a single number — requires cross-thread communication and is disproportionately slow relative to its FLOP count. **The saving is memory-traffic and synchronization, not multiplications.**
2. **The $\beta$ parameter, and its gradient.** Removed entirely.

At 7–10% of layer-norm time, and with normalization called twice per transformer block across dozens of blocks, this is a small but entirely free win — the kind that is worth taking precisely because it costs nothing in quality.

> **Where this came from.** **Biao Zhang and Rico Sennrich** proposed RMSNorm in 2019, in the machine-translation community rather than the language-modelling one. Their argument was an ablation, not a theory: they separated LayerNorm's two operations — re-centring and re-scaling — and measured each. Re-scaling accounted for essentially all the benefit; re-centring accounted for essentially none. **The paper's contribution is a deletion.** It is a good example of a  underrated kind of research: taking a technique everyone uses, asking which of its parts are load-bearing, and removing the rest. Every major open-weight language model family now uses it.

### Where to put it: pre-LN vs post-LN

**Post-LN (original Transformer):** $x_{\ell+1} = \mathrm{LN}(x_\ell + F(x_\ell))$
**Pre-LN (everything modern):** $x_{\ell+1} = x_\ell + F(\mathrm{LN}(x_\ell))$

▸ In pre-LN there is a **clean identity path from input to output with no normalization on it**, so the gradient reaches layer 1 undiminished. In post-LN, every residual addition is followed by a normalization that rescales the gradient, and the product over $L$ layers grows like $O(\sqrt L)$ at initialization — which is why post-LN transformers *require* warmup and pre-LN often doesn't.

**The trade-off:** pre-LN is far more stable but slightly underperforms post-LN at matched size, because the residual stream's magnitude grows with depth and later layers contribute proportionally less. Fixes: **DeepNorm** (post-LN with an $\alpha$-scaled residual and $\beta$-scaled init, allowing 1000-layer transformers), and **sandwich/peri-LN** (normalize both before and after the block).

#### Pre-LN vs post-LN, decoded — where the parenthesis goes

**The entire difference is where one closing bracket sits.** Compare:

| Variant | Formula | Read aloud |
|---|---|---|
| Post-LN | $x_{\ell+1} = \mathrm{LN}(x_\ell + F(x_\ell))$ | "Add the block's output, **then** normalize the sum" |
| Pre-LN | $x_{\ell+1} = x_\ell + F(\mathrm{LN}(x_\ell))$ | "Normalize a copy, feed it to the block, **then** add" |

▸ **Now trace the backward path in each, using the rule from Chapter 6 that addition passes gradients through untouched.** In pre-LN, follow $x_{\ell+1}$ back to $x_\ell$ through the $+$: you arrive **with the gradient completely unmodified**, because nothing sits on that route. Repeat for all $L$ layers and the gradient reaches layer 1 at full strength, having passed through nothing but additions. In post-LN, the LN wraps the *whole sum*, so **every single one of the $L$ hops passes through a normalization** that rescales the gradient. There is no clean path. The rescalings compound, and the accumulated factor across depth grows like $O(\sqrt L)$ at initialization.

Put a number on it: at $L = 100$, that is roughly a $\sqrt{100} = 10\times$ imbalance in gradient scale between the ends of the network at step zero. A single learning rate cannot be right for both ends.

**And that is exactly what warmup is for.** Learning-rate warmup starts at nearly zero and ramps up over the first few thousand steps. Its job is to let the network reach a state where the gradients are better balanced *before* taking any large steps. **Pre-LN often does not need warmup because it never had the imbalance**; post-LN requires it because the first few large steps into an imbalanced network are exactly what produces divergence. If you have ever wondered why transformer recipes universally include a warmup phase that nobody can quite justify from first principles, this is the origin.

> **Analogy.** A relay race with a checkpoint after every leg. **Post-LN** puts the checkpoint on the main track: every runner must pass through it, and every pass costs a little time. **Pre-LN** puts the checkpoint on a side loop: runners can hand the baton straight down the track if they want, and only the ones who take the loop get checked. The baton is guaranteed to arrive — but the main track's traffic is now unregulated, which is exactly the trade-off named below.

**Reading the trade-off honestly.** Pre-LN's identity path is never normalized, so the residual stream's magnitude grows with depth (Ch. 6 §6.4: variances add). By layer 80 the stream is large and layer 80's contribution is proportionally small — so **later layers matter less than they should**, and some capacity is wasted. Post-LN keeps every layer's contribution comparable, which is better *if* you can train it. **DeepNorm** is the attempt to have both: keep post-LN's placement, but scale the residual by $\alpha$ and shrink the initialization by $\beta$ so that the compounding is cancelled by construction. It is what makes 1000-layer transformers trainable.

> **Where this came from.** The original 2017 Transformer used post-LN, and its recipe required a warmup schedule that was widely copied without much explanation. GPT-2 (2019) moved the normalization to the input of each sub-block, and **Xiong et al.** published the gradient analysis in 2020 showing why: post-LN's expected gradient scale at initialization depends on depth while pre-LN's does not. **DeepNorm** came from a Microsoft team in 2022 with the memorable demonstration of a 1,000-layer transformer. There is a pattern here worth noticing — **the fix arrived years after the practice, and the practice was warmup.** For roughly three years the field's standard answer to an architectural instability was a hyperparameter schedule that concealed it.

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

#### Scale invariance, decoded — the strangest true fact in this book

**Start with the invariance itself.** $\mathrm{Norm}((cW)x) = \mathrm{Norm}(Wx)$ says: *"multiply all the weights of this layer by any number you like — 2, 100, one millionth — and the layer's output is bit-for-bit identical."*

Why: multiplying the weights by $c$ multiplies the pre-activations by $c$, which multiplies both their mean and their standard deviation by $c$ — and the normalization computes $(cx - c\mu)/(c\sigma) = (x-\mu)/\sigma$. **The $c$ cancels.** It is not approximately invariant, or invariant near initialization; it is exactly invariant, always. (One caveat worth stating: this holds for the linear layer *feeding* a normalization. It is not true of $\gamma$, $\beta$, or the final output layer — see the corollary.)

> **Analogy.** A thermometer that reports temperature as a percentile against the last hour's readings. Change the units from Celsius to Fahrenheit, or to some absurd scale where you multiply everything by 1000, and the percentile is unchanged. **The weight norm has become a unit, and units are not physics.** Whatever the optimizer does to $\|W\|$, it is not changing what the network computes.

▸ **Now the disorienting consequence.** In a normalized network, **most of the parameter space is redundant.** The set of weight matrices $\{cW : c > 0\}$ — an entire ray through the origin — all compute the same function. What matters is $W/\|W\|$, the *direction*. The optimizer, however, does not know this, and continues to move $\|W\|$ around. **The weight norm has become an optimizer state variable that masquerades as a parameter**, and §7.4 is the story of what it silently controls.

**Reading $\langle\nabla_W\mathcal{L},\, W\rangle = 0$.** Read aloud: *"the gradient and the weight vector have inner product zero"* — which, from Chapter 0 §0.8, means **perpendicular.**

The derivation is a one-liner worth walking. Since $\mathcal{L}(cW) = \mathcal{L}(W)$ for every $c$, the loss does not change as you slide along the ray. So its derivative in that direction is zero. And "the direction along the ray" *is* the direction of $W$ itself. **A function that is flat along a direction has zero gradient component along it.** Nothing more sophisticated is happening.

**Now the Pythagoras step, which is the beautiful part.** An SGD step is $W_{t+1} = W_t - \eta\nabla\mathcal{L}$, and we have just established that $\nabla\mathcal{L} \perp W_t$. **So the step is a right angle to the current position** — and the new length is the hypotenuse of a right triangle:

$$\|W_{t+1}\|^2 = \|W_t\|^2 + \eta^2\|\nabla\mathcal{L}\|^2$$

▸ **The correction term is a sum of squares, so it can never be negative. The weight norm can only grow.** Not "tends to grow," not "usually grows" — under plain SGD in a scale-invariant layer, it is monotonically non-decreasing, step after step, for the entire run. The optimizer is trying to move sideways and the geometry keeps pushing it outwards.

**Put numbers on it.** Take $\|W_t\| = 10$ and $\eta\|\nabla\mathcal{L}\| = 1$. Then $\|W_{t+1}\| = \sqrt{100 + 1} = 10.0499$ — a growth of half a percent in one step. Sustained over ten thousand steps, that is not a rounding error; it is a compounding drift in a quantity nobody is watching and that does not appear in the loss.

> **Analogy.** Walking on the surface of a balloon while always stepping exactly tangent to it — never up, never down, only *around*. Yet each straight-line tangent step lands you slightly outside the sphere you started on, so the balloon you are standing on is imperceptibly bigger every time. You never once tried to inflate it. **The inflation is a consequence of walking straight on a curved surface, and it is entirely invisible from the loss.**

**2. The gradient scales as $1/\|W\|$.** By homogeneity, $\nabla_{cW}\mathcal{L} = \frac1c\nabla_W\mathcal{L}$. Therefore the *angular* update — the only thing that changes the function — has size

▸ $$\Delta\angle \approx \frac{\eta\|\nabla\mathcal{L}\|}{\|W\|} \propto \frac{\eta}{\|W\|^2}$$

▸ **The effective learning rate is $\eta_{\text{eff}} = \eta/\|W\|^2$.** Weight norm growth is therefore *automatic learning-rate decay*, and **weight decay's real job in a normalized network is not to regularize but to keep $\|W\|$ small so the effective LR stays high.** This resolves a long-standing puzzle: why does weight decay help even when it provably cannot reduce the function class (since the function is scale-invariant)? Because it controls the effective step size. The equilibrium norm where growth and decay balance is
$$\|W\|^4_{\text{eq}} \approx \frac{\eta\,G^2}{2\lambda},\qquad G \equiv \|W\|\cdot\|\nabla_W\mathcal{L}\|\ \ (\text{scale-invariant, so } G \text{ does not depend on } \|W\|)$$
$$\Rightarrow\quad \eta_{\text{eff}} = \frac{\eta}{\|W\|^2} \approx \frac{\sqrt{2\eta\lambda}}{G}\ \propto\ \sqrt{\eta\lambda}$$

*(Derivation: $\|W\|^2$ grows by $\eta^2\|\nabla\mathcal{L}\|^2 = \eta^2G^2/\|W\|^2$ per step and decays by a factor $(1-\eta\lambda)^2 \approx 1-2\eta\lambda$. Setting growth equal to decay gives $2\eta\lambda\|W\|^2 = \eta^2G^2/\|W\|^2$.)*

▸ **So in a normalized network, $\eta$ and $\lambda$ are not independent knobs — only the product $\eta\lambda$ matters** (approximately). This is a  useful, non-obvious, and frequently-asked fact.

**Corollary:** never apply weight decay to LayerNorm/RMSNorm gains or to biases. They are *not* scale-invariant (the gain directly sets the layer's output magnitude), so decaying them shrinks the network's actual output for no benefit.

#### The effective learning rate, decoded

**Why the gradient shrinks as the weights grow.** "Homogeneity" here means: scale the input to a function by $c$ and its derivative scales by $1/c$. Concretely, since $\mathcal{L}(cW) = \mathcal{L}(W)$, differentiating both sides with respect to $W$ and applying the chain rule gives $c\,\nabla_{cW}\mathcal{L} = \nabla_W\mathcal{L}$, hence $\nabla_{cW}\mathcal{L} = \frac1c\nabla_W\mathcal{L}$.

▸ **Doubling the weights halves the gradient.** The loss surface has not changed shape — you have simply moved to a part of parameter space where the same function change requires twice the parameter change, so the slope is half as steep.

**Now assemble the effective learning rate.** Only the *direction* of $W$ affects the function, so the meaningful quantity is how far the direction rotates per step. That is roughly (step size) ÷ (radius):

$$\Delta\angle \approx \frac{\eta\|\nabla\mathcal{L}\|}{\|W\|} \ \propto\ \frac{\eta}{\|W\|^2}$$

The first fraction is the arc-length formula: an arc of length $\eta\|\nabla\mathcal{L}\|$ at radius $\|W\|$ subtends an angle of their ratio. The second step substitutes $\|\nabla\mathcal{L}\| \propto 1/\|W\|$ from just above, and the two factors of $\|W\|$ multiply into a square.

> **Analogy.** You are pushing a point around a circular track, and the push always lands tangentially. **The same-sized push moves you a smaller angle on a bigger track** — and here the push itself also shrinks as the track grows, so the angle you turn falls off as the *square* of the radius. Since only your angular position means anything, a growing radius is indistinguishable from a shrinking learning rate.

**Sit with the number.** $\eta_{\text{eff}} = \eta/\|W\|^2$. Double the weight norm over a training run — which the Pythagoras argument says happens automatically — and **your effective learning rate has quietly fallen by a factor of four**, while the number in your configuration file has not moved.

▸ **Weight norm growth is automatic learning-rate decay.** Nobody scheduled it, nobody logged it, and it is happening in every normalized layer of every network you have ever trained with plain SGD.

**Now weight decay's real job.** In a scale-invariant layer, weight decay provably cannot change the *function class*, because the function does not depend on $\|W\|$ at all. So the textbook justification — "it shrinks the hypothesis space and reduces overfitting" — is unavailable. And yet removing weight decay reliably hurts.

▸ **The resolution: weight decay's job is to fight the norm growth so that $\eta_{\text{eff}}$ stays high.** It is not a regularizer here. **It is a learning-rate controller wearing a regularizer's clothes.** This is one of the  satisfying explanations in deep learning, because it takes a practice everyone follows for a reason everyone can now see is wrong, and supplies a reason that is right.

**Reading the equilibrium.** Two opposing forces act on $\|W\|^2$ each step:

| Force | Size per step | Direction |
|---|---|---|
| Pythagoras growth | $\eta^2\|\nabla\mathcal{L}\|^2 = \eta^2G^2/\|W\|^2$ | up |
| Weight decay | $\approx 2\eta\lambda\|W\|^2$ | down |

Growth is *strong when $\|W\|$ is small*; decay is *strong when $\|W\|$ is large*. So there is exactly one norm where they balance, and the system settles there regardless of where it started. Setting the two equal and solving gives $\|W\|_{\text{eq}}^4 = \eta G^2/(2\lambda)$, and substituting into $\eta_{\text{eff}} = \eta/\|W\|^2$:

$$\eta_{\text{eff}} \approx \frac{\sqrt{2\eta\lambda}}{G} \ \propto\ \sqrt{\eta\lambda}$$

(Here $G \equiv \|W\|\cdot\|\nabla_W\mathcal{L}\|$ is the product of a quantity that grows with $\|W\|$ and one that shrinks with it, so $G$ itself does not depend on $\|W\|$ — which is exactly why it can be treated as a constant while solving.)

▸ **The practical payload: $\eta$ and $\lambda$ are not two knobs. They are one knob, $\eta\lambda$, plus a redundant direction.** Halve the learning rate and double the weight decay and the equilibrium effective learning rate is unchanged. Quadruple the product $\eta\lambda$ and you double $\eta_{\text{eff}}$. **This is why a hyperparameter sweep that grids over $\eta$ and $\lambda$ independently is largely wasting its budget on a diagonal** — most of the grid is the same setting, sampled repeatedly.

**Why the corollary follows immediately.** The whole argument required scale invariance, and scale invariance came from having a normalization layer downstream to cancel the scale. Ask which parameters have that property:

| Parameter | Scale-invariant? | Weight decay? |
|---|---|---|
| Linear/conv weights feeding a norm layer | ✓ (the norm cancels $c$) | Yes — it controls $\eta_{\text{eff}}$ |
| LayerNorm / RMSNorm gain $\gamma$ | ✗ — $\gamma$ is *after* the normalization, so nothing cancels it | **No** |
| Biases, $\beta$ | ✗ — a shift changes the output directly | **No** |

▸ **Decaying $\gamma$ toward zero is instructing the layer to output less.** There is no compensating mechanism, no cancellation, and no benefit — you are simply turning the network's volume down. This is why every serious training script contains a parameter-group split that excludes gains and biases from weight decay, and why forgetting it is a real and quietly costly bug.

> **Where this came from.** The observation that weight decay acts as a learning-rate control in normalized networks was made in a short 2017 note by **Twan van Laarhoven**, and developed by several groups over the following two years — including work by **Elad Hoffer and colleagues**, and by **Zhiyuan Li and Sanjeev Arora**, whose 2019 paper pushed the logic to its striking conclusion: because only the product $\eta\lambda$ matters, a conventional decaying learning-rate schedule can be *exactly* traded for an exponentially **increasing** one, with equivalent training dynamics. It is a good demonstration that the result is not a heuristic but a  equivalence. Note the ordering, which is the recurring shape of this chapter: **weight decay was standard practice for decades before anyone understood what it does inside a normalized network.** The practice was right; the stated reason was not.

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

#### The two penalties, decoded

**Reading the ridge shrinkage formula.** $\hat\theta_i^{\text{ridge}} = \frac{\lambda_i}{\lambda_i + \lambda}\hat\theta_i^{\text{OLS}}$ — and the notation collision here is  nasty, so name it first: **$\lambda_i$ with a subscript is the $i$-th eigenvalue of the Hessian** (how sharply the loss curves in direction $i$), while **$\lambda$ without one is the weight-decay strength you chose.** Two different quantities, one letter, one formula. **OLS** is Ordinary Least Squares — the unregularized solution.

Read the fraction as a **dimmer switch**, one per direction, with a value between 0 and 1:

| Curvature $\lambda_i$ (with $\lambda = 1$) | Fraction kept | Interpretation |
|---|---|---|
| $100$ | $100/101 = 0.99$ | Data pinned this direction down firmly. Keep it |
| $1$ | $1/2 = 0.50$ | Comparable evidence and prior. Split the difference |
| $0.01$ | $0.01/1.01 = 0.0099$ | Data says almost nothing here. Discard it |

▸ **Directions the data determined confidently survive; directions the data barely constrained are erased.** And "curvature" is precisely "how much the loss objects when you move this way" — a flat direction is one the data has no opinion about. **Ridge is not shrinking everything equally; it is shrinking in inverse proportion to evidence,** which is the correct behaviour and the reason it works.

> **Analogy.** A committee vote where each member's influence is weighted by how strongly they feel. Someone who is adamant ($\lambda_i = 100$) is heard almost in full. Someone who is nearly indifferent ($\lambda_i = 0.01$) is overruled by the house default. **Weight decay is the house default, and $\lambda$ is how loudly the house speaks.**

**The Bayesian reading, decoded.** A **prior** is what you believed before seeing data; **MAP (maximum a posteriori)** estimation picks the parameters that best combine prior and data. Saying "$\ell_2$ is a Gaussian prior $\mathcal{N}(0, 1/\lambda)$" means: *"before seeing anything, I expect the weights to be smallish and clustered around zero, in a bell-shaped way."* Taking the negative logarithm of a Gaussian density gives exactly a squared term — which is where $\frac\lambda2\|\theta\|^2$ comes from, and why the penalty is quadratic rather than some other shape. Note that a bigger $\lambda$ means a *narrower* prior (variance $1/\lambda$): stronger belief that weights are small.

**Why $\ell_1$ produces exact zeros, in terms you can say out loud.** The absolute value $|\theta|$ has a **corner** at zero — approach from the left and the slope is $-1$; from the right, $+1$. It has no single slope there. The **subgradient** is the set of all slopes that would work, here the whole interval $[-\lambda, \lambda]$.

▸ **The consequence: as long as the data's pull on a coordinate is weaker than $\lambda$, the penalty can exactly match it, and the coordinate sits pinned at precisely zero.** Not $10^{-6}$ — zero. Contrast $\ell_2$, whose derivative $2\lambda\theta$ goes to zero as $\theta$ does, so the shrinking force fades exactly as you approach the origin and never quite arrives.

**The geometric reading, with a picture in words.** In two dimensions, the $\ell_2$ ball is a circle and the $\ell_1$ ball is a diamond with its corners on the axes. The optimum is where an expanding contour of the data loss first touches the constraint region. A circle has no special points — first contact can be anywhere. A diamond has four sharp corners that stick out furthest, and **a corner on an axis is a solution with a coordinate equal to zero.** In $d$ dimensions the $\ell_1$ ball has $2d$ such corners plus lower-dimensional edges and faces, and almost all of them lie on some axis. **Sparsity is a fact about the shape of the constraint set.**

> **Where this came from.** Ridge regression was published by **Arthur Hoerl and Robert Kennard** in 1970 in *Technometrics*, an applied-statistics journal. Hoerl worked in industry at DuPont, and the problem was thoroughly practical: chemical-process data in which the input variables were strongly correlated, making the least-squares solution numerically unstable and wildly variable. The name reportedly comes from "ridge analysis" in response-surface methodology, where contour plots of the fitted surface show ridge-like structures. The same mathematics had appeared earlier and independently in the Soviet Union, in **Andrey Tikhonov's** work on ill-posed inverse problems from the 1940s onward — which is why the identical idea is called **Tikhonov regularization** in numerical analysis and physics. **LASSO** was named by **Robert Tibshirani** in 1996; he has credited Leo Breiman's *non-negative garrote* as the inspiration, and an $\ell_1$ penalty had already been used for a related purpose in geophysics by **Santosa and Symes** in 1986, for deconvolving seismic traces. ▸ **The pattern is worth noticing: every technique in this catalogue was invented to solve an unstable *estimation* problem in a field with small, messy, expensive data — chemistry, geophysics, inverse problems. None of them were designed for overparameterized networks, and yet they transferred.**

#### Examples and non-examples: what actually shrinks a weight

"Weight decay" names a *behaviour*: at every step, multiply the weight by something slightly less than one. "$\ell_2$ penalty" names a *term in the loss*. With plain SGD they coincide exactly. With anything adaptive they come apart, and the gap is the single most-missed detail in this chapter.

**✅  weight decay — the weight is multiplied by $(1 - \eta\lambda)$ each step**

| Example | Why it qualifies |
|---|---|
| `torch.optim.SGD(params, lr=0.1, weight_decay=1e-4)` | SGD adds $\lambda\theta$ to the gradient and then multiplies by $\eta$; the update is $\theta \leftarrow (1-\eta\lambda)\theta - \eta g$. Multiplicative shrinkage, identical for every coordinate |
| `torch.optim.AdamW(params, weight_decay=0.01)` | The decay is applied *outside* the adaptive rescaling — literally $\theta \leftarrow \theta - \eta\lambda\theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ |
| Adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss and optimizing with vanilla SGD | Differentiates to $\lambda\theta$; see the first row. **This is the case where the two names are  the same thing** |
| A hard projection back onto the ball $\|\theta\| \le c$ after each step | Not identical, but it is unambiguously a constraint on weight magnitude with no coordinate-dependent distortion |

**❌ Near-misses — shrink weights, but not in the way you specified**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| `torch.optim.Adam(params, weight_decay=1e-4)` | Adam adds $\lambda\theta$ to the gradient, then divides the whole thing by $\sqrt{\hat v}$. The effective decay on coordinate $i$ becomes $\eta\lambda\theta_i / \sqrt{\hat v_i}$ — **inversely proportional to that coordinate's gradient scale** | An $\ell_2$ penalty passed through an adaptive preconditioner. Not weight decay, despite the argument name |
| Weight decay applied to LayerNorm's $\gamma$ | $\gamma$ sits *after* the normalization, so no scale invariance cancels it. Shrinking it just turns the layer's output down | A capacity reduction you did not intend (§7.4) |
| Early stopping at 30 epochs | Never touches the loss and never multiplies a weight by anything | An implicit $\ell_2$ with strength $\approx 1/(\eta t)$ — the same *effect*, a completely different *mechanism* |
| Gradient clipping | Bounds the size of the *update*; a weight can still drift arbitrarily far given enough steps | A stability control, not a capacity control |
| Adding $\frac{\lambda}{2}\|\theta\|^2$ to a network whose next layer is a BatchNorm | Shrinking those weights changes nothing about the function — BN cancels the scale | A **learning-rate control** in disguise (§7.4), which is the actual reason it helps |

**Put a number on the Adam/AdamW gap.** Take two weights, both currently $\theta = 0.5$, with $\lambda = 10^{-2}$ and $\eta = 10^{-3}$. Weight A sits in a direction with RMS gradient $\sqrt{\hat v_A} = 1.0$; weight B in a quiet direction with $\sqrt{\hat v_B} = 0.01$.

| | Decoupled (AdamW) | Coupled ($\ell_2$ into Adam's gradient) |
|---|---|---|
| Shrinkage on A | $\eta\lambda\theta = 5\times10^{-6}$ | $\eta\lambda\theta/1.0 = 5\times10^{-6}$ |
| Shrinkage on B | $\eta\lambda\theta = 5\times10^{-6}$ | $\eta\lambda\theta/0.01 = 5\times10^{-4}$ |
| Ratio B:A | $1{:}1$ | $\mathbf{100{:}1}$ |

▸ **The boundary:** weight decay is *multiplicative shrinkage applied uniformly*; an $\ell_2$ penalty is *an additive gradient term that the optimizer is then free to distort*. They are the same operation only when the optimizer applies a single global scalar to the gradient — which SGD does and Adam, by construction, does not.

> **Common misconception.** *"Weight decay and $\ell_2$ regularization are two names for the same thing."* They are two names for the same thing **only under SGD**. Under Adam the $\ell_2$ term is fed into the gradient and then divided by $\sqrt{\hat v}$ along with everything else, so coordinates that happen to receive small gradients get their weights shrunk up to a hundred times harder than coordinates that receive large ones — a regularization strength determined by gradient statistics rather than by you. The belief is tempting because the identity really is exact for SGD, it is how every textbook introduces the topic, and PyTorch spells the argument `weight_decay=` in **both** optimizers, which actively invites the confusion. Loshchilov and Hutter's 2017 paper exists solely to separate them, and the "W" in AdamW stands for exactly this fix. ▸ **If you are using Adam and you meant to regularize, use AdamW.**

> **Common misconception.** *"More regularization is safer — if in doubt, turn it up."* Regularization is a bias–variance trade, and both ends of that trade cost you. A model that underfits because $\lambda$ was too large produces a training curve that looks *calm and healthy* — smooth loss, no overfitting gap, no instability — while quietly leaving accuracy on the table. Overfitting announces itself with a widening train/validation gap; over-regularization announces nothing at all. That asymmetry in *visibility*, not in cost, is why the belief takes hold. The honest diagnostic is the training loss itself: **if you are not able to overfit your training set when you deliberately try, your regularization is too strong, not your model too small.**

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

#### Dropout, decoded

**Reading the formula.** $\tilde h = h \odot m/(1-p)$ with $m_i \sim \mathrm{Bernoulli}(1-p)$:

- $m$ is a **mask** — a vector of independent coin flips, each landing 1 (keep) with probability $1-p$ and 0 (drop) with probability $p$. **Note the direction: $p$ is the probability of being *dropped*.** With $p = 0.5$, half the units are zeroed on any given forward pass, and a different half next time.
- $\odot$ multiplies entry by entry, so $\tilde h_i$ is either $h_i$ (scaled) or exactly zero.
- The tilde marks a perturbed version of $h$ (Chapter 0 §0.6).

**Why divide by $1-p$.** Without it, the expected activation would be $\mathbb{E}[h_i m_i] = h_i(1-p)$ — systematically smaller than $h_i$, so the next layer would receive a quieter signal in training than in evaluation, and every layer's learned scale would be wrong at test time. Dividing by $1-p$ restores $\mathbb{E}[\tilde h] = h$ exactly. With $p = 0.5$ the surviving units are simply **doubled**. ▸ **"Inverted dropout" means the correction is applied during training rather than at inference, so inference is the identity — no flag to remember, no rescaling to forget.** Nearly every framework does this; the alternative convention is a well-known source of bugs when porting old code.

> **Analogy.** A workshop where, every morning, half the staff are randomly told to stay home — and the remaining half must still ship the day's work. Nobody can build a process that depends on one specific irreplaceable colleague, because that colleague is absent half the time. Everyone ends up broadly capable and mutually redundant. **Dropout prevents co-adaptation: it forbids a unit from relying on the presence of any particular other unit.** The division by $1-p$ is the instruction that the half who show up must work twice as hard, so total output stays constant.

**Reading the three interpretations.**

**1. Ensembling, with a number.** A network with $n$ droppable units has $2^n$ possible masks, hence $2^n$ sub-networks that share weights. With a modest $n = 1000$, that is $2^{1000} \approx 10^{301}$ sub-networks — for comparison, the observable universe holds roughly $10^{80}$ atoms. Training visits an utterly negligible fraction of them, but because they share weights, improving one improves enormous numbers of others. At inference, using the full network with the $1/(1-p)$ scaling approximates the **geometric mean** of all of them (the geometric mean, not the arithmetic one, because probabilities multiply). For a single-layer linear-softmax model this approximation is exactly right; for deep networks it is an approximation that works better than it has any right to.

**2. Explicit regularization, decoded.** The identity shown says that for linear regression, dropout on the inputs is *exactly* $\ell_2$ regularization with a per-feature strength of $\frac{p}{1-p}\mathbb{E}[x_j^2]$. Read the coefficient: features with **large typical magnitude get penalized more.** That is a meaningful difference from ordinary weight decay, which penalizes every weight identically regardless of what it multiplies. ▸ **Dropout is data-dependent $\ell_2$** — it knows something about your inputs that plain weight decay does not. At $p = 0.5$ the factor $p/(1-p) = 1$; at $p = 0.1$ it is $0.11$, so the regularization strength rises steeply with $p$ and is not remotely linear in it.

**3. Approximate Bayesian inference, in one paragraph.** If dropout training is a form of variational inference, then the randomness is not a training trick to be discarded — it is a posterior to be sampled. So: leave dropout **on** at test time, run $T$ forward passes, and treat the spread of the predictions as an uncertainty estimate. That is **MC dropout (Monte Carlo dropout)**, and it is one of the cheapest usable uncertainty methods available: no retraining, no ensemble to store, just $T$ forward passes.

**Reading the variants.** Each fixes a case where per-element dropout does too little:

| Variant | Drops | Why plain dropout fails there |
|---|---|---|
| Spatial / 2D dropout | An entire channel | Adjacent pixels are highly correlated, so zeroing individual pixels removes almost no information — the neighbours give it straight back |
| DropConnect | Individual weights | A finer-grained version, more sub-networks, rarely worth the cost |
| DropPath / stochastic depth | A whole residual block | Makes the network's *depth* random; $p_\ell$ rises linearly with depth so early layers are almost always present |
| Attention dropout | Individual attention weights | Prevents over-reliance on one token position |

▸ **Why dropout disappeared from large-scale pretraining is worth understanding rather than memorizing.** Dropout combats **overfitting**, which is what happens when a model sees the same examples repeatedly and begins memorizing them. A frontier language model makes roughly one pass over a corpus it will never see again — **there is nothing to memorize, because nothing repeats.** Meanwhile dropout's cost is real: it injects noise into every gradient and effectively reduces capacity. When the disease is absent, the treatment is pure side effect. It remains standard in fine-tuning and small-data regimes, where examples  do repeat. **The general lesson: a regularizer is not a virtue, it is a trade, and the trade depends on the data-to-parameter ratio.**

> **Where this came from.** Dropout was introduced by **Geoffrey Hinton and collaborators** in 2012, with the full treatment by **Nitish Srivastava, Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov** in 2014. Hinton has described two sources of inspiration in talks over the years. The first was noticing that bank tellers were frequently rotated between positions, and speculating that this was to make it hard for any stable group of employees to co-operate on a fraud — the same logic, applied to neurons conspiring to co-adapt. The second was **sexual reproduction**: genes must work with a randomly chosen half of a partner's genome, which penalizes fragile gene-complexes and rewards robust individually-useful ones. (These are recollections offered in talks rather than a documented design process, so treat them as illustration rather than history.) One firmly documented and frequently surprising fact: **Google was granted a US patent covering dropout in 2016** — a reminder that a technique which now appears as a one-line default in every framework was, briefly, intellectual property.

#### Examples and non-examples: is that dropout?

Dropout has a tight definition: **zero out units at random, independently, with a fixed probability, resampling the mask every forward pass, during training only, with a compensating rescale.** Drop any clause and you have a different technique.

**✅  dropout and its family**

| Example | Why it qualifies |
|---|---|
| `nn.Dropout(0.1)` in a transformer feed-forward block, model in `train()` | Independent Bernoulli mask, fresh each step, scaled by $1/0.9 = 1.11$, identity at eval |
| `nn.Dropout2d(0.2)` zeroing whole conv channels | Same rule; the *unit* is a channel rather than a scalar, because neighbouring pixels are correlated |
| Stochastic depth skipping residual block 14 with $p_{14}=0.15$ | Bernoulli, resampled per step, disabled at eval; the *unit* is a whole block |
| Attention dropout zeroing individual attention weights | Same rule applied to the post-softmax matrix |
| DropConnect zeroing individual entries of $W$ | Same rule applied to weights instead of activations |
| Leaving dropout **on** at test time and averaging 30 forward passes | Deliberate MC dropout (Ch. 33). Same operation, reinterpreted as posterior sampling — and you must *choose* it explicitly |

**❌ Near-misses — zero things out, but aren't dropout**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| ReLU setting negative pre-activations to 0 | Deterministic and input-dependent — the same input always zeroes the same units, and there is no rescale | A **nonlinearity**. Roughly half the units are zero, which is why it gets confused for dropout |
| The causal mask in a decoder setting future positions to $-\infty$ | Fixed, known, identical every step, and active at inference | An **architectural constraint** enforcing autoregressive structure (Ch. 13) |
| Padding masks zeroing positions past a sequence's end | Data-dependent, deterministic, active at inference | **Bookkeeping** — those positions are not data |
| Masking 15% of tokens for BERT-style pretraining | It defines the *training objective*; the model is asked to predict what was masked | **Masked language modelling** — a task, not a regularizer |
| Magnitude pruning that sets 90% of weights to zero and keeps them there | Permanent, chosen by magnitude rather than by coin flip, and the zeros persist at inference | **Sparsification / pruning** (Ch. 17) |
| Cutout / random erasing a $16\times16$ patch of the input image | Applied in *input* space by the data pipeline, before the model sees anything | **Data augmentation.** The closest  relative — but it is not inside the network |
| Dropping units and *not* dividing by $1-p$ | Expected activations shrink by a factor $1-p$ per layer; over 12 layers at $p=0.1$ that is $0.9^{12} = 0.28$ | A bug — or "classic" dropout, which then requires multiplying by $1-p$ at inference instead |

▸ **The boundary:** dropout's mask is **random, independent of the data, resampled every step, and switched off at inference.** Anything deterministic is architecture; anything data-dependent is masking; anything permanent is pruning; anything in input space is augmentation.

> **Common misconception.** *"Dropout is applied at inference too — that's how the network stays regularized."* Regularization happens during *training*; it shapes which weights you end up with. At inference the standard behaviour is the identity function: `nn.Dropout` in `eval()` mode returns its input unchanged. Leaving it on would make your model's prediction for a fixed input **different on every call**, which is exactly what you do not want from a classifier. The belief is tempting because dropout is usually described as "ensembling $2^n$ sub-networks," and ensembles obviously operate at prediction time — so it feels as though the sampling must continue. It does not: the $1/(1-p)$ scaling applied during training is precisely what makes the *single* full network at inference approximate the whole ensemble's geometric mean in one pass. The one exception is MC dropout, where you deliberately re-enable it to *get* the stochasticity — and there you run $T$ passes and average, because a single stochastic pass is not a prediction, it is a sample.

> **Common misconception.** *"`nn.Dropout(0.8)` keeps 80% of the units."* It drops 80% and keeps 20%, and multiplies the survivors by $1/(1-0.8) = 5$. PyTorch, TensorFlow's `tf.nn.dropout` since TF 2.0, and this book all use $p$ = probability of being **dropped** — but the original 2014 paper's notation, TensorFlow 1.x's `keep_prob` argument, and a great deal of older code use the opposite convention. The belief is tempting because "0.8" reads like a retention rate in almost every other context (a pass rate, a survival rate, a keep-alive). ▸ **When porting code across framework generations, check this argument first** — the failure is silent, the model still trains, and it simply trains badly.

### Label smoothing

▸ $$y^{\text{LS}} = (1-\alpha)y_{\text{one-hot}} + \frac{\alpha}{K}\mathbf{1},\qquad \alpha\approx0.1$$

Prevents the logit gap from diverging: the optimal logit gap becomes finite, $\log\frac{(1-\alpha)(K-1)}{\alpha} + \text{const}$, instead of $\infty$.

**Effects:** better calibration (Ch. 33), better generalization, **tighter clustering of penultimate-layer features** — Müller et al. showed label smoothing induces equidistant class clusters, which is a deliberate acceleration of *neural collapse* (Ch. 31).

**Cost:** it destroys information useful for knowledge distillation. A teacher trained with label smoothing gives worse students, because the relative "dark knowledge" among wrong classes is erased.

#### Label smoothing, decoded

**Reading the formula.** $y^{\text{LS}} = (1-\alpha)y_{\text{one-hot}} + \frac\alpha K\mathbf{1}$ says: *"take 90% of the confident answer and spread the other 10% evenly over all classes."* Here $\mathbf{1}$ is the vector of all ones and $K$ is the number of classes, so $\frac\alpha K\mathbf{1}$ is a flat sliver given to everyone including the correct answer.

**With numbers, $K = 1000$ and $\alpha = 0.1$:**

| Class | Original target | Smoothed target |
|---|---|---|
| The correct one | $1.0$ | $0.9 + 0.1/1000 = 0.9001$ |
| Each of the 999 others | $0.0$ | $0.1/1000 = 0.0001$ |

**Why this fixes a real pathology.** Cross-entropy asks the model to push the correct class's probability to 1. But softmax reaches 1 only in the limit of an **infinite** logit gap, so a model trained to convergence on hard targets is being asked to chase a target it can never reach — and it responds by growing its logits without bound, forever, long after its *decisions* have stopped changing. With smoothing, the optimal gap becomes finite:

$$\log\frac{(1-\alpha)(K-1)}{\alpha} = \log\frac{0.9 \times 999}{0.1} = \log(8991) \approx 9.1$$

▸ **A finite, computable target instead of infinity.** The model reaches a gap of about 9.1 and stops. This is the entire mechanism, and it explains the effects: **better calibration** (the model's confidence is no longer artificially inflated by a runaway objective) and **tighter feature clusters** (there is now a specific configuration to converge to, rather than a direction to run in indefinitely).

> **Analogy.** An exam where a perfect score is impossible by design — the best available answer is worth 90%, and the remaining 10% is distributed as participation credit. A student aiming for 100% will study forever and become increasingly, unjustifiably certain. A student aiming for 90% studies until they get there and stops, and their stated confidence ends up matching their actual accuracy. **Label smoothing is telling the model that certainty is not on the menu.**

**Reading the cost, which is the interesting part.** **Knowledge distillation** trains a small student model on a large teacher's full probability *distribution* rather than on the hard labels. Its power comes from what practitioners call **dark knowledge**: an image of a Bedlington terrier might get $0.85$ dog, $0.10$ sheep, $0.02$ cat. That "sheep" is real information about visual similarity — the sort of thing a hard label can never convey.

Label smoothing overwrites exactly that. It forces all $K-1$ wrong classes toward the *same* value $\alpha/K$, flattening the meaningful structure among them. ▸ **The teacher is better calibrated and simultaneously a worse teacher**, because the property being improved and the property being destroyed are the same property: how much the wrong classes differ from each other. If you plan to distil, do not smooth.

> **Where this came from.** Label smoothing was introduced by **Christian Szegedy and colleagues** at Google in 2016, in the paper that presented Inception-v3 — where it appeared as one item in a list of engineering refinements, worth a few tenths of a percent, with a brief and somewhat informal justification. It took until 2019 for **Rafael Müller, Simon Kornblith, and Geoffrey Hinton** to study it properly and produce both the geometric picture (equidistant, tightly clustered penultimate-layer features) and the distillation warning. **A one-paragraph trick in an architecture paper turned out to have a specific geometric effect on representations and a specific incompatibility with a major technique, neither of which anyone noticed for three years.**

### Data augmentation and mixup

Augmentation is the highest-value regularizer in vision and audio, and it works by enlarging the effective support of $p_{\text{data}}$ along directions you know are label-preserving. It is an **encoding of invariances**, i.e. a prior.

**Mixup:** $\tilde x = \lambda x_i + (1-\lambda)x_j$, $\tilde y = \lambda y_i+(1-\lambda)y_j$, $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$.
Forces linear behaviour between examples. Improves calibration and adversarial robustness. **CutMix** pastes a patch instead of blending, preserving local statistics; usually better for classification.

#### Augmentation and mixup, decoded

**What "encoding an invariance" means.** You know things about your problem that your dataset does not contain. A photograph of a cat, flipped horizontally, is still a cat. Rotated five degrees, still a cat. Brightened, still a cat. Your training set may contain none of those variants — so you manufacture them, and in doing so you tell the network a fact about the world that no amount of data-fitting would supply.

▸ **This is why augmentation is a *prior*, in exactly the Bayesian sense.** It is knowledge injected before the data speaks. And it is why the choice must be domain-correct: horizontal flips are label-preserving for cats and **destructive for handwritten digits**, where a flipped 2 is not a 2 and a flipped 6 is arguably a 9. **An augmentation that is not label-preserving is not a regularizer, it is label noise you have added on purpose.**

**Reading mixup.** $\tilde x = \lambda x_i + (1-\lambda)x_j$ and $\tilde y = \lambda y_i + (1-\lambda)y_j$ — *"blend two training images, and blend their labels by the same amount."* With $\lambda = 0.7$ you get an image that is 70% cat and 30% dog, and a target that says 70% cat and 30% dog. The image is visually nonsense — a semi-transparent cat superimposed on a dog — and that turns out not to matter.

**What $\mathrm{Beta}(\alpha,\alpha)$ is doing.** The Beta distribution lives on $[0,1]$, which is exactly the range $\lambda$ needs, and the parameter $\alpha$ controls its shape:

| $\alpha$ | Shape of the distribution | Effect |
|---|---|---|
| $0.2$ | U-shaped — mass piled near 0 and 1 | Most pairs are barely mixed; occasional heavy mixes |
| $1.0$ | Flat (uniform) | Every blend ratio equally likely |
| Large | Peaked at $0.5$ | Nearly every pair is a 50/50 blend |

Common values are $\alpha \in [0.1, 0.4]$, i.e. the U-shape — **mostly-clean examples with a light dusting of mixing**, not a training set of blurry composites.

▸ **What mixup actually enforces:** *"between any two training points, the model's output should move in a straight line."* Neural networks left to themselves behave erratically off the data manifold — confidently predicting nonsense in the gaps between training examples, which is precisely where adversarial examples live (Ch. 1 §1.1.4). Mixup fills those gaps with a stated expectation. **That is why it improves calibration and adversarial robustness together: both are symptoms of wild behaviour between data points, and mixup is a direct instruction about what should happen there.**

> **Analogy.** Teaching someone to judge temperature by showing them only ice water and boiling water is unreliable — they may form a bizarre theory about everything in between. Also showing them mixtures, and telling them the mixture ratio, forces the sensible interpolation. **Mixup does not add information about cats or dogs; it adds information about what the space *between* them should look like.**

**Why CutMix often works better.** Blending two images produces pixel statistics — ghostly semi-transparency, halved contrast — that never occur in real photographs, so some of what the network learns is about an artefact of the augmentation. CutMix instead **cuts a rectangular patch from one image and pastes it into the other**, mixing the labels in proportion to the patch's area. Every pixel remains a real pixel from a real photograph; only the *composition* is synthetic. This also mimics  occlusion, which is a thing that actually happens in the world.

> **Where this came from.** **Mixup** was introduced by Hongyi Zhang, Moustapha Cisse, Yann Dauphin, and David Lopez-Paz in 2018. It is a notably good example of a result that sounds like it should not work — averaging two unrelated images and their labels is not an obviously sensible operation, and the resulting inputs are not members of the data distribution by any reasonable definition. It works consistently anyway, across datasets and architectures. **CutMix** followed in 2019 from a team at Naver, combining mixup's label interpolation with the older idea of Cutout (masking a random region). ▸ **Data augmentation as a whole remains the highest-value regularizer in vision and audio and one of the lowest-status research areas** — a persistent mismatch between what works and what gets attention, and one worth remembering when deciding where to spend your own effort on a real problem.

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

#### Early stopping, decoded — why it is secretly ridge regression

**Reading $\theta_i(t) = (1 - (1-\eta\lambda_i)^t)\theta_i^*$.** The setting: a quadratic loss (the local picture near any minimum), gradient descent started from $\theta_0 = 0$, viewed one Hessian eigendirection at a time. Here $\theta_i^*$ is where direction $i$ would eventually land, $\lambda_i$ is that direction's curvature, and $t$ is the step count.

The factor $(1 - (1-\eta\lambda_i)^t)$ is a **fraction of the way there**, running from 0 at $t=0$ toward 1 as $t$ grows. Its rate is set by $\eta\lambda_i$ — **the product of the step size and the curvature.**

▸ **So different directions converge at wildly different speeds, and the speed is proportional to the curvature.** Sharply-curved directions arrive almost immediately; nearly-flat directions crawl. Put numbers on it with $\eta = 0.1$:

| Curvature $\lambda_i$ | $\eta\lambda_i$ | Fraction of the way there after 100 steps |
|---|---|---|
| $1.0$ | $0.1$ | $1 - 0.9^{100} = 0.99997$ |
| $0.1$ | $0.01$ | $1 - 0.99^{100} = 0.63$ |
| $0.001$ | $0.0001$ | $1 - 0.9999^{100} = 0.0100$ |

**Now compare the two shrinkage factors.** Write $u = \eta\lambda_i t$, a dimensionless "how much progress has this direction had" number:

| Method | Fraction of $\theta^*_i$ retained |
|---|---|
| Early stopping | $1 - e^{-u}$ (for small $\eta\lambda_i$) |
| Ridge, with $\lambda = 1/(\eta t)$ | $u/(1+u)$ |

At $u = 0.1$: $0.095$ versus $0.091$. At $u = 1$: $0.63$ versus $0.50$. At $u = 10$: essentially 1 versus $0.91$. **Both start out equal to $u$ and both approach 1; the curves have the same shape and cross at moderate values.** They are the same operation described in two different vocabularies.

▸ **Early stopping is $\ell_2$ regularization with $\lambda = 1/(\eta t)$, and stopping earlier is regularizing harder.** Concretely, at $\eta = 10^{-3}$: stopping at 10,000 steps corresponds to $\lambda = 0.1$; running to 100,000 steps corresponds to $\lambda = 0.01$, ten times weaker. **Training length is a regularization hyperparameter, whether or not you were treating it as one.**

> **Analogy.** Developing a photograph in a chemical bath. High-contrast areas (high curvature) appear almost immediately; faint detail (low curvature) emerges only after a long soak. Pull the print early and you keep the bold structure and lose the faint detail — which is exactly what ridge does, keeping well-determined directions and discarding poorly-determined ones. **The stopwatch and the chemical concentration are two controls on the same outcome**, and this is why $\lambda \approx 1/(\eta t)$ has a $t$ in it.

**Why the ordering is not a coincidence.** Gradient descent from the origin explores directions in decreasing order of curvature, and curvature is precisely "how firmly the data constrains this direction." So the directions that arrive first are the well-evidenced ones and the stragglers are the noise-driven ones. **Stopping early is a way of only accepting conclusions the data reached quickly.** The connection is old: essentially the same result appears in the inverse-problems literature under the name **iterative regularization**, where truncating an iterative solver is a standard alternative to adding an explicit penalty term.

▸ **Practical consequence you can act on today:** if you use early stopping *and* weight decay *and* a learning-rate schedule, you are setting the same underlying quantity three times, through three interfaces, and they interact. This is a large part of why hyperparameter searches in deep learning have so much redundancy in them — and, together with §7.4's $\eta\lambda$ result, why a sweep that treats every hyperparameter as independent spends most of its budget re-sampling the same effective configuration.

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

#### Reading the decision table

Every row is a consequence of something derived above, and it is worth being able to name which:

| Row | The reason, in one line |
|---|---|
| CNN, batch ≥ 32 → BatchNorm | The batch is large enough for reliable statistics ($\approx25\%$ error at $B{=}32$), and the batch noise is a useful free regularizer |
| CNN, batch < 16 → GroupNorm | Below that, you are dividing by a random number (§7.2). GroupNorm's statistics do not span the batch, so batch size is irrelevant to them |
| Transformer → Pre-LN RMSNorm | Pre-LN for the clean gradient path; RMSNorm because re-centring was measured and found not to matter |
| >100 layers → DeepNorm | Pre-LN's residual stream grows like $\sqrt L$, so very deep models need the growth cancelled explicitly rather than tolerated |
| Small dataset → augmentation ≫ dropout > weight decay | Augmentation adds *knowledge* (invariances you know are true); the others only add *constraint*. Adding knowledge beats adding constraint whenever you have knowledge to add |
| LLM pretraining → weight decay only, dropout 0 | A single pass means nothing repeats, so there is nothing to overfit. Weight decay stays — not to regularize, but as the effective-learning-rate control from §7.4 |
| Fine-tuning → dropout 0.1, low LR, early stop | Now examples *do* repeat and the dataset is small, so overfitting is real again. All three of these are the same lever (§7.5) applied through three interfaces |
| Need calibration → label smoothing | It caps the logit gap at a finite value, which stops confidence from inflating past accuracy |

▸ **The unifying question behind the whole table is one ratio: how many times will the model see each example?** Once (frontier pretraining) → regularization is nearly pointless and mostly harmful. Hundreds of times (fine-tuning, small vision datasets) → regularization is the difference between a usable model and a memorized one. **Everything else is detail.**

#### Examples and non-examples: is that a regularizer?

A **regularizer** biases *which* solution the training procedure lands on, among the many that fit the training data equally well. That is a statement about the *selection*, not about the model's capacity and not about the optimizer's stability — and the three get conflated constantly.

**✅  regularizers**

| Example | What it biases the solution toward |
|---|---|
| Weight decay $\lambda = 0.1$ on a ResNet's conv weights | Small-norm solutions; shrinks low-curvature directions hardest (§7.5) |
| Dropout $p = 0.1$ in a fine-tuning run | Solutions with no unit that any other unit depends on |
| Label smoothing $\alpha = 0.1$ | Solutions with bounded logit gaps — capped confidence |
| Random horizontal flips on CIFAR-10 | Solutions that are invariant to left–right mirroring. **Adds knowledge**, not just constraint |
| Mixup with $\alpha = 0.2$ | Solutions that behave linearly between training points |
| Stopping at epoch 30 instead of 300 | High-curvature (well-evidenced) directions only — equivalent to ridge with $\lambda = 1/(\eta t)$ |
| SGD's minibatch noise itself | Flatter minima. Implicit — nobody typed it, and it is doing real work |

**❌ Near-misses — improve generalization, but are not regularizers**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Collecting 10× more training data | The set of solutions that fit the data *shrinks* — you removed candidates rather than choosing among them | More **evidence**. Strictly better, and always the first thing to try |
| Using a 3-layer model instead of a 30-layer one | The good solutions no longer exist to be selected from | **Capacity reduction** |
| Gradient clipping at norm 1.0 | Stops a single bad batch from destroying the run; at convergence it is inactive and biases nothing | **Optimization stability** |
| A cosine learning-rate schedule | Governs *how fast* you travel, not *where you settle* — except through the $\eta\lambda$ coupling of §7.4 | A **schedule**. Its regularizing effect is an indirect side channel |
| Averaging 5 independently trained models | Happens entirely at inference; each member was trained exactly as before | **Ensembling** — variance reduction at prediction time |
| BatchNorm | Its regularizing effect (batch noise) is a *by-product* of a layer added for conditioning — and it vanishes as batch size grows | A **normalization layer** that happens to leak noise |
| Freezing the first 20 layers during fine-tuning | Certain solutions become unreachable rather than merely disfavoured | A **hard constraint** on the parameter space |
| Adding random noise to 10% of your labels | Actively destroys evidence; the model fits a worse target | **Label noise.** It sometimes helps, which is why it gets miscategorized |

▸ **The boundary:** a regularizer leaves the set of representable functions intact and changes which one the training run prefers. **If it removed candidates from the hypothesis space, it is a constraint; if it changed how you got there, it is optimization; if it changed the data, it is evidence or corruption.** Weight decay, dropout, and augmentation are the only rows above that leave the model equally expressive and simply express a preference.

---

## Did you know?

- **BatchNorm's stated explanation was overturned three years and tens of thousands of citations after publication.** "Internal covariate shift" was the story from 2015 to 2018. The decisive test was to *deliberately inject* distribution shift right after each BatchNorm layer — training was essentially unaffected. The technique survived; the explanation did not.

- **LayerNorm was invented for recurrent networks, and the transformer did not exist yet.** Ba, Kiros, and Hinton published it in 2016 specifically because BatchNorm fails when sequence lengths vary. The Transformer arrived the following year. The normalization now running inside essentially every large language model was designed for the architecture it would help replace.

- **InstanceNorm came out of artistic style transfer.** The insight was that an image's overall contrast and brightness are exactly what you want to *discard* before restyling it — so normalize them away per image, per channel. A technique from a visual-effects problem became a standard building block.

- **In a normalized network, the weight norm can only ever grow under plain SGD.** Because the gradient is exactly perpendicular to the weights, each step is the leg of a right triangle and the new norm is the hypotenuse: $\|W_{t+1}\|^2 = \|W_t\|^2 + \eta^2\|\nabla\mathcal{L}\|^2$. There is no negative term available. The optimizer inflates the weights while trying only to rotate them.

- **Weight decay in a normalized network is not a regularizer.** It provably cannot restrict the function class, because the function does not depend on the weight norm at all. What it actually does is hold the norm down so the *effective* learning rate $\eta/\|W\|^2$ stays high. A practice followed for decades under one justification turned out to work for an entirely different reason.

- **Your learning rate and your weight decay are one hyperparameter, not two.** At equilibrium the effective learning rate depends only on the product $\eta\lambda$. Halve one and double the other and you have changed nothing — which means a grid search over both independently spends most of its budget re-testing the same configuration along a diagonal.

- **Every LayerNorm output has a length of exactly $\sqrt d$.** For $d = 768$, that is $27.7$ — not approximately, but exactly, for every token, on every layer, on every input, forever. LayerNorm squeezes all of $\mathbb{R}^d$ onto a sphere of dimension $d-2$.

- **Dropout defines more sub-networks than there are atoms in the observable universe.** With 1,000 droppable units there are $2^{1000} \approx 10^{301}$ masks; the observable universe holds roughly $10^{80}$ atoms. Training samples one per step, forever, and never repeats.

- **Google was granted a patent on dropout in 2016.** A technique that today appears as a one-line default in every deep learning framework was, for a period, formally claimed intellectual property.

- **Label smoothing makes a model a better predictor and a worse teacher.** By flattening all the wrong classes toward one shared value, it destroys the relative structure among them — the "dark knowledge" that knowledge distillation depends on. The property being improved and the property being destroyed are the same property.

- **Early stopping is ridge regression with $\lambda = 1/(\eta t)$.** Training length is a regularization strength, exactly and quantitatively, at least for quadratic losses. Stopping at 10,000 steps regularizes ten times more strongly than stopping at 100,000.

- **Ridge regression was invented for chemical process data, and LASSO's $\ell_1$ penalty appeared first in geophysics.** Hoerl and Kennard published ridge in 1970 in an applied statistics journal, dealing with correlated industrial measurements; an $\ell_1$ penalty was used for deconvolving seismic traces in 1986, a decade before Tibshirani named the LASSO. The same mathematics had already appeared in Tikhonov's Soviet work on ill-posed inverse problems from the 1940s.

- **Warmup existed for three years before anyone explained what it was hiding.** Post-LN transformers have a gradient imbalance that grows like $\sqrt L$ with depth; warmup conceals it by taking tiny steps until the network settles. Pre-LN removes the imbalance and often removes the need for warmup with it. **The field's standard response to an architectural instability was a hyperparameter schedule.**

---

## Check for Understanding

**Normalization's most important effect is not fixing covariate shift but making the layer scale-invariant, which turns the weight norm into an inverse effective learning rate — so in a normalized network, weight decay is a learning-rate control and only the product $\eta\lambda$ matters.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What does normalization actually do for optimization, if not fix covariate shift?** (Answer in terms of fog and terrain, not Lipschitz constants.)
2. **Why do $\gamma$ and $\beta$ not simply undo the normalization and make the whole thing pointless?**
3. **Why is BatchNorm's backward pass harder than every other layer's?** What two directions does it delete from the gradient, and why are those two specifically?
4. **Why does BatchNorm break at batch size 2?** Give the statistical reason, not "it's noisy."
5. **What is the single structural difference between BatchNorm and every other member of the normalization family**, and how does it explain the train/eval gap, the padding problem, and the contrastive-learning leak all at once?
6. **Why did transformers end up using LayerNorm?** (Two reasons: sequence length, and the train/eval discrepancy.)
7. **Why can multiplying a layer's weights by 100 leave its output completely unchanged?** And what does that imply about what the optimizer is really doing to those weights?
8. **Why does the weight norm only ever grow under plain SGD?** (The answer is Pythagoras.)
9. **Why does weight decay help in a normalized network even though it cannot change what functions the network can express?**
10. **Why should you never apply weight decay to LayerNorm gains or biases?**
11. **Why does $\ell_1$ produce exact zeros while $\ell_2$ only produces small numbers?** Explain it with the shape of the penalty, then with the shape of the ball.
12. **Why did dropout disappear from large-scale language model pretraining, and why is it still standard for fine-tuning?**
13. **In what sense is early stopping the same thing as weight decay?**
14. **Why does label smoothing improve calibration but ruin a model as a distillation teacher?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 08 — Convolutions, ResNets & Vision Architectures](08-convolutions-resnets-vision.md)
