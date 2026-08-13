# Chapter 4 — Optimization I: Gradient Descent

> **Prerequisites:** Ch. 1 (matrix calculus, eigendecomposition).
> **Goal:** by the end you should be able to say, for any optimizer, *what quantity it is estimating and what its convergence rate depends on* — and to explain why the condition number is the villain in every story.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, $`A^\top`$, or $`\mathcal{O}(\cdot)`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. This chapter needs three things from it: $`\nabla`$ is "which way is uphill," $`\mathbb{E}`$ is "the average of," and $`\theta_{t+1} = \theta_t - \dots`$ is a *line of code*, not an equation to solve.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`\theta_t`$ | "theta at step $`t`$" | All the model's parameters, stacked into one long vector, after $`t`$ updates |
| $`\eta`$ | "eta" | The **learning rate** — how big a step you take. Also called the step size |
| $`\nabla\mathcal{L}(\theta)`$ | "grad L of theta" | The direction of steepest **increase** of the loss. You always move against it |
| $`g_t`$ | "g at $`t`$" | Shorthand for the gradient actually used at step $`t`$ — often a noisy minibatch estimate |
| $`\theta^*,\ \mathcal{L}^*`$ | "theta-star, L-star" | The best parameters, and the loss they achieve. The target |
| $`L`$ | "the smoothness constant" | A **ceiling** on curvature: how sharply the loss can bend anywhere |
| $`\mu`$ | "mu" | A **floor** on curvature: how strongly the loss bends in its flattest direction |
| $`\kappa = L/\mu`$ | "kappa" | **Condition number** — the aspect ratio of the valley. The villain of this chapter |
| $`\lambda_{\max},\ \lambda_{\min}`$ | "lambda-max, lambda-min" | Largest and smallest eigenvalues of the Hessian: sharpest and flattest curvature |
| $`H = \nabla^2\mathcal{L}`$ | "the Hessian" | The grid of second derivatives — the table of curvatures |
| $`\langle a, b\rangle`$ | "inner product of a and b" | The dot product $`a^\top b`$: one number measuring alignment |
| $`\beta`$ | "beta" | The **momentum coefficient** — what fraction of the old velocity you keep |
| $`v_t`$ | "v at $`t`$" | The momentum **buffer** (velocity): a decaying running sum of past gradients |
| $`B`$, $`\mathcal{B}_t`$ | "B", "batch at $`t`$" | Minibatch size, and the set of examples drawn at step $`t`$ |
| $`\Sigma(\theta)`$ | "capital sigma of theta" | **Covariance of the per-example gradients** — how much the minibatch gradient wobbles |
| $`\mathrm{tr}\,\Sigma`$ | "trace of Sigma" | Sum of the diagonal entries: total gradient noise across all coordinates |
| $`\mathcal{O}(1/T)`$ | "big-O of one over T" | "Falls at least as fast as $`1/T`$," with constants deliberately ignored |
| $`dW`$ | "d-W" | An infinitesimal random kick (Brownian motion) inside a stochastic differential equation |
| $`F`$ | "the Fisher" | Fisher information matrix — curvature measured in *probability* space, not parameter space |
| $`\otimes`$ | "Kronecker product" | Build a big matrix by scaling a whole matrix by every entry of another |
| $`\bar\theta_t`$ | "theta-bar" | Here, the **averaged weights**. ⚠ Not a gradient — Chapter 1's bar convention does not apply in §4.8 |

**Full forms for the abbreviations in this chapter:**

| Short | Full form |
|---|---|
| GD | Gradient Descent |
| SGD | Stochastic Gradient Descent |
| LR | Learning Rate |
| EMA | Exponential Moving Average |
| SWA | Stochastic Weight Averaging |
| SDE | Stochastic Differential Equation |
| SVRG | Stochastic Variance Reduced Gradient |
| BFGS / L-BFGS | Broyden–Fletcher–Goldfarb–Shanno / its Limited-memory version |
| K-FAC | Kronecker-Factored Approximate Curvature |
| KL | Kullback–Leibler (divergence) |
| PSD | Positive Semi-Definite |
| TRPO | Trust Region Policy Optimization |
| DDPM | Denoising Diffusion Probabilistic Model |
| DiT | Diffusion Transformer |
| AdamW | Adam with decoupled **W**eight decay |

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
| Convex + smooth | $`L`$-Lipschitz gradient | $`\mathcal{L}(\theta_T)-\mathcal{L}^* = O(1/T)`$ |
| Strongly convex + smooth | also $`\mu`$-strongly convex | linear rate $`O((1-\mu/L)^T)`$ |
| Nonconvex + smooth | just smoothness | $`\min_t\|\nabla\mathcal{L}(\theta_t)\|^2 = O(1/T)`$ (stationary point only) |

Deep learning is the third row, which is why theory gives so little and practice gives so much.

#### Reading the update rule in plain English

$$\theta_{t+1} = \theta_t - \eta\,\nabla\mathcal{L}(\theta_t)$$

Read aloud: *"theta at step t-plus-one equals theta at step t, minus eta times grad-L evaluated at theta-t."* In English: **"the new parameters are the old parameters, nudged a little way in the downhill direction."** Every symbol:

| Piece | Read aloud | Job |
|---|---|---|
| $`\theta_t`$ | "theta sub t" | Every weight in the model at step $`t`$, laid end to end in one vector. For a 100-million-parameter model, $`\theta`$ is a list of $`10^8`$ numbers |
| $`\nabla\mathcal{L}(\theta_t)`$ | "grad L of theta-t" | A vector of the **same length** as $`\theta`$, one entry per parameter, saying "raise this weight and the loss goes up by this much per unit" |
| the minus sign | "minus" | The gradient points **uphill**. You want down. Hence the minus |
| $`\eta`$ | "eta" | How far to walk. A single scalar multiplying the whole vector |
| $`t{+}1`$ vs $`t`$ | — | This is an **assignment**, not an equation. Read it as `theta = theta - eta * grad`. Solving it as an equation would give the nonsense $`\eta\nabla\mathcal{L} = 0`$ |

▸ **The whole of deep learning is this one line, run a few hundred thousand times.** Everything else in Chapters 4 and 5 — momentum, Adam, warmup, cosine decay, clipping — is a modification of either $`\eta`$ or $`\nabla\mathcal{L}`$ in that single expression. Nothing replaces it.

**Put a number in.** Take a one-parameter model with $`\mathcal{L}(\theta) = \theta^2`$, so $`\nabla\mathcal{L} = 2\theta`$. Start at $`\theta_0 = 1`$ with $`\eta = 0.1`$:

- $`\theta_1 = 1 - 0.1(2)(1) = 0.8`$
- $`\theta_2 = 0.8 - 0.1(2)(0.8) = 0.64`$
- $`\theta_3 = 0.512`$, and in general $`\theta_t = 0.8^t`$.

Each step multiplies the distance-to-zero by $`0.8`$. That is **geometric decay**, and it is the whole story of gradient descent on a well-behaved problem: you don't march to the answer, you shrink toward it by a constant fraction each step. Twenty steps gets you to $`0.8^{20}\approx0.012`$.

Now set $`\eta = 1.1`$ instead: $`\theta_1 = 1 - 1.1(2)(1) = -1.2`$, then $`\theta_2 = 1.44`$, then $`-1.728`$. **It diverges, flipping sign every step.** The threshold between these two behaviours is what §4.3 calls the stability condition, and it is the first thing to check when a training run explodes.

#### Unpacking the three regimes

The table is a hierarchy of *promises*, and each row buys a stronger promise with a stronger assumption. Decoding the vocabulary:

- **Convex** — the loss surface is bowl-shaped: any straight line drawn between two points on the surface stays above the surface. Consequence: **there are no local minima to get stuck in.** Every low point is *the* low point.
- **Smooth** ($`L`$-Lipschitz gradient) — the slope is not allowed to change abruptly. No cliffs, no kinks. This is what lets you trust a slope you measured *here* to still be roughly right a small step away.
- **Strongly convex** — convex, plus the bowl  curves upward everywhere (it never flattens into a plain). This rules out long flat troughs.
- **Nonconvex** — none of the above. Bumps, ridges, saddles, plateaus, multiple valleys of differing depth.
- **Stationary point** — a place where $`\nabla\mathcal{L} = 0`$. That could be a minimum, a maximum, or a saddle. The theory in row three does not promise you which.

> **Analogy.** The three rows are three kinds of terrain-survey guarantee. Row 1: "the land is a smooth bowl — walk downhill and you'll reach the bottom, and here's how fast." Row 2: "the bowl is  steep-sided, so you'll reach the bottom *exponentially* fast." Row 3: "the land is ordinary countryside. Walk downhill and eventually you will pass through somewhere flat. We cannot tell you whether it is the sea, a lake, or a mountain pass."

▸ **The gap between row 3 and rows 1–2 is the honest statement of deep learning's theoretical position.** The only thing proved about the algorithm training every model in this book is that it will eventually visit a place where the gradient is small. It is not proved to find a minimum, let alone the best one, let alone one that generalizes. That it works anyway is an empirical fact that the theory is still catching up with — and it is why the rest of this chapter spends its time on *rates* and *mechanisms* rather than on guarantees.

> **Where this came from.** Gradient descent is older than computing. **Augustin-Louis Cauchy** described it in a short note to the French Academy of Sciences in **1847**, as a general method for solving systems of simultaneous equations — his motivating problems came from astronomy, where fitting orbits produced systems too messy to solve exactly. He had no computer; the "iterations" were done by hand. The method was then largely a curiosity until **Haskell Curry** analyzed the choice of step length for nonlinear problems in **1944**, work motivated by wartime calculation. Curry is far better known in computer science for the lambda-calculus ideas that gave us "currying" and the Haskell programming language. **The algorithm that trains every neural network in this book was invented for hand-computed astronomy, roughly a century before the first electronic computers.**

#### Examples and non-examples: what "converged" actually means

The word does three different jobs in three different sentences, and the three claims have wildly different strengths. Keeping them apart is most of what separates a practitioner who can diagnose a run from one who can only restart it.

**✅  convergence claims**

| Example | Why it qualifies |
|---|---|
| Full-batch gradient descent on logistic regression with $`\eta < 2/L`$: loss falls monotonically to the global minimum | Convex and smooth — row 1 of the table. The claim is proved, and the rate is known |
| A quadratic direction with $`\eta = 1/\lambda_i`$: that coordinate hits exactly zero in **one step** | $`(1-\eta\lambda_i) = 0`$. Exact, not asymptotic |
| SGD with $`\eta_t = \eta_0/t`$ on a strongly convex loss: $`\theta_t \to \theta^*`$ | The Robbins–Monro conditions hold ($`\sum\eta_t=\infty`$, $`\sum\eta_t^2<\infty`$) |
| A nonconvex run whose gradient norm falls $`12.0 \to 0.003`$ | This is precisely, and only, what row 3 promises: you arrive somewhere the gradient is small |

**❌ Near-misses — sound like convergence, aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "Loss has been flat for 5,000 steps, so we converged" | At constant $`\eta`$, SGD does not converge — it orbits a noise ball of radius $`\eta\,\mathrm{tr}\Sigma/2\mu B`$. The loss is flat; the weights are still moving | Statistical equilibrium (§4.6) |
| "Training loss is $`0.001`$, so the optimizer found a good solution" | That is a statement about the training set. Zero training loss is achievable on **random labels** | Interpolation. Says nothing whatsoever about $`R(h)`$ |
| "The gradient norm is $`10^{-4}`$, so we're at a minimum" | $`\nabla\mathcal{L}=0`$ is also true at maxima, saddles, and plateaus | A stationary point, of unknown type |
| "Theory proves SGD finds the minimum of a deep network" | Row 3 promises $`\min_t\|\nabla\mathcal{L}\|^2 = \mathcal{O}(1/\sqrt{T})`$ — a small gradient somewhere along the path | $`\epsilon`$-stationarity, which is a much weaker sentence than it sounds |
| "Loss went down every step for 2,000 steps, so $`\eta`$ is safe" | A single direction with $`\eta\lambda_i`$ just past $`2`$ grows silently while the other $`10^8`$ directions dominate the total — until it doesn't | Latent instability, typically discovered as a `NaN` at step 12,000 |
| "Two runs ended at the same loss, so they found the same solution" | Two draws from the same noise ball, at a distance of $`2\sqrt{\eta\,\mathrm{tr}\Sigma/2\mu B}`$ from each other | Equal loss, different weights (§4.6, §4.8) |

▸ **The boundary:** "converged" can mean *the iterates stopped moving*, *the gradient reached zero*, or *the model is good*. These are three distinct claims about three distinct objects ($`\theta`$, $`\nabla\mathcal{L}`$, and $`R(h)`$), and **every theorem in this chapter proves only the middle one.**

> **Common misconception.** *"If the optimizer converged, it found a good solution."* Convergence is a statement about the gradient going to zero, nothing more. The random-label experiment in Chapter 2 is a fully converged run — training loss zero, gradient tiny, every optimization criterion satisfied — that produces a model with 10% test accuracy. **The belief is tempting because in the convex world it is simply true**: for a convex loss there is one minimum, the gradient vanishes only there, and reaching it means winning. Every proof in §4.3 lives in that world, and the vocabulary came with it. In the nonconvex world the vocabulary survived and the guarantee did not.

> **Common misconception.** *"The loss stopped moving, so training is finished."* At a fixed learning rate the loss curve flattening means the *signal* term $`\|\nabla\mathcal{L}\|^2`$ has dropped below the *noise* term $`\mathrm{tr}\Sigma/B`$ — the two terms in §4.6's identity have swapped ranks. The weights have not stopped; they are circling. Drop the learning rate by $`10\times`$ at that exact moment and the loss falls again, visibly, in a step — the famous cliff in every published training curve. **The belief is tempting because a flat line  does mean "nothing more is happening" for a deterministic algorithm**, which is the mental model most people are carrying. For a stochastic one, a flat line means "the ball has reached its equilibrium radius," and shrinking the ball is a separate action you have to take.

---

## 4.2 The definitions that make the proofs work

**$`L`$-smooth** (gradient is $`L`$-Lipschitz):
$$\|\nabla\mathcal{L}(x)-\nabla\mathcal{L}(y)\| \le L\|x-y\| \iff \lambda_{\max}(\nabla^2\mathcal{L}) \le L$$

The key consequence — the **descent lemma**:
▸ $$\mathcal{L}(y) \le \mathcal{L}(x) + \langle\nabla\mathcal{L}(x), y-x\rangle + \frac{L}{2}\|y-x\|^2$$

*Read it as:* the function is below a quadratic bowl that touches it at $`x`$. So if you minimize the bowl, you're guaranteed to decrease the function.

**$`\mu`$-strongly convex:**
$$\mathcal{L}(y) \ge \mathcal{L}(x) + \langle\nabla\mathcal{L}(x),y-x\rangle + \frac{\mu}{2}\|y-x\|^2 \iff \lambda_{\min}(\nabla^2\mathcal{L}) \ge \mu > 0$$

**Condition number:** $`\kappa = L/\mu`$. This is the ratio of the steepest to the shallowest curvature. **It is the single number that determines how hard your optimization problem is.**

#### Unpacking $`L`$-smoothness

$$\|\nabla\mathcal{L}(x)-\nabla\mathcal{L}(y)\| \le L\|x-y\|$$

Read aloud: *"the norm of the difference between the gradient at x and the gradient at y is at most L times the norm of x minus y."* In English: **"move a distance $`d`$ through parameter space, and the slope can change by at most $`L\cdot d`$."** It is a speed limit on how fast the terrain can change its tilt.

Decoding each part:

- $`\|\cdot\|`$ is the ordinary length of a vector (§0.6). $`\|x - y\|`$ = "how far apart are these two parameter settings."
- $`\nabla\mathcal{L}(x) - \nabla\mathcal{L}(y)`$ = "how different are the two slopes."
- $`L`$ is the **worst-case exchange rate** between the two. Small $`L`$ = gently rolling terrain. Large $`L`$ = the tilt can swing wildly over a short distance.
- $`\iff \lambda_{\max}(\nabla^2\mathcal{L}) \le L`$ says the same thing in curvature language: the Hessian's largest eigenvalue — the sharpest bend anywhere — never exceeds $`L`$. **Smoothness and maximum curvature are the same quantity seen from two sides**, because curvature *is* the rate at which slope changes.

> **Analogy.** $`L`$ is the tightest hairpin allowed on a mountain road. A road with a low $`L`$ has only long sweeping bends: if you glance at the road ahead and it looks straight, you can safely drive fifty metres without looking again. A road with a huge $`L`$ can turn 90° in a metre, so any information you gathered is stale almost immediately. **Gradient descent's step size is exactly "how far you drive between glances," and $`L`$ is what makes that safe or reckless.**

#### The descent lemma, decoded

▸ $$\mathcal{L}(y) \le \mathcal{L}(x) + \langle\nabla\mathcal{L}(x), y-x\rangle + \frac{L}{2}\|y-x\|^2$$

Three terms, and each one has a plain job:

| Term | Read aloud | What it is |
|---|---|---|
| $`\mathcal{L}(x)`$ | "L of x" | Where you are now — the current loss |
| $`\langle\nabla\mathcal{L}(x),\ y-x\rangle`$ | "the inner product of the gradient with the displacement" | The **linear prediction**: slope times distance. If the surface were a flat ramp, this would be exact |
| $`\frac{L}{2}\|y-x\|^2`$ | "L over two, times the squared distance" | The **penalty for the ramp being a lie.** It grows with the *square* of how far you moved |

So the sentence is: *"the true loss after a move is at most the loss you'd predict from the slope, plus a fudge term that grows quadratically with step length."*

**Make it concrete in one dimension.** Let $`\mathcal{L}(\theta) = \theta^2`$, so $`L = 2`$ (the second derivative). At $`x = 3`$: $`\mathcal{L}(3)=9`$, $`\nabla\mathcal{L}(3)=6`$. Move to $`y = 2`$:

- Linear prediction: $`9 + 6(2-3) = 3`$.
- Truth: $`\mathcal{L}(2) = 4`$.
- The bound: $`9 + 6(-1) + \frac{2}{2}(1)^2 = 4`$. ✓ **Exactly tight**, because a parabola *is* its own quadratic upper bound.

The linear prediction (3) was too optimistic — the true value (4) is higher. That gap is precisely what the $`\frac{L}{2}\|y-x\|^2`$ term pays for.

> **Analogy.** You're driving downhill and the road ahead is hidden by fog. Your speedometer and inclinometer tell you the slope right where you are. The descent lemma says: "assume the hill continues at this slope, then subtract a safety margin that grows with the square of how far you commit." **It converts a local measurement into a guaranteed global statement**, which is the only reason any of the proofs in §4.3 work.

▸ **Why it is called the descent lemma:** if the true function is always *below* this quadratic bowl and the bowl touches it at $`x`$, then minimizing the bowl gives a point where the real function is at least that low. **You get a guaranteed decrease by minimizing something you can actually compute.** That single move is the engine of first-order optimization.

#### $`\mu`$-strong convexity, decoded

$$\mathcal{L}(y) \ge \mathcal{L}(x) + \langle\nabla\mathcal{L}(x),y-x\rangle + \frac{\mu}{2}\|y-x\|^2$$

**Identical shape, inequality flipped, $`L`$ replaced by $`\mu`$.** Where smoothness said "the surface is never *above* this bowl," strong convexity says "the surface is never *below* that bowl." One is a lid; the other is a floor.

▸ **The pair together sandwiches your loss surface between two parabolas** — a wide one ($`\mu`$) underneath and a narrow one ($`L`$) on top. Every convergence rate in this chapter is squeezed out of that sandwich, and nothing more.

Why the floor matters: it forbids **long flat troughs.** Without it, a function can approach its minimum arbitrarily slowly because the slope keeps vanishing before you arrive. Strong convexity guarantees that being far from the optimum forces you to have a real gradient — so *there is always a usable signal telling you which way home is.*

#### The condition number $`\kappa`$, and why it is the villain

$$\kappa = \frac{L}{\mu} = \frac{\text{sharpest curvature}}{\text{flattest curvature}}$$

Read aloud: *"kappa equals L over mu."* In English: **"how lopsided is this bowl?"**

- $`\kappa = 1`$ — a perfectly round bowl. The gradient points straight at the minimum from everywhere. One well-sized step gets you there.
- $`\kappa = 10^4`$ — a valley 10,000 times narrower across than it is long. The gradient points almost entirely *across* the valley rather than *along* it.

> **Analogy.** Picture a running track versus a circular bowl. In a round bowl, downhill is always toward the centre. In a long, steeply-banked track, "downhill" as felt underfoot means "toward the inside rail" — almost perpendicular to the direction you actually need to travel to reach the finish. **$`\kappa`$ measures how badly the direction that feels downhill differs from the direction that gets you there.** A ball dropped into a bathtub sloshes side to side many times while creeping slowly toward the drain; $`\kappa`$ is the ratio of the sloshing to the creeping.

**A one-line numerical demonstration.** Take $`\mathcal{L}(\theta) = \frac12(100\theta_1^2 + 0.01\theta_2^2)`$. Then $`L = 100`$, $`\mu = 0.01`$, $`\kappa = 10^4`$. Stability forces $`\eta < 2/100 = 0.02`$. At $`\eta = 0.01`$, the $`\theta_1`$ coordinate shrinks by a factor $`|1 - 0.01\times100| = 0`$ — it is solved in a single step. The $`\theta_2`$ coordinate shrinks by $`|1 - 0.01\times0.01| = 0.9999`$ — it needs about **6,900 steps to halve**, and about 23,000 to reach a tenth of where it started. *One coordinate is finished in a single step; the other takes tens of thousands.* You cannot use a bigger step, because doing so would blow up the first coordinate.

▸ **This is the entire problem, and every optimizer in Chapters 4 and 5 is an attempt to solve it.** Momentum attacks it by averaging away the sloshing. Adam attacks it by giving each coordinate its own step size. Newton's method attacks it by rescaling space so the valley becomes round. Normalization layers (Ch. 7) attack it by keeping the loss surface from becoming lopsided in the first place. **When you understand $`\kappa`$, the rest of optimization stops looking like a bag of tricks and starts looking like one problem with several attacks.**

---

## 4.3 The convergence proof for smooth convex GD

Plug $`y = \theta_{t+1} = \theta_t - \eta g_t`$ into the descent lemma:

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta\|g_t\|^2 + \frac{L\eta^2}{2}\|g_t\|^2 = \mathcal{L}(\theta_t) - \eta\left(1 - \frac{L\eta}{2}\right)\|g_t\|^2$$

▸ **Stability condition:** we need $`1 - L\eta/2 > 0`$, i.e. $`\eta < 2/L`$. With $`\eta = 1/L`$ we get the clean

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \frac{1}{2L}\|g_t\|^2$$

**Every step decreases the loss by an amount proportional to the squared gradient norm, divided by the sharpness.** That's the whole mechanism of gradient descent in one line.

Summing over $`T`$ steps (nonconvex case):
$$\frac{1}{2L}\sum_{t=0}^{T-1}\|g_t\|^2 \le \mathcal{L}(\theta_0) - \mathcal{L}^* \implies \min_{t<T}\|g_t\|^2 \le \frac{2L(\mathcal{L}(\theta_0)-\mathcal{L}^*)}{T}$$

So gradient norm shrinks as $`O(1/\sqrt T)`$ in the nonconvex case. **This is all the guarantee deep learning has.** Not a global minimum, not even a local minimum — just "you'll eventually pass near a point with small gradient."

**With convexity added**, a few more lines give $`\mathcal{L}(\theta_T) - \mathcal{L}^* \le \frac{\|\theta_0-\theta^*\|^2}{2\eta T}`$, the $`O(1/T)`$ rate.

**With strong convexity:**
▸ $$\|\theta_T - \theta^*\|^2 \le \left(1-\frac{\mu}{L}\right)^T\|\theta_0-\theta^*\|^2 = \left(1-\frac{1}{\kappa}\right)^T\|\theta_0-\theta^*\|^2$$

To reach error $`\epsilon`$: $`T = O(\kappa\log(1/\epsilon))`$ iterations.

**Number:** if $`\kappa = 10^4`$ (typical for a poorly-conditioned deep network), you need $`\sim 10^4`$ iterations per digit of accuracy. If $`\kappa = 10`$, you need $`\sim10`$. **This is why conditioning is everything, and why normalization layers (Ch. 7) are worth more than most architectural cleverness.**

#### Reading the proof, line by line

The proof is four lines and each is a small, mechanical move. Here is the same argument in slow motion.

**Line 1 — substitute your own step into the bound.** The descent lemma held for *any* $`y`$. Choose $`y`$ to be exactly where gradient descent is about to go, $`y = \theta_t - \eta g_t`$. Then $`y - x = -\eta g_t`$, and the two terms become:

- $`\langle\nabla\mathcal{L}(x),\ -\eta g_t\rangle = -\eta\|g_t\|^2`$, since the gradient is being dotted with a scaled copy of itself. *A vector dotted with itself is its squared length* — that is where $`\|g_t\|^2`$ comes from.
- $`\frac{L}{2}\|-\eta g_t\|^2 = \frac{L\eta^2}{2}\|g_t\|^2`$, since pulling a scalar out of a norm squares it.

**Line 2 — read the sign.** Factor out $`\eta\|g_t\|^2`$:

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \eta\left(1 - \frac{L\eta}{2}\right)\|g_t\|^2$$

Everything hinges on the bracket $`\left(1 - \frac{L\eta}{2}\right)`$. If it is **positive**, you are subtracting something positive from the loss: **guaranteed improvement.** If it is negative, the inequality permits the loss to *rise*, and the guarantee is gone.

▸ **So the stability condition $`\eta < 2/L`$ is not a heuristic — it is the exact point at which the sign of a bracket flips.** Below it, every step provably helps. Above it, nothing is promised, and in practice the loss goes to `NaN` within a few dozen steps. **The most common cause of a diverging training run is that you crossed $`2/L`$.**

**Line 3 — the clean choice.** Setting $`\eta = 1/L`$ makes the bracket exactly $`\tfrac12`$, giving

$$\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \frac{1}{2L}\|g_t\|^2$$

Read aloud: *"the new loss is at most the old loss, minus one over two-L, times the squared gradient norm."* In English: **"each step buys you a loss reduction equal to the squared steepness divided by twice the sharpness."**

> **Analogy.** Steepness is your reward for stepping; sharpness is your tax. A steep, gently-curving slope (big $`\|g\|`$, small $`L`$) is a bargain — you can commit to a long stride and cash in. A steep but violently-curving slope (big $`\|g\|`$, big $`L`$) is a trap: the slope you measured stops being true almost immediately, so you must take small steps and progress crawls.

**Line 4 — the telescoping trick.** Every step buys at least $`\frac{1}{2L}\|g_t\|^2`$ of loss reduction. But the *total* loss reduction available in the whole run is finite — it can never exceed $`\mathcal{L}(\theta_0) - \mathcal{L}^*`$, the drop from your starting point to the best possible. So:

$$\underbrace{\frac{1}{2L}\sum_{t=0}^{T-1}\|g_t\|^2}_{\text{total bought}} \ \le\ \underbrace{\mathcal{L}(\theta_0) - \mathcal{L}^*}_{\text{total available}}$$

▸ **This is a budget argument, and it is worth admiring.** You have a fixed amount of loss to spend. Every large gradient spends some of it. Therefore **you cannot have many large gradients** — you would run out of loss. Since the *smallest* of $`T`$ numbers is no bigger than their average, $`\min_{t<T}\|g_t\|^2 \le \frac{2L(\mathcal{L}(\theta_0)-\mathcal{L}^*)}{T}`$.

**Read the conclusion honestly.** It says: *somewhere in your first $`T`$ steps, you passed a point where the gradient was small.* It does not say the last step was good. It does not say you are near a minimum. It does not even tell you **which** step it was, so you could not go back and retrieve it. This is what "all the guarantee deep learning has" means, precisely.

#### What the three rates mean in wall-clock terms

The three rates look similar on paper and are wildly different in practice. Put the same target accuracy through each.

| Rate | Form | Steps to gain one decimal digit | Character |
|---|---|---|---|
| Nonconvex | $`\min_t\|g_t\|^2 \sim 1/T`$ | needs **10×** the total steps so far | brutal; you buy accuracy by multiplying your budget |
| Convex | $`\mathcal{L}(\theta_T)-\mathcal{L}^* \sim 1/T`$ | needs **10×** the total steps so far | same shape, but now it is the *loss* that's converging, not just the gradient |
| Strongly convex | $`(1-1/\kappa)^T`$ | a **fixed** $`\approx 2.3\kappa`$ more steps | each digit costs the same as the last |

▸ **The difference between $`1/T`$ and $`(1-1/\kappa)^T`$ is the difference between "progress gets harder forever" and "progress costs a constant rent."** The strongly convex rate is called **linear convergence** — confusingly, because the *log* of the error falls linearly with $`T`$. Each digit costs a fixed toll. Under a $`1/T`$ rate, the first digit might cost 10 steps, the second 100, the third 1,000. **That is why practitioners care so much about anything that restores curvature: it changes the shape of the cost curve, not just its constant.**

**Where $`2.3\kappa`$ comes from.** To get $`(1 - 1/\kappa)^T = 0.1`$, take logs: $`T\ln(1-1/\kappa) = \ln(0.1)`$. For large $`\kappa`$, $`\ln(1-1/\kappa)\approx -1/\kappa`$, so $`T \approx \kappa\ln 10 = 2.303\kappa`$. With $`\kappa = 10^4`$ that is **23,000 steps per digit** — and with $`\kappa = 10`$, just 23. Same algorithm, same code, a thousandfold difference in cost, decided entirely by the shape of the surface.

> **Where this came from.** The modern habit of proving *rates* rather than mere convergence comes largely from the Soviet optimization school of the 1960s–80s — **Boris Polyak**, **Arkadi Nemirovski**, **David Yudin**, and **Yurii Nesterov**. Nemirovski and Yudin's 1983 book established **lower** bounds as well: results of the form "no method that only looks at gradients can possibly do better than this." That reframing matters. Before it, a new optimizer was judged by whether it worked; afterwards, it could be judged against a known ceiling, and $`\mathcal{O}(1/T^2)`$ could be recognized as *optimal* rather than merely good.

---

## 4.4 The quadratic model: seeing $`\kappa`$ do its damage

Take $`\mathcal{L}(\theta) = \frac12\theta^\top H\theta`$ with $`H = Q\Lambda Q^\top`$. In the eigenbasis ($`u = Q^\top\theta`$), GD decouples completely:

$$u^{(i)}_{t+1} = (1-\eta\lambda_i)\,u^{(i)}_t \implies u^{(i)}_t = (1-\eta\lambda_i)^t u^{(i)}_0$$

▸ Each eigendirection contracts independently by factor $`|1-\eta\lambda_i|`$.

- **Convergence requires** $`|1-\eta\lambda_i|<1`$ for all $`i`$, i.e. $`\eta < 2/\lambda_{\max}`$. **This is the stability threshold** — remember it for Edge of Stability in Ch. 5.
- The **slowest** direction is $`\lambda_{\min}`$: contraction $`1-\eta\lambda_{\min} = 1-\lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa`$ at the largest stable LR.
- The **optimal** LR balances the two extremes: $`\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}}`$, giving contraction $`\frac{\kappa-1}{\kappa+1}`$.

**The picture:** a long narrow valley. GD zig-zags across the narrow direction (which is near its stability limit) while crawling along the long direction. You've seen this plot; now you know it's $`|1-\eta\lambda_i|`$ close to $`-1`$ in one direction and close to $`+1`$ in the other.

**Numbers.** $`\lambda_{\max}=100`$, $`\lambda_{\min}=0.01`$, $`\kappa=10^4`$. Max stable $`\eta = 0.02`$. Contraction in the slow direction: $`1 - 0.02\times0.01 = 0.9998`$. To reduce that component by $`e^{-1}`$ takes $`5{,}000`$ steps. To reduce it by $`10^{-3}`$ takes ~35,000 steps. **That's a long training run.**

#### The quadratic model, decoded

This section is the most valuable half-page in the chapter, because it turns "$`\kappa`$ is bad" from a slogan into arithmetic you can do. Take it slowly.

**Why a quadratic at all?** Near any minimum, *every* smooth function looks like a quadratic bowl — that is what a Taylor expansion says. The gradient vanishes at the bottom, so the first surviving term is the curvature term. So $`\mathcal{L}(\theta) = \frac12\theta^\top H\theta`$ is not a toy: it is what your real loss surface looks like once you are close enough to care.

**Reading $`\mathcal{L}(\theta) = \frac12\theta^\top H\theta`$.** The expression $`\theta^\top H\theta`$ is a **quadratic form** (§1.1.2) — a single number, "the height of a bowl in direction $`\theta`$." In one dimension with $`H = a`$, it is just $`\frac12 a\theta^2`$: the familiar parabola. The $`\frac12`$ is there purely so the gradient comes out clean, $`\nabla\mathcal{L} = H\theta`$, with no stray factor of 2.

**The change of variables, which is the whole trick.** Write $`H = Q\Lambda Q^\top`$ (§1.1.2): rotate, stretch along the axes, rotate back. Now define $`u = Q^\top\theta`$ — *"describe the parameters in the coordinate system where the bowl's axes line up with the grid."*

> **Analogy.** You are given a tilted, oval swimming pool and asked to describe how water sloshes in it. Instead of wrestling with the tilt, you walk around the pool until you are standing along its long axis. Now "length" and "width" are separate, independent questions. **The rotation $`Q^\top`$ is walking around the pool.** Nothing about the pool changed; you changed where you stand.

In these coordinates the update **decouples**: each direction evolves entirely on its own, with no interference:

$$u^{(i)}_{t+1} = (1-\eta\lambda_i)\,u^{(i)}_t$$

Read aloud: *"u-superscript-i at t-plus-one equals one minus eta lambda-i, times u-superscript-i at t."* Here $`u^{(i)}`$ means **the $`i`$-th coordinate** in the rotated frame, and $`\lambda_i`$ is the curvature along that axis. So a $`10^8`$-dimensional coupled optimization problem becomes $`10^8`$ *independent* one-dimensional problems, each of the trivial form "multiply by a number each step."

▸ **The number you multiply by is $`(1-\eta\lambda_i)`$, and everything follows from where it sits on the number line:**

| $`1-\eta\lambda_i`$ | Behaviour of that direction | Feels like |
|---|---|---|
| between $`0`$ and $`1`$ | shrinks steadily toward zero | crawling downhill |
| exactly $`0`$ | solved in **one step** | a perfect step size for that axis |
| between $`-1`$ and $`0`$ | shrinks, but **flips sign** each step | overshooting, oscillating, converging |
| exactly $`-1`$ | bounces forever, never shrinking | permanent oscillation |
| beyond $`-1`$ | grows, flipping sign | **divergence**, and `NaN` shortly after |

**Numbers on all five rows**, with $`\lambda_i = 100`$ throughout, varying only $`\eta`$:

- $`\eta = 0.001`$: factor $`0.9`$ → slow, steady.
- $`\eta = 0.01`$: factor $`0`$ → done in one step.
- $`\eta = 0.015`$: factor $`-0.5`$ → overshoots past the bottom, comes back, halving each time.
- $`\eta = 0.02`$: factor $`-1`$ → oscillates forever at fixed amplitude. This is the stability edge, $`\eta = 2/\lambda_{\max}`$.
- $`\eta = 0.025`$: factor $`-1.5`$ → each swing is 50% wider than the last. After 40 steps, $`1.5^{40}\approx 1.1\times10^7`$.

▸ **Notice what the single scalar $`\eta`$ must do:** it is one number applied to *every* coordinate at once, and each coordinate wants a different one. The sharpest direction ($`\lambda_{\max}`$) sets a hard ceiling — exceed $`2/\lambda_{\max}`$ and that direction explodes, taking the whole model with it. The flattest direction ($`\lambda_{\min}`$) then has to make do with whatever is left, which is $`\kappa`$ times too small for it. **One knob, a million conflicting demands, and the loudest one wins.**

**Why the zig-zag picture looks the way it does.** In the sharp direction, $`1-\eta\lambda_{\max}`$ sits near $`-1`$: sign flips every step, so the path crosses back and forth over the valley floor. In the flat direction, $`1-\eta\lambda_{\min}`$ sits near $`+1`$: essentially no movement per step. Draw both at once and you get the classic hairpin path — **furious lateral sloshing plus imperceptible forward progress.** The picture is not an artist's impression; it is $`(-0.9)^t`$ plotted against $`(0.9998)^t`$.

**Verifying the book's numbers.** With $`\lambda_{\max}=100`$, stability caps $`\eta`$ at $`2/100 = 0.02`$. The slow direction then contracts by $`1 - 0.02(0.01) = 0.9998`$ per step, i.e. it loses $`0.02\%`$ of its error each step. To fall by a factor $`e \approx 2.718`$ takes $`1/0.0002 = 5{,}000`$ steps. To fall by $`1000\times`$ takes $`\ln(1000)/0.0002 \approx 34{,}500`$ steps. **And that is the best case** — with the largest step size that does not immediately destroy the run.

#### Examples and non-examples: problems the learning rate can fix

The quadratic model gives you the exact tool for this: $`\eta`$ enters the dynamics only through the contraction factor $`(1-\eta\lambda_i)`$. Any pathology you can describe as "that factor sits in the wrong place on the number line" is an $`\eta`$ problem. Nothing else is.

**✅  learning-rate problems**

| Example | Why it qualifies |
|---|---|
| Loss becomes `NaN` within 30 steps; halving $`\eta`$ fixes it permanently | $`\eta > 2/\lambda_{\max}`$, so $`\lvert 1-\eta\lambda_{\max}\rvert > 1`$ and that direction grows geometrically. At factor $`-1.5`$, forty steps gives $`1.5^{40} \approx 1.1\times10^7`$ |
| Loss visibly bounces up and down with a period of about two steps | $`1-\eta\lambda_{\max}`$ sits just inside $`-1`$: the sign flips every step. This is the stability edge, and it is diagnosable by eye |
| Loss falls steadily but $`10\times`$ too slowly, and $`\eta \times 8`$ gives a clean $`8\times`$ speedup | You were sitting at $`1-\eta\lambda_i \approx 0.99`$ everywhere, far below the ceiling. Free money, and rare |
| Turning on momentum with $`\beta=0.9`$ causes divergence, and cutting $`\eta`$ by 10 restores it | Momentum multiplied the effective step by $`1/(1-\beta) = 10`$. It really was a step-size problem — you just made it without meaning to |

**❌ Near-misses — look like learning-rate problems, aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Loss plateaus at $`2.31`$ for 20,000 steps, and no value of $`\eta`$ helps | With $`\kappa = 10^4`$, the flat direction contracts by $`0.9998`$ per step at the **largest stable** $`\eta`$. That is $`34{,}500`$ steps to fall $`1000\times`$, and $`\eta`$ cannot be raised because $`\lambda_{\max}`$ forbids it | **Conditioning.** Fixed by preconditioning, normalization, or better architecture — never by the step size |
| Training diverges in fp16 but is stable in fp32 at the identical $`\eta`$ | fp16's largest representable value is $`65{,}504`$; the gradient overflowed before any dynamics were involved | Numerical range. Fixed by loss scaling (Ch. 14) |
| One loss spike every few thousand steps, recovering afterwards | A rare high-curvature batch. Lowering $`\eta`$ globally taxes all 5,000 well-behaved steps to police one bad one | An outlier-batch problem. Fixed by gradient clipping (§4.8) |
| Halving $`\eta`$ and halving $`B`$ together changes nothing | They appear only as the ratio $`\eta/B`$ (§4.6). You changed both halves of a fraction | A no-op, dressed as two experiments |
| Loss trains beautifully; test accuracy is poor | $`\eta/B`$ does influence which minimum you land in — but the training dynamics were never the failure | A generalization problem (Ch. 2), not an optimization one |
| Loss decreases faster with $`\eta = 0.01`$ than $`\eta = 0.001`$ over 100 steps, so $`0.01`$ is better | Early-run speed and final quality are different questions; the larger step also sets a larger noise-ball floor | A measurement over the wrong horizon |

▸ **The boundary:** $`\eta`$ multiplies **every** eigenvalue at once, which is exactly why it cannot change $`\kappa = \lambda_{\max}/\lambda_{\min}`$ — the ratio is invariant to scaling both terms. **The learning rate sets your position between "too slow" and "unstable"; the condition number sets how wide that gap is.** No amount of tuning one fixes the other.

> **Common misconception.** *"The learning rate is the hyperparameter that really matters — get it right and the rest is detail."* It is the most *sensitive* knob, which is not the same as the most *important* one. Its useful range is bounded above by $`2/\lambda_{\max}`$, a number set by your architecture, initialization, and normalization — none of which are $`\eta`$. And the number of steps you need is set by $`\kappa`$, which $`\eta`$ provably cannot touch. The entire remainder of this chapter, and all of Chapter 5, exists because tuning $`\eta`$ hits a wall: momentum attacks $`\kappa`$ with a square root, preconditioning attacks it directly, and normalization (Ch. 7) attacks $`\lambda_{\max}`$ so the ceiling itself moves. **The belief is tempting because $`\eta`$ is the one knob whose effect is instant and unmistakable** — wrong by $`3\times`$ and your run dies in the first minute. Knobs with immediate, dramatic feedback feel more important than knobs whose effect is a 30% change in total training time, even when the second is worth more.

> **Common misconception.** *"A smaller learning rate is always safer."* Safer against divergence, yes. But $`\eta`$ also sets the noise-ball radius $`\eta\,\mathrm{tr}\Sigma/2\mu B`$ and, through $`\eta/B`$, the temperature that determines *which* basin you settle in. Training at $`\eta = 10^{-6}`$ from step one is not a cautious version of training at $`10^{-3}`$ — it is a different algorithm that explores less, lands in a sharper minimum, and takes a thousand times as long to get there. **The belief is tempting because every failure mode you can see is caused by $`\eta`$ being too large**, and none of the failure modes caused by $`\eta`$ being too small announce themselves. They just look like a run that finished.

---

## 4.5 Momentum

### The one-line idea

Accumulate a running average of past gradients so that consistent directions build up speed and oscillating directions cancel out.

### The analogy

A ball rolling down the valley instead of a hiker taking discrete steps. The ball's inertia carries it along the valley floor and averages out the side-to-side sloshing.

### Heavy-ball (Polyak, 1964)

▸ $$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

(PyTorch's convention. The "$`\ (1-\beta)g_t`$" variant just rescales $`\eta`$.)

**What it does, quantitatively.** For constant $`g`$, the velocity converges to $`v_\infty = g/(1-\beta)`$.

▸ $$\text{effective step} = \frac{\eta}{1-\beta}\qquad\Rightarrow\qquad \beta=0.9 \text{ multiplies your step size by } 10.$$

This is why you must lower the LR when you turn on momentum, and why "momentum 0.9 vs 0.99" is a bigger change than it looks (10× vs 100×).

The **memory horizon** is $`1/(1-\beta)`$ steps: $`\beta=0.9 \to 10`$ steps, $`\beta=0.99\to 100`$.

#### Unpacking the momentum update

$$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

Read aloud: *"v at t-plus-one equals beta times v at t, plus g at t; then theta at t-plus-one equals theta at t minus eta times v at t-plus-one."* In English: **"keep a running tally of recent gradients that fades by a factor $`\beta`$ each step, and step along the tally instead of along the raw gradient."**

| Symbol | Read aloud | Job |
|---|---|---|
| $`v_t`$ | "v at t" | The **velocity buffer** — a vector the same shape as $`\theta`$, stored alongside the weights |
| $`\beta`$ | "beta" | How much of yesterday survives into today. $`\beta = 0`$ recovers plain gradient descent exactly |
| $`g_t`$ | "g at t" | Today's fresh gradient, added on top |
| $`\eta v_{t+1}`$ | — | You now step along the *accumulated* direction, not the instantaneous one |

**Unroll it and the structure appears.** Substituting repeatedly:

$$v_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \beta^3 g_{t-3} + \dots$$

▸ **Momentum is a weighted vote over all past gradients, with older votes weighted $`\beta^k`$.** With $`\beta = 0.9`$, the gradient from 10 steps ago still carries $`0.9^{10} = 0.35`$ of a vote; from 50 steps ago, $`0.9^{50} = 0.005`$ — effectively nothing. That is why $`1/(1-\beta)`$ is called the memory horizon: it is roughly where the weights have faded to insignificance.

> **Analogy.** Plain gradient descent is a hiker who re-reads the compass at every footfall and instantly obeys it, including when the reading is nonsense. Momentum is a **loaded shopping trolley.** Push it left, then right, then left again and it barely deviates — the conflicting shoves cancel. Push it consistently in one direction and it builds up speed and keeps rolling even over a flat patch. **Consistency is amplified; contradiction is cancelled.** That is the entire mechanism, and it is exactly what a long narrow valley needs: the across-valley shoves alternate and cancel, the along-valley shoves agree and accumulate.

**Where $`\eta/(1-\beta)`$ comes from — do the geometric series.** Suppose the gradient is a constant $`g`$ every step. Then $`v_\infty = g + \beta g + \beta^2 g + \dots = g\,(1 + \beta + \beta^2 + \dots) = \frac{g}{1-\beta}`$, because a geometric series with ratio $`\beta < 1`$ sums to $`1/(1-\beta)`$.

▸ **Put the number in and it stops being abstract.** $`\beta = 0.9 \Rightarrow \frac{1}{1-0.9} = 10`$. **Turning on momentum with $`\beta = 0.9`$ silently multiplies your step size by ten.** $`\beta = 0.99`$ multiplies it by a hundred. This is the single most common cause of "I added momentum and it diverged" — you did not add a stabilizer, you added a 10× learning-rate increase. **Nudging $`\beta`$ from $`0.99`$ down to $`0.98`$ halves the effective step; nudging it from $`0.9`$ up to $`0.99`$ multiplies it tenfold.** The knob is wildly nonlinear near 1, which is why $`\beta`$ is always quoted to two or three decimal places while $`\eta`$ is quoted to one significant figure.

**Two conventions, and why you must check which one you have.** PyTorch uses $`v \leftarrow \beta v + g`$. Many textbooks use $`v \leftarrow \beta v + (1-\beta)g`$, which is a true exponential moving average and has $`v_\infty = g`$ — no amplification at all. **The two differ by exactly the factor $`1/(1-\beta)`$**, so copying a learning rate between frameworks without checking the convention can be a 10× error. The book's parenthetical about "just rescales $`\eta`$" is this fact.

> **Where this came from.** The heavy-ball method is **Boris Polyak's**, from a 1964 paper in a Soviet computational-mathematics journal on speeding up iterative methods. The name is literal: Polyak's analysis models a **physical ball with mass rolling down the loss surface under friction**, where $`\beta`$ plays the role of one-minus-the-friction. Momentum then arrived in neural networks by a separate route — it appears in the 1986 Rumelhart–Hinton–Williams backpropagation paper as a practical fix for slow, oscillating training, presented as an engineering tweak rather than as Polyak's theorem. **The optimization theory and the deep-learning practice were, for a couple of decades, two communities using the same algorithm without much conversation.**

> **The story behind the caveat.** Heavy-ball's beautiful $`\sqrt{\kappa}`$ guarantee holds for *quadratics*. It was shown much later — by **Laurent Lessard, Benjamin Recht, and Andrew Packard in 2016**, using tools borrowed from control theory — that there exist strongly convex, smooth functions on which heavy-ball with its optimal parameters **fails to converge at all**, cycling forever. Nesterov's method has no such counterexample. This is a rare and instructive case: an algorithm used by everyone for fifty years, with a hole in its guarantee found by importing machinery from a different engineering discipline.

**Analysis on the quadratic.** In eigendirection $`i`$, the coupled recursion has characteristic polynomial
$$z^2 - (1+\beta-\eta\lambda_i)z + \beta = 0$$
Optimal tuning $`\beta^* = \left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2`$, $`\eta^* = \frac{4}{(\sqrt{L}+\sqrt\mu)^2}`$ gives contraction rate

▸ $$1 - \frac{1}{\sqrt\kappa}\quad\text{instead of}\quad 1-\frac{1}{\kappa}$$

**This is the acceleration.** For $`\kappa=10^4`$: plain GD needs $`10^4`$ iterations per digit, momentum needs $`10^2`$. **A 100× speedup for one extra buffer.** That is the best deal in optimization.

#### Where the square root comes from, in plain English

The formulas above are dense, so here is what they are saying and why the result is startling.

**"Characteristic polynomial"** is the standard tool for a recursion that depends on the *last two* steps rather than one. Plain gradient descent had $`u_{t+1} = c\,u_t`$ — one previous value, one multiplier, done. Momentum's $`u_{t+1}`$ depends on both $`u_t`$ and $`u_{t-1}`$ (because $`v`$ carries history), which makes it a **second-order recursion** — mathematically the same object as a mass on a spring, or the Fibonacci sequence. For such a recursion you find the growth factors by solving a quadratic, and $`z^2 - (1+\beta-\eta\lambda_i)z + \beta = 0`$ is that quadratic. The two roots are the two rates at which that direction can decay.

**The key structural fact:** the product of the two roots of $`z^2 + bz + c`$ is $`c`$ — here, $`\beta`$. So both roots have magnitude $`\sqrt\beta`$ when they are a complex-conjugate pair. **That $`\sqrt{\ }`$ is the entire source of the acceleration**, and it is the reason the answer is $`1 - 1/\sqrt\kappa`$ rather than $`1 - 1/\kappa`$.

> **Analogy.** Plain gradient descent is a hiker in the sloshing bathtub, obediently walking whichever way is steepest — so mostly across the tub, wasting almost all motion. Momentum is a **swinging pendulum**: the sideways sloshes turn into a regular oscillation that cancels itself, while the persistent drift toward the drain adds up undisturbed. The optimizer stops fighting the oscillation and starts *tuning* it, so it rings at a frequency where it cancels.

▸ **Sit with the size of the win.** At $`\kappa = 10^4`$: $`\sqrt{\kappa} = 100`$. Gradient descent needs about $`2.3\times10^4`$ steps per decimal digit; momentum needs about $`2.3\times10^2`$. **A run that would take a hundred days finishes in one**, and the price is one extra buffer of the same size as the weights. In a field where most improvements are worth a few percent, a 100× speedup for one array is close to unique.

**Why "one extra buffer" is the honest cost.** $`v`$ has exactly as many entries as $`\theta`$. For a 1-billion-parameter model in 32-bit floats, that is 4 GB of extra memory — real, but a straightforward trade. (Adam, in Chapter 5, keeps *two* such buffers, which is why optimizer state is often the largest single consumer of memory in large-scale training.)

### Nesterov accelerated gradient

Evaluate the gradient at the *look-ahead* point:
$$v_{t+1} = \beta v_t + \nabla\mathcal{L}(\theta_t - \eta\beta v_t),\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}$$

**Intuition:** heavy-ball looks at where it is and then jumps; Nesterov jumps first and then corrects. If momentum is about to overshoot, Nesterov feels the upward slope *before* committing, and brakes.

Nesterov achieves the optimal $`O(1/T^2)`$ rate for smooth convex problems (vs $`O(1/T)`$ for GD), and this is provably optimal for any first-order method (Nemirovski–Yudin lower bound). Heavy-ball achieves $`\sqrt\kappa`$ acceleration only for quadratics; Nesterov achieves it in general.

**Practical note:** the difference between heavy-ball and Nesterov in deep learning is small (a few percent). The difference between momentum and no momentum is enormous.

#### Nesterov's look-ahead, decoded

$$v_{t+1} = \beta v_t + \nabla\mathcal{L}(\theta_t - \eta\beta v_t),\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}$$

The **only** difference from heavy-ball is *where the gradient is measured.* Heavy-ball evaluates $`\nabla\mathcal{L}`$ at $`\theta_t`$ — where you currently stand. Nesterov evaluates it at $`\theta_t - \eta\beta v_t`$ — **where your existing momentum is about to carry you anyway, before today's gradient is even consulted.**

> **Analogy.** You are sprinting down a corridor and a wall is coming. Heavy-ball checks the wall's position from where its feet are and then leaps; if the leap ends past the wall, it discovers this on landing. Nesterov asks "**given my current speed, where will I be in a moment?**" and reads the slope *there* first. If that spot is already up the far wall, it starts braking on this step instead of the next one. **One step of anticipation, and that is the whole idea.**

▸ **Why anticipation helps at exactly the moment it matters.** The dangerous situation is overshoot: momentum has built up and is about to carry you past the bottom and up the other side. Heavy-ball keeps feeling downhill slope right up until the instant it is past the bottom — it gets no warning. Nesterov, evaluating one look-ahead in front, **feels the upslope one step early and applies a correcting gradient before committing.** The correction arrives when it is useful rather than one step late.

**What "optimal" means here, precisely.** The Nemirovski–Yudin lower bound is a statement about the *whole class* of algorithms that only ever look at gradients: **no such method, however clever, can beat $`\mathcal{O}(1/T^2)`$ on smooth convex problems in the worst case.** Nesterov's method attains it. So "optimal" is not a compliment about implementation quality; it means **there is provably nothing left on the table** within that class. Doing better requires more information — curvature (§4.7), structure, or randomness.

| Method | Rate on smooth convex | Rate on strongly convex |
|---|---|---|
| Gradient descent | $`\mathcal{O}(1/T)`$ | $`(1-1/\kappa)^T`$ |
| Heavy-ball | $`\mathcal{O}(1/T)`$ in general | $`(1-1/\sqrt\kappa)^T`$ **for quadratics only** |
| Nesterov | $`\mathcal{O}(1/T^2)`$ — optimal | $`(1-1/\sqrt\kappa)^T`$ in general |

**Why the practical difference is nevertheless small.** The gap between heavy-ball and Nesterov shows up in worst-case guarantees on adversarially-chosen functions. A real deep-learning loss surface is not adversarial, minibatch noise blurs the look-ahead advantage, and PyTorch's `nesterov=True` costs one extra line. Take it if it is free; do not expect a transformation. **The 100× is in having momentum at all, not in which flavour.**

> **Where this came from.** **Yurii Nesterov** published the accelerated method in **1983**, in a Soviet mathematics journal, as a solution to a specific open question: the lower bound said $`\mathcal{O}(1/T^2)`$ ought to be reachable, and nobody had produced a method that reached it. His scheme did. For decades it was regarded as an elegant but somewhat mysterious construction — the proof works, but the algorithm does not obviously *look* like it should. A substantial line of later research exists purely to explain **why** it works, including a well-known interpretation as the discretization of a particular second-order differential equation. **An algorithm can be proved optimal long before anyone can say what it is doing**, and Nesterov acceleration is the standard example.

#### Examples and non-examples: what momentum is

Momentum is the most misdescribed algorithm in deep learning, because its unrolled form $`v_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \dots`$  *is* a weighted moving average — which makes "it smooths the gradient" sound like a complete explanation. It isn't, and the gap between the two descriptions is where the $`100\times`$ lives.

**✅  examples of what momentum does**

| Example | Why it qualifies |
|---|---|
| $`\kappa = 10^4`$: steps per decimal digit fall from $`\approx 2.3\times10^4`$ to $`\approx 2.3\times10^2`$ | The **asymptotic rate** changed, $`1-1/\kappa \to 1-1/\sqrt\kappa`$. This is a different exponent, not a better constant |
| An across-valley gradient alternating $`+g,-g,+g,\dots`$ accumulates to $`g(1-0.9+0.81-\dots) = g/1.9 \approx 0.53g`$, while an along-valley constant $`g`$ accumulates to $`g/0.1 = 10g`$ | A **19× relative reweighting** between agreeing and disagreeing directions, computable exactly from two geometric series. This is the whole mechanism, in two lines of arithmetic |
| Momentum accelerating on a **noiseless** quadratic, where there is nothing to smooth | Proof by elimination that the benefit is not variance reduction: remove all noise and the $`100\times`$ remains |
| $`\beta = 0.9`$ requiring $`\eta`$ to drop roughly $`10\times`$ | $`1/(1-\beta) = 10`$, the geometric-series sum, and it is a prediction you can verify on any run |
| Nesterov's method attaining the $`\mathcal{O}(1/T^2)`$ lower bound for smooth convex problems | It is provably **optimal** among first-order methods. No filter can be optimal for a class it doesn't change the rate on |

**❌ Near-misses — described as momentum, but different objects**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A uniform boxcar average of the last 10 gradients, then a plain step |  smoothing. It reduces variance and leaves the rate at $`1-1/\kappa`$; the $`\sqrt{\ }`$ never appears | A moving-average filter |
| "Momentum reduces gradient noise" | It does average $`\approx 1/(1-\beta)`$ gradients, so variance does drop — but that is a side effect, absent on the deterministic problems where momentum shines most | A real but incidental benefit |
| "$`\beta = 0.99`$ is a bit more momentum than $`\beta = 0.9`$" | $`1/(1-\beta)`$ goes $`10 \to 100`$. The memory horizon **and** the effective step size both multiply by ten | A $`10\times`$ change wearing a $`1\%`$ costume |
| EMA of the **weights** (§4.8) | Acts on the trajectory after the fact; the update rule itself is untouched | Weight averaging (SWA/EMA) — a different object at a different point in the pipeline |
| "Momentum stabilizes training, so you can raise $`\eta`$" | It multiplies the effective step by $`1/(1-\beta)`$, so the nominal $`\eta`$ must come **down** | The single most common cause of "I added momentum and it diverged" |
| The textbook form $`v \leftarrow \beta v + (1-\beta)g`$ compared with PyTorch's $`v \leftarrow \beta v + g`$ | The first is a true EMA with $`v_\infty = g`$; the second has $`v_\infty = g/(1-\beta)`$ | The same algorithm at two learning rates differing by $`10\times`$ — check which one your framework uses |

▸ **The boundary:** smoothing changes the **variance** of the update; momentum changes the **rate**, $`1-1/\kappa \to 1-1/\sqrt\kappa`$. The mechanism is that feeding the average back into the state makes the recursion **second-order** — mathematically a mass on a spring — and the two roots of its characteristic polynomial multiply to $`\beta`$, so each has magnitude $`\sqrt\beta`$. **The square root comes from the feedback, not from the averaging.** Any scheme that filters the gradient and then takes a first-order step cannot produce it.

> **Common misconception.** *"Momentum is just an exponential moving average of the gradient."* The unrolled form is a weighted average, so the description isn't wrong — it is incomplete in the one place that matters. A filter applied to gradients before an ordinary step gives you a quieter version of the same algorithm, with the same $`1-1/\kappa`$ rate. Momentum *stores* the average in state and updates from it, which makes $`\theta_{t+1}`$ depend on both $`\theta_t`$ and $`\theta_{t-1}`$ — a second-order recursion, with the $`\sqrt{\ }`$ falling out of its characteristic polynomial. **The belief is tempting precisely because the unrolled algebra looks so much like smoothing**, and because the physical name ("heavy ball") suggests inertia, which sounds like a synonym for damping. Polyak's ball is not damping the path; it is *resonating* — tuning the oscillation so that lateral sloshing cancels itself while forward drift accumulates undisturbed.

> **Common misconception.** *"Nesterov is a better momentum, so switch and expect a speedup."* Nesterov's advantage is in worst-case guarantees on adversarially chosen functions, and heavy-ball's known failure case (Lessard–Recht–Packard, 2016) is a constructed counterexample, not a description of a loss surface. On a real deep network with minibatch noise blurring the look-ahead, the difference is small enough to be hard to measure. **The belief is tempting because "provably better" reads as "better," full stop**, and because `nesterov=True` costs one keyword argument. Take it — it is free. Expect nothing. **The $`100\times`$ is in having momentum at all, not in which variety.**

---

## 4.6 Stochastic gradient descent

### The one-line idea

Estimate the gradient from a minibatch instead of the full dataset — accept noise in exchange for doing $`n/B`$ times more steps per epoch.

### The analogy

Rather than polling every voter in the country before making each decision, poll 64 people. Your direction is noisier, but you can make 2,274 decisions in the time it took to make one.

### Setup

$$g_t = \frac1B\sum_{i\in\mathcal{B}_t}\nabla\ell_i(\theta_t),\qquad \mathbb{E}[g_t]=\nabla\mathcal{L}(\theta_t),\qquad \mathrm{Cov}(g_t) = \frac{\Sigma(\theta_t)}{B}$$

**Unbiased but noisy, with variance $`\propto 1/B`$.**

#### Unpacking the minibatch gradient

$$g_t = \frac1B\sum_{i\in\mathcal{B}_t}\nabla\ell_i(\theta_t),\qquad \mathbb{E}[g_t]=\nabla\mathcal{L}(\theta_t),\qquad \mathrm{Cov}(g_t) = \frac{\Sigma(\theta_t)}{B}$$

Read the first expression aloud: *"g at t equals one over B, times the sum over i in batch-t, of grad-little-ell-i at theta-t."* In English: **"pick $`B`$ training examples at random, compute each one's gradient, and average them."**

| Symbol | Read aloud | Meaning |
|---|---|---|
| $`\mathcal{B}_t`$ | "script B at t" | The set of example indices drawn for step $`t`$ — a random subset of the dataset |
| $`B`$ | "B" | How many examples are in it. Typically 32 to 4,096 |
| $`\ell_i`$ | "little ell i" | The loss on **one** example. Lowercase for per-example; $`\mathcal{L}`$ (script L) for the full-dataset average (§0.4) |
| $`\frac1B\sum`$ | "one over B, sum" | An ordinary average. The `for` loop plus a division |
| $`\mathbb{E}[g_t]`$ | "the expectation of g" | The average over all possible batch draws — what you'd get if you repeated the random draw forever |
| $`\mathrm{Cov}(g_t)`$ | "the covariance of g" | How much $`g_t`$ scatters around that average, per coordinate and between coordinates |

**"Unbiased" means precisely this:** $`\mathbb{E}[g_t] = \nabla\mathcal{L}(\theta_t)`$. The minibatch gradient is *not* the true gradient — on any given step it points somewhere else — but it is **not systematically wrong in any direction.** Average enough of them and the errors cancel.

> **Analogy.** A political poll of 1,000 people is not the election result. But it is unbiased: it is not tilted toward one party, it is merely imprecise. Poll ten times and average, and you get closer. **Stochastic gradient descent never bothers to poll ten times — it just acts on each noisy poll immediately, on the reasonable theory that ten cheap decisions beat one expensive one.**

**Why the variance is $`\Sigma/B`$, with numbers.** Averaging $`B`$ independent noisy quantities divides the variance by $`B`$ — the same $`\sqrt{n}`$ law as the standard error in §1.3.1. So the *standard deviation* of the gradient falls as $`1/\sqrt B`$:

| Batch size $`B`$ | Gradient noise (relative) | Cost per step |
|---|---|---|
| 16 | $`1.00`$ | $`1\times`$ |
| 64 | $`0.50`$ | $`4\times`$ |
| 256 | $`0.25`$ | $`16\times`$ |
| 1024 | $`0.125`$ | $`64\times`$ |

▸ **Read those two columns against each other and the entire economics of batch size falls out.** Sixteen times the compute buys you four times less noise. **You are always paying quadratically for linear precision** — which is why "just use a huge batch" is not free progress, and why the correct question is never "what batch size is best?" but "what is the *cheapest* noise level that still trains?"

**What $`\Sigma(\theta)`$ actually is.** It is the covariance of the *per-example* gradients: how much individual training examples disagree about which way is downhill. If every example wants the same update, $`\Sigma \approx 0`$ and a batch of 1 is as good as a batch of 10,000. If examples disagree violently — early in training, or with heterogeneous data — $`\Sigma`$ is large and small batches thrash. Note that $`\Sigma`$ depends on $`\theta`$: **the noise level changes as training proceeds**, typically shrinking as the model fits the easy examples and the remaining disagreement concentrates on the hard ones.

> **Where this came from.** The idea of taking a downhill step using a *noisy* estimate of the slope predates neural networks by decades. **Herbert Robbins and Sutton Monro** published *A Stochastic Approximation Method* in **1951**, and their motivating problem was not machine learning at all: it was experimental design. Suppose you want the dose of a chemical at which a specified fraction of subjects respond, but each experiment gives a noisy yes/no answer. You cannot compute the curve; you can only probe it, noisily, one point at a time. Their answer — nudge the dose in the observed direction with a step size that shrinks on a schedule — **is stochastic gradient descent, invented for bioassay.** The convergence conditions in this section still carry their names.

### The convergence rate and why constant LR plateaus

Repeat the descent-lemma argument with $`\mathbb{E}\|g_t\|^2 = \|\nabla\mathcal{L}\|^2 + \frac{\mathrm{tr}\Sigma}{B}`$:

$$\mathbb{E}[\mathcal{L}(\theta_{t+1})] \le \mathcal{L}(\theta_t) - \eta\|\nabla\mathcal{L}\|^2 + \frac{L\eta^2}{2}\left(\|\nabla\mathcal{L}\|^2 + \frac{\mathrm{tr}\Sigma}{B}\right)$$

▸ The last term **does not vanish as $`\nabla\mathcal{L}\to0`$**. Near a minimum, SGD with fixed $`\eta`$ doesn't converge — it settles into a noise ball of radius

$$\mathbb{E}\|\theta-\theta^*\|^2 \approx \frac{\eta\,\mathrm{tr}\Sigma}{2\mu B}$$

**This is why LR decay exists.** Not because "the optimizer gets tired," but because the *stationary noise floor is proportional to $`\eta/B`$*, and the only way to get below it is to shrink $`\eta`$ or grow $`B`$.

For convergence you need the **Robbins–Monro conditions**:
▸ $$\sum_t \eta_t = \infty \quad\text{(can still travel far)},\qquad \sum_t \eta_t^2 < \infty \quad\text{(noise is summable)}$$
$`\eta_t = \eta_0/t`$ satisfies both. $`\eta_t = \eta_0/\sqrt t`$ satisfies the first but not the second — it converges in the weaker averaged sense with rate $`O(1/\sqrt T)`$, which is optimal for nonsmooth stochastic convex problems.

**Constant $`\eta`$ satisfies neither**, which is exactly your setup. So your model is *not* converging to a point; it is diffusing in a noise ball around a region of low loss. That is a completely standard and often *good* place to be — but it means your parameters at epoch 43 are meaningfully different from those at epoch 42, even if the loss is the same, and it means an **EMA of the weights** (§4.8) will beat any single checkpoint.

#### The noise ball, decoded

This is one of the most practically important paragraphs in the book, and it is easy to skim past. Here it is slowly.

**The step that does the work.** In the deterministic case, the descent lemma gave a term $`\|\nabla\mathcal{L}\|^2`$ that shrinks to zero as you approach the minimum, so progress smoothly stops. With minibatches, the quantity that appears is not $`\|\nabla\mathcal{L}\|^2`$ but $`\mathbb{E}\|g_t\|^2`$, and there is a standard identity for it:

$$\mathbb{E}\|g_t\|^2 = \underbrace{\|\nabla\mathcal{L}\|^2}_{\text{signal}} + \underbrace{\frac{\mathrm{tr}\,\Sigma}{B}}_{\text{noise}}$$

Read aloud: *"the expected squared norm of g equals the squared norm of the true gradient, plus the trace of Sigma over B."* This is the "mean squared equals squared mean plus variance" rule from §1.3, applied to a vector. $`\mathrm{tr}\,\Sigma`$ ("trace of Sigma") is the sum of the diagonal entries — **the total gradient variance summed over every coordinate.**

▸ **Now the crucial observation.** As you reach the minimum, the signal term $`\|\nabla\mathcal{L}\|^2`$ goes to zero. **The noise term does not.** Each example still has its own opinion about which way is downhill even when the *average* opinion is "stay put." So the descent lemma's damage term $`\frac{L\eta^2}{2}\cdot\frac{\mathrm{tr}\Sigma}{B}`$ survives, permanently, and the algorithm never stops moving.

> **Analogy.** Imagine a marble in a shallow bowl, and the bowl is sitting on a table that is being constantly and randomly jiggled. The marble rolls toward the bottom — but it never *rests* at the bottom, because the jiggling keeps kicking it back up the sides. It settles into a statistical equilibrium: **it wanders around inside a small region whose size is set by how hard the table is being shaken and how steep the bowl is.** That region is the noise ball. Shake harder (bigger $`\eta`$, smaller $`B`$) and the ball grows. Use a steeper bowl (bigger $`\mu`$) and it shrinks.

**Reading the radius formula.**

$$\mathbb{E}\|\theta-\theta^*\|^2 \approx \frac{\eta\,\mathrm{tr}\Sigma}{2\mu B}$$

Every symbol in it earns its place: larger step $`\eta`$ → bigger ball (harder kicks). Larger batch $`B`$ → smaller ball (kicks averaged down). Noisier data $`\mathrm{tr}\Sigma`$ → bigger ball. Steeper bowl $`\mu`$ → smaller ball (stronger restoring force). ▸ **The single actionable consequence: the only two knobs you actually control are $`\eta`$ and $`B`$, and they appear only as the ratio $`\eta/B`$.** Halving $`\eta`$ and halving $`B`$ leaves you exactly where you were.

**The Robbins–Monro conditions, in English.**

| Condition | Reads as | Why you need it |
|---|---|---|
| $`\sum_t \eta_t = \infty`$ | "the step sizes add up to infinity" | You must be able to travel an **unlimited total distance**. If steps shrink too fast, you run out of road before reaching the optimum — like a car whose fuel is rationed geometrically |
| $`\sum_t \eta_t^2 < \infty`$ | "the squared step sizes add up to something finite" | The accumulated *noise* must be finite. Noise enters through $`\eta^2`$ (from the descent lemma's quadratic term), so a finite sum means the total jitter injected over infinite time is bounded |

▸ **Together they say: shrink your steps, but not too fast.** Too fast and you stall short of the answer; too slow and the noise never settles. $`\eta_t = \eta_0/t`$ threads the needle: $`\sum 1/t`$ diverges (the harmonic series — famously, barely), while $`\sum 1/t^2`$ converges (to $`\pi^2/6`$). Those two facts, one from each side, are exactly the two conditions.

**Why this is not an academic point for you.** Constant $`\eta`$ satisfies neither, so a model trained at constant learning rate is *by construction* not converging. Three consequences you can act on today:

1. **Two consecutive checkpoints with identical loss can have  different weights.** They are two random draws from the same ball.
2. **Comparing single checkpoints across runs measures ball position as much as ball quality.** This is the checkpoint-selection bias of Chapter 3, arriving from the optimizer's side.
3. **Averaging the iterates (§4.8) moves you toward the ball's centre**, which is closer to $`\theta^*`$ than almost any individual point in it. That is not a trick — it is the direct consequence of the marble wandering symmetrically around a low point.

### SGD as an SDE — the temperature

For small $`\eta`$, SGD is well approximated by the stochastic differential equation

▸ $$d\theta = -\nabla\mathcal{L}(\theta)\,dt + \sqrt{\frac{\eta}{B}\Sigma(\theta)}\,dW$$

The stationary distribution (for isotropic $`\Sigma = \sigma^2 I`$) is approximately Gibbs:
$$p(\theta) \propto \exp\left(-\frac{2\mathcal{L}(\theta)}{T}\right),\qquad \boxed{T = \frac{\eta\sigma^2}{B}}$$

▸ **The learning-rate-to-batch-size ratio $`\eta/B`$ is a temperature.** High temperature ⇒ the walker escapes narrow basins and settles in wide ones. This is the mechanistic core of "SGD prefers flat minima" (Ch. 19), and it immediately explains:

- **The linear scaling rule** (Goyal et al.): if you multiply $`B`$ by $`k`$, multiply $`\eta`$ by $`k`$ to hold temperature fixed. Works up to a critical batch size.
- **Large-batch generalization gap:** big $`B`$ at fixed $`\eta`$ = low temperature = sharp minima = worse generalization.
- Why "just use a bigger batch, it's faster" often silently costs you accuracy.

**Numbers for Case Study A:** $`\eta=3\times10^{-4}`$, $`B=64`$, so $`\eta/B = 4.7\times10^{-6}`$. If you moved to $`B=256`$ for speed, holding accuracy would want $`\eta \approx 1.2\times10^{-3}`$ (linear rule) or $`\eta\approx6\times10^{-4}`$ (square-root rule, often better for Adam). Changing batch size without changing LR is changing two things at once.

#### The temperature $`\eta/B`$, decoded

This is the deepest idea in the chapter, and the notation is the least familiar. Take it in three passes.

**Pass 1 — what a stochastic differential equation is.** Read

$$d\theta = -\nabla\mathcal{L}(\theta)\,dt + \sqrt{\tfrac{\eta}{B}\Sigma(\theta)}\,dW$$

aloud as: *"the change in theta equals minus the gradient of L times the change in time, plus the square root of eta-over-B times Sigma, times d-W."* In English: **"in each instant, drift downhill a little, and get kicked in a random direction a little."**

| Piece | Read aloud | Job |
|---|---|---|
| $`d\theta`$ | "d theta" | An infinitesimal change in the parameters — one very small step |
| $`-\nabla\mathcal{L}(\theta)\,dt`$ | "minus grad L, d t" | The **drift**: the deterministic downhill pull. This is ordinary gradient descent, written continuously |
| $`dW`$ | "d W" | A **random kick** drawn fresh each instant, from a standard Gaussian. ($`W`$ is a Wiener process — Brownian motion. Nothing more exotic than "pollen jiggling in water") |
| $`\sqrt{\frac{\eta}{B}\Sigma}`$ | "root eta over B Sigma" | How **hard** the kicks are, and in which directions they prefer to point |

▸ **The whole equation says: gradient descent with minibatches is a particle drifting downhill while being randomly buffeted.** The discrete algorithm and the continuous equation agree well as long as $`\eta`$ is small, which is why this picture is trustworthy in practice and not merely poetic.

**Pass 2 — what "stationary distribution" means.** The particle never settles at a point, so asking "where does it end up?" is the wrong question. The right question is **"how often is it found where?"** — a probability distribution over parameter space. That is the stationary distribution, and it turns out to be

$$p(\theta) \propto \exp\left(-\frac{2\mathcal{L}(\theta)}{T}\right),\qquad T = \frac{\eta\sigma^2}{B}$$

Read: *"the probability of finding the particle at theta is proportional to e-to-the-minus-two-L-of-theta-over-T."* $`\propto`$ means "proportional to" — equal up to a constant that only ensures the probabilities sum to 1 (§0.9). ▸ **Low-loss regions are exponentially more likely to be occupied than high-loss ones, and $`T`$ controls how much more.**

**Pass 3 — why $`T`$ is called a temperature, and why that is not a metaphor.** This is *literally* the **Boltzmann distribution** from statistical physics, where $`p \propto e^{-E/kT}`$ with $`E`$ the energy and $`T`$ the temperature (the same distribution that reappears as the softmax in §1.3.4). Loss plays the role of energy; $`\eta/B`$ plays the role of temperature.

> **Analogy.** Shake a tray of ball bearings with dimples of varying width and depth. **Shake hard (high temperature)** and the bearings hop constantly; they cannot stay in a narrow dimple, because any narrow dimple's walls are easy to jump out of. Over time they collect in the **wide, shallow basins**, which are simply harder to leave by accident. **Shake gently (low temperature)** and each bearing settles into whatever dimple it happened to be nearest, however narrow. The tray has not changed. Only the shaking has, and it has completely changed which minima are found.

▸ **Three practical facts drop straight out of this one picture:**

- **Linear scaling rule.** Temperature is $`\eta/B`$. Multiply $`B`$ by $`k`$ and the temperature falls by $`k`$ — so multiply $`\eta`$ by $`k`$ to restore it. **The rule is not empirical folklore; it is "hold the temperature fixed."** (It breaks down past a critical batch size, where the gradient becomes essentially noise-free and there is no noise left to preserve.)
- **The large-batch generalization gap.** Increase $`B`$ and forget to raise $`\eta`$: you have *cooled* the system. Cold systems settle into narrow basins. Narrow basins generalize worse. **The gap is not caused by large batches; it is caused by accidentally lowering the temperature while changing something else.**
- **Why the noise is not a bug.** Minibatch noise is the only thing preventing your model from settling into the first narrow crevice it meets. Remove it — perfectly, as SVRG tries to — and you remove the exploration that finds wide basins.

**Numbers, so the scale is concrete.** At $`\eta = 3\times10^{-4}`$ and $`B=64`$, the temperature is $`4.7\times10^{-6}`$. Moving to $`B=256`$ without touching $`\eta`$ divides the temperature by 4 — the same as quartering the learning rate, which nobody would do accidentally, yet it is exactly what "I doubled the batch twice to use the GPU better" does. **Changing batch size is changing the learning rate. There is no way to change one without the other unless you deliberately compensate.**

> **Where this came from.** The **linear scaling rule** was made famous by a 2017 paper from **Priya Goyal and colleagues at Facebook AI Research**, *Accurate, Large Minibatch SGD*, which trained a ResNet-50 on ImageNet in one hour by scaling to a batch of 8,192 — combining linear LR scaling with a gradual warmup. An earlier note by **Alex Krizhevsky** in 2014 had argued for a **square-root** scaling instead, on the grounds that it holds the *variance of the update* fixed rather than the temperature. Both rules are still in use, and which is right depends on which quantity you believe matters; the square-root rule is generally the better default for Adam. The **large-batch generalization gap** was documented carefully by **Nitish Keskar and colleagues in 2017**, who tied it explicitly to the sharpness of the minima that large-batch training finds — one of the papers that put "flat minima" into the working vocabulary of the field.

### Variance reduction (why it doesn't help in deep learning)

SVRG maintains a snapshot $`\tilde\theta`$ and uses $`g_t = \nabla\ell_i(\theta_t) - \nabla\ell_i(\tilde\theta) + \nabla\mathcal{L}(\tilde\theta)`$, which is unbiased with variance $`\to0`$ as $`\theta_t\to\tilde\theta`$. It gives linear convergence for strongly convex finite sums.

▸ **It reliably fails to help deep networks.** Reasons: (1) data augmentation makes the "finite sum" not finite; (2) the snapshot goes stale within a few hundred steps because $`\theta`$ moves fast; (3) most importantly, **the noise is doing useful work** (it's the temperature above) — removing it removes the implicit regularization. A rare case where a theoretically superior algorithm is practically worse *because* the theory optimized the wrong objective.

#### What SVRG is doing, and why the noise turns out to be load-bearing

**SVRG** stands for **Stochastic Variance Reduced Gradient**. The trick is a **control variate** — a standard statistical device for making a noisy estimate less noisy without making it biased.

$$g_t = \underbrace{\nabla\ell_i(\theta_t)}_{\text{noisy, want this}} - \underbrace{\nabla\ell_i(\tilde\theta)}_{\text{noisy, know its average}} + \underbrace{\nabla\mathcal{L}(\tilde\theta)}_{\text{that average, exactly}}$$

Read it as: *"take one example's gradient here, subtract that same example's gradient at a snapshot point, and add back the exact full-dataset gradient at the snapshot."*

> **Analogy.** You want to know today's temperature in a city, and you have one thermometer at one street corner — noisy and unrepresentative. But you also have last week's reading from that *same* corner, and last week's official city-wide average. So you compute *"this corner today, minus this corner last week, plus the whole city last week."* The corner's personal quirks — it's in a wind tunnel, it's next to a vent — appear in both of the first two terms and **cancel.** What survives is a much cleaner estimate of the city-wide change. **The subtraction removes the part of the noise that is a stable property of the example rather than of the moment.**

**Why it is unbiased.** The second and third terms have the same expectation ($`\mathbb{E}_i[\nabla\ell_i(\tilde\theta)] = \nabla\mathcal{L}(\tilde\theta)`$), so they cancel *in expectation* and leave $`\mathbb{E}[g_t] = \nabla\mathcal{L}(\theta_t)`$ — still correct on average. And as $`\theta_t \to \tilde\theta`$, the first two terms cancel *exactly*, driving the variance to zero. That is the beautiful part: **near the snapshot, the estimator becomes noise-free**, and you recover the fast linear convergence of full-batch gradient descent at the cost of single-example steps.

▸ **And it does not help deep networks. Read the three reasons as three different lessons:**

| Reason | The general lesson |
|---|---|
| Data augmentation means the "finite sum" is not finite — every epoch sees a *different* crop, flip, and noise draw | **Theory assumptions are load-bearing.** "Finite sum" was not a technicality; without it, the whole construction is unfounded |
| The snapshot $`\tilde\theta`$ goes stale within a few hundred steps, because $`\theta`$ moves fast | **Timescales matter.** A method built for slow, careful convergence is mismatched to a system that traverses its own landscape rapidly |
| The noise was doing useful work — it *is* the temperature of the previous section | **Check what you are removing.** Variance was never the enemy; it was the exploration mechanism |

▸ **The third reason is the one worth carrying with you.** The theory measured success by "how fast does the training loss fall," and by that measure SVRG  wins. But nobody deploys training loss. **When an algorithm that provably optimizes your stated objective performs worse in practice, the usual explanation is that your stated objective was not your real one** — here, generalization was, and the discarded noise was quietly providing it.

#### Examples and non-examples: gradient noise, and what the batch size buys

Every formula in this section contains $`\eta`$ and $`B`$ only as the ratio $`\eta/B`$. That single algebraic fact settles most arguments about batch size before they start.

**✅  consequences of the noise analysis**

| Example | Why it qualifies |
|---|---|
| $`B: 256 \to 1024`$ with $`\eta`$ fixed shrinks the noise-ball radius² by exactly $`4\times`$ | $`\mathbb{E}\|\theta-\theta^*\|^2 \approx \eta\,\mathrm{tr}\Sigma/2\mu B`$ — $`B`$ appears in the denominator, linearly |
| $`B: 256 \to 1024`$ **and** $`\eta: 10^{-3} \to 4\times10^{-3}`$ leaves the statistics identical while quartering the step count | $`\eta/B`$ unchanged. This is the linear scaling rule, and it is a corollary rather than a heuristic |
| Two checkpoints 500 steps apart with identical loss but measurably different weights | Two draws from the same equilibrium distribution. Exactly what the marble-in-a-jiggled-bowl picture predicts |
| Dropping $`\eta`$ by $`10\times`$ late in training causes a visible cliff in the loss curve | You shrank the ball's radius by $`10\times`$; the loss floor it was orbiting drops with it |
| Averaging the last $`k`$ iterates beating every individual iterate | The marble wanders roughly symmetrically about a low point, so the mean of the wander is closer to it than the wander is (§4.8) |

**❌ Near-misses — plausible statements about noise that don't survive the formula**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "Bigger batch means a better gradient, so bigger is always better" | Variance falls as $`1/B`$, so the **standard deviation** falls as $`1/\sqrt B`$: $`4\times`$ the compute buys $`2\times`$ less gradient noise. And it silently lowers $`\eta/B`$ | Diminishing returns, plus an unintended temperature change |
| "Full-batch GD is SGD with the noise removed, so it must be at least as good" | It also removes the exploration term, and §4.6's variance-reduction discussion is a whole subsection on that backfiring | A different algorithm with a different implicit bias |
| "My gradients are noisy because my labels are noisy" | $`\Sigma`$ is the covariance of **per-example gradients**. On perfectly clean labels, a cat photo and a truck photo still pull the weights in different directions | Disagreement between examples — present even on noiseless data |
| "Variance reduction (SVRG) provably converges faster, so it should help here" | It does converge faster on the training loss, and it performs worse in deep learning, for the three reasons listed above | An algorithm optimizing a stated objective that wasn't the real one |
| "$`\eta`$ and $`B`$ are two independent knobs, so tune them separately" | They enter the dynamics as one number. Halving both is a no-op | One statistical knob ($`\eta/B`$) and one hardware knob ($`B`$) |
| "Constant learning rate, loss is flat, therefore converged" | Constant $`\eta`$ satisfies neither Robbins–Monro condition, so the run is **by construction** not converging | An orbit at equilibrium radius |

▸ **The boundary:** $`B`$ has two independent jobs and they are constantly conflated. Statistically, it appears only inside $`\eta/B`$ — so changing $`B`$ alone changes the temperature, and changing $`B`$ with $`\eta`$ in proportion changes nothing at all. Computationally, it sets how much work you do per step. **Choose $`B`$ for the hardware; choose $`\eta`$ to put $`\eta/B`$ where you want it.**

> **Common misconception.** *"Gradient noise is a defect of minibatching that we tolerate for speed."* The noise is doing work. It sets the temperature $`\eta/B`$ in §4.6's stochastic differential equation, and that temperature determines which basin you settle in — which is a statement about generalization, not about optimization. The cleanest evidence is SVRG: an algorithm designed specifically to remove this noise, with better convergence theory, that performs *worse* on deep networks. **The belief is tempting because in every other estimation problem you have met, variance is unambiguously bad** — a noisy thermometer is a worse thermometer. Here the estimator's noise feeds back into the search, and a search that never makes a mistake never leaves the first valley it finds.

---

## 4.7 Second-order and quasi-Newton methods

**Newton's method:** $`\theta_{t+1} = \theta_t - H^{-1}g_t`$. Converges quadratically near the optimum and is **affine invariant** — $`\kappa`$ disappears entirely. Cost: $`O(p^3)`$ to invert. Dead on arrival for $`p=10^8`$.

**Natural gradient:** replace $`H`$ with the Fisher information $`F = \mathbb{E}[\nabla\log p_\theta \nabla\log p_\theta^\top]`$. This makes the step invariant to *reparameterization of the model's distribution* rather than of the parameters — you're doing steepest descent in KL-divergence geometry, not Euclidean geometry.

▸ $$\Delta\theta^* = \arg\min_{\Delta\theta} \mathcal{L}(\theta+\Delta\theta)\ \text{ s.t. }\ \mathrm{KL}(p_\theta\|p_{\theta+\Delta\theta})\le\epsilon \implies \Delta\theta \propto -F^{-1}g$$

This is the mathematical foundation of TRPO (Ch. 17).

**K-FAC** approximates $`F`$ as a Kronecker product per layer: $`F_\ell \approx A_{\ell-1}\otimes G_\ell`$ where $`A`$ is the input covariance and $`G`$ the output-gradient covariance. Inversion becomes $`(A\otimes G)^{-1} = A^{-1}\otimes G^{-1}`$, cost $`O(d^3)`$ per layer instead of $`O(d^6)`$.

**Shampoo / Muon** are the modern descendants — preconditioners built from $`\left(\sum g g^\top\right)^{-1/4}`$ per tensor mode. Muon in particular orthogonalizes the momentum matrix (via Newton–Schulz iteration for the "sign of a matrix"), which is a spectral-norm-controlled step. These are currently the strongest challengers to AdamW at scale (Ch. 5).

**L-BFGS** builds an implicit inverse-Hessian from the last $`m`$ (gradient, step) pairs. Excellent for deterministic full-batch problems (it's what `scipy.optimize` uses). **Poor with minibatch noise** — the curvature pairs are corrupted, and the line search it needs is impractical.

#### Second-order methods, decoded

Everything in this section is one idea in five costumes: **stop using a single global step size, and instead rescale space so the valley becomes round.**

**Newton's method, and why $`\kappa`$ vanishes.** $`\theta_{t+1} = \theta_t - H^{-1}g_t`$. Read: *"subtract H-inverse times g."* The gradient says *which way* is downhill; the inverse Hessian says *how far to go in each direction given how sharply it curves.* Divide by curvature and every direction is treated on equal terms.

**Do it in one dimension and it's obvious.** $`\mathcal{L}(\theta) = \frac12 a\theta^2`$ gives $`g = a\theta`$ and $`H = a`$. Then $`\theta_{t+1} = \theta_t - \frac{a\theta_t}{a} = 0`$. **One step, exactly the minimum, for any $`a`$ whatsoever.** The sharpness cancelled. That cancellation, happening independently in every eigendirection, is what "affine invariant" and "$`\kappa`$ disappears" mean.

> **Analogy.** Gradient descent is navigating a city with a rule like "always walk 100 metres per step." Down a wide boulevard that is far too timid; in a narrow alley it puts you through a wall. Newton's method **redraws the map** so that every street is the same width, then walks a sensible distance on the redrawn map. The territory did not change; the units did.

▸ **And it is unusable at scale, for a reason worth stating in raw numbers.** $`H`$ is $`p\times p`$. At $`p = 10^8`$ parameters, $`H`$ has $`10^{16}`$ entries — around 40 million gigabytes in 32-bit floats — and inverting it costs $`\mathcal{O}(p^3) = 10^{24}`$ operations. A modern accelerator doing $`10^{15}`$ operations per second would need about $`10^9`$ seconds — **roughly thirty years, for a single step.** This is not an engineering problem to be optimized away. **Everything else in this section is a way to get some of Newton's benefit without ever forming $`H`$.**

**Natural gradient, decoded.** The Fisher information matrix $`F = \mathbb{E}[\nabla\log p_\theta \nabla\log p_\theta^\top]`$ is an outer product (§0.8) averaged over data: an "average of gradient-of-log-likelihood times its own transpose." What makes it interesting is what it measures.

$$\Delta\theta^* = \arg\min_{\Delta\theta} \mathcal{L}(\theta+\Delta\theta)\ \text{ s.t. }\ \mathrm{KL}(p_\theta\|p_{\theta+\Delta\theta})\le\epsilon \implies \Delta\theta \propto -F^{-1}g$$

Read aloud: *"the best update is the one that reduces the loss most, subject to the constraint that the Kullback–Leibler divergence between the old and new model distributions is at most epsilon."* In English: **"change the model's *behaviour* by a bounded amount, and within that budget, improve as much as you can."**

> **Analogy.** Ordinary gradient descent limits how far the **dials** move. Natural gradient limits how far the **output** moves. Two dials on a mixing desk might have the same physical travel while one barely alters the sound and the other makes it unrecognizable; a sound engineer thinks in decibels of change, not millimetres of knob. ▸ **Natural gradient is steepest descent measured in "how different does the model behave" rather than "how far did the numbers move," and that is a better notion of distance because it does not depend on how you happened to parameterize the model.**

This is why it underpins trust-region policy methods in reinforcement learning: there, a small parameter change can catastrophically alter a policy, and the quantity you actually need to keep small is the behavioural change.

**K-FAC and the Kronecker product.** $`A \otimes G`$ ("A Kronecker G") means: **take $`A`$, and replace each of its entries $`a_{ij}`$ by the whole matrix $`a_{ij}G`$.** A $`d\times d`$ Kronecker a $`d\times d`$ gives $`d^2 \times d^2`$ — exactly the size of the Hessian block for a $`d\times d`$ weight matrix.

The magic is the identity $`(A\otimes G)^{-1} = A^{-1}\otimes G^{-1}`$: **inverting the big matrix reduces to inverting the two small ones.** Numbers: $`d = 1024`$ gives $`d^2 = 10^6`$ parameters in the layer, so the exact block would be $`10^6\times10^6`$ and cost $`10^{18}`$ to invert. Two $`1024\times1024`$ inversions cost about $`2\times10^9`$ — **a saving of roughly $`5\times10^8`$**, in exchange for assuming the curvature factorizes into "input side" times "output side." That assumption is wrong, but usefully close.

**L-BFGS, decoded.** **BFGS** is named for four people; **L** is for **Limited-memory**. The idea: you cannot store $`H^{-1}`$, but you *can* remember the last $`m`$ pairs of (how the parameters changed, how the gradient changed). Each pair is a measurement of curvature **along one direction** — if the gradient shifted a lot for a small parameter move, that direction is sharp. Keep $`m \approx 10`$ such pairs and you have a rank-10 sketch of the curvature, at the cost of 10 extra buffers rather than $`p^2`$.

▸ **Why minibatch noise destroys it.** L-BFGS infers curvature by *differencing two gradients.* If both gradients carry noise of comparable size to the real difference, the inferred curvature is mostly noise — and you are dividing your update by a noisy estimate, which is far more dangerous than adding noise to it. **Differencing is a noise amplifier**, and that single fact explains why deep learning's second-order methods (K-FAC, Shampoo, Muon) all estimate curvature from *accumulated averages* rather than from differences.

> **Where this came from.** **BFGS** is the field's best story about simultaneous discovery: **Charles Broyden, Roger Fletcher, Donald Goldfarb, and David Shanno** each published the same update rule, independently, in **1970**. None was aware of the others until afterwards, and the algorithm ended up carrying all four surnames — which is why it looks like a law firm. The **limited-memory** variant that makes it usable on large problems is due to **Jorge Nocedal** in 1980. The **natural gradient** is **Shun-ichi Amari's**, from 1998, and it comes out of *information geometry* — a research programme that treats families of probability distributions as curved surfaces with their own notion of distance. Amari's argument was that the Fisher information *is* the correct metric on that surface, and that steepest descent should therefore be measured with it. **K-FAC** is **James Martens and Roger Grosse**, 2015, and **Shampoo** is **Vineet Gupta, Tomer Koren, and Yoram Singer**, 2018 — the name is a pun on *preconditioner*.

#### Examples and non-examples: stationary points, saddles, and "getting stuck"

Now that the Hessian is on the table, the vocabulary of §4.1 can be made precise. $`\nabla\mathcal{L} = 0`$ tells you that you have stopped; **the eigenvalues of $`H`$ tell you what you have stopped on.** Those are different measurements, and only the first one is cheap.

**✅  examples, classified by the Hessian**

| Example | Why it qualifies |
|---|---|
| Bottom of $`\mathcal{L} = \frac12\theta^\top H\theta`$ with $`H = \mathrm{diag}(100, 0.01)`$ | All eigenvalues positive $`\Rightarrow`$ a strict local minimum. Also global, because the problem is convex |
| A point with $`H = \mathrm{diag}(3, -0.5)`$ and $`\nabla\mathcal{L} = 0`$ | A **saddle**: a valley along one axis, a ridge along the other. Stationary, and not a minimum |
| A plateau where every $`\lvert\lambda_i\rvert < 10^{-6}`$ and $`\|\nabla\mathcal{L}\| = 10^{-8}`$ | Nearly flat in every direction. Not a minimum, not a saddle — a region where the gradient signal has simply run out |
| Two deep-network checkpoints with identical loss but permuted hidden units |  distinct points in parameter space representing the **same function**. A network with 1,024 hidden units in a layer has $`1024!`$ such copies — more than $`10^{2600}`$ |
| A direction of negative curvature found at a stalled checkpoint | Good news, and worth stating: it means a descent direction exists and you are not at a minimum at all |

**❌ Near-misses — the standard diagnoses that are usually wrong**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "Training is stuck in a local minimum" | For $`\theta`$ to be a local minimum, **all** $`p \approx 10^8`$ eigenvalues must be $`\ge 0`$. A crude coin-flip heuristic puts that at $`2^{-10^8}`$; the serious random-matrix analyses (Dauphin and colleagues, 2014) reach the same qualitative conclusion — high-loss critical points in high dimensions are overwhelmingly saddles | A **saddle or a plateau**: a descent direction exists, it is just very shallow |
| "The gradient norm is $`0.001`$, so we're at a critical point" | $`0.001`$ compared with what? A gradient of that size spread over $`10^8`$ coordinates is a per-coordinate signal of $`10^{-7}`$ | An unnormalized number. Compare against the gradient norm at initialization, or against $`\|\theta\|`$ |
| "$`H`$ has a negative eigenvalue, so the loss is unbounded below here" | Curvature is local. A negative eigenvalue says "downhill that way, right now" | A descent direction — the thing you were looking for |
| "This run ended at a higher training loss, so it found a worse minimum" | Different loss on the training set is not different quality, and the two runs may be in the same basin up to a permutation of units | A comparison that has to be made on held-out data (Ch. 2, Ch. 3) |
| "SGD's noise is what lets it escape local minima" | The noise does matter, but what it selects between is **flat versus sharp basins** via the temperature $`\eta/B`$ — a different phenomenon from escaping a trap that mostly isn't there | Basin selection, which is about generalization rather than about getting unstuck |
| "Second-order methods help because they escape local minima" | Newton's method is *attracted* to stationary points of every type, including saddles — it solves $`\nabla\mathcal{L}=0`$ and does not care which solution it gets | Curvature rescaling, which fixes $`\kappa`$, not landscape topology |

▸ **The boundary:** *stationary* is a property of $`\nabla\mathcal{L}`$; *minimum* is a property of the eigenvalues of $`H`$. In two dimensions they nearly coincide, which is why every drawn picture of a loss landscape misleads. In $`10^8`$ dimensions, "every single one of a hundred million curvatures happens to be positive" is an extraordinary coincidence, and **almost everything you will ever call a local minimum is a saddle or a plateau.**

> **Common misconception.** *"Neural networks are hard to train because gradient descent gets trapped in bad local minima."* This was the field's standard explanation through the 1990s, and it is largely wrong for large models. Being at a local minimum requires all $`p`$ eigenvalues of the Hessian to be non-negative simultaneously; in high dimensions, stationary points essentially always have at least one negative direction, and the fraction of negative directions falls as the loss falls. What actually slows training is **saddles and plateaus** — places where the gradient is small in every direction you can cheaply see, but a descent direction still exists. **The belief is tempting because every loss landscape ever drawn is two-dimensional**, and in 2D a bowl and a saddle look about equally likely. The picture is not merely simplified; on this specific question it is actively misleading, and the correction — that dimension makes minima *rarer*, not commoner — is one of the  counterintuitive results in optimization.

> **Common misconception.** *"A flat loss curve means I need a bigger learning rate to escape."* Sometimes. But the three causes of a flat curve want three different responses: a **noise-ball plateau** wants a *smaller* $`\eta`$ (the cliff appears immediately); a **saddle or plateau** wants momentum or preconditioning, since raising $`\eta`$ multiplies a gradient that is near zero and therefore does almost nothing; an **ill-conditioned valley** wants neither, because the slow direction contracts by $`0.9998`$ per step at the largest $`\eta`$ that $`\lambda_{\max}`$ permits. **The belief is tempting because raising $`\eta`$ is the cheapest experiment available** and it occasionally works, which is enough to keep the habit alive. The diagnostic that separates the three: drop $`\eta`$ by $`10\times`$ for 200 steps. If the loss falls, you were in a noise ball. If nothing happens, the landscape is the problem.

---

## 4.8 Two practical things that are worth more than they look

### Gradient clipping

▸ $$g \leftarrow g\cdot\min\left(1, \frac{c}{\|g\|}\right)$$

Clip by **global norm** across all parameters, not per-tensor (per-tensor clipping changes the *direction* of the update, global clipping only its length).

**Why it works:** the smoothness constant $`L`$ isn't uniform. Deep nets have regions ("cliffs") where $`L`$ is enormous, and a normal-size step there overshoots catastrophically. Clipping enforces a trust region: **it is an adaptive step size for the local Lipschitz constant.** Theoretically justified under $`(L_0,L_1)`$-smoothness, where $`\|\nabla^2\mathcal{L}\| \le L_0 + L_1\|\nabla\mathcal{L}\|`$ — which empirically holds for transformers.

Typical $`c`$: 1.0 for transformers. Log $`\|g\|`$ during training; if it's routinely 100× the clip value you have a problem elsewhere.

#### Gradient clipping, decoded

$$g \leftarrow g\cdot\min\left(1, \frac{c}{\|g\|}\right)$$

Read aloud: *"g becomes g times the minimum of one and c over the norm of g."* The $`\min`$ is doing an `if`:

- If $`\|g\| \le c`$ (the gradient is already small), then $`c/\|g\| \ge 1`$, the $`\min`$ picks $`1`$, and **nothing happens.**
- If $`\|g\| > c`$, the $`\min`$ picks $`c/\|g\| < 1`$, which rescales $`g`$ to have length exactly $`c`$.

▸ **So clipping is: "if the gradient is longer than $`c`$, shorten it to $`c`$; otherwise leave it alone."** The **direction is completely untouched** — you multiplied every component by the same positive number. Only the length changes.

> **Analogy.** A voltage regulator. Under normal conditions it is invisible and passes the signal through unchanged. When a spike arrives it caps the output, protecting whatever is downstream. It does not smooth, filter, or alter the shape of the signal — it just refuses to let it exceed a ceiling.

**Why global norm and not per-tensor.** Suppose your model has two parameter tensors and today's gradient is $`(10, 1)`$ across them. Global clipping to $`c=1`$ gives $`(0.995, 0.0995)`$ — **the same direction**, scaled down. Per-tensor clipping to 1 gives $`(1, 1)`$ — **a different direction entirely**, in which the second tensor has been silently promoted to equal importance. ▸ **Global clipping changes your step length; per-tensor clipping changes your step direction.** One of those is a safety mechanism and the other is an unplanned optimizer redesign.

**Why it works at all — the $`(L_0, L_1)`$-smoothness idea.** Section 4.2 assumed a *single* smoothness constant $`L`$ for the entire loss surface. Real networks violate this badly: most of the landscape is gently curved, and a few regions ("cliffs") bend enormously. The relaxed condition $`\|\nabla^2\mathcal{L}\| \le L_0 + L_1\|\nabla\mathcal{L}\|`$ says something more realistic — **the curvature is allowed to be large exactly where the gradient is large.**

Now recall the stability rule: safe steps need $`\eta < 2/L`$. If $`L`$ grows with $`\|\nabla\mathcal{L}\|`$, then the safe step size must *shrink* when the gradient is big — which is precisely the opposite of what plain gradient descent does, since its step length $`\eta\|g\|`$ *grows* with the gradient. **Clipping restores the correct behaviour: in the sharpest, most dangerous regions it holds the step length constant instead of letting it explode.** That is why "it is an adaptive step size for the local Lipschitz constant" is the right description rather than a rationalization.

**A number that makes the danger concrete.** Suppose the loss briefly produces a gradient 1,000× its typical size — one pathological batch, a numerical near-singularity, a rare token. Without clipping, that single step moves the weights 1,000× further than usual, almost certainly out of the basin the model spent hours finding. With $`c = 1.0`$, that step is the same size as every other step. **Clipping is cheap insurance against events that are individually rare and individually fatal.**

> **Where this came from.** Norm-based gradient clipping was introduced by **Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio** in a 2013 paper on the difficulty of training recurrent networks. Their diagnosis was geometric: repeatedly applying the same weight matrix through time produces a loss surface with **cliff-like walls** (the $`\lambda^k`$ effect from §1.1.2), and the trouble is not the direction of the gradient at a cliff — that is fine — but its magnitude, which launches the parameters into a distant, worthless region. Their fix was deliberately the least invasive thing that could work: keep the direction, cap the length. **It was proposed for recurrent networks, and it is now standard in transformer training, which has no recurrence at all** — the cliffs turn out to be a general feature of deep composition, not of recurrence specifically.

### Weight averaging (EMA / SWA)

▸ $$\bar\theta_t = \gamma\bar\theta_{t-1} + (1-\gamma)\theta_t,\qquad \gamma \in [0.999, 0.9999]$$

Because SGD with constant LR *diffuses in a noise ball* (§4.6), the individual iterates are all slightly off-center. Averaging them lands you closer to the center of the basin, which is both lower-loss and flatter.

▸ **For diffusion models specifically, EMA weights are not optional — they are standard practice and typically worth more than any architectural change you're considering.** DDPM, Stable Diffusion, DiT, and essentially every strong diffusion model reports EMA weights. If Case Study A doesn't maintain an EMA, adding one is very likely the highest-return change available to you, and it also solves the checkpoint-selection-bias problem from Ch. 3 (you stop needing to pick a lucky epoch).

With $`\gamma=0.9999`$ and 2,274 steps/epoch, the EMA horizon is $`1/(1-\gamma)=10{,}000`$ steps $`=4.4`$ epochs. That is a sensible setting for Case Study A.

#### Weight averaging, decoded

$$\bar\theta_t = \gamma\bar\theta_{t-1} + (1-\gamma)\theta_t$$

▸ **First, a notation warning.** In Chapter 1 a bar meant *a gradient flowing backwards* ($`\bar y \equiv \partial\mathcal{L}/\partial y`$). **Here it does not.** $`\bar\theta`$ is a plain average of weights. This is the kind of collision Chapter 0 warns about; context is the only guide, and the context here is "there is no backward pass in sight."

Read aloud: *"theta-bar at t equals gamma times theta-bar at t-minus-one, plus one-minus-gamma times theta at t."* In English: **"keep a shadow copy of the weights that is 99.99% its old self and 0.01% the current weights, updated every step."**

- **EMA** = **Exponential Moving Average**. The weights fade exponentially: the parameters from $`k`$ steps ago contribute $`\gamma^k`$.
- **SWA** = **Stochastic Weight Averaging** — the same idea with a flat (uniform) window instead of an exponential one.
- $`\gamma`$ here plays the identical role to $`\beta`$ in momentum: horizon $`= 1/(1-\gamma)`$ steps. $`\gamma = 0.9999`$ means a memory of 10,000 steps.

**The shadow copy is not used for training.** Gradients are computed from $`\theta`$, updates are applied to $`\theta`$, and $`\bar\theta`$ just watches and averages. At evaluation time you *swap it in*. This is why EMA is nearly free to try: it cannot destabilize training, because training never reads it.

> **Analogy.** A long-exposure photograph of a candle flame. Any single frame shows an irregular, flickering shape. Hold the shutter open for ten seconds and you get a smooth, symmetric teardrop — **the shape the flame is "trying" to be**, with the flicker averaged away. §4.6 established that constant-learning-rate SGD is a marble rattling around inside a noise ball; each checkpoint is one frame of the flicker. **The EMA is the long exposure.**

**Why the average is better than any of its inputs, in one line.** Each iterate is $`\theta_t = \theta_{\text{centre}} + \text{noise}_t`$. Average $`N`$ of them and the centre stays put while the noise, being roughly zero-mean and only weakly correlated across a long horizon, shrinks. ▸ **You get closer to the centre of the basin without computing a single extra gradient.** And because the loss surface curves upward away from the centre, "closer to the centre" means "lower loss" — and, by the flat-minima argument, "in a flatter spot" too.

**Why this matters more than it sounds.** It also disposes of a methodological problem from Chapter 3. If you evaluate 43 checkpoints and keep the best, some of that "best" is  quality and some is a lucky position in the noise ball — you have run a 43-way selection on a noisy metric. **Evaluating an EMA instead removes the lottery**: there is one set of weights, it is stable across epochs, and it is not the winner of a noise contest.

**A practical trap worth knowing.** With $`\gamma = 0.9999`$, the EMA needs roughly 10,000 steps before it stops being dominated by its initialization. Evaluate it at step 500 and you are largely evaluating the random initial weights, and it will look terrible. Either start the EMA after a warmup period, or apply a bias correction of the same form as Adam's (Ch. 5) — divide by $`1-\gamma^t`$. **"My EMA is much worse than my raw weights" almost always means "my EMA is younger than its own horizon."**

> **Where this came from.** Averaging the iterates of a noisy stochastic method is a classical result, established independently by **David Ruppert** in 1988 and by **Boris Polyak and Anatoli Juditsky** in 1992 — the same Polyak as the heavy-ball method, twenty-eight years later. Their theorem is striking: averaging the iterates of a *simply-tuned* stochastic approximation scheme achieves the same asymptotic efficiency as much more carefully tuned methods. **You can be sloppy about the trajectory if you are careful about the average.** The technique re-entered deep learning as **Stochastic Weight Averaging** through work by **Pavel Izmailov, Dmitry Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson** in 2018, who showed the averaged solution sits in a flatter region than any of the individual points averaged — connecting a 1980s convergence result to the modern flat-minima story.

---

## 4.9 Summary table

| Method | Update | Rate (strongly cvx) | Memory | Notes |
|---|---|---|---|---|
| GD | $`-\eta g`$ | $`(1-1/\kappa)^T`$ | $`0`$ | baseline |
| Heavy-ball | $`-\eta v`$, $`v=\beta v+g`$ | $`(1-1/\sqrt\kappa)^T`$ | $`p`$ | quadratics only |
| Nesterov | look-ahead gradient | $`(1-1/\sqrt\kappa)^T`$ | $`p`$ | general, optimal |
| SGD | minibatch $`g`$ | noise floor $`\eta\sigma^2/B`$ | $`0`$ | needs decay |
| Newton | $`-H^{-1}g`$ | quadratic | $`p^2`$ | $`O(p^3)`$ solve |
| Natural grad | $`-F^{-1}g`$ | — | $`p^2`$ | KL geometry |
| K-FAC | Kronecker $`F^{-1}`$ | — | $`\sim2pd`$ | practical 2nd order |
| Adam(W) | Ch. 5 | — | $`2p`$ | the default |

#### Reading the summary table

The **Memory** column is the one people skim and then regret skimming. It counts **extra numbers stored per parameter**, where $`p`$ is the parameter count:

| Entry | Read aloud | What it costs at $`p = 10^9`$ (fp32) |
|---|---|---|
| $`0`$ | "zero" | Nothing beyond the weights and the current gradient |
| $`p`$ | "p" | One extra buffer the same size as the model — **4 GB** |
| $`2p`$ | "two p" | Two extra buffers — **8 GB**, which is why Adam's state often exceeds the model itself |
| $`\sim2pd`$ | "about two p d" | K-FAC's two factor matrices per layer; modest, because $`d \ll p`$ |
| $`p^2`$ | "p squared" | $`10^{18}`$ numbers. **Four billion gigabytes.** Not a large number — an impossible one |

▸ **This column is why the "best" algorithms in the Rate column are not the ones anyone uses.** Newton's method has the finest convergence rate on the page and is unusable; Adam has no rate entry at all and trains essentially every model in this book. **Optimization at scale is a memory-bandwidth problem wearing a mathematics costume.**

The **Rate** column compares like with like only in the strongly-convex row of §4.1 — which deep learning is not in. Read those entries as *"how this method behaves in the one setting where we can prove anything,"* and treat the ranking as a strong hint rather than a promise. The em-dashes are honest: for natural gradient and K-FAC applied to real networks, there is no clean rate to quote.

**The "Notes" column contains the actual decision procedure.** "Quadratics only" is the heavy-ball caveat from §4.5. "Needs decay" is the noise ball from §4.6. "$`\mathcal{O}(p^3)`$ solve" is the thirty-years-per-step calculation from §4.7. Every note is a one-line summary of a section you have now read.

---

## Did you know?

- **Gradient descent predates the electronic computer by a century, and was invented for astronomy.** Augustin-Louis Cauchy described it in an 1847 note to the French Academy of Sciences as a way to solve the messy systems of equations that came out of fitting orbits. Every iteration he had in mind was to be performed by hand.

- **The BFGS algorithm is named after four people who all invented it in the same year without knowing about each other.** Broyden, Fletcher, Goldfarb, and Shanno each published the same quasi-Newton update in **1970**. Rather than adjudicate priority, the field simply concatenated the surnames — which is why one of numerical optimization's workhorses is named like a law firm.

- **Momentum's original justification was physics, not statistics.** Boris Polyak's 1964 "heavy ball" method literally models a ball with mass rolling down the loss surface subject to friction, and $`\beta`$ is one minus the friction. The now-standard reading — "momentum is an exponential moving average of gradients" — is a later reinterpretation of the same two lines of algebra.

- **Turning on momentum at $`\beta = 0.9`$ multiplies your effective learning rate by ten.** The steady-state velocity is $`g/(1-\beta)`$, so $`\beta = 0.99`$ multiplies it by a hundred. A large share of "I enabled momentum and it diverged" reports are really "I raised my learning rate by an order of magnitude without noticing."

- **Heavy-ball momentum can fail to converge on problems where the theory says it should.** In 2016, Lessard, Recht, and Packard built a strongly convex, smooth function on which optimally-tuned heavy-ball cycles forever, using analysis tools imported from control theory. The method had been in continuous use for over fifty years.

- **Nesterov's accelerated method was proved optimal decades before anyone could explain why it works.** The 1983 scheme attains the theoretical speed limit for gradient-only methods, and a whole later literature exists purely to answer the question "but what is it *doing*?" — including the now-popular reading of it as a discretized differential equation.

- **Stochastic gradient descent was invented for medical dose-finding.** Robbins and Monro's 1951 paper was about finding the dose at which a specified fraction of subjects respond, when each trial gives only a noisy yes-or-no. Their step-size conditions — sum to infinity, squares sum to something finite — are still the ones quoted in §4.6.

- **The ratio $`\eta/B`$ is a temperature in the literal thermodynamic sense.** SGD's stationary distribution is $`p(\theta)\propto e^{-2\mathcal{L}/T}`$, the Boltzmann distribution from 19th-century statistical mechanics, with the loss playing the role of energy. The same distribution shows up in this book as the softmax, where the temperature knob on a language model is the *same parameter with its original meaning intact*.

- **You cannot change batch size without changing your learning rate.** Since only $`\eta/B`$ matters, quadrupling the batch to use a GPU better silently divides the temperature by four — the same intervention as quartering the learning rate, which nobody would do by accident. The "large-batch generalization gap" is largely this effect, misattributed.

- **Removing the noise from SGD makes deep networks worse.** Variance-reduction methods like SVRG provably converge faster on the training objective and reliably underperform in practice, because the noise they eliminate is the exploration mechanism that finds wide basins. It is one of the cleanest examples in the field of an algorithm optimizing the metric you wrote down instead of the one you meant.

- **A single Newton step on a modern large model would take about thirty years.** Inverting a $`10^8\times10^8`$ Hessian costs on the order of $`10^{24}`$ operations. This is not an implementation problem — it is why every practical "second-order" method in deep learning is a scheme for approximating curvature without ever writing the matrix down.

- **Gradient clipping was invented for recurrent networks and is now standard for transformers, which have no recurrence.** Pascanu, Mikolov, and Bengio proposed it in 2013 after diagnosing that the danger at a loss "cliff" is the gradient's *magnitude*, not its direction. Keeping the direction and capping the length turned out to be a general property of deep composition rather than of recurrence.

- **Averaging your weights is a 1980s theorem that deep learning rediscovered.** Ruppert (1988) and Polyak–Juditsky (1992) proved that averaging the iterates of a crudely-tuned stochastic method matches the efficiency of a carefully-tuned one. Thirty years later, EMA weights are mandatory practice in diffusion models — where they routinely outperform any architectural change under consideration.

---

## Check for Understanding

**Gradient descent's speed is governed by the condition number, momentum buys you a square root of it for the price of one extra buffer, and SGD's minibatch noise is not a defect to be minimized but a temperature $`\eta/B`$ that determines which basin you end up in.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What is the learning rate, and what physically goes wrong if it is too large?** (The answer should mention a number crossing $`-1`$, not "the model gets confused.")
2. **What is the condition number, and why is a long narrow valley harder than a round bowl?** Why does the *sharpest* direction get to dictate the step size for every other direction?
3. **What does the descent lemma buy you?** (Correct shape of answer: it converts a slope you measured *here* into a guarantee about somewhere you have not been yet.)
4. **Why does momentum help in a narrow valley?** Explain it in terms of which shoves cancel and which accumulate, with no algebra.
5. **Why does turning on momentum at $`\beta=0.9`$ require you to lower the learning rate?**
6. **What does Nesterov's look-ahead actually look ahead at, and at which moment does that help?**
7. **Why does stochastic gradient descent with a constant learning rate never converge?** Describe where the model ends up instead, and what sets the size of that region.
8. **Why is $`\eta/B`$ a temperature?** What does raising it do to which minimum you find, and why is that sometimes a good thing?
9. **Why does making the gradient less noisy make deep networks generalize worse?**
10. **Why can't you use Newton's method on a large model?** Give a number, not a feeling.
11. **Why does gradient clipping keep the direction and change only the length — and what breaks if you clip per-tensor instead?**
12. **Why does averaging your weights beat picking your best checkpoint?** Answer in terms of a long-exposure photograph, and say what problem from Chapter 3 it also solves.

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 05 — Optimization II: Adaptive Methods & Schedules](05-optimization-ii-adaptive-methods-and-schedules.md)
