# Chapter 5 — Optimization II: Adaptive Methods & Learning-Rate Schedules

> **Prerequisites:** Ch. 4.
> **This chapter directly answers the two questions from the Case Study A log:** what "adaptive" actually means in AdamW, and what every learning-rate decay technique is.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, $`\odot`$, or $`\lfloor\cdot\rfloor`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. This chapter leans hardest on two of them: $`\odot`$ means "multiply matching entries and keep them separate" (§0.8), and $`\mathbb{E}`$ means "the average of" (§0.5).

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`\odot`$ | "elementwise times" | Multiply matching entries and **keep them separate** — no summing. `*` in NumPy/PyTorch |
| $`g_t^2`$ | "g squared" | Every entry of the gradient squared **individually**, not a dot product |
| $`D_t`$ | "D at t" | A **diagonal preconditioner**: one personal step-size multiplier per parameter |
| $`G_t`$ | "G at t" | AdaGrad's running **sum** of squared gradients. Only ever grows |
| $`m_t`$ | "m at t" | Adam's **first moment** — a decaying average of the gradient itself (its mean) |
| $`v_t`$ | "v at t" | Adam's **second moment** — a decaying average of the gradient *squared* |
| $`\hat m_t,\ \hat v_t`$ | "m-hat, v-hat" | The same two, after **bias correction** for having started at zero |
| $`\beta_1,\ \beta_2`$ | "beta one, beta two" | How long each moment remembers. Defaults $`0.9`$ and $`0.999`$ |
| $`\epsilon`$ | "epsilon" | A tiny constant that stops a division by zero. Secretly an Adam↔SGD dial |
| $`\lambda`$ | "lambda" | **Weight-decay strength** here — not an eigenvalue |
| $`\eta_t`$ | "eta at t" | The learning rate **as a function of the step number** — i.e. a schedule |
| $`T`$ | "T" | Total planned training steps. Schedules need to know where the finish line is |
| $`\gamma`$ | "gamma" | A multiplicative decay factor, e.g. $`\times0.1`$ at a milestone |
| $`\lfloor x\rfloor`$ | "floor of x" | Round **down** to a whole number. Turns a smooth curve into stairs |
| $`t_{\text{warmup}}`$ | "t warmup" | How many steps you spend ramping the learning rate up from near zero |
| $`\rho_t,\ \rho_\infty`$ | "rho at t, rho infinity" | RAdam's estimate of how many samples the second moment has effectively seen |
| $`r_t`$ | "r at t" | RAdam's **rectification factor** — a derived warmup multiplier in $`[0,1]`$ |
| $`\lambda_{\max}(H)`$ | "lambda-max of H" | The **sharpness**: largest curvature of the loss surface at the current point |
| $`\mathrm{sign}(x)`$ | "sign of x" | $`+1`$, $`-1`$, or $`0`$. Throws away magnitude, keeps direction |
| $`\rho`$ | "rho" | SAM's neighbourhood radius — how far it looks for a worse nearby point |
| $`\mathrm{orth}(M) = UV^\top`$ | "orthogonalize M" | Strip a matrix's stretch factors, keep only its rotation (§1.1.3) |
| $`d_{\text{model}}`$ | "d model" | The transformer's hidden width |

**Full forms for the abbreviations in this chapter:**

| Short | Full form |
|---|---|
| AdaGrad | **Ada**ptive **Grad**ient algorithm |
| RMSProp | **R**oot **M**ean **S**quare **Prop**agation |
| Adam | **Ada**ptive **M**oment estimation (not an acronym for a person or a phrase) |
| AdamW | Adam with decoupled **W**eight decay |
| RAdam | **R**ectified **Adam** |
| Lion | **E**vo**L**ved S**i**gn M**o**me**n**tum |
| SGDR | **S**tochastic **G**radient **D**escent with warm **R**estarts |
| WSD | **W**armup–**S**table–**D**ecay (schedule) |
| SAM | **S**harpness-**A**ware **M**inimization |
| SOAP | Shampoo with **A**dam in the **P**reconditioner's eigenbasis |
| EoS | **E**dge **o**f **S**tability |
| SMA | **S**imple **M**oving **A**verage |
| EMA | **E**xponential **M**oving **A**verage |
| LR | **L**earning **R**ate |
| SNR | **S**ignal-to-**N**oise **R**atio |
| RMS | **R**oot **M**ean **S**quare |
| LN / RMSNorm | **L**ayer **N**ormalization / Root-Mean-Square Normalization |
| $`\mu`$P | **M**aximal **U**pdate **P**arameterization |
| BERT | **B**idirectional **E**ncoder **R**epresentations from **T**ransformers |
| KL | **K**ullback–**L**eibler (divergence) |

---

## 5.1 The idea behind adaptivity

### The one-line idea

Different parameters have wildly different gradient scales. Give each parameter its own step size, inferred from its own gradient history.

### The analogy

You're managing a team where one person's estimates are always in dollars and another's in millions of dollars. Rather than shouting the same correction at both, you normalize each person's feedback by how much they typically vary. Someone who always reports huge numbers gets scaled down; someone who whispers gets scaled up.

### The problem being solved

In Ch. 4 we saw everything depends on $`\kappa = \lambda_{\max}/\lambda_{\min}`$. In deep nets, curvature varies by orders of magnitude *across coordinates*: embedding rows for rare tokens get gradients thousands of times smaller than the final layer bias. A single global $`\eta`$ is either too big for one and too small for the other.

▸ Adaptive methods approximate a **diagonal preconditioner**: $`\theta_{t+1} = \theta_t - \eta D_t^{-1}g_t`$ with $`D_t`$ diagonal. This is a cheap ($`O(p)`$) stand-in for $`H^{-1}`$, capturing per-coordinate scale but not cross-coordinate rotation.

#### What "diagonal preconditioner" actually says

$$\theta_{t+1} = \theta_t - \eta D_t^{-1}g_t$$

Read aloud: *"theta at t-plus-one equals theta at t, minus eta times D-inverse times g."* Compare it to plain gradient descent, $`\theta_{t+1} = \theta_t - \eta g_t`$: **the only change is that one matrix has been inserted.** Everything in this chapter is about what to put in that slot.

**"Preconditioner"** is the general word for a matrix you insert to make a badly-shaped problem better-shaped before solving it. **"Diagonal"** means it has entries only on its main diagonal and zeros everywhere else.

▸ **And a diagonal matrix does exactly one thing: it multiplies each coordinate by its own number.** So $`D_t^{-1}g_t`$ is not really a matrix multiply — it is $`p`$ independent scalar divisions, one per parameter, which is why the cost is $`\mathcal{O}(p)`$ rather than $`\mathcal{O}(p^2)`$. In code it is a single elementwise divide. Nobody ever builds the matrix.

| Preconditioner | What it can fix | Memory | Cost per step |
|---|---|---|---|
| $`D^{-1}`$, diagonal (Adam) | Each coordinate's **scale** | $`\mathcal{O}(p)`$ | $`\mathcal{O}(p)`$ |
| $`H^{-1}`$, full (Newton) | Scale **and** rotation — all cross-coordinate structure | $`\mathcal{O}(p^2)`$ | $`\mathcal{O}(p^3)`$ |

> **Analogy.** Chapter 4 compared Newton's method to redrawing the city map so every street is the same width. A diagonal preconditioner is the **budget version**: you cannot redraw the map, but you can put a different speed limit on each street. If a street runs diagonally to the grid — that is, if two parameters interact — you cannot help it. But if the only problem is that some streets are motorways and others are alleys, per-street speed limits solve it completely, for almost no money.

**What "captures scale but not rotation" costs you, concretely.** Take a two-parameter loss whose sharp direction runs at 45° to the axes — the parameters are correlated. Its Hessian has large off-diagonal entries, and a diagonal preconditioner cannot see them: it will apply the same correction to both coordinates and the valley stays diagonal and narrow. **If the valley happens to line up with the coordinate axes, a diagonal preconditioner fixes it perfectly; if it runs at an angle, the diagonal method gets nothing.**

▸ **The bet Adam makes — and it is a bet — is that in real neural networks, most of the conditioning damage is per-coordinate scale rather than cross-coordinate rotation.** That bet is largely right, which is why Adam works, and only partly right, which is why Shampoo and Muon (§5.7) can still beat it by paying more.

**Why the scales  differ this much.** Consider a token appearing once in 100,000. Its embedding row receives a nonzero gradient on roughly one step in a thousand at batch 64, and zero the rest of the time. The final layer's bias receives a gradient on **every** step, from every example. ▸ **A single global $`\eta`$ has to serve both, and there is no value that works: large enough to move the rare embedding is large enough to destroy the bias, and small enough to be safe for the bias means the rare embedding never learns anything at all.** That is the problem adaptivity exists to solve, and it is not a subtle one — the scales differ by orders of magnitude, not percentages.

#### Examples and non-examples: what counts as an adaptive method

**✅  adaptive — the step size differs *per parameter*, derived from observed gradients**

| Example | The per-coordinate quantity it adapts on |
|---|---|
| AdaGrad | $`\sqrt{\sum_{s\le t} g_s^2}`$ — the all-time gradient energy of that coordinate |
| RMSProp | $`\sqrt{\mathrm{EMA}(g^2)}`$ — the recent gradient energy |
| Adam / AdamW | Same as RMSProp, plus a momentum numerator |
| Adafactor | A rank-one factorization of the same second-moment matrix, to save memory |
| Shampoo, Muon | Adapt on gradient *structure*, not just scale — strictly more than diagonal |

**❌ Near-misses — things that change the step size but are not adaptive methods**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A cosine learning-rate schedule | $`\eta_t`$ changes with $`t`$, but **every parameter gets the same $`\eta_t`$** — the shape was decided before training started | A **schedule**: a function of the step counter, not of the gradients |
| Warmup | Same — it is a fixed ramp, blind to what the gradients are doing | A schedule (and a stabilizer; see §5.4) |
| Gradient clipping | Rescales the *whole* gradient vector by one shared factor when it is too big | A safety rail on the update **norm** |
| Plain momentum (heavy-ball / Nesterov) | Changes the *direction* by averaging past gradients; the scaling is still one global $`\eta`$ | Directional smoothing |
| `ReduceLROnPlateau` | It does react to data — but to the *loss*, globally, with one number for all parameters | A **reactive schedule**; §5.3 treats it as a measurement device |
| Per-layer learning rates you set by hand | Not derived from gradients, and fixed for the whole run | Manual tuning — a hyperparameter, not an algorithm |
| Newton's method | Adaptive in the strongest sense, but it is not *diagonal* — it corrects rotation too | A full second-order method, $`\mathcal{O}(p^3)`$ per step |

▸ **The boundary:** a method is adaptive when **the effective step size of parameter $`i`$ is computed from the gradient history of parameter $`i`$.** Anything that hands the same number to all $`p`$ parameters is a schedule, however cleverly that number is chosen.

> **Common misconception.** *"Adam means you don't have to tune the learning rate."* Adam removes the need to tune the learning rate **per parameter**; it does nothing about the global $`\eta`$ multiplying everything. The range of workable $`\eta`$ for Adam is narrower than people expect — the usual $`10^{-3}`$ to $`10^{-4}`$ band spans one order of magnitude, and stepping outside it still diverges or stalls. The misconception is tempting because Adam's *default* $`\eta = 10^{-3}`$ happens to work on an unusually wide range of problems, so many people have  never had to change it. That is luck about defaults, not an absence of the hyperparameter.

> **Common misconception.** *"A diagonal preconditioner is an approximation to the Hessian, so it's a cheap second-order method."* $`\sqrt{\mathbb{E}[g^2]}`$ is **not** an approximation to the diagonal of the Hessian. It is the root-mean-square of the *gradient*, which is a first-order quantity — closest in spirit to the diagonal of the Fisher information / Gauss–Newton matrix, and even that identification only holds under assumptions Adam does not enforce. **Adam is a first-order method with per-coordinate normalization.** The confusion is tempting because dividing by a curvature estimate is exactly what Newton does, and the update *looks* like a Newton step. The two agree on what shape of problem they fix, not on how they measure it.

---

## 5.2 The lineage: AdaGrad → RMSProp → Adam → AdamW

### AdaGrad (2011)

$$G_t = G_{t-1} + g_t^2 \quad\text{(elementwise)},\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\odot g_t$$

**Property:** parameters with historically large gradients get small steps. Excellent for **sparse features** — a rare token's embedding has $`G`$ near zero, so it gets a large step when it finally appears. Provably good regret for online convex optimization.

▸ **Fatal flaw:** $`G_t`$ is a monotonically increasing sum. The effective LR $`\eta/\sqrt{G_t}`$ decays to zero *no matter what*, on a schedule dictated by accumulated history rather than by need. In deep learning this kills training prematurely.

#### Unpacking AdaGrad

$$G_t = G_{t-1} + g_t^2 \quad\text{(elementwise)},\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\odot g_t$$

Read aloud: *"G at t equals G at t-minus-one plus g-squared, elementwise; then theta at t-plus-one equals theta at t, minus eta over root-G-plus-epsilon, elementwise-times g."*

**The word "elementwise" is doing all the work, so be precise about it.** $`g_t^2`$ does **not** mean $`g_t^\top g_t`$ (a single number) or $`g_tg_t^\top`$ (a matrix). It means: take each entry of the gradient and square it, keeping them in a vector of the same length. If $`g = (3, -4, 0.1)`$ then $`g^2 = (9, 16, 0.01)`$. Likewise $`\sqrt{G_t}`$ takes the square root of each entry separately, and $`\odot`$ multiplies matching entries (§0.8).

**Do the whole thing on three parameters.** Suppose over 100 steps, parameter A always sees gradients around $`1.0`$, parameter B around $`0.01`$, and parameter C sees $`0`$ for 99 steps and then $`1.0`$ once.

| Parameter | $`G_{100}`$ | $`\sqrt{G_{100}}`$ | Step it gets when its gradient is $`g`$ |
|---|---|---|---|
| A (loud, constant) | $`\approx 100`$ | $`10`$ | $`\eta g/10`$ — **damped 10×** |
| B (quiet, constant) | $`\approx 0.01`$ | $`0.1`$ | $`\eta g/0.1`$ — **amplified 10×** |
| C (rare, one spike) | $`\approx 1`$ | $`1`$ | $`\eta g`$ — **full-size step on the one occasion it matters** |

▸ **Read row C twice. That is the entire reason AdaGrad was invented.** A parameter that almost never receives a gradient has accumulated almost no history, so when its moment finally comes it gets a full-strength update rather than a whisper. For sparse features — rare words, rare categories, rare users — this is exactly right, and plain gradient descent gets it exactly wrong.

> **Analogy.** A conversation where everyone gets equal *speaking time* rather than equal *airtime per sentence*. The person who talks constantly gets progressively interrupted; the person who has said nothing all meeting is given the floor immediately when they finally raise a hand. **AdaGrad normalizes by how much you have already said.**

**Now the flaw, in numbers.** $`G_t`$ is a **running sum with no forgetting** — it can only grow. If a parameter sees gradients of typical size $`1`$, then $`G_t \approx t`$, so $`\sqrt{G_t} \approx \sqrt t`$ and the effective learning rate is $`\eta/\sqrt t`$. At $`t = 10^4`$ that is $`\eta/100`$; at $`t = 10^6`$, $`\eta/1000`$.

▸ **This is a hard-wired learning-rate schedule that you did not choose and cannot switch off.** Worse, it is indexed by *accumulated gradient history* rather than by progress. A model that is training beautifully at step 100,000 gets its learning rate cut by 300× for no reason other than that time has passed. For convex problems with a finite amount to learn, that is exactly right and provably optimal. For a deep network that is still discovering structure at step 500,000, it is fatal.

> **Where this came from.** AdaGrad was published in **2011** by **John Duchi, Elad Hazan, and Yoram Singer**, out of the **online convex optimization** tradition — a setting where an adversary hands you one example at a time and you are judged by *regret*, meaning how much worse you did than the single best fixed decision in hindsight. In that world, a learning rate that decays like $`1/\sqrt t`$ is not a compromise; it is what the optimal regret bound requires. Essentially the same algorithm was reached independently by **Brendan McMahan and Matthew Streeter** in 2010. Both lines of work were motivated by  sparse, high-dimensional problems — web-scale text and advertising features, where most coordinates are zero most of the time. **AdaGrad's "fatal flaw" in deep learning is a direct consequence of the guarantee it was designed to achieve.** It is not a bug; it is a theorem, applied outside its assumptions.

### RMSProp (Hinton, unpublished, 2012)

Fix the flaw: replace the sum with an **exponential moving average**.

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t}+\epsilon}\odot g_t$$

Now $`v_t`$ is a *running estimate of $`\mathbb{E}[g^2]`$* over a window of $`\frac{1}{1-\beta_2}`$ steps, and it can go down as well as up. This is a real second-moment *estimator*, not an accumulator.

#### Accumulator versus estimator — the one-word fix

The change from AdaGrad to RMSProp is a single line of code, and it is worth being able to state the difference in one sentence.

| | AdaGrad | RMSProp |
|---|---|---|
| Update | $`G_t = G_{t-1} + g_t^2`$ | $`v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2`$ |
| In English | "add today's square to the pile" | "**forget** a fraction of the pile, then add today's square" |
| Can it decrease? | **No.** Never | **Yes** — if gradients shrink, so does $`v`$ |
| What it measures | Total gradient energy since step 0 | Typical gradient energy **lately** |
| Horizon | All of history | $`1/(1-\beta_2)`$ steps |

The two coefficients $`\beta_2`$ and $`(1-\beta_2)`$ are chosen so the weights sum to 1: it is a **weighted average**, not a sum. That single fact is why $`v_t`$ stays on the same scale as $`g^2`$ forever, while $`G_t`$ grows without bound.

> **Analogy.** AdaGrad is your **lifetime odometer**; RMSProp is your **average speed over the last hour**. The odometer is a fine measurement, but you would not use it to decide how hard to press the accelerator right now. Once the question is "how fast am I going *at the moment*," a quantity that only ever increases is the wrong instrument.

**Numbers.** $`\beta_2 = 0.999`$ gives a horizon of $`1/(1-0.999) = 1{,}000`$ steps. A gradient from 1,000 steps ago retains weight $`0.999^{1000} \approx 0.37`$; from 5,000 steps ago, $`0.999^{5000}\approx 0.007`$ — gone. ▸ **So $`v_t`$ answers "how big have this parameter's gradients been over roughly the last thousand steps," and nothing more.** Hold on to that sentence; §5.2 uses it to settle a question about validation loss that has nothing to do with optimization at all.

**Why "root mean square."** $`\sqrt{v_t}`$ is the square **root** of the **mean** of the **squares** — RMS, the same quantity electrical engineers use for the effective magnitude of an alternating signal. It is a measure of typical size that ignores sign, which is exactly what you want here: a parameter whose gradient alternates $`+1, -1, +1, -1`$ has mean zero but RMS one, and it should get *small* steps, not large ones.

> **Where this came from.** RMSProp has the unusual distinction of being one of the most-used and most-cited algorithms in machine learning **that was never published in a paper.** Geoffrey Hinton presented it in a lecture of his 2012 Coursera course on neural networks, and for years the standard citation in the literature was to a specific slide of that lecture. It was proposed as a straightforward repair of AdaGrad's monotone decay, with no theory attached. **Some of deep learning's most reliable tools arrived as lecture notes and stayed that way** — a reminder that the field's practice has often run well ahead of its bibliography.

### Adam (Kingma & Ba, 2015)

Add momentum on the first moment too.

▸ $$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t &&\text{(1st moment: mean of gradient)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 &&\text{(2nd moment: uncentered variance)}\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t} &&\text{(bias correction)}\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$

Defaults: $`\beta_1=0.9`$, $`\beta_2=0.999`$, $`\epsilon=10^{-8}`$.

#### Reading Adam's four lines in plain English

Adam is the most-used algorithm in this book and it fits on a postcard. Here is each line as an English sentence.

| Line | Read aloud | What it says |
|---|---|---|
| $`m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t`$ | "m equals beta-one m, plus one-minus-beta-one g" | **"Which way has this parameter been pushed lately, on average?"** A decaying average of the gradient. This is exactly momentum (§4.5), written in the averaging convention |
| $`v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2`$ | "v equals beta-two v, plus one-minus-beta-two g squared" | **"How *big* have the pushes been lately, ignoring direction?"** Squaring destroys the sign, so cancellation is impossible |
| $`\hat m_t = m_t/(1-\beta_1^t)`$, $`\hat v_t = v_t/(1-\beta_2^t)`$ | "m-hat equals m over one minus beta-one to the t" | **"Correct for the fact that both averages started at zero and are therefore too small at first"** |
| $`\theta_t = \theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)`$ | "theta minus eta, m-hat over root v-hat plus epsilon" | **"Step in the average direction, scaled down by how erratic that parameter has been"** |

▸ **The two moments answer two different questions about the same stream of numbers: *where is it pointing* and *how loud is it*.** The update is the ratio of the two. Everything Adam does follows from that ratio.

**Why "moment" is the right word.** In statistics, the **first moment** of a random variable is its mean and the **second moment** is the mean of its square. $`m_t`$ estimates $`\mathbb{E}[g]`$ and $`v_t`$ estimates $`\mathbb{E}[g^2]`$. Note that $`v`$ is the **uncentered** second moment — it is $`\mathbb{E}[g^2]`$, not the variance $`\mathbb{E}[g^2] - \mathbb{E}[g]^2`$. Adam never subtracts the mean, which matters: for a parameter with a strong consistent gradient, $`\mathbb{E}[g^2]`$ is dominated by the *signal*, not the noise.

> **Analogy.** Two microphones on the same speaker. The first records **what** they are saying, averaged over the last ten seconds. The second records **how loudly** they have been talking over the last thousand seconds, regardless of content. Adam's decision is: *say the thing the first microphone heard, at a volume inversely proportional to what the second one measured.* Someone who has been shouting gets turned down; someone who has been murmuring gets turned up. Neither is being censored — **only the volume is normalized, never the message.**

**Why $`\beta_2 \gg \beta_1`$ — the asymmetry is deliberate.** $`\beta_1 = 0.9`$ is a 10-step memory; $`\beta_2 = 0.999`$ is a 1,000-step memory. The direction of the gradient  changes as you move through the landscape, so averaging it over too long a window would have you steering by stale information. The *scale* of the gradient changes far more slowly — it is a property of the parameter's role in the network, not of your current position. ▸ **Estimate the fast-moving thing over a short window and the slow-moving thing over a long one.** That is why the two betas differ by two orders of magnitude in their horizons, and it is not arbitrary.

#### Deriving the bias correction (do this once)

Unroll $`m_t`$ with $`m_0=0`$:
$$m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$$

Assume $`g_i`$ are drawn with stationary mean $`\mathbb{E}[g]`$:
$$\mathbb{E}[m_t] = \mathbb{E}[g](1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g](1-\beta_1)\frac{1-\beta_1^t}{1-\beta_1} = \mathbb{E}[g]\left(1-\beta_1^t\right)$$

▸ So $`m_t`$ **underestimates** the true mean by the factor $`(1-\beta_1^t)`$, purely because it started at zero. Dividing by $`(1-\beta_1^t)`$ makes it unbiased. Same argument for $`v_t`$ with $`\beta_2`$. ∎

**Why it matters most for $`v`$:** at $`t=1`$, $`1-\beta_2^1 = 0.001`$. Without correction, $`v_1 = 0.001g_1^2`$, so $`\sqrt{v_1} = 0.032|g_1|`$, and the update would be $`\eta g_1/(0.032|g_1|) = 31\eta`$ — a **31× too-large first step**. Bias correction turns that into exactly $`\eta\cdot\mathrm{sign}(g_1)`$. Without it, Adam blows up in the first few dozen steps.

#### Bias correction, decoded

The derivation above is four lines of algebra with one idea in it. Here is the idea.

**Where the bias comes from.** You initialize $`m_0 = 0`$ — you must, since you have no gradients yet. But zero is not a neutral starting value; it is a *vote for "no gradient at all"*, and it is mixed into every subsequent average. After one step, $`m_1 = 0.9(0) + 0.1g_1 = 0.1g_1`$: **the estimate of the average gradient is ten times too small,** purely because it is 90% composed of a made-up zero.

> **Analogy.** You are averaging customer ratings, and you seed the average with a fake 0-star review to get started. After the first  5-star review, your displayed rating is not 5 — it is dragged toward the phantom zero. **Bias correction is dividing by the fraction of the average that is real data**, so the phantom stops counting. After a thousand real reviews it makes no difference; on day one it makes all the difference.

**The correction is exactly that fraction.** The derivation shows $`\mathbb{E}[m_t] = \mathbb{E}[g](1-\beta_1^t)`$, so $`(1-\beta_1^t)`$ **is** the fraction of the average that comes from real gradients. Dividing by it rescales the estimate back to full size. Watch it evaporate on its own:

| $`t`$ | $`1-\beta_1^t`$ ($`\beta_1=0.9`$) | $`1-\beta_2^t`$ ($`\beta_2=0.999`$) |
|---|---|---|
| 1 | $`0.100`$ | $`0.001`$ |
| 10 | $`0.651`$ | $`0.010`$ |
| 100 | $`0.99997`$ | $`0.095`$ |
| 1,000 | $`\approx 1`$ | $`0.632`$ |
| 10,000 | $`\approx 1`$ | $`0.99995`$ |

▸ **Read the two columns against each other.** The first-moment correction has done its job within about 50 steps. The second-moment correction is still meaningfully active at step **1,000** — because $`\beta_2`$ is so close to 1 that $`v`$ takes a thousand steps to fill up with real data. **Bias correction is not a formality; for $`v`$ it is an automatic, self-retiring warmup covering the first thousand steps of every Adam run.**

**Why an uncorrected $`v`$ is so much more dangerous than an uncorrected $`m`$.** An under-estimated $`m`$ makes your step *too small* — harmless, and it fixes itself. An under-estimated $`v`$ sits in a **denominator**, so it makes your step *too large*. At $`t=1`$: $`v_1 = 0.001g_1^2`$, so $`\sqrt{v_1} = 0.0316|g_1|`$, and the update is $`\eta g_1/(0.0316|g_1|) \approx 31.6\,\eta`$. ▸ **The very first step of training — the one taken from random initialization, when the model is most fragile — would be thirty-two times too large.** Errors in numerators are forgiving; errors in denominators are not. That asymmetry is worth remembering well beyond Adam.

**And notice what the corrected first step is.** $`\hat m_1/\sqrt{\hat v_1} = g_1/|g_1| = \mathrm{sign}(g_1)`$ exactly. **Adam's first step is a step of size exactly $`\eta`$ in the direction of the gradient's sign, regardless of how big or small that gradient was.** That is not a coincidence — it is a preview of the mental model in the next subsection.

#### What the update actually looks like

▸ $$\frac{\hat m_t}{\sqrt{\hat v_t}} \approx \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}} = \frac{\text{mean}}{\text{RMS}} \in [-1, 1]$$

**Adam's per-coordinate step size is bounded by roughly $`\eta`$**, regardless of gradient magnitude. In the limit of a perfectly consistent gradient (no noise), $`\hat m/\sqrt{\hat v}\to \mathrm{sign}(g)`$ and Adam becomes **signSGD with step $`\eta`$**. In the limit of pure noise with zero mean, the ratio is $`\approx \sqrt{1-\beta_1}\cdot`$something small, and the step shrinks.

▸ **The single most useful mental model: Adam takes a step of size $`\eta`$ in the direction of the gradient's *sign*, damped by how inconsistent the gradient has been.** It is a signal-to-noise ratio detector, per coordinate.

This tells you immediately why $`\eta=3\times10^{-4}`$ is the "magic" Adam LR: it's the *actual displacement per parameter per step*, in raw parameter units. For weights of typical magnitude $`\sim0.02`$ (He init at width 1024 gives $`\sigma=\sqrt{2/1024}=0.044`$), a step of $`3\times10^{-4}`$ is a **0.7% relative change per step.** Over 2,274 steps in one epoch, if steps were coherent, a weight could travel $`0.68`$ — far more than its own magnitude. They aren't coherent, so it doesn't; but that's the scale you're working at.

#### Why the ratio $`\hat m/\sqrt{\hat v}`$ lives in $`[-1,1]`$

$$\frac{\hat m_t}{\sqrt{\hat v_t}} \approx \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}} = \frac{\text{mean}}{\text{RMS}}$$

**The claim is a fact about averages, not about optimizers.** For any list of numbers, the mean can never exceed the root-mean-square. Check it on two examples:

- Gradients $`(1, 1, 1, 1)`$ — perfectly consistent. Mean $`=1`$, RMS $`= 1`$. Ratio $`= 1`$. **Maximum.**
- Gradients $`(1, -1, 1, -1)`$ — pure disagreement. Mean $`= 0`$, RMS $`= 1`$. Ratio $`= 0`$. **Nothing.**
- Gradients $`(1, -1, 1, 1)`$ — mostly agreeing. Mean $`= 0.5`$, RMS $`= 1`$. Ratio $`= 0.5`$.

▸ **The ratio is a consistency score between 0 and 1** (or $`-1`$ and 1, with sign). It measures nothing about how *large* the gradients were — only how much they **agreed with each other.** Scale every gradient by a million and the ratio is unchanged, because the scale appears in the numerator and the denominator equally and cancels.

> **Analogy.** A signal-to-noise ratio, exactly as an engineer means it. A faint but steady radio station comes through clearly; a loud burst of static carries no information at all. **Adam judges each parameter's gradient the way a receiver judges a channel: not "how strong is this?" but "how much of this is signal?"** — and it turns the volume up on the clear channels regardless of how quiet they are.

**The two limits, spelled out.**

| Situation | $`\hat m/\sqrt{\hat v}`$ | Adam's behaviour |
|---|---|---|
| Gradient perfectly consistent (no noise) | $`\to \mathrm{sign}(g)`$ | **A step of exactly $`\eta`$**, direction only — this is signSGD |
| Gradient pure zero-mean noise | $`\to 0`$ | Step shrinks toward nothing; the parameter is left alone |
| In between (real training) | between $`0`$ and $`1`$ | Step is $`\eta`$, damped by inconsistency |

▸ **This is why $`\eta`$ means something so different in Adam than in SGD.** In SGD, $`\eta`$ multiplies a gradient of unknown magnitude, so the resulting displacement could be anything and you must tune $`\eta`$ against the gradient scale. **In Adam, $`\eta`$ is a hard ceiling on how far any single parameter can move in one step, measured in the parameter's own units.** That is why the same $`3\times10^{-4}`$ works across wildly different architectures and losses, while a good SGD learning rate varies by orders of magnitude between them.

**Follow the arithmetic on the "magic" number.** With $`\eta = 3\times10^{-4}`$ and typical weight magnitude $`0.044`$, each step moves a weight by at most $`0.7\%`$ of its own size. **That is the real unit to think in — not "learning rate" but "percent of a weight, per step."** A useful diagnostic falls straight out of it, and it is the last item in §5.8: log the ratio $`\|\Delta\theta_\ell\|/\|\theta_\ell\|`$ per layer. It should sit near $`10^{-3}`$. A layer at $`10^{-1}`$ is rewriting itself every ten steps and cannot be learning anything stable; a layer at $`10^{-5}`$ is frozen. **One logged number, two failure modes caught.**

> **Where this came from.** Adam was introduced by **Diederik Kingma and Jimmy Ba** in a paper first posted in late 2014 and presented at ICLR 2015. The name is not a person and not really an acronym — the authors state it derives from **ada**ptive **m**oment estimation. Its two ingredients were both already in circulation: momentum on the first moment (Polyak, 1964) and RMSProp's second moment (Hinton, 2012). ▸ **The paper's  new contribution was the bias correction** — the small, unglamorous fix that makes the combination stable from step one. It has since become one of the most-cited papers in all of science, with well over a hundred thousand citations. **Combining two known ideas correctly, and handling the boring edge case properly, turned out to be worth more than either idea alone.**

> **The story behind Adam's proof.** The original paper included a convergence proof for the online convex setting. In 2018, **Sashank Reddi, Satyen Kale, and Sanjiv Kumar** showed the proof contained an error, and — more damagingly — constructed a simple convex problem on which Adam provably **fails to converge**, oscillating between the wrong answers. Their paper won a Best Paper award at ICLR 2018 and proposed a fix (AMSGrad, which forces $`v`$ to be non-decreasing). The twist: **AMSGrad is barely used, and Adam remains the default everywhere.** The algorithm with the broken proof kept its job. This is a  instructive episode about the relationship between theory and practice in this field — the proof mattered enough to fix, and not enough to change anyone's optimizer.

#### The $`\epsilon`$ placement matters

- PyTorch: $`\dfrac{\hat m}{\sqrt{\hat v}+\epsilon}`$
- Original paper's alternative: $`\dfrac{\hat m}{\sqrt{\hat v + \epsilon}}`$

They differ when $`\hat v \ll \epsilon^2`$. In fp16/bf16 training, $`\epsilon = 10^{-8}`$ can *underflow*, so people use $`10^{-6}`$ or even $`10^{-4}`$. Note that increasing $`\epsilon`$ makes Adam more like SGD (large $`\epsilon`$ ⇒ the denominator is constant ⇒ plain momentum). **$`\epsilon`$ is secretly an interpolation knob between Adam and SGD.**

#### Unpacking $`\epsilon`$, the most underrated hyperparameter

$`\epsilon`$ ("epsilon") is introduced everywhere as "a small number to avoid dividing by zero," which is true and is not the whole story.

**Why the placement changes anything.** Compare the two forms when the gradient has been tiny for a long stretch, say $`\sqrt{\hat v} = 10^{-9}`$, with $`\epsilon = 10^{-8}`$:

| Form | Denominator | Resulting step |
|---|---|---|
| $`\hat m/(\sqrt{\hat v}+\epsilon)`$ | $`10^{-9} + 10^{-8} = 1.1\times10^{-8}`$ | $`\epsilon`$ **dominates** — the step becomes proportional to $`\hat m`$, i.e. plain momentum |
| $`\hat m/\sqrt{\hat v+\epsilon}`$ | $`\sqrt{10^{-18} + 10^{-8}} = 10^{-4}`$ | A **completely different** number — off by four orders of magnitude |

They agree whenever $`\hat v`$ is comfortably above $`\epsilon^2`$, which is most of the time — and disagree wildly for exactly the quiet parameters where adaptivity was supposed to help. **This is the sort of discrepancy that makes a result fail to reproduce across frameworks while every line of the training script matches.**

▸ **The interpolation, stated cleanly.** As $`\epsilon \to 0`$, the denominator is all $`\sqrt{\hat v}`$ and you have full Adam: sign-like steps, per-coordinate normalization. As $`\epsilon`$ grows past every $`\sqrt{\hat v}`$ in the model, the denominator is *the same constant everywhere*, the per-coordinate scaling vanishes, and what remains is $`\theta \leftarrow \theta - (\eta/\epsilon)\hat m`$ — **plain SGD with momentum, at learning rate $`\eta/\epsilon`$.** So $`\epsilon`$ slides continuously between the two families. It is not a numerical guard with a side effect; **it is the dial between the chapter's two protagonists, disguised as a numerical guard.**

> **Analogy.** A noise gate on an audio channel. Set it very low and every whisper is amplified — including the hiss. Set it high and quiet channels are simply passed through untouched at fixed gain. $`\epsilon`$ is the gate threshold: it decides **which parameters are quiet enough to stop treating individually.**

**The precision trap, concretely.** In bfloat16 the smallest normal positive value is around $`10^{-38}`$, so $`10^{-8}`$ itself is representable — but $`\hat v`$ is an average of *squared* gradients, and squaring a gradient of $`10^{-4}`$ gives $`10^{-8}`$, right at the same scale. Once $`\hat v`$ and $`\epsilon`$ are comparable, small rounding differences in $`\hat v`$ change the update substantially. **This is why large-scale runs commonly use $`\epsilon = 10^{-6}`$ or larger** — not for numerical safety in the naive sense, but to keep the denominator far away from the region where reduced precision is deciding the answer. If a mixed-precision run is mysteriously unstable and everything else checks out, $`\epsilon`$ is worth suspecting before anything else.

#### $`\beta_2`$ and the memory horizon — *this settles the Case Study A question*

The EMA with decay $`\beta_2`$ has effective memory $`\frac{1}{1-\beta_2}`$ steps.

▸ For $`\beta_2=0.999`$: **1,000 steps.**
▸ In Case Study A at 2,274 steps/epoch: **0.44 epochs.**

The optimizer's entire representation of the loss landscape spans less than half an epoch of data. It has no state variable that persists over 13 epochs. It has no state variable indexed by epoch at all. And it never sees `best`, `val_realCE`, or any validation quantity — those live in Python outside the optimizer's `state` dict, and the optimizer's `step()` receives only `p.grad`.

▸ **Conclusion, stated precisely: there is no channel through which a validation dry spell could influence AdamW's behaviour, and no state with a long enough horizon to encode one even if there were.**

(For very large batch / long training, $`\beta_2=0.95`$ is often better — shorter memory tracks a fast-changing landscape. For small batches, $`\beta_2=0.999`$ or higher smooths more noise. Your $`B=64`$ justifies 0.999.)

#### What the optimizer can and cannot know, decoded

The conclusion in this subsection is stated compactly, and it is the answer to a question people ask constantly in different words: *"is my optimizer somehow reacting to my validation loss?"* Here is the argument in full.

**Step 1 — inventory the optimizer's entire memory.** AdamW stores exactly two things per parameter, and nothing else:

| Stored quantity | What it holds | How far back it remembers |
|---|---|---|
| $`m`$ | Decaying average of the gradient | $`1/(1-\beta_1) = 10`$ steps |
| $`v`$ | Decaying average of the squared gradient | $`1/(1-\beta_2) = 1{,}000`$ steps |

Plus one integer, the step count $`t`$, used only for bias correction. **That is the complete state.** There is no third buffer, no history log, no epoch counter, no "how are we doing" variable.

**Step 2 — convert the horizon into your units.** At 2,274 steps per epoch, a 1,000-step memory is **0.44 epochs**. ▸ **The optimizer's longest-lived memory is shorter than a single pass through your dataset.** Anything that happened one epoch ago has faded by a factor $`0.999^{2274} \approx 0.10`$; two epochs ago, $`0.01`$. A pattern spanning 13 epochs is invisible to a quantity that has forgotten 99% of what happened two epochs back.

**Step 3 — inventory what enters the optimizer.** `step()` receives `p.grad` and reads the optimizer's own state. That is the whole interface. Validation loss is computed in Python, compared to a stored best in Python, and printed in Python. **It never crosses into the optimizer**, and there is no mechanism by which it could — not a hidden one, not a subtle one. The channel does not exist.

> **Analogy.** A thermostat controls a room's heating using a thermometer. If you ask "is the thermostat reacting to what the neighbours are saying about my house?" the answer is not "probably not" or "only weakly." **It is that the thermostat has one wire and that wire goes to a thermometer.** No amount of neighbourly opinion enters. AdamW has one wire, and it goes to `p.grad`.

▸ **The general skill here is worth more than the specific answer.** When you suspect a system is doing something mysterious, the productive move is not to argue about plausibility — it is to **enumerate its state and enumerate its inputs.** If the quantity you are worried about is in neither list, the question is settled, and no amount of suggestive correlation in the logs should reopen it. Most "the optimizer is doing something weird" reports dissolve at this step.

**What *does* legitimately span many epochs**, so you know where to look instead: the weights themselves ($`\theta`$ carries everything, forever), the learning-rate schedule if you have one, the weight-decay shrinkage compounding at $`(1-\eta\lambda)`$ per step, and your own data ordering. **Those are the long-horizon objects in a training run. The optimizer's moment estimates are not among them.**

### AdamW — decoupled weight decay (Loshchilov & Hutter, 2019)

This distinction is subtle, frequently misunderstood, and directly relevant to your `weight_decay=0.01`.

**L2 regularization** adds $`\frac\lambda2\|\theta\|^2`$ to the loss, so the gradient becomes $`g_t + \lambda\theta_t`$. In Adam, that modified gradient goes *through the preconditioner*:

$$\theta_{t+1} = \theta_t - \eta\frac{\widehat{m_t(g+\lambda\theta)}}{\sqrt{\widehat{v_t(g+\lambda\theta)}}+\epsilon}$$

▸ **The bug:** the decay term gets divided by $`\sqrt{\hat v}`$. Parameters with large gradients (large $`\hat v`$) receive **less** decay; parameters with small gradients receive **more**. That is exactly backwards from what you want, and it means the strength of your regularization depends on gradient magnitudes you don't control.

**AdamW** decouples it:

▸ $$\boxed{\ \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\right)\ }$$

Now the shrinkage is a clean multiplicative $`\theta \leftarrow (1-\eta\lambda)\theta`$ applied uniformly.

**Numbers for Case Study A.** $`\eta\lambda = 3\times10^{-4}\times0.01 = 3\times10^{-6}`$ per step.
- Per epoch (2,274 steps): $`(1-3\times10^{-6})^{2274} = 0.99320`$ → **0.68% norm decay per epoch** with zero gradient.
- Over 43 epochs: $`0.9932^{43} = 0.745`$ → weights would shrink to 74.5% of initial norm.

This is real but gentle. Note it's coupled to $`\eta`$ in PyTorch's AdamW (decay $`=\eta\lambda\theta`$), so **if you ever add an LR schedule, your weight decay decays with it.** Some implementations decouple fully ($`\lambda\theta`$ without the $`\eta`$); know which you have.

**Practical:** exclude biases, LayerNorm/RMSNorm gains, and (usually) embeddings from weight decay. Decaying a LayerNorm gain toward zero is actively harmful — it shrinks the layer's output scale for no regularization benefit (see the scale-invariance discussion in Ch. 7).

#### Decoupled weight decay, decoded

This is the "W" in AdamW, and it is one line of code that took the field several years to get right.

**First, what weight decay is for.** Left alone, weights grow. Large weights mean a function with steep, jagged responses — the kind that fits training data exactly and predicts nonsense between the points. Weight decay is a constant, gentle pull of every weight toward zero, and its purpose is to make the model prefer the simplest function that fits.

**Two ways to implement the same intention:**

| Approach | Mechanism | Reads as |
|---|---|---|
| **$`\ell_2`$ regularization** | Add $`\frac\lambda2\|\theta\|^2`$ to the *loss*, so the gradient gains a term $`+\lambda\theta`$ | "Tell the optimizer that being large is expensive, and let it work out what to do" |
| **Decoupled weight decay** | Leave the loss alone; multiply the weights by $`(1-\eta\lambda)`$ after each update | "Just shrink the weights. Do not involve the optimizer at all" |

▸ **With plain SGD these two are algebraically identical.** With Adam they are not, and that is the entire point of the AdamW paper.

**Why Adam breaks the equivalence.** Under $`\ell_2`$ regularization, the decay term $`\lambda\theta`$ is bundled into the gradient — and Adam divides the whole gradient by $`\sqrt{\hat v}`$. So the decay a parameter actually experiences is $`\lambda\theta/\sqrt{\hat v}`$: **it is divided by that parameter's own recent gradient magnitude.**

Put numbers on it. Two weights, both of size $`1.0`$, both nominally decaying at $`\lambda = 0.01`$:

| Weight | $`\sqrt{\hat v}`$ | Decay it actually receives |
|---|---|---|
| A — busy, large gradients | $`1.0`$ | $`0.01`$ — the intended amount |
| B — quiet, small gradients | $`0.01`$ | $`1.0`$ — **a hundred times too much** |

▸ **This is exactly backwards.** The parameters you would most want to regularize are the ones doing a lot of work; instead, the quiet parameters get pulverized while the busy ones are barely touched. And it is worse than merely backwards: **the strength of your regularization is being set by gradient magnitudes you neither chose nor monitor.** Change your data, your batch size, or your loss scaling, and your effective weight decay silently changes with it.

**The fix, in one line.**

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\right)$$

The $`\lambda\theta_t`$ term now sits **outside** the fraction. It never touches $`\sqrt{\hat v}`$. Every parameter shrinks by the same multiplicative factor $`(1-\eta\lambda)`$ per step, exactly as intended.

> **Analogy.** A hotel where every room's thermostat is supposed to be set to 20°C. The broken version routes each thermostat through a valve whose opening depends on how much hot water that room happened to use yesterday — so the rooms that used the least end up scalding. **The fix is not a better valve. It is noticing that the thermostat should not have been going through the valve.**

**Reading the arithmetic.** $`\eta\lambda = 3\times10^{-4}\times 0.01 = 3\times10^{-6}`$ per step. Over one epoch of 2,274 steps, $`(1-3\times10^{-6})^{2274} = 0.9932`$ — a $`0.68\%`$ shrink. Over 43 epochs, $`0.9932^{43} = 0.745`$. ▸ **With zero gradient signal, your weights would end at 75% of their starting norm.** Real training pushes back against this constantly, so the observed norm does whatever the balance of the two forces dictates — but the pull is real and it compounds.

**The coupling nobody expects.** In PyTorch's `AdamW`, the decay term is multiplied by $`\eta`$. So **decaying your learning rate also decays your weight decay.** Add a cosine schedule that ends at $`0.1\eta_0`$ and your regularization at the end of training is one tenth of what it was at the start — a change you did not request and probably do not want, arriving as a side effect of a scheduling decision. Some implementations apply $`\lambda\theta`$ without the $`\eta`$ factor, which decouples them fully. **Check which one you have before you conclude a schedule "hurt regularization."**

**Why some parameters must be excluded.** A LayerNorm gain multiplies its layer's output. Shrinking it toward zero does not simplify the function in any useful sense — it just quietly turns the layer down, and the network compensates by inflating the next weight matrix instead. Biases are similar: they shift, they do not scale, so shrinking them constrains nothing meaningful. ▸ **Weight decay is a statement about the *complexity* of a function, and it only means that for parameters that actually control complexity.** Applying it to everything because it is one fewer line of configuration is a real and common mistake.

> **Where this came from.** **Ilya Loshchilov and Frank Hutter** posted the decoupled-weight-decay paper in **2017**, originally titled around the idea of *fixing* weight-decay regularization in Adam. It took until **ICLR 2019** — and a retitling to *Decoupled Weight Decay Regularization* — to be published. Meanwhile the change was, and is, a handful of lines. The same two authors are responsible for **SGDR**, the warm-restarts paper of 2017 whose cosine curve is the default schedule in §5.3. **One research group is behind both the optimizer and the schedule that essentially every large model in this book is trained with**, and in both cases the lasting contribution was a small correction to something everyone was already doing.

#### Examples and non-examples: is this weight decay or is it $`\ell_2`$?

**✅  decoupled weight decay**

| Example | Why it qualifies |
|---|---|
| `torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)` | The $`\lambda\theta`$ term is applied outside the $`\sqrt{\hat v}`$ division |
| Plain SGD with `weight_decay=1e-4` | With SGD the two formulations coincide exactly, so this *is* decoupled decay |
| Multiplying every weight by $`0.9999`$ after each `optimizer.step()`, by hand | Literally the definition: a uniform multiplicative shrink |
| A parameter group with `weight_decay=0.0` for all LayerNorm gains and biases | Correct exclusion — those parameters do not control function complexity |

**❌ Near-misses — things that look like weight decay but behave differently**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| `torch.optim.Adam(params, weight_decay=0.01)` | Adam adds $`\lambda\theta`$ **into the gradient**, so it passes through $`\sqrt{\hat v}`$ and the quiet parameters get up to $`100\times`$ the intended pull | $`\ell_2`$ regularization inside an adaptive optimizer — the exact bug AdamW fixed |
| Adding `0.5 * lam * (w**2).sum()` to your loss and calling `.backward()` | Same thing by a different route: it enters as a gradient | $`\ell_2`$ regularization |
| Weight decay applied to LayerNorm gains | Shrinking a gain does not simplify the function; the next layer inflates to compensate | A silent scale change, no regularization benefit |
| Weight decay applied to embeddings | Rare tokens receive almost no gradient, so decay dominates and their vectors drift toward zero | Slow deletion of your rare vocabulary |
| Gradient clipping | Bounds the update size; does nothing about weight magnitude between updates | A stability mechanism |
| Early stopping | Also limits how large weights become — but by stopping, not by pulling | Implicit regularization (Chapter 7) |

▸ **The boundary:** decoupled weight decay is a **multiplicative shrink of the parameters themselves**, $`\theta \leftarrow (1-\eta\lambda)\theta`$, that never passes through the optimizer's preconditioner. $`\ell_2`$ regularization is **an added term in the loss**, which by construction does pass through it. With SGD the distinction is invisible; with any adaptive method it is the difference between the regularization you asked for and one scaled by an unmonitored quantity.

> **Common misconception.** *"`Adam(weight_decay=...)` and `AdamW(weight_decay=...)` are the same thing with different spelling."* They are different algorithms and, for the same $`\lambda`$, they produce meaningfully different models. The tempting part is that both arguments are called `weight_decay` and both make weights smaller on average, so the difference only shows up as a few points of validation performance — attributable to anything. **If you port a $`\lambda`$ from an Adam recipe to AdamW, or vice versa, you have not ported the regularization strength.**

> **Common misconception.** *"Weight decay of 0.01 means my weights shrink 1% per step."* It means $`\eta\lambda`$ per step in PyTorch's coupling — $`3\times10^{-4} \times 0.01 = 3\times10^{-6}`$, which is 0.0003%. Over 2,274 steps that compounds to 0.68% per epoch. **The learning rate is inside your weight decay**, which is why the two hyperparameters cannot be tuned independently, and why halving $`\eta`$ also halves your effective regularization.

> **Common misconception.** *"Weight decay works by preventing overfitting, so more is always safer."* Decay is a force, and training is a balance between it and the gradient. Push $`\lambda`$ high enough and the balance point moves to weights too small to represent the function at all — the loss rises for both train and validation, which is underfitting, not safety. There is a real optimum and it is usually within a factor of 3 of $`0.01`$–$`0.1`$ for transformers; **the failure at high $`\lambda`$ looks like a bad model, not like "too much regularization," which is why it goes undiagnosed.**

---

## 5.3 Every learning-rate schedule, with formulas

### Why schedules exist, in one sentence

From Ch. 4 §4.6: SGD with constant $`\eta`$ equilibrates in a noise ball of radius $`\propto \eta\sigma^2/(\mu B)`$. Decaying $`\eta`$ shrinks the ball. **A schedule is a cooling schedule** — you anneal the temperature $`\eta/B`$.

### The catalogue

| Schedule | Formula | Notes |
|---|---|---|
| **Constant** | $`\eta_t=\eta_0`$ | your current setup. Never converges; diffuses. |
| **Step** | $`\eta_t = \eta_0\gamma^{\lfloor t/s\rfloor}`$ | classic ResNet recipe: $`\times0.1`$ at epochs 30/60/90. Loss shows characteristic cliffs. |
| **Multi-step** | drops at specified milestones | same, hand-placed |
| **Exponential** | $`\eta_t=\eta_0 e^{-kt}`$ or $`\eta_0\gamma^t`$ | smooth version of step |
| **Polynomial / linear** | $`\eta_t=\eta_0(1-t/T)^p`$ | $`p=1`$ is linear-to-zero; the standard for BERT-family and, per Chinchilla-era practice, very strong |
| **Cosine** | $`\eta_t = \eta_{\min} + \frac{\eta_0-\eta_{\min}}{2}\left(1+\cos\frac{\pi t}{T}\right)`$ | the modern default; slow start, fast middle, slow finish |
| **Cosine w/ warm restarts (SGDR)** | cosine over cycles $`T_i`$, $`T_{i+1}=T_i\cdot T_{\text{mult}}`$ | each restart escapes a basin; gives free ensembles |
| **1/t (Robbins–Monro)** | $`\eta_t=\eta_0/(1+kt)`$ | theoretically optimal for strongly convex; too aggressive in practice |
| **Inverse-sqrt** | $`\eta_t = \eta_0/\sqrt{\max(t,t_w)}`$ | the original Transformer schedule |
| **One-cycle** | LR up to $`\eta_{\max}`$ then down below $`\eta_0`$; momentum moves inversely | Smith's "super-convergence" |
| **ReduceLROnPlateau** | if no improvement for `patience` epochs: $`\eta\leftarrow\gamma\eta`$ | *reactive*, not scheduled |
| **WSD (warmup–stable–decay)** | warmup, long constant plateau, short sharp decay at the end | lets you extend training without recommitting to $`T`$; now common for LLMs |

#### Reading the catalogue: what these formulas are actually shaped like

Twelve formulas is a lot of notation for what are really about five shapes. Here is what each one *draws*.

**Shared vocabulary first:**

- $`\eta_t`$ — the learning rate **at step $`t`$**. The subscript is the whole idea: it is now a function of time, not a constant.
- $`T`$ — the **total** number of steps you plan to train for. Several schedules need this in advance, which is a real commitment (and the reason WSD exists).
- $`\gamma`$ — a multiplicative decay factor, e.g. $`\gamma = 0.1`$ means "cut to a tenth."
- $`\lfloor t/s\rfloor`$ — "floor of $`t`$ over $`s`$": divide and **round down** to a whole number. It stays at 0 for the first $`s`$ steps, then 1, then 2. ▸ **The floor function is what turns a smooth curve into a staircase** — that and nothing else is why step schedules produce those characteristic cliffs in the loss curve.

**Now the shapes:**

| Shape | Which schedules | What it looks like drawn |
|---|---|---|
| **Flat** | Constant | A horizontal line. Never converges — it diffuses in the noise ball of §4.6 |
| **Staircase** | Step, Multi-step | Flat, sudden drop, flat, sudden drop. Each drop produces a visible cliff in the loss as the noise ball shrinks |
| **Smooth decay** | Exponential, Polynomial, $`1/t`$, Inverse-sqrt | A curve falling monotonically to (or toward) zero. They differ only in *how fast the falling gets slower* |
| **Bell-ish** | Cosine, One-cycle | Ramp up, sail through the middle, ease down. Gentle at both ends, aggressive in the middle |
| **Plateau + cliff** | WSD | Long flat middle, then a short sharp drop right at the end |

**Cosine, unpacked, since it is the default.**

$$\eta_t = \eta_{\min} + \frac{\eta_0-\eta_{\min}}{2}\left(1+\cos\frac{\pi t}{T}\right)$$

The only moving part is $`\cos(\pi t/T)`$. At $`t=0`$ the angle is $`0`$ and $`\cos = +1`$; at $`t = T`$ the angle is $`\pi`$ (180°) and $`\cos = -1`$. So the bracket $`(1+\cos)`$ slides from $`2`$ down to $`0`$, the halving turns that into $`1`$ down to $`0`$, and the whole thing sweeps $`\eta`$ from $`\eta_0`$ to $`\eta_{\min}`$. ▸ **Everything else in the formula is packaging; one cosine is doing the work.**

Why this particular curve rather than a straight line: the cosine is **flat at both ends** (its derivative is zero at $`t=0`$ and $`t=T`$) and steepest in the middle. That gives you a gentle start while the model is fragile, fast decay through the productive middle, and a long slow finish that lets the noise ball settle without a jolt. **A linear ramp does the same job with a hard corner at each end**, which is why cosine is preferred despite being, per §4.6's theory, no better justified.

> **Analogy.** Landing an aircraft. You do not descend at a constant rate and then hit the ground — you flare, easing the descent to nearly zero right at touchdown. **The last few percent of a training run is a flare**, and the reason is the noise ball: the point of a low final learning rate is to shrink the ball around wherever you already are, not to travel anywhere new.

**The Robbins–Monro row deserves a comment.** $`\eta_t = \eta_0/(1+kt)`$ is the schedule the theory of §4.6 actually endorses — it is the one that provably converges. It is **too aggressive in practice** because it cuts the learning rate hard while the model is still finding structure: by step 10,000 with $`k=1`$ it has fallen by a factor of 10,000. ▸ **The theoretically optimal schedule is not the practically good one, and the reason is that the theory optimizes final convergence while you care about where you converge to.** Cosine, which no theorem endorses, wins on every leaderboard.

**Why ReduceLROnPlateau is in a different category.** Every other row is a *function of $`t`$* — you could print the entire schedule before training starts. ReduceLROnPlateau is **reactive**: it reads a metric and decides. That makes it the only entry that can respond to something going wrong, and the only entry whose behaviour depends on measurement noise. The next subsection is about why that trade is usually a bad one.

> **Where this came from.** The cosine schedule arrived in **Ilya Loshchilov and Frank Hutter's** 2017 **SGDR** paper — *Stochastic Gradient Descent with Warm Restarts*. The paper's headline contribution was the **restarts**: run cosine down to near zero, then jump the learning rate back up and do it again, on the theory that each restart lets the optimizer escape its current basin and that the pre-restart checkpoints make a free ensemble. **The restarts are now rarely used. The cosine curve inside them became the field's default schedule.** It is a nice example of a paper's supporting detail outliving its thesis. The **one-cycle** policy and the "super-convergence" phenomenon come from **Leslie Smith**, working at the US Naval Research Laboratory, whose earlier work on cyclical learning rates also produced the LR range test below. The **linear-to-zero** schedule became standard through the BERT-family training recipes and has held up remarkably well; several careful comparisons in the Chinchilla era found it competitive with or better than cosine.

### The Transformer schedule, decoded

$$\eta_t = d_{\text{model}}^{-0.5}\cdot\min\left(t^{-0.5},\ t\cdot t_{\text{warmup}}^{-1.5}\right)$$

The two branches meet at $`t=t_{\text{warmup}}`$. Before: linear warmup. After: inverse-sqrt decay. The $`d_{\text{model}}^{-0.5}`$ prefactor is an early, crude version of $`\mu`$P (Ch. 21) — LR should shrink with width.

#### Unpacking the two branches

$$\eta_t = d_{\text{model}}^{-0.5}\cdot\min\left(t^{-0.5},\ t\cdot t_{\text{warmup}}^{-1.5}\right)$$

**The $`\min`$ is an `if` statement wearing a disguise** — it silently picks whichever of two curves is smaller, which is a compact way of writing a piecewise function without writing "if."

| Branch | Formula | Shape | When it wins |
|---|---|---|---|
| Warmup | $`t\cdot t_{\text{warmup}}^{-1.5}`$ | **Grows** linearly in $`t`$ | Early, while $`t`$ is small |
| Decay | $`t^{-0.5} = 1/\sqrt t`$ | **Falls** as $`t`$ grows | Later |

Early on, the rising branch is below the falling one, so the $`\min`$ picks it and the learning rate ramps up. At some point they cross; after that the falling branch is smaller and the learning rate decays. ▸ **The crossover is exactly $`t = t_{\text{warmup}}`$, by construction** — the exponent $`-1.5`$ was chosen so that the two expressions are equal there. Check it: at $`t = t_w`$, the first branch is $`t_w^{-0.5}`$ and the second is $`t_w \cdot t_w^{-1.5} = t_w^{-0.5}`$. ✓ Equal, so the curve is continuous with no jump.

> **Analogy.** A car's speed on a motorway journey. Accelerate steadily up to cruising speed, then let speed gradually bleed off as you approach the destination. Written as a single expression, that is "the smaller of (how fast I've managed to accelerate so far) and (how fast is still safe given how close I am)." **The $`\min`$ is exactly that "whichever is smaller" rule**, and it is a common trick for writing piecewise schedules on one line.

**Numbers, using the original transformer's settings.** With $`d_{\text{model}} = 512`$ and $`t_{\text{warmup}} = 4000`$: the prefactor is $`512^{-0.5} = 0.0442`$. Peak learning rate occurs at $`t = 4000`$, where $`\eta = 0.0442 \times 4000^{-0.5} = 0.0442\times0.0158 = 7.0\times10^{-4}`$. By $`t = 100{,}000`$ it has fallen to $`0.0442\times 0.00316 = 1.4\times10^{-4}`$ — a factor of 5 below peak. ▸ **Inverse-square-root decay is *gentle*.** Going from step 4,000 to step 100,000 — a 25× increase in training — reduces the learning rate only 5-fold, because $`\sqrt{25} = 5`$. That mildness is deliberate: the schedule was designed for runs of unknown length, and it never needs to know $`T`$.

**The width prefactor, and why it is there.** $`d_{\text{model}}^{-0.5}`$ says: **a wider model needs a smaller learning rate.** The reason is a shape argument. Each unit in a layer sums contributions from $`d`$ inputs; random-ish contributions accumulate like $`\sqrt d`$ (the same law as §1.1.5 and §1.3.1), so gradients — and therefore updates — grow with width. To keep the *relative* size of each update fixed as you widen the model, the learning rate must shrink like $`1/\sqrt d`$.

▸ **This matters far beyond one schedule.** It means **a learning rate tuned at one model width is wrong at another**, which is why hyperparameters found on a small pilot model so often fail to transfer to the large run. Maximal update parameterization ($`\mu`$P) is the systematic treatment of this idea: choose the scaling of initialization, learning rates, and multipliers so that tuning transfers exactly across widths, letting you sweep hyperparameters on a cheap small model and use them unchanged at full scale. **The transformer paper's single prefactor was an early, hand-tuned instance of a principle that later got worked out properly.**

> **Where this came from.** This schedule is from **Vaswani et al., *Attention Is All You Need* (2017)** — the transformer paper itself — with $`t_{\text{warmup}} = 4000`$ stated in the paper. It was a practical necessity rather than a theoretical contribution: early transformers were  unstable without warmup, and the warmup was included because training failed without it. **A one-line hyperparameter schedule tucked into a Section 5 became one of the paper's most durable practical legacies**, and §5.4 is the still-incomplete effort to explain why it was needed at all.

### On ReduceLROnPlateau specifically

Your project notes correctly observe this isn't in your setup, and correctly note that if it *were*, a dry spell would **lower** the LR rather than waste anything. Let me sharpen why that's the right instinct — and add the caveat.

The mechanism: you're in a noise ball of radius $`\propto\eta`$. If the loss has stopped improving, either (a) you're at the basin floor and the noise ball is now the limiting factor — shrinking $`\eta`$  helps, or (b) you're on a plateau and shrinking $`\eta`$ makes it worse. ReduceLROnPlateau bets on (a).

▸ **The caveat, which follows directly from Ch. 3:** ReduceLROnPlateau's trigger is `no improvement in best for patience epochs`, and we showed that under pure noise a 13-epoch gap occurs 63% of the time. **A plateau scheduler fed a noisy metric will fire on noise.** With your $`\mathrm{SE}\approx0.15`$ and `patience=10`, it would have cut your LR at least once during a period when the model was in fact improving steadily. If you ever add one:
- feed it a **smoothed** metric (EMA over 5–10 epochs), and
- set `threshold` to something meaningful relative to your measured noise (e.g. `threshold=0.05`, `threshold_mode='abs'`), and
- set `cooldown` so it doesn't fire twice on the same noise excursion.

Honestly, for your situation **cosine decay to $`0.1\eta_0`$ over your planned epoch budget** is a better choice than plateau-based reduction, because it doesn't depend on a noisy signal at all. And keeping the recipe identical across the four arms is the right scientific call — a scheduler that fires on noise would fire at *different times* in different arms and destroy the comparison.

#### Why a reactive scheduler is a measurement device, and should be treated as one

The three settings named above (`threshold`, `threshold_mode`, `cooldown`) look like configuration trivia. They are not — each one patches a specific failure that follows from a fact you already know.

**The core problem, stated plainly.** ReduceLROnPlateau's trigger is *"the best validation number has not improved for `patience` epochs."* But your validation number carries measurement noise with standard error $`\approx 0.15`$. ▸ **A stretch of epochs with no new best is exactly what a noisy measurement of a slowly-improving quantity looks like.** The scheduler cannot distinguish "the model stopped improving" from "the improvement is smaller than my instrument's resolution" — because those two produce the *same observation*.

> **Analogy.** A bathroom scale reading to the nearest kilogram, used to decide whether a diet is working. Someone losing 200 grams a week will show no change for weeks at a time, purely because of rounding. **A rule that says "if the scale hasn't moved in three weeks, change the diet" will fire constantly on a diet that is working perfectly.** The rule is not measuring the diet; it is measuring the scale.

**Each fix, mapped to the failure it prevents:**

| Setting | What it does | Which failure it prevents |
|---|---|---|
| Feed a **smoothed** metric (EMA over 5–10 epochs) | Averages down the noise before the comparison | Firing on a single unlucky epoch |
| `threshold` set relative to measured noise | Requires improvement *larger than the error bar* to count | Treating a noise-sized wobble as real evidence |
| `threshold_mode='abs'` | Compares in absolute units, not percentages | Percentage thresholds shift meaning as the loss falls |
| `cooldown` | Refuses to fire again immediately | Two cuts from one noise excursion, halving the learning rate for a reason that never existed |

▸ **The general principle, which outlives this particular scheduler: any automated rule that reads a noisy metric is a hypothesis test, and it has a false-positive rate whether or not you computed one.** `patience=10` is a decision rule. Chapter 3 gives you the tools to compute how often that rule fires on pure noise. **If you would not accept the false-positive rate in a paper, do not accept it in a callback.**

**And the reason cosine wins here is not that it is a better schedule.** It is that it is a *function of $`t`$* — it fires at exactly the same step in every arm of your experiment, regardless of what the noise did. **A reactive scheduler makes your four arms four different experiments**, because each one cuts its learning rate at a different moment for reasons unrelated to the intervention you are studying. That is not a tuning concern; it is a confound.

#### Examples and non-examples: a schedule that is doing something, versus one that is reacting to noise

**✅ Schedules whose decisions do not depend on a noisy measurement**

| Example | Why it qualifies |
|---|---|
| Cosine decay from $`\eta_0`$ to $`0.1\eta_0`$ over a fixed 43-epoch budget | Every value of $`\eta_t`$ is known before training starts |
| Linear warmup for 2,000 steps, then inverse-square-root decay | A pure function of the step counter |
| Step decay $`\times 0.1`$ at epochs 30 and 40, fixed in advance | Pre-registered, identical across experimental arms |
| A constant $`\eta`$ | The degenerate schedule, and a perfectly legitimate baseline |

**❌ Near-misses — schedules whose behaviour is set by your instrument, not your model**

| Looks like it | Why it isn't safe | What it actually is |
|---|---|---|
| `ReduceLROnPlateau(patience=10)` on a raw validation loss with $`\mathrm{SE}\approx 0.15`$ | A 13-epoch dry spell happens 63% of the time under a flat null (Chapter 3) — it will fire on noise | A **measurement device** whose readings you have not calibrated |
| Early stopping on an unsmoothed metric | Same trigger, same failure, and it ends the run rather than adjusting it | A noise detector wearing a stopping rule's clothes |
| Plateau reduction used in **one arm** of a four-arm comparison | It fires at different times in different arms, so the arms no longer differ only in the thing you changed | A confounded experiment |
| "I lowered the LR manually when the curve flattened" | Human pattern-matching on 43 noisy points is not more reliable than the scheduler | An undocumented, unreproducible schedule |
| Plateau reduction with `threshold` in percent while the loss is falling | A 1% threshold means 0.015 nats at loss 1.5 and 0.020 at loss 2.0 — the rule changes meaning as training proceeds | A drifting criterion |

▸ **The boundary:** a schedule is safe when **its trajectory is a function of the step counter alone.** The moment $`\eta_t`$ depends on a measured quantity, the schedule inherits that quantity's error bar — and must be given a smoothed input, an absolute threshold sized to the noise, and a cooldown, or it is measuring your evaluation set rather than your model.

> **Common misconception.** *"A reactive scheduler is strictly better than a fixed one, because it responds to what's actually happening."* It responds to what your *instrument reports* is happening, and Chapter 3 shows those two are far apart at 43 noisy epochs. A fixed cosine schedule makes no measurement and therefore cannot make a measurement error. **Reactivity is only an advantage when your signal is cleaner than your decision threshold**, and for validation loss on a small evaluation set it usually is not.

### Choosing $`\eta_0`$: the LR range test

Run a short training with $`\eta`$ increasing exponentially from $`10^{-8}`$ to $`10`$. Plot loss vs $`\log\eta`$. You'll see: flat, then descending, then a minimum, then divergence.

▸ **Pick $`\eta_0`$ about one order of magnitude below the divergence point**, or at the steepest descent point. Costs a few hundred steps. Almost nobody does it, and almost everyone should.

#### The LR range test, decoded

This is the cheapest high-value experiment in the chapter and it takes five minutes, so it is worth being able to describe exactly.

**The procedure.** Start a training run at an absurdly small learning rate. After every step, multiply the learning rate by a constant slightly above 1, so that it climbs **exponentially** — from $`10^{-8}`$ to $`10`$ over a few hundred steps. Record the loss at each step. Plot loss on the vertical axis against $`\log\eta`$ on the horizontal.

**Why logarithmic, not linear?** Because learning rates matter multiplicatively. The gap between $`10^{-5}`$ and $`10^{-4}`$ is the same *kind* of change as between $`10^{-3}`$ and $`10^{-2}`$, and a linear sweep would spend 99% of its steps in a region where nothing happens. **A log sweep gives each order of magnitude equal attention**, which is what you want when you have no idea which one you are in.

**Reading the four regions of the plot:**

| Region | What you see | What it means |
|---|---|---|
| **Flat** | Loss barely moves | $`\eta`$ is so small that nothing is learned. Useless, but confirms the test is working |
| **Descending** | Loss falls, increasingly steeply | Real learning. The steepest point is the most efficient learning rate |
| **Minimum** | Loss bottoms out | Roughly the largest rate that is still productive |
| **Divergence** | Loss shoots up, often to `NaN` within a few steps | You crossed the stability threshold — $`\eta > 2/L`$ from §4.4 |

▸ **The divergence point is a direct empirical measurement of $`2/\lambda_{\max}`$.** You are not guessing at a hyperparameter; you are measuring the sharpness of your own loss surface with a ruler. That is why the test is so much more informative than a random search over three candidate values — **it tells you where the wall is**, and every other decision is then relative to a known landmark.

> **Analogy.** Finding the redline on an unfamiliar engine. You do not consult a table of typical redlines for engines of this displacement; you rev it up carefully until it protests, note the number, and then run comfortably below it. **Ten minutes of deliberate probing beats an afternoon of educated guessing**, and unlike the guess it is a fact about *your* engine.

**Why "one order of magnitude below" rather than at the minimum.** The test is run at a fixed set of weights near initialization, on a handful of batches. The real run will visit sharper regions (§5.5's progressive sharpening says it will, systematically), and the noise in a real run is larger than in a short test. **The order of magnitude is margin against those two facts**, not timidity. If you also use warmup, you can afford to sit closer to the edge, because warmup buys you time for the sharpness to settle.

**A practical caveat that saves confusion.** Run the test from a fresh initialization, and reset the weights afterwards — the sweep itself will have partly trained (and, at the high end, damaged) the model. It is a measurement, not a warm start.

---

## 5.4 Warmup, and why it exists

Three explanations, all partially true:

**1. Adam's variance at small $`t`$.** $`\hat v_t`$ is estimated from few samples. Its relative variance is large early, so $`1/\sqrt{\hat v_t}`$ has a heavy tail and occasionally produces enormous steps. **RAdam** makes this precise: define the "length of the approximated SMA"
$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1-\beta_2^t},\qquad \rho_\infty = \frac{2}{1-\beta_2}-1$$
and apply the rectification factor
▸ $$r_t = \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_\infty}{(\rho_\infty-4)(\rho_\infty-2)\rho_t}}$$
only when $`\rho_t>4`$; otherwise fall back to SGD-with-momentum. This is warmup *derived* rather than tuned. For $`\beta_2=0.999`$, $`\rho_\infty=1999`$ and $`\rho_t>4`$ first happens around $`t\approx5`$ — but $`r_t`$ stays well below 1 for several thousand steps, which is why warmup periods of ~2,000–10,000 steps are standard.

**2. Curvature is highest at initialization.** $`\lambda_{\max}(H)`$ typically spikes early. Since stability needs $`\eta<2/\lambda_{\max}`$, you need a small $`\eta`$ early. (See §5.5.)

**3. Large-batch loss landscapes.** With large $`B`$ the gradient is nearly deterministic, so a too-large first step commits hard to a bad direction. Warmup lets the model "find its feet."

**Practical:** linear warmup over 1–5% of total steps, from 0 (or $`\eta_0/100`$) to $`\eta_0`$. For Case Study A, ~2,000 steps ≈ 1 epoch would be typical. **Case Study A has no warmup and hasn't diverged, which suggests $`3\times10^{-4}`$ is comfortably below your stability threshold** — possibly meaning you could train faster with a higher peak LR plus warmup.

#### Warmup, decoded — three explanations for one practice

Warmup is the clearest case in this book of a technique that unquestionably works while its explanation remains contested. All three accounts below are partly right, and they are not in competition — they describe different failure modes that warmup happens to prevent simultaneously.

**Explanation 1 — Adam's early variance, and what RAdam adds.**

At step 10, $`v`$ has effectively seen about ten squared gradients. An average of ten noisy numbers is itself noisy — and this one sits in a **denominator**, under a square root. When your denominator is an underestimate, your step is an *over*estimate, and the errors are one-sided in the dangerous direction (§5.2's bias-correction argument, in a different costume).

RAdam turns this into arithmetic. Read the symbols first:

| Symbol | Read aloud | Meaning |
|---|---|---|
| $`\rho_t`$ | "rho at t" | How many samples the second-moment estimate has **effectively** averaged, at step $`t`$ |
| $`\rho_\infty = \frac{2}{1-\beta_2}-1`$ | "rho infinity" | The eventual value, once the average is full. For $`\beta_2 = 0.999`$: $`2000 - 1 = 1999`$ |
| $`r_t`$ | "r at t" | A multiplier in $`[0,1]`$ applied to the step — **a warmup you did not have to choose** |

**"SMA" is Simple Moving Average** — a flat window rather than an exponential one. $`\rho_t`$ answers "if I replaced this exponential average with a plain window, how many samples wide would it be right now?" ▸ **The whole scheme is: estimate how much data your variance estimate is really built on, and shrink your step in proportion to how little that is.** When $`\rho_t \le 4`$ the variance is not even defined, and RAdam falls back to plain momentum rather than trusting it at all.

The formula for $`r_t`$ is not something to memorize — it is the correction factor that makes the variance of the adaptive term match its asymptotic value. What matters is its *shape*: it starts near 0 and climbs toward 1, and for $`\beta_2 = 0.999`$ it stays well below 1 for several thousand steps. ▸ **That is the origin of the "2,000–10,000 steps of warmup" folklore. It is not a tuned convention; it is the shape of a curve.**

> **Analogy.** A newly-hired analyst gives you a forecast on their first day. The forecast might be right, but you would not bet the company on it — you would weight it lightly and increase your trust as their track record accumulates. **RAdam applies exactly that policy to the model's own variance estimate**, and derives the trust curve rather than guessing it.

**Explanation 2 — curvature is highest at initialization.**

Stability requires $`\eta < 2/\lambda_{\max}`$ (§4.4). At a random initialization, $`\lambda_{\max}`$ is typically at its largest — the network is a badly-conditioned, uncoordinated object before its layers have adapted to one another. So the *safe* learning rate is smallest exactly when training begins, and rises as the model organizes itself. ▸ **Warmup is not "easing the model in gently" in some vague pedagogical sense. It is tracking a stability threshold that  moves during the first few thousand steps.**

**Explanation 3 — large batches commit too hard, too early.**

With a large batch the gradient is nearly noise-free, so the first few steps point *confidently* somewhere. But at initialization, "confidently" and "correctly" are different things: the direction is determined by random weights. A large, confident, wrong first step drives the model into a region it then spends thousands of steps escaping. ▸ **Small batches are protected from this by their own noise** — they cannot commit hard, because they do not agree with themselves. This is why warmup became essential at the same historical moment that batch sizes grew into the thousands.

**How to read three explanations for one phenomenon.** ▸ **The temptation is to ask which is right. The better question is which failure mode *your* run has.** Adam with default $`\beta_2`$ at a modest batch size: explanation 1 dominates. SGD at a huge batch: explanation 3. A deep transformer trained at an aggressive learning rate: explanation 2. Warmup is cheap enough that it is worth applying without diagnosing — but if you are debugging an early-training divergence, knowing which of the three you are looking at tells you what else to change.

> **Where this came from.** **RAdam** — *On the Variance of the Adaptive Learning Rate and Beyond* — is due to **Liyuan Liu and colleagues**, published in 2020. Its contribution is conceptual as much as practical: **it takes a piece of tuning folklore and derives it.** Warmup had been standard since the 2017 transformer paper, chosen because training failed without it, with no account of why. RAdam supplied an account and a formula. In practice, plain linear warmup remains more common than RAdam — the derivation ended up being more valuable than the algorithm it produced, which happens more often in this field than the literature admits.

---

#### Examples and non-examples: what warmup is, and what it is being confused with

**✅  warmup**

| Example | Why it qualifies |
|---|---|
| $`\eta_t = \eta_0 \cdot t/2000`$ for $`t < 2000`$, then cosine decay | The learning rate rises from near zero to its peak over a fixed prefix of training |
| The transformer paper's second branch, $`\eta_t \propto t \cdot T_{\text{warm}}^{-3/2}`$ | That branch is exactly a linear ramp |
| RAdam's $`r_t`$ multiplier | A warmup *derived* from the variance of $`\hat v`$ rather than chosen |
| Ramping the LR over the first 5% of steps when moving from batch 256 to batch 8192 | The classic large-batch use — explanation 3 above |

**❌ Near-misses — often called warmup, or mistaken for its purpose**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Cosine or linear **decay** from step 0 | Goes the wrong way — the step is largest exactly when the model is least stable | A decay schedule with no warmup |
| Adam's bias correction | It *divides* by $`1-\beta_1^t`$, which makes early steps **larger**, not smaller | A correction for the zero-initialized EMA, not a ramp |
| Freezing layers for the first epoch | Changes which parameters move, not how far | Staged unfreezing |
| Training on short sequences first | Changes the data, not the step size | Curriculum learning |
| Gradient clipping during early steps | Truncates outsized updates after the fact, and only on the steps that misbehave | A rail, not a ramp |
| A "warm restart" (the SGDR cycle) | Resets $`\eta`$ *upward* mid-training, to escape a basin | A restart schedule — the shared word "warm" is a coincidence of naming |

▸ **The boundary:** warmup is a **monotone increase of the global learning rate over a prefix of training**, and it exists because the largest safe step size is  smallest at initialization — for three separable reasons: $`\hat v`$ is unreliable, $`\lambda_{\max}`$ is at its peak, and a confident first step is a confidently wrong one.

> **Common misconception.** *"Warmup lets the model settle in gently before real training starts."* That framing sounds reasonable and predicts nothing. Each of the three mechanisms above predicts something checkable: the RAdam account says warmup length should scale with $`1/(1-\beta_2)`$; the curvature account says it should track how fast $`\lambda_{\max}`$ falls; the large-batch account says warmup matters more as batch size grows and should be nearly unnecessary at batch 32. **Those predictions differ, and they are separable in an afternoon of experiments** — which is exactly why "settling in" is not a substitute for knowing which one you are hitting.

> **Common misconception.** *"If training diverges, lower the learning rate."* That works, and it is often the wrong fix, because it lowers $`\eta`$ for **all** of training in order to survive the first 500 steps. Warmup lets you keep the peak $`\eta`$ that the rest of training wants. **The question is not "is my learning rate too high?" but "too high *when*?"** — and the answer is almost always "only at the start."

---

## 5.5 Edge of Stability — the phenomenon that broke the classical picture

### The classical claim

From Ch. 4 §4.4: GD on a quadratic is stable iff $`\eta < 2/\lambda_{\max}`$. So training should require $`\lambda_{\max} < 2/\eta`$, and if $`\lambda_{\max}`$ exceeds that, training diverges.

### What actually happens (Cohen et al., 2021)

Train a real network with full-batch GD at fixed $`\eta`$ and track $`\lambda_{\max}(H)`$ over time:

1. **Progressive sharpening:** $`\lambda_{\max}`$ *increases* steadily during training. The optimizer walks toward sharper regions.
2. It rises until it hits $`2/\eta`$.
3. Then it **stops** and hovers just above $`2/\eta`$, oscillating.
4. **The loss keeps decreasing anyway**, non-monotonically, in a sawtooth.

▸ $$\lambda_{\max}(H) \approx \frac{2}{\eta}\quad\text{for most of training}$$

This is called the **Edge of Stability (EoS)**.

### Why it's not a contradiction

The quadratic analysis is local and assumes fixed $`H`$. In reality: when $`\eta\lambda_{\max}`$ slightly exceeds 2, the top eigendirection oscillates with growing amplitude — but the *third-order* term means that as the oscillation grows, the model moves to a region of *lower* $`\lambda_{\max}`$. This is a self-stabilizing negative feedback loop. The system finds an equilibrium at the boundary.

### The consequences you should internalize

▸ **The learning rate directly sets the sharpness of the solution you find.** $`\lambda_{\max}\approx 2/\eta`$ means a *smaller* LR gives you a *sharper* minimum. Combined with "flat minima generalize better" (Ch. 19), this is a mechanistic reason why larger learning rates often generalize better — the LR is an *implicit sharpness regularizer*, not just a speed knob.

For Case Study A: $`\eta = 3\times10^{-4}`$ implies an EoS sharpness of $`\lambda_{\max}\approx 6{,}667`$. (Adam complicates this — the relevant quantity is the sharpness of the *preconditioned* landscape, and the Adaptive-EoS threshold is roughly $`\lambda_{\max}(\text{diag}(\hat v)^{-1/2}H) \approx 38/\eta`$ empirically — but the qualitative story survives.)

▸ **Loss going up for a few steps is not a bug.** At EoS, training is *supposed* to be non-monotone. If you see small sawteeth in your training loss at constant LR, that's the mechanism working, not instability.

**Practical implication:** if your model isn't at EoS, your LR is probably too low and you're leaving both speed and generalization on the table. You can measure this: power-iterate $`Hv`$ (Ch. 1 §1.2.4) for 20 iterations every few hundred steps and plot $`\lambda_{\max}\eta`$. If it's well below 2, raise $`\eta`$.

#### Edge of Stability, decoded

This is the most surprising result in the chapter, so it is worth stating what is surprising about it before decoding the mechanism.

**The setup in plain terms.** $`\lambda_{\max}(H)`$ is the **sharpness** — the largest curvature of the loss surface at your current position. Chapter 4 proved that gradient descent on a quadratic is stable only when $`\eta\lambda_{\max} < 2`$. The natural expectation follows immediately: either your learning rate is small enough for the surface you are on, or training blows up. Two outcomes.

**What Cohen and colleagues observed is a third outcome, and it is bizarre.** Sharpness does not sit still and it does not run away. It **climbs**, deliberately, until it touches exactly the value that should destroy training — and then stops there and stays, for the rest of the run, while the loss continues to fall in a jagged sawtooth.

> **Analogy.** You are told a bridge collapses above 10 tonnes. You drive a truck across, adding cargo as you go, expecting either to stay safely under 10 tonnes or to fall in. Instead the truck **loads itself up to exactly 10.01 tonnes, hovers there, creaks alarmingly, and drives across successfully** — and does this every single time, on every bridge. That is progressive sharpening plus the Edge of Stability, and it is why the result changed how people think about training.

**Reading the mechanism, term by term.**

1. **Progressive sharpening.** For reasons not fully settled, gradient descent on real networks drifts toward *sharper* regions of the loss surface. Sharpness rises steadily during training rather than falling.
2. **The threshold is reached.** $`\lambda_{\max}`$ climbs until $`\eta\lambda_{\max} = 2`$.
3. **The top eigendirection starts to oscillate.** Just past the threshold, the factor $`|1-\eta\lambda_{\max}|`$ from §4.4 exceeds 1, so displacement along that one direction grows and flips sign each step.
4. **The oscillation self-corrects.** The quadratic model assumed $`H`$ is fixed. It is not. As the oscillation's amplitude grows, the parameters swing into regions where $`\lambda_{\max}`$ is **lower** — the third-order structure of the real loss surface curves away. Lower sharpness restores stability, the oscillation shrinks, sharpening resumes, and the cycle repeats.

▸ **This is a negative feedback loop, and it means $`\lambda_{\max}\approx 2/\eta`$ is not a coincidence — it is an equilibrium the system actively maintains.** Push sharpness up and the resulting instability pushes it back down. The system sits at the boundary because the boundary is the only place both forces balance.

**The consequence that should change how you set $`\eta`$.** Rearrange:

$$\lambda_{\max} \approx \frac{2}{\eta}$$

Read aloud: *"the sharpness of the solution you end up in is about two over your learning rate."* ▸ **Your learning rate does not merely control how fast you get somewhere — it selects *where you end up*.** Halve $`\eta`$ and you will finish in a minimum roughly twice as sharp. Combined with the flat-minima story, that inverts the usual instinct: a large learning rate is not a risky shortcut you should back away from once training is stable. **It is a regularizer, and lowering it below what stability requires costs you generalization for nothing.**

**Numbers.** At $`\eta = 3\times10^{-4}`$, the predicted equilibrium sharpness is $`2/(3\times10^{-4}) = 6{,}667`$. At $`\eta = 10^{-3}`$ it would be $`2{,}000`$ — a substantially flatter solution. **These are not close numbers, and the only thing that changed was one hyperparameter.**

**The Adam caveat, in plain terms.** Adam does not descend on the raw loss surface; it descends on a *rescaled* one, where each coordinate has been divided by $`\sqrt{\hat v}`$. So the relevant sharpness is that of the **preconditioned** landscape, $`\mathrm{diag}(\hat v)^{-1/2}H`$, and the empirical threshold is around $`38/\eta`$ rather than $`2/\eta`$. ▸ **The constant changed; the story did not.** Adaptive methods also sharpen progressively, also reach a stability boundary, and also hover there. If you take one thing from the caveat, take this: **the number 2 belongs to plain gradient descent, and you should not apply it to an Adam run.**

**Why "the loss went up for a few steps" is not a bug.** At the equilibrium, the top eigendirection is *supposed* to be mildly unstable — that instability is the feedback signal. Small sawteeth in a constant-learning-rate training curve are the mechanism working correctly. ▸ **A perfectly monotone training loss at constant learning rate is weak evidence that your learning rate is too low.** That inverts most people's instinct, which is to treat any non-monotonicity as a problem to be tuned away.

> **Where this came from.** The Edge of Stability was documented by **Jeremy Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar** in a 2021 paper whose contribution was, in a sense, purely observational: they measured $`\lambda_{\max}(H)`$ during full-batch gradient descent on real networks and plotted it against $`2/\eta`$. **The tool required — power iteration on the Hessian — had existed for decades, and the experiment could have been run at any point in the previous ten years.** The classical picture said the measurement would be uninteresting, so it was not made. This is a recurring pattern in deep learning: **a confident theoretical expectation suppresses the measurement that would have contradicted it.** The adaptive-method version, with its threshold near $`38/\eta`$, was worked out in follow-up work shortly afterwards.

---

#### Examples and non-examples: is this instability, or is it the Edge of Stability?

**✅  Edge-of-Stability behaviour — leave it alone**

| Example | Why it qualifies |
|---|---|
| Training loss at constant $`\eta`$ that falls overall but with sawteeth of a few percent | The signature: non-monotone locally, monotone in trend |
| Measured $`\lambda_{\max}(H)`$ that rises, touches $`2/\eta`$, and then plateaus there for the rest of the run | Exactly the predicted equilibrium |
| Larger $`\eta`$ producing a flatter final solution and slightly better validation, at the same training loss | $`\lambda_{\max}\approx 2/\eta`$ says a bigger step selects a flatter basin |
| An Adam run whose preconditioned sharpness settles near $`38/\eta`$ | Same mechanism, different constant |

**❌ Near-misses — non-monotone loss that is *not* the Edge of Stability**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Loss that rises and **never comes back down**, going to NaN in a few hundred steps | EoS oscillations self-correct; divergence does not |  divergence — $`\eta`$ is past what the feedback loop can absorb |
| A loss spike exactly when the data loader wraps to a new shard | Correlated with an external event, not with curvature | A data problem (an unshuffled or corrupted shard) |
| Sawteeth in **validation** loss at 43 measurements | That is measurement noise, not optimization dynamics — see Chapter 3 | Sampling noise on the evaluation set |
| Loss jumping every time the LR schedule steps down | Caused by your schedule, on a schedule | A step-decay artefact |
| Loss noise from a batch size of 8 | Mini-batch gradient noise dominates; sharpness has nothing to do with it | Stochastic gradient noise, which scales as $`1/\sqrt{B}`$ |
| Periodic spikes in a mixed-precision run when the loss scaler backs off | An overflow-and-retry cycle in fp16 | A numerics event, logged by the GradScaler |

▸ **The boundary:** Edge-of-Stability sawteeth are **self-correcting oscillations along the single sharpest eigendirection at a fixed learning rate**, and they leave the downward trend intact. If the loss does not recover on its own within a handful of steps, or if the spikes line up with anything external, the cause is elsewhere.

> **Common misconception.** *"A perfectly smooth, monotone training loss means training is going well."* At constant $`\eta`$ it is weak evidence that $`\eta`$ is **too low** — the feedback loop that produces sawteeth only engages once you are at the stability boundary, and being far below the boundary means you are selecting a sharper minimum than you needed to. The misconception is tempting because a clean curve looks like control, and every other kind of engineering rewards smooth traces. **Here, smoothness may be the symptom.**

> **Common misconception.** *"Since $`\lambda_{\max}\approx 2/\eta`$, I should apply the number 2 to my Adam run."* Adam descends on a preconditioned surface, and the empirical constant there is around $`38/\eta`$ — nearly twenty times different. Plugging $`\eta = 3\times10^{-4}`$ into $`2/\eta`$ gives 6,667 for gradient descent, while the Adam-relevant threshold on the preconditioned landscape is a different quantity entirely. **The qualitative story transfers; the constant does not**, and quoting the wrong one produces confident, wrong sharpness estimates.

---

## 5.6 Adam vs SGD: the generalization question

**The empirical fact:** on image classification, well-tuned SGD+momentum often beats Adam on *test* accuracy despite Adam winning on *training* loss. On language and generative modelling (including diffusion), Adam wins decisively and SGD often fails to train at all.

**Why Adam is essential for transformers:**
- Gradient scales differ enormously across layers/parameter types (embeddings vs attention vs LayerNorm gains). Diagonal preconditioning fixes this; SGD can't.
- Heavy-tailed gradient noise. Adam's normalization is robust to outlier gradients; SGD's isn't.
- Token frequency is Zipfian — rare-token embeddings need AdaGrad-like treatment.

**Why Adam can generalize worse (when it does):**

▸ **The implicit bias argument.** On separable data with logistic loss, GD converges in direction to the **max-$`\ell_2`$-margin** solution (Ch. 19). Adam/signSGD converges toward a **max-$`\ell_\infty`$-margin**-like solution. These are different inductive biases, and $`\ell_2`$ margin happens to match the geometry of vision problems better.

▸ **The sharpness argument.** Adam's effective per-coordinate LR is larger in low-curvature directions, so it happily settles in sharper minima that SGD's noise would escape.

**Middle grounds:** decoupled weight decay (AdamW) closes much of the gap; **SAM** (Sharpness-Aware Minimization) closes more:
$$\min_\theta \max_{\|\delta\|\le\rho}\mathcal{L}(\theta+\delta) \quad\Rightarrow\quad \tilde g = \nabla\mathcal{L}\left(\theta + \rho\frac{\nabla\mathcal{L}(\theta)}{\|\nabla\mathcal{L}(\theta)\|}\right)$$
Two forward-backward passes per step (2× cost) to explicitly optimize for flatness. Consistently improves generalization by 0.5–2% on vision; mixed results on generative modelling.

#### The generalization question, decoded

The headline fact is strange enough to state twice: **on some problems, an optimizer that reaches a lower training loss produces a worse model.** If minimizing the loss were the whole job, that could not happen. It happens routinely.

**Why Adam is not optional for transformers.** Three separate reasons, each sufficient on its own:

| Reason | Mechanism | Why SGD cannot cope |
|---|---|---|
| Gradient scales vary hugely across parameter types | An embedding row, an attention projection, and a LayerNorm gain live on different scales entirely | A single global $`\eta`$ must serve all of them. There is no value that works |
| Gradient noise is **heavy-tailed** | Occasional gradients arrive orders of magnitude larger than typical — a rare token, an unusual sequence | SGD steps in proportion to gradient size, so one outlier moves the weights enormously. Adam's ratio is bounded by 1, so an outlier is *capped* |
| Token frequency is **Zipfian** | Word frequency falls roughly as $`1/\text{rank}`$: a handful of tokens dominate, and a long tail appears once in millions | Rare embeddings need AdaGrad-style treatment (§5.2) or they never learn at all |

▸ **"Heavy-tailed" is the one worth internalizing.** It means the distribution of gradient sizes has no comfortable typical scale — the occasional enormous value is not an error to be filtered but a normal feature of language data. Adam's normalization means an outlier gradient produces a step of the same size as any other. **That is a robustness property, and it is the main reason plain SGD often fails to train a transformer at all rather than merely training it more slowly.**

**Implicit bias, decoded.** When a model can fit the training data perfectly, there are usually *infinitely many* parameter settings that do. Training does not select among them at random — the optimizer's dynamics quietly prefer one kind. That preference is its **implicit bias**, and nobody wrote it down; it falls out of the update rule.

- **Gradient descent** on separable data with logistic loss drifts toward the **maximum-$`\ell_2`$-margin** solution: the separating boundary that leaves the largest straight-line gap between classes.
- **Adam and signSGD** drift toward something closer to a **maximum-$`\ell_\infty`$-margin** solution, where "gap" is measured by the single worst coordinate rather than by straight-line distance.

> **Analogy.** Two surveyors asked to place a fence exactly between two properties. One measures distance as the crow flies; the other measures it as "the largest single east–west or north–south offset." **Both fences are legitimately "in the middle." They are in different places.** Neither surveyor is wrong; they were handed different definitions of distance without either being told there was a choice.

▸ **The point is not that $`\ell_2`$ margin is correct.** It is that $`\ell_2`$ margin happens to match the geometry of image data well, so on vision benchmarks the optimizer with that bias wins. On language, the ranking flips. **"Which optimizer generalizes better" is not a property of the optimizer — it is a property of the match between the optimizer's implicit bias and the problem's geometry**, which is why the vision-versus-language split in the empirical facts is not a puzzle but the expected outcome.

**SAM, decoded.**

$$\min_\theta \max_{\|\delta\|\le\rho}\mathcal{L}(\theta+\delta)$$

Read aloud: *"minimize over theta, the maximum over all perturbations delta of norm at most rho, of the loss at theta plus delta."* The nesting is the whole idea: **"find weights such that even the worst nearby point is good."**

The two-line implementation follows directly. First find the worst nearby point by stepping *uphill* — $`\theta + \rho\nabla\mathcal{L}/\|\nabla\mathcal{L}\|`$ moves a distance $`\rho`$ in the steepest-ascent direction, which is the fastest route to a nearby bad point. Then compute the gradient **there** and use it to update $`\theta`$. Two forward-backward passes, hence 2× cost.

> **Analogy.** You are choosing where to pitch a tent in the dark. Ordinary optimization finds the single lowest spot your foot can detect — which might be a narrow crack. SAM asks a different question: **"if I woke up a metre in any direction, would I still be somewhere reasonable?"** A crack fails that test; a broad hollow passes. $`\rho`$ is how far you imagine rolling.

▸ **This makes the flat-minima preference *explicit* rather than incidental.** Section 4.6 got flatness as a side effect of minibatch temperature; the Edge of Stability got it as a side effect of learning rate. SAM asks for it directly and pays 2× compute. **That the explicit method gains only 0.5–2% over the implicit ones is a  compliment to how much regularization SGD's noise was providing for free.**

> **Where this came from.** The observation that adaptive methods can generalize worse was pressed hardest by **Ashia Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, and Benjamin Recht** in a 2017 paper, *The Marginal Value of Adaptive Gradient Methods in Machine Learning*, which showed carefully-tuned SGD matching or beating adaptive methods on a range of vision tasks and argued the difference was a matter of implicit bias rather than tuning effort. The implicit-bias result for gradient descent on separable data — that it converges in direction to the max-margin solution — is due to **Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro**, also 2017–2018. **SAM** is **Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur**, 2021. The through-line across all three is a shift in what the field measures: **from "how low is the loss" to "what shape of solution did the algorithm prefer."**

---

#### Examples and non-examples: implicit bias

**✅  implicit biases — preferences nobody wrote into the objective**

| Example | The preference it produces |
|---|---|
| Gradient descent on separable data with logistic loss | Converges in direction to the maximum-$`\ell_2`$-margin separator |
| Adam / signSGD on the same data | Drifts toward a maximum-$`\ell_\infty`$-margin-like separator instead |
| Small-batch SGD | Prefers flatter minima, because gradient noise cannot sit still in a narrow crack (§4.6) |
| A large learning rate | Selects $`\lambda_{\max}\approx 2/\eta`$ — a flatter solution, by the Edge-of-Stability equilibrium |
| Early stopping | Prefers solutions reachable in few steps, which are the low-frequency components first |

**❌ Near-misses — preferences that are explicit, not implicit**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Weight decay | You wrote $`\lambda`$ in the config; the preference for small norms is stated | **Explicit** regularization |
| SAM | The min-max objective *asks* for flatness in writing | Explicit flat-minima regularization, at 2× compute |
| Dropout | An explicit stochastic modification of the forward pass | Explicit regularization (Chapter 7) |
| Data augmentation | You chose which invariances to demand | Explicit prior injection through the data |
| A well-chosen architecture (convolutions for images) | Hard-coded into the function class | An **inductive bias** of the model, not of the optimizer |

▸ **The boundary:** an implicit bias is a preference that **falls out of the update rule** among solutions the loss cannot distinguish. Test: if you could delete the line of code that causes it, is it explicit — and if the preference survives even when the training loss is exactly zero for every candidate, it is implicit.

> **Common misconception.** *"Adam generalizes worse than SGD."* The empirical record does not support a blanket ranking. On image classification, well-tuned SGD with momentum often wins; on language and diffusion models, SGD frequently fails to train at all. **"Which optimizer generalizes better" is a statement about the match between an optimizer's implicit bias and a problem's geometry**, not a property of the optimizer. The misconception is tempting because the 2017 result that started this conversation was demonstrated on vision benchmarks, and a result demonstrated on one domain travels as if it were universal.

> **Common misconception.** *"A lower training loss means a better model, so the optimizer that reaches lower loss wins."* Once a model can fit the training set perfectly, the training loss no longer distinguishes between infinitely many solutions — and the optimizer, not the loss, picks which one. **At that point the training loss has stopped being an objective and become a tie.** This is the single strangest fact in the chapter and it is why "what shape of solution did the algorithm prefer" replaced "how low is the loss" as the interesting question.

---

## 5.7 The modern optimizers worth knowing

**Lion** (evolved by search, 2023):
$$c_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad \theta_t = \theta_{t-1} - \eta\big(\mathrm{sign}(c_t) + \lambda\theta_{t-1}\big),\quad m_t = \beta_2 m_{t-1}+(1-\beta_2)g_t$$
Only **one** state buffer instead of Adam's two — 50% optimizer-memory saving. Pure sign update. Needs $`\eta`$ about $`3`$–$`10\times`$ smaller than Adam and $`\lambda`$ correspondingly larger. Competitive on vision and language.

**Shampoo / SOAP:** full-matrix preconditioners per tensor mode. $`P_L = (\sum_t G_tG_t^\top)^{-1/4}`$, $`P_R = (\sum_t G_t^\top G_t)^{-1/4}`$, update $`= P_L G P_R`$. Captures cross-coordinate structure Adam can't. 1.3–2× wall-clock speedup at scale; expensive matrix roots (amortized over many steps).

**Muon** (2024): orthogonalize the momentum matrix via Newton–Schulz iteration, i.e. take $`\mathrm{orth}(M) = UV^\top`$ from $`M=U\Sigma V^\top`$. The update has all singular values equal to 1, which is a spectral-norm-controlled step. Currently gives the best measured speedups on transformer pretraining among cheap methods. Used for 2-D weight matrices only; keep AdamW for embeddings, biases, norms.

#### The modern optimizers, decoded

All three of these are answers to the same question: **Adam normalizes each coordinate independently, so what would you gain by respecting the fact that weights come arranged in matrices rather than as an unstructured list?**

**Lion, decoded.** The name expands to **EvoLved Sign Momentum**, and every word is literal.

$$\theta_t = \theta_{t-1} - \eta\big(\mathrm{sign}(c_t) + \lambda\theta_{t-1}\big)$$

$`\mathrm{sign}(x)`$ returns $`+1`$, $`-1`$, or $`0`$ — it **discards magnitude entirely** and keeps only direction. So every parameter moves by exactly $`\eta`$ every step, up or down. Section 5.2 showed that noise-free Adam converges to signSGD; **Lion simply starts there**, skipping the second moment altogether.

▸ **Read the memory arithmetic, because it is the actual selling point.** Adam stores $`m`$ and $`v`$: two buffers, $`2p`$ numbers. Lion stores one momentum buffer: $`p`$. At a billion parameters in 32-bit floats, that is **4 GB saved**, on hardware where memory is usually the binding constraint. And the required learning rate is 3–10× smaller precisely because the update no longer shrinks for inconsistent parameters — every step is full size, so the step must be smaller.

> **Analogy.** Adam is a driver who modulates the accelerator continuously. Lion presses it fully or not at all, and compensates by having a gentler engine. The trip works out about the same, and the car needs one fewer sensor.

**"Evolved" is not a metaphor.** Lion was **discovered by an automated symbolic search over program space** — a search procedure composed candidate optimizers out of primitive operations, evaluated them by training small models, and kept what worked. **Nobody derived Lion.** It was found, then explained afterwards. That is a  new mode of algorithm discovery in this field, and it is worth noticing that what the search converged on was something simpler than Adam rather than more complex.

**Shampoo and SOAP, decoded.** Adam treats a $`1024\times1024`$ weight matrix as a bag of a million unrelated numbers. Shampoo does not: it keeps two preconditioners, one for the row space and one for the column space.

$$P_L = \left(\textstyle\sum_t G_tG_t^\top\right)^{-1/4},\qquad P_R = \left(\textstyle\sum_t G_t^\top G_t\right)^{-1/4},\qquad \text{update} = P_L\,G\,P_R$$

$`G_tG_t^\top`$ is a $`\text{rows}\times\text{rows}`$ matrix — "how do the *output* directions of this layer covary?" $`G_t^\top G_t`$ is $`\text{cols}\times\text{cols}`$ — "how do the *input* directions covary?" ▸ **Adam can only ask "is this individual weight's gradient large?" Shampoo can ask "is this entire output direction over-represented?"** — a question about structure that a diagonal method cannot express at all.

The $`-1/4`$ exponent looks odd until you notice that the two preconditioners multiply from both sides, so their exponents add to $`-1/2`$ — matching Adam's single $`1/\sqrt{\hat v}`$. **The fourth root is there so that two half-measures compose into the right whole one.** The cost is computing matrix roots, which is expensive; in practice they are refreshed every few hundred steps rather than every step, amortizing the expense. **SOAP** (Shampoo with Adam in the Preconditioner's eigenbasis) is a refinement that runs Adam in the coordinate system Shampoo identifies.

**Muon, decoded.** Recall the SVD from §1.1.3: any matrix factors as $`M = U\Sigma V^\top`$ — rotate, stretch by the singular values, rotate. **Muon throws away the stretch.** $`\mathrm{orth}(M) = UV^\top`$ keeps both rotations and replaces every singular value with 1.

▸ **What that means in one sentence: the update keeps its *directions* and equalizes its *magnitudes*.** If the raw momentum matrix wanted to move 100× further along one direction than another, Muon overrules it and moves equally in both. It is the matrix-valued analogue of what $`\mathrm{sign}(\cdot)`$ does to a scalar — which is why Lion and Muon feel related despite looking nothing alike.

> **Analogy.** An equalizer that flattens a mixing desk's frequency response. The music — which instruments are playing, in what pattern — is untouched. The tendency of one frequency band to dominate purely because of the equipment is removed. **Muon flattens the update's spectrum, not its content.**

Why this is a **spectral-norm-controlled step**: $`\|UV^\top\|_2 = 1`$ exactly (§1.1.4), so the update's largest possible stretch is capped by construction. Section 1.1.4 showed that a network's amplification is the product of its layers' spectral norms; controlling each update's spectral norm controls how fast that product can drift. **Newton–Schulz** is an iterative scheme that computes $`UV^\top`$ using only matrix multiplications — no actual SVD, which would be far too slow. That is the engineering trick that makes the method affordable.

**Why 2-D weight matrices only.** All three methods exploit *matrix* structure. An embedding table, a bias vector, and a LayerNorm gain either have no meaningful row/column structure or have rows that are semantically independent. ▸ **Applying a matrix-aware method to something that is not meaningfully a matrix gains nothing and can cost you** — which is why the standard recipe is Muon for the 2-D weights and AdamW for everything else, in the same run.

> **Where this came from.** **Lion** came out of Google in 2023, from work on symbolic discovery of optimization algorithms — an automated search that wrote and tested candidate optimizers. **Shampoo** is **Vineet Gupta, Tomer Koren, and Yoram Singer**, 2018; the name is a pun on *preconditioner*. **Muon** is due to **Keller Jordan** and collaborators in 2024, and its name is an acronym for *MomentUm Orthogonalized by Newton-schulz*. ▸ **Muon's provenance is unusual and worth noting: it was developed and validated largely through open speedrun competitions to train a small GPT model to a fixed loss in the least wall-clock time**, with results reproduced publicly on identical hardware. **A benchmark with an unambiguous stopping condition and public reproduction turned out to be an extremely effective research instrument** — arguably more effective, for this particular question, than the conference publication cycle.

---

## 5.8 What I would change in Case Study A

Concretely, given everything above:

1. ▸ **Add an EMA of the weights** ($`\gamma=0.9999`$). Highest expected return, near-zero risk, standard for diffusion. Evaluate the EMA weights, not the raw ones.
2. ▸ **Run an LR range test.** You may be well below EoS at $`3\times10^{-4}`$ with $`B=64`$; $`1\times10^{-3}`$ with 2,000 steps of warmup is a plausible faster recipe.
3. ▸ **If you add a schedule, use cosine to $`0.1\eta_0`$**, not plateau-based. It doesn't depend on your noisy metric, and it stays identical across all four arms.
4. ▸ **Exclude LayerNorm gains and biases from weight decay** if you haven't. One line, small consistent gain.
5. **Log $`\|g\|`$, $`\|\theta\|`$, and the per-layer update-to-weight ratio $`\eta\|\Delta\theta_\ell\|/\|\theta_\ell\|`$.** That last one should sit around $`10^{-3}`$; if a layer is at $`10^{-1}`$ or $`10^{-5}`$ you've found a bug or a scaling problem.

Keeping the recipe identical across the four arms is correct and I'd keep doing it — change the recipe for *all* arms or none.

#### Reading the recommendations as a ranking

The list is ordered, and the ordering encodes a principle worth extracting: **expected gain divided by risk**, not expected gain alone.

| # | Change | Why it ranks there |
|---|---|---|
| 1 | EMA of the weights | Cannot destabilize training (§4.8 — the shadow copy is never read during training), and diffusion models reliably benefit. **High gain, near-zero risk** |
| 2 | LR range test | Costs a few hundred steps and returns a *measurement*, not an opinion. It cannot make anything worse |
| 3 | Cosine schedule, not plateau | Removes a dependence on a noisy signal, and stays identical across arms |
| 4 | Exclude norms and biases from decay | One line, well-understood mechanism, small consistent gain |
| 5 | Log $`\|g\|`$, $`\|\theta\|`$, update-to-weight ratio | Changes nothing about the model. Buys you the ability to diagnose the next problem |

▸ **Notice that the last item improves nothing and is still on the list.** The per-layer update-to-weight ratio $`\|\Delta\theta_\ell\|/\|\theta_\ell\|`$ should sit near $`10^{-3}`$: about a tenth of a percent of each weight, per step. A layer at $`10^{-1}`$ is rewriting itself every ten steps and cannot hold anything stable; a layer at $`10^{-5}`$ is effectively frozen and is contributing nothing. **Both failures are invisible in the loss curve and obvious in this one number.** Instrumentation that catches two distinct failure modes for one line of logging is worth more than most tuning.

---

## Did you know?

- **RMSProp has never been published.** Geoffrey Hinton presented it in a 2012 Coursera lecture, and for years the standard academic citation was to a specific slide of that lecture. It remains one of the most widely used and most cited unpublished results in machine learning.

- **"Adam" is not named after a person and is not really an acronym.** The authors state it comes from **ada**ptive **m**oment estimation. Both of its ingredients already existed — momentum from 1964, RMSProp from 2012 — and the paper's  new contribution was the bias correction, the least glamorous line in the algorithm.

- **Adam's convergence proof was wrong, and Adam is still the default.** In 2018, Reddi, Kale, and Kumar found an error in the original proof and constructed a simple convex problem where Adam provably fails to converge. The paper won a Best Paper award at ICLR 2018 and proposed a fix called AMSGrad. Almost nobody uses AMSGrad.

- **Without bias correction, Adam's first step would be about 32 times too large.** At $`t=1`$ with $`\beta_2 = 0.999`$, the second-moment estimate is a thousand times too small, and it sits in a denominator under a square root. Errors in numerators are forgiving; errors in denominators are not.

- **Adam's second-moment memory is shorter than one epoch for most training runs.** At $`\beta_2 = 0.999`$ the horizon is 1,000 steps. At a couple of thousand steps per epoch, the optimizer's entire picture of the loss landscape covers less than half a pass through the data.

- **$`\epsilon`$ is a hidden dial between Adam and SGD.** Push it large enough and the denominator becomes constant across all parameters, the per-coordinate adaptation disappears, and what remains is plain momentum at learning rate $`\eta/\epsilon`$. The "small constant for numerical stability" is a continuous interpolation between the chapter's two main algorithms.

- **The famous "$`3\times10^{-4}`$" Adam learning rate entered the field partly through a half-joking tweet.** Andrej Karpathy's 2016 remark naming it the best learning rate for Adam was not offered as a research finding, and it became a de facto default anyway. It is defensible for a real reason — §5.2 shows $`\eta`$ is the actual per-parameter displacement per step — but the mechanism by which it spread was social, not empirical.

- **The cosine schedule is the surviving side-effect of a paper about something else.** Loshchilov and Hutter's 2017 SGDR paper was about **warm restarts**; the cosine curve was the shape used between restarts. The restarts are now uncommon. The curve became the field's default schedule.

- **AdamW's fix is a handful of lines and took about two years and a retitling to get published.** The decoupled-weight-decay paper first appeared in 2017 and was published at ICLR 2019. The same two authors wrote SGDR — meaning one group is behind both the optimizer and the schedule that most large models in this book are trained with.

- **Applying weight decay through Adam's preconditioner regularizes exactly the wrong parameters.** The decay gets divided by each parameter's recent gradient magnitude, so quiet parameters are crushed and busy ones are barely touched — and the strength of your regularization ends up set by gradient scales you never chose.

- **The original transformer's learning-rate schedule was a practical necessity, not a theoretical contribution.** Early transformers simply did not train without warmup. The one-line formula tucked into *Attention Is All You Need* has outlived a great deal of that paper's architecture, and §5.4 shows the field still offers three competing explanations for why it was needed.

- **The Edge of Stability could have been discovered ten years earlier.** Measuring $`\lambda_{\max}(H)`$ during training needs only power iteration, a technique from the 1920s. The classical theory said the answer would be boring — sharpness below $`2/\eta`$ or divergence — so the measurement was not made. It turned out that sharpness climbs to exactly $`2/\eta`$ and stays there while the loss falls in a sawtooth.

- **Your learning rate sets how sharp a minimum you land in.** Since $`\lambda_{\max}\approx 2/\eta`$ at equilibrium, halving your learning rate roughly doubles the sharpness of the solution you converge to. Lowering the learning rate "to be safe" once training is stable is not a free precaution — it changes the answer.

- **A perfectly smooth training loss can be a warning sign.** At the Edge of Stability, training is *supposed* to be non-monotone; the small sawteeth are the self-stabilizing feedback loop working. Monotone loss at constant learning rate is weak evidence that the learning rate is lower than it needs to be.

- **Lion was not designed by anyone.** It was found by an automated symbolic search over candidate optimizer programs, then explained afterwards. What the search converged on was *simpler* than Adam — one buffer instead of two, and a pure sign update.

- **Muon was developed largely through public speedrun competitions.** The goal was to train a small GPT model to a fixed loss in the least wall-clock time, with results reproduced publicly on identical hardware. An unambiguous stopping condition and open reproduction turned out to be a remarkably effective research instrument.

---

## Check for Understanding

**Adam is a per-coordinate signal-to-noise detector that takes steps of size $`\eta`$ in the sign direction of the gradient, its entire memory spans $`1/(1-\beta_2)`$ steps and contains nothing about validation, and a learning-rate schedule is a cooling schedule for the temperature $`\eta/B`$ that determines both how tightly you converge and how sharp a minimum you converge to.**

### Can you explain these out loud?

The real test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What problem does adaptivity solve that a well-tuned single learning rate cannot?** (The rare embedding versus the final-layer bias is the whole answer.)
2. **What is a diagonal preconditioner, in terms of speed limits on streets?** What kind of badly-shaped problem can it *not* fix?
3. **Why did AdaGrad stop working on long runs, and what one word did RMSProp change to fix it?**
4. **Why does Adam need bias correction at all?** What is wrong with the estimate on step 1 if you skip it?
5. **Why is Adam's update roughly $`\pm\eta`$ regardless of gradient size?** What does that make $`\eta`$ mean, in units?
6. **What does $`\beta_2 = 0.999`$ mean in steps, and why does that number decide whether the optimizer could possibly "know" about your validation curve?**
7. **What is $`\epsilon`$ actually protecting you from, and why is $`10^{-8}`$ sometimes the wrong value?**
8. **Explain the difference between $`\ell_2`$ regularization and weight decay to someone who has only used SGD** — where they will insist, correctly, that the two are the same thing.
9. **Why does an adaptive optimizer break that equivalence?** Which parameters get pulverized, and why is that backwards?
10. **What is warmup for?** Give all three reasons, and say which one applies to a batch-8192 run.
11. **What is the Edge of Stability, in terms of a bridge and a truck?** Why does it mean your learning rate chooses your solution's sharpness?
12. **Why is a perfectly smooth training loss at constant learning rate mild evidence that something is wrong?**
13. **Why does an optimizer that reaches a lower training loss sometimes produce a worse model?** (The words "implicit bias" should come *after* the explanation, not instead of it.)
14. **Why would you not put a plateau-based scheduler on a noisy validation metric?** Connect it to the 63% figure from Chapter 3.

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 06 — Neural Networks & Backpropagation](06-neural-networks-and-backpropagation.md)
