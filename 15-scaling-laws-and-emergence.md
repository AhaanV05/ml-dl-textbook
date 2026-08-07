# Chapter 15 — Scaling Laws & Emergence

> **Prerequisites:** Ch. 14 (the $C=6ND$ formula).

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $L(N)$, $L(N,D)$ | "L of N", "L of N and D" | Test loss, **written as a function of** what you spent |
| $`\alpha_N,\ \alpha_D,\ \alpha_C`$ | "alpha-N, alpha-D, alpha-C" | Power-law **exponents** — the slope of the line on a log-log plot |
| $`N_c,\ D_c,\ C_c`$ | "N-c, D-c, C-c" | Fitted scale constants. They carry no meaning on their own |
| $E$ | "E" | The **irreducible loss** — the floor no model can go below |
| $A,\ B$ | "A, B" | Fitted numerators in the Chinchilla law. **$B$ is not batch size here** |
| $\alpha,\ \beta$ | "alpha, beta" | The Chinchilla exponents. **Not a learning rate and not Adam's momentum** |
| $\propto$ | "is proportional to" | Equal after multiplying by some constant you don't care about |
| $`N_{\text{opt}},\ D_{\text{opt}}`$ | "N-opt, D-opt" | The compute-optimal model size and token count |
| $\Theta(1)$ | "big-theta of one" | "Of order exactly 1" — bounded **above and below**. Stronger than $\mathcal{O}$ |
| $\mathrm{fan\_in}$ | "fan-in" | How many inputs feed a single unit — the width of the incoming layer |
| $\mu$P | "mu-P" | Maximal Update Parametrization |
| $p^k$ | "p to the k" | Per-token accuracy $p$, raised to the length $k$ of the required answer |
| $`D_{\text{inf}}`$ | "D-inference" | Tokens the model will serve over its whole deployed life |
| $\boxed{\ \cdot\ }$ | (a box) | The book's marker for "this is the result to carry away" |

▸ **A three-chapter Greek warning.** In Chapter 14, $\beta$ was Adam's momentum coefficient. Here it is the **data-scaling exponent** in the Chinchilla law. In Chapter 16 it will be the **weight on a KL penalty**. Similarly $\alpha$ is a scaling exponent here, a learning rate elsewhere, and the z-loss coefficient in §14.6. **The letters are recycled; take the meaning from the sentence.**

### Abbreviations used in this chapter

| Short | Full form |
|---|---|
| CoT | Chain of Thought |
| FLOP | FLoating-point OPeration |
| LLM | Large Language Model |
| LR | Learning Rate |
| MCTS | Monte Carlo Tree Search |
| $\mu$P | Maximal Update Parametrization |
| PRM | Process Reward Model (scores each reasoning *step*, not just the answer) |
| RL | Reinforcement Learning |
| RM | Reward Model |
| SP | Standard Parametrization (the ordinary way of setting init and learning rates) |

---

## 15.1 The empirical finding

### The one-line idea

Test loss falls as a **power law** in model size, data, and compute — smoothly, predictably, over many orders of magnitude, with no sign of a natural ceiling.

### The analogy

The learning curve of a manufacturing process. Every doubling of cumulative production reduces unit cost by a fixed percentage — Wright's law. It has held for aircraft, solar panels, and transistors across a century. Scaling laws are the same shape: not "more is better" but "more is better by an exactly predictable amount," which is what makes them useful for planning rather than merely encouraging.

### The Kaplan form

▸ $$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\qquad L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D},\qquad L(C)=\left(\frac{C_c}{C}\right)^{\alpha_C}$$

with $`\alpha_N\approx0.076`$, $`\alpha_D\approx0.095`$, $`\alpha_C\approx0.050`$ for language modelling.

▸ **Read the exponents as effort-per-improvement.** $`\alpha_N = 0.076`$ means a 10× larger model reduces loss by a factor $10^{-0.076} = 0.84$ — a **16% reduction**. To halve the loss you need $`10^{\log_{10}2/0.076} = 10^{3.96} \approx 9{,}000\times`$ the parameters. **Progress is real, cheap improvements are not.** This single calculation explains why frontier labs spend what they spend.

#### Reading a power law

**What the notation says.** $`L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}`$ reads: *"the loss you get from a model of size $N$ equals some fixed model size $`N_c`$ divided by yours, raised to a small power."* Three symbols, three jobs:

| Symbol | What it is | Does it matter? |
|---|---|---|
| $N$ | Your model's parameter count | This is the thing you control |
| $`N_c`$ | A fitted constant, in units of parameters | **Not really** — it just sets where the line sits vertically |
| $`\alpha_N`$ | The exponent | **Everything.** It sets how fast the line falls |

▸ **A power law is a straight line on log-log axes, and the exponent is the slope.** Take logs of both sides:

$$\log L = \alpha_N\log N_c - \alpha_N \log N$$

That is $`y = c - \alpha_N x`$ — a straight line with slope $`-\alpha_N`$. **This is why every scaling plot you will ever see has logarithmic axes.** On ordinary linear axes a power law looks like an unremarkable curve that flattens out, and you cannot read anything off it. On log-log axes it is a ruler-straight line you can extend with a pencil, which is precisely what makes forecasting possible.

> **Analogy — compound interest, running backwards.** Savings grow by a fixed *percentage* per year, so on a log axis the balance is a straight line. Scaling laws are the mirror image: loss falls by a fixed percentage per *doubling of spending*. In both cases the interesting number is not the balance or the loss, it is **the percentage** — and that percentage is the exponent.

**Now feel how small $0.076$ is.** Each row below is the loss multiplier $10^{-0.076k}$ for a $10^k$-fold increase in parameters:

| Model size × | Loss falls to | Reduction |
|---|---|---|
| 10× | $0.84$ | 16% |
| 100× | $0.71$ | 29% |
| 1,000× | $0.59$ | 41% |
| 9,000× | $0.50$ | **half** |

▸ **And the compute exponent is worse.** With $`\alpha_C = 0.050`$, halving the loss requires $10^{0.301/0.050} = 10^{6}$ — **a million times the compute.** Chapter 14's 9.6-day run becomes 26,000 years on the same cluster. **That number, more than any other in this book, explains both why the field's progress is real and why it is expensive.**

**One honest gap in the Kaplan form, worth noticing now.** As $N\to\infty$, $`(N_c/N)^{\alpha_N}\to 0`$. Written literally, the formula promises **zero loss** for an infinite model. That cannot be true: language has  irreducible uncertainty, and no predictor can beat the entropy of the source (Ch. 1 §1.4). The three-term Chinchilla law in §15.2 fixes exactly this by adding an explicit floor $E$. **Read the Kaplan form as an accurate description of the region that was measured, not as an extrapolation to infinity.**

> **Where this came from.** The shape is much older than machine learning. In 1936 **Theodore Wright**, an aeronautical engineer, published an analysis of aircraft manufacturing costs showing that unit cost fell by a fixed percentage for every doubling of cumulative production — now called **Wright's law**, and since found to hold across a startling range of technologies including solar panels and transistors. The deep learning version has a similarly quiet origin: a team at **Baidu**, led by Joel Hestness, published *Deep Learning Scaling is Predictable, Empirically* in **2017**, three years before Kaplan et al., reporting power-law scaling across machine translation, speech recognition, and image classification. **It was largely ignored.** The 2020 OpenAI paper landed differently, partly because it was about language models and partly because by then someone was prepared to act on it. This is a recurring pattern: the observation and the willingness to bet on it are separate events, often years apart.

**Additional Kaplan findings that hold up:**
- Architecture details (depth vs width, aspect ratio) matter far less than $N$ — within a broad range, only total parameter count matters.
- Larger models are more **sample-efficient**: at a fixed number of tokens, bigger is better.
- Performance is limited by whichever of $N$, $D$, $C$ is the binding constraint; the others give no benefit.

---

## 15.2 Chinchilla — derive the compute-optimal allocation

### The question

Given a fixed compute budget $C = 6ND$, how should you split it between model size $N$ and tokens $D$?

Kaplan's answer (2020) was $N\propto C^{0.73}$ — mostly grow the model. **This was wrong**, and the error cost the field a generation of undertrained models (GPT-3, at 175B params on 300B tokens, is severely undertrained).

### The Chinchilla fit

Hoffmann et al. (2022) fit a three-term law over 400+ runs:

▸ $$L(N,D) = \underbrace{E}_{\text{irreducible}} + \underbrace{\frac{A}{N^\alpha}}_{\text{finite model}} + \underbrace{\frac{B}{D^\beta}}_{\text{finite data}}$$

Fitted: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$.

**Read the three terms.** $E$ is the entropy of natural language — no model gets below it (Ch. 1 §1.4). $A/N^\alpha$ is what you lose by having a finite model. $B/D^\beta$ is what you lose by having finite data.

#### Unpacking the three-term law

**The equation is an itemized bill.** Your loss is the sum of three separate charges, and you can only argue with two of them.

> **Analogy — why a photograph is blurry.** There are three causes, and they are independent. The lens has a physical diffraction limit you cannot buy past ($E$). The sensor may have too few pixels ($A/N^\alpha$ — buy a bigger sensor). And you may not have collected enough light ($B/D^\beta$ — leave the shutter open longer). **Spending your entire budget on a bigger sensor when the problem is darkness is wasted money, and the reverse is equally wasted.** Getting that balance right is the whole of §15.2.

**Reading the shape of each term.** $A/N^\alpha$ can be written $AN^{-\alpha}$ — as $N$ grows, this shrinks toward zero. Same for $B/D^{-\beta}$ in $D$. So the two "penalty" terms fade as you spend, and what remains is $E$, sitting there permanently.

**Now put real numbers through it.** Take a 7B model trained on 140B tokens (Chinchilla-optimal, by the rule below):

$$N^\alpha = (7\times10^9)^{0.34} \approx 2{,}223 \quad\Rightarrow\quad \frac{A}{N^\alpha} = \frac{406.4}{2223} = 0.18$$
$$D^\beta = (1.4\times10^{11})^{0.28} \approx 1{,}321 \quad\Rightarrow\quad \frac{B}{D^\beta} = \frac{410.7}{1321} = 0.31$$
$$L = \underbrace{1.69}_{\text{floor}} + \underbrace{0.18}_{\text{model}} + \underbrace{0.31}_{\text{data}} = 2.18\ \text{nats}$$

▸ **Look at the proportions: $1.69/2.18 = 77\%$ of the loss is the floor.** Everything the entire industry competes over — every architecture, every data pipeline, every optimizer — is fighting for the remaining 23%. **This is the most sobering number in the chapter**, and it is why loss differences that look tiny (2.18 versus 2.12) correspond to large differences in capability: the reducible part moved by a quarter.

**What "1.69 nats" actually means.** A nat is a unit of information, like a bit but using base $e$ instead of base 2 (Ch. 1 §1.4). Converting: $1.69/\ln 2 \approx 2.44$ bits per token. So even a perfect model would still need roughly two and a half bits to say which token comes next. In perplexity terms — $e^{L}$, the effective number of equally likely choices — the floor is $e^{1.69}\approx 5.4$ and the fitted model lands at $e^{2.18}\approx 8.9$. **The model is choosing between about nine plausible tokens where the language itself leaves about five  open.**

*(Two honest caveats. $E$ is fitted on one particular corpus with one particular tokenizer, so it is not "the entropy of English" in any absolute sense — change the data mix and $E$ moves. And 2.44 bits per token over roughly four characters per token is around 0.6 bits per character, which is broadly in the range Shannon obtained in 1951 by having people guess the next letter of printed English — an agreement pleasant enough to notice and too loose to lean on.)*

**Why $A = 406.4$ and $B=410.7$ look absurd.** They are not losses. They are numerators sitting on top of $N^{0.34}$ and $D^{0.28}$, quantities in the thousands, so the constants have to be large for the ratio to come out near 1. ▸ **Fitted constants in a power law carry no interpretation on their own** — only the exponents and the resulting ratios mean anything. Do not try to give $A$ a story.

> **Where this came from.** DeepMind named the model **Chinchilla** in keeping with its animal-named series — Gopher, Flamingo, Gato. The result was not a new architecture and not a new algorithm; it was **400-odd training runs and a curve fit**, which is unusual for a paper of that influence. It is worth registering how much of the value came from the sheer number of runs: the fit needed enough points across enough of the $(N, D)$ plane that the two exponents could be separated, and nobody had previously been willing to spend that compute on measurement rather than on a headline model.

### The optimization, done properly

Minimize $L$ subject to $C=6ND$. Substitute $D=\frac{C}{6N}$:

$$L(N) = E + AN^{-\alpha} + B\left(\frac{C}{6N}\right)^{-\beta} = E + AN^{-\alpha} + B\left(\frac{6N}{C}\right)^{\beta}$$

$$\frac{dL}{dN} = -\alpha AN^{-\alpha-1} + \beta B\,6^\beta C^{-\beta}N^{\beta-1} = 0$$

$$\alpha A N^{-\alpha-1} = \beta B\,6^\beta C^{-\beta}N^{\beta-1}$$

$$N^{\alpha+\beta} = \frac{\alpha A}{\beta B\,6^\beta}\,C^{\beta}$$

▸ $$\boxed{\ N_{\text{opt}} \propto C^{\frac{\beta}{\alpha+\beta}},\qquad D_{\text{opt}}\propto C^{\frac{\alpha}{\alpha+\beta}}\ }$$

**Plug in the numbers:** $\frac{\beta}{\alpha+\beta} = \frac{0.28}{0.62}=0.452$ and $\frac{\alpha}{\alpha+\beta}=\frac{0.34}{0.62}=0.548$.

▸ **Both exponents are close to $\tfrac12$.** Hence the famous headline: **scale model and data equally.** Every doubling of compute should be spent on $\sqrt2\times$ more parameters and $\sqrt2\times$ more tokens.

#### The derivation, line by line

Five lines of algebra just decided how every frontier model since 2022 was sized. Here is each line, slowly.

**Line 1 — why substitute at all.** You have two unknowns, $N$ and $D$, and one constraint, $C = 6ND$. Use the constraint to eliminate one of them, and a two-variable optimization becomes a one-variable one you can differentiate. Rearranging $C=6ND$ gives $D = \frac{C}{6N}$, which is the substitution.

**Line 2 — the step that trips people.** $\left(\frac{C}{6N}\right)^{-\beta}$ becomes $\left(\frac{6N}{C}\right)^{\beta}$. This is nothing but the rule $x^{-\beta} = (1/x)^{\beta}$: **a negative exponent flips the fraction.** No insight, just bookkeeping — but it is where a reader following along usually loses the thread.

**Line 3 — differentiate.** The power rule, $\frac{d}{dN}N^{p} = pN^{p-1}$, applied to each term. Everything that isn't $N$ — the $6^\beta$, the $C^{-\beta}$, the $A$ and $B$ — is a constant along for the ride:

$$\frac{d}{dN}\big[AN^{-\alpha}\big] = -\alpha A N^{-\alpha-1},\qquad \frac{d}{dN}\big[B\,6^\beta C^{-\beta}N^{\beta}\big] = \beta B\,6^\beta C^{-\beta}N^{\beta-1}$$

**Line 4 — why setting it to zero is the whole economic content.** Notice the signs. The first derivative is **negative**: more parameters always reduce the model term. The second is **positive**: at a fixed budget, more parameters mean fewer tokens, which *increases* the data term. Setting the sum to zero says:

▸ **The optimum is where one more parameter buys exactly as much as the token you had to give up to afford it.**

> **Analogy.** You have a fixed budget for a dinner party and must split it between quality and quantity. The best split is where the last pound spent on better wine adds exactly as much to the evening as the last pound spent on more of it. Everyone applies this rule by instinct; the derivation above is that instinct with a power law attached.

**Line 5 — collect the powers of $N$.** Divide both sides by $N^{\beta-1}$: the left exponent becomes $-\alpha-1-(\beta-1) = -(\alpha+\beta)$. Move constants across, and $N^{\alpha+\beta} = \frac{\alpha A}{\beta B 6^\beta}C^\beta$. Take the $(\alpha+\beta)$-th root of both sides, and everything that isn't $C$ collapses into the $\propto$:

$$N_{\text{opt}}\propto C^{\frac{\beta}{\alpha+\beta}},\qquad D_{\text{opt}} = \frac{C}{6N_{\text{opt}}} \propto C^{1 - \frac{\beta}{\alpha+\beta}} = C^{\frac{\alpha}{\alpha+\beta}}$$

▸ **A free consistency check you should always run.** The two exponents *must* sum to exactly 1, because $N\times D\propto C$ and exponents add when powers multiply. Here: $0.452 + 0.548 = 1.000$. ✓ **If your algebra ever produces exponents that do not sum to 1, you dropped a term** — and this check costs nothing.

▸ **Now the part that looks like a typo and is not.** $`N_{\text{opt}}`$'s exponent is $\beta$ — the **data** exponent — and $`D_{\text{opt}}`$'s is $\alpha$, the **model** exponent. They have swapped. The reason: **a large exponent means that resource is efficient, so a modest amount of it already flattens its term, and the budget should go to the other one.** Push it to the extreme to check. If $\alpha\to\infty$ — the model term vanishing the instant you add a parameter — then $\frac{\beta}{\alpha+\beta}\to 0$, so $`N_{\text{opt}}`$ stops growing with compute at all: you would use a tiny model and pour everything into data. That is exactly right, and it is what the formula says.

**And the doubling arithmetic, checked.** $2^{0.452} = 1.368$ and $2^{0.548} = 1.462$ — both close to $\sqrt2 = 1.414$, which is where the headline comes from. Note that $1.368\times1.462 = 2.00$: the two increases multiply to the doubling you paid for, exactly as the budget requires.

### The rule of thumb

▸ $$\frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20\ \text{tokens per parameter}$$

**Examples:** 1B → 20B tokens; 7B → 140B; 70B → 1.4T.

**The demonstration:** Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) on nearly every benchmark, at **identical training compute** — and is 4× cheaper to run.

### Why almost nobody follows Chinchilla exactly

Chinchilla optimizes **training** compute only. Real deployments pay **inference** compute forever.

▸ **Inference-aware scaling** (Sardana et al.): minimize $`C_{\text{train}} + C_{\text{inference}}`$ over the model's lifetime. If you will serve $`D_{\text{inf}}`$ tokens, total cost $`\approx 6ND_{\text{train}} + 2ND_{\text{inf}}`$. Since inference cost is linear in $N$ and independent of $`D_{\text{train}}`$, the optimum shifts sharply toward **smaller models trained on far more data.**

This is why LLaMA-3-8B was trained on 15T tokens — a ratio of **1,875 tokens per parameter**, roughly 94× past Chinchilla-optimal. It is *not* compute-optimal to train, and it is *far* better to deploy. **A model trained past Chinchilla still improves, just with diminishing returns; a model too large to serve is worthless.**

### Data-constrained scaling

When unique data runs out (Muennighoff et al., 2023): repeating data has value that decays with epoch count. Up to **~4 epochs** is nearly as good as fresh data; by ~16 epochs, additional repetition contributes essentially nothing. Beyond that, extra parameters also stop helping.

▸ Since high-quality text is finite (single-digit trillions of tokens on the open web), this is a  constraint, and it is the strategic reason for the intense current focus on **synthetic data**, **multimodal data**, and **test-time compute** (§15.5).

---

## 15.3 $\mu$P — hyperparameter transfer

### The problem

Optimal learning rate depends on model width. Tuning a 70B model directly is unaffordable, so you tune a small proxy — but the optimum shifts, so the transfer fails.

### The solution

**Maximal Update Parametrization** ($\mu$P, Yang & Hu) chooses per-layer scalings of initialization variance and learning rate such that, in the infinite-width limit, **every layer's activations and updates stay $\Theta(1)$**. The consequence:

▸ **Optimal hyperparameters become width-independent, so they transfer from a small proxy to a large model exactly.**

### The prescriptions

| | Standard (SP) | $\mu$P |
|---|---|---|
| Input/embedding LR | $\eta$ | $\eta$ |
| Hidden weight init var | $1/\mathrm{fan\_in}$ | $1/\mathrm{fan\_in}$ |
| **Hidden weight LR (Adam)** | $\eta$ | $\eta/\mathrm{fan\_in}$ |
| **Output layer init** | $1/\mathrm{fan\_in}$ | $1/\mathrm{fan\_in}^2$ |
| **Output logits** | — | multiply by $1/\mathrm{fan\_in}$ |
| Attention scale | $`1/\sqrt{d_k}`$ | $`1/d_k`$ |

### The reasoning, in one line

If a layer's update is $\Delta W$ and it acts on an activation $x\in\mathbb{R}^{n}$, then $\Delta W x$ is a sum of $n$ terms. For the *change in output* to remain $\Theta(1)$ as $n\to\infty$, the per-entry update must scale as $1/n$ when the terms are correlated (which they are after the first step, unlike at initialization where the $1/\sqrt n$ of Ch. 6 suffices). **$\mu$P is Chapter 6's variance analysis extended from initialization to the entire training trajectory.**

▸ **Practical payoff:** tune LR, init scale, and warmup on a 40M-parameter model, transfer directly to 7B or 70B. This is now standard practice at every serious lab, and it saves an enormous fraction of a pretraining budget.

Also note: $\mu$P changes attention to $`1/d_k`$ rather than $`1/\sqrt{d_k}`$ — worth knowing as a rare, principled exception to Chapter 11's rule.

---

## 15.4 Emergence, and the argument against it

### The claim

Some capabilities appear **abruptly** at a scale threshold: three-digit arithmetic, word unscrambling, multi-step reasoning show near-zero accuracy until a critical size, then rise sharply. Wei et al. (2022) catalogued dozens.

If true, this is important and alarming: capabilities cannot be forecast from smaller models.

### The critique (Schaeffer, Miranda & Koyejo, 2023)

▸ **Emergence may be an artifact of the metric, not the model.**

Consider exact-match accuracy on a $k$-token answer. If per-token accuracy is $p$, then
$$\Pr(\text{exact match}) = p^k$$
Suppose $p$ improves *smoothly* with scale — say $p$ goes $0.5\to0.9$ over three orders of magnitude. With $k=5$:
$$0.5^5 = 0.031 \quad\to\quad 0.9^5 = 0.59$$
The exact-match curve is sharply convex and looks like a phase transition. **The underlying per-token improvement was perfectly smooth.**

The general point: **discontinuous or nonlinear metrics manufacture discontinuities.** Exact match, multiple-choice accuracy, and any thresholded score do this. Continuous metrics — token edit distance, Brier score, log-likelihood of the correct answer — show smooth improvement on the *same* model outputs.

▸ **The honest position, which is what to say if asked:** the *underlying capability* improves smoothly and predictably; the *user-visible usefulness* can change abruptly, because usefulness often depends on crossing a reliability threshold. Both statements are true and they are not in conflict. Emergence is real as a fact about task success rates and mostly not real as a fact about the model's internals.

**A  residual:** some capabilities do look sharp even under continuous metrics (in-context learning's onset alongside induction-head formation, Ch. 13 §13.3). Phase transitions in *mechanisms* appear to be real even where phase transitions in *metrics* are artifacts.

#### Examples and non-examples: is that emergence?

This is the single most misreported topic in the chapter, so it is worth drawing the boundary precisely.

**✅  sharp, under continuous metrics too**

| Case | Why it survives scrutiny |
|---|---|
| Induction-head formation | A visible mechanism appears in a narrow window; the loss curve itself has a bump |
| Grokking (Ch. 30) | Test accuracy jumps long after training loss flattens, under plain accuracy *and* continuous measures |
| Phase changes in learned circuits | The internals  reorganize |

**❌ Near-misses — look like emergence, but are metric artifacts**

| Looks emergent | Why it isn't | What's really happening |
|---|---|---|
| Exact-match accuracy on multi-token answers | $p^k$ is sharply convex even when $p$ moves smoothly | Smooth improvement, convex readout |
| Multiple-choice accuracy crossing chance | Thresholding at "is the argmax right?" discards all margin information | The margin was improving all along |
| Any pass/fail benchmark | A threshold *manufactures* a cliff | Continuous progress, binarized |
| A capability appearing "suddenly" in a product | Usefulness requires crossing a reliability bar | Real for users, not a fact about the model |
| Log-scale x-axis with few model sizes | Three points can't distinguish a curve from a step | Insufficient resolution |

▸ **The boundary:** ask *"does the sharpness survive if I switch to a continuous metric on the same outputs?"* If token edit distance, Brier score, or log-likelihood of the correct answer all move smoothly, the discontinuity lived in your ruler, not in the model. **This single test resolves most emergence claims.**

> **Common misconception.** *"Emergent abilities prove that scaling produces unpredictable jumps in capability."* Mostly the opposite: the underlying capability improves smoothly and *predictably*, and a thresholded metric converts that smooth curve into a cliff. But do not overcorrect — the honest position holds two things at once. **The model's competence moves smoothly; the user's experience can change abruptly**, because a tool that succeeds 40% of the time and one that succeeds 90% of the time are different products. Both statements are true.

> **Common misconception.** *"A bigger model is always better."* Only if the binding constraint is model size. Chinchilla's whole point is that GPT-3 was *undertrained* — at 175B parameters on 300B tokens it would have been beaten by a much smaller model given the same compute spent on more data. **Scaling the wrong axis buys nothing**, which is what the itemized-bill reading of the three-term law is for.

---

## 15.5 Test-time compute scaling

The newest scaling axis, and the one most likely to matter over the next few years.

▸ Instead of scaling training, spend more compute **at inference**: sample many chains of thought, search over them, verify, and select.

**Methods:**
- **Self-consistency:** sample $k$ chains, take the majority answer. Accuracy improves roughly logarithmically in $k$.
- **Best-of-$n$ with a verifier / reward model:** sample $n$, score, pick the best. Gains depend on verifier quality; a weak verifier saturates or degrades (reward hacking, Ch. 16).
- **Search:** tree search / MCTS over reasoning steps with a process reward model.
- **Learned long reasoning:** train the model via RL to produce long internal chains before answering (o-series, R1-style). This converts test-time compute into a *trained* behaviour rather than an external procedure.

▸ **The finding that reframed the field:** on many reasoning benchmarks, the loss/accuracy curve against *inference* FLOPs is also a clean power law, and there are regimes where a smaller model with more test-time compute beats a larger model with greedy decoding **at equal total FLOPs**. Training compute and inference compute are, to a degree, substitutable.

**The catch:** the substitution works best where answers are *verifiable* (math, code, formal tasks) and much worse where they are not, because the selection step needs a reliable signal.

---

## 15.6 Using scaling laws in practice

The workflow that a serious lab actually runs:

1. Train a ladder of small models ($10^{17}$–$10^{20}$ FLOPs) across a range of $N$ and $D$.
2. Fit $L(N,D) = E + AN^{-\alpha}+BD^{-\beta}$ by Huber-loss regression in log space (Huber, not squared error, because outlier runs are common).
3. Solve for the optimal $(N,D)$ at your target budget, adjusted for expected inference volume.
4. Use $\mu$P to transfer hyperparameters from the ladder.
5. **Predict** the final loss before launching, and treat a deviation as a bug signal.

▸ **Scaling laws' real value is as a forecasting and debugging instrument**, not a philosophy. GPT-4's technical report notes the final loss was predicted from runs using $10{,}000\times$ less compute. If your big run misses its predicted loss, you have an infrastructure or data problem, and you know that on day two rather than day forty.

**Caveats to state:** the exponents are dataset- and architecture-dependent; laws fitted on one data mixture do not transfer to another; they describe pretraining loss, which after post-training is only loosely coupled to usefulness; and extrapolating many orders of magnitude beyond the fitted range is an act of faith.

---

## Did you know?

- **Scaling laws were published three years before anyone acted on them.** A team at Baidu led by Joel Hestness reported power-law scaling across translation, speech, and vision in 2017. The paper was largely ignored. The 2020 OpenAI paper said something similar and reshaped an industry — the observation and the willingness to bet on it are separate events.

- **The shape is older than machine learning by eighty years.** In 1936 the aeronautical engineer Theodore Wright found that aircraft unit costs fell by a fixed percentage per doubling of cumulative production. **Wright's law** has since been found to hold for solar panels, transistors, and a startling range of other technologies.

- **Halving a language model's loss requires roughly a *million* times the compute.** With $`\alpha_C \approx 0.05`$, that's $10^{0.301/0.05} = 10^6$. A training run of ten days becomes twenty-six thousand years on the same hardware. This one number explains both why progress is real and why it costs what it costs.

- **About 77% of a well-trained model's loss is irreducible.** In the Chinchilla fit, the floor $E = 1.69$ nats out of a total near 2.18. Every architecture, data pipeline, and optimizer the entire industry competes over is fighting for the remaining 23% — which is why loss differences that look trivial correspond to large capability gaps.

- **GPT-3 was severely undertrained, and nobody knew for two years.** Kaplan's 2020 analysis said "mostly grow the model," so GPT-3 used 175 billion parameters on only 300 billion tokens. Chinchilla showed in 2022 that a much smaller model trained on far more data wins at equal compute. A generation of models was built on the wrong recipe.

- **Chinchilla's contribution was not an architecture or an algorithm — it was 400 training runs and a curve fit.** The result required enough points across the size–data plane to separate two exponents, and nobody had previously been willing to spend that much compute on *measurement* rather than on a headline model.

- **DeepMind's models were named after animals** — Gopher, Chinchilla, Flamingo, Gato. Chinchilla was so named because it was the small, efficient counterpart to the much larger Gopher, which it outperformed.

- **The rule of thumb is about 20 tokens per parameter**, and it falls out of the two Chinchilla exponents being nearly equal ($\alpha = 0.34$, $\beta = 0.28$). If they were far apart, the optimal recipe would be lopsided toward one axis.

- **Most "emergent abilities" are artifacts of the measuring instrument.** Exact-match accuracy on a $k$-token answer goes like $p^k$, which is sharply convex — so a perfectly smooth improvement in per-token accuracy produces what looks like a phase transition. Switch to a continuous metric and the cliff disappears.

- **But some sharpness is real.** In-context learning's onset coincides with induction heads forming, and it shows up as a visible bump in the loss curve itself. Phase transitions in *mechanisms* appear  even where phase transitions in *metrics* are illusions.

- **The Chinchilla constants $A = 406.4$ and $B = 410.7$ are not losses and have no interpretation.** They're numerators sitting atop quantities in the thousands. Fitted constants in a power law mean nothing on their own — only exponents and ratios carry information.

- **Chinchilla's irreducible loss works out to roughly 0.6 bits per character**, which is broadly in the range Shannon obtained in 1951 by having human subjects guess the next letter of printed English. The agreement is pleasant to notice and far too loose to lean on.

- **Compute-optimal is not deployment-optimal.** Chinchilla answers "best model for a fixed *training* budget." If you're going to serve a model billions of times, it's rational to overtrain a smaller one well past the Chinchilla point — inference cost, not training cost, dominates. This is why modern small models are trained on far more tokens than Chinchilla would prescribe.

---

## Check for Understanding

**Loss falls as a power law in compute with an exponent near $0.05$, meaning improvements are predictable but expensive; the compute-optimal split is roughly equal scaling of parameters and data (~20 tokens per parameter) because the two Chinchilla exponents are close to equal; and most apparent "emergence" is the effect of putting a smoothly improving model through a discontinuous metric.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is a power law a straight line on log-log axes**, and why does every scaling plot use them?
2. **What does an exponent of 0.076 actually mean** in terms of effort per improvement?
3. **Why does halving the loss take a million times the compute?**
4. **Read the three-term Chinchilla law as an itemized bill.** Which charge can you never argue with? (The blurry-photograph analogy.)
5. **Why is 77% of the loss irreducible**, and why does that make small loss differences matter so much?
6. **Why was GPT-3 undertrained**, and what did Kaplan get wrong that Chinchilla corrected?
7. **Where does "20 tokens per parameter" come from?** What would change it?
8. **Why do the fitted constants $A$ and $B$ look absurd, and why shouldn't you try to interpret them?**
9. **How can exact-match accuracy manufacture a fake phase transition** out of smooth improvement?
10. **What is the one test that resolves most emergence claims?**
11. **Why are "the model improves smoothly" and "the capability appeared suddenly" both true?**
12. **Why is compute-optimal not the same as deployment-optimal?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 16 — Post-Training: SFT, RLHF, DPO & Reasoning](16-post-training-rlhf-dpo-reasoning.md)
