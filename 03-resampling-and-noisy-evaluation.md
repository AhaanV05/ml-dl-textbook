# Chapter 3 — Resampling & the Statistics of Noisy Evaluation

> **Prerequisites:** Ch. 1 (§1.3), Ch. 2 (§2.3).
> **Why this chapter is placed here:** resampling is how you attach an error bar to any number you compute. It is also the diagnostic manual for reading a training curve, and §3.6 is the single most transferable section in Part I.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\mathrm{Var}$, or $\sim$ are unfamiliar — or if you have never had a hat ($\hat\theta$), a star ($\theta^*$), and a bar ($\bar\theta$) distinguished from one another — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; refer back as needed. Each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\hat\theta$ | "theta-hat" | **An estimate.** Any number you computed from data: an accuracy, a loss, an AUC |
| $s(X_1,\dots,X_n)$ | "s of X-one through X-n" | The **recipe** that turns a dataset into that number |
| $\hat\theta^{*(b)}$ | "theta-hat star, b" | The same recipe re-run on the $b$-th **resampled** dataset. The star means "from a resample" |
| $B$ | "B" | How many resamples you draw. 1,000 for a standard error, 10,000 for an interval |
| $\widehat{\mathrm{SE}}$ | "SE-hat" | Estimated **standard error** — how much $\hat\theta$ would wobble on fresh data |
| $F$, $\hat F_n$ | "F", "F-hat-n" | The true distribution; the **empirical** one made of your $n$ points, each weighted $1/n$ |
| $\alpha$ | "alpha" | The **miss rate** you accept. A 95% interval means $\alpha = 0.05$ |
| $z_{1-\alpha/2}$ | "z, one minus alpha over two" | A normal quantile. $z_{0.975} = 1.96$, the number behind every "±2 SE" |
| $\Phi$, $\Phi^{-1}$ | "Phi", "Phi inverse" | Normal CDF and its inverse — they convert z-scores ↔ probabilities |
| $q_\alpha$ | "q sub alpha" | The $\alpha$-quantile of the bootstrap distribution |
| $\hat\theta_{(i)}$ | "theta-hat, paren i" | The estimate with observation $i$ **deleted**. Parenthesised index = "left out" |
| $\bar{\hat\theta}_{(\cdot)}$ | "theta-hat-bar, paren dot" | The average of all those leave-one-out estimates. The dot means "over every $i$" |
| $\rho$, $\rho_k$ | "rho" | Correlation; $\rho_k$ is the correlation between values $k$ steps apart |
| $\widehat{\mathrm{CV}}_k$ | "CV-hat-k" | The $k$-fold cross-validation estimate of error |
| $\#\{\,\cdot\,\}$ | "the number of" | A count of how many items satisfy the condition inside |
| $H_n$ | "the n-th harmonic number" | $1 + \tfrac12 + \tfrac13 + \dots + \tfrac1n \approx \ln n + 0.577$ |
| $n_{\text{eff}}$ | "n effective" | How many **independent** observations your correlated ones are actually worth |
| $\gamma$ | "gamma" | The decay rate of an exponential moving average |
| $\mathrm{Var}(X\mid t)$ | "variance of X given t" | The variance **once $t$ is known.** The bar means "given," never "divide" |
| $\lceil x\rceil,\ \lfloor x\rfloor$ | "ceiling of x", "floor of x" | Round up; round down |
| $\xrightarrow{n\to\infty}$ | "tends to, as n goes to infinity" | Where the quantity settles given unlimited data |

Two warnings before you start.

- **The star $^*$ means something different here than in Chapter 2.** In Chapter 2, $R^*$ and $\theta^*$ meant *optimal*. In this chapter, $\hat\theta^{*(b)}$ and $x^*_i$ mean *resampled* — a quantity computed on a synthetic dataset rather than the real one. Same glyph, different job. Context is the only guide, and in this chapter the star always sits next to a resample.
- **Three kinds of "spread" appear and get confused constantly.** The **standard deviation** $\hat\sigma$ describes how much individual *data points* differ from each other. The **standard error** $\widehat{\mathrm{SE}}$ describes how much your *estimate* would differ across repeat experiments — it is smaller by a factor of $\sqrt n$. The **confidence interval** is the standard error dressed up with a coverage claim. Reporting an SD where you meant an SE is the single most common numerical error in empirical machine learning papers.

### Full forms for this chapter's abbreviations

| Short | Full form |
|---|---|
| SE | Standard Error |
| SD | Standard Deviation |
| CI | Confidence Interval |
| MCSE | Monte Carlo Standard Error |
| BCa | Bias-Corrected and accelerated (bootstrap interval) |
| CV | Cross-Validation |
| LOO | Leave-One-Out (cross-validation) |
| OOB | Out-Of-Bag |
| EMA | Exponential Moving Average |
| MCMC | Markov Chain Monte Carlo |
| MC | Monte Carlo |
| CDF | Cumulative Distribution Function |
| CE | Cross-Entropy |
| AUC | Area Under the Curve |
| ROC-AUC | Receiver Operating Characteristic Area Under Curve |
| FID | Fréchet Inception Distance |
| RL | Reinforcement Learning |
| LR | Learning Rate |
| AdamW | Adam with decoupled **W**eight decay |
| i.i.d. | independent and identically distributed |

---

## 3.1 The problem resampling solves

### The one-line idea

You have one dataset but you need to know how much your answer would have wobbled if you'd gotten a *different* dataset. Resampling fakes the other datasets by reusing the one you have.

### The analogy

You want to know how reliable a restaurant is, but you can only eat there once. Resampling is the trick of asking each of your ten dinner companions separately what they thought, and treating the spread of their opinions as a stand-in for the spread you'd see across ten different visits. It's not perfect — they all ate on the same night — but it's far better than reporting one opinion with no error bar.

### The formal problem

You have an estimator $\hat\theta = s(X_1,\dots,X_n)$ (a test accuracy, a correlation, a model's loss). You want its sampling distribution — its standard error, a confidence interval, a $p$-value. Classical statistics gives you these in closed form for simple estimators (sample mean) and gives you nothing for complicated ones (median of a cross-validated AUC of a gradient-boosted model). Resampling gives you all of them, at the cost of compute.

#### What a "sampling distribution" is, and why it is the whole game

Four words in that paragraph do all the work. Take them in order.

**Estimator.** $\hat\theta = s(X_1,\dots,X_n)$ reads *"theta-hat equals s of X-one through X-n."* The hat means **computed from data** (Chapter 0 §0.6); $s$ is any recipe at all. Your test accuracy is an estimator. So is your validation loss, your correlation coefficient, your F1 score, your FID. **Anything you compute from a finite dataset and then quote in a table is an estimator**, and everything in this chapter applies to it.

**Sampling distribution.** This is the concept people skip, and it is the one that matters. Imagine you could collect your dataset again — same size, same source, fresh draw — and recompute $\hat\theta$. You'd get a slightly different number. Do it a thousand times and you'd have a thousand numbers with a spread and a centre. **That histogram is the sampling distribution.** It is not a distribution over data points; it is a distribution over *the answer you would have reported*.

> **Analogy.** You weigh yourself once and the scale says 74.3 kg. Step off and on again: 74.6. Again: 74.1. **The number on the display is not your weight — it is one draw from a distribution centred near your weight.** The sampling distribution is that histogram of readings, and the standard error is its width. You cannot report the reading responsibly without some sense of the width, and you got only one reading.

**Standard error.** The standard deviation of that histogram. Not the standard deviation of the *data* — the standard deviation of the *answer*. For the sample mean these differ by exactly $\sqrt n$: $\mathrm{SE} = \hat\sigma/\sqrt n$. With $\hat\sigma = 0.5$ and $n = 1024$, the data has spread 0.5 while the answer has spread $0.5/32 = 0.016$. **Thirty-two times tighter.** Confusing the two is how a paper ends up quoting error bars thirty-two times too wide, or thirty-two times too narrow.

**The problem, stated exactly.** You want the histogram. You have **one** draw from it. That is the whole difficulty, and it sounds insurmountable — and for the sample mean, mathematics solves it outright: the central limit theorem hands you the width without any repetition. But nobody's real quantity is a sample mean.

**Where classical formulas run out, concretely.** Consider the estimator in the text: *the median of a cross-validated AUC of a gradient-boosted model.* Try to derive its standard error in closed form and you must characterize, analytically: how boosting responds to resampled data, how cross-validation folds interact, how AUC (a rank statistic) distributes, and then how a median of all that behaves. **Nobody can do this. Nobody will ever do this.** And this describes essentially every number in a modern machine learning paper.

▸ **The bootstrap's move is to replace a derivation you cannot do with a simulation you can.** Where the mathematician asks *"what is the distribution of $s$ under $F$?"*, the bootstrap answers *"I'll run the experiment many times and look."* It converts a mathematics problem into a compute problem — and compute is the one resource this field has in abundance. **That trade is the reason a 1979 idea became universal only after 1990.**

> **Analogy (extending the book's dinner-companions image).** You cannot revisit the restaurant, but you can interview the companions you brought. It is an imperfect substitute — they all ate on the same night, so a bad night contaminates every opinion at once. That limitation is real and shows up later as *the bootstrap cannot tell you about anything your one sample failed to contain* (§3.2, "when the bootstrap fails"). But it is enormously better than quoting one score with no interval, which is what most of the field still does.

#### Examples and non-examples: what has a sampling distribution

**✅  estimators — every one of these has a sampling distribution and therefore an error bar**

| Example | Why it qualifies |
|---|---|
| Test accuracy $= 0.873$ on 1,000 held-out images | Computed from a finite random sample; a different 1,000 images gives a different number |
| Validation cross-entropy $= 1.524$ nats over 1,024 molecules | Same — the 1,024 were drawn, and could have been drawn differently |
| The Pearson correlation between two model scores on 200 prompts | A function of a finite draw |
| The **median** across 5 seeds of a fine-tuning run | The seeds are a random sample from the space of initializations |
| "GPT-4 beats GPT-3.5 on 62% of 300 head-to-head comparisons" | 62% is $\hat\theta$; 300 is $n$; the error bar is roughly $\pm 5.5$ points |

**❌ Near-misses — numbers that look like estimators but have no sampling distribution**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Your model's **parameter count**, 7.2 B | Not computed from data; recount it and you get 7.2 B again | A property of the architecture — exact |
| **Training loss on the exact batch you just optimized** | The "sample" was chosen *because* the model does well on it | A fitting diagnostic, not an estimate of anything |
| The number of layers, the learning rate, $\beta_2 = 0.999$ | You chose them; they were not drawn | Hyperparameters — constants |
| The **population** accuracy $\theta$ itself | It has one fixed value; it does not wobble | The thing being estimated, forever unobserved |
| Accuracy on the **whole finite universe** you care about (e.g. all 50,000 SKUs in your catalogue, evaluated on all 50,000) | You measured every unit; there is no sampling step | A census — the answer, not an estimate |

▸ **The boundary:** a number has a sampling distribution **if and only if repeating the data-collection step would change it.** Ask "if I re-ran the draw, would this number move?" If yes, it needs an interval. If no, quoting an interval on it is nonsense.

> **Common misconception.** *"Standard error and standard deviation are two names for the same thing."* They differ by a factor of $\sqrt n$, which at $n=1024$ is a factor of 32 — not a rounding difference, a different order of magnitude. **The standard deviation describes how spread out your data points are; the standard error describes how spread out your *answer* is.** The confusion is tempting because both are computed with the same `std()` call and both are reported with a $\pm$. A concrete tell: if your reported $\pm$ does not shrink when you collect more data, you have quoted a standard deviation by mistake — standard deviation converges to a constant, standard error goes to zero.

> **Common misconception.** *"My test set is large, so my measurement is basically exact."* "Large" is meaningless without the effect size you are trying to resolve. With $n=1{,}024$ and $\hat\sigma = 0.5$, your standard error is $0.016$ nats. **A 0.032-nat improvement — a very typical late-training gain — is exactly two standard errors, which is right at the edge of detectability.** The test set is simultaneously "large" and "far too small," and only the comparison to the effect size tells you which.

---

## 3.2 The bootstrap

### Mechanism

▸ **Nonparametric bootstrap.** Given data $X = (x_1,\dots,x_n)$:
> 1. Draw $x^{*}_1,\dots,x^{*}_n$ **with replacement** from $X$. This is one bootstrap sample.
> 2. Compute $\hat\theta^{*(b)} = s(x^*_1,\dots,x^*_n)$.
> 3. Repeat for $b=1,\dots,B$ (typically $B$ = 1,000 for SEs, 10,000 for CIs).
> 4. The empirical distribution of $\{\hat\theta^{*(b)}\}$ approximates the sampling distribution of $\hat\theta$.

$$\widehat{\mathrm{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B\left(\hat\theta^{*(b)} - \bar{\hat\theta^*}\right)^2}$$

#### Reading the bootstrap recipe in plain English

**The whole method in one sentence:** *treat your dataset as if it were the entire universe, and sample from it the way you originally sampled from the real universe.*

**"With replacement" is the load-bearing phrase, and it is worth being concrete.** You have 5 validation losses: $[0.9,\ 1.4,\ 1.1,\ 2.0,\ 1.6]$. To draw one bootstrap sample of size 5, you pick an index at random, write it down, **put it back**, and repeat five times. A typical result:

$$x^* = [1.4,\ 0.9,\ 1.4,\ 2.0,\ 1.4]$$

Note that $1.4$ appears three times and $1.1$ and $1.6$ do not appear at all. **That is not a bug — it is the entire mechanism.** Sampling without replacement would just hand back your original five numbers in a different order, and the mean would be identical every time, spread zero, no information. **The duplicates and the omissions are what generate the variation you are trying to measure.**

Compute the mean of each resample and you get a spread of values: $1.46$ here, maybe $1.28$ next time, $1.62$ after that. **That spread is your standard error.**

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $x^*_1,\dots,x^*_n$ | "x-star one through x-star n" | One resampled dataset. Same size $n$ as the original — this matters |
| $b$ | "b" | Which resample you're on, $1$ to $B$ |
| $\hat\theta^{*(b)}$ | "theta-hat star, b" | Your statistic recomputed on resample $b$ |
| $\bar{\hat\theta^*}$ | "theta-hat-star bar" | The average across all $B$ resamples |
| $B - 1$ | — | Bessel's correction, the same $n-1$ you use for a sample variance |

▸ **Why the resample must be the same size $n$ as the original.** The standard error depends on $n$ — it shrinks like $1/\sqrt n$. Resample 50 points instead of 100 and you will measure the wobble of a 50-point experiment, which is $\sqrt2$ times too large. **You are simulating *your* experiment, so the simulated experiment must have your experiment's size.**

**How big should $B$ be?** $B$ controls only how precisely you have measured the width — it adds no information about the data. The Monte Carlo error in $\widehat{\mathrm{SE}}$ falls like $1/\sqrt{B}$, so $B = 1{,}000$ pins the standard error to about 2% of itself, which is plenty. **Confidence intervals need more** ($B = 10{,}000$) because a 2.5% quantile is estimated from only the most extreme $2.5\%$ of your draws — with $B = 1{,}000$ that is 25 numbers, and 25 numbers make a noisy quantile. **The rule of thumb: $B$ for standard errors, $10B$ for tails.**

> **Analogy.** You have a bag of 100 marbles that you scooped from a lake containing millions. You want to know how much a *different* scoop from the lake would have differed. You cannot go back to the lake. So you scoop repeatedly **from your bag**, replacing each marble before the next draw, and watch how much those scoops differ from each other. **If your bag is a fair miniature of the lake, the variability of scoops-from-bag mimics the variability of scoops-from-lake.** The entire validity of the method rests on that "if," which is why §3.2's failure cases are all situations where the bag is *not* a fair miniature.

> **Where this came from.** The bootstrap is due to **Bradley Efron**, in a 1979 paper in the *Annals of Statistics* titled "Bootstrap Methods: Another Look at the Jackknife." The name refers to the image of pulling yourself up by your own bootstraps — and the phrase was originally, in 19th-century American usage, a description of something **physically impossible**, an absurdity; only later did it flip in ordinary speech to mean admirable self-reliance. Efron intended the older, ironic reading: you are manufacturing information about the sampling distribution using nothing but the sample itself, which looks as if it ought not to work. The method was also, in 1979, near the edge of what computers could do — a thousand refits of anything was a serious undertaking — which is why an idea from the 1970s became routine only in the 1990s. **The bootstrap is the first major statistical method designed on the assumption that computation is cheaper than cleverness**, and it will not be the last. Efron received the International Prize in Statistics in 2019, in large part for it.

#### Examples and non-examples: what counts as a bootstrap resample

Start from the same five validation losses: $X = [0.9,\ 1.4,\ 1.1,\ 2.0,\ 1.6]$, so $n=5$ and $\bar X = 1.40$.

**✅  bootstrap resamples**

| Resample | Mean | Why it qualifies |
|---|---|---|
| $[1.4,\ 0.9,\ 1.4,\ 2.0,\ 1.4]$ | $1.42$ | Size 5, drawn with replacement, duplicates allowed |
| $[0.9,\ 0.9,\ 0.9,\ 0.9,\ 0.9]$ | $0.90$ | Legal and important — probability $(1/5)^5 = 1/3125$. Rare draws populate the tails |
| $[2.0,\ 1.6,\ 1.4,\ 1.1,\ 0.9]$ | $1.40$ | The original values in a new order — legal, just uninformative on its own |
| $[1.1,\ 1.1,\ 2.0,\ 0.9,\ 1.6]$ | $1.34$ | Typical draw: one value twice, one value missing |

**❌ Near-misses — procedures that look like bootstrapping but break it**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Shuffling $X$ and recomputing | Sampling *without* replacement returns the same multiset; the mean is $1.40$ every single time, spread exactly $0$ | A permutation of the data — the engine of §3.5, not of §3.2 |
| Drawing 3 of the 5 points | Measures the wobble of a 3-point experiment, which is $\sqrt{5/3}\approx 1.29\times$ too wide | Subsampling (the $m$-out-of-$n$ bootstrap — a real method, but it needs a correction factor) |
| Drawing 500 points from the 5 | Simulates an experiment 100× larger than yours; standard error comes out $\sqrt{100}=10\times$ too small | A fantasy about a dataset you do not have |
| Adding Gaussian noise to each $x_i$ | Invents variability that was never in the data | A smoothed bootstrap (legitimate, but a *different* method with a bandwidth to choose) |
| Bootstrapping the 5 **timesteps** of one molecule and calling it a molecule-level error bar | Resamples the wrong unit — the units you resample define the population you generalize to | A within-molecule error bar. See §3.6 |

▸ **The boundary:** a bootstrap resample must have **the same size $n$** and be drawn **with replacement** from **the same unit of randomness** as the original experiment. Get the size wrong and your error bar is scaled by $\sqrt{n/m}$; get the unit wrong and your error bar answers a question nobody asked.

> **Common misconception.** *"The bootstrap creates new data, so it gives you more information than you had."* It creates no information at all. Every resample is built entirely from values already in $X$ — the bootstrap can never produce a value of $1.75$ from the five numbers above, because $1.75$ is not in the bag. **What the bootstrap manufactures is not data but the *shape of the wobble*.** The misconception is tempting because $B = 10{,}000$ resamples of 1,024 points really does involve ten million numbers, and it feels as though ten million numbers must be worth more than 1,024. They are not: the information content is frozen at $n = 1{,}024$, and pushing $B$ higher only sharpens your picture of a width that was already determined.

> **Common misconception.** *"More bootstrap resamples means a tighter confidence interval."* Raising $B$ from 1,000 to 100,000 makes your interval **more stable**, not narrower. The width is set by $n$. If you re-ran the whole thing with a different random seed, the endpoints would move less with larger $B$ — that is all you bought. **$B$ buys precision about the interval; $n$ buys the interval.**

### Why it works

The bootstrap replaces the unknown true distribution $F$ with the empirical distribution $\hat F_n$ (mass $1/n$ on each observed point). The **plug-in principle**: whatever functional of $F$ you wanted, compute it on $\hat F_n$ instead. Since $\hat F_n \to F$ (Glivenko–Cantelli, uniformly), and the functional is smooth, the bootstrap distribution converges to the true sampling distribution.

**Sanity check with the mean.** If $s$ = sample mean, the bootstrap SE is
$$\widehat{\mathrm{SE}} = \sqrt{\frac{1}{n^2}\sum_i (x_i-\bar x)^2} = \frac{\hat\sigma}{\sqrt n}$$
which reproduces the textbook formula exactly. Good — a method that gets easy cases wrong isn't worth trusting on hard ones.

#### The plug-in principle, decoded

**What $\hat F_n$ is, concretely.** $F$ is the true distribution — the lake. $\hat F_n$ is the **empirical distribution**: a distribution that puts probability exactly $1/n$ on each point you observed, and zero everywhere else. With the five losses $[0.9, 1.4, 1.1, 2.0, 1.6]$, $\hat F_n$ is a distribution over those five numbers and nothing else, each with probability $0.2$. **It is your dataset, reinterpreted as a probability distribution.** Once you see that, "drawing with replacement from your data" and "sampling from $\hat F_n$" are revealed to be the same sentence.

**The plug-in principle in one line.** Whatever you wanted to know about $F$ — its mean, its median, the distribution of some complicated statistic computed from it — **compute the same thing about $\hat F_n$ instead, and call that your estimate.** No new mathematics is required, because $\hat F_n$ is a completely known distribution you can sample from as often as you like.

▸ **This is why the bootstrap works on things nobody can analyse.** You never needed a formula for how your statistic behaves — you only needed a distribution you could *draw from*. Analysis is replaced by simulation, and simulation does not care whether your statistic is a mean or a median-of-cross-validated-AUC.

**Why $\hat F_n \to F$ — the Glivenko–Cantelli theorem.** As $n$ grows, the empirical distribution converges to the true one, and — this is the strong part — it converges **uniformly**, meaning the *worst* discrepancy anywhere on the whole curve goes to zero, not merely the discrepancy at any single point. That uniformity is exactly what you need, because your statistic might depend on any part of the distribution.

> **Analogy.** Take a photograph of a crowd. With 10 people photographed, the picture is a poor guide to the crowd's height distribution. With 10,000, it is an excellent one — and not just for the average height, but for the tallest decile, the median, the spread, everything at once. **Glivenko–Cantelli is the promise that the photograph eventually resembles the crowd in every respect simultaneously.** The bootstrap then measures the crowd by measuring the photograph.

**Reading the sanity check.** The text verifies the bootstrap against the one case where the true answer is known. For the sample mean, the exact bootstrap standard error works out to $\sqrt{\frac{1}{n^2}\sum_i(x_i-\bar x)^2}$. Pull the $\frac{1}{n^2}$ out as $\frac1n$ outside the square root and you have $\frac1n\sqrt{\sum_i(x_i-\bar x)^2} = \frac{1}{\sqrt n}\sqrt{\frac1n\sum_i(x_i-\bar x)^2} = \frac{\hat\sigma}{\sqrt n}$. **The textbook formula, recovered exactly, with no assumption of normality anywhere in the derivation.**

▸ **This kind of check deserves a name and a habit.** Before trusting a general method on a hard problem, run it on the one problem where you already know the answer. If it disagrees, you have found a bug in the method or in your implementation, cheaply. If it agrees, you have earned a little trust. **A method that gets the easy case wrong is not a method with a small flaw; it is a method you do not understand.**

**One caveat the plug-in principle carries.** $\hat F_n$ can only ever contain values you observed. If the real distribution has a long tail and your $n$ points missed it, no amount of resampling invents it — **the bootstrap cannot tell you about a region of the distribution you failed to sample.** This is the honest limitation behind every failure case in the next section.

> **Where this came from.** The theorem underwriting all of this was proved twice in 1933, by **Valery Glivenko** and **Francesco Cantelli**, publishing in the same Italian actuarial journal in the same year. It is sometimes called the **fundamental theorem of statistics** — a grand title for the modest-sounding claim that if you look at enough of something, what you have seen resembles what is there. Modest or not, it is the licence for every empirical measurement anyone has ever made.

### The 0.632 fact

The probability a given point is *omitted* from a bootstrap sample:
$$\left(1-\frac1n\right)^n \xrightarrow{n\to\infty} e^{-1} = 0.368$$

▸ So **63.2% of unique points appear** in each bootstrap sample, and 36.8% are held out. Those held-out points form a natural test set ("out-of-bag"), which is:
- how random forests get free validation error without a holdout set (Ch. 14),
- the origin of the **.632 estimator**: $\widehat{\mathrm{Err}}_{.632} = 0.368\,\overline{\mathrm{err}}_{\text{train}} + 0.632\,\widehat{\mathrm{Err}}_{\text{OOB}}$, which corrects OOB's pessimism.

#### Where 0.632 comes from, and why it's free validation

**Derive it in three lines.** You draw $n$ times with replacement from $n$ points. Focus on one particular point, say $x_7$.

1. On a single draw, the chance you *miss* $x_7$ is $1 - \frac1n$ (you had $n$ choices and only one of them was $x_7$).
2. The $n$ draws are independent, so the chance you miss it **every time** is $\left(1-\frac1n\right)^n$.
3. As $n$ grows this converges to $e^{-1} = 0.3679$.

$$\left(1-\tfrac1n\right)^n \xrightarrow{n\to\infty} e^{-1} = 0.368$$

**It converges fast.** $n=10$: $0.349$. $n=100$: $0.366$. $n=1000$: $0.3677$. $n=\infty$: $0.36788$. **So "about 37% left out, about 63% in" is accurate for any dataset you'd actually use.** With $n = 15{,}864$ molecules, roughly 5,838 are absent from any given bootstrap resample.

**Why $e$ appears here, of all places.** $e$ is *defined* as the limit of $(1+\frac1n)^n$, and $(1-\frac1n)^n$ is its reciprocal in the limit. The appearance is not a coincidence or a numerical accident: **$e^{-1}$ is what you always get when you take many independent chances to avoid something and each chance has probability $1/n$ of catching you.** The same number governs the derangement problem (the probability that no one at a party gets their own coat back is also $\to 1/e$), and the secretary problem's optimal stopping rule.

> **Analogy.** A raffle with $n$ tickets, and $n$ prizes drawn one at a time with the winning ticket returned to the drum each time. Buy one ticket and, however large $n$ is, your chance of winning nothing at all settles at 37%. Roughly a third of ticket-holders go home empty-handed, and roughly a third of prizes go to someone who has already won one. **The duplicates and the omissions are the same phenomenon counted from two sides.**

▸ **Out-of-bag validation is the payoff, and it is close to free.** Every bootstrap resample leaves ~37% of the data untouched. That 37% is a  held-out set *for the model trained on that resample* — never seen, never fit. Average the errors over all resamples and you have an honest validation estimate **without ever setting aside a permanent test split.** For a random forest of 500 trees, each data point is out-of-bag for roughly $500\times0.368 = 184$ of them, so every point gets an honest prediction from 184 models. **You paid nothing: the trees had to be trained anyway.**

**Why OOB is pessimistic, and what the .632 formula does about it.** Each bootstrap model is trained on only $0.632n$ *distinct* points, not $n$. Less data means a slightly worse model, so its held-out error is slightly too high. Meanwhile training error is famously too *low*. The .632 estimator splits the difference by exactly the right weights:

$$\widehat{\mathrm{Err}}_{.632} = 0.368\,\overline{\mathrm{err}}_{\text{train}} + 0.632\,\widehat{\mathrm{Err}}_{\text{OOB}}$$

**Put numbers on it.** Training error $0.10$, OOB error $0.30$: $0.368(0.10) + 0.632(0.30) = 0.037 + 0.190 = \mathbf{0.227}$. A blend that pulls the pessimistic estimate down by an amount calibrated to how much data the model actually saw.

▸ **A warning, and it is a real one.** The .632 estimator breaks badly for models that interpolate — a 1-nearest-neighbour classifier has *zero* training error by construction, so the first term contributes nothing but the weight $0.368$ is still assigned to it, dragging the estimate far too low. Efron and Tibshirani's **.632+** estimator (1997) adjusts the weights using an estimate of how much the model overfits. **Any modern deep network also has near-zero training error**, so treat plain .632 as a classical-models tool and reach for straightforward cross-validation otherwise.

> **Where this came from.** Out-of-bag estimation and the whole bootstrap-in-machine-learning tradition trace to **Leo Breiman**, who introduced **bagging** in 1996 — a contraction of "**b**ootstrap **agg**regat**ing**" — and **random forests** in 2001. Breiman had an unusual career for a statistician: after teaching mathematics at UCLA in the 1960s he left academia entirely and spent thirteen years as a full-time consultant, returning to a statistics chair at Berkeley in 1980 with a strong conviction that the field had drifted too far from prediction problems that actually arise. That conviction produced his 2001 essay "Statistical Modeling: The Two Cultures," which argued that statistics had over-invested in assuming data-generating models and under-invested in algorithms judged by predictive accuracy — an argument that reads today like a description of what machine learning became.

#### Examples and non-examples: when out-of-bag validation is honest

**✅ Situations where OOB is a  held-out estimate**

| Example | Why it qualifies |
|---|---|
| A 500-tree random forest on 15,864 molecules; each tree scored on the ~5,838 molecules it never saw | Each tree was fit without those points. No leakage, no separate split |
| Bagged decision stumps, OOB error averaged over resamples | Same argument, one tree at a time |
| Bagged linear regressions where hyperparameters were fixed **before** any bagging began | The choice of hyperparameters did not consult OOB scores |

**❌ Near-misses — OOB numbers that are no longer honest**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Tuning `max_depth` by picking the depth with the lowest OOB error, then quoting that OOB error | The number was minimized over choices, so it inherits the selection bias of §3.6 | A **training** signal for hyperparameters. You now need a fresh holdout |
| OOB error from a forest trained on data that was standardized using the **full** dataset's mean and variance | The preprocessing already saw the held-out points | Leakage — a mild but real optimistic bias |
| OOB error when rows are **duplicated** in the source data (the same molecule appearing under two IDs) | A "held-out" row can be an exact copy of a training row | Memorization measured and mislabelled as generalization |
| OOB error for time series, where row $t$ is out-of-bag but rows $t-1$ and $t+1$ trained the tree | Neighbouring timesteps carry nearly the same information | An estimate of interpolation, not of forecasting |
| .632 error for a 1-nearest-neighbour classifier | Training error is exactly 0 by construction, so $0.368 \times 0 = 0$ silently deletes a third of the estimate | A number biased far too low — this is why .632+ exists |

▸ **The boundary:** a held-out estimate is honest exactly when **nothing about the held-out points influenced the model or the choice of model.** Being absent from the training resample is necessary, not sufficient — preprocessing, duplicates, temporal neighbours, and hyperparameter selection all leak information across the boundary while leaving the row itself technically "out of bag."

> **Common misconception.** *"Out-of-bag validation is free, so I never need a test set."* It is free for estimating the error **of a fixed procedure**. The moment you use OOB scores to *choose* something — a depth, a feature set, a model family — you have optimized against them, and the winning score is biased low by exactly the mechanism §3.6 quantifies. The misconception is tempting because OOB  does eliminate the need for a *validation* split in the classical setting; what it never eliminates is the need for data untouched by selection.

> **Common misconception.** *"63.2% is a property of the bootstrap that you can tune."* It is not a knob; it falls out of drawing $n$ times from $n$ with replacement, and it is $1 - e^{-1}$ for any $n$ above about 20. If you want a different holdout fraction you must change the *procedure* (draw $m \ne n$ points), which changes the standard error the resample is measuring. **The 0.632 and the correct error bar are locked together — you cannot adjust one without disturbing the other.**

### Confidence intervals, in increasing order of correctness

1. **Normal:** $\hat\theta \pm z_{1-\alpha/2}\widehat{\mathrm{SE}}_{\text{boot}}$. Requires approximate normality.
2. **Percentile:** take the $\alpha/2$ and $1-\alpha/2$ quantiles of $\{\hat\theta^{*(b)}\}$. Transformation-respecting but biased if $\hat\theta$ is biased.
3. **Basic / pivotal:** $[2\hat\theta - q_{1-\alpha/2},\ 2\hat\theta - q_{\alpha/2}]$. Reflects the bootstrap distribution — correct when the *shape* is right but the location is shifted.
4. **BCa (bias-corrected and accelerated):** adjusts the percentile levels using
$$\alpha_1 = \Phi\!\left(\hat z_0 + \frac{\hat z_0 + z_{\alpha/2}}{1 - \hat a(\hat z_0 + z_{\alpha/2})}\right)$$
with $\hat z_0 = \Phi^{-1}\!\left(\frac{\#\{\hat\theta^{*(b)} < \hat\theta\}}{B}\right)$ (bias correction) and $\hat a$ from the jackknife skewness. **Second-order accurate** ($O(1/n)$ coverage error vs $O(1/\sqrt n)$ for percentile). Use this if you're publishing.

#### The four intervals, decoded

**First, what a confidence interval claims.** A 95% interval does **not** say "there is a 95% chance the truth is in this range" — the truth is a fixed number, not a random one. It says: *"if I repeated this whole experiment many times and built an interval this way each time, 95% of those intervals would contain the truth."* **The randomness is in the interval, not in the truth.** The word for how often it succeeds is **coverage**, and an interval that claims 95% but achieves 88% is said to *undercover*. Every ranking below is a ranking by coverage.

**The shared notation:**

- $\alpha$ — the miss rate. A 95% interval has $\alpha = 0.05$, so $\alpha/2 = 0.025$ in each tail.
- $z_{1-\alpha/2}$ — the standard normal quantile. $z_{0.975} = 1.96$. **This is the 1.96 in "±1.96 standard errors."**
- $\Phi$ — the normal CDF: feed it a z-score, get back a probability. $\Phi(1.96) = 0.975$.
- $\Phi^{-1}$ — its inverse: feed it a probability, get back a z-score. $\Phi^{-1}(0.975) = 1.96$.
- $q_\alpha$ — the $\alpha$-quantile of your $B$ bootstrap values: sort them, take the one $\alpha$ of the way along.
- $\#\{\hat\theta^{*(b)} < \hat\theta\}$ — read *"the number of bootstrap replicates that came out below the original estimate."*

**Method 1 — Normal.** $\hat\theta \pm 1.96\,\widehat{\mathrm{SE}}$. You used the bootstrap only to get the *width*, then assumed the shape is a bell curve. With $\hat\theta = 1.524$ and $\widehat{\mathrm{SE}} = 0.153$: $[1.224,\ 1.824]$. Fast, familiar, and wrong whenever the sampling distribution is skewed — which for ratios, correlations, and anything bounded, it is.

**Method 2 — Percentile.** Sort your 10,000 bootstrap values and read off the 250th and the 9,750th. **No normality assumed at all**, and it has a lovely property: it is *transformation-respecting*. If you build an interval for $\theta$ and then take logs of the endpoints, you get exactly the interval you'd have built for $\log\theta$ directly. Intervals built the normal way do not have this property, and it matters because the log/logit scale is usually where symmetry lives.

**Method 3 — Basic / pivotal.** $[2\hat\theta - q_{1-\alpha/2},\ 2\hat\theta - q_{\alpha/2}]$. **Note the reversal**: the *upper* bootstrap quantile forms the *lower* endpoint. That looks like an error and is not.

> **Analogy for the reversal.** A rifle whose sights are misaligned puts most shots 4 cm right of the bullseye. To find the bullseye given a shot, you move **left**. The bootstrap distribution shows you where estimates land *relative to a known centre* (your $\hat\theta$); the pivotal interval flips that displacement around to say where the truth might be relative to your estimate. **Percentile assumes the bootstrap distribution has the truth's shape; pivotal assumes it has the truth's *error* shape, mirrored.** They differ exactly when the distribution is asymmetric, which is exactly when it matters.

**Method 4 — BCa.** Two corrections stacked on the percentile method.

- **Bias correction $\hat z_0$.** Count how many bootstrap replicates fell below your original estimate. If the answer is exactly half, $\Phi^{-1}(0.5) = 0$ and no correction is applied. If only 40% fell below, $\hat z_0 = \Phi^{-1}(0.40) = -0.253$, and the interval shifts to compensate for a systematically off-centre estimator. **It is a measurement of your estimator's bias, taken from the bootstrap itself.**
- **Acceleration $\hat a$.** Computed from the jackknife (§3.3), it measures how fast the standard error *changes with* the value of $\theta$. Many estimators are noisier at large values than at small ones — a proportion near 0.5 has more variance than one near 0.02 — so a symmetric interval is wrong even after centring. $\hat a$ makes the two tails different widths.

▸ **"Second-order accurate" means the coverage error shrinks like $O(1/n)$ instead of $O(1/\sqrt n)$.** With $n = 100$: percentile intervals are off by roughly $1/\sqrt{100} = 10\%$ — a nominal 95% interval delivering maybe 88%. BCa is off by roughly $1/100 = 1\%$, delivering about 94%. **A tenfold improvement in honesty for a modest amount of extra arithmetic**, which is why the text's advice is unambiguous: if the number is going in a paper, use BCa.

**A practical ordering.** Exploring, want a rough width: percentile with $B = 1{,}000$. Publishing, or the distribution looks skewed: BCa with $B = 10{,}000$. Reaching for the normal interval: only when you have checked that the bootstrap histogram actually looks like a bell — **and you can check, because you have 10,000 draws sitting right there. Plot them.**

> **Where this came from.** The BCa interval is Efron's own refinement, published in 1987 as "Better Bootstrap Confidence Intervals." A recurring pattern in this chapter: the person who invents a method spends the following decade discovering the ways it is subtly wrong and repairing them. The gap between "the idea works" and "the idea is trustworthy at the third decimal place" is usually about eight years of careful, unglamorous follow-up.

#### Examples and non-examples: what "95% confidence" licenses you to say

Take a concrete result: a 95% bootstrap interval for validation cross-entropy of $[1.42,\ 1.63]$, with point estimate $1.524$.

**✅ Statements this interval  supports**

| Statement | Why it qualifies |
|---|---|
| "The procedure that produced this interval captures the true value 95% of the time it is run." | This is the literal definition of coverage |
| "A value of 1.70 is not well supported by this data." | 1.70 lies outside the interval, so the corresponding null would be rejected at $\alpha = 0.05$ |
| "I cannot distinguish this model from one whose true loss is 1.60." | 1.60 is inside the interval |
| "Quadrupling my evaluation set would roughly halve this width to about $\pm 0.05$." | Width scales as $1/\sqrt{n}$ |

**❌ Near-misses — statements that sound identical but are not licensed**

| Sounds right | Why it isn't | What it actually is |
|---|---|---|
| "There is a 95% probability the true loss lies in $[1.42,\ 1.63]$." | $\theta$ is a fixed number; it is either in there or not, with probability 1 or 0. The randomness lives in the *interval* | A **credible** interval — the Bayesian object, which requires a prior |
| "95% of future validation losses will land in this range." | The interval is about the *mean*, not about individual draws — those are $\sqrt{n}$ times more spread out | A **prediction** interval, which is much wider |
| "95% of my bootstrap replicates fell in here, so the truth probably does too." | True of the percentile method by construction, but the bootstrap distribution is centred on $\hat\theta$, not on $\theta$ | A description of your resamples, not an inference |
| "Model A's interval overlaps Model B's, therefore they are not significantly different." | Overlapping intervals can still correspond to a significant *difference*; the correct object is an interval for the **paired difference** | A conservative and often wrong shortcut |
| "The interval is narrow, so the estimate is accurate." | Narrow means low **variance**. A systematically biased estimator gives tight intervals around the wrong number | A statement about precision, silent about bias |

▸ **The boundary:** a confidence interval is a statement about **the long-run behaviour of the procedure**, never about the probability of a particular claim. Everything you may say with it must survive the rewrite "if I re-ran this experiment many times…".

> **Common misconception.** *"Overlapping error bars mean no significant difference."* Two 95% intervals can overlap substantially while a paired test on the same data returns $p < 0.01$. The reason is correlation: if Model A and Model B are evaluated on **the same** molecules, the two errors move together, and the difference $A - B$ has far less variance than either estimate alone. **Always bootstrap the difference, not the two numbers separately.** This one mistake is probably the single most common statistical error in machine learning papers, and it costs the field real discoveries — it throws away true effects, not just false ones.

> **Common misconception.** *"BCa is more accurate, so it always gives a tighter interval."* BCa gives a *better-centred* interval. Applied to a skewed sampling distribution it frequently gives a **wider** one on the side where the estimator has a long tail, and narrower on the other. Accuracy here means "the coverage is closer to 95%," which for an undercovering percentile interval means getting **bigger**, not smaller.

### When the bootstrap fails

- **Extremes.** Bootstrapping $\max(X_i)$ is inconsistent — the max of a resample is one of finitely many observed values, so the bootstrap distribution is atomic and never converges to the continuous true one. *Anything that depends on the maximum or minimum of your data is bootstrap-hostile.* **This includes "best validation loss."** Keep that in mind for §3.6.
- **Dependent data.** i.i.d. resampling destroys autocorrelation. Use **block bootstrap** (resample contiguous blocks of length $\ell \sim n^{1/3}$) for time series — or for training curves.
- **Very small $n$** ($<20$) or heavy tails without finite variance.
- **Parameters on a boundary** (e.g. variance component at 0).

#### Why maxima break the bootstrap — the one failure you will actually hit

**Work the failure with tiny numbers, because it is completely concrete.** Five observations: $[3.1,\ 4.8,\ 2.2,\ 7.9,\ 5.0]$. The maximum is $7.9$. Now bootstrap it.

- The maximum of any resample can **only** be one of those five values — you are drawing from the values you have, so nothing above $7.9$ can ever appear.
- The chance that $7.9$ is included in a given resample is $1 - 0.368 = 0.632$, and when it is included it is the max.
- So the bootstrap distribution of the maximum is: $7.9$ with probability $0.632$, $5.0$ with most of the remainder, and so on. **A lumpy staircase sitting on five specific values.**

The *true* sampling distribution of the maximum is smooth and continuous, and — crucially — it puts real probability on values **above** $7.9$, because a fresh sample from the world could contain a larger number. **The bootstrap assigns that region probability zero, forever, and increasing $n$ does not fix it.** That is what "inconsistent" means: the estimate does not converge to the right answer no matter how much data you collect.

> **Analogy.** You want to know how tall the tallest person in the next room might be. Measure everyone in *this* room, then repeatedly re-poll this room. **However many times you re-poll, you will never produce a number taller than the tallest person you have already met.** The quantity you are estimating lives specifically in the part of the distribution your sample failed to reach, and re-examining your sample cannot reach it. Averages are safe from this because the middle of a distribution is well populated; extremes are not, because the edge of a distribution is by definition where you have almost no data.

▸ **The general rule, worth stating as a principle: the bootstrap works for statistics that are *smooth* functions of the data, and fails for statistics that depend on one specific observation.** A mean uses every point a little. A maximum uses one point entirely, and which point that is jumps discontinuously as the data changes. Medians sit in between, which is why the jackknife (a linear approximation) fails on them while the bootstrap survives.

**And now the sentence that should make you sit up: "This includes 'best validation loss.'"**

Your `best.pt` checkpoint is selected by taking a **minimum over epochs**. That is an extreme-value statistic of a noisy sequence, and everything above applies to it directly:

- The reported best is one specific observation, not a smooth summary.
- Its distribution is dominated by the luckiest draw, not by the typical one.
- Resampling your epochs cannot tell you about the better value a re-run would have found.
- And it is **biased optimistic by construction**, because a minimum over noise is systematically below the truth.

▸ **§3.6 computes the size of that bias for a real run and gets 0.30 nats — two full standard deviations of pure self-deception, from a procedure everybody uses and nobody questions.** This section is where that result is set up; keep it in mind for twenty pages.

**The other three failure modes, briefly, each with its tell:**

| Failure | Tell | Fix |
|---|---|---|
| **Dependent data** | Successive points are correlated (time series, training epochs) | **Block bootstrap**: resample contiguous blocks of length $\ell\sim n^{1/3}$ so within-block correlation is preserved |
| **Very small $n$** ($<20$) | $\hat F_n$ is too coarse to be a fair miniature of $F$ | Parametric bootstrap, or an exact test, or collect more data |
| **Boundary parameters** | The truth sits at the edge of the allowed range (a variance of exactly 0) | Constrained or parametric methods |

**Why dependence breaks it, in one sentence.** I.i.d. resampling shuffles your points into a random order, which **destroys** the autocorrelation that was carrying information — the resamples look far more random than reality, so you underestimate the true uncertainty. Training curves are strongly autocorrelated ($\rho_1 \approx 0.7$ routinely, per §3.7), so **if you ever bootstrap a training curve, use blocks.**

> **Where this came from.** The general conditions under which the bootstrap is and is not consistent were worked out by **Peter Bickel and David Freedman** in 1981, two years after Efron's paper — and the maximum is the canonical counterexample, the one that appears in every subsequent textbook. **The failure modes were mapped almost immediately, which is a mark of a healthy idea:** the interesting thing about the bootstrap was never that it always works, but that it works so broadly and fails in ways you can name.

### Variants worth knowing

- **Parametric bootstrap:** fit a model $F_{\hat\eta}$, simulate from it. Lower variance if the model is right, badly wrong if it isn't.
- **Bayesian bootstrap:** instead of multinomial counts, draw Dirichlet$(1,\dots,1)$ weights. Smoother; equivalent to a noninformative Bayesian posterior over $F$.
- **Wild bootstrap:** for regression with heteroskedastic errors, resample residuals multiplied by random signs.

---

## 3.3 The jackknife

**Mechanism:** leave one observation out at a time. $\hat\theta_{(i)} = s(x_1,\dots,x_{i-1},x_{i+1},\dots,x_n)$.

▸ $$\widehat{\mathrm{Bias}}_{\text{jack}} = (n-1)\left(\bar{\hat\theta}_{(\cdot)} - \hat\theta\right),\qquad \widehat{\mathrm{SE}}_{\text{jack}} = \sqrt{\frac{n-1}{n}\sum_{i}\left(\hat\theta_{(i)} - \bar{\hat\theta}_{(\cdot)}\right)^2}$$

Note the unusual $(n-1)/n$ **multiplier** rather than $1/(n(n-1))$ — because deleting one point out of $n$ perturbs $\hat\theta$ by only $O(1/n)$, so you must inflate the observed spread.

**Relationship to the bootstrap:** the jackknife is a *linear approximation* to the bootstrap (it estimates the influence function $\mathrm{IF}(x_i) \approx (n-1)(\bar{\hat\theta}_{(\cdot)}-\hat\theta_{(i)})$). It's cheaper ($n$ refits vs $B$) but fails for non-smooth statistics — famously, it fails for the median.

**Where it survives in ML:** the "acceleration" constant $\hat a$ in BCa; influence functions for data attribution and for identifying mislabeled training points.

#### Unpacking the jackknife

**The mechanism, in words.** Delete observation 1, recompute your statistic. Put it back, delete observation 2, recompute. Carry on through all $n$. You now have $n$ estimates, each built from $n-1$ points, and their spread tells you how sensitive your answer is to any single data point.

**The notation trips people up, so:** $\hat\theta_{(i)}$ — with the index **in parentheses** — means "the estimate computed with point $i$ **left out**." Compare $\hat\theta_i$ (no parentheses), which would mean "the $i$-th component of $\theta$." **Parentheses mean deletion.** And $\bar{\hat\theta}_{(\cdot)}$ — bar on top, dot in the parentheses — means "average all $n$ of those deletion estimates"; the dot is a placeholder reading *"over every index."*

**Work it with five numbers.** Data $[2, 4, 6, 8, 10]$, statistic = the mean, so $\hat\theta = 6$.

| Left out | Remaining | $\hat\theta_{(i)}$ |
|---|---|---|
| 2 | 4,6,8,10 | 7.0 |
| 4 | 2,6,8,10 | 6.5 |
| 6 | 2,4,8,10 | 6.0 |
| 8 | 2,4,6,10 | 5.5 |
| 10 | 2,4,6,8 | 5.0 |

$\bar{\hat\theta}_{(\cdot)} = 6.0$. Now the two formulas:

**Bias estimate.** $(n-1)(\bar{\hat\theta}_{(\cdot)} - \hat\theta) = 4(6.0 - 6.0) = 0$. Correct — the sample mean is unbiased, and the jackknife knows it.

**Standard error.** $\sqrt{\frac{n-1}{n}\sum_i(\hat\theta_{(i)} - 6)^2} = \sqrt{\frac45(1 + 0.25 + 0 + 0.25 + 1)} = \sqrt{\frac45(2.5)} = \sqrt{2} = 1.414$.

Check against the textbook answer: $\hat\sigma/\sqrt n$ with $\hat\sigma = \sqrt{10} = 3.162$ gives $3.162/\sqrt5 = 1.414$. ✓ **Exact agreement**, same sanity check the bootstrap passed in §3.2.

▸ **Now the multiplier that looks wrong.** For an ordinary sample variance you divide by $n(n-1)$; here you *multiply* by $(n-1)/n$ — a factor differing by $(n-1)^2$. The reason is that leave-one-out estimates are **artificially similar to each other**: deleting 1 point out of 1,000 changes the answer by about $1/1000$, so the observed spread is roughly $n$ times too small to represent real sampling variability. **The multiplier inflates a deliberately-shrunken spread back to full size.** Whenever you see a strange-looking constant in a resampling formula, this is usually the reason: something was measured on the wrong scale and has to be rescaled.

> **Analogy.** To measure how much a bridge sways in a storm, you cannot summon a storm — so you push it gently by hand and measure the millimetre of deflection, then scale that up by a known factor. The $(n-1)/n$ multiplier is that scaling. **The jackknife nudges; the bootstrap actually shakes the bridge.**

**"A linear approximation to the bootstrap," explained.** The **influence function** $\mathrm{IF}(x_i)$ measures how much your answer moves per unit of weight placed on point $i$ — the derivative of your statistic with respect to that observation's presence. The jackknife estimates that derivative by finite differences: *remove the point, see how far the answer moved.* It then assumes the statistic behaves **linearly** in the data, and adds up the individual influences.

▸ **That linearity assumption is the whole story of when it works and when it doesn't.** The bootstrap makes large, simultaneous perturbations (many points duplicated, many dropped) and needs no linearity. The jackknife makes one tiny perturbation at a time and extrapolates. **A statistic with a kink — a median, a maximum, a rank — has a derivative that does not describe its behaviour, and the extrapolation fails.**

**Why the median specifically.** With an odd number of points, the median is one particular observation. Delete a point above it and the median shifts to the next value down; delete one below and it shifts up. **The $n$ leave-one-out medians take only two or three distinct values**, so their spread reflects the gap between adjacent order statistics rather than  sampling variability. The bootstrap handles medians fine; the jackknife does not. **The rule: use the jackknife on smooth statistics only, and "smooth" excludes anything defined by sorting.**

**Why it survives at all.** It is $n$ refits rather than $B = 10{,}000$ — cheap when a refit is expensive. And it gives you the influence of each individual point, which is a *different and often more useful* output than an error bar. In machine learning that shows up as **influence functions**: rank your training points by how much each one moved the model, and the top of that list is an excellent detector of mislabeled data. **The jackknife's by-product turned out to be more valuable to this field than its main product.**

> **Where this came from.** **Maurice Quenouille** proposed the leave-one-out bias correction in 1949 and developed it in 1956. **John Tukey** extended it to variance estimation in 1958 and gave it the name: a **jackknife** is the folding pocket knife you carry everywhere — not the best tool for any particular job, but always available and adequate for most. It was a deliberately modest name for a deliberately rough-and-ready tool. Tukey had a gift for this: he also coined "**bit**" (from *binary digit*), "**software**," and the box plot, and co-invented the fast Fourier transform. Efron's 1979 bootstrap paper is subtitled *"Another Look at the Jackknife"* precisely because the bootstrap was conceived as the jackknife's more general successor — **which makes the jackknife's survival inside BCa a small piece of poetic justice: the superseded method is a component of its own replacement.**

---

## 3.4 Cross-validation

### Mechanism and the bias–variance of $k$

Partition into $k$ folds; train on $k-1$, test on 1; average.

$$\widehat{\mathrm{CV}}_k = \frac1k\sum_{j=1}^k \hat R^{(j)}$$

| $k$ | Training set size | Bias of the estimate | Variance | Cost |
|---|---|---|---|---|
| 2 | $n/2$ | high (pessimistic) | low | 2 fits |
| 5 | $0.8n$ | moderate | moderate | 5 fits |
| 10 | $0.9n$ | small | moderate | 10 fits |
| $n$ (LOO) | $n-1$ | ~none | **high** | $n$ fits |

#### Reading the cross-validation table

**The mechanism, without symbols.** Cut your data into $k$ equal piles. Set pile 1 aside, train on the other $k-1$, and measure the error on pile 1. Put it back; set pile 2 aside; repeat. After $k$ rounds every point has been predicted exactly once by a model that never saw it. Average the $k$ error measurements. **That average is $\widehat{\mathrm{CV}}_k$**, and $\hat R^{(j)}$ is the error on fold $j$ — the superscript in parentheses means "fold number," not a power.

**Why bias and variance move in opposite directions as $k$ grows** — this is the entire content of the table, and it has two separate causes.

*The bias, and why small $k$ is pessimistic.* With $k=2$ you train on half your data. A model trained on 50% of the data is  worse than the one you will finally ship (trained on 100%), so its measured error is too high. **You are honestly measuring the wrong model.** With $k=10$ you train on 90%, nearly the real thing, so the bias nearly vanishes. Leave-one-out trains on $n-1$ points and is essentially unbiased.

*The variance, and why large $k$ is noisy.* This is the counterintuitive half, and the formula below is the explanation.

**Choosing $k$, with the cost column in view.** $k = 5$ costs 5 fits and gets you 80% training data. $k = n$ costs $n$ fits — for $n = 15{,}864$, that is 15,864 full training runs — and buys you a small reduction in bias in exchange for a large increase in variance. **Nobody runs leave-one-out on a neural network, and the reason is not only compute.**

> **Analogy.** You want to know how a student will perform on an exam. Give them ten practice tests, each covering 90% of the syllabus, and average the scores: each test is a fair proxy for the real thing, and the ten scores differ enough to tell you about consistency. Now give them fifteen thousand practice tests, each omitting exactly one question. **Every test is nearly identical to every other, so the fifteen thousand scores tell you almost nothing more than the first one did** — you have spent an enormous amount of effort re-measuring the same thing.

▸ **Why LOO has high variance:** the $n$ training sets are nearly identical (they overlap in $n-2$ points), so the $n$ estimates are highly *correlated*. Averaging correlated things doesn't reduce variance:
$$\mathrm{Var}\!\left(\frac1n\sum \hat R^{(i)}\right) = \frac{\sigma^2}{n}\big(1 + (n-1)\rho\big) \xrightarrow{\rho\to1} \sigma^2$$
When $\rho \approx 1$, you have effectively **one** sample no matter how large $n$ is. This formula — the correlated-average variance — is one of the most useful in all of ML. It explains LOO's variance, it explains why ensembles of correlated models don't help (Ch. 14), and it explains why the 16 validation batches don't average away as much noise as you'd hope.

#### The correlated-average formula, decoded — learn this one

The text calls it one of the most useful formulas in machine learning, and that is not overstatement. Read it aloud: *"the variance of the average of n things, each with variance sigma-squared and pairwise correlation rho, equals sigma-squared over n times the quantity one plus n minus one times rho."*

$$\mathrm{Var}\!\left(\frac1n\sum_i \hat R^{(i)}\right) = \frac{\sigma^2}{n}\big(1 + (n-1)\rho\big)$$

**Every symbol:**

- $\sigma^2$ — the variance of any one of the things you're averaging.
- $\rho$ — the **pairwise correlation** between any two of them. $\rho = 0$ means unrelated; $\rho = 1$ means identical.
- $n$ — how many you're averaging.

**Check the two ends first, because they anchor everything.**

- $\rho = 0$ (independent): the bracket becomes $1$, so $\mathrm{Var} = \sigma^2/n$. **The familiar result** — averaging $n$ independent things divides the variance by $n$.
- $\rho = 1$ (identical): the bracket becomes $1 + (n-1) = n$, so $\mathrm{Var} = \sigma^2$. **No reduction whatsoever.** Averaging the same measurement $n$ times teaches you nothing, which is obvious once said and invisible in the formula until you evaluate it.

**Now the middle, where reality lives.** Rewrite it as $\sigma^2\left(\rho + \frac{1-\rho}{n}\right)$ — the same expression, rearranged so the structure is visible. **The first term has no $n$ in it.** As $n\to\infty$ the second term vanishes and you are left with $\sigma^2\rho$, a floor you cannot cross.

▸ **The correlation sets a floor that no amount of averaging can break through.** This is the single most consequential fact in the formula, and it recurs everywhere in this book.

**Put numbers on it — $\sigma^2 = 1$, and watch the collapse:**

| $\rho$ | $n=10$ | $n=100$ | $n=\infty$ (the floor) |
|---|---|---|---|
| 0.0 | 0.100 | 0.010 | 0 |
| 0.1 | 0.190 | 0.109 | 0.10 |
| 0.5 | 0.550 | 0.505 | 0.50 |
| 0.9 | 0.910 | 0.901 | 0.90 |
| 0.99 | 0.991 | 0.990 | 0.99 |

**At $\rho = 0.9$, going from 10 samples to 100 improves your variance by 1%.** You did ten times the work for nothing. The effective sample size is roughly $\frac{1}{\rho}$ once $n$ is large — **at $\rho = 0.9$, an infinite number of correlated samples is worth about 1.1 independent ones.**

> **Analogy.** You want an unbiased opinion of a film. Ask a hundred people who all watched it together in the same room, on the same night, after the same dinner conversation. Their opinions are heavily correlated, so the hundredth opinion adds almost nothing to the first ten. **A handful of  independent viewers beats a hundred correlated ones, and the crossover happens far earlier than intuition suggests.**

**Now the three places it explains something, exactly as the text promises.**

1. **Leave-one-out's variance.** Two LOO training sets share $n-2$ of their $n-1$ points, so $\rho \approx 1$. The $n$ fold-estimates are nearly the same number, the bracket goes to $n$, and $\mathrm{Var}\to\sigma^2$. **You ran $n$ training jobs and got the statistical power of roughly one.**
2. **Ensembles of correlated models.** Ten models trained on the same data with the same architecture and different seeds might have $\rho = 0.8$; averaging them gets you $\sigma^2(0.8 + 0.02) = 0.82\sigma^2$, an 18% reduction, not 90%. **This is why random forests deliberately handicap each tree with random feature subsets** — accepting a worse individual tree in exchange for a lower $\rho$ is a good trade, because $\rho$ is what sets the floor.
3. **The 16 validation batches in §3.6.** They share the model, they share the epoch, and they are drawn from a common timestep distribution. They do not average down like 16 independent measurements, and §3.6 computes exactly how badly.

▸ **The transferable habit: whenever you average things to reduce noise, ask what they have in common.** Whatever that is, it does not average away. **You are not buying $1/\sqrt n$; you are buying $\sqrt{\rho + (1-\rho)/n}$**, and if $\rho$ is not near zero, most of the reduction you were counting on never arrives.

**$k=5$ or $10$ is the standard compromise.** Bengio & Grandvalet (2004) proved there is **no unbiased estimator of the variance of $k$-fold CV**, so the error bars people put on CV numbers are all heuristic.

### The variants you must not confuse

- **Stratified $k$-fold:** preserve class proportions per fold. Mandatory for imbalanced data.
- **Grouped $k$-fold:** all rows sharing a group ID (patient, molecule scaffold, user) go in the same fold. **For molecular data this is essential** — scaffold splitting rather than random splitting, or your model gets credit for memorizing a scaffold it saw in training.
- **Time-series CV:** expanding or rolling window, never shuffle. Random CV on time series leaks the future.
- **Nested CV:** an outer loop for performance estimation, an inner loop for hyperparameter selection. **If you tune hyperparameters on the same CV you report, your number is optimistically biased** — the amount of bias grows with the number of configurations tried (see §3.6, it's the same maximum-of-noise problem).

#### The four variants, and the leak each one plugs

Every variant exists because **random splitting quietly assumes your rows are exchangeable**, and in real data they usually aren't. Each variant names a specific way that assumption fails and repairs it.

| Variant | The assumption that fails | What leaks if you ignore it |
|---|---|---|
| **Stratified** | Every fold has the same class mix | With 2% positives and 10 folds, a fold might contain zero positives. Metrics become undefined or wildly noisy |
| **Grouped** | Every row is a separate observation | Ten measurements of the same patient split across folds: the model recognizes the patient, not the disease |
| **Time-series** | Order doesn't matter | Training on Friday to predict Thursday. **You have given the model the future** |
| **Nested** | The number you report wasn't used to make choices | You tuned on the score you're quoting, so you're quoting a maximum over noise |

**Grouped $k$-fold, in molecular terms, since the book flags it as essential.** A **scaffold** is a molecule's core skeleton; a chemical series may contain fifty near-identical compounds sharing one scaffold with small substitutions. Split randomly and forty of them land in training, ten in test. **The model does not need to learn chemistry — it needs to learn "I have seen this skeleton, the answer was about 5.2."** Test performance looks superb, and the model is worthless on a  new scaffold, which is the only case anyone cares about.

▸ **The diagnostic that catches this in five minutes: if your random-split score is far better than your grouped-split score, the gap is the amount of memorization you were about to publish as understanding.** Report the grouped number. The same test applies to patient IDs in medical data, user IDs in recommendation, document IDs in text, and near-duplicate images in vision — the last of which has quietly inflated more benchmark results than any other single cause.

**Time-series leakage, stated as concretely as possible.** Shuffle a stock price series and train on a random 80%. Some of your training rows are from *after* the rows you're testing on. The model can learn "the price on day $t$ is close to the price on day $t+1$," which is true, useless, and unavailable at prediction time. **Your backtest is spectacular; your deployment loses money.** The fix is an expanding or rolling window: always train on the past, always test on the future, never shuffle.

**Nested CV, and why the inner loop is not optional.** Suppose you try 50 hyperparameter configurations and pick the best by cross-validation score. That best score is a **minimum over 50 noisy draws** — the exact phenomenon §3.6 dissects. With per-configuration noise of $\sigma$, the expected optimism is roughly $\sigma\,\Phi^{-1}(1-\frac{1}{51}) \approx 2.05\sigma$. **Two standard errors of pure selection bias, before the model has done anything.** Nested CV fixes this by holding out an *outer* fold that the tuning loop never touches: the inner loop chooses, the outer loop reports, and the two never share data.

▸ **The one-line principle behind all four variants, and it is the principle behind this whole chapter: whatever you select on, you cannot also report.** Selection consumes the honesty of a measurement. If you tuned on it, tested on it, or took a minimum over it, you need a fresh untouched slice to report from — and if you don't have one, say so.

> **Where this came from.** Cross-validation as a formal method arrived essentially simultaneously from two directions: **Mervyn Stone**'s "Cross-Validatory Choice and Assessment of Statistical Predictions" (1974) and **Seymour Geisser**'s "The Predictive Sample Reuse Method" (1975), with related work by David Allen the same year. Stone showed in 1977 that leave-one-out cross-validation and the Akaike information criterion (AIC) are asymptotically equivalent — a satisfying bridge, since one is a brute-force computational procedure and the other an analytic penalty derived from information theory, and nobody had expected them to be the same thing. **Bengio and Grandvalet's 2004 result** that there is *no unbiased estimator of the variance of $k$-fold cross-validation* is worth sitting with: cross-validation is one of the most widely used procedures in the field, and the error bars routinely printed beside its output are, without exception, heuristic.

#### Examples and non-examples: is this split actually held out?

**✅  held-out evaluations**

| Example | Why it qualifies |
|---|---|
| Scaffold-split test set: every Bemis–Murcko scaffold appears in exactly one fold | The model cannot have seen the skeleton it is being asked about |
| Time-series: train on Jan–Sep, validate Oct, test Nov–Dec, in that order | No information flows backwards in time |
| Patient-grouped folds: all 14 scans of patient #221 in fold 3 only | The unit of generalization (a *new patient*) matches the unit of splitting |
| Normalization statistics ($\mu, \sigma$) computed on the training fold and **applied** to the test fold | The test rows contributed nothing to the transform |
| Nested CV: inner 5-fold picks $\lambda$, outer fold reports the score | The reported number never influenced a choice |

**❌ Near-misses — splits that look held out and aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| `train_test_split(X, y, random_state=42)` on a table where each patient has 14 rows | Rows of the same patient land on both sides | A test of patient re-identification |
| Fitting `StandardScaler` on all data, *then* splitting | Test means and variances are baked into the training features | Preprocessing leakage — typically worth a few percent of illusory accuracy |
| SMOTE / oversampling applied before splitting | Synthetic test points are interpolations *of training points* | Near-duplicate leakage; can push a 0.70 AUC to 0.95 |
| Random $k$-fold on stock prices | Some training rows postdate some test rows | A model that exploits the future |
| Reporting the best of 50 hyperparameter configurations' CV scores | Best-of-50 is a maximum over noise, biased by $\approx 2.05\sigma$ | A *selection* score, not a performance estimate |
| A "fresh" test set scraped from the same website as training, months apart | Near-duplicate pages, boilerplate, and reposts recur | Partial memorization measured as generalization |
| Deduplicating with exact string match on a corpus full of paraphrases | Exact match misses the near-duplicates, which are the ones that matter | An incomplete dedup — check with MinHash or embeddings |

▸ **The boundary:** a split is honest when **the unit you split on is the unit you claim to generalize to**, and when **no computation touching the test rows preceded the split.** Both halves are required. Grouping alone does not save you from preprocessing leakage, and correct preprocessing does not save you from splitting on the wrong unit.

> **Common misconception.** *"Leakage is a bug you would notice, because the score would be suspiciously perfect."* Most leakage is subtle — a few points of AUC, well within the range you were hoping for, which is precisely why it survives review. **Leakage that produces 0.999 gets caught in an afternoon; leakage that produces 0.87 instead of 0.81 gets published.** The tempting belief here is that a plausible number is a trustworthy one, and plausibility is exactly what a small leak manufactures.

> **Common misconception.** *"Leave-one-out cross-validation is the best kind because it uses the most training data."* LOOCV has the lowest bias and often the **highest variance** — the $n$ models are trained on almost identical datasets, so their errors are highly correlated, and the correlated-average formula above shows correlation is what stops averaging from helping. $\rho$ near 1 means the variance barely falls with $n$. It is also $n$ times the compute. **Five- or ten-fold usually beats it on both axes**, which is why $k=5$ and $k=10$ became the defaults despite LOOCV being the "purer" idea.

---

## 3.5 Permutation tests

**The question:** is my measured effect bigger than chance?

**Mechanism:** shuffle the labels $y$ many times, recompute the statistic each time, and see where your real statistic sits in that null distribution.

▸ $$p = \frac{1 + \#\{b : T^{(b)}_{\text{perm}} \ge T_{\text{obs}}\}}{1 + B}$$

The $+1$ in numerator and denominator is not a fudge — it makes the test **exact** (valid at level $\alpha$ for finite $B$) by including the observed permutation in the reference set.

**Assumption:** exchangeability under the null. Shuffle only what the null says is exchangeable. For grouped data, permute *within* groups.

**Where you should use it:** "is model A actually better than model B on my test set?" Compute per-example loss differences $d_i = \ell_A(i) - \ell_B(i)$, then permute the *signs* of $d_i$ (sign-flip test). This is the correct test and almost nobody in ML runs it.

#### Permutation tests, decoded

**The idea is the most intuitive in all of statistics, and needs no distributions at all.** You measured an effect. To decide whether it is real, you **deliberately destroy** the thing that could have caused it, remeasure, and repeat many times. If your real effect looks like the destroyed ones, you have nothing. If it stands outside them, you have something.

**The procedure, concretely, on the model-comparison example.** You have 1,000 test molecules. For each, model A's loss and model B's loss. Compute $d_i = \ell_A(i) - \ell_B(i)$ — a per-example difference. Suppose the mean difference is $\bar d = -0.032$ (A better by 0.032). Now:

1. Flip a fair coin for each of the 1,000 differences. Heads, keep $d_i$; tails, negate it. **Under the null hypothesis that the two models are equally good, the sign of each difference is arbitrary**, so a sign-flipped dataset is just as plausible as the real one.
2. Recompute the mean. Get, say, $+0.004$.
3. Repeat $B = 10{,}000$ times. You now have 10,000 mean differences from a world where the models are identical.
4. Ask: how many of those 10,000 are as extreme as $-0.032$? If the answer is 3, your $p$-value is about $4/10001 = 0.0004$ and the difference is real. If the answer is 4,200, it isn't.

**Every symbol in the $p$-value formula:**

- $T_{\text{obs}}$ — the statistic you actually measured.
- $T^{(b)}_{\text{perm}}$ — the statistic on the $b$-th shuffled dataset.
- $\#\{b : T^{(b)}_{\text{perm}} \ge T_{\text{obs}}\}$ — read *"the number of $b$ for which the shuffled statistic was at least as extreme as the observed one."*
- $B$ — how many shuffles.

▸ **Why the $+1$ top and bottom is not a fudge.** Without it, a statistic that beat all $B$ shuffles would get $p = 0/B = 0$ — a claim of *literal impossibility* from a finite simulation, which is nonsense. The $+1$ says: **include the observed arrangement itself in the reference set**, because under the null it is one of the equally likely arrangements. That makes the test **exact** — its false-positive rate is  at most $\alpha$, for any finite $B$, with no approximation. **The smallest $p$-value you can ever report is $1/(B+1)$**, which is an honest statement about the resolution of your simulation. With $B = 10{,}000$ that floor is $10^{-4}$; if you need to claim smaller, run more permutations.

> **Analogy.** You believe a particular coach improves player performance. Take the season's results and **randomly reassign which players the coach worked with**, then recompute the improvement. Do it ten thousand times. If a randomly assigned "coach" produces an effect as large as the real one in 40% of shuffles, the coach has shown you nothing. If it happens 3 times in 10,000, something is going on. **You never needed to know how player performance is distributed — you only needed to be able to shuffle.** That freedom from distributional assumptions is the entire selling point.

**Exchangeability, the one assumption — and the one people break.** The test is valid only if, under the null, the shuffled arrangements are  as likely as the observed one. **Shuffle only what the null says is interchangeable.** Two common failures:

- **Grouped data.** Ten measurements per patient: shuffling labels across patients breaks the patient structure and produces a null that is far too optimistic. **Permute within groups**, or permute whole groups.
- **Time series.** Shuffling destroys the autocorrelation, again making the null too tight. Use block permutations, for the same reason §3.2 recommends the block bootstrap.

**Why the sign-flip test is the right one for model comparison, and why it is so rarely run.** The standard alternative is a two-sample test comparing model A's losses against model B's losses — which throws away the **pairing**. Both models saw the *same* molecules, and molecule difficulty varies enormously (some are hard for everyone). The paired difference $d_i$ cancels that shared difficulty out entirely.

▸ **Numbers: if per-molecule loss has SD 0.5 but the *difference* between models has SD 0.05, the paired test is ten times more sensitive on identical data.** A comparison that unpaired analysis calls insignificant, paired analysis resolves cleanly. **It costs nothing but the discipline of logging per-example losses instead of only their averages** — and that single logging habit is what separates a test you can run from a test you can't.

> **Where this came from.** The permutation test is **R. A. Fisher's**, introduced in *The Design of Experiments* (1935) and illustrated with the famous **lady tasting tea**: a colleague, Muriel Bristol, claimed she could tell whether milk had been poured into the cup before or after the tea. Fisher's response was to design an experiment — eight cups, four prepared each way, presented in random order — and then to observe that the randomization *itself* supplies the reference distribution: under the null that she is guessing, all $\binom{8}{4} = 70$ ways of choosing four cups are equally likely, so getting all four right by luck has probability $1/70 = 0.014$. **No normal distribution, no variance estimate, no assumptions whatsoever — the probability comes from the design of the experiment.** (Accounts of how well Bristol actually did vary, and Fisher did not record the outcome in the book; the story is told for its logic rather than its result.) **E. J. G. Pitman** developed permutation tests systematically in 1937–38. Fisher regarded randomization tests as the foundation and the familiar normal-theory tests as *convenient approximations to them* — an ordering the field has largely inverted, mostly because permutation tests were computationally impossible for the fifty years after he proposed them. **They stopped being impossible around 1980 and most of the field never revisited the default.**

#### Examples and non-examples: what a $p$-value of 0.03 does and does not mean

**✅ Statements a $p = 0.03$  licenses**

| Statement | Why it qualifies |
|---|---|
| "If the two models were truly identical, data this lopsided would arise about 3% of the time." | This is the definition: $P(\text{data this extreme} \mid \text{null})$ |
| "At a pre-registered $\alpha = 0.05$, I reject the null." | 0.03 < 0.05 |
| "Running this analysis on 100 -null comparisons would produce about 5 rejections." | The false-positive rate is the thing $\alpha$ controls |
| "This says nothing about *how much* better A is — for that I need the effect size and its interval." | Correct, and the most under-used sentence in the field |

**❌ Near-misses — readings of $p = 0.03$ that are simply wrong**

| Sounds right | Why it isn't | What it actually is |
|---|---|---|
| "There is a 97% chance model A is better." | Reverses the conditioning: $P(\text{data}\mid\text{null}) \ne P(\text{null}\mid\text{data})$ | The posterior probability, which needs a prior |
| "There is a 3% chance the result is a fluke." | Same reversal. The 3% is computed *assuming* the null, so it cannot be the probability of the null | A common paraphrase in press releases, and wrong |
| "$p = 0.03$ means the effect is big." | $p$ mixes effect size with $n$. A trivial 0.001-nat gap reaches $p=0.03$ given enough examples | Evidence *against a null*, silent about magnitude |
| "$p = 0.06$ means no effect." | Absence of evidence at an arbitrary threshold is not evidence of absence | An underpowered result, usually |
| "I ran 20 comparisons and one hit $p = 0.03$, so that one is real." | At $\alpha = 0.05$, one hit in 20 is exactly what pure noise produces | Multiple-comparisons inflation; needs Bonferroni or FDR control |
| "$p = 0$" from 10,000 permutations | The floor is $1/(B+1) \approx 10^{-4}$ | "$p < 10^{-4}$" — say that instead |

▸ **The boundary:** a $p$-value is $P(\text{data at least this extreme} \mid \text{the null is true})$ — a probability computed **in a world you do not believe in**, about data, not about hypotheses. Every wrong reading above comes from flipping the conditioning bar.

> **Common misconception.** *"Statistical significance means practical significance."* With 15,864 validation molecules, a difference of 0.004 nats can be highly significant and completely irrelevant — it will not change a single downstream decision. Conversely, a 0.05-nat improvement measured on 200 examples may be practically enormous and statistically invisible. **Significance answers "is it there?"; effect size answers "does it matter?"** The misconception is tempting because significance is a single number with a bright line at 0.05, and effect sizes require you to know your own domain.

> **Common misconception.** *"Permutation tests are assumption-free."* They are free of *distributional* assumptions — you never assume normality. They are **not** free of the exchangeability assumption, and that one is easy to violate silently: shuffling labels across patients when ten rows belong to one patient produces a null distribution that is far too narrow, and therefore $p$-values that are far too small. **You traded an assumption you could check for an assumption you must think about.**

> **Common misconception.** *"The two models' mean losses are 1.524 and 1.556, so A is better."* That comparison ignores the pairing and the noise. A 0.032-nat gap against a per-molecule standard error near 0.016 is exactly two standard errors — borderline — but the *paired* sign-flip test on the same data may resolve it decisively, because paired differences have far smaller spread. **The right test can turn "borderline" into "certain" without collecting one extra molecule.**

---

## 3.6 The statistical autopsy of the Case Study A validation curve

Everything above now gets applied to one fully specified training run — **Case Study A**, the book's reference example. The numbers are realistic and the reasoning transfers to any noisy evaluation metric.

### The setup, restated with numbers

```
Val set:      15,864 molecules
Measured on:  16 batches × 64 = 1,024 molecules  (6.45%)
Sampling:     fresh random subset each epoch, fresh random t per batch
Metric:       val_realCE
Observed:     best@22 = 1.556 → best@36 = 1.547 → best@37 = 1.524 → best@43
```

#### Reading the setup block

Before the arithmetic, make sure the five lines in that box mean something to you — every number in this section is derived from them.

| Line | What it says | Why it will matter |
|---|---|---|
| `Val set: 15,864` | The full validation pool | The population you're sampling *from* |
| `Measured on: 16 × 64 = 1,024` | Only 6.45% is evaluated each epoch | **Noise source 1**: you measured a sample, not the pool |
| `fresh random subset each epoch` | Different molecules every time | Fresh noise per epoch — nothing cancels between epochs |
| `fresh random t per batch` | **Only 16 draws of $t$** | **Noise source 2**: the killer, because 16 is a very small number |
| `Observed: 1.556 → 1.547 → 1.524` | The improvements being chased | Total movement: **0.032 nats** |

**What $t$ is, in one sentence, for readers who haven't reached Chapter 20.** In a diffusion model, $t$ is the **noise level** applied to an example before the model sees it: at $t\approx 0$ the input is nearly clean and prediction is easy; at $t\approx T$ it is nearly destroyed and prediction is nearly impossible. **The loss therefore depends enormously on which $t$ you happened to draw** — and that single fact is what §3.6 is about.

**A nat** is a unit of information, the natural-logarithm cousin of a bit ($1$ nat $\approx 1.44$ bits). Cross-entropy in nats is your model's average surprise at the truth (Chapter 1 §1.4).

▸ **Fix the shape of the problem in your mind before the algebra: the quantity being chased is 0.032, and the section is about to show that the measurement's own wobble is roughly 0.15.** Everything that follows is the arithmetic of that mismatch — and the mismatch, not the model, is what explains every pattern in the training curve.

### Noise source 1: molecule sampling

You measure 1,024 of 15,864. If the per-molecule CE has standard deviation $\sigma_{\text{mol}}$ (for molecular CE this is typically 0.3–0.8 nats; call it 0.5):

$$\mathrm{SE}_{\text{mol}} = \frac{\sigma_{\text{mol}}}{\sqrt{1024}} \cdot \underbrace{\sqrt{1 - \tfrac{1024}{15864}}}_{\text{finite-population correction} = 0.967} = \frac{0.5}{32}\times 0.967 = \mathbf{0.0151}$$

Already comparable to the 0.032 improvement being chased. But this is **not the dominant term.**

#### Unpacking the first standard error

**The base formula is the $\sigma/\sqrt n$ from Chapter 1 §1.3.1**, applied to molecules: measure 1,024 of them, and the average has a standard error of $\sigma_{\text{mol}}/\sqrt{1024} = 0.5/32 = 0.0156$. The $\sqrt{1024} = 32$ is worth noticing on its own — **a thousand samples buys you a factor of thirty-two, not a factor of a thousand.**

**The new piece is the finite-population correction.** Read $\sqrt{1 - \frac{n}{N}}$ as: *"the fraction of the population you have* not *measured, square-rooted."*

- $n = 1{,}024$ measured, $N = 15{,}864$ total, so $n/N = 0.0645$.
- $\sqrt{1 - 0.0645} = \sqrt{0.9355} = 0.967$.
- Multiply: $0.0156 \times 0.967 = \mathbf{0.0151}$.

▸ **Why sampling a *finite* pool is easier than sampling an infinite one.** The usual $\sigma/\sqrt n$ assumes you draw from an inexhaustible population, so there is always more you haven't seen. But if you measured **all** 15,864 molecules, there would be no sampling error at all — you'd have the exact answer. The correction interpolates smoothly between the two: at $n = N$ it becomes $\sqrt{1-1} = 0$, and at $n \ll N$ it becomes $\approx 1$ and disappears.

> **Analogy.** Estimating the average age in a town of 15,864 by surveying 1,024 people leaves plenty of unknown. Surveying 15,000 of them leaves almost none — the remaining 864 can only shift the answer so far. **The correction is the mathematics of "you're nearly done."** At 6.45% you are nowhere near done, which is why it only buys 3.3%.

**How much would it help to evaluate the full pool?** Setting $n = N$ removes noise source 1 entirely, saving $0.0151$ nats. Hold that number: it is why §3.6's fix list ranks "evaluate on the full val set" *sixth*, not first. **Removing 0.015 of noise from a total of 0.153 is not where the leverage is** — and identifying which noise term dominates *before* optimizing is the entire method being demonstrated here.

**On the $\sigma_{\text{mol}} = 0.5$ estimate.** It is a stated assumption, not a measurement, and the text is explicit about that. **You could measure it in one line** — take the per-molecule losses you already computed and call `.std()` on them. If you are ever in this situation, do that instead of assuming; the whole point of §3.7 is that you should already be logging the per-example values that make it possible.

### Noise source 2: timestep sampling — this is the killer

You draw a **fresh random $t$ per batch**. Only 16 batches ⇒ only **16 draws of $t$**.

Apply the law of total variance (Ch. 1 §1.3.2):

▸ $$\mathrm{Var}(\widehat{\mathrm{CE}}) = \underbrace{\frac{\mathbb{E}_t[\mathrm{Var}(\text{CE}\mid t)]}{1024}}_{\text{molecule noise}} + \underbrace{\frac{\mathrm{Var}_t\big(\mathbb{E}[\text{CE}\mid t]\big)}{16}}_{\text{timestep noise}}$$

The second term divides by **16**, not 1024. And $\mathrm{Var}_t(\mathbb{E}[\text{CE}\mid t])$ is *large* — in a discrete diffusion model, CE at $t\approx0$ (nearly clean input) might be 0.2 nats, and at $t\approx T$ (fully corrupted) it approaches $\log K$. If the conditional mean CE ranges over roughly $[0.3, 2.4]$, a rough uniform-range estimate gives $\sigma_t \approx (2.4-0.3)/\sqrt{12} = 0.61$.

$$\mathrm{SE}_{t} = \frac{0.61}{\sqrt{16}} = \mathbf{0.152}$$

▸ **Total:** $\mathrm{SE} = \sqrt{0.0151^2 + 0.152^2} = 0.153$ nats.

**The measurement noise is roughly 5× larger than the improvement you're trying to detect.**

Even if my $\sigma_t$ estimate is off by 3×, you'd still have $\mathrm{SE} \approx 0.05 > 0.032$. The conclusion is robust to the assumption.

#### The law of total variance, decoded — this is the section's engine

**The law in one sentence:** *"Total variance = the average of the within-group variances, plus the variance of the group averages."* Every symbol in the two-term formula is one of those two ideas.

**Read the pieces:**

- $\mathrm{Var}(\text{CE}\mid t)$ — read *"the variance of cross-entropy **given** $t$."* The bar means "given," never "divide" (Chapter 0, Trap 5). This is: **fix the noise level, then ask how much molecules differ from each other.**
- $\mathbb{E}_t[\mathrm{Var}(\text{CE}\mid t)]$ — average that within-$t$ variance over the $t$ values you might draw. **Noise *within* groups.**
- $\mathbb{E}[\text{CE}\mid t]$ — the *mean* cross-entropy at noise level $t$. A number for each $t$.
- $\mathrm{Var}_t(\mathbb{E}[\text{CE}\mid t])$ — how much those group means differ from each other. **Noise *between* groups.**

▸ **And now the crucial part, which is not the formula but the divisors.** The first term is divided by **1,024** — one draw per molecule. The second is divided by **16** — one draw per batch. **You drew $t$ sixty-four times less often than you drew molecules, and variance divides by the number of draws.** That single asymmetry is the whole diagnosis.

> **Analogy.** You want the average height of adults in a country. You visit 16 towns and measure 64 people in each. Two things vary: people differ *within* a town, and towns differ *from each other*. Your person-to-person noise is averaged over all 1,024 people and shrinks fast. But your **town-to-town** noise is averaged over only **16 towns**, and if towns  differ — one is a fishing village, another a basketball academy — that term dominates completely. **You do not have a sample of 1,024. You have a sample of 16 towns, with the within-town question answered very precisely.** Measuring 640 people per town instead of 64 would barely help; visiting 160 towns would.

**Now the numbers, and they are stark.**

*The within-$t$ term:* $\frac{0.5^2}{1024} = \frac{0.25}{1024} = 2.4\times10^{-4}$. Square root: $0.0156$.

*The between-$t$ term.* Where does $\sigma_t \approx 0.61$ come from? The mean CE ranges over roughly $[0.3, 2.4]$ across noise levels — near-clean inputs are easy (0.3 nats), near-destroyed inputs approach the entropy of a uniform guess. For a quantity spread roughly uniformly over a range $[a,b]$, the standard deviation is $(b-a)/\sqrt{12}$, so $(2.4-0.3)/3.464 = 0.61$. Then:
$$\frac{0.61^2}{16} = \frac{0.372}{16} = 0.0233 \quad\Rightarrow\quad \sqrt{0.0233} = \mathbf{0.152}$$

*Combining.* Variances add, standard errors don't: $\sqrt{0.0156^2 + 0.152^2} = \sqrt{0.000243 + 0.0231} = \sqrt{0.0233} = \mathbf{0.153}$.

▸ **Look at what happened to the small term.** $0.0156$ contributed $0.000243$ to a total of $0.0233$ — **one percent.** Because variances add as squares, **the larger noise source doesn't merely dominate, it annihilates the smaller one.** Eliminating molecule noise entirely would move the total from $0.1526$ to $0.1520$. This is the general law of noise budgets: **find the biggest term and fix that; everything else is decoration** — and it is exactly why "evaluate on the full validation set" is fix number six.

**The ratio, stated plainly.** $0.153$ of measurement noise against a $0.032$ improvement. **The ruler's tick marks are five times wider than the thing being measured.** No amount of staring at the number 1.524 recovers information the measurement never collected.

**On robustness.** The text notes that even a 3× error in $\sigma_t$ leaves $\mathrm{SE}\approx 0.05 > 0.032$. That is the right way to close an estimate built on assumptions: **don't defend the number, show that the conclusion survives the number being wrong.** A conclusion that needs $\sigma_t$ to be exactly 0.61 would be worthless; one that survives $\sigma_t$ anywhere from 0.2 to 1.8 is a finding.

> **A note on the name.** The law of total variance is sometimes taught as **"Eve's law,"** a mnemonic for its shape: **E**xpectation of the **V**ariance, plus **V**ariance of the **E**xpectation. Its partner, the law of total expectation ($\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X\mid Y]]$), is then "Adam's law." Silly names,  effective at making the structure stick — you can reconstruct the formula from the mnemonic alone, which is more than most people can do from the derivation.

### The consequence: what a "best" record actually means

If per-epoch reads have SD $\approx 0.15$ around a slowly-drifting true value, then:

- A single epoch dipping to 1.524 tells you almost nothing about whether the true CE is 1.52 or 1.60.
- The recorded "best" is $\min$ over 43 noisy draws. **The minimum of noisy draws is a biased estimator of the true minimum.** With 43 draws from $\mathcal{N}(\mu, 0.15^2)$, the expected minimum is
$$\mathbb{E}[\min] \approx \mu - 0.15\cdot\Phi^{-1}\!\left(1-\tfrac{1}{44}\right) \approx \mu - 0.15\times 2.02 = \mu - 0.30$$
Two full standard deviations of **optimistic bias**. The saved `best.pt` checkpoint is, with high probability, *not* the best model — it's the model that got the luckiest validation draw.

This is exactly the bootstrap-fails-on-maxima problem from §3.2, and it's the same phenomenon as hyperparameter-selection bias in §3.4. **Selecting on noise is selecting noise.**

#### Why the minimum of noisy draws lies to you

**The claim, stated without symbols: if you take 43 noisy readings of a fixed quantity and keep the lowest, that lowest reading is not an estimate of the quantity. It is an estimate of "the lowest of 43 readings," which is a different and systematically smaller thing.**

**Decode the formula.** $\mathbb{E}[\min] \approx \mu - \sigma\,\Phi^{-1}\!\left(1 - \frac{1}{n+1}\right)$:

- $\mu$ — the true value you want.
- $\sigma = 0.15$ — the noise on each reading, from the previous section.
- $n = 43$ — how many readings.
- $\Phi^{-1}$ — the inverse normal CDF: hand it a probability, it returns how many standard deviations out that is.
- $1 - \frac{1}{n+1} = 1 - \frac{1}{44} = 0.9773$ — read as *"where does the most extreme of 43 draws typically sit?"* If you slice a distribution into $n+1$ equal-probability strips, the extreme draw typically lands near the outermost boundary.
- $\Phi^{-1}(0.9773) = 2.00$ — **two standard deviations.**

So $\mathbb{E}[\min] \approx \mu - 0.15\times 2.00 = \mu - 0.30$.

▸ **Your recorded best is about 0.30 nats below the truth — and the improvement you are trying to detect is 0.032.** The selection bias is **nine times larger than the effect.** The number in `best.pt`'s filename is dominated by luck, not by quality.

**How the bias grows with the number of tries:**

| Epochs $n$ | $\Phi^{-1}(1-\frac{1}{n+1})$ | Optimism at $\sigma = 0.15$ |
|---|---|---|
| 5 | 0.97 | 0.15 nats |
| 20 | 1.67 | 0.25 nats |
| 43 | 2.00 | **0.30 nats** |
| 100 | 2.33 | 0.35 nats |
| 1,000 | 3.09 | 0.46 nats |

**Notice the shape: it grows, but slowly — roughly like $\sqrt{2\ln n}$.** Doubling your epochs does not double the bias. **But it never stops growing, and it never goes the other way.** Train longer and your reported best gets a little more optimistic, forever, entirely independent of whether the model improved.

> **Analogy — the winner's curse.** Twenty oil companies bid on a drilling lease. Each estimates the field's value honestly, with error. The company that wins is the one whose estimate was *highest*, which means the winner is systematically the one that overestimated most. **The winner tends to overpay, not because anyone was foolish, but because winning an auction is a selection event.** Your `best.pt` won an auction against 42 other epochs, and it won by having the most favourable measurement error. **Selection is not measurement.**

**Three consequences that should change how you work:**

1. **Your `best.pt` is probably not your best model.** With $\sigma = 0.15$ and true differences between epochs of ~0.03, which epoch "wins" is decided mostly by the draw. The checkpoint you shipped is the model that got the friendliest validation subset and the friendliest set of 16 timesteps.
2. **The number you report is inflated and you cannot correct it after the fact** — not from the same data. The only honest fix is a fresh, untouched evaluation of the selected checkpoint, which is fix number six in the list below and takes minutes.
3. **This is the identical mechanism as hyperparameter-selection bias.** Trying 50 configurations and reporting the best is taking a minimum over 50 noisy draws: $\Phi^{-1}(1-\frac{1}{51}) = 2.05$, so about $2.05\sigma$ of optimism. **Same formula, same table, different label on the axis.** Anywhere a maximum or minimum is taken over noisy candidates, this bias is present and quantifiable.

▸ **The connection to §3.2 is now exact, not metaphorical.** The bootstrap fails on maxima because a maximum depends on one lucky observation rather than smoothly on all of them. "Best validation loss" *is* a maximum-type statistic. **It inherits every pathology of extreme-value estimation, including the fact that resampling your existing data cannot repair it.** The phrase "selecting on noise is selecting noise" is not rhetoric — it is a summary of this table.

#### Examples and non-examples: is this number contaminated by selection?

**✅ Numbers that are clean — no maximum was taken over noise**

| Example | Why it qualifies |
|---|---|
| Validation loss at **epoch 30**, chosen in advance because that is where the schedule ends | The epoch was fixed before any score was seen |
| The **mean** of all 43 epoch losses | An average, not an extreme — averaging is not selection |
| Loss of `best.pt` re-evaluated on a **fresh** set of molecules never used for checkpointing | The new draw is independent of the selection event |
| A model chosen on validation, then scored once on a sequestered test set | The test set had no vote in the choice |
| Loss at the epoch where a **pre-registered** early-stopping rule (patience 10) first fired | The rule, not the scores, made the decision |

**❌ Near-misses — numbers that look like measurements but are maxima over noise**

| Looks like a measurement | Why it isn't | What it actually is |
|---|---|---|
| The loss printed in `best.pt`'s filename | It is the minimum of 43 noisy reads: biased low by about $2.00\sigma = 0.30$ nats | An order statistic — $\mathbb{E}[\min]$, not $\mu$ |
| "Best-of-5-seeds" result quoted as the model's performance | $\Phi^{-1}(1-\tfrac16) = 0.97$, so about $0.97\sigma$ optimistic | The extreme of 5 draws |
| Best cross-validation score across 50 hyperparameter configurations | About $2.05\sigma$ optimistic | A selection score |
| "Best checkpoint on the test set" | Selecting on test converts the test set into a validation set | A validation score, with no test set remaining |
| A leaderboard's top entry, where 200 teams submitted | The winner is the team whose noise was friendliest, by about $2.7\sigma$ | An extreme-value statistic about the *leaderboard* |
| The peak of a smoothed validation curve | Smoothing shrinks $\sigma$, so it shrinks the bias — it does not remove it | A smaller, still-present selection bias |

▸ **The boundary:** a number is contaminated exactly when **the same data both chose it and scored it.** Averaging over candidates is safe; taking an extreme over candidates is not. The size of the contamination is $\sigma \cdot \Phi^{-1}\!\left(1-\frac{1}{n+1}\right)$, which you can compute in one line and should.

> **Common misconception.** *"Selection bias will average out if I repeat the experiment."* It is a **bias**, not noise: it points the same direction every single time. Run 100 training runs and take each one's best epoch, and all 100 reported numbers are too low, by roughly the same 0.30 nats. **Averaging 100 biased numbers gives a very precise estimate of the wrong quantity.** The misconception is tempting because most problems in this chapter *do* shrink with repetition — this one does not, and telling the two apart is the point of the whole section.

> **Common misconception.** *"A longer training run gives you a better `best.pt`, because you had more chances."* More chances raises the *reported* number's optimism — 0.30 nats at 43 epochs, 0.46 at 1,000, from the table above — while doing nothing for the model's true quality unless the model was  still improving. **You are buying a better-looking number, not a better model**, and you can tell the two apart only by re-evaluating on data that took no part in the choosing.

### Record statistics: the exact null

Here is the rigorous version of "is a 13-epoch dry spell meaningful?"

**Null hypothesis:** the true performance is *flat*, and epoch reads are i.i.d. continuous draws.

Under this null, for continuous i.i.d. draws, the probability that draw $n$ is a new minimum is exactly $1/n$ (by exchangeability — each of the first $n$ draws is equally likely to be the smallest). Therefore:

▸ $$\Pr(\text{no new record in epochs } 23\ldots35) = \prod_{k=23}^{35}\left(1-\frac1k\right) = \prod_{k=23}^{35}\frac{k-1}{k} = \frac{22}{35} = \mathbf{0.629}$$

The product telescopes. **A 13-epoch gap after a record at epoch 22 happens 63% of the time under pure noise with zero improvement.** It is not evidence of anything. It is the single most likely category of outcome.

More generally, the expected number of records in $n$ epochs is the harmonic number:
$$\mathbb{E}[\#\text{records}] = H_n = \sum_{k=1}^n \frac1k \approx \ln n + 0.5772$$
For $n=43$: $H_{43} \approx 4.34$. **You should expect about 4 "best" records in 43 epochs even from a model that is not improving at all.** Records get rarer as $1/n$; that's arithmetic, not optimizer behaviour.

### The part that *is* evidence: the cluster

Under the same null, records at epochs 36 **and** 37 back-to-back:

▸ $$\Pr = \frac{1}{36}\times\frac{1}{37} = \frac{1}{1332} = 7.5\times10^{-4}$$

Under the null, that's a 1-in-1300 event. Under the alternative — a , continuously improving model whose true CE has drifted below the old floor — consecutive records are *expected*, because once the true mean drops well below 1.556, most epochs clear the bar.

▸ **The asymmetry is the whole lesson: the dry spell carries no information; the cluster carries all of it.**

You can turn this into a proper test. Under the null, the indicator that epoch $k$ is a record is Bernoulli$(1/k)$, and these indicators are **independent** across $k$ (a classical and slightly surprising result). So the number of records in epochs $[a,b]$ has a known distribution and you can compute an exact $p$-value for "more records than chance," and even a runs test for clustering. If you log full curves, do this — it's 10 lines of code and it converts vibes into a number.

### Diagnosing whether the model is improving during the "quiet" stretch

The right question isn't "did the record move?" It's "did the *trend* move?" Fit a regression to the epoch-wise CE:

$$\text{CE}_e = \alpha + \beta e + \varepsilon_e$$

With $\sigma_\varepsilon = 0.15$ and $E = 43$ epochs, the standard error of the slope is
$$\mathrm{SE}(\hat\beta) = \frac{\sigma_\varepsilon}{\sqrt{\sum_e (e-\bar e)^2}} = \frac{0.15}{\sqrt{E(E^2-1)/12}} = \frac{0.15}{\sqrt{43\cdot1848/12}} = \frac{0.15}{81.4} = 0.0018\ \text{nats/epoch}$$

▸ So a **regression on the whole curve can detect a true drift of 0.004 nats/epoch at 2σ**, while a single-epoch record comparison needs a jump of ~0.3 nats. The trend line is roughly **80× more statistically powerful** than the min-tracker, using exactly the same data you already logged.

This is the single most important practical takeaway in the chapter: **stop reading the min, start reading the slope.**

### Fixes, ranked by leverage

1. ▸ **Fix the validation $t$ values across epochs.** Use a fixed stratified grid — e.g. for 16 batches, assign $t_j = \lfloor T(j-0.5)/16 \rfloor$ for $j=1..16$, or use fixed antithetic pairs. This is **common random numbers**, and it kills noise source 2 entirely for *epoch-to-epoch comparisons*: the between-$t$ variance becomes a constant offset shared by every epoch rather than fresh noise.

   Formally, for the difference between epochs, $\mathrm{Var}(\hat A - \hat B) = \mathrm{Var}(\hat A)+\mathrm{Var}(\hat B) - 2\mathrm{Cov}(\hat A,\hat B)$. With shared $t$ and shared molecules, $\mathrm{Cov}$ is nearly as large as the variances, and the difference variance collapses by an order of magnitude. **This costs zero extra compute.**

2. ▸ **Fix the validation molecule subset** (same 1,024, same seed, every epoch). Same argument. Between them, fixes 1 and 2 typically reduce the effective noise from ~0.15 to ~0.02.

3. **Stratify $t$ rather than sampling it uniformly at random.** Stratified sampling with $m$ strata reduces variance to the within-stratum variance only:
   $$\mathrm{Var}_{\text{strat}} = \frac{1}{m}\sum_j \frac{\sigma_j^2}{n_j} \ll \frac{\sigma^2_{\text{total}}}{n}$$
   For a quantity whose mean varies strongly with $t$ (yours does), this is a huge win.

4. **Report a per-$t$ breakdown.** A single scalar averaging over all noise levels hides where the model improved. Log CE in, say, 10 $t$-buckets. You will very likely find the model is improving steadily at low $t$ while high-$t$ CE is pinned at its entropy floor, and the aggregate is being dominated by the pinned part.

5. **Use an EMA of the validation metric for checkpoint selection**, not the raw min:
   $$\tilde{\text{CE}}_e = \gamma\tilde{\text{CE}}_{e-1} + (1-\gamma)\text{CE}_e,\quad \gamma\approx0.8$$
   This has effective sample size $\frac{1+\gamma}{1-\gamma} = 9$ epochs, cutting the SE by $3\times$ and removing most of the min-selection bias.

6. **Evaluate the checkpoint you actually care about on the full 15,864-molecule val set**, once, at the end. Cheap, and removes noise source 1 completely.

7. **Bootstrap a CI on your final number.** Resample the per-molecule losses $B=10{,}000$ times and report the 2.5/97.5 percentiles. If a paper says "1.524" without an interval, it is claiming a precision nobody measured.

### On the optimizer question, settled quantitatively

Your text correctly says AdamW has no knowledge of `best`. Chapter 5 does the mechanism; here's the timing argument, which is the decisive one:

- Steps per epoch: $\lceil 145{,}515/64\rceil = \mathbf{2{,}274}$.
- The "quiet" 13 epochs = $13\times2274 = \mathbf{29{,}562}$ parameter updates.
- AdamW's second-moment memory horizon is $\frac{1}{1-\beta_2} = 1{,}000$ steps $= \mathbf{0.44}$ epochs.

▸ The optimizer's entire memory of the loss landscape is **shorter than half an epoch**. It could not represent a 13-epoch dry spell even if it wanted to. And with a constant LR and no scheduler, there is nothing in the update rule that is a function of epoch index at all. The clustering is a property of *your measurement process*, not of the optimizer.

---

## 3.7 Monte Carlo standard error — the habit to build

Any time you report a number computed from sampling — FID, sample validity, docking scores, generation diversity, RL episode return — report its MCSE:

$$\mathrm{MCSE} = \frac{\hat\sigma}{\sqrt{n_{\text{eff}}}}$$

For correlated draws (e.g. MCMC, or successive training epochs), the **effective sample size** is
▸ $$n_{\text{eff}} = \frac{n}{1 + 2\sum_{k=1}^{\infty}\rho_k}$$
where $\rho_k$ is the lag-$k$ autocorrelation. Training curves have $\rho_1 \approx 0.7$ routinely, which alone gives $n_{\text{eff}} \approx n/6$.

**The rule:** a number without an error bar is not a measurement, it's an anecdote.

#### Monte Carlo standard error and $n_{\text{eff}}$, decoded

**MCSE** stands for **Monte Carlo standard error** — "Monte Carlo" being the general name for any method that answers a question by random sampling (see the *Did you know?* below for where that name comes from). It is the ordinary $\hat\sigma/\sqrt n$ of §1.3.1 wearing a different hat: **the part of your uncertainty that exists only because you sampled instead of enumerating.**

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $\hat\sigma$ | "sigma-hat" | The standard deviation of your $n$ individual draws |
| $n_{\text{eff}}$ | "n effective" | How many *independent* draws your $n$ correlated draws are worth |
| $\rho_k$ | "rho-k" | The correlation between a draw and the draw $k$ steps later |
| $\sum_{k=1}^{\infty}\rho_k$ | "sum over k of rho-k" | Total memory in the sequence, added up across all lags |

**Why correlation costs you sample size — with numbers.** Suppose you log validation loss every epoch and the lag-1 autocorrelation is $\rho_1 = 0.7$, with each further lag decaying by the same factor ($\rho_k = 0.7^k$, the standard AR(1) shape). Then

$$1 + 2\sum_{k\ge1} 0.7^k = 1 + 2\cdot\frac{0.7}{1-0.7} = 1 + 4.67 = 5.67$$

so $n_{\text{eff}} \approx n/5.67$. **Forty-three logged epochs are worth about 7.6 independent measurements.** Your error bar is $\sqrt{5.67} = 2.4$ times wider than the naive $\hat\sigma/\sqrt{43}$ would suggest.

> **Analogy.** You want to know the average temperature of a city, so you take 3,600 readings — one per second for an hour. You do not have 3,600 independent measurements of anything; the temperature barely moved. You have roughly *one* measurement, taken very carefully. **Correlated samples are repetitions, not replications**, and $n_{\text{eff}}$ is the arithmetic that converts one into the other.

▸ **The failure mode this prevents is quoting a confidently tiny error bar on a highly autocorrelated sequence.** Reinforcement-learning return curves, MCMC chains, and epoch-by-epoch training metrics are all strongly autocorrelated. Divide by $\sqrt n$ when you should have divided by $\sqrt{n_{\text{eff}}}$ and your interval is too narrow by a factor of $\sqrt{1+2\sum\rho_k}$ — commonly 2× to 3×, which is the difference between "significant" and "nothing here."

#### Examples and non-examples: what needs an MCSE

**✅ Numbers that require one**

| Example | The sampling step that creates the uncertainty |
|---|---|
| FID computed from 10,000 generated images | Which 10,000 you generated |
| "94.2% of sampled molecules are valid," from 2,000 samples | Which 2,000 you drew — SE is $\sqrt{0.942 \cdot 0.058/2000} = 0.5\%$ |
| Mean episode return over 100 RL rollouts | Which 100 episodes, and their heavy tails |
| Validation loss estimated from 16 of 1,000 diffusion timesteps | Which 16 timesteps — the dominant term in §3.6 |
| A win rate from 300 pairwise human preference judgements | Which 300 prompts, and which annotators |

**❌ Numbers that do not need one — because nothing was sampled**

| Example | Why there is no Monte Carlo error |
|---|---|
| Model size, 7.24 B parameters | Counted, not sampled |
| Wall-clock training time of a specific run | A measured fact about one event (it has *measurement* error, not sampling error) |
| Exact accuracy over the complete, enumerated test set of 10,000 MNIST images, evaluated deterministically | Every unit was used; the only randomness left is over *hypothetical other test sets*, which is a different question |
| The value of $\log 10 = 2.303$ | A mathematical constant |
| Peak GPU memory in bytes, read from the allocator | Instrumented, not estimated |

▸ **The boundary:** MCSE exists when **re-running your evaluation with a different random seed would change the number.** That is a five-second test — change the seed, run it twice, see how far apart the answers land. If they differ, you have just measured your MCSE empirically, and you now have no excuse for reporting the number without it.

> **Common misconception.** *"I used the whole validation set, so there is no sampling error."* There are usually *two* sampling steps and using all of one does not remove the other. Case Study A uses all 15,864 molecules but only 16 of 1,000 possible diffusion timesteps per molecule — and §3.6 shows the timestep draw contributes the larger share of the variance. **"I used everything" is only true if you enumerated every source of randomness**, and almost nobody does.

---

## Did you know?

- **Efron nearly gave the bootstrap a different name.** He has recounted considering several alternatives before settling — candidates reported to have been in the running include "Swiss Army knife," "meat axe," and "shotgun," each meant to convey a crude tool that works on almost anything. "Bootstrap" won, and the field ended up with a name whose original 19th-century meaning was **an impossibility**, not a virtue.

- **Student's $t$-distribution was published under a pseudonym because of beer.** William Sealy Gosset worked as a brewer at Guinness in Dublin, which forbade employees from publishing — reportedly after a colleague had leaked trade information. Gosset published his 1908 small-sample work in *Biometrika* as "Student," and the name stuck so firmly that the distribution underlying half of all statistical practice is named after an anonymity clause in a brewery's employment contract.

- **The Monte Carlo method is named after a casino, and was invented during a game of solitaire.** Stanislaw Ulam, recovering from illness at Los Alamos in 1946, wondered what fraction of solitaire deals are winnable, found the combinatorics hopeless, and realized he could just *deal many hands and count*. Nicholas Metropolis proposed the code name after the Monte Carlo casino in Monaco, where Ulam's uncle had gambled.

- **The Bonferroni correction was not proposed by Bonferroni.** Carlo Emilio Bonferroni proved the probability inequalities that bear his name in the 1930s; the multiple-comparison procedure built on them was developed and popularized by **Olive Jean Dunn** in 1961. The single most-used correction in applied statistics is named after the person who proved the lemma rather than the person who saw what it was for.

- **The $p < 0.05$ threshold is an arbitrary convenience, and its author said so.** Fisher suggested 0.05 in *Statistical Methods for Research Workers* (1925) as a handy round number, and later argued explicitly against treating any fixed level as a universal standard. A convention chosen for the convenience of printed lookup tables now decides which drugs get approved.

- **The American Statistical Association felt the need to publish a formal statement explaining what a $p$-value is.** Its 2016 statement on statistical significance was an unusual step for an organization founded in 1839 — it does not issue statements on statistical practice — and it was prompted by the sheer volume of misinterpretation in the published literature. Point one of the statement is that $p$-values do not measure the probability that a hypothesis is true.

- **The "winner's curse" was discovered by petroleum engineers, not statisticians.** Capen, Clapp, and Campbell, working at Atlantic Richfield, published it in 1971 in the *Journal of Petroleum Technology* after noticing that companies winning offshore oil-lease auctions were systematically overpaying. The mechanism they described — the winner is whoever's estimation error was most favourable — is exactly what selects your `best.pt`.

- **A million random draws contain only about fourteen record-breaking values.** The expected number of running records in $n$ draws is the harmonic number $H_n \approx \ln n + 0.577$. For $n = 10^6$ that is $14.4$. Records become vanishingly rare *purely because $n$ grows*, which is why a long stretch without a new best tells you almost nothing about whether anything has changed.

- **Kaggle's private leaderboard exists because of the arithmetic in §3.6.** Teams see their score on one slice of the test data and are ranked, at the end, on a slice they never saw. Without that split, the winner would be whichever team's overfitting to the public slice was luckiest — and the "leaderboard shakeup," where the public top-ten collapses on the private split, is selection bias made visible in public.

- **A large replication effort in psychology found roughly a third of results replicated.** The Open Science Collaboration's 2015 attempt to repeat 100 published studies found that about 36% of replications produced a statistically significant effect in the same direction, with effect sizes about half the size originally reported. **Halved effect sizes are exactly the signature of selection on noise** — the published literature is a maximum taken over many analyses.

- **The bootstrap was too expensive to use for a decade after it was invented.** A thousand refits of a model in 1979 was a serious computing job. The method's rise in the 1990s tracked the price of cycles rather than any statistical development — an early and clean instance of a pattern this book returns to repeatedly: **the binding constraint on a method is often hardware, not ideas.**

- **There is no unbiased estimator of the variance of $k$-fold cross-validation.** Bengio and Grandvalet proved this in 2004. Every $\pm$ printed beside a cross-validated score in every paper you have ever read is a heuristic, and the field has simply agreed to live with it.

---

## Check for Understanding

**Resampling is how you recover the error bar that your single dataset hid from you — and once you have that error bar, most of the patterns you thought you saw in your training curve turn out to be the arithmetic of taking minima over noise, while the ones that survive (clusters, slopes) are the ones actually telling you the model is learning.**

### Can you explain these out loud?

The real test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What is a sampling distribution, and why does it exist even though you only ever collect one dataset?** (If the words "if I ran the experiment again" don't appear, try again.)
2. **Why does the bootstrap need to sample *with* replacement?** What would happen if it sampled without?
3. **Why must a bootstrap resample be the same size as the original dataset?**
4. **What does $B = 10{,}000$ buy you that $B = 1{,}000$ doesn't — and what does neither of them buy?**
5. **Where does 63.2% come from, and why does $e$ show up in a question about drawing marbles?**
6. **What does a 95% confidence interval actually claim?** State it without using the word "probability" about the true value.
7. **Two models' error bars overlap. Why is that not a reason to say they're indistinguishable?**
8. **Explain a permutation test to someone who has never heard of a null distribution.** (The word "shuffle" should do most of the work.)
9. **Why is the loss written in your `best.pt` filename systematically too good?** How much too good, roughly, and how would you compute that number?
10. **Why is a 13-epoch stretch without a new best exactly what you should expect from a model that has stopped improving *and* from one that never was?**
11. **Why does correlation between successive measurements cost you sample size?** What is $n_{\text{eff}}$ in one sentence?
12. **What single logging change would let you run the right statistical test on your next model comparison?** (Answer: log per-example losses, not just their average.)
13. **What is the one-sentence rule that connects nested cross-validation, held-out test sets, out-of-bag validation, and the winner's curse?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 04 — Optimization I: Gradient Descent](04-optimization-i-gradient-descent.md)
