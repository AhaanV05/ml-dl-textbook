# Chapter 3 — Resampling & the Statistics of Noisy Evaluation

> **Prerequisites:** Ch. 1 (§1.3), Ch. 2 (§2.3).
> **Why this chapter is placed here:** resampling is how you attach an error bar to any number you compute. It is also the diagnostic manual for reading a training curve, and §3.6 is the single most transferable section in Part I.

---

## 3.1 The problem resampling solves

### The one-line idea

You have one dataset but you need to know how much your answer would have wobbled if you'd gotten a *different* dataset. Resampling fakes the other datasets by reusing the one you have.

### The analogy

You want to know how reliable a restaurant is, but you can only eat there once. Resampling is the trick of asking each of your ten dinner companions separately what they thought, and treating the spread of their opinions as a stand-in for the spread you'd see across ten different visits. It's not perfect — they all ate on the same night — but it's far better than reporting one opinion with no error bar.

### The formal problem

You have an estimator $\hat\theta = s(X_1,\dots,X_n)$ (a test accuracy, a correlation, a model's loss). You want its sampling distribution — its standard error, a confidence interval, a $p$-value. Classical statistics gives you these in closed form for simple estimators (sample mean) and gives you nothing for complicated ones (median of a cross-validated AUC of a gradient-boosted model). Resampling gives you all of them, at the cost of compute.

---

## 3.2 The bootstrap

### Mechanism

▸ **Nonparametric bootstrap.** Given data $X = (x_1,\dots,x_n)$:
> 1. Draw $x^{*}_1,\dots,x^{*}_n$ **with replacement** from $X$. This is one bootstrap sample.
> 2. Compute $\hat\theta^{*(b)} = s(x^*_1,\dots,x^*_n)$.
> 3. Repeat for $b=1,\dots,B$ (typically $B$ = 1,000 for SEs, 10,000 for CIs).
> 4. The empirical distribution of $\{\hat\theta^{*(b)}\}$ approximates the sampling distribution of $\hat\theta$.

$$\widehat{\mathrm{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B\left(\hat\theta^{*(b)} - \bar{\hat\theta^*}\right)^2}$$

### Why it works

The bootstrap replaces the unknown true distribution $F$ with the empirical distribution $\hat F_n$ (mass $1/n$ on each observed point). The **plug-in principle**: whatever functional of $F$ you wanted, compute it on $\hat F_n$ instead. Since $\hat F_n \to F$ (Glivenko–Cantelli, uniformly), and the functional is smooth, the bootstrap distribution converges to the true sampling distribution.

**Sanity check with the mean.** If $s$ = sample mean, the bootstrap SE is
$$\widehat{\mathrm{SE}} = \sqrt{\frac{1}{n^2}\sum_i (x_i-\bar x)^2} = \frac{\hat\sigma}{\sqrt n}$$
which reproduces the textbook formula exactly. Good — a method that gets easy cases wrong isn't worth trusting on hard ones.

### The 0.632 fact

The probability a given point is *omitted* from a bootstrap sample:
$$\left(1-\frac1n\right)^n \xrightarrow{n\to\infty} e^{-1} = 0.368$$

▸ So **63.2% of unique points appear** in each bootstrap sample, and 36.8% are held out. Those held-out points form a natural test set ("out-of-bag"), which is:
- how random forests get free validation error without a holdout set (Ch. 14),
- the origin of the **.632 estimator**: $\widehat{\mathrm{Err}}_{.632} = 0.368\,\overline{\mathrm{err}}_{\text{train}} + 0.632\,\widehat{\mathrm{Err}}_{\text{OOB}}$, which corrects OOB's pessimism.

### Confidence intervals, in increasing order of correctness

1. **Normal:** $\hat\theta \pm z_{1-\alpha/2}\widehat{\mathrm{SE}}_{\text{boot}}$. Requires approximate normality.
2. **Percentile:** take the $\alpha/2$ and $1-\alpha/2$ quantiles of $\{\hat\theta^{*(b)}\}$. Transformation-respecting but biased if $\hat\theta$ is biased.
3. **Basic / pivotal:** $[2\hat\theta - q_{1-\alpha/2},\ 2\hat\theta - q_{\alpha/2}]$. Reflects the bootstrap distribution — correct when the *shape* is right but the location is shifted.
4. **BCa (bias-corrected and accelerated):** adjusts the percentile levels using
$$\alpha_1 = \Phi\!\left(\hat z_0 + \frac{\hat z_0 + z_{\alpha/2}}{1 - \hat a(\hat z_0 + z_{\alpha/2})}\right)$$
with $\hat z_0 = \Phi^{-1}\!\left(\frac{\#\{\hat\theta^{*(b)} < \hat\theta\}}{B}\right)$ (bias correction) and $\hat a$ from the jackknife skewness. **Second-order accurate** ($O(1/n)$ coverage error vs $O(1/\sqrt n)$ for percentile). Use this if you're publishing.

### When the bootstrap fails

- **Extremes.** Bootstrapping $\max(X_i)$ is inconsistent — the max of a resample is one of finitely many observed values, so the bootstrap distribution is atomic and never converges to the continuous true one. *Anything that depends on the maximum or minimum of your data is bootstrap-hostile.* **This includes "best validation loss."** Keep that in mind for §3.6.
- **Dependent data.** i.i.d. resampling destroys autocorrelation. Use **block bootstrap** (resample contiguous blocks of length $\ell \sim n^{1/3}$) for time series — or for training curves.
- **Very small $n$** ($<20$) or heavy tails without finite variance.
- **Parameters on a boundary** (e.g. variance component at 0).

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

▸ **Why LOO has high variance:** the $n$ training sets are nearly identical (they overlap in $n-2$ points), so the $n$ estimates are highly *correlated*. Averaging correlated things doesn't reduce variance:
$$\mathrm{Var}\!\left(\frac1n\sum \hat R^{(i)}\right) = \frac{\sigma^2}{n}\big(1 + (n-1)\rho\big) \xrightarrow{\rho\to1} \sigma^2$$
When $\rho \approx 1$, you have effectively **one** sample no matter how large $n$ is. This formula — the correlated-average variance — is one of the most useful in all of ML. It explains LOO's variance, it explains why ensembles of correlated models don't help (Ch. 14), and it explains why the 16 validation batches don't average away as much noise as you'd hope.

**$k=5$ or $10$ is the standard compromise.** Bengio & Grandvalet (2004) proved there is **no unbiased estimator of the variance of $k$-fold CV**, so the error bars people put on CV numbers are all heuristic.

### The variants you must not confuse

- **Stratified $k$-fold:** preserve class proportions per fold. Mandatory for imbalanced data.
- **Grouped $k$-fold:** all rows sharing a group ID (patient, molecule scaffold, user) go in the same fold. **For molecular data this is essential** — scaffold splitting rather than random splitting, or your model gets credit for memorizing a scaffold it saw in training.
- **Time-series CV:** expanding or rolling window, never shuffle. Random CV on time series leaks the future.
- **Nested CV:** an outer loop for performance estimation, an inner loop for hyperparameter selection. **If you tune hyperparameters on the same CV you report, your number is optimistically biased** — the amount of bias grows with the number of configurations tried (see §3.6, it's the same maximum-of-noise problem).

---

## 3.5 Permutation tests

**The question:** is my measured effect bigger than chance?

**Mechanism:** shuffle the labels $y$ many times, recompute the statistic each time, and see where your real statistic sits in that null distribution.

▸ $$p = \frac{1 + \#\{b : T^{(b)}_{\text{perm}} \ge T_{\text{obs}}\}}{1 + B}$$

The $+1$ in numerator and denominator is not a fudge — it makes the test **exact** (valid at level $\alpha$ for finite $B$) by including the observed permutation in the reference set.

**Assumption:** exchangeability under the null. Shuffle only what the null says is exchangeable. For grouped data, permute *within* groups.

**Where you should use it:** "is model A actually better than model B on my test set?" Compute per-example loss differences $d_i = \ell_A(i) - \ell_B(i)$, then permute the *signs* of $d_i$ (sign-flip test). This is the correct test and almost nobody in ML runs it.

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

### Noise source 1: molecule sampling

You measure 1,024 of 15,864. If the per-molecule CE has standard deviation $\sigma_{\text{mol}}$ (for molecular CE this is typically 0.3–0.8 nats; call it 0.5):

$$\mathrm{SE}_{\text{mol}} = \frac{\sigma_{\text{mol}}}{\sqrt{1024}} \cdot \underbrace{\sqrt{1 - \tfrac{1024}{15864}}}_{\text{finite-population correction} = 0.967} = \frac{0.5}{32}\times 0.967 = \mathbf{0.0151}$$

Already comparable to the 0.032 improvement being chased. But this is **not the dominant term.**

### Noise source 2: timestep sampling — this is the killer

You draw a **fresh random $t$ per batch**. Only 16 batches ⇒ only **16 draws of $t$**.

Apply the law of total variance (Ch. 1 §1.3.2):

▸ $$\mathrm{Var}(\widehat{\mathrm{CE}}) = \underbrace{\frac{\mathbb{E}_t[\mathrm{Var}(\text{CE}\mid t)]}{1024}}_{\text{molecule noise}} + \underbrace{\frac{\mathrm{Var}_t\big(\mathbb{E}[\text{CE}\mid t]\big)}{16}}_{\text{timestep noise}}$$

The second term divides by **16**, not 1024. And $\mathrm{Var}_t(\mathbb{E}[\text{CE}\mid t])$ is *large* — in a discrete diffusion model, CE at $t\approx0$ (nearly clean input) might be 0.2 nats, and at $t\approx T$ (fully corrupted) it approaches $\log K$. If the conditional mean CE ranges over roughly $[0.3, 2.4]$, a rough uniform-range estimate gives $\sigma_t \approx (2.4-0.3)/\sqrt{12} = 0.61$.

$$\mathrm{SE}_{t} = \frac{0.61}{\sqrt{16}} = \mathbf{0.152}$$

▸ **Total:** $\mathrm{SE} = \sqrt{0.0151^2 + 0.152^2} = 0.153$ nats.

**The measurement noise is roughly 5× larger than the improvement you're trying to detect.**

Even if my $\sigma_t$ estimate is off by 3×, you'd still have $\mathrm{SE} \approx 0.05 > 0.032$. The conclusion is robust to the assumption.

### The consequence: what a "best" record actually means

If per-epoch reads have SD $\approx 0.15$ around a slowly-drifting true value, then:

- A single epoch dipping to 1.524 tells you almost nothing about whether the true CE is 1.52 or 1.60.
- The recorded "best" is $\min$ over 43 noisy draws. **The minimum of noisy draws is a biased estimator of the true minimum.** With 43 draws from $\mathcal{N}(\mu, 0.15^2)$, the expected minimum is
$$\mathbb{E}[\min] \approx \mu - 0.15\cdot\Phi^{-1}\!\left(1-\tfrac{1}{44}\right) \approx \mu - 0.15\times 2.02 = \mu - 0.30$$
Two full standard deviations of **optimistic bias**. The saved `best.pt` checkpoint is, with high probability, *not* the best model — it's the model that got the luckiest validation draw.

This is exactly the bootstrap-fails-on-maxima problem from §3.2, and it's the same phenomenon as hyperparameter-selection bias in §3.4. **Selecting on noise is selecting noise.**

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

Under the null, that's a 1-in-1300 event. Under the alternative — a genuinely, continuously improving model whose true CE has drifted below the old floor — consecutive records are *expected*, because once the true mean drops well below 1.556, most epochs clear the bar.

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

---

## Check for Understanding

**Resampling is how you recover the error bar that your single dataset hid from you — and once you have that error bar, most of the patterns you thought you saw in your training curve turn out to be the arithmetic of taking minima over noise, while the ones that survive (clusters, slopes) are the ones actually telling you the model is learning.**

---

**Next:** [Chapter 04 — Optimization I: Gradient Descent](04-optimization-i-gradient-descent.md)
