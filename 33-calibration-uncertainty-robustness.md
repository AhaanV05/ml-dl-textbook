# Chapter 33 — Calibration, Uncertainty & Robustness

> **Prerequisites:** Ch. 2, Ch. 3, Ch. 7.

---

## 33.1 Calibration

### The one-line idea

A model is calibrated if, among all the predictions it makes with 70% confidence, 70% turn out to be correct — which is a completely different property from being accurate.

### The analogy

A weather forecaster. One who says "70% chance of rain" and is right 70% of those times is *calibrated* and useful, even if they're often uncertain. One who says "100% rain" every day and is right 60% of the time is badly calibrated even though they might beat the first on some accuracy metrics. **You can act on a calibrated forecast; you cannot act on a confident wrong one.**

### The definition

▸ $$\mathbb{P}\big(\hat y = y \ \big|\ \hat p = p\big) = p\qquad\forall p\in[0,1]$$

**Accuracy and calibration are independent axes.** A model that outputs the base rate for every input is perfectly calibrated and useless.

### Measuring it

**Expected Calibration Error** — bin predictions by confidence, compare accuracy to confidence in each bin:
▸ $$\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\Big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\Big|$$

▸ **The problems with ECE, which you should be able to state:**
- **Binning-dependent.** More bins ⇒ higher measured ECE. It is not comparable across papers using different $M$.
- **Biased upward**, and the bias grows with $M$ and shrinks with $n$ — because within-bin sampling noise is counted as miscalibration. With $M=15$ bins and $n=1000$, a *perfectly calibrated* model shows nonzero ECE. (This is Chapter 3's point: an estimator with sampling noise has an expected absolute deviation greater than zero.)
- **Only measures top-label calibration** by default; the full predicted distribution can be badly wrong. Use classwise-ECE or adaptive binning (equal-mass bins).

**Proper scoring rules** avoid binning entirely and are strictly better as objectives:
- **Brier score:** $\frac1n\sum_i(p_i-y_i)^2$. Decomposes as reliability − resolution + uncertainty.
- **Negative log-likelihood:** $-\frac1n\sum_i\log p_i(y_i)$ — the training loss itself.

▸ A **proper** scoring rule is one whose expectation is uniquely minimized by reporting the true probability. **Cross-entropy and Brier are proper; accuracy is not** — which is precisely why optimizing accuracy produces miscalibrated models.

**Reliability diagram:** plot accuracy against confidence per bin. Above the diagonal = underconfident; below = overconfident.

### Why modern networks are miscalibrated

▸ Guo et al. (2017): LeNet (1998) was well calibrated; ResNet (2016) is badly **overconfident**. Causes:
- **Capacity.** With enough capacity, the model drives training NLL toward zero, which requires pushing probabilities to 1 — and it keeps doing so after accuracy has saturated.
- **Reduced regularization** (weight decay was reduced as BatchNorm arrived).
- **BatchNorm** itself worsens calibration, for reasons still not fully settled.
- **Cross-entropy's implicit bias** toward maximizing margin (Ch. 31 §31.3) drives logits apart without bound.

---

## 33.2 Fixing calibration

**Temperature scaling.** Learn a single scalar $T$ on a *held-out validation set* by minimizing NLL:
▸ $$p = \mathrm{softmax}(z/T)$$

▸ **It cannot change accuracy** — dividing all logits by a positive scalar preserves the argmax. It is a pure calibration fix, has one parameter, and is astonishingly effective (ECE often drops 10×). **This should be the default for any deployed classifier.**

**Platt scaling:** $p=\sigma(az+b)$ — two parameters, for binary problems.
**Vector/matrix scaling:** per-class parameters; more flexible, can change accuracy, needs more data.
**Isotonic regression:** a nonparametric monotone map. More flexible, needs more validation data, and can overfit.

**Training-time methods:** label smoothing (Ch. 7 §7.5), mixup, focal loss (which is implicitly a confidence penalty), and ensembles.

▸ **The critical caveat:** temperature scaling calibrates **in-distribution only.** Under distribution shift, the calibrated model becomes overconfident again — the temperature was fitted for a distribution that no longer holds. This is why §33.3–33.6 exist.

---

## 33.3 Aleatoric vs epistemic uncertainty

▸ **Aleatoric** — irreducible noise in the data. Two identical inputs with different labels. **More data does not help.** (This is Chapter 2's $\sigma^2$ and Chapter 15's $E$ term.)
▸ **Epistemic** — uncertainty about the model. **More data does help.** This is what a Bayesian posterior over parameters captures.

**Why the distinction is operationally important:** epistemic uncertainty tells you *where to collect more data* (active learning), and *when not to trust the model* (OOD detection). Aleatoric uncertainty tells you the *irreducible error floor* — and chasing it wastes resources.

**Decomposition** for an ensemble:
▸ $$\underbrace{H\big[\bar p(y\mid x)\big]}_{\text{total}} = \underbrace{\mathbb{E}_\theta\big[H[p(y\mid x,\theta)]\big]}_{\text{aleatoric}} + \underbrace{I(y;\theta\mid x)}_{\text{epistemic (mutual information)}}$$

**Epistemic uncertainty is the *disagreement* among plausible models.** A softmax probability from a single network conflates the two and cannot distinguish them.

---

## 33.4 Estimating uncertainty

| Method | Mechanism | Cost | Quality |
|---|---|---|---|
| **Deep ensembles** | train $M$ (5–10) models with different seeds; average | $M\times$ train and infer | ▸ **the strongest baseline, consistently** |
| **MC dropout** | keep dropout on at test time; $T$ forward passes | $T\times$ inference | cheap, weaker; underestimates uncertainty |
| **SWAG** | fit a Gaussian to the SGD iterate trajectory | ~1× train | good value for money |
| **Laplace approximation** | Gaussian posterior with covariance $H^{-1}$ at the MAP | post-hoc, cheap with K-FAC | good, principled, underrated |
| **Variational BNN** | learn a posterior over weights | 2× parameters, hard | rarely worth it in practice |
| **Evidential / Dirichlet** | predict distribution parameters directly | ~free | single pass, but sensitive to the regularizer |
| **Conformal** | §33.5 | ~free | **distribution-free guarantee** |

▸ **Why deep ensembles beat more principled Bayesian methods:** different random initializations land in genuinely *different* loss basins (Ch. 31 §31.5), so the members are functionally diverse. Variational methods and MC dropout explore only the neighbourhood of a single mode, so their disagreement understates the real epistemic uncertainty. **Multi-modality is what matters, and ensembles get it for free.**

**Cheap approximations to ensembles:** snapshot ensembles (cyclic LR, save at each minimum), BatchEnsemble (rank-1 per-member perturbations), and Monte Carlo over data augmentations at test time.

---

## 33.5 Conformal prediction

### The one-line idea

Instead of a point prediction, output a *set* of labels guaranteed to contain the truth with probability $1-\alpha$ — with a proof that requires no assumption about the model or the data distribution.

### The analogy

A weather service that, instead of guessing tomorrow's temperature, gives a range and *guarantees* the true value falls inside 90% of the time. It makes no claim about how good the underlying forecasting model is; if the model is bad the range is wide, and if the model is good the range is narrow. **The guarantee is about the procedure, not the model.**

### Split conformal, in full

1. Split the data into training and **calibration** sets.
2. Train any model on the training set.
3. Define a **nonconformity score** $s(x,y)$ — higher means worse agreement. E.g. $s = 1-\hat p(y\mid x)$.
4. Compute scores $s_1,\dots,s_n$ on the calibration set.
5. Let $\hat q$ be the $\lceil(n+1)(1-\alpha)\rceil/n$ empirical quantile of those scores.
6. **Predict the set** $C(x_{\text{new}}) = \{y : s(x_{\text{new}},y)\le\hat q\}$.

### The guarantee

▸ $$\mathbb{P}\big(y_{\text{new}}\in C(x_{\text{new}})\big)\ \ge\ 1-\alpha$$

**Requires only exchangeability** — that the calibration points and the test point are exchangeable (weaker than i.i.d.). **No assumption about the model, the architecture, or the data distribution.**

### The proof, in three lines

Under exchangeability, $s_{\text{new}}$ is exchangeable with $s_1,\dots,s_n$. So the rank of $s_{\text{new}}$ among the $n+1$ scores is uniform on $\{1,\dots,n+1\}$. Hence
$$\mathbb{P}\big(s_{\text{new}}\le s_{(\lceil(n+1)(1-\alpha)\rceil)}\big) = \frac{\lceil(n+1)(1-\alpha)\rceil}{n+1}\ \ge\ 1-\alpha$$
and $s_{\text{new}}\le\hat q$ is exactly the event $y_{\text{new}}\in C(x_{\text{new}})$. ∎

▸ **That is the entire proof.** It is a statement about ranks, which is why it needs nothing from the model. This makes conformal prediction one of the few genuinely rigorous guarantees available in modern ML, and it is increasingly the standard for high-stakes deployment.

### What you must understand about it

- ▸ **Coverage is *marginal*, not conditional.** You get 90% coverage averaged over the population, **not** 90% for every subgroup. A model can achieve 90% overall while covering 99% of easy cases and 40% of a minority subgroup. **Mondrian/group-conditional conformal** restores per-group coverage by calibrating within groups — do this whenever fairness matters.
- **Adaptive set sizes are the useful signal.** A good score function produces small sets on easy inputs and large sets on hard ones. **APS** (accumulate sorted probabilities) and **RAPS** (with a regularizer to prevent huge tails) are the standard choices for classification.
- **Regression:** conformalized quantile regression gives interval predictions with the same guarantee.
- **Distribution shift breaks it.** Exchangeability fails. Weighted conformal prediction handles covariate shift if you can estimate the likelihood ratio.

---

## 33.6 Out-of-distribution detection

**Scores, in rough order of strength:**

- **Maximum softmax probability.** The baseline. ▸ **Fails badly** because a network can be confidently wrong far from the data — ReLU networks are provably arbitrarily confident sufficiently far from the training data (Hein et al.).
- **Energy score:** $E(x) = -T\log\sum_j e^{z_j/T}$. Theoretically better aligned with $\log p(x)$ than the softmax, which normalizes away the very magnitude that carries the signal. Simple, strong, and free.
- **ODIN:** temperature scaling plus a small input perturbation in the direction that increases the max logit — in-distribution inputs respond more.
- **Mahalanobis distance** in feature space, per class, with a shared covariance. Strong, especially on far-OOD.
- **kNN distance** in a normalized feature space — remarkably strong and nonparametric.
- **Ensemble disagreement / mutual information** — principled, expensive.
- **Outlier exposure:** train with an auxiliary outlier dataset and a uniform-output loss. The single biggest improvement when auxiliary data is available.

▸ **The result everyone should know:** deep generative models assign **higher likelihood** to OOD data than to in-distribution data — a Glow or PixelCNN trained on CIFAR-10 gives higher likelihood to SVHN images (Nalisnick et al., 2019). The cause is that likelihood in high dimensions is dominated by low-level statistics (SVHN is simply *smoother*, so its pixels are easier to predict), not semantics. ▸ **Likelihood is not typicality**, and a high-likelihood point can be far outside the typical set. Fixes use likelihood *ratios* against a background model, or explicit typicality tests.

**Evaluation:** AUROC and FPR@95%TPR, on both near-OOD (CIFAR-10 vs CIFAR-100 — hard) and far-OOD (CIFAR vs noise — easy). Report both; far-OOD numbers alone are close to meaningless.

---

## 33.7 Distribution shift

| Type | Definition | Example |
|---|---|---|
| **Covariate shift** | $p(x)$ changes, $p(y\mid x)$ fixed | new camera, new hospital |
| **Label shift** | $p(y)$ changes, $p(x\mid y)$ fixed | disease prevalence rises |
| **Concept drift** | $p(y\mid x)$ changes | fraud tactics evolve |
| **Domain shift** | joint change | sim → real |
| **Subpopulation shift** | group proportions change | deployment demographics differ |

▸ **The distinction is not academic:** covariate shift is correctable by importance weighting $\frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}$; label shift by prior correction (BBSE/EM on the confusion matrix); **concept drift by nothing except new labels.** Diagnosing which one you have determines whether the problem is solvable without relabelling.

**Detection:** monitor input statistics (drift tests on features), monitor confidence and prediction distributions, and — the only reliable signal — collect labels on a sample of production data.

**Handling:** domain adaptation (DANN's adversarial domain-invariant features, CORAL's covariance matching), test-time adaptation (TENT — update only BatchNorm statistics and affine parameters by minimizing prediction entropy), and simple retraining, which is usually the right answer.

---

## 33.8 Adversarial robustness

### The phenomenon

▸ An imperceptible perturbation $\|\delta\|_\infty\le8/255$ flips a classifier from 99% "panda" to 99% "gibbon."

**Why it happens (Goodfellow's linear explanation):** for a linear model, $w^\top(x+\delta)=w^\top x + w^\top\delta$, and with $\delta=\epsilon\,\mathrm{sign}(w)$ the change is $\epsilon\|w\|_1$ — which grows with dimension. **In high dimensions, many tiny coordinated changes sum to a large logit change.** Networks are locally near-linear enough for this to apply.

▸ **The deeper account (Ilyas et al., 2019): adversarial examples arise from *non-robust features* that are genuinely predictive.** They demonstrated this by constructing a dataset labelled only via non-robust features, on which standard training yields good accuracy on the *clean* test set. **Adversarial vulnerability is not a bug in the model; it is a property of the data that the model faithfully learned.** This is the answer that distinguishes a deep understanding from a superficial one.

### Attacks

**FGSM:** $x' = x+\epsilon\,\mathrm{sign}(\nabla_x\mathcal{L})$ — one step.
**PGD** (the standard): iterated FGSM with random start and projection back into the $\epsilon$-ball:
▸ $$x^{t+1} = \Pi_{B_\epsilon(x)}\Big(x^t + \alpha\,\mathrm{sign}\big(\nabla_x\mathcal{L}(x^t,y)\big)\Big)$$
**C&W:** an optimization-based attack minimizing perturbation size subject to misclassification; the strongest white-box attack.
**AutoAttack:** an ensemble of parameter-free attacks — ▸ **the current standard for honest evaluation**, because it removes the tuning advantage that made many defences look better than they were.
**Black-box:** transfer attacks (adversarial examples transfer between models remarkably well), score-based, and decision-based.

### Defences

▸ **Adversarial training** — the only reliably effective empirical defence:
$$\min_\theta\ \mathbb{E}_{(x,y)}\Big[\max_{\|\delta\|\le\epsilon}\mathcal{L}(f_\theta(x+\delta),y)\Big]$$
Solve the inner max with PGD each step. **Costs 5–10× training time**, and reduces clean accuracy by 5–15% on CIFAR-10.

**TRADES** decomposes the objective into natural error plus a boundary term, giving an explicit, tunable accuracy/robustness knob.

▸ **Certified defences** give *provable* guarantees. **Randomized smoothing** is the practical one: define $g(x)=\arg\max_c\mathbb{P}_{\eta\sim\mathcal{N}(0,\sigma^2I)}\big[f(x+\eta)=c\big]$. Then $g$ is provably constant within an $\ell_2$ ball of radius
$$R = \frac{\sigma}{2}\big(\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B})\big)$$
Scales to ImageNet. Interval-bound propagation and convex relaxations give tighter certificates on smaller models.

▸ **The graveyard of broken defences.** Dozens of published defences were later broken, almost all by **gradient masking** — they made gradients uninformative (via non-differentiable ops, randomness, or numerical saturation) rather than making the model robust. Detect it by: black-box attacks beating white-box ones, unbounded attacks failing to reach 0% accuracy, or one-step attacks beating iterative ones. **Always evaluate with AutoAttack and adaptive attacks designed against your specific defence.**

▸ **The robustness–accuracy trade-off is real and provable** in some settings (Tsipras et al.): robust and standard classification can require genuinely different features, so you cannot have both for free.

---

## 33.9 Spurious correlations and group robustness

▸ **The problem:** the model latches onto a feature that is predictive in training but not causal — background instead of object (cows on grass), hospital ID instead of pathology, "no" appearing in the hypothesis instead of entailment. Simplicity bias (Ch. 31 §31.3) makes this the *default* outcome, because the spurious feature is usually the easier one to extract.

**Methods:**
- **GroupDRO:** minimize the *worst-group* loss rather than the average, with the group weights updated multiplicatively. ▸ Requires group labels, and works well when you have them. Critically, it needs **strong regularization** to work at all — otherwise the model interpolates every group and the worst-group loss goes to zero without fixing anything.
- **IRM:** seek a representation such that the optimal classifier is the same across environments. ▸ **Theoretically elegant, empirically disappointing** — the practical penalty term often fails to identify the invariant predictor, and IRM frequently underperforms plain ERM on realistic benchmarks. Worth knowing as a cautionary example of a beautiful idea that has not delivered.
- **JTT / LfF:** train once, identify the errors (which concentrate in the minority group), and upweight them. **No group labels needed** — this is the practical option.
- **Reweighting/subsampling** to balance groups: simple, and a surprisingly strong baseline.
- **Data augmentation and counterfactual augmentation** that break the correlation directly. Usually the most effective fix when feasible.

▸ **The finding that reframes the area (Kirichenko et al., 2023): the features are usually fine; the *last layer* is the problem.** Retraining only the final linear layer on a small group-balanced set recovers most of the worst-group accuracy. The representation had learned the core feature all along; the classifier had just learned to rely on the spurious one. **This makes the problem far cheaper to fix than it appears, and it's the most practically useful thing in this section.**

---

## Check for Understanding

**Accuracy and calibration are independent, and modern networks are overconfident because cross-entropy keeps pushing logits apart after accuracy saturates — fixable in-distribution by a single temperature parameter, but not under shift, where you need epistemic uncertainty (best estimated by ensembles, because they span different loss basins) or conformal prediction, whose coverage guarantee is a statement about the ranks of calibration scores and therefore holds for any model and any distribution, though only marginally and not per subgroup.**

---

**Next:** [Chapter 34 — The Interview Bank](34-interview-bank.md)
