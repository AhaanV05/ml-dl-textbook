# Chapter 33 — Calibration, Uncertainty & Robustness

> **Prerequisites:** Ch. 2, Ch. 3, Ch. 7.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

> **Why this chapter exists:** every other chapter asks *is the model right?* This one asks the four questions you need before anyone is allowed to act on its answer — **does it know when it's unsure** (calibration), **can it tell "the world is noisy" from "I haven't seen this before"** (uncertainty), **does it still work when the data moves** (shift), and **does it survive an adversary** (robustness). These are the chapters of a deployment review, not a leaderboard.

### Symbols introduced in this chapter

Skim once now; each is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`\hat p`$ | "p-hat" | The **confidence** the model states — a number in $`[0,1]`$ |
| $`\hat y`$ | "y-hat" | The label the model predicts, versus the truth $`y`$ |
| $`B_m`$ | "B-m" | The $`m`$-th **confidence bin** — all predictions whose confidence falls in one range |
| $`\lvert B_m\rvert`$ | "size of B-m" | **How many** predictions landed in that bin. The bars are a count, not absolute value |
| $`\mathrm{acc}(B_m)`$, $`\mathrm{conf}(B_m)`$ | "accuracy / confidence of bin m" | Fraction actually correct, and average stated confidence, inside that bin |
| $`z`$ | "z" | The **logits** — raw pre-softmax scores, any real number |
| $`T`$ | "T" | **Temperature.** Divide the logits by it before the softmax |
| $`H[p]`$ | "entropy of p" | How spread out a distribution is (Ch. 1 §1.4). Big = uncertain |
| $`I(y;\theta\mid x)`$ | "mutual information of y and theta given x" | How much the prediction and the *choice of model* share — the disagreement |
| $`\alpha`$ | "alpha" | **Miscoverage rate.** You demand coverage $`1-\alpha`$; $`\alpha=0.1`$ means 90% |
| $`s(x,y)`$ | "s of x, y" | **Nonconformity score** — bigger means "$`y`$ fits $`x`$ worse" |
| $`\hat q`$ | "q-hat" | The calibration **cutoff**: a quantile of the observed scores |
| $`C(x)`$ | "C of x" | The predicted **set** of labels, not a single label |
| $`\lceil\,\cdot\,\rceil`$ | "ceiling" | Round **up** to the next whole number |
| $`\delta`$ | "delta" | The adversarial **perturbation** added to an input |
| $`\|\delta\|_\infty \le \epsilon`$ | "infinity-norm at most epsilon" | **No single pixel** moves by more than $`\epsilon`$ |
| $`\mathrm{sign}(\cdot)`$ | "sign" | $`+1`$, $`0`$, or $`-1`$ — keeps direction, throws away magnitude |
| $`\Pi_{B_\epsilon(x)}`$ | "project onto the epsilon-ball" | Snap back inside the allowed perturbation region |
| $`\Phi^{-1}`$ | "phi-inverse" | Inverse Gaussian CDF: *"how many standard deviations out is this probability?"* |

**Full forms for the abbreviations in this chapter** (the complete book-wide list is in §0.13):

| Short | Full form |
|---|---|
| ECE | Expected Calibration Error |
| NLL | Negative Log-Likelihood |
| OOD / ID | Out-Of-Distribution / In-Distribution |
| MSP | Maximum Softmax Probability |
| ODIN | Out-of-DIstribution detector for Neural networks |
| APS / RAPS | Adaptive Prediction Sets / Regularized APS |
| AUROC | Area Under the Receiver Operating Characteristic curve |
| FPR@95%TPR | False Positive Rate when the True Positive Rate is 95% |
| FGSM | Fast Gradient Sign Method |
| PGD | Projected Gradient Descent |
| C&W | Carlini & Wagner (attack) |
| TRADES | TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization |
| DRO | Distributionally Robust Optimization |
| ERM / IRM | Empirical / Invariant Risk Minimization |
| JTT / LfF | Just Train Twice / Learning from Failure |
| SWAG | Stochastic Weight Averaging – Gaussian |
| BNN | Bayesian Neural Network |
| K-FAC | Kronecker-Factored Approximate Curvature |
| MC | Monte Carlo |
| BBSE | Black-Box Shift Estimation |
| DANN | Domain-Adversarial Neural Network |
| CORAL | CORrelation ALignment |
| TENT | Test-time ENTropy minimization |
| BN | Batch Normalization |
| SVHN | Street View House Numbers (dataset) |

---

## 33.1 Calibration

### The one-line idea

A model is calibrated if, among all the predictions it makes with 70% confidence, 70% turn out to be correct — which is a completely different property from being accurate.

### The analogy

A weather forecaster. One who says "70% chance of rain" and is right 70% of those times is *calibrated* and useful, even if they're often uncertain. One who says "100% rain" every day and is right 60% of the time is badly calibrated even though they might beat the first on some accuracy metrics. **You can act on a calibrated forecast; you cannot act on a confident wrong one.**

### The definition

▸ $$\mathbb{P}\big(\hat y = y \ \big|\ \hat p = p\big) = p\qquad\forall p\in[0,1]$$

**Accuracy and calibration are independent axes.** A model that outputs the base rate for every input is perfectly calibrated and useless.

#### Reading the calibration condition in plain English

$$\mathbb{P}\big(\hat y = y \ \big|\ \hat p = p\big) = p\qquad\forall p\in[0,1]$$

**Symbol by symbol.**

- $`\hat y`$ ("y-hat") is the label the model **guessed**; $`y`$ is the label that was actually right. So $`\hat y = y`$ is the event *"the model got it right."*
- The vertical bar $`\mid`$ means **"given"** (§0.9, Trap 5) — not divide, not absolute value. It restricts attention to a sub-population.
- $`\hat p`$ is the confidence the model **stated**.
- $`\forall`$ means **"for all"** (§0.11). The condition must hold at every confidence level, not just on average.

▸ **Read the whole thing aloud:** *"Among all the times the model said 'I'm $`p`$ confident,' it should be right exactly a $`p`$ fraction of the time — and that must hold for every value of $`p`$."*

**The procedure it describes.** Take every prediction the model made with confidence exactly $`0.70`$ — say there are $`400`$ of them. Count how many were correct. Calibrated means the count is about $`280`$. Do it again for $`0.80`$, for $`0.95`$, for $`0.30`$. ▸ **Calibration is not a property of any single prediction. It is a property of a *pile* of predictions**, which is why you can never say a particular forecast was miscalibrated, and why it takes a lot of data to measure.

**Accuracy and calibration, as a two-by-two.** These really are independent, and all four boxes are occupied by real systems:

| | **Calibrated** | **Miscalibrated** |
|---|---|---|
| **Accurate** | What you want. Says 95% and is right 95% of the time | The modern deep network. Right 95% of the time, says 99.5% |
| **Inaccurate** | The base-rate predictor. Says 51% every time, right 51% of the time — honest and worthless | Says 99%, right 60%. The worst quadrant, and common |

**The base-rate model, with numbers.** A disease affects 1% of patients. Build a model that ignores the patient entirely and outputs $`0.01`$ every time. Among all its $`0.01`$ predictions, exactly 1% are positive. ▸ **It is perfectly calibrated and contains zero information.** This is why calibration is a *necessary* property, never a sufficient one — you always report it alongside a discrimination measure like accuracy or AUROC.

#### Why calibration is not a nicety — a decision you can only make with real probabilities

Suppose a fraud model scores a transaction. A missed fraud costs 100 units of currency in chargebacks; a manual review costs 2 units in analyst time. Reviewing is worth it exactly when

$$\hat p \times 100 \;>\; 2 \qquad\Longleftrightarrow\qquad \hat p > 0.02$$

▸ **That threshold, $`0.02`$, is derived entirely from the two costs — and it is only meaningful if $`\hat p`$ means what it says.** If the model's stated $`0.03`$ is really $`0.15`$, you are reviewing transactions you should be blocking outright, and the whole cost calculation is fiction. Every threshold anyone ever sets on a model output is secretly this computation, whether or not they write it down.

> **Analogy.** A fuel gauge. An *accurate* car is one that goes where you steer it. A *calibrated* fuel gauge is one where "quarter tank" means a quarter of a tank. You can drive a car with a broken gauge — right up until you plan a journey. **Accuracy tells you the model works; calibration tells you whether you can plan around it.** Nobody notices a bad gauge in city traffic and everybody notices it in the desert, which is the same reason miscalibration surfaces under distribution shift.

> **Where this came from.** Calibration is a **weather** idea, and the analogy in the text is not decorative — it is the history. **Glenn W. Brier**, working at the US Weather Bureau, published *Verification of Forecasts Expressed in Terms of Probability* in 1950, introducing the score that still bears his name. He was solving an institutional problem: forecasters had begun stating probabilities rather than yes/no predictions, and there was no agreed way to tell a good probabilistic forecaster from a lucky one — nor any way to stop a forecaster gaming a yes/no score by hedging. The Weather Bureau began issuing public probability-of-precipitation forecasts in the mid-1960s, and **Allan Murphy** and **Robert Winkler** built the modern verification framework, including reliability diagrams, through the 1970s. ▸ **The result that should have travelled faster: professional weather forecasters are among the best-calibrated forecasters ever measured** — decades of daily feedback on scored probabilistic predictions does what nothing else does. By contrast, Philip Tetlock's long study of expert political forecasting, published as *Expert Political Judgment* in 2005, found expert confidence largely unrelated to expert accuracy. **The difference is not intelligence; it is whether anyone kept score.**

### Measuring it

**Expected Calibration Error** — bin predictions by confidence, compare accuracy to confidence in each bin:
▸ $$\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\Big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\Big|$$

▸ **The problems with ECE, which you should be able to state:**
- **Binning-dependent.** More bins ⇒ higher measured ECE. It is not comparable across papers using different $`M`$.
- **Biased upward**, and the bias grows with $`M`$ and shrinks with $`n`$ — because within-bin sampling noise is counted as miscalibration. With $`M=15`$ bins and $`n=1000`$, a *perfectly calibrated* model shows nonzero ECE. (This is Chapter 3's point: an estimator with sampling noise has an expected absolute deviation greater than zero.)
- **Only measures top-label calibration** by default; the full predicted distribution can be badly wrong. Use classwise-ECE or adaptive binning (equal-mass bins).

**Proper scoring rules** avoid binning entirely and are strictly better as objectives:
- **Brier score:** $`\frac1n\sum_i(p_i-y_i)^2`$. Decomposes as reliability − resolution + uncertainty.
- **Negative log-likelihood:** $`-\frac1n\sum_i\log p_i(y_i)`$ — the training loss itself.

▸ A **proper** scoring rule is one whose expectation is uniquely minimized by reporting the true probability. **Cross-entropy and Brier are proper; accuracy is not** — which is precisely why optimizing accuracy produces miscalibrated models.

**Reliability diagram:** plot accuracy against confidence per bin. Above the diagonal = underconfident; below = overconfident.

#### Computing ECE by hand, once

$$\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\Big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\Big|$$

**Every symbol.** $`M`$ is the number of bins; $`B_m`$ is the set of predictions in bin $`m`$; $`\lvert B_m\rvert`$ is **how many** predictions that is (bars as a count, not absolute value — the *outer* bars, around the difference, are the ordinary absolute value); $`n`$ is the total. $`\mathrm{acc}(B_m)`$ is the fraction actually correct in that bin, $`\mathrm{conf}(B_m)`$ the average confidence stated in it.

▸ **In English:** *"chop the confidence range into buckets; in each bucket compute how far the truth is from the claim; average those gaps, weighting each bucket by how many predictions fell into it."*

**Do it.** $`n = 1000`$ predictions, $`M = 3`$ bins:

| Bin | Range | $`\lvert B_m\rvert`$ | $`\mathrm{conf}`$ | $`\mathrm{acc}`$ | Gap | Weight | Contribution |
|---|---|---|---|---|---|---|---|
| 1 | $`[0,\,0.6)`$ | $`200`$ | $`0.55`$ | $`0.50`$ | $`0.05`$ | $`0.2`$ | $`0.010`$ |
| 2 | $`[0.6,\,0.9)`$ | $`300`$ | $`0.75`$ | $`0.70`$ | $`0.05`$ | $`0.3`$ | $`0.015`$ |
| 3 | $`[0.9,\,1.0]`$ | $`500`$ | $`0.96`$ | $`0.88`$ | $`0.08`$ | $`0.5`$ | $`0.040`$ |

$$\mathrm{ECE} = 0.010+0.015+0.040 = 0.065$$

Report it as **6.5%**: *"on average, this model's stated confidence is off by six and a half percentage points."*

#### Why ECE's three problems are worse than they sound

**Problem 1 — binning-dependence, demonstrated.** Split bin 3 in two and nothing about the model changes, but the number does. Suppose the $`500`$ predictions in bin 3 were really $`200`$ at confidence $`0.925`$/accuracy $`0.95`$ (**under**confident) and $`300`$ at confidence $`0.983`$/accuracy $`0.833`$ (badly **over**confident) — which averages to exactly the $`0.96`$/$`0.88`$ above. Recompute:

$$0.2\times\lvert 0.95-0.925\rvert + 0.3\times\lvert 0.833-0.983\rvert = 0.005 + 0.045 = 0.050$$

The chapter's ECE rises from $`0.065`$ to $`0.075`$. ▸ **The coarse bin let a $`+0.025`$ error cancel a $`-0.15`$ error.** Finer bins stop the cancellation, so **more bins can only reveal more miscalibration, never less** — which means a reported ECE without a stated $`M`$ is not a measurement, and comparing two papers with different $`M`$ is meaningless.

**Problem 2 — upward bias, with the actual number.** Take a **perfectly calibrated** model, $`M = 15`$ bins, $`n = 1000`$, so roughly $`67`$ predictions per bin. In a bin whose true accuracy is $`0.9`$, the observed accuracy is a coin-flip average with standard deviation

$$\sqrt{\frac{0.9\times 0.1}{67}} = \sqrt{0.00134} = 0.037$$

and the expected *absolute* deviation of a roughly-normal quantity is $`\sigma\sqrt{2/\pi}\approx 0.8\sigma \approx 0.029`$. ▸ **A perfect model measures $`\mathrm{ECE}\approx 0.029`$ — about 3% — purely from sampling noise.** Since many papers report ECEs in the 2–5% range, a substantial fraction of published "miscalibration" may be the estimator's own noise. This is exactly Chapter 3's point: an absolute deviation cannot average to zero, because the absolute value turns cancelling errors into accumulating ones.

Check the direction: with $`M = 5`$ bins ($`200`$ per bin) the same computation gives $`\sqrt{0.09/200}\times 0.8 = 0.017`$. **Fewer bins, smaller measured error, same model.**

**Problem 3 — top-label only.** By default ECE looks at the *most likely* class and ignores everything else. A model that puts $`0.90`$ on the right answer and splits the remaining $`0.10`$ across wildly implausible classes scores identically to one that puts it on the plausible runner-up. **Classwise-ECE** evaluates every class's probabilities, and **adaptive binning** uses equal-*mass* bins (same number of points each) rather than equal-*width* ones, which fixes the pathology where the top bin holds 90% of the data.

#### Why "proper" is the word that matters

A scoring rule takes a stated probability $`q`$ and an outcome, and returns a penalty. It is **proper** if, when the truth is $`p`$, the *expected* penalty is minimized by stating $`q = p`$ — honesty is optimal. It is **strictly** proper if $`q = p`$ is the *only* minimizer.

**Brier is proper, in two lines.** With truth $`p`$ and statement $`q`$, the expected Brier penalty is

$$\mathbb{E}[(q-y)^2] = p(q-1)^2 + (1-p)q^2 = q^2 - 2pq + p$$

Differentiate with respect to $`q`$: $`2q - 2p = 0 \Rightarrow q = p`$. ▸ **The unique best answer is the truth, and there is no way to game it.**

**Accuracy is not proper, and the reason is one sentence.** Accuracy only sees the $`\arg\max`$ (§0.3). If the true probability of class 1 is $`0.51`$, accuracy is maximized by predicting class 1 — and it is *equally* maximized whether you claim $`0.51`$ or $`0.9999`$. ▸ **Accuracy is completely blind to the difference between "barely sure" and "certain," so nothing in accuracy-driven training ever penalizes overclaiming.** That is not a side-effect of modern architectures; it is what the metric is.

**Put numbers on Brier.** Forecast $`0.7`$ for rain. If it rains: $`(0.7-1)^2 = 0.09`$. If not: $`(0.7-0)^2 = 0.49`$. Over many days where it truly rains 70% of the time:

| Your forecast | Expected Brier |
|---|---|
| $`0.7`$ (honest) | $`0.7(0.09)+0.3(0.49) = \mathbf{0.21}`$ |
| $`1.0`$ (overconfident) | $`0.7(0)+0.3(1.0) = 0.30`$ |
| $`0.5`$ (hedging) | $`0.7(0.25)+0.3(0.25) = 0.25`$ |

Honesty wins, and the winning value $`0.21`$ is exactly $`p(1-p)`$ — the **irreducible** part, the aleatoric noise of §33.3 showing up early. **Reliability − resolution + uncertainty** is that same split written out: how far off your probabilities are, minus how much they usefully vary across cases, plus the world's own randomness.

> **Where this came from.** The idea that a scoring rule should reward honesty is due to **Leonard J. Savage**, whose 1971 paper *Elicitation of Personal Probabilities and Expectations* set out the general theory of proper scoring rules, building on the subjective-probability foundations laid by **Bruno de Finetti** and **Frank Ramsey** in the 1920s and 1930s. The motivating problem was economic rather than statistical: how do you pay someone for a forecast in a way that makes lying unprofitable? ▸ **Cross-entropy — the loss you train every classifier with — is a strictly proper scoring rule, which means your training objective was already the right calibration objective.** The miscalibration in §33.1 does not come from the loss being wrong; it comes from optimizing it past the point where accuracy stops improving.

### Why modern networks are miscalibrated

▸ Guo et al. (2017): LeNet (1998) was well calibrated; ResNet (2016) is badly **overconfident**. Causes:
- **Capacity.** With enough capacity, the model drives training NLL toward zero, which requires pushing probabilities to 1 — and it keeps doing so after accuracy has saturated.
- **Reduced regularization** (weight decay was reduced as BatchNorm arrived).
- **BatchNorm** itself worsens calibration, for reasons still not fully settled.
- **Cross-entropy's implicit bias** toward maximizing margin (Ch. 31 §31.3) drives logits apart without bound.

#### The mechanism, with arithmetic

Here is exactly why "drive NLL toward zero" and "stay calibrated" are in conflict. Take a binary problem and let $`g`$ be the **logit gap** — how far ahead the correct class's score is. Then the stated confidence is $`\sigma(g) = 1/(1+e^{-g})`$ and the loss on that example is $`-\log\sigma(g) = \log(1+e^{-g})`$.

| Logit gap $`g`$ | Stated confidence | NLL on this example | Is it classified correctly? |
|---|---|---|---|
| $`2`$ | $`0.881`$ | $`0.127`$ | yes |
| $`4`$ | $`0.982`$ | $`0.0181`$ | yes |
| $`8`$ | $`0.99966`$ | $`0.000335`$ | yes |
| $`16`$ | $`0.9999999`$ | $`0.00000011`$ | yes |

▸ **Read the last column.** From $`g=2`$ onward, **accuracy never changes** — the example was already correct and stays correct. But the loss keeps falling, by a factor of about $`e`$ for every unit of gap, forever. So gradient descent keeps pushing, because there is always more loss to remove, and the only way to remove it is to state more extreme confidence.

▸ **A network with enough capacity to fit the training set therefore has an unbounded incentive to become more confident about things it already gets right.** That confidence is fitted on the *training* set. On held-out data the model might be correct 95% of the time while still saying $`0.9997`$, which is an ECE of roughly $`0.05`$ — and none of it came from a bug.

> **Analogy.** A student graded only on their final answers, who is then told they can earn extra credit for *underlining* the answer more emphatically. Nothing about their actual knowledge improves. Their underlining becomes unhinged. **Cross-entropy after the accuracy plateau is extra credit for emphasis.**

**Why regularization is on the list.** Weight decay caps the size the weights can reach, which caps how large logits can get, which caps confidence. As BatchNorm arrived and weight decay was dialled down, that cap loosened. Label smoothing (Ch. 7 §7.5) attacks it directly by making the *target* $`0.95`$ instead of $`1.0`$, so the gradient reverses once the model gets too confident and the incentive to inflate $`g`$ disappears. Focal loss does the same thing by another route — it down-weights examples the model already gets right, so confident-correct examples stop contributing gradient at all.

> **Where this came from.** The observation is from **Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Weinberger's** *On Calibration of Modern Neural Networks* (2017), and its rhetorical force came from the comparison: **LeNet-5** (LeCun, 1998) was *already* well calibrated, and **ResNet-110** (2016) — far more accurate on the same task — was badly overconfident. ▸ **The field had spent eighteen years making models more accurate and, without noticing, less honest.** Nobody had been measuring the second thing, because accuracy was the leaderboard column. The paper's other contribution was to test a battery of sophisticated calibration methods and find that the simplest — one scalar — beat them all.

---

## 33.2 Fixing calibration

**Temperature scaling.** Learn a single scalar $`T`$ on a *held-out validation set* by minimizing NLL:
▸ $$p = \mathrm{softmax}(z/T)$$

▸ **It cannot change accuracy** — dividing all logits by a positive scalar preserves the argmax. It is a pure calibration fix, has one parameter, and is astonishingly effective (ECE often drops 10×). **This should be the default for any deployed classifier.**

#### Temperature scaling, worked

**What a logit is.** The raw score a network emits for each class before the softmax — any real number, positive or negative, with no probabilistic meaning of its own. The softmax (§1.3.4) turns a vector of them into probabilities by exponentiating and normalizing.

**Run one example.** Suppose the logits are $`z = (4.0,\ 1.0,\ 0.5)`$.

| $`T`$ | $`z/T`$ | After $`\exp`$ | Softmax | Top confidence |
|---|---|---|---|---|
| $`0.5`$ | $`(8,\ 2,\ 1)`$ | $`(2981,\ 7.39,\ 2.72)`$ | $`(0.9966,\ 0.0025,\ 0.0009)`$ | $`99.7\%`$ |
| $`1.0`$ | $`(4,\ 1,\ 0.5)`$ | $`(54.60,\ 2.72,\ 1.65)`$ | $`(0.926,\ 0.046,\ 0.028)`$ | $`92.6\%`$ |
| $`2.0`$ | $`(2,\ 0.5,\ 0.25)`$ | $`(7.39,\ 1.65,\ 1.28)`$ | $`(0.716,\ 0.160,\ 0.124)`$ | $`71.6\%`$ |

▸ **The winner is class 1 in every row.** Dividing by a positive number cannot reorder anything, so the argmax — and therefore the accuracy, and therefore every accuracy-derived metric — is untouched. What changes is *only* how emphatically the model states its answer: $`T > 1`$ softens, $`T < 1`$ sharpens.

**The two limits.** As $`T \to \infty`$ every logit goes to $`0`$ and the output becomes uniform: total ignorance. As $`T \to 0`$ the largest logit dominates completely and the output becomes one-hot: total certainty. ▸ **$`T`$ is a single dial running from "I have no idea" to "I am certain," and calibration is finding where on that dial the model's honesty lies.**

> **Analogy.** A contrast knob on a monitor. It cannot change which pixel is brightest, so it cannot change what the picture *is* — it only changes how starkly the bright parts stand out from the dark. A network trained to near-zero loss has the contrast cranked to maximum. Temperature scaling turns it back down until the image matches reality.

**How you fit it, and the one mistake to avoid.** Choose $`T`$ to minimize NLL on a **held-out validation set**. It is a one-dimensional search that takes seconds. ▸ **Fitting $`T`$ on the training set gives you $`T \approx 1`$ and no benefit**, because on the training set the model's confidence is roughly right — it memorized those labels. **The overconfidence only exists on data the model has not seen, so the fix can only be measured on data the model has not seen.**

**Why one parameter is enough.** It is an empirical fact, not a theorem: modern networks are miscalibrated in a remarkably *uniform* way — every prediction is over-sharpened by roughly the same factor, rather than some classes being inflated and others deflated. When that holds, one scalar undoes it. When it does not — heavily imbalanced classes, for instance — vector or matrix scaling, with per-class parameters, does better at the cost of needing far more validation data and gaining the ability to change accuracy (and therefore to overfit).

**Where it sits in the family.** Temperature scaling is **Platt scaling with the offset removed and the slope shared**: $`p = \sigma(az+b)`$ becomes $`p=\sigma(z/T)`$ when $`b = 0`$ and $`a = 1/T`$. Fewer parameters, less flexibility, far less to overfit — which is exactly why it wins on typical validation-set sizes.

**Platt scaling:** $`p=\sigma(az+b)`$ — two parameters, for binary problems.
**Vector/matrix scaling:** per-class parameters; more flexible, can change accuracy, needs more data.
**Isotonic regression:** a nonparametric monotone map. More flexible, needs more validation data, and can overfit.

**Training-time methods:** label smoothing (Ch. 7 §7.5), mixup, focal loss (which is implicitly a confidence penalty), and ensembles.

▸ **The critical caveat:** temperature scaling calibrates **in-distribution only.** Under distribution shift, the calibrated model becomes overconfident again — the temperature was fitted for a distribution that no longer holds. This is why §33.3–33.6 exist.

#### Why the caveat is fatal rather than annoying

$`T`$ is a number fitted on a **specific sample from a specific distribution**. It encodes "on data like *this*, the model overstates by about this much." Change the data and the sentence is no longer about anything.

Worse, the failure is in the dangerous direction. Under shift, accuracy drops but stated confidence barely moves — the model's logits do not know they are looking at a new hospital's scanner. ▸ **So the gap between confidence and accuracy widens exactly when you most need it to be trustworthy, and a temperature fitted in the old world makes the numbers *look* fine.** A calibrated-in-distribution model under shift is a fuel gauge that was calibrated in the garage and is now in the desert.

**And the deeper structural point.** A softmax probability is a single number. It cannot distinguish "this input is  ambiguous" from "I have never seen anything like this input." ▸ **One number cannot carry two independent quantities**, and that is not a training failure that a better $`T`$ could fix — it is a shortage of channel capacity. §33.3 gives the two quantities names; §33.4 gives you machinery that reports both; §33.5 sidesteps the problem entirely by giving up on point predictions.

> **Where this came from.** The one-parameter fix has an older sibling. **John Platt**, at Microsoft Research in 1999, faced the opposite version of the same problem: support vector machines output an uncalibrated margin — a distance, not a probability — and practitioners needed probabilities to make decisions. His solution was to fit a sigmoid, $`p = \sigma(az+b)`$, to the margin on held-out data. **Bianca Zadrozny and Charles Elkan** generalized it in 2002 to **isotonic regression**, a nonparametric monotone map fitted by the pool-adjacent-violators algorithm — a procedure from 1950s order-restricted statistics. Guo and colleagues' 2017 contribution was partly negative and all the more useful for it: they tested the whole family and found that on modern networks the *most* constrained member, one scalar with no offset, beat everything more flexible. ▸ **Two decades of increasingly sophisticated calibration methods, and the winner was the simplest thing that could possibly work.**

---

## 33.3 Aleatoric vs epistemic uncertainty

▸ **Aleatoric** — irreducible noise in the data. Two identical inputs with different labels. **More data does not help.** (This is Chapter 2's $`\sigma^2`$ and Chapter 15's $`E`$ term.)
▸ **Epistemic** — uncertainty about the model. **More data does help.** This is what a Bayesian posterior over parameters captures.

**Why the distinction is operationally important:** epistemic uncertainty tells you *where to collect more data* (active learning), and *when not to trust the model* (OOD detection). Aleatoric uncertainty tells you the *irreducible error floor* — and chasing it wastes resources.

**Decomposition** for an ensemble:
▸ $$\underbrace{H\big[\bar p(y\mid x)\big]}_{\text{total}} = \underbrace{\mathbb{E}_\theta\big[H[p(y\mid x,\theta)]\big]}_{\text{aleatoric}} + \underbrace{I(y;\theta\mid x)}_{\text{epistemic (mutual information)}}$$

**Epistemic uncertainty is the *disagreement* among plausible models.** A softmax probability from a single network conflates the two and cannot distinguish them.

#### The two words, and the two questions they answer

The vocabulary is off-putting and the ideas are not.

- **Aleatoric** — from the Latin *alea*, **a die** (as in *alea iacta est*, "the die is cast"). Dice-uncertainty. The world itself is random and no amount of study changes that.
- **Epistemic** — from the Greek *epistēmē*, **knowledge** (the root of *epistemology*). Ignorance-uncertainty. You could in principle fix this by learning more.

▸ **The operational test that separates them in one question: "if I collected a million more labelled examples of exactly this kind, would this uncertainty shrink?"** Yes → epistemic. No → aleatoric.

| | Aleatoric | Epistemic |
|---|---|---|
| Source | noise in the world | ignorance in the model |
| Shrinks with more data? | **no** | **yes** |
| Example | a fair coin; two radiologists disagreeing on a  borderline scan | a scan from a scanner model you have never trained on |
| Right response | accept it, or get better *features* (not more rows) | **collect labels here** (active learning), or **refuse to answer** (OOD detection) |
| Where else in this book | Ch. 2's $`\sigma^2`$, Ch. 15's $`E`$ term | Ch. 31 §31.5's distinct loss basins |

#### Decoding the ensemble decomposition, with numbers

$$\underbrace{H\big[\bar p(y\mid x)\big]}_{\text{total}} = \underbrace{\mathbb{E}_\theta\big[H[p(y\mid x,\theta)]\big]}_{\text{aleatoric}} + \underbrace{I(y;\theta\mid x)}_{\text{epistemic}}$$

**Every symbol.**

- $`\theta`$ indexes **which member** of the ensemble you are talking about (§0.4: parameters).
- $`p(y\mid x,\theta)`$ is **one member's** prediction. $`\bar p(y\mid x)`$ is the **average** over members — the ensemble's answer.
- $`H[\cdot]`$ is entropy (§1.4.1): how spread out a distribution is. $`0`$ = certain, $`\log K`$ = maximally unsure over $`K`$ classes.
- $`\mathbb{E}_\theta[\cdot]`$ averages over members. So $`\mathbb{E}_\theta[H[\cdot]]`$ is **"average how confused each individual member is."**
- $`I(y;\theta\mid x)`$ is mutual information (§1.4.3) between the answer and the choice of model — **"how much does knowing which member you asked tell you about the answer?"**

▸ **In English: total confusion = the confusion everyone shares + the confusion caused by them disagreeing.** Rearranged, epistemic uncertainty is *total minus shared*, which is exactly **disagreement**.

**Two cases with identical totals and opposite meanings.** Binary problem, two ensemble members, entropies in nats ($`\ln`$).

**Case A — the coin.** Both members say $`0.5`$.

- $`\bar p = 0.5`$, so total $`= H[0.5] = -2(0.5\ln 0.5) = 0.693`$ nats.
- Each member's own entropy is also $`0.693`$, so aleatoric $`= 0.693`$.
- Epistemic $`= 0.693 - 0.693 = \mathbf{0}`$.

**Case B — the unknown input.** One member says $`0.02`$, the other says $`0.98`$.

- $`\bar p = 0.5`$, so total $`= 0.693`$ nats — **identical to Case A.**
- $`H[0.02] = -(0.02\ln 0.02 + 0.98\ln 0.98) = 0.078 + 0.020 = 0.098`$ nats, and the same for $`0.98`$. So aleatoric $`= 0.098`$.
- Epistemic $`= 0.693 - 0.098 = \mathbf{0.595}`$ — **86% of the total.**

▸ **Same total uncertainty, same averaged prediction of $`0.5`$, and completely different situations.** In Case A every model agrees the world is a coin flip and no data will help. In Case B both models are *individually confident* and they flatly contradict each other, which is the signature of an input outside what the training data pinned down.

▸ **A single network outputs $`0.5`$ in both cases and there is no way to tell them apart.** That is the shortage of channel capacity from §33.2 made precise: one number, two questions.

> **Analogy.** Ask five doctors for a diagnosis. If all five say "it's a coin flip between A and B," you have a  ambiguous case, and a sixth opinion will not help — you need a different *test*, not another opinion. If two say "definitely A" and three say "definitely B," you have a case outside the reliable range of medical consensus, and that is precisely when you seek more information. **Averaging the five opinions gives you the same number in both situations, and throws away the only thing that mattered.**

> **Where this came from.** The aleatory/epistemic split is much older than machine learning. It was standard vocabulary in **engineering risk analysis** — nuclear-plant safety assessment, structural reliability, seismic hazard modelling — where the distinction has enormous practical consequences: aleatory hazard (will there be an earthquake) is priced in, while epistemic uncertainty (is our fault model right) is what expert review and further study are supposed to reduce. **Alex Kendall and Yarin Gal's** 2017 paper *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* brought the terminology into deep learning and showed how to model both in a single network. ▸ **The pattern is one this book keeps repeating: the concept a field needs has usually been sitting in a neighbouring discipline for thirty years under a name nobody thought to look up.**

---

## 33.4 Estimating uncertainty

| Method | Mechanism | Cost | Quality |
|---|---|---|---|
| **Deep ensembles** | train $`M`$ (5–10) models with different seeds; average | $`M\times`$ train and infer | ▸ **the strongest baseline, consistently** |
| **MC dropout** | keep dropout on at test time; $`T`$ forward passes | $`T\times`$ inference | cheap, weaker; underestimates uncertainty |
| **SWAG** | fit a Gaussian to the SGD iterate trajectory | ~1× train | good value for money |
| **Laplace approximation** | Gaussian posterior with covariance $`H^{-1}`$ at the MAP | post-hoc, cheap with K-FAC | good, principled, underrated |
| **Variational BNN** | learn a posterior over weights | 2× parameters, hard | rarely worth it in practice |
| **Evidential / Dirichlet** | predict distribution parameters directly | ~free | single pass, but sensitive to the regularizer |
| **Conformal** | §33.5 | ~free | **distribution-free guarantee** |

▸ **Why deep ensembles beat more principled Bayesian methods:** different random initializations land in  *different* loss basins (Ch. 31 §31.5), so the members are functionally diverse. Variational methods and MC dropout explore only the neighbourhood of a single mode, so their disagreement understates the real epistemic uncertainty. **Multi-modality is what matters, and ensembles get it for free.**

**Cheap approximations to ensembles:** snapshot ensembles (cyclic LR, save at each minimum), BatchEnsemble (rank-1 per-member perturbations), and Monte Carlo over data augmentations at test time.

#### The table, one row at a time

Each row is a different answer to the same question: *"I want to know how much the answer would change if I had trained a slightly different model — how do I get more than one model without training a thousand?"*

- **Deep ensembles.** Train $`M`$ networks from scratch with different random seeds. Different seed → different initialization → different data ordering → a  different solution. Average their predictions; use their **disagreement** as epistemic uncertainty. With $`M = 5`$ that is $`5\times`$ the training cost and $`5\times`$ the serving cost — for a 7-billion-parameter model, 35 billion parameters to host.
- **MC dropout.** Dropout is normally switched off at test time. Leave it **on**, run the same input $`T`$ times, and you get $`T`$ slightly different networks for free. Cost: $`T\times`$ inference, $`1\times`$ training.
- **SWAG.** During the tail of training, SGD's iterates wander around the minimum rather than sitting in it. Record where they wander; fit a Gaussian to that cloud; sample from it. **The optimizer's own jitter is treated as a free posterior sample generator.**
- **Laplace approximation.** Near a minimum, the loss looks like a quadratic bowl. A Gaussian is exactly the distribution whose log-density *is* a quadratic bowl. So place a Gaussian at the trained solution with covariance $`H^{-1}`$, the inverse Hessian (§1.2.4). ▸ **Sharp minimum (large curvature) → narrow posterior → confident. Flat minimum → wide posterior → uncertain.** That is a satisfying, geometric statement of what confidence *is*, and it is done entirely after training.
- **Variational BNN.** Learn a whole distribution over every weight — a mean and a variance per parameter, so twice the parameters and a much harder optimization.
- **Evidential / Dirichlet.** Have the network output the parameters of a distribution *over* probability vectors, rather than a probability vector. One forward pass, but the answer depends heavily on a regularizer that has no obviously right setting.
- **Conformal.** Refuses to estimate uncertainty at all and instead outputs a *set* with a coverage guarantee. §33.5.

**Why $`H^{-1}`$ cannot be computed directly.** For a model with $`P = 7\times10^9`$ parameters, the Hessian has $`P^2 = 4.9\times 10^{19}`$ entries. At 4 bytes each that is roughly **200 exabytes**, which is why every practical Laplace method uses a structured approximation — diagonal, or **K-FAC** (Kronecker-factored), which represents the curvature of each layer as a product of two much smaller matrices.

#### Why the simple method beats the principled ones

The claim to internalize: **deep ensembles win because their members are in *different valleys*, and everything else stays in one valley.**

| Method | What varies between "samples" | Where the samples live |
|---|---|---|
| Deep ensemble | initialization, data order — everything | **different loss basins** (Ch. 31 §31.5) |
| MC dropout | which units are switched off | one basin, jittered |
| SWAG | SGD's tail noise | one basin, its actual shape |
| Variational BNN | a learned Gaussian per weight | one basin, by construction |

▸ **Two networks in different basins are different *functions*: they make different mistakes, on different inputs.** So when they agree, that agreement is real evidence; when they disagree, you have learned something. Two samples from within one basin are near-copies: they make the *same* mistakes, agree confidently on inputs where the single solution is confidently wrong, and therefore report low uncertainty exactly where the danger is.

> **Analogy.** You want to know whether a bridge design is sound. **Deep ensemble:** commission five independent engineering firms who have never met. If all five say it holds, that means something; if two say it collapses, you have learned a great deal. **MC dropout:** ask one firm the same question five times, on five different afternoons, with a different junior engineer typing the answer. The five answers will be similar and the similarity tells you nothing. **Independence is the whole product**, and it is the one thing you cannot get cheaply.

**The honest caveat.** Ensembles reduce overconfidence under shift; they do not eliminate it. Five networks trained on the same shifted-away-from data can be confidently wrong *together*, because they share the training distribution even though they do not share a basin. **No method in this table gives you a guarantee. §33.5 is the one that does, and it buys the guarantee by answering a weaker question.**

> **Where this came from.** Bayesian neural networks are not new — they were largely worked out in the early 1990s. **David MacKay's** 1992 Caltech doctoral work, *A Practical Bayesian Framework for Backpropagation Networks*, laid out the Laplace approximation for neural networks, used it for model comparison, and derived automatic complexity control from it; **Radford Neal's** 1995 thesis developed Hamiltonian Monte Carlo for the same problem and proved the now-famous result that an infinitely wide neural network with a suitable prior *is* a Gaussian process (the ancestor of Chapter 30's neural tangent kernel). Then the field ignored all of it for roughly two decades, because the networks that mattered got too large. MacKay went on to write *Information Theory, Inference, and Learning Algorithms* and, later, the energy book *Sustainable Energy — Without the Hot Air*; he died in 2016. ▸ **The best joke in this section is that the same DeepMind group produced both sides of the argument:** Charles Blundell was first author of *Weight Uncertainty in Neural Networks* (2015), the elegant variational method — and a co-author of *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles* (Lakshminarayanan, Pritzel, Blundell, 2017), the crude method that beat it. The 2017 paper is explicit that it is not Bayesian and does not pretend to be.

---

## 33.5 Conformal prediction

### The one-line idea

Instead of a point prediction, output a *set* of labels guaranteed to contain the truth with probability $`1-\alpha`$ — with a proof that requires no assumption about the model or the data distribution.

### The analogy

A weather service that, instead of guessing tomorrow's temperature, gives a range and *guarantees* the true value falls inside 90% of the time. It makes no claim about how good the underlying forecasting model is; if the model is bad the range is wide, and if the model is good the range is narrow. **The guarantee is about the procedure, not the model.**

### Split conformal, in full

1. Split the data into training and **calibration** sets.
2. Train any model on the training set.
3. Define a **nonconformity score** $`s(x,y)`$ — higher means worse agreement. E.g. $`s = 1-\hat p(y\mid x)`$.
4. Compute scores $`s_1,\dots,s_n`$ on the calibration set.
5. Let $`\hat q`$ be the $`\lceil(n+1)(1-\alpha)\rceil/n`$ empirical quantile of those scores.
6. **Predict the set** $`C(x_{\text{new}}) = \{y : s(x_{\text{new}},y)\le\hat q\}`$.

#### Running split conformal on actual numbers

Take a 4-class problem with $`10{,}000`$ labelled examples and a demand for **90% coverage**, so $`\alpha = 0.10`$.

**Step 1–2 — split and train.** Hold out $`n = 1000`$ examples as the **calibration** set; train whatever model you like on the other $`9{,}000`$. The model is now frozen and its internals are irrelevant to everything that follows.

**Step 3 — the nonconformity score.** $`s(x,y)`$ answers *"how badly does label $`y`$ fit input $`x`$?"*, **higher is worse**. The standard choice is

$$s(x,y) = 1 - \hat p(y\mid x)$$

If the model gives the true class $`0.95`$, the score is $`0.05`$ — excellent agreement. If it gives the true class $`0.20`$, the score is $`0.80`$ — poor agreement. ▸ **Nothing requires this particular score. Any function of $`(x,y)`$ works and the guarantee still holds; the choice only affects how *large* the sets come out.**

**Step 4 — score the calibration set.** For each of the $`1000`$ held-out points, compute the score **of its true label**. You now have $`1000`$ numbers describing how well this model tends to fit data it has not seen.

**Step 5 — the quantile.** Decode the index first:

$$\frac{\lceil (n+1)(1-\alpha)\rceil}{n} = \frac{\lceil 1001 \times 0.9\rceil}{1000} = \frac{\lceil 900.9\rceil}{1000} = \frac{901}{1000}$$

▸ **So $`\hat q`$ is the 901st smallest of your 1000 scores.** The $`\lceil\cdot\rceil`$ (round up) and the $`n+1`$ are not cosmetic — they are what turn an approximation into a guaranteed inequality, as the proof below shows. Say the 901st value is $`\hat q = 0.62`$.

**Step 6 — predict sets.** Include every label whose score is at most $`0.62`$, i.e. every label with $`\hat p(y\mid x) \ge 1 - 0.62 = 0.38`$:

| New input's $`\hat p`$ | Labels with $`\hat p \ge 0.38`$ | Set | Reading |
|---|---|---|---|
| $`(0.55,\ 0.30,\ 0.10,\ 0.05)`$ | class 1 | $`\{1\}`$ | easy — a confident single answer |
| $`(0.40,\ 0.38,\ 0.12,\ 0.10)`$ | classes 1, 2 | $`\{1,2\}`$ |  ambiguous between two |
| $`(0.30,\ 0.28,\ 0.22,\ 0.20)`$ | none | $`\varnothing`$ | **no plausible answer at all** |

▸ **The set size is the output.** A one-element set is a confident prediction; a three-element set says "it's one of these"; an empty set says the input does not resemble anything the model handles. (Empty sets are legitimate and still satisfy the coverage guarantee on average, but they are often unwelcome in production, so implementations either force in the top class or switch to the APS score, which cannot produce one.)

> **Analogy.** A tailor who has measured a thousand customers and recorded how far off their first guess was each time. They now know that 90% of the time their guess is within one size. So for a new customer they hand over a *range* of sizes wide enough to cover that 90% — narrow when the customer is a typical build, wide when they are unusual. **The tailor is not claiming to be a good judge of size. They are reporting, honestly, how often they have been wrong before.** That is the entire content of conformal prediction.

### The guarantee

▸ $$\mathbb{P}\big(y_{\text{new}}\in C(x_{\text{new}})\big)\ \ge\ 1-\alpha$$

**Requires only exchangeability** — that the calibration points and the test point are exchangeable (weaker than i.i.d.). **No assumption about the model, the architecture, or the data distribution.**

### The proof, in three lines

Under exchangeability, $`s_{\text{new}}`$ is exchangeable with $`s_1,\dots,s_n`$. So the rank of $`s_{\text{new}}`$ among the $`n+1`$ scores is uniform on $`\{1,\dots,n+1\}`$. Hence
$$\mathbb{P}\big(s_{\text{new}}\le s_{(\lceil(n+1)(1-\alpha)\rceil)}\big) = \frac{\lceil(n+1)(1-\alpha)\rceil}{n+1}\ \ge\ 1-\alpha$$
and $`s_{\text{new}}\le\hat q`$ is exactly the event $`y_{\text{new}}\in C(x_{\text{new}})`$. ∎

▸ **That is the entire proof.** It is a statement about ranks, which is why it needs nothing from the model. This makes conformal prediction one of the few  rigorous guarantees available in modern ML, and it is increasingly the standard for high-stakes deployment.

### What you must understand about it

- ▸ **Coverage is *marginal*, not conditional.** You get 90% coverage averaged over the population, **not** 90% for every subgroup. A model can achieve 90% overall while covering 99% of easy cases and 40% of a minority subgroup. **Mondrian/group-conditional conformal** restores per-group coverage by calibrating within groups — do this whenever fairness matters.
- **Adaptive set sizes are the useful signal.** A good score function produces small sets on easy inputs and large sets on hard ones. **APS** (accumulate sorted probabilities) and **RAPS** (with a regularizer to prevent huge tails) are the standard choices for classification.
- **Regression:** conformalized quantile regression gives interval predictions with the same guarantee.
- **Distribution shift breaks it.** Exchangeability fails. Weighted conformal prediction handles covariate shift if you can estimate the likelihood ratio.

---

## 33.6 Out-of-distribution detection

**Scores, in rough order of strength:**

- **Maximum softmax probability.** The baseline. ▸ **Fails badly** because a network can be confidently wrong far from the data — ReLU networks are provably arbitrarily confident sufficiently far from the training data (Hein et al.).
- **Energy score:** $`E(x) = -T\log\sum_j e^{z_j/T}`$. Theoretically better aligned with $`\log p(x)`$ than the softmax, which normalizes away the very magnitude that carries the signal. Simple, strong, and free.
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
| **Covariate shift** | $`p(x)`$ changes, $`p(y\mid x)`$ fixed | new camera, new hospital |
| **Label shift** | $`p(y)`$ changes, $`p(x\mid y)`$ fixed | disease prevalence rises |
| **Concept drift** | $`p(y\mid x)`$ changes | fraud tactics evolve |
| **Domain shift** | joint change | sim → real |
| **Subpopulation shift** | group proportions change | deployment demographics differ |

▸ **The distinction is not academic:** covariate shift is correctable by importance weighting $`\frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}`$; label shift by prior correction (BBSE/EM on the confusion matrix); **concept drift by nothing except new labels.** Diagnosing which one you have determines whether the problem is solvable without relabelling.

**Detection:** monitor input statistics (drift tests on features), monitor confidence and prediction distributions, and — the only reliable signal — collect labels on a sample of production data.

**Handling:** domain adaptation (DANN's adversarial domain-invariant features, CORAL's covariance matching), test-time adaptation (TENT — update only BatchNorm statistics and affine parameters by minimizing prediction entropy), and simple retraining, which is usually the right answer.

---

## 33.8 Adversarial robustness

### The phenomenon

▸ An imperceptible perturbation $`\|\delta\|_\infty\le8/255`$ flips a classifier from 99% "panda" to 99% "gibbon."

**Why it happens (Goodfellow's linear explanation):** for a linear model, $`w^\top(x+\delta)=w^\top x + w^\top\delta`$, and with $`\delta=\epsilon\,\mathrm{sign}(w)`$ the change is $`\epsilon\|w\|_1`$ — which grows with dimension. **In high dimensions, many tiny coordinated changes sum to a large logit change.** Networks are locally near-linear enough for this to apply.

▸ **The deeper account (Ilyas et al., 2019): adversarial examples arise from *non-robust features* that are  predictive.** They demonstrated this by constructing a dataset labelled only via non-robust features, on which standard training yields good accuracy on the *clean* test set. **Adversarial vulnerability is not a bug in the model; it is a property of the data that the model faithfully learned.** This is the answer that distinguishes a deep understanding from a superficial one.

### Attacks

**FGSM:** $`x' = x+\epsilon\,\mathrm{sign}(\nabla_x\mathcal{L})`$ — one step.
**PGD** (the standard): iterated FGSM with random start and projection back into the $`\epsilon`$-ball:
▸ $$x^{t+1} = \Pi_{B_\epsilon(x)}\Big(x^t + \alpha\,\mathrm{sign}\big(\nabla_x\mathcal{L}(x^t,y)\big)\Big)$$
**C&W:** an optimization-based attack minimizing perturbation size subject to misclassification; the strongest white-box attack.
**AutoAttack:** an ensemble of parameter-free attacks — ▸ **the current standard for honest evaluation**, because it removes the tuning advantage that made many defences look better than they were.
**Black-box:** transfer attacks (adversarial examples transfer between models remarkably well), score-based, and decision-based.

### Defences

▸ **Adversarial training** — the only reliably effective empirical defence:
$$\min_\theta\ \mathbb{E}_{(x,y)}\Big[\max_{\|\delta\|\le\epsilon}\mathcal{L}(f_\theta(x+\delta),y)\Big]$$
Solve the inner max with PGD each step. **Costs 5–10× training time**, and reduces clean accuracy by 5–15% on CIFAR-10.

**TRADES** decomposes the objective into natural error plus a boundary term, giving an explicit, tunable accuracy/robustness knob.

▸ **Certified defences** give *provable* guarantees. **Randomized smoothing** is the practical one: define $`g(x)=\arg\max_c\mathbb{P}_{\eta\sim\mathcal{N}(0,\sigma^2I)}\big[f(x+\eta)=c\big]`$. Then $`g`$ is provably constant within an $`\ell_2`$ ball of radius
$$R = \frac{\sigma}{2}\big(\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B})\big)$$
Scales to ImageNet. Interval-bound propagation and convex relaxations give tighter certificates on smaller models.

▸ **The graveyard of broken defences.** Dozens of published defences were later broken, almost all by **gradient masking** — they made gradients uninformative (via non-differentiable ops, randomness, or numerical saturation) rather than making the model robust. Detect it by: black-box attacks beating white-box ones, unbounded attacks failing to reach 0% accuracy, or one-step attacks beating iterative ones. **Always evaluate with AutoAttack and adaptive attacks designed against your specific defence.**

▸ **The robustness–accuracy trade-off is real and provable** in some settings (Tsipras et al.): robust and standard classification can require  different features, so you cannot have both for free.

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

#### Examples and non-examples: what "90% confident" means

Confidence is the most misread number a model produces. The boundary here is worth drawing very carefully, because the errors have real consequences.

**✅ Correct readings**

| Statement | Why it's right |
|---|---|
| "Across all predictions where the model said 90%, about 90% were correct" | **This is the definition of calibration.** It's a claim about a *set*, not a case |
| "This model is well-calibrated in-distribution" | A meaningful, testable claim |
| "A 90% conformal set contains the truth 90% of the time, on average over the test distribution" | Correct — and note "on average" |

**❌ Near-misses — what it does *not* mean**

| Tempting reading | Why it's wrong |
|---|---|
| "There's a 90% chance this particular answer is right" | A single prediction has no frequency. Calibration is a **population** property |
| "High confidence means the model knows the answer" | Confidence measures the softmax's shape, not knowledge. Networks are confidently wrong routinely |
| "The model is 95% accurate, so it's well-calibrated" | **Accuracy and calibration are independent.** A model can be 95% accurate and say 99.9% every time |
| "Low confidence means the input is out-of-distribution" | Softmax confidence is a poor OOD detector — networks are often *more* confident on garbage |
| "It's calibrated, so I can trust it under shift" | Temperature scaling fixes calibration **in-distribution only**. Shift breaks it |
| "My 90% conformal set covers each subgroup 90% of the time" | Coverage is **marginal**, not conditional. It can be 99% for one group and 60% for another |

▸ **The boundary:** calibration is a statement about **long-run frequencies over a population of predictions**, not about any single prediction, and it says nothing about accuracy. **A model that outputs "70%" on every input and is right 70% of the time is perfectly calibrated and completely useless.** Calibration and accuracy are two independent axes, and you need both.

> **Common misconception.** *"Modern networks are overconfident because they're too big."* Size is not the mechanism. Cross-entropy keeps rewarding larger logit gaps *after* the model already classifies correctly — there's always a little more loss to squeeze out by being more certain. Training past the point of fitting therefore inflates confidence with no accuracy gain. **Overconfidence is a property of the objective, not of capacity**, which is why a single temperature parameter fitted on held-out data can largely undo it.

> **Common misconception.** *"Temperature scaling improves the model."* It changes **no** predictions — dividing all logits by a constant leaves the argmax untouched, so accuracy is bit-for-bit identical. It only rescales the confidences. This is precisely why it's safe and why it's the default fix: you get honest probabilities at zero cost to performance.

> **Common misconception.** *"Ensembles work because averaging reduces variance."* That's part of it, but the reason ensembles beat multiple samples from *one* model is that independently-initialized networks land in **different loss basins** and therefore make  different errors. Monte Carlo dropout samples stay in one basin and consequently underestimate uncertainty. **Diversity of solutions, not just averaging, is the mechanism** — which is also why ensembles remain the strongest practical epistemic-uncertainty estimator despite being the least clever.

---

## Did you know?

- **Neural networks got *worse* at calibration as they got better at accuracy.** A 2017 paper by Chuan Guo and colleagues showed that LeNet from the 1990s was better calibrated than the far more accurate ResNets that succeeded it. Progress on one axis moved the other backwards, and nobody had been watching.

- **The fix for that turned out to be a single number.** Temperature scaling — dividing all logits by one scalar fitted on a validation set — recovers most of the lost calibration. One parameter, no retraining, no change to any prediction.

- **Accuracy and calibration are completely independent properties.** A model that says "70%" on every input and is right 70% of the time is perfectly calibrated and worthless. A model that is 99% accurate but always claims 99.99% certainty is excellent and badly calibrated. You need both, and measuring one tells you nothing about the other.

- **Expected Calibration Error is a biased estimator, and the bias depends on your bin count.** Change the number of bins and the number changes. Papers reporting ECE without stating the binning scheme are reporting something not comparable to anyone else's figure — a close cousin of the FID sample-count problem in Chapter 19.

- **Conformal prediction's coverage guarantee holds for *any* model and *any* distribution**, with no assumptions about either. It requires only that the data be exchangeable. The guarantee comes from the ranks of calibration scores, which is why the model can be arbitrarily bad and the guarantee still holds — a badly wrong model simply produces enormous prediction sets.

- **That guarantee is marginal, not conditional, and the distinction matters enormously.** A 90% conformal predictor may cover one subgroup 99% of the time and another 60%, averaging to 90%. It is a real guarantee about a population and no guarantee at all about your patient.

- **Adversarial examples were found by accident.** A 2013 Google team including Christian Szegedy was probing what individual neurons represented and discovered that imperceptible perturbations flipped confident classifications entirely. The initial reaction in the field was to suspect a bug in the experiments.

- **The explanation turned out to be deflating: networks are too *linear*, not too complex.** Ian Goodfellow's 2014 argument showed that small perturbations compound through many near-linear layers — the $`1.2^{50} \approx 9100`$ amplification of Chapter 1 §1.1.4. The exotic-sounding phenomenon is a consequence of multiplying fifty numbers slightly larger than one.

- **Adversarial training buys robustness at a measurable cost in clean accuracy.** This appears to be a  trade-off rather than an engineering shortfall, and there is theoretical work arguing the two objectives are fundamentally in tension.

- **Most published adversarial defences were later broken.** A series of papers by Anish Athalye, Nicholas Carlini, and David Wagner showed that many defences worked by "obfuscated gradients" — making the attacker's optimization fail rather than making the model robust. Stronger attacks defeated them. The field now expects adaptive-attack evaluation as standard.

- **A model can achieve high accuracy by learning the background instead of the object.** Classifiers have been found keying on snow to detect wolves, on hospital-specific markings to detect pneumonia, and on watermarks to detect horses. The simplicity bias of Chapter 31 is the mechanism: the network latches onto the *easiest* predictive feature, and that is frequently a spurious one.

- **Softmax confidence is a poor out-of-distribution detector**, and often confidently high on inputs that are pure noise. This is unintuitive — you'd expect nonsense to produce uncertainty — and it is why energy-based scores, Mahalanobis distance, and ODIN exist as separate machinery.

---

## Check for Understanding

**Accuracy and calibration are independent, and modern networks are overconfident because cross-entropy keeps pushing logits apart after accuracy saturates — fixable in-distribution by a single temperature parameter, but not under shift, where you need epistemic uncertainty (best estimated by ensembles, because they span different loss basins) or conformal prediction, whose coverage guarantee is a statement about the ranks of calibration scores and therefore holds for any model and any distribution, though only marginally and not per subgroup.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What does "90% confident" actually mean?** Why is it a claim about a *set* of predictions rather than about one?
2. **Why are accuracy and calibration independent?** Give an example of a model that's excellent at one and terrible at the other.
3. **Why did networks get *less* calibrated as they got more accurate?** (What is cross-entropy still optimizing after accuracy saturates?)
4. **Why does temperature scaling change no predictions at all?**
5. **Why is Expected Calibration Error's value dependent on your bin count**, and what should you always report alongside it?
6. **Distinguish aleatoric from epistemic uncertainty.** Which one can more data fix?
7. **Why do deep ensembles beat Monte Carlo dropout** for epistemic uncertainty? (Think about loss basins.)
8. **How can conformal prediction guarantee coverage without any assumption about the model?** What is it actually using?
9. **Why is "marginal coverage" a much weaker promise than it sounds**, and when would that hurt you?
10. **Why does an imperceptible perturbation flip a confident classifier?** (The answer is about linearity, not complexity.)
11. **What are "obfuscated gradients"**, and why were so many adversarial defences broken?
12. **Why might a pneumonia classifier be reading the hospital's scanner markings**, and what does that have to do with simplicity bias?

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 34 — The Interview Bank](34-interview-bank.md)
