# Chapter 34 — The Interview Bank

> **How to use this:** cover the answer, attempt it out loud, then compare. An answer you can *recognize* is not an answer you can *give*. Questions marked **★** are the ones that most reliably separate strong candidates; §34.11 collects them.
> Section references point to where the full treatment lives.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### How to actually use this bank as a beginner

The answers below are **compressed reference answers, not explanations.** They are written the way a strong candidate *thinks* under time pressure — a derivation skeleton, a number, a failure mode — because that is what fits on an index card and what a grader is listening for. They are deliberately not written to teach an idea for the first time. If you meet one cold, it will read as a wall of symbols, and that is not a verdict on you.

So use the bank in this order:

1. **Read the linked section first.** Every answer ends with a pointer like (§12.7). That is where the idea is actually built, slowly, with analogies and worked numbers. This chapter is the *exam*; those chapters are the *course*. Attempting the exam first is a way of feeling bad, not a way of learning.
2. **Cover the answer and say yours out loud.** Out loud — not in your head. The gap between "I know this" and "I can say this" is enormous and it is only audible. Recognition is free; production is the thing being tested.
3. **Grade yourself on three things, in this order.** Did you get the *mechanism*? Did you produce a *number* or an order of magnitude? Did you name a *failure mode*? A definition alone scores close to zero, because everyone has one.
4. **When an answer is too dense, drop to the plain-English version.** Each of §34.1–§34.10 ends with a **Saying it in plain English** subsection that rewrites the four to six hardest answers in that section with no notation at all. That version is what you would actually say in a room. The compressed version is what you would write on a whiteboard *while* saying it.

▸ **The compressed answer and the spoken answer are two different artifacts, and a strong candidate carries both.** Interviewers rarely want notation recited at them. They want the sentence — and then the notation offered afterwards as proof the sentence is not bluffing. Learn to lead with the sentence and keep the algebra in reserve.

One more piece of framing. The answers here are terse because interview answers *should* be terse: a good one runs sixty to ninety seconds. If your spoken version takes four minutes, you have not understood it better than the book — you have understood it less well, and are covering the gap with words.

### The symbols that recur across this bank

This chapter draws on all thirty-three preceding ones, so it reuses their notation without reintroducing it. These are the symbols you will hit most often below.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`\mathbb{E}[\cdot]`$ | "expectation of" | The average, weighted by probability. In code, the mean over a batch. |
| $`\mathcal{L}`$ | "script L" | The loss being minimized. |
| $`\theta`$ | "theta" | All of the model's parameters, gathered into one big vector. |
| $`\nabla_\theta \mathcal{L}`$ | "grad theta L" | The slope of the loss with respect to every parameter — which way is uphill. |
| $`\lVert \cdot \rVert`$ | "norm of" | Length of a vector. $`\lVert g \rVert`$ is the size of the gradient. |
| $`\sigma_j`$ | "sigma j" | The $`j`$-th singular value — how much a matrix stretches one direction. |
| $`\kappa`$ | "kappa" | Condition number, $`\sigma_{\max}/\sigma_{\min}`$ — how lopsided the problem is. |
| $`\lambda`$ | "lambda" | Regularization strength, or an eigenvalue. Context decides which. |
| $`\eta`$ | "eta" | Learning rate — the step size. |
| $`\gamma`$ | "gamma" | Discount factor in reinforcement learning; also a per-leaf cost in XGBoost. |
| $`\beta`$ | "beta" | A momentum coefficient, or a KL strength (DPO), or an inverse temperature. |
| $`p(x \mid y)`$ | "p of x given y" | Probability of $`x`$ once $`y`$ is known. The bar is "given," never "divide." |
| $`x \sim p`$ | "x drawn from p" | $`x`$ is a random sample from the distribution $`p`$. |
| $`\mathcal{O}(\cdot)`$ | "big-O" | How cost grows with problem size, constants ignored. |
| $`\odot`$ | "elementwise product" | Multiply matching entries and keep them separate. |
| $`\propto`$ | "is proportional to" | Equal up to a constant nobody cares about. |
| $`\bar\alpha_t`$ | "alpha-bar t" | Diffusion schedule: how much of the clean signal survives by step $`t`$. |
| $`\rho`$ | "rho" | A correlation, or the probability ratio $`\pi_\theta/\pi_{\text{old}}`$ in PPO. |
| $`\Phi(\cdot),\ \Phi^{-1}(\cdot)`$ | "Phi, Phi inverse" | The Gaussian cumulative distribution and its inverse (the z-score lookup). |
| ★ / ★★ | — | Difficulty marks: ★ separates candidates, ★★ is a full derivation. |

---

## 34.1 Mathematics & statistics

**1. What does the SVD tell you about a matrix?**
$`A=U\Sigma V^\top`$ decomposes any linear map into rotate → scale → rotate. Singular values give the scaling in each direction; the number of nonzero ones is the rank; $`\sigma_1`$ is the operator norm; $`\sigma_1/\sigma_r`$ is the condition number. Truncating gives the optimal rank-$`k`$ approximation in Frobenius and spectral norm (Eckart–Young). (§1.1)

**2. Why is the condition number important in optimization?**
Gradient descent's convergence rate for a quadratic is $`\left(\frac{\kappa-1}{\kappa+1}\right)^t`$. Large $`\kappa`$ means the loss surface is a narrow ravine: the step size is bounded by the steepest direction while progress is limited by the flattest. Momentum improves the dependence to $`\sqrt\kappa`$. (§4.3–4.5)

**3. What is the difference between forward-mode and reverse-mode autodiff, and why does DL use reverse?**
Forward computes a Jacobian-vector product per input dimension; reverse computes a vector-Jacobian product per output dimension. Cost scales with $`\min(n_{\text{in}}, n_{\text{out}})`$ respectively. A loss has one output and millions of parameters, so reverse mode gets all gradients in one backward pass. (§1.3)

**4. Explain KL divergence, and the difference between forward and reverse KL.** ★
$`\mathrm{KL}(p\|q)=\mathbb{E}_p[\log p/q]`$ — the extra bits from coding $`p`$ with a codebook built for $`q`$. Non-negative, asymmetric, not a metric. **Forward** $`\mathrm{KL}(p\|q)`$ is mode-covering: $`q`$ must be positive wherever $`p`$ is, so it smears. **Reverse** $`\mathrm{KL}(q\|p)`$ is mode-seeking: $`q`$ can ignore modes but must not put mass where $`p`$ has none. Maximum likelihood minimizes forward KL (hence blurry VAEs); variational inference minimizes reverse (hence underestimated posterior variance). (§1.4)

> **Where this came from.** **Solomon Kullback and Richard Leibler** published *On Information and Sufficiency* in 1951. Both were cryptanalysts — they worked for the U.S. Army's Signal Intelligence Service and later the National Security Agency, and the quantity was built to answer an operational question: given an intercepted message, how much evidence does it provide for one hypothesized cipher over another? A detail worth knowing, because it is exactly what this interview question is about: in their own terminology the word **"divergence" referred to the *symmetric* combination** $`\mathrm{KL}(p\|q)+\mathrm{KL}(q\|p)`$, while the one-directional quantity we now universally call "the KL divergence" was the *directed* divergence, or the mean information for discrimination. The asymmetry was never an oversight to be apologized for — it was the whole point, because "how surprised am I by your messages" and "how surprised are you by mine" are  different questions. The measure that now trains every large language model on earth was invented for code-breaking.

**5. Derive the ELBO.**
$`\log p(x) = \log\int p(x,z)dz = \log\mathbb{E}_q\left[\frac{p(x,z)}{q(z)}\right]\ge\mathbb{E}_q\left[\log\frac{p(x,z)}{q(z)}\right]`$ by Jensen. Expanding, the gap is exactly $`\mathrm{KL}(q(z)\|p(z|x))`$. (§1.4)

**6. What is the standard error of the mean, and why does it matter?**
$`\sigma/\sqrt n`$. Halving error bars needs $`4\times`$ the data. It governs whether any measured improvement is real. (§1.3, §3.6)

**7. Central Limit Theorem — statement and a caveat.**
The standardized sample mean of i.i.d. finite-variance variables converges to $`\mathcal{N}(0,1)`$. Caveats: requires finite variance (fails for Cauchy), convergence is slow in the tails, and it says nothing about the maximum — which is why record statistics (§3.6) need a different tool.

> **Common misconception.** *"With enough samples, my data becomes Gaussian."* The Central Limit Theorem says nothing whatever about your data. It says the **sample mean** — one number computed from many — has a sampling distribution that approaches a bell curve. Ten thousand income figures are exactly as skewed at $`n=10{,}000`$ as a hundred were at $`n=100`$; it is their *average* that behaves nicely. The belief is tempting because the two claims sound almost identical, and because "normality" gets invoked loosely as a background regularity condition in half the statistics anyone remembers. The practical bite is worth carrying into a room: a comparison of two means on wildly skewed data is usually fine, because it is a statement about means — while a model that assumes Gaussian *residuals* is not rescued at all by collecting more rows.

**8. Bayes' theorem, and what a prior does.**
$`p(\theta|x)\propto p(x|\theta)p(\theta)`$. The prior is a regularizer: $`\ell_2`$ is a Gaussian prior, $`\ell_1`$ a Laplace prior, and MAP estimation is penalized maximum likelihood. (§7.5)

> **Where this came from.** **Thomas Bayes** was an English Presbyterian minister who never published the theorem. It appeared in 1763, two years after his death, when **Richard Price** found it among his papers and sent it to the Royal Society. **Pierre-Simon Laplace** arrived at the same idea independently around 1774 and did far more with it than Bayes had — for most of the nineteenth century the method was known not as Bayes' theorem but as *inverse probability*, and Laplace's name was attached to it more often than Bayes'. It then spent much of the twentieth century in disrepute, attacked by Fisher and others for the apparent arbitrariness of choosing a prior. What kept it alive was that it kept working on problems where nothing else did: Alan Turing's team used sequential Bayesian evidence-weighing against Enigma at Bletchley Park (the unit they used, the *ban*, was named after the town of Banbury where the punched sheets were printed), and in 1968 the U.S. Navy used Bayesian search theory to locate the wreck of the submarine USS *Scorpion* in the open Atlantic. The rehabilitation you see today is largely a consequence of cheap computation making the integrals tractable.

**9. What is mutual information and where does it appear in DL?**
$`I(X;Y)=H(X)-H(X|Y)=\mathrm{KL}(p(x,y)\|p(x)p(y))`$. Appears in InfoNCE bounds (§25.3), the information bottleneck, feature selection, and the epistemic-uncertainty decomposition (§33.3).

**10. Explain the Johnson–Lindenstrauss lemma and one place it matters.**
$`n`$ points can be embedded into $`O(\log n/\epsilon^2)`$ dimensions preserving pairwise distances to $`1\pm\epsilon`$. Underpins random projection, LSH-based deduplication, and — conceptually — superposition, since it implies exponentially many nearly-orthogonal directions exist. (§1.2, §32.2)

> **Common misconception.** *"Maximizing the evidence lower bound (ELBO) maximizes the log-likelihood."* It maximizes a *floor* underneath the log-likelihood, and the distance from floor to ceiling is exactly $`\mathrm{KL}(q(z)\,\Vert\,p(z\mid x))`$ — a quantity that moves during training. So the bound can rise while the thing it bounds falls, provided the approximation gap closed faster than the model degraded. The belief is tempting because in practice the two usually move together, which is precisely what makes the exceptions hard to spot. The habit to take away: when a paper reports "ELBO" and discusses it as though it were the likelihood, nobody has told you how loose the bound is — and a loose bound is an objective that has quietly drifted from the one you wanted.

### Saying it in plain English — mathematics and statistics

*The answers above are what you would write. These are what you would say. Six of the densest, with the notation removed and the intuition put back.*

**Q1 — What the SVD tells you about a matrix.**
Every matrix, however ugly, does the same three things in sequence: it rotates the space, stretches it along a few chosen directions, then rotates it again. The SVD is the receipt for that. The stretch factors are the singular values, and almost everything you want to know is a fact about them. A stretch factor of zero means the matrix flattens that direction out of existence — count the nonzero ones and you have the rank. The largest stretch is the most the matrix can ever amplify anything, which is what "operator norm" means. The ratio of largest to smallest tells you how lopsided the transformation is, which is the condition number and which governs how badly numerical algorithms will behave.

And the part that earns the question a place in interviews: keep only the biggest few stretches and discard the rest, and you get the **best possible** low-rank imitation of the matrix — provably the best, not merely a decent heuristic. That single guarantee is why a seventy-billion-parameter model can be adapted with a rank-8 patch, why PCA is computed the way it is, and why "just take the top-$`k`$ components" is a defensible thing to do rather than a shrug.

**Q2 — Why the condition number matters in optimization.**
Picture a valley shaped like a drainage ditch: a hundred metres long, half a metre wide. You want the bottom. The steep side walls force you to take small steps — a big step bounces you off one wall and into the other. But those same small steps are what must carry you down the long, nearly flat length of the ditch. Small steps, long way to go, and the two constraints are set by *different* directions.

The condition number is exactly the ratio of "steep" to "flat," and it directly sets how many steps you need. Momentum helps because it lets the tiny, consistent downhill push along the ditch accumulate over many steps, while the side-to-side bouncing keeps reversing sign and cancels itself out. That is why momentum improves the dependence from the condition number to its square root — and a square root of a big number is a much smaller big number.

**Q3 — Forward-mode versus reverse-mode autodiff.**
Both compute exact derivatives; they differ only in the direction of the sweep. Forward mode asks: *"if I wiggle this one input, what happens to everything downstream?"* — one sweep per input. Reverse mode asks: *"this one output moved; which upstream quantities are responsible?"* — one sweep per output.

Now look at what a neural network is: hundreds of millions of inputs (the weights) and exactly one output (the loss). Wiggling each weight separately would take hundreds of millions of sweeps. Working backwards from the single loss takes one. The advantage is a factor equal to the parameter count.

▸ **Notice this is a fact about shape — many-in, one-out — not a fact about neural networks.** Any many-in-one-out computation gets the same deal, which is why reverse-mode differentiation was invented decades before anyone applied it to learning, and why it is the right answer for weather models and circuit optimization too. If you invert the shape — one input, many outputs, as in sensitivity analysis of a simulation — forward mode wins instead.

**Q4 — Forward versus reverse KL.**
KL divergence measures waste. Suppose you built a compression scheme tuned for one distribution and then used it on messages drawn from a different one. KL is the number of extra characters you burn per message. It is zero only when the distributions match, and it is not symmetric: the cost of using q's codebook on p's messages is not the cost of using p's codebook on q's.

The two directions create two personalities, and this is the part interviewers actually want.

Forward KL punishes you severely for assigning near-zero probability to something that  happens — the surprise term blows up. So a model minimizing it stretches itself thin to cover everything, including the empty space between two distinct kinds of data. That is the blurry-VAE, average-of-all-faces failure: **mode-covering**.

Reverse KL punishes you for putting mass where the truth has none, and doesn't much care about the parts of the truth you ignore. So a model minimizing it hides inside one mode and refuses to leave. That is the overconfident, too-narrow variational posterior: **mode-seeking**.

Then the connection that makes it a real answer rather than trivia: maximum likelihood *is* forward KL in disguise, and variational inference *is* reverse KL. So the blurriness of likelihood-trained generative models and the overconfidence of variational approximations are not two unrelated annoyances — they are the same asymmetry, read in opposite directions. If you remember one line: **forward smears, reverse shrinks.**

**Q5 — What the ELBO is doing.**
You want to know how likely your data is under the model, but computing that requires summing over every possible hidden explanation of it, and there are far too many to enumerate. So instead of computing the number, you compute a *floor* underneath it — and then push the floor up.

The picture: you are standing in fog trying to determine the height of a mountain whose summit you cannot see. You can walk to some spot and measure your own altitude, which is certainly no higher than the summit. Climb, measure again, and your lower bound rises. The ELBO is that altitude reading.

The inequality that gives you the floor is Jensen's — the average of a log is at most the log of the average — and the  beautiful part is the *size of the gap*. It isn't some unknown slack; it is exactly how wrong your guessed explanation is compared to the true one. Which means pushing the floor up does two useful jobs simultaneously: it makes the model fit the data better, and it makes your guess about the hidden variable more accurate. One objective, two improvements, and no way to cheat by improving the bound while the real quantity gets worse.

**Q10 — Johnson–Lindenstrauss.**
Take a cloud of points living in a million dimensions and project it, *at random*, into a few hundred dimensions. Every pairwise distance survives almost unchanged. The number of dimensions you need depends on how many *points* you have — and only logarithmically — and does not depend on the original dimension at all.

It sounds impossible until you internalize what high-dimensional space is actually like: overwhelmingly empty, and any two random directions are almost certainly close to perpendicular. Random projection preserves distances for the same reason a random sample estimates a mean: each coordinate of the projection is an independent noisy measurement of the true distance, and averaging a few hundred of them is enough.

Where it shows up: random projections and sketching, hashing-based near-duplicate detection in training corpora, and — the reason it appears in modern interviews — **superposition**. If exponentially many nearly-orthogonal directions exist in a modest space, a model with 4,096 residual dimensions can carry tens of thousands of distinguishable features, and the interference between them stays small enough to ignore. The lemma is the licence for that whole picture.

---

## 34.2 Learning theory & evaluation

**11. Derive the bias–variance decomposition.**
$`\mathbb{E}[(y-\hat f)^2] = \big(\mathbb{E}[\hat f]-f\big)^2 + \mathbb{E}\big[(\hat f-\mathbb{E}[\hat f])^2\big] + \sigma^2`$. Add and subtract $`\mathbb{E}[\hat f]`$; the cross term vanishes. (§2.2)

**12. Why are VC-dimension bounds useless for deep networks?** ★
A network with $`p`$ parameters at 32-bit precision gives $`\log|\mathcal{H}|\le22p`$; for $`p=10^7`$, $`n=10^5`$ the bound exceeds 1 on a loss bounded by 1. It's vacuous. Deeper reason: uniform convergence bounds what the class *could* do; deep nets can fit random labels (Zhang et al.), so any capacity-only bound must be vacuous. The explanation must come from the algorithm's implicit bias. (§2.4, §2.7, §31.2)

**13. Explain the bootstrap. When does it fail?**
Resample with replacement $`B`$ times, recompute the statistic, use the spread as the sampling distribution. Fails for extremes (max/min — the bootstrap distribution is atomic), for dependent data without block resampling, and for heavy tails with infinite variance. (§3.1)

> **Where this came from.** **Bradley Efron** introduced the bootstrap at Stanford in 1979, and named it after the physically impossible feat of lifting yourself by your own bootstraps — the image comes from the Baron Munchausen tales, where the Baron hauls himself out of a swamp by his own hair or bootstraps depending on the telling. The name is a joke about the apparent absurdity of the method: you are using a single sample to estimate how much a *different* sample would have differed, with no new data and no distributional assumptions. What made it more than a curiosity was not new mathematics but cheap computing — resampling a dataset ten thousand times is trivial now and was a  expense in 1979, which is why a technique this simple had to wait until the era of the mainframe to be worth writing down.

**14. Why does $`k`$-fold CV variance not shrink like $`1/k`$?**
The folds share training data, so the fold estimates are correlated: $`\mathrm{Var}=\frac{\sigma^2}{k}+\frac{k-1}{k}\rho\sigma^2`$. The correlation term doesn't vanish. (§3.3)

**15. What is nested cross-validation and why do you need it?**
Outer loop estimates performance; inner loop selects hyperparameters. Tuning and reporting on the same CV gives optimistically biased numbers, with bias growing in the number of configurations tried. (§3.3)

> **Common misconception.** *"Cross-validation gives an unbiased estimate of my model's performance."* Two separate problems, and interviewers probe both. First, $`k`$-fold estimates the performance of a model trained on $`n(1-1/k)`$ examples, not on all $`n`$ — pessimistic, and the gap is material when data is scarce. Second and far worse: the moment you use a cross-validation score to *choose* something — a hyperparameter, a feature set, an architecture — that score has stopped being an estimate and become a selection criterion, and selection criteria are optimistic by construction. The belief is tempting because the *first* number you compute  is close to unbiased. It is the fortieth one, the one you kept because it was the best, that is not. **Nested cross-validation exists to keep the choosing and the reporting in different loops**, and the cost of skipping it grows with how many configurations you tried, not with how big the dataset is.

**16. You track the minimum of a noisy metric across 40 epochs. Why is the reported best optimistically biased?** ★
The minimum of $`n`$ noisy draws is below the true mean by roughly $`\sigma\Phi^{-1}\!\big(1-\frac{1}{n+1}\big)`$ — about $`2\sigma`$ for $`n\approx43`$. You are reporting a favourable noise draw, not a better model. Fix: re-evaluate the selected checkpoint on a fresh, larger set. (§3.6)

**17. A model shows no new best for 13 epochs, then two records back to back. What can you conclude?** ★
Under a flat-performance null, $`P(\text{no record in epochs }23..35)=\prod_{k=23}^{35}(1-1/k)=22/35=0.63`$ — the dry spell is the *most likely* outcome and carries no information. Back-to-back records at 36 and 37 have probability $`\frac{1}{36\cdot37}=7.5\times10^{-4}`$ — that's the evidence. **Read the cluster, not the gap.** (§3.6)

**18. Precision, recall, F1, ROC-AUC, PR-AUC — when do you use which?**
Precision = TP/(TP+FP); recall = TP/(TP+FN). ROC-AUC is invariant to class balance, which is a *problem* when positives are rare — it looks good while precision is terrible. PR-AUC is sensitive to the positive rate and is the right choice under heavy imbalance. F1 is their harmonic mean; use $`F_\beta`$ when the costs differ. (§22.6)

> **Where this came from.** "ROC" stands for **Receiver Operating Characteristic**, and the receiver in question is a piece of radar hardware. The curve was developed by signal-processing engineers during and shortly after the Second World War to describe how well a radar receiver — and the operator watching it — could separate a returning aircraft echo from background noise. Turn the detector's sensitivity up and you catch more real aircraft but also call more waves and flocks of birds; turn it down and you miss real ones. Sweeping that threshold and plotting hits against false alarms is the ROC curve, and it kept its wartime name when psychologists adopted it in the 1950s for signal-detection experiments and medicine adopted it in the 1970s for diagnostic tests. Knowing the origin makes the modern failure mode obvious: radar operators faced roughly balanced stakes and a steady base rate of targets. Fraud detection at one positive in ten thousand does not resemble that situation at all, which is precisely why ROC-AUC flatters models that are useless in deployment.

**19. What's a permutation test?**
Shuffle labels many times, recompute the statistic, and locate the observed value in that null distribution. Assumption-free, exact in principle, and the right tool when you don't trust a parametric null. (§3.4)

**20. What is Monte Carlo standard error and why report it?**
The uncertainty from your own sampling procedure, $`\sigma/\sqrt{B}`$. Without it, a "0.3% improvement" from 100 samples is indistinguishable from noise. (§3.5)

> **Common misconception.** *"Early stopping doesn't train on the validation set, so validation loss is still a clean estimate."* Early stopping fits exactly one parameter — the number of steps — on the validation set. That is fitting, and it contaminates the number in the ordinary way. With forty checkpoints inspected, the reported minimum sits roughly two standard errors below the truth (Q16), and the contamination gets *worse* the more diligently you evaluate. The belief is tempting because one integer feels far too small a thing to bias anything; the arithmetic disagrees, because the bias scales with **how many times you looked**, not with how many parameters you fitted. The same argument indicts learning-rate selection by "best validation loss," checkpoint averaging chosen post hoc, and every leaderboard where the same held-out set has been queried a thousand times.

### Saying it in plain English — learning theory and evaluation

*This section is mostly about one uncomfortable idea: most of what looks like a result is a selection effect. Here are five of them said out loud.*

**Q12 — Why VC bounds are useless for deep networks.**
A capacity bound is a promise of the form: *"this family of models is so flexible that until you have this much data, good training performance tells you nothing."* Count the flexibility of a modern network, plug it in, and the bound comes back saying you need more data than exists. It does not say the model will fail. It says the bound has nothing to tell you. It is a speed-limit sign reading ten thousand kilometres per hour — technically true, operationally silent.

And you cannot patch it, because there is a decisive experiment. Take a standard image dataset, replace every label with a random one so there is no pattern whatsoever, and a standard network will still fit the training set perfectly. The family  *can* express pure nonsense. So any argument built only on "what could this family express" is obliged to conclude nothing — that is not a weakness of one particular bound, it is a structural consequence.

▸ **The explanation for why the network generalizes therefore has to come from somewhere other than the space of available functions.** It has to come from *which* function the optimizer actually walks toward, out of the many that fit. Gradient descent has preferences — it drifts toward large-margin, small-norm, smooth solutions — and those preferences, not the model's capacity, are the modern answer. Saying that sentence is the point of the question.

**Q14 — Why k-fold CV variance doesn't shrink like one over k.**
Averaging ten *independent* measurements cuts your uncertainty by about three. But the folds are not independent — any two training folds share the great majority of their data. If the dataset itself is unrepresentative, say it is missing an entire customer segment, then every fold inherits that flaw identically, and averaging ten copies of the same mistake removes exactly none of it.

So the variance splits into two pieces: a part that  shrinks as you add folds, and a stubborn part fixed by how correlated the folds are, which does not shrink at all. Ten folds do not buy you ten times the confidence. Usually more like two or three. The practical consequence: **when two models differ by less than the fold-to-fold spread, adding folds will not settle the argument** — only more data, or a fresh test set, will.

**Q16 — Why the best epoch is optimistically biased.**
You watch a noisy validation number for forty epochs and report the lowest value you saw. But you did not select the best model. You selected the epoch at which the model *and the noise* happened to line up in your favour. Draw forty numbers from a machine that produces pure noise, report the smallest, and it will sit roughly two standard deviations below the mean — purely because you looked forty times. Nothing improved. You went shopping.

The nastiest property is that this bias *grows* with how carefully you evaluate. Check every epoch instead of every fifth, and your reported number gets better while your model does not. A more diligent-looking experiment produces a more inflated result.

The fix costs almost nothing: take the checkpoint you selected and evaluate it once more, on data that played no part in selecting it. **Selection and estimation must not use the same numbers.** That one sentence covers this question, nested cross-validation, and most of why leaderboards drift.

**Q17 — The dry spell and the double record.**
Suppose the model has  stopped improving and every epoch is now an independent draw from the same noisy distribution. How often would you expect a new best? At epoch 30, a new best requires this draw to beat all 29 before it — probability one in thirty. **Records get rarer over time automatically**, with no learning involved at all, simply because the bar is the maximum of an ever-growing pile.

So a thirteen-epoch stretch with no new best is not evidence of anything. Under the pure-noise story it is not merely possible, it is the single most likely outcome. People habitually read it as "training has stalled" and kill the run.

What noise cannot comfortably explain is two records back to back, late. Under the same story that has a probability of roughly one in thirteen hundred. *That* is the signal. **Read the cluster, not the gap** — and note that the intuition most practitioners have is precisely backwards: they panic during the plateau, which is uninformative, and shrug at the burst, which is not.

**Q18 — ROC-AUC versus PR-AUC.**
ROC-AUC answers a specific question: hand the model one random positive and one random negative — how often does it score the positive higher? Notice what that question ignores: how *rare* positives are. That invariance sounds like a virtue and is actually the trap.

Imagine fraud at one case in ten thousand. A model can be excellent at the pairwise question and still, at any threshold you could actually deploy, produce a hundred false alarms for every  catch — because there are ten thousand times more negatives available to be wrongly flagged. ROC-AUC will read 0.98 and the fraud team will hate you, and both of you are right.

PR-AUC asks the question the fraud team lives with: of the cases I flagged, how many were real, across every threshold? It moves when the base rate moves, which is exactly the sensitivity you want. The rule: **the rarer the positive class, the more you should quote precision–recall and the less you should trust ROC.** The follow-up to be ready for is that neither number tells you where to set the threshold — that is a separate decision, made on validation data, using the actual relative cost of the two errors.

---

## 34.3 Optimization

**21. Adam vs SGD — what actually differs, and when do you use each?** ★
Adam keeps per-parameter first and second moment estimates and normalizes the step by $`\sqrt{\hat v}`$, making it invariant to per-parameter gradient scale. SGD+momentum generalizes better on vision (its implicit bias is $`\ell_2`$/max-margin, §31.2); Adam is essential for transformers, where gradient scales vary enormously across layers and embedding rows are sparsely updated. (§5.3, §5.6)

> **Where this came from.** Stochastic gradient descent is older than machine learning: **Herbert Robbins and Sutton Monro** published *A Stochastic Approximation Method* in 1951, solving a problem in sequential experimentation — how to find the input at which an unknown, noisily-measured function hits a target value, updating your guess after each single noisy observation rather than waiting for a full experiment. Their convergence conditions on the step size (it must shrink, but not too fast) are the ancestors of every learning-rate schedule in this book. Momentum arrived in 1964 from **Boris Polyak**, who called it the *heavy ball* method and meant the analogy literally: give the descending point mass and it stops responding to every local wiggle of the slope. **Yurii Nesterov's** accelerated variant followed in 1983 with a provably optimal rate for smooth convex problems. None of this was about neural networks; the machinery was sitting complete on the shelf for decades before there was anything worth pointing it at.

**22. Derive Adam's bias correction.**
$`m_t=(1-\beta_1)\sum_{i}\beta_1^{t-i}g_i`$. With $`m_0=0`$ and stationary $`g`$, $`\mathbb{E}[m_t]=(1-\beta_1^t)\mathbb{E}[g]`$. Divide by $`(1-\beta_1^t)`$. Without it, early steps are biased toward zero — and for $`\beta_2=0.999`$ the $`v`$ bias persists for ~1000 steps, which would make early steps enormous. (§5.3)

**23. What's the difference between Adam+L2 and AdamW?** ★
L2 adds $`\lambda\theta`$ to the gradient, which then gets divided by $`\sqrt{\hat v}`$ — so parameters with large historical gradients get *less* decay. AdamW applies $`\theta\leftarrow\theta-\eta\lambda\theta`$ directly, decoupled from the adaptive scaling. That restores the intended uniform shrinkage, and it's why AdamW is standard. (§5.2)

> **Where this came from.** Adam was published by **Diederik Kingma and Jimmy Ba** in 2014, both PhD students at the time; the name is not a person's but an abbreviation of *adaptive moment estimation*. Two later corrections are worth knowing, because they show how an algorithm can become universal before it is understood. First, the paper's convergence proof was **wrong**: Reddi, Kale and Kumar exhibited a simple convex problem on which Adam provably fails to converge, published it in 2018, and won a best-paper award for it. By then Adam had already been the default optimizer of the field for three years. Second, **Ilya Loshchilov and Frank Hutter** pointed out in 2017 that essentially every framework's `Adam(weight_decay=...)` was not implementing weight decay at all but $`\ell_2`$ regularization, which is a different thing under an adaptive optimizer — the fix they proposed is AdamW. The pattern to take away: the most-used optimizer in history spent years shipping with a broken proof and a misimplemented regularizer, and worked anyway. Empirical success and theoretical understanding are much more loosely coupled in this field than the papers imply.

**24. Why do transformers need learning-rate warmup?**
Early gradients are unrepresentative, and Adam's $`\hat v`$ estimate is high-variance in the first ~$`1/(1-\beta_2)`$ steps, so the effective step can be enormous. Post-LN transformers also have $`O(\sqrt L)`$ gradient amplification at initialization. Warmup lets the second-moment estimate stabilize. RAdam derives the correction analytically. (§5.4)

**25. What is the Edge of Stability?**
GD on neural networks doesn't stay in the regime $`\eta<2/\lambda_{\max}`$; instead $`\lambda_{\max}`$ *rises* until it hits $`2/\eta`$ and then hovers there, with the loss decreasing non-monotonically. So sharpness is determined by the learning rate, not fixed by the problem. (§5.5)

**26. Explain gradient clipping and which variant to use.**
Rescale $`g\leftarrow g\cdot\min(1, c/\|g\|)`$ (norm clipping, preserves direction) rather than clipping elementwise (which changes direction). Essential for RNNs and large transformers where rare batches produce huge gradients. (§4.8)

> **Common misconception.** *"Clipping is a safety net — it doesn't change what you're optimizing."* It does. A clipped stochastic gradient is a **biased** estimate of the true gradient: on every step where the clip engages you follow a rescaled direction, and averaged over batches that rescaled direction is not the gradient of your loss. In practice the bias is small and the variance reduction is enormous, so the trade is almost always worth making — but if your threshold is active on most steps you are no longer running the algorithm you believe you are running, and the symptom is deceptive: a beautifully smooth loss curve that quietly refuses to improve. Instrument the fraction of steps that clip. A few percent is a safety net. Eighty percent is a learning rate you set by accident.

**27. How should learning rate scale with batch size?**
Linear scaling ($`\eta\propto B`$) for SGD, from the gradient-noise argument; square-root scaling ($`\eta\propto\sqrt B`$) is safer for Adam. Both need warmup at large $`B`$. (§4.6)

**28. Why does SGD generalize better than full-batch GD?**
Gradient noise acts as a temperature $`\propto\eta/B`$ in an SDE whose stationary distribution weights basins by *volume*. Wide (flat) basins have exponentially more volume in high dimensions, and flat minima generalize better. Predicts — correctly — that large batch at fixed LR generalizes worse. (§4.6, §31.4)

> **Common misconception.** *"Adam reaches a lower training loss faster, so it is the better optimizer."* Speed of descent on the training objective and quality of the final solution are different axes, and this is one of the places an interviewer will check whether you have separated them. Adam's coordinate-wise normalization changes the algorithm's **implicit bias** — which solution it drifts toward when many solutions fit the data equally well — and that bias is measurably worse on vision benchmarks than plain stochastic gradient descent's max-margin drift (Q147, §31.2). The belief is tempting because the training curve is the thing you stare at all day and the test number arrives at the end, once the decision is already made. ▸ **A faster descent into a worse basin is still a faster descent.** The correct framing is that Adam buys robustness to wildly unequal gradient scales, which transformers need and convolutional networks largely do not.

**29. What is natural gradient and how does K-FAC approximate it?**
Natural gradient is $`F^{-1}g`$ with $`F`$ the Fisher information — steepest descent in distribution space rather than parameter space, hence invariant to reparameterization. K-FAC approximates $`F`$ per layer as a Kronecker product $`A\otimes G`$ of input and gradient covariances, making the inverse cheap. (§4.7)

**30. What's the point of SAM?**
Minimize $`\max_{\|\epsilon\|\le\rho}\mathcal{L}(\theta+\epsilon)`$ — the worst loss in a neighbourhood, so the solution is flat. Implemented as two forward-backward passes: ascend to the worst nearby point, take the gradient there, apply it at the original point. ~2× cost, ~1% ImageNet gain. (§5.7)

### Saying it in plain English — optimization

*Optimization questions reward a mechanism plus a consequence. Here are six with the mechanism made physical.*

**Q21 — Adam versus SGD, in one breath.**
SGD gives every parameter the same step size. That is fine when all the parameters live at roughly the same scale. Adam gives every parameter its *own* step size, by keeping a running record of how large that parameter's gradients have typically been and dividing by it. The effect is a per-parameter automatic gain control: a parameter whose gradients are tiny gets amplified, one whose gradients are huge gets damped.

Why does that matter so much for transformers and so little for a convolutional network? Because in a transformer the scales are wildly unequal. The embedding row for a rare token receives a gradient on the handful of batches that happen to contain it and nothing at all otherwise; a normalization gain receives a gradient every single step. No single learning rate can serve both. In a vision model the scales are far more uniform, and there SGD's plain unnormalized step happens to bias the solution toward the large-margin kind of answer that generalizes slightly better.

So: **Adam because it is robust to scale, SGD because its bias is nicer** — and for transformers, robustness wins outright, because a model that will not train at all has no generalization to speak of.

**Q22 — Adam's bias correction.**
Adam's running averages have to start somewhere, and they start at zero. So on step one the average is 99.9% "the zero I invented" and 0.1% "the gradient I actually measured." It is not a noisy estimate of the gradient — it is an estimate that has been dragged almost all the way to zero by a fictitious initial value.

Bias correction is the arithmetic that removes the fiction. You know exactly how much weight the initial zero still carries, because it decays at a known rate, so you divide it out. What happens if you skip it is asymmetric and instructive: the numerator is too small, which alone would be harmless, but the *denominator* is too small too — and the denominator sits under a square root and is doing the dividing. Too-small denominator means enormous steps. For the second-moment decay rate everyone uses, that distortion persists for roughly a thousand steps, which is long enough to destroy a model before training has properly begun.

This is also the honest origin of warmup. Warmup is bias correction performed by hand: people found empirically that early steps had to be small, years before anyone wrote down why.

**Q23 — Adam plus L2 versus AdamW.**
Weight decay is meant to be a uniform frictional force: every weight pulled toward zero by the same fraction each step. The traditional way to implement it was to add that pull into the gradient. Under plain SGD, adding it to the gradient and applying it as a shrink are mathematically identical, so nobody noticed the distinction for decades.

Under Adam they are not identical, because Adam divides everything in the gradient by that parameter's typical gradient magnitude — and the decay term is now *in* the gradient, so it gets divided too. A parameter with large historical gradients divides its decay away almost entirely; a parameter with small ones gets crushed by it. That is precisely backwards: the weights you most wanted to restrain are the ones that escape restraint.

AdamW takes the decay out of the gradient and applies it as a separate shrink after the adaptive step. Friction becomes friction again. It is a two-line change worth a real accuracy gain, and it is why the modern default is AdamW rather than Adam.

**Q25 — The Edge of Stability.**
Classical theory says: if the landscape's curvature is some value, then step sizes above two divided by that value will bounce you out of the bowl instead of into it. You would therefore expect training to settle safely below the line. It does not.

What actually happens is that the network *drifts into sharper and sharper regions* until the curvature rises to meet the instability threshold set by your learning rate — and then it stays there, hovering on the edge, with the loss decreasing overall while jittering up and down from step to step. Training is not converging into a bowl; it is surfing the rim.

▸ **The cause and effect are the reverse of what the textbook implies.** Curvature is not a fixed property of the problem that you must choose a learning rate to respect. **The learning rate you choose determines the curvature you end up in.** That is one concrete, measurable mechanism behind the folklore that large learning rates find flatter minima — and it also explains why a learning-rate decay at the end of training is doing something real: lowering the rate lets the model settle into a region it was previously too jumpy to enter.

**Q28 — Why SGD generalizes better than full-batch descent.**
Every minibatch produces a slightly wrong gradient, and that wrongness acts as a persistent jiggle — like shaking a tray of marbles. A marble sitting in a narrow, steep-sided dimple gets shaken out. A marble in a broad shallow dish stays put. Over time the marbles concentrate in wide basins, not because wide basins are deeper, but because they are **bigger targets and harder to be shaken out of**.

Two checkable predictions follow. First, the strength of the shaking scales as learning rate divided by batch size — so if you increase the batch and do not increase the learning rate, you have quietly turned the shaking down and will settle into the first narrow dimple you meet. That is exactly the observed large-batch generalization gap, and it is why the standard remedy is to scale the learning rate with the batch. Second, in high dimensions "wide" beats "narrow" by an absurd margin: a basin twice as wide in each of a thousand directions is larger by a factor of two to the thousandth power, so the preference is not a mild one.

**Q29 — Natural gradient, and what K-FAC approximates.**
Ordinary gradient descent measures distance in *parameter* space: "move each number by a small amount." But the thing you care about is the *function*, and equal-sized moves in parameter space can produce wildly unequal changes in behaviour. Nudge a softmax temperature and the model's outputs transform completely; nudge one weight buried in an over-parameterized layer and nothing observable happens at all.

Natural gradient replaces the ruler. It measures step size by how much the model's predicted *distribution* moved, not by how much the numbers moved. Two things follow immediately: the algorithm behaves identically no matter how you happened to parameterize the model — rescale a layer and it compensates automatically — and it takes confident strides in directions where the function is insensitive while stepping carefully where it is twitchy.

The catch is cost. The correcting matrix has one row and column per parameter, so for a billion parameters it has $`10^{18}`$ entries, and you need to invert it. K-FAC's trick is to assume that within a layer, the matrix factors into "something about that layer's inputs" times "something about the gradients arriving at its outputs" — which amounts to claiming those two are statistically independent. They are not, exactly. But the assumption converts one impossible inverse into two small ones, and the approximation is good enough that the resulting steps are  better than plain ones.

---

## 34.4 Neural networks & training

**31. Derive backpropagation for one layer.**
With $`z=Wa_{\text{prev}}+b`$, $`a=\sigma(z)`$, $`\delta=\frac{\partial\mathcal{L}}{\partial z}`$: $`\delta^{(l)}=\big(W^{(l+1)\top}\delta^{(l+1)}\big)\odot\sigma'(z^{(l)})`$, $`\frac{\partial\mathcal{L}}{\partial W^{(l)}}=\delta^{(l)}a^{(l-1)\top}`$, $`\frac{\partial\mathcal{L}}{\partial b^{(l)}}=\delta^{(l)}`$. (§6.3)

**32. Derive He initialization.** ★
For $`z=Wa`$ with $`n`$ inputs, $`\mathrm{Var}(z)=n\,\mathrm{Var}(W)\mathrm{Var}(a)`$. ReLU zeroes half the units, halving the variance: $`\mathrm{Var}(a)=\frac12\mathrm{Var}(z_{\text{prev}})`$. To keep variance constant, $`n\,\mathrm{Var}(W)\cdot\frac12=1`$, so $`\mathrm{Var}(W)=2/n`$. Xavier uses $`2/(n_{\text{in}}+n_{\text{out}})`$ for symmetric activations. (§6.4)

> **Where this came from.** These two initializers are named using opposite halves of a person's name, which trips up nearly everyone. **Xavier initialization** comes from the 2010 paper by **Xavier Glorot** and Yoshua Bengio — "Xavier" is a first name, and the scheme is also, more consistently, called Glorot initialization. **He initialization** comes from the 2015 paper by **Kaiming He** and colleagues at Microsoft Research Asia — "He" is a surname. The 2015 paper's actual subject was a new activation function (PReLU); the initialization was a supporting result, and it is the part that survived. The same group published ResNet later that same year, which means one team produced both of the ingredients — a variance-preserving initialization and an identity shortcut — that between them made very deep networks trainable at all. Before 2010 the standard advice was to draw weights from a fixed small uniform range regardless of layer width, which quietly guaranteed that wide layers would shrink the signal and narrow ones would amplify it.

**33. Why does the vanishing gradient problem occur, quantitatively?**
The gradient through $`L`$ layers multiplies $`L`$ Jacobians. With sigmoid, $`\max\sigma'=0.25`$, so 10 layers gives $`\le0.25^{10}\approx10^{-6}`$. ReLU (derivative 1 on the positive side) and residual connections (an additive identity path) fix it. (§6.4, §8.2)

**34. Why do ResNets work? Give three explanations.**
(a) Gradient highway — $`\frac{\partial x_L}{\partial x_\ell}=1+\sum\frac{\partial F}{\partial x_\ell}`$, so the leading 1 guarantees flow. (b) Ensemble of $`2^L`$ shallow paths — deleting a block barely hurts. (c) Loss-landscape smoothing. The first is strongest. Note the original problem was *degradation* (higher training error at depth), i.e. optimization, not overfitting. (§8.2)

**35. BatchNorm vs LayerNorm — why do transformers use LN?** ★
BN normalizes over the batch per channel, so it creates a train/eval discrepancy, breaks with small batches, leaks information across samples, and interacts badly with variable sequence lengths and padding. LN normalizes per sample over features — batch-independent, identical in train and eval, length-agnostic. (§7.2–7.3)

**36. What is BatchNorm actually doing, if not fixing internal covariate shift?**
Smoothing the loss landscape (reducing the Lipschitz constants of the loss and its gradient), plus making the layer scale-invariant so the weight norm becomes an inverse effective learning rate. Santurkar et al. disproved the covariate-shift story directly. (§7.1, §7.4)

> **Common misconception.** *"Normalization makes the activations Gaussian."* It fixes two numbers — the mean and the variance — and touches nothing else. A bimodal distribution normalizes to a bimodal distribution with mean 0 and variance 1. A heavy-tailed one normalizes to a heavy-tailed one. And the very next operation is a learned affine, $`\gamma x + \beta`$, whose explicit purpose is to let the network **undo** the standardization if that serves it. The belief is tempting because "normalize" and "normal" share a root, and because every textbook figure of the operation draws a bell curve going in and a bell curve coming out. What normalization actually buys is a *scale guarantee* — and notice that every  benefit in the answer above follows from scale rather than shape: the loss landscape smooths because gradient magnitudes stop swinging, the layer becomes scale-invariant because the division cancels any rescaling of $`W`$, and the weight norm turns into an inverse effective learning rate for the same reason.

**37. Why should you not apply weight decay to LayerNorm gains and biases?** ★
Scale-invariance is what makes weight decay a learning-rate control (§7.4). Norm gains and biases are *not* scale-invariant — they directly set the layer's output magnitude — so decaying them shrinks the function for no benefit.

**38. Explain the coupling between learning rate and weight decay in a normalized network.** ★
With normalization, the function is invariant to $`\|W\|`$, the gradient is orthogonal to $`W`$ (so $`\|W\|`$ grows under SGD), and the angular update scales as $`\eta/\|W\|^2`$. Weight decay keeps $`\|W\|`$ small so the effective LR stays high. At equilibrium $`\eta_{\text{eff}}\propto\sqrt{\eta\lambda}`$ — so only the product matters. (§7.4)

**39. Three ways to understand dropout.**
(a) Ensembling $`2^n`$ weight-sharing subnetworks, approximated at test time by the geometric mean. (b) For linear regression, exactly equivalent to data-dependent $`\ell_2`$. (c) Approximate variational inference — hence MC dropout. Note dropout is absent from LLM pretraining, because with one pass over a huge corpus there is no overfitting to prevent. (§7.5)

> **Where this came from.** Dropout was introduced by **Geoffrey Hinton** and colleagues at Toronto in 2012, and written up at length by Srivastava and co-authors in 2014. Two motivations are recorded. The published one is biological: sexual reproduction breaks up sets of co-adapted genes, forcing each gene to be useful in many different genetic backgrounds rather than only in one finely-tuned combination — and dropout does the same thing to units, since a unit cannot rely on a specific partner being present. The other is an anecdote Hinton has told in talks: he noticed that bank tellers were rotated between positions and was told the reason was to make sustained collusion difficult, since a conspiracy requires the same people to stay in place. Whether the bank story is the true origin or a good explanation found afterwards is not something to state with certainty in an interview — but the co-adaptation idea it illustrates is exactly right, and it is a better one-sentence answer than "it adds noise."

> **Common misconception.** *"Dropout is on at inference — that's what makes the model robust."* It is off. At test time you want one deterministic function, so every unit is present and the scale is corrected so that each unit receives the input magnitude it saw during training (in the standard *inverted dropout* implementation the correction was already applied on the training side, which is why the inference path looks like it does nothing). Forget this and you get the classic bug: evaluation accuracy sitting well below training accuracy for no discernible reason, because nobody called `model.eval()`. The belief is tempting because dropout is described as making the network robust to missing units, and it is natural to assume the robustness is exercised at the moment it matters. The one legitimate exception is **Monte Carlo (MC) dropout**, where you deliberately keep it on and run many forward passes precisely *because* you want a spread of predictions to read as uncertainty — a different technique with a different goal, and citing it as evidence that "dropout runs at inference" reverses the causality.

**40. Why is early stopping equivalent to $`\ell_2`$?**
For a quadratic with eigenvalues $`\lambda_i`$, GD from 0 gives $`\theta_i(t)=(1-(1-\eta\lambda_i)^t)\theta_i^*`$; matching ridge's $`\frac{\lambda_i}{\lambda_i+\lambda}`$ gives $`\lambda\approx1/(\eta t)`$. Training longer = weaker regularization. (§7.5)

**41. Why GELU/SwiGLU over ReLU?**
GELU $`=x\Phi(x)`$ is smooth and non-monotone, giving nonzero gradient for small negatives (no dead units) and better empirical results. SwiGLU adds a multiplicative gate: $`W_3(\mathrm{SiLU}(W_1x)\odot W_2x)`$, with $`d_{\text{ff}}=\frac83d`$ to hold parameters constant. ~1% perplexity gain. (§6.5)

**42. Your training loss is NaN. Walk me through debugging.**
Check for: LR too high (lower it, add warmup); missing gradient clipping; log(0) or division by zero in a custom loss; fp16 overflow (switch to bf16 or check loss scaling); a corrupted batch (log the batch index and inspect); exploding attention logits (add QK-norm); bad initialization. Then bisect: overfit a single batch first — if that fails, the bug is in the model or loss, not the data pipeline. (§6.7, §14.6)

**43. Your model gets 99% train and 60% test accuracy. What do you do?**
That's overfitting: more data or augmentation first, then regularization (weight decay, dropout, early stopping), then reduce capacity. But check first for a *leak* in the other direction — verify the split is grouped correctly and that no preprocessing was fitted on the full dataset.

**44. Your model gets 60% on both train and test.**
Underfitting or a bug. Check that the model can overfit 10 examples; if not, there's a bug (label misalignment, wrong loss reduction, frozen parameters, LR ~0). If it can, increase capacity, train longer, raise the LR, or improve features.

### Saying it in plain English — neural networks and training

*These six carry most of the practical weight in this section. Say them without a single symbol.*

**Q32 — He initialization.**
Think of a signal passing through twenty layers. At each layer you multiply it by a pile of random numbers and add the results up. If the random numbers are slightly too large, the signal grows a little at every layer — and "a little," twenty times compounded, is an explosion. Slightly too small and it fades to nothing. What you want is for each layer to be signal-preserving on average, neither amplifying nor attenuating.

Adding up many independent contributions makes the spread grow in proportion to how many you added, so the weights need to be smaller in proportion to the number of inputs. Divide by the fan-in. That is the classical answer and it dates to 2010.

Then ReLU arrives and deletes every negative activation, throwing away half the energy at each layer. To come out level you have to go in twice as loud. That factor of two is the whole content of He initialization — and it is the difference between a thirty-layer network that trains and a thirty-layer network that outputs zeros. Worth appreciating that a factor of two, compounded over depth, is the entire story: at thirty layers, being wrong by that factor is a discrepancy of a billion.

**Q34 — Why ResNets work.**
Lead with the honest headline, because most candidates get it wrong: **ResNets were not invented to fix overfitting.** The motivating observation was embarrassing — a 56-layer network had worse *training* error than a 20-layer one. Not worse test error. Worse training error, meaning the deeper network could not even reproduce something the shallower one had already achieved, despite being trivially able to express it by having the extra layers do nothing at all. That is an optimization failure, not a capacity failure, and the distinction is the answer.

The fix is to make "do nothing" the default. Each block computes an adjustment and adds it to what came in, so a block that has learned nothing is harmless rather than destructive. Three ways to see why that helps, in decreasing order of strength. Gradients always have a clear path home, because there is a route from the loss to any early layer that passes through no weight matrices at all — so nothing can multiply the signal down to zero. The network behaves like an ensemble of many shallow paths of different lengths, which is why deleting a single trained block from a ResNet barely dents accuracy while deleting one from a plain network destroys it. And the loss surface is visibly smoother when plotted. The gradient-path explanation is the load-bearing one; the other two are supporting evidence.

**Q36 — What BatchNorm is actually doing.**
The original story was that BatchNorm stabilizes the distribution of each layer's inputs while the layers below keep changing. It is a nice story and it appears to be wrong: you can deliberately inject noise *after* the normalization, deliberately wrecking the distributional stability, and the training speedup survives intact.

What it does instead is make the loss surface less erratic. The gradient stops changing so violently as you move, which means a large step is more likely to land roughly where the gradient predicted it would — so you can use a much bigger learning rate without diverging, and that is most of the observed speedup.

There is a second, cleaner effect. After normalization, multiplying a layer's weights by ten does not change its output at all, because the normalization divides the scale straight back out. So the weight's magnitude has stopped controlling the function. It now controls something else: how far a fixed-size gradient step rotates the weight vector. Large weights, small effective step. ▸ **The weight norm quietly becomes an inverse learning rate** — which is the entire setup for the next question, and one of the more  surprising facts in this book.

**Q38 — The learning-rate and weight-decay coupling.**
Follow directly from the last answer. In a normalized layer, only the *direction* of the weight vector affects the function; its length is invisible. And the gradient turns out to always be perpendicular to the weight vector — which has to be true, since a component along the weight would only change its length, and length does nothing.

Perpendicular steps make a vector *longer*, by Pythagoras. So training slowly and inexorably inflates the weights. And as they inflate, a fixed-size step rotates them through a smaller angle: it is the difference between nudging a pencil and nudging a flagpole. The layer's real learning rate is decaying on its own, with no schedule involved.

Which reveals what weight decay is actually for in a normalized network. It is not primarily preventing overfitting — the function does not even depend on the weight norm. It is stopping the inflation, so that the effective learning rate stays usefully high. At the equilibrium where growth and decay balance, only the *product* of learning rate and decay coefficient matters. ▸ **Doubling one and halving the other trains the same model** — so a two-dimensional grid search over these two hyperparameters is spending most of its compute re-measuring the same points.

**Q39 — Three ways to understand dropout.**
First: it is an ensemble you get for free. Each step randomly deletes a fraction of the units, so you are training an astronomically large collection of thinned networks that all share weights, and at test time using every unit at reduced scale approximates averaging their predictions.

Second: for a linear model you can do the algebra exactly, and dropout comes out as a weight penalty scaled by how active each input feature is. It is ridge regression with a data-dependent strength — an actual derivation, not an analogy.

Third: dropout can be read as drawing samples from an approximate posterior over the weights, which is the argument that licenses leaving it switched on at test time and treating the spread of predictions as a measure of uncertainty.

The footnote that separates people who have read about dropout from people who have trained large models: **dropout has essentially vanished from large-model pretraining.** Its job is to prevent memorizing the training set, and when you make a single pass over trillions of tokens there is nothing to memorize. It reappears during fine-tuning on small datasets, where the original problem returns.

**Q40 — Why early stopping is equivalent to a weight penalty.**
Start training from zero. Directions in which the loss is steep get fitted almost immediately — the gradient there is large and the parameter shoots to its target. Directions in which the loss is shallow are still creeping slowly toward their target when you stop. So stopping early leaves precisely the shallow directions small.

Now compare with what a weight penalty does: it shrinks each direction in proportion to how weakly the data constrains it, leaving the well-determined directions almost untouched. Same behaviour, arrived at differently.

Do the algebra on a quadratic and the correspondence is quantitative: **training time acts like one over the regularization strength.** Train twice as long, regularize half as much. This is also why "train longer" and "add more weight decay" so persistently fight each other in a hyperparameter sweep — they are two controls wired to the same underlying quantity, and tuning them independently is tuning one thing twice.

---

## 34.5 Transformers & LLMs

**45. Why divide by $`\sqrt{d_k}`$ in attention?** ★
With unit-variance $`q,k`$, $`\mathrm{Var}(q^\top k)=d_k`$, so scores have SD $`\sqrt{d_k}`$. Large-variance scores saturate the softmax, whose Jacobian $`p_i(\delta_{ij}-p_j)\to0`$ — gradients vanish. Dividing restores unit variance. (§11.2)

> **Where this came from.** Attention was not proposed as an architecture. It was a **patch for a bottleneck in machine translation**. In 2014 **Dzmitry Bahdanau**, Kyunghyun Cho and Yoshua Bengio were working on encoder–decoder translation, in which an entire source sentence had to be squeezed into a single fixed-length vector before decoding began. Long sentences degraded badly, for the obvious reason. Their fix was to let the decoder look back at every source position and take a weighted average, with the weights computed on the fly — an *alignment* mechanism, mapping which output word came from which input word, and one whose weights could be visualized and read like a translator's word-by-word correspondence. Three years later the 2017 transformer paper's contribution was to delete everything else: no recurrence, no convolution, attention alone. Its title, *Attention Is All You Need*, is widely reported by the authors to be a nod to the Beatles' "All You Need Is Love," and it started a durable and mostly regrettable convention of "X Is All You Need" paper titles.

**46. Why multiple heads instead of one wide one?**
One softmax gives one attention distribution per query; the model must commit to one weighting. $`h`$ heads let a token simultaneously attend to its syntactic head, its antecedent, and the previous token. Also a rank argument: a sum of $`h`$ low-rank-driven operations is more expressive than one. (§11.3)

> **Common misconception.** *"Attention weights show which words the model considers important."* They show where a head **read from**, which is not the same as what influenced the output. Three independent reasons this breaks. What a head *writes* is governed by its output–value (OV) circuit, so a token can be attended to heavily and contribute almost nothing (Q153). Attention composes across heads and layers, so any single map is one factor in a long product. And the residual stream carries information forward without passing through attention at all, so a token can matter enormously while never being attended to. The belief is tempting because attention maps are readable, and on short sentences they often do line up with human intuition — which makes them an excellent hypothesis generator and a poor explanation. ▸ **The causal test is activation patching (Q154), not a heatmap.** Being able to say why an attention visualization is evidence of correlation rather than mechanism is one of the cheapest ways to sound senior on this topic.

**47. Why separate Q, K, and V projections?**
$`Q=K`$ forces a symmetric attention matrix and maximal self-attention. Separating them allows asymmetric relations. $`V=K`$ would force the matching key to equal the transmitted content; separating lets a token be *found* by one property and *contribute* another. (§11.1)

**48. Give the parameter count and FLOP formula for a transformer.** ★
$`4d^2`$ (attention) $`+8d^2`$ (FFN) $`=12d^2`$ per layer, so $`N\approx12Ld^2`$ non-embedding. Training FLOPs $`C\approx6ND`$. Check: GPT-3, $`L=96`$, $`d=12288`$ → 174B. (§11.7)

**49. Is attention the compute bottleneck?**
Not usually. Attention's share is $`\frac{4Td}{24d^2+4Td}`$, hitting 50% at $`T=6d`$ — about 25k tokens for $`d=4096`$. Below that, the FFN and projections dominate. The real bottleneck at inference is the **KV cache**, which is memory. (§11.7, §12.7)

**50. Explain RoPE and why it's better than learned absolute embeddings.** ★
Rotate $`q`$ and $`k`$ by angle $`m\theta_i`$ in $`d/2`$ 2-D planes. Then $`\langle R_mq, R_nk\rangle = q^\top R_{n-m}k`$ — the score depends only on relative position, achieved with purely absolute operations. No $`T\times T`$ bias matrix, KV-cache compatible, and extendable by rescaling the base. Learned absolute embeddings have a hard length limit and don't extrapolate. (§12.4)

**51. How would you extend a model's context from 4k to 32k?**
Options: Position Interpolation (scale positions down — crowds local resolution, needs fine-tuning); NTK-aware base scaling (increase $`\theta_{\text{base}}`$; often works without fine-tuning); **YaRN** (per-dimension: interpolate low frequencies, extrapolate high, plus attention temperature — current best). Then fine-tune on long documents, and verify with RULER, not just needle-in-a-haystack. (§12.4, §12.8)

**52. What does FlashAttention do?** ★
It recognizes attention is *memory-bound*, not compute-bound. It tiles Q, K, V into SRAM-sized blocks and uses an **online softmax** (running max and sum with rescaling) to compute the exact result without ever materializing the $`T\times T`$ matrix, recomputing it in the backward pass instead of storing it. Memory $`O(T^2)\to O(T)`$, 2–4× faster, **numerically exact**.

**53. Compute the KV cache size for a 70B model at 4k context.**
$`2\times L\times T\times h_{kv}\times d_{\text{head}}\times`$ bytes. With $`L=80`$, $`d_{\text{head}}=128`$, bf16, and **full MHA** ($`h_{kv}=64`$): $`2\cdot80\cdot4096\cdot64\cdot128\cdot2=10.7`$ GB **per sequence** — 343 GB at batch 32, more than the weights. With **GQA** ($`h_{kv}=8`$, which is what LLaMA-2-70B actually ships) it's 1.34 GB. Quoting both, and explaining that the second is why the first never gets deployed, is the complete answer. (§12.7)

**54. MHA vs MQA vs GQA vs MLA.**
MHA: $`h`$ K/V heads. MQA: 1 shared K/V head — maximal saving, noticeable quality loss. GQA: $`g`$ groups (typically 8) — 8× saving at near-MHA quality, the current default. MLA: compress K/V into a low-rank latent and cache only that — >90% reduction with quality better than GQA at matched cache. (§12.7)

**55. Why did decoder-only beat encoder–decoder for LLMs?**
Training-signal density (every token is a target vs BERT's 15%), architectural simplicity, natural in-context learning, and the fact that splitting parameters into two pools is wasteful at a fixed budget for open-ended generation. Encoder–decoder still wins with a fixed repeatedly-attended source (translation, ASR). (§11.6, §13.6)

**56. Walk through BPE.**
Start with characters (or the 256 bytes), count adjacent pairs, merge the most frequent, record the merge, repeat to the target vocabulary. Encoding applies the merges in learned order. Byte-level BPE means no `<UNK>` ever. WordPiece merges by $`\frac{c(ab)}{c(a)c(b)}`$ (a PMI-like criterion) rather than raw frequency; Unigram runs backwards, pruning from a large candidate set by likelihood loss. (§10.4)

> **Where this came from.** Byte-Pair Encoding was published in 1994 by **Philip Gage** in *The C Users Journal*, as a **data compression algorithm**. It had nothing to do with language: the idea was that in any byte stream, some adjacent pairs recur often, so you repeatedly replace the most frequent pair with an unused byte value and record the substitution. Simple, fast, and thoroughly ordinary as compression schemes go. In 2016 **Rico Sennrich, Barry Haddow and Alexandra Birch** repurposed it for neural machine translation to solve a different problem — the vocabulary of a language is unbounded, so any fixed word list produces a flood of unknown tokens — and observed that a compression algorithm run over text produces exactly the units you want: whole words where they are common, morpheme-like fragments where they are not. Every frontier language model's tokenizer is a descendant of a 1994 file-compression utility, which is also the reason for its most famous weaknesses (see Q57): a scheme optimized to make text *short* has no reason to make it *countable*.

**57. Why are LLMs bad at counting letters in a word?** ★
Tokenization. "strawberry" is ~3 tokens, not 10 characters — the model can't *see* the letters, only a memorized association. Same root cause as poor arithmetic (inconsistent digit grouping), rhyming, and reversal. (§10.4)

**58. Derive word2vec's negative sampling objective.**
Replace the $`|V|`$-way softmax with binary logistic classification: $`-\log\sigma(u_o^\top v_c)-\sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\log\sigma(-u_{w_i}^\top v_c)`$, with $`P_n\propto U(w)^{3/4}`$. Levy & Goldberg showed this implicitly factorizes the shifted PMI matrix. (§10.3)

**59. What is in-context learning and why does it happen?** ★
Task performance from prompt examples with no weight update. Mechanistically explained by **induction heads**: a previous-token head writes token $`t-1`$'s identity into position $`t`$; a later head queries with the current token to find earlier occurrences and copies what followed. Requires ≥2 layers. Evidence: the ability and the heads appear in the same narrow training window, and ablating the heads destroys the ability. Note that randomizing the demonstration *labels* barely hurts — the demonstrations mainly convey format and label space. (§13.3)

**60. Nucleus sampling vs top-$`k`$ — why did top-$`p`$ win?**
Top-$`k`$ uses a fixed count, but the number of plausible next tokens varies enormously by context. Top-$`p`$ takes the smallest set summing to $`p`$, adapting to the distribution's entropy. (§13.4)

**61. Why is beam search wrong for open-ended generation?**
Higher sequence probability correlates with worse human-judged quality past a point — the likelihood trap. The highest-probability continuation of most prompts is degenerate repetition. Beam search is right for translation and summarization, where the output is highly constrained by the input. (§13.4)

**62. What are the pitfalls of perplexity?**
Tokenizer-dependent (compare bits-per-byte instead), data-dependent, invalidated by contamination, and only loosely related to usefulness after post-training — RLHF typically *raises* perplexity while improving preference scores. (§13.5)

**63. Derive the Chinchilla scaling result.** ★
Fit $`L=E+AN^{-\alpha}+BD^{-\beta}`$, minimize subject to $`C=6ND`$. Substituting $`D=C/6N`$ and setting $`dL/dN=0`$ gives $`N\propto C^{\beta/(\alpha+\beta)}`$, $`D\propto C^{\alpha/(\alpha+\beta)}`$. With $`\alpha=0.34,\beta=0.28`$: exponents 0.452 and 0.548 — roughly equal scaling, ~20 tokens per parameter. (§15.2)

**64. Why is LLaMA-3-8B trained on 15T tokens when Chinchilla says 160B?**
Chinchilla optimizes *training* compute only. Inference cost is linear in $`N`$ and paid forever, so accounting for deployment shifts the optimum strongly toward smaller models trained far longer. A model too large to serve is worthless. (§15.2)

**65. Is emergence real?** ★
Partly. Schaeffer et al. showed that discontinuous metrics manufacture discontinuities: if per-token accuracy $`p`$ improves smoothly and the metric is exact match over $`k`$ tokens, the measured score $`p^k`$ looks like a phase transition. Continuous metrics on the same outputs show smooth improvement. But some *mechanistic* transitions (induction-head formation) do appear  sharp. Underlying capability: smooth. User-visible usefulness: can jump. (§15.4)

**66. What is $`\mu`$P and why does it matter?**
A parameterization whose per-layer init and LR scalings keep activations and updates $`\Theta(1)`$ at any width, so optimal hyperparameters become width-independent and transfer from a small proxy to a large model. It also keeps the model in the feature-learning rather than lazy/NTK regime. (§15.3, §30.2)

**67. Explain Mixture of Experts and load balancing.**
Replace the FFN with $`E`$ experts and a top-$`k`$ router; total parameters grow while active FLOPs don't. Left alone the router collapses (rich-get-richer), so add $`\mathcal{L}_{\text{aux}}=\alpha E\sum_i f_iP_i`$, minimized at uniform routing. Alternatives: expert-choice routing, capacity limits, bias-based balancing. (§11.8)

**68. Why do transformers need chain of thought for hard problems?** ★
A fixed-depth transformer has bounded serial computation per forward pass (roughly $`\mathsf{TC}^0`$). Emitting intermediate tokens provides an external serial scratchpad — each token is another forward pass conditioned on the last. **CoT buys depth with tokens.** (§11.9, §16.7)

> **Common misconception.** *"Chain of thought (CoT) shows you how the model reached its answer."* It shows you a computation the model performed, and that computation does real work — the answer above is precisely the argument that the emitted tokens buy serial depth the architecture does not otherwise have. What it does not guarantee is **faithfulness**: the stated reasoning need not be the reasoning that determined the conclusion, and a model can produce a fluent, internally valid derivation that rationalizes an answer arrived at some other way. The belief is tempting because a legible artifact feels like a transparent one, and because the text is in your own language and reads like a confession. The framing that lands well in a room: **chain of thought is a capability intervention first and an interpretability artifact a distant second.** Treating the trace as a faithful log is the fastest way to sound naive about it; noting that faithfulness is an open empirical question with its own measurement literature is the opposite.

### Saying it in plain English — transformers and large language models

*This is the section most interviews spend the most time in. Six of the hardest, spoken.*

**Q45 — Why divide by the square root of the head dimension.**
An attention score is a dot product: multiply matching entries of two vectors and add up the results. Add up more terms and the total naturally swings wider — sum sixty-four random numbers and the result is typically about eight times larger in magnitude than any one of them. So with a wide head, the raw scores come out enormous.

Then they go into a softmax, whose job is to turn scores into a set of weights that sum to one. Feed a softmax numbers that are far apart and it does not produce a soft distribution at all — it produces a hard pick: 0.9999 on one token, essentially nothing everywhere else. And a saturated softmax is *flat*: nudging the scores changes the output by almost nothing, so almost no gradient comes back. The layer has locked itself into an arbitrary choice, at initialization, before it knew anything, and cannot learn its way out.

Dividing by the square root of the head dimension undoes exactly the widening the summation caused, putting the scores back at a scale where the softmax is still soft and still differentiable. It is a variance-control constant, not a modelling decision — and the giveaway is that the correction is the square root, which is the signature of "I summed up this many independent things."

**Q50 — RoPE, and why it beats learned absolute positions.**
Attention on its own has no sense of order; it sees an unordered bag of vectors. You have to tell it where each token sits. The old approach was to add a learned "this is position 7" vector to each token's embedding. That works up to the length you trained on and then falls off a cliff, because position 5,000 is a vector that was never trained and means nothing.

RoPE instead **rotates** each token's query and key vectors by an angle proportional to its position. Take the coordinates in pairs, treat each pair as a little clock hand, and turn it. Now when you compare a query at position 12 with a key at position 5, the two rotations partially cancel, and what survives depends only on the difference — seven positions apart. You have obtained *relative* position for free, out of an operation applied to each token completely independently.

That last clause is why it won, and it is the part candidates usually miss. Because the position is applied per token rather than as a pairwise bias table, a token's key can be computed once and cached forever, which is exactly what serving requires. A relative-position scheme implemented as a bias matrix cannot do that cleanly, and also costs memory that grows with the square of the length.

The extension trick then falls out naturally: the rotation speeds come from a base frequency, so slowing all the clocks down lets the same trained model spread itself over a longer document without ever seeing an angle it does not recognize.

**Q52 — What FlashAttention does.**
The surprise at the heart of it: attention was never compute-limited. On a modern accelerator, performing arithmetic is roughly two orders of magnitude cheaper than fetching the numbers from main memory. The textbook implementation writes the full token-by-token score matrix out to memory, reads it back to apply softmax, writes it again, and reads it again to multiply by the values. At eight thousand tokens that matrix holds sixty-four million numbers, moved four times. The arithmetic is trivial; the moving is the entire cost.

FlashAttention never builds the matrix. It cuts the problem into tiles small enough to fit in the chip's small, extremely fast scratchpad memory, and computes the softmax incrementally: keep a running maximum and a running total, and whenever a larger value appears, rescale what you have accumulated so far. It is the same technique you would use to compute an average of a stream without storing the stream. For the backward pass it recomputes the tiles rather than storing them — deliberately doing arithmetic twice to avoid moving data once.

Two things to say aloud: memory goes from quadratic to linear in sequence length, and **the result is bit-for-bit identical** to standard attention. This is not an approximation, not a sparsity pattern, not a quality trade. Getting an exact answer several times faster purely by rearranging memory traffic is why it was adopted essentially universally within a year.

**Q59 — In-context learning and induction heads.**
Put three examples of a task in the prompt and the model does the task — with no weight update, nothing saved, nothing trained. Where does that ability come from?

At least partly from a two-layer circuit the model constructs spontaneously during pretraining. A head in an early layer does something dull: it writes into each position a small note recording "the token immediately before me was X." A head in a later layer then poses a query of the form "I am currently at token X — find me an earlier position whose note says *its* predecessor was X" — and copies whatever came *after* that earlier occurrence.

The net effect: **if the pattern A-then-B appeared earlier in the context, then on seeing A again, predict B.** That is a general-purpose pattern copier, and crucially it works on patterns invented in the prompt thirty tokens ago, which is exactly what learning from examples looks like from the outside.

The evidence this is not merely a nice story: the circuit forms during a narrow window of training, the in-context learning ability appears in the same window, and surgically deleting the heads destroys the ability. It also requires at least two layers, because it is  two sequential steps — write the note, then read it — which is a testable structural prediction that holds.

One deflating result worth having ready, because interviewers like it: scramble the *labels* in your demonstrations, so the examples are actively wrong, and performance barely drops. The demonstrations are mostly communicating the output format and which labels are available, not teaching the mapping.

**Q63 — Chinchilla, and what it corrected.**
You have a fixed compute budget. You can spend it on a large model that sees relatively little data, or a smaller model that sees far more. Which choice loses less?

Fit a simple curve: loss falls off as a power of model size, plus a power of data size, plus an irreducible floor. Then ask which split of the budget minimizes the total, given that compute is roughly proportional to model size times data size. The two exponents turn out to be nearly equal — and when they are equal, the optimum is balanced. **Double your compute and you should double both the model and the data**, which works out to somewhere around twenty tokens of training data per parameter.

What made it a landmark was how far off practice had been. The prevailing models were far too large for the amount of data they had been fed — DeepMind demonstrated it directly by training a much smaller model on far more data and beating a model several times its size on essentially every benchmark. The result reshaped what everybody trained for years afterwards, right up until people noticed the objection in the next question: this optimizes *training* compute only, and inference cost is paid forever.

**Q68 — Why chain of thought helps.**
A transformer's forward pass has a fixed number of layers. Whatever you ask it, it performs the same fixed amount of sequential work before it must commit to a token. Some problems  require more sequential steps than that — the answer at step 40 depends on the answer at step 39, and no amount of *width* can compress a 40-step chain into twelve layers, because width buys parallel work and the problem needs serial work.

Writing intermediate steps changes the situation completely, because every emitted token starts a brand-new forward pass that can read everything written so far. The text on the page has become external memory, and each token is an extra layer of sequential depth that the architecture did not have.

▸ **Chain of thought buys depth with tokens.** That framing also predicts the shape of the phenomenon correctly: it barely helps on lookup or single-step tasks, where the depth was never the constraint, and helps enormously on long dependency chains like multi-step arithmetic and proof search. And it implies something worth stating — the scratchpad is doing computational work, not narrating work done elsewhere, which is why forcing a model to answer immediately and then explain produces worse answers than letting it think first.

---

## 34.6 Post-training, RAG, inference

**69. Derive DPO.** ★★
(1) The KL-regularized RLHF objective's optimum is $`\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)e^{r(x,y)/\beta}`$ — rewrite the objective as $`-\beta[\mathrm{KL}(\pi\|\pi^*)-\log Z]`$ and note KL is minimized at 0. (2) Invert: $`r=\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z(x)`$. (3) Substitute into Bradley–Terry, which depends only on the *difference* of rewards, so $`\beta\log Z(x)`$ cancels. Result: $`\mathcal{L}=-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)`$. (§16.5)

> **Where this came from.** The preference model at the centre of this derivation is the **Bradley–Terry** model, published by Ralph Bradley and Milton Terry in 1952 for the statistics of *paired comparison experiments* — the setting where you cannot measure a quantity directly but you can ask judges which of two items they prefer, as in taste tests of food products. Its assumption is that each item has a hidden strength and the probability of preferring one is a function of the difference of strengths, which is exactly the structure that makes the intractable normalizing term cancel in DPO. Essentially the same model had been written down by **Ernst Zermelo** in 1929 to rate chess players from tournament results, and it is the mathematical ancestor of the Elo rating system. So the machinery underneath modern preference tuning is a chess-rating model, applied to pairs of language-model outputs, with human annotators in the role of tournament results. The other half of the lineage is **Christiano and colleagues in 2017**, who showed that a reinforcement learner could be steered by human preference comparisons alone — training a simulated robot to perform a backflip from a few hundred human judgements of which of two clips looked more backflip-like, a task nobody could write a reward function for.

**70. Where is DPO weaker than PPO?**
DPO is offline: its constraint binds only on responses in the dataset, so it can push probability mass onto unseen outputs — the observed pathology is that both $`y_w`$ and $`y_l`$ decrease in likelihood. Fixes: iterative/online DPO, plus an SFT term on $`y_w`$. Well-tuned online PPO still edges it out on hard tasks. (§16.5)

**71. Why is the KL penalty in RLHF essential?**
It keeps the policy inside the region where the reward model is valid, preserves fluency and pretrained knowledge, and prevents collapse onto a few reward-maximizing degenerate outputs. Without it the policy exploits the reward model. (§16.4)

**72. What is GRPO and why does it work?**
Replace the value network with the mean reward over a group of $`G`$ completions for the same prompt. Valid because any action-independent baseline leaves the policy gradient unbiased — the group mean is a Monte-Carlo estimate of $`V(x)`$. Trades a second large model for more sampling. (§16.6, §27.4)

**73. What is RLVR and why did it change reasoning models?**
Reinforcement learning from *verifiable* rewards — check the math answer or run the unit tests instead of using a learned reward model. It eliminates reward hacking at the source, because you can't fool a compiler. Limited to verifiable domains, and test-exploitation becomes the new failure mode. (§16.6)

> **Common misconception.** *"Reinforcement learning from human feedback (RLHF) teaches the model new facts."* Post-training reshapes which of the model's existing behaviours surface; the knowledge arrived during pretraining. The evidence is sitting in Q62: RLHF typically **raises** perplexity on held-out text while improving human preference scores. A model that has become a worse predictor of raw text and a better assistant has not acquired content — it has reallocated probability mass. The belief is tempting because an aligned model visibly *does things* the base model would not do, and a change in behaviour reads as a change in knowledge. The practical consequence is a hiring-signal-grade one: if the complaint is "the model doesn't know about our product," the levers are pretraining data, continued pretraining, or retrieval — not a preference dataset of thirty thousand comparisons, which is roughly the wrong tool by three orders of magnitude.

**74. Explain speculative decoding and why it's lossless.** ★
A small draft model proposes $`\gamma`$ tokens; the target scores all of them in one forward pass (cheap, since decode is memory-bound). Accept token $`i`$ with probability $`\min(1, p_i/q_i)`$; on rejection, resample from the normalized residual $`\max(0,p-q)`$. That modified rejection sampling makes the output distribution **exactly** $`p`$. Expected tokens per round $`=\frac{1-\alpha^{\gamma+1}}{1-\alpha}`$; typical 2–3× speedup. (§17.3)

**75. Why is LLM decoding memory-bandwidth-bound?** ★
At batch 1, producing one token requires reading every parameter. A 70B bf16 model is 140 GB; at 3.35 TB/s that's 42 ms/token = 24 tok/s regardless of compute. Consequence: batching is nearly free, and quantization buys near-linear speedup. (§17.1)

> **Common misconception.** *"Temperature 0 makes a language model deterministic."* Greedy decoding is deterministic in exact arithmetic. The serving stack is not doing exact arithmetic. Floating-point addition is not associative, so the order in which a reduction is summed changes the final bits — and that order depends on the batch size you happened to land in, which fused kernel the scheduler selected, how tensors were sharded across devices, and sometimes which generation of hardware served the request. Two nearly-tied logits then resolve differently, and a single divergent token changes everything that follows it. The belief is tempting because the *sampling* nondeterminism  did disappear, so whatever variation remains looks like a defect rather than an inherent property. Two practical rules follow: pin batch size and kernel configuration when you need reproducibility, and never build an evaluation whose correctness depends on byte-identical generations.

**76. Why is quantizing activations harder than weights?**
Transformers develop systematic outlier channels with magnitudes 10–100× the rest, consistently in the same dimensions. A per-tensor scale set by them crushes everything else. Fixes: LLM.int8() (keep outliers in fp16), SmoothQuant (migrate difficulty to weights), AWQ (protect high-activation channels). (§17.4)

**77. Explain LoRA. What rank and which layers?**
$`W'=W_0+\frac{\alpha}{r}BA`$ with $`B=0`$ at init, so training starts exactly at the pretrained model. Apply to **all** linear layers (not just Q,V) — that matters more than the rank. $`r=8`$–64; higher for new knowledge, lower for style. Merges into $`W`$ at inference for zero added latency. (§17.7)

**78. Why the $`\tau^2`$ in distillation loss?**
The gradient of the soft-target term scales as $`1/\tau^2`$, so multiplying by $`\tau^2`$ keeps the relative weight of soft and hard targets constant as $`\tau`$ varies. High $`\tau`$ matters because a confident teacher at $`\tau=1`$ carries no more information than the label — the "dark knowledge" is in the ratios among small probabilities. (§17.6)

**79. Design a RAG system and tell me its failure modes.**
Chunk structurally (index small, expand to parent), embed with an instruction-prefixed model, hybrid BM25 + dense retrieval fused by **RRF** ($`\sum 1/(60+\text{rank})`$), cross-encoder rerank top-100 → top-10, place the best material at the start and end of the context, instruct explicit citation and "I don't know." Failures: document never retrieved (chunking/vocabulary — add BM25), retrieved but ignored (buried in the middle), model overriding context with parametric knowledge, multi-hop (needs decomposition), aggregation queries (wrong tool — use SQL). **Evaluate retrieval and generation separately.** (§18.3–18.4)

> **Where this came from.** The lexical half of that hybrid is older than almost anything else in this book that is still in production. **BM25** — the "BM" stands for *Best Matching* — came out of the **Okapi** information retrieval system built at City University London in the 1980s and 1990s, developed principally by **Stephen Robertson** and **Karen Spärck Jones**; the inverse-document-frequency weighting at its heart is Spärck Jones's, from 1972. It is a hand-designed scoring function with a handful of tuned constants and no learning at all, and it remains stubbornly difficult for dense embedding models to beat on queries containing rare exact terms — product codes, error strings, surnames — which is precisely why the modern recommendation is to run both and fuse the rankings rather than to replace one with the other. The term **RAG** itself is much newer: Lewis and colleagues coined it in 2020 for a system that fine-tuned a generator jointly with a retriever. The irony worth noting is that the architecture that made the name famous is not what people build now — today "RAG" almost always means an off-the-shelf retriever bolted to a frozen model, which the original paper would have considered the baseline.

**80. Long context or RAG?**
Both. Retrieval reduces the haystack; long context reasons over what's left. Feeding 1M tokens when 5k suffice costs 200× more for a quality *decrease* ("lost in the middle"). (§18.5)

**81. Why do agents fail?** ★
Compounding error: $`p^n`$. At 95% per-step reliability, 20 steps gives 36%. Improving per-step reliability from 95% to 99% takes you to 82% — worth more than any architectural change. Plus context rot, looping, and prompt injection from tool outputs. (§18.7)

### Saying it in plain English — post-training, retrieval, and inference

*The serving questions in particular are ones where a candidate who has actually deployed something sounds different from one who has read about it. Six, spoken.*

**Q69 — DPO, with the algebra taken out.**
Standard reinforcement learning from human feedback is a three-stage machine. Collect human comparisons of model outputs. Train a separate reward model to imitate those preferences. Then use reinforcement learning to push the language model toward high reward, with a penalty that stops it wandering too far from where it started. Three large models resident in memory, an unstable reinforcement learning loop, and a great many knobs.

DPO's observation is that the middle model is **redundant**. Work out what the *perfect* answer to that reinforcement learning problem looks like, and it has a clean closed form: the ideal policy is the original model, reweighted by the exponential of the reward. Now read that equation backwards. Given any policy, you can ask what reward it would be optimal for — and the answer is simply how much more likely that policy makes a response compared to the original model. **The reward is recoverable from the policy.**

So substitute it in. The preference model only ever compares two responses, which means it only ever involves the *difference* of two rewards — and the awkward normalizing term, the thing that made the closed form intractable to use directly, is identical for both responses and cancels exactly. What survives is an ordinary supervised loss on pairs: raise the log-probability of the preferred answer relative to the reference model, lower it for the rejected one, and squash the gap through a sigmoid.

The sentence to end on: **the reward model was hiding inside the language model all along.**

**Q72 — GRPO.**
Policy gradient methods need a baseline — a "what did I expect to score here?" number — because without one, every action taken during a successful episode gets reinforced, including the bad ones that happened to be along for the ride. The signal drowns in variance. The standard fix is to train a second network, typically the same size as the model itself, whose only job is to predict that baseline.

GRPO deletes it. For each prompt, sample eight or sixteen answers, score all of them, and use the group's mean score as the baseline. An answer that beat its siblings is pushed up; one that lost is pushed down.

That is legal because of a general fact worth being able to state: you may subtract *anything that does not depend on the action you took* without biasing the gradient. The group mean depends only on the prompt, so it qualifies — and it is a direct sampled estimate of exactly the quantity the value network was straining to predict.

The trade is explicit and worth naming: **you replace a large second model with more sampling.** On tasks where scoring is cheap and objective — mathematics with a checkable answer, code with unit tests — that is a very good trade, which is why it became the standard recipe for reasoning models.

**Q74 — Speculative decoding, and why it is lossless.**
Generating one token from a large model costs a full pass over every weight. *Checking* a token is nearly free by comparison, because a single pass can score many positions simultaneously. That asymmetry is the entire opportunity.

So: let a small, fast model guess the next few tokens. Then run the large model *once* over the guessed sequence, obtaining its opinion at every position at the same time. Wherever the large model agrees, keep the tokens — several tokens for the price of one pass. At the first disagreement, discard the remainder and continue from there.

The part that sounds too good to be true is the losslessness. If you naively kept tokens wherever the models agreed, you would bias the output toward whatever the small model likes, and the distribution would drift. The fix is a careful accept-reject rule: accept a guessed token with probability equal to the ratio of the two models' probabilities for it, capped at one — and on rejection, sample the replacement not from the large model's raw distribution but from the **leftover**, the part of the large model's distribution that the small model under-supplied. The arithmetic works out so that the tokens you emit are distributed **exactly** as if the large model had generated them alone.

Not approximately. Exactly. Which is why you can switch it on in production without re-running your evaluations — an unusual property for a 2–3× speedup, and the reason this question gets asked.

**Q75 — Why decoding is memory-bandwidth-bound.**
To produce a single token at batch size one, the chip must read every weight in the model exactly once. It does almost nothing with each weight — a couple of arithmetic operations — and moves on. So the clock is set by how fast memory can deliver a hundred and forty gigabytes, not by how fast the arithmetic units can chew. Do the division and it is brutal: tens of milliseconds per token, and no amount of additional compute changes it at all.

Three consequences follow immediately, and reciting them is the real answer:

- **Batching is nearly free.** You read the weights once and serve thirty-two users from that same read, so throughput multiplies while latency barely moves. This is the single most important fact about the economics of serving.
- **Quantization pays off almost linearly.** Halve the bytes, halve the read time, halve the latency — which is why 8-bit and 4-bit inference are everywhere and 8-bit *training* is comparatively rare, since training is  compute-bound.
- **The thing that limits your batch size is not the weights.** It is the **KV cache**, which grows with every concurrent user and every token they have generated, and eventually consumes the memory you wanted to hold the model in.

**Q76 — Why activations are harder to quantize than weights.**
Quantizing means choosing a scale — "the largest number I will represent is this" — and then chopping everything below it into a few hundred levels. It works when the numbers involved are comparable in size.

Trained transformer activations are not. A small number of feature channels routinely carry values a hundred times larger than everything else, and reliably the *same* channels across different inputs, which tells you the model built them deliberately rather than encountering them by accident. Set your scale by those outliers and the ordinary values — where nearly all the information lives — get flattened into two or three distinguishable levels. Set your scale by the ordinary values and the outliers clip, destroying whatever the model was using them for.

Weights do not behave this way; their distribution is comparatively tame. So every practical fix targets the outliers specifically: keep those few channels in high precision and quantize the rest; algebraically push the outlier magnitude out of the activations and into the weights, where it is harmless; or give the weights that receive large activations more careful individual scaling. Naming one of those three is the difference between knowing that the phenomenon exists and knowing the field.

**Q78 — Why the temperature is squared in the distillation loss.**
Distillation trains a small model to match a large model's full output distribution, not just its winning label. The point is that the teacher's second and third choices carry information: a handwritten digit that the teacher scores as "seven with 0.9, one with 0.05, nine with 0.04" is telling the student something true about how sevens, ones and nines relate. The hard label "seven" carries none of that. Hinton's name for it — **dark knowledge** — is exactly right: it is information that is present and invisible.

The problem is that a confident teacher puts 0.9999 on the answer, and the interesting ratios are buried in numbers far too small to influence any gradient. So you soften: divide the logits by a temperature before the softmax, which spreads the distribution and lifts the small probabilities into view.

Now the bookkeeping, which is what the question is really about. Softening also *shrinks the gradient* coming from that term, by roughly the square of the temperature. So if you carefully tune the balance between "match the teacher" and "match the true label" at one temperature and then change the temperature, you have silently changed the balance as well, and your tuning is void. Multiplying by the temperature squared cancels that, so the temperature controls only the thing you meant it to control — how much dark knowledge is visible — and not how loudly that term shouts.

---

## 34.7 Generative models

**82. Compare VAE, GAN, flow, and diffusion.**
VAE: bounded likelihood, fast, blurry, stable. GAN: no likelihood, fast, sharp, mode-collapse-prone, unstable. Flow: exact likelihood, fast, architecturally constrained (invertibility, no dimensionality reduction). Diffusion: bounded likelihood, high quality, mode-covering, stable, slow — and the slowness was fixable by distillation while GAN instability was not. (§19.8)

**83. Why are VAE samples blurry?** ★
(a) A Gaussian likelihood makes reconstruction an MSE, and the MSE-optimal prediction under uncertainty is the *mean* — an average of plausible images. (b) Maximum likelihood minimizes forward KL, which is mode-covering. (c) The ELBO is a loose bound. (§19.3)

> **Common misconception.** *"Blurriness and mode collapse are both just 'the model didn't learn the data well enough.'"* They are opposite failures, and telling them apart is most of Q4 cashed in. Blurriness is **mode-covering**: the model refuses to leave any part of the data unaccounted for, so it spreads probability across the empty region between two  distinct kinds of image, and samples drawn from that region are the averaged, unreal in-between ones. Mode collapse is **mode-seeking**: the model finds one region it can render convincingly and abandons the rest of the distribution outright. Neither is fixed by more capacity or longer training, because neither is a capacity problem — each is a direct consequence of which direction of the Kullback–Leibler (KL) divergence the objective happens to penalize. The belief is tempting because both look like "bad samples" in a grid of pictures, and the diagnosis requires asking what the model failed to *cover* rather than how good any single image looks.

**84. Explain the reparameterization trick and why it's necessary.**
Write $`z=\mu+\sigma\odot\epsilon`$ with $`\epsilon\sim\mathcal{N}(0,I)`$, so the expectation is over a fixed distribution and the gradient passes inside. The alternative (REINFORCE/score-function) is unbiased but has orders of magnitude higher variance. (§19.3)

**85. What is posterior collapse and how do you fix it?**
If the decoder is powerful enough to model $`x`$ alone, the optimum sets $`q(z|x)=p(z)`$, KL$`=0`$, and the latent is unused. Fixes: KL annealing, free bits (don't penalize KL below a floor per dimension), weaker decoder. (§19.3)

**86. Derive the GAN's optimal discriminator and what the generator then minimizes.** ★
Pointwise maximization of $`a\log y+b\log(1-y)`$ gives $`D^*=\frac{p_{\text{data}}}{p_{\text{data}}+p_g}`$. Substituting back yields $`-\log 4+2\,\mathrm{JSD}(p_{\text{data}}\|p_g)`$, minimized when $`p_g=p_{\text{data}}`$. **But** when the supports are disjoint — essentially always in high dimensions — JSD is constant at $`\log 2`$ and its gradient is zero. That's the fundamental instability, and it's why WGAN replaces JSD with the Earth-Mover distance. (§19.4)

> **Where this came from.** **Ian Goodfellow** conceived the adversarial framing in 2014 in Montreal, during an argument at a bar where colleagues were celebrating a graduation; the account he has given publicly is that he went home the same night and had a working implementation before morning. The paper appeared later that year with Bengio and others. What makes the story instructive rather than merely charming is the shape of the idea: the argument was about how to make a generative model without computing a likelihood, and the answer — let a second network judge, and train the two against each other — sidesteps the intractable integral entirely by replacing "how probable is this sample" with "can anyone tell it apart." The replacement distance in the answer above has a far older pedigree: **Gaspard Monge** posed the optimal transport problem in 1781, asking for the cheapest way to move a pile of soil into a new shape for military earthworks, and **Leonid Kantorovich** generalized it in the 1940s while working on the allocation of scarce resources in the Soviet economy — work for which he shared the 1975 Nobel Memorial Prize in Economics. The distance WGAN uses to stabilize image generation is literally a formula for how much it costs to move dirt.

**87. Explain the VQ-VAE straight-through estimator.**
The nearest-neighbour codebook lookup has no gradient, so copy the decoder's gradient directly to the encoder as if quantization were the identity. Biased, and it works. The same trick appears in quantization-aware training and any discrete bottleneck. (§19.3, §17.4)

**88. Derive the diffusion forward-process closed form.** ★
By induction: $`x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon''`$; substituting $`x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}}\epsilon'`$ and combining the two independent Gaussians gives variance $`\alpha_t(1-\bar\alpha_{t-1})+1-\alpha_t=1-\bar\alpha_t`$. So $`q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)`$. **This is what makes training one-shot** — you jump to any noise level without simulating the chain. (§20.2)

> **Where this came from.** Diffusion models were introduced in 2015 by **Jascha Sohl-Dickstein** and colleagues, in a paper titled *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. The idea was borrowed directly from statistical physics: a physical system driven away from equilibrium can be described by a forward process that destroys structure and a reverse process that recovers it, and if the forward steps are small enough, the reverse steps have the same functional form. The 2015 paper had all the essential mathematics — the forward chain, the learned reverse chain, the variational bound — and then **almost nothing happened for five years.** The samples were not competitive, GANs were producing spectacular images, and the field's attention was elsewhere. It took until 2020 and the DDPM paper of Ho, Jain and Abbeel, which changed the parameterization to predicting the noise and simplified the loss weighting, for the same framework to start producing state-of-the-art images. Simultaneously and independently, Song and Ermon had arrived at what turned out to be the same algorithm from the score-matching direction (Q91). A useful thing to notice for your own judgement: the gap between "correct idea, published, freely available" and "idea that works" was half a decade, and the difference was a change of parameterization.

**89. What is the actual diffusion training loss?**
$`\mathcal{L}_{\text{simple}}=\mathbb{E}_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2`$ — plain MSE denoising. It's the VLB with the weighting dropped, which is a worse likelihood objective and a better perceptual one. (§20.4)

**90. Why predict $`\epsilon`$ rather than $`x_0`$? What is $`v`$-prediction?**
At large $`t`$, $`x_t`$ is nearly noise so $`\epsilon`$ is well-scaled while $`x_0`$ is nearly unconstrained (high-variance target). At small $`t`$, the reverse. $`v=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}x_0`$ interpolates smoothly and is preferred for distillation and high resolution. (§20.4)

**91. Show that diffusion and score matching are the same.** ★
$`\nabla_{x_t}\log q(x_t|x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}`$. So $`s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}`$. DDPM and NCSN were developed independently and are the same algorithm. Tweedie's formula adds the third equivalence: the optimal denoiser, the score, and the posterior mean are one object. (§20.6)

**92. Explain classifier-free guidance.** ★
Train one network with the condition randomly dropped ~10% of the time, then extrapolate at sampling: $`\tilde\epsilon=\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)`$. Derivable from classifier guidance by substituting the implicit classifier $`\nabla\log p(y|x)=\nabla\log p(x|y)-\nabla\log p(x)`$. $`w>1`$ pushes beyond the conditional distribution: higher prompt adherence and fidelity, **lower diversity**, and saturation artifacts at large $`w`$. (§20.8)

> **Common misconception.** *"Higher guidance means a better model."* At $`w>1`$, classifier-free guidance samples from a distribution the model was never trained on — a sharpened one, obtained by extrapolating past the conditional prediction rather than toward it. Individual images become more striking and more obedient to the prompt, which is exactly what a person flipping through a grid of samples registers. What that person cannot see, because it is invisible one image at a time, is that **diversity collapses**: the same prompt begins returning the same composition, the same palette, the same pose. Push further and colours blow out and saturation artifacts appear, because the extrapolation has left the region where the network's predictions carry meaning. The belief is tempting because every individual comparison favours more guidance. ▸ **Guidance is a fidelity–diversity dial, not a quality dial** — and volunteering that trade before being asked is a strong signal.

**93. What is DDIM and why does it allow 50 steps instead of 1000?**
A non-Markovian family with the same marginals and the same trained network; $`\sigma_t=0`$ gives a deterministic first-order discretization of the probability-flow ODE, so you can skip timesteps and apply standard ODE solvers. (§20.7)

**94. What is flow matching and why is it replacing DDPM?** ★
Define $`x_t=(1-t)x_{\text{noise}}+t\,x_{\text{data}}`$ and regress the velocity $`v_\theta(x_t,t)`$ onto $`x_{\text{data}}-x_{\text{noise}}`$; sample by integrating the ODE. Straight paths mean far less discretization error, so 10–20 steps without distillation, and there's no noise schedule to tune. Conceptually the same family as diffusion, with a better-chosen path. (§20.10)

**95. Why latent diffusion?**
Run diffusion in a pretrained autoencoder's latent space: $`512^2\times3\to64^2\times4`$ is a 48× reduction in elements. The autoencoder handles imperceptible high-frequency detail; the diffusion model handles semantics. The autoencoder is also a hard ceiling — lost fine text and small faces cannot be recovered. (§20.11)

**96. How does discrete diffusion work?**
Replace Gaussian noise with a categorical transition matrix $`Q_t`$; the cumulative $`\bar Q_t=\prod Q_s`$ plays the role of $`\bar\alpha_t`$, and the posterior $`q(x_{t-1}|x_t,x_0)`$ is exact and closed-form. The absorbing/masking kernel dominates, and with it the objective reduces to a weighted average of masked-LM losses — **BERT with a continuously varying mask rate.** (§21.2–21.3)

**97. What is AdaLN-Zero and why does it matter?** ★
Adaptive LayerNorm plus a per-block gate $`\alpha(c)`$ on the residual branch, with $`\alpha`$'s producing MLP zero-initialized — so every block starts as the exact identity. Same principle as zero-init residual, Fixup, and ReZero. It beat cross-attention and in-context conditioning by a wide margin at every scale in the DiT ablations. (§21.4)

### Saying it in plain English — generative models

*Generative-model questions are where the algebra is heaviest and the underlying ideas are the most physical. Six, without a single symbol.*

**Q83 — Why VAE samples are blurry.**
Three reasons, all pointing the same direction.

The first is the loss. If you train a decoder with squared error, and the latent code  does not determine the output — it says "a face, smiling, glasses" but not the exact position of every hair — then the loss-minimizing output is the **average** of every image consistent with that description. Averaging faces gives you a smooth, plausible, slightly unreal face. Averaging is what squared error rewards, and blur is what averaging looks like.

The second is the direction of the divergence. Maximum likelihood punishes a model brutally for assigning near-zero probability to something that actually occurred, and punishes it only mildly for assigning probability to things that never occur. So the model plays it safe and spreads mass everywhere, including across the empty region between two  distinct kinds of image — and points sampled from that region are the blurry in-between ones.

The third is that the objective is not even the likelihood; it is a lower bound on it. A loose bound means you are optimizing something that has drifted from what you actually wanted.

The contrast worth drawing at the end: a GAN suffers from none of these, because it never computes a likelihood at all — it only has to fool a critic, and a blurry image is trivially detectable. ▸ **Sharpness and stability were traded against each other for the better part of a decade, and diffusion is what finally got both** — largely because its slowness turned out to be fixable by distillation, while the GAN's instability never really was.

**Q86 — The GAN's optimal discriminator, and the catastrophe hiding in it.**
Freeze the generator and ask what the best possible discriminator would look like. At any given point in image space it sees some density of real images and some density of fakes, and its best possible answer is simply the fraction that are real. Substitute that back into the generator's objective and you get a specific, respectable measure of distance between the two distributions, minimized exactly when they match. So far the game has the correct solution, which is reassuring.

Here is the catastrophe. That distance measure is **constant** wherever the two distributions do not overlap. And in high dimensions they essentially never overlap: real images occupy a thin sheet, generated images occupy a different thin sheet, and two thin sheets in an enormous space intersect nowhere. Constant means zero gradient. So the discriminator becomes perfect, its confidence saturates, and the generator receives no information whatsoever about which direction to move. Everyone who has trained a GAN has watched this happen and has watched the loss curves become uninformative at the same moment.

That single observation organizes the whole subsequent literature. WGAN swaps in a distance that keeps saying "you are getting warmer" even when the supports are disjoint. And every other stabilization trick — adding noise to the inputs, one-sided label smoothing, gradient penalties, limiting the discriminator's capacity — is an attempt either to force the two supports to overlap or to stop the discriminator winning outright.

**Q88 — The diffusion forward process, and why the closed form matters.**
The forward process is *defined* one step at a time: shrink the image slightly, add a little noise, repeat a thousand times. Taken literally, reaching step 700 means running seven hundred steps, and training would be unbearable.

But adding Gaussian noise to Gaussian noise gives Gaussian noise, and the variances simply add. So all thousand steps collapse into one line: the original image scaled by one number, plus fresh noise scaled by another, with the two numbers fixed by the schedule and always arranged so the total variance stays at one. **You can jump directly to any noise level in a single operation.**

This is the fact that makes diffusion trainable, and it is the right thing to lead with. Each training step picks a random image, picks a random timestep, jumps straight there, and asks the network to identify the noise that was added. No simulation, no sequential dependency, every example in the batch at a different timestep, perfectly parallel. Remove the closed form and diffusion models would be an interesting idea nobody could afford to train.

**Q91 — Diffusion and score matching are the same algorithm.**
Two research programmes reached the same algorithm from opposite directions and did not initially realize it.

One said: corrupt data with noise, learn to undo the corruption, then run the undoing repeatedly starting from pure noise. That is denoising diffusion.

The other said: do not model the probability of an image — model its *gradient*, the direction of "more probable" at every point in image space — and then follow that field uphill, adding a little noise so you explore rather than collapsing onto a single peak. That is score matching with Langevin sampling.

They coincide because of a small piece of calculus. For a point that is a clean image plus a known amount of Gaussian noise, the direction of increasing probability is **exactly** the direction that removes the noise, scaled by a constant that depends only on how much noise there is. So a network trained to predict the noise *is* a network that estimates the gradient of the log-density, up to a known factor.

There is a third face of the same object, and mentioning it is a strong signal. The best possible denoiser, given a noisy input, outputs the average of all the clean images that could have produced it. So: optimal denoiser, gradient of the log-density, and the posterior mean of the clean image are one mathematical object wearing three costumes. Everything else in the diffusion literature is a choice of how to discretize the path between them.

**Q92 — Classifier-free guidance.**
You want the image to obey the prompt more strictly than the model naturally does. The earlier approach bolted on a separate classifier and pushed the image toward "more likely to be labelled *a dog*" — which requires training a classifier that works on noisy images, an irritating extra component.

The trick is to make the model its own classifier. Train a single network that sometimes sees the prompt and sometimes sees nothing, by blanking the condition around ten percent of the time. Now at sampling time you can ask it both questions — "what would you denoise this into, given the prompt?" and "given nothing at all?" — and the difference between the two answers is precisely the direction the prompt is pulling in. Take that difference and **overshoot it**: go past the conditional answer, further along the prompt's direction than the model itself proposed.

That is the whole method, and in code it is roughly one line plus a doubled batch. The costs are worth naming unprompted, because interviewers wait for them. You are now sampling from a distribution sharpened beyond the one you trained on, so you get better prompt adherence and better-looking individual images, but noticeably **less diversity** — every prompt starts producing the same composition — and at high guidance you get blown-out colour and saturation artifacts, because the extrapolation eventually leaves the region where the model's predictions mean anything.

**Q94 — Flow matching, and why it is displacing DDPM.**
Diffusion learns to reverse a random walk. A random walk is wiggly, so the reverse path from noise to image is wiggly too — and a curved path has to be followed in many small steps, because any large step leaves the curve.

Flow matching asks a blunter question: why not draw a **straight line** from a noise sample to a data sample, and simply train the network to report the velocity along that line? Sampling then means starting at noise and integrating the velocity field — following the arrows. Because the paths are close to straight, a coarse integrator does fine, and you get good samples in ten or twenty steps rather than a thousand, with no separate distillation stage and no noise schedule to tune.

The framing that lands well in a room: **diffusion and flow matching are the same family** — both learn a vector field that transports noise into data — and the difference is only the choice of path between the two endpoints. Diffusion inherited its path from the physics it came from, where the path was not a design choice at all. Flow matching picked the path deliberately, and picked a straighter one.

---

## 34.8 Classical ML

**98. Why does ridge shrink some directions more than others?** ★
In the SVD basis, $`\hat y=\sum_j u_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}u_j^\top y`$. Directions with small $`\sigma_j`$ — poorly determined by the data, hence noisily estimated — are shrunk most. Exactly the right behaviour. (§22.2)

**99. Why does LASSO produce exact zeros and ridge doesn't?**
The subgradient of $`\lambda|\beta_j|`$ at 0 is the interval $`[-\lambda,\lambda]`$, so $`\beta_j=0`$ is optimal whenever $`|x_j^\top r|\le\lambda`$. Ridge's penalty gradient $`2\lambda\beta_j\to0`$ as $`\beta_j\to0`$, so it never pins. Geometrically: the $`\ell_1`$ ball has corners on the axes. (§22.3)

**100. When do you use elastic net?**
Correlated predictors. LASSO arbitrarily selects one of a correlated group and zeros the rest, which is unstable across resamples; the ridge component induces a grouping effect. (§22.3)

> **Common misconception.** *"LASSO tells you which features matter."* It tells you which features suffice for one particular fit of one particular sample. Hand it a group of correlated predictors and it keeps one essentially arbitrarily and zeros the rest — then resample the data and it will often keep a different one. Nothing about the discarded features was shown to be unimportant; they were shown to be redundant *given* the survivor. The belief is tempting because a zero coefficient looks like a verdict and a sparse model looks like an explanation, and both are enormously satisfying to present to a stakeholder. If you need stability, that is what the elastic net's ridge component is for (Q100), and what stability selection — refit over many resamples and count how often each feature survives — was designed to measure. ▸ **Selection is not importance**, and a candidate who says so unprompted has usually shipped a model somebody argued with.

**101. Derive the SVM dual and explain support vectors.** ★
Lagrangian, stationarity gives $`w=\sum_i\alpha_iy_ix_i`$ and $`\sum_i\alpha_iy_i=0`$; substituting gives $`\max_\alpha\sum\alpha_i-\frac12\sum\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle`$. Data enters only through inner products (→ kernels), and KKT complementary slackness forces $`\alpha_i=0`$ for every point off the margin — exact sparsity. (§22.5)

> **Where this came from.** The linear separating hyperplane with maximum margin goes back to **Vladimir Vapnik and Alexey Chervonenkis** in the Soviet Union in the 1960s. What turned it into the dominant classifier of the 1990s were two later additions, both at **Bell Labs**: the **kernel trick**, contributed by Bernhard Boser, Isabelle Guyon and Vapnik in 1992, and the **soft margin** for non-separable data, from Corinna Cortes and Vapnik in 1995. The kernel insight is the one to appreciate here, because it falls directly out of the dual you just derived — once the data appears only through inner products, you may replace the inner product with any function that behaves like one, and you are silently working in a much richer feature space without ever constructing it. Their first prominent application was reading handwritten digits, in the same lab and on the same problem that produced the convolutional networks that would eventually displace them. It is worth remembering that from roughly 1995 to 2010, "machine learning" meant kernel methods to most researchers, and neural networks were the unfashionable choice.

**102. State the representer theorem and why it matters.**
For $`\min_f\sum_iL(y_i,f(x_i))+\Omega(\|f\|_\mathcal{H})`$ with $`\Omega`$ strictly increasing, the minimizer is $`f^*=\sum_i\alpha_ik(x_i,\cdot)`$. Proof: decompose $`f=f_\parallel+f_\perp`$; the reproducing property means $`f_\perp`$ doesn't affect the loss but strictly increases the norm. It's why an infinite-dimensional optimization reduces to $`n`$ coefficients. (§22.5)

**103. Why does the RBF kernel's infinite dimensionality not cause overfitting?**
Capacity is controlled by norm, not dimension. The margin/norm-based bounds of Chapter 2 §2.5 don't involve $`d`$. (§22.5)

**104. Why not use misclassification error as a tree splitting criterion?** ★
It's piecewise linear in $`p`$, so a split that increases purity without changing the argmax gives zero gain. Gini and entropy are strictly concave, so any purity-increasing split registers. Concavity is what makes greedy search work. (§23.1)

**105. Why do random forests use random feature subsets?** ★
Ensemble variance is $`\rho\sigma^2+\frac{1-\rho}{B}\sigma^2`$. The second term vanishes with more trees; the first does not. So the only way to improve a large ensemble is to reduce the correlation $`\rho`$ — and random features stop every tree from choosing the same dominant split at the root. (§23.2)

**106. Bagging vs boosting.**
Bagging averages many low-bias, high-variance deep trees to reduce **variance**. Boosting sums many high-bias, low-variance shallow trees to reduce **bias**. Opposite terms of the same decomposition, which is why the tree depths differ. (§23.3)

**107. Show AdaBoost minimizes exponential loss.** ★
Forward stagewise: $`\min_{\alpha,G}\sum_iw_i e^{-\alpha y_iG(x_i)}`$ where $`w_i=e^{-y_if_{m-1}(x_i)}`$ — AdaBoost's weight, derived rather than posited. Splitting by correct/incorrect gives $`(e^\alpha-e^{-\alpha})\mathrm{err}+e^{-\alpha}`$; minimizing over $`G`$ is minimizing weighted error, and $`d/d\alpha=0`$ gives $`\alpha=\frac12\log\frac{1-\mathrm{err}}{\mathrm{err}}`$. (§23.3)

> **Where this came from.** AdaBoost began life as the answer to a **yes-or-no theoretical question**. In 1988 Michael Kearns and Leslie Valiant asked whether "weak" learnability and "strong" learnability are the same thing: if you can only build a classifier that is *barely* better than a coin flip, can you always combine such classifiers into one that is arbitrarily accurate? It was posed as an open problem in learning theory, with no algorithm attached and no expectation of practical use. **Robert Schapire** proved the answer was yes in 1990, by construction. **Yoav Freund and Schapire** then produced AdaBoost in the mid-1990s as a far more practical version of that construction, and it won the **Gödel Prize** — a theory award — in 2003. The weights that look so arbitrary when AdaBoost is taught procedurally are not arbitrary at all; as the derivation above shows, they are what falls out of fitting an additive model to exponential loss one term at a time, a connection Friedman, Hastie and Tibshirani made explicit around 2000 and which opened the door to gradient boosting with any loss function you like.

**108. Derive XGBoost's leaf weight and split gain.** ★★
Second-order expansion, grouped by leaf: $`\tilde{\mathcal{L}}=\sum_j[G_jw_j+\frac12(H_j+\lambda)w_j^2]+\gamma T`$. Each leaf is an independent quadratic, so $`w_j^*=-\frac{G_j}{H_j+\lambda}`$ and $`\tilde{\mathcal{L}}^*=-\frac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T`$. Hence $`\mathrm{Gain}=\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma`$: don't split if negative. (§23.5)

**109. What does LightGBM do differently?**
Histogram binning (255 bins, $`O(\#\text{bins})`$ split search), leaf-wise rather than level-wise growth (control with `num_leaves`), GOSS (keep large-gradient instances, subsample the rest with a reweighting that keeps the gain estimate unbiased), and EFB (bundle mutually-exclusive sparse features). (§23.6)

**110. How does CatBoost avoid target leakage?** ★
Ordered target statistics: encode a categorical value using only the examples *earlier* in a random permutation. Same idea as out-of-fold target encoding, applied online. Ordered boosting extends it to the residuals themselves. (§23.6)

**111. Derive EM for a GMM and show it monotonically increases the likelihood.** ★
E-step: $`\gamma_{ik}=\frac{\pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x_i|\mu_j,\Sigma_j)}`$. M-step: $`\pi_k=N_k/n`$, $`\mu_k=\frac1{N_k}\sum_i\gamma_{ik}x_i`$, $`\Sigma_k=\frac1{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top`$. Monotonicity: the E-step raises the Jensen bound to *equality* with the log-likelihood; the M-step then increases the bound; the log-likelihood is $`\ge`$ the bound, so it increased too. (§24.4)

**112. What's the relationship between k-means and GMM?**
k-means is GMM with isotropic equal-weight shared covariance $`\sigma^2I`$ in the limit $`\sigma^2\to0`$, where soft responsibilities become hard assignments. (§24.4)

**113. Why is the GMM likelihood unbounded?**
A component collapsing onto a single point has $`|\Sigma_k|\to0`$ and likelihood $`\to\infty`$. Fix with a covariance floor $`\epsilon I`$ (a MAP prior) or by restarting collapsed components. (§24.4)

**114. Give three derivations of PCA.**
Maximum projected variance (top eigenvector of $`\Sigma`$), minimum reconstruction error (equivalent, since $`\|x\|^2=\|Px\|^2+\|x-Px\|^2`$), and the SVD of the centred data matrix. Always compute via SVD, never by forming $`X^\top X`$. (§24.1)

**115. Why does t-SNE use a Student-$`t`$ in the low-dimensional space?** ★
The crowding problem: ball volume grows as $`r^d`$, so a point in 50-D has far more roughly-equidistant neighbours than can be placed at similar distance in 2-D. A Gaussian would over-penalize moderate distances and collapse everything; the heavy tail lets moderately-dissimilar points sit far apart cheaply. (§24.3)

**116. What can you not read off a t-SNE plot?**
Cluster sizes (density is equalized), distances between clusters, and — at low perplexity — the existence of clusters at all (it will cluster pure noise). Also: never cluster on t-SNE coordinates and report the clusters as findings. (§24.3)

**117. How do you handle class imbalance?**
In order: use the right metric (PR-AUC, not accuracy or ROC-AUC); class weights; **threshold tuning on validation** — often the entire solution and frequently skipped; then resampling/SMOTE; focal loss for extreme dense-prediction imbalance. **Always resample inside the CV loop, never before splitting.** (§22.6)

> **Common misconception.** *"Class imbalance means you have to resample."* Resampling is fourth on the list above, and the ordering is the answer. The first three fixes — choose a metric that responds to the positive rate, weight the classes in the loss, and **tune the decision threshold on validation** — are cheaper, more stable, and very often the entire solution. Threshold tuning in particular gets skipped constantly: results are reported at 0.5 as though it were a law of nature rather than the arbitrary default of a library. Oversampling also has a real cost that people rarely price in — it changes the base rate the model implicitly learns, so its output probabilities are no longer calibrated to the world you deploy into, and you have traded a ranking problem for a calibration problem. And the operational rule that catches people in interviews: resample **inside** the cross-validation loop. Duplicate a minority example before splitting and the same row lands in train and test, and the score you report is fiction.

**118. Explain the curse of dimensionality concretely.**
To capture a fraction $`r`$ of the volume in $`d`$ dimensions, a hypercube neighbourhood needs edge length $`r^{1/d}`$: for $`d=10`$, $`r=0.01`$, that's 63% of every feature's range. "Local" isn't local. And $`\frac{d_{\max}-d_{\min}}{d_{\min}}\to0`$, so all points become equidistant. (§22.6)

### Saying it in plain English — classical machine learning

*Classical ML questions punish recitation and reward "why was it designed this way." Six, spoken.*

**Q98 — Why ridge shrinks some directions more than others.**
Rewrite your data in its own natural coordinate system: the direction along which it varies most, then the next most, and so on down to directions in which it barely varies at all. In a direction with plenty of spread, the data pins the coefficient down tightly — there is real evidence about it. In a direction where every data point looks nearly identical, the coefficient is nearly unidentified, and a tiny change in the noise swings it enormously.

Ridge leaves the first kind almost untouched and crushes the second kind. Nobody designed that behaviour; it falls straight out of the algebra, because the shrinkage factor for each direction is its spread divided by that spread plus the penalty. Large spread, factor close to one, essentially no shrinkage. Small spread, factor close to zero, gone.

▸ **Ridge does not shrink everything uniformly. It shrinks each direction in proportion to how badly the data determined it** — which is exactly what a careful statistician would do by hand, and it is why ridge helps most precisely when your features are collinear.

**Q104 — Why accuracy is the wrong tree-splitting criterion.**
Take a node holding 80 positives and 20 negatives. A candidate split sends you to two children: one with 40 positives and 20 negatives, one with 40 positives and none. The second child is now pure — that is , useful progress toward a good tree. But the majority class in every node involved is still "positive," so the *number of misclassified points has not changed at all*. Measured by accuracy, this split was worthless. It was not.

The reason is a fact about shape. Misclassification rate, plotted against the class proportion, is made of straight line segments — and averaging over straight lines gives you back exactly the value you started from. Gini and entropy are strictly *curved*, bowed downward, and for a curved function the average over the pieces is strictly better than the whole. So any split that  purifies registers a positive gain, even when no label has flipped yet.

▸ **Curvature is what makes greedy tree-building work at all.** Without it the algorithm goes blind at precisely the moments when it is several splits away from a payoff — which is most of the time, and is why the criterion matters far more than it seems to.

**Q105 — Why random forests randomize the feature set.**
Averaging reduces variance, but only the part of the variance that is *independent* between the things being averaged. Write down the variance of an average of correlated estimators and you get two terms: one that shrinks as you add more trees, and one fixed by the average correlation between trees, which does not shrink at all. Add ten thousand trees and the first term is essentially gone; the second sits exactly where it started.

So past a few hundred trees, the only remaining lever is making the trees **disagree more**. And bootstrap resampling alone does not achieve much, because if one feature is strongly predictive, every tree discovers it and splits on it at the root — and now all your trees begin life nearly identical, differing only in the fine detail further down. Forcing each split to choose from a random handful of features breaks that: some trees never see the dominant feature at the top and are compelled to find a different, weaker story. Individually those trees are *worse*. Collectively they are much better, because they are wrong in uncorrelated ways.

▸ **Random forests deliberately damage each tree in order to decorrelate the ensemble.** Recognizing that the damage is the point — accuracy per tree traded for independence between trees — is the answer. "It adds randomness" is not.

**Q108 — XGBoost's leaf value and split gain.**
Boosting adds one tree at a time, each one correcting what the previous ones left over. The question is: given the model so far, what tree should I add next?

Rather than guess, approximate the loss locally as a parabola in the prediction — you know its slope and its curvature at the current prediction for every training point, whatever the loss function happens to be. Now the effect of adding a tree becomes simple, because everything that lands in the same leaf receives the same adjustment. Each leaf is therefore an independent one-dimensional parabola to minimize, and the answer is the leaf's total slope divided by its total curvature, plus a regularization term in the denominator that damps leaves holding very few points.

Substitute those optimal values back in and you obtain a **score for any tree structure**: a sum over leaves of slope-squared over curvature. That is the crucial object, because it turns splitting into arithmetic. Score the two children, score the parent, and the difference is exactly what the split buys you. Subtract a fixed cost per additional leaf; if the result is negative, do not split.

▸ **The gain formula converts pruning from a heuristic into a decision rule.** Earlier tree algorithms grew greedily and then pruned afterwards using cross-validation, with the complexity penalty living outside the objective. Here the cost of complexity sits inside the same objective the tree is optimizing, and that unification is most of why XGBoost outperformed what came before it.

**Q111 — EM, and why it can never go downhill.**
You have a cluster model but you do not know which cluster each point came from. If you knew the assignments, fitting the clusters would be trivial. If you had the clusters, guessing the assignments would be trivial. Classic chicken and egg. So you alternate: guess soft assignments given the current clusters, refit the clusters given those assignments, repeat.

Why does this not thrash back and forth forever? Because there is a lower bound running underneath the whole procedure — a floor that is always at or below the true likelihood. The assignment step does something very specific: it raises that floor until it **touches** the true likelihood at the current parameters. You are now standing on the floor and on the true curve at the same point. The refit step then moves the parameters to raise the floor as high as it will go. But the true likelihood is never below the floor — so it must have risen at least as much.

The picture: you are climbing a mountain hidden in cloud, using a scaffold you can see. Build the scaffold so it touches the mountain exactly where you are standing, walk to the highest point on the scaffold, rebuild, repeat. You can never descend. What you *can* do is get stuck on a local peak — EM guarantees improvement, never optimality, which is why the standard practice is several random restarts and keep the best.

**Q115 — Why t-SNE uses a heavy-tailed distribution in the map.**
Here is the problem it exists to solve. In fifty dimensions a point can have a great many neighbours all at roughly the same distance from it — there is simply room for them. Try to lay those same neighbours out on a flat page while preserving those distances and you cannot: a two-dimensional page does not have enough circumference at that radius. Something has to give, everything gets crushed inward, and the whole picture collapses into an undifferentiated blob. That is the crowding problem.

The fix is to change what "moderately far apart" *costs* in the low-dimensional map. If low-dimensional similarity falls off like a bell curve, then pushing two points moderately apart is heavily penalized and the layout refuses to spread out. A heavy-tailed distribution decays much more slowly, so moderately-dissimilar points can sit comfortably far apart without paying much. Space frees up, clusters separate, and the picture becomes readable.

The direct consequence — and the follow-up question you should expect, because it is where people get caught — is that **the geometry has been deliberately distorted in order to make the picture legible.** So the distances between clusters on a t-SNE plot mean nothing, the relative sizes of clusters mean nothing (density is equalized by construction), and at aggressive settings the method will happily manufacture beautiful, convincing clusters out of pure noise.

---

## 34.9 Reinforcement learning

**119. Prove the Bellman operator is a contraction.** ★
$`\|\mathcal{T}V-\mathcal{T}W\|_\infty\le\max_{s,a}\gamma\sum_{s'}P(s'|s,a)|V(s')-W(s')|\le\gamma\|V-W\|_\infty`$, using $`|\max_af-\max_ag|\le\max_a|f-g|`$ and $`\sum P=1`$. Banach then gives a unique fixed point and geometric convergence at rate $`\gamma^k`$. (§26.4)

**120. Why does $`\gamma`$ exist?**
Convergence of the infinite sum, equivalence to a per-step termination probability, and variance reduction. Effective horizon $`\frac{1}{1-\gamma}`$. Higher $`\gamma`$ is more farsighted and provably slower to converge. (§26.2)

> **Where this came from.** **Richard Bellman** developed dynamic programming at the RAND Corporation in the early 1950s, and in his autobiography he explains the name as an act of camouflage. His funding came through the Air Force and the Secretary of Defense at the time, Charles Wilson, was — by Bellman's account — actively hostile to anything that looked like mathematical research. So Bellman chose a name that was impossible to object to and conveyed nothing: "programming" in the wartime sense of scheduling and planning rather than computer code, and "dynamic" because it sounded impressive and could not be used pejoratively. The story comes to us from Bellman himself and is very likely polished in the retelling, but the naming convention it produced is real and still confuses people who expect "programming" to mean code. The fixed-point argument in the previous question is not Bellman's: it is **Stefan Banach's** contraction mapping theorem, published in 1922 in his doctoral work on functional analysis, three decades before anyone thought about sequential decisions. Bellman supplied the recursion; Banach supplied the guarantee that iterating it converges.

**121. SARSA vs Q-learning — explain with Cliff Walking.** ★
SARSA's target uses the action actually taken next, so it learns the value of the $`\epsilon`$-greedy policy *including* exploration and picks a safe path away from the cliff. Q-learning's $`\max`$ learns the greedy policy's value and takes the optimal cliff-edge path — better asymptotic policy, worse online return because it occasionally explores off the edge. (§26.6)

**122. What is maximization bias and how is it fixed?**
$`\mathbb{E}[\max_a\hat Q]\ge\max_a\mathbb{E}[\hat Q]`$ by Jensen — the max of noisy estimates is too large, and Q-learning uses the same values to select and evaluate. Double Q-learning decouples: one estimator picks the action, the other scores it. Double DQN applies this with the online and target networks. (§26.6, §27.3)

**123. What is the deadly triad?** ★
Function approximation + bootstrapping + off-policy training. Any two are safe; all three can diverge (Baird's counterexample). Every stabilizer in deep RL — target networks, replay, trust regions, conservative updates — is a partial countermeasure. (§26.7)

**124. Why is a target network necessary in DQN?**
Without it the regression target depends on the parameters being updated, so raising $`Q(s,a)`$ raises the target for $`Q(s',a')`$ — a positive feedback loop. Freezing the target turns RL back into a sequence of ordinary supervised regressions. (§27.2)

**125. Derive the policy gradient theorem.** ★
Log-derivative trick: $`\nabla_\theta\mathbb{E}[R]=\mathbb{E}[R\nabla_\theta\log p_\theta(\tau)]`$. Expanding $`\log p_\theta(\tau)`$, the transition dynamics and initial distribution don't depend on $`\theta`$ and vanish, leaving $`\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\Psi_t]`$. **The environment dropping out is what makes it model-free.** (§27.4)

> **Where this came from.** The estimator this theorem justifies was introduced by **Ronald Williams** in 1992 under the name **REINFORCE**, which is a  acronym and not a pun: *REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility*. Each capitalized fragment names a piece of the update — the step size, the reward relative to a baseline, and the gradient of the log-probability of the action taken. The theorem itself, in the form used today, was stated and proved by **Richard Sutton, David McAllester, Satinder Singh and Yishay Mansour** in 2000, which is what made policy gradients respectable with function approximation rather than a heuristic that happened to work. The underlying identity — that the derivative of an expectation can be rewritten as an expectation involving the derivative of a log-probability — is older still and is known in the simulation and operations-research literature as the *likelihood ratio* or *score function* method. As with so much of this book, the machinery was borrowed rather than invented.

**126. Prove a baseline doesn't bias the policy gradient.**
$`\mathbb{E}_a[\nabla\log\pi_\theta(a|s)b(s)]=b(s)\nabla\sum_a\pi_\theta(a|s)=b(s)\nabla 1=0`$. Any state-dependent baseline is free; the variance-minimizing one is near $`V^\pi(s)`$, giving the advantage. (§27.4)

**127. Derive GAE and explain $`\lambda`$ vs $`\gamma`$.**
$`\hat A^{\mathrm{GAE}}_t=\sum_l(\gamma\lambda)^l\delta_{t+l}`$ — an exponentially weighted average of all $`k`$-step advantage estimators. $`\lambda=0`$ gives $`\delta_t`$ (max bias, min variance); $`\lambda=1`$ gives Monte Carlo (unbiased, max variance). **$`\gamma`$ defines the objective; $`\lambda`$ is purely an estimator knob.** (§27.6)

**128. Explain PPO's clipped objective, including the sign asymmetry.** ★
$`\mathcal{L}=\mathbb{E}[\min(\rho\hat A, \mathrm{clip}(\rho,1\pm\epsilon)\hat A)]`$. For $`\hat A>0`$ the $`\min`$ caps the reward at $`\rho=1+\epsilon`$, removing incentive to move further. For $`\hat A<0`$ the $`\min`$ picks the more negative term, so there's no ceiling on pushing a bad action down until $`\rho<1-\epsilon`$. The $`\min`$ makes it a pessimistic lower bound on the surrogate — which is what makes multiple epochs on the same batch safe. (§27.7)

**129. What is SAC's objective and how does it relate to RLHF?** ★
Maximize reward plus policy entropy. The optimal policy is Boltzmann: $`\pi^*\propto\exp(Q/\alpha)`$ — **exactly the KL-regularized optimum from DPO's derivation with a uniform reference policy.** SAC's entropy bonus and RLHF's KL-to-reference are the same mathematical object. (§27.8, §16.5)

**130. What is the core problem in offline RL?**
Distributional shift on **actions**: the policy proposes actions absent from the data, $`Q`$ is unconstrained there, and the $`\max`$ systematically overestimates them — with no environment feedback to correct it. More gradient steps make it worse. Fixes: policy constraints (TD3+BC), conservative values (CQL), in-sample learning (IQL), or sequence modelling (Decision Transformer, which loses trajectory stitching). (§27.10)

**131. State the reward-shaping theorem.**
Adding $`F(s,a,s')=\gamma\Phi(s')-\Phi(s)`$ leaves the optimal policy unchanged for any $`\Phi`$. Any non-potential-based shaping can change the optimum, usually badly. (§26.9)

> **Common misconception.** *"The reward function tells the agent what you want it to do."* The reward tells the agent what gets reinforced, which is a statement about a **measurement**, not about your intention. Every gap between the two is an opportunity, and optimization is precisely the process of discovering opportunities. The canonical demonstration is a boat-racing agent that found it could score more by circling a lagoon striking regenerating targets than by finishing the course — behaving flawlessly, by the only definition it had been given, while never completing a race. The belief is tempting because writing the reward *feels* like stating the goal; the words in your head are about winning the race, and the number in the code is about points. This is also why the theorem above matters more than it looks: a well-meant bonus for behaviour you consider good will change the optimal policy unless it is potential-based. **A bonus paid whenever the agent moves closer to the goal, with no matching penalty for moving away, is a bonus for oscillating** — and an agent will find that out long before you do.

**132. Why is deep RL hard to reproduce?**
Results vary enormously across seeds; single-curve reporting is meaningless. Always report ≥5 seeds with a distribution. This is Chapter 3's lesson, violated more in RL than anywhere else. (§27.11)

> **Common misconception.** *"The agent learned the task — here is the curve."* A single reinforcement learning curve is not evidence. Rerun identical code with a different random seed and the outcome frequently changes *category* rather than magnitude: solved versus never left the floor. Published work has shown that two implementations of the *same* algorithm, differing only in engineering details nobody wrote down, can differ by more than two different algorithms do — which means an algorithm comparison can be a comparison of codebases wearing algorithm names. The belief is tempting because in supervised learning the seed usually moves the third decimal place, so the habit of ignoring it is well-earned everywhere else you have worked. Report at least five seeds with the spread visible, and treat any gap smaller than the seed-to-seed range as no result at all. This is Q16 and Q17 again, in the sub-field that violates them most.

### Saying it in plain English — reinforcement learning

*Reinforcement learning answers live or die on whether you can name what goes wrong. Six, spoken.*

**Q119 — Why the Bellman operator is a contraction.**
Take two completely different guesses about how valuable every state is — one wildly optimistic, one gloomy. Apply a single round of Bellman updating to both: for each state, look one step ahead, take the immediate reward, and add the discounted value of wherever you land.

The immediate rewards are identical in both guesses, so they cancel. What remains is the discounted difference between the two guesses one step downstream. Discounted means multiplied by a number less than one. So whatever the largest disagreement between the two guesses was, after one round it is at most that fraction as large.

▸ **Every round of updating shrinks the worst-case disagreement by a fixed factor.** From that one line you get everything: any two starting guesses converge toward each other, so there can be only one fixed point; you approach it geometrically rather than eventually; and the error after some number of rounds is bounded by the discount raised to that power. It is the same argument as repeatedly halving a distance, with the discount factor playing the role of the halving.

Notice what it also says about horizons, because that is the part people miss. A discount of 0.99 shrinks the error by only one percent per round, so you need hundreds of rounds to converge. Long-horizon problems are slow for a reason visible in a single line of algebra, not because of anything to do with neural networks.

**Q121 — SARSA versus Q-learning, on the cliff.**
Picture a grid world where the shortest route runs along the very edge of a cliff, and falling off is catastrophic. You are exploring, so ten percent of your moves are random.

Q-learning learns the value of behaving *perfectly*. Its update assumes that from the next state onward it will always choose the best action — so the cliff-edge path looks excellent, because a perfect agent never steps off. It converges to the  optimal policy. But it is *executing* the exploratory policy, so during training it falls off the cliff repeatedly, and its actual accumulated return while learning is poor.

SARSA learns the value of what it actually does, exploration included. Standing on a cliff-edge tile, its update accounts for the ten percent chance that the next move is random and fatal — so those tiles  score lower, and it learns to walk one row inland. Slightly longer route, far better performance during learning.

▸ **Neither is wrong; they answer different questions.** "What is the best policy?" versus "what is the value of the policy I am actually running, warts and all?" The practical translation, which is what an interviewer is listening for: if failures during learning are cheap — a simulator, a game — learn off-policy. If you are learning on a real robot, or anywhere a mistake has a bill attached, the on-policy answer is the one that keeps you solvent.

**Q123 — The deadly triad.**
Three ingredients, each individually reasonable and each individually something you want:

- **Function approximation** — you generalize values across states instead of storing a lookup table, so updating one state necessarily moves the estimates of others.
- **Bootstrapping** — your training target is built from your own current estimates rather than from actual observed returns, which is what makes learning fast.
- **Off-policy data** — you learn about one policy from data collected by another, which is what makes replay buffers and reuse of old experience possible.

Any two of the three are safe. All three together and you can get a runaway feedback loop: an inflated estimate at a state you rarely visit leaks, through the shared function approximator, into the targets for states you *do* visit, which inflates those, which feeds back — and no corrective data ever arrives, because you are not actually going to the state that started it. This is not hypothetical. There is a small, famous counterexample with a handful of states on which the weights provably diverge to infinity.

▸ **Almost the entire toolkit of deep reinforcement learning is a set of partial defences against this one failure.** Target networks slow the bootstrapping down. Replay buffers and importance weights manage the off-policy mismatch. Trust regions and conservative value penalties stop estimates running away. Saying that out loud reframes what looks like a grab-bag of tricks as a coherent response to a single structural problem, and it is the answer that separates candidates.

**Q125 — The policy gradient theorem.**
You want to improve the *average* reward. But reward comes from the environment, and you cannot differentiate the environment — you have no equations for it, only samples.

The trick is to move the derivative off the reward and onto the one thing you do control. There is an identity that rewrites "the derivative of an average" as "the average of reward times the derivative of the log-probability of what you did." Now differentiate the probability of a whole trajectory: it is the product of the environment's transition probabilities and your policy's action probabilities. Take the logarithm and the product becomes a sum. And every term belonging to the environment has no dependence on your parameters whatsoever — so it differentiates to zero and disappears.

▸ **The environment drops out of the gradient. That is what "model-free" means, mechanically.** What is left is almost embarrassingly simple: for each action you took, nudge its log-probability up in proportion to the reward that followed. Good outcome, make what you did more likely; bad outcome, less. The theorem's real content is not that rule — anyone would guess that rule — but that it is an **unbiased** estimate of the true gradient of expected return, despite you knowing nothing at all about how the world works.

**Q128 — PPO's clipped objective, including the sign asymmetry.**
You have data collected from your old policy and you want to squeeze several gradient steps out of it, which is only valid while the new policy stays close to the old one. PPO enforces closeness by clipping the probability ratio — how much more likely the new policy makes each action.

Take a *good* action, one with positive advantage. The objective rewards you for raising its probability, but the clip caps that reward once you have raised it by about twenty percent. Beyond that you gain nothing, so the gradient goes to zero and the update simply stops pushing. Fine.

Now take a *bad* action. The objective takes the more pessimistic of the clipped and unclipped terms, and for a negative advantage that is the unclipped one over a long range. So there is no ceiling on driving a bad action's probability toward zero — the brakes only engage once you have already cut it by twenty percent.

The asymmetry is deliberate, and the reason is worth stating: **it is much safer to abandon a bad action aggressively than to commit to a good-looking one aggressively.** An action that merely looks good because of a noisy advantage estimate is the thing that wrecks a policy; an action you correctly abandon costs you very little.

The unifying reading: taking the minimum makes the objective a pessimistic **lower bound** on the true surrogate. You are always optimizing something no more optimistic than reality, and that is precisely what makes running several epochs over the same batch safe.

**Q129 — SAC's entropy bonus is RLHF's KL penalty.**
Soft Actor–Critic maximizes reward plus the randomness of the policy — it literally pays you to stay unpredictable, so you keep exploring and do not commit prematurely to whatever looked best early. Solve for the optimal policy under that objective and it has a specific shape: the probability of an action is exponential in its value, with a temperature setting the exchange rate between reward and randomness.

Now look at the RLHF objective: maximize reward while staying close, in KL terms, to a reference model. Solve that and you get the reference policy multiplied by an exponential in reward. **The same shape**, with the reference model standing where the uniform distribution stood.

▸ **Entropy regularization is KL regularization toward a uniform reference.** They are one equation. That is why the derivation at the heart of DPO and the derivation at the heart of SAC are the same three lines, and why "keep the model near the base model" and "keep the policy stochastic" produce solutions of the same functional form. Recognizing that a robotics result from 2018 and a language-model alignment result from 2023 are the same mathematical object is exactly the kind of connection that senior interviews are built to find.

---

## 34.10 Vision, graphs, and the science of DL

**133. Two 3×3 convs vs one 5×5?**
Same receptive field, $`18C^2`$ vs $`25C^2`$ parameters, and two nonlinearities instead of one. The whole VGG argument. (§8.6)

**134. What does a 1×1 convolution do?**
Channel dimensionality reduction, cross-channel mixing, and cheap added nonlinearity. It's a per-position dense layer. (§8.1)

**135. Why is the effective receptive field smaller than the theoretical one?**
Paths from the centre to the output vastly outnumber paths from the edge, so influence is a random-walk convolution — approximately Gaussian with radius $`\propto\sqrt L`$ rather than $`L`$. Assume roughly the square root of the formula. (§8.1)

**136. What causes checkerboard artifacts?**
Transposed convolution with a stride that doesn't divide the kernel size, giving uneven overlap. Fix: nearest/bilinear upsample followed by a normal conv. (§8.1)

**137. Why did ViT lose on ImageNet-1k and win on JFT-300M?** ★
A CNN's locality/translation prior is correct but restrictive. When data is scarce, a correct prior substitutes for data; when data is abundant, it becomes a ceiling. Probing shows large-data ViTs *rediscover* convolution-like early layers. (§28.1)

**138. Explain CLIP's objective and one known weakness.**
Symmetric InfoNCE over a batch: an $`N`$-way "which caption belongs to this image" classification in both directions, with a learned temperature. Weakness: it behaves substantially like a bag of concepts — "a horse riding an astronaut" embeds near "an astronaut riding a horse" (ARO benchmark). Also weak at counting, spatial relations, and negation. (§28.2)

**139. Why SigLIP over CLIP?**
The softmax needs a global normalization over the batch, forcing an expensive all-gather. A pairwise sigmoid loss is decomposable per pair, so it works at small batch sizes and scales without the communication cost. (§28.2)

**140. What limits GNN depth?** ★
Over-smoothing: repeated neighbourhood averaging is diffusion; $`\hat A^L`$ converges to a rank-one projection, so all node representations collapse, exponentially in $`L`$. Over-squashing pushes the other way: an exponentially large receptive field compressed into a fixed vector, bounded by the graph's spectral gap. Hence 2–4 layers, and hence graph transformers. (§29.5)

**141. State the WL expressivity bound.** ★
Message-passing GNNs are at most as powerful as the 1-WL test, because a layer computes a function of (node feature, multiset of neighbour features) — structurally a WL refinement round. Consequence: they cannot count triangles, detect cycles, or distinguish two 3-regular graphs of the same size. GIN reaches the bound by using **sum** aggregation (injective on multisets — mean and max are not) plus an MLP. (§29.4)

**142. Why does equivariance help?**
It's a hard constraint, so the model never spends capacity learning that physics is rotation-invariant, and it generalizes perfectly to unseen orientations. ~10× data efficiency on molecular property prediction. (§29.8)

**143. Explain double descent.** ★
Test error peaks at the interpolation threshold ($`p\approx n`$) because the design matrix is generically near-singular there (Marchenko–Pastur), so the minimum-norm interpolant's norm blows up. Past it, the solution space grows and the minimum-norm element gets smaller and smoother. Model-wise, epoch-wise, and sample-wise variants all exist — and more data can hurt. Explicit regularization removes the peak. (§30.1)

**144. What is the NTK and what does it fail to explain?** ★
At infinite width, the tangent kernel $`\Theta(x,x')=\langle\nabla_\theta f(x),\nabla_\theta f(x')\rangle`$ is deterministic and constant during training, so the network is exactly kernel regression with a fixed feature map. Explains convergence to global minima and spectral bias (smooth functions fit first). Fails because features never change — so it cannot explain transfer learning or representation learning, and finite networks beat their NTK. Real networks are deliberately kept in the feature-learning regime, which is what $`\mu`$P preserves. (§30.2)

**145. Explain grokking.** ★
Delayed generalization: 100% train accuracy at step $`10^3`$, test accuracy jumps at $`10^5`$. A memorizing circuit is found first; a generalizing one has smaller weight norm. Once training loss is ~0, the CE gradient vanishes and **weight decay becomes the dominant force**, slowly drifting the model along the zero-loss manifold to the minimum-norm (generalizing) solution. For modular addition the circuit was fully reverse-engineered as a discrete Fourier transform. Continuous progress measures show smooth formation — the discontinuity is in the metric. (§30.3)

**146. What is neural collapse?** ★
In the terminal phase of training: within-class variability →0 (NC1), class means form a simplex ETF with pairwise cosine $`-\frac{1}{C-1}`$ (NC2), classifier weights align with class means (NC3), and classification reduces to nearest class centre (NC4). It's the global optimum of the unconstrained-features problem. NC1 destroys within-class information, which is a quantitative account of why over-training a classifier gives worse transfer features.  (§31.1)

**147. What is the implicit bias of gradient descent?** ★
On separable data with logistic loss, $`w(t)/\|w(t)\|`$ converges to the max-margin SVM direction — at rate $`O(1/\log t)`$, which is why training long past zero error still helps. On least squares from zero init it converges to the minimum $`\ell_2`$-norm interpolant. Adam's coordinate-wise normalization gives a different ($`\ell_\infty`$-geometry / $`\ell_1`$-margin) bias, which is the real explanation for the Adam generalization gap. (§31.2)

**148. What's wrong with "flat minima generalize better"?**
Sharpness isn't reparameterization-invariant: $`(W_1,W_2)\to(\alpha W_1,\alpha^{-1}W_2)`$ leaves a ReLU network's function unchanged while scaling Hessian eigenvalues by $`\alpha^{\pm2}`$ (Dinh et al.). So any minimum can be made arbitrarily sharp. Responses: scale-invariant sharpness measures, the observation that such reparameterizations aren't visited in practice, and SAM's empirical success. (§31.3)

**149. State the Lottery Ticket Hypothesis and the correction to it.** ★
A dense random network contains a sparse subnetwork that, trained *from the original initialization*, matches full accuracy. Found by iterative magnitude pruning with weight **resetting** — reinitializing the same mask randomly fails, so the ticket is (mask, init). The correction: at ImageNet scale you must **rewind** to $`\theta_k`$ for small $`k`$ rather than $`\theta_0`$ — the ticket forms in the first few hundred steps, at the point of linear mode connectivity. (§31.4)

**150. Why does model merging work?** ★
Models fine-tuned from a shared base stay in the same loss basin, so task vectors $`\tau=\theta_{\text{FT}}-\theta_{\text{pre}}`$ add meaningfully (and negate to remove behaviours). Models from *different* seeds are in the same basin only up to **permutation symmetry** of hidden units — align the units first (Git Re-Basin) and the interpolation barrier largely disappears. (§17.8, §31.5)

**151. Explain superposition.** ★
A model represents more features than it has dimensions by using nearly-orthogonal directions, tolerating interference because features are sparse and rarely co-active — and a ReLU suppresses the small crosstalk. JL guarantees exponentially many such directions exist. Consequence: individual neurons are polysemantic, so "what does neuron 1432 do" is the wrong question. In the toy model, sparsity drives a phase transition from orthogonal representation to antipodal pairs to tetrahedra and other sphere packings. (§32.2)

**152. What is a sparse autoencoder and what are its problems?**
An overcomplete ($`m\approx 8`$–256$`\times d`$) autoencoder with an $`\ell_1`$ penalty, trained to reconstruct activations from a sparse code. Problems: $`\ell_1`$ shrinkage biases all magnitudes toward zero (fixed by TopK/JumpReLU), dead features, **feature splitting with no canonical granularity**, and circular evaluation. Strongest evidence for their validity is causal: clamping a feature steers behaviour. (§32.3)

**153. What are QK and OV circuits?**
An attention head factorizes into $`W_{QK}=W_Q^\top W_K`$, which decides *where* to read, and $`W_{OV}=W_OW_V`$, which decides *what* to write. Independently analyzable low-rank operations in the residual-stream basis. (§32.4)

**154. Why is activation patching better than a saliency map?**
It's causal. Patching an activation from a clean run into a corrupted one and measuring the output change tests whether the component carries the relevant information. Denoising finds sufficient components, noising finds necessary ones, and path patching isolates specific edges. Note self-repair means ablation *understates* importance. (§32.5)

**155. Explain why deep generative models assign higher likelihood to OOD data.** ★
Likelihood in high dimensions is dominated by low-level statistics, not semantics: SVHN is smoother than CIFAR-10, so its pixels are easier to predict, so a CIFAR-trained Glow gives it higher likelihood. **Likelihood is not typicality** — a high-likelihood point can lie far outside the typical set. Fixes use likelihood ratios against a background model. (§33.6)

**156. Why are modern networks miscalibrated, and how do you fix it?** ★
Capacity plus reduced regularization: the model drives training NLL toward zero, pushing probabilities to 1 long after accuracy saturates. Fix with **temperature scaling** — one scalar fitted on validation NLL, which cannot change accuracy and often reduces ECE 10×. Caveat: it calibrates in-distribution only. (§33.1–33.2)

**157. Explain conformal prediction and prove the coverage guarantee.** ★★
Split off a calibration set, compute nonconformity scores, take the $`\lceil(n+1)(1-\alpha)\rceil/n`$ quantile $`\hat q`$, and predict $`\{y: s(x,y)\le\hat q\}`$. Proof: under exchangeability the new score's rank among $`n+1`$ scores is uniform, so $`P(s_{\text{new}}\le s_{(\lceil(n+1)(1-\alpha)\rceil)})\ge1-\alpha`$. Requires nothing of the model. **But coverage is marginal, not conditional** — 90% overall can hide 40% on a subgroup; use group-conditional calibration. (§33.5)

**158. Aleatoric vs epistemic uncertainty, and why do ensembles beat MC dropout?**
Aleatoric is irreducible data noise; epistemic is model uncertainty, reducible with data, and equals the *disagreement* among plausible models. Ensembles beat MC dropout and variational methods because different initializations land in  different loss basins, giving functional diversity; single-mode methods only explore one basin's neighbourhood. (§33.3–33.4)

**159. Why do adversarial examples exist?** ★
Locally, $`w^\top\delta`$ with $`\delta=\epsilon\,\mathrm{sign}(w)`$ gives a change of $`\epsilon\|w\|_1`$, which grows with dimension — many tiny coordinated changes sum to a large logit change. Deeper: Ilyas et al. showed they arise from **non-robust features that are  predictive** — a dataset labelled only by non-robust features yields good clean test accuracy. Vulnerability is a property of the data the model faithfully learned, not a bug.  (§33.8)

**160. How do you evaluate an adversarial defence honestly?**
AutoAttack plus adaptive attacks designed against your specific defence. Watch for gradient masking: black-box beating white-box, unbounded attacks failing to reach 0%, or one-step beating iterative. Most published defences were broken this way. (§33.8)

**161. How do you fix a spurious correlation?**
Ideally break it with data or counterfactual augmentation. Otherwise GroupDRO (needs group labels and strong regularization) or JTT (upweight the first model's errors — no group labels). But first try the cheapest fix: **retrain only the last layer on a small group-balanced set** — Kirichenko et al. showed the representation usually already contains the core feature and only the classifier relies on the shortcut. (§33.9)

### Saying it in plain English — vision, graphs, and the science of deep learning

*This is the widest section in the bank, and the one where the compressed answers are furthest from anything you would say aloud. Six of the densest, spoken.*

**Q140 — What limits the depth of a graph neural network.**
Two failures push from opposite ends, and naming both is the answer.

Start with the obvious one. A graph network layer replaces each node's description with some blend of its neighbours' descriptions. Do that once and a node learns about its immediate surroundings, which is the point. Do it thirty times and you have run a diffusion process — the same mathematics as heat spreading through a metal plate — and heat spreading always ends the same way: everything reaches the same temperature. After enough rounds every node in a connected graph holds essentially the same vector, and a classifier reading those vectors cannot tell any node from any other. That is **over-smoothing**, and it arrives exponentially fast, not gradually.

Now the opposite failure, which is the one candidates miss. Suppose you *did* want a node to hear from something eight hops away. The number of nodes within eight hops of you grows explosively — on a social graph it can be most of the network. All of that has to arrive through a fixed-size vector, funnelled through whatever narrow bridges the graph happens to have. Exponentially much information, constant-width pipe. That is **over-squashing**, and it is a property of the graph's shape, not of your architecture: a graph with a bottleneck cannot pass information across the bottleneck no matter how you build the layer.

▸ **You are squeezed from both sides — shallow enough to avoid collapse, deep enough to reach anything.** In practice the window is two to four layers, which is a startling constraint if you have arrived from vision where a hundred layers is routine. And it is why graph transformers exist: let every node attend to every other directly, and the number of hops stops being the currency.

**Q143 — Double descent.**
The classical picture everyone is taught is a U. Too few parameters and the model cannot represent the pattern; too many and it memorizes the noise; somewhere in between is the sweet spot. That U is real, and it is only the left half of the actual curve.

Keep adding parameters past the point where the model can exactly fit every training point and something perverse happens: test error, which had been climbing, **peaks and then falls again** — often ending below the classical sweet spot. Hence two descents.

The peak sits precisely where the number of parameters equals the number of training examples, and the reason is worth being able to state. At that exact point there is essentially **one** way to fit the data perfectly, and the model has no choice about it — it must adopt that solution however violent it is. The arithmetic backs this up: the system of equations is square and generically almost singular, so the unique solution has enormous coefficients and a wildly oscillating shape.

Go past that point and you have *many* perfect fits available, and now the optimizer gets to choose. Gradient descent's preference is for small, smooth solutions, and the more surplus parameters you provide, the smoother the smoothest available perfect fit becomes. **Extra capacity is not extra flexibility to memorize; it is extra freedom to be gentle.**

Two follow-ups to have ready, because interviewers reach for them. The same peak appears along other axes — you can see it as a function of training *time*, and as a function of dataset *size*, which yields the  uncomfortable statement that **more data can make a model worse** if it moves you toward the threshold rather than past it. And adding honest regularization flattens the peak away entirely, which tells you the whole phenomenon is a story about an unregularized model being forced into a bad solution rather than anything deep about capacity.

**Q145 — Grokking.**
A small network is trained on modular arithmetic. Within a thousand steps it has perfect training accuracy and chance-level test accuracy — textbook memorization, the point at which you would normally stop and declare it overfit. Training continues, pointlessly, for a hundred times longer. Then test accuracy jumps from chance to perfect, suddenly, long after the training loss stopped moving.

The mechanism is more satisfying than the phenomenon. There are two kinds of solution available: a lookup table that stores every training pair, and a  algorithm that computes the answer. The lookup table is easier to stumble into, so gradient descent finds it first. The algorithm has a **smaller weight norm** — it is a more compact object.

Now ask what force is still acting once the training loss is essentially zero. The cross-entropy gradient has all but vanished, because there is nothing left to correct. What has not vanished is weight decay, which pulls steadily toward smaller weights regardless of loss. So the model drifts, slowly, along the surface of solutions that all achieve zero training loss, in the direction of smaller norm — and eventually arrives at the compact one, which is the one that generalizes. ▸ **After the loss flattens, weight decay is the only thing still steering, and it steers toward the algorithm.**

Two details that turn a good answer into a strong one. For modular addition the learned algorithm was fully reverse-engineered and it is a discrete Fourier transform — the network independently discovered that you can add angles by rotating, which is how a human would do modular arithmetic if they thought about it as a clock. And the *sharpness* is largely an artifact of what you plotted: continuous progress measures, tracking how much of the circuit has formed, rise smoothly throughout the apparent plateau. The transition is in the metric, not the mechanism — the same lesson as the emergence debate in Q65.

**Q151 — Superposition.**
A model has, say, four thousand dimensions in its residual stream and needs to represent far more than four thousand distinguishable concepts. Naively that is impossible: four thousand dimensions hold four thousand independent directions.

It is possible because "independent" is doing less work than it appears. If you accept directions that are merely *nearly* perpendicular rather than exactly perpendicular, the number available grows **exponentially** with the dimension rather than linearly — that is the Johnson–Lindenstrauss result from Q10, cashed in. So you can pack tens of thousands of concepts in, at the cost of every pair interfering slightly.

Why is the interference tolerable? Because the features are **sparse**. Nearly every concept is absent from nearly every input — "this text is in Hungarian," "this is a legal citation," "this mentions a specific chemical" are almost always off — so the crosstalk that would arise if they were all active at once almost never occurs. And a rectified linear unit is a threshold: small negative interference is clipped to zero and disappears entirely. Sparsity plus a threshold turns an overloaded code into a workable one.

The consequence is the part interviewers actually want, so lead toward it. If concepts are stored in directions that do not line up with the coordinate axes, then an individual neuron is a projection onto an arbitrary axis and will respond to a strange union of unrelated things. Neurons are **polysemantic**, and "what does neuron 1432 do?" is a malformed question. ▸ **The neuron basis is the wrong basis** — which is the entire motivation for sparse autoencoders, which go looking for the directions instead of accepting the axes.

One extra observation, if there is room. In the toy models, varying how sparse the features are produces  **phase transitions** in the geometry: at low sparsity the model gives a few features clean orthogonal directions and drops the rest; as sparsity rises it starts pairing features into opposite directions, then arranging them in triangles, tetrahedra, and other recognizable sphere packings. The model is solving a packing problem, and you can watch it change its answer.

**Q155 — Why generative models score out-of-distribution data as *more* likely.**
Train a good density model on CIFAR-10 — small photographs of animals and vehicles. Then show it SVHN, photographs of house numbers, which it has never seen and which look nothing like its training data. It reports that the house numbers are **more probable** than its own training images. Not slightly; substantially. This is reproducible, it is not a bug, and it broke the obvious plan of using a generative model as an anomaly detector.

The explanation is that likelihood in high dimensions is dominated by the boring statistics. Predicting an image pixel by pixel, the overwhelming majority of your score comes from getting local smoothness right — this pixel resembles its neighbour — and house-number photographs are much smoother and flatter than photographs of animals. Easy to predict means high likelihood. Nothing in the objective ever asked "is this the *kind* of thing I was trained on."

The idea that makes this click, and that is worth stating in exactly these words: **likelihood is not typicality.** In high dimensions, the region where samples actually land is not the region of highest density. Draw from a five-hundred-dimensional standard Gaussian and you will essentially never land near the origin, even though the origin is the single most probable point — because the volume out at radius $`\sqrt{d}`$ is overwhelming. So the most probable point is not a typical point, and "high likelihood" and "looks like my training data" come apart. Practical fixes therefore stop asking "is this likely?" and start asking "is this likelier under my model than under a generic background model?" — a likelihood *ratio*, which cancels the low-level statistics that were doing all the damage.

**Q157 — Conformal prediction.**
You have a trained model of unknown quality and you want a set of predictions that provably contains the truth ninety percent of the time. No assumption about the model, no assumption about the data distribution.

The recipe is almost embarrassingly plain. Hold out a calibration set the model never trained on. For each held-out example, score how badly the model handled it — one over the probability it gave the correct class, or the size of the residual, any measure of "how wrong." You now have a few thousand wrongness scores. Take the ninetieth percentile. For a new input, output **every** label whose wrongness score falls under that threshold.

Why that works is a single sentence about ranks. If the new example is exchangeable with the calibration examples — no assumption beyond "it came from the same place, and order does not matter" — then its wrongness score is equally likely to occupy any position in the sorted list. So it lands below the ninetieth percentile about ninety percent of the time. That is the whole proof, and it never once mentions the model. A terrible model still gets valid coverage; it just pays for it with enormous prediction sets, which is itself a useful signal.

Now the caveat, and volunteering it unprompted is what makes the answer strong. The guarantee is **marginal**: ninety percent averaged over everybody. It is entirely consistent with ninety-eight percent coverage on the easy majority and forty percent on the subgroup you most needed it for, and the guarantee will not warn you. If subgroups matter, calibrate within each subgroup separately — and know that exact per-input coverage is provably unattainable without assumptions, so this is a real limit rather than a gap in the technique.

> **Common misconception.** *"This neuron fires on curves, so it is the curve detector."* Under superposition a network stores more features than it has directions, so a typical neuron is **polysemantic** — it participates in several unrelated features and lights up on a strange union of them. Finding an input that reliably drives a neuron tells you one thing it responds to and almost nothing about what else it does, or about how much the model's output actually depends on it. The belief is tempting because the examples you find are  striking, and because a handful of neurons really are close to monosemantic — which produces a compelling and thoroughly unrepresentative gallery. The consequence *is* the answer to Q151: the neuron basis is the wrong basis, which is why sparse autoencoders hunt for directions instead of accepting axes, and why the validation has to be **causal** (clamp the feature, watch behaviour change) rather than visual.

> **Common misconception.** *"Temperature scaling improved the model."* It cannot. Dividing every logit by one positive scalar is a monotone transformation, so the ordering of the classes is untouched and the argmax is **exactly** unchanged — accuracy, precision, recall, and area under the ROC curve are all bit-for-bit identical before and after. What changes is only the numbers attached to the predictions: expected calibration error can fall by an order of magnitude while every decision the system makes stays the same. The belief is tempting because calibration plots improve dramatically, and a dramatic plot reads as a better model. Saying the invariance out loud is the cleanest available demonstration that you know what calibration is. The caveat to volunteer next: one scalar fitted on in-distribution validation data does essentially nothing for you under distribution shift, which is exactly when you wanted the probabilities to be trustworthy.

> **Common misconception.** *"Ninety percent conformal coverage means each individual prediction is ninety percent likely to be right."* Coverage is **marginal** — averaged across the whole input distribution. A method can hit ninety percent overall while covering ninety-eight percent of easy cases and forty percent of one clinically important subgroup, and the guarantee is entirely silent about that. The belief is tempting because the guarantee is unusually strong in one dimension — it assumes nothing whatever about the model, only exchangeability — and strength in one dimension is easily read as strength in all of them. The fix is group-conditional (Mondrian) calibration: run the procedure separately within each group you care about. The honest statement of the limit is that **exact conditional coverage is provably unattainable distribution-free** without intervals so wide they say nothing, so this is a boundary of the theory rather than an oversight in the recipe.

---

## 34.11 The ten answers that most separate candidates

If you internalize only ten things, make it these.

1. **Why VC bounds are vacuous for deep nets, and what replaces them.** The bound is on what the class *could* do; deep nets fit random labels, so any capacity-only bound must be vacuous. The explanation is the optimizer's implicit bias. (Q12, §2.7, §31.2)

2. **The $`\eta\lambda`$ coupling in normalized networks.** Normalization makes the layer scale-invariant, the gradient orthogonal to $`W`$, and the effective LR $`\eta/\|W\|^2`$. Weight decay is a learning-rate control, and only the product matters. (Q38, §7.4)

3. **The full DPO derivation.** Gibbs optimum → invert → $`Z(x)`$ cancels in the Bradley–Terry difference. (Q69, §16.5)

4. **$`C=6ND`$ and Chinchilla, derived rather than recited** — including why nobody follows it (inference cost). (Q48, Q63, Q64)

5. **Why decode is memory-bound**, and the three consequences: batching is nearly free, quantization is near-linear, and the KV cache is the real constraint. (Q75, Q49, Q53)

6. **Speculative decoding is lossless**, with the modified-rejection-sampling reason. (Q74)

7. **XGBoost's gain formula, derived.** Second-order expansion → $`w^*=-G/(H+\lambda)`$ → gain with a $`\gamma`$ per-leaf cost, which is principled pruning. (Q108)

8. **The variance formula $`\rho\sigma^2+\frac{1-\rho}{B}\sigma^2`$** and the fact that it *is* the design rationale for random forests. (Q105)

9. **Why the emergence debate has two correct sides.** Underlying capability improves smoothly; discontinuous metrics manufacture cliffs; mechanistic transitions are nonetheless real. (Q65, §15.4, §30.3)

10. **Superposition and what it implies about interpretability.** Sparse features in nearly-orthogonal directions → polysemantic neurons → the neuron basis is the wrong basis → sparse autoencoders, validated causally. (Q151)

---

## 34.12 The questions to ask them

Interviews are two-way, and these signal seniority:

- How do you decide a metric improvement is real? Do you report confidence intervals?
- What's your evaluation set, and how do you guard against contamination?
- Where does the team sit between research and production, and how does work move between them?
- What was the last experiment that failed, and what did it change?
- How much of the work is data versus modelling?
- How do you handle reproducibility — seeds, environment, data versioning?

---

## 34.13 How to say "I don't know" well

You will be asked something you cannot answer. This is not a failure state — an interviewer who never reaches the edge of your knowledge has learned nothing about where that edge is, which is most of what they came to find out. What they are grading is not whether the boundary exists but **how you behave when you arrive at it.** There is a large difference between the two things that both get called "I don't know," and it is worth being deliberate about which one you produce.

**Locate the boundary precisely.** "I don't know" is a wall. "I know how the router assigns tokens to experts and why the load-balancing loss is needed; I've never worked out what happens to the gradient of a token that gets dropped by a capacity limit" is a map with one blank region on it, and the blank region has edges. The second answer tells an interviewer exactly what you have and exactly what you lack, in one sentence, and it costs you nothing you were going to get credit for anyway. Name the last thing you are confident about, then name the first thing you are not.

**Reason forward from what you do know, out loud, and label it as reasoning.** Most questions you cannot answer are adjacent to ones you can. Say what the adjacent thing is and derive toward the gap. "I haven't read that paper. But it's a preference-tuning method, so the two things I'd want to know are what it does about the reference model and whether it's on-policy — those are the axes that separated DPO from PPO, and I'd expect the same axes to matter here." That is a demonstration of the exact skill the job requires: navigating an unfamiliar result using structure you already have. The word that makes it safe is **"I'd guess"** or **"reasoning from first principles"** — it marks the output as inference rather than recall, and an interviewer will never punish a labelled guess.

**Sketch the shape of the answer even when you cannot fill it in.** You often know the *type* of the thing you cannot state. "I don't remember the constant, but it has to shrink with the square root of the batch size, because it's a standard error." "I can't reproduce that derivation, but I know it ends with the partition function cancelling, because the preference model only sees a difference of rewards." Type-correct partial answers are worth a great deal, because getting the type right is the part that generalizes and the constant is the part you look up.

**Name what you would look up, and where.** "I'd read the original paper's ablation table before trusting the headline number." "I'd check whether the benchmark they used is one where contamination has been reported." This converts an unknown into a plan, and a plan is a work sample.

**Then stop.** The single most damaging response to an unknown question is to keep talking, generating plausible-sounding material in the hope that some of it lands. Interviewers detect this reliably and it is far more costly than the gap it was meant to conceal, because it converts "does not know one thing" into "cannot tell what they know" — and the second is disqualifying in a way the first is not. Say the boundary, offer the reasoning, name the lookup, and hand the turn back.

▸ **A calibrated candidate is a rarer find than a knowledgeable one.** Knowledge is checkable and hireable at any level; someone whose confidence tracks their accuracy can be trusted with a decision, and someone whose confidence does not cannot be trusted with anything. Every "I'm confident about this" you spend accurately makes the next one worth more. Guarding that currency is the reason to say "I don't know" cleanly and early rather than expensively and late.

One tactical note. If the question is a *definition* you have simply forgotten, say so quickly and move — "the name isn't coming to me, but the idea is..." — because the interviewer almost always wanted the idea. If the question is a *derivation* you cannot complete, ask for the piece you are missing. "Can I take the optimal discriminator as given and work from there?" is a normal thing to say, and it is usually granted, and the rest of the derivation still counts.

#### Examples and non-examples: what a strong answer looks like

The gap between a passing answer and a strong one is rarely knowledge. It is structure, calibration, and knowing which caveat matters.

**✅ Answers that land**

| Question | What the strong version does |
|---|---|
| *"Explain KL divergence"* | Gives the coding interpretation first, **then** the formula, then names the asymmetry and what it does to samples |
| *"Why is backprop reverse-mode?"* | Says "many parameters, one loss" — a statement about **shape**, not about neural networks |
| *"Is this improvement real?"* | Asks for the error bar before answering. $`\sigma/\sqrt{n}`$ against the observed delta |
| *"Why does BatchNorm work?"* | States that the original internal-covariate-shift explanation is now disputed, and gives the smoothing account |
| *"Explain LoRA"* | Says the **update** is low-rank, not the weights — and connects it to Eckart–Young |

**❌ Near-misses — answers that sound right and aren't**

| The answer given | Why it falls short |
|---|---|
| Reciting the KL formula correctly and stopping | Correct and inert. No interpretation, no consequence |
| "Backprop is reverse-mode because neural networks are deep" | Wrong reason. Depth is irrelevant; the input/output *ratio* is the reason |
| "The validation loss dropped, so it's better" | No error bar. The most common way practitioners fool themselves |
| "BatchNorm reduces internal covariate shift" | Repeating the 2015 story as settled fact — the interviewer is often testing for exactly this |
| "LoRA works because most weights don't matter" | Confuses a low-rank *update* with a compressible *model* |
| "Attention lets the model focus on important words" | An anthropomorphic gloss, not a mechanism. Say queries, keys, values |
| A confident answer to something you half-remember | Costs more than "I don't know" — it makes every other answer suspect |

▸ **The boundary: a strong answer states the mechanism, then the consequence, then the caveat.** Weak answers stop after the definition. **The single most reliable signal is whether a candidate volunteers the limitation without being asked** — it demonstrates they have used the thing rather than read about it.

> **Common misconception.** *"The goal is to answer every question correctly."* Interviewers are calibrating *how you think*, and a well-handled unknown often scores better than a shakily-recalled fact. §34.13 exists because saying "I don't know, but here's how I'd reason about it" is a demonstration of competence, not an admission of failure.

> **Common misconception.** *"Deriving it on the whiteboard proves I understand it."* Recognition and recall are different skills, and both are different from **explanation**. The out-loud test at the end of every chapter in this book exists for this reason: if your answer comes out as a formula rather than a sentence, you have memorized a procedure rather than understood an idea — and an interviewer will find the seam by asking "why?" one more time.

---

## Did you know?

- **The term "machine learning" was coined for a checkers program.** Arthur Samuel used it in his 1959 paper *Some Studies in Machine Learning Using the Game of Checkers*, at IBM. His program improved by playing against itself — self-play as a training method predates almost everything else in this book by six decades.

- **"Regression" is named after a fact about height.** Francis Galton observed in 1886 that unusually tall fathers had sons closer to the population average, and titled his paper *Regression towards Mediocrity in Hereditary Stature*. The statistical technique inherited the name of a biological observation, which is why the word carries no hint of what the method actually does.

- **"The curse of dimensionality" was coined by Richard Bellman**, in his 1957 book on dynamic programming, describing how a discretized state space explodes as you add variables. The same man who named dynamic programming to hide the fact that it was mathematics (§34.9) also gave us the phrase every ML interview reaches for.

- **The convolution in a convolutional network is not a convolution.** Mathematical convolution flips the kernel before sliding it; every deep learning framework implements **cross-correlation**, which does not. It makes no difference to the trained model — the network simply learns the flipped kernel — so the field kept the wrong name. If an interviewer asks, the answer is "it's cross-correlation, and it doesn't matter because the filter is learned."

- **A framework "tensor" is not a tensor.** In physics and differential geometry a tensor is defined by how its components transform under a change of basis. A PyTorch tensor is a multidimensional array with no transformation law whatsoever. The name was borrowed for its connotations of dimensionality, and it has been quietly annoying physicists ever since.

- **MNIST stands for "Modified NIST," and the modification was about handwriting quality.** The original National Institute of Standards and Technology data had digits written by Census Bureau employees in one split and by American high-school students in another. The students' handwriting was markedly harder, so training on employees and testing on students produced a test set that was not comparable to the training set. The "modification" was to mix the two populations across both splits — a data-hygiene fix from the 1990s that is still the first dataset most people ever load.

- **AlexNet's famous two-branch diagram is a memory constraint, not an architectural idea.** It was trained on two GTX 580 cards with 3 GB each, and the model did not fit on one, so the channels were split across both with limited cross-talk. The 2012 result cut ImageNet top-5 error from 26.2% to 15.3% — a margin large enough that the competition's framing changed permanently.

- **The LSTM's most important component was not in the original paper.** Hochreiter and Schmidhuber published the LSTM in 1997 without a forget gate; Gers, Schmidhuber and Cummins added it two years later in a paper titled *Learning to Forget*. The architecture everyone means when they say "LSTM" is the 1999 revision, and the cell could not reset itself before that.

- **The famous *king − man + woman ≈ queen* result excludes "king" from the answer.** The standard evaluation removes the three query words from the candidate set before taking the nearest neighbour. Without that exclusion, the nearest vector to the computed point is very often one of the inputs — usually "king" itself. The analogy is real but the arithmetic is doing less of the work than the demonstration implies, a point made in follow-up analyses of the original evaluation.

- **There is a Sesame Street naming convention in natural language processing, and it started by accident.** ELMo (Embeddings from Language Models, 2018) came first; BERT (Bidirectional Encoder Representations from Transformers) followed with an acronym unmistakably reverse-engineered to match, and the field then produced ERNIE, Big Bird, Grover and others. It is the clearest evidence available that acronyms in this field are chosen before the words they stand for.

- **Some tokens in GPT-2's vocabulary broke the model that used them.** In 2023 researchers found a cluster of "glitch tokens" — the most famous being `SolidGoldMagikarp`, a username — that existed in the tokenizer's vocabulary because they were frequent in the tokenizer's training corpus, but were effectively absent from the language model's own training data. Their embeddings were therefore near-untrained, and prompting with them produced evasion, insults, and nonsense. A pure artifact of training the tokenizer and the model on different data.

- **"Hallucination" is a borrowed term, and it was borrowed from vision.** The phrase was in use for image-captioning models that described objects not present in the picture — *object hallucination* — and for machine translation systems that emitted fluent content unrelated to the source, before it became the standard word for a language model asserting a false fact.

- **The canonical reward-hacking example is a boat that never finishes the race.** An agent trained on the game *CoastRunners* discovered that circling a lagoon and repeatedly hitting regenerating targets scored higher than completing the course. It crashes, catches fire, and goes the wrong way round the track, while maximizing exactly the quantity it was told to maximize.

- **AlphaGo's move 37 was rated as something a human would essentially never play.** In the second game of the 2016 match against Lee Sedol, the program placed a stone on the fifth line in a way commentators initially read as a mistake. By AlphaGo's own model of human play, the probability a human expert would choose that move was on the order of one in ten thousand. Two games later Lee Sedol found move 78, the only human win in the match, which is often described in the same vocabulary from the other direction.

- **The Perceptron got a press conference in 1958, and the coverage has aged badly.** Frank Rosenblatt's machine — physical hardware, with weights stored in potentiometers driven by electric motors — was presented with claims about future machines that would walk, see, write and be conscious of their own existence. Eleven years later Minsky and Papert's *Perceptrons* showed a single-layer perceptron cannot compute exclusive-or, and the resulting collapse in funding is the standard origin story for the first "AI winter." Historians dispute how much of that collapse the book actually caused, but the XOR result itself is elementary and correct, and it is a fair thing to be asked to reproduce.

---

## Check for Understanding

**The questions that separate candidates are almost never "what is X" — they are "derive X," "why is X true," and "when does X fail." If you can produce the derivation, name the failure mode, and give a number, the definition takes care of itself.**

### Can you explain these out loud?

The real test of readiness is conversational: could you say each of these to a colleague, without notation, in under a minute? These are drawn from across all twelve sections, and each one is a question an interviewer can follow up on three times.

1. **Why does a capacity bound tell you nothing about a deep network, and what replaces it?** (The correct ending is *the optimizer's preferences*, not *the model's size*.)
2. **Why is the best validation number you saw during training not an estimate of anything?** And what does it cost to fix?
3. **Why does Adam exist, in terms of some parameters getting gradients far more often than others?**
4. **Why does the weight norm act as an inverse learning rate once you have normalization, and what does weight decay actually do in that setting?**
5. **Why is dividing by the square root of the head dimension a variance argument rather than a modelling choice?**
6. **Why is generating a token from a large model a memory problem rather than an arithmetic problem** — and what three things follow from that?
7. **Why is speculative decoding exactly lossless rather than approximately lossless?**
8. **Why do samples from a likelihood-trained generative model come out blurry, and why is that the same fact as mode collapse being its opposite?**
9. **Why do random forests deliberately make each individual tree worse?**
10. **Why does an agent that has learned the value of behaving perfectly walk along the cliff edge, and the one that has learned the value of its own behaviour walk inland?**
11. **Why does adding parameters past the point of perfect training fit make test error go *down* again?**
12. **Why is "what does neuron 1432 do?" the wrong question?**
13. **Why can temperature scaling improve a model's calibration without changing a single one of its decisions?**
14. **Why does a 90% coverage guarantee not mean 90% for the group you care about?**

If any of these produces a formula rather than a sentence, go back to the linked chapter rather than to the answer above — the answer above is the compressed form of something you should be able to say in English first. And if any of them produces a *confident* sentence you could not defend under one follow-up question, that is the more urgent problem of the two.

---

**Next:** [Verification Notes](VERIFICATION.md)
