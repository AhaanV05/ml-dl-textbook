# Chapter 2 — Learning Theory & Generalization

> **Prerequisites:** Ch. 1 (probability, expectation, variance).
> **Payoff:** this is the chapter that makes double descent (Ch. 18) shocking rather than confusing. You cannot appreciate why deep learning broke the theory without knowing what the theory said.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar — or if you have never had $\Pr$, $\delta$, and $\epsilon$ explained as the three dials of a confidence statement — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; refer back as needed. Each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\mathcal{D}$ | "script D" | The true, unknown distribution your data comes from |
| $\mathcal{H}$ | "script H" | The **hypothesis class** — every model your architecture could possibly become |
| $h \in \mathcal{H}$ | "h in script H" | One specific model, i.e. one particular setting of all the weights |
| $\ell(h(x),y)$ | "ell of h-of-x and y" | The loss on **one** example. Script-$\ell$ is *per-example* |
| $R(h)$ | "the risk of h" | Average loss over **all data that could ever arrive** — what you want |
| $\hat R_n(h)$ | "R-hat-n of h" | Average loss over **the $n$ points you have** — what you can measure |
| $R^*$ | "R-star" | The lowest risk **any** function could achieve. The Bayes risk. A floor |
| $\hat h$ | "h-hat" | The model that empirical risk minimization actually picks |
| $\inf$ | "infimum" | A careful $\min$: the smallest value, or the value it creeps down to |
| $\sup$ | "supremum" | A careful $\max$: the largest value, or the value it creeps up to |
| $\lvert\mathcal{H}\rvert = M$ | "the size of script H" | How many distinct models the class contains |
| $\Pr(\cdot)$ | "the probability that" | A number between 0 and 1 |
| $\delta$ | "delta" | The **failure probability** you'll tolerate. $1-\delta$ is your confidence |
| $\epsilon$ | "epsilon" | The **error tolerance** — how wrong you're willing to be |
| $\mathbb{1}[\,\cdot\,]$ | "indicator" | 1 if the statement inside is true, 0 if false. A switch |
| $d_{\mathrm{VC}}$ | "the VC dimension" | The most points the class can label in **every** possible way |
| $\hat{\mathfrak{R}}_S(\mathcal{F})$ | "Rademacher complexity" | How well the class can fit **pure coin flips** |
| $\sigma_i \in \{\pm1\}$ | "sigma-i" | A **Rademacher variable**: a fair coin flip, written $+1$ or $-1$ |
| $\mathrm{KL}(Q\|P)$ | "KL of Q from P" | Extra bits to describe $Q$ when you had budgeted for $P$ |
| $\binom{n}{i}$ | "n choose i" | The number of ways to pick $i$ items out of $n$ |
| $\lesssim$ | "is at most, up to constants" | $\le$, ignoring factors nobody bothers to track |
| $\perp$ | "is independent of" | Knowing one tells you nothing about the other |
| $\forall$ | "for all" | True for every one of them, *simultaneously* |
| w.p. | "with probability" | Attached to a bound, it names how often the bound holds |

Three of these deserve a warning before you meet them.

- **$\ell$ versus $R$ versus $\hat R$.** $\ell$ is the loss on one example, $R$ is its average over the whole world, $\hat R_n$ is its average over your sample. Nearly every equation in this chapter is a statement about how far apart the last two can be.
- **$\sigma$ changes jobs mid-chapter.** In §2.2 it is a standard deviation ($\sigma^2$ = noise variance). In §2.5 it is a coin flip ($\sigma_i = \pm1$). Chapter 0's rule still works: squared means standard deviation, indexed-and-in-a-sum means something else.
- **$\epsilon$ and $\varepsilon$ are different variables here.** Straight $\epsilon$ is an error tolerance you choose; curly $\varepsilon$ in §2.2 is the random noise in the data. The book keeps them typographically distinct on purpose.

### Full forms for this chapter's abbreviations

| Short | Full form |
|---|---|
| ERM | Empirical Risk Minimization |
| VC | Vapnik–Chervonenkis (dimension) |
| PAC | Probably Approximately Correct |
| PAC-Bayes | Probably Approximately Correct, Bayesian version |
| KL | Kullback–Leibler (divergence) |
| MDL | Minimum Description Length |
| SVM | Support Vector Machine |
| SGD | Stochastic Gradient Descent |
| LR | Learning Rate |
| AdamW | Adam with decoupled **W**eight decay |
| RHS / LHS | Right-Hand Side / Left-Hand Side |
| i.i.d. | independent and identically distributed |
| CIFAR-10 | Canadian Institute For Advanced Research (a 10-class image dataset) |
| MNIST | Modified National Institute of Standards and Technology (handwritten digits) |
| ResNet | Residual Network |

---

## 2.1 The setup: empirical risk minimization

### The one-line idea

You want low error on data you've never seen. You can only measure error on data you have. ERM is the decision to optimize the thing you can measure and hope it transfers.

### The analogy

You're a chef hiring based on a tasting. The tasting score is *empirical risk*. How good the food will be over a year of service is *true risk*. A chef who memorized one dish perfectly will ace the tasting and fail the job. That's overfitting, and everything in this chapter is about bounding the gap between the tasting and the job.

### Formalism

Data $(x,y) \sim \mathcal{D}$, unknown. Hypothesis class $\mathcal{H}$ (e.g., all networks of a given architecture). Loss $\ell(h(x),y)$.

$$\underbrace{R(h) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(h(x),y)]}_{\text{true / population risk}} \qquad \underbrace{\hat R_n(h) = \frac1n\sum_{i=1}^n \ell(h(x_i),y_i)}_{\text{empirical risk}}$$

▸ ERM: $\hat h = \arg\min_{h\in\mathcal{H}} \hat R_n(h)$.

The **generalization gap** is $R(\hat h) - \hat R_n(\hat h)$. All of classical learning theory is the project of bounding this.

#### Reading the two risks in plain English

Two formulas, one idea. Take them symbol by symbol.

**$(x,y)\sim\mathcal{D}$.** The squiggle $\sim$ is read *"is drawn from"* — not "approximately". $\mathcal{D}$ is the process that generates examples: photographs and their labels, molecules and their energies, sentences and their next words. **You never see $\mathcal{D}$.** You only ever see samples that fell out of it. Everything in this chapter is an attempt to say something about an object nobody has access to.

**$\mathcal{H}$, the hypothesis class.** Read as *"the menu."* Fix an architecture — say a 3-layer network with a particular width — and $\mathcal{H}$ is the set of every function you get by choosing the weights differently. Each $h \in \mathcal{H}$ is one dish on the menu. Training is choosing a dish.

**True risk, $R(h) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(h(x),y)]$.** Read aloud: *"R of h equals the expectation, over pairs $(x,y)$ drawn from $\mathcal{D}$, of the loss between what $h$ predicts and the truth."* In English: **the average mistake this model would make across every example the world will ever hand you.** It is a single number attached to a single model.

**Empirical risk, $\hat R_n(h) = \frac1n\sum_{i=1}^n \ell(h(x_i),y_i)$.** The hat means *estimated from data* (Chapter 0, §0.6); the subscript $n$ says how many points went into it. The $\sum_{i=1}^n$ is a `for` loop over your dataset and the $\frac1n$ turns the total into an average. In English: **the average mistake on the examples you happen to own.**

▸ **The two formulas are identical except that $\mathbb{E}$ has been replaced by $\frac1n\sum$.** That single swap — a true average over the world, traded for an average over a sample — is the entire subject of this chapter, and arguably the entire subject of statistics.

> **Analogy.** $R(h)$ is a restaurant's real quality, averaged over every meal it will ever serve. $\hat R_n(h)$ is the average of the $n$ reviews on the internet. The reviews are all you can read; the quality is all you care about. And the reviews are not a neutral sample — which is exactly the wrinkle that makes learning theory hard.

**Now the sting, and it is subtle.** For a *fixed* model chosen before you looked at the data, $\hat R_n(h)$ is an unbiased estimate of $R(h)$ — average of $n$ independent draws, no bias, error bar $\sigma/\sqrt n$. Straightforward. But $\hat h = \arg\min_{h\in\mathcal{H}}\hat R_n(h)$ was **chosen by looking at the data**, specifically by looking for whichever model the data flattered most.

> **Analogy.** Roll 1,000 dice once each and keep the one that came up 6. That die is not a lucky die. Its measured average of 6.0 is not an estimate of its true average of 3.5 — it is an estimate of *the largest of a thousand noisy readings*. Picking the winner and then quoting the winner's score is a systematically optimistic procedure, and no amount of honesty in the individual measurements repairs it.

**Put numbers on it.** Suppose every model in $\mathcal{H}$ is  equally bad, with true risk $0.50$, and your empirical estimate of each has a standard error of $0.02$. With $M = 1$ model, your reported training loss averages $0.50$. With $M = 1{,}000$ models to choose the best of, the minimum of 1,000 draws from $\mathcal{N}(0.50, 0.02^2)$ sits around $0.50 - 3.2\times0.02 = 0.436$. You will report a training loss of $0.44$ and a test loss of $0.50$, and call the $0.064$ difference "overfitting". **Nothing overfit. You took a minimum over noise.**

▸ **The generalization gap $R(\hat h) - \hat R_n(\hat h)$ is the price of having searched.** The larger the menu you searched, the more the winner's score was flattered by luck, and the bigger the gap. Every bound in this chapter is a different way of charging you for the size of the search. (Chapter 3 §3.6 shows this same arithmetic wrecking a real validation curve, where the "search" is nothing more exotic than taking a minimum over epochs.)

**Why $\arg\min$ and not $\min$.** $\min_h \hat R_n(h)$ is the best *loss value* you achieved; $\arg\min_h \hat R_n(h)$ is the *model that achieved it*. You want the model — you cannot deploy a number. Chapter 0 §0.3 has the mountain-summit version of this distinction.

### The error decomposition

$$R(\hat h) - R^* = \underbrace{\big(R(\hat h) - \inf_{h\in\mathcal{H}} R(h)\big)}_{\text{estimation error}} + \underbrace{\big(\inf_{h\in\mathcal{H}} R(h) - R^*\big)}_{\text{approximation error}}$$

- **Approximation error** shrinks as $\mathcal{H}$ grows. (Bigger models can express more.)
- **Estimation error** *classically* grows as $\mathcal{H}$ grows. (Bigger models overfit more.)

The classical story says these trade off, and there's a sweet spot. Chapter 18 shows that the second claim is false for modern models. Hold that thought.

#### Unpacking the error decomposition

This equation looks like bookkeeping, and it *is* bookkeeping — but it is the bookkeeping that tells you which of your problems is fixable by which action.

**Decoding the pieces:**

- $R^*$ — the **Bayes risk**. The best any function whatsoever could do, including functions outside your architecture, including functions nobody could ever write down. It is not zero: if two identical-looking molecules have different measured energies, no function can get both right. $R^*$ is the world's own irreducible ambiguity.
- $\inf_{h\in\mathcal{H}} R(h)$ — read *"the infimum over h in script H of R of h."* **The best true risk available anywhere on your menu.** Infimum rather than minimum because the best value may only be approached, never attained (as $\|w\|\to\infty$, say). For intuition, read $\inf$ as $\min$ and $\sup$ as $\max$; the distinction is a technicality that almost never changes the meaning.
- The whole left side, $R(\hat h) - R^*$, is **your total regret**: how much worse your trained model is than the theoretical best possible.

**The trick used to build it — add and subtract the same thing.** Take $R(\hat h) - R^*$, insert $-\inf_h R(h) + \inf_h R(h)$ in the middle (which changes nothing, since it sums to zero), and regroup. That gives two differences instead of one. This "insert a zero, regroup" move appears again in §2.2's derivation, and it is the single most common manoeuvre in all of statistics.

**The two terms, in English:**

| Term | The question it answers | How to fix it |
|---|---|---|
| **Approximation error** $\inf_h R(h) - R^*$ | "Is the right answer even *on* my menu?" | Bigger model, better architecture |
| **Estimation error** $R(\hat h) - \inf_h R(h)$ | "Given that it's on the menu, did I find it?" | More data, better optimizer, regularization |

> **Analogy.** You want the best possible meal. **Approximation error** is how much worse the best dish on this restaurant's menu is than the best dish in the city — a fact about the restaurant, unaffected by how carefully you order. **Estimation error** is how much worse the dish you actually ordered is than the best dish on this menu — a fact about your ordering process, driven by how much information you had. Reading more reviews (more data) helps the second and does nothing at all for the first.

▸ **The two are fixed by opposite actions, which is why the trade-off exists.** Enlarge $\mathcal{H}$ and the approximation error can only go down — a bigger menu contains the old menu. But classically, enlarging $\mathcal{H}$ makes the estimation error go up, because a bigger menu is a bigger search, and a bigger search flatters its winner more (see the dice above). One term falls, one rises, so a minimum sits somewhere in the middle. **That is the entire content of "there is a sweet spot."**

**Put a number on it.** Fitting $f^*(x)=\sin(2\pi x)$ with a straight line: the approximation error is large and *no amount of data removes it* — a million points still leaves a line unable to bend. Fitting it with a degree-19 polynomial on 20 points: the approximation error is essentially zero, and the estimation error is enormous. §2.2's table measures exactly these two quantities.

**Why the caveat "*classically*" is doing so much work.** The claim "estimation error grows with $\lvert\mathcal{H}\rvert$" is not a theorem. It is a theorem *about the worst case* — about the largest gap achievable by any hypothesis in the class. Real optimizers do not return the worst case. They return a very particular, non-random, heavily-biased-toward-simple solution, and Chapter 18 is what happens when you take that seriously.

> **Where this came from.** Empirical risk minimization as a named principle comes from **Vladimir Vapnik and Alexey Chervonenkis**, working at the Institute of Control Sciences in Moscow in the 1960s and 1970s. Their foundational paper on uniform convergence appeared in Russian in 1968 and in English translation in 1971, and it went largely unread in the West for over a decade — partly language, partly the Cold War, partly that neural networks were out of fashion. Vapnik emigrated to the United States in 1990 and joined AT&T Bell Labs, where the theory finally met an application: the support vector machine (SVM), developed there with Bernhard Boser, Isabelle Guyon, and Corinna Cortes. Vapnik was fond of quoting the social psychologist Kurt Lewin's line that *"nothing is more practical than a good theory"* — a fair description of a career spent proving results about learning two decades before anyone had the computers to need them.

#### Examples and non-examples: overfitting

Everyone can define overfitting. Almost nobody can reliably *identify* it, because four different things produce the same surface symptom — a test number worse than a train number.

**✅  examples**

| Example | Why it qualifies |
|---|---|
| Degree-19 polynomial through 20 jittered points: train MSE $0.000$, test MSE $\approx 4.9$ | Redraw the jitter and the fitted curve changes completely. The model absorbed variation that will not recur |
| A decision tree grown until every leaf holds exactly one training example | Each leaf stores one label. A new point's prediction is decided by whichever training point happened to land nearby |
| A ResNet at 100% training accuracy on **randomly relabelled** CIFAR-10, 10% on test | There is no signal to learn — every bit of the fit is memorized noise, by construction |
| Trying 50 hyperparameter configurations and reporting the best validation score as your result | The reported number is a **maximum over 50 noisy draws**. This is overfitting *to the validation set*, and it is the most common kind in practice |
| $k$-nearest-neighbours with $k=1$ on measurement-noisy data | Training error is exactly zero by definition (each point is its own neighbour) and carries no information at all |

**❌ Near-misses — look like overfitting, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Train loss $0.31$, test loss $0.34$, both still falling | A small, stable gap with both curves improving. Nothing is being absorbed that isn't also generalizing | Healthy training |
| 100% train accuracy, 96% test accuracy, both flat for 20 epochs | Zero training error on its own is not overfitting. The model interpolates *and* generalizes | Benign interpolation (Ch. 18, Ch. 30) |
| A 175B-parameter model beating the 7B one on held-out data | Capacity went up and the gap went *down* — the opposite of the predicted direction | Scaling (Ch. 15) |
| Test loss far above train loss because the test set came from a different hospital | The gap is caused by the test distribution, not by fitting training noise. More training data will not close it | Distribution shift (Ch. 33) |
| Validation loss ticks up for 200 steps right after a learning-rate warm restart, then falls below its old value | The optimizer was deliberately kicked out of a basin. No memorization happened in 200 steps | Transient optimization dynamics (Ch. 5) |
| Both train and test error are terrible | Nothing was fit too closely — nothing was fit at all | Underfitting (high approximation error) |

▸ **The boundary:** overfitting is not "zero training error" and not "a bad test number." It is **test error rising because the model absorbed variation that will not recur in fresh data.** The operational test: hold the data fixed, increase the fit, and watch whether the *gap* grows. If the gap is stable while both losses fall, you are not overfitting no matter how many parameters you have; if the gap grows while train error keeps falling, you are, no matter how few.

> **Common misconception.** *"More parameters always overfit."* Parameter count is neither necessary nor sufficient. Not necessary: the one-parameter classifier $\mathbb{1}[\sin(\omega x)>0]$ can produce any labeling of any number of points — infinite capacity from a single knob. Not sufficient: GPT-scale models have more parameters than training tokens and generalize better than their smaller siblings, and §2.9 ranks parameter count **dead last** among the seven levers that control generalization. **The reason this belief is so tempting is that it is exactly true inside the one family everyone meets first** — polynomials of increasing degree, where degree $d$ contributes $d+1$ coefficients *and* $d+1$ units of wiggliness at the same time. In that family, count and capacity are the same dial. In almost every family you will actually use, they have come apart. Chapter 30's double descent curve is what happens when you keep turning the count dial past the point where the folklore said the world ends.

> **Common misconception.** *"Approximation error and estimation error are both fixed by better training."* Only the second one is. Approximation error is a fact about the *menu* — the best function your architecture can express, compared with the best function that exists. Training harder, longer, or with a better optimizer cannot move it by a single decimal, because it is defined by an infimum over the whole class, taken before any data arrives. Fit a straight line to $\sin(2\pi x)$ with ten points or ten billion and the approximation error is identical. **The belief is tempting because in practice you never observe the two terms separately** — you see one training curve, and a stubbornly high loss looks the same whether the answer isn't on your menu or you failed to find it. The diagnostic that separates them: if your model cannot even fit the *training* set, the problem is approximation; if it fits training beautifully and tests poorly, the problem is estimation.

---

## 2.2 The bias–variance decomposition, derived

### The claim

For squared loss and a *fixed* test point $x$, with randomness over the draw of the training set $S$:

▸ $$\mathbb{E}_S\big[(y - \hat f_S(x))^2\big] = \underbrace{\big(\bar f(x) - f^*(x)\big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat f_S(x) - \bar f(x))^2\big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

where $\bar f(x) = \mathbb{E}_S[\hat f_S(x)]$ is the average prediction over training sets, $f^*$ is the true regression function, and $y = f^*(x)+\varepsilon$ with $\mathbb{E}[\varepsilon]=0$, $\mathrm{Var}(\varepsilon)=\sigma^2$.

#### Reading the bias–variance decomposition in plain English

The hardest thing about this formula is not the algebra — it is understanding **what is random**. Get that right and the rest follows.

**What is random here is the training set, not the test point.** The subscript on $\mathbb{E}_S$ says so: $S$ is the training set, and $\mathbb{E}_S[\cdot]$ means *"imagine collecting a fresh training set of the same size, retraining from scratch, and repeating that forever; average the result."* The test point $x$ is nailed down. You are asking: **at this one location, how much does my prediction wobble as the data I trained on changes?**

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $S$ | "S" | One training set — one draw of $n$ points from $\mathcal{D}$ |
| $\hat f_S$ | "f-hat-sub-S" | The model you get **after training on $S$**. Different $S$, different model |
| $f^*(x)$ | "f-star of x" | The truth at $x$. The best any function could say |
| $\bar f(x)$ | "f-bar of x" | The **average prediction across all possible training sets** |
| $\varepsilon$ | "epsilon" | The noise in the label: $y = f^*(x) + \varepsilon$ |
| $\sigma^2$ | "sigma squared" | How big that noise is, on average, squared |
| $\mathbb{E}_S[\cdot]$ | "expectation over S" | Average over the infinity of training sets you could have had |

Note that $\bar f$ is a **thought experiment, not a model you can build.** Nobody has infinitely many training sets. It exists to make the algebra split cleanly, and the practical stand-in for it — retrain on resampled data and average — is exactly bagging (Chapter 23) and exactly the bootstrap (Chapter 3).

> **Analogy — three ways a rifle misses.** You are shooting at a target.
> - **Bias** is the sights being misaligned: every shot lands 4 cm left, consistently. Averaging a thousand shots does not help, because they are all wrong in the same direction. You need a different rifle.
> - **Variance** is a shaky grip: shots scatter widely but centre on the bullseye. Averaging *does* help — that is why shooting from a rest, or averaging an ensemble, works.
> - **Noise** is wind you cannot see or predict. It moves every bullet, it is nobody's fault, and no rifle and no grip removes it.
>
> **These are three  different failures, and they have three  different fixes.** That is the whole reason the decomposition is worth knowing.

**Put real numbers in.** Fix a test point where the truth is $f^*(x) = 2.0$ and the label noise has $\sigma = 0.3$. You train on five different datasets and your model predicts $1.5,\ 1.7,\ 1.6,\ 1.4,\ 1.8$.

- $\bar f(x) = (1.5+1.7+1.6+1.4+1.8)/5 = 1.60$.
- $\mathrm{Bias}^2 = (1.60 - 2.00)^2 = 0.16$.
- $\mathrm{Variance} = \frac15\big[(-0.1)^2+(0.1)^2+(0)^2+(-0.2)^2+(0.2)^2\big] = 0.020$.
- $\mathrm{Noise} = \sigma^2 = 0.09$.
- Expected squared error $= 0.16 + 0.020 + 0.09 = 0.27$.

▸ **Read that split as a to-do list.** Bias contributes $0.16$ — by far the largest piece — so this model is *systematically* low, and collecting more data will not fix it. You need a more expressive model. Had the numbers come out the other way, more data or an ensemble would be the move. **The decomposition converts "my model is bad" into "my model is bad *in this specific way*," which is the difference between debugging and guessing.**

**Why squared, and why it splits at all.** Squaring is what makes the cross-term vanish. The middle term of every expansion below contains $\mathbb{E}_S[\bar f - \hat f]$, which is zero *by the definition of $\bar f$ as the average*. Squared loss is the unique loss where "deviation from the mean" and "the mean itself" are perfectly separable in this way — which is the same reason variances add for independent variables, and the same reason $\ell_2$ is the natural geometry of Pythagoras. Change to absolute error or cross-entropy and the clean split disappears.

### The derivation (do this once, it's short)

Write $\hat f = \hat f_S(x)$, $f^* = f^*(x)$.

$$
\mathbb{E}_{S,\varepsilon}\big[(y-\hat f)^2\big] = \mathbb{E}\big[(f^* + \varepsilon - \hat f)^2\big]
$$

Expand, using $\mathbb{E}[\varepsilon]=0$ and $\varepsilon \perp \hat f$:

$$= \mathbb{E}[(f^*-\hat f)^2] + 2\underbrace{\mathbb{E}[\varepsilon]}_{0}\mathbb{E}[f^*-\hat f] + \mathbb{E}[\varepsilon^2] = \mathbb{E}_S[(f^*-\hat f)^2] + \sigma^2$$

Now insert $\pm\bar f$ into the first term:

$$\mathbb{E}_S[(f^* - \bar f + \bar f - \hat f)^2] = (f^*-\bar f)^2 + 2(f^*-\bar f)\underbrace{\mathbb{E}_S[\bar f - \hat f]}_{=0} + \mathbb{E}_S[(\bar f - \hat f)^2]$$

$$= \mathrm{Bias}^2 + \mathrm{Variance}. \qquad \blacksquare$$

#### Following the derivation, line by line

Four lines, two tricks, and both tricks are used everywhere else in the book. Worth the ten minutes.

**Notation first.** $\mathbb{E}_{S,\varepsilon}$ means *"average over both sources of randomness"* — which training set you drew, and which noise landed on this test label. $\varepsilon \perp \hat f$ reads *"epsilon is independent of f-hat"* — the noise on **this** test label had no influence on the model, because the model was fit on a *different* set of points. That independence is the load-bearing assumption of the whole derivation. (If your test point leaked into training, it fails, and so does everything downstream. This is the mathematics behind "never touch the test set.") $\blacksquare$ at the end is just a full stop meaning "proof over."

**Line 1 — substitute the truth.** $y = f^* + \varepsilon$, so $(y - \hat f)^2 = (f^* + \varepsilon - \hat f)^2$. Nothing has happened yet; we have only written $y$ in terms of what generated it.

**Line 2 — expand, and watch two of three terms simplify.** Group as $\big[(f^*-\hat f) + \varepsilon\big]^2 = (f^*-\hat f)^2 + 2\varepsilon(f^*-\hat f) + \varepsilon^2$. Take expectations of each piece:

- $\mathbb{E}[(f^*-\hat f)^2]$ — keep it.
- $2\,\mathbb{E}[\varepsilon]\,\mathbb{E}[f^*-\hat f]$ — the factorization into two separate expectations is *only legal because of independence*, and then $\mathbb{E}[\varepsilon]=0$ kills it. **Zero.**
- $\mathbb{E}[\varepsilon^2] = \mathrm{Var}(\varepsilon) = \sigma^2$, since $\mathbb{E}[\varepsilon]=0$ makes the variance and the mean square the same thing.

▸ **The noise term $\sigma^2$ has now detached itself and will never interact with anything again.** That is why it is called *irreducible*: the algebra sets it aside on line 2 and no later choice of model can reach it.

**Line 3 — the add-and-subtract trick.** Write $f^* - \hat f = (f^* - \bar f) + (\bar f - \hat f)$. We inserted $-\bar f + \bar f$, which is zero, so the quantity is unchanged — but it is now split into *"how far the average model is from the truth"* plus *"how far this model is from the average model."* Square that sum and, again, three terms appear:

- $(f^*-\bar f)^2$ — no $S$ in it at all, so the expectation passes straight through. This is $\mathrm{Bias}^2$.
- $2(f^*-\bar f)\,\mathbb{E}_S[\bar f - \hat f]$ — and $\mathbb{E}_S[\hat f] = \bar f$ *by definition*, so the bracket is exactly zero. **This is the step the whole derivation is built around.**
- $\mathbb{E}_S[(\bar f - \hat f)^2]$ — the average squared deviation of the model from its own average. This is $\mathrm{Variance}$.

▸ **Both cross-terms died for the same reason: something averaged to zero.** Once you notice that, the derivation is not four lines of algebra, it is one idea applied twice — *split a difference around a convenient midpoint, and the cross-term evaporates because deviations from an average average to nothing.*

> **Analogy.** Measure how far a dart is from the bullseye by first walking to the *centre of the dart cluster*, then from there to the bullseye. The trip splits into two legs, and the legs are perpendicular — so their lengths combine by Pythagoras with no interaction term. Bias is the leg from cluster-centre to bullseye; variance is the average leg from each dart to the cluster-centre. **The reason there is no cross-term is that the two legs are at right angles**, which is a geometric restatement of "deviations from the mean average to zero."

### Reading it

- **Bias**: are you systematically wrong even with infinite data from this hypothesis class? A linear model fitting a sine wave has bias.
- **Variance**: how much does your fitted function jiggle when you resample the training set? A degree-20 polynomial on 20 points has enormous variance.
- **Noise $\sigma^2$**: irreducible. Also called the Bayes error. **In a diffusion model this is $H(p_{\text{data}}\mid t)$** — the part of the data that is  ambiguous at noise level $t$. No model reaches below it.

### Important caveats people get wrong

1. **This decomposition is specific to squared loss.** For 0-1 loss and cross-entropy there are analogous decompositions but they don't split as cleanly, and the "variance" term can be *helpful* (averaging over an ensemble can push a point across a decision boundary in the right direction).
2. **The classical U-shaped curve is a claim about how bias and variance move with capacity, not a theorem.** The decomposition itself is exact and always true. The U-shape is empirical folklore that turns out to be a special case (Ch. 18).
3. Variance is measured over *training set resampling*, which is precisely what the bootstrap estimates (Ch. 3). That's not a coincidence — bagging works by driving the variance term down.

#### Why those three caveats matter more than the formula

**Caveat 1, concretely.** For classification with 0–1 loss, an *ensemble* can be better than every member of the ensemble — a thing squared loss cannot do. Suppose three classifiers each get a given example right 60% of the time and err independently. Majority vote is right whenever at least two are right: $3(0.6)^2(0.4) + (0.6)^3 = 0.432 + 0.216 = \mathbf{0.648}$. **Variance became your friend**: the disagreement among members is what let the majority land on the correct side of the decision boundary. Under squared loss variance is a pure cost, always. Under 0–1 loss it can pay. Do not carry squared-loss intuitions into a classifier without checking.

**Caveat 2, stated bluntly.** The equation is a *theorem* — exact, unconditional, no assumptions beyond the setup. The **U-shaped curve is a picture**, and pictures are not theorems. Every textbook prints them next to each other, which trains people to believe they have the same status. They do not. The decomposition survives Chapter 18 completely intact; the U-shape does not.

**Caveat 3 — the connection to Chapter 3, which is the practical one.** "Variance over training-set resampling" is not a metaphor: it is literally a quantity you can estimate, by resampling your training set and retraining. That is the bootstrap. And **bagging** — bootstrap aggregating — is the observation that if you can *measure* the variance term by resampling, you can also *shrink* it by resampling and averaging. Averaging $B$ models with pairwise correlation $\rho$ leaves you variance $\sigma^2\left(\rho + \frac{1-\rho}{B}\right)$; at $\rho = 0.6$ and $B = 100$ that is $0.604\sigma^2$, not $0.01\sigma^2$.

▸ **Note where the floor comes from: $\rho$, not $B$.** No number of correlated models beats the correlation. That is why random forests inject *extra* randomness (random feature subsets) that hurts each individual tree — the point is to buy a lower $\rho$, and a lower $\rho$ is worth more than a better tree. The identical formula reappears in Chapter 3 §3.4 explaining why leave-one-out cross-validation is noisy, and again in §3.6 explaining why 16 validation batches don't average away as much as you'd hope.

> **Where this came from.** The bias–variance trade-off entered machine learning through a 1992 *Neural Computation* paper, "Neural Networks and the Bias/Variance Dilemma," by **Stuart Geman, Elie Bienenstock, and René Doursat**. Their conclusion was pessimistic: they argued that flexible, non-parametric methods such as neural networks face a variance problem so severe that learning anything  complex would require impractical quantities of data, and that useful systems would have to come loaded with strong built-in structure instead. It was a careful, correct-for-its-time argument that has aged into one of the most instructive wrong predictions in the field — and Chapter 18 is, in effect, the explanation of what they could not have known.

### A number

Consider fitting polynomials of degree $d$ to $n=20$ noisy points from $f^*(x)=\sin(2\pi x)$, $\sigma=0.3$.

| $d$ | Bias² | Variance | Total (+ $\sigma^2 = 0.09$) |
|---|---|---|---|
| 1 | 0.31 | 0.01 | 0.41 |
| 3 | 0.02 | 0.03 | 0.14 |
| 9 | 0.001 | 0.19 | 0.28 |
| 19 | 0.000 | 4.8 | 4.9 |

At $d=19$ (= $n-1$, exact interpolation) the variance explodes. **This is the interpolation threshold.** Classical theory stops the story here. Chapter 18 continues it past $d = 20$, where something surprising happens.

#### What the polynomial table is actually showing

**The experiment, stated without symbols.** Take the wave $\sin(2\pi x)$. Sample 20 points from it and add a little random jitter to each ($\sigma = 0.3$, so each point sits about 0.3 above or below the true curve). Now fit a polynomial of degree $d$ through those points and see how close the fitted curve is to the real wave. Repeat with fresh jitter, many times. Bias² is how wrong the *average* fitted curve is; variance is how much the fitted curves disagree with each other.

**Why the degree $d$ is the capacity dial.** A degree-$d$ polynomial has $d+1$ free coefficients — $d+1$ knobs. Degree 1 is a straight line (2 knobs), degree 3 can have two bends, degree 19 has 20 knobs for 20 data points. **When knobs equal data points, the fit passes exactly through every point.** That is what "interpolation" means and why $d = n-1 = 19$ is the special column.

**Read the table as a story in four acts:**

| $d$ | What the curve looks like | What's failing |
|---|---|---|
| 1 | A straight line through a wave | **Bias 0.31** — hopeless shape. It cannot bend, so it is wrong everywhere in the same way, every time |
| 3 | A gentle S that tracks the wave | Nearly nothing. Bias down 15×, variance still tiny. **The sweet spot** |
| 9 | Tracks the wave but ripples | Bias gone (0.001), but variance up 6× — it is now fitting jitter |
| 19 | Threads every point exactly | **Variance 4.8** — 50× worse total error than $d{=}3$ |

▸ **Watch the two columns move in opposite directions, and watch the total find a floor.** Bias falls $0.31 \to 0.02 \to 0.001 \to 0.000$, monotonically. Variance rises $0.01 \to 0.03 \to 0.19 \to 4.8$, monotonically. The total is their sum plus the fixed $0.09$, and it bottoms out at $d = 3$. **This table *is* the U-shaped curve, in numbers rather than in a drawing.**

**Why the last row explodes so violently.** A degree-19 polynomial forced through 20 jittered points has to make wild excursions between them to land on all of them. Nudge one point by 0.3 and the curve does not shift by 0.3 — it re-shapes globally, swinging by many units between neighbouring samples. That is the $4.8$: **near-perfect agreement on the data, near-total disagreement about everywhere else.**

> **Analogy.** Twenty pins are set in a board and you must lay a flexible steel batten so it touches all of them. A short, stiff batten (low $d$) touches few pins but sits smoothly. A very long, floppy batten touches every pin — by whipping violently up and down between them. **Passing through every data point is not a virtue; it is a constraint, and something has to absorb the strain.**

**Note the $0.09$ floor.** Every row's total includes $\sigma^2 = 0.09$, and the best row achieves $0.14$ — so the *best possible* model here is only $0.05$ better than the noise. That is the practical shape of most real problems: the headroom is much smaller than the raw error, which is why estimating the floor before optimizing (Chapter 1, §1.5) saves so much time.

▸ **And now the cliffhanger, precisely.** Classical theory says the story ends at $d = 19$ because there is nothing to say past it: at $d = 20$ or $d = 500$ there are infinitely many polynomials through the 20 points, so "the fit" isn't even well-defined. Chapter 18's move is to *define* it — among all interpolating solutions, take the one your optimizer actually finds, which is typically the smallest-norm one — and then the variance comes back **down**. **The classical curve doesn't stop at the interpolation threshold because the world ends there; it stops because the question was ill-posed, and modern optimizers answer it implicitly.**

#### Examples and non-examples: bias and variance

Both words are badly overloaded — "bias" has at least four unrelated meanings in machine learning, and "variance" has three. The decomposition uses one specific sense of each, and it is defined entirely by **what you average over: fresh draws of the training set.**

**✅  examples**

| Example | Which term, and why it qualifies |
|---|---|
| Straight line fit to $\sin(2\pi x)$: bias² $=0.31$, variance $=0.01$ | **Bias.** Average the fitted line over a thousand resampled datasets and it is *still* a line — wrong in the same way every time. Averaging cannot rescue it |
| Degree-19 polynomial on 20 points: bias² $=0.000$, variance $=4.8$ | **Variance.** The average of the fits is essentially correct; the individual fits disagree violently with each other |
| The $\sigma^2 = 0.09$ appearing in every row of the table | **Noise.** Identical at $d{=}1$ and $d{=}19$, because it is a property of the data-generating process, not of anything you chose |
| 1-NN versus 15-NN on the same noisy dataset | The dial in miniature: $k=1$ is near-zero bias and high variance; raising $k$ smooths over neighbours, buying bias and selling variance, monotonically |
| Bagging 100 trees drops test error but each individual tree is unchanged | Diagnostic proof that the error was **variance** — averaging only ever attacks that term |

**❌ Near-misses — the word is right, the concept isn't**

| Looks like bias or variance | Why it isn't | What it actually is |
|---|---|---|
| The $b$ in $Wx + b$ | An additive offset parameter, fitted like any other | A bias *term*. Shares a name with the decomposition and nothing else |
| A model that scores worse for one demographic group | A statement about groups within one fitted model, with no resampling anywhere | Fairness bias — an important, entirely unrelated sense |
| The spread of a classifier's output probabilities across the 10 classes | Computed from a single fixed model on a single input | Predictive entropy / confidence (Ch. 33) |
| Test error changes by $\pm 0.4\%$ across five random seeds, so "the model is high-variance" | The expectation in the theorem is over resampled *training sets*, not over seeds with the data held fixed | Optimization / initialization variance — real, measurable, and a different quantity |
| A U-shaped curve as you widen a modern network | The classical U is a claim about capacity *below* the interpolation threshold; past it the curve descends again | Double descent (Ch. 30) |
| Applying the exact bias²$+$variance$+$noise split to a classifier's 0–1 error | The clean additive split is a theorem **about squared loss**. Under 0–1 loss, variance can be *helpful* — see the majority-vote arithmetic in the caveats above | An analogy, not an identity |

▸ **The boundary:** bias and variance are defined by an expectation over **fresh draws of the training set $\mathcal{D}$**. If the quantity you are describing does not involve re-drawing $\mathcal{D}$ and retraining, it is not a term in this decomposition — whatever it is called.

> **Common misconception.** *"The bias–variance trade-off proves that a bigger model must eventually get worse."* The **decomposition** is a theorem: exact, unconditional, still true in every modern network. The **U-shaped curve** is a picture summarizing how the two terms happened to move in the model families of the 1990s. Textbooks print them on the same page, which trains readers to give them the same authority. They do not have it. Widen a network past the interpolation threshold and test error falls again — the variance term  comes back down, because among the infinitely many interpolating solutions the optimizer keeps choosing a smooth one. **The belief is tempting because the U-curve is drawn in every introductory course and almost never labelled with its domain of validity**, so it gets remembered as a law of nature rather than as an empirical observation about polynomials.

---

## 2.3 Concentration: why finite samples say anything at all

### Hoeffding's inequality

For independent $X_i \in [a,b]$ with mean $\mu$:

▸ $$\Pr\left(\left|\tfrac1n\sum X_i - \mu\right| \ge \epsilon\right) \le 2\exp\!\left(\frac{-2n\epsilon^2}{(b-a)^2}\right)$$

**Reading:** the probability of being wrong by $\epsilon$ decays *exponentially in $n$* and in $\epsilon^2$. Invert it: with probability $1-\delta$,

$$\left|\hat R_n(h) - R(h)\right| \le (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}$$

**Number:** for a bounded loss in $[0,1]$, $n=1024$ samples, $\delta=0.05$: the bound is $\sqrt{\log(40)/2048} = \sqrt{3.69/2048} = 0.042$. So a single evaluation on 1,024 samples pins the true loss to within $\pm 0.042$ **at 95% confidence, worst case.**

▸ **This is directly relevant to you.** Your validation uses ~1,024 molecules. Hoeffding's worst-case bound is $\pm0.042$; the improvement you measured was $1.556 \to 1.524 = 0.032$. The improvement is *smaller than the worst-case error bar on a single measurement.* Chapter 3 does the sharper, variance-based version, but the order of magnitude is the point: **this is an experiment operating at the edge of its own measurement precision.**

#### Reading Hoeffding's inequality in plain English

Read the whole thing aloud first: *"The probability that the sample average differs from the true mean by at least epsilon is at most two, times e to the minus two-n-epsilon-squared over the range squared."*

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $X_i$ | "X-sub-i" | One measurement. Here: the loss on one validation example |
| $[a,b]$ | "the interval a to b" | The range each measurement is guaranteed to fall in |
| $\mu$ | "mu" | The true mean you are trying to learn |
| $\tfrac1n\sum X_i$ | "the sample average" | What you actually computed |
| $\lvert\,\cdot\,\rvert$ | "the absolute value of" | Distance, sign discarded — wrong high or wrong low, both count |
| $\epsilon$ | "epsilon" | How wrong you're asking about |
| $\Pr(\cdot) \le \cdot$ | "the probability is at most" | An upper bound on bad luck |
| $2\exp(\cdot)$ | "two times e to the" | The 2 is because there are two ways to be wrong: too high, too low |

**Where each part of the exponent comes from.**

- **$n$ in the exponent** — more samples, exponentially less chance of being badly off. This is stronger than it sounds; compare with Chebyshev's inequality, which only gives you $1/n$ decay.
- **$\epsilon^2$ in the exponent** — asking to be twice as accurate costs you *four times* in the exponent. This is the $\sqrt n$ law wearing different clothes.
- **$(b-a)^2$ in the denominator** — the wider each individual measurement can swing, the weaker the guarantee. Bounded range is the *only* assumption Hoeffding needs: no normality, no known variance, nothing about the shape of the distribution at all.

> **Analogy.** You flip a possibly-biased coin $n$ times to estimate its bias. Hoeffding says the chance your estimate is off by more than $\epsilon$ shrinks like $e^{-2n\epsilon^2}$ — and it says this **without you knowing anything about the coin.** You do not need to assume the flips are normally distributed (they obviously aren't — each one is 0 or 1). You only need to know that no single flip can be worth more than 1. That "no single observation can dominate" is the entire content of the boundedness assumption, and it is why a *heavy-tailed* metric breaks Hoeffding: one observation *can* dominate.

**Work the numbers, twice.**

*Forward direction* — with $n = 1024$, $\epsilon = 0.042$, losses in $[0,1]$:
$$2\exp\!\left(\frac{-2(1024)(0.042)^2}{1}\right) = 2\exp(-3.61) = 2(0.027) = 0.054 \approx 5\%$$
So: *"there's about a 5% chance my measured loss is off by more than 0.042."*

*Inverted direction* — fix $\delta = 0.05$ and solve for $\epsilon$ to get the confidence-interval form. $\sqrt{\log(2/0.05)/2048} = \sqrt{3.689/2048} = \sqrt{0.0018} = 0.042$. **Same number, read as a question instead of an answer.** The forward form asks "how likely am I to be this wrong?"; the inverted form asks "how wrong might I be, at this confidence?" In practice you always want the second.

**Now scale it, because the scaling is the lesson.** How many samples to pin the loss to $\pm 0.01$ at 95%?

$$n = \frac{\log(2/\delta)}{2\epsilon^2} = \frac{3.689}{2(0.0001)} = 18{,}445$$

▸ **Going from $\pm0.042$ to $\pm0.01$ — a 4× tighter bound — costs 18× the data.** Halving an error bar costs 4× the samples, and it costs that *every single time*. There is no regime where this gets easier. This is the same $\sigma/\sqrt n$ wall from Chapter 1 §1.3.1, and it is the reason nobody can cheaply distinguish two models that differ by 0.3%.

**One honest caveat about the $\pm0.042$ figure.** Hoeffding is a **worst-case** bound: it assumes the per-example losses could be as adversarial as their range allows. Real losses have variance far below the worst case, so the true error bar is usually several times smaller. Bernstein's and Bennett's inequalities keep the exponential decay but replace $(b-a)^2$ with the actual variance, which is much tighter when the variance is small. Chapter 3 does this properly with the bootstrap. **Hoeffding's value is not precision — it is that it holds no matter what, with no assumptions to check.**

> **Where this came from.** **Wassily Hoeffding** published the inequality in 1963 in the *Journal of the American Statistical Association*, in a paper titled simply "Probability Inequalities for Sums of Bounded Random Variables." He spent most of his career at the University of North Carolina at Chapel Hill and is at least as well known for founding the theory of **U-statistics** in 1948. The inequality has ancestors: **Sergei Bernstein** proved closely related exponential bounds in the 1920s, and **Herman Chernoff** published the bounding technique now universally called the "Chernoff bound" in 1952 — a name Chernoff himself has repeatedly disclaimed, saying in later interviews that the key idea was suggested to him by his colleague Herman Rubin and that the bound should really carry Rubin's name. It is a rare case of a scientist campaigning, unsuccessfully, *against* having something named after him.

### Union bound and the first real generalization bound

For a *finite* hypothesis class $|\mathcal{H}| = M$, apply Hoeffding to each $h$ and union-bound:

$$\Pr\left(\exists h: |\hat R_n(h)-R(h)| \ge \epsilon\right) \le 2M e^{-2n\epsilon^2}$$

Set the RHS to $\delta$ and solve:

▸ $$R(h) \le \hat R_n(h) + \sqrt{\frac{\log M + \log(2/\delta)}{2n}} \quad \forall h\in\mathcal{H},\ \text{w.p. } 1-\delta$$

**Read the shape:** generalization gap $\sim \sqrt{\frac{\text{complexity}}{n}}$. Every classical bound has this form. $\log M$ is "bits needed to describe the hypothesis" — a **description-length** measure. This connects directly to PAC-Bayes and MDL below.

**Number:** a network with $p$ parameters at 32-bit precision has $M \le 2^{32p}$, so $\log M \le 32p\log 2 = 22p$. For $p = 10^7$ and $n=10^5$: bound $= \sqrt{2.2\times10^8/2\times10^5} = \sqrt{1100} = 33$. The bound says "your error is at most your training error plus 33," on a loss bounded by 1. **The bound is vacuous.** This is not a small technical problem — it is *the* problem with classical theory applied to deep nets.

#### The union bound, decoded — and why the result goes vacuous

**What the union bound says.** In one line: *"the chance that at least one of several bad things happens is at most the sum of their individual chances."* Formally $\Pr(A_1 \cup A_2 \cup \cdots) \le \sum_k \Pr(A_k)$. It is almost embarrassingly simple — you are just refusing to subtract the overlaps — and it is one of the two or three most-used tools in all of theoretical computer science.

**Decoding the symbols:**

- $\lvert\mathcal{H}\rvert = M$ — the class has exactly $M$ models in it. Finite. (Real parameter spaces are continuous and therefore infinite; §2.4 fixes that.)
- $\exists h$ — read *"there exists an h such that."* The bad event is *"**some** model in my class has a misleading training score."*
- $\forall h \in \mathcal{H}$ — read *"for all h."* The conclusion holds for **every** model at once, which is the point.

▸ **Why you must bound all $M$ at once, and not just the one you picked.** This is the step everyone skips and it is the crux. You did not choose $\hat h$ in advance — you chose it *because* its training score was low. So a guarantee about "any fixed model" is worthless; you need a guarantee that no model *anywhere* on the menu got a badly misleading score, because the selection procedure hunts precisely for the ones that did. **Uniform convergence is the tax you pay for the freedom to look before choosing.**

> **Analogy.** One weather forecaster being wrong by 10 °C is a 1-in-1000 event. But if you consult 1,000 forecasters and quote the one whose prediction you liked best, the chance *someone* was off by 10 °C is now roughly certain — and you have specifically gone looking for them. To trust the winner, you must first bound the chance that *anyone* was badly wrong. That is the union bound, and $M$ is how many forecasters you consulted.

**Where the $\log M$ comes from.** Set $2Me^{-2n\epsilon^2} = \delta$ and solve for $\epsilon$. Take logs: $\log 2 + \log M - 2n\epsilon^2 = \log\delta$, so $\epsilon = \sqrt{\frac{\log M + \log(2/\delta)}{2n}}$. **The $M$ entered as a multiplier and came out as a logarithm, because it had to pass through an exponential to get there.** That is enormously lucky: it means a class ten thousand times larger costs you only $\sqrt{\log 10^4} \approx 3$ times more. Without the exponential decay in Hoeffding, learning would be impossible.

**Why $\log M$ is "bits."** Describing which of $M$ objects you mean takes $\log_2 M$ bits — that is the definition of a bit. So $\log M$ is literally **the length of the shortest name for your model**, and the bound reads:

$$\text{generalization gap} \ \lesssim\ \sqrt{\frac{\text{bits needed to name your model}}{\text{number of examples}}}$$

▸ **This one sentence is the shape of every classical bound in this chapter**, and of the PAC-Bayes bound in §2.6, and of minimum description length (MDL). VC dimension and Rademacher complexity are both just cleverer ways of counting the bits when $M$ is infinite.

**Now the vacuity computation, slowly, because it is the hinge of the chapter.**

1. Every parameter is a 32-bit float, so it has $2^{32}$ possible values.
2. With $p$ parameters, the number of distinguishable models is at most $(2^{32})^p = 2^{32p}$.
3. $\log M \le 32p\log 2 = 22.2p$ nats.
4. With $p = 10^7$: $\log M \le 2.2\times10^8$.
5. With $n = 10^5$: $\sqrt{2.2\times10^8 / (2\times10^5)} = \sqrt{1100} \approx 33$.

**Now sit with what "33" means.** The loss is bounded in $[0,1]$, so the *worst conceivable* error is 1. A bound of "training error plus 33" is like a weather forecast that says tomorrow's temperature will be somewhere between $-3000$ °C and $+3000$ °C. It is not wrong. It is **true and useless**, which is the technical meaning of **vacuous**.

▸ **And you cannot patch it by being cleverer with the constants.** The bound is off by a factor of ~100, not ~2. Every subsequent section of this chapter is an attempt to find a quantity other than "parameter count" to put in that numerator — §2.4 tries expressivity, §2.5 tries norms, §2.6 tries information — and only the last one gets a number below 1.

**A useful sanity check on the shape.** To make the bound non-vacuous ($\le 1$) with $10^7$ parameters, you would need $n \ge \log M / 2 = 1.1\times10^8$ examples — roughly **eleven examples per parameter**. The folk rule "you need about ten data points per parameter" is not folklore at all; it is this bound, quoted without its derivation. **And every large model in this book violates it by two to four orders of magnitude while generalizing fine.** That contradiction is what the rest of the chapter, and Chapter 18, exist to resolve.

#### Examples and non-examples: what a generalization bound actually tells you

A bound is a specific kind of object with a specific kind of claim, and reading it as a forecast produces nonsense in both directions — panic when it is large, false comfort when it is small.

**✅ Things a bound legitimately gives you**

| Example | Why it qualifies |
|---|---|
| "With probability $\ge 0.95$ over the draw of the training set, $R(\hat h) \le \hat R(\hat h) + 0.08$" | A claim about the **procedure**, valid before you see any particular dataset. This is what the theorem literally states |
| The $\sqrt{\log M/2n}$ shape | The *dependence* is the useful part: $4\times$ the data halves the gap, independent of every constant you don't know |
| A PAC-Bayes bound of $0.21$ against a measured test error of $0.17$ | **Non-vacuous.** It brackets reality, so the quantities inside it ($\|\hat\theta\|^2$, $\sigma$) are plausibly the ones that matter — which is what makes §2.9's ranking possible |
| Comparing what two bounds put in the numerator — parameter count versus weight norm | Bounds are most valuable as a **ranking of candidate explanations**, and the one that stays below 1 wins |
| "This class has infinite VC dimension, so it is not distribution-free PAC-learnable" | A **negative** result, and negative results from bounds are airtight — no cleverness recovers what the theorem forbids |

**❌ Near-misses — reading a bound as something it isn't**

| Looks like it says this | Why it doesn't | What it actually is |
|---|---|---|
| "The bound is $33$, so my test error will be around 33" | An upper limit is not an estimate. Loss lives in $[0,1]$, so "$\le 33$" excludes nothing | A **vacuous** bound: true and uninformative |
| "The bound is $0.40$ and my error is $0.05$, so the bound is wrong" | $0.05 \le 0.40$. The theorem held perfectly; it simply never promised tightness | Looseness, not falsity |
| "I violate the 10-points-per-parameter rule, so my model will fail" | That rule is the *converse* of a bound. Failing to earn a certificate is not the same as being forbidden | Absence of a guarantee |
| "95% confidence means 95% of test points fall under the bound" | The probability is over the draw of the **training sample**, not over individual test examples | A statement about how often the whole guarantee holds across hypothetical retrainings |
| "I tightened my bound, so my model generalizes better now" | You changed the analysis; the weights are byte-for-byte identical | A better certificate for the same object — unless the tightening action (weight decay) also changed the model |
| "The bound applies to my model" | Uniform-convergence bounds hold for **every** $h$ in $\mathcal{H}$ simultaneously — that's where the $\log M$ came from | A worst-case statement over a class, paid for in the union bound |

▸ **The boundary:** a bound constrains the **worst case over a class**, with the probability taken over the **training sample**; a prediction estimates the typical case for **one model**. Bounds fail upward and never downward, which is exactly why "$\le 33$" is worthless while "$\le 0.21$" is a  scientific result.

> **Common misconception.** *"A generalization bound predicts my test error."* It bounds it, from one side only, in the worst case over an entire hypothesis class, with probability over training-set draws. Every one of those five qualifiers weakens the claim, and they compound. A bound of $0.40$ next to a measured error of $0.05$ is not evidence of anything being broken — it is the ordinary state of affairs. **The belief is tempting because a bound is a number with the same units as test error, printed next to test error.** Numbers that share units invite subtraction, and subtracting them is meaningless here. The right question is never "how close is the bound to my error?" but "**is the bound below 1?**" — because a bound above 1 has told you literally nothing about a loss that already lives in $[0,1]$.

> **Common misconception.** *"The bounds come out vacuous, so statistical learning theory doesn't apply to deep learning."* Every theorem in this chapter is still true of your ResNet. Hoeffding holds. The union bound holds. The VC bound holds. What failed is not the *machinery* but a **choice of what to put in the numerator** — parameter count, then expressivity, then dimension. Swap in the weight norm (§2.5) and the number shrinks; swap in a posterior-to-prior KL (§2.6) and it drops below 1 and starts being useful. **The belief is tempting because the failure is so spectacular** — a bound of 33 on a quantity that cannot exceed 1 feels like a refutation rather than a mis-parameterization. It is worth noticing that the fix was not a new theorem; it was a new quantity fed to an old one.

---

## 2.4 VC dimension

### The one-line idea

VC dimension measures how many points a hypothesis class can label *arbitrarily* — a capacity measure that doesn't care about how many parameters you have, only about how expressive you actually are.

### The analogy

Think of a hypothesis class as a set of cookie cutters. VC dimension is the largest number of scattered crumbs such that, no matter which subset you want inside the cutter and which outside, some cutter in your set can do it.

### Definition

$\mathcal{H}$ **shatters** a set of $m$ points if for all $2^m$ labelings, some $h\in\mathcal{H}$ realizes it. $\mathrm{VC}(\mathcal{H})$ = largest $m$ shatterable.

**Examples:**
- Thresholds on $\mathbb{R}$ ($h(x) = \mathbb{1}[x>a]$): VC = 1.
- Intervals on $\mathbb{R}$: VC = 2.
- Linear classifiers in $\mathbb{R}^d$ (with bias): VC = $d+1$.
- Axis-aligned rectangles in $\mathbb{R}^2$: VC = 4.
- $h(x)=\mathbb{1}[\sin(\omega x)>0]$, one parameter $\omega$: **VC = $\infty$.** ⇒ *parameter count is not capacity.* Remember this when someone says "more parameters = more overfitting."

#### Shattering, decoded

**"Shatters" is a vivid word for a simple test.** Put $m$ points on a table. Now consider *every possible way* of colouring them red or blue — there are $2^m$ such colourings, because each point independently gets one of two colours. Your class $\mathcal{H}$ **shatters** those points if, for **every single one** of those $2^m$ colourings, some model in your class produces exactly that pattern. Not most of them. All of them.

▸ **VC dimension is then: the largest $m$ for which you can find *some* arrangement of $m$ points that your class shatters.** Two quantifiers, and their direction matters enormously. You get to **choose** the point positions favourably; you must then handle **all** labelings. That asymmetry is why VC dimension is a measure of *potential*, not of typical behaviour.

**Every symbol:**

- $\mathbb{1}[x > a]$ — the indicator function. Read *"one if x is greater than a, zero otherwise."* It is an `if` statement written as mathematics; here it defines a classifier that says "positive" to everything right of $a$.
- $2^m$ — the number of ways to label $m$ points with two classes. For $m = 10$ that is 1,024; for $m = 50{,}000$ it is a number with 15,000 digits.
- $\mathrm{VC}(\mathcal{H})$ — a single integer summarizing the whole class.

**Work the smallest example completely: thresholds on the line, VC = 1.**

- *One point.* Put it at $x = 0$. Want it labelled 1? Choose $a = -1$, so $\mathbb{1}[0 > -1] = 1$. ✓ Want it labelled 0? Choose $a = 1$. ✓ Both labelings achievable, so **1 point is shattered.**
- *Two points*, at $x_1 = 0$ and $x_2 = 1$. There are four labelings. Three are easy. But $(x_1{=}1, x_2{=}0)$ — "the left one positive, the right one negative" — is **impossible**, because a threshold rule always says positive to everything on the right. One labeling out of four is unreachable, so 2 points are not shattered.
- Therefore VC = 1. **And note: failing on even a single labeling is enough to fail.**

**Now the pattern in the other examples.**

| Class | VC | Why that number |
|---|---|---|
| Thresholds on $\mathbb{R}$ | 1 | One boundary, and it always points the same way |
| Intervals on $\mathbb{R}$ | 2 | Two boundaries. Fails on 3 points labelled $+,-,+$ |
| Linear classifiers in $\mathbb{R}^d$ | $d+1$ | $d$ weights plus a bias — here capacity *does* equal parameter count |
| Axis-aligned rectangles in $\mathbb{R}^2$ | 4 | Four edges. Place points at N/S/E/W extremes; any subset can be boxed in |
| $\mathbb{1}[\sin(\omega x) > 0]$ | $\infty$ | **One parameter.** Read the next paragraph |

**The sine example is the one that should change your mind.** A single real number $\omega$ controls how fast $\sin(\omega x)$ oscillates. Crank $\omega$ up and the sign flips arbitrarily fast; a well-chosen $\omega$ can be made to have its positive half-cycles land on exactly the points you want positive, for *any* labeling of *any* number of points (placed, say, at $x_i = 2^{-i}$). One parameter, infinite VC dimension.

> **Analogy.** A single dial on an old radio can, in principle, address any station on the band, including ones you never imagined. The *count* of dials tells you nothing about how many stations are reachable — that depends entirely on the mechanism behind the dial. **A parameter is a dial, not a unit of capacity.** A real number contains infinitely many bits; how many of them your model can actually *use* is a completely separate question.

▸ **Hence the standing rule: parameter counting is not capacity measurement, in either direction.** The sine has one parameter and infinite capacity. A 175-billion-parameter language model constrained by weight decay, finite precision, and an optimizer that only travels a short distance from initialization may have vastly *less* effective capacity than its count suggests. **Both errors are common, and the second is the expensive one.**

**Why the definition is built this way.** It might look arbitrary, but shattering is exactly the condition that makes the union bound collapse. If your class can realize every one of the $2^m$ labelings on $m$ points, then on those points it is effectively a class of size $2^m$, and $\log M = m\log 2$ grows *linearly* in the sample size — which makes $\sqrt{\log M / n}$ stop shrinking, and learning becomes impossible. **VC dimension is precisely the point at which that catastrophe stops happening**, which §2.4's Sauer–Shelah lemma makes exact.

### The VC bound

With probability $\ge 1-\delta$, for all $h$:

▸ $$R(h) \le \hat R_n(h) + \sqrt{\frac{8\big(d_{\mathrm{VC}}\log\frac{2en}{d_{\mathrm{VC}}} + \log\frac{4}{\delta}\big)}{n}}$$

**Sauer–Shelah lemma** is the technical engine: a class with VC dimension $d$ can realize at most $\sum_{i=0}^{d}\binom{n}{i} \le (en/d)^d$ labelings of $n$ points — *polynomial*, not exponential. That polynomial growth is what saves the union bound.

#### Reading the VC bound, and the lemma that makes it work

**The bound's shape is the same as §2.3's**, and once you see that, the intimidating expression becomes readable:

$$R(h) \ \le\ \underbrace{\hat R_n(h)}_{\text{what you measured}} \ +\ \underbrace{\sqrt{\frac{\text{complexity} + \text{confidence}}{n}}}_{\text{what you might be off by}}$$

Everything inside the square root is either a complexity term ($d_{\mathrm{VC}}\log\frac{2en}{d_{\mathrm{VC}}}$) or a confidence term ($\log\frac4\delta$), and $n$ is downstairs. **$\log M$ from §2.3 has simply been replaced by $d_{\mathrm{VC}}\log\frac{2en}{d_{\mathrm{VC}}}$** — that is the only structural change, and it is the entire achievement of VC theory: it gives you a finite complexity number for an *infinite* hypothesis class.

**The pieces you haven't met:**

- $\binom{n}{i}$ — "n choose i", the number of ways to select $i$ items from $n$. $\binom{5}{2} = 10$.
- $e \approx 2.718$ — Euler's number. It appears because of a standard bound on binomial sums; carry no meaning into it.
- $\sum_{i=0}^{d}\binom{n}{i}$ — "the number of ways to choose *at most* $d$ items out of $n$." Read this as **the number of distinct behaviours the class can display on $n$ points.**

**The crucial comparison, in numbers.** How many labelings of $n$ points are there in total, versus how many your class can achieve?

| $n$ | All labelings, $2^n$ | Achievable, $\le (en/d)^d$ with $d = 10$ | Ratio |
|---|---|---|---|
| 20 | $1.0\times10^6$ | $\approx 1.4\times10^7$ | class can still shatter |
| 100 | $1.3\times10^{30}$ | $\approx 2.2\times10^{14}$ | $10^{-16}$ |
| 1,000 | $\approx 10^{301}$ | $\approx 2.2\times10^{24}$ | $10^{-277}$ |

▸ **That collapse is the whole game.** Below the VC dimension the class can do anything; above it, the fraction of behaviours it can express falls off a cliff. **Exponential growth would kill the union bound ($\log M \propto n$, so $\sqrt{\log M/n}$ stops shrinking); polynomial growth saves it ($\log M \propto d\log n$, so $\sqrt{d\log n/n} \to 0$).** The Sauer–Shelah lemma is the theorem that says the transition from "anything" to "almost nothing" is total and immediate — there is no gradual middle.

> **Analogy.** A key with $d$ notches can open a great many locks — but not *every* lock. Below some number of locks, the key blank is flexible enough to be cut for any of them. Past that number, the space of lock patterns grows exponentially while the space of cuttable keys grows only polynomially, and the fraction of locks you can open collapses to essentially zero. **Learning is possible precisely because your model cannot express most functions**, which is a  strange thing to have to say out loud.

**Now the number for neural networks, and it is bad.** For a ReLU network, $\mathrm{VC} = O(WL\log W)$: weights times depth times a log. With $W = 10^7$ and $L = 12$, $\log W \approx 16$, giving $\mathrm{VC}\approx 1.9\times10^9$ — *nineteen thousand times more than $n = 10^5$*. Push it through the bound: $\sqrt{1.9\times10^9/10^5} \approx 138$. **Vacuous by two orders of magnitude, on a loss bounded by 1.**

▸ **VC theory did not fail because it is wrong. It failed because it is *tight*.** There  exist datasets on which a network of this size overfits catastrophically — §2.4's random-label experiment builds one. A bound that must cover the worst case has no choice but to be this large. **The problem is that the worst case is not what your optimizer does**, and Chapter 19 is what happens when the field finally takes that seriously.

> **Where this came from.** The counting lemma was proved *three times independently, all within about a year.* **Vladimir Vapnik and Alexey Chervonenkis** proved it in Moscow as the engine of their learning theory (published in Russian in 1968, English in 1971). **Norbert Sauer** proved it in 1972 as pure combinatorics, answering a question posed by Paul Erdős. **Saharon Shelah** proved it in 1972 too, with Micha Perles, as a tool in model theory — a branch of mathematical logic with no connection to statistics whatsoever. Three fields, three motivations, one theorem, essentially simultaneously; it is now variously called the Sauer–Shelah lemma, the Sauer–Shelah–Perles lemma, or the Vapnik–Chervonenkis lemma depending on whose textbook you're holding. When an object gets discovered this many times this quickly, it is usually a sign that it was sitting at a natural junction of several problems — and this one sits at the junction of counting, logic, and generalization.

### VC dimension of neural networks

For a ReLU network with $W$ weights and $L$ layers:

$$\mathrm{VC} = O(WL\log W)$$

**Number:** a modest network with $W=10^7$, $L=12$: $\mathrm{VC} \approx 10^7 \cdot 12 \cdot 16 = 1.9\times10^9$. With $n=10^5$ training points, the bound gives $\sqrt{1.9\times10^9/10^5} \approx 138$. Vacuous by two orders of magnitude.

▸ **The Zhang et al. (2017) experiment.** Take CIFAR-10, replace all labels with random noise, train a standard ResNet. It reaches **zero training error**. So $\mathcal{H}$ shatters 50,000 points; VC dimension is at least 50,000; uniform-convergence bounds are useless. And yet *the same architecture trained on real labels generalizes fine.* Conclusion: generalization in deep learning is **not** a property of the hypothesis class. It's a property of the class *plus the optimizer* — which is why Chapter 19 (implicit bias) exists.

#### What the random-label experiment actually proves

**The experiment, stated so you could run it tomorrow.** Take CIFAR-10: 50,000 photographs, each labelled with one of ten categories. Now throw the labels away and replace each one with a **uniformly random** category. A picture of a dog is now labelled "truck", another dog is "frog", and there is no pattern whatsoever — no function of the pixels predicts these labels, because the labels were generated by a random number generator that never looked at the pixels. Train a standard ResNet on this. It reaches **100% training accuracy.** It memorizes all fifty thousand arbitrary assignments.

**Why that single sentence demolishes an entire theory.** Reaching zero error on 50,000 randomly-labelled points means the class realized *that particular* labeling out of $10^{50000}$ possible ones. Run it again with different random labels and it does it again. Conclusively: **this architecture can express essentially any labeling of CIFAR-10**, so its VC dimension is at least 50,000, which is at least $n$ — and $\sqrt{d_{\mathrm{VC}}/n} \ge 1$ makes every uniform-convergence bound in this chapter numerically useless. Not loose. **Useless.**

**Now the part that makes it a *paradox* rather than merely bad news.** The identical architecture, identical optimizer, identical hyperparameters, trained on the *real* labels, generalizes to unseen photographs at around 90% accuracy. Same $\mathcal{H}$. Same $n$. Same bound. Wildly different outcomes.

▸ **A bound that depends only on $\mathcal{H}$ and $n$ produces the same number for both runs.** Therefore no such bound can possibly explain the difference. Not a loose bound, not a bad bound — *no bound of that form*, ever, however clever. **The explanation must involve something the bound doesn't mention.** That is a proof by elimination, and it is why this paper reset the field's agenda.

**What the difference actually was, and the clue everyone points to.** The random-label run took far longer to converge, and it did not converge to a *simple* solution — it converged to a memorization. The real-label run found a solution fast, and the solution had structure that transferred. **The architecture was capable of both; the data plus the optimizer chose which.** That is the observation that names the successor research programme: **implicit bias** — the study of which solution gradient descent picks when many are available.

> **Analogy.** A blank notebook can hold Shakespeare or fifty thousand random digits; its page count tells you nothing about which it contains. Bounding a student's exam performance by the capacity of the notebook they were issued is doomed, because everyone gets the same notebook. **You have to describe what they actually wrote.** Classical theory measures the notebook. Chapters 18 and 19 measure the writing.

**One consequence worth stating plainly**, since it inverts a piece of common advice. "This model has enough capacity to memorize the training set, therefore it will overfit" is **false as an inference**. Every modern network has that capacity, including the ones that generalize best. Capacity establishes what is *possible*, and possibility has turned out to be nearly uninformative about what happens.

> **Where this came from.** The paper is "Understanding Deep Learning Requires Rethinking Generalization," by **Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals**, presented at ICLR 2017, where it won a best-paper award. Its impact came less from technical difficulty — the central experiment is a few lines of code that any graduate student could have run at any point in the previous five years — than from the fact that somebody ran it and stated the consequence without softening it. It is a good argument for occasionally testing the assumption everyone treats as too obvious to check.

> **The story behind VC theory's authors.** **Vladimir Vapnik** and **Alexey Chervonenkis** met at the Institute of Control Sciences in Moscow in the early 1960s and collaborated for decades on a theory that, at the time, had almost no computational outlet — the machines to exploit it did not exist. Their work reached the West slowly and in translation, and for years was better known among Soviet mathematicians than among the American researchers building the systems it described. Chervonenkis remained in Russia, teaching and working on data analysis. In 2014 he went walking in Losiny Ostrov, the large national park on the edge of Moscow, became lost, and died of hypothermia; he was 76. Vapnik's *The Nature of Statistical Learning Theory* (1995) remains the standard exposition of the ideas the two of them built, and the initials of both are attached, permanently, to the single most-cited capacity measure in machine learning.

#### Examples and non-examples: what VC dimension explains

VC dimension is not a failed idea. It is a **correct** idea with a precisely knowable domain of validity, and the interesting exercise is drawing the line between the problems where it works beautifully and the ones where it says nothing.

**✅ Where VC dimension  answers the question**

| Example | Why it qualifies |
|---|---|
| Linear classifiers in $\mathbb{R}^{100}$: $d_{\mathrm{VC}} = 101$, with $n = 10^5$ | $\sqrt{101/10^5} \approx 0.03$. A 3% predicted gap, and observed gaps for well-conditioned linear models really are that size. The bound is **usable** |
| Axis-aligned rectangles in the plane: $d_{\mathrm{VC}} = 4$ | Four points can be shattered, five never can. A finite, computable, tight-ish number that tells you how many samples you need |
| Deciding whether a degree-$d$ polynomial threshold family is learnable from $n$ points | Capacity and parameter count coincide in this family, so the ratio $d/n$  governs the gap |
| "Infinite VC dimension $\Rightarrow$ not distribution-free PAC-learnable" | The **negative** direction of the fundamental theorem, and it remains completely valid for deep nets. No amount of engineering repeals it |
| $\mathbb{1}[\sin(\omega x)>0]$ has $d_{\mathrm{VC}} = \infty$ with one parameter | The theory correctly flags a one-knob family as uncontrollable. It caught something parameter-counting would have missed entirely |

**❌ Near-misses — VC-shaped reasoning that doesn't hold**

| Looks like it works | Why it doesn't | What it actually is |
|---|---|---|
| "A ResNet's $d_{\mathrm{VC}}$ is roughly its parameter count, so $p/n$ estimates my gap" | For ReLU networks $d_{\mathrm{VC}}$ scales like $O(WL\log W)$ — parameters *times depth* — and the random-label experiment shows it empirically exceeds $n$ | A ratio $\ge 1$: a vacuous bound in disguise |
| "VC dimension counts parameters" | One parameter, infinite VC dimension; and conversely two models with equal parameter counts can differ in $d_{\mathrm{VC}}$ by orders of magnitude | Parameter count — a different number that sometimes coincides |
| "Same architecture and same $n$, so the same generalization behaviour" | Random labels and real labels give the same $\mathcal{H}$, the same $n$, and outcomes 80 accuracy points apart | Proof that the answer lives in the **data and the optimizer**, which $\mathcal{H}$-only quantities cannot see |
| "This network has the capacity to memorize the training set, so it will overfit" | Every network that generalizes well also has that capacity. The premise is true of the best models in the world | Confusing what is **possible** with what **happens** |
| "Shattering means the model fits the data well" | Shattering means realizing *every one* of the $2^d$ labelings of some cleverly placed $d$ points | Worst-case expressivity on an adversarially chosen set — not accuracy on yours |
| "$d_{\mathrm{VC}} \ge 50{,}000$, therefore this model is bad" | It is also $\ge 50{,}000$ for the run that hits 90% test accuracy | A statement about the hypothesis class that both runs share |

▸ **The boundary:** VC dimension is a function of $\mathcal{H}$ **alone**. It cannot see the data distribution, the optimizer, or which of the many fitting solutions you actually landed on. It is exactly right when those three don't matter — low-dimensional linear models, small classical families — and exactly useless when they are the entire story, which is the case for every network in this book.

> **Common misconception.** *"VC dimension explains deep networks — it just gives loose numbers."* Not loose: **structurally incapable**. The random-label experiment is a proof by elimination. Two runs share $\mathcal{H}$, share $n$, and differ enormously in test error; any bound that is a function of only $\mathcal{H}$ and $n$ returns the *same number* for both, so no such bound — however tight, however clever, however many years of future work — can distinguish them. **The belief is tempting because "loose bound" is the failure mode we're used to**, and it comes with a comforting implicit promise that someone will eventually tighten it. Here there is nothing to tighten. The explanation has to mention something the quantity doesn't contain, which is why the field's agenda moved to implicit bias (Chapters 18–19) rather than to sharper VC constants.

> **Common misconception.** *"A model that can memorize random labels will memorize the real ones too."* Capacity is a statement about what is reachable, not about where gradient descent goes. The very same weights, the very same learning rate, and the very same 50,000 images produce memorization on shuffled labels and structured generalization on true labels. **The belief is tempting because it is a perfectly sound inference in the classical setting**, where the hypothesis class is small enough that "can" and "will" nearly coincide — if a degree-19 polynomial *can* thread 20 points, it will. With a class large enough to realize essentially any function, "can" stops carrying information, and the optimizer's preferences take over as the thing worth studying.

---

## 2.5 Rademacher complexity

### The one-line idea

Instead of counting hypotheses, directly measure: how well can my model class fit *pure noise*? That number is the complexity.

### Definition

Given samples $S = \{z_1,\dots,z_n\}$ and Rademacher variables $\sigma_i \in \{\pm1\}$ uniform:

▸ $$\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma\left[\sup_{f\in\mathcal{F}}\frac1n\sum_{i=1}^n \sigma_i f(z_i)\right]$$

This is a *correlation with random labels*, maximized over the class. If $\mathcal{F}$ can fit any noise, $\hat{\mathfrak{R}} \to 1$. If $\mathcal{F}$ is a single constant function, $\hat{\mathfrak{R}} = 0$.

#### Unpacking Rademacher complexity

**Read the formula aloud first:** *"R-hat-sub-S of script-F equals the expectation over sigma of the supremum over f in script-F of one-over-n times the sum over i of sigma-i times f of z-i."* Now translate it into a procedure, because that is what it is.

**The procedure, in four steps:**

1. Take your $n$ data points $z_1,\dots,z_n$.
2. **Flip a fair coin for each one.** $\sigma_i = +1$ for heads, $-1$ for tails. These are the *Rademacher variables*, and they carry no information about anything — they are pure noise, generated with no reference to your data.
3. Ask your model class: *"which of you correlates best with this meaningless pattern?"* That is the $\sup_{f\in\mathcal{F}}$ — read it as $\max$; the sup is used only because the class may be infinite and the best value may only be approached.
4. Repeat with fresh coin flips and average. That is the $\mathbb{E}_\sigma$.

▸ **Rademacher complexity is: how well can my model class fit pure noise, on average, on this actual dataset?** That is a shockingly direct definition of capacity — no counting, no combinatorics, no shattering. Just: try it and see.

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $\mathcal{F}$ | "script F" | The function class (here written as functions, not classifiers) |
| $S = \{z_1,\dots,z_n\}$ | "S" | Your actual, specific sample — note it appears as a subscript |
| $\sigma_i \in \{\pm1\}$ | "sigma-i" | One coin flip, written as $+1$ or $-1$ |
| $\mathbb{E}_\sigma$ | "expectation over sigma" | Average over all the coin-flip patterns |
| $\sup_{f\in\mathcal{F}}$ | "the supremum over f" | The best any function in the class can do. A careful $\max$ |
| $\frac1n\sum_i \sigma_i f(z_i)$ | — | The **correlation** between the function's outputs and the coin flips |
| $\hat{\mathfrak{R}}$ vs $\mathfrak{R}$ | "empirical" vs "expected" | Hat = on your sample; no hat = averaged over samples too |

The letter $\mathfrak{R}$ is **Fraktur R** (blackletter). It is a font choice, not an operator; it exists so that $\mathfrak{R}$, $R$ (risk), and $\hat R$ (empirical risk) can coexist on the same page.

**Work both extremes with real numbers, $n = 4$.**

*A class containing only the constant function $f \equiv 1$*: the sum is $\frac14\sum_i\sigma_i$, the average of four coin flips, whose expectation is $0$. With no choice available, there is no way to exploit the noise. $\hat{\mathfrak{R}} = 0$ — well, $\mathbb{E}\lvert\frac14\sum\sigma_i\rvert$ is small and, with a  single fixed function and no absolute value, exactly 0.

*A class rich enough to output anything in $\{\pm1\}$ at each point*: pick $f(z_i) = \sigma_i$ for every $i$. Then $\frac14\sum_i \sigma_i\cdot\sigma_i = \frac14\sum_i 1 = 1$. **Perfect correlation with noise, every time.** $\hat{\mathfrak{R}} = 1$.

*Something in between* — a class that can match, say, 3 of the 4 signs: $\frac14(1+1+1-1) = 0.5$.

▸ **So the number is a dial from 0 (rigid) to 1 (can fit anything), read directly off your data.** And it is measurable: you can literally estimate it by generating random labels and training. Which means §2.4's random-label experiment was not just a demonstration — **it was an estimate of a Rademacher complexity, and the answer came back 1.**

> **Analogy.** You want to know how gullible a conspiracy theorist is. Don't count how many books they own (VC dimension). Instead, tell them a story you made up on the spot and see how confidently they explain it. **Someone who can produce a compelling account of *any* random noise you feed them has told you exactly how much to trust their account of the real thing.** That is Rademacher complexity, and its brilliance is that it tests the mechanism rather than inventorying the parts.

**Why the bound has the form $R(f) \le \hat R_n(f) + 2\mathfrak{R}_n(\mathcal{F}) + 3\sqrt{\log(2/\delta)/2n}$.** Three terms, three jobs: what you measured, a penalty for how much your class can chase noise, and a small confidence term for finite-sample luck. The factor of 2 comes from a proof technique called **symmetrization** — you compare your sample to an imaginary second "ghost" sample, and the coin flips $\sigma_i$ are what decide which of the two each point came from. The 2 and the 3 are artifacts of that argument, not insights.

> **Where this came from.** **Hans Rademacher** introduced the $\pm1$-valued functions now bearing his name in 1922, in a paper on systems of orthogonal functions — pure analysis, no statistics anywhere near it. He held a chair at the University of Breslau until 1934, when he was dismissed by the Nazi regime on political grounds (he had been active in pacifist and human-rights organizations), and emigrated to the United States, spending the rest of his career at the University of Pennsylvania. He is best remembered in mathematics not for these functions but for the **Rademacher exact formula** of 1937, a convergent series for the number of ways to write an integer as a sum of positive integers, sharpening the celebrated asymptotic result of Hardy and Ramanujan. Rademacher complexity as a capacity measure in statistical learning theory arrived only around 2000–2002, developed by **Vladimir Koltchinskii** and, independently, by **Peter Bartlett and Shahar Mendelson** — seventy-eight years after the functions were defined for an entirely different purpose.

### The bound

▸ $$R(f) \le \hat R_n(f) + 2\mathfrak{R}_n(\mathcal{F}) + 3\sqrt{\frac{\log(2/\delta)}{2n}}$$

**Why it's better than VC:** it's *data-dependent* and *scale-sensitive*. VC dimension of linear classifiers in $\mathbb{R}^d$ is $d+1$ regardless of margins; Rademacher complexity of *norm-bounded* linear predictors is

$$\mathfrak{R}_n(\{x\mapsto \langle w,x\rangle : \|w\|\le B\}) \le \frac{B\max_i\|x_i\|}{\sqrt n}$$

— **independent of dimension $d$.** This is why SVMs generalize in infinite-dimensional feature spaces. Norm, not dimension, is the capacity that matters.

#### Why norm beats dimension

**Read the formula:** *"the Rademacher complexity of the class of linear functions $x \mapsto \langle w,x\rangle$ with $\|w\| \le B$ is at most $B$ times the largest data norm, over root n."*

- $x \mapsto \langle w,x\rangle$ — read *"the function that maps x to the dot product of w with x."* The $\mapsto$ (bar-arrow) means "sends to" and defines a function without naming it, the way a lambda does in code.
- $\|w\| \le B$ — the weights are constrained to a ball of radius $B$. **This is the only thing restricting the class**, and $d$ is nowhere in it.
- $\max_i\|x_i\|$ — the length of the longest data point you actually have. Note that this is a property of *your dataset*, which is what "data-dependent" means.

**Put numbers on it.** Inputs normalized so $\|x_i\| \le 1$, weights constrained to $\|w\| \le 5$, and $n = 10{,}000$ samples:

$$\mathfrak{R}_n \le \frac{5 \times 1}{\sqrt{10{,}000}} = \frac{5}{100} = 0.05$$

▸ **Now change $d$ from 10 to 10 million and recompute.** The answer is still $0.05$. **The dimension does not appear.** Compare this with VC dimension, which for linear classifiers is exactly $d+1$ — so VC theory would say the 10-million-dimensional class is a million times more complex, and Rademacher theory says the two are identical. **Rademacher is right and VC is not wrong**; they are answering different questions. VC asks what the class could do with unlimited weights; Rademacher asks what it can do with the weights you're actually allowing.

**Why the constraint changes everything.** With unbounded $w$, a linear model in $d$ dimensions really can shatter $d+1$ points — that is what VC = $d+1$ says. But shattering requires *large* weights: to force a specific sign pattern on nearly-collinear points, some coefficient has to become enormous. **Cap the norm and you have quietly removed exactly the configurations that made the class dangerous, without removing any dimensions.** Capacity turns out to live in the *size* of the weights, not in how many of them there are.

> **Analogy.** A library's capacity to mislead you is not the number of books on the shelves — it is how far any one book is allowed to stray from the truth. Add a million dull, cautious volumes and nothing changes. Add one wild one and everything does. **VC dimension counts books; norm-based complexity measures how wild any one of them is allowed to be.**

**Why this explains SVMs, in one paragraph.** A support vector machine maps inputs into a feature space that can be *infinite*-dimensional (the radial-basis-function kernel does exactly this), so its VC dimension is infinite and VC theory says nothing at all. But the SVM's training objective explicitly minimizes $\|w\|^2$ — it *maximizes the margin*, which is the same thing, since margin $= 1/\|w\|$ for a separating hyperplane. **The algorithm was designed, before this theory existed, to control precisely the quantity the theory later identified as the one that matters.** Vapnik built the margin in for geometric reasons; the norm-based analysis explained afterwards why it worked in infinite dimensions.

▸ **This is the pivot on which the rest of the chapter turns.** Once capacity is a *norm* rather than a *count*, weight decay stops being a heuristic and becomes a direct intervention on your generalization bound. §2.6 makes that connection exact, and §2.9 item 4 is the same fact stated as practical advice.

**Where the $1/\sqrt n$ comes from — it is the same $\sqrt n$ as everywhere else.** The quantity $\frac1n\sum_i\sigma_i x_i$ is an average of $n$ random $\pm$ vectors. Random terms partially cancel, so their sum grows like $\sqrt n$ rather than $n$, and the average therefore shrinks like $1/\sqrt n$. Chapter 1 §1.3.1 has this as the standard error; Chapter 0 §0.8 has it as why random high-dimensional vectors are near-orthogonal. **It is one fact, wearing three costumes.**

### Contraction and layer-wise bounds for networks

Talagrand's contraction lemma: if $\phi$ is $L$-Lipschitz, $\mathfrak{R}(\phi\circ\mathcal{F}) \le L\,\mathfrak{R}(\mathcal{F})$. ReLU is 1-Lipschitz, so it's free.

Chaining through $L$ layers with spectral norm bounds gives things like

$$\mathfrak{R}_n \lesssim \frac{\prod_{\ell=1}^L \|W_\ell\|_2 \cdot \left(\sum_\ell \frac{\|W_\ell\|_{2,1}^{2/3}}{\|W_\ell\|_2^{2/3}}\right)^{3/2}}{\sqrt n}$$

(Bartlett–Foster–Telgarsky). This is *norm-based* and can be non-vacuous for small networks. For large ones it's still typically vacuous, because $\prod\|W_\ell\|_2$ is astronomically large in practice.

#### Reading the layer-wise bound, and why it still fails

**The contraction lemma first, because it is the easy half.** Talagrand's contraction lemma says: *"if you pass a function class through an $L$-Lipschitz squashing function $\phi$, its Rademacher complexity grows by at most a factor of $L$."* Recall from Chapter 1 §1.1.4 that a Lipschitz constant is a speed limit: nudge the input by $\epsilon$, the output moves by at most $L\epsilon$.

▸ **ReLU has $L = 1$**, so composing with a ReLU multiplies the complexity by 1 — it costs nothing. That is what "it's free" means, and it is a  convenient accident: the most popular activation function in deep learning happens to be the one that a capacity analysis can pass straight through. (Sigmoid and tanh also have $L = 1$; the property is not unique to ReLU, but ReLU is where it matters most.)

**Now the big expression, which is less fearsome than it looks:**

- $\prod_{\ell=1}^L\|W_\ell\|_2$ — the product of the spectral norms, layer by layer. **This is exactly the network's overall Lipschitz constant** from Chapter 1 §1.1.4: the maximum amplification the whole stack can apply. Every term in the product is "how much can this layer stretch."
- $\|W_\ell\|_{2,1}$ — a **mixed norm**: take the $\ell_2$ length of each row, then $\ell_1$-sum those lengths. It is sensitive to how many rows are meaningfully non-zero, so it acts as a rough sparsity measure.
- The ratio $\|W_\ell\|_{2,1}/\|W_\ell\|_2$ inside the sum measures **how much of the layer's capacity is spread across many directions** versus concentrated in one — a cousin of the stable rank from Chapter 1 §1.1.3.
- $\lesssim$ — "is at most, up to constants nobody tracks."
- $/\sqrt n$ — the same $\sqrt n$ as always.

**Where it breaks, with numbers.** Suppose a 50-layer network has $\|W_\ell\|_2 = 1.5$ at every layer — a completely unremarkable value that no practitioner would flag.

$$\prod_{\ell=1}^{50}1.5 = 1.5^{50} \approx 6.4\times10^{8}$$

Divide by $\sqrt{n}$ with $n = 10^5$: $6.4\times10^8/316 \approx 2\times10^6$. **A bound of two million on a loss bounded by one.** And this is the *best* of the norm-based bounds.

▸ **The failure has a precise cause, and it is the same cause as adversarial examples.** A product of $L$ numbers each slightly above 1 is astronomically large — $1.2^{50}\approx 9100$, $1.5^{50}\approx 6\times10^8$. Chapter 1 §1.1.4 used exactly this arithmetic to explain why imperceptible input perturbations flip a classifier's output. **The same product that makes networks fragile makes their generalization bounds vacuous, because both quantities are the network's worst-case amplification.** The bound is measuring something real. It is measuring the worst case, and the worst case has very little to do with the typical case.

> **Analogy.** Rate a delivery company by multiplying together the maximum possible delay of every leg of the journey. Fifty legs, each capable of a 50% overrun, yields a predicted arrival time hundreds of millions of times the schedule. Every individual bound is honest and none of them ever compound that way in practice, because the delays are not adversarially aligned. **A worst-case bound over a long composition is almost always so pessimistic as to be useless, and no amount of tightening the individual legs fixes it — the problem is the multiplication.**

> **A note on Talagrand.** **Michel Talagrand**, a French mathematician at the CNRS, built much of the modern theory of concentration of measure — results about why functions of many independent random variables are overwhelmingly close to their averages. His concentration inequalities are the machinery underneath a large fraction of statistical learning theory, including the contraction lemma quoted here. He was awarded the **Abel Prize in 2024**, one of the highest honours in mathematics, for work on probability and functional analysis. He has also written publicly about having lost the sight in one eye as a child and spending long hospital stays that pushed him toward mathematics.

---

## 2.6 PAC-Bayes — the bound that actually works

### The one-line idea

Don't bound the risk of a single hypothesis. Bound the *average* risk of a distribution over hypotheses, and pay a KL-divergence price for how far that distribution moved from your prior.

### The analogy

A prior is a codebook you agreed on before seeing data. The posterior is where you actually ended up. The KL term is the number of extra bits you need to transmit to tell someone where you ended up, given the codebook. **The fewer extra bits, the more you can trust the result** — because a hypothesis you could have described cheaply in advance is less likely to be a coincidence.

### The McAllester bound

For a prior $P$ over $\mathcal{H}$ chosen *before* seeing data, and any posterior $Q$ (which may depend on data), with probability $\ge 1-\delta$:

▸ $$\mathbb{E}_{h\sim Q}[R(h)] \le \mathbb{E}_{h\sim Q}[\hat R_n(h)] + \sqrt{\frac{\mathrm{KL}(Q\|P) + \log\frac{2\sqrt n}{\delta}}{2n}}$$

#### Reading the McAllester bound in plain English

**The shape is the same as every other bound in this chapter** — measured error, plus square root of complexity over $n$ — with two changes, and both changes are the whole idea.

**Change 1: you no longer bound a single model. You bound a *cloud* of models.**

- $P$ — the **prior**: a probability distribution over models, fixed **before you look at the data**. Typically "the distribution from which you drew your random initialization."
- $Q$ — the **posterior**: any distribution over models you like, chosen **after** seeing the data. Typically "a small Gaussian blob centred on the weights you trained to."
- $\mathbb{E}_{h\sim Q}[R(h)]$ — read *"the expected true risk, when h is drawn from Q."* You are not asking about one model; you are asking about the **average performance of a randomly perturbed version of your model.**

▸ **That change of question is what buys the whole result.** It is much harder to guarantee that *one specific* parameter setting generalizes than to guarantee that *a whole neighbourhood* of settings generalizes on average — because a fluke can be a single point, but a fluke cannot be a whole region.

**Change 2: the complexity term is $\mathrm{KL}(Q\|P)$ — information, not size.**

Kullback–Leibler (KL) divergence, from Chapter 1 §1.4, measures **how many extra nats you waste encoding samples from $Q$ using a code designed for $P$.** Here that reads: *how much did the training data have to tell you, to move you from where you started to where you ended up?*

- Never moved at all ($Q = P$)? $\mathrm{KL} = 0$, zero penalty. Obviously — the data taught you nothing, so it could not have overfit you.
- Moved to a razor-thin spike far from the prior? Large KL, large penalty.
- Note there is **no $p$**, no parameter count, no dimension. A 100-million-parameter network that barely budged from initialization pays almost nothing.

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $h \sim Q$ | "h drawn from Q" | Sample a model from your posterior cloud |
| $\mathbb{E}_{h\sim Q}[R(h)]$ | "expected true risk under Q" | Average true error of the cloud |
| $\mathbb{E}_{h\sim Q}[\hat R_n(h)]$ | "expected training risk under Q" | Average training error of the cloud — measurable! |
| $\mathrm{KL}(Q\|P)$ | "KL of Q from P" | Information injected by the data, in nats |
| $\log\frac{2\sqrt n}{\delta}$ | — | The confidence term. Grows like $\log$, so it barely matters |
| $2n$ | — | Twice your sample size, downstairs, as always |

> **Analogy — the book's codebook framing, made concrete.** Before seeing any data, you and a colleague agree on a codebook $P$: common models get short names, unusual ones get long names. You train, and end up at some model. Now you must **telegraph your result to them**, and you pay per character. $\mathrm{KL}(Q\|P)$ is your telegram bill. **A result you can describe cheaply, in a language agreed on in advance, is unlikely to be a coincidence** — because there simply are not many cheap descriptions, so you couldn't have found one by luck. An expensive description, on the other hand, is one of astronomically many, and finding *some* expensive description that fits your data is trivially easy. **The bill is the evidence.**

**Put numbers on it.** With $n = 10^5$ and $\delta = 0.05$:

| $\mathrm{KL}(Q\|P)$ | Bound's extra term $\sqrt{(\mathrm{KL} + \log\frac{2\sqrt n}{\delta})/2n}$ |
|---|---|
| 0 nats | $\sqrt{7.4/2\times10^5} = 0.006$ |
| 100 nats | $\sqrt{107/2\times10^5} = 0.023$ |
| 10,000 nats | $\sqrt{10007/2\times10^5} = 0.22$ |
| 1,000,000 nats | $\sqrt{10^6/2\times10^5} = 2.24$ — vacuous |

▸ **Compare with §2.3's $\log M \le 2.2\times10^8$ for the same network.** The KL term is small whenever training moved you a short distance in *information* terms, regardless of how many parameters there are to move. **The bound stopped charging you for capacity you didn't use.** That single reframing — from "how much could you have done" to "how much did you actually do" — is why this is the only bound in the chapter that produces a number below 1 for a real network.

**Why flatness buys you a tighter bound, spelled out.** Suppose the loss surface around your solution $\hat\theta$ is a wide, flat basin. Then you can smear $Q$ out over that whole basin — use a large $\sigma$ — and the *average* training error $\mathbb{E}_{h\sim Q}[\hat R_n(h)]$ barely rises, because every model in the basin is nearly as good. But a wide $Q$ is *closer* to a wide prior, so $\mathrm{KL}(Q\|P)$ is small. **You get the same first term and a smaller second term. The bound improves for free.**

Now suppose $\hat\theta$ sits in a needle-thin crevice. Smearing $Q$ at all sends most of its mass up the walls, so $\mathbb{E}_{h\sim Q}[\hat R_n(h)]$ explodes and you are forced to keep $Q$ narrow — which makes $\mathrm{KL}(Q\|P)$ large. **You pay either way.**

▸ **This is the rigorous content of "flat minima generalize better."** It is not an analogy and it is not folklore: flatness is literally the geometric condition that lets you choose a low-information posterior, and low information is literally what the bound charges for. Chapter 19 builds on this; sharpness-aware minimization (SAM) is an optimizer that attacks this quantity directly.

> **Where this came from.** The **PAC** framework — "probably approximately correct" — was introduced by **Leslie Valiant** in 1984 in a paper called "A Theory of the Learnable," which founded computational learning theory as a field and contributed to his Turing Award in 2010. The name captures the two dials exactly: *approximately* correct (within $\epsilon$) and *probably* so (with probability $1-\delta$). Valiant's insight was that demanding exact learning with certainty is hopeless, but demanding "close enough, usually" is achievable and quantifiable. The **PAC-Bayesian** hybrid — keep PAC's worst-case guarantees but let the object being bounded be a distribution over hypotheses — came from **John Shawe-Taylor and Robert Williamson** in 1997 and was developed into the standard form above by **David McAllester** in 1998–1999. It sat as a mostly theoretical curiosity for nearly two decades before anyone managed to make it produce a useful number for a neural network.

### Why this one succeeds where VC fails

- $\mathrm{KL}(Q\|P)$ measures **how much information the training data injected into the parameters**, not how many parameters exist. A 100M-parameter network that barely moved from initialization has small KL.
- It's naturally connected to **flat minima**: if the loss is flat around $\hat\theta$, you can use a wide $Q$ (large-variance Gaussian around $\hat\theta$) without hurting $\mathbb{E}_Q[\hat R_n]$. A wide $Q$ has *small* $\mathrm{KL}$ to a wide prior. **So flatness literally buys you a tighter generalization bound.** This is the rigorous version of "flat minima generalize better" (Ch. 19).

Formally: let $Q = \mathcal{N}(\hat\theta, \sigma^2 I)$, $P = \mathcal{N}(0,\sigma_0^2 I)$. Then
$$\mathrm{KL}(Q\|P) = \frac{\|\hat\theta\|^2}{2\sigma_0^2} + \frac{p}{2}\left(\frac{\sigma^2}{\sigma_0^2} - 1 - \log\frac{\sigma^2}{\sigma_0^2}\right)$$
The larger $\sigma$ you can tolerate (i.e. the flatter the minimum), the closer $\sigma \to \sigma_0$ and the smaller the second term. And the whole bound only depends on $\|\hat\theta\|^2$ — **weight norm**, not parameter count.

#### Unpacking the Gaussian KL

This formula looks forbidding and is  simple once split in two. **The setup:** $Q = \mathcal{N}(\hat\theta,\sigma^2 I)$ is a spherical Gaussian blob of radius $\sigma$ centred on your trained weights; $P = \mathcal{N}(0,\sigma_0^2 I)$ is a spherical blob of radius $\sigma_0$ centred at the origin, chosen before training. $I$ is the identity matrix, so "spherical" — the same spread in every direction, no correlations.

**Term 1 — $\dfrac{\|\hat\theta\|^2}{2\sigma_0^2}$: the cost of having *moved*.**

$\|\hat\theta\|^2$ is the squared length of your weight vector, i.e. how far training carried you from the origin, measured in units of the prior's own spread. **This is exactly what weight decay minimizes.** Set $\hat\theta = 0$ and it vanishes; double every weight and it quadruples.

**Term 2 — $\dfrac{p}{2}\left(\dfrac{\sigma^2}{\sigma_0^2} - 1 - \log\dfrac{\sigma^2}{\sigma_0^2}\right)$: the cost of having *narrowed*.**

Write $r = \sigma^2/\sigma_0^2$ (how much narrower your posterior is than the prior). The bracket is $r - 1 - \log r$, and this little function is worth knowing:

| $r = \sigma^2/\sigma_0^2$ | $r - 1 - \log r$ | Meaning |
|---|---|---|
| 1.0 | **0** | Posterior as wide as prior — you learned nothing, you pay nothing |
| 0.5 | 0.193 | Half as wide |
| 0.1 | 1.397 | Ten times narrower |
| 0.01 | 3.605 | A hundred times narrower |
| 0.001 | 5.908 | A thousand times narrower |

▸ **The function is zero at $r=1$ and positive everywhere else** — it is a standard, and rather elegant, fact that $r - 1 - \log r \ge 0$ for all $r > 0$, with equality only at $r = 1$. So the penalty is *exactly zero* when you don't narrow at all, and grows slowly (logarithmically) as you narrow. **Committing to a precise answer costs information; staying vague is free.**

**The $p$ out front is the sting, and it explains the flatness argument completely.** That per-dimension penalty is multiplied by the number of parameters. With $p = 10^7$ and $r = 0.1$, term 2 is $\frac{10^7}{2}(1.397) = 7\times10^6$ nats — vacuous again. With $r = 0.99$, the bracket is $5\times10^{-5}$ and term 2 is only $250$ nats. **A 0.1% narrowing of the posterior costs 250 nats; a 90% narrowing costs seven million.**

▸ **So the entire PAC-Bayes game is: how close can you keep $\sigma$ to $\sigma_0$ without your average training loss falling apart?** And that is precisely a question about how flat the minimum is. A flat basin lets you keep $r \approx 1$ and pay almost nothing. A sharp minimum forces $r \ll 1$ and the $p/2$ multiplier annihilates you. **Flatness isn't a nice-to-have in this bound; it is the only thing standing between you and a factor of ten million.**

> **Analogy.** You must state your answer to a question, and you are billed by how precise you make it. "Somewhere in Europe" is nearly free. "Latitude 48.8566, longitude 2.3522" costs a great deal. **If the true answer is a broad region, you can give the cheap answer and still be right — but if it is a single point, you have no choice but to buy the expensive one.** Flat minima are broad regions. Sharp minima are single points.

**Where the two terms point in practice.** Term 1 says *shrink your weights* — that is weight decay, §2.9 item 4. Term 2 says *find flat regions* — that is large learning rates, small batches, and sharpness-aware minimization, §2.9 item 3. **Both of the top mechanical levers in §2.9's ranked list appear here as separate terms of one formula**, which is a good reason to believe the formula is describing something real.

▸ **Dziugaite & Roy (2017)** optimized this bound directly and obtained the first **non-vacuous** generalization bound for a real neural network on MNIST (bound ≈ 0.16 error, actual ≈ 0.03). Not tight, but finite — a  milestone.

### Connection to MDL / Occam

$\mathrm{KL}(Q\|P)$ has units of nats = information. The bound says:

> generalization gap $\lesssim \sqrt{\dfrac{\text{bits to describe your model given a prior}}{n}}$

This is Occam's razor made quantitative, and it's the same $\log M$ from §2.3 in continuous clothing.

#### Occam, made quantitative

**Why "nats", and why that word matters.** A **nat** is a unit of information, like a bit but using base $e$ instead of base 2: one nat $= 1/\log 2 \approx 1.44$ bits. KL divergence, entropy, and cross-entropy in this book are all measured in nats because the mathematics uses natural logarithms throughout. **So $\mathrm{KL}(Q\|P) = 100$ nats literally means "144 bits of information" — about eighteen bytes.** That is a real, physical, countable quantity, not a metaphor.

▸ **This makes the bound say something startling: your generalization guarantee is a function of how many bytes it takes to describe your trained model, given a description scheme fixed in advance.** Not how many parameters. Not how many layers. **Bytes.** Two networks with identical architectures can have wildly different bounds if one of them is describable more cheaply.

**How the three complexity measures line up.** They are the same quantity at increasing levels of sophistication:

| Section | Complexity term | What it counts |
|---|---|---|
| §2.3 | $\log M$ | Bits to name one model out of $M$ — requires a finite class |
| §2.4 | $d_{\mathrm{VC}}\log\frac{2en}{d_{\mathrm{VC}}}$ | Bits to name one *behaviour on $n$ points* — handles infinite classes |
| §2.5 | $\mathfrak{R}_n(\mathcal{F})$ | Measured directly, on your data, by fitting noise |
| §2.6 | $\mathrm{KL}(Q\|P)$ | Bits the data actually injected — the only one that stays small |

**Minimum description length (MDL), stated as a principle.** Choose the model that minimizes *(bits to describe the model) + (bits to describe the data given the model)*. A model that memorizes needs no bits for the second term and enormous bits for the first; a model that says nothing needs no bits for the first and enormous bits for the second. **The best model is the one that pays the smallest total bill**, and compression and generalization turn out to be the same objective viewed from two sides.

> **Analogy.** You are given a page of numbers and must transmit it. Option A: send the digits — no cleverness, high cost. Option B: notice they are the first 500 primes and send the sentence "the first 500 primes" — five words. **The compression *is* the understanding.** Option B generalizes: it predicts the 501st number. Option A cannot. This is not an analogy for learning; on the MDL view it is a definition of it.

**Now the honest caveat about Occam's razor.** The 14th-century English Franciscan **William of Ockham** argued repeatedly for parsimony in explanation, but the famous Latin formulation — *entia non sunt multiplicanda praeter necessitatem*, "entities should not be multiplied beyond necessity" — does not appear in his surviving writings. It seems to have been coined later; it is commonly traced to the Irish scholar John Punch in 1639, three centuries after Ockham's death. The *name* "Occam's razor" appears to be a 19th-century coinage. **A principle named after a man who never stated it in the form everyone quotes** is a fair emblem for how much of this chapter's vocabulary was assembled after the fact.

▸ **And the mathematically important caveat: simplicity is only meaningful relative to a language.** $\mathrm{KL}(Q\|P)$ has a $P$ in it, and $P$ is *your choice*. There is no universal notion of a "simple" model; there is only "cheap to describe **in the code you committed to beforehand**." Change $P$ and every model's complexity changes. **This is not a weakness of the bound — it is the bound being honest about something the informal version of Occam's razor hides.** An architecture is a prior. A random initialization scheme is a prior. Convolution is a prior that makes translation-invariant functions cheap. §2.9 item 5 is this observation, stated as practical advice.

> **Where this came from.** **Dziugaite and Roy's 2017 result** — the first non-vacuous generalization bound for a real neural network — worked by *optimizing the bound itself*: rather than training the network and then computing the bound, they treated the right-hand side as an objective and searched for a posterior $Q$ making it small, effectively looking for the widest tolerable blob around a solution. The reported bound of roughly 0.16 error against a true error of about 0.03 is loose by a factor of five, which sounds unimpressive until you compare it with §2.3's bound of 33 and §2.4's bound of 138 on comparable problems. **It was the first time in the history of the subject that a rigorous bound for a real network was even the right order of magnitude.**

---

## 2.7 Uniform convergence may be unable to explain deep learning

Nagarajan & Kolter (2019) constructed settings where the learned classifier generalizes well, but **any** uniform-convergence bound (over any hypothesis set containing the learned predictors) must be vacuous. The mechanism: SGD's solutions have a complicated data-dependent boundary that hugs the training points; there exists a set of "adversarial" resampled datasets on which the same solution misclassifies badly, which uniform convergence must account for.

**The takeaway that should reshape your intuition:**

▸ Generalization in deep learning is **not** explained by the hypothesis class. It is explained by the *combination* of (architecture, optimizer, initialization, data). The optimizer's implicit bias is a first-class citizen, not an implementation detail. That's why Ch. 19 spends so long on it.

#### What "uniform convergence" means, and why it can fail on principle

**Define the term precisely, because the whole result hangs on it.** A bound is a **uniform-convergence** bound if it has this form:

$$\sup_{h\in\mathcal{H}}\ \big\lvert R(h) - \hat R_n(h)\big\rvert \ \le\ \text{something small}$$

Read aloud: *"the supremum over all h in the class of the absolute difference between true and empirical risk is small."* In English: **"no model anywhere in my class has a badly misleading training score."** Every bound in §§2.3–2.6 is of this form. It is not one technique; it is the entire classical toolkit.

**Why it seemed unavoidable.** You choose $\hat h$ by looking at training scores. If some model in the class had a wildly misleading score, the selection procedure would be *drawn toward it* — that is what minimizing does. So it looked as though controlling every model at once was simply what "guaranteeing the winner" requires.

**What Nagarajan and Kolter showed.** They built settings — deliberately simple ones, where the learned classifier provably generalizes well — in which **any** uniform-convergence bound must be near-vacuous. Not "the known bounds are loose." *Any* bound of that form, including ones not yet invented, applied to any hypothesis set containing the learned predictors.

**The mechanism, in plain terms.** Gradient descent produces a decision boundary that is mostly simple but has small, complicated wrinkles that hug the specific training points it saw. Now imagine resampling the training set adversarially: because the boundary's wrinkles are *keyed to the original points*, there exist alternative datasets — drawn from the very same distribution — on which that same learned function performs terribly. Uniform convergence has to account for those datasets, since it quantifies over everything. **So the bound is dragged down to the level of a failure mode that the actual training process never encounters.**

> **Analogy.** A tailor makes suits that fit every one of their customers beautifully, and does so reliably, year after year. You want to certify their skill. Uniform convergence insists on a guarantee of the form "*any* suit this tailor could produce fits *any* body" — and that is flatly false, because a suit cut for one customer fits another badly. **The certification fails even though the tailor never fails**, because the certification is quantifying over pairings that never occur. The tailor's skill lies in the *matching* of suit to customer, and a guarantee that ignores the matching cannot see it.

▸ **The logical structure here is worth naming, because it is rare.** Most negative results in a field say "our current tools are not good enough yet." This one says **the tools are the wrong shape, and sharpening them cannot help.** You are not being told to find a tighter bound; you are being told that no bound in this family exists. **That is why the field's attention moved from bounding $\mathcal{H}$ to characterizing what SGD actually converges to** — which is implicit bias, flat minima, margin maximization, the neural tangent kernel, and everything in Chapters 18, 19, and 30–31.

**One caution against over-reading it.** The result does not say that learning theory is useless, or that generalization is inexplicable. It says one specific proof strategy has a ceiling. PAC-Bayes bounds are *not* uniform-convergence bounds in the relevant sense — they bound a posterior's average risk rather than every hypothesis's risk — which is exactly why §2.6 survives and gets a number under 1.

> **Where this came from.** "Uniform Convergence May Be Unable to Explain Generalization in Deep Learning," by **Vaishnavh Nagarajan and J. Zico Kolter** (Carnegie Mellon), appeared at NeurIPS 2019 and received an outstanding-paper award. Its rhetorical achievement was to state, carefully and provably, a suspicion the field had held informally since the Zhang et al. experiment two years earlier — that the problem was structural rather than technical. **The gap between "everyone suspects it" and "someone proved it" is where a surprising fraction of scientific progress lives.**

---

## 2.8 The classical picture in one figure (in words)

```
Test error
   ^
   |  \                              CLASSICAL VIEW
   |   \                          (what you were taught)
   |    \        _____
   |     \      /
   |      \    /
   |       \__/  <- "sweet spot"
   |
   +---------------------------------> model capacity
      underfit   optimal    overfit
```

Everything on this page is correct and rigorously derived. And the picture is **incomplete**. Extend the x-axis far enough right and the curve comes back down. That's Chapter 18.

#### Reading the classical curve

**What each axis is.** Horizontal: *model capacity* — polynomial degree, number of parameters, VC dimension, whatever dial makes the class richer. Vertical: *test error* — the thing you actually care about, $R(\hat h)$. The curve is the composition of two monotone trends that §2.1's decomposition already gave you:

| Region | Bias | Variance | What you'd see in practice |
|---|---|---|---|
| Left (underfit) | high | low | Training and test error both high, and **close together** |
| Middle (sweet spot) | low | low | Training error low, test error low, small gap |
| Right (overfit) | ~zero | high | Training error ~0, test error rising, **large gap** |

▸ **The single most useful diagnostic that follows from this picture: look at the *gap*, not at the test error.** A large gap between training and test error means you are to the right — reduce capacity, add data, add regularization. A small gap with both errors high means you are to the left — the model is not expressive enough, and regularizing harder will make things strictly worse. **These two states look identical if you only track test error, and they call for opposite actions.**

**Map §2.2's table onto the drawing.** $d = 1$ is the left arm (bias 0.31). $d = 3$ is the trough (total 0.14). $d = 9$ is climbing the right arm (variance 0.19). $d = 19$ is off the top of the page (variance 4.8). **The picture is a smoothed version of that table, and the table is where the picture's authority comes from.**

**Why it went unquestioned for so long, which is the interesting part.** The curve is not wrong. It is a correct description of a regime — the regime where you have more data than parameters, which described essentially all of statistics from Gauss to about 2012. Fitting a 20-parameter model to 10,000 observations, you will observe this curve, every time. **The picture became misleading only when the field walked past the right-hand edge of the graph and kept going.**

> **Analogy.** A map of the known world drawn in 1400 is not a lie. It is accurate about everything its makers could reach, and the sea monsters at the margin are an honest notation for "we stopped here." The error is not in the map; it is in reading the edge of the parchment as the edge of the world. **Everything on this page is true out to the interpolation threshold. The parchment simply ended there.**

▸ **What sits just past the right margin.** At capacity $= n$ the model interpolates exactly, and the error peaks — that is the $4.8$ in §2.2's last row. Push *further*, to capacity well beyond $n$, and there are now infinitely many models that fit the data perfectly, so the optimizer gets to choose among them. It chooses a smooth one. **The error comes back down, often below the classical sweet spot.** Chapter 18 draws the missing half.

**A historical note on how well-hidden this was.** The peak at the interpolation threshold was not entirely unknown: the statistical-physics-of-learning literature of the late 1980s and early 1990s had already described error curves with a spike where the number of parameters matches the number of examples. But the observation lived in a different research community with different vocabulary, and it did not reach mainstream statistics or machine learning teaching. **The name "double descent" and the demonstration that it appears across modern models are due to Belkin, Hsu, Ma, and Mandal in 2019** — thirty years after the first sightings.

---

## 2.9 Practical translation

What actually controls generalization in a deep model you're training today, ranked by effect size:

1. **Amount and quality of data.** Dominates everything.
2. **Data augmentation / the effective size of the training distribution.**
3. **Optimizer + LR + batch size** (implicit regularization; $\eta/B$ is a temperature — Ch. 19).
4. **Weight decay** (directly shrinks $\|\theta\|$, which is the quantity in the PAC-Bayes bound).
5. **Architecture priors** (convolution = translation equivariance; attention = permutation equivariance + learned structure).
6. **Explicit regularizers** (dropout, label smoothing).
7. **Parameter count.** Weakest and most non-monotone of the lot.

Case Study A uses `weight_decay=0.01` with AdamW at `lr=3e-4`. In AdamW the decay is decoupled, so the per-step shrinkage is exactly $\theta \leftarrow (1-\eta\lambda)\theta = (1 - 3\times10^{-6})\theta$. Over 2,274 steps/epoch that's a factor $(1-3\times10^{-6})^{2274} = 0.9932$ per epoch of pure shrinkage — about **0.68% of the weight norm removed per epoch**, in the absence of gradients. Over 43 epochs, if gradients were zero, weights would shrink to $0.9932^{43} = 74\%$ of their initial norm. That's a real, non-negligible regularization pressure — and, per PAC-Bayes, it is *directly* the quantity your generalization bound depends on.

#### Reading the ranked list, and unpacking the weight-decay arithmetic

**On the ranking itself.** Notice that **parameter count is dead last**, below dropout. That ordering is the chapter's whole argument compressed into a list: the quantity classical theory put in the numerator of every bound turns out to be the weakest lever you have, and the quantity nobody bounded — *how much and what kind of data you show the model* — dominates everything. If you find yourself debating model size before you have exhausted items 1 and 2, you are optimizing the seventh-most-important variable.

**How each item maps back to the theory:**

| # | Lever | The theory it acts on |
|---|---|---|
| 1–2 | Data and augmentation | The $n$ in every $\sqrt{\cdot/n}$ in this chapter. Quadruple $n$, halve every bound |
| 3 | Optimizer, LR, batch size | Which minimum you land in → the $\sigma$ you can tolerate in §2.6's Gaussian KL |
| 4 | Weight decay | $\|\hat\theta\|^2$ — term 1 of the same KL, and the $B$ in §2.5's norm bound |
| 5 | Architecture priors | The choice of $P$ in $\mathrm{KL}(Q\|P)$. A prior that makes the right functions cheap |
| 6 | Explicit regularizers | Mostly act through 3 and 4 by other routes |
| 7 | Parameter count | The $p$ in §2.3's vacuous bound. Non-monotone (Ch. 18) |

▸ **Items 1 and 2 attack the denominator; items 3–6 attack the numerator; item 7 attacks a numerator that turned out not to be the right one.**

#### Unpacking the weight-decay arithmetic

Every number in that paragraph is worth deriving yourself once, because the same calculation applies to any run you will ever configure.

**Step 1 — the per-step shrinkage.** $\theta \leftarrow (1-\eta\lambda)\theta$ with learning rate $\eta = 3\times10^{-4}$ and decay $\lambda = 0.01$:
$$\eta\lambda = 3\times10^{-4}\times10^{-2} = 3\times10^{-6}$$
So each step multiplies the weights by $0.999997$. **Three parts in a million.** In isolation this looks like nothing.

**Step 2 — compound it over an epoch.** 2,274 steps per epoch:
$$(1 - 3\times10^{-6})^{2274} \approx e^{-3\times10^{-6}\times 2274} = e^{-0.00682} = 0.9932$$
(The $e^{-x}$ shortcut works because $(1-x)^k \approx e^{-kx}$ when $x$ is tiny — worth having at hand.) So **0.68% of the weight norm gone per epoch.**

**Step 3 — compound it over the run.** 43 epochs:
$$0.9932^{43} = e^{-0.00682\times43} = e^{-0.293} = 0.746$$
**A quarter of the weight norm removed over the run**, if gradients contributed nothing.

▸ **Sit with the ratio: three parts per million becomes a 25% effect.** This is the compounding arithmetic from Chapter 1 §1.1.2 ($\lambda^k$) and §1.1.4 ($1.2^{50}\approx9100$) once more — **any per-step multiplier that is not exactly 1 becomes a large number when raised to the power of a training run.** It is why learning-rate and weight-decay settings that look inconsequential move final results, and why "I only changed it slightly" is not a defence.

**What "decoupled" means and why it matters here.** In classic $\ell_2$ regularization you add $\frac\lambda2\|\theta\|^2$ to the loss, so the decay term passes through Adam's adaptive rescaling and gets divided by each parameter's own gradient-magnitude estimate — meaning parameters with small gradients get decayed *harder*, which nobody intended. **AdamW's "W" is the fix**: apply the shrinkage directly to $\theta$, outside the adaptive machinery. That is what makes the arithmetic above exactly right rather than approximately right; with coupled $\ell_2$ you could not compute it at all without knowing the gradient statistics.

**The connection back to §2.6, stated plainly.** The Gaussian PAC-Bayes KL had two terms, and the first was $\|\hat\theta\|^2/2\sigma_0^2$. Weight decay reduces $\|\hat\theta\|^2$. **So this hyperparameter is not acting on a proxy for generalization — it is acting on the literal quantity appearing in the only non-vacuous bound in the chapter.** Shrinking the weight norm by 25% cuts that KL term by $1 - 0.746^2 = 44\%$.

▸ **This is the rarest thing in applied deep learning: a knob whose effect on a rigorous guarantee you can compute exactly.** Almost everything else in §2.9's list works through mechanisms the theory can only gesture at. Weight decay is the one where the arithmetic closes.

#### Examples and non-examples: an honest held-out evaluation

Every bound in this chapter assumes the test sample is independent of everything used to produce the model. That assumption is cheap to state and remarkably easy to break — and when it breaks, the failure is silent and always in the flattering direction. This is the one place in the chapter where the theory touches your keyboard.

**✅  held out**

| Example | Why it qualifies |
|---|---|
| Medical images split **by patient**: all 40 scans of patient #17 land on one side | The unit of independence is the patient, and it was respected. The model cannot succeed by recognizing an individual |
| Time series: train on Jan–Sep, test on Oct–Dec | Test data is strictly in the future, which is the situation at deployment. No information flows backwards |
| Fitting the normalizer's mean and standard deviation on the **training fold only**, then applying those fixed numbers to test | Zero test information entered any fitted quantity, including the boring ones |
| Deduplicating near-identical documents **before** splitting | Removes the route by which the same content appears on both sides |
| A test set opened once, at the very end, after all decisions were frozen | Each look spends a little of its independence; one look spends the least |
| Choosing the epoch by **validation** loss and reporting **test** loss | The selection noise lands on the validation set, where you have already conceded it |

**❌ Near-misses — the code looks correct and the number is a lie**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| `train_test_split(X, y, random_state=42)` on a table with duplicate rows | The duplicate lands on both sides. The model is scored on rows it literally trained on | Memorization, measured and reported as generalization |
| Normalizing the whole dataset, *then* splitting | The mean and standard deviation carry test information into training | Preprocessing leakage — the most common leak in published notebooks |
| Random split of 100,000 frames drawn from 100 videos | Adjacent frames are near-identical images. The effective sample size is nearer 100 than 100,000 | Correlated samples: §1.3.1's $\sqrt{n}$ trap, with $n$ overcounted by $1000\times$ |
| Selecting the checkpoint with the lowest **test** loss | You just optimized against the test set. The number is a minimum over ~50 correlated noisy draws | Using the test set as a validation set; optimism of roughly one standard error, unbudgeted |
| Reporting the best of 200 runs against the same held-out set | Two hundred comparisons, one reported | Adaptive overfitting — the sequential-selection version of the dice example in §2.1 |
| Oversampling the minority class, *then* splitting | Copies of the same minority row are distributed across both sides | Duplicate leakage wearing a class-balance costume |
| A "fresh" test set collected by the same annotators, same week, same pipeline | Independent of the training *rows*, not of the training *distribution* | An i.i.d. check, not a robustness check. Says nothing about shift (Ch. 33) |

▸ **The boundary:** a split is honest only if **no quantity fitted on the training side — and no decision made by you — used information from the test side.** The unit of independence is never the row; it is whatever generated the correlated rows: a patient, a document, a video, a trading day, a web domain.

> **Common misconception.** *"I used a random split, so there's no leakage."* Randomness protects against *selection* bias, not against *dependence*. If your rows are correlated — frames of a video, sentences from one document, repeated visits by one patient — a uniformly random split scatters members of each correlated group across both sides, which is the worst possible outcome. Splitting by group is the fix, and it will make your reported numbers go down, which is the point. **The belief is tempting because `random_state=42` looks like the careful, unbiased choice**, and because the leak produces no error, no warning, and a better score. Every incentive points the wrong way. The habit worth building: before splitting, name out loud the thing that generated correlated rows, and split on *that*.

> **Common misconception.** *"My test set is clean — I've only checked it a dozen times."* A test set consulted repeatedly becomes a validation set, gradually and without announcing itself. Twelve looks with a decision after each is twelve rounds of selection, and the resulting optimism compounds much as the dice example in §2.1 does: the maximum of many noisy numbers drifts upward even when nothing improved. **The belief is tempting because each individual look feels harmless** — one glance at one number can hardly corrupt 10,000 examples. The corruption isn't in the glance; it's in the decision you make afterwards, because that decision transmits the test set's noise into your model. This is the argument for a  locked final test set, and Chapter 3 is the chapter about measuring anything at all under this kind of noise.

---

## Did you know?

- **A model with exactly one parameter can have infinite capacity.** The classifier $\mathbb{1}[\sin(\omega x) > 0]$ has a single real-valued knob $\omega$, and it can produce *any* labeling of *any* number of suitably placed points. Every claim of the form "more parameters means more overfitting" is refuted by one sine wave.

- **The counting lemma at the heart of learning theory was proved three times in about a year, in three unrelated fields.** Vapnik and Chervonenkis proved it in Moscow for statistical learning (1968/1971); Norbert Sauer proved it in 1972 as combinatorics, answering a question from Paul Erdős; Shelah and Perles proved it in 1972 as a tool in mathematical logic. It is now called the Sauer–Shelah lemma, the Sauer–Shelah–Perles lemma, or the Vapnik–Chervonenkis lemma, depending on whose book you're holding.

- **Occam never wrote Occam's razor.** William of Ockham argued for parsimony, but the famous Latin line — *entities should not be multiplied beyond necessity* — does not appear in his surviving works; it is usually traced to John Punch in 1639, roughly three centuries later. The English name "Occam's razor" appears to be a 19th-century coinage.

- **The theory of learning was published before there were computers worth applying it to.** Vapnik and Chervonenkis's foundational work appeared in Russian in 1968, and much of it went unread in the West for over twenty years — a delay of language, politics, and fashion. By the time the field caught up, the theory was waiting.

- **The most influential experiment in modern learning theory is a few lines of code.** Zhang, Bengio, Hardt, Recht, and Vinyals (2017) simply replaced CIFAR-10's labels with random noise and trained a network anyway. It hit 100% training accuracy — proving no capacity-based bound could ever explain why the same network generalizes on real labels. Anyone could have run it at any point in the preceding five years. Nobody did.

- **In 2019 someone proved that a whole family of proofs cannot work.** Nagarajan and Kolter showed that *any* uniform-convergence bound must be vacuous in settings where the learned model demonstrably generalizes. Most negative results say "our tools aren't sharp enough yet." This one says the tools are the wrong shape.

- **Statistical learning theory was about thirty years old before anyone computed a meaningful bound for a working neural network.** Dziugaite and Roy managed it in 2017, on MNIST, getting roughly 0.16 against a true error near 0.03. Loose by 5×, and a landmark — because the previous generation of bounds returned numbers like 33 and 138 on a quantity that cannot exceed 1.

- **The most cited pessimistic prediction in the field was careful, rigorous, and wrong.** Geman, Bienenstock and Doursat argued in 1992 that the variance of flexible models like neural networks was so severe that useful systems would need built-in structure rather than learned capacity. Given what was known in 1992, it was the reasonable conclusion.

- **The quantity SVMs were designed to maximize is exactly the quantity theory later identified as the right one.** Vapnik's margin is $1/\|w\|$, so maximizing margin is minimizing weight norm — and norm-based Rademacher bounds are *dimension-free*, which is why a support vector machine can generalize in an infinite-dimensional feature space. The algorithm got there first, for geometric reasons; the explanation arrived afterwards.

- **Herman Chernoff has spent decades trying to give away the Chernoff bound.** He has said in interviews that the key step was suggested by his colleague Herman Rubin and that the result should carry Rubin's name. It is one of the few known cases of a scientist campaigning against an eponym in his own favour.

- **The capacity measure named after Hans Rademacher arrived in machine learning seventy-eight years after he defined the functions.** Rademacher introduced his $\pm1$-valued functions in 1922 in a paper on orthogonal function systems; they became a learning-theory complexity measure around 2000–2002. He was dismissed from his chair at Breslau by the Nazi regime in 1934 for his pacifist and human-rights activity, and spent the rest of his career at the University of Pennsylvania.

- **CIFAR-10 is named after a funding agency.** It stands for the **Canadian Institute For Advanced Research**, which supported the group that assembled it. The dataset's name says nothing about images, categories, or resolution — a small monument to how research infrastructure gets named.

- **The "ten data points per parameter" rule of thumb is a real theorem in disguise.** Push §2.3's union-bound calculation through and non-vacuity requires roughly eleven examples per parameter. Every large model in this book violates that by two to four orders of magnitude — and generalizes anyway. The rule is not folklore; it is a  bound applied outside the regime where it means anything.

- **Michel Talagrand, whose concentration inequalities underwrite much of this chapter's machinery, won the Abel Prize in 2024.** The contraction lemma that makes ReLU "free" in the layer-wise bound is a small corner of a body of work about why functions of many independent random variables are overwhelmingly close to their averages.

---

## Check for Understanding

**Classical theory bounds the generalization gap by the complexity of what your model *could* have done, which for deep networks is so large the bounds say nothing — so the real explanation must live in what the optimizer *actually* does, which is why implicit bias, flatness, and description length turn out to matter more than parameter count.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What is the difference between true risk and empirical risk, and why is the second one a biased view of the first *only after you've chosen a model*?** (If you reach for "overfitting," push further — say the thousand-dice version.)
2. **Why does searching a larger set of models make your training score less trustworthy, even when every individual measurement is honest?**
3. **What are the three things that make a prediction wrong, and why does each one need a different fix?** Use the rifle, not the formula.
4. **Why do both cross-terms vanish in the bias–variance derivation?** (Correct answer: because deviations from an average average to zero — the same reason twice.)
5. **Why does a degree-19 polynomial through 20 points have zero bias and catastrophic error?**
6. **What does Hoeffding's inequality assume, and what does it deliberately not assume?** Why does halving your error bar cost four times the data, every time?
7. **What does it mean for a bound to be "vacuous," and why is a bound of 33 on a quantity that cannot exceed 1 a scientific problem rather than a technical one?**
8. **What does it mean for a hypothesis class to "shatter" a set of points, and why does a single-parameter sine wave shatter arbitrarily many?**
9. **Why did training a network on random CIFAR-10 labels demolish an entire branch of theory?** State the argument as an elimination: what is it that no capacity-based bound can possibly explain?
10. **Why is the norm of the weights a better capacity measure than the number of weights?** Why does that make a support vector machine work in infinite dimensions?
11. **What is PAC-Bayes charging you for, in terms of a telegram bill?** Why does that quantity stay small for a huge network that barely moved from its initialization?
12. **Why does a flat minimum give you a tighter generalization guarantee?** Say it in terms of how precisely you're forced to state your answer.
13. **Why is "simplicity" only meaningful relative to a prior — and in what sense is an architecture a prior?**
14. **Why does a per-step weight shrinkage of three parts in a million remove a quarter of the weight norm over a run?**

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 03 — Resampling & the Statistics of Noisy Evaluation](03-resampling-and-noisy-evaluation.md)
