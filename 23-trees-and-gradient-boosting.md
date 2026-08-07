# Chapter 23 — Trees & Gradient Boosting

> **Prerequisites:** Ch. 2 (bias–variance), Ch. 22.
> **Why this chapter matters more than its reputation suggests:** gradient boosting is still the best method for tabular data, it wins most tabular Kaggle competitions, and it is what most production ML systems outside of vision/NLP actually run. It is also a beautiful piece of mathematics.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $I(\text{node})$ | "impurity of the node" | How **mixed** the labels are in a group. Zero means all one class. |
| $p_k$ | "p-k" | The fraction of points in this node whose label is class $k$ |
| $n_L,\ n_R$ | "n-left, n-right" | How many training points land on the left / right side of a split |
| $\min_{j,s}$ | "minimize over j and s" | "Try every feature $j$ and every threshold $s$; keep the best pair" |
| $\lvert T_{\text{leaves}}\rvert$ | "the number of leaves of T" | The tree's size — its complexity |
| $\rho$ | "rho" | **Correlation** between two ensemble members' predictions |
| $\sigma^2$ | "sigma squared" | **Variance** of a single ensemble member |
| $B$ | "B" | How many trees are in the ensemble |
| $\mathbb{1}[\,\cdot\,]$ | "the indicator of" | **1 if the statement inside is true, 0 if false.** A yes/no written as a number. |
| $\eta$ | "eta" | **Learning rate / shrinkage** — what fraction of each new tree you actually add |
| $F_m(x)$ | "F-m of x" | The whole ensemble's prediction after $m$ rounds |
| $r_i^{(m)}$ | "r-i at round m" | **Pseudo-residual** — the direction example $i$ wants the prediction moved |
| $g_i,\ h_i$ | "g-i, h-i" | First and second derivative of the loss at example $i$ |
| $G_j,\ H_j$ | "big G-j, big H-j" | Those same derivatives **summed over every example sitting in leaf $j$** |
| $w_j$ | "w-j" | The single number the tree outputs for anything that lands in leaf $j$ |
| $\lambda,\ \gamma$ | "lambda, gamma" | Ridge penalty on leaf values; fixed toll charged per extra leaf |
| $\phi_j$ | "phi-j" | The **SHAP** credit assigned to feature $j$ for one prediction |

### Full forms for the abbreviations in this chapter

| Short | Full form |
|---|---|
| CART | Classification And Regression Trees |
| ID3 / C4.5 | Iterative Dichotomiser 3 / its successor, Quinlan's tree algorithms |
| MSE | mean squared error |
| SSE | sum of squared errors |
| OOB | out-of-bag |
| MDI | mean decrease in impurity |
| XGBoost | eXtreme Gradient Boosting |
| LightGBM | Light Gradient Boosting Machine |
| CatBoost | Categorical Boosting |
| GOSS | Gradient-based One-Side Sampling |
| EFB | Exclusive Feature Bundling |
| SHAP | SHapley Additive exPlanations |
| MART | Multiple Additive Regression Trees (Friedman's own name for gradient boosting with trees) |
| PAM | Partitioning Around Medoids |
| GBDT / GBM | gradient boosted decision trees / gradient boosting machine |

---

## 23.1 Decision trees

### The one-line idea

Recursively split the feature space on one variable at a time, choosing each split to make the resulting groups as pure as possible, until further splitting stops helping.

### The analogy

Twenty questions. Each question partitions the space of possible answers; a good question halves it. A decision tree is a game of twenty questions where the questions are chosen greedily to be maximally informative given what you've already asked.

### CART

At each node, search over all features $j$ and all thresholds $s$ for the split minimizing the weighted child impurity:

▸ $$\min_{j,s}\ \frac{n_L}{n}I(\text{left}) + \frac{n_R}{n}I(\text{right})$$

**Impurity measures.** For a node with class proportions $p_k$:

| Measure | Formula | Note |
|---|---|---|
| **Gini** | $1-\sum_k p_k^2$ | probability two random draws differ; CART default |
| **Entropy** | $-\sum_k p_k\log p_k$ | information-theoretic; ID3/C4.5 |
| Misclassification | $1-\max_k p_k$ | **not used for splitting** |
| **MSE** (regression) | $\frac1n\sum_i(y_i-\bar y)^2$ | |

▸ **Why misclassification error is a bad splitting criterion, even though it's the thing you care about:** it is piecewise-linear in $p$, so a split that moves probability mass without changing the argmax gives *zero* improvement. Gini and entropy are strictly concave, so **any** split that increases purity registers a gain. Concavity is what makes greedy search work at all. This is a  good interview question.

Gini and entropy agree on the chosen split ~98% of the time. Gini is marginally cheaper (no log).

**Information gain** $= H(\text{parent}) - \sum\frac{n_c}{n}H(\text{child})$, which is exactly the mutual information between the split variable and the label.

**Regression trees** predict the mean of each leaf and split to minimize the sum of squared errors.

#### Reading the split criterion in plain English

$$\min_{j,s}\ \frac{n_L}{n}I(\text{left}) + \frac{n_R}{n}I(\text{right})$$

Read aloud: *"over all features $j$ and all thresholds $s$, find the pair that minimizes the size-weighted average impurity of the two resulting groups."*

Every symbol:

| Symbol | Read aloud | What it is |
|---|---|---|
| $j$ | "j" | Which **column** of your table you're splitting on — age, income, pixel 47 |
| $s$ | "s" | The **cut point** on that column — "age $\le 38.5$" |
| $n$ | "n" | How many training points arrived at this node |
| $n_L,\ n_R$ | "n-left, n-right" | How many go left, how many go right. Always $n_L + n_R = n$. |
| $I(\cdot)$ | "impurity of" | A number saying **how mixed** the labels are in that group |
| $n_L/n$ | "n-left over n" | The **fraction** of points going left — a weight, so a big group counts more |

> **Analogy.** You run a mailroom with one big unsorted bin. You are allowed to install a single sorting rule — "postcode below 50000 goes in bin A, everything else in bin B" — and you want the two bins that come out to each be as *uniform* as possible. Impurity is "how jumbled is this bin." The weighting $n_L/n$ exists because making a bin of 3 letters perfectly pure while leaving 997 letters in chaos is not progress.

**Real numbers.** Take a node with 100 points: 50 spam, 50 not-spam. Try the split "contains the word *free*":

- **Left** ($n_L = 40$): 35 spam, 5 not. So $p_{\text{spam}} = 0.875$.
- **Right** ($n_R = 60$): 15 spam, 45 not. So $p_{\text{spam}} = 0.25$.

Gini of the parent: $1 - (0.5^2 + 0.5^2) = 0.5$ — the worst possible value for two classes.
Gini left: $1 - (0.875^2 + 0.125^2) = 1 - (0.766 + 0.016) = 0.219$.
Gini right: $1 - (0.25^2 + 0.75^2) = 1 - (0.0625 + 0.5625) = 0.375$.

Weighted child impurity: $0.4(0.219) + 0.6(0.375) = 0.0875 + 0.225 = 0.3125$.

▸ **The split bought you $0.5 - 0.3125 = 0.1875$ of impurity reduction.** The algorithm computes exactly this number for every column and every threshold — perhaps a million candidate splits — and keeps the single best one. Then it recurses on each child. That is the entire CART (Classification And Regression Trees) algorithm. There is no cleverness beyond exhaustive search plus recursion.

#### The impurity measures, decoded

All four rows of the table above answer the same question — *"how mixed is this group?"* — and differ only in how they punish mixture.

Take a two-class node and let $p$ be the fraction of class 1:

| $p$ (fraction class 1) | Gini $1-\sum_k p_k^2$ | Entropy $-\sum_k p_k\log_2 p_k$ | Misclassification $1-\max_k p_k$ |
|---|---|---|---|
| $1.0$ (pure) | $0$ | $0$ | $0$ |
| $0.9$ | $0.18$ | $0.469$ | $0.1$ |
| $0.75$ | $0.375$ | $0.811$ | $0.25$ |
| $0.5$ (worst) | $0.5$ | $1.0$ | $0.5$ |

- **Gini, $1-\sum_k p_k^2$.** Read: "one minus the sum over classes of the squared class fractions." The interpretation in the table — *"probability two random draws differ"* — is exact and worth checking: draw two points at random with replacement; the chance they match is $\sum_k p_k^2$; so the chance they differ is $1$ minus that. Gini is the **probability of disagreement**.
- **Entropy, $-\sum_k p_k\log p_k$.** Read: "minus the sum of p-log-p." This is Shannon's surprise measure (Ch. 1 §1.4): the average number of yes/no questions needed to identify a point's class. A 50/50 node needs exactly 1 bit; a pure node needs 0.
- **Misclassification, $1-\max_k p_k$.** Read: "one minus the biggest class fraction." If you had to guess one label for the whole node, this is your error rate. It is *the thing you actually care about at prediction time* — which makes it all the more surprising that it is a bad splitting criterion.
- **MSE (mean squared error), $\frac1n\sum_i(y_i-\bar y)^2$.** The regression version. $\bar y$ (read "y-bar") is the average target in the node. Impurity here means "spread": a leaf whose targets are all $7.2$ has impurity 0.

#### Why concavity is the whole trick

The ▸ note above is one of the sharpest ideas in the chapter, so here it is with numbers.

Take a node of 100 points, 60 class-A and 40 class-B. Misclassification error $= 1 - 0.6 = 0.4$. Now split it into two halves of 50:

- **Left:** 40 A, 10 B → error $= 0.2$.
- **Right:** 20 A, 30 B → error $= 0.4$.
- Weighted: $0.5(0.2) + 0.5(0.4) = 0.3$. Improvement $= 0.1$. Fine.

But try a *different* split, also into halves:

- **Left:** 35 A, 15 B → error $=0.3$.
- **Right:** 25 A, 25 B → error $=0.5$.
- Weighted: $0.5(0.3)+0.5(0.5) = 0.4$. Improvement $= \mathbf{0}$.

The second split clearly separated the classes better than no split at all — the left group went from 60% A to 70% A — and misclassification error **reported zero progress**, because in both children the majority label was unchanged.

Gini on the same second split: parent $=0.48$; left $=2(0.7)(0.3) = 0.42$; right $=0.5$; weighted $=0.46$. Improvement $=0.02$. Small, but **positive**.

▸ **This is what concavity buys you.** A strictly concave impurity function has the property that *any* redistribution of probability mass into two more-extreme groups strictly lowers the weighted average — the chord always lies below the curve. Misclassification error is piecewise **linear** in $p$, so the chord lies exactly *on* the curve over each linear piece and the improvement is exactly zero. A greedy algorithm that sees zero improvement stops. **Gini and entropy are curved so that greedy search always has a gradient to follow.**

> **Analogy.** You are hiking to the summit in fog and can only feel the ground under your feet. Misclassification error is a landscape made of flat terraces: you can be a hundred metres from a huge climb and feel perfectly level ground, so you stop. Gini and entropy are the same mountain resurfaced so that it always tilts *somewhere*. The peak is in the same place; you can now actually walk to it.

> **Where this came from.** Decision trees were invented independently in at least three fields. The earliest recognizable ancestor is **AID (Automatic Interaction Detection)**, built by **James Morgan and John Sonquist** at the University of Michigan's Survey Research Center and published in 1963 — they were social scientists trying to find *interactions* in survey data (does education affect income differently for men and women?) and were explicitly frustrated that regression assumed additivity. In the 1980s two lines matured in parallel: **CART**, from the 1984 book by **Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone**, which brought Gini, surrogate splits for missing values, and cost-complexity pruning; and **ID3 / C4.5**, from the Australian computer scientist **Ross Quinlan**, which came out of the artificial-intelligence tradition and used entropy and information gain. Statistics and AI arrived at nearly the same algorithm from opposite directions and with almost no shared vocabulary — which is why the same object has two names for everything.

> **The story behind "Gini."** The measure is named for **Corrado Gini**, the Italian statistician who introduced an index of statistical dispersion in 1912 — as a way of quantifying **income inequality**, which is still its most famous use. The identical quantity has been rediscovered repeatedly under other names: in ecology it is the **Simpson diversity index** (Edward H. Simpson, 1949), used to measure how many species a habitat contains; in economics, its complement $\sum_k p_k^2$ is the **Herfindahl–Hirschman index**, which United States antitrust regulators use to decide whether a proposed merger makes a market too concentrated. The number your gradient-boosting library computes several billion times per training run is the same one a competition authority computes to block a merger. (Gini was also an active supporter of Italian Fascism and wrote in defence of it; the mathematics is untouched by this, but the name carries the history.)

### Complexity control

- **Pre-pruning:** `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.
- **Cost-complexity (post-)pruning:** grow fully, then minimize
▸ $$R_\alpha(T) = R(T) + \alpha|T_{\text{leaves}}|$$
This traces a nested sequence of subtrees as $\alpha$ increases; pick $\alpha$ by cross-validation. Post-pruning is better than pre-pruning because a split that looks useless can enable a useful one below it (the XOR problem).

#### Unpacking cost-complexity pruning

$$R_\alpha(T) = R(T) + \alpha\lvert T_{\text{leaves}}\rvert$$

Read aloud: *"the alpha-penalized cost of tree $T$ equals its error plus alpha times how many leaves it has."*

- $T$ — a particular tree (a specific set of splits).
- $R(T)$ — the tree's **error** on the training data: misclassification rate, or sum of squared errors for regression.
- $\lvert T_{\text{leaves}}\rvert$ — the **number of leaves**, i.e. how many distinct predictions the tree can make. This is the tree's complexity. The vertical bars mean "size of the set," exactly as $\lvert S\rvert$ means "how many elements are in $S$."
- $\alpha$ — the **price per leaf**, a number you choose. It has units of "error per leaf."

> **Analogy.** You are furnishing a room and each extra piece of furniture costs rent. $R(T)$ is how uncomfortable the room is; $\alpha\lvert T_{\text{leaves}}\rvert$ is the monthly bill. At $\alpha = 0$ rent is free, so you cram in every chair you own — that is the fully grown tree, which fits the training data perfectly and generalizes terribly. Raise the rent and you start throwing out the chairs nobody sits in. Raise it far enough and you keep one chair — a stump — and eventually none at all.

**Real numbers.** Suppose a fully grown tree has 400 leaves and 0 training errors out of 1000 points, so $R(T) = 0$. A pruned version has 12 leaves and 60 errors, so $R = 0.06$.

- At $\alpha = 0.0001$: full tree scores $0 + 0.04 = 0.04$; pruned scores $0.06 + 0.0012 = 0.0612$. **Full tree wins.**
- At $\alpha = 0.0005$: full scores $0.20$; pruned scores $0.066$. **Pruned wins.**

Somewhere between those two values the winner flips. ▸ **The important structural fact is that as you sweep $\alpha$ from 0 upward, the winning tree changes only finitely many times, and each new winner is a subtree of the previous one.** That is why the text says it "traces a nested sequence." You do not have to search over the astronomically many possible subtrees — you get a short list, maybe twenty candidates, and cross-validation picks among them. A search problem was converted into a sorting problem, the same move Eckart–Young makes for matrices (Ch. 1 §1.1.3).

**Why post-pruning beats pre-pruning — the XOR problem, concretely.** Take two binary features $x_1, x_2$ and the label $y = x_1 \oplus x_2$ (exclusive-or: 1 when exactly one of them is 1). Split on $x_1$ alone: the left child is 50/50, the right child is 50/50. **Zero impurity reduction.** Same for $x_2$. A pre-pruning rule like `min_impurity_decrease` looks at this and stops immediately — declaring the data unlearnable. But split on $x_1$ *anyway*, and then split each child on $x_2$, and you get four perfectly pure leaves.

▸ **Greedy search is myopic by construction, and the fix is to be greedy on the way down and thoughtful on the way up.** Grow the tree past the point of apparent usefulness, then prune back with the benefit of hindsight. This is a general pattern well beyond trees: exploring past a local plateau and retracting later beats refusing to explore.

### Properties

**Good:** no scaling needed, handles mixed types and missing values natively (surrogate splits), captures interactions automatically, interpretable, invariant to monotone feature transforms.

**Bad:** ▸ **extremely high variance** — change one data point near a root split and the entire tree below changes. Axis-aligned splits struggle with diagonal boundaries. Cannot extrapolate outside the training range. Biased toward high-cardinality features.

That high variance is the exact weakness that ensembling was invented to fix.

#### What "extremely high variance" actually means here

**Variance**, in the bias–variance sense (Ch. 2), is not about the spread of your data. It is about how much your *fitted model* would change if you had been handed a different random sample from the same source. High variance means: **train on a slightly different dataset, get a noticeably different model.**

For a tree, the mechanism is brutally direct. The root split is chosen by an argmax over maybe a million candidates. Suppose the best split scores $0.1875$ and the runner-up scores $0.1871$. Delete one training row, and those two numbers can swap places. The tree now splits on a *different feature at the root* — and every node beneath it is re-grown on entirely different subsets of the data. **One row changed; ninety percent of the model changed.**

> **Analogy.** A tree is a tournament bracket. The champion is determined by a chain of pairwise contests, and a single upset in the first round rewrites the entire remainder of the bracket. Averaging the results of a hundred tournaments is far more stable than any one of them — which is precisely the ensembling argument in §23.2.

Contrast this with linear regression, where one deleted row moves each coefficient by roughly $O(1/n)$ — the model changes a little, everywhere, rather than a lot, discontinuously. ▸ **Trees are high-variance because their output is a discrete structure chosen by argmax, and argmax is discontinuous.** That single sentence explains why every practical tree method you will actually deploy is an ensemble.

**The other weaknesses, made concrete:**

- **Axis-aligned splits and diagonal boundaries.** A tree can only ask "is $x_1 > 3$?", never "is $x_1 + x_2 > 3$?". To approximate the diagonal line $x_2 = x_1$ it must build a staircase. With depth $d$ it gets $2^d$ steps — so depth 10 gives about a thousand steps, which is visually fine but cost a thousand leaves to express what linear regression writes with two coefficients.
- **Cannot extrapolate.** A leaf predicts a constant. If every training house cost under \$2M, a tree can never predict \$5M for a mansion — the prediction is capped by the largest leaf mean it ever saw. Linear regression happily extrapolates (sometimes to nonsense, but it *can*). ▸ **This is the single most common surprise when a tree model is deployed on data that drifted upward**: forecasts flatten out at the ceiling of the training range and stay there.
- **Biased toward high-cardinality features.** A feature with 1000 distinct values offers 999 candidate thresholds; a binary feature offers one. More lottery tickets means a better chance of a spuriously good score, so the high-cardinality feature wins the argmax more often than it deserves. This is the same multiple-comparisons problem that makes p-hacking work, wearing a different hat.

#### Examples and non-examples: what a decision tree can represent

**✅ Things a CART tree handles natively**

| Example | Why it qualifies |
|---|---|
| "If `income > 50000` **and** `age < 30`, approve" | A root-to-leaf path is a conjunction of axis-aligned tests — that is literally what a tree is |
| Rescaling `income` from dollars to thousands of dollars | Splits are chosen from rank order, so any monotone transform yields an identical tree |
| An XOR target on two binary features | Depth 2 solves it exactly: split on $x_1$, then on $x_2$ in both children |
| A  step change in the target at `age = 65` | A leaf predicts a constant, so steps are the tree's native output shape |
| A column with 40% missing values | Surrogate splits (or a learned default direction) route the missing rows without imputation |

**❌ Near-misses — look tree-friendly, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| The boundary $x_1 + x_2 > 3$ | The tree can only ask about one feature per node, so it must build a staircase | An **oblique** boundary — needs an oblique tree or a linear model |
| Predicting \$5M for a mansion when every training house was under \$2M | Every leaf is a constant learned from training rows; there is no slope to ride outward | **Extrapolation**, which trees structurally cannot do |
| A smooth relationship such as $y = \log x$ | Approximated by a staircase; a good fit costs many leaves | Piecewise-constant approximation, not a smooth fit |
| "Trees are invariant to feature transformations" | Only **monotone** ones. Rotate the feature space by 45° and you get a completely different tree | Monotone invariance, which is narrower than the slogan suggests |
| A `customer_id` column landing near the top of the split ranking | It offers $n-1$ candidate thresholds, so one of them looks good by luck | **High-cardinality bias**, a multiple-comparisons artifact |

▸ **The boundary:** a tree carves the space into axis-aligned boxes and predicts a constant inside each one. Anything requiring a slope, a diagonal, or a value outside the observed range is outside what the model can express — and no depth setting repairs it.

> **Common misconception.** *"Decision trees are interpretable, so tree ensembles are interpretable too."* A single depth-3 tree is  readable: eight leaves, a page of if-statements you could hand to a regulator. **A tuned gradient-boosting model is 800 trees of depth 6, and the prediction is the sum of 800 numbers — nobody reads that.** And even the single tree's interpretability is fragile in exactly the way this section describes: delete one row, the root split flips, and the explanation you presented last week is a different explanation of the same data. The belief is tempting because trees have been marketed on interpretability for forty years and the illustration in every textbook has four nodes. **This is precisely why §23.8 exists — once you ensemble, reading the model stops being an option and you need SHAP or something like it.**

---

## 23.2 Bagging and random forests

### The variance formula — the heart of the matter

For $B$ estimators each with variance $\sigma^2$ and pairwise correlation $\rho$:

▸ $$\mathrm{Var}\left(\frac1B\sum_{b}f_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

**Read it carefully.** The second term vanishes as $B\to\infty$. The first does not.

▸ **Averaging more trees can never reduce variance below $\rho\sigma^2$. Therefore the only way to improve a large ensemble is to reduce the correlation $\rho$ between its members.** This single formula explains the entire design of random forests, and it is the thing to say if asked "why random feature subsets?"

#### Reading the ensembling variance formula in plain English

$$\mathrm{Var}\left(\frac1B\sum_{b}f_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Read aloud: *"the variance of the average of B predictors equals rho sigma-squared, plus one-minus-rho over B, sigma-squared."*

Every symbol:

| Symbol | Read aloud | What it is |
|---|---|---|
| $f_b$ | "f-b" | The prediction of tree number $b$ |
| $B$ | "B" | How many trees you are averaging |
| $\frac1B\sum_b f_b$ | "one over B, sum over b, of f-b" | The ensemble's prediction — just the plain average |
| $\mathrm{Var}(\cdot)$ | "variance of" | How much this quantity would jump around if you re-drew the training data |
| $\sigma^2$ | "sigma squared" | The variance of **one** tree on its own |
| $\rho$ | "rho" | How **correlated** two different trees' predictions are, from $0$ (independent) to $1$ (identical) |

> **Analogy — the polling analogy is exact.** You want to know who will win an election, and you can commission $B$ polls. If every pollster works independently, averaging their results shrinks your error toward zero as you buy more polls. But suppose all of them buy their phone lists from the *same vendor*, and the list systematically under-represents rural voters. Then every poll is wrong in the same direction. Commissioning a thousandth poll does not help at all: you have bought a thousand copies of the same bias. **$\rho$ is "how much do the pollsters share a blind spot," and $\rho\sigma^2$ is the error that no amount of money removes.**

**Now put numbers on it.** Take $\sigma^2 = 1$ (one tree's variance, in whatever units) and watch what the two terms do:

| $\rho$ | $B=1$ | $B=10$ | $B=100$ | $B=\infty$ (the floor) |
|---|---|---|---|---|
| $0.0$ | $1.00$ | $0.100$ | $0.010$ | $\mathbf{0.00}$ |
| $0.2$ | $1.00$ | $0.280$ | $0.208$ | $\mathbf{0.20}$ |
| $0.5$ | $1.00$ | $0.550$ | $0.505$ | $\mathbf{0.50}$ |
| $0.9$ | $1.00$ | $0.910$ | $0.901$ | $\mathbf{0.90}$ |

Read the last column. At $\rho = 0.9$ — trees that mostly agree — you can average a *million* of them and still carry 90% of a single tree's variance. At $\rho = 0.2$ you carry 20%. **The entire benefit of ensembling lives in that column, and $B$ barely appears in it.**

Look also at how fast the $B$-dependent term dies. At $\rho = 0.5$, going from $B=10$ to $B=100$ improves variance from $0.550$ to $0.505$ — under one percent of the total. ▸ **This is why "more trees is monotonically better but rapidly boring."** Past a few hundred trees you are paying linearly more compute for effectively nothing. If you want a better forest, you cannot buy it with $B$; you have to buy it with $\rho$.

**Sanity-check the two extreme cases**, which is the fastest way to convince yourself a formula is right:

- $\rho = 1$ (all trees identical): formula gives $1\cdot\sigma^2 + 0 = \sigma^2$. Correct — averaging a hundred copies of the same thing is the same thing.
- $\rho = 0$ (all trees independent): formula gives $0 + \sigma^2/B$. Correct — this is the ordinary $1/n$ variance-of-a-mean rule from Ch. 1 §1.3.1, the same $\sigma/\sqrt{n}$ that governs every average you have ever computed.

▸ **So the formula is a smooth interpolation between "you have $B$ opinions" and "you have one opinion repeated $B$ times," and $\rho$ is the dial.** Every design decision in the next two sections — bootstrap resampling, random feature subsets, random thresholds — is an attempt to turn that dial down without wrecking $\sigma^2$ in the process.

### Bagging

Train each tree on a bootstrap resample. Reduces $\rho$ somewhat (different data), but the trees still see the same features and tend to pick the same strong splits at the root, so $\rho$ stays high.

**Out-of-bag estimation:** each bootstrap omits $(1-1/n)^n\to e^{-1}=36.8\%$ of the data. Evaluate each point using only the trees that didn't see it — **a free cross-validation estimate** with no extra fitting.

#### Where the 36.8% comes from, and why it is free

**What a bootstrap resample is.** You have $n$ training rows. Draw $n$ rows *with replacement* — meaning after each draw you put the row back, so it can be drawn again. The result is a dataset the same size as the original, but with some rows appearing twice or three times and some rows missing entirely. Each tree in a bagged ensemble gets its own such resample.

**The arithmetic, step by step.** Focus on one particular row, say row 7.

- The chance that a single draw is *not* row 7 is $1 - \frac1n$. (With $n = 1000$, that's $0.999$.)
- There are $n$ independent draws, so the chance row 7 is missed by all of them is $\left(1 - \frac1n\right)^n$.
- With $n = 1000$: $0.999^{1000} = 0.3677$.
- And as $n$ grows this converges to $e^{-1} = 0.3679$, one of the standard limits of calculus: $\left(1-\frac1n\right)^n \to e^{-1}$.

▸ **So roughly 37% of your rows are left out of any given bootstrap, and 63% are in — and this ratio is essentially the same whether $n$ is 100 or 100 million.** It does not depend on your data at all. It is a fact about drawing marbles.

> **Analogy.** A raffle with 1000 tickets, where 1000 winners are drawn and each ticket is returned to the drum after being drawn. About 632 distinct people win something; about 368 go home empty-handed, no matter how large you make the raffle.

**Why this gives free validation.** Row 7 was omitted from about 37% of your trees. Those trees have never seen it — as far as they are concerned it is held-out test data. So: predict row 7 using *only* the trees that omitted it, and you have an honest out-of-sample prediction. Do this for every row and you get a full validation curve.

**Real numbers.** With $B = 500$ trees, each row is out-of-bag for about $500 \times 0.368 \approx 184$ of them. That is a substantial sub-ensemble — enough to give a stable estimate. ▸ **Out-of-bag (OOB) error is approximately the leave-one-out cross-validation error of the full forest, obtained at zero additional cost**, because the omissions you needed for validation were already happening for a different reason. That is a  rare thing in machine learning: a diagnostic that comes free with the training you were doing anyway.

**The caveat worth stating:** OOB error is computed with sub-ensembles of size $\approx 0.37B$ rather than $B$, so it is very slightly *pessimistic*. With a few hundred trees this bias is negligible. With twenty trees it is not.

> **Where this came from.** The **bootstrap** was introduced by **Bradley Efron** at Stanford in 1979, generalizing an older technique called the **jackknife** (Maurice Quenouille, 1949; developed by John Tukey in the 1950s). Tukey named the jackknife after the pocket knife — a rough, all-purpose tool you carry because it does many jobs adequately rather than one job perfectly. Efron's bootstrap took its name from the phrase "to pull oneself up by one's bootstraps," describing the apparently impossible feat the method performs: **estimating how much your answer would vary across datasets you do not have, using only the one dataset you do have.** **Leo Breiman** applied it to model ensembling in 1996, coining "bagging" from **b**ootstrap **agg**regat**ing** — and adding out-of-bag estimation as a side benefit of the resampling that was already occurring.

### Random forests

Bagging **plus** a random subset of $m$ features considered at each split.

▸ $m=\sqrt p$ for classification, $p/3$ for regression.

**This is the key addition:** by preventing every tree from using the same dominant feature at the top, it directly attacks $\rho$ in the formula above. The individual trees get slightly worse (higher $\sigma^2$), but $\rho$ drops more, and the ensemble improves. **A deliberate bias-for-decorrelation trade.**

**Extremely Randomized Trees (ExtraTrees):** also choose the split *threshold* at random rather than optimally. Even lower $\rho$, even higher individual bias, faster to train. Often competitive.

#### Unpacking $m=\sqrt{p}$ — why crippling each tree improves the forest

Notation first: $p$ is the **number of features** (columns) in your data, and $m$ is how many of them each individual split is allowed to look at, chosen fresh and at random at *every node*. Not once per tree — once per split.

With $p = 100$ features, $m = \sqrt{100} = 10$. At each node the tree flips a coin to pick 10 of the 100 columns, finds the best split among only those, and ignores the other 90.

**This sounds like sabotage, and it is.** The obvious objection: if feature 3 is  the most informative column, why would you deliberately hide it from 90% of the nodes?

The answer is the variance formula. Run the numbers:

- **Plain bagging.** Every tree sees all 100 features at the root. Feature 3 is the best split in most bootstrap samples, so almost every tree splits on feature 3 at the root, and often on feature 7 at depth 1, and so on. The trees look nearly identical. Say $\rho = 0.75$, $\sigma^2 = 1$. Floor: $\rho\sigma^2 = \mathbf{0.75}$.
- **Random forest.** Feature 3 is available at the root of only $10/100 = 10\%$ of trees. The other 90% must build a different structure from the start. Each tree is  worse — say $\sigma^2$ rises from $1$ to $1.3$. But $\rho$ falls from $0.75$ to $0.30$. Floor: $0.30 \times 1.3 = \mathbf{0.39}$.

▸ **Individual quality got 30% worse and the ensemble got roughly twice as good.** That is the trade, and the variance formula is what makes it legible: $\rho$ enters the floor multiplicatively and you were able to move it much further than you moved $\sigma^2$.

> **Analogy.** You are assembling a panel of forecasters. Every one of them reads the same newspaper. Forbid each of them, at random, from reading most of the newspaper, and individually they become less accurate — but they now make *different* mistakes, and the panel average is better than before. Diversity of error is worth more than individual accuracy once you are averaging.

**Why $\sqrt{p}$ and $p/3$ specifically?** These are empirical defaults from Breiman's experiments, not theorems. The logic behind the classification/regression split: regression trees extract less signal per split (they are fitting a continuous target, so a single split reduces less of the total variance), so they need a larger candidate pool to make progress. Both are worth tuning — on data with many junk columns, a larger $m$ helps because small $m$ too often offers a node nothing but noise.

**ExtraTrees pushes the same dial one notch further.** Instead of finding the *best* threshold among the $m$ features, it draws a threshold uniformly at random for each and keeps the best of those random cuts. $\rho$ drops further still, individual bias rises further still, and — crucially — it is much **faster**, because no sorting of feature values is needed at all. The candidate list went from "every value in the column" to "one number I made up."

> **Where this came from.** **Random forests** were published by **Leo Breiman** in 2001, but he was assembling parts that already existed. **Tin Kam Ho**, at Bell Labs, had published the **random subspace method** in 1995 — build each tree on a random subset of features — and **Yali Amit and Donald Geman** had published randomized trees in 1997 while working on shape recognition. Breiman's contribution was to combine bootstrap sampling with *per-node* feature randomization and to supply the analysis explaining why it works: the generalization error of a forest is bounded by a quantity involving exactly the strength/correlation trade-off above. Breiman came to this late and from an unusual direction — he had left academia in 1967 to work as a full-time statistical consultant for thirteen years before returning to Berkeley, and his industrial experience shows in the design: random forests need almost no tuning, which is a priority you develop by shipping models rather than proving theorems. **ExtraTrees** came from **Pierre Geurts, Damien Ernst, and Louis Wehenkel** at Liège in 2006.

#### Examples and non-examples: what bagging actually averages away

The variance formula $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ says averaging destroys only the part of the error that is **uncorrelated across members**. That one fact settles what does and does not benefit from being bagged.

**✅  variance reduction by averaging**

| Example | Why it qualifies |
|---|---|
| 500 deep trees, each grown on a different bootstrap sample | The trees make *different* mistakes, so the second term shrinks like $1/B$ |
| Restricting each node to $m=\sqrt p$ features | Attacks $\rho$ itself, which lowers the **floor** rather than the term above it |
| ExtraTrees drawing thresholds at random | Same mechanism pushed harder: more diversity, lower $\rho$ |
| Averaging 5 neural networks trained from different seeds | Different initializations  decorrelate the errors |

**❌ Near-misses — averaging that buys nothing**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Bagging 500 ordinary linear regressions | Linear regression is already low-variance, so $\rho \approx 1$ and the floor equals the original error | Wasted compute — you get back roughly what you started with |
| Bagging 500 depth-1 stumps | Each stump is nearly determined by the data, so they are near-identical | A **bias** problem, which averaging cannot touch |
| Training 500 trees on the same rows with no randomization | $\rho = 1$ exactly | One tree, computed 500 times |
| Averaging the 500 trees' training accuracies | An average of scores is not an ensemble prediction | A summary statistic |
| Bagging to fix predictions that are systematically 10% too low | Averaging removes scatter, never offset | Bias — boosting is the tool for that |

▸ **The boundary:** bagging removes the part of your error that would change if you had been handed a different sample. If your model would make the same mistake on every resample, averaging is arithmetic with no effect.

> **Common misconception.** *"More trees means more capacity, so a 5000-tree random forest will overfit."* It will not. Each new tree is one more sample in an average, and averages converge — as $B$ grows, the ensemble's variance descends to the floor $\rho\sigma^2$ and stops there. **You cannot overfit a random forest by adding trees; you can only spend time and memory.** (You *can* overfit one by growing the trees too deep on too few rows — but that is a different knob, and it is a property of the members, not of how many there are.) The belief is tempting because in nearly every other model family "more of it" means more capacity — more layers, more parameters, more boosting rounds — and boosting lives two sections away in this same chapter behaving in exactly the opposite way. **The question that settles it every time: are the members being averaged, or added?**

**Feature importance.** Two kinds, and the difference matters:
- **MDI (mean decrease in impurity):** sum of impurity reductions, weighted by samples. Fast, but ▸ **biased toward high-cardinality and continuous features**, and computed on training data.
- **Permutation importance:** shuffle a feature on held-out data and measure the performance drop. Slower, unbiased with respect to cardinality, but ▸ **misleading with correlated features** — shuffling one of two correlated features shows no drop because the other covers for it, so both look unimportant.

#### Feature importance, decoded — and why both kinds lie

**MDI (mean decrease in impurity)**, sometimes called "Gini importance," is computed like this: every time any tree splits on feature $j$, add up the impurity reduction that split achieved, weighted by how many samples reached that node. Sum over all nodes and all trees. Normalize so the numbers add to 1.

It is free — the impurity reductions were already computed during training — which is why it is the default in almost every library. It is also **measured on training data**, which is the tell: a quantity computed on the data the model memorized cannot be trusted to say what the model *needs*.

**Permutation importance** asks a cleaner question. Take your held-out validation set. Record the model's score. Now take column $j$ and shuffle its values randomly among the rows — destroying any relationship it had with the label while keeping its distribution intact. Score again. The drop is the importance.

> **Analogy.** MDI is asking a factory "how many times did you reach for this tool?" Permutation importance is hiding the tool and seeing whether production falls. The second question is better, but it has its own blind spot: if there is an identical spare tool on the next bench, hiding one changes nothing, and you conclude — wrongly — that neither tool matters.

**The correlated-feature failure, with numbers.** Suppose `height_cm` and `height_inches` are both in your table, perfectly correlated. Any model can use either.

- **Permutation importance** of `height_cm`: shuffle it, and the trees that used it now get noise — but the model has plenty of splits on `height_inches` too, and predictions barely move. Drop $\approx 0$. Same for `height_inches`. **Both features score near zero, and height looks irrelevant.** It is in fact the most important variable you have.
- **MDI** does something different but equally wrong: the two features *split* the credit between them, so each gets roughly half of what height deserves, and both are ranked below a single uncorrelated feature of  lower importance.

▸ **Neither method measures "how much does the world depend on feature $j$." Both measure "how much does *this fitted model* depend on feature $j$," and the fitted model made arbitrary choices among interchangeable columns.** The practical rule: cluster your features by correlation first, and report importance at the cluster level, not the column level. Failing that, permute correlated groups together.

**The high-cardinality bias in MDI, made concrete.** Add a column of pure random unique IDs to your data — a customer ID, say. It carries zero information. But it offers $n-1$ candidate thresholds, so somewhere among them there is a split that looks good on the training data by chance. Trees will use it, impurity reductions will accumulate, and MDI will rank a meaningless ID column above real predictors. **This is not a subtle effect; it is routinely dramatic**, and it is the single most common reason a feature-importance chart is misleading. A useful sanity check: add a column of pure random noise and see where it ranks. Anything scoring below it is noise.

**Practical guidance:** random forests need almost no tuning (more trees is monotonically better, just slower), are hard to overfit, and make an excellent baseline. But on most tabular problems, well-tuned gradient boosting beats them.

#### Examples and non-examples: what a feature-importance number tells you

**✅ Claims a feature-importance number  supports**

| Example | Why it qualifies |
|---|---|
| "This model leans heavily on `tenure_months`" | Both MDI and permutation measure exactly that — the fitted model's reliance |
| "Removing `zip_code` would degrade *this* model" | Permutation importance on held-out data measures that directly |
| "Half the importance mass sits in three columns, so the rest may be droppable" | A statement about model dependence, and testable by retraining without them |
| "A pure-noise column ranks 7th, so everything below rank 7 is noise" | The null-column sanity check — a real, decidable comparison |

**❌ Near-misses — claims it does not support**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "`num_prior_ER_visits` is important, so reducing ER visits improves outcomes" | Importance is read off a predictive model fit to observational data; the arrow may run the other way | A **correlational** finding dressed up as an intervention |
| "`height_inches` scored 0.001, so height doesn't matter" | Its perfectly correlated twin `height_cm` covered for it during permutation | A **correlated-feature artifact** — permute the group together |
| "`customer_id` is our strongest predictor" | $n-1$ thresholds means $n-1$ chances to win the argmax by luck | High-cardinality bias in MDI |
| "Feature A scored twice feature B, so it matters twice as much" | The numbers have no units and do not form a ratio scale | An ordering, and a shaky one |
| A SHAP value of $+0.4$ for a row | It exactly attributes *this model's output* on *this row* | A faithful attribution of a prediction — still not a causal effect |
| An importance computed on the training set | The model memorized that data; reliance there need not mean necessity | MDI, which is why held-out permutation exists |

▸ **The boundary:** every importance method answers "how much does **this fitted model** depend on this column?" — never "how much does **the world** depend on it." The model chose arbitrarily among interchangeable columns, and it learned from data in which nothing was randomized.

> **Common misconception.** *"The importance chart shows what drives the outcome, so it shows me what to change."* It shows what the model leaned on — a fact about your fitted model and your data-collection process, not about cause and effect. The standard cautionary case: a hospital-readmission model puts "number of prior ER visits" at the top, but prior visits do not *cause* readmission, they mark severity; a policy that discourages ER visits moves the marker while making outcomes worse. **Causal claims require randomization or an explicit causal model, and a gradient-boosting fit supplies neither.** The belief is tempting because the output is a tidy ranked bar chart with the word "importance" printed on it, and because the model  does predict well — but predicting well and knowing why are separate achievements, and nothing in the training objective ever asked for the second one.

---

## 23.3 Boosting

### The one-line idea

Instead of averaging independent strong learners, build weak learners **sequentially**, each one fixing what the current ensemble gets wrong.

### The analogy

Bagging is asking a hundred people independently and averaging. Boosting is a relay of specialists: the first gives a rough answer, the second is hired specifically to fix the first's mistakes, the third to fix what remains. Each is weak alone; the sequence is strong.

▸ **The bias–variance contrast:** bagging reduces **variance** (average many low-bias, high-variance trees). Boosting reduces **bias** (sum many high-bias, low-variance trees — "stumps" of depth 3–6). They are attacking opposite terms of the same decomposition, which is why boosting uses shallow trees and forests use deep ones.

#### Bagging versus boosting, decoded

The bias–variance decomposition (Ch. 2) says your expected error splits into three pieces:

$$\text{error} = \underbrace{\text{bias}^2}_{\text{systematically wrong}} + \underbrace{\text{variance}}_{\text{unstable}} + \underbrace{\text{noise}}_{\text{irreducible}}$$

- **Bias** = how wrong your model is *on average across all possible training sets*. A model too simple to represent the truth has high bias, and more data will not save it.
- **Variance** = how much your model jumps around *between* training sets. Covered above in §23.1.
- **Noise** = the part of the target nothing can predict. It is a property of the world, not of you.

Now the two strategies, side by side:

| | Bagging / random forests | Boosting |
|---|---|---|
| Trees are trained | **in parallel**, independently | **sequentially**, each depending on all previous |
| Each tree is | deep, complex, **overfit on purpose** | shallow, weak, **underfit on purpose** |
| Individual tree has | low bias, high variance | high bias, low variance |
| The ensemble is combined by | **averaging** | **adding up** |
| The combination fixes | variance | bias |
| Add too many trees and | nothing bad happens | you eventually overfit |
| Parallelizes | trivially | not across trees (only within a tree's split search) |

> **Analogy.** Bagging is polling a hundred experts at once and taking the average — you are cancelling out their individual quirks. Boosting is a production line: the first worker roughs out the shape, the second is hired specifically to fix what the first left wrong, the third fixes what remains. **Averaging cancels noise; a production line accumulates skill.** You would not average the outputs of a production line, and you would not run pollsters in sequence.

▸ **The tell that distinguishes them in one glance: does adding more members ever hurt?** For a forest, no — more trees only refines the average, and the variance formula shows why it converges rather than degrading. For boosting, yes — each new tree is a  addition to model capacity, so eventually you start fitting noise. That is why boosting needs early stopping and forests do not, and it is the practical reason §23.7 puts `n_estimators` with early stopping at the top of the tuning list.

**Why shallow trees for boosting?** A tree of depth $d$ can express interactions among at most $d$ features (each path from root to leaf asks $d$ questions). Depth 1 — a "stump" — gives a purely **additive** model: no interactions at all. Depth 3 allows three-way interactions. Since boosting will build hundreds of these and *add* them together, each one only needs to contribute a small correction; making each one deep would hand it enough capacity to overfit its own residuals in a single step. ▸ **`max_depth` in a boosted model is not really a capacity knob — it is an "interaction order" knob**, and that is the more useful way to think when tuning it.

#### Examples and non-examples: bagging versus boosting

**✅  bagging**

| Example | Why it qualifies |
|---|---|
| A random forest: 500 deep trees on bootstrap samples, predictions **averaged** | Parallel, independent members combined by averaging; it attacks variance |
| ExtraTrees | The same recipe with extra randomization at the threshold |
| Averaging 5 neural networks trained from different seeds | Independent members, averaged |
| Out-of-bag error estimation | Only definable because each member never saw about 36.8% of the rows |

**✅  boosting**

| Example | Why it qualifies |
|---|---|
| AdaBoost: each stump trained on data **reweighted** by the previous round's errors | Sequential and dependent, combined by a weighted **sum** |
| Gradient boosting: each tree fit to the current ensemble's pseudo-residuals | Tree $m$'s target is defined by trees $1$ through $m-1$ |
| XGBoost, LightGBM, CatBoost | All gradient boosting; they differ in split-finding, regularization, and encoding |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Weighted bagging — fixed class weights on every bootstrap draw | The weights are set once, before training, and depend on no model | **Cost-sensitive** bagging |
| Growing 500 forest trees and keeping the best 50 | Selection after the fact, not sequential correction | Ensemble **pruning** |
| Stacking: several models, then a meta-model on their predictions | The base models are mutually independent; the meta-model is one extra layer, not a residual chain | **Stacking / blending** |
| "Gradient boosting does gradient descent on the trees' parameters" | The derivative is taken with respect to the **function value** $F(x_i)$ at each training point, not with respect to any split or weight | Gradient descent **in function space** |
| Boosting depth-20 trees | Each round then has the capacity to fit its own residuals outright, which voids the weak-learner premise | A model that overfits in a handful of rounds |
| A forest of trees trained on progressively larger subsets | The members still never see each other's errors | Bagging with a curriculum |

▸ **The boundary:** ask what the members do to *each other*. **Bagging members are strangers and get averaged; boosting members are a relay team and get added.** Averaging cancels scatter (variance); adding accumulates corrections (bias).

> **Common misconception.** *"Boosting is just bagging with weights on the examples."* The weights are not the distinction — the **dependency** is. In bagging, every member could be trained on a separate machine that never learns the others exist, and that independence is exactly what licenses the $1/B$ term in the variance formula. In boosting, tree $m$'s training target is *computed from* the output of trees $1$ through $m-1$; delete tree 3 from a trained model and every tree after it is now fitting the wrong thing. **That is why forests parallelize trivially across trees and boosting cannot, and why forests are immune to extra members while boosting requires early stopping.** The belief is tempting because AdaBoost's per-round example weights really do look like a weighted resample — and AdaBoost was first described in exactly those terms, years before the forward-stagewise view in this section revealed what the algorithm was actually doing.

> **Common misconception.** *"Random forests and gradient boosting are basically the same thing — both are lots of trees."* They agree on the base learner and on almost nothing else. A forest grows **deep** trees, deliberately overfit, and cancels their variance by averaging. A boosted model grows **shallow** trees, deliberately underfit, and cancels their bias by summing. Set `max_depth=2` in a random forest and you get a poor model; set `max_depth=20` in a boosted model and you get a poor model — the same knob points in opposite directions. **They are opposite answers to opposite halves of the bias–variance decomposition that happen to share a data structure.** The belief is tempting because the libraries expose nearly identical APIs, `n_estimators` sounds like it means the same thing in both, and the phrase "tree ensemble" covers both without ever warning you which failure mode is being attacked.

### AdaBoost

**The algorithm.** Initialize weights $w_i=1/n$. For $m=1..M$:
1. Fit a weak classifier $G_m$ on weighted data.
2. Compute weighted error $\mathrm{err}_m = \frac{\sum_i w_i\mathbb{1}[y_i\ne G_m(x_i)]}{\sum_i w_i}$.
3. ▸ $\alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$
4. ▸ $w_i \leftarrow w_i\exp\big(\alpha_m\mathbb{1}[y_i\ne G_m(x_i)]\big)$ — **misclassified points get up-weighted.**

Final: $G(x)=\mathrm{sign}\left(\sum_m\alpha_mG_m(x)\right)$.

#### AdaBoost, one round at a time, with real numbers

First the notation, since this algorithm is written almost entirely in it:

| Symbol | Read aloud | What it is |
|---|---|---|
| $w_i$ | "w-i" | How much attention example $i$ currently gets. Starts equal for everyone. |
| $G_m$ | "G-m" | The $m$-th weak classifier. Outputs $+1$ or $-1$. |
| $y_i$ | "y-i" | The true label of example $i$, coded as $+1$ or $-1$ |
| $\mathbb{1}[y_i\ne G_m(x_i)]$ | "indicator that y-i is not G-m of x-i" | **1 if this example was misclassified, 0 if correct** |
| $\mathrm{err}_m$ | "err-m" | The weighted fraction of examples that $G_m$ got wrong |
| $\alpha_m$ | "alpha-m" | How much of a **vote** classifier $m$ gets in the final decision |
| $\mathrm{sign}(\cdot)$ | "sign of" | Return $+1$ if the argument is positive, $-1$ if negative |

The labels being $\pm1$ rather than $0/1$ is not cosmetic — it makes $y_iG(x_i)$ equal $+1$ when correct and $-1$ when wrong, which is what lets a single expression handle both cases later.

**Step 2, the weighted error.** $\mathrm{err}_m = \frac{\sum_i w_i\mathbb{1}[y_i\ne G_m(x_i)]}{\sum_i w_i}$ reads: *"add up the weights of the examples we got wrong; divide by the total weight."* The indicator function is doing the filtering — it multiplies by 1 on mistakes and 0 on successes, so the numerator is "total weight of mistakes." It is a weighted error rate, nothing more.

**Step 3, the vote.** $\alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$. The fraction inside is the **odds of being right**. Numbers:

| $\mathrm{err}_m$ | odds correct | $\alpha_m$ | Interpretation |
|---|---|---|---|
| $0.01$ | $99$ | $+4.60$ | Nearly perfect — enormous vote |
| $0.20$ | $4$ | $+1.39$ | Solidly useful |
| $0.40$ | $1.5$ | $+0.41$ | Barely better than a coin — small vote |
| $0.50$ | $1$ | $\mathbf{0}$ | **Useless — zero vote, contributes nothing** |
| $0.70$ | $0.43$ | $-0.85$ | **Worse than chance — negative vote, so we believe the opposite** |

▸ **The zero at $\mathrm{err}=0.5$ is the elegant part.** A classifier that is right half the time carries no information, and the formula gives it exactly no influence — automatically, with no special-casing. And a classifier reliably *worse* than chance gets a negative weight, which means AdaBoost happily uses it by inverting it. Being consistently wrong is just as informative as being consistently right.

**Step 4, the reweighting.** $w_i \leftarrow w_i\exp(\alpha_m\mathbb{1}[y_i\ne G_m(x_i)])$. The indicator makes this a two-case rule:

- **Got it right:** the indicator is 0, so $w_i \leftarrow w_i e^0 = w_i$. **Unchanged.**
- **Got it wrong:** the indicator is 1, so $w_i \leftarrow w_i e^{\alpha_m}$. **Multiplied up.**

With $\mathrm{err}_m = 0.2$, $e^{\alpha_m} = e^{1.386} = 4$. Every misclassified point becomes **four times** as important for the next round.

**Walk one round on ten examples.** Start with $w_i = 0.1$ each. Round 1's stump misclassifies examples 3, 7 — so $\mathrm{err}_1 = 0.2$, $\alpha_1 = 1.386$, and those two weights become $0.4$. Total weight is now $8(0.1) + 2(0.4) = 1.6$. Renormalized, examples 3 and 7 each carry $0.4/1.6 = 25\%$ of the total attention, up from 10%.

▸ **Round 2 is now fitting a substantially different problem.** Half the training signal comes from two examples. A stump that gets those two right will look excellent even if it misses several easy ones — which is exactly the point. **The hard examples are auctioned to whichever weak learner can handle them.**

> **Analogy.** A tutor with a fixed hour and ten topics. After each practice test the tutor reallocates time toward whatever the student got wrong, and stops spending time on what they already know. Any single hour's teaching is narrow and unbalanced. The sequence of hours covers everything, with effort concentrated where it was needed.

**And the failure mode is visible in the same analogy.** If one of the practice questions has a *typo in the answer key*, the student keeps getting it "wrong," the tutor keeps escalating time on it, and eventually the entire hour is spent on a question that has no correct answer. That is AdaBoost with a mislabelled data point: its weight grows by a factor of roughly $e^{\alpha}$ every round, exponentially, forever. ▸ **AdaBoost's sensitivity to label noise is not a tuning problem — it is written into step 4**, and the fix is to replace the exponential loss, which is where §23.4 goes.

> **Where this came from.** Boosting began as a **theoretical question with a yes/no answer**, not as an algorithm. In 1988 **Michael Kearns**, then a graduate student, posed what he called the *hypothesis boosting problem*: in Valiant's PAC learning framework, is a "weak" learner — one that is only guaranteed to do slightly better than random guessing — secretly as powerful as a "strong" learner that can achieve arbitrarily small error? Most people expected the answer to be no. In 1990 **Robert Schapire** proved the answer is **yes**, and did it constructively: his proof was a recursive procedure for combining weak learners. **Yoav Freund** improved the construction shortly after. The two then collaborated on **AdaBoost** (short for *adaptive* boosting) in 1995, which replaced the awkward recursive scheme with the simple reweighting loop above and worked spectacularly in practice. Freund and Schapire received the **Gödel Prize in 2003** for it — an award for theoretical computer science, which is a fair reflection of where the idea was born. ▸ **The lesson worth keeping: AdaBoost is one of the very few widely deployed algorithms that was derived from a complexity-theoretic proof rather than discovered by experiment.**

> **The story behind the exponential loss.** Nobody designed AdaBoost to minimize $\sum_i e^{-y_if(x_i)}$. The weights, the $\log$-odds vote, the reweighting rule — all of it came out of the PAC-learning proof, and for five years the algorithm's excellent performance was explained by a *margin* theory. Then **Jerome Friedman, Trevor Hastie, and Robert Tibshirani** published *Additive Logistic Regression: A Statistical View of Boosting* in 2000, showing that this apparently ad-hoc procedure is exactly greedy coordinate descent on the exponential loss. The reaction in the boosting community was not universally warm — there was a  and long-running disagreement about whether the loss-function view or the margin view explains AdaBoost's resistance to overfitting, and the published discussion accompanying the paper is unusually pointed for a statistics journal. Both views turned out to be productive: the margin theory explains *why it generalizes*, and the loss-function theory explains *how to generalize it*, which is gradient boosting.

### AdaBoost is forward stagewise fitting of the exponential loss — derive it

**Claim:** AdaBoost greedily minimizes $\mathcal{L}=\sum_i \exp(-y_if(x_i))$ with $f=\sum_m\alpha_mG_m$.

At step $m$ with current $f_{m-1}$, we solve
$$\min_{\alpha,G}\sum_i \exp\big(-y_i(f_{m-1}(x_i)+\alpha G(x_i))\big) = \min_{\alpha,G}\sum_i w_i^{(m)}\exp(-\alpha y_iG(x_i))$$
where $w_i^{(m)}=\exp(-y_if_{m-1}(x_i))$ — **this is exactly AdaBoost's weight**, and it appears automatically rather than by design.

Split the sum by correct/incorrect (using $y_iG(x_i)=\pm1$):
$$= e^{-\alpha}\!\!\sum_{y_i=G(x_i)}\!\! w_i + e^{\alpha}\!\!\sum_{y_i\ne G(x_i)}\!\! w_i = (e^\alpha - e^{-\alpha})\sum_i w_i\mathbb{1}[y_i\ne G(x_i)] + e^{-\alpha}\sum_i w_i$$

For fixed $\alpha>0$, minimizing over $G$ means minimizing the weighted error — **step 1**. Then differentiate with respect to $\alpha$:
$$(e^\alpha+e^{-\alpha})\,\mathrm{err} - e^{-\alpha}=0 \quad\Longrightarrow\quad \alpha = \tfrac12\log\frac{1-\mathrm{err}}{\mathrm{err}}$$
— **step 3**, up to the factor of 2 absorbed into the weight update. ∎

▸ **Why this derivation mattered historically:** AdaBoost was invented as an algorithm with a margin-based analysis, and this later reinterpretation (Friedman, Hastie & Tibshirani, 2000) revealed it as coordinate descent on a specific loss. **That immediately generalized it: swap the exponential loss for any differentiable loss and you get gradient boosting.**

*(Note the exponential loss's weakness: $e^{-yf}$ grows without bound, so AdaBoost is very sensitive to label noise and outliers. Logistic loss, which grows linearly, is more robust — hence LogitBoost and gradient boosting's usual choice.)*

#### Reading the forward-stagewise derivation

This derivation is four lines of algebra carrying a large idea, so here it is slowly.

**"Forward stagewise" means: add one piece at a time and never revise.** At step $m$ you have $f_{m-1}$, a fixed sum of the pieces you already added. You choose one new piece $\alpha G$ to append, choosing it to make the total loss as small as possible *given that everything before it is frozen*. You never go back and re-tune $\alpha_3$. This is why it is "greedy": it is coordinate descent where each coordinate is an entire weak learner, and each coordinate is visited once.

**The quantity $y_if(x_i)$ is called the margin, and it is the object the whole derivation is about.** Since $y_i \in \{-1,+1\}$ and $f$ is the ensemble's raw score:

- $y_if(x_i) > 0$ — **correct**, and the magnitude says how confidently.
- $y_if(x_i) < 0$ — **wrong**, and the magnitude says how confidently wrong.

So $\exp(-y_if(x_i))$ is small when you are confidently right and large when you are confidently wrong. Numbers: margin $+3$ costs $e^{-3} = 0.050$; margin $0$ costs $1$; margin $-3$ costs $e^{3} = 20.1$; margin $-10$ costs $22026$.

**Line 1 — where the weights come from.** Split the exponential using $e^{a+b} = e^ae^b$:

$$\exp\big(-y_i(f_{m-1}(x_i)+\alpha G(x_i))\big) = \underbrace{\exp(-y_if_{m-1}(x_i))}_{\text{call this }w_i^{(m)}}\cdot \exp(-\alpha y_iG(x_i))$$

▸ **This is the derivation's punchline and it arrives in the first line.** The factor $w_i^{(m)}$ depends only on the *old* ensemble, so as far as the current step is concerned it is a constant — a per-example weight. And it is literally AdaBoost's weight: an example the current ensemble already handles confidently ($y_if_{m-1}$ large and positive) gets a tiny $w_i$; one the ensemble is getting badly wrong gets a huge one. **Nobody put the reweighting rule into AdaBoost by design. It falls out of the algebra of the exponential function.** Freund and Schapire wrote it down for entirely different reasons and it turned out to be this.

**Line 2 — the two-case split.** Because $y_iG(x_i)$ is exactly $+1$ (correct) or $-1$ (wrong), $\exp(-\alpha y_iG(x_i))$ is exactly $e^{-\alpha}$ or $e^{+\alpha}$. So the sum breaks into "correct examples, each paying $e^{-\alpha}$" plus "wrong examples, each paying $e^{+\alpha}$." Rearranged, the whole thing is

$$(e^\alpha - e^{-\alpha})\cdot(\text{weighted error}) + e^{-\alpha}\cdot(\text{total weight}).$$

Since $\alpha>0$ makes $e^\alpha - e^{-\alpha}$ positive, the only way to lower this is to **lower the weighted error** — which is step 1 of the algorithm, recovered.

**Line 3 — solving for $\alpha$.** Differentiate the expression above with respect to $\alpha$, set to zero, and out drops $\alpha = \frac12\log\frac{1-\mathrm{err}}{\mathrm{err}}$ — step 3, recovered, with the factor of $\frac12$ absorbed into how the weights are normalized. **Every line of the original algorithm has now been re-derived from a single choice of loss function.**

#### Why the exponential loss is the weak point

Compare how the two losses punish a badly wrong prediction:

| Margin $yf(x)$ | Exponential $e^{-yf}$ | Logistic $\log(1+e^{-yf})$ |
|---|---|---|
| $+2$ | $0.135$ | $0.127$ |
| $0$ | $1.00$ | $0.693$ |
| $-2$ | $7.39$ | $2.13$ |
| $-5$ | $148$ | $5.01$ |
| $-10$ | $\mathbf{22026}$ | $\mathbf{10.0}$ |

Read the bottom row. At margin $-10$ the exponential loss charges **twenty-two thousand**; the logistic loss charges **ten**. And since AdaBoost's weights *are* the loss values, that single example now carries more weight than several thousand ordinary ones.

▸ **A mislabelled row does not merely add noise to AdaBoost — it eventually becomes the entire objective.** The model spends its remaining capacity trying to fit a point whose label is wrong, and drags the decision boundary across the correct data to reach it. The logistic loss grows only *linearly* in the margin for large negative margins, so a hopeless example asymptotically caps out at a bounded influence: the model tries, fails, and moves on. **This is the same "bounded gradients are a safety property" argument that makes cross-entropy the standard classification loss (Ch. 1 §1.5)** — and it is why essentially every gradient-boosting library defaults to logistic rather than exponential loss today.

---

## 23.4 Gradient boosting

### The one-line idea

Do gradient descent in **function space**: at each step, fit a tree to the negative gradient of the loss with respect to the current predictions.

### The derivation

We want to minimize $\mathcal{L}=\sum_i L(y_i, F(x_i))$ over functions $F$. Treat the vector of predictions $(F(x_1),\dots,F(x_n))$ as the parameter. The negative gradient is

▸ $$r_i^{(m)} = -\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

These are **pseudo-residuals**. Gradient descent would set $F_m = F_{m-1}+\eta r$ — but that only defines an update at the training points. To generalize, **fit a tree $h_m$ to the pseudo-residuals** and step in that direction:

▸ $$F_m(x) = F_{m-1}(x) + \eta\,h_m(x)$$

**For squared loss**, $L=\frac12(y-F)^2$ gives $r_i = y_i-F(x_i)$ — the ordinary residual. That is the special case people remember, and the general case is what makes the method powerful.

| Loss | Pseudo-residual |
|---|---|
| Squared error | $y_i - F_i$ |
| Absolute error | $\mathrm{sign}(y_i-F_i)$ |
| Huber | residual, clipped |
| Logistic (binary) | $y_i - \sigma(F_i)$ |
| Poisson | $y_i - e^{F_i}$ |

▸ **Any differentiable loss works.** This is why gradient boosting handles ranking (LambdaRank), survival analysis, quantile regression, and custom business objectives — you supply the gradient.

#### What "gradient descent in function space" actually means

This phrase is the conceptual heart of the chapter, and it sounds far more exotic than it is.

**Ordinary gradient descent**, the kind in Ch. 4, adjusts *numbers*. You have parameters $\theta \in \mathbb{R}^p$, you compute $\nabla_\theta\mathcal{L}$ — a list of $p$ numbers saying which way to nudge each parameter — and you step: $\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}$.

**Gradient boosting does the same thing, but the "parameter" is the model's list of predictions.** Here is the trick, stated flatly: forget that $F$ is a function. On your $n$ training points, $F$ is fully described by the $n$ numbers $\big(F(x_1), F(x_2), \dots, F(x_n)\big)$. Treat *that vector* as the parameter. It lives in $\mathbb{R}^n$, and you can take a gradient with respect to it exactly as usual.

$$r_i^{(m)} = -\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

Read aloud: *"r-i at round m is minus the derivative of the loss for example i, with respect to that example's own prediction, evaluated at the current ensemble."*

- The square brackets with the subscript $F = F_{m-1}$ mean **"evaluate this derivative at the current model"** — the derivative is a formula; the subscript says where to plug in.
- The minus sign is the usual one: the gradient points uphill on the loss, and we want to go downhill.
- $r_i$ answers a very concrete question: ***"if I could change this one prediction by hand, which way should I move it, and how urgently?"***

**Now the problem, and the fix.** Gradient descent would say $F_m(x_i) = F_{m-1}(x_i) + \eta r_i$ for each $i$. That is a perfectly good instruction — for the $n$ points you trained on. But it says nothing whatsoever about a new customer who walks in tomorrow. You have improved a lookup table, not a model.

▸ **The fix is the entire idea of gradient boosting: fit a tree to the vector $r$, and use the tree's output as the step direction.** The tree is a *smoother* — it looks at the $n$ desired nudges, notices that "everyone with income under 40k and age over 60 wants to move down by about 0.3," and turns a table of $n$ instructions into a rule that applies to anyone. **The gradient tells you where to go; the tree tells you how to get there for points you have never seen.**

> **Analogy.** You are the manager of a delivery fleet and every driver phones in a complaint: "my route is 4 minutes too long," "mine is 2 minutes too short." That list of complaints is the gradient — precise, and completely specific to today's drivers. You cannot hand it to tomorrow's drivers. So instead you look for **patterns** in the complaints — "all the northside afternoon routes are too long" — and issue a *rule*. The rule is the tree. It fits the complaints imperfectly, which is fine, because you will collect complaints again tomorrow and issue another rule.

#### Reading the pseudo-residual table

Each row is one loss function and the "nudge direction" it produces. Work through them:

- **Squared error, $L=\frac12(y-F)^2$.** Differentiate with respect to $F$: $\partial L/\partial F = -(y-F)$, so $r_i = y_i - F_i$. **The ordinary residual** — how far off you are, signed. If you predicted 7 and the truth is 10, the tree is asked to predict $+3$ there. This is why the naive description "boosting fits the errors" is *almost* right: it is right for exactly one loss function.
- **Absolute error, $L = \lvert y-F\rvert$.** The derivative of $\lvert u\rvert$ is $\mathrm{sign}(u)$, so $r_i = \mathrm{sign}(y_i - F_i)$ — **just $+1$ or $-1$.** Being off by 3 and being off by 300 produce *the same* pseudo-residual. ▸ **That is exactly what makes absolute error robust:** an outlier can shout no louder than anyone else, because the loss only ever asks for direction, never magnitude.
- **Huber.** The residual, but clipped at some threshold $\delta$. Squared-error behaviour for small errors (efficient, uses magnitude), absolute-error behaviour for large ones (robust, ignores magnitude). The deliberate compromise.
- **Logistic (binary), $r_i = y_i - \sigma(F_i)$.** Here $\sigma$ is the sigmoid (Ch. 1 §1.3.4), $F_i$ is a **log-odds score**, and $\sigma(F_i)$ converts it to a probability. So the pseudo-residual is **"true label minus predicted probability."** If $y_i=1$ and your model says $0.7$, the pseudo-residual is $+0.3$. Note this is bounded in $[-1,1]$ no matter how wrong you are — the safety property from §23.3 again.
- **Poisson, $r_i = y_i - e^{F_i}$.** For count data. $F_i$ is a log-rate, so $e^{F_i}$ is the predicted count, and the residual is observed-minus-predicted counts. The exponential link is what guarantees the prediction is never negative — you cannot predict $-3$ customer visits.

▸ **The pattern across the whole table: the pseudo-residual is always "what we wanted minus what we got," measured in whatever currency the loss uses.** Change the currency, and the identical algorithm now does quantile regression, ranking, or survival analysis. **You never modify the boosting loop; you modify one line that computes a derivative.** That modularity is why gradient boosting outlived AdaBoost.

> **Where this came from.** The generalization was found twice, in the same period, from opposite directions. **Jerome Friedman** — a co-author of CART and, before that, a physicist at the Stanford Linear Accelerator Center — published *Greedy Function Approximation: A Gradient Boosting Machine* in 2001, having derived it directly from the 2000 statistical reinterpretation of AdaBoost he had co-authored: once you know AdaBoost is coordinate descent on a loss, swapping the loss is the obvious next move. Independently, **Llew Mason, Jonathan Baxter, Peter Bartlett, and Marcus Frean** presented boosting as gradient descent in function space in 1999, from a machine-learning-theory rather than statistical starting point, in a framework they called **AnyBoost**. Friedman's name for the tree version was **MART** — Multiple Additive Regression Trees — which is why some older codebases and the commercial TreeNet software use that acronym for what everyone now calls GBDT.

### The three regularizers that make it work

1. **Learning rate (shrinkage)** $\eta\approx0.01$–$0.1$. ▸ **Lower $\eta$ with more trees is nearly always better** (the classic $\eta$–$M$ trade-off). This is the single most important hyperparameter.
2. **Subsampling** (stochastic gradient boosting): use a random 50–80% of rows per tree. Adds variance reduction on top of bias reduction, and speeds training.
3. **Tree constraints:** depth 3–8, min samples per leaf, and the $\ell_2$ penalty on leaf weights (below).

#### Why shrinkage works, with numbers

$$F_m(x) = F_{m-1}(x) + \eta\,h_m(x)$$

Read aloud: *"the new ensemble equals the old ensemble plus eta times the new tree."* The tree $h_m$ was fitted to close the whole gap — and then you deliberately apply only a small fraction $\eta$ of it.

**This looks wasteful and is not.** Suppose the tree says "increase this prediction by 3.0" and $\eta = 0.05$. You move it by $0.15$. Next round you recompute the residuals — the gap is now $2.85$ — and another tree, fitted fresh on the *new* residual pattern, moves it a bit further. You needed about 20 trees to travel a distance one tree offered to cover in a single step.

▸ **What you bought is that 20 different trees voted on the direction rather than one.** Any single tree's structure is high-variance (§23.1) — it may have split on a spurious threshold. Taking 5% of twenty different opinions is far more stable than taking 100% of one. **Shrinkage converts a sequence of confident wrong turns into a consensus.**

> **Analogy.** Steering a ship with a rudder. You could yank it hard over the instant you see the buoy is off to port — and overshoot, then over-correct, and zigzag your way in. Or make many small corrections, re-checking the buoy between each. The path is smoother and you arrive closer. $\eta$ is how hard you are willing to pull the rudder in one go.

**The $\eta$–$M$ trade-off, quantified.** The rough empirical rule is that halving the learning rate requires roughly doubling the number of trees to reach the same training loss — and usually yields slightly better test loss:

| $\eta$ | Trees needed (typical) | Relative training time | Test performance |
|---|---|---|---|
| $0.3$ | $\sim 100$ | $1\times$ | baseline |
| $0.1$ | $\sim 300$ | $3\times$ | better |
| $0.05$ | $\sim 600$ | $6\times$ | slightly better still |
| $0.01$ | $\sim 3000$ | $30\times$ | marginal further gain |

▸ **The returns diminish while the cost does not, which is why $0.05$ and $0.1$ are the practical sweet spot** — and why "lower $\eta$ is nearly always better" should be read as "better per tree, not better per hour." If you have a fixed compute budget the optimum is interior, and the way to find it is to fix $\eta$ and let early stopping choose $M$.

**Subsampling** (row sampling, typically 50–80% of rows per tree) does two things at once: it lowers $\rho$ between successive trees exactly as bagging does, and it makes each tree cheaper. Friedman found in 2002 that adding this to gradient boosting improved accuracy as well as speed — a  free lunch, and one of the reasons it became a default. **Column sampling** (`colsample_bytree`, 60–90% of features) is the random-forest idea imported wholesale into boosting, and it earns its keep for the same reason: decorrelation.

#### Examples and non-examples: regularizing a boosted model

**✅  boosting regularizers**

| Example | Why it qualifies |
|---|---|
| $\eta = 0.05$ with $M \approx 600$ chosen by early stopping | Shrinkage: each tree delivers 5% of what it wanted, so ~20 trees vote per unit of progress |
| `subsample = 0.7` | Row sampling decorrelates successive trees, importing bagging's benefit into a boosted model |
| `max_depth = 4` | Caps the **interaction order** at four features per root-to-leaf path |
| The $\lambda$ penalty on leaf weights (§23.5) | Shrinks each leaf's output toward zero inside the closed-form optimum |
| `min_child_weight` | Refuses leaves supported by too little second-order mass — a data-quantity floor per leaf |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Lowering $\eta$ without raising $M$ | You shortened the journey rather than smoothing it | Undertraining, not regularization |
| Setting `n_estimators = 5000` with no validation set | Every extra tree here is  added capacity | The classic boosting overfit |
| Treating `max_depth` as a plain capacity dial | Its first-order effect is *which interactions are expressible* | An interaction-order dial |
| Early stopping on the **training** loss | Training loss falls monotonically by construction and never signals the turn | Not a stopping rule at all |
| Adding more stumps to fix underfitting from `max_depth=1` | Stumps can only ever sum to a purely **additive** model, however many you stack | A structural limit; you need depth $\ge 2$ for any interaction |
| Bagging the boosted model | Legitimate, but it addresses a different term | Variance reduction on top of a bias-reduction method |

▸ **The boundary:** in boosting, every regularizer works by limiting **how much a single round is allowed to commit** — a fraction of the step, a subset of the rows, a bound on the interactions. None of it resembles the "just add more members" safety of a forest.

> **Common misconception.** *"More trees can't hurt — that's true for forests, and boosting is an ensemble too."* In a forest, each new tree is one more sample in an average. In a boosted model, each new tree is a fresh block of capacity **added onto the running total**. Train long enough and the ensemble starts fitting the noise in its own residuals: test loss traces a U, falling to a minimum at some $M^*$ and then climbing, while training loss keeps dropping the whole time. **`n_estimators` in a boosted model is not a resource setting, it is the primary capacity hyperparameter** — which is why early stopping on a validation set sits at the top of §23.7's tuning order, and why the honest way to report a boosted model is "$M$ selected by early stopping" rather than a round number. The belief is tempting because both objects are called ensembles, both libraries name the parameter `n_estimators`, and the forest advice — "more is always better, just slower" — gets repeated so often that it is transplanted without anyone checking whether the members are averaged or summed.

---

## 23.5 XGBoost — derive the objective

XGBoost's contribution was to write down the *exact* objective a tree should optimize, using a second-order expansion, rather than fitting to gradients heuristically.

### The regularized objective

▸ $$\mathcal{L}^{(m)} = \sum_{i=1}^n L\big(y_i,\ F_{m-1}(x_i)+f_m(x_i)\big) + \Omega(f_m),\qquad \Omega(f)=\gamma T + \tfrac12\lambda\sum_{j=1}^{T}w_j^2$$

where $T$ = number of leaves and $w_j$ = the value at leaf $j$.

#### Unpacking the regularized objective

$$\mathcal{L}^{(m)} = \sum_{i=1}^n L\big(y_i,\ F_{m-1}(x_i)+f_m(x_i)\big) + \Omega(f_m),\qquad \Omega(f)=\gamma T + \tfrac12\lambda\sum_{j=1}^{T}w_j^2$$

Read aloud: *"the objective at round m is the sum over all n examples of the loss — evaluated at the old prediction plus the new tree's contribution — plus a complexity penalty on the new tree."*

| Symbol | Read aloud | What it is |
|---|---|---|
| $\mathcal{L}^{(m)}$ | "script-L, superscript m" | The thing round $m$ is trying to minimize |
| $F_{m-1}(x_i)$ | "F-m-minus-1 of x-i" | What the ensemble already predicts for example $i$. **Fixed — a known number.** |
| $f_m(x_i)$ | "f-m of x-i" | What the *new* tree will add for example $i$. **The unknown we are solving for.** |
| $\Omega(f)$ | "Omega of f" | Greek capital O — the **complexity penalty** on the new tree |
| $T$ | "T" | Number of leaves in the new tree |
| $w_j$ | "w-j" | The value the new tree outputs at leaf $j$ |
| $\gamma$ | "gamma" | Price charged **per leaf** — the same idea as $\alpha$ in cost-complexity pruning |
| $\lambda$ | "lambda" | Ridge penalty pushing every leaf value toward **zero** |

▸ **The one-sentence summary of XGBoost's contribution: it wrote the regularization *into the objective the tree is optimizing*, instead of applying it afterwards as a pruning heuristic.** Friedman's gradient boosting fits a tree to the pseudo-residuals using ordinary CART machinery — impurity, `min_samples_leaf`, and so forth — which means the tree is optimizing a *different* criterion from the one you actually care about. XGBoost closes that gap: the split-finding criterion is derived from the loss you specified, complete with its penalties.

**The two penalty terms do different jobs:**

- $\gamma T$ — a **flat toll per leaf.** Adding a leaf must buy at least $\gamma$ worth of loss reduction or it is not worth it. This is a *structural* penalty: it controls how many splits exist.
- $\tfrac12\lambda\sum_j w_j^2$ — a **quadratic penalty on leaf values.** This is a *magnitude* penalty: the tree may have many leaves, but each is discouraged from making a bold prediction. This is exactly ridge regression (Ch. 22 §22.2) with the leaf values as coefficients.

> **Analogy.** You are approving a company's proposed org chart. $\gamma T$ is a fixed cost for every headcount you approve — it limits how many boxes appear on the chart. $\lambda\sum w_j^2$ limits how much authority any one box may wield. You can have a large flat organization of cautious people, or a small organization of decisive ones, and the two knobs price those choices separately.

### Second-order expansion

Let $g_i = \partial_F L(y_i,F_{m-1})$ and $h_i = \partial_F^2 L(y_i,F_{m-1})$. Taylor-expand:

$$\mathcal{L}^{(m)}\approx\sum_i\left[L(y_i,F_{m-1}) + g_if_m(x_i)+\tfrac12h_if_m(x_i)^2\right]+\Omega(f_m)$$

Drop the constant. A tree assigns every $x$ in leaf $j$ the same value $w_j$, so group the sum by leaves. Let $I_j$ be the set of instances in leaf $j$, $G_j=\sum_{i\in I_j}g_i$, $H_j=\sum_{i\in I_j}h_i$:

▸ $$\tilde{\mathcal{L}} = \sum_{j=1}^{T}\left[G_jw_j + \tfrac12(H_j+\lambda)w_j^2\right] + \gamma T$$

### The optimal leaf weight

Each leaf is now an independent one-dimensional quadratic. Differentiate and set to zero:

▸ $$\boxed{\ w_j^* = -\frac{G_j}{H_j+\lambda}\ }$$

Substituting back gives the **structure score** — the quality of a given tree shape:

▸ $$\boxed{\ \tilde{\mathcal{L}}^* = -\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda} + \gamma T\ }$$

### The split gain

Splitting a leaf into $L$ and $R$ changes the score by:

▸ $$\boxed{\ \mathrm{Gain} = \frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma\ }$$

▸ **This is the formula to know.** Read it: the first two terms are the children's scores, the third is the parent's, and $\gamma$ is a fixed cost per additional leaf. **If the gain is negative, don't split** — $\gamma$ acts as automatic pre-pruning with a principled threshold, rather than a heuristic like `min_impurity_decrease`.

Note also that $H_j+\lambda$ in the denominator means **leaves with little curvature (few or low-confidence samples) get shrunk toward zero.** That is ridge regularization appearing again (Ch. 22 §22.2), now on leaf values.

#### The second-order expansion, decoded

**What a Taylor expansion is doing here.** You have a complicated loss $L$ and you are about to change its input by a small amount $f_m(x_i)$. Rather than working with $L$ exactly — which could be logistic, Poisson, or something you wrote yourself — you replace it locally by a **parabola** that matches its value, slope, and curvature at the current point:

$$L(y, F + f) \approx \underbrace{L(y,F)}_{\text{value}} + \underbrace{g\,f}_{\text{slope}} + \underbrace{\tfrac12 h\,f^2}_{\text{curvature}}$$

- $g_i = \partial_F L(y_i, F_{m-1})$ — read "**the first derivative of the loss at example $i$**." Which way and how urgently this example wants its prediction moved. (Note: $g_i = -r_i$, the negative of §23.4's pseudo-residual. Same information, opposite sign convention.)
- $h_i = \partial_F^2 L(y_i, F_{m-1})$ — read "**the second derivative**," the curvature. How quickly the urgency changes as you move. High $h$ means "this example is confident about what it wants"; low $h$ means "this example is nearly indifferent."

▸ **This is the same move Newton's method makes (Ch. 4), and it buys the same thing: a step size that is derived rather than guessed.** Plain gradient boosting knows only the slope, so it must be told how far to step. Second-order boosting knows the curvature too, so it can compute the exact bottom of the local parabola.

**Concrete $g$ and $h$ for logistic loss.** With $p = \sigma(F)$ the predicted probability:

$$g = p - y, \qquad h = p(1-p)$$

Now read what $h$ is telling you:

| Prediction $p$ | $h = p(1-p)$ | Meaning |
|---|---|---|
| $0.5$ | $0.25$ | Maximum uncertainty — **maximum curvature, maximum influence** |
| $0.9$ | $0.09$ | Fairly confident |
| $0.99$ | $0.0099$ | Very confident — **barely moves the fit** |
| $0.999$ | $0.000999$ | Effectively settled |

▸ **$h_i$ is a natural, automatic per-example weight, and it weights by uncertainty.** Examples the model is already confident about contribute almost nothing to the next tree — not because anyone wrote a rule saying so, but because the loss surface is flat there. AdaBoost achieved something similar by explicitly rewriting weights each round; XGBoost gets it from the second derivative for free.

**Why grouping by leaves works.** A tree assigns *the same number* $w_j$ to every example landing in leaf $j$. So in the sum $\sum_i [g_if_m(x_i) + \frac12 h_if_m(x_i)^2]$, every $i$ in leaf $j$ contributes $g_iw_j + \frac12h_iw_j^2$. Factor out $w_j$ and $w_j^2$:

$$\sum_{i \in I_j}\Big(g_iw_j + \tfrac12h_iw_j^2\Big) = \Big(\underbrace{\textstyle\sum_{i\in I_j}g_i}_{G_j}\Big)w_j + \tfrac12\Big(\underbrace{\textstyle\sum_{i\in I_j}h_i}_{H_j}\Big)w_j^2$$

- $I_j$ — read "**the index set of leaf $j$**": the list of which training rows land there.
- $G_j$ — the **total** first-derivative pressure in that leaf.
- $H_j$ — the **total** curvature in that leaf.

▸ **A sum over $n$ examples became a sum over $T$ leaves, and the $T$ terms don't interact.** Every leaf is now an independent one-dimensional parabola in its own $w_j$, and one-dimensional parabolas are the easiest optimization problem that exists. That collapse — from a coupled $n$-dimensional problem to $T$ separate scalar ones — is what makes the whole derivation go through in closed form.

#### The optimal leaf weight, with numbers

$$w_j^* = -\frac{G_j}{H_j+\lambda}$$

This is just "vertex of a parabola." For $ax + \frac12 bx^2$, the minimum sits at $x = -a/b$. Here $a = G_j$ and $b = H_j + \lambda$, and the $\lambda$ arrives from the ridge term $\frac12\lambda w_j^2$ folding into the quadratic coefficient.

**Work an example.** A leaf holds 20 examples, all currently predicted at $p = 0.5$, of which 15 are truly class 1.

- $g_i = p - y_i$, so the 15 positives each give $0.5 - 1 = -0.5$, and the 5 negatives each give $0.5 - 0 = +0.5$. Total: $G_j = 15(-0.5) + 5(0.5) = -5.0$.
- $h_i = p(1-p) = 0.25$ for all twenty. Total: $H_j = 5.0$.
- With $\lambda = 1$: $w_j^* = -(-5.0)/(5.0 + 1) = \mathbf{+0.833}$.
- With $\lambda = 0$: $w_j^* = 5.0/5.0 = \mathbf{+1.0}$.

The leaf pushes the log-odds up, which is right — this leaf is 75% positive and the current prediction is 50%. And **$\lambda = 1$ shrank the step by 17%**.

Now shrink the leaf to **2 examples** (both positive, both at $p=0.5$): $G_j = -1.0$, $H_j = 0.5$.

- With $\lambda = 1$: $w_j^* = 1.0/(0.5+1) = \mathbf{0.667}$ — shrunk by **33%**.
- With $\lambda = 0$: $w_j^* = 1.0/0.5 = \mathbf{2.0}$.

▸ **Same $\lambda$, and the tiny leaf was shrunk twice as hard as the large one.** That is the whole point of putting $\lambda$ in the denominator next to $H_j$: $H_j$ grows with the number of examples in the leaf, so $\lambda$ is large *relative to* $H_j$ exactly when the leaf has little evidence. **A fixed penalty automatically becomes a strong prior on small leaves and a weak one on large leaves.** You get sample-size-aware shrinkage from one constant.

> **Analogy.** $\lambda$ is a sceptical editor. A claim backed by twenty sources gets published nearly as written; the same claim backed by two sources gets toned down substantially. The editor applies one policy; the effect scales with the evidence.

#### Reading the split gain

$$\mathrm{Gain} = \frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$$

Substituting $w^*$ back into the leaf objective gives that leaf's contribution as $-\frac12\frac{G_j^2}{H_j+\lambda}$ — so the quantity $\frac{G_j^2}{H_j+\lambda}$ is exactly **"how much loss this leaf removes,"** and the whole formula is bookkeeping:

$$\mathrm{Gain} = \tfrac12\big[\underbrace{\text{score}_L + \text{score}_R}_{\text{after the split}} - \underbrace{\text{score}_{\text{parent}}}_{\text{before}}\big] - \underbrace{\gamma}_{\text{cost of the extra leaf}}$$

Note $G_L + G_R$ and $H_L + H_R$ are the parent's totals — because splitting a leaf does not create or destroy examples, it just partitions them. That is why the parent term is written in terms of the children's sums.

**Real numbers.** A node with $G = -5.0$, $H = 5.0$, $\lambda = 1$, $\gamma = 0.5$. Two candidate splits:

**Split A — a  informative split.** Left gets the 15 positives ($G_L = -7.5$, $H_L = 3.75$); right gets the 5 negatives ($G_R = +2.5$, $H_R = 1.25$).
- $\text{score}_L = 7.5^2/(3.75+1) = 56.25/4.75 = 11.84$
- $\text{score}_R = 2.5^2/(1.25+1) = 6.25/2.25 = 2.78$
- $\text{score}_{\text{parent}} = 5.0^2/(5.0+1) = 25/6 = 4.17$
- $\mathrm{Gain} = \frac12[11.84 + 2.78 - 4.17] - 0.5 = \frac12(10.45) - 0.5 = \mathbf{+4.72}$. **Split it.**

**Split B — a useless split** that divides the node into two halves with the same class balance. Then $G_L = G_R = -2.5$, $H_L = H_R = 2.5$.
- $\text{score}_L = \text{score}_R = 6.25/3.5 = 1.79$
- $\mathrm{Gain} = \frac12[1.79 + 1.79 - 4.17] - 0.5 = \frac12(-0.59) - 0.5 = \mathbf{-0.79}$. **Do not split.**

▸ **Note that split B produced a *negative* gain even before $\gamma$ was subtracted.** That is the $\lambda$ working: two small leaves each pay the $\lambda$ tax separately, whereas one large leaf pays it once. **The regularizer makes uninformative splits actively costly rather than merely neutral**, which is a much stronger form of pruning than a heuristic threshold on impurity decrease.

**What $\gamma$ gives you that `min_impurity_decrease` does not:** it is measured in the units of the loss you actually chose. A gain of $+4.72$ means "this split reduces my logistic loss by 4.72 nats-worth of objective." $\gamma = 0.5$ means "I refuse any split worth less than half that unit." The threshold has a meaning; a threshold on Gini impurity does not translate into anything you care about.

> **Where this came from.** **XGBoost** began around 2014 as a research project by **Tianqi Chen**, then a PhD student at the University of Washington working with Carlos Guestrin, and was open-sourced before the paper describing it appeared at KDD in 2016. Its reputation was made on Kaggle rather than in the literature: the paper's own accounting notes that **among the 29 winning solutions published on Kaggle's blog during 2015, 17 used XGBoost** — a rate of adoption that essentially ended the debate about tabular defaults. What is interesting is that a large share of the contribution is *systems engineering* rather than statistics: a cache-aware block structure for out-of-core data, a sparsity-aware split finder that learns a default direction for missing values instead of imputing them, parallelized split search, and an approximate quantile sketch weighted by $h_i$ so that split candidates could be proposed without sorting the entire dataset. ▸ **The second-order objective above is the part everyone quotes, but the reason XGBoost won was that it was roughly an order of magnitude faster than the alternatives on the hardware people actually had.**

### XGBoost's other contributions

- **Sparsity-aware split finding:** learn a default direction for missing values at each node, rather than imputing.
- **Weighted quantile sketch:** approximate split candidates using $h_i$ as weights, so it scales to data that doesn't fit in memory.
- **Column and row subsampling**, cache-aware access patterns, out-of-core computation, and parallel split finding.

---

## 23.6 LightGBM and CatBoost

### LightGBM

**Histogram-based splitting.** Bin continuous features into 255 buckets once; split-finding then costs $O(\#\text{bins})$ instead of $O(n\log n)$ sorting. ▸ **This is a 10–20× speedup and is now standard in XGBoost too** (`tree_method='hist'`).

**Leaf-wise (best-first) growth** instead of level-wise: always split the leaf with the highest gain, anywhere in the tree. Lower loss for the same number of leaves; **but** it produces deep, unbalanced trees that overfit small data — control with `num_leaves` (the real capacity knob) and `min_data_in_leaf`.

**GOSS (Gradient-based One-Side Sampling):** keep all instances with large gradients (the under-fitted ones) and randomly sample from the small-gradient ones, upweighting the sample by $\frac{1-a}{b}$ to keep the gain estimate unbiased. Large-gradient instances carry most of the information about where to split.

**EFB (Exclusive Feature Bundling):** in sparse data, features that are rarely nonzero simultaneously (e.g. one-hot columns) can be bundled into one feature without loss. Reduces effective dimensionality substantially.

#### LightGBM's four ideas, decoded

**1. Histogram binning — where the 10–20× comes from.** The naive way to find the best threshold on a continuous column is to sort its $n$ values and sweep, which costs $O(n\log n)$ *per feature per node*. With $n = 10^6$ rows and $p = 100$ features and a few thousand nodes, that is a catastrophic number of comparisons.

Histogram binning does the sorting **once, at the start**. Each continuous column is replaced by a bin index from 0 to 254 — so a value of $37{,}412.55$ becomes bin `143`. Now finding a split means sweeping 255 bin boundaries and accumulating $(G, H)$ sums, which costs $O(255)$ regardless of $n$.

**Real numbers.** With $n = 10^6$: sorting costs about $10^6\times 20 = 2\times10^7$ operations per feature per node; binning costs $255$. The build of the histogram itself costs $O(n)$, but there is a further trick — **the histogram subtraction identity**: a node's two children partition its examples, so $\text{hist}_{\text{right}} = \text{hist}_{\text{parent}} - \text{hist}_{\text{left}}$. Build the histogram for the *smaller* child only and subtract to get the other, halving the work at every level.

▸ **The cost of binning is a loss of threshold precision, and it turns out not to matter.** With 255 bins you can only split at 255 places instead of $n$; but the extra precision was mostly fitting noise anyway, so binning often acts as a mild regularizer. This is why it stopped being LightGBM's differentiator — XGBoost adopted it as `tree_method='hist'` and it is now everyone's default.

**2. Leaf-wise versus level-wise growth.** Level-wise (XGBoost's original) grows the tree in complete layers: split every node at depth 1, then every node at depth 2. Leaf-wise (LightGBM) keeps a priority queue of all current leaves and always splits **whichever leaf has the highest gain anywhere in the tree.**

> **Analogy.** You have a budget to fix potholes. Level-wise is repaving every street in the city block by block, in order. Leaf-wise is going straight to the worst pothole in the city, wherever it is, then the next worst. The second gets more repair per pound — and produces a very uneven-looking city.

Given a budget of 31 leaves, leaf-wise reaches a lower training loss than level-wise, because it spent all 31 where they helped most. But the resulting tree can be depth 15 down one branch and depth 2 down another, and a depth-15 path is asking 15 questions to isolate perhaps four training points. ▸ **This is why `num_leaves` — not `max_depth` — is LightGBM's real capacity knob, and why the standard first mistake is porting an XGBoost `max_depth=6` into LightGBM and setting `num_leaves=64` ($2^6$).** A level-wise tree of depth 6 rarely fills all 64 slots; a leaf-wise tree always will, and it will put them somewhere deep and specific. Start well below $2^{\text{depth}}$.

**3. GOSS, decoded.** The name expands to **Gradient-based One-Side Sampling**. The insight: after a few rounds, most examples are already well-fit, and a well-fit example has $g_i \approx 0$ — it contributes nothing to any $G_j$ and therefore nothing to any split decision. So why pay to scan it?

GOSS keeps the top $a$ fraction by $\lvert g_i\rvert$ (say 20%), randomly samples a fraction $b$ (say 10%) from the remaining 80%, and **multiplies the sampled small-gradient examples' contributions by $\frac{1-a}{b}$** — here $\frac{0.8}{0.1} = 8$. That factor is an importance-sampling correction: you kept one in eight of them, so each stands in for eight, and the estimated $G_j$ stays unbiased. **The result is a scan over 30% of the data that estimates the same split gains.**

▸ **This is the same "one-sided" logic as hard-negative mining in contrastive learning (Ch. 25) and as focal loss: the examples you already handle carry no information about how to improve.** The subtlety GOSS gets right is the reweighting — sampling without the $\frac{1-a}{b}$ correction would systematically bias every split toward the hard examples' preferences.

**4. EFB, decoded.** One-hot encoding a categorical column with 1000 levels creates 1000 columns, of which **exactly one is nonzero in any row**. Such columns are *mutually exclusive*: they never fire together. So you can pack them into a single column by offsetting their value ranges — column A uses codes 1–10, column B uses 11–25, column C uses 26–40 — and a split on the packed column at, say, code $12.5$ is a perfectly valid split. Split-finding now scans one histogram instead of a thousand.

Finding the best bundling is a graph-colouring problem (build a graph where features conflict if they are ever nonzero together; colours are bundles), which is NP-hard, so LightGBM uses a greedy approximation and tolerates a small conflict rate. ▸ **The gain is proportional to how sparse your data is**, which is why EFB matters enormously for one-hot text or click-log features and not at all for a dense table of sensor readings.

### CatBoost

**Ordered target statistics.** Encoding a categorical feature by its target mean leaks the label. CatBoost fixes this by using, for each example, only the examples *before it* in a random permutation:
▸ $$\hat x_i = \frac{\sum_{j<i}\mathbb{1}[x_j=x_i]y_j + a\cdot p}{\sum_{j<i}\mathbb{1}[x_j=x_i] + a}$$
▸ **This is the same idea as out-of-fold target encoding**, applied online. Target leakage in categorical encoding is one of the most common silent bugs in applied ML, and knowing this fix is valuable well beyond CatBoost.

#### Reading the ordered target statistic — and the bug it fixes

$$\hat x_i = \frac{\sum_{j<i}\mathbb{1}[x_j=x_i]y_j + a\cdot p}{\sum_{j<i}\mathbb{1}[x_j=x_i] + a}$$

Read aloud: *"the encoded value for row $i$ is the sum of the labels of all earlier rows that share row $i$'s category, plus a prior, divided by the count of those earlier rows, plus the prior's weight."*

| Symbol | Read aloud | What it is |
|---|---|---|
| $x_i$ | "x-i" | Row $i$'s **category** — "London", "customer segment C" |
| $\hat x_i$ | "x-i hat" | The **number** we will replace that category with |
| $j < i$ | "j less than i" | "Only rows that came **before** row $i$ in a random permutation" |
| $\mathbb{1}[x_j=x_i]$ | "indicator x-j equals x-i" | 1 if row $j$ is in the same category as row $i$, else 0 |
| $y_j$ | "y-j" | The **label** of that earlier row |
| $p$ | "p" | A **prior** — usually the overall mean of $y$ across the dataset |
| $a$ | "a" | How many imaginary prior observations to add — the smoothing strength |

Strip the $a$ and $p$ terms and it reads simply: **"the average label among earlier rows of my category."** The $a\cdot p$ in the numerator and $a$ in the denominator are Laplace-style smoothing, which stops the first occurrence of a rare category from being encoded as a bare $0$ or $1$.

**The bug it fixes, concretely.** Naive target encoding replaces each category with the mean label of *all* rows in that category — including row $i$ itself. Now consider a category that occurs exactly **once** in your data, say `city = Reykjavík`, with label $y = 1$.

- Naive encoding: $\hat x = 1.0$. The encoded feature **is** the label, exactly.
- The tree splits on it, gets that row perfectly right, and your training accuracy looks superb.
- At prediction time, a new Reykjavík row is encoded using the training statistic — which was computed from a *different* row's label. **The information the model relied on does not exist for new data.** Validation collapses, and if your validation set was encoded using statistics computed on the full data, validation collapses only in production.

▸ **This is the single most common silent leak in applied tabular machine learning**, and it is silent precisely because the model looks *better*, not worse, at every stage before deployment. High-cardinality ID-like columns — user IDs, postcodes, SKUs — are where it bites, and those are exactly the columns people reach for target encoding to handle.

**How the "$j<i$" fixes it.** Row $i$'s encoding uses only rows that came before it in a random ordering, so **row $i$'s own label can never appear in its own feature.** It is the same principle as out-of-fold encoding (compute the statistic on the other $K-1$ folds), just applied one row at a time instead of one fold at a time.

**The cost, and CatBoost's fix for the cost.** Early rows in the permutation have almost no history — row 3 might be encoded from a single prior observation, which is nearly pure noise. So CatBoost maintains **several independent random permutations** and averages, so no single row is unlucky in all of them. The same permutation machinery drives **ordered boosting**: to compute row $i$'s residual, use a model trained only on rows before $i$, which removes the subtle bias where residuals are computed by a model that already fitted those very rows.

> **Analogy for prediction shift.** A teacher grades a student's mock exam, then uses those marks to decide what to teach next, then sets the *same* questions on the real exam. The class does brilliantly and has learned nothing generalizable. Ordered boosting insists the teacher only ever plan tomorrow's lesson using yesterday's students' results.

**Oblivious trees, decoded.** CatBoost's trees use **the same split condition at every node of a given depth**. A depth-6 oblivious tree asks the same 6 questions of every input, in the same order — so the tree is really a list of 6 conditions, and evaluating it means computing 6 bits and using them as an index into a table of $2^6 = 64$ leaf values. **No branching, no pointer chasing, fully vectorizable.** It is a much weaker model per tree (it cannot ask a different question down the left branch than the right), which is exactly the point: it is a strong regularizer, and CatBoost compensates with more trees.

> **Where this came from.** **CatBoost** was released by **Yandex** in 2017, with the ordered-boosting analysis published by **Liudmila Prokhorenkova and colleagues** in 2018. Yandex had a long institutional investment in this area: its production search-ranking model, **MatrixNet**, had used oblivious-tree boosting since around 2009, which is why the oblivious-tree design shows up in CatBoost rather than in the American libraries. **LightGBM** came from **Microsoft Research** in 2016–2017 (Guolin Ke and colleagues), also driven by a search-ranking workload. ▸ **All three of the major boosting libraries were built by organizations that needed to rank things — web results, ads, recommendations — at a scale where a 10× speedup was worth more than a 1% accuracy gain.** Their design differences trace directly to the differences in those workloads: Yandex had enormous categorical vocabularies, Microsoft had enormous sparse feature counts, and the academic XGBoost project had to run well on a laptop.

**Ordered boosting.** The same permutation trick applied to the residuals themselves: the model used to compute example $i$'s residual is trained only on examples before $i$. This removes the subtle **prediction shift** bias in standard gradient boosting (where residuals are computed using a model that has already seen those examples). Matters most on small datasets.

**Oblivious trees:** the same split condition at every node of a given depth. A strong regularizer and extremely fast at inference (the tree becomes a lookup into a $2^{\text{depth}}$ table).

### Choosing

| | Use when |
|---|---|
| **XGBoost** | default; best-documented; most robust |
| **LightGBM** | large data, many features, speed matters |
| **CatBoost** | many categorical features, small data, minimal tuning |

Differences are usually within noise after tuning. **Try all three; it takes an hour.**

---

## 23.7 Tuning, in priority order

1. **`n_estimators` with early stopping** on a validation set. Set it high and let early stopping decide.
2. **`learning_rate`** — 0.05 or 0.01; lower needs more trees.
3. **`max_depth`** (3–8) or **`num_leaves`** (LightGBM, ~$2^{\text{depth}}$ but usually less).
4. **`min_child_weight`** / `min_data_in_leaf` — the main overfitting control on noisy data.
5. **`subsample`** and **`colsample_bytree`** (0.6–0.9).
6. **`reg_lambda`** ($\lambda$) and **`reg_alpha`**.
7. **`gamma`** — minimum split gain.

▸ **The single highest-value habit: always use early stopping with a validation set.** It removes the most important hyperparameter from the search entirely.

#### What each knob actually does

The seven parameters above are not seven unrelated dials — they are three groups attacking the same thing from different angles.

| Group | Parameters | What it controls | Turn it which way to reduce overfitting? |
|---|---|---|---|
| **How long you train** | `n_estimators`, `learning_rate` | Total distance travelled through function space | Fewer trees, or smaller steps |
| **How complex each tree is** | `max_depth`, `num_leaves`, `min_child_weight`, `gamma` | Capacity of one correction | Shallower, fewer leaves, larger minimums, higher toll |
| **How much randomness you inject** | `subsample`, `colsample_bytree` | Decorrelation between successive trees | Lower fractions (more randomness) |

**`min_child_weight` is the one whose name misleads everyone.** It is not a count of rows — it is a **minimum required $H_j$**, the summed second derivative in a leaf. For squared error, $h_i = 1$ for every example, so $H_j$ *is* the row count and the name is accurate. For logistic loss, $h_i = p(1-p)$, so a leaf holding 100 examples the model is already confident about ($p = 0.99$) has $H_j \approx 1$, and `min_child_weight=5` will refuse to create it.

▸ **That is better behaviour than a raw row count, not worse: it asks for a minimum amount of *evidence*, not a minimum number of *rows*.** A hundred examples the model has already settled carry roughly one example's worth of information about where to split next, and the parameter is measuring exactly that.

**Why early stopping is worth more than everything else combined.** Boosting's loss curve on held-out data is U-shaped: it falls, flattens, and eventually rises as the ensemble starts fitting noise. Early stopping simply watches the validation metric and halts when it has not improved for `early_stopping_rounds` (typically 50–100) iterations, keeping the best-scoring model. Set `n_estimators = 10000` and let it stop wherever it stops.

▸ **This converts your most sensitive hyperparameter from something you tune by trial into something that is *measured*.** And it interacts correctly with the others: halve the learning rate and early stopping automatically runs about twice as long, with no intervention. Every other tuning decision becomes cheaper because you are no longer re-tuning tree count alongside it.

**One trap:** the early-stopping set is now part of your model selection, so its score is optimistically biased. If you are reporting a number to anyone, hold out a *third* split that early stopping never saw (Ch. 3).

---

## 23.8 Interpretation with SHAP

Shapley values from cooperative game theory, applied to feature attribution:

▸ $$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{j\})-f(S)\big]$$

The average marginal contribution of feature $j$ over all possible orderings of the features.

**Why it's the standard:** it is the *unique* attribution satisfying local accuracy (attributions sum to the prediction minus the base value), missingness, and consistency (if a feature's contribution increases in every subset, its attribution cannot decrease). **TreeSHAP** computes it exactly in $O(TLD^2)$ for tree ensembles rather than the exponential general case — which is why SHAP took off in the tabular world specifically.

**Caveats:** SHAP explains the *model*, not the world — a large SHAP value is not evidence of causation. Correlated features split credit in ways that can mislead, and the choice of background distribution changes the numbers.

#### Reading the Shapley formula

$$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{\lvert S\rvert!\,(\lvert F\rvert-\lvert S\rvert-1)!}{\lvert F\rvert!}\big[f(S\cup\{j\})-f(S)\big]$$

This is the most notation-dense formula in the chapter, and every piece of it is bookkeeping around one simple idea.

| Symbol | Read aloud | What it is |
|---|---|---|
| $F$ | "F" | The **full set of features** — all your columns |
| $j$ | "j" | The one feature whose credit we are computing |
| $F\setminus\{j\}$ | "F minus j" | All the features **except** $j$. The backslash means set subtraction. |
| $S$ | "S" | A **subset** — some coalition of the other features |
| $S\subseteq F\setminus\{j\}$ | "S a subset of F-minus-j" | "Sum over **every possible** such coalition" |
| $\lvert S\rvert$ | "the size of S" | How many features are in that coalition |
| $!$ | "factorial" | $4! = 4\times3\times2\times1 = 24$ |
| $f(S)$ | "f of S" | The model's prediction **when only the features in $S$ are known** |
| $f(S\cup\{j\}) - f(S)$ | "f of S-union-j, minus f of S" | **How much the prediction changes when you add feature $j$ to that coalition** |
| $\phi_j$ | "phi-j" | Feature $j$'s total credit for this one prediction |

**The idea, without notation.** Line the features up in a random order and reveal them one at a time, watching the prediction move. Feature $j$'s contribution *in that ordering* is however much the prediction moved at the moment $j$ was revealed. **Do this for every possible ordering and average.** That average is $\phi_j$.

**The scary fraction is just a counting weight.** $\frac{\lvert S\rvert!\,(\lvert F\rvert-\lvert S\rvert-1)!}{\lvert F\rvert!}$ is the fraction of the $\lvert F\rvert!$ possible orderings in which exactly the features of $S$ come before $j$: $\lvert S\rvert!$ ways to arrange the ones before, $(\lvert F\rvert - \lvert S\rvert - 1)!$ ways to arrange the ones after. **It is there so that summing over subsets gives the same answer as averaging over orderings.**

> **Analogy — and it is the original one.** Three people share a taxi home. Alone, Ana's fare would be £10, Ben's £20, Carl's £30. Sharing, various pairs and the trio cost less. How should the £30 total be split? Any single ordering ("Ana was picked up first, so she pays her £10 and the others pay the extra") is arbitrary and favours whoever you happened to list first. **Shapley's answer: average over all orderings.** Each person pays their average marginal cost, and the result is provably the only split satisfying a short list of fairness axioms. A machine learning prediction is the taxi fare; the features are the passengers.

**Work a two-feature example.** Model $f$, base value (prediction with no features known) $= 0.30$. Features: `age` and `income`.

| Known features | Prediction |
|---|---|
| none | $0.30$ |
| age only | $0.50$ |
| income only | $0.40$ |
| both | $0.80$ |

Two orderings, each with probability $\frac12$:

- **age then income:** age contributes $0.50 - 0.30 = 0.20$; income contributes $0.80-0.50 = 0.30$.
- **income then age:** income contributes $0.40-0.30 = 0.10$; age contributes $0.80-0.40 = 0.40$.

Average: $\phi_{\text{age}} = \frac{0.20+0.40}{2} = 0.30$, and $\phi_{\text{income}} = \frac{0.30+0.10}{2} = 0.20$.

**Check the sum:** $0.30 + 0.30 + 0.20 = 0.80$. ✓ The base value plus all attributions equals the prediction exactly. ▸ **That is the "local accuracy" property, and it is the reason SHAP displaces every earlier attribution method: the numbers *add up*.** You can hand a customer a sentence of the form "your score was 0.80: everyone starts at 0.30, your age added 0.30, your income added 0.20" and it is arithmetically true, not a metaphor.

Notice also that neither feature's credit equals what it does alone — age alone adds $0.20$ but is credited $0.30$, because the two features interact and the surplus has to be shared. **Splitting the interaction surplus fairly is the entire problem Shapley solved.**

#### Why TreeSHAP was the unlock

The sum runs over **every subset** of the other features: $2^{\lvert F\rvert - 1}$ of them. With 30 features that is over half a billion evaluations **per prediction explained**. Explaining a thousand rows would take longer than training the model several thousand times over. For a decade this made Shapley attribution a lovely idea nobody could use.

▸ **TreeSHAP exploits the fact that a tree is not an arbitrary function: it is a set of paths.** By pushing all subsets through the tree simultaneously and tracking, at each node, what fraction of subsets would have gone which way, it computes the exact Shapley values in $O(TLD^2)$ — number of trees, times leaves, times depth squared. For a typical model ($T=500$, $L=32$, $D=6$) that is about $6\times10^5$ operations instead of $10^9$, per row, and **exact rather than sampled.**

**Real numbers on why this mattered.** Explaining 10,000 rows of a 30-feature model: exact enumeration, comfortably beyond a day on one machine; TreeSHAP, a few seconds. That is the difference between "interesting in principle" and "runs in the nightly job," and it is why SHAP became the default explanation tool in the tabular world specifically and not, say, in vision.

> **Where this came from.** The **Shapley value** was defined by **Lloyd Shapley** in 1953 — in cooperative game theory, answering how a coalition should divide a jointly-earned payoff. It is characterized axiomatically: efficiency (the shares sum to the total), symmetry (identical players get identical shares), null player (a player who adds nothing to every coalition gets nothing), and additivity. Shapley's proof shows it is the **unique** rule satisfying all four, which is where the "it is the only method with these properties" claim in the text comes from. Shapley won the Nobel Memorial Prize in Economics in 2012 — but for **matching theory** and the deferred-acceptance algorithm developed with David Gale, not for the Shapley value. The application to machine learning came from **Scott Lundberg and Su-In Lee** in 2017, whose contribution was twofold: showing that several existing attribution methods (LIME, DeepLIFT, layer-wise relevance propagation) were all approximating the same underlying quantity, and then finding the polynomial-time tree algorithm that made it practical. ▸ **A 1953 theorem about dividing spoils among coalitions became the standard tool of regulated model governance, because a bank explaining a declined loan needs attributions that provably sum to the decision.**

---

## 23.9 Why trees still beat deep learning on tabular data

The empirical finding (Grinsztajn et al., 2022, and repeatedly since) with the reasons:

1. **Tabular decision boundaries are often axis-aligned and piecewise constant.** Trees represent this natively; MLPs have a smoothness bias that fights it.
2. **Uninformative features.** Trees simply never split on them; MLPs must learn to ignore them, and their performance degrades measurably as junk features are added.
3. **Heavy-tailed, non-Gaussian, mixed-type features.** Trees are invariant to any monotone transform; neural networks need careful preprocessing.
4. **No natural weight-sharing structure.** The inductive biases that make CNNs and transformers work (locality, translation invariance, sequence order) have no analogue in a table of unrelated columns.
5. **Small $n$.** Most real tabular datasets are $10^3$–$10^6$ rows, which is the regime where a strong prior beats a flexible model.

▸ **The honest current state:** deep tabular models (FT-Transformer, TabPFN, TabM) have closed much of the gap and TabPFN-style in-context models are  better on very small datasets. But **gradient boosting remains the correct default**, and saying so in an interview signals judgement rather than fashion-following.

#### The five reasons, decoded

The list above is really one idea seen from five angles: **a table is not an image, and every inductive bias that makes deep learning work was designed for data with geometry.**

**1. Axis-aligned, piecewise-constant boundaries.** A neural network with smooth activations approximates smooth functions efficiently and step functions badly — to make a sharp jump at `age = 65` it must push weights large, which weight decay is actively fighting. A tree makes that jump with one comparison. **Real tabular targets are full of such jumps**, because they were often created by rules: retirement at 65, tax brackets, shipping tiers, credit thresholds. The data-generating process was literally a decision tree, so a decision tree fits it in a handful of splits.

**2. Uninformative features.** Add 50 columns of pure noise to a dataset. A tree evaluates them at every node, finds they buy no gain, and never splits on them — the cost is compute, not accuracy. An MLP (multi-layer perceptron) multiplies every input by a weight from the very first step, so noise enters every hidden unit and must be *learned away* over many epochs. Grinsztajn and colleagues measured this directly and found deep models degrade measurably as junk features are added while trees barely move. ▸ **Trees have feature selection built into their forward pass; networks have to learn it.**

**3. Heavy tails and mixed types.** A tree only ever asks "is this value above that value?", so replacing income by $\log(\text{income})$ — or by its rank, or by any other increasing function — changes **nothing about the fitted tree**. That invariance is free. A network sees a column with values spanning $10^0$ to $10^7$ and either saturates or gets swamped, so you must standardize, log-transform, clip, and check. Every one of those is a modelling decision you can get wrong.

**4. No weight-sharing structure to exploit.** A convolution works because pixel $(i,j)$ and pixel $(i,j{+}1)$ are *the same kind of thing* in a spatial relationship, so one filter can be reused everywhere. Attention works because tokens are the same kind of thing in a sequence. **Column 3 of your table is "customer tenure in months" and column 4 is "country code" — they are not the same kind of thing and there is no meaningful ordering between them.** Permute your columns and the problem is unchanged, which is exactly the situation in which the architecture has nothing to exploit.

**5. Small $n$.** With $10^4$ rows you cannot afford to *learn* the inductive biases above; you need them supplied. ▸ **A strong prior beats a flexible model precisely when data is scarce — which is the same reason ridge beats ordinary least squares in Ch. 22 and the same reason scaling laws (Ch. 15) point the other way once $n$ reaches $10^{12}$ tokens.** There is no contradiction between "deep learning won at language" and "trees win at tables"; the two domains sit at opposite ends of the same axis.

---

## Did you know?

- **Boosting was a theorem before it was an algorithm.** Michael Kearns asked in 1988 whether a learner that is only guaranteed to beat a coin flip could be amplified into an arbitrarily accurate one. Most people expected "no." Robert Schapire proved "yes" in 1990, and the proof itself was the first boosting algorithm. Freund and Schapire won the Gödel Prize — a theory award — for AdaBoost in 2003.

- **"Random Forests" is a registered trademark.** Leo Breiman and Adele Cutler trademarked the name and licensed it exclusively to a commercial vendor. This is why the technique is universally described in lowercase in papers and libraries, and why nobody sells a product called Random Forest.

- **The Gini impurity is an income-inequality statistic.** Corrado Gini introduced it in 1912 to measure dispersion in wealth. The identical formula is the Simpson diversity index in ecology, and its complement $\sum_k p_k^2$ is the Herfindahl–Hirschman index that antitrust regulators use to decide whether a merger concentrates a market too far.

- **The word "matrix" of trees — CART — was written by four authors and one of them had already left academia.** Leo Breiman spent thirteen years as a full-time statistical consultant before returning to Berkeley in 1980, four years before the CART book. His later polemic *Statistical Modeling: The Two Cultures* (2001) argued that academic statistics had wasted decades on models it could prove things about rather than algorithms that predicted well — a position that reads as obvious now and was contentious then.

- **XGBoost's reputation was made on a leaderboard, not in a journal.** Its own KDD 2016 paper reports that of the 29 winning Kaggle solutions published during 2015, **17 used XGBoost** — and the library had already been open for two years by the time the paper appeared.

- **Boosted decision trees are standard equipment in particle physics.** Ensembles of boosted trees are used throughout the analysis pipelines at the Large Hadron Collider to separate rare signal events from overwhelming background, including in the analyses that established the Higgs boson. A method invented to answer a question in computational learning theory is now part of how a fundamental particle gets confirmed.

- **The bootstrap is named after an impossible act.** "Pulling yourself up by your own bootstraps" describes something that cannot physically be done — which is roughly what the method appears to do: estimate how much your answer would vary across datasets you were never given, using only the one you have. Its predecessor, the **jackknife**, was named by John Tukey after the pocket knife: a crude tool that does many jobs adequately.

- **About 63.2% of your rows appear in any bootstrap sample, and this number never changes.** $1 - e^{-1} = 0.632$ holds whether you have a thousand rows or a billion. The constant appears so often in the resampling literature that there are bootstrap variance estimators literally named ".632" and ".632+".

- **Yandex's boosting library exists because Russian search queries have enormous categorical vocabularies.** CatBoost's distinguishing features — ordered target statistics and oblivious trees — both trace to Yandex's production search ranker MatrixNet, which had used symmetric trees since roughly 2009 for inference speed. The three major boosting libraries were each built by an organization that needed to rank things fast, and their differences are the fingerprints of three different workloads.

- **SHAP is a 1953 game theory result, and Shapley's Nobel was for something else.** Lloyd Shapley shared the 2012 Nobel Memorial Prize in Economics with Alvin Roth for matching theory and the deferred-acceptance algorithm — not for the Shapley value, which is by far his most-used contribution outside economics.

- **A decision tree cannot predict a number it has never seen the neighbourhood of.** Every prediction is a leaf mean, so the output is bounded by the range of the training targets. Fit a tree to house prices from 2015 and deploy it in 2025 and it will confidently report 2015 prices forever, flat, with no warning that anything is wrong.

- **Misclassification error is the thing you care about and the thing you must not optimize.** Because it is piecewise linear in the class proportions, a split that  purifies both children can register exactly zero improvement — and a greedy algorithm that sees zero improvement stops. Gini and entropy exist because greedy search needs a surface that always tilts.

- **The name for the technique everyone calls GBDT was originally MART.** Jerome Friedman called his tree version Multiple Additive Regression Trees, and the commercial implementation he was associated with shipped under that name. "Gradient boosting" won because it says what the method does.

---

## Check for Understanding

**Ensembling variance is $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$, so random forests exist to reduce the correlation $\rho$ that averaging cannot touch; boosting attacks the other term, fitting shallow trees sequentially to the negative gradient of any differentiable loss, and XGBoost made this exact by second-order expansion — giving the leaf weight $-G/(H+\lambda)$ and a split gain that subtracts a fixed per-leaf cost, which is principled pruning rather than a heuristic.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why does averaging a million trees stop helping long before a million?** (Say the word "correlation" and then say what it means in terms of shared blind spots.)
2. **Why do random forests deliberately hide the best feature from most of their trees?** Which term of the variance formula does that improve, and what does it cost?
3. **Why is misclassification error a bad splitting criterion when it is exactly the thing you want to minimize?**
4. **Why is a decision tree "high variance"?** Answer in terms of what happens to the tree when you delete one row.
5. **What is the difference between bagging and boosting, in terms of what each one is trying to cancel out?** Why does one of them need early stopping and the other doesn't?
6. **What does "gradient descent in function space" mean?** (Correct answer names what the parameter vector is, and says why you need a tree rather than just applying the gradient directly.)
7. **Why is AdaBoost so fragile to a single mislabelled row?** Trace it through the reweighting step.
8. **Why do gradient boosting libraries use shallow trees and random forests use deep ones?**
9. **What is $\lambda$ doing in $w^* = -G/(H+\lambda)$, and why does a fixed $\lambda$ shrink small leaves harder than large ones?**
10. **Why does target encoding leak, and how does "only use rows before me" fix it?**
11. **What does a SHAP value mean, in terms of splitting a taxi fare?** Why does it matter that the attributions sum to the prediction?
12. **Why do trees still beat neural networks on tabular data?** Give at least three reasons, and say what changes if you suddenly have a billion rows.
13. **Why is 36.8% of your data left out of every bootstrap sample, and what do you get for free because of it?**
14. **Why does a lower learning rate with more trees usually win — and when is that advice wrong?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 24 — Unsupervised Learning & Dimensionality Reduction](24-unsupervised-learning.md)
