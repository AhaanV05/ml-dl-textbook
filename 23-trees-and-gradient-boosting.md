# Chapter 23 — Trees & Gradient Boosting

> **Prerequisites:** Ch. 2 (bias–variance), Ch. 22.
> **Why this chapter matters more than its reputation suggests:** gradient boosting is still the best method for tabular data, it wins most tabular Kaggle competitions, and it is what most production ML systems outside of vision/NLP actually run. It is also a beautiful piece of mathematics.

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

▸ **Why misclassification error is a bad splitting criterion, even though it's the thing you care about:** it is piecewise-linear in $p$, so a split that moves probability mass without changing the argmax gives *zero* improvement. Gini and entropy are strictly concave, so **any** split that increases purity registers a gain. Concavity is what makes greedy search work at all. This is a genuinely good interview question.

Gini and entropy agree on the chosen split ~98% of the time. Gini is marginally cheaper (no log).

**Information gain** $= H(\text{parent}) - \sum\frac{n_c}{n}H(\text{child})$, which is exactly the mutual information between the split variable and the label.

**Regression trees** predict the mean of each leaf and split to minimize the sum of squared errors.

### Complexity control

- **Pre-pruning:** `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.
- **Cost-complexity (post-)pruning:** grow fully, then minimize
▸ $$R_\alpha(T) = R(T) + \alpha|T_{\text{leaves}}|$$
This traces a nested sequence of subtrees as $\alpha$ increases; pick $\alpha$ by cross-validation. Post-pruning is better than pre-pruning because a split that looks useless can enable a useful one below it (the XOR problem).

### Properties

**Good:** no scaling needed, handles mixed types and missing values natively (surrogate splits), captures interactions automatically, interpretable, invariant to monotone feature transforms.

**Bad:** ▸ **extremely high variance** — change one data point near a root split and the entire tree below changes. Axis-aligned splits struggle with diagonal boundaries. Cannot extrapolate outside the training range. Biased toward high-cardinality features.

That high variance is the exact weakness that ensembling was invented to fix.

---

## 23.2 Bagging and random forests

### The variance formula — the heart of the matter

For $B$ estimators each with variance $\sigma^2$ and pairwise correlation $\rho$:

▸ $$\mathrm{Var}\left(\frac1B\sum_{b}f_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

**Read it carefully.** The second term vanishes as $B\to\infty$. The first does not.

▸ **Averaging more trees can never reduce variance below $\rho\sigma^2$. Therefore the only way to improve a large ensemble is to reduce the correlation $\rho$ between its members.** This single formula explains the entire design of random forests, and it is the thing to say if asked "why random feature subsets?"

### Bagging

Train each tree on a bootstrap resample. Reduces $\rho$ somewhat (different data), but the trees still see the same features and tend to pick the same strong splits at the root, so $\rho$ stays high.

**Out-of-bag estimation:** each bootstrap omits $(1-1/n)^n\to e^{-1}=36.8\%$ of the data. Evaluate each point using only the trees that didn't see it — **a free cross-validation estimate** with no extra fitting.

### Random forests

Bagging **plus** a random subset of $m$ features considered at each split.

▸ $m=\sqrt p$ for classification, $p/3$ for regression.

**This is the key addition:** by preventing every tree from using the same dominant feature at the top, it directly attacks $\rho$ in the formula above. The individual trees get slightly worse (higher $\sigma^2$), but $\rho$ drops more, and the ensemble improves. **A deliberate bias-for-decorrelation trade.**

**Extremely Randomized Trees (ExtraTrees):** also choose the split *threshold* at random rather than optimally. Even lower $\rho$, even higher individual bias, faster to train. Often competitive.

**Feature importance.** Two kinds, and the difference matters:
- **MDI (mean decrease in impurity):** sum of impurity reductions, weighted by samples. Fast, but ▸ **biased toward high-cardinality and continuous features**, and computed on training data.
- **Permutation importance:** shuffle a feature on held-out data and measure the performance drop. Slower, unbiased with respect to cardinality, but ▸ **misleading with correlated features** — shuffling one of two correlated features shows no drop because the other covers for it, so both look unimportant.

**Practical guidance:** random forests need almost no tuning (more trees is monotonically better, just slower), are hard to overfit, and make an excellent baseline. But on most tabular problems, well-tuned gradient boosting beats them.

---

## 23.3 Boosting

### The one-line idea

Instead of averaging independent strong learners, build weak learners **sequentially**, each one fixing what the current ensemble gets wrong.

### The analogy

Bagging is asking a hundred people independently and averaging. Boosting is a relay of specialists: the first gives a rough answer, the second is hired specifically to fix the first's mistakes, the third to fix what remains. Each is weak alone; the sequence is strong.

▸ **The bias–variance contrast:** bagging reduces **variance** (average many low-bias, high-variance trees). Boosting reduces **bias** (sum many high-bias, low-variance trees — "stumps" of depth 3–6). They are attacking opposite terms of the same decomposition, which is why boosting uses shallow trees and forests use deep ones.

### AdaBoost

**The algorithm.** Initialize weights $w_i=1/n$. For $m=1..M$:
1. Fit a weak classifier $G_m$ on weighted data.
2. Compute weighted error $\mathrm{err}_m = \frac{\sum_i w_i\mathbb{1}[y_i\ne G_m(x_i)]}{\sum_i w_i}$.
3. ▸ $\alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$
4. ▸ $w_i \leftarrow w_i\exp\big(\alpha_m\mathbb{1}[y_i\ne G_m(x_i)]\big)$ — **misclassified points get up-weighted.**

Final: $G(x)=\mathrm{sign}\left(\sum_m\alpha_mG_m(x)\right)$.

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

### The three regularizers that make it work

1. **Learning rate (shrinkage)** $\eta\approx0.01$–$0.1$. ▸ **Lower $\eta$ with more trees is nearly always better** (the classic $\eta$–$M$ trade-off). This is the single most important hyperparameter.
2. **Subsampling** (stochastic gradient boosting): use a random 50–80% of rows per tree. Adds variance reduction on top of bias reduction, and speeds training.
3. **Tree constraints:** depth 3–8, min samples per leaf, and the $\ell_2$ penalty on leaf weights (below).

---

## 23.5 XGBoost — derive the objective

XGBoost's contribution was to write down the *exact* objective a tree should optimize, using a second-order expansion, rather than fitting to gradients heuristically.

### The regularized objective

▸ $$\mathcal{L}^{(m)} = \sum_{i=1}^n L\big(y_i,\ F_{m-1}(x_i)+f_m(x_i)\big) + \Omega(f_m),\qquad \Omega(f)=\gamma T + \tfrac12\lambda\sum_{j=1}^{T}w_j^2$$

where $T$ = number of leaves and $w_j$ = the value at leaf $j$.

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

### CatBoost

**Ordered target statistics.** Encoding a categorical feature by its target mean leaks the label. CatBoost fixes this by using, for each example, only the examples *before it* in a random permutation:
▸ $$\hat x_i = \frac{\sum_{j<i}\mathbb{1}[x_j=x_i]y_j + a\cdot p}{\sum_{j<i}\mathbb{1}[x_j=x_i] + a}$$
▸ **This is the same idea as out-of-fold target encoding**, applied online. Target leakage in categorical encoding is one of the most common silent bugs in applied ML, and knowing this fix is valuable well beyond CatBoost.

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

---

## 23.8 Interpretation with SHAP

Shapley values from cooperative game theory, applied to feature attribution:

▸ $$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{j\})-f(S)\big]$$

The average marginal contribution of feature $j$ over all possible orderings of the features.

**Why it's the standard:** it is the *unique* attribution satisfying local accuracy (attributions sum to the prediction minus the base value), missingness, and consistency (if a feature's contribution increases in every subset, its attribution cannot decrease). **TreeSHAP** computes it exactly in $O(TLD^2)$ for tree ensembles rather than the exponential general case — which is why SHAP took off in the tabular world specifically.

**Caveats:** SHAP explains the *model*, not the world — a large SHAP value is not evidence of causation. Correlated features split credit in ways that can mislead, and the choice of background distribution changes the numbers.

---

## 23.9 Why trees still beat deep learning on tabular data

The empirical finding (Grinsztajn et al., 2022, and repeatedly since) with the reasons:

1. **Tabular decision boundaries are often axis-aligned and piecewise constant.** Trees represent this natively; MLPs have a smoothness bias that fights it.
2. **Uninformative features.** Trees simply never split on them; MLPs must learn to ignore them, and their performance degrades measurably as junk features are added.
3. **Heavy-tailed, non-Gaussian, mixed-type features.** Trees are invariant to any monotone transform; neural networks need careful preprocessing.
4. **No natural weight-sharing structure.** The inductive biases that make CNNs and transformers work (locality, translation invariance, sequence order) have no analogue in a table of unrelated columns.
5. **Small $n$.** Most real tabular datasets are $10^3$–$10^6$ rows, which is the regime where a strong prior beats a flexible model.

▸ **The honest current state:** deep tabular models (FT-Transformer, TabPFN, TabM) have closed much of the gap and TabPFN-style in-context models are genuinely better on very small datasets. But **gradient boosting remains the correct default**, and saying so in an interview signals judgement rather than fashion-following.

---

## Check for Understanding

**Ensembling variance is $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$, so random forests exist to reduce the correlation $\rho$ that averaging cannot touch; boosting attacks the other term, fitting shallow trees sequentially to the negative gradient of any differentiable loss, and XGBoost made this exact by second-order expansion — giving the leaf weight $-G/(H+\lambda)$ and a split gain that subtracts a fixed per-leaf cost, which is principled pruning rather than a heuristic.**

---

**Next:** [Chapter 24 — Unsupervised Learning & Dimensionality Reduction](24-unsupervised-learning.md)
