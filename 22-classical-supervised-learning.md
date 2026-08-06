# Chapter 22 — Classical Supervised Learning

> **Prerequisites:** Ch. 1, Ch. 2.
> **Why it still matters:** on tabular data, small datasets, and any problem where you must explain the model, these methods win. They are also where the clearest mathematics in ML lives, which is why interviews return to them.

---

## 22.1 Linear regression

### The one-line idea

Find the linear function whose predictions have the smallest squared error, which geometrically means projecting the target vector onto the column space of the design matrix.

### The analogy

You have a target point floating in space and a flat sheet of paper (the space of all achievable predictions). The best you can do is the point on the paper closest to the target — the shadow it casts straight down. The error is the perpendicular from the target to the paper, which is why the residual is orthogonal to every predictor.

### The normal equations

$$\hat\beta = \arg\min_\beta\|y-X\beta\|^2$$

$$\nabla_\beta\|y-X\beta\|^2 = -2X^\top(y-X\beta)=0$$

▸ $$\boxed{\ X^\top X\hat\beta = X^\top y\quad\Longrightarrow\quad \hat\beta = (X^\top X)^{-1}X^\top y\ }$$

**The geometry:** $\hat y = X\hat\beta = \underbrace{X(X^\top X)^{-1}X^\top}_{H,\ \text{the hat matrix}}y$. $H$ is the orthogonal projection onto $\mathrm{col}(X)$: $H^2=H$, $H^\top=H$, $\mathrm{tr}(H)=p$ (the number of parameters — this is the "degrees of freedom").

**The residual is orthogonal to every column of $X$:** $X^\top(y-X\hat\beta)=0$. That single equation *is* the normal equations, and it is the cleanest way to remember them.

**Never compute $(X^\top X)^{-1}$ numerically.** The condition number of $X^\top X$ is the *square* of that of $X$. Use QR ($X=QR$, solve $R\hat\beta = Q^\top y$) or SVD.

### Gauss–Markov

Under (i) linearity, (ii) $\mathbb{E}[\varepsilon]=0$, (iii) homoskedasticity and no autocorrelation ($\mathrm{Var}(\varepsilon)=\sigma^2I$), OLS is the **B**est **L**inear **U**nbiased **E**stimator — minimum variance among unbiased linear estimators. Normality is *not* required for this; it is required for exact $t$ and $F$ tests.

▸ **The word doing the work is "unbiased."** Ridge is biased and often has lower total error. Gauss–Markov is not a claim that OLS is best, only that it is best in a class you may not want to restrict yourself to.

---

## 22.2 Ridge regression

$$\hat\beta_{\text{ridge}} = \arg\min_\beta\ \|y-X\beta\|^2 + \lambda\|\beta\|^2 \quad\Longrightarrow\quad \hat\beta = (X^\top X+\lambda I)^{-1}X^\top y$$

### The SVD reading — the most illuminating one

Let $X = U\Sigma V^\top$ with singular values $\sigma_j$. Then

▸ $$\hat y_{\text{ridge}} = \sum_{j=1}^{p}u_j\underbrace{\frac{\sigma_j^2}{\sigma_j^2+\lambda}}_{\text{shrinkage factor}}u_j^\top y$$

versus OLS's $\hat y = \sum_j u_ju_j^\top y$ (all factors 1).

▸ **Ridge shrinks each principal direction by $\frac{\sigma_j^2}{\sigma_j^2+\lambda}$ — barely at all for high-variance directions ($\sigma_j^2\gg\lambda$) and almost to zero for low-variance ones.** Directions the data barely constrains are exactly the ones estimated with the most noise, so this is precisely the right thing to do. This is the same statement as Ch. 7 §7.5's eigenbasis shrinkage, and it is the best one-line justification of $\ell_2$ regularization in existence.

**Effective degrees of freedom:** $\mathrm{df}(\lambda)=\sum_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}$, ranging from $p$ (at $\lambda=0$) to 0. **A continuous model-complexity dial** — and the right $x$-axis for a bias–variance plot.

**Always standardize before ridge**, and never penalize the intercept.

---

## 22.3 LASSO and elastic net

$$\hat\beta_{\text{lasso}}=\arg\min_\beta\ \tfrac{1}{2}\|y-X\beta\|^2+\lambda\|\beta\|_1$$

**No closed form** (the $\ell_1$ term is non-differentiable at 0), but for orthonormal $X$ the solution is exactly the **soft-threshold**:

▸ $$\hat\beta_j = \mathcal{S}_\lambda(\hat\beta_j^{\text{OLS}}) = \mathrm{sign}(\hat\beta_j^{\text{OLS}})\big(|\hat\beta_j^{\text{OLS}}|-\lambda\big)_+$$

**Derivation of the sparsity.** The subgradient of $\lambda|\beta_j|$ at $\beta_j=0$ is the *interval* $[-\lambda,\lambda]$. Optimality requires $0\in -x_j^\top(y-X\beta)+\lambda\partial|\beta_j|$, so $\beta_j=0$ is optimal whenever $|x_j^\top r|\le\lambda$.
▸ **Any coordinate whose correlation with the residual is below $\lambda$ is set to exactly zero.** Ridge's penalty has gradient $2\lambda\beta_j\to0$ as $\beta_j\to0$, so it never produces exact zeros. **That is the entire difference between the two.**

**Geometric picture:** the $\ell_1$ ball is a cross-polytope with vertices on the axes. An elliptical level set of the loss expanding until it touches the ball will, generically, touch at a vertex — where coordinates are zero.

**Elastic net:** $\lambda\left(\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|^2\right)$. Needed when predictors are correlated: LASSO arbitrarily picks one of a correlated group and zeros the rest, which is unstable across resamples; the ridge term induces a **grouping effect** so correlated predictors get similar coefficients.

**Solved by:** coordinate descent (cycle over $j$, apply soft-thresholding — fast and the standard) or LARS (traces the whole regularization path).

---

## 22.4 Logistic regression

$$p(y=1\mid x)=\sigma(\beta^\top x),\qquad \mathcal{L}=-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]$$

### The gradient (derive it once)

Using $\sigma'=\sigma(1-\sigma)$:
▸ $$\nabla_\beta\mathcal{L} = \sum_i (p_i-y_i)x_i = X^\top(p-y)$$

**Identical in form to linear regression's $X^\top(\hat y-y)$.** This is not a coincidence — it holds for every generalized linear model with its canonical link, and it's the cleanest thing to say if asked why cross-entropy is "the right" loss for logistic regression.

**Hessian:** $H = X^\top SX$ with $S=\mathrm{diag}(p_i(1-p_i))$, which is **PSD** ⇒ the loss is convex ⇒ a unique global optimum. Newton's method here is **IRLS** (iteratively reweighted least squares).

▸ **Perfect separation** makes the MLE diverge: $\|\beta\|\to\infty$ pushes probabilities to 0/1 and the loss to 0 without ever attaining it. Any $\ell_2$ penalty fixes it. This is also the setting in which the *implicit bias* of gradient descent selects the max-margin direction (Ch. 31 §31.3).

**Interpretation:** $\beta_j$ is the change in **log-odds** per unit of $x_j$. $e^{\beta_j}$ is the odds ratio.

**Multiclass:** softmax regression, gradient $X^\top(P-Y)$ — the same shape again (Ch. 1 §1.3.4).

### Generalized linear models

$g(\mathbb{E}[y]) = \beta^\top x$ for a link $g$ and an exponential-family response.

| Response | Distribution | Canonical link |
|---|---|---|
| continuous | Gaussian | identity |
| binary | Bernoulli | logit |
| count | Poisson | log |
| count, overdispersed | Neg. binomial | log |
| positive skewed | Gamma | inverse or log |

---

## 22.5 Support vector machines

### The one-line idea

Among all separating hyperplanes, choose the one that is as far as possible from the nearest points of both classes — because the widest margin is the most robust to perturbation.

### The analogy

Building a road between two villages. Any road that doesn't cross a house works, but you want the *widest* road possible, so a car drifting slightly doesn't hit anything. The houses touching the road's edge are the support vectors; the ones far back are irrelevant to where you put the road.

### The margin

With $y_i\in\{-1,+1\}$ and the canonical scaling $\min_i y_i(w^\top x_i+b)=1$, the distance from a point to the hyperplane is $\frac{|w^\top x+b|}{\|w\|}$, so the **margin** is $\frac{2}{\|w\|}$.

Maximizing the margin = minimizing $\|w\|$:

▸ $$\min_{w,b}\ \tfrac12\|w\|^2\quad\text{s.t.}\quad y_i(w^\top x_i+b)\ge1\ \ \forall i$$

### The dual

Lagrangian $L = \frac12\|w\|^2-\sum_i\alpha_i\left[y_i(w^\top x_i+b)-1\right]$, $\alpha_i\ge0$.

Stationarity: $\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_i\alpha_iy_ix_i$; $\frac{\partial L}{\partial b}=0\Rightarrow \sum_i\alpha_iy_i=0$. Substituting:

▸ $$\max_\alpha\ \sum_i\alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\langle x_i,x_j\rangle\quad\text{s.t.}\quad \alpha_i\ge0,\ \sum_i\alpha_iy_i=0$$

▸ **Two facts fall out, and both matter:**
1. **The data enters only through inner products $\langle x_i,x_j\rangle$** → the kernel trick.
2. **KKT complementary slackness:** $\alpha_i\big[y_i(w^\top x_i+b)-1\big]=0$. So $\alpha_i>0$ **only** for points exactly on the margin. All other points have $\alpha_i=0$ and can be deleted without changing the solution. Those are the **support vectors**, and the sparsity is exact, not approximate.

### Soft margin

$$\min\ \tfrac12\|w\|^2 + C\sum_i\xi_i\quad\text{s.t. } y_i(w^\top x_i+b)\ge1-\xi_i,\ \xi_i\ge0$$

Equivalent unconstrained form with the **hinge loss**:
▸ $$\min_w\ \frac{1}{2}\|w\|^2 + C\sum_i\max\big(0,\ 1-y_i(w^\top x_i+b)\big)$$

The dual is unchanged except $0\le\alpha_i\le C$ (a "box constraint"). **Small $C$ = wide margin, more violations, more regularization.**

**Hinge vs logistic loss:** hinge is exactly zero once the margin is met (⇒ sparsity, only support vectors matter); logistic is always positive (⇒ every point contributes, and you get calibrated probabilities). That trade — sparsity vs probabilities — is the practical difference between SVM and logistic regression.

### Kernels

Replace $\langle x_i,x_j\rangle$ with $k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$. **$\phi$ is never computed.**

| Kernel | $k(x,x')$ | Note |
|---|---|---|
| Linear | $x^\top x'$ | |
| Polynomial | $(\gamma x^\top x'+r)^d$ | finite feature space |
| **RBF/Gaussian** | $\exp(-\gamma\|x-x'\|^2)$ | **infinite-dimensional**; the default |
| Laplacian | $\exp(-\gamma\|x-x'\|_1)$ | |
| Sigmoid | $\tanh(\gamma x^\top x'+r)$ | not PSD for all params |

**Mercer's condition:** $k$ is a valid kernel iff the Gram matrix $K_{ij}=k(x_i,x_j)$ is PSD for every finite sample.

▸ **The RBF kernel's feature map is infinite-dimensional** (expand the exponential in a Taylor series — every polynomial degree appears). So the SVM fits a linear classifier in an infinite-dimensional space, and it generalizes anyway — because, as Chapter 2 §2.5 showed, **the capacity that matters is the norm, not the dimension.** This is the single best illustration of that point.

$\gamma$ controls locality: large $\gamma$ means each point influences only its immediate neighbourhood → high variance, complex boundary.

### The representer theorem

▸ For any regularized loss of the form $\min_f \sum_i L(y_i,f(x_i)) + \Omega(\|f\|_{\mathcal{H}})$ over an RKHS $\mathcal{H}$, with $\Omega$ strictly increasing, **the minimizer has the form**
$$f^*(\cdot)=\sum_{i=1}^{n}\alpha_i k(x_i,\cdot)$$

*Sketch:* decompose $f = f_\parallel + f_\perp$ where $f_\parallel$ lies in the span of $\{k(x_i,\cdot)\}$. By the reproducing property, $f(x_i)=\langle f,k(x_i,\cdot)\rangle$ depends only on $f_\parallel$, so the loss term is unchanged by $f_\perp$; but $\|f\|^2=\|f_\parallel\|^2+\|f_\perp\|^2$, so the penalty is strictly increased by any nonzero $f_\perp$. Hence $f_\perp=0$ at the optimum. ∎

▸ **This is why kernel methods are computationally possible at all:** an infinite-dimensional optimization problem provably reduces to $n$ coefficients.

**Scaling:** SVM training is $O(n^2)$–$O(n^3)$, and prediction is $O(n_{\text{SV}}d)$. **Do not use kernel SVMs beyond ~100k samples.** Use linear SVM (LIBLINEAR), or approximate the kernel with random Fourier features: $k(x,x')\approx z(x)^\top z(x')$ with $z(x)=\sqrt{2/D}\cos(\omega^\top x+b)$, $\omega\sim\mathcal{N}(0,2\gamma I)$ — then run a linear model.

---

## 22.6 The rest of the classical toolkit

### Naive Bayes

$$\hat y=\arg\max_c\ p(c)\prod_j p(x_j\mid c)$$

The independence assumption is essentially always false, yet classification accuracy is often good — **because argmax only needs the ranking to be right, not the probabilities.** (The probabilities are badly miscalibrated, typically pushed to 0 or 1.) Variants: Multinomial (text counts), Bernoulli (binary), Gaussian (continuous). **Laplace smoothing** $\frac{count+\alpha}{total+\alpha K}$ is mandatory to avoid zero probabilities.

Extremely fast, one pass, a strong text baseline, and it works with very little data.

### k-Nearest Neighbours

No training; predict by majority/mean over the $k$ nearest points.

▸ **The curse of dimensionality**, made concrete: in $d$ dimensions, to capture a fraction $r$ of the data volume, a hypercube neighbourhood must have edge length $r^{1/d}$. For $d=10$, $r=0.01$: edge $=0.63$ — **63% of the range of every feature.** "Local" is not local. Relatedly, the ratio $\frac{\text{dist}_{\max}-\text{dist}_{\min}}{\text{dist}_{\min}}\to0$ as $d\to\infty$, so all points become equidistant and "nearest" stops meaning anything.

**The 1-NN bound** is worth knowing: as $n\to\infty$, the 1-NN error is at most twice the Bayes error. A remarkably strong guarantee for so simple a method.

### Discriminant analysis

**LDA** assumes Gaussian classes with a **shared** covariance ⇒ linear boundary; it is also a supervised dimensionality reduction (project onto directions maximizing $\frac{\text{between-class scatter}}{\text{within-class scatter}}$). **QDA** allows per-class covariance ⇒ quadratic boundary, $O(Cd^2)$ parameters.

### Imbalanced data

Ranked by what actually works:
1. **Use the right metric.** Accuracy is meaningless at 1% positives. Use **PR-AUC** (which, unlike ROC-AUC, is sensitive to the positive rate), F1, or a cost-weighted metric.
2. **Class weights / cost-sensitive loss.** Usually sufficient, and it's one parameter.
3. **Threshold tuning.** Train normally, then choose the decision threshold on a validation set to optimize your actual objective. ▸ **Frequently the whole solution**, and frequently skipped.
4. **Resampling.** Undersample the majority (fast, discards data), oversample the minority (overfits), or **SMOTE** (interpolate between a minority point and its neighbours — note it can create points inside the majority region, and it interacts badly with high dimensions).
5. **Focal loss** (Ch. 8 §8.5) for extreme imbalance in dense prediction.

▸ **Always resample *inside* the cross-validation loop, never before splitting** — otherwise synthetic points derived from training data leak into the validation fold and your score is fiction. This is one of the most common serious errors in applied ML.

---

## 22.7 When to use what

| Situation | Method |
|---|---|
| $n<1000$, need interpretability | regularized linear/logistic |
| Tabular, $n$ = 1k–1M | **gradient boosting** (Ch. 23) |
| Very wide, $p\gg n$ (genomics) | LASSO / elastic net |
| Text classification, fast baseline | linear SVM or NB on TF-IDF |
| Small $n$, complex boundary | kernel SVM |
| Need calibrated probabilities | logistic regression, or calibrate afterwards (Ch. 33) |
| Images, audio, text, sequences | deep learning |

---

## Check for Understanding

**Ridge shrinks each principal direction by $\sigma_j^2/(\sigma_j^2+\lambda)$ so poorly-determined directions are damped most, LASSO's non-differentiable corner at zero is what produces exact sparsity, and the SVM's dual shows the data entering only through inner products — which is what lets a linear classifier in an infinite-dimensional space be computed with $n$ coefficients and still generalize, because capacity is controlled by norm rather than dimension.**

---

**Next:** [Chapter 23 — Trees & Gradient Boosting](23-trees-and-gradient-boosting.md)
