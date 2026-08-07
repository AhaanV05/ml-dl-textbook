# Chapter 22 — Classical Supervised Learning

> **Prerequisites:** Ch. 1, Ch. 2.
> **Why it still matters:** on tabular data, small datasets, and any problem where you must explain the model, these methods win. They are also where the clearest mathematics in ML lives, which is why interviews return to them.

> **New to the notation?** If symbols like $\in$, $\sum$, $\arg\min$, $\nabla$, $X^\top$, or $\lVert\cdot\rVert$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. Nothing in this chapter needs more than that plus patience.

### Symbols introduced in this chapter

Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $X \in \mathbb{R}^{n\times p}$ | "the design matrix" | Your spreadsheet: one **row** per example, one **column** per feature |
| $\beta$ | "beta" | The coefficient vector — one weight per feature |
| $\hat\beta$ | "beta-hat" | The **estimated** coefficients. The hat means "our guess," not the truth |
| $\lVert y - X\beta\rVert^2$ | "norm of y minus X-beta, squared" | Total squared error, added up over all $n$ examples |
| $H$ | "the hat matrix" | The machine that turns $y$ into $\hat y$ — a projection, nothing more |
| $\lambda$ | "lambda" | **Regularization strength.** Bigger $\lambda$ = simpler, more shrunken model |
| $\sigma_j$ | "sigma-j" | A **singular value** of $X$ — how strongly the data pins down direction $j$ |
| $\lVert\beta\rVert_1$ | "the one-norm of beta" | $\sum_j \lvert\beta_j\rvert$ — add up the sizes, ignoring signs |
| $\mathcal{S}_\lambda$ | "soft-threshold at lambda" | Pull toward zero by $\lambda$; if you would overshoot, stop **at** zero |
| $(z)_+$ | "z-plus" / "positive part" | $\max(0, z)$. A ReLU wearing a subscript |
| $\sigma(z)$ | "sigmoid of z" | $1/(1+e^{-z})$ — squashes any real number into $(0,1)$ |
| $\alpha_i$ | "alpha-i" | A **dual variable**: how hard example $i$ pushes on the solution |
| $\xi_i$ | "xi-i" (rhymes with "sigh") | **Slack**: how far example $i$ is permitted to trespass over the margin |
| $C$ | "C" | The SVM's price of trespassing. Large $C$ = strict, small $C$ = forgiving |
| $k(x, x')$ | "the kernel of x and x-prime" | A similarity score between two examples |
| $\phi(x)$ | "phi of x" | The **feature map** — where $x$ would live in the expanded space, if you built it |
| $\mathcal{H}$ | "script H" | A reproducing kernel Hilbert space: the set of functions a kernel can build |
| $\gamma$ | "gamma" | The RBF kernel's width dial. Large $\gamma$ = each point's influence is very local |
| s.t. | "subject to" | "…while obeying the constraints that follow" |
| $\partial \lvert\beta_j\rvert$ | "the subdifferential" | The *set* of slopes at a corner where the derivative doesn't exist |

**Every abbreviation in this chapter, spelled out.** Read each full form aloud once; most acronyms stop being frightening the moment you hear what they stand for.

| Short | Full form |
|---|---|
| BLUE | Best Linear Unbiased Estimator |
| GLM | Generalized Linear Model |
| IRLS | Iteratively Reweighted Least Squares |
| KKT | Karush–Kuhn–Tucker (optimality conditions) |
| kNN | k-Nearest Neighbours |
| LARS | Least Angle Regression |
| LASSO | Least Absolute Shrinkage and Selection Operator |
| LDA | Linear Discriminant Analysis |
| MLE | Maximum Likelihood Estimation |
| NB | Naive Bayes |
| OLS | Ordinary Least Squares |
| PR-AUC | Precision–Recall Area Under Curve |
| PSD | Positive Semi-Definite |
| QDA | Quadratic Discriminant Analysis |
| QR | (not an acronym — the $Q$–$R$ matrix factorization) |
| RBF | Radial Basis Function |
| RKHS | Reproducing Kernel Hilbert Space |
| ROC-AUC | Receiver Operating Characteristic Area Under Curve |
| SMOTE | Synthetic Minority Over-sampling TEchnique |
| SVD | Singular Value Decomposition |
| SVM | Support Vector Machine |
| TF-IDF | Term Frequency–Inverse Document Frequency |

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

#### Reading the normal equations in plain English

Four objects, and each one is a shape you can picture.

| Symbol | Shape | Read aloud | What it actually is |
|---|---|---|---|
| $y$ | $n\times 1$ | "y" | The column of true answers — one number per example |
| $X$ | $n\times p$ | "the design matrix" | Your spreadsheet: row $i$ is example $i$, column $j$ is feature $j$ |
| $\beta$ | $p\times 1$ | "beta" | One weight per feature. **This is the unknown you are solving for** |
| $X\beta$ | $n\times 1$ | "X beta" | The predictions those weights would make, all $n$ of them |

So $\lVert y-X\beta\rVert^2$ reads aloud as: *"for each example, subtract the prediction from the truth, square it, and add up all $n$ of them."* The $\arg\min_\beta$ in front means **"give me the weights that make that total smallest"** — the recipe, not the score (§0.3).

The second line, $\nabla_\beta\lVert y-X\beta\rVert^2 = -2X^\top(y-X\beta) = 0$, is the standard find-the-bottom-of-the-bowl move: differentiate with respect to every entry of $\beta$, stack the answers into a vector, and set that vector to zero. At the lowest point of a bowl, the slope in *every* direction is flat simultaneously.

**Now put numbers in it.** Three houses. One feature (size, in arbitrary units) plus an intercept column of ones:

$$X = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3\end{pmatrix},\qquad y = \begin{pmatrix} 2 \\ 3 \\ 5\end{pmatrix}$$

Build the two ingredients:

$$X^\top X = \begin{pmatrix} 3 & 6 \\ 6 & 14\end{pmatrix},\qquad X^\top y = \begin{pmatrix} 10 \\ 23 \end{pmatrix}$$

The top-left $3$ is "how many rows"; the $6$ is $1+2+3$; the $14$ is $1+4+9$. **$X^\top X$ is nothing but a small table of feature-by-feature dot products** — for $p$ features it is $p\times p$ regardless of how many million rows you have. That is why linear regression scales to enormous $n$.

Solving $3\hat\beta_0 + 6\hat\beta_1 = 10$ and $6\hat\beta_0+14\hat\beta_1=23$ gives

$$\hat\beta_0 = \tfrac13,\qquad \hat\beta_1 = \tfrac32$$

Predictions $\hat y = (11/6,\ 10/3,\ 29/6) \approx (1.833,\ 3.333,\ 4.833)$, residuals $r = y - \hat y = (1/6,\ -1/3,\ 1/6)$.

**Check the orthogonality claim with arithmetic.** The residual must be perpendicular to every column of $X$:

- Against the intercept column $(1,1,1)$: $\tfrac16 - \tfrac13 + \tfrac16 = 0$ ✓
- Against the size column $(1,2,3)$: $\tfrac16 - \tfrac23 + \tfrac12 = 0$ ✓

▸ **That is the whole theorem, verified by hand.** "The residual is orthogonal to every predictor" is not an abstraction — it is two sums that come out to exactly zero. And it is *forced*: if the residual had any leftover component along a feature, you could reduce the error by moving $\hat\beta$ a little in that direction, so you weren't at the minimum.

> **Analogy.** You are shining a torch straight down onto a tabletop while holding a ball above it. $y$ is the ball. The tabletop is $\mathrm{col}(X)$ — every prediction the model is *capable* of making. $\hat y$ is the shadow. The residual is the vertical line from ball to shadow, and it is perpendicular to the table for the same reason a plumb line is: **any tilt in it would mean the shadow isn't the closest point on the table.** The normal equations say "make the error vertical," and "normal" here is the geometer's word for perpendicular, not the statistician's word for Gaussian.

#### Unpacking the hat matrix $H$

$H = X(X^\top X)^{-1}X^\top$ is called the hat matrix because **it puts the hat on $y$**: $\hat y = Hy$. That naming is not a joke someone made once; it is the standard term in every regression textbook.

Three properties, decoded:

| Property | Read aloud | What it means |
|---|---|---|
| $H^2 = H$ | "H is idempotent" | Projecting something that is *already* on the table does nothing. Shadow of a shadow is the shadow. |
| $H^\top = H$ | "H is symmetric" | The projection is **orthogonal** (straight down), not oblique (slanted). |
| $\mathrm{tr}(H) = p$ | "the trace of H is p" | The table is $p$-dimensional. You spent exactly $p$ degrees of freedom. |

In the numeric example above, $p=2$: you used two numbers to explain three, so exactly one degree of freedom is left over for the residual. Fit three points with three free parameters and $H = I$, the residual is exactly zero, and you have learned nothing about the world.

▸ **$\mathrm{tr}(H)$ is the honest definition of "how complex is this model."** It counts dimensions of prediction space you have claimed. Ridge (§22.2) will turn this integer into a smooth dial, and that single move — from counting parameters to measuring effective parameters — is the bridge from classical statistics to everything in Chapter 2.

#### What breaks when $X^\top X$ is singular

Record height twice: once in centimetres and once in inches. Column 2 is exactly $2.54\times$ column 1 (or in round numbers, take $x_1 = (1,2,3)$ and $x_2 = (2,4,6)$).

Now $\beta = (1, 0)$, $\beta = (0, 0.5)$, and $\beta = (3, -1)$ all produce **identical predictions** $(1,2,3)$. Check the third: $3(1,2,3) - 1(2,4,6) = (1,2,3)$ ✓.

- The *predictions* are perfectly well-determined.
- The *coefficients* are completely undetermined — an infinite line of solutions.
- $X^\top X$ has a zero eigenvalue, so $(X^\top X)^{-1}$ does not exist, and `numpy.linalg.inv` will either raise or return catastrophic garbage.

▸ **Collinearity does not hurt prediction; it destroys interpretation.** If someone reports "controlling for inches, centimetres have a negative effect," this is what happened. Ridge (§22.2) fixes it by adding $\lambda I$ to the diagonal, which makes the matrix invertible again and picks the smallest-norm member of that infinite family.

**And the condition-number warning, made concrete.** If $X$ has condition number $\kappa(X) = 10^4$ — mild, entirely ordinary for real data — then $\kappa(X^\top X) = 10^8$. Double-precision floats carry about 16 significant digits, so forming $X^\top X$ throws away eight of them before you have solved anything. QR and SVD work with $X$ directly and never pay the squaring. **This is why `sklearn`'s `LinearRegression` calls a least-squares solver, not a matrix inverse.**

#### Examples and non-examples: what counts as "linear regression"

**✅  examples**

| Example | Why it qualifies |
|---|---|
| $\hat y = \beta_0 + \beta_1(\text{sqft}) + \beta_2(\text{bedrooms})$ | Linear in $\beta$, fit by least squares |
| **Polynomial regression** $\hat y = \beta_0+\beta_1x+\beta_2x^2+\beta_3x^3$ | Curved in $x$, but perfectly linear in $\beta$. Just put $x^2, x^3$ in as columns |
| Regression on a Fourier basis, $\hat y = \sum_k \beta_k\sin(kx)$ | Same story — the basis functions are fixed, the weights are free |
| One-hot encoding a categorical variable | Each level becomes a column; the coefficients are group means |
| A neural network's final layer, holding the body frozen | Fixed features in, linear weights out. Literally OLS on learned features |

**❌ Near-misses — look like linear regression, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| $\hat y = \beta_0 e^{\beta_1 x}$ | Nonlinear **in the parameters** — no normal equations exist | Nonlinear regression; needs iterative optimization |
| Logistic regression | The response is passed through $\sigma(\cdot)$; no closed form | A generalized linear model (§22.4) |
| Fitting a line by minimizing $\sum_i \lvert y_i - \hat y_i\rvert$ | Linear model, but the *loss* is not squared error | Least absolute deviations / quantile regression |
| $\hat y = mx + c$, called "a linear function" | $f(2x)\ne 2f(x)$ once $c\ne 0$ | **Affine** (§0.12b) — and everybody calls it linear anyway |
| Correlation between $x$ and $y$ | A summary statistic, not a fitted model | Pearson's $r$. It happens to equal $\hat\beta_1$ when both are standardized |

▸ **The boundary:** linear regression is linear **in the parameters**, never necessarily in the inputs. Ask "if I doubled $\beta$, would the prediction double?" If yes, the normal equations apply, no matter how curved the picture looks.

> **Common misconception.** *"Linear regression can only fit straight lines."* It fits a straight line **in whatever feature space you hand it**. Give it the columns $(x,\ x^2,\ x^3,\ \sin x,\ \log x)$ and it will happily fit a wiggling curve, in closed form, in one shot. This belief is tempting because the textbook picture is always a scatter plot with a line through it — but the line lives in feature space, and you choose that space. Kernel methods (§22.5) are the extreme version of this observation: take the feature space to be infinite-dimensional and never build it.

> **Common misconception.** *"$R^2 = 0.9$ means the model is good."* $R^2$ is $1 - \mathrm{RSS}/\mathrm{TSS}$, and it **never decreases** when you add a column, even a column of pure noise. Add $n-1$ random columns to $n$ data points and $R^2$ hits exactly 1. The belief is tempting because $R^2$ is bounded in $[0,1]$ and feels like a percentage grade. It is a measure of *fit to the data you already have*, which is precisely the quantity Chapter 2 warns you not to trust.

> **Where this came from.** Least squares has one of the ugliest priority disputes in mathematics. **Adrien-Marie Legendre** published the method — and coined the name *méthode des moindres carrés* — in 1805, in an appendix to a book on determining the orbits of comets. **Carl Friedrich Gauss** published it in 1809 in *Theoria Motus*, and claimed he had been using it since 1795, at the age of eighteen. Legendre was furious, and the exchange between them stayed sour for years. Gauss almost certainly *did* have it first (he had used it in 1801 to predict where the newly discovered and then-lost asteroid Ceres would reappear, and astronomers found it exactly where he said it would be) — but he had not published, and the modern norm that publication establishes priority was not yet settled. Gauss's contribution was in any case the deeper one: he supplied the *probabilistic* justification, showing that least squares is the right thing to do if errors follow what we now call the Gaussian distribution.

> **The story behind the word "regression."** It comes from a phenomenon, not a method. **Francis Galton**, in the 1870s–1880s, measured sweet pea seeds and then human heights and noticed that unusually tall parents had children who were tall but *less* tall — closer to the average. He called this "regression towards mediocrity," publishing on hereditary stature in 1886. "Regression" therefore originally named the *finding* — that extremes pull back toward the mean — and only later became the name for the *technique* used to detect it. **Regression to the mean is still one of the most under-appreciated confounders in applied work:** the students who scored worst on a test improve after tutoring whether or not the tutoring did anything, and so do the hospitals that had the worst quarter.

### Gauss–Markov

Under (i) linearity, (ii) $\mathbb{E}[\varepsilon]=0$, (iii) homoskedasticity and no autocorrelation ($\mathrm{Var}(\varepsilon)=\sigma^2I$), OLS is the **B**est **L**inear **U**nbiased **E**stimator — minimum variance among unbiased linear estimators. Normality is *not* required for this; it is required for exact $t$ and $F$ tests.

▸ **The word doing the work is "unbiased."** Ridge is biased and often has lower total error. Gauss–Markov is not a claim that OLS is best, only that it is best in a class you may not want to restrict yourself to.

#### Gauss–Markov, decoded assumption by assumption

Every word of "**B**est **L**inear **U**nbiased **E**stimator" is load-bearing, and three of the four are restrictions rather than compliments.

| Word | Read it as | The restriction it hides |
|---|---|---|
| **Best** | "smallest variance" | Only variance. Not smallest *error* — see below |
| **Linear** | "of the form $\hat\beta = My$ for some fixed matrix $M$" | Rules out anything that inspects $y$ before deciding how to combine it |
| **Unbiased** | $\mathbb{E}[\hat\beta] = \beta$ | Right *on average across imaginary repeated datasets*. Says nothing about your one dataset |
| **Estimator** | "a recipe for turning data into a number" | — |

And the three conditions:

- **(i) Linearity.** The truth really is $y = X\beta + \varepsilon$. If the world is quadratic and you fit a line, no theorem saves you.
- **(ii) $\mathbb{E}[\varepsilon]=0$.** The noise has no systematic offset. Read aloud: *"the average error is zero."* If your scale is 2 kg heavy, this fails, and the bias lands entirely in the intercept.
- **(iii) $\mathrm{Var}(\varepsilon) = \sigma^2 I$.** Two claims in one symbol. The diagonal being **constant** is **homoskedasticity** (from Greek *homos*, same, and *skedasis*, dispersion) — every example is measured with the same precision. The off-diagonals being **zero** means no autocorrelation — knowing example 3's error tells you nothing about example 4's.

**Where (iii) fails in practice, concretely:**

| Situation | Which part fails | Consequence |
|---|---|---|
| Predicting income; rich people's incomes vary more | Constant diagonal (heteroskedasticity) | $\hat\beta$ still unbiased, but **standard errors are wrong** — your p-values lie |
| Daily stock returns | Zero off-diagonals | Effective sample size is far below $n$; confidence intervals far too narrow |
| Repeated measurements on the same patient | Zero off-diagonals | Same problem; needs mixed-effects or clustered errors |

▸ **Note what does *not* break.** Violating (iii) leaves OLS unbiased. It breaks the *uncertainty estimates*, not the point estimate. This is the single most commonly muddled point about Gauss–Markov.

#### Examples and non-examples: unbiased estimators

**✅  unbiased**

| Example | Why |
|---|---|
| $\hat\beta_{\text{OLS}}$ under assumptions (i)–(ii) | $\mathbb{E}[\hat\beta] = \beta$ exactly |
| The sample mean $\bar x = \frac1n\sum_i x_i$ | $\mathbb{E}[\bar x] = \mu$ for any distribution with a mean |
| Sample variance with the $\frac{1}{n-1}$ divisor | The $-1$ exists precisely to remove the bias |

**❌ Near-misses — biased, and often better anyway**

| Looks fine | Why it's biased | Why you might want it |
|---|---|---|
| Ridge, $\hat\beta = (X^\top X+\lambda I)^{-1}X^\top y$ | Systematically shrunk toward zero | Lower **total** error: bias² + variance is smaller |
| Sample variance with $\frac1n$ | Underestimates $\sigma^2$ by a factor $\frac{n-1}{n}$ | It's the maximum-likelihood estimate |
| The James–Stein estimator | Deliberately shrinks toward a common point | **Strictly beats** the sample mean in 3+ dimensions, for every true $\mu$ |
| Any early-stopped gradient descent fit | Stopping early = implicit shrinkage | Standard practice in all of deep learning |

▸ **The boundary:** unbiased means *centred on the truth across hypothetical repetitions of the whole experiment*. It says nothing about being *close* to the truth in the one experiment you ran. An estimator that is always off by exactly $+0.01$ beats one that is unbiased but scattered by $\pm 50$, and mean squared error — $\text{bias}^2 + \text{variance}$ — is the quantity that notices.

> **Common misconception.** *"Gauss–Markov proves OLS is the best possible estimator."* It proves OLS is best inside a fence, and the fence has two sides: **linear** and **unbiased**. Step over either side and better estimators exist and are used constantly — ridge, LASSO, James–Stein, gradient boosting, neural networks. The misconception is tempting because "BLUE" sounds like an unqualified award, and because introductory courses present the theorem before presenting anything outside its scope.

> **Where this came from.** Gauss proved the result in the early 1820s, in *Theoria combinationis observationum erroribus minimis obnoxiae*, and notably he did **not** assume normally distributed errors — his proof needs only the variance conditions, which is exactly why the theorem is so widely applicable. **Andrey Markov's** name became attached much later. Accounts of exactly why differ; the usual telling is that Markov restated the result in the 1910s, and that Jerzy Neyman, writing in the 1930s, drew attention to it and credited Markov — apparently without realising Gauss had done it a century earlier. The double name stuck. Markov, incidentally, is far better known for Markov chains, which he developed partly to settle an argument about whether the law of large numbers requires independence — he demonstrated it does not, using letter sequences from Pushkin's *Eugene Onegin* as his data.

---

## 22.2 Ridge regression

$$\hat\beta_{\text{ridge}} = \arg\min_\beta\ \|y-X\beta\|^2 + \lambda\|\beta\|^2 \quad\Longrightarrow\quad \hat\beta = (X^\top X+\lambda I)^{-1}X^\top y$$

#### Reading the ridge objective in plain English

Two terms, pulling in opposite directions, with $\lambda$ as the referee.

| Piece | Read aloud | What it wants |
|---|---|---|
| $\lVert y - X\beta\rVert^2$ | "squared error" | **Fit the data.** Push $\beta$ wherever it takes to match $y$ |
| $\lambda\lVert\beta\rVert^2$ | "lambda times the squared length of beta" | **Stay small.** Push $\beta$ toward the origin |
| $\lambda$ | "lambda" | Who wins. $\lambda=0$: pure OLS. $\lambda\to\infty$: $\hat\beta\to 0$ |

$\lVert\beta\rVert^2$ is $\beta_1^2+\beta_2^2+\dots+\beta_p^2$ — the squared length of the coefficient vector. Nothing more exotic than Pythagoras.

> **Analogy.** You're tuning a graphic equalizer to match a reference track. Left to yourself you would slam sliders to extremes to chase every detail, including the hiss. Ridge is a spring attached to every slider pulling it back toward flat. A slider only travels far if the *evidence* is strong enough to stretch the spring. $\lambda$ is the stiffness of the springs.

**And the algebra, in one sentence:** $(X^\top X + \lambda I)^{-1}$ is the OLS solution with $\lambda$ added to the diagonal. Adding a positive number to the diagonal of a symmetric matrix adds $\lambda$ to every eigenvalue, so a matrix that was singular (some eigenvalue exactly 0) becomes invertible (that eigenvalue is now $\lambda > 0$). ▸ **Ridge is invertible even when $p > n$ and OLS has no unique answer at all.** That is not a side benefit; for genomics or text data it is the entire reason to reach for it.

**Now run the numbers on the three-house example from §22.1.** Recall $X^\top X = \begin{pmatrix}3&6\\6&14\end{pmatrix}$, $X^\top y = \begin{pmatrix}10\\23\end{pmatrix}$, and OLS gave $\hat\beta = (0.333,\ 1.500)$.

| $\lambda$ | $X^\top X + \lambda I$ | $\hat\beta_{\text{ridge}}$ | Slope, relative to OLS |
|---|---|---|---|
| $0$ | $\begin{pmatrix}3&6\\6&14\end{pmatrix}$ | $(0.333,\ 1.500)$ | 100% |
| $1$ | $\begin{pmatrix}4&6\\6&15\end{pmatrix}$ | $(0.500,\ 1.333)$ | 89% |
| $10$ | $\begin{pmatrix}13&6\\6&24\end{pmatrix}$ | $(0.370,\ 0.866)$ | 58% |
| $100$ | $\begin{pmatrix}103&6\\6&114\end{pmatrix}$ | $(0.086,\ 0.197)$ | 13% |

▸ **Watch the slope decay: 1.50 → 1.33 → 0.87 → 0.20.** Ridge never reaches exactly zero, no matter how large $\lambda$ gets — it approaches it asymptotically. Hold on to that, because §22.3 is entirely about the fact that LASSO *does* reach zero, and why.

*(This example penalizes the intercept, which you should never do in practice — see the end of this section. It is shown that way here only so the arithmetic stays small enough to check by hand.)*

### The SVD reading — the most illuminating one

Let $X = U\Sigma V^\top$ with singular values $\sigma_j$. Then

▸ $$\hat y_{\text{ridge}} = \sum_{j=1}^{p}u_j\underbrace{\frac{\sigma_j^2}{\sigma_j^2+\lambda}}_{\text{shrinkage factor}}u_j^\top y$$

versus OLS's $\hat y = \sum_j u_ju_j^\top y$ (all factors 1).

▸ **Ridge shrinks each principal direction by $\frac{\sigma_j^2}{\sigma_j^2+\lambda}$ — barely at all for high-variance directions ($\sigma_j^2\gg\lambda$) and almost to zero for low-variance ones.** Directions the data barely constrains are exactly the ones estimated with the most noise, so this is precisely the right thing to do. This is the same statement as Ch. 7 §7.5's eigenbasis shrinkage, and it is the best one-line justification of $\ell_2$ regularization in existence.

**Effective degrees of freedom:** $\mathrm{df}(\lambda)=\sum_j\frac{\sigma_j^2}{\sigma_j^2+\lambda}$, ranging from $p$ (at $\lambda=0$) to 0. **A continuous model-complexity dial** — and the right $x$-axis for a bias–variance plot.

**Always standardize before ridge**, and never penalize the intercept.

#### Unpacking the shrinkage factor $\frac{\sigma_j^2}{\sigma_j^2+\lambda}$

This is the most useful single fraction in regularization, and it takes ninety seconds to make it obvious.

$\sigma_j$ is the $j$-th **singular value** of $X$ (§1.1.3): *how much the data actually varies along direction $j$*. Large $\sigma_j$ = the data spreads out a lot in that direction = you have plenty of evidence about it. Small $\sigma_j$ = the data barely moves that way = whatever you estimate there is mostly noise.

Now feed numbers into $\frac{\sigma_j^2}{\sigma_j^2+\lambda}$, with $\lambda = 1$:

| $\sigma_j$ | $\sigma_j^2$ | Shrinkage factor | Read aloud |
|---|---|---|---|
| $10$ | $100$ | $100/101 = 0.990$ | "Keep 99% of it. The data is certain here." |
| $3$ | $9$ | $9/10 = 0.900$ | "Keep 90%." |
| $1$ | $1$ | $1/2 = 0.500$ | "Halve it. Evidence and prior are exactly balanced." |
| $0.3$ | $0.09$ | $0.09/1.09 = 0.083$ | "Keep 8%. Mostly noise." |
| $0.1$ | $0.01$ | $0.01/1.01 = 0.0099$ | "Delete it. This direction is unmeasured." |

▸ **$\lambda$ is a threshold expressed in units of squared singular value.** Directions with $\sigma_j^2 \gg \lambda$ pass through untouched; directions with $\sigma_j^2 \ll \lambda$ are erased; the crossover is exactly at $\sigma_j^2 = \lambda$, where you keep half. **Choosing $\lambda$ is choosing how much evidence a direction must supply before you believe it.**

> **Analogy.** A newsroom with a rule: run the story only if enough independent sources confirm it. A rumour from one anonymous tip gets spiked; a fact confirmed by a hundred witnesses runs verbatim; something with a handful of sources runs, hedged. $\sigma_j^2$ is the number of sources, $\lambda$ is the editor's threshold, and the shrinkage factor is how much of the claim survives to print.

**Effective degrees of freedom, with numbers.** Suppose $p=3$ and $X$ has singular values $\sigma = (10,\ 1,\ 0.1)$. At $\lambda = 1$:

$$\mathrm{df}(1) = 0.990 + 0.500 + 0.0099 = 1.50$$

▸ **You fit three parameters and spent one and a half.** OLS would have spent exactly 3 ($\mathrm{df}(0) = 1 + 1 + 1$). This is the moment "model complexity" stops being an integer you can count and becomes a real number you can dial — which is why $\mathrm{df}(\lambda)$, not $\lambda$, is the correct $x$-axis for a bias–variance plot. Plotting against $\lambda$ compresses everything interesting into a corner of the graph.

#### Examples and non-examples: is that $\ell_2$ regularization?

**✅  the same mathematical object**

| Example | Why it's the same thing |
|---|---|
| Ridge regression, $+\lambda\lVert\beta\rVert^2$ | The definition |
| **Weight decay** in a neural net | Adds $\lambda\lVert W\rVert^2$ to the loss; identical gradient $2\lambda W$ |
| **Tikhonov regularization** in inverse problems | The same formula, from a different literature |
| **MAP estimation with a Gaussian prior** $\beta\sim\mathcal{N}(0,\tau^2 I)$ | The negative log-prior *is* $\lVert\beta\rVert^2/2\tau^2$; $\lambda = \sigma^2/\tau^2$ |
| Adding fake rows $\sqrt{\lambda}\,I$ to $X$ with zero targets | Algebraically identical to ridge. Sometimes literally implemented this way |

**❌ Near-misses — regularization, but not $\ell_2$ shrinkage**

| Looks like it | Why it differs | What it actually is |
|---|---|---|
| **AdamW's** decoupled weight decay | Subtracts $\lambda\theta$ from the weights *directly*, outside the adaptive scaling | Decoupled decay. Provably **not** the same as adding $\lambda\lVert\theta\rVert^2$ to the loss under Adam |
| Dropout | Randomly zeroes activations, not weights | Stochastic ensembling; only $\ell_2$-like in restricted linear cases |
| Early stopping | Never writes down a penalty at all | Implicit shrinkage — for linear models it is *approximately* ridge, with $\lambda \approx 1/(\eta t)$ |
| Batch normalization | Rescales activations | Not regularization in the penalty sense, though it has regularizing side effects |
| Clipping $\lVert\beta\rVert \le c$ | A hard constraint, not a soft penalty | The **constrained** form. Equivalent for *some* $\lambda$, but you can't name which in advance |

▸ **The boundary:** $\ell_2$ regularization means a term proportional to the **squared** length of the parameters is added to the objective, so its gradient is proportional to the parameters themselves. That last clause is the tell: a penalty whose gradient shrinks to zero as $\beta\to 0$ can never push $\beta$ all the way to zero.

> **Common misconception.** *"Ridge does feature selection by driving useless coefficients to zero."* It never sets anything to exactly zero. The gradient of $\lambda\beta_j^2$ is $2\lambda\beta_j$, which vanishes as $\beta_j$ approaches zero — the push weakens exactly when you'd need it to finish the job. You get a hundred tiny coefficients, not ten large ones and ninety zeros. The misconception is tempting because ridge coefficients *print* as things like `3.2e-08`, which looks like zero, and because ridge and LASSO are always taught in the same breath.

> **Common misconception.** *"Standardizing before ridge is just tidiness."* It changes the answer. The penalty $\sum_j \beta_j^2$ treats every coefficient identically, but a coefficient's size depends on its feature's units. Measure a distance in metres and $\beta = 2$; switch to kilometres and the same relationship needs $\beta = 2000$, which the penalty now punishes a million times harder. **Un-standardized ridge silently regularizes your small-unit features into oblivion.** The intercept is left unpenalized for the same reason: shrinking it toward zero would mean "when all features are at their average, predict zero," which is a claim about your units, not about the world.

> **Where this came from.** Ridge regression was published in 1970 by **Arthur Hoerl and Robert Kennard** in *Technometrics*, under the title "Ridge Regression: Biased Estimation for Nonorthogonal Problems." The word "ridge" is inherited from Hoerl's earlier work on **ridge analysis** in response-surface methodology, where you trace the crest of a response surface — the picture the name evokes is the ridgeline of contours, not a mountain. The paper had a hostile reception: deliberately introducing bias offended a discipline that had spent fifty years proving things about unbiased estimators, and reviewers and discussants pushed back hard. It is now one of the most-used ideas in applied statistics.

> **The story behind the other name.** The same formula had already appeared in the Soviet Union, from **Andrey Tikhonov**, who was attacking a completely different problem: **ill-posed inverse problems**, such as reconstructing a physical cause from noisy measurements of its effect. His work on regularizing such problems dates to the 1940s–1960s. Neither line of work knew of the other for years, and today the identical formula carries three names in three fields — *ridge regression* in statistics, *Tikhonov regularization* in numerical analysis and geophysics, and *weight decay* in neural networks. ▸ **When the same equation is independently discovered three times, that is evidence it is a natural object rather than a trick.**

---

## 22.3 LASSO and elastic net

$$\hat\beta_{\text{lasso}}=\arg\min_\beta\ \tfrac{1}{2}\|y-X\beta\|^2+\lambda\|\beta\|_1$$

**No closed form** (the $\ell_1$ term is non-differentiable at 0), but for orthonormal $X$ the solution is exactly the **soft-threshold**:

▸ $$\hat\beta_j = \mathcal{S}_\lambda(\hat\beta_j^{\text{OLS}}) = \mathrm{sign}(\hat\beta_j^{\text{OLS}})\big(|\hat\beta_j^{\text{OLS}}|-\lambda\big)_+$$

**Derivation of the sparsity.** The subgradient of $\lambda|\beta_j|$ at $\beta_j=0$ is the *interval* $[-\lambda,\lambda]$. Optimality requires $0\in -x_j^\top(y-X\beta)+\lambda\partial|\beta_j|$, so $\beta_j=0$ is optimal whenever $|x_j^\top r|\le\lambda$.
▸ **Any coordinate whose correlation with the residual is below $\lambda$ is set to exactly zero.** Ridge's penalty has gradient $2\lambda\beta_j\to0$ as $\beta_j\to0$, so it never produces exact zeros. **That is the entire difference between the two.**

#### Reading the soft-threshold operator in plain English

$$\mathcal{S}_\lambda(z) = \mathrm{sign}(z)\big(\lvert z\rvert - \lambda\big)_+$$

Three pieces, read right to left:

| Piece | Read aloud | Job |
|---|---|---|
| $\lvert z\rvert$ | "the absolute value of z" | Forget the sign; how big is it? |
| $(\ \cdot\ - \lambda)_+$ | "minus lambda, positive part" | Subtract $\lambda$. **If that goes negative, return 0 instead.** This is a ReLU |
| $\mathrm{sign}(z)$ | "sign of z" | Put the original sign back on |

In one English sentence: ▸ **"Move the coefficient $\lambda$ closer to zero — and if it would cross zero on the way, park it exactly at zero."**

In code, it is three lines:

```python
def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)
```

**Now compare, with numbers.** Take an orthonormal $X$ so both methods have closed forms, and set $\lambda = 1$. For ridge on orthonormal $X$, $\hat\beta_j = \hat\beta_j^{\text{OLS}}/(1+\lambda)$, i.e. simply halved.

| $\hat\beta_j^{\text{OLS}}$ | LASSO $\mathcal{S}_1(\cdot)$ | Ridge $(\cdot)/2$ | What happened |
|---|---|---|---|
| $3.0$ | $2.0$ | $1.5$ | Both shrink; LASSO takes a fixed $1.0$, ridge takes 50% |
| $1.2$ | $0.2$ | $0.6$ | LASSO nearly kills it |
| $0.7$ | $\mathbf{0}$ | $0.35$ | **LASSO deletes the feature. Ridge keeps a small one** |
| $-2.5$ | $-1.5$ | $-1.25$ | Sign preserved, magnitude reduced by exactly 1 |
| $-0.4$ | $\mathbf{0}$ | $-0.2$ | Deleted |
| $0.0$ | $0.0$ | $0.0$ | Nothing to do |

▸ **Read down the two shrinkage columns and the entire ridge/LASSO debate is visible.** LASSO subtracts a **constant amount** from everything, so small coefficients hit zero and stop. Ridge multiplies by a **constant fraction**, so a small coefficient just becomes a smaller one — halving a number can never reach zero.

> **Analogy.** Two tax regimes. Ridge is a flat 50% tax: earn £1, keep 50p; earn £1M, keep £500k. Nobody is ever taxed out of existence. LASSO is a fixed £1,000 fee to participate: if you earn £5,000 you pay it and keep £4,000; if you earn £700 you don't bother showing up at all. **A fixed fee eliminates small participants; a percentage never does.** That is the whole of sparsity.

#### Why the corner produces zeros — the subgradient, without the jargon

$\lvert\beta\rvert$ has a sharp corner at $\beta=0$. Approach from the right and the slope is $+1$; from the left, $-1$. At exactly zero there is no single slope — so mathematics does the obvious thing and admits the **whole interval** $[-1,+1]$ as the set of valid slopes there. That set is the **subdifferential**, written $\partial\lvert\beta\rvert$.

Optimality at a corner then means: *is zero among the available slopes?* Rather than the single equation "gradient $=0$," you get the condition "$0$ belongs to the set," which is a much easier condition to satisfy — and satisfying it *is* what pins $\beta_j$ at exactly zero.

**Concretely.** Suppose feature $j$ has correlation with the residual $x_j^\top r = 0.6$, and $\lambda = 1$. The condition for $\beta_j = 0$ to be optimal is $\lvert x_j^\top r\rvert \le \lambda$, and $0.6 \le 1$. ✓ So $\beta_j$ stays at exactly zero — **nudging it in either direction increases the objective, because the penalty's slope of $1$ outweighs the fit's payoff of $0.6$.** The coefficient is not "close to zero"; it is *stuck* at zero, held there by a corner.

▸ **Compare with ridge at the same point.** Ridge's penalty has slope $2\lambda\beta_j = 0$ at $\beta_j=0$. So *any* correlation, however small, tips the balance and moves the coefficient off zero. **A smooth penalty can never hold a coefficient still. A kinked one can.** Everything about $\ell_1$ follows from that corner.

> **Analogy for the kink.** Ridge is a rubber band: pull an inch, feel a little tension; pull ten inches, feel ten times as much. Near its rest point it barely resists at all. LASSO is static friction: it takes a fixed force to make the block move *at all*, and below that threshold nothing happens whatsoever. Sparsity is static friction; smooth shrinkage is a spring.

**Geometric picture:** the $\ell_1$ ball is a cross-polytope with vertices on the axes. An elliptical level set of the loss expanding until it touches the ball will, generically, touch at a vertex — where coordinates are zero.

**Elastic net:** $\lambda\left(\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|^2\right)$. Needed when predictors are correlated: LASSO arbitrarily picks one of a correlated group and zeros the rest, which is unstable across resamples; the ridge term induces a **grouping effect** so correlated predictors get similar coefficients.

**Solved by:** coordinate descent (cycle over $j$, apply soft-thresholding — fast and the standard) or LARS (traces the whole regularization path).

#### Unpacking the geometric picture

"The $\ell_1$ ball is a cross-polytope with vertices on the axes" sounds worse than it is. In two dimensions:

- The $\ell_2$ ball $\{\beta : \beta_1^2+\beta_2^2 \le c\}$ is a **circle**. Perfectly smooth everywhere.
- The $\ell_1$ ball $\{\beta : \lvert\beta_1\rvert+\lvert\beta_2\rvert \le c\}$ is a **diamond** — a square rotated 45°, with its four corners sitting on the axes at $(\pm c, 0)$ and $(0,\pm c)$.

A corner on an axis means one coordinate is exactly zero. That is the whole geometric content.

Now the picture. The constrained view of both methods is "minimize squared error, subject to staying inside the ball." The squared-error contours are **ellipses** centred at $\hat\beta^{\text{OLS}}$. Inflate the ellipse from the OLS solution outward until it first kisses the ball; that touch point is the answer.

| Ball | Where the ellipse touches | Consequence |
|---|---|---|
| Circle ($\ell_2$) | Some point on a smooth curve. Every point looks the same | Generically **no** coordinate is zero |
| Diamond ($\ell_1$) | Very often a **corner**, because corners stick out | A coordinate is **exactly** zero |

▸ **Corners catch things.** A pointy set has a disproportionate chance of being touched first at a point, and its points lie on the axes. In $p$ dimensions the cross-polytope has $2p$ vertices, $2^p$ flat faces, and an enormous amount of edge and corner structure at every intermediate level of sparsity — which is why LASSO can produce any number of zeros, not just all-or-nothing.

#### Examples and non-examples: sparsity

**✅  sparse**

| Example | Why it qualifies |
|---|---|
| $\hat\beta = (0,\ 2.1,\ 0,\ 0,\ -0.7)$ from LASSO | Three coefficients are **exactly** 0.0, bit-for-bit |
| A pruned network with weights removed from the graph | The weights are absent, not small |
| A ReLU's output, roughly half zeros | Exact zeros produced by a kink, same mechanism |
| An SVM's dual $\alpha$ vector | Exactly zero for every non-support-vector (§22.5) |

**❌ Near-misses — look sparse, aren't**

| Looks sparse | Why it isn't | What it actually is |
|---|---|---|
| Ridge coefficients like $(3\!\times\!10^{-9},\ 2.1,\ \dots)$ | It's $10^{-9}$, not $0$. Every feature is still in the model | **Shrunk**, not selected |
| Coefficients thresholded by hand after fitting | The fit never knew about the threshold; the survivors weren't re-estimated | Hard thresholding — a heuristic, not an optimum |
| A dense embedding with mostly small values | No zeros at all | A **dense** vector with heavy-tailed magnitudes |
| Dropout masking activations to zero | Different zeros every step; nothing is removed | Stochastic regularization |
| A sparse *matrix format* (CSR) holding a dense matrix | A storage format, not a property of the numbers | An inefficient storage choice |

▸ **The boundary:** sparsity means **exact zeros produced by the optimization itself**, so the feature can be deleted from the model without changing a single prediction. Small is not zero, and post-hoc thresholding is not selection.

> **Common misconception.** *"LASSO finds the true set of relevant features."* It finds *a* set that predicts well. With two nearly identical predictors — say height in centimetres and height in inches, or two co-expressed genes — LASSO keeps **one, essentially arbitrarily**, and zeroes the other. Refit on a bootstrap resample and it may well swap them. The misconception is tempting because a coefficient of exactly zero *feels* like a verdict of irrelevance, while it is really a verdict of redundancy. If you need the stable answer, use the elastic net, or run LASSO on many resamples and count how often each feature survives (stability selection).

> **Common misconception.** *"LASSO is just ridge that's better at feature selection."* They solve different problems and fail in different ways. LASSO can select **at most $n$** features — a hard structural limit, painful when $p = 20{,}000$ genes and $n = 200$ patients. LASSO is also unstable under correlated predictors. Ridge has neither limitation but never selects. The elastic net exists because in $p \gg n$ problems you routinely need both properties at once.

> **What breaks if you don't standardize.** The penalty $\lambda\sum_j\lvert\beta_j\rvert$ compares coefficients directly, so it compares *units*. A feature measured in millimetres needs a coefficient a thousand times smaller than the same feature in metres, and so pays a thousand times less penalty — it will survive selection purely for being measured in large units. **Un-standardized LASSO does feature selection on your unit choices.** Every serious implementation standardizes internally by default; know that yours does, and know what it does with the intercept.

> **Where this came from.** **Robert Tibshirani** introduced the LASSO in 1996 in the *Journal of the Royal Statistical Society*, and has said it was inspired by **Leo Breiman's** non-negative garrote (1995), which shrinks OLS coefficients by non-negative multipliers and can zero them out. Tibshirani's move was to put the constraint directly into the fitting problem rather than applying it afterwards. Related ideas arrived independently from other directions: an $\ell_1$ penalty was used in **geophysics in the mid-1980s** to recover sparse spike trains from band-limited seismic data, and **basis pursuit** (Chen, Donoho and Saunders, late 1990s) arrived from signal processing. The whole area then exploded in the 2000s under the name **compressed sensing**, on the strength of a startling result: under the right conditions, minimizing $\lVert\beta\rVert_1$ recovers the *exact* sparse signal, not an approximation of it.

> **Where the algorithms came from.** For years the LASSO was considered expensive to fit. **LARS** (Efron, Hastie, Johnstone and Tibshirani, 2004) changed that by computing the entire regularization path — every solution for every $\lambda$ — for roughly the cost of one OLS fit, exploiting the fact that the path is piecewise linear in $\lambda$. Then, in the late 2000s, **coordinate descent** turned out to be even faster in practice: cycle through the coefficients one at a time, applying the soft-threshold formula above, and repeat. It is almost embarrassingly simple, and it is what the widely used `glmnet` package does. ▸ **A twenty-year gap between "this is a good idea" and "this is cheap enough to be the default" is entirely typical in statistics and machine learning.**

> **The story behind the elastic net.** **Hui Zou and Trevor Hastie** introduced it in 2005, explicitly to fix two named LASSO failures: the "at most $n$ features" ceiling in $p\gg n$ settings, and the arbitrary tie-breaking among correlated predictors. The name is the image: a stretchable net that keeps all the big fish and lets small ones through, where the ridge term is the stretch that keeps a correlated group together rather than snapping down on one member.

---

## 22.4 Logistic regression

$$p(y=1\mid x)=\sigma(\beta^\top x),\qquad \mathcal{L}=-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]$$

#### Reading logistic regression in plain English

Two formulas, and each contains exactly one idea.

**Formula 1 — the model.** $p(y=1\mid x) = \sigma(\beta^\top x)$ reads: *"the probability that this example is a positive, **given** its features, equals sigmoid of the weighted sum of its features."* The bar $\mid$ means "given" (§0.9, Trap 5), not division.

The problem it solves: $\beta^\top x$ is a linear score and can be any real number — $-40$, $0.3$, $10^6$. Probabilities must live in $(0,1)$. The **sigmoid** is the squasher:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

| $z = \beta^\top x$ | $e^{-z}$ | $\sigma(z)$ | In words |
|---|---|---|---|
| $-3$ | $20.1$ | $0.047$ | "Almost certainly negative" |
| $-1$ | $2.72$ | $0.269$ | "Probably negative" |
| $0$ | $1$ | $0.500$ | "No idea" |
| $+2$ | $0.135$ | $0.881$ | "Probably positive" |
| $+5$ | $0.0067$ | $0.993$ | "Almost certainly positive" |

▸ **Sigmoid never reaches 0 or 1.** It approaches them asymptotically, which turns out to be the source of both its best property (gradients are always defined) and its worst pathology (perfect separation, below).

*"Sigmoid" simply means S-shaped* — Greek *sigma* plus *eidos*, "form." The name describes the picture, not the mathematics.

**Formula 2 — the loss.** $\mathcal{L} = -\sum_i[y_i\log p_i + (1-y_i)\log(1-p_i)]$ looks like two terms, but since $y_i$ is $0$ or $1$, **exactly one of them is alive at a time**:

```python
loss_i = -log(p_i)      if y_i == 1
loss_i = -log(1 - p_i)  if y_i == 0
```

The $y_i$ and $(1-y_i)$ are a switch written as arithmetic — the same trick as the Kronecker delta in §0.6. In one sentence: ▸ **"How surprised were you by the answer that actually occurred?"** Measured in nats, since the log is natural.

| True $y$ | Predicted $p$ | Loss $= -\log(\cdot)$ | Reaction |
|---|---|---|---|
| $1$ | $0.99$ | $0.010$ | "Called it." |
| $1$ | $0.90$ | $0.105$ | "Fine." |
| $1$ | $0.50$ | $0.693$ | "Coin flip." ($\log 2$) |
| $1$ | $0.10$ | $2.303$ | "Badly wrong." |
| $1$ | $0.01$ | $4.605$ | "Confidently, catastrophically wrong." |

▸ **The loss for being confidently wrong grows without bound, while the reward for being confidently right saturates.** Going from $p=0.9$ to $p=0.99$ saves you $0.095$; going from $p=0.1$ to $p=0.01$ costs you $2.30$. **Cross-entropy is far more interested in punishing arrogance than in rewarding certainty**, and that asymmetry is why it produces usable probabilities where squared error does not.

> **Analogy.** A weather forecaster paid by a scoring rule. If she says "90% chance of rain" and it rains, she loses a little; if she says "99%" and it stays dry, she loses a fortune. Under this contract the only way to maximize long-run pay is to state **what you actually believe** — no hedging, no bravado. That property has a name (a *proper scoring rule*), and it is the entire reason cross-entropy, rather than accuracy, is the training objective.

#### Unpacking the gradient $X^\top(p-y)$

Read it aloud: *"design matrix transpose, times prediction-minus-truth."* Then read what it means: **each feature's gradient is that feature's values, weighted by how wrong you were on each example.**

For a single example, $\nabla_\beta\mathcal{L} = (p_i - y_i)\,x_i$, and everything about training is visible in that product:

| $y_i$ | $p_i$ | $p_i - y_i$ | Effect on $\beta$ |
|---|---|---|---|
| $1$ | $0.5$ | $-0.5$ | Push $\beta$ **along** $x_i$ — raise this example's score |
| $1$ | $0.99$ | $-0.01$ | Barely move. Already right |
| $0$ | $0.9$ | $+0.9$ | Push $\beta$ **against** $x_i$, hard. Badly wrong |
| $0$ | $0.01$ | $+0.01$ | Barely move |

**Watch one gradient step.** Take $x = (1, 2)$, $y = 1$, and start at $\beta = (0,0)$, so $p = \sigma(0) = 0.5$.

- Gradient: $(0.5 - 1)\cdot(1,2) = (-0.5,\ -1.0)$
- Step with $\eta = 0.1$: $\beta \leftarrow (0,0) - 0.1(-0.5,-1.0) = (0.05,\ 0.10)$
- New score: $0.05 + 0.20 = 0.25$, so $p = \sigma(0.25) = 0.562$

▸ It moved from $0.50$ toward $1$, which is exactly what it should have done, and *nothing about the update knew it was doing classification*. The identical formula with $p$ replaced by $\hat y$ is linear regression's gradient. **This is the deepest structural fact in the whole chapter, and §22.4's GLM table is its explanation.**

#### What "log-odds" actually means

**Odds** are probability rewritten as a ratio: $\text{odds} = \frac{p}{1-p}$. A probability of $0.8$ is odds of $4$, spoken as "four to one on."

| $p$ | Odds $\frac{p}{1-p}$ | $\log$ odds |
|---|---|---|
| $0.5$ | $1$ | $0$ |
| $0.75$ | $3$ | $1.10$ |
| $0.88$ | $7.33$ | $1.99$ |
| $0.25$ | $0.333$ | $-1.10$ |
| $0.99$ | $99$ | $4.60$ |

Notice the third row: $p = 0.881$ gave log-odds $1.99 \approx 2$ — and $2$ is exactly the $\beta^\top x$ that produced it in the sigmoid table above. That is not a coincidence, it is an identity:

$$\log\frac{p}{1-p} = \beta^\top x$$

▸ **The sigmoid and the log-odds are the same statement read in opposite directions.** Logistic regression is a perfectly ordinary *linear* model — it is linear in the log-odds. All the curvature you see in the fitted probability comes from the change of units, not from the model.

**So what is a coefficient?** $\beta_j = 0.7$ means: *"one extra unit of feature $j$ adds $0.7$ to the log-odds."* Exponentiate to get the **odds ratio**: $e^{0.7} = 2.01$, so **each extra unit roughly doubles the odds.** Doubling the odds is not doubling the probability — from $p=0.1$ (odds $0.111$) it takes you to odds $0.222$, i.e. $p = 0.182$; from $p = 0.5$ it takes you to $p = 0.667$. Same coefficient, different absolute effect, depending on where you start.

> **Analogy.** Log-odds are to probability what decibels are to sound pressure. Both compress a bounded, multiplicative quantity into an unbounded, additive one where equal steps mean equal *ratios*. "+3 dB" always means twice the power, whether you were whispering or shouting; "+0.7 log-odds" always means twice the odds, whether you were at 1% or 40%. The linear model is doing its work on the decibel scale.

### The gradient (derive it once)

Using $\sigma'=\sigma(1-\sigma)$:
▸ $$\nabla_\beta\mathcal{L} = \sum_i (p_i-y_i)x_i = X^\top(p-y)$$

**Identical in form to linear regression's $X^\top(\hat y-y)$.** This is not a coincidence — it holds for every generalized linear model with its canonical link, and it's the cleanest thing to say if asked why cross-entropy is "the right" loss for logistic regression.

**Hessian:** $H = X^\top SX$ with $S=\mathrm{diag}(p_i(1-p_i))$, which is **PSD** ⇒ the loss is convex ⇒ a unique global optimum. Newton's method here is **IRLS** (iteratively reweighted least squares).

▸ **Perfect separation** makes the MLE diverge: $\|\beta\|\to\infty$ pushes probabilities to 0/1 and the loss to 0 without ever attaining it. Any $\ell_2$ penalty fixes it. This is also the setting in which the *implicit bias* of gradient descent selects the max-margin direction (Ch. 31 §31.3).

**Interpretation:** $\beta_j$ is the change in **log-odds** per unit of $x_j$. $e^{\beta_j}$ is the odds ratio.

**Multiclass:** softmax regression, gradient $X^\top(P-Y)$ — the same shape again (Ch. 1 §1.3.4).

#### What breaks under perfect separation

Take the smallest possible example. One feature. Two points: $x = -1$ with $y = 0$, and $x = +1$ with $y = 1$. A single weight $\beta$, no intercept.

| $\beta$ | $p$ at $x{=}{+}1$ | Total loss |
|---|---|---|
| $1$ | $0.731$ | $0.627$ |
| $5$ | $0.993$ | $0.0134$ |
| $10$ | $0.99995$ | $0.000091$ |
| $100$ | $1 - 4\times10^{-44}$ | $\approx 10^{-43}$ |
| $\infty$ | $1$ | $0$ — **never attained** |

▸ **There is no best $\beta$.** Every value is beaten by a larger one, forever. The infimum of the loss is zero and the argmin does not exist, so gradient descent walks off to infinity, slowly, forever, and your coefficients print as `[1.7e+03]` with standard errors that are pure nonsense. This is not a numerical bug — the maximum-likelihood estimate  does not exist.

**Three fixes, in order of how often they're the right one:**

| Fix | What it does | Cost |
|---|---|---|
| Any $\ell_2$ penalty, even $\lambda = 10^{-4}$ | Adds $\lambda\beta^2$, whose growth eventually beats the loss's decay. The optimum becomes finite and unique | Trivially small bias. **This is why `sklearn`'s `LogisticRegression` regularizes by default** |
| Firth's penalized likelihood | A principled bias correction from the statistics literature | Less standard tooling |
| Notice the leaked feature | Perfect separation often means a column encodes the answer | Free, and frequently the real diagnosis |

▸ **In practice, perfect separation on real data is usually a data-leakage alarm, not a modelling problem.** If one feature separates your classes flawlessly, ask what it is before you regularize it away — it is often the label in disguise (a "discharge date" that only exists for discharged patients, an "account closed" flag on churn).

#### Examples and non-examples: is that logistic regression?

**✅  examples**

| Example | Why it qualifies |
|---|---|
| Predicting click / no-click from ad features with a sigmoid output | Linear score → sigmoid → cross-entropy |
| A neural network's **final layer** with one output and BCE loss | Logistic regression on learned features. Literally |
| Softmax regression over 5 classes | The multiclass generalization; identical gradient shape |
| Fitting log-odds by weighted least squares (IRLS) | Newton's method on this exact objective |

**❌ Near-misses — commonly confused with it**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| **Probit** regression, $p = \Phi(\beta^\top x)$ | Uses the Gaussian CDF instead of the sigmoid | Probit. Nearly identical fits; different coefficient scale |
| Linear regression on 0/1 targets | Can predict $-0.3$ or $1.4$; heteroskedastic by construction | The **linear probability model**. Still used in econometrics for its interpretability |
| Fitting a sigmoid with **squared** error | Non-convex, and gradients vanish when badly wrong | A worse objective. The $\sigma'$ factor no longer cancels |
| A model with a sigmoid *hidden* activation | Sigmoid inside, not at the output | An MLP |
| **Platt scaling** an SVM's scores | 1-D logistic regression on a *fitted model's* output | Calibration (Ch. 33), not classification |
| The **logistic curve** in a growth model | Same S-shaped function, no classification | Population dynamics |

▸ **The boundary:** logistic regression is *linear in the log-odds*, trained by maximizing the Bernoulli likelihood. Change the link (probit) or change the loss (squared error) and you have a different, usually worse-behaved, estimator — even though the pictures look identical.

> **Common misconception.** *"Logistic regression is a classification algorithm, so the 'regression' in the name is a mistake."* It is a regression — it regresses the **log-odds** on the features, and produces a number in $(0,1)$, not a class. Classification happens afterwards, when *you* choose a threshold, and that threshold is a business decision, not a modelling one. The default of $0.5$ is a convention with no special authority; for a rare disease or a fraud screen it is usually badly wrong (§22.6).

> **Common misconception.** *"A predicted probability of 0.9 means the model is 90% confident."* It means: **among all examples the model scored 0.9, about 90% should be positives** — if the model is calibrated, which it may not be. Confidence in the everyday sense (how sure the model is about its own estimate) is a different quantity entirely, and requires a distribution *over* the probability, not a point estimate of it. Logistic regression trained by maximum likelihood on plenty of data does tend to be well-calibrated, which is exactly why it survives as the workhorse of credit scoring and clinical risk models; a deep network, as Chapter 33 documents at length, usually is not.

> **Common misconception.** *"Cross-entropy is used because it's differentiable and accuracy isn't."* True but not the reason. Squared error is differentiable too, and it does badly here — pair it with a sigmoid and the gradient acquires a $\sigma'(z) = \sigma(1-\sigma)$ factor that goes to zero exactly when the model is confidently wrong, so the worst-behaved examples produce the smallest updates. Cross-entropy's $\sigma'$ cancels against the log's derivative, leaving the clean $(p-y)x$. ▸ **The right story is that cross-entropy is the negative log-likelihood of the Bernoulli model, and its convenient gradient is a consequence, not a coincidence.**

> **Where this came from.** The logistic function is older than statistics as a discipline. **Pierre-François Verhulst**, a Belgian mathematician, introduced it in the late 1830s and 1840s to model population growth under a resource ceiling — Malthus predicted exponential growth, Verhulst added a term for finite capacity, and the S-curve fell out. He named it *la courbe logistique*. The curve was independently rediscovered in the 1920s by Raymond Pearl and Lowell Reed, who used it to fit the population of the United States.
>
> The *statistical* use came much later and out of rivalry. **Chester Bliss** coined "**probit**" in the 1930s, from "**prob**ability un**it**," for bioassay work on insecticide dosages. **Joseph Berkson**, a physician-statistician at the Mayo Clinic, argued the logistic function was easier to compute and fitted just as well, and in 1944 he coined "**logit**" — a deliberate play on Bliss's word. The naming skirmish was a  and somewhat testy disagreement in the biostatistics literature, and Berkson's word eventually won. **David Cox's** 1958 paper on the regression analysis of binary sequences is generally credited with establishing logistic regression in its modern form.
>
> ▸ **The practical reason logistic beat probit was arithmetic.** In the era before computers, $1/(1+e^{-z})$ could be evaluated from a table of exponentials; $\Phi(z)$, the Gaussian CDF, has no closed form at all. The default classifier of the twenty-first century won its position because it was easier to compute with a slide rule.

### Generalized linear models

$g(\mathbb{E}[y]) = \beta^\top x$ for a link $g$ and an exponential-family response.

| Response | Distribution | Canonical link |
|---|---|---|
| continuous | Gaussian | identity |
| binary | Bernoulli | logit |
| count | Poisson | log |
| count, overdispersed | Neg. binomial | log |
| positive skewed | Gamma | inverse or log |

#### The GLM table, decoded

$g(\mathbb{E}[y]) = \beta^\top x$ reads: *"apply some fixed function $g$ to the expected value of the response, and **that** is what the linear model predicts."* Three moving parts:

| Part | Name | Job |
|---|---|---|
| $\beta^\top x$ | the **linear predictor** | The only thing you actually fit. Unbounded, real-valued |
| $g$ | the **link function** | Translates between the unbounded score and the constrained response |
| The distribution | the **random component** | Says how noise behaves — which determines the loss |

▸ **A GLM is one idea applied five ways: keep the linear score, and change the translator on the way out.** The distribution then tells you what loss to use, because the loss is always the negative log-likelihood of that distribution.

**Reading each row aloud, with a concrete task:**

| Response | Concrete task | Link, in words | What the model predicts |
|---|---|---|---|
| Continuous | House price | Identity: do nothing | $\mathbb{E}[y] = \beta^\top x$. Ordinary least squares |
| Binary | Will this email be opened? | Logit: log-odds | $\log\frac{p}{1-p} = \beta^\top x$. Logistic regression |
| Count | Visits to a page per hour | Log | $\log\mathbb{E}[y] = \beta^\top x$, so $\mathbb{E}[y] = e^{\beta^\top x}$ — **always positive, automatically** |
| Overdispersed count | Comments per post (a few posts go viral) | Log | Same, but with a free variance parameter |
| Positive skewed | Insurance claim size, time-to-failure | Log or inverse | Positive, right-skewed, variance grows with the mean |

**Why the link is not arbitrary.** Counts cannot be negative, but $\beta^\top x$ can. Put a log link in and the constraint is enforced structurally: $e^{\text{anything}} > 0$. ▸ **The link exists to make impossible predictions unrepresentable, rather than merely unlikely.**

**A Poisson coefficient, concretely.** With a log link and $\beta_j = 0.4$: one extra unit of feature $j$ multiplies the expected count by $e^{0.4} = 1.49$ — a 49% increase. Exactly as in logistic regression, ▸ **coefficients under a log-type link are multiplicative, not additive**, so "a one-unit increase adds 0.4 visits" is always the wrong reading.

**And why "canonical."** For each exponential-family distribution there is one link — the *canonical* one — that makes the gradient come out as $X^\top(\hat\mu - y)$ with no extra factors. It is not the only legal link, but it is the one where the algebra is clean, the loss is convex, and the residuals have the interpretation you expect. **All three "the same shape again" observations in this section are the same theorem viewed from three rows of that table.**

#### Examples and non-examples: is that a GLM?

**✅  examples**

| Example | Link + distribution |
|---|---|
| Ordinary least squares | Identity + Gaussian |
| Logistic regression | Logit + Bernoulli |
| Poisson regression on event counts | Log + Poisson |
| Softmax regression | Multinomial logit + Categorical |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| $\hat y = \beta_0 e^{\beta_1 x}$ fitted by least squares | Nonlinear in $\beta$, and the noise model is wrong | Nonlinear least squares |
| Log-transforming $y$ then running OLS | Models $\mathbb{E}[\log y]$, **not** $\log\mathbb{E}[y]$ — Jensen's inequality makes these different | A log-linear model. Fine, but a different estimand |
| A generalized **additive** model, $g(\mathbb{E}[y]) = \sum_j f_j(x_j)$ | The $f_j$ are learned smooth functions, not coefficients | A GAM — strictly more general |
| A neural network with a sigmoid output | The "linear predictor" is a learned nonlinear function of $x$ | A GLM head bolted onto a nonlinear body |

▸ **The boundary:** in a GLM the predictor must be *linear in $\beta$* and the response must come from an exponential-family distribution, with the link mediating between them. Relax the first and you get a GAM or a network; relax the second and maximum likelihood no longer hands you a convex problem.

> **Where this came from.** **John Nelder and Robert Wedderburn** unified the family in a 1972 paper in the *Journal of the Royal Statistical Society*. Before it, linear regression, logistic regression, probit analysis, log-linear models for contingency tables, and Poisson regression were taught as five separate techniques with five separate literatures and five separate computer programs. The paper showed they are one model with a swappable link, fitted by one algorithm — **iteratively reweighted least squares** — which meant a single piece of software could do all of them. Nelder went on to lead the development of that software, GLIM, at Rothamsted Experimental Station. ▸ **Unification papers rarely contain a new technique; their contribution is to collapse a taxonomy into a parameter.** The same move is what Chapter 20 does for diffusion models.

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

#### Reading the margin problem in plain English

Three notational choices are doing quiet work here, and none of them is deep.

**1. Labels are $\{-1,+1\}$, not $\{0,1\}$.** This is pure convenience. It makes $y_i(w^\top x_i + b)$ a single quantity that is **positive when correct and negative when wrong**, regardless of class. That quantity has a name — the **functional margin** — and everything in this section is a statement about it.

| $y_i$ | $w^\top x_i+b$ | $y_i(w^\top x_i+b)$ | Meaning |
|---|---|---|---|
| $+1$ | $+2.0$ | $+2.0$ | Correct, comfortably |
| $-1$ | $-2.0$ | $+2.0$ | Also correct, also comfortably. **Same number** |
| $+1$ | $+0.3$ | $+0.3$ | Correct but inside the margin |
| $-1$ | $+0.5$ | $-0.5$ | Wrong |

**2. "The canonical scaling $\min_i y_i(w^\top x_i+b) = 1$."** The hyperplane $w^\top x + b = 0$ is unchanged if you double both $w$ and $b$ — same set of points, same decision boundary. So there is a free scale factor floating around, and the constraint "$\ge 1$" simply spends it: **we agree to measure $w$ in units where the closest point scores exactly 1.** Without this convention "maximize the margin" has no answer, since you could make the margin any number you like by rescaling.

**3. $\text{s.t.}$ means "subject to," and $\forall i$ means "for every example."** So the whole line reads: *"make $w$ as short as possible, while keeping every single training point at least one unit away on the correct side."*

**Put numbers on it.** Two dimensions, $w = (1,1)$, $b = -3$. The boundary is $x_1+x_2 = 3$.

| Point | Label | $w^\top x + b$ | $y_i(w^\top x_i+b)$ | Distance to boundary |
|---|---|---|---|---|
| $(2,2)$ | $+1$ | $4-3 = 1$ | $1$ | $1/\sqrt2 = 0.707$ |
| $(1,1)$ | $-1$ | $2-3 = -1$ | $1$ | $0.707$ |
| $(3,3)$ | $+1$ | $6-3=3$ | $3$ | $2.12$ |

$\lVert w\rVert = \sqrt{1^2+1^2} = \sqrt2 = 1.414$, so the margin $\frac{2}{\lVert w\rVert} = 1.414$ — and indeed $0.707 + 0.707 = 1.414$. ▸ **The margin is the total width of the empty corridor: one closest point on each side, each $1/\lVert w\rVert$ away.**

**And why minimizing $\lVert w\rVert$ maximizes the margin.** The margin is $\frac{2}{\lVert w\rVert}$. A fraction gets bigger when its denominator gets smaller. That is the entire argument — no calculus required. (The $\tfrac12$ and the square are cosmetic: squaring removes a square root, and the half cancels when you differentiate. Neither changes the argmin.)

> **Analogy, extending the book's road.** Two villages of houses, and you must lay a straight road between them without demolishing anything. Any legal road works, but you want the **widest** one, so a car drifting slightly stays on tarmac. Here is the part the picture makes obvious and the algebra hides: **only the houses touching the kerb constrain you.** A house 400 metres back is irrelevant — you could delete it, or move it further back, and the road would not shift a centimetre. Those kerbside houses are the support vectors, and the SVM's defining strangeness is that it is a model fitted to a handful of edge cases while ignoring the bulk of the data.

> **Common misconception.** *"The SVM finds the boundary that best separates the classes, so it's using all the data."* It uses whichever points end up on the margin — sometimes three of them, out of fifty thousand. Move a point deep inside its own class anywhere else deep inside its own class and the model is bit-for-bit identical. The misconception is tempting because every other method in this chapter *does* use every point: OLS, ridge, logistic regression, and naive Bayes all shift, however slightly, when any single observation moves. ▸ **The flip side is the SVM's real fragility: a single mislabelled point sitting near the boundary can swing the solution violently, because it becomes a support vector.** That fragility is exactly what the soft margin's $C$ is for.

### The dual

Lagrangian $L = \frac12\|w\|^2-\sum_i\alpha_i\left[y_i(w^\top x_i+b)-1\right]$, $\alpha_i\ge0$.

Stationarity: $\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_i\alpha_iy_ix_i$; $\frac{\partial L}{\partial b}=0\Rightarrow \sum_i\alpha_iy_i=0$. Substituting:

▸ $$\max_\alpha\ \sum_i\alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\langle x_i,x_j\rangle\quad\text{s.t.}\quad \alpha_i\ge0,\ \sum_i\alpha_iy_i=0$$

▸ **Two facts fall out, and both matter:**
1. **The data enters only through inner products $\langle x_i,x_j\rangle$** → the kernel trick.
2. **KKT complementary slackness:** $\alpha_i\big[y_i(w^\top x_i+b)-1\big]=0$. So $\alpha_i>0$ **only** for points exactly on the margin. All other points have $\alpha_i=0$ and can be deleted without changing the solution. Those are the **support vectors**, and the sparsity is exact, not approximate.

#### The dual, decoded — what a Lagrange multiplier *is*

Constrained optimization has one core trick: **turn a hard constraint into a price.**

Instead of forbidding a violation outright, charge for it. $\alpha_i$ is the price attached to example $i$'s constraint, and the Lagrangian $L = \frac12\lVert w\rVert^2 - \sum_i\alpha_i[y_i(w^\top x_i+b)-1]$ reads: *"the thing I want to minimize, minus a payment for every unit of slack each constraint has left over."*

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\alpha_i \ge 0$ | "alpha-i, non-negative" | The **price** of example $i$'s constraint. Prices can't be negative |
| $\alpha_i = 0$ | — | "This constraint isn't binding. Ignore this point entirely" |
| $\alpha_i > 0$ | — | "This point is pressed right up against the boundary and is pushing" |
| $w = \sum_i \alpha_i y_i x_i$ | "w is a weighted sum of the data" | **The answer is built out of the training points themselves** |
| $\sum_i \alpha_i y_i = 0$ | — | The pushes from the two classes must balance, or the boundary slides |

▸ **Read $w = \sum_i\alpha_iy_ix_i$ carefully — it is the sentence the entire kernel edifice rests on.** The weight vector is not some independent object; it is a *linear combination of the training examples*, with positive examples pushing one way and negative ones the other, weighted by $\alpha_i$. Since $\alpha_i = 0$ for almost everything, $w$ is a combination of a handful of points.

> **Analogy for complementary slackness.** A door held shut by several people leaning on it. Everyone within arm's reach is exerting some force ($\alpha_i > 0$). Everyone standing back in the corridor is exerting exactly zero ($\alpha_i = 0$) — and the door does not care whether they are one metre away or a hundred. The condition $\alpha_i[y_i(w^\top x_i+b)-1] = 0$ is precisely "**either you're touching the door, or you're pushing with zero force.**" One of the two factors must vanish; both cannot be non-zero at once.

**Why the dual is written with $\langle x_i, x_j\rangle$ and why that changes everything.** Look at the dual objective: $\sum_i\alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle$. The features $x$ appear **only** inside dot products with each other. Never alone. Never squared. Never individually.

So the optimizer never needs to know what $x_i$ *is* — only how similar each pair of examples is. Hand it a table of $n^2$ similarity numbers and it can solve the problem without ever seeing a feature vector. ▸ **That single structural observation is the kernel trick, and it was sitting in the algebra before anyone thought to exploit it.**

**Where the numbers land, concretely.** Suppose 1,000 training points and the solver returns $\alpha$ with 12 non-zero entries. Then:

- Prediction cost drops from "touch 1,000 points" to "touch 12 points."
- You may **delete 988 rows of your training set** and refit; you will get the identical model.
- Add a new point far from the boundary and nothing changes — not the boundary, not $\alpha$, not a single prediction.

#### Examples and non-examples: support vectors

**✅  support vectors**

| Example | Why |
|---|---|
| A point sitting exactly on the margin, $y_i(w^\top x_i+b) = 1$ | $\alpha_i > 0$; it is holding the boundary in place |
| In the soft-margin case, a point inside the margin ($0 < \xi_i < 1$) | Still constrains; $\alpha_i = C$ |
| A **misclassified** point ($\xi_i > 1$) | Also a support vector, at $\alpha_i = C$ — it pushes maximally and loses |

**❌ Near-misses**

| Looks like one | Why it isn't | What it actually is |
|---|---|---|
| The point closest to the *other class's* centroid | Distance to a centroid has nothing to do with the margin | Just a point. $\alpha_i$ is probably 0 |
| An outlier far outside its own class, away from the boundary | Comfortably correct ⇒ $\alpha_i = 0$ ⇒ invisible to the model | An outlier the SVM ignores completely |
| A high-leverage point in linear regression | Leverage is $H_{ii}$, a projection quantity from a different method | An influential observation in OLS |
| A "hard negative" mined for a contrastive loss (Ch. 25) | Related in spirit, but selected by heuristic, not by an optimality condition | A hard negative |

▸ **The boundary:** a support vector is defined by an *optimality condition* — $\alpha_i > 0$ — not by being unusual, extreme, or interesting. It is the set of points whose removal would change the answer, and the SVM computes that set exactly rather than estimating it.

> **Common misconception.** *"Outliers wreck an SVM, because it's fitted to the extreme points."* An outlier far from the boundary on the correct side is the *most* ignorable point there is: $\alpha_i = 0$, and it could be at infinity for all the model cares. Compare OLS, where a single distant point can drag the entire line toward it because squared error grows quadratically. **The SVM is robust to distant outliers and fragile to near-boundary noise; OLS is the reverse.** The misconception is tempting because "it only uses the extreme points" sounds like "it only uses the outliers," and those are entirely different notions of extreme.

> **Common misconception.** *"The dual is just a mathematical curiosity; you could solve the primal."* You can, and for linear SVMs on large $n$ you should — that is exactly what LIBLINEAR does. But the dual is what makes kernels *possible*, because the primal contains $w$, which lives in the feature space, and in an infinite-dimensional feature space $w$ cannot be written down. ▸ **The dual is where you go when the primal's variables don't fit in memory — or in the universe.**

### Soft margin

$$\min\ \tfrac12\|w\|^2 + C\sum_i\xi_i\quad\text{s.t. } y_i(w^\top x_i+b)\ge1-\xi_i,\ \xi_i\ge0$$

Equivalent unconstrained form with the **hinge loss**:
▸ $$\min_w\ \frac{1}{2}\|w\|^2 + C\sum_i\max\big(0,\ 1-y_i(w^\top x_i+b)\big)$$

The dual is unchanged except $0\le\alpha_i\le C$ (a "box constraint"). **Small $C$ = wide margin, more violations, more regularization.**

**Hinge vs logistic loss:** hinge is exactly zero once the margin is met (⇒ sparsity, only support vectors matter); logistic is always positive (⇒ every point contributes, and you get calibrated probabilities). That trade — sparsity vs probabilities — is the practical difference between SVM and logistic regression.

#### Unpacking slack, $C$, and the hinge

**$\xi_i$ (xi, rhyming with "sigh") is a permission slip.** The hard constraint said "score at least 1." The soft one says "score at least $1 - \xi_i$, and pay $C\xi_i$ for the privilege."

| $\xi_i$ | $y_i(w^\top x_i+b)$ | Where the point sits |
|---|---|---|
| $0$ | $\ge 1$ | Outside the margin, correct. **Free** |
| $0.3$ | $0.7$ | Inside the margin, still on the correct side |
| $1.0$ | $0$ | Exactly on the decision boundary |
| $1.7$ | $-0.7$ | **Misclassified**, and $\xi_i > 1$ says so |

▸ **$\xi_i > 1$ is exactly the condition for a training error**, so $\sum_i\xi_i$ is an upper bound on the number of mistakes. The soft-margin objective is therefore, quite literally, "keep the corridor wide, and pay $C$ per unit of mistake-ness."

**Now $C$.** Divide the whole objective by $C$ and it becomes $\frac{1}{2C}\lVert w\rVert^2 + \sum_i\xi_i$. ▸ **So $\frac1C$ plays the role that $\lambda$ played in ridge: it is the regularization strength wearing a reciprocal.** Every confusion about "does larger $C$ mean more or less regularization" dissolves the moment you write it that way.

| $C$ | Behaviour | Failure mode |
|---|---|---|
| $C \to \infty$ | No violation tolerated. Reduces to the hard margin | Fails outright if the data isn't separable; chases every noisy point |
| $C = 100$ | Narrow corridor, few violations | Overfits; support-vector count small but boundary jagged |
| $C = 1$ | The usual default | — |
| $C = 0.01$ | Wide corridor, many violations allowed | Underfits; nearly every point becomes a support vector |
| $C \to 0$ | The penalty vanishes; $w \to 0$ | Predicts one class for everything |

**The hinge, and the trade against logistic.** $\max(0, 1-z)$ is the same positive-part operator $(\cdot)_+$ from soft-thresholding in §22.3 — and, for that matter, the same shape as a ReLU, flipped. Write $z = y_i(w^\top x_i + b)$:

| $z$ | Hinge $\max(0,1-z)$ | Logistic $\log(1+e^{-z})$ | Difference |
|---|---|---|---|
| $5.0$ | $\mathbf{0}$ | $0.0067$ | Hinge: "done." Logistic: "could be better" |
| $2.0$ | $\mathbf{0}$ | $0.127$ | |
| $1.0$ | $\mathbf{0}$ | $0.313$ | Hinge switches off exactly here |
| $0.5$ | $0.5$ | $0.474$ | Nearly identical in the middle |
| $0$ | $1.0$ | $0.693$ | |
| $-1.0$ | $2.0$ | $1.313$ | Hinge grows linearly, logistic asymptotically linearly |
| $-5.0$ | $6.0$ | $5.007$ | Both roughly linear far out. **Both robust to gross outliers** |

▸ **The exact zero in the hinge column is the whole story.** A loss that is *identically* zero over a region contributes *identically zero* gradient there, so those points drop out of the solution entirely — that is where the SVM's exact sparsity comes from, and it is the same mechanism (a kink) that gives LASSO its exact zeros. Logistic's $0.0067$ at $z=5$ is small but never zero, so every point keeps a vote forever, which is exactly what you need if you want the output to be a calibrated probability rather than a decision.

#### Examples and non-examples: margin-based losses

**✅  margin losses**

| Example | Why it qualifies |
|---|---|
| Hinge, $\max(0, 1-z)$ | A function of $z = y\cdot\text{score}$ alone; penalizes small margins |
| Squared hinge, $\max(0,1-z)^2$ | Same idea, differentiable everywhere; used by LIBLINEAR |
| Logistic, $\log(1+e^{-z})$ | Also a function of $z$ alone; a "soft" hinge |
| Exponential, $e^{-z}$ | The margin loss AdaBoost minimizes (Ch. 23) |

**❌ Near-misses**

| Looks like one | Why it isn't | What it actually is |
|---|---|---|
| 0–1 loss, $\mathbb{1}[z < 0]$ | Zero gradient everywhere it's defined; NP-hard to minimize | The thing all the others are **surrogates for** |
| Squared error on $\pm1$ labels | Punishes $z = 3$ *more* than $z = 1$ — being too right is penalized | Regression, misapplied |
| Accuracy | Not a loss at all; not differentiable, not decomposable | An evaluation metric |
| Triplet loss, $\max(0, d_+ - d_- + m)$ | Also a hinge, but on a *difference of distances*, not on $y\cdot\text{score}$ | A metric-learning loss (Ch. 25) |

▸ **The boundary:** a margin loss is a decreasing function of the single quantity $z = y\cdot\text{score}$, so it rewards being right *by a comfortable amount*, not merely right. Squared error fails the test because it is not monotone in $z$ — it wants your score to be exactly $\pm1$ and objects to anything more confident.

> **Common misconception.** *"The SVM maximizes accuracy on the training set."* It minimizes hinge loss plus a norm penalty, and those are different objectives. It will happily accept a few more training errors in exchange for a wider corridor — that is what small $C$ *asks* it to do. A model with 97% training accuracy and a wide margin usually beats one with 100% and a razor-thin one, and Chapter 2's norm-based generalization bounds say why. The misconception is tempting because the SVM is presented as a classifier and classifiers are scored by accuracy.

> **Where this came from.** The maximum-margin idea is much older than its fame. **Vladimir Vapnik and Alexey Chervonenkis** developed the "generalized portrait" algorithm in the early 1960s at the Institute of Control Sciences in Moscow, alongside the statistical learning theory (VC dimension, Ch. 2) that explains why margins help. It sat largely unused in the West for nearly thirty years, in part because the linear, hard-margin version is not much use on real data. Two additions turned it into the dominant classifier of the 1990s: **kernels**, added by Bernhard Boser, Isabelle Guyon and Vapnik in a 1992 conference paper, and the **soft margin**, added by Corinna Cortes and Vapnik in 1995 in a paper titled "Support-Vector Networks" — the word "networks" in the title being a nod to the neural networks the method was, at that moment, beating. Vapnik had by then moved to AT&T Bell Labs, which also housed Yann LeCun's convolutional network group; the two approaches competed directly on handwritten-digit recognition down the hall from each other.

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

#### The kernel trick, worked by hand

The claim "$\phi$ is never computed" sounds like a slogan. Here it is as arithmetic you can check.

Take the quadratic kernel in two dimensions, $k(x,x') = (x^\top x')^2$. Expand it:

$$(x_1x_1' + x_2x_2')^2 = x_1^2x_1'^2 + 2x_1x_2x_1'x_2' + x_2^2x_2'^2$$

Now stare at the right-hand side and notice it is a dot product of two 3-vectors:

$$\phi(x) = \big(x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2\big) \quad\Longrightarrow\quad (x^\top x')^2 = \langle \phi(x), \phi(x')\rangle$$

**Check it with numbers.** Let $x = (1,2)$ and $x' = (3,1)$.

| Route | Computation | Result |
|---|---|---|
| **Kernel** | $x^\top x' = 1(3)+2(1) = 5$, then $5^2$ | $\mathbf{25}$ |
| **Explicit** | $\phi(x) = (1,\ 2.828,\ 4)$, $\phi(x') = (9,\ 4.243,\ 1)$, then their dot product: $9 + 12 + 4$ | $\mathbf{25}$ |

▸ **Identical answers. Two multiplications and a square, versus building two 3-vectors and taking a 3-dimensional dot product.** With two features the saving is trivial. With 100 features and degree 5, the explicit feature space has about **92 million** dimensions — while the kernel is still one dot product and one exponentiation. **That gap is the trick.**

> **Analogy.** You want the distance between two cities. One route: fly to each, walk the ground between them, report the number. The other: look up their coordinates and use a formula. The formula never visits either city, and it gives the same answer. **The feature space is the territory you never have to visit** — the kernel is the formula that answers questions about it from where you're standing.

#### The RBF kernel, with numbers

$k(x,x') = \exp(-\gamma\lVert x-x'\rVert^2)$ reads: *"take the squared distance between the two points, scale it by $\gamma$, negate it, exponentiate."* It is a similarity that is 1 when the points coincide and decays toward 0 as they separate.

| $\lVert x-x'\rVert^2$ | $\gamma = 0.1$ | $\gamma = 1$ | $\gamma = 10$ |
|---|---|---|---|
| $0$ | $1.000$ | $1.000$ | $1.000$ |
| $1$ | $0.905$ | $0.368$ | $4.5\times10^{-5}$ |
| $4$ | $0.670$ | $0.018$ | $4.2\times10^{-18}$ |
| $9$ | $0.407$ | $1.2\times10^{-4}$ | $\approx 0$ |
| $25$ | $0.082$ | $1.4\times10^{-11}$ | $\approx 0$ |

▸ **Read down the $\gamma = 10$ column: every point is a stranger to every other point.** The kernel matrix becomes essentially the identity, the model memorizes each training point in its own tiny bubble, and it predicts the majority class everywhere else. Read down $\gamma = 0.1$: everything is similar to everything, the kernel matrix is nearly all-ones, and the model reduces to a constant. **$\gamma$ is a bandwidth, and both extremes fail — in opposite directions.**

**Why the RBF's feature space is infinite-dimensional, concretely.** Factor it:

$$e^{-\gamma\lVert x-x'\rVert^2} = e^{-\gamma\lVert x\rVert^2}\,e^{-\gamma\lVert x'\rVert^2}\,e^{2\gamma\,x^\top x'}$$

and expand the last factor as its Taylor series:

$$e^{2\gamma\,x^\top x'} = \sum_{k=0}^{\infty}\frac{(2\gamma)^k}{k!}\,(x^\top x')^k$$

▸ **That is a weighted sum of *every* polynomial kernel, of every degree, out to infinity** — degree 0, degree 1, degree 2, and on forever, with weights $(2\gamma)^k/k!$ that eventually decay. So the RBF feature space contains every monomial in your features. And it still generalizes, because (Ch. 2 §2.5) capacity is governed by the **norm** of the solution, not the dimension of the space it lives in.

#### Examples and non-examples: valid kernels

A kernel is valid (Mercer) exactly when every Gram matrix $K_{ij} = k(x_i,x_j)$ it produces is symmetric and positive semi-definite. That is not decoration: PSD is what guarantees the dual is a **convex** problem with a unique optimum, and it is what guarantees the implied $\phi$ exists at all.

**✅ Valid kernels**

| Example | Why |
|---|---|
| $k(x,x') = x^\top x'$ | $K = XX^\top$, PSD by construction |
| $\exp(-\gamma\lVert x-x'\rVert^2)$ | Provably PSD for all $\gamma>0$ |
| $(\gamma x^\top x' + r)^d$ with $r\ge0$, $d$ a positive integer | Products and sums of valid kernels are valid |
| $k_1 + k_2$, or $c\cdot k_1$ for $c>0$, or $k_1 k_2$ | **Kernels are closed under these operations** — build new ones this way |
| A string kernel counting shared subsequences | Never writes down a feature vector at all, and is PSD |

**❌ Near-misses — similarity scores that are not kernels**

| Looks like one | Why it fails | Consequence |
|---|---|---|
| $k(x,x') = \lVert x - x'\rVert$ | It's a **distance**, which grows with dissimilarity. Not PSD | No feature space exists; the dual is non-convex |
| $\tanh(\gamma x^\top x' + r)$, the "sigmoid kernel" | Not PSD for most parameter settings | Used anyway, historically, and it sometimes silently misbehaves |
| A hand-built similarity matrix with some negative eigenvalue | Fails PSD by inspection | Solvers may not converge, or converge to nonsense |
| Cosine similarity on unnormalized data, computed inconsistently | The issue is bookkeeping, not the formula | Cosine similarity **is** a valid kernel when applied consistently |

▸ **The boundary:** a valid kernel is secretly a dot product in *some* space, and PSD is the exact certificate that such a space exists. Similarity is not enough; a distance is the wrong sign, and an arbitrary similarity table is usually not PSD.

> **Common misconception.** *"The kernel trick maps your data into a high-dimensional space."* It maps *nothing*. No feature vector in the expanded space is ever constructed, stored, or looked at, and for the RBF kernel it could not be — it has infinitely many coordinates. What is computed is a table of pairwise similarities. The misconception is tempting because every textbook picture shows points being lifted onto a paraboloid, and that picture is a *proof device*, not an account of the computation.

> **Common misconception.** *"An infinite-dimensional model must overfit — you have more parameters than data."* The RBF SVM has effectively infinite capacity by the dimension-counting standard and generalizes well anyway. Dimension is simply the wrong meter. What is being controlled is $\lVert w\rVert$, the norm of the solution, and $C$ controls it directly. ▸ **This is the cleanest available demonstration that "number of parameters" is not a measure of model complexity** — the same lesson Chapter 30 delivers for overparameterized neural networks, arrived at thirty years earlier by a completely different route.

> **Where this came from.** The three ingredients arrived decades apart and from different fields. **James Mercer** proved the condition that bears his name in 1909, in pure analysis, working on integral equations — no data, no learning, no computers. The idea of using kernels for pattern recognition was published in 1964 by **Mark Aizerman, Emmanuel Braverman and Lev Rozonoer** in the Soviet Union, under the name the "method of potential functions," where each training point was imagined as an electric charge whose potential field extends around it — the RBF kernel *is* a potential field, and the physical metaphor was the original motivation. And **Thomas Cover** proved in 1965 that a set of points is more likely to be linearly separable the higher the dimension you cast it into, which is the theoretical licence for the whole enterprise. Yet the pieces sat unconnected for nearly thirty years. The 1992 paper of Boser, Guyon and Vapnik put margins and kernels together, and the field changed within about three years. ▸ **All three ingredients had been publicly available since 1965. Nobody assembled them.**

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

## Did you know?

- **Least squares was invented to find a lost asteroid.** Ceres was discovered in January 1801 and then vanished behind the sun. From 41 days of observations, the 24-year-old Carl Friedrich Gauss predicted where it would reappear; astronomers pointed their telescopes there and found it. Gauss became famous across Europe, and the method became the foundation of statistics.

- **There was a bitter priority dispute over least squares.** Adrien-Marie Legendre published it first, in 1805. Gauss claimed in 1809 that he had been using it since 1795 — and was probably telling the truth, but had not published. Legendre was, by all accounts, furious. Gauss's habit of not publishing results he considered unfinished caused several such disputes.

- **"Regression" is named after a phenomenon that has nothing to do with fitting lines.** Francis Galton, studying heredity in the 1880s, observed that tall parents tend to have children closer to average height — "regression toward mediocrity." The statistical technique inherited the name of the biological observation it was used to study.

- **LASSO stands for Least Absolute Shrinkage and Selection Operator**, coined by Robert Tibshirani in 1996. The acronym was reverse-engineered to spell a word — a small tradition in statistics that also gave us BIC, GLM, and a great many strained abbreviations.

- **The reason $\ell_1$ produces exact zeros is geometric.** The $\ell_1$ constraint region is a diamond with sharp corners on the axes; the $\ell_2$ region is a smooth ball. A contour of the loss touching a diamond most often touches it *at a corner* — and a corner is where a coordinate equals exactly zero. A ball has no corners, so $\ell_2$ shrinks without ever zeroing.

- **Support vector machines dominated machine learning for roughly fifteen years.** From the mid-1990s until deep learning's 2012 breakthrough, SVMs with kernels were the default strong baseline for most classification problems. Vladimir Vapnik, their co-inventor, also developed the VC dimension of Chapter 2.

- **The kernel trick computes inner products in infinite-dimensional spaces without ever visiting them.** The radial basis function kernel corresponds to a feature map with infinitely many dimensions, yet evaluating it costs the same as measuring a distance between two ordinary vectors. You get the expressiveness without paying for the coordinates.

- **An SVM's decision boundary depends only on a handful of points.** The support vectors — the examples sitting on the margin — fully determine the classifier. Delete every other training point and the boundary is unchanged, which is a strikingly different notion of "what the model learned" from a neural network's.

- **Logistic regression's loss has no closed-form solution**, unlike ordinary least squares. There is no formula for the optimal weights; you must iterate. This is the earliest point in a statistics education where the answer stops being an equation and starts being an algorithm.

- **Ridge regression was invented twice, in different fields, for different reasons.** Statisticians derived it to handle correlated predictors; numerical analysts derived essentially the same fix (Tikhonov regularization, named for Andrey Tikhonov) to stabilize ill-posed inverse problems. Adding $\lambda$ to the diagonal solves both problems because they are the same problem.

- **Ridge shrinks different directions by different amounts**, which is invisible in the usual formula. Written through the singular value decomposition, direction $j$ is multiplied by $\sigma_j^2/(\sigma_j^2 + \lambda)$ — so well-determined directions (large $\sigma_j$) pass through nearly untouched while poorly-determined ones are crushed. It is a *targeted* penalty, not a blanket one.

- **These methods are not obsolete, and on tabular data they frequently win.** Gradient-boosted trees (Chapter 23) and regularized linear models still outperform neural networks on the majority of real-world tabular problems — a fact that surprises people whose exposure to machine learning began after 2015.

---

## Check for Understanding

**Ridge shrinks each principal direction by $\sigma_j^2/(\sigma_j^2+\lambda)$ so poorly-determined directions are damped most, LASSO's non-differentiable corner at zero is what produces exact sparsity, and the SVM's dual shows the data entering only through inner products — which is what lets a linear classifier in an infinite-dimensional space be computed with $n$ coefficients and still generalize, because capacity is controlled by norm rather than dimension.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why does ridge regression shrink some directions more than others?** What makes a direction "poorly determined"?
2. **Why does $\ell_1$ give exact zeros when $\ell_2$ doesn't?** (Diamond versus ball — say it geometrically.)
3. **When would you choose ridge over LASSO**, and when elastic net over either?
4. **Why does logistic regression need an iterative solver when linear regression doesn't?**
5. **What is a support vector**, and why does deleting all the other training points change nothing?
6. **What is the kernel trick actually doing?** How can you work in infinite dimensions cheaply?
7. **Why doesn't an SVM in an infinite-dimensional space overfit catastrophically?** (What controls capacity, if not dimension?)
8. **What does the "dual" formulation buy you**, and why does it matter that data enters only through inner products?
9. **Why is "regression" called regression?** (The answer is about Victorian heredity, not mathematics.)
10. **Why is adding $\lambda$ to the diagonal both a statistical fix and a numerical one?**
11. **What does the margin mean**, and why does maximizing it tend to generalize well?
12. **Why do these methods still beat neural networks on tabular data?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 23 — Trees & Gradient Boosting](23-trees-and-gradient-boosting.md)
