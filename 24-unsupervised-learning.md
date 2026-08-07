# Chapter 24 — Unsupervised Learning & Dimensionality Reduction

> **Prerequisites:** Ch. 1 (§1.1 SVD, §1.4), Ch. 22.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\Sigma$ | "capital Sigma" | The **covariance matrix** — a grid saying how every pair of features moves together |
| $\Sigma w = \lambda w$ | "Sigma w equals lambda w" | "$w$ is a direction the covariance only **stretches**, never turns" |
| $\lambda_j$ | "lambda-j" | The **variance captured** along principal direction $j$ |
| $\|w\|=1$ | "norm of w equals one" | "$w$ is a pure direction — an arrow of length one, no magnitude" |
| $P$ | "P" | A **projection** — the operator that flattens points onto a subspace |
| $z$ | "z" | A **latent variable**: unobserved coordinates presumed to have generated the data |
| $\mathcal{N}(\mu,\Sigma)$ | "normal with mean mu, covariance Sigma" | The bell curve, in many dimensions at once |
| $p_{j\mid i}$ | "p of j given i" | Probability that point $i$ would pick point $j$ as its neighbour |
| $q_{ij}$ | "q-i-j" | The same neighbour probability, but computed in the 2-D picture |
| $\mathrm{KL}(P\|Q)$ | "KL of P from Q" | How badly $Q$ fails as a stand-in for $P$ (Ch. 1 §1.4) |
| $\mu_k$ | "mu-k" | The **centroid** — the average point of cluster $k$ |
| $S_k$ | "S-k" | The **set** of data points currently assigned to cluster $k$ |
| $\pi_k$ | "pi-k" | The **mixing weight**: what fraction of all data comes from component $k$ |
| $\gamma_{ik}$ | "gamma-i-k" | The **responsibility**: probability that point $i$ came from component $k$ |
| $L = D - W$ | "L equals D minus W" | The **graph Laplacian** — a matrix encoding a graph's connectivity |
| $\varepsilon$, `minPts` | "epsilon, min-points" | DBSCAN's radius, and how many neighbours make a point "core" |
| $s_i$ | "s-i" | The **silhouette** of point $i$: how much better its own cluster fits than the next-best |

### Full forms for the abbreviations in this chapter

| Short | Full form |
|---|---|
| PCA | principal component analysis |
| SVD | singular value decomposition |
| ICA | independent component analysis |
| NMF | non-negative matrix factorization |
| CLT | central limit theorem |
| RKHS | reproducing kernel Hilbert space |
| t-SNE | t-distributed stochastic neighbour embedding |
| UMAP | Uniform Manifold Approximation and Projection |
| GMM | Gaussian mixture model |
| EM | expectation–maximization |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| HDBSCAN | Hierarchical DBSCAN |
| SSE | sum of squared errors |
| PAM | Partitioning Around Medoids |
| LOF | local outlier factor |
| SVM | support vector machine |
| MAD | median absolute deviation |
| OOD | out-of-distribution |
| ARI / NMI | adjusted Rand index / normalized mutual information |
| PR-AUC / ROC-AUC | area under the precision–recall / receiver-operating-characteristic curve |
| LDA | linear discriminant analysis |
| VAE | variational autoencoder |
| k-NN | k-nearest-neighbours |

---

## 24.1 Principal Component Analysis

### The one-line idea

Find the orthogonal directions along which the data varies most, and keep only those — because directions with little variance usually carry little signal.

### The analogy

A cloud of points shaped like a flattened, tilted pancake. There are three coordinates, but the pancake is essentially two-dimensional. PCA finds the plane of the pancake and reports each point's position within it, discarding the negligible thickness. The trick is that the pancake's plane is not aligned with any of your original axes — PCA finds the *rotation* that makes the flatness obvious.

### Three derivations, all giving the same answer

Assume $X$ is centred ($\frac1n\sum_i x_i = 0$). **Centring is mandatory** — without it the first component points at the mean.

**(1) Maximum variance.** Find the unit vector $w$ maximizing the variance of the projection:
$$\max_{\|w\|=1}\ \frac1n\sum_i(w^\top x_i)^2 = \max_{\|w\|=1}\ w^\top \Sigma w,\qquad \Sigma=\tfrac1nX^\top X$$
Lagrangian $w^\top\Sigma w - \lambda(w^\top w-1)$, stationarity gives
▸ $$\Sigma w = \lambda w$$
The maximizer is the **top eigenvector of the covariance matrix**, with variance $\lambda_1$.

**(2) Minimum reconstruction error.** Find the rank-$k$ projection $P$ minimizing $\sum_i\|x_i - Px_i\|^2$. Since $\|x\|^2 = \|Px\|^2+\|x-Px\|^2$, minimizing reconstruction error is *identical* to maximizing projected variance. Same answer.

**(3) SVD.** With $X = U\Sigma_{\text{sv}}V^\top$:
▸ Principal directions = **columns of $V$** (right singular vectors); scores = $U\Sigma_{\text{sv}}$; explained variance of component $j$ = $\sigma_j^2/\sum_k\sigma_k^2$.

**Always compute PCA via SVD of $X$, never by forming $X^\top X$** — the condition number squares (Ch. 22 §22.1).

#### Reading the three derivations in plain English

First, the setup notation, because everything depends on it:

| Symbol | Read aloud | What it is |
|---|---|---|
| $X$ | "X" | Your data as a grid: $n$ rows (points), $d$ columns (features) |
| $x_i$ | "x-i" | One row — a single data point, a list of $d$ numbers |
| $w$ | "w" | A candidate **direction** in feature space, an arrow of length 1 |
| $\|w\|=1$ | "norm of w equals one" | The constraint that $w$ is a pure direction with no magnitude of its own |
| $w^\top x_i$ | "w-transpose x-i" | The **projection** of point $i$ onto direction $w$ — one number: how far along $w$ the point sits |
| $\Sigma = \frac1nX^\top X$ | "Sigma equals one-over-n X-transpose X" | The **covariance matrix**: entry $(a,b)$ says how features $a$ and $b$ vary together |
| $\lambda$ | "lambda" | The **variance** you get along the corresponding direction |

**Why centring is mandatory, concretely.** Suppose every one of your points sits near $(100, 100)$ with a tiny spread. Uncentred, $\frac1n\sum_i(w^\top x_i)^2$ is dominated by the fact that all the points are 141 units from the origin, so the "direction of maximum variance" comes out pointing at $(100,100)$ — it has found where your data *is*, not how it *varies*. Subtract the mean and the origin sits inside the cloud, where the question becomes meaningful. ▸ **PCA measures spread about the mean, so if you forget to remove the mean, component 1 is the mean.**

**Derivation 1 — maximum variance, step by step.**

$$\max_{\|w\|=1}\ \frac1n\sum_i(w^\top x_i)^2 = \max_{\|w\|=1}\ w^\top \Sigma w$$

Read aloud: *"among all unit directions $w$, find the one that maximizes the average squared projection."* The equality is a rearrangement — $\sum_i (w^\top x_i)^2 = w^\top\big(\sum_i x_ix_i^\top\big)w$, and $\frac1n\sum_i x_ix_i^\top$ is the definition of $\Sigma$.

The constraint $\|w\|=1$ is doing real work: without it you would simply make $w$ enormous and the "variance" would be unbounded. **A Lagrangian** is the standard tool for optimizing under a constraint — you add a penalty term $-\lambda(w^\top w - 1)$ that is zero when the constraint holds and charges you otherwise, then optimize freely. Setting the derivative to zero gives

$$\Sigma w = \lambda w$$

▸ **which is the definition of an eigenvector (Ch. 1 §1.1.2).** So the answer to "which direction has the most spread?" is "an eigenvector of the covariance," and multiplying both sides by $w^\top$ shows $w^\top\Sigma w = \lambda$ — **the variance along that direction is exactly its eigenvalue.** So you want the largest one. Nothing about PCA is a search; the answer is handed to you by a matrix factorization.

**Derivation 2 — minimum reconstruction error, and why it's the same problem.** The identity

$$\|x\|^2 = \|Px\|^2 + \|x - Px\|^2$$

is **Pythagoras**. $Px$ is the shadow the point casts on your chosen plane; $x - Px$ is the perpendicular gap between the point and its shadow; and the original vector is the hypotenuse. Since $\|x\|^2$ is fixed — it is a property of your data, not of your choice of plane — **making the shadow long is exactly the same act as making the gap short.** One equation, two names.

> **Analogy.** You are choosing where to stand to photograph a flock of birds in flight. "Maximize the spread of the birds in my photo" and "minimize how much depth I'm losing" sound like two different goals; Pythagoras says they are one goal. The total is fixed by the flock; you are only deciding how to split it between what the photo captures and what it discards.

**Derivation 3 — SVD, and the notation collision.** The text writes $X = U\Sigma_{\text{sv}}V^\top$ with a subscript because **$\Sigma$ is already being used for the covariance matrix.** This clash is universal in the literature and is worth naming out loud: capital sigma means "covariance matrix" in statistics and "the diagonal matrix of singular values" in linear algebra, and PCA is precisely where the two meet.

- **Principal directions are the columns of $V$** — the right singular vectors, living in feature space ($\mathbb{R}^d$). These are the new axes.
- **Scores are $U\Sigma_{\text{sv}}$** — where each data point lands in the new coordinate system. This is what you feed downstream.
- **Explained variance of component $j$ is $\sigma_j^2/\sum_k\sigma_k^2$** — read: "this component's squared singular value as a fraction of the total." Since $\lambda_j = \sigma_j^2/n$, this is just "what share of the total spread does direction $j$ own."

**Real numbers.** Take $n = 3$ points in 2-D, already centred: $(-2,-1)$, $(0,0)$, $(2,1)$. They lie exactly on the line $y = x/2$.

$$\Sigma = \tfrac13\begin{pmatrix} 4+0+4 & 2+0+2 \\ 2+0+2 & 1+0+1\end{pmatrix} = \tfrac13\begin{pmatrix} 8 & 4 \\ 4 & 2\end{pmatrix}$$

The direction $w = \frac{1}{\sqrt5}(2,1)$ gives $\Sigma w = \frac13\frac{1}{\sqrt5}(20, 10) = \frac{10}{3}w$. So $\lambda_1 = 10/3 \approx 3.33$. The perpendicular direction $\frac1{\sqrt5}(-1,2)$ gives $\Sigma w = 0$, so $\lambda_2 = 0$. ▸ **Two features, one nonzero eigenvalue: the data is  one-dimensional, and PCA has said so exactly.** The second component explains $0/(3.33) = 0\%$ of the variance and can be discarded with no loss whatsoever. Real data never gives an exact zero, but it routinely gives $0.001$, which is the same message with noise on top.

**Why never form $X^\top X$.** The condition number $\kappa$ measures how much a matrix amplifies small numerical errors. Squaring the matrix squares the condition number. If $X$ has $\kappa = 10^6$ — unremarkable for real data with mixed units — then $X^\top X$ has $\kappa = 10^{12}$, which exceeds the precision of a 64-bit float. **You would be computing eigenvectors of a matrix whose small entries are entirely rounding noise.** The SVD works on $X$ directly and never forms the square, which is why every serious implementation uses it.

> **Where this came from.** PCA was invented twice, from the two derivations above, thirty-two years apart. **Karl Pearson** published it in 1901 under the title *On Lines and Planes of Closest Fit to Systems of Points in Space* — the geometric, minimum-reconstruction-error framing, motivated by biometric measurement data, and written before matrix algebra was a standard tool for statisticians (his paper is full of explicit geometry). **Harold Hotelling** rederived it in 1933, from the maximum-variance direction, working on psychological test scores; it is Hotelling who named them "principal components" and who connected the method to eigenvalue problems. That both routes exist, and that Pythagoras is all that separates them, was not obvious to either of them. And underneath both sits the SVD, which **Beltrami** and **Jordan** had discovered in 1873–74 while doing pure geometry, decades before anyone had data to apply it to (Ch. 1 §1.1.3).

> **The story behind the scree plot.** The **scree plot** — eigenvalues sorted descending, and you look for the "elbow" — was named by the psychologist **Raymond Cattell** in 1966. *Scree* is a geological term for the loose rubble that accumulates at the base of a cliff. Cattell's point was visual: the meaningful components form the cliff face, and the noise components form the flat rubble pile at the bottom, and you cut where the cliff ends. It is one of the more honest names in statistics, because it admits the criterion is a judgement about a picture rather than a test. **Parallel analysis** — comparing your eigenvalues against those from randomly permuted data — is the principled version: it asks "would a component this large have appeared by chance if my features were unrelated?"

### Practical notes

- **Standardize** when features have different units; otherwise the largest-unit feature dominates. Standardizing means you're doing eigendecomposition of the *correlation* matrix.
- **Choosing $k$:** cumulative explained variance (90–95%), the scree-plot elbow, or parallel analysis (compare eigenvalues against those from permuted data).
- **Whitening:** divide scores by $\sigma_j$ so all components have unit variance. Useful as a preprocessing step; amplifies noise in small components.
- **Complexity:** $O(\min(n^2d, nd^2))$ exact; use randomized SVD for large $d$.

▸ **The limitation that matters:** PCA is linear. Data on a curved manifold (a Swiss roll) is not compressible by any linear projection. Also, **high variance is not the same as high information** — in a classification problem the discriminative direction may have small variance, which is why LDA (Ch. 22 §22.6) exists.

#### The practical notes, decoded

**Standardizing, and why it silently changes the answer.** PCA maximizes variance, and variance has units — squared units of whatever the column is measured in. Put `salary` (values around $60{,}000$) and `years_experience` (values around $8$) in the same table and the variance of salary is roughly $10^8$ times larger. **Component 1 will be "salary," to nine decimal places, and it will tell you nothing.** Worse, switch salary from dollars to thousands of dollars and PCA returns a different answer — a method whose output depends on your choice of units is not measuring anything about the world.

Standardizing (subtract each column's mean, divide by its standard deviation) forces every feature to variance 1, which makes the units cancel. The text's note that this "means you're doing eigendecomposition of the *correlation* matrix" is exact: **correlation is covariance with the units divided out.**

▸ **The rule: standardize whenever the columns have different units; do not standardize when they share units and the size differences are real.** Pixel intensities are all in the same units and a low-variance pixel  matters less — standardizing there would amplify the corners of the image, which are mostly noise, into principal components.

**Choosing $k$, with the actual numbers.** Cumulative explained variance is the ratio $\frac{\sum_{j\le k}\sigma_j^2}{\sum_{\text{all }j}\sigma_j^2}$. A typical table:

| Component | $\sigma_j^2$ | Share | Cumulative |
|---|---|---|---|
| 1 | $42.0$ | $52.5\%$ | $52.5\%$ |
| 2 | $18.0$ | $22.5\%$ | $75.0\%$ |
| 3 | $9.0$ | $11.3\%$ | $86.3\%$ |
| 4 | $5.0$ | $6.3\%$ | $92.5\%$ |
| 5–80 | $6.0$ total | $7.5\%$ | $100\%$ |

At $k=4$ you have 92.5% of the variance in 4 numbers instead of 80. **That is a 20× compression for a 7.5% loss of spread** — and the discarded 7.5% is spread thinly over 76 directions, which is exactly the signature of noise rather than structure.

**Whitening, decoded.** After projecting, divide each score by $\sigma_j$ so every component has variance exactly 1. The result is a data cloud that is perfectly spherical: no direction is privileged. This helps any downstream method that assumes isotropy — including k-means (§24.4), whose objective is isotropic squared distance. ▸ **The danger is right there in the arithmetic: dividing by $\sigma_j$ means dividing by a small number for the last components, which multiplies their noise up to the same scale as your signal.** Whiten the top few components, not all of them.

**Complexity, decoded.** $O(\min(n^2d,\,nd^2))$ reads: *"whichever of these two is smaller."* Two regimes:
- **Many points, few features** ($n \gg d$): the $nd^2$ branch wins. With $n=10^6$, $d=100$: $10^{10}$ operations. Fine.
- **Few points, many features** ($d\gg n$): the $n^2d$ branch wins. With $n=1000$ gene-expression samples and $d=20{,}000$ genes: $2\times10^{10}$ rather than $4\times10^{11}$. **You work with the $n\times n$ Gram matrix instead of the $d\times d$ covariance** — the same trick that makes kernel methods possible.
- **Randomized SVD** sidesteps both: multiply $X$ by a random $d\times(k{+}p)$ matrix to sketch its column space, then factor the small result. Cost drops to roughly $O(ndk)$. For $k=10$ components of a $10^6\times 10^4$ matrix that is the difference between minutes and never.

**Why PCA cannot unroll a Swiss roll.** A Swiss roll is a 2-D sheet rolled up in 3-D. Two points on opposite sides of the roll are **near in straight-line distance and far along the sheet**. Any linear projection preserves straight-line structure, so it must keep those two points near each other — it cannot possibly unroll anything, because unrolling requires *stretching* the space differently in different places, and "differently in different places" is the definition of nonlinear. ▸ **This one picture is the reason §24.3 exists.**

**"High variance is not high information," concretely.** Imagine measuring people's height (spread: tens of centimetres) and one blood marker that is either $0.31$ or $0.33$ depending on whether they have a disease. The marker has minuscule variance and perfect predictive power. **PCA will discard it first.** PCA is unsupervised — it has never been told what you are trying to predict, so it optimizes spread as a proxy for information, and the proxy fails whenever the signal is small and the noise is large. LDA (linear discriminant analysis) is the supervised repair: it maximizes *between-class* separation relative to within-class spread, and would seize on the marker immediately.

### Probabilistic PCA

Model $x = Wz + \mu + \varepsilon$, $z\sim\mathcal{N}(0,I)$, $\varepsilon\sim\mathcal{N}(0,\sigma^2I)$. Then $x\sim\mathcal{N}(\mu, WW^\top+\sigma^2I)$, and the maximum-likelihood $W$ spans the top-$k$ eigenspace.

▸ **Why this framing matters:** it gives PCA a likelihood (so you can compare models, handle missing data, and use EM), and as $\sigma^2\to0$ it recovers classical PCA exactly. **A VAE (Ch. 19 §19.3) is probabilistic PCA with neural networks replacing the linear maps** — that is the cleanest way to see what a VAE is.

### Kernel PCA

Run PCA in an RKHS: centre the kernel matrix $\tilde K = K - \mathbf{1}_nK - K\mathbf{1}_n+\mathbf{1}_nK\mathbf{1}_n$, then eigendecompose $\tilde K$. Nonlinear components at $O(n^2)$–$O(n^3)$ cost. The out-of-sample projection requires the pre-image problem, which has no exact solution.

#### Probabilistic PCA, decoded

$$x = Wz + \mu + \varepsilon,\qquad z\sim\mathcal{N}(0,I),\qquad \varepsilon\sim\mathcal{N}(0,\sigma^2I)$$

Read aloud: *"a data point is a linear map $W$ applied to a hidden low-dimensional code $z$, plus an overall offset $\mu$, plus isotropic Gaussian noise."*

| Symbol | Read aloud | What it is |
|---|---|---|
| $z$ | "z" | The **latent code** — $k$ numbers, unobserved, that "explain" the point |
| $z\sim\mathcal{N}(0,I)$ | "z is distributed standard normal" | The code is assumed to be a random draw from a plain unit Gaussian |
| $W$ | "W" | The $d\times k$ **loading matrix** that expands the short code into a full data point |
| $\mu$ | "mu" | The overall mean of the data |
| $\varepsilon$ | "epsilon" | Noise, the same size in every direction ($\sigma^2 I$ means "$\sigma^2$ on every diagonal, zero off it") |
| $\sim$ | "is distributed as" | Not equality — "is a random draw from" |

▸ **The conceptual move is that PCA stops being a *procedure* and becomes a *story about how the data was made.*** Classical PCA says "here is an algorithm: form the covariance, take eigenvectors." Probabilistic PCA says "I claim each data point was generated by drawing a short random code and expanding it linearly, with a little noise added." Once you have a generative story, you have a likelihood, and once you have a likelihood you can do everything statistics knows how to do.

**What "$x\sim\mathcal{N}(\mu, WW^\top+\sigma^2I)$" is telling you.** Sum of independent Gaussians is Gaussian, and covariances add. The $Wz$ part contributes $WW^\top$ — a rank-$k$ matrix, since $W$ is $d\times k$ — and the noise contributes $\sigma^2I$, a full-rank but featureless blob. **So the model says: the data's covariance is a $k$-dimensional pancake plus a uniform fuzz of thickness $\sigma$.** That is precisely the picture in this section's opening analogy, now written as an equation.

**Why $\sigma^2\to0$ recovers classical PCA.** Shrink the fuzz to nothing and every point must lie exactly in the span of $W$; the maximum-likelihood fit degenerates to the exact orthogonal projection, which is classical PCA. ▸ **Classical PCA is the zero-noise limit of a probabilistic model** — the same relationship k-means has to Gaussian mixtures in §24.4, and the same relationship a hard argmax has to a softmax. **This pattern — "the crisp classical algorithm is the zero-temperature limit of a probabilistic one" — recurs throughout the book, and each time the probabilistic version buys uncertainty estimates.**

**What the likelihood buys you, concretely:**
- **Missing data.** A row with three missing features is no longer a problem — you marginalize them out. Classical PCA has no principled answer beyond imputing first and hoping.
- **Model comparison.** "Is $k=4$ or $k=7$ better?" becomes a likelihood question you can answer with held-out data, rather than a judgement about a scree plot.
- **EM instead of SVD.** You can fit $W$ by expectation–maximization (§24.4), which is $O(ndk)$ per iteration rather than $O(nd^2)$ — a real win when $d$ is enormous.
- **A generative model.** You can *sample* new data: draw $z$, push it through $W$, add noise.

▸ **And then the punchline: replace $Wz + \mu$ with a neural network and replace the exact posterior with a learned approximate one, and you have a VAE (Ch. 19 §19.3).** The encoder is the posterior $p(z\mid x)$; the decoder is the generative map; the "reconstruction plus KL" loss is the evidence lower bound (ELBO) for exactly this model. **If VAEs have ever felt arbitrary, this is the sentence to hold onto: a VAE is probabilistic PCA with the linear maps swapped for networks.**

#### Kernel PCA, decoded

**The idea in one line: if your data is not linearly separable where it lives, do PCA somewhere else.** The kernel trick (Ch. 22) lets you compute inner products in a vastly higher-dimensional feature space without ever constructing a point in it. Since PCA can be written entirely in terms of inner products between data points, it transplants directly.

- $K$ — the **kernel matrix**, $n\times n$, where $K_{ij}$ is the similarity between points $i$ and $j$ in the implicit feature space. For the Gaussian kernel, $K_{ij} = \exp(-\|x_i-x_j\|^2/2\sigma^2)$.
- **RKHS** stands for **reproducing kernel Hilbert space** — the name for the implicit space the kernel corresponds to. You never visit it; you only ever compute inner products in it.
- $\mathbf{1}_n$ — the $n\times n$ matrix with every entry $1/n$. The four-term formula $\tilde K = K - \mathbf{1}_nK - K\mathbf{1}_n + \mathbf{1}_nK\mathbf{1}_n$ is **centring, done in the feature space you cannot see.** In ordinary PCA you centre by subtracting the mean from every point; here you cannot touch the points, so you subtract the row means, the column means, and add back the grand mean (which you subtracted twice). It is exactly the algebra of centring, expressed only through inner products.

**The costs, honestly.** The matrix is $n\times n$ regardless of how many features you had, so memory is $O(n^2)$ and eigendecomposition is $O(n^3)$. At $n=50{,}000$ the kernel matrix alone is 20 GB in double precision. **Kernel PCA is a small-$n$ method, and that is why it lost ground to the neighbour-graph methods of §24.3, which only need each point's nearest few neighbours.**

**The pre-image problem, decoded.** Ordinary PCA lets you go backwards: take a low-dimensional score and reconstruct an approximate data point. In kernel PCA the reconstruction lives in the implicit feature space, and there may be **no point in your original space that maps to it** — the feature space is much larger than the image of your data under the feature map, so most of it corresponds to nothing. Finding the closest actual point is called the pre-image problem; it is solved approximately by iterative optimization, and it is why kernel PCA is used for analysis rather than as a preprocessing step you invert later.

---

## 24.2 The other linear decompositions

### ICA

PCA finds **uncorrelated** components; ICA finds **statistically independent** ones. Independence is strictly stronger than uncorrelatedness except for Gaussians.

▸ **The central insight:** by the CLT, a mixture of independent sources is *more Gaussian* than the sources. So maximize **non-Gaussianity** to unmix. Measure it with negentropy $J(y)=H(y_{\text{gauss}})-H(y)$, approximated by $\left(\mathbb{E}[G(y)]-\mathbb{E}[G(\nu)]\right)^2$ with $G(u)=\log\cosh(u)$ or $-e^{-u^2/2}$; kurtosis is the simplest but is outlier-sensitive.

▸ **ICA fails entirely if the sources are Gaussian** — a rotation of independent Gaussians is still independent Gaussians, so the model is unidentifiable. Also, sign, scale, and ordering of components are unidentifiable in general.

Classic application: blind source separation (the cocktail-party problem), EEG/fMRI artifact removal.

#### ICA, decoded — why "more Gaussian" is the clue

**The setup.** Three people talk simultaneously in a room with three microphones. Each microphone records a different *mixture* of the three voices — mic 1 hears mostly Ana but some Ben, and so on. You have the three recordings; you want the three voices. Formally, you observe $x = As$ where $s$ is the vector of unknown sources and $A$ is an unknown mixing matrix, and you want to recover both.

**Why this looks impossible and isn't.** You have $n$ equations and $2n$ unknowns per time step — there are infinitely many $(A, s)$ pairs consistent with your data. The extra assumption that breaks the tie is **statistical independence**: whatever Ana says is unrelated to whatever Ben says.

**Uncorrelated versus independent — the distinction the whole method turns on.**

- **Uncorrelated** means $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$: no *linear* relationship.
- **Independent** means the joint distribution factorizes, $p(x,y) = p(x)p(y)$: **no relationship of any kind**, linear or otherwise.

Independence implies uncorrelatedness. The converse fails. Concrete counterexample: let $X$ be $-1$, $0$, or $+1$ with equal probability, and let $Y = X^2$. Then $\mathbb{E}[XY] = \mathbb{E}[X^3] = 0 = \mathbb{E}[X]\mathbb{E}[Y]$ — **perfectly uncorrelated.** But knowing $X$ tells you $Y$ exactly. ▸ **PCA can only see the covariance matrix, so it can only ever deliver uncorrelated components. Independence requires looking at higher moments, and that is the entire difference between the two methods.**

**Why non-Gaussianity is the signal, via the central limit theorem.** The CLT says a sum of many independent random variables tends toward a Gaussian. A microphone recording is a *sum* of the sources. Therefore:

▸ **Each mixture is more Gaussian than any of the sources that made it. So if you search over unmixing directions for the one whose output looks *least* Gaussian, you are searching for an original source.** That is a  lovely inversion: the CLT is normally used to argue that averages become Gaussian, and ICA runs it backwards as a detector.

> **Analogy.** Mix all your paints together and you get brown. Every mixture is browner than its ingredients. So if you are trying to recover the original colours from a set of muddy mixtures, the right search criterion is "find the combination that is as *un*-brown as possible." Gaussian is the brown of probability distributions — the thing everything tends toward when combined.

**Negentropy, decoded.** $J(y) = H(y_{\text{gauss}}) - H(y)$ reads: *"the entropy of a Gaussian with the same variance, minus the entropy of $y$."* A foundational result of information theory is that **among all distributions with a given variance, the Gaussian has the largest entropy.** So $J(y) \ge 0$ always, with equality only when $y$ *is* Gaussian. **Negentropy is a distance from Gaussianity that is zero exactly at Gaussian and positive otherwise** — precisely the objective you want to maximize.

Entropy is expensive to estimate from samples, so in practice one uses the approximation $\big(\mathbb{E}[G(y)] - \mathbb{E}[G(\nu)]\big)^2$ where $\nu$ is a standard Gaussian variable and $G$ is a nonlinear "contrast function." Reading the choices:

- $G(u) = \log\cosh(u)$ — grows roughly like $\lvert u\rvert$ for large $u$, so it does not over-react to a single extreme value. **The robust default.**
- $G(u) = -e^{-u^2/2}$ — saturates completely, even more robust, good for very heavy-tailed sources.
- **Kurtosis** ($\mathbb{E}[y^4]-3$ for unit variance) is the simplest measure of non-Gaussianity, but a fourth power means one outlier at $y=10$ contributes $10{,}000$ — a single corrupted sample can dominate the entire objective.

**Why Gaussian sources kill the method.** Take two independent standard Gaussians and rotate the pair by any angle. The result is *still* two independent standard Gaussians — the isotropic Gaussian is the one distribution that looks identical from every direction. ▸ **So if the sources are Gaussian there is no "least Gaussian direction"; every direction is equally Gaussian, and the mixing matrix is unrecoverable in principle.** This is not a weakness of any particular algorithm; it is an identifiability result, and no cleverness escapes it. At most **one** source may be Gaussian.

**The three inherent ambiguities**, and why they rarely matter:
- **Scale.** Doubling a source and halving the corresponding column of $A$ gives identical data. Fixed by convention: force unit variance.
- **Sign.** Flipping a source and its column also gives identical data. An audio waveform inverted still sounds the same; an EEG component inverted is still the same artifact.
- **Order.** Nothing distinguishes "source 1" from "source 3." Unlike PCA, whose components come sorted by variance, **ICA components arrive in arbitrary order** — so "the first independent component" is a meaningless phrase, and you must sort them yourself by some criterion you care about.

> **Where this came from.** ICA came out of a neurophysiological question, not a statistical one. **Jeanny Hérault and Christian Jutten**, working in Grenoble in the mid-1980s, were trying to model how a nervous system could decode joint position from muscle sensors that each report a *mixture* of contraction and stretch information — and they built a neuromimetic network to unmix the signals. The problem was formalized and named by **Pierre Comon** in 1994, and given its fast, practical algorithm by **Aapo Hyvärinen and Erkki Oja** in Helsinki in the late 1990s (FastICA), with **Anthony Bell and Terrence Sejnowski**'s Infomax approach arriving from the neural-network side in 1995. The **cocktail-party problem** is older still and comes from psychoacoustics: **Colin Cherry** coined the phrase in 1953 while studying how a human listener manages to attend to one conversation in a crowded room — a question about attention, which is exactly where machine learning would pick the word up again half a century later.

### NMF

$X\approx WH$ with $W,H\ge0$. Non-negativity forces a **parts-based, additive** decomposition (no cancellation), which is why NMF on face images yields nose/eye/mouth components while PCA yields ghostly whole-face eigenfaces. Used for topic modelling and spectral unmixing. Non-convex; solved by multiplicative updates or alternating least squares.

### Random projection

By Johnson–Lindenstrauss (Ch. 1 §1.2), projecting onto $k = O(\log n/\epsilon^2)$ random directions preserves all pairwise distances to within $1\pm\epsilon$ — **with no reference to the data at all.** Extremely cheap and a good baseline before any learned method.

#### NMF, decoded — why non-negativity produces parts

$X \approx WH$ with $W, H \ge 0$ reads: *"approximate the data matrix as a product of two matrices, both of which are constrained to contain no negative numbers."*

Concretely, with $X$ being $n\times d$ (documents × words), $W$ is $n\times k$ and $H$ is $k\times d$:

- **Each row of $H$ is a "part"** — a pattern over the original features. In topic modelling, a row of $H$ is a topic: a non-negative weight for every word.
- **Each row of $W$ says how much of each part is in that data point.** Document 17 is $0.6\times$ topic 3 plus $0.3\times$ topic 8.
- **The reconstruction is $x_i \approx \sum_k W_{ik}H_{k:}$** — a **non-negatively weighted sum of parts.**

▸ **The entire behavioural difference from PCA comes from one word: cancellation.** PCA's components can be negative, so it is free to build a face by saying "take this ghostly average face, then *subtract* 0.4 of this other ghostly face." Reconstruction by subtraction is extremely efficient — it is why PCA is optimal in reconstruction error (Ch. 1 §1.1.3) — but it means no individual component has to correspond to anything. NMF cannot subtract. **Every part must be something you would actually want to add**, and the only way to build a face by pure addition is to have parts that are nose-shaped, eye-shaped, mouth-shaped.

> **Analogy.** PCA is a sculptor with a block of marble: it removes material, and no intermediate state resembles a body part. NMF is building with Lego: everything you place stays, so the pieces themselves have to look like things. **Additive-only construction forces interpretable components, and it costs you reconstruction accuracy** — the Lego model is a worse likeness than the sculpture.

**Why it is harder to solve than PCA.** The objective $\|X - WH\|_F^2$ is convex in $W$ with $H$ fixed, and convex in $H$ with $W$ fixed — but **not convex in both together.** So there is no closed-form answer and no uniqueness: different initializations give different, equally valid decompositions. The standard solvers alternate — hold $H$, solve for $W$; hold $W$, solve for $H$ — either by multiplicative update rules (which preserve non-negativity automatically, since you only ever multiply by non-negative factors) or by non-negative least squares. ▸ **Run NMF twice with different seeds and compare: components that survive both runs are real, and components that do not are artifacts of the initialization.** This is the standard reliability check, and it has no analogue in PCA because PCA has a unique answer.

> **Where this came from.** NMF was published first by **Pentti Paatero and Unto Tapper** in 1994 under the name **positive matrix factorization**, in the environmental-science literature — they were doing *source apportionment* for atmospheric particulates: given chemical measurements at monitoring stations, work out which pollution sources contributed how much. The non-negativity was not a mathematical preference but a physical fact, since a smokestack cannot emit a negative quantity of sulphate. The method reached the machine-learning community through **Daniel Lee and H. Sebastian Seung**'s 1999 *Nature* paper, whose faces-decompose-into-facial-parts figure is one of the most reproduced images in the field. ▸ **A constraint imposed by chemistry turned out to be the mechanism that produces interpretability.**

#### Random projection, decoded — the baseline that embarrasses everything else

**The procedure, in full:** generate a $d\times k$ matrix $R$ of random numbers (Gaussian, or even just $\pm1$), compute $XR$, and scale. That is the entire method. **You do not look at the data.** There is no fitting, no iteration, no hyperparameter beyond $k$.

And by Johnson–Lindenstrauss (Ch. 1 §1.1.5), all pairwise distances survive to within $1\pm\epsilon$ as long as $k = O(\log n/\epsilon^2)$.

**Real numbers.** With $n = 100{,}000$ points and $\epsilon = 0.2$ (all distances correct to ±20%): $\log n = 11.5$, so $k$ on the order of a few hundred suffices — and **it does not matter whether those points started in 5,000 dimensions or 5 million.** The starting dimension never appears in the bound.

▸ **Why you should actually run this before anything else.** It costs one matrix multiply. If random projection to 300 dimensions gives you 95% of the downstream performance of a carefully tuned learned embedding, you have learned something extremely important: **your problem was about distances, and distances were free.** If it gives you 40%, you have learned that your method is  finding structure and you now have an honest floor to measure against. Either answer is worth more than the twenty minutes it costs.

**Where it beats PCA:** no need to compute a covariance or an SVD (so it works on streaming data, and on matrices too large to hold); it is data-independent, so the same projection can be shared across machines and applied to data that has not arrived yet; and it is the mathematical basis of locality-sensitive hashing and much of what approximate nearest-neighbour libraries do internally.

**Where it loses:** for a *fixed* small $k$ (say 10), PCA's directions are chosen to be the best possible and random ones are not, so PCA wins decisively in the very-low-dimensional regime. Random projection's advantage begins where $k$ is in the hundreds and $n$ or $d$ is large enough that fitting anything is painful.

---

## 24.3 Nonlinear embedding: t-SNE and UMAP

### t-SNE

**Goal:** preserve *local neighbourhoods*, not global distances.

1. In high-dim, define conditional similarities with a Gaussian kernel, symmetrize:
$$p_{j|i}=\frac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)},\qquad p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n}$$
Each $\sigma_i$ is set by binary search so the perplexity $2^{H(P_i)}$ matches a target (5–50) — an adaptive bandwidth that handles varying density.

2. In low-dim, use a **Student-$t$ with one degree of freedom** (a Cauchy):
▸ $$q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}}$$

3. Minimize $\mathrm{KL}(P\|Q)$ by gradient descent.

▸ **Why the heavy tail matters (the "crowding problem"):** the volume of a ball grows as $r^d$, so a point in 50 dimensions can have far more roughly-equidistant neighbours than can fit at similar distance in 2-D. With a Gaussian in the low-dim space, moderate distances would be over-penalized and everything would collapse to the centre. The $t$-distribution's heavy tail lets moderately-dissimilar points sit far apart cheaply, which is exactly what creates the visible gaps between clusters.

▸ **The forward-KL choice explains the failure mode.** $\mathrm{KL}(P\|Q)$ heavily penalizes $q_{ij}$ small where $p_{ij}$ is large (neighbours pushed apart) but barely penalizes $q_{ij}$ large where $p_{ij}$ is small (distant points placed nearby). **So local structure is faithful and global structure is not.**

**What you must not read off a t-SNE plot:**
- Cluster *sizes* — meaningless (density is equalized by the per-point $\sigma_i$).
- Distances *between* clusters — meaningless.
- Apparent clusters at low perplexity — t-SNE will produce clusters in pure Gaussian noise.
Always run several perplexities before believing anything.

#### Reading the t-SNE similarities in plain English

$$p_{j|i}=\frac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}$$

Read aloud: *"p of j given i equals e-to-the-minus distance-squared over two sigma-i-squared, divided by the same quantity summed over every other point."*

| Piece | Read aloud | What it is |
|---|---|---|
| $\|x_i-x_j\|^2$ | "norm of x-i minus x-j, squared" | Squared distance between the two points in the **original** high-dimensional space |
| $\exp(-\text{dist}^2/2\sigma_i^2)$ | "e to the minus…" | A **Gaussian bump** centred on point $i$: near points get a value near 1, far points near 0 |
| $\sigma_i$ | "sigma-i" | The **bandwidth** — how wide $i$'s bump is. Note the subscript: **every point gets its own.** |
| The denominator | "sum over k not equal to i" | Normalization, so the numbers for point $i$ add to 1 |
| $p_{j\mid i}$ | "p of j given i" | **"If point $i$ had to pick one neighbour at random, weighted by closeness, the chance it picks $j$."** |

▸ **The conceptual move: distances are replaced by probabilities.** t-SNE never tries to preserve "these points are 4.7 units apart." It tries to preserve "if this point picked a friend, it would probably pick that one." That is a much weaker, much more achievable requirement — and it is why t-SNE preserves neighbourhoods while destroying distances.

**Real numbers.** Point $i$ with $\sigma_i = 1$, and three candidate neighbours at distances $0.5$, $2$, and $5$:

- $\exp(-0.25/2) = 0.882$
- $\exp(-4/2) = 0.135$
- $\exp(-25/2) = 0.0000037$

Normalized: $0.867$, $0.133$, $0.0000036$. **The point at distance 5 is, for all purposes, invisible.** A Gaussian falls off so fast that beyond about $3\sigma$ nothing exists. This is exactly why the low-dimensional kernel has to be different, and why that choice is the whole method.

**The symmetrization, $p_{ij} = \frac{p_{j|i}+p_{i|j}}{2n}$.** The conditional version is asymmetric: a lonely outlier's nearest neighbour might be a point deep inside a dense cluster, and that cluster point would never reciprocate. Averaging the two directions and dividing by $n$ produces a proper joint distribution over pairs. ▸ **The practical effect is that no point can be ignored: every point's total $p$ is guaranteed to be at least $\frac{1}{2n}$**, which prevents outliers from being placed arbitrarily since they still exert a pull on someone.

**Perplexity, decoded.** $2^{H(P_i)}$ where $H$ is the entropy of point $i$'s neighbour distribution. Entropy measures how spread out that distribution is; $2^H$ converts it into a **count**:

- If $i$ splits its attention evenly over exactly 30 neighbours, $H = \log_2 30$ and $2^H = 30$.
- If $i$ puts everything on one neighbour, $H=0$ and $2^H = 1$.

▸ **So perplexity is "the effective number of neighbours each point is paying attention to"** — the same "effective count" construction as stable rank in Ch. 1 §1.1.3 and as perplexity in language modelling (Ch. 1 §1.4.2). It is the same formula each time, and it always means "how many things is this distribution really spread over."

**Why the binary search for $\sigma_i$ matters more than it sounds.** You specify perplexity; the algorithm then solves for whatever $\sigma_i$ achieves it, separately for each point. In a dense region a small $\sigma$ already reaches 30 effective neighbours; in a sparse region $\sigma$ must be large. **The result is that density is equalized — every point ends up with the same number of effective neighbours regardless of how crowded its neighbourhood actually was.** This is what makes t-SNE robust to varying density, and it is *also* precisely why cluster sizes in the output are meaningless: the algorithm deliberately erased the density information that would have made them meaningful.

#### The crowding problem, and why the heavy tail solves it

**The setup, with numbers.** In $d$ dimensions, the volume of a ball of radius $r$ scales as $r^d$. So the number of points that can sit at roughly distance $r$ from a centre while staying spread out from each other grows explosively with $d$.

Concretely: in **2 dimensions** you can place about 6 points around a centre such that all of them are the same distance from it and from their neighbours (a hexagon). In **3 dimensions**, about 12. In **50 dimensions**, the number of roughly-equidistant, roughly-mutually-distant points is astronomically larger — this is the Johnson–Lindenstrauss counting argument of Ch. 1 §1.1.5 in another costume.

▸ **So a point with 50 equally-close neighbours in 50-D has no way to keep them all equally close in 2-D. There is simply not enough room.** Some of them must be placed further away. The question is what the loss function does about that.

**With a Gaussian in the low-dimensional space:** placing a moderate-similarity neighbour at moderate distance produces a $q_{ij}$ that has already collapsed to near-zero (see the numbers above — $\exp(-12.5)$). The KL penalty for "$p$ is large but $q$ is tiny" is severe, so the optimizer responds by hauling everything inward, trying to fit all those neighbours close. **The result is the classic failure: one indistinguishable blob.**

**With the Student-$t$ (Cauchy), $q_{ij} \propto (1+\|y_i-y_j\|^2)^{-1}$:**

| Distance $\|y_i-y_j\|$ | Gaussian $\exp(-\text{d}^2/2)$ | Student-$t$ $(1+\text{d}^2)^{-1}$ | Ratio |
|---|---|---|---|
| $1$ | $0.607$ | $0.500$ | $0.8\times$ |
| $3$ | $0.011$ | $0.100$ | $9\times$ |
| $5$ | $3.7\times10^{-6}$ | $0.038$ | $10{,}000\times$ |
| $10$ | $1.9\times10^{-22}$ | $0.0099$ | $5\times10^{19}\times$ |

▸ **Read the last column. At distance 10 the heavy-tailed kernel assigns a similarity twenty billion billion times larger than the Gaussian would.** That means placing a moderately-similar point far away is *cheap*. The optimizer no longer has to cram; it can push mildly-dissimilar points out to arm's length and pay almost nothing. **And the empty space it creates by doing so is exactly the visible gap between clusters in every t-SNE plot you have ever seen.** The gaps are not measurements of anything. They are the byproduct of a kernel chosen so that pushing things apart is affordable.

> **Analogy.** You are seating 200 wedding guests in a hall that is far too small. With a rule that says "anyone who knows each other must sit within one metre," the whole party ends up in a scrum in the centre. Change the rule to "friends within one metre, everyone else just needs to be *somewhere*," and suddenly the friendship groups form recognizable tables with aisles between them. **The aisles' widths mean nothing. They are what is left over.**

#### Why the KL direction determines the failure mode

$\mathrm{KL}(P\|Q) = \sum_{ij}p_{ij}\log\frac{p_{ij}}{q_{ij}}$ — read: *"sum over all pairs of p times log p over q."* The weighting by $p_{ij}$ is the entire story:

- **$p_{ij}$ large, $q_{ij}$ small** (true neighbours placed far apart): the term is $\text{large}\times\log(\text{large}) $ — **a huge penalty.**
- **$p_{ij}$ small, $q_{ij}$ large** (distant points placed adjacent): the term is $\text{tiny}\times\log(\text{tiny})$ — **almost no penalty at all.**

▸ **The loss function only ever punishes you for separating things that belong together. It essentially never punishes you for joining things that don't.** So local structure is a hard constraint and global structure is free to be whatever the optimizer finds convenient. **Every documented misuse of a t-SNE plot is a consequence of this one asymmetry**, and it is the same forward-versus-reverse KL asymmetry that governs mode-covering versus mode-seeking behaviour in variational inference (Ch. 1 §1.4.3).

**The four things you must not read off the plot, restated with the mechanism:**

| Do not read | Because |
|---|---|
| Cluster **sizes** | The per-point $\sigma_i$ binary search deliberately equalizes density |
| Distances **between** clusters | Reverse-direction KL errors are unpenalized, so between-cluster geometry is unconstrained |
| Cluster **shapes** | The layout is a local optimum of a non-convex objective; shapes vary run to run |
| Apparent clusters at low perplexity | At perplexity 5, t-SNE will produce crisp clusters from **pure Gaussian noise** |

▸ **That last row deserves emphasis: t-SNE run on random noise produces a picture that looks exactly like a discovery.** The only defence is to run several perplexities (5, 30, 50) and several random seeds, and to trust only what persists across all of them.

> **Where this came from.** **SNE (stochastic neighbour embedding)** was introduced by **Geoffrey Hinton and Sam Roweis** in 2002, using a Gaussian in both spaces — and suffering exactly the crowding collapse described above. **Laurens van der Maaten**, working with Hinton, published **t-SNE** in 2008; the "t" is the single change that made the method work, and it was motivated directly by the crowding argument. Speed came later: van der Maaten's 2014 Barnes–Hut variant borrowed the **Barnes–Hut algorithm from computational astrophysics** — invented by **Josh Barnes and Piet Hut in 1986** to simulate gravitational N-body problems by approximating a distant cluster of stars as a single point mass. The repulsive forces between distant points in a t-SNE layout are mathematically the same problem as gravity between distant stars, so the astronomers' quadtree carries over unchanged and turns $O(n^2)$ into $O(n\log n)$. **A method for simulating galaxies is why you can embed a million cells.**

> **The story behind the noise warning.** The best-known demonstration that t-SNE plots can be over-read is the 2016 interactive article *How to Use t-SNE Effectively* by **Martin Wattenberg, Fernanda Viégas, and Ian Johnson**, published in *Distill*. Its most quoted figure shows t-SNE run on pure high-dimensional Gaussian noise at low perplexity, producing crisp, convincing, entirely fictitious clusters. The piece is widely credited with changing how the method is reported in biology and single-cell genomics, where t-SNE plots had by then become a standard figure in published papers.

### UMAP

Builds a fuzzy $k$-NN graph with a locally-adaptive metric (theoretically motivated by Riemannian geometry and category theory), then optimizes a low-dimensional graph layout with a cross-entropy objective and negative sampling.

**Versus t-SNE:** much faster ($O(n^{1.14})$ vs $O(n\log n)$ with Barnes–Hut, but with far better constants), better preservation of global structure, and it supports a `transform` for new points. **Key parameters:** `n_neighbors` (local↔global balance) and `min_dist` (how tightly points pack).

▸ **Both are visualization tools, not preprocessing.** Do not cluster on t-SNE/UMAP coordinates and report the clusters as findings — the embedding manufactures separations. Cluster in the original or PCA space; use the embedding only to *look*.

#### UMAP, decoded

**What "fuzzy k-NN graph with a locally-adaptive metric" means, stripped of the theory.** For each point, find its $k$ nearest neighbours. Rather than recording binary "is a neighbour / isn't," record a **membership strength** between 0 and 1 that decays with distance — that is the "fuzzy" part. The "locally-adaptive metric" is UMAP's counterpart of t-SNE's per-point $\sigma_i$: each point's distances are rescaled so that its nearest neighbour sits at distance 0, and the decay rate is set so the memberships sum to a target. **Same purpose, same effect: density is normalized away, point by point.**

Then you have two fuzzy graphs — the one from the high-dimensional data and the one implied by your current 2-D layout — and you move the 2-D points to make the graphs agree, using cross-entropy as the measure of agreement.

▸ **The one substantive difference from t-SNE's objective: UMAP's cross-entropy has a term that penalizes $q$ being large where $p$ is small.** t-SNE's forward KL does not (see above). That extra term is a  repulsion between non-neighbours, and it is the main reason UMAP's global arrangement is more trustworthy — though "more trustworthy" is a long way from "trustworthy."

**Negative sampling, decoded.** Computing the repulsion between all $n^2$ pairs is impossible at scale. Instead, for each edge you actually optimize, sample a handful of random non-neighbours and push against those. The expected force is right; the variance is high; the cost is $O(n)$ rather than $O(n^2)$. ▸ **This is the same trick as word2vec's negative sampling (Ch. 10) and the same idea as the negatives in contrastive learning (Ch. 25): approximate a sum over everything by a few random draws from everything.**

**The two parameters, and what they actually control.**

| Parameter | Typical | Turn it up and… |
|---|---|---|
| `n_neighbors` | 5–50 (default 15) | The graph reaches further, so more **global** structure is preserved and fine local detail is smoothed away. Small values give many small islands. |
| `min_dist` | 0.0–0.99 (default 0.1) | Points are forbidden from packing tighter than this, so clusters look **looser and more spread**. Set it to 0 for maximally crisp clumps, which is what you want if you are going to look at cluster membership rather than shape. |

**The `transform` method matters more than the speed.** t-SNE fits an embedding *of a fixed set of points*; there is no function to apply to a new point, so adding one row means re-running everything and getting a different picture. UMAP learns something you can apply to new data. ▸ **That is the difference between a figure and a component of a pipeline** — although the warning above still stands, because the coordinates a `transform` produces are just as geometrically untrustworthy as the ones the fit produced.

**Why "do not cluster on the embedding" is not a stylistic preference.** Both methods have an objective that *rewards* producing separated blobs — that is what the heavy tail and the repulsion terms are for. Running a clustering algorithm on the output and reporting the clusters is therefore circular: you have run an algorithm designed to manufacture separation, then used a second algorithm to detect the separation it manufactured, and reported the detection as a finding about the data. ▸ **Cluster in the original space (or in the top PCA components, which is a faithful linear summary), then colour the embedding by the cluster labels.** The picture is then a visualization of a result rather than the source of it.

> **Where this came from.** **UMAP** was published in 2018 by **Leland McInnes, John Healy, and James Melville**. Its derivation is unusually abstract for a practical tool — it is built on fuzzy simplicial sets and borrows language from category theory and Riemannian geometry — and the honest situation is that the algorithm's success and its theoretical derivation are only loosely coupled: several subsequent analyses have found that UMAP and t-SNE, despite very different-looking derivations, are doing closely related things, and that the practical differences come down to initialization and the specific attraction/repulsion balance rather than to the topology. McInnes also implemented and popularized **HDBSCAN** (§24.4), which is why the two are so often used together — the pairing is a piece of software history as much as a methodological recommendation.

---

## 24.4 Clustering

### k-means

▸ $$\min_{\{S_k\},\{\mu_k\}}\ \sum_{k=1}^{K}\sum_{x\in S_k}\|x-\mu_k\|^2$$

**Lloyd's algorithm:** alternate (a) assign each point to its nearest centroid, (b) set each centroid to the mean of its members. Each step is non-increasing in the objective, and the number of partitions is finite, so it **converges in finite time — to a local optimum.**

**k-means++ initialization:** choose the first centre uniformly; choose each subsequent centre with probability $\propto D(x)^2$ (squared distance to the nearest existing centre). ▸ Gives an $O(\log K)$-competitive expected cost guarantee and, in practice, is what makes k-means reliable. Always use it.

**Assumptions, all of which are failure modes:**
- Clusters are **spherical and equally sized** (the objective is isotropic squared distance).
- $K$ is known.
- Every point belongs to exactly one cluster (hard assignment).
- Sensitive to outliers (means are not robust) — use k-medoids/PAM for robustness.

▸ **k-means is GMM with isotropic, equal-weight, shared covariance $\sigma^2I$ in the limit $\sigma^2\to0$**, where the soft responsibilities become hard assignments. That is the cleanest statement of its relationship to the next section.

**Choosing $K$:** elbow on inertia (unreliable — inertia always decreases), **silhouette score**, gap statistic, or a downstream task metric. In practice, $K$ is usually determined by the application, not the data.

**Mini-batch k-means** for large $n$; **spherical k-means** (cosine distance) for text/embeddings.

#### Reading the k-means objective in plain English

$$\min_{\{S_k\},\{\mu_k\}}\ \sum_{k=1}^{K}\sum_{x\in S_k}\|x-\mu_k\|^2$$

Read aloud: *"choose the K groups and the K centres that minimize the total squared distance from every point to the centre of its own group."*

| Piece | Read aloud | What it is |
|---|---|---|
| $K$ | "K" | How many clusters you decided to look for — **you must supply this** |
| $S_k$ | "S-k" | The **set** of points assigned to cluster $k$ |
| $\mu_k$ | "mu-k" | Cluster $k$'s **centre**, a point in the same space as the data |
| $\{S_k\},\{\mu_k\}$ | "the S's and the mu's" | Curly braces mean "the whole collection of them" — we optimize over **both** at once |
| $\sum_{x\in S_k}$ | "sum over x in S-k" | "Add this up over every point assigned to cluster $k$" |
| $\|x-\mu_k\|^2$ | "norm of x minus mu-k, squared" | Squared straight-line distance from the point to its centre |

The quantity being minimized has a name: **inertia**, or within-cluster sum of squares.

▸ **The reason this is hard is the joint optimization.** If someone hands you the assignments, computing the best centres is trivial (take means). If someone hands you the centres, computing the best assignments is trivial (nearest centre). You have neither, and the two depend on each other. **k-means is  NP-hard** — you cannot brute-force it, and the number of ways to partition $n$ points into $K$ groups is astronomically larger than the number of atoms in the observable universe for even modest $n$.

**Lloyd's algorithm is the obvious response: guess one, solve the other, repeat.**

1. **Assign:** each point goes to its nearest centre. (Best assignments given the centres.)
2. **Update:** each centre moves to the mean of its members. (Best centres given the assignments.)
3. Repeat until nothing changes.

**Why it always terminates.** Step 1 cannot increase the objective — moving a point to a *closer* centre reduces its contribution. Step 2 cannot increase it either, because the mean is provably the point minimizing the sum of squared distances to a set (differentiate $\sum_i (x_i - \mu)^2$ and you get $\mu = \bar x$). So the objective decreases or stays flat at every step. Since there are only finitely many possible assignments and the objective never goes back up, **you can never revisit a configuration, so you must stop.**

▸ **Note what this does and does not promise. It promises you will stop. It does not promise you stopped anywhere good.** Lloyd's algorithm reaches a *local* optimum, and the local optima can be arbitrarily bad. This is the single most important practical fact about k-means, and it is what §24.4's initialization discussion exists to address.

> **Analogy.** You are placing $K$ mobile phone masts to serve a city, wanting everyone close to a mast. Step 1: everyone connects to whichever mast is nearest. Step 2: each mast relocates to the geographic centre of its own subscribers. Iterate. Everybody's connection improves at each round and eventually nothing moves. But if you happened to drop all your masts in one suburb, they will politely spread out within that suburb and never discover the other half of the city.

#### k-means++, decoded, with numbers

**The failure it fixes.** Suppose your data has 3 well-separated clusters and you initialize 3 centres uniformly at random from your data. If two of them land in the same cluster, Lloyd's algorithm will split that cluster in half and merge the other two — and it will never escape, because no single reassignment or centre move improves things. **The result can be several times worse than optimal, and it will look perfectly converged.**

**The fix:** choose centres one at a time, and make each new one *far from the ones you already have*.

1. First centre: pick a data point uniformly at random.
2. For every point, compute $D(x)$ = distance to the nearest centre chosen so far.
3. Pick the next centre randomly, with probability proportional to $D(x)^2$.
4. Repeat until you have $K$.

**Real numbers.** Three points at distances $1$, $3$, and $10$ from the existing centre. Their $D^2$ values are $1$, $9$, $100$. Normalized: $0.9\%$, $8.2\%$, $90.9\%$. ▸ **The far point is a hundred times more likely to be chosen than the near one** — but the near one is not impossible, and that matters: pure "pick the furthest point" is deterministic and would seize on outliers every single time. **The squaring makes it greedy; the randomness makes it robust.**

**The guarantee.** k-means++ initialization gives an expected cost within $O(\log K)$ of the true optimum — **before you have run a single iteration of Lloyd's algorithm.** With $K=10$ that is a factor of about 2.3, versus no bound at all for uniform initialization. It costs $K$ extra passes over the data, which is negligible. **There is no situation in which you should not use it, and it has been the default in every serious library for over a decade.**

#### The four assumptions, and how each one fails

| Assumption | What breaks it | What it looks like |
|---|---|---|
| **Spherical, equal-sized clusters** | The objective is $\|x-\mu\|^2$ — isotropic. It has no way to express "elongated in this direction." | An elongated cluster gets chopped crosswise into two round pieces |
| **$K$ is known** | Nothing in the objective can tell you $K$ — inertia decreases monotonically with $K$, hitting exactly 0 at $K=n$ | The elbow plot has no elbow |
| **Hard assignment** | A point exactly between two clusters is forced to pick one, and contributes fully to that centre's mean | Boundary points drag centroids toward each other |
| **Means are not robust** | One point at distance 1000 contributes $10^6$ to the objective, more than a thousand ordinary points | A single outlier pulls a centroid out of its cluster, or claims a cluster of its own |

▸ **All four failures are the same failure: the objective is a sum of squared Euclidean distances, and that formula encodes a specific belief about what a cluster is.** If your clusters are not round blobs of similar size, you have not found a bug — you have used a method whose definition of "cluster" disagrees with yours. **k-medoids / PAM** (Partitioning Around Medoids) fixes the robustness leg by using actual data points as centres and absolute rather than squared distance; §24.4's later methods fix the others.

**Why $\sigma^2\to0$ turns a GMM into k-means.** In a Gaussian mixture with shared isotropic covariance $\sigma^2I$, the responsibility of component $k$ for point $i$ is proportional to $\exp(-\|x_i-\mu_k\|^2/2\sigma^2)$. As $\sigma^2$ shrinks, the exponent's denominator shrinks, so tiny differences in distance become enormous differences in the exponent. Numbers: two centres at distances $2.0$ and $2.1$.

| $\sigma^2$ | Responsibility of the nearer centre |
|---|---|
| $1.0$ | $0.551$ |
| $0.1$ | $0.756$ |
| $0.01$ | $0.982$ |
| $0.001$ | $\approx 1.000$ |

▸ **The soft assignment hardens into an argmax.** This is the same softmax-becomes-argmax limit that governs temperature in language models (Ch. 1 §1.3.4) and in DINO's sharpening (Ch. 25 §25.4). **k-means is a Gaussian mixture at zero temperature**, which is simultaneously the cleanest statement of the relationship and an explanation of all four assumptions above: k-means inherits "spherical" from the isotropic covariance, "equally sized" from the shared mixing weight, and "hard" from the limit.

**Choosing $K$, honestly.** Inertia always decreases as $K$ grows, so "pick the $K$ that minimizes inertia" returns $K=n$. The elbow method looks for where the *rate* of decrease changes, which is a judgement about a picture and frequently has no clear answer. The **silhouette score** (below) is better because it measures separation as well as compactness. The **gap statistic** compares your inertia against inertia on uniformly random data with the same bounding box — the same "compare against a null" logic as parallel analysis in §24.1. ▸ **But in practice $K$ is usually set by what you will do with the clusters: a marketing team can act on 5 segments and not on 47, and that constraint is more informative than any internal index.**

> **Where this came from.** k-means has one of the strangest publication histories in the field. The algorithm now universally called **Lloyd's algorithm** was written down by **Stuart Lloyd** at Bell Labs in **1957** — as a method for designing quantizers for **pulse-code modulation**, that is, for choosing the discrete levels to which an analogue telephone signal should be rounded. It circulated as an internal Bell Labs document and **was not formally published until 1982**, twenty-five years later, by which time the method had been reinvented several times over. **Joel Max** derived essentially the same procedure independently in 1960, which is why signal-processing texts say "Lloyd–Max quantizer." **Hugo Steinhaus** described the partitioning problem in 1956, and **James MacQueen** gave the method the name **"k-means"** in a 1967 paper. **k-means++ initialization** — the single most valuable improvement anyone has made to it — is comparatively recent: **David Arthur and Sergei Vassilvitskii**, 2007. ▸ **The dominant clustering algorithm of the 21st century was designed to compress telephone calls and spent a quarter of a century as an unpublished memo.**

### Gaussian mixture models and EM

▸ $$p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k),\qquad \sum_k\pi_k=1$$

Direct maximum likelihood is intractable (a log of a sum). EM makes it tractable.

**Derivation.** Introduce latent assignments $z_i\in\{1..K\}$. For any distribution $q(z)$, Jensen gives (Ch. 1 §1.4.4):
$$\log p(x\mid\theta)\ \ge\ \mathbb{E}_{q}\big[\log p(x,z\mid\theta)\big] + H(q)$$
with equality iff $q(z)=p(z\mid x,\theta)$.

**E-step** — set $q$ to the exact posterior, closing the gap:
▸ $$\gamma_{ik} = p(z_i=k\mid x_i) = \frac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$$

**M-step** — maximize the bound over $\theta$ with $\gamma$ fixed. Let $N_k=\sum_i\gamma_{ik}$:
▸ $$\pi_k = \frac{N_k}{n},\qquad \mu_k = \frac{1}{N_k}\sum_i\gamma_{ik}x_i,\qquad \Sigma_k = \frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$$

▸ **Why EM never decreases the likelihood:** the E-step raises the bound to exactly equal the log-likelihood; the M-step then increases the bound. Since the log-likelihood is always ≥ the bound, it must have increased too. **Monotone convergence to a local optimum, guaranteed.** This proof is short and comes up often — learn it.

**Failure modes:** ▸ **singularities.** A component that collapses onto a single point has $|\Sigma_k|\to0$ and likelihood $\to\infty$. The likelihood surface is  unbounded. Fix by adding $\epsilon I$ to covariances (a regularization/MAP prior) or restarting collapsed components. Also: local optima (many restarts), and $O(Kd^2)$ parameters per full covariance (use `diag` or `tied` when $d$ is large).

#### Reading the mixture model in plain English

$$p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k),\qquad \sum_k\pi_k=1$$

Read aloud: *"the probability of seeing point x is a weighted sum of K Gaussian bells, and the weights add to one."*

| Piece | Read aloud | What it is |
|---|---|---|
| $\pi_k$ | "pi-k" | The **mixing weight** — what fraction of the population belongs to group $k$. Nothing to do with 3.14159. |
| $\mathcal{N}(x\mid\mu_k,\Sigma_k)$ | "normal of x, given mu-k and Sigma-k" | The **height of Gaussian bell $k$** at the location $x$ |
| $\mu_k$ | "mu-k" | Where bell $k$ is centred |
| $\Sigma_k$ | "Sigma-k" | Bell $k$'s **covariance** — its width, and crucially its *orientation and shape* |
| $\sum_k\pi_k = 1$ | "the pi's sum to one" | The weights are proportions of a whole |

▸ **The generative story: to make a data point, first roll a weighted die to pick which group it belongs to, then draw from that group's Gaussian.** You observe the point; you do not observe the die roll. **That hidden die roll is the latent variable $z_i$, and everything about EM is a consequence of it being hidden.**

**Why a GMM is strictly more expressive than k-means.** k-means has one shared, isotropic, implicit covariance. A GMM gives each component its own full $\Sigma_k$, so a component can be a long thin diagonal ellipse while its neighbour is a small circle. It also has $\pi_k$, so a component may legitimately be small. **Elongated, differently sized, differently oriented clusters are all representable** — the first three of k-means's four failure modes are repaired in one stroke.

**Why direct maximum likelihood is intractable.** The log-likelihood is $\sum_i\log\sum_k \pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)$ — **a logarithm of a sum.** The log cannot be pushed inside, so the parameters of all $K$ components stay coupled inside one logarithm and the derivative has no closed-form root. If you *knew* which component generated each point, the sum would collapse to a single term, the log would land directly on the Gaussian, and you would just be fitting $K$ separate Gaussians — which is trivial. ▸ **The entire difficulty is the not-knowing, and EM's strategy is to guess.**

#### The EM derivation, decoded

$$\log p(x\mid\theta)\ \ge\ \mathbb{E}_{q}\big[\log p(x,z\mid\theta)\big] + H(q)$$

Read aloud: *"the log-likelihood is at least the expected complete-data log-likelihood under any distribution q, plus the entropy of q."* The right-hand side is the **evidence lower bound (ELBO)** (Ch. 1 §1.4.4).

| Piece | Read aloud | What it is |
|---|---|---|
| $\theta$ | "theta" | All the parameters: every $\pi_k$, $\mu_k$, $\Sigma_k$ |
| $z$ | "z" | The hidden assignments — which component made each point |
| $q(z)$ | "q of z" | **Your current guess** about the hidden assignments. Any distribution at all. |
| $\mathbb{E}_q[\cdot]$ | "expectation under q" | Average the thing inside, weighting by how likely $q$ says each $z$ is |
| $H(q)$ | "entropy of q" | How uncertain your guess is (Ch. 1 §1.4) |
| $\ge$ | "is at least" | **A bound, not an equation** — and this is the whole trick |

> **Analogy.** You want to know the height of a mountain permanently hidden in cloud. You cannot measure the peak. But you can build a scaffold beneath it and measure the scaffold — that is a **lower bound**, and you know the mountain is at least that tall. EM alternates two moves: **raise the scaffold until it touches the peak** (E-step), then **move the whole mountain upward** (M-step). Because the scaffold was touching when you moved, and the mountain is always above the scaffold, the peak is now  higher than before.

**The E-step, decoded.**

$$\gamma_{ik} = p(z_i=k\mid x_i) = \frac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$$

This is **Bayes' rule**, nothing more. Numerator: prior probability of group $k$ times the likelihood of seeing this point under group $k$. Denominator: the same thing summed over all groups, so the answers add to 1.

$\gamma_{ik}$ is called the **responsibility**: read it as *"how much of point $i$ does component $k$ own?"* A point sitting squarely in one cluster might have $\gamma = (0.99, 0.01, 0.00)$; a point between two might have $(0.48, 0.50, 0.02)$.

**Real numbers.** Two components, equal weights $\pi = 0.5$, both unit-variance 1-D Gaussians centred at $0$ and $4$. A point at $x = 1$:
- Density under component 1: $\frac{1}{\sqrt{2\pi}}e^{-0.5} = 0.242$
- Density under component 2: $\frac{1}{\sqrt{2\pi}}e^{-4.5} = 0.0044$
- $\gamma_{i1} = \frac{0.5(0.242)}{0.5(0.242)+0.5(0.0044)} = \mathbf{0.982}$

**98% of that point belongs to component 1, and 2% to component 2** — and the 2% is not discarded, it  contributes to component 2's mean and covariance in the M-step. That is the difference between soft and hard assignment.

**The M-step, decoded.** With $N_k = \sum_i\gamma_{ik}$ (read: *"the effective number of points owned by component $k$"* — a fractional count):

- $\pi_k = N_k/n$ — **"what share of the total data does this component own?"**
- $\mu_k = \frac{1}{N_k}\sum_i\gamma_{ik}x_i$ — **a weighted average**, where each point contributes in proportion to how much this component owns it.
- $\Sigma_k = \frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$ — the same weighted average, applied to outer products. The outer product $(x-\mu)(x-\mu)^\top$ is a $d\times d$ matrix (Ch. 0 §0.8) whose $(a,b)$ entry is how much feature $a$ and feature $b$ deviate together.

▸ **Every M-step formula is the ordinary maximum-likelihood formula for a single Gaussian, with each data point weighted by its responsibility.** If the responsibilities were all 0 or 1, these would be exactly "the mean and covariance of the points in this group." Soft assignment changes nothing except that the counts become fractional.

**Why the likelihood never decreases — the proof, in three sentences.** (1) The bound is *always* below the log-likelihood, for any $q$. (2) The E-step sets $q$ to the exact posterior, which is the unique choice making the bound **equal** to the log-likelihood — the gap between them is exactly $\mathrm{KL}(q\|p(z\mid x))$, which is zero precisely when they match. (3) The M-step then increases the bound. Since the log-likelihood started equal to the bound and is always at least the bound, it has increased by at least as much. ∎

▸ **This proof is short, comes up constantly, and generalizes far beyond mixtures — it is the same argument that justifies variational inference and the VAE's training objective (Ch. 19).** Learn it once and it pays for itself repeatedly.

#### Why singularities are a real hole, not a numerical annoyance

Take one component and place its mean exactly on a single data point, then shrink its covariance. The Gaussian density at its own centre is $\frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}$ — and as $\lvert\Sigma\rvert \to 0$ that goes to **infinity**.

Numbers, in 1-D at the centre: $\sigma = 0.1$ gives density $3.99$; $\sigma = 0.01$ gives $39.9$; $\sigma = 10^{-6}$ gives $399{,}000$. The likelihood of the whole dataset is a product, so one term going to infinity takes the whole thing with it.

▸ **The likelihood function is  unbounded above, so "the maximum likelihood estimate" does not exist.** This is not a bug in EM or a floating-point issue — the objective itself has no maximum. What EM finds in practice is a good *local* optimum, and it finds one only because it usually does not stumble into a collapse. **A GMM run that reports a spectacular likelihood is more likely to have collapsed than to have succeeded**, and the diagnostic is to look at the smallest component's determinant.

**The fixes, and what each one really is:**
- **Add $\epsilon I$ to every covariance.** This is `reg_covar` in most libraries. It puts a floor under the width of any component, so the density has a ceiling. Formally it is a MAP estimate under an inverse-Wishart prior — **a prior that says "I do not believe in clusters of literally zero width."**
- **Restart collapsed components.** Detect a component whose $N_k$ has fallen below a threshold and re-seed it somewhere else.
- **Constrain the covariance shape.** `spherical` ($\sigma^2 I$), `diag` (axis-aligned ellipse), `tied` (one shared $\Sigma$ for all components), `full` (unrestricted). Parameter counts per component in $d$ dimensions: $1$, $d$, and $d(d+1)/2$. ▸ **At $d = 100$, a full covariance costs 5,050 parameters per component — so a 10-component full-covariance GMM has 50,500 covariance parameters, and fitting that from a few thousand points is hopeless.** `diag` costs 100. **This is the single most common reason a GMM underperforms in high dimensions, and the fix is a one-word argument.**

> **Where this came from.** The **EM algorithm** was named and unified by **Arthur Dempster, Nan Laird, and Donald Rubin** at Harvard in **1977** — a paper that has become one of the most cited in the history of statistics. Their contribution was largely one of recognition: a dozen apparently unrelated iterative procedures scattered across genetics, econometrics, and signal processing were all the same algorithm, and once you see that, the monotonicity proof applies to all of them at once. One of those special cases had been developed a decade earlier at the Institute for Defense Analyses in Princeton, where **Leonard Baum, Lloyd Welch** and colleagues worked out what is now called the **Baum–Welch algorithm** for fitting hidden Markov models — the classified applications were in cryptanalysis and signal analysis, and the same machinery later underpinned speech recognition for thirty years. ▸ **A tool that turned out to be a general principle of statistics was worked out first, in a specialized form, inside a defence research organization.**

### DBSCAN and HDBSCAN

**DBSCAN:** a point is *core* if it has $\ge \texttt{minPts}$ neighbours within $\varepsilon$. Clusters are maximal sets of density-connected points; everything else is **noise**.

▸ **What it buys:** arbitrary cluster shapes, no $K$, and explicit outlier detection — three things k-means cannot do. **What it costs:** sensitivity to $\varepsilon$, and failure when clusters have very different densities (one $\varepsilon$ cannot suit both).

**HDBSCAN** fixes the density problem by building a hierarchy over all $\varepsilon$ and extracting the most *stable* clusters. Only `min_cluster_size` needs setting. ▸ **The best general-purpose default clustering algorithm** when you don't know $K$ and the geometry isn't spherical.

#### DBSCAN, decoded — clustering by crowd, not by centre

Every method so far has defined a cluster by a **centre**. DBSCAN defines it by **crowding**, and that single change is what buys arbitrary shapes.

The two parameters and the three point types:

| Term | Read aloud | Definition |
|---|---|---|
| $\varepsilon$ | "epsilon" | A **radius**. "Neighbour" means "within $\varepsilon$." |
| `minPts` | "min points" | How many neighbours you need to count as being in a crowd |
| **Core point** | | Has at least `minPts` neighbours within $\varepsilon$ — **it is inside a dense region** |
| **Border point** | | Not itself crowded, but within $\varepsilon$ of a core point — **on the edge** |
| **Noise point** | | Neither — **belongs to nothing, and DBSCAN says so** |

**The cluster-building rule:** start at any core point and repeatedly absorb everything within $\varepsilon$; whenever you absorb another core point, continue outward from it too. Stop when the frontier contains only border points. That connected blob is one cluster.

▸ **Because the cluster grows by chaining from core point to core point, it can wander into any shape at all — a crescent, a spiral, two interlocking rings.** k-means cannot produce a crescent no matter how you set it, because "everything nearer to $\mu_1$ than to $\mu_2$" always carves space with straight lines. **DBSCAN never asks "which centre is nearest"; it only asks "can I walk there through crowded ground."**

> **Analogy.** A city at night seen from the air. k-means says "there are five cities; assign every light to its nearest city hall." DBSCAN says "wherever the lights are dense, keep walking; wherever they thin out, stop." The second finds a city shaped like a river valley. It also correctly identifies the isolated farmhouse as **not part of any city** — which is a category k-means does not possess, since k-means must assign every point to something.

**Real numbers on the $\varepsilon$ sensitivity.** Points along a line at spacing 1.0, with one gap of 2.5 in the middle, `minPts = 3`:

| $\varepsilon$ | Result |
|---|---|
| $0.9$ | **Every point is noise** — nobody has 3 neighbours |
| $1.5$ | **Two clusters**, split at the gap. Correct. |
| $3.0$ | **One cluster** — the gap is now bridgeable |

▸ **Three settings, three completely different answers, and no internal signal that any of them is wrong.** The standard heuristic is to plot each point's distance to its $k$-th nearest neighbour, sorted — the "knee" of that curve is a reasonable $\varepsilon$, because it is where you cross from "inside a cluster" to "in the gaps." It is the same judgement-about-a-picture problem as the scree plot.

**The varying-density failure, concretely.** Suppose cluster A has points every $0.5$ units and cluster B has points every $3$ units. Set $\varepsilon = 1$ and B dissolves entirely into noise. Set $\varepsilon = 4$ and A merges with everything near it. **There is no single $\varepsilon$ that works, because $\varepsilon$ is a global constant and density is a local property.**

**How HDBSCAN escapes.** Rather than committing to one $\varepsilon$, it runs the whole family at once. Imagine sweeping $\varepsilon$ from large to small: clusters appear, then split into children, then dissolve into noise. That sweep is a tree. HDBSCAN then asks, for each candidate cluster in the tree, **"over how wide a range of $\varepsilon$ did this cluster persist unchanged?"** — its **stability** — and keeps the ones that survive longest. A dense cluster persists over a wide range of small $\varepsilon$; a sparse cluster persists over a wide range of large $\varepsilon$; **both can be selected simultaneously, because stability is measured relative to each cluster's own scale.**

▸ **You are left with one parameter — `min_cluster_size` — whose meaning is fully intuitive ("how many members before I'm willing to call it a cluster?").** Compare that to k-means, which needs $K$; GMM, which needs $K$ and a covariance type; spectral, which needs a graph construction and $K$. **This is why HDBSCAN is the right default when you  do not know what is in your data**, which is the situation clustering is supposed to be for.

> **Where this came from.** **DBSCAN** was published at KDD in **1996** by **Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu**, working on spatial databases in Munich — the motivating problem was geographic, finding shaped regions in point data such as satellite imagery and geological surveys, which is precisely the setting where "clusters are round" is obviously false. It received the **KDD Test of Time Award in 2014**, eighteen years after publication. **HDBSCAN** came from **Ricardo Campello, Davoud Moulavi, and Jörg Sander** in 2013 — with Sander, notably, an author of both.

### Spectral clustering

1. Build a similarity graph $W$ (k-NN or Gaussian kernel).
2. Form the Laplacian $L=D-W$, or the normalized $L_{\text{sym}}=I-D^{-1/2}WD^{-1/2}$.
3. Take the eigenvectors of the $k$ smallest eigenvalues, stack as columns, row-normalize.
4. Run k-means on those rows.

▸ **Why it works:** $x^\top Lx = \frac12\sum_{ij}w_{ij}(x_i-x_j)^2$, so minimizing the Rayleigh quotient finds a soft assignment that is smooth over the graph — i.e. cuts few edges. It is a **continuous relaxation of the normalized-cut problem**, which is NP-hard. The **multiplicity of the eigenvalue 0 equals the number of connected components**, which is why the eigengap indicates $K$.

Handles non-convex shapes; costs $O(n^3)$ for the eigendecomposition (use sparse graphs + Lanczos).

### Hierarchical clustering

Agglomerative: start with singletons, repeatedly merge the closest pair. **Linkage** determines the geometry:

| Linkage | Distance | Behaviour |
|---|---|---|
| Single | $\min$ | chaining; finds elongated shapes; outlier-sensitive |
| Complete | $\max$ | compact, roughly equal-diameter clusters |
| Average | mean | compromise |
| **Ward** | increase in within-cluster SSE | spherical, balanced; the usual default |

Output is a dendrogram — cut it at any height for any $K$. $O(n^2\log n)$ time, $O(n^2)$ memory.

### Validity metrics

- **Silhouette:** $s_i = \frac{b_i-a_i}{\max(a_i,b_i)}$ where $a_i$ is mean intra-cluster distance and $b_i$ the mean distance to the nearest other cluster. Range $[-1,1]$.
- **Davies–Bouldin** (lower better), **Calinski–Harabasz** (higher better).
- With labels: **Adjusted Rand Index**, **Normalized Mutual Information**, V-measure.

▸ All internal metrics encode a geometric assumption (usually compactness and separation), so **they favour the algorithm whose assumptions match them** — silhouette systematically prefers k-means-style spherical solutions. Never select an algorithm by an internal index alone.

---

## 24.5 Anomaly detection

| Method | Mechanism | Note |
|---|---|---|
| $z$-score / MAD | distance from centre | univariate; MAD is robust |
| **Mahalanobis** | $\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$ | accounts for correlation; assumes Gaussian |
| **Isolation Forest** | random splits; anomalies isolate in few splits | ▸ scores by *path length*, not density — fast, scales well, handles high $d$ |
| LOF | ratio of local density to neighbours' | finds local anomalies a global method misses |
| One-class SVM | smallest enclosing region | sensitive to $\nu$ and kernel width |
| Autoencoder | reconstruction error | for images/sequences; can generalize too well and reconstruct anomalies |
| Density models | likelihood under a fitted model | ▸ **caution:** deep generative models notoriously assign *higher* likelihood to some OOD data (Ch. 33 §33.6) |

**The evaluation problem:** anomalies are rare and often unlabelled, so use PR-AUC rather than ROC-AUC, and expect wide confidence intervals (Ch. 3).

---

#### Examples and non-examples: reading a t-SNE or UMAP plot

These plots are the most over-interpreted images in machine learning, so the boundary is worth stating precisely.

**✅ What you may legitimately conclude**

| Observation | Valid reading |
|---|---|
| Points form tight, separated blobs | The data has local neighbourhood structure |
| A known class lands in one blob | That class is locally coherent in the original space |
| A point sits inside the wrong blob | It has neighbours of that class — worth investigating |

**❌ Near-misses — things the plot does *not* say**

| Tempting reading | Why it's wrong |
|---|---|
| "These two clusters are far apart, so they're very different" | **Between-cluster distances are meaningless.** t-SNE only preserves local neighbourhoods |
| "This cluster is bigger, so it has more variance" | Cluster *sizes* and densities are not preserved either |
| "There are exactly 5 clusters" | Perplexity and random seed change the apparent count |
| "The x-axis means something" | The axes have no interpretation whatsoever — no units, no direction |
| "I ran it twice and got different pictures, so it's broken" | It is stochastic; different runs  differ |
| "The clusters are real" | t-SNE will produce convincing clusters from **pure random noise** |

▸ **The boundary:** t-SNE and UMAP preserve *who is near whom*, and discard *how far apart* and *how big*. **They are neighbourhood-preserving, not distance-preserving** — which makes them excellent for spotting structure and worthless for measuring it. Any quantitative claim read off one of these plots is almost certainly wrong.

> **Common misconception.** *"t-SNE found clusters, so the data is clustered."* Run t-SNE on uniformly random points and it will still produce clean-looking islands, because the algorithm's objective actively pushes dissimilar points apart. **Apparent clusters are partly an artifact of the method.** Confirm any structure you see with a method that operates in the original space — silhouette scores, a clustering algorithm, or a downstream task.

> **Common misconception.** *"PCA and t-SNE are both dimensionality reduction, so they're interchangeable."* PCA is linear, deterministic, invertible, and preserves *global* variance structure — you can project new points and reconstruct originals. t-SNE is nonlinear, stochastic, has no out-of-sample extension in its basic form, and deliberately distorts global structure. **PCA is for analysis; t-SNE is for looking.**

---

## Did you know?

- **Principal Component Analysis was invented twice, 32 years apart.** Karl Pearson described it in 1901 as finding "lines and planes of closest fit"; Harold Hotelling independently derived and named it in 1933 from a different starting point. Pearson was minimizing reconstruction error, Hotelling was maximizing variance — and the fact that these give the same answer is the central theorem of the method.

- **Pearson's 1901 paper was about biology, not data science.** He was working on problems in heredity and biometrics. He also founded the world's first university statistics department and, less admirably, was a prominent eugenicist — a piece of the field's history that its textbooks often omit.

- **The Expectation–Maximization algorithm is a Jensen's-inequality argument in disguise.** The E-step tightens a lower bound until it touches the true likelihood; the M-step then pushes that bound up. Because the bound never exceeds the truth, likelihood cannot decrease — which is exactly the ELBO logic of Chapter 1 §1.4.4, applied to a different problem.

- **k-means is not guaranteed to find the best clustering** — it converges to a local optimum that depends entirely on initialization. The standard fix, k-means++, simply chooses smarter starting centres, and it improves results enough that it has been the default in most libraries for years.

- **k-means implicitly assumes your clusters are round and equally sized.** It minimizes squared distance to a centre, which makes it structurally incapable of finding elongated or nested clusters. When it produces a bad answer on such data, it isn't malfunctioning — you asked it a question it cannot express.

- **DBSCAN doesn't need you to specify the number of clusters** and can label points as noise, which k-means cannot. It finds clusters of arbitrary shape by following density. The price is two parameters that are  hard to set, and poor behaviour when densities vary across the dataset.

- **t-SNE's heavy-tailed Student-t distribution in the low-dimensional space exists to solve the "crowding problem."** In high dimensions there is far more room at moderate distances than in two dimensions, so mapping faithfully would jam everything together. The heavy tail gives distant points somewhere to go — and is precisely why between-cluster distances become meaningless.

- **t-SNE will produce beautiful, convincing clusters from pure random noise.** The objective pushes dissimilar points apart whether or not there is structure to find. This is the single most important caveat about the most widely-shared plot in the field.

- **UMAP is generally faster than t-SNE and preserves more global structure**, but "more" is not "enough" — its inter-cluster distances are still not reliable. It rests on some  deep mathematics (Riemannian geometry and algebraic topology) that most users, reasonably, never touch.

- **The "curse of dimensionality" makes anomaly detection by distance nearly useless in high dimensions.** As dimension grows, all pairwise distances converge toward each other, so "unusually far away" stops being a meaningful category. This is the same concentration phenomenon that makes random vectors near-orthogonal in Chapter 1 §1.1.5 — a blessing there, a curse here.

- **Independent Component Analysis was developed to solve the "cocktail party problem"** — separating individual voices from a mixture of microphone recordings. Unlike PCA, which seeks uncorrelated components, ICA seeks *statistically independent* ones, and the crucial technical requirement is that the sources be non-Gaussian.

- **Non-negative Matrix Factorization produces parts-based representations precisely because it forbids subtraction.** With only additive combinations allowed, a face decomposes into noses, eyes, and mouths rather than the ghostly whole-face "eigenfaces" that PCA yields. Constraining the model made it more interpretable — a rare and instructive trade.

---

## Check for Understanding

**PCA finds the eigenvectors of the covariance because maximizing projected variance and minimizing reconstruction error are the same problem; EM works because the E-step tightens a Jensen bound to equality and the M-step then raises it, guaranteeing monotone likelihood improvement; and t-SNE's heavy-tailed low-dimensional kernel exists to solve the crowding problem, which is also why the distances between its clusters mean nothing.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why do "maximize projected variance" and "minimize reconstruction error" give the same answer** in PCA?
2. **Why are the principal components the eigenvectors of the covariance matrix?**
3. **How do you choose how many components to keep**, and why is the scree plot a judgment call rather than a calculation?
4. **Why does PCA require centring the data first?** What breaks if you skip it?
5. **Why is EM guaranteed never to decrease the likelihood?** (Say it in terms of a bound being tightened, then raised.)
6. **What does k-means assume about the shape of your clusters**, and when does that assumption break?
7. **Why does k-means need k-means++**, and what problem is it solving?
8. **What is the crowding problem**, and why does t-SNE use a heavy-tailed distribution to fix it?
9. **Why are distances between t-SNE clusters meaningless?** What *is* preserved?
10. **Why will t-SNE show clusters even in pure noise**, and how should that change how you read one?
11. **When would you use DBSCAN instead of k-means**, and what does it cost you?
12. **Why does distance-based anomaly detection fail in high dimensions?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 25 — Self-Supervised & Representation Learning](25-self-supervised-representation-learning.md)
