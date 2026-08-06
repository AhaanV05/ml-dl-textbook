# Chapter 24 — Unsupervised Learning & Dimensionality Reduction

> **Prerequisites:** Ch. 1 (§1.1 SVD, §1.4), Ch. 22.

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

### Practical notes

- **Standardize** when features have different units; otherwise the largest-unit feature dominates. Standardizing means you're doing eigendecomposition of the *correlation* matrix.
- **Choosing $k$:** cumulative explained variance (90–95%), the scree-plot elbow, or parallel analysis (compare eigenvalues against those from permuted data).
- **Whitening:** divide scores by $\sigma_j$ so all components have unit variance. Useful as a preprocessing step; amplifies noise in small components.
- **Complexity:** $O(\min(n^2d, nd^2))$ exact; use randomized SVD for large $d$.

▸ **The limitation that matters:** PCA is linear. Data on a curved manifold (a Swiss roll) is not compressible by any linear projection. Also, **high variance is not the same as high information** — in a classification problem the discriminative direction may have small variance, which is why LDA (Ch. 22 §22.6) exists.

### Probabilistic PCA

Model $x = Wz + \mu + \varepsilon$, $z\sim\mathcal{N}(0,I)$, $\varepsilon\sim\mathcal{N}(0,\sigma^2I)$. Then $x\sim\mathcal{N}(\mu, WW^\top+\sigma^2I)$, and the maximum-likelihood $W$ spans the top-$k$ eigenspace.

▸ **Why this framing matters:** it gives PCA a likelihood (so you can compare models, handle missing data, and use EM), and as $\sigma^2\to0$ it recovers classical PCA exactly. **A VAE (Ch. 19 §19.3) is probabilistic PCA with neural networks replacing the linear maps** — that is the cleanest way to see what a VAE is.

### Kernel PCA

Run PCA in an RKHS: centre the kernel matrix $\tilde K = K - \mathbf{1}_nK - K\mathbf{1}_n+\mathbf{1}_nK\mathbf{1}_n$, then eigendecompose $\tilde K$. Nonlinear components at $O(n^2)$–$O(n^3)$ cost. The out-of-sample projection requires the pre-image problem, which has no exact solution.

---

## 24.2 The other linear decompositions

### ICA

PCA finds **uncorrelated** components; ICA finds **statistically independent** ones. Independence is strictly stronger than uncorrelatedness except for Gaussians.

▸ **The central insight:** by the CLT, a mixture of independent sources is *more Gaussian* than the sources. So maximize **non-Gaussianity** to unmix. Measure it with negentropy $J(y)=H(y_{\text{gauss}})-H(y)$, approximated by $\left(\mathbb{E}[G(y)]-\mathbb{E}[G(\nu)]\right)^2$ with $G(u)=\log\cosh(u)$ or $-e^{-u^2/2}$; kurtosis is the simplest but is outlier-sensitive.

▸ **ICA fails entirely if the sources are Gaussian** — a rotation of independent Gaussians is still independent Gaussians, so the model is unidentifiable. Also, sign, scale, and ordering of components are unidentifiable in general.

Classic application: blind source separation (the cocktail-party problem), EEG/fMRI artifact removal.

### NMF

$X\approx WH$ with $W,H\ge0$. Non-negativity forces a **parts-based, additive** decomposition (no cancellation), which is why NMF on face images yields nose/eye/mouth components while PCA yields ghostly whole-face eigenfaces. Used for topic modelling and spectral unmixing. Non-convex; solved by multiplicative updates or alternating least squares.

### Random projection

By Johnson–Lindenstrauss (Ch. 1 §1.2), projecting onto $k = O(\log n/\epsilon^2)$ random directions preserves all pairwise distances to within $1\pm\epsilon$ — **with no reference to the data at all.** Extremely cheap and a good baseline before any learned method.

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

### UMAP

Builds a fuzzy $k$-NN graph with a locally-adaptive metric (theoretically motivated by Riemannian geometry and category theory), then optimizes a low-dimensional graph layout with a cross-entropy objective and negative sampling.

**Versus t-SNE:** much faster ($O(n^{1.14})$ vs $O(n\log n)$ with Barnes–Hut, but with far better constants), better preservation of global structure, and it supports a `transform` for new points. **Key parameters:** `n_neighbors` (local↔global balance) and `min_dist` (how tightly points pack).

▸ **Both are visualization tools, not preprocessing.** Do not cluster on t-SNE/UMAP coordinates and report the clusters as findings — the embedding manufactures separations. Cluster in the original or PCA space; use the embedding only to *look*.

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

**Failure modes:** ▸ **singularities.** A component that collapses onto a single point has $|\Sigma_k|\to0$ and likelihood $\to\infty$. The likelihood surface is genuinely unbounded. Fix by adding $\epsilon I$ to covariances (a regularization/MAP prior) or restarting collapsed components. Also: local optima (many restarts), and $O(Kd^2)$ parameters per full covariance (use `diag` or `tied` when $d$ is large).

### DBSCAN and HDBSCAN

**DBSCAN:** a point is *core* if it has $\ge \texttt{minPts}$ neighbours within $\varepsilon$. Clusters are maximal sets of density-connected points; everything else is **noise**.

▸ **What it buys:** arbitrary cluster shapes, no $K$, and explicit outlier detection — three things k-means cannot do. **What it costs:** sensitivity to $\varepsilon$, and failure when clusters have very different densities (one $\varepsilon$ cannot suit both).

**HDBSCAN** fixes the density problem by building a hierarchy over all $\varepsilon$ and extracting the most *stable* clusters. Only `min_cluster_size` needs setting. ▸ **The best general-purpose default clustering algorithm** when you don't know $K$ and the geometry isn't spherical.

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

## Check for Understanding

**PCA finds the eigenvectors of the covariance because maximizing projected variance and minimizing reconstruction error are the same problem; EM works because the E-step tightens a Jensen bound to equality and the M-step then raises it, guaranteeing monotone likelihood improvement; and t-SNE's heavy-tailed low-dimensional kernel exists to solve the crowding problem, which is also why the distances between its clusters mean nothing.**

---

**Next:** [Chapter 25 — Self-Supervised & Representation Learning](25-self-supervised-representation-learning.md)
