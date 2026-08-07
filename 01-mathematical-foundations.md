# Chapter 1 — Mathematical Foundations

> **Prerequisites:** calculus, basic linear algebra.
> **Why this chapter exists:** every derivation later in the book leans on about fifteen facts. Here they are, derived, so you never have to take one on faith.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, $A^\top$, or $\bar{y}$ are unfamiliar — or if you have ever wondered why $\sigma$ seems to mean four different things — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It is a decoder ring for every symbol used here, and it takes about twenty minutes. Nothing in this chapter assumes more than that.

### Symbols introduced in this chapter

Skim this once now; refer back as needed. Each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $A \in \mathbb{R}^{m\times n}$ | "A in R m-by-n" | A grid of numbers, $m$ rows by $n$ columns |
| $Q\Lambda Q^\top$ | "Q Lambda Q-transpose" | "Rotate → stretch along axes → rotate back" |
| $\lambda_i$ | "lambda-i" | An **eigenvalue** — a stretch factor |
| $\sigma_i$ | "sigma-i" | A **singular value** — a stretch factor for non-square matrices |
| $U\Sigma V^\top$ | "the SVD" | "Rotate → stretch (and reshape) → rotate" |
| $\|x\|$ | "norm of x" | The **length** of a vector |
| $\|A\|_2$ | "spectral norm" | The **largest stretch** a matrix can apply |
| $\nabla_\theta \mathcal{L}$ | "grad theta L" | Which way to nudge each parameter to increase the loss |
| $J$ | "the Jacobian" | The table of *all* input→output sensitivities |
| $H$ | "the Hessian" | The table of curvatures (how the slope itself changes) |
| $\bar{y}$ | "y-bar" | **Shorthand for $\partial\mathcal{L}/\partial y$** — the gradient arriving at $y$ |
| $\mathbb{E}[X]$ | "expectation" | The probability-weighted average of $X$ |
| $\mathrm{KL}(p\|q)$ | "KL of p from q" | How much you lose by believing $q$ when reality is $p$ |

---

## 1.1 Linear algebra, but only the parts that show up

### The one-line idea

A neural network is a stack of linear maps with nonlinearities wedged between them, so understanding what a matrix *does to a space* is understanding what a layer does to data.

### The analogy

A matrix is a machine that stretches, squashes, and rotates space. Eigenvectors are the directions the machine doesn't rotate — it only stretches them. Eigenvalues are how much. Singular values are the same idea for machines that also change the dimension of the space.

### 1.1.1 The four things a matrix is

For $A \in \mathbb{R}^{m \times n}$:

- **A linear map** $\mathbb{R}^n \to \mathbb{R}^m$.
- **A collection of $n$ column vectors** in $\mathbb{R}^m$. $Ax = \sum_j x_j a_{:,j}$ — matrix-vector multiply is a *weighted sum of columns*. This is the single most useful reinterpretation in deep learning; an embedding lookup is exactly this with $x$ a one-hot vector.
- **A collection of $m$ row vectors.** $(Ax)_i = \langle a_{i,:}, x\rangle$ — each output is a dot product, i.e. a *similarity measurement*. Attention is built on this reading.
- **A sum of rank-one pieces**: $A = \sum_k \sigma_k u_k v_k^\top$ (the SVD). LoRA is built on this reading.

#### Unpacking those four readings

This list is dense, so here is each line again, slowly. **They are four ways of looking at the same grid of numbers** — like describing a building by its floor plan, its facade, its room list, or its structural frame. Same building, four useful languages.

Take a concrete tiny matrix throughout:

$$A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \\ 4 & 1\end{pmatrix} \in \mathbb{R}^{3\times 2}, \qquad x = \begin{pmatrix} 5 \\ 10 \end{pmatrix} \in \mathbb{R}^2$$

**Reading 1 — "a linear map $\mathbb{R}^n \to \mathbb{R}^m$."** This says: *$A$ is a machine that eats a 2-number list and spits out a 3-number list.* "Linear" means it obeys two rules: doubling the input doubles the output, and feeding in a sum gives the sum of the outputs. That's it — that is the entire definition of linear, and it is why so much is provable about these objects.

**Reading 2 — "a weighted sum of columns."** The formula $Ax = \sum_j x_j a_{:,j}$ uses $a_{:,j}$ to mean "**column $j$ of $A$**" (the colon means "all rows," exactly as in NumPy). So:

$$Ax = 5\begin{pmatrix}1\\0\\4\end{pmatrix} + 10\begin{pmatrix}2\\3\\1\end{pmatrix} = \begin{pmatrix}25\\30\\30\end{pmatrix}$$

▸ **The input vector is a recipe of how much of each column to mix.** Now the payoff: if $x$ is **one-hot** (all zeros except a single 1 in position $j$), the mixture is just *column $j$, unchanged*. That is why "an embedding lookup is exactly this" — an embedding table is a matrix, a token ID is a one-hot vector, and looking up a word's vector *is* a matrix multiply. In practice nobody actually multiplies (it would be wasteful); they index into the array. But mathematically it's the same operation, which is why gradients flow through embedding layers correctly.

**Reading 3 — "a collection of $m$ row vectors."** Here $a_{i,:}$ means "**row $i$**," and $\langle\cdot,\cdot\rangle$ is the dot product. So output entry $i$ is the dot product of row $i$ with $x$:

$$(Ax)_1 = \langle (1,2),\ (5,10)\rangle = 1(5)+2(10) = 25 \ \checkmark$$

▸ Since a dot product measures **alignment**, each row is a *question being asked of the input*: "how much does $x$ look like me?" A layer with 3 rows asks 3 such questions. This is the reading that makes attention make sense — attention scores are literally "how much does this query look like that key."

**Reading 4 — "a sum of rank-one pieces."** A **rank-one matrix** is any matrix you can build from one column vector times one row vector ($uv^\top$, the outer product from §0.8). It's the simplest possible non-trivial matrix — a single pattern, scaled. The claim is that *any* matrix is a **stack of such simple patterns**, each weighted by $\sigma_k$:

$$A = \sigma_1 u_1v_1^\top + \sigma_2u_2v_2^\top + \dots$$

▸ Since the $\sigma_k$ come out sorted **largest first**, the early terms carry most of the matrix. Keep the first few and drop the rest and you have a very good approximation using far fewer numbers. That is compression, PCA, and LoRA — all three are this one sentence.

> **Analogy for the four readings.** A recipe can be read as (1) a process that turns ingredients into a dish, (2) a shopping list of ingredients with quantities, (3) a set of taste-tests you perform, or (4) "it's basically a carbonara plus a small tweak." Each reading is complete, and each makes a different question easy to answer. Fluency is knowing which to reach for.

#### Examples and non-examples: what a matrix can and cannot do

**✅ Things a matrix  does**

| Example | Concretely |
|---|---|
| Rotate a 2-D vector 90° | $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ sends $(1,0)\mapsto(0,1)$ |
| Scale one axis | $\mathrm{diag}(3,1)$ triples the $x$-coordinate, leaves $y$ |
| Project onto a line | $\mathrm{diag}(1,0)$ flattens everything onto the $x$-axis |
| Look up an embedding | $E^\top e_i$ = row $i$ of the table |
| Mix features | Any dense layer, before the activation |

**❌ Near-misses — things people assume matrices do, but they can't**

| Assumed | Why it's impossible | What's actually needed |
|---|---|---|
| Shift/translate a vector | $A\mathbf{0} = \mathbf{0}$ always — a linear map **pins the origin** | A bias term $b$ (making it *affine*) |
| Apply ReLU or any threshold | Requires an `if`; linear maps have no branching | A nonlinear activation |
| Normalize a vector to length 1 | Dividing by $\|x\|$ depends on $x$ itself | A nonlinear operation (LayerNorm) |
| Sort the entries | Order depends on values | Not expressible as a fixed matrix |

▸ **The boundary:** a matrix does exactly one thing — **it takes weighted sums of its inputs.** Everything a network does beyond weighted sums (thresholding, normalizing, gating, selecting) is contributed by the *non*-linear parts. This is why removing all activations collapses a deep network to a single layer.

> **Common misconception.** *"A deeper network is more expressive because it has more layers."* Only if the layers are separated by nonlinearities. $W_3W_2W_1x$ is a single matrix wearing three hats — a 50-layer purely-linear network can represent nothing that a 1-layer network can't. **Depth without nonlinearity buys exactly zero expressive power.** The reason this misconception is tempting is that depth *does* buy enormous power in real networks; it's easy to credit the depth rather than the interaction between depth and nonlinearity.

> **Common misconception.** *"An embedding layer is a matrix multiplication, so it's expensive."* Mathematically it is a multiplication by a one-hot vector; computationally it is an array index. Doing it literally with $\lvert V\rvert = 128{,}000$ and $d = 4096$ would be 524 million multiply-adds — almost all by zero — to fetch 4,096 numbers. Frameworks index instead, at $O(1)$. **The math and the implementation deliberately diverge here**, which is why `nn.Embedding` exists as its own layer type rather than being an `nn.Linear`.

### 1.1.2 Eigendecomposition

For square symmetric $A \in \mathbb{R}^{n\times n}$:

▸ $$A = Q\Lambda Q^\top, \qquad Q^\top Q = I, \qquad \Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$$

The columns of $Q$ are orthonormal eigenvectors. Every symmetric real matrix has this (spectral theorem). **The Hessian of a loss is symmetric**, which is why the whole of optimization theory can be phrased in eigenvalues.

Key consequences you'll use constantly:

- $A^k = Q\Lambda^k Q^\top$. Repeated application amplifies the largest-$|\lambda|$ direction exponentially. *This is exactly why gradients vanish or explode in deep nets* (Ch. 6) and why power iteration works.
- $A$ is **positive semi-definite** (PSD) iff all $\lambda_i \ge 0$ iff $x^\top A x \ge 0\ \forall x$. A local minimum of a loss has PSD Hessian.
- $\mathrm{tr}(A) = \sum_i \lambda_i$, $\det(A) = \prod_i \lambda_i$.

#### Reading $A = Q\Lambda Q^\top$ in plain English

An **eigenvector** is a direction that a matrix does not turn — it only lengthens or shortens it. The **eigenvalue** is the factor by which it does so. Formally, $Av = \lambda v$: *"applying $A$ to $v$ gives back $v$ itself, just scaled by $\lambda$."*

> **Analogy.** Push on a rectangular sheet of rubber. Most drawn arrows both stretch *and* swing round to a new angle. But arrows drawn exactly along the sheet's two natural axes only stretch — they keep pointing the same way. Those are the eigenvectors, and how much they stretch are the eigenvalues.

Now the decomposition, term by term:

| Piece | Shape | What it does |
|---|---|---|
| $Q^\top$ | $n\times n$ | **Rotate** into the special coordinate system where $A$ is simple |
| $\Lambda$ | $n\times n$ | **Stretch** each axis independently by $\lambda_1,\dots,\lambda_n$ |
| $Q$ | $n\times n$ | **Rotate back** to the original coordinates |

▸ **So $A = Q\Lambda Q^\top$ says: "every symmetric matrix is secretly just a stretch, viewed from a tilted angle."** Rotate so the tilt goes away, stretch along the axes, rotate back. A complicated-looking matrix is a simple one in disguise, and $Q$ is the disguise.

**Decoding the surrounding notation:**

- $\mathrm{diag}(\lambda_1,\dots,\lambda_n)$ — a matrix that is **zero everywhere except the diagonal**. Zeros off-diagonal mean the axes don't interact: each is scaled on its own. That is exactly what makes $\Lambda$ easy.
- $Q^\top Q = I$ — this says $Q$ is **orthonormal**: its columns are all length 1 ("normal") and mutually perpendicular ("ortho"). A matrix like this is a pure rotation/reflection — it never changes any length, it only re-aims. It also means $Q^\top = Q^{-1}$, so "rotate back" is free: just transpose.
- **Symmetric** means $A = A^\top$: the entry at row $i$ column $j$ equals the one at row $j$ column $i$. The grid is a mirror image across its diagonal.

**Why $A^k = Q\Lambda^k Q^\top$ matters so much.** Apply $A$ $k$ times and the middle rotations cancel in pairs ($Q^\top Q = I$), leaving only $\Lambda^k$ — and raising a diagonal matrix to a power just raises each diagonal entry: $\lambda_i^k$.

▸ **Now put numbers on it.** With $\lambda = 1.1$ and $k = 50$: $1.1^{50}\approx 117$. With $\lambda = 0.9$: $0.9^{50}\approx 0.005$. The *same* matrix applied fifty times amplifies one direction 117-fold while crushing another to nothing. **That is the vanishing/exploding gradient problem in a single line** — a deep network applies related matrices over and over, and any eigenvalue not very close to 1 gets raised to the power of the depth.

**Positive semi-definite (PSD), demystified.** $x^\top A x \ge 0$ for every $x$ means: *whatever direction you probe in, the answer is never negative.* The scalar $x^\top A x$ is called a **quadratic form**; think of it as "the height of a bowl in direction $x$."

- All $\lambda_i > 0$ → a bowl curving up in every direction → **a minimum**.
- All $\lambda_i < 0$ → a dome → a maximum.
- Mixed signs → a **saddle**: up one way, down another (a Pringle). Saddles, not local minima, are the dominant stationary points in high-dimensional deep learning — with millions of eigenvalues, having *all* of them be positive by chance is vanishingly unlikely.

**Trace and determinant, intuitively.** $\mathrm{tr}(A)$ is just the sum of the diagonal entries, and it equals the sum of eigenvalues — the *total* stretch. $\det(A)$ is the product — the **volume scale factor**. If $\det = 0$, some eigenvalue is 0, the matrix squashes space flat into a lower dimension, and the operation cannot be undone (that's what "singular" / "non-invertible" means).

> **Where this came from.** The word "eigenvalue" is a half-translation: German *eigen* means "own" or "characteristic," so an eigenvector is a matrix's *own* vector. David Hilbert used *Eigenwert* in his work on integral equations around 1904, and English-speaking mathematicians borrowed the German prefix rather than translating it. Earlier, Augustin-Louis Cauchy had proved in 1829 that real symmetric matrices always have real eigenvalues — while studying the rotation of rigid bodies and the geometry of quadric surfaces, not anything resembling data analysis. The **spectral theorem** gets its name from the same root as the optical *spectrum*: Hilbert's "spectrum" of values, later found to literally correspond to the emission spectra of atoms in quantum mechanics. It is a striking accident of naming that a term chosen by analogy turned out to be the physical truth.

### 1.1.3 SVD, for non-square matrices

▸ $$A = U\Sigma V^\top,\quad U \in \mathbb{R}^{m\times m},\ \Sigma \in \mathbb{R}^{m\times n} \text{ diagonal},\ V \in \mathbb{R}^{n\times n}$$

$\sigma_i \ge 0$ are the singular values, $\sigma_i^2$ are the eigenvalues of $A^\top A$.

**Why you care:** the *best rank-$r$ approximation* of $A$ in Frobenius norm is $A_r = \sum_{k\le r}\sigma_k u_k v_k^\top$ (Eckart–Young theorem). This is:

- PCA (Ch. 15),
- LoRA fine-tuning (a weight update $\Delta W$ constrained to rank $r$),
- the reason "effective rank" of a weight matrix is a meaningful diagnostic.

**Effective rank / stable rank:** $\displaystyle \mathrm{srank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i\sigma_i^2}{\sigma_1^2}$. A $1024\times1024$ matrix can have full mathematical rank but stable rank 12 — meaning it  only uses 12 directions. Worth logging during training.

#### What the SVD actually says

The eigendecomposition only works for square symmetric matrices. Most matrices in deep learning are neither. The **SVD (singular value decomposition)** is the repair: it works for *every* matrix, of any shape, with no conditions whatsoever. That universality is why it is arguably the most important factorization in applied mathematics.

Same "rotate → stretch → rotate" story as before, with one new move:

| Piece | What it does |
|---|---|
| $V^\top$ | **Rotate** in the input space ($\mathbb{R}^n$) |
| $\Sigma$ | **Stretch** each axis by $\sigma_i \ge 0$, *and change the number of dimensions* |
| $U$ | **Rotate** in the output space ($\mathbb{R}^m$) |

▸ **The difference from eigendecomposition:** the entry and exit rotations are now *different matrices* ($V$ and $U$), because the input and output live in differently-sized spaces. A $3\times 2$ matrix takes 2-D things to 3-D things, so "rotate back" isn't even a meaningful instruction — you have to rotate in the destination space instead.

**Why singular values are never negative.** $\sigma_i \ge 0$ by construction; any minus sign gets absorbed into the direction of $u_i$ or $v_i$ (flipping an arrow is a rotation, which belongs in $U$ or $V$). So $\sigma_i$ is a pure *magnitude* — "how much stretch," never "which way."

**"$\sigma_i^2$ are the eigenvalues of $A^\top A$"** — a bookkeeping fact you'll use constantly. $A^\top A$ is always square and symmetric no matter what shape $A$ is, so it *does* have an eigendecomposition. Squaring appears for the same reason it does in Pythagoras: it's the natural currency of lengths.

**Reading Eckart–Young.** "The best rank-$r$ approximation of $A$ is $A_r = \sum_{k \le r}\sigma_k u_kv_k^\top$" means: **to approximate a matrix with a simpler one, keep the biggest few rank-one pieces and throw the rest away — and no cleverer method exists.** The theorem's real force is the *no cleverer method exists* part. It converts "compress this matrix" from an open-ended search problem into a sorting problem.

**Concrete magnitudes.** A $1000\times 1000$ matrix holds $10^6$ numbers. Its rank-10 approximation needs $10 \times (1000 + 1000 + 1) \approx 2\times10^4$ — a **50× saving**. This is exactly the LoRA bargain: rather than updating all of a $\Delta W$, constrain the update to rank $r$ and store two thin matrices instead of one fat one.

**Stable rank, unpacked.** Ordinary ("mathematical") rank counts how many $\sigma_i$ are *nonzero* — a brittle measure, since $\sigma = 10^{-9}$ counts exactly as much as $\sigma = 10^{3}$. Stable rank asks the better question: **"how many directions actually carry meaningful weight?"**

Work through the ratio $\sum_i \sigma_i^2 / \sigma_1^2$:

- If one direction dominates ($\sigma_1$ huge, rest tiny) → ratio $\approx 1$ → the matrix is essentially rank one.
- If $k$ directions are equally strong → the sum is $k\sigma_1^2$ → ratio $= k$.

▸ So stable rank is an **"effective number of directions in use,"** the same style of quantity as perplexity in §1.4.2 — a count that responds smoothly to how spread-out a distribution of magnitudes is. A weight matrix whose stable rank collapses during training is telling you the layer has stopped using most of its capacity.

> **Where this came from.** The SVD was discovered twice, independently, in consecutive years: by the Italian geometer **Eugenio Beltrami** in 1873 and the Frenchman **Camille Jordan** in 1874, neither aware of the other. Both were studying bilinear forms, a question in pure geometry with no application in sight. It was rediscovered again by James Joseph Sylvester in 1889. The optimality result that makes it indispensable — that truncating the SVD is the *best possible* low-rank approximation — came much later, from **Carl Eckart and Gale Young in 1936**, working in psychometrics: they were trying to summarize tables of psychological test scores. A theorem invented for 19th-century geometry, sharpened for 1930s psychology, is now the reason you can fine-tune a 70-billion-parameter language model on a single GPU.

### 1.1.4 Norms

| Norm | Definition | Where it appears |
|---|---|---|
| $\ell_2$ | $\|x\|_2 = \sqrt{\sum x_i^2}$ | weight decay, gradient clipping |
| $\ell_1$ | $\|x\|_1 = \sum_i \lvert x_i \rvert$ | sparsity, LASSO, sparse autoencoders |
| $\ell_\infty$ | $\|x\|_\infty = \max_i \lvert x_i \rvert$ | adversarial robustness, Adam's implicit bias |
| Frobenius | $\|A\|_F = \sqrt{\sum_{ij}A_{ij}^2}$ | weight norm |
| Spectral | $\|A\|_2 = \sigma_{\max}(A)$ | Lipschitz constants, spectral norm regularization |

#### What a norm is, and what each one is for

**A norm is a way of answering "how big is this thing?" with a single number.** The double bars $\|\cdot\|$ are the notation for it; the subscript says *which* notion of size you mean. (In $\ell_p$, the $\ell$ is a script "L," and the subscript $p$ picks the family member.)

There is more than one norm because "big" is  ambiguous. If a delivery van drives 3 blocks east and 4 blocks north, is the distance travelled 5 (as the crow flies) or 7 (as the van drives)? Both answers are correct; they answer different questions.

Take $x = (3, -4)$ throughout:

| Norm | Computed on $x=(3,-4)$ | Read aloud | What it measures |
|---|---|---|---|
| $\ell_2$ | $\sqrt{9+16} = 5$ | "the two-norm" | **Straight-line length.** Ordinary Pythagoras. |
| $\ell_1$ | $3 + 4 = 7$ | "the one-norm" | **Total absolute size**, city-block distance. |
| $\ell_\infty$ | $\max(3,4) = 4$ | "the infinity-norm" | **The single worst entry.** |

Decoding the pieces:

- $\lvert x_i\rvert$ means **absolute value** — drop the minus sign. (Note: this is a *different* use of vertical bars from "given" in probability. Context distinguishes them; single bars around a number mean magnitude.)
- $\max_i \lvert x_i\rvert$ means "scan every entry, report the largest magnitude." The name "infinity-norm" comes from a limit: the general $\ell_p$ norm is $(\sum_i \lvert x_i\rvert^p)^{1/p}$, and as $p \to \infty$ the largest term overwhelms all others, so the whole expression converges to the maximum.

▸ **Why $\ell_1$ causes sparsity and $\ell_2$ doesn't** — the one fact to take from this table. Suppose you must shrink a total "budget" of weights. Under $\ell_1$, moving a weight from $0.1$ to $0$ saves exactly $0.1$ of penalty — *the same saving as moving from $10.1$ to $10.0$*. The reward for zeroing out a small weight is undiminished, so small weights get driven to exactly zero. Under $\ell_2$ the penalty is *squared*, so shrinking $0.1 \to 0$ saves only $0.01$ while shrinking $10.1\to 10.0$ saves about $2$. $\ell_2$ therefore spends its effort on large weights and leaves small ones hovering near — but never at — zero. **$\ell_1$ gives you exact zeros; $\ell_2$ gives you uniformly small numbers.** That single asymmetry is the whole basis of LASSO, of sparse autoencoders, and of most of interpretability research's tooling.

**The two matrix norms.**

- **Frobenius** $\|A\|_F = \sqrt{\sum_{ij}A_{ij}^2}$ — pretend the matrix is one long vector and take its $\ell_2$ length. It ignores structure entirely; it just asks "how much stuff is in here?" The double subscript $A_{ij}$ means "the entry in row $i$, column $j$," and $\sum_{ij}$ is the nested loop over the whole grid.
- **Spectral** $\|A\|_2 = \sigma_{\max}(A)$ — the **largest singular value**, i.e. the biggest stretch the matrix can apply to any input. This one is about worst-case behaviour, not total content.

> **Analogy.** Frobenius is the *total weight* of a car. Spectral is its *top speed*. Two very different questions about the same object, and you would not use one to answer the other.

▸ The **spectral norm is the Lipschitz constant of the linear map**: $\|Ax\| \le \|A\|_2\|x\|$, with equality for $x = v_1$. A network of $L$ layers has Lipschitz constant at most $\prod_\ell \|W_\ell\|_2 \cdot \prod_\ell \mathrm{Lip}(\sigma_\ell)$. If each $\|W_\ell\|_2 = 1.2$ and $L=50$, the network can amplify an input perturbation by $1.2^{50} \approx 9100\times$. That is the entire theory of adversarial examples in one line.

#### Reading the Lipschitz result

A **Lipschitz constant** is a speed limit on how fast a function's output can change when its input changes. If a function has Lipschitz constant $K$, then nudging the input by $\epsilon$ can move the output by at most $K\epsilon$ — never more. Formally, $\|f(a) - f(b)\| \le K\|a - b\|$.

So $\|Ax\| \le \|A\|_2\|x\|$ reads: *"the output can never be longer than the input times the largest available stretch."* And "with equality for $x = v_1$" means that bound is **actually reached** — feed in the top right-singular vector and you get the full stretch. It's a tight speed limit, not a loose one.

**Why the constants multiply through a network.** Layer 1 can amplify by at most $\|W_1\|_2$; layer 2 takes that already-amplified signal and can amplify by $\|W_2\|_2$ again. Amplifications *compound* — same as compound interest, and same as $\lambda^k$ in §1.1.2. The $\mathrm{Lip}(\sigma_\ell)$ term covers the activation functions ($\mathrm{Lip} = 1$ for ReLU and tanh, so they don't make it worse).

▸ **Sit with the number: $1.2^{50}\approx 9100$.** Each layer is only mildly expansive — a 20% stretch, which no one would flag as pathological. Yet fifty of them compose into a system where a change to the input invisible to a human can produce a completely different output. **Adversarial examples are not a bug in any particular network; they are what happens when you multiply fifty numbers slightly larger than one.** This is also precisely why spectral normalization (forcing $\|W\|_2 \approx 1$) is a defence: keep every factor at 1 and the product stays at 1.

#### Examples and non-examples: norms and "size"

**✅  norms**

| Example | Value on $x = (3,-4)$ |
|---|---|
| $\ell_2$ (Euclidean length) | $5$ |
| $\ell_1$ (sum of magnitudes) | $7$ |
| $\ell_\infty$ (largest entry) | $4$ |

**❌ Near-misses — measure "size" but aren't norms**

| Looks like a norm | Why it fails | Consequence |
|---|---|---|
| $\sum_i x_i$ (plain sum, no absolute value) | $(3,-4)$ gives $-1$ — **negative**, and $(1,-1)$ gives 0 for a nonzero vector | Not a size at all |
| Variance | $\mathrm{Var}(x+y)\ne\mathrm{Var}(x)+\mathrm{Var}(y)$ in general | Fails the triangle inequality |
| Cosine **similarity** | Bigger means *more alike*, not *larger* | It's an angle, and it ignores length entirely |
| Squared $\ell_2$, $\|x\|^2$ | $\|2x\|^2 = 4\|x\|^2$, not $2\|x\|^2$ | Not a norm, though constantly used as a loss |
| KL divergence | Asymmetric, no triangle inequality | A **divergence**, not a distance (§1.4.1) |

▸ **The boundary:** a norm must be zero only at zero, scale linearly ($\|ax\| = \lvert a\rvert\|x\|$), and satisfy the triangle inequality. **Squared $\ell_2$ fails the scaling rule** — which is why "$\ell_2$ loss" and "the $\ell_2$ norm" are  different objects despite the shared name.

> **Common misconception.** *"$\ell_1$ and $\ell_2$ regularization do the same thing, one's just harsher."* They do qualitatively different things. $\ell_2$ shrinks all weights toward zero but essentially never *to* zero; $\ell_1$ drives a large fraction to **exactly** zero. If you want feature selection or an interpretable sparse code, $\ell_2$ will not deliver it no matter how large you set the coefficient. The reason is the constant-versus-vanishing gradient argument above.

> **Common misconception.** *"A large Frobenius norm means the layer is doing something drastic."* Not necessarily. Frobenius measures total content; **spectral** norm measures worst-case amplification. A matrix can have a big Frobenius norm spread harmlessly across a thousand directions, or a small one concentrated into a single explosive direction. For stability questions — gradient explosion, adversarial robustness, Lipschitz bounds — the spectral norm is the one that matters, and the two can point in opposite directions.

> **Where this came from.** **Rudolf Lipschitz** introduced his condition in 1864, in the theory of differential equations — it was the ingredient needed to guarantee that a differential equation has exactly one solution. **Ferdinand Georg Frobenius** lent his name to the matrix norm (though he did not invent it, and it is sometimes called the Hilbert–Schmidt norm). Neither had any notion of a "network." The vocabulary of modern robustness research was assembled almost entirely from 19th-century analysis, which is why so much of this chapter's terminology sounds like it came from a different discipline: it did.

> **The adversarial-examples story.** In 2013, a team at Google including **Christian Szegedy** published *Intriguing Properties of Neural Networks*. They had gone looking for something else entirely — they were probing what individual neurons represented — and stumbled on the fact that an imperceptible, carefully-chosen perturbation could flip a state-of-the-art classifier from "school bus" to "ostrich" with high confidence. The finding was so counterintuitive that the initial reaction in the field was to suspect a bug. Ian Goodfellow's follow-up in 2014 supplied the deflating explanation: the culprit isn't some exotic nonlinearity, it's that networks are *too linear*, and linear things multiply their stretches. The formula above is that argument.

### 1.1.5 The Johnson–Lindenstrauss lemma (needed for Ch. 20)

▸ For any $\epsilon \in (0,1)$ and $N$ points in $\mathbb{R}^D$, there exists a linear map into $\mathbb{R}^d$ with
$$d = O\!\left(\frac{\log N}{\epsilon^2}\right)$$
that preserves all pairwise distances to within a factor $1\pm\epsilon$.

Read it backwards: **in $d$ dimensions you can place $N = e^{O(\epsilon^2 d)}$ vectors that are all nearly orthogonal to each other.** In $d = 768$ dimensions you can fit *millions* of directions with pairwise cosine similarity under $0.1$. This is the mathematical permission slip for superposition — a network storing far more features than it has neurons is not cheating, it's exploiting high-dimensional geometry.

**Numbers:** random unit vectors in $\mathbb{R}^d$ have expected cosine similarity $0$ and standard deviation $1/\sqrt{d}$. For $d = 768$, that's $0.036$. Two random directions in a transformer's residual stream are, for practical purposes, orthogonal.

#### Reading Johnson–Lindenstrauss slowly

This lemma is short and its consequences are enormous, so let's take it one clause at a time.

- **"$N$ points in $\mathbb{R}^D$"** — you have $N$ data points, each described by $D$ numbers. Think $N = 10{,}000$ documents, each a $50{,}000$-dimensional word-count vector.
- **"there exists a linear map into $\mathbb{R}^d$"** — you can squash them into far fewer dimensions, using nothing fancier than a matrix multiply.
- **"preserves all pairwise distances to within $1\pm\epsilon$"** — after squashing, every distance between every pair is still correct to within a small percentage. $\epsilon = 0.1$ means all distances are accurate to ±10%.
- **"$d = O(\log N/\epsilon^2)$"** — and here is the shock. The required dimension depends on **$\log N$**, the number of *points*, and **not at all on $D$**, the dimension you started in.

▸ **Sit with that.** Squashing 10,000 points works in the same number of dimensions whether they started in 50,000-D or 50-million-D. The original dimension is irrelevant. Ten thousand points simply do not need many dimensions to keep their distances apart, no matter how they were originally described.

> **Analogy.** Photograph a crowd. A photo is a 2-D squashing of a 3-D scene and it destroys some information — but the *relative* arrangement of people survives well enough to recognise the group. J–L says that if you choose the camera angle randomly, and you're willing to accept small errors, the number of "pixels" you need scales with the *log* of the number of people, not with the size of the room.

**Now "read it backwards," as the text says.** Instead of "given $N$ points, how few dimensions do I need?", ask "**given $d$ dimensions, how many near-orthogonal directions fit?**" Invert $d \approx \log N/\epsilon^2$ and you get $N \approx e^{\epsilon^2 d}$ — an **exponential** in the dimension.

The consequence is  counterintuitive. In $d$ dimensions you can only fit $d$ *exactly* perpendicular directions — that's the definition of dimension. But if you relax "exactly perpendicular" to "almost perpendicular," you can fit **exponentially many**. In 768 dimensions: 768 exactly-orthogonal directions, versus *millions* that are near-orthogonal.

▸ **This is the mathematical permission slip for superposition.** A network with 768 neurons appears able to store only 768 features. In fact it can store millions, each on its own nearly-orthogonal direction, because near-orthogonal directions barely interfere. The interference is not zero — it shows up as a small amount of noise — but it is small enough to be worth the trade. When Chapter 32 says a network represents more features than it has neurons, this is why it isn't cheating.

**Where the $1/\sqrt{d}$ comes from.** Two random unit vectors have expected cosine similarity 0 (no reason to align) with standard deviation $1/\sqrt{d}$ — the same $\sqrt{n}$ law as the standard error in §1.3.1, and for the same reason: a dot product is a sum of $d$ random terms, and random terms accumulate like $\sqrt{d}$ rather than $d$ because they partially cancel. At $d = 768$ that's $0.036$, so two random directions have a cosine similarity of about 3.6% — **effectively perpendicular.** High-dimensional space is overwhelmingly made of near-perpendicular directions, which is why "everything is far from everything else" up there.

> **Where this came from.** **William B. Johnson and Joram Lindenstrauss** proved this in 1984 — and it was not the point of their paper. Their actual subject was a technical question in functional analysis about extending Lipschitz maps into Hilbert space, and the dimension-reduction result appeared as a *lemma*: a stepping stone on the way to the theorem they cared about. It has since been cited vastly more than the theorem it was built to support, and it underwrites locality-sensitive hashing, sketching algorithms, random-projection methods, modern vector databases, and the theory of superposition in neural networks. Lindenstrauss died in 2012; his son Elon Lindenstrauss won the Fields Medal in 2010. It is a good illustration of a recurring pattern in this book: the tool that turns out to matter is frequently the one the author considered a technicality.

---

## 1.2 Matrix calculus

### The one-line idea

Differentiating with respect to a vector gives you a vector of the same shape; differentiating with respect to a matrix gives a matrix of the same shape. If your shapes don't match, you made an error.

### 1.2.1 The shape rule

If $\mathcal{L} \in \mathbb{R}$ (a scalar loss) and $\theta \in \mathbb{R}^{m\times n}$, then $\nabla_\theta \mathcal{L} \in \mathbb{R}^{m\times n}$. Always. Every backprop bug you will ever have is a shape or transpose error, and this rule catches most of them.

### 1.2.2 The identities you need

Let $x \in \mathbb{R}^n$, $A \in \mathbb{R}^{m\times n}$, $y = Ax$.

| Function | Derivative |
|---|---|
| $y = Ax$ | $\dfrac{\partial y}{\partial x} = A$ (Jacobian, $m\times n$) |
| $s = a^\top x$ | $\nabla_x s = a$ |
| $s = x^\top A x$ | $\nabla_x s = (A + A^\top)x$; $= 2Ax$ if symmetric |
| $s = \|x\|^2$ | $\nabla_x s = 2x$ |
| $s = \mathrm{tr}(A^\top B)$ | $\nabla_A s = B$ |
| $y = Ax$, given $\bar y = \partial\mathcal{L}/\partial y$ | $\nabla_A \mathcal{L} = \bar y\, x^\top$, $\ \nabla_x\mathcal{L} = A^\top \bar y$ |

That last row **is** backprop through a linear layer. Memorize it. The two rules are:

▸ $$\boxed{\ \bar A = \bar y\, x^\top \quad\text{(outer product)},\qquad \bar x = A^\top \bar y \quad\text{(transpose-multiply)}\ }$$

Sanity check the shapes: $\bar y \in \mathbb{R}^m$, $x^\top \in \mathbb{R}^{1\times n}$, so $\bar y x^\top \in \mathbb{R}^{m\times n} = $ shape of $A$. ✓

#### The bar notation, decoded — read this before the table

The table above uses a convention that trips up almost everyone meeting it for the first time. **The bar is not an average.** It is shorthand:

$$\bar{y} \;\equiv\; \frac{\partial \mathcal{L}}{\partial y} \qquad\text{("the gradient of the loss with respect to } y\text{")}$$

$\bar y$ is called the **adjoint** of $y$, or more colloquially the **upstream gradient** — the sensitivity signal flowing *backwards* from the loss into $y$. It answers: *"if $y$ were slightly larger, how much would the final loss change?"* The bar exists purely because writing $\partial\mathcal{L}/\partial y$ several hundred times would be unbearable.

**Now the two boxed rules in English.**

Setting: a linear layer computes $y = Ax$. Someone hands you $\bar y$ (how much the loss cares about each output). You need two things: how to update $A$, and what to hand backwards to the previous layer.

**Rule 1 — the weight gradient: $\bar A = \bar y\,x^\top$.**

Read entry by entry: $\bar A_{ij} = \bar y_i \, x_j$. In words: **"the gradient for the weight connecting input $j$ to output $i$ is (how wrong output $i$ was) × (how active input $j$ was)."**

▸ This is one of the most intuitive facts in all of deep learning, and it is worth saying without symbols: **blame is assigned in proportion to participation.** If input $j$ was zero, it contributed nothing to output $i$, so it receives no blame and its weight isn't updated. If input $j$ was large and output $i$ was badly wrong, that connection gets a large correction. Hebbian learning ("neurons that fire together wire together") is this formula with the error signal in place of the second firing rate.

**Rule 2 — the input gradient: $\bar x = A^\top \bar y$.**

This is what gets passed to the previous layer, so that it can run the same two rules itself. Read entry by entry: $\bar x_j = \sum_i A_{ij}\bar y_i$ — *"how much the loss cares about input $j$ = sum over all the outputs it fed, of (its connection strength) × (how much the loss cares about that output)."*

▸ **Why the transpose appears.** Forward, $A$ sends information from inputs to outputs. Backward, you need to send information from outputs back to inputs — **the same connections, traversed the other way.** Transposing a matrix is exactly "reverse the direction of every arrow." The $^\top$ is not an algebraic trick; it is the statement that backpropagation runs the network's wiring in reverse.

**Chaining these two rules layer by layer, from the loss back to the first layer, *is* backpropagation.** There is nothing else to it. Every deep learning framework you will ever use is, at its core, a program that stores which operations were performed on the way forward, then walks that list backwards applying rules like these.

> **Where backpropagation came from.** The algorithm has been invented at least four times. **Seppo Linnainmaa**, a Finnish master's student, published the general method of reverse-mode automatic differentiation in his 1970 thesis — with no reference to neural networks at all; he was concerned with rounding-error accumulation in numerical computation. **Paul Werbos** proposed applying it to neural networks in his 1974 Harvard PhD thesis, which went largely unnoticed. The idea was rediscovered independently by David Parker and by Yann LeCun in the early 1980s. It only became famous with the 1986 *Nature* paper by **David Rumelhart, Geoffrey Hinton, and Ronald Williams**, whose contribution was less the derivation than the demonstration that it *worked* — that a multi-layer network trained this way learned useful internal representations. Hinton has been notably generous about this history, repeatedly pointing out that he did not invent backpropagation. The lesson is a familiar one: an idea belongs to whoever convinces the field it matters, not always to whoever wrote it down first.

### 1.2.3 Jacobians and the two modes of autodiff

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is $J_{ij} = \partial f_i/\partial x_j \in \mathbb{R}^{m\times n}$.

**You never build $J$.** For a network with $p = 10^8$ parameters and scalar loss, $J$ would be $1 \times 10^8$ — that one is fine — but intermediate Jacobians are catastrophically large. Instead autodiff computes products:

- **Forward mode** computes the **Jacobian–vector product** $Jv$: cost $\approx$ one forward pass per input direction. Efficient when $n \ll m$.
- **Reverse mode** computes the **vector–Jacobian product** $v^\top J$: cost $\approx$ one forward + one backward pass per *output* direction. Efficient when $m \ll n$.

▸ Deep learning has $m = 1$ (scalar loss) and $n = p$ (millions of parameters). So reverse mode wins by a factor of $p$. **That's the whole reason backpropagation is reverse-mode.** It is not a deep insight about neural networks; it's a statement about the shape of the problem.

**Cost:** reverse-mode AD gives $\nabla_\theta\mathcal{L}$ at roughly $2$–$3\times$ the cost of a forward pass, *independent of $p$*. This is the Baur–Strassen result and it is  remarkable.

#### What a Jacobian is, and why there are two modes

**The Jacobian is a table of every input→output sensitivity.** For $f:\mathbb{R}^n\to\mathbb{R}^m$, the entry $J_{ij} = \partial f_i/\partial x_j$ answers: *"if I nudge input $j$, how much does output $i$ move?"* Row $i$ collects everything affecting output $i$; column $j$ collects everything input $j$ affects.

It is simply the gradient generalized to functions with many outputs. If $m = 1$ (one output), the Jacobian is a single row — and that row is the gradient.

**Why you never build it.** Consider one layer mapping 4,096 activations to 4,096 activations. Its Jacobian is $4096\times4096 \approx 17$ million numbers — **for one layer, for one example in the batch.** Multiply by 100 layers and a batch of 64 and no machine can hold it. But here is the saving observation: *you never actually need the table.* You only ever need the answer to "what does this table do to one particular vector," and that can be computed without ever forming the table.

> **Analogy.** You want to know how much your monthly budget changes if rent goes up. You do not need to compute the full sensitivity table of every expense against every other — you need one number. Building the whole table to read one entry is the mistake; autodiff is the discipline of never building it.

**The two modes, in plain terms.**

- **Forward mode ($Jv$)** — pick an input direction, push it through the network, and see how all the outputs move. *One sweep answers "what does perturbing this one input do to everything?"* Cost: one pass **per input direction**.
- **Reverse mode ($v^\top J$)** — pick an output, walk backwards, and see how sensitive it is to all the inputs. *One sweep answers "what does everything do to this one output?"* Cost: one pass **per output**.

▸ **So the choice is decided purely by shape — count your inputs and outputs.** Many inputs, one output → reverse mode. One input, many outputs → forward mode.

**Now apply that to deep learning.** A network has $p = 10^8$ parameters (inputs to the derivative question) and exactly **one** scalar loss (output). One output, a hundred million inputs.

- Forward mode: one pass per parameter → $10^8$ passes. At even a millisecond each, that's over three years for a single gradient step.
- Reverse mode: one pass per output → **one** pass.

▸ **That is the whole reason backpropagation is reverse-mode**, and the text's point deserves emphasis: it is not a fact about neural networks at all. It is a fact about the *shape* of the problem — many knobs, one number to minimize. Any optimization with that shape wants reverse mode, whether it involves neural networks, fluid dynamics, or financial models.

**Why the loss must be a scalar.** Now you can see why every objective in this book collapses to one number. If your loss were a 10-vector, reverse mode would cost 10 passes. Scalar losses aren't an aesthetic preference — they are what makes training affordable. When you see a multi-objective setup, note that it is always reduced to a *weighted sum* first: $\mathcal{L} = \lambda_1\mathcal{L}_1 + \lambda_2\mathcal{L}_2$. That sum exists to preserve this property.

**The Baur–Strassen result, and why it is astonishing.** The cost of the gradient is $2$–$3\times$ a forward pass **regardless of how many parameters you have.** Naively, computing $10^8$ derivatives ought to cost roughly $10^8$ times more than computing one value. Instead you get all hundred million of them for the price of about two forward passes.

▸ **Without this single fact, deep learning would not exist as a field.** Not "would be slower" — would not exist. Every other component (GPUs, data, architectures) is an amplifier on top of an algorithm that must first be affordable at all.

> **Where this came from.** **Walther Baur and Volker Strassen** proved the cost bound in 1983 — the same Strassen famous for discovering in 1969 that matrices can be multiplied faster than the obvious $n^3$ algorithm, a result that shocked the field by refuting the widespread assumption that $n^3$ was optimal. The gradient-cost theorem is sometimes called the "cheap gradient principle." Like the Johnson–Lindenstrauss lemma, it came out of theoretical computer science with no application to learning in mind, and like J–L, it turned out to be load-bearing for an entire industry built decades later.

### 1.2.4 The Hessian and what "sharpness" means

$H_{ij} = \partial^2\mathcal{L}/\partial\theta_i\partial\theta_j$, symmetric, $p \times p$. You cannot store it ($p=10^8 \Rightarrow 10^{16}$ entries $= 40$ petabytes in fp32). You *can* compute Hessian–vector products cheaply:

▸ $$Hv = \nabla_\theta\big(\langle \nabla_\theta\mathcal{L},\ v\rangle\big)$$

i.e. differentiate the dot product of the gradient with a constant vector. Cost: one extra backward pass. This is the **Pearlmutter trick**, and it's how people actually measure $\lambda_{\max}(H)$ (power iteration on $Hv$) to study sharpness, flat minima, and Edge of Stability (Ch. 5, 19).

**Gauss–Newton / Fisher decomposition.** For a loss $\mathcal{L} = \ell(f_\theta(x), y)$,

$$H = \underbrace{J_f^\top \nabla^2_f \ell\, J_f}_{\text{Gauss–Newton, PSD}} + \underbrace{\sum_k (\nabla_f \ell)_k \nabla^2_\theta f_k}_{\text{curvature of the model}}$$

Near a good fit the second term is small (residuals $\to 0$), so $H \approx$ GGN, which is PSD. This is why practical second-order methods (K-FAC, Shampoo, natural gradient) approximate the GGN/Fisher rather than the true Hessian — the true Hessian has negative eigenvalues and inverting it would push you *uphill*.

#### The Hessian in plain English

If the gradient is the **slope**, the Hessian is the **curvature** — how fast the slope itself is changing. $H_{ij} = \partial^2\mathcal{L}/\partial\theta_i\partial\theta_j$ is a second derivative: *"as I nudge parameter $j$, how much does the gradient with respect to parameter $i$ change?"*

> **Analogy.** Driving: position is the loss, speed is the gradient, acceleration is the Hessian. The gradient tells you which way is downhill. The Hessian tells you whether the hill is a gentle bowl you can stride across or a narrow ravine where a normal-sized step overshoots and lands you on the opposite wall.

**Why "sharpness" is an eigenvalue.** Because $H$ is symmetric it has an eigendecomposition (§1.1.2), and its eigenvalues are the curvature along each principal direction. $\lambda_{\max}(H)$ is the curvature in the *most sharply curving* direction — the narrowest part of the ravine.

▸ **This directly sets your maximum stable learning rate.** For a quadratic bowl, gradient descent diverges once $\eta > 2/\lambda_{\max}$. Too big a step in the sharpest direction and you bounce up the far wall higher than you started, then higher again, and the loss explodes. **When a training run diverges after a learning-rate increase, this inequality is the reason.** Chapter 5's "Edge of Stability" is the observation that real training tends to hover right at this boundary rather than staying safely below it.

**Why you cannot store it.** $H$ is $p\times p$. For $p = 10^8$ that is $10^{16}$ entries — **40 petabytes**, versus roughly 0.4 GB for the gradient. This is the whole reason second-order optimization is hard: the information is wonderful, and completely unaffordable.

**Reading the Pearlmutter trick.** $Hv = \nabla_\theta(\langle\nabla_\theta\mathcal{L}, v\rangle)$ looks circular but is a clean piece of sleight of hand:

1. Compute the gradient $\nabla_\theta\mathcal{L}$ — a vector, as usual.
2. Dot it with a **fixed** vector $v$ to get a single number, $\langle\nabla_\theta\mathcal{L}, v\rangle$.
3. Differentiate *that number* again.

Since $v$ is constant, differentiating the dot product hands you exactly $Hv$. **You obtain a Hessian-vector product without ever forming the Hessian** — the same discipline as never building the Jacobian, for the same reason. Cost: one extra backward pass.

▸ And $Hv$ is enough to find $\lambda_{\max}$ via **power iteration**: repeatedly apply $H$ to a random vector and renormalize. By §1.1.2, $H^k v$ amplifies the largest-eigenvalue direction exponentially over all others, so after a few dozen iterations the vector has swung round to point along the top eigenvector. The exponential amplification that ruins gradient flow in deep networks is, here, a useful measuring instrument.

**The Gauss–Newton decomposition, term by term.** The formula splits the curvature into two sources:

- $J_f^\top\nabla_f^2\ell\,J_f$ — curvature that comes from **the loss function's own shape**, viewed through the model. Always PSD (curves upward everywhere).
- $\sum_k(\nabla_f\ell)_k\nabla^2_\theta f_k$ — curvature from **the model itself bending**, weighted by how wrong each output currently is. Can be negative.

The second term carries a factor of the residual — *how wrong you are*. Fit the data well and the residuals go to zero, so this term fades and $H \approx$ GGN, which is guaranteed PSD.

▸ **Why this matters practically.** Newton's method steps by $-H^{-1}\nabla\mathcal{L}$. If $H$ has a **negative** eigenvalue, inverting it flips the sign along that direction and the "descent" step confidently walks *uphill*. Using the GGN instead guarantees all-positive curvature and therefore an honest downhill step. **K-FAC, Shampoo, and natural gradient all approximate the GGN/Fisher precisely to buy this guarantee** — they are not merely cheaper approximations to the Hessian, they are approximations to a *better-behaved* matrix than the Hessian.

> **Where the names came from.** **Ludwig Otto Hesse** introduced his matrix in the 1840s studying curves and surfaces; Sylvester later named it after him. **Carl Gustav Jacob Jacobi** gave his name to the Jacobian in the 1840s. **Carl Friedrich Gauss** developed least squares around 1795 (publishing in 1809) to predict the orbit of the newly-discovered asteroid **Ceres**, which astronomers had lost behind the sun — Gauss's prediction of where it would reappear was correct, and made him famous across Europe at 24. **Ronald Fisher** introduced his information matrix in the 1920s while working at Rothamsted agricultural research station, analysing crop-fertilizer experiments. **Barak Pearlmutter** published the Hessian-vector trick in 1994, in a paper titled simply *Fast Exact Multiplication by the Hessian*. Orbital mechanics, agriculture, and 19th-century surface geometry: the optimizer in your training script is an assembly of parts built for entirely unrelated purposes.

---

## 1.3 Probability

### 1.3.1 The three-line core

$$\mathbb{E}[X] = \int x\,p(x)\,dx,\qquad \mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2,\qquad \mathrm{Cov}(X,Y) = \mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$$

**Linearity of expectation holds always** (even for dependent variables): $\mathbb{E}[aX+bY] = a\mathbb{E}[X]+b\mathbb{E}[Y]$.
**Variance is not linear**: $\mathrm{Var}(aX+bY) = a^2\mathrm{Var}(X)+b^2\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)$.

▸ For $n$ i.i.d. variables, $\mathrm{Var}\!\left(\frac{1}{n}\sum X_i\right) = \frac{\sigma^2}{n}$, so the **standard error is $\sigma/\sqrt{n}$.** To halve your error bars you need $4\times$ the data. This single fact governs Chapter 3 entirely, and it is why a 16-batch validation estimate is noisy.

#### Decoding the three-line core

**Expectation $\mathbb{E}[X]$** — the probability-weighted average of $X$. The integral $\int x\,p(x)\,dx$ is the continuous version of "multiply each value by how likely it is, then add up" (§0.3: read $\int$ as $\sum$).

**Variance $\mathrm{Var}(X)$** — how spread out $X$ is. The formula $\mathbb{E}[X^2] - \mathbb{E}[X]^2$ reads *"the average of the squares minus the square of the average,"* and the gap between those two is exactly the spread. Sanity check: if $X$ is always $5$, both terms are $25$ and the variance is $0$. Correct — no spread.

**Covariance $\mathrm{Cov}(X,Y)$** — whether two quantities move together. Positive: when one is above its average, so is the other. Negative: they move oppositely. Zero: no *linear* relationship (note: zero covariance does not mean independent — it only rules out straight-line association).

**Why linearity of expectation is such a big deal.** $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ holds **always**, even when $X$ and $Y$ are tangled up in complicated ways. You almost never get a rule this unconditional in probability, and much of the field's cleverness consists of rewriting a hard quantity as a sum so this rule applies.

**Why variance is not linear.** Because of the $2ab\,\mathrm{Cov}(X,Y)$ cross-term. Spreads only add cleanly when the things are uncorrelated. If two errors tend to occur together, their combined variance is *more* than the sum of parts; if they cancel, less.

**Now the punchline, derived.** For $n$ independent samples, all covariances vanish, so:

$$\mathrm{Var}\left(\frac1n\sum_i X_i\right) = \frac{1}{n^2}\sum_i \mathrm{Var}(X_i) = \frac{1}{n^2}\cdot n\sigma^2 = \frac{\sigma^2}{n}$$

The $1/n$ comes out **squared** (from $\mathrm{Var}(aX) = a^2\mathrm{Var}(X)$), while there are only $n$ terms to add — hence $\sigma^2/n$, and taking the square root gives the **standard error $\sigma/\sqrt{n}$**.

▸ **The $\sqrt{n}$ is the most financially consequential exponent in experimental machine learning.** Because error shrinks like $\sqrt{n}$ and not $n$:

| To improve precision by | You need this much more data |
|---|---|
| $2\times$ | $4\times$ |
| $10\times$ | $100\times$ |

> **Analogy.** Polling. To halve the margin of error on an election poll you must quadruple the number of people you call. This is why national polls stall around ±3% — going to ±1.5% costs four times as much for a modest gain, and nobody pays it.

**Why your validation number moves when nothing changed.** If per-example loss has standard deviation $\sigma \approx 1$ and you evaluate on 1,024 samples, your standard error is $1/\sqrt{1024} = 0.031$ nats. Compare that with §1.4.2, where a ** late-training improvement is $0.032$ nats. **The measurement noise is the same size as the effect you are trying to detect.** Two runs can differ by 0.03 nats with no real difference between them at all. This is the single most common way practitioners fool themselves, and it is why Chapter 3 exists.

#### Examples and non-examples: when is $\sigma/\sqrt{n}$ the right error bar?

**✅ The formula  applies**

| Situation | Why |
|---|---|
| 1,024 validation examples drawn independently | Independent, identically distributed — exactly the assumption |
| Averaging loss over a shuffled batch | Shuffling supplies the independence |
| Monte Carlo estimates with fresh random draws | Each draw is independent by construction |

**❌ Near-misses — looks like $n$ samples, but the formula lies**

| Situation | Why it breaks | What actually happens |
|---|---|---|
| 1,000 sentences from **10** documents | Sentences within a document are correlated | Effective $n$ is nearer 10; your error bar is far too small |
| $k$-fold cross-validation scores | Folds share training data, so scores are correlated | The naive standard error **understates** the true one (Chapter 3) |
| Time-series points measured every second | Consecutive points are nearly identical | Effective sample size collapses |
| The same 1,024 validation examples, re-evaluated each epoch | It's the *same* sample every time | Reduces run-to-run noise, but the bias against that fixed set never averages away |
| Data with duplicates | Duplicates carry no new information | $n$ counts rows, not information |

▸ **The boundary:** the $\sqrt{n}$ law needs **independent** samples. When samples are correlated, $n$ in the formula must be replaced by an *effective* sample size, which can be dramatically smaller. **Counting rows in your validation set is not the same as counting independent observations**, and the gap between the two is where overconfident conclusions come from.

> **Common misconception.** *"My validation set has 10,000 examples, so my measurement is precise."* Only if those 10,000 are independent and representative. Ten thousand sentences scraped from a hundred web pages behave statistically much more like a hundred samples than ten thousand. This is the most common reason a model that looked clearly better in evaluation fails to be better in production.

> **Common misconception.** *"Validation loss went down, so the model improved."* Only if the change exceeds the noise. With standard error 0.03 nats, a 0.02 improvement is indistinguishable from nothing at all. Practitioners routinely ship changes that were pure noise, and — worse — routinely *discard* good changes that happened to land on an unlucky evaluation.

> **Where this came from.** The $1/\sqrt{n}$ law traces to **Jacob Bernoulli's** *Ars Conjectandi*, published posthumously in 1713 — he worked on it for two decades and died before finishing. The Gaussian itself was first written down by **Abraham de Moivre** in 1733 as an approximation to coin-flip probabilities, and he could not find a use for it; **Gauss** and **Laplace** later established it as the law of errors. Bernoulli's original motivating question was, essentially, "how many observations do I need before I can trust an estimate?" — which is exactly the question a modern practitioner asks when deciding how large a validation set must be. Three hundred years later, the answer is still $\sqrt{n}$.

### 1.3.2 Law of total expectation / variance

$$\mathbb{E}[X] = \mathbb{E}_Y\big[\mathbb{E}[X\mid Y]\big]$$
▸ $$\mathrm{Var}(X) = \underbrace{\mathbb{E}_Y[\mathrm{Var}(X\mid Y)]}_{\text{within-group}} + \underbrace{\mathrm{Var}_Y(\mathbb{E}[X\mid Y])}_{\text{between-group}}$$

The variance decomposition is the engine behind:
- the **bias–variance decomposition** (Ch. 2),
- why a diffusion model's validation loss is extra noisy: one samples a random timestep $t$ per batch, so the between-$t$ variance $\mathrm{Var}_t(\mathbb{E}[\text{loss}\mid t])$ gets *added* on top of the within-batch sampling variance. Loss at $t=999$ and loss at $t=5$ are wildly different numbers. (Ch. 3, Ch. 12.)

### 1.3.3 Gaussians

$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\!\left(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\right)$$

The two facts that make diffusion models tractable:

▸ **Closure under affine maps:** if $x\sim\mathcal{N}(\mu,\Sigma)$ then $Ax+b \sim \mathcal{N}(A\mu+b,\ A\Sigma A^\top)$.

▸ **Closure under convolution:** if $x_1\sim\mathcal{N}(\mu_1,\sigma_1^2)$ and $x_2\sim\mathcal{N}(\mu_2,\sigma_2^2)$ independent, then $x_1+x_2 \sim \mathcal{N}(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$.

Together these give the closed-form $q(x_t\mid x_0)$ in DDPM (Ch. 11) — you can jump to any noise level in one step instead of simulating $t$ steps. Without this, diffusion training would be intractable.

**Reparameterization.** $x\sim\mathcal{N}(\mu,\sigma^2) \iff x = \mu + \sigma\epsilon,\ \epsilon\sim\mathcal{N}(0,1)$. This is what makes VAEs and diffusion differentiable — you move the randomness out of the computational path so gradients can flow through $\mu$ and $\sigma$ (Ch. 10).

#### Reading the Gaussian, and the two closure properties

The density formula is decoded piece by piece in [§0.12 of the primer](00-notation-and-math-primer.md) as a worked example — if it looks impenetrable, read that first. The one-sentence version: **"likelihood falls off exponentially with squared distance from the centre, measured in units of the distribution's own spread."**

**"Closure" means: do this operation to a Gaussian and you still have a Gaussian.** This is a rare and precious property. Most distributions, put through a transformation, turn into something with no name and no formula. Gaussians stubbornly refuse to.

**Closure under affine maps** — stretch, rotate, and shift a Gaussian and you get another Gaussian, with mean $A\mu + b$ and covariance $A\Sigma A^\top$. ("Affine" = linear map plus a shift.) The mean transforms the way a point would; the covariance gets sandwiched by $A$ on both sides because variance is a *squared* quantity, so the stretch factor has to be applied twice.

**Closure under convolution** — add two independent Gaussians and you get a Gaussian, with **means adding and variances adding**. Note carefully: *variances* add, not standard deviations. So combining two noise sources of $\sigma = 1$ each gives $\sigma = \sqrt{2} \approx 1.41$, not 2. Noise accumulates by $\sqrt{\cdot}$ — the same law as §1.3.1, and for the same reason.

▸ **Why these two facts make diffusion models possible.** A diffusion model's forward process adds a little Gaussian noise, a thousand times over. Naively, finding the image at step 500 requires simulating 500 steps. But "add Gaussian noise" is an affine map plus an independent Gaussian — *both* closure properties — so all thousand steps collapse into a single Gaussian you can write down in closed form. **You jump straight to any noise level in one step.** Training samples a random $t$ and needs $x_t$ immediately; without closure that would mean simulating hundreds of steps per training example, and diffusion models would be computationally hopeless. Chapter 20's $q(x_t\mid x_0)$ is this collapse.

**The reparameterization trick, and the problem it solves.** The notation $x = \mu + \sigma\epsilon$ says: *instead of drawing $x$ from a distribution whose parameters you're trying to learn, draw a fixed, parameter-free noise sample $\epsilon\sim\mathcal{N}(0,1)$ and then* **deterministically** *reshape it.*

Why this is necessary: you cannot differentiate through a sampling operation. "Draw a random number" has no derivative — there's no sense in which nudging $\mu$ nudges the output of a random draw. So the gradient can't reach $\mu$ and $\sigma$, and learning stalls.

▸ The trick moves the randomness **out of the path between the parameters and the loss.** In $x = \mu + \sigma\epsilon$, the randomness sits in $\epsilon$, which is off to one side and depends on nothing. The route from $\mu$ to $x$ is now plain arithmetic — $\partial x/\partial\mu = 1$, $\partial x/\partial\sigma = \epsilon$ — and gradients flow.

> **Analogy.** You want to know how a factory's output depends on its thermostat setting, but the room temperature also fluctuates randomly. Measuring "temperature" directly confounds the two. Instead, record the random fluctuation separately and write temperature = (setting) + (fluctuation). Now you can isolate the effect of the setting, because the noise has been factored out into its own term rather than being entangled with the thing you control.

> **Where this came from.** The reparameterization trick was introduced by **Diederik Kingma and Max Welling** in the 2013 paper *Auto-Encoding Variational Bayes* — the paper that introduced the VAE — and independently by Rezende, Mohamed, and Wierstra in 2014. The underlying statistical idea is much older, known in simulation as the "push-out" method or inverse-transform sampling, but its use to make a *neural network* differentiable through a sampling step was the unlock. Kingma is also the first author on Adam (Chapter 5), making him responsible for two of the most-cited pieces of machinery in modern deep learning; the Adam paper appeared in 2014, while he was still a PhD student.

### 1.3.4 Categorical distributions and the softmax

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Its Jacobian (needed constantly):
▸ $$\frac{\partial\, \mathrm{softmax}(z)_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

Combined with cross-entropy loss $\mathcal{L} = -\log p_y$, the gradient collapses beautifully:
▸ $$\frac{\partial \mathcal{L}}{\partial z_j} = p_j - \mathbb{1}[j=y]$$

**Derivation.** $\mathcal{L} = -z_y + \log\sum_k e^{z_k}$. Then $\partial\mathcal{L}/\partial z_j = -\delta_{jy} + \frac{e^{z_j}}{\sum_k e^{z_k}} = p_j - \delta_{jy}$. ∎

This is why the softmax–cross-entropy pair is numerically well behaved: **the gradient is just "predicted minus actual."** Never implement them separately; the fused version avoids computing $\log$ of a possibly-zero probability.

**Numerical stability:** always compute $\mathrm{logsumexp}(z) = z_{\max} + \log\sum_j e^{z_j - z_{\max}}$. Without the shift, $e^{z}$ overflows fp32 at $z \approx 88$.

**Temperature:** $\mathrm{softmax}(z/T)$. As $T\to 0$ it becomes argmax; as $T\to\infty$ it becomes uniform. Entropy is monotonically increasing in $T$.

#### Softmax, unpacked

**The job:** a network's final layer outputs $K$ unconstrained numbers called **logits** — any real values, positive or negative, not probabilities. You need probabilities: non-negative, summing to 1. Softmax is the converter.

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Two steps only:

1. **Exponentiate** every logit. $e^{z}$ is positive for any input, which handles the "non-negative" requirement, and it magnifies differences — turning a modest lead in logits into a large lead in probability.
2. **Divide by the total.** Now they sum to 1. The denominator $\sum_j e^{z_j}$ is the same for every $i$; it is purely a normalizer.

Worked example with $z = (2, 1, 0)$: exponentiate → $(7.39, 2.72, 1.00)$, total $11.11$, divide → $(0.67, 0.24, 0.09)$. A 1-unit logit lead became roughly a 3:1 probability ratio.

▸ **The name is a mild misnomer.** It is not a soft version of *max* (which would return one number); it's a soft version of **argmax** — a smoothed version of the one-hot "the winner is #2" vector. "Softargmax" would be accurate and nobody uses it.

**Why "$e$" specifically?** Because it makes the derivative clean, which is the next result.

**Reading the Jacobian $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$.** The Kronecker delta $\delta_{ij}$ is 1 when $i=j$ and 0 otherwise (§0.6), so this is two cases in one line:

- $i = j$: $p_i(1 - p_i)$ — **positive.** Raising a logit raises its own probability.
- $i \ne j$: $-p_ip_j$ — **negative.** Raising one logit lowers everyone else's.

▸ That negative off-diagonal term is softmax's defining behaviour: **probabilities compete.** Because they must total 1, one class can only gain at another's expense. This is why softmax classifiers handle mutually-exclusive labels well and multi-label problems badly — for "this image contains both a dog and a beach" you want independent sigmoids, since those facts don't compete.

**The gradient that makes everything work.** Combined with cross-entropy loss:

$$\frac{\partial\mathcal{L}}{\partial z_j} = p_j - \mathbb{1}[j = y]$$

where $\mathbb{1}[j=y]$ is 1 for the correct class and 0 otherwise. So the gradient is literally **"predicted minus actual"**:

| Case | Gradient | Effect |
|---|---|---|
| Correct class | $p_y - 1$ (negative) | Push this logit **up** |
| Wrong class | $p_j - 0$ (positive) | Push these logits **down** |
| Perfect prediction | $0$ | No update — nothing to learn |

▸ Notice how the messy $p_i(\delta_{ij}-p_j)$ Jacobian and the $-\log$ of cross-entropy **annihilate each other**, leaving pure subtraction. The $\log$ in cross-entropy and the $\exp$ in softmax are inverses; pairing them cancels both. This is not a coincidence — cross-entropy is the loss *designed* to pair with softmax, and the whole reason to use it over, say, squared error.

**Why you must fuse them in code.** Computing softmax then $\log$ separately means computing $\log(p)$ where $p$ might have underflowed to exactly $0$, giving $-\infty$ and a NaN that poisons the run. The fused version never forms $p$ at all, using $\mathcal{L} = -z_y + \mathrm{logsumexp}(z)$ instead. This is why every framework provides `cross_entropy_with_logits` and why passing it already-softmaxed inputs is a classic bug.

**The logsumexp shift, concretely.** $\mathrm{logsumexp}(z) = z_{\max} + \log\sum_j e^{z_j - z_\max}$. Subtracting the max makes the largest exponent exactly $e^0 = 1$, so nothing can overflow; the identity holds because factoring $e^{z_\max}$ out of the sum turns into an additive $z_{\max}$ once you take the log. Without it, $e^{89}$ overflows fp32 to infinity — and logits of 90 are entirely reachable in a large model.

**Temperature, intuitively.** Dividing logits by $T$ before the softmax rescales how decisive the distribution is:

- $T < 1$: gaps widen → **more confident**, more repetitive text.
- $T = 1$: the model's own distribution, untouched.
- $T > 1$: gaps shrink → **more uniform**, more surprising (and more error-prone) text.
- $T\to 0$: pure argmax, fully deterministic.

When you move a temperature slider in a chat interface, this division is the only thing happening.

> **Where this came from.** The formula is the **Boltzmann (or Gibbs) distribution** from 1868 statistical mechanics, where it describes the probability that a physical system occupies a given energy state — and $T$ there is *literal temperature*. Heat a gas and its molecules spread across more states; cool it and they settle into the lowest-energy one. The machine-learning name "softmax" was popularized by **John S. Bridle** around 1989–90. So when you raise an LLM's temperature to get more creative output, you are invoking a 19th-century thermodynamics equation with its original physical meaning surprisingly intact: more heat, more disorder, more entropy.

---

## 1.4 Information theory

### The one-line idea

Entropy measures how surprised you should expect to be. Cross-entropy measures how surprised you *actually* are when you use the wrong beliefs. KL divergence is the gap — the extra surprise you pay for being wrong.

### The analogy

You're designing a code to transmit messages. Entropy is the shortest average message length possible if you know the true frequencies. Cross-entropy is your average message length if you built your codebook from *wrong* frequencies. KL is the wasted bits.

### 1.4.1 Definitions

$$H(p) = -\sum_x p(x)\log p(x) \qquad \text{(entropy)}$$
$$H(p,q) = -\sum_x p(x)\log q(x) \qquad \text{(cross-entropy)}$$
▸ $$\mathrm{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)$$

Properties:
- $\mathrm{KL}(p\|q) \ge 0$, with equality iff $p=q$ (Gibbs' inequality, provable by Jensen).
- **Not symmetric.** $\mathrm{KL}(p\|q)\ne\mathrm{KL}(q\|p)$, and the asymmetry matters enormously.
- Not a metric (no triangle inequality).

**Forward vs reverse KL — the mode-covering / mode-seeking distinction.**

- $\mathrm{KL}(p\|q)$ ("forward", used in MLE): the expectation is over $p$. Wherever $p$ has mass, $q$ *must* have mass, or you pay $\log(p/0) = \infty$. ⇒ **mode-covering**, $q$ spreads out and blurs. This is why maximum-likelihood generative models produce blurry samples.
- $\mathrm{KL}(q\|p)$ ("reverse", used in variational inference and some RL): the expectation is over $q$. $q$ can safely ignore regions of $p$ by putting zero mass there. ⇒ **mode-seeking**, $q$ collapses onto one mode. This is why VI underestimates posterior variance, and why some RLHF objectives collapse diversity.

▸ **Maximum likelihood = minimizing forward KL.** Proof: $\arg\max_\theta \frac1n\sum_i \log p_\theta(x_i) \to \arg\max_\theta \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] = \arg\min_\theta \mathrm{KL}(p_{\text{data}}\|p_\theta)$ because $H(p_{\text{data}})$ is a constant w.r.t. $\theta$. ∎

So **every time you minimize cross-entropy loss, you are minimizing a KL divergence to the data distribution.** Your `val_realCE` is literally an estimate of $H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}}\|p_\theta)$ — the first term is a fixed floor you can never get below.

#### The three quantities, built from scratch

First, the abbreviations used throughout this section, in full:

| Short form | Full form |
|---|---|
| KL | **Kullback–Leibler** divergence |
| MLE | **Maximum likelihood estimation** |
| CE | **Cross-entropy** |
| VI | **Variational inference** |
| RLHF | **Reinforcement learning from human feedback** |

**Start with "surprise."** If something certain happens, you learn nothing. If something you thought impossible happens, you learn a great deal. Information theory makes this precise by defining the surprise of an outcome as

$$\text{surprise}(x) = -\log p(x)$$

Check that it behaves correctly: $p = 1$ gives $-\log 1 = 0$ (certain, no surprise). $p = 0.001$ gives $-\log(0.001) = 6.9$ (very surprising). As $p\to 0$, surprise $\to\infty$.

Why the logarithm specifically? Because surprise should **add** when independent things happen. Two unrelated events with probabilities $p_1$ and $p_2$ have joint probability $p_1p_2$, and $\log$ is the unique function turning that product into a sum: $-\log(p_1p_2) = -\log p_1 - \log p_2$. Learning two independent facts should be twice the information of learning one, and only the logarithm delivers that.

**Entropy $H(p)$ — your *expected* surprise.**

$$H(p) = -\sum_x p(x)\log p(x) = \mathbb{E}_{x\sim p}[\text{surprise}(x)]$$

Read: *"for every outcome, multiply how surprising it is by how often it happens, and add up."* It measures **how unpredictable a source is on average.**

- A coin that always lands heads: $H = 0$. Nothing to learn.
- A fair coin: $H = \log 2 = 0.693$ nats (= 1 bit). Maximum for two outcomes.
- A fair 10-sided die: $H = \log 10 = 2.303$ nats.

▸ **Entropy is maximized by the uniform distribution and minimized by certainty.** "How uncertain am I?" and "how much information do I gain by being told the answer?" turn out to be *the same number* — which is the central insight of the field.

**Cross-entropy $H(p,q)$ — your surprise when your beliefs are wrong.**

$$H(p,q) = -\sum_x p(x)\log q(x)$$

Look closely at the two different letters. Outcomes occur with the **real** frequencies $p$, but your surprise is computed from your **believed** probabilities $q$. *Reality is $p$; you think it's $q$; the cost is $H(p,q)$.*

**Kullback–Leibler divergence — the penalty for being wrong.**

$$\mathrm{KL}(p\|q) = H(p,q) - H(p)$$

**Your actual surprise, minus the minimum surprise that was achievable.** It is pure waste — the part of your confusion attributable to holding the wrong model rather than to the world being  random.

> **Analogy (the book's coding story, made concrete).** You are assigning Morse-code-like symbols to letters. English uses "e" constantly and "z" rarely, so you give "e" a short code and "z" a long one. Entropy is the best possible average message length given the true letter frequencies. Now suppose you built your codebook from *Polish* frequencies and used it for English: your messages get longer. Cross-entropy is that inflated length. KL divergence is the wasted characters — and notice it can only be zero or positive, since a wrong codebook can never beat the right one.

**Why $\mathrm{KL} \ge 0$ always.** Because $H(p,q)\ge H(p)$: no codebook beats the one built from the true frequencies. Equality only when $q = p$.

▸ **Why the asymmetry matters so much in practice.** $\mathrm{KL}(p\|q)\neq\mathrm{KL}(q\|p)$ because the expectation is taken over *different* distributions — the first weights by reality, the second by your beliefs. This is not a technicality; it determines what your generative model does when it cannot represent the data perfectly:

| | Forward $\mathrm{KL}(p\|q)$ | Reverse $\mathrm{KL}(q\|p)$ |
|---|---|---|
| Average over | reality $p$ | your model $q$ |
| Punishes | missing any real mode | putting mass where there's none |
| Result | **mode-covering** — spreads out | **mode-seeking** — collapses to one |
| Symptom | blurry samples | reduced diversity |
| Used in | maximum likelihood estimation | variational inference, some reinforcement learning from human feedback |

> **Analogy.** You must describe "food people eat for breakfast" with one dish. Forward KL forces you to cover everything — you invent an unappetizing average of cereal, eggs, and toast that nobody actually eats (blurry). Reverse KL lets you say "toast" and ignore the rest — a real breakfast, but you've silently dropped most of the distribution (mode collapse). **Neither is wrong; they answer different questions.** Blurry maximum-likelihood samples and collapsed reinforcement-learning diversity are the same mathematical fact wearing two hats.

**"Maximum likelihood = minimizing forward KL," read slowly.** The proof line says: maximizing average log-probability of your data is the same as minimizing $\mathrm{KL}(p_{\text{data}}\|p_\theta)$, because the two differ only by $H(p_{\text{data}})$ — a property of the *dataset*, not of $\theta$. Adding a constant doesn't move an argmin.

▸ **Consequence for reading your own training logs.** Your loss is $H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}}\|p_\theta)$: an **irreducible floor** plus **your model's error**. Only the second term is trainable. This is why a loss curve flattening at 1.4 rather than 0 may mean the model has *finished*, not that it has *failed*, and why "how close am I to the floor?" is a far more useful question than "how close am I to zero?" A cross-entropy of zero would require the data to be perfectly predictable, which for natural language it emphatically is not.

#### Examples and non-examples: entropy, cross-entropy, and KL

**✅ Worked numeric examples** — a 4-class problem, true label = class 1.

| Model's prediction $q$ | Cross-entropy $-\log q_1$ | Reading |
|---|---|---|
| $(1.00,\ 0,\ 0,\ 0)$ | $0$ | Perfect and certain |
| $(0.90,\ 0.05,\ 0.03,\ 0.02)$ | $0.105$ | Confident and right |
| $(0.25,\ 0.25,\ 0.25,\ 0.25)$ | $1.386 = \log 4$ | Knows nothing |
| $(0.05,\ 0.90,\ 0.03,\ 0.02)$ | $3.00$ | Confident and **wrong** — the expensive case |
| $(0.001,\ 0.999,\ 0,\ 0)$ | $6.91$ | Catastrophically overconfident |

▸ **Notice the asymmetry of the penalty.** Being right and confident saves you 0.105 versus a uniform guess's 1.386 — a gain of about 1.3. Being *wrong* and confident costs 3.0, and being extremely wrong costs 6.9. **Cross-entropy punishes confident errors far more than it rewards confident successes**, which is exactly why models trained on it learn to hedge, and why calibration (Chapter 33) is a live concern.

**❌ Near-misses — commonly confused with entropy or KL**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Accuracy | Ignores confidence entirely — 51% and 99% both count as "right" | A **threshold** metric, not a proper scoring rule |
| $\mathrm{KL}(p\|q)$ as a "distance" | Asymmetric, violates the triangle inequality | A **divergence** |
| Cross-entropy of zero as the goal | Impossible whenever data is  ambiguous | The goal is $H(p_{\text{data}})$, the irreducible floor |
| Perplexity compared across tokenizers | Different vocabularies, different $K$ | Not comparable — a smaller vocabulary flatters perplexity |
| Entropy of the model's output | That's the model's *confidence* | Not $H(p_{\text{data}})$, which is a property of the world |

▸ **The boundary:** entropy is a property of a **distribution**; cross-entropy involves **two** distributions (reality and belief); KL is the **gap** between them. Confusing "the model is uncertain" (high output entropy) with "the data is ambiguous" (high $H(p_\text{data})$) is the most consequential mix-up here — the first is fixable by training, the second is not.

> **Common misconception.** *"A perfect model reaches zero loss."* Only if the world is deterministic. For next-token prediction on natural text, many continuations are  valid; the entropy of the data is a hard floor that no architecture or scale can breach. A model at 1.52 nats against a floor of 1.40 has captured nearly everything available — and a team chasing the remaining 0.12 without knowing the floor exists can burn a quarter for nothing.

> **Common misconception.** *"Lower perplexity always means a better language model."* Perplexity depends on the tokenizer. A model with a larger vocabulary emits fewer tokens per sentence, each carrying more information, which changes per-token perplexity without changing the model's quality at all. **Comparing perplexity across tokenizers is meaningless**, and it is a mistake that has appeared in published papers.

> **Common misconception.** *"KL divergence measures how different two distributions are, like a distance."* It's asymmetric — $\mathrm{KL}(p\|q)$ and $\mathrm{KL}(q\|p)$ can differ enormously, and one can be infinite while the other is finite. That asymmetry isn't a defect to be tolerated; as the mode-covering/mode-seeking table shows, **it is the whole reason different training objectives produce qualitatively different failure modes.**

> **Where this came from.** **Claude Shannon** created information theory essentially single-handedly in *A Mathematical Theory of Communication* (1948), written at Bell Labs — a paper that arrived so complete that colleagues described it as having no visible antecedents. He had done wartime cryptography work with Alan Turing, who visited Bell Labs in 1943, and the two took tea together while both were forbidden from discussing their actual projects. The name "entropy" came from a suggestion by **John von Neumann**, whose reported advice was that Shannon should use it because the same formula appears in statistical mechanics and, besides, "nobody knows what entropy really is, so in a debate you will always have the advantage." The anecdote may be embellished — it comes to us secondhand — but the mathematical parallel with Boltzmann's thermodynamic entropy is exact, not merely poetic. **Solomon Kullback and Richard Leibler** published their divergence in 1951 in *On Information and Sufficiency*; both were cryptanalysts who worked for the U.S. Army's Signal Intelligence Service and later the National Security Agency, where the measure was used to quantify how well an intercepted message matched a hypothesized cipher. The quantity now used to train every large language model on earth was invented for code-breaking.

### 1.4.2 Calibrating cross-entropy numbers

Cross-entropy in nats. **Perplexity** $= e^{H}$, interpretable as "effective number of equally-likely choices."

| CE (nats) | Perplexity | Reading |
|---|---|---|
| $\log 2 = 0.693$ | 2 | coin flip |
| $1.0$ | 2.72 | ~2.7 effective options |
| $1.524$ | **4.59** | Case Study A, epoch 37 |
| $1.556$ | 4.74 | Case Study A, epoch 22 |
| $\log 10 = 2.303$ | 10 | uniform over 10 classes |

▸ Going from $1.556 \to 1.524$ is $\Delta = 0.032$ nats, which shrinks effective branching from $4.74$ to $4.59$ — about a **3.2% reduction in perplexity.** That is the size of a typical late-training improvement. Hold that number in mind for Chapter 3: is a 1,024-sample measurement precise enough to resolve 0.032 nats?

**Rule of thumb for calibration:** if a vocabulary has $K$ symbols and the model were uniform, CE $=\log K$. A CE of 1.52 against $K=10$ ($\log 10 = 2.30$) means the model has removed about 34% of the uncertainty. Always compare a cross-entropy to $\log K$, never to zero.

#### Making cross-entropy numbers mean something

**Nats versus bits.** A **nat** ("natural unit of information") is the unit you get using natural logarithm $\log_e$; a **bit** ("binary digit") uses $\log_2$. They differ only by a constant: $1$ nat $= 1/\ln 2 \approx 1.44$ bits. Machine learning uses nats because $\log_e$ differentiates cleanly; information theory traditionally uses bits because they correspond to yes/no answers. Same quantity, different rulers — like metres and feet.

**Perplexity, and why it is the useful number.** $\text{perplexity} = e^{H}$ converts a cross-entropy back into an intuitive count:

▸ **"The model is as confused as if it were choosing uniformly among this many options."** A perplexity of 4.59 means the model has narrowed things down to the equivalent of a fair 4.59-sided die at each step. That is a statement you can hold in your head; "1.524 nats" is not.

Why exponentiating does this: a uniform distribution over $K$ options has entropy $\log K$, so $e^{\log K} = K$ recovers the count exactly. For non-uniform distributions it returns an *effective* count — exactly the same construction as stable rank in §1.1.3, which converted a spread of singular values into an "effective number of directions."

**Reading the table.** Cross-entropy of $\log 2 = 0.693$ nats → perplexity 2 → "as uncertain as a coin flip." Cross-entropy of $\log 10 = 2.303$ → perplexity 10 → "no better than guessing among 10 classes."

▸ **Why you must always compare to $\log K$ and never to zero.** Zero cross-entropy means perfect prediction, which is only achievable if the data is  deterministic. The meaningful baseline is $\log K$ — what you would score knowing nothing. A cross-entropy of 1.52 against a baseline of 2.30 means you have removed $(2.30-1.52)/2.30 \approx 34\%$ of the initial uncertainty. **A raw loss number is meaningless without its baseline**; a cross-entropy of 4.0 is excellent for a 50,000-word vocabulary ($\log 50000 = 10.8$) and catastrophic for a binary task ($\log 2 = 0.69$).

**Why the 3.2% improvement is worth a paragraph.** Going $1.556\to1.524$ looks negligible written as nats. In perplexity it is $4.74\to4.59$ — the model went from juggling 4.74 effective options to 4.59. That is a real gain, and it is *typical* of late-stage training. Now recall §1.3.1: measuring on 1,024 samples carries a standard error near $0.031$ nats, **essentially the size of the improvement itself.** The improvement is real but your instrument can barely see it. Holding both numbers simultaneously — the effect size and the measurement noise — is the difference between an engineer and someone reading tea leaves.

> **Where perplexity came from.** The measure was introduced by **Frederick Jelinek's** speech recognition group at IBM in the 1970s, who needed a way to say how hard a recognition task was before running an experiment. Jelinek is also the source of machine learning's most-quoted joke — "Every time I fire a linguist, the performance of the speech recognizer goes up" — a remark from around 1985 that he later said he regretted, and which captured the then-heretical view that statistical methods would beat hand-built linguistic rules. That bet is the direct ancestor of every large language model in this book. The word "bit," incidentally, was coined by **John Tukey** at Bell Labs as a contraction of "binary digit," and Shannon credited him for it in the 1948 paper.

### 1.4.3 Mutual information

$$I(X;Y) = \mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big) = H(X) - H(X\mid Y)$$

"How many nats of uncertainty about $X$ does knowing $Y$ remove." Symmetric. $\ge 0$.

Appears in: InfoNCE contrastive learning (Ch. 15, where the loss is a lower bound on $I$), the information bottleneck, and disentanglement objectives ($\beta$-VAE).

#### Mutual information, decoded

**Mutual information $I(X;Y)$ measures how much knowing one thing tells you about another.** The two formulas are two ways to see it:

- $\mathrm{KL}(p(x,y)\,\|\,p(x)p(y))$ — *"how far are $X$ and $Y$ from being independent?"* If they were independent, the joint distribution would exactly equal the product of the individual ones, and the Kullback–Leibler divergence would be 0. So mutual information is **the distance from independence.**
- $H(X) - H(X\mid Y)$ — *"my uncertainty about $X$ before knowing $Y$, minus my uncertainty after."* The **reduction in confusion**.

Here $H(X\mid Y)$ is **conditional entropy**: how uncertain you remain about $X$ once $Y$ is revealed.

> **Analogy.** $X$ = "is it raining?", $Y$ = "is the pavement wet?" Before looking outside, you're quite uncertain about the rain. Told the pavement is wet, you're much less uncertain. That drop in uncertainty, measured in nats, is the mutual information. And symmetry means it works equally well in reverse — learning it's raining tells you just as much about the pavement.

▸ **Symmetric and non-negative.** $I(X;Y) = I(Y;X)$ — information is a two-way street, even though the Kullback–Leibler divergence it's built from is famously *not* symmetric. And $I \ge 0$: learning something can never, on average, make you *more* uncertain. (It can on any particular occasion — a surprising observation can  confuse you — but not in expectation.)

**Full forms of the abbreviations here:** *InfoNCE* is **Information Noise-Contrastive Estimation**; *$\beta$-VAE* is a **beta-weighted variational autoencoder**. The "information bottleneck" is the principle that a good representation should keep the information relevant to the task and discard everything else — stated precisely, maximize $I(\text{representation}; \text{label})$ while minimizing $I(\text{representation}; \text{input})$.

> **Where this came from.** Mutual information is Shannon's, from the same 1948 paper. The **information bottleneck** principle was proposed by **Naftali Tishby**, Fernando Pereira, and William Bialek in 1999, and Tishby later argued it explains what deep networks do during training: an early "fitting" phase followed by a long "compression" phase in which the network discards input information irrelevant to the label. The claim set off one of the more heated methodological arguments in deep learning around 2017–2018, with subsequent work showing the compression phase depends on choices like the activation function and does not appear universally. It remains  unsettled — a useful reminder that this book's later chapters describe an empirical science with live disputes, not a closed body of theory.

### 1.4.4 Jensen's inequality and the ELBO

For convex $\varphi$: $\varphi(\mathbb{E}[X]) \le \mathbb{E}[\varphi(X)]$. For concave (like $\log$): $\log\mathbb{E}[X] \ge \mathbb{E}[\log X]$.

**The ELBO derivation** — this appears in VAEs, diffusion, and variational inference, so derive it once here properly.

$$
\log p_\theta(x) = \log\int p_\theta(x,z)\,dz = \log\int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}dz = \log\mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
$$

Apply Jensen ($\log$ is concave):

▸ $$\log p_\theta(x) \ \ge\ \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] \ =:\ \mathrm{ELBO}(\theta,\phi;x)$$

And the **gap is exactly a KL**:

$$\log p_\theta(x) - \mathrm{ELBO} = \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big) \ \ge 0$$

*Proof of the gap:* expand $\mathrm{ELBO} = \mathbb{E}_q[\log p_\theta(z\mid x) + \log p_\theta(x) - \log q_\phi(z\mid x)] = \log p_\theta(x) - \mathrm{KL}(q_\phi\|p_\theta(\cdot\mid x))$. ∎

**Why this matters:** maximizing the ELBO does two things at once — it pushes up the true log-likelihood *and* pushes $q_\phi$ toward the true posterior. You never need the intractable posterior. Every diffusion loss in Chapters 11 and 12 is a rearranged ELBO.

Standard rewriting:
$$\mathrm{ELBO} = \underbrace{\mathbb{E}_q[\log p_\theta(x\mid z)]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}(q_\phi(z\mid x)\,\|\,p(z))}_{\text{regularization}}$$

#### Jensen's inequality and the ELBO, decoded

**ELBO** stands for **Evidence Lower BOund**. "Evidence" is the statistician's name for $p_\theta(x)$ — how probable your model thinks the observed data is — and this quantity is a *lower bound* on it. The name is a literal description.

**Jensen's inequality first.** For a **concave** function like $\log$ (one that curves downward, like a hill):

$$\log\mathbb{E}[X] \ge \mathbb{E}[\log X]$$

*"The log of the average is at least the average of the logs."*

> **Analogy.** Average two points on a dome and the midpoint sits *inside* the dome — below the surface. Reading the surface height at the average position gives a higher number than averaging the two heights. That gap is Jensen's inequality, and it is nothing more than the geometric fact that a curved surface lies above its own chords.

**The problem the ELBO solves.** You want to maximize $\log p_\theta(x)$ — make the observed data probable. But computing it requires $\int p_\theta(x,z)\,dz$: summing over **every possible value of the hidden variable $z$**. For a latent vector of even modest size that integral has no closed form and cannot be computed. You are stuck before you begin.

**The three-step trick.** Follow the chain in the derivation above:

1. **Multiply by 1.** Insert $\frac{q_\phi(z\mid x)}{q_\phi(z\mid x)}$ — legal, changes nothing, but now a distribution $q_\phi$ you *do* control appears in the expression.
2. **Recognize an expectation.** $\int q_\phi(z\mid x)\,[\cdots]\,dz$ is by definition $\mathbb{E}_{q_\phi}[\cdots]$ — and expectations can be *estimated by sampling*, even when integrals can't be computed.
3. **Apply Jensen.** Move the $\log$ inside the expectation. It's an inequality, so you get a quantity that is guaranteed **less than or equal to** what you wanted.

▸ **The move in one sentence: you cannot compute the thing you want, so you construct something computable that is guaranteed to sit below it, and push *that* up instead.** Since your bound never exceeds the true value, driving the bound up drives the true value up with it.

> **Analogy.** You want to know the height of a mountain hidden in cloud. You can't measure it. But you can measure a ridge you know lies below the summit. Push the ridge measurement as high as you can and you have both a guaranteed floor on the summit's height *and*, as the ridge approaches the peak, an increasingly accurate estimate of it.

**The gap is exactly a Kullback–Leibler divergence** — the most elegant part:

$$\log p_\theta(x) - \mathrm{ELBO} = \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)\ \ge 0$$

The slack between your bound and the truth is precisely *how wrong your guessed distribution $q_\phi$ is about the true posterior $p_\theta(z\mid x)$*. Two consequences:

- Because Kullback–Leibler divergence is never negative, the ELBO is **always** a valid lower bound. No conditions.
- Maximizing the ELBO does **two useful things at once**: raises the true log-likelihood, *and* squeezes the gap by dragging $q_\phi$ toward the true posterior. You get approximate inference for free as a side effect of fitting.

**The standard rewriting, in plain terms.** The final form splits into two competing jobs:

| Term | Reads as | Wants |
|---|---|---|
| $\mathbb{E}_q[\log p_\theta(x\mid z)]$ | **reconstruction** | "From the code $z$, rebuild the input accurately" |
| $-\mathrm{KL}(q_\phi(z\mid x)\|p(z))$ | **regularization** | "Keep the codes close to a simple prior distribution" |

▸ **These pull against each other, productively.** Reconstruction alone would memorize, giving every input its own private code in a scattered, meaningless latent space. The regularizer forces codes into a tidy, well-populated region — which is what makes it possible to *sample* a new code and decode it into something plausible. **The tension between the two terms is what turns an autoencoder into a generative model.** Chapters 19 and 20 are, in large part, variations on how to weight and rearrange these two terms.

> **Where this came from.** **Johan Jensen**, a Danish mathematician, proved his inequality in 1906 — and was an amateur, in the sense that he never held an academic post: he worked his entire career as an engineer for the Copenhagen Telephone Company, doing mathematics in his spare time and eventually becoming head of its technical department. Variational methods themselves descend from the calculus of variations developed by Euler and Lagrange in the 1700s for problems like "what shape does a hanging chain take?" The ELBO reached machine learning through the variational Bayes literature of the 1990s (Michael Jordan, Zoubin Ghahramani, Tommi Jaakkola, Lawrence Saul), and became ubiquitous in 2013 when Kingma and Welling used it for the variational autoencoder. Every diffusion model loss in Chapters 20 and 21 is this same bound, rearranged.

---

## 1.5 A worked example tying it together

**Setup.** Your model outputs logits $z \in \mathbb{R}^{K}$ over $K$ atom types for each position. True type is $y$. Loss is CE.

1. Forward: $p = \mathrm{softmax}(z)$, $\mathcal{L} = -\log p_y$.
2. Gradient w.r.t. logits: $\bar z = p - e_y$ (§1.3.4). Note $\|\bar z\|\le\sqrt2$ always — **cross-entropy has bounded logit gradients**, which is why it's stable and why MSE on logits isn't.
3. If $z = Wh + b$ with hidden state $h$: $\bar W = \bar z h^\top$, $\bar h = W^\top\bar z$, $\bar b = \bar z$ (§1.2.2).
4. If the model is perfect, $p = e_y$, so $\bar z = 0$ and training stops. If the model is uniform, $p_y = 1/K$, $\mathcal{L} = \log K$, and $\bar z$ has magnitude $\approx 1$.
5. In expectation over the data, $\mathbb{E}[\mathcal{L}] = H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}}\|p_\theta)$. The floor $H(p_{\text{data}})$ is the **irreducible entropy of the data** — for molecules with  chemical ambiguity at a given noise level, this is not small.

▸ **Practical implication:** if $H(p_{\text{data}} \mid \text{noise level } t) \approx 1.4$ nats, then a measured CE of $1.524$ means the model is only $0.12$ nats from optimal — and further gains are necessarily tiny and hard to measure. Estimating the irreducible floor (e.g. by fitting a much larger model, or by measuring CE at $t\to 0$) tells you whether the remaining headroom actually exists.

### Walking through the worked example

This example is where the whole chapter converges, so here is each step with its notation spelled out.

**The setup.** The model emits $K$ **logits** — raw, unconstrained scores, one per atom type. $y$ is the correct type. $e_y$ is the **one-hot vector** for $y$: all zeros except a single 1 at position $y$.

**Step 1 — forward.** Softmax turns logits into probabilities (§1.3.4); the loss is $-\log$ of the probability assigned to the correct answer. Assign it 90% and you pay $-\log(0.9) = 0.105$. Assign it 10% and you pay $2.303$. **The loss is your surprise at the truth.**

**Step 2 — the gradient, and why it is bounded.** $\bar z = p - e_y$ is "predicted minus actual." The claim $\|\bar z\|\le\sqrt2$ follows from $p$ being a probability vector: the worst case is total confidence in the wrong class, giving a vector with a $+1$ and a $-1$, whose length is $\sqrt{1^2+1^2}=\sqrt2$.

▸ **This bound is why cross-entropy trains stably.** No matter how catastrophically wrong the model is, the gradient at the logits cannot exceed $\sqrt2$ — errors can't produce unbounded updates. Squared error on logits has no such ceiling: being wrong by 100 produces a gradient of 200, and one bad batch can destroy a run. **Bounded gradients are a safety property, not an accident**, and they are the main reason cross-entropy rather than squared error is used for classification.

**Step 3 — backprop.** Apply the two rules from §1.2.2 to $z = Wh + b$. Note $\bar b = \bar z$ because the bias adds directly to the output, so its sensitivity passes through unchanged.

**Step 4 — the two extremes.** Perfect model: $p = e_y$, so $\bar z = 0$ and updates stop — **there is nothing left to learn, and the mathematics knows it.** Ignorant model: $p_j = 1/K$ for all $j$, loss $= \log K$, gradients of order 1. The gradient's *size* is itself a readout of how wrong the model is.

**Step 5 — the decomposition, in practical terms.** $\mathbb{E}[\mathcal{L}] = H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}}\|p_\theta)$ splits your loss into **the world's inherent randomness** plus **your model's error**. Only the second is trainable.

▸ **The practical implication is the most useful paragraph in this chapter.** If the irreducible floor is 1.4 nats and you measure 1.524, you have 0.12 nats of headroom left — and your measurement noise (§1.3.1) is about 0.03 nats. You are within a factor of four of the noise floor. **Knowing this saves months.** The common failure is spending a quarter chasing an improvement that was never mathematically available, because nobody estimated the floor. Whenever a loss curve flattens, the first question is not "what should I try next?" but "how do I know there is anything left to get?"

---

## Did you know?

- **The word "matrix" means "womb."** James Joseph Sylvester coined the term in 1850, from the Latin for womb, because he thought of it as the thing *from which* determinants are born — a matrix generates determinants the way a womb generates offspring. Sylvester coined a startling amount of standard mathematical vocabulary, including "discriminant" and "graph" in the network sense.

- **Eigenvalues were named in German and never fully translated.** *Eigen* means "own" or "characteristic," so an eigenvector is a matrix's "own vector." English absorbed the German prefix rather than translating it, which is why "eigenvalue" is a linguistic hybrid. French and Spanish did translate it — *valeur propre* and *valor propio*, "proper value."

- **The Singular Value Decomposition was discovered twice in two years by mathematicians who never learned of each other's work.** Eugenio Beltrami published in 1873, Camille Jordan in 1874. Neither had any application in mind; both were studying bilinear forms as pure geometry.

- **Backpropagation was published in 1970 in Finnish, in a master's thesis, about rounding errors.** Seppo Linnainmaa described reverse-mode automatic differentiation with no reference to neural networks whatsoever. It took sixteen years and the 1986 Rumelhart–Hinton–Williams paper for the field to notice.

- **Shannon's information theory paper arrived essentially without precedent.** Colleagues at Bell Labs described *A Mathematical Theory of Communication* (1948) as appearing fully formed. Shannon also built a mechanical mouse that could solve mazes, a rocket-powered frisbee, a calculator that worked in Roman numerals which he named THROBAC, and a machine whose sole function was to switch itself off — a device he called the "Ultimate Machine."

- **The Kullback–Leibler divergence was developed by code-breakers.** Solomon Kullback and Richard Leibler were cryptanalysts for the U.S. Army's Signal Intelligence Service and later the National Security Agency. The measure that trains every modern language model was built to test whether an intercepted message matched a guessed cipher.

- **Jensen's inequality was proved by a telephone engineer.** Johan Jensen never held an academic position; he ran the technical department of the Copenhagen Telephone Company and did mathematics in his spare time.

- **The softmax function is 19th-century thermodynamics.** It is the Boltzmann distribution from 1868, where the temperature parameter $T$ is *literal temperature*. Raising an LLM's temperature to make it more creative applies a physics equation with its original meaning nearly intact: more heat, more disorder.

- **Gauss invented least squares to find a lost asteroid.** Ceres was discovered in January 1801 and then vanished behind the sun. From 41 days of observation, the 24-year-old Gauss predicted where it would reappear. Astronomers pointed their telescopes there and found it, making him instantly famous across Europe.

- **The Johnson–Lindenstrauss lemma was a footnote to its own paper.** The 1984 result was a technical stepping stone toward a theorem about extending Lipschitz maps. It is now cited far more than the theorem it was built to prove, and it underwrites vector databases, locality-sensitive hashing, and the theory of superposition.

- **"Bit" is a contraction invented by a statistician.** John Tukey coined it at Bell Labs from "binary digit," and Shannon credited him in the 1948 paper. Tukey also coined "software."

- **In 768 dimensions, essentially every pair of random directions is perpendicular.** Two random unit vectors have expected cosine similarity 0 with standard deviation $1/\sqrt{768}\approx 0.036$. High-dimensional space is overwhelmingly empty and overwhelmingly orthogonal — an intuition that fails completely in the two and three dimensions humans can picture.

---

## Check for Understanding

**Every loss you minimize is a KL divergence in disguise, every gradient is a vector–Jacobian product computed in reverse because the loss is scalar, and every estimate you make from finite samples carries an error bar of $\sigma/\sqrt{n}$ that no amount of clever modelling will shrink.**

### Can you explain these out loud?

The real test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is a matrix multiply "a weighted sum of columns," and why does that make an embedding lookup a matrix multiply?**
2. **What is an eigenvector, in terms of stretching rather than equations?** Why does $\lambda^k$ explain both vanishing gradients and power iteration?
3. **Why does $\ell_1$ regularization produce exact zeros when $\ell_2$ doesn't?**
4. **Why is backpropagation reverse-mode?** (Correct answer: because there are many parameters and one loss. It is a fact about shape, not about neural networks.)
5. **Why can you fit millions of near-orthogonal directions into 768 dimensions?**
6. **Why does halving your error bars require four times the data?**
7. **What does KL divergence measure, in terms of codebooks and wasted characters?** Why is it asymmetric, and what does that asymmetry do to your generated samples?
8. **Why is a cross-entropy of 1.52 meaningless until you know $\log K$?**
9. **What is the ELBO doing, in terms of a mountain hidden in cloud?**
10. **Why does cross-entropy have bounded gradients, and why does that matter more than it sounds?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 02 — Learning Theory & Generalization](02-learning-theory-and-generalization.md)
