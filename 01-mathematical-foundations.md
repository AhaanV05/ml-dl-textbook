# Chapter 1 — Mathematical Foundations

> **Prerequisites:** calculus, basic linear algebra.
> **Why this chapter exists:** every derivation later in the book leans on about fifteen facts. Here they are, derived, so you never have to take one on faith.

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

### 1.1.2 Eigendecomposition

For square symmetric $A \in \mathbb{R}^{n\times n}$:

▸ $$A = Q\Lambda Q^\top, \qquad Q^\top Q = I, \qquad \Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$$

The columns of $Q$ are orthonormal eigenvectors. Every symmetric real matrix has this (spectral theorem). **The Hessian of a loss is symmetric**, which is why the whole of optimization theory can be phrased in eigenvalues.

Key consequences you'll use constantly:

- $A^k = Q\Lambda^k Q^\top$. Repeated application amplifies the largest-$|\lambda|$ direction exponentially. *This is exactly why gradients vanish or explode in deep nets* (Ch. 6) and why power iteration works.
- $A$ is **positive semi-definite** (PSD) iff all $\lambda_i \ge 0$ iff $x^\top A x \ge 0\ \forall x$. A local minimum of a loss has PSD Hessian.
- $\mathrm{tr}(A) = \sum_i \lambda_i$, $\det(A) = \prod_i \lambda_i$.

### 1.1.3 SVD, for non-square matrices

▸ $$A = U\Sigma V^\top,\quad U \in \mathbb{R}^{m\times m},\ \Sigma \in \mathbb{R}^{m\times n} \text{ diagonal},\ V \in \mathbb{R}^{n\times n}$$

$\sigma_i \ge 0$ are the singular values, $\sigma_i^2$ are the eigenvalues of $A^\top A$.

**Why you care:** the *best rank-$r$ approximation* of $A$ in Frobenius norm is $A_r = \sum_{k\le r}\sigma_k u_k v_k^\top$ (Eckart–Young theorem). This is:

- PCA (Ch. 15),
- LoRA fine-tuning (a weight update $\Delta W$ constrained to rank $r$),
- the reason "effective rank" of a weight matrix is a meaningful diagnostic.

**Effective rank / stable rank:** $\displaystyle \mathrm{srank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i\sigma_i^2}{\sigma_1^2}$. A $1024\times1024$ matrix can have full mathematical rank but stable rank 12 — meaning it genuinely only uses 12 directions. Worth logging during training.

### 1.1.4 Norms

| Norm | Definition | Where it appears |
|---|---|---|
| $\ell_2$ | $\|x\|_2 = \sqrt{\sum x_i^2}$ | weight decay, gradient clipping |
| $\ell_1$ | $\|x\|_1 = \sum|x_i|$ | sparsity, LASSO, sparse autoencoders |
| $\ell_\infty$ | $\max_i |x_i|$ | adversarial robustness, Adam's implicit bias |
| Frobenius | $\|A\|_F = \sqrt{\sum_{ij}A_{ij}^2}$ | weight norm |
| Spectral | $\|A\|_2 = \sigma_{\max}(A)$ | Lipschitz constants, spectral norm regularization |

▸ The **spectral norm is the Lipschitz constant of the linear map**: $\|Ax\| \le \|A\|_2\|x\|$, with equality for $x = v_1$. A network of $L$ layers has Lipschitz constant at most $\prod_\ell \|W_\ell\|_2 \cdot \prod_\ell \mathrm{Lip}(\sigma_\ell)$. If each $\|W_\ell\|_2 = 1.2$ and $L=50$, the network can amplify an input perturbation by $1.2^{50} \approx 9100\times$. That is the entire theory of adversarial examples in one line.

### 1.1.5 The Johnson–Lindenstrauss lemma (needed for Ch. 20)

▸ For any $\epsilon \in (0,1)$ and $N$ points in $\mathbb{R}^D$, there exists a linear map into $\mathbb{R}^d$ with
$$d = O\!\left(\frac{\log N}{\epsilon^2}\right)$$
that preserves all pairwise distances to within a factor $1\pm\epsilon$.

Read it backwards: **in $d$ dimensions you can place $N = e^{O(\epsilon^2 d)}$ vectors that are all nearly orthogonal to each other.** In $d = 768$ dimensions you can fit *millions* of directions with pairwise cosine similarity under $0.1$. This is the mathematical permission slip for superposition — a network storing far more features than it has neurons is not cheating, it's exploiting high-dimensional geometry.

**Numbers:** random unit vectors in $\mathbb{R}^d$ have expected cosine similarity $0$ and standard deviation $1/\sqrt{d}$. For $d = 768$, that's $0.036$. Two random directions in a transformer's residual stream are, for practical purposes, orthogonal.

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

### 1.2.3 Jacobians and the two modes of autodiff

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is $J_{ij} = \partial f_i/\partial x_j \in \mathbb{R}^{m\times n}$.

**You never build $J$.** For a network with $p = 10^8$ parameters and scalar loss, $J$ would be $1 \times 10^8$ — that one is fine — but intermediate Jacobians are catastrophically large. Instead autodiff computes products:

- **Forward mode** computes the **Jacobian–vector product** $Jv$: cost $\approx$ one forward pass per input direction. Efficient when $n \ll m$.
- **Reverse mode** computes the **vector–Jacobian product** $v^\top J$: cost $\approx$ one forward + one backward pass per *output* direction. Efficient when $m \ll n$.

▸ Deep learning has $m = 1$ (scalar loss) and $n = p$ (millions of parameters). So reverse mode wins by a factor of $p$. **That's the whole reason backpropagation is reverse-mode.** It is not a deep insight about neural networks; it's a statement about the shape of the problem.

**Cost:** reverse-mode AD gives $\nabla_\theta\mathcal{L}$ at roughly $2$–$3\times$ the cost of a forward pass, *independent of $p$*. This is the Baur–Strassen result and it is genuinely remarkable.

### 1.2.4 The Hessian and what "sharpness" means

$H_{ij} = \partial^2\mathcal{L}/\partial\theta_i\partial\theta_j$, symmetric, $p \times p$. You cannot store it ($p=10^8 \Rightarrow 10^{16}$ entries $= 40$ petabytes in fp32). You *can* compute Hessian–vector products cheaply:

▸ $$Hv = \nabla_\theta\big(\langle \nabla_\theta\mathcal{L},\ v\rangle\big)$$

i.e. differentiate the dot product of the gradient with a constant vector. Cost: one extra backward pass. This is the **Pearlmutter trick**, and it's how people actually measure $\lambda_{\max}(H)$ (power iteration on $Hv$) to study sharpness, flat minima, and Edge of Stability (Ch. 5, 19).

**Gauss–Newton / Fisher decomposition.** For a loss $\mathcal{L} = \ell(f_\theta(x), y)$,

$$H = \underbrace{J_f^\top \nabla^2_f \ell\, J_f}_{\text{Gauss–Newton, PSD}} + \underbrace{\sum_k (\nabla_f \ell)_k \nabla^2_\theta f_k}_{\text{curvature of the model}}$$

Near a good fit the second term is small (residuals $\to 0$), so $H \approx$ GGN, which is PSD. This is why practical second-order methods (K-FAC, Shampoo, natural gradient) approximate the GGN/Fisher rather than the true Hessian — the true Hessian has negative eigenvalues and inverting it would push you *uphill*.

---

## 1.3 Probability

### 1.3.1 The three-line core

$$\mathbb{E}[X] = \int x\,p(x)\,dx,\qquad \mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2,\qquad \mathrm{Cov}(X,Y) = \mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$$

**Linearity of expectation holds always** (even for dependent variables): $\mathbb{E}[aX+bY] = a\mathbb{E}[X]+b\mathbb{E}[Y]$.
**Variance is not linear**: $\mathrm{Var}(aX+bY) = a^2\mathrm{Var}(X)+b^2\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)$.

▸ For $n$ i.i.d. variables, $\mathrm{Var}\!\left(\frac{1}{n}\sum X_i\right) = \frac{\sigma^2}{n}$, so the **standard error is $\sigma/\sqrt{n}$.** To halve your error bars you need $4\times$ the data. This single fact governs Chapter 3 entirely, and it is why a 16-batch validation estimate is noisy.

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

### 1.4.3 Mutual information

$$I(X;Y) = \mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big) = H(X) - H(X\mid Y)$$

"How many nats of uncertainty about $X$ does knowing $Y$ remove." Symmetric. $\ge 0$.

Appears in: InfoNCE contrastive learning (Ch. 15, where the loss is a lower bound on $I$), the information bottleneck, and disentanglement objectives ($\beta$-VAE).

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

---

## 1.5 A worked example tying it together

**Setup.** Your model outputs logits $z \in \mathbb{R}^{K}$ over $K$ atom types for each position. True type is $y$. Loss is CE.

1. Forward: $p = \mathrm{softmax}(z)$, $\mathcal{L} = -\log p_y$.
2. Gradient w.r.t. logits: $\bar z = p - e_y$ (§1.3.4). Note $\|\bar z\|\le\sqrt2$ always — **cross-entropy has bounded logit gradients**, which is why it's stable and why MSE on logits isn't.
3. If $z = Wh + b$ with hidden state $h$: $\bar W = \bar z h^\top$, $\bar h = W^\top\bar z$, $\bar b = \bar z$ (§1.2.2).
4. If the model is perfect, $p = e_y$, so $\bar z = 0$ and training stops. If the model is uniform, $p_y = 1/K$, $\mathcal{L} = \log K$, and $\bar z$ has magnitude $\approx 1$.
5. In expectation over the data, $\mathbb{E}[\mathcal{L}] = H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}}\|p_\theta)$. The floor $H(p_{\text{data}})$ is the **irreducible entropy of the data** — for molecules with genuine chemical ambiguity at a given noise level, this is not small.

▸ **Practical implication:** if $H(p_{\text{data}} \mid \text{noise level } t) \approx 1.4$ nats, then a measured CE of $1.524$ means the model is only $0.12$ nats from optimal — and further gains are necessarily tiny and hard to measure. Estimating the irreducible floor (e.g. by fitting a much larger model, or by measuring CE at $t\to 0$) tells you whether the remaining headroom actually exists.

---

## Check for Understanding

**Every loss you minimize is a KL divergence in disguise, every gradient is a vector–Jacobian product computed in reverse because the loss is scalar, and every estimate you make from finite samples carries an error bar of $\sigma/\sqrt{n}$ that no amount of clever modelling will shrink.**

---

**Next:** [Chapter 02 — Learning Theory & Generalization](02-learning-theory-and-generalization.md)
