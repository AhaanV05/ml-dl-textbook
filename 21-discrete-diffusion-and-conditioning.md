# Chapter 21 — Discrete Diffusion & Conditional Generation

> **Prerequisites:** Ch. 20, Ch. 11.
> **Scope:** two topics that belong together because discrete diffusion is almost always used conditionally, and because the DiT architecture is the standard backbone for both.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\odot$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

This chapter is Chapter 20 with the Gaussians swapped out for matrices. **Every structural idea carries over unchanged** — a fixed forward corruption, a closed form for jumping to any noise level, an exact posterior, a clean-data parameterization, a variational bound made of KL terms — and the only thing that changes is what "add noise" means when the data is a token rather than a pixel. If you can hold the shape of Chapter 20 in your head, this chapter is mostly translation.

### Symbols introduced in this chapter

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $K$ | "K" | The **vocabulary size** — how many distinct tokens exist |
| $x_t$ (here) | "x-t" | A token at corruption level $t$, written as a **one-hot row vector** of length $K$ |
| $Q_t$ | "Q-t" | The **transition matrix** for one step. $K\times K$, rows sum to 1 |
| $[Q_t]_{ij}$ | "Q-t, i-j" | Probability that token $i$ becomes token $j$ in one step |
| $\bar Q_t$ | "Q-bar-t" | $Q_1Q_2\cdots Q_t$ — the $t$-step transition matrix. The discrete $\bar\alpha_t$ |
| $\mathrm{Cat}(x;\ p)$ | "categorical" | A weighted die: outcome $j$ with probability $p_j$ |
| $x_{t-1}Q_t$ | "x times Q" | Row-vector times matrix — **looks up the row of $Q_t$ for the current token** |
| $\odot$ | "elementwise" / "Hadamard" | Multiply matching entries, keep them separate (Ch. 0 §0.8) |
| $\mathbf{1}$ | "the all-ones vector" | A column of $K$ ones |
| $\mathbf{1}\mathbf{1}^\top$ | "one one-transpose" | The $K\times K$ matrix of all ones (an outer product) |
| $e_{[\text{MASK}]}$ | "e-mask" | The one-hot vector selecting the `[MASK]` token |
| $\tilde p_\theta(x_0\mid x_t)$ | "p-tilde-theta" | The network's guess at **the original clean token** |
| $\lambda$ | "lambda" | Here: the weight on the auxiliary loss. Not an eigenvalue |
| $\gamma(c),\ \beta(c)$ | "gamma of c", "beta of c" | Learned **scale** and **shift**, computed from the condition $c$ |
| $\alpha(c)$ | "alpha of c" | The AdaLN-Zero **gate** — starts at exactly 0 |
| $c$ | "c" | The **conditioning vector** — timestep, class, or text, mixed into one vector |
| $\ell$ | "logits" | Pre-softmax scores. Here: **not** a loss and **not** a layer index |
| $w$ | "the guidance scale" | Same dial as Ch. 20 §20.8, now applied to logits |
| $p$ (in DiT) | "patch size" | Side length of the square patch turned into one token |
| $\varnothing$ | "null" | "No condition given" |

### Abbreviations used in this chapter, spelled out

| Short | Full form |
|---|---|
| D3PM | Discrete Denoising Diffusion Probabilistic Models |
| SEDD | Score Entropy Discrete Diffusion |
| MDLM | Masked Diffusion Language Model |
| MD4 | Masked Discrete Diffusion (a 2024 model family) |
| DiT | Diffusion Transformer |
| MM-DiT | Multi-Modal Diffusion Transformer |
| FiLM | Feature-wise Linear Modulation |
| AdaGN / AdaLN | Adaptive Group Normalization / Adaptive Layer Normalization |
| LN | Layer Normalization |
| BERT | Bidirectional Encoder Representations from Transformers |
| CFG | Classifier-Free Guidance |
| ELBO / VLB | Evidence Lower BOund / Variational Lower Bound |
| KL | Kullback–Leibler (divergence) |
| CE | Cross-Entropy |
| EMA | Exponential Moving Average |
| MLP / FFN | Multi-Layer Perceptron / Feed-Forward Network |
| KV | Key–Value (cache) |
| RoPE | Rotary Position Embedding |
| LoRA | Low-Rank Adaptation |
| SD3 | Stable Diffusion 3 |
| logP / QED | octanol–water partition coefficient / Quantitative Estimate of Drug-likeness (molecular property scores) |

---

## Part A — Discrete diffusion

## 21.1 Why continuous diffusion doesn't transfer

Text, molecular graphs, and code are **categorical**. "Add Gaussian noise to a token ID" is meaningless — token 47 is not between 46 and 48.

Two responses:
1. **Embed and diffuse in continuous space** (Diffusion-LM, Bit Diffusion). Works, but the rounding step at the end is lossy and the model spends capacity modelling embedding geometry that doesn't matter.
2. **Define diffusion directly on the categorical simplex.** This is D3PM, and it is the cleaner answer.

#### Why "token 47 is not between 46 and 48" is the whole problem

Chapter 20's forward process was $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon$. Every operation in that line — multiply by a fraction, add a small number — assumes the values live on a **number line**, where "a little bit more" and "halfway between" are meaningful.

Pixel values pass that test. A pixel of brightness 128 really is halfway between 127 and 129, and nudging it by 0.4 produces a valid, slightly different pixel.

**Token IDs fail it completely.** In a typical vocabulary, token 4711 might be `" cat"`, 4712 `" catalog"`, and 4713 `" catch"`. Adding 0.4 to `" cat"` gives 4711.4, which is not a word — it is not even a *thing*. The integer is a **name**, not a quantity. It is the same category error as computing the average of two telephone numbers.

▸ **Categorical data has no metric, no ordering, and no notion of "small perturbation."** Every design decision in this chapter follows from that one sentence.

> **Analogy — smudging a photograph versus smudging a sentence.** You can smudge a photograph a little and get a slightly blurred photograph. There is no operation that smudges the word "cat" a little. The nearest thing you can do is **replace** it — with a random word, or with a blank. That is exactly the choice this chapter makes, and the two options become the two transition kernels of §21.2.

**What "the categorical simplex" means.** The **simplex** is the set of all valid probability distributions over $K$ outcomes — every vector of $K$ non-negative numbers that sums to 1. For $K=3$ it is a triangle; for $K=50{,}000$ it is a $49{,}999$-dimensional object you should not try to visualize. A one-hot token sits at a *corner* of this shape; a fully corrupted token sits at the centre (uniform) or at a different corner (mask). **Diffusion on the simplex means moving between distributions rather than between numbers**, and matrix multiplication is exactly the operation that moves you around a simplex while keeping you on it.

**Why response 1 is unsatisfying, spelled out.** Embed each token as a vector, diffuse the vectors continuously, then round to the nearest embedding at the end. Two costs:

- **The rounding is lossy and unprincipled.** The final continuous vector lands somewhere between embeddings; snapping it to the nearest one is a decision the model never got to make probabilistically, and small geometric errors become wrong words.
- **The model wastes capacity on irrelevant geometry.** Half the embedding space's directions do not correspond to any token at all. The network has to learn the shape of a cloud of $K$ points in $d$ dimensions — information that has nothing to do with language and that the discrete formulation never needs to represent.

#### Examples and non-examples:  categorical data

The whole chapter turns on one test: **does "halfway between" mean anything?**

**✅  categorical**

| Example | Why it qualifies |
|---|---|
| Tokens in a 50,000-word vocabulary | Halfway between `" cat"` (4711) and `" catch"` (4713) is `" catalog"` (4712), which has nothing to do with either. The index is a **name** |
| Atom types in a molecule (C, N, O, S, …) | There is no element halfway between carbon and nitrogen that you could put in a bond |
| Amino acids in a protein sequence | 20 discrete labels. Averaging leucine and lysine gives nothing |
| Which of 8 layout templates a page uses | Template 3.5 does not exist |
| Programming-language keywords | `whilf` is not between `while` and `if` |

**❌ Near-misses — look categorical, and are secretly continuous or ordinal**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Pixel intensities discretized to 0–255 | 128 really is between 127 and 129, and a nudge of $0.4$ gives a valid pixel | **Ordinal** with a metric — exactly the case for the band kernel $Q_{ij}\propto\exp(-\lvert i-j\rvert^2/\sigma^2)$ |
| Star ratings 1–5 | 3 is  between 2 and 4. An error of one star is smaller than an error of three | Ordinal — treating it as unordered throws away real structure |
| Quantized audio samples | The quantization is an artefact of storage; the underlying signal is a waveform | Continuous data, discretized for convenience |
| Bucketed ages (0–9, 10–19, …) | The buckets sit on a line and the line is the point | Binned continuous data |
| A one-hot vector of $K$ classes, softened to $(0.7, 0.2, 0.1)$ | This lives *inside* the simplex, not at a corner. It is a distribution over categories, not a category | A probability vector — the thing the model **outputs**, not the thing it models |

▸ **The boundary:** data is categorical when the labels carry **no metric and no order** — when the only meaningful question you can ask about two values is "same or different?" The moment "closer" means something, you have ordinal or continuous data, and a corruption process that respects that closeness (the band kernel, or plain Gaussian diffusion) will beat one that does not.

> **Common misconception.** *"Tokens have embeddings, and embeddings are continuous vectors, so continuous diffusion works fine on text."* Embeddings are continuous, but only $K$ of the infinitely many points in that space are **valid tokens** — the rest are nothing at all. Continuous diffusion happily produces a vector 0.3 of the way between `" cat"` and `" democracy"`, and there is no token there; you must round, and the rounding is a decision the model never made probabilistically. The belief is tempting because Diffusion-LM and Bit Diffusion do work, and work reasonably well. **They work despite the geometry, not because of it** — and the capacity spent learning the shape of a $K$-point cloud in $d$ dimensions is capacity not spent on language.

> **Common misconception.** *"The simplex is where the tokens live."* Tokens live at the **corners** of the simplex — the $K$ one-hot vectors. The interior is where *distributions over tokens* live. This matters because it locates the two objects the chapter juggles: $x_t$ is always a corner (a real token, or `[MASK]`), while $x_{t-1}Q_t$ and $\tilde p_\theta(x_0\mid x_t)$ are interior points you sample from. **Confusing the two is the source of most sign and shape errors when implementing D3PM.**

---

## 21.2 D3PM

### The one-line idea

Replace "add Gaussian noise" with "randomly resample tokens according to a transition matrix," and everything from Chapter 20 goes through with matrix multiplication where the Gaussians were.

### The analogy

A game of telephone with a specific corruption rule. At each round, every word independently has some chance of being replaced — either by a random word (uniform kernel) or by the word "[BLANK]" (absorbing kernel). After enough rounds the message is unrecognizable. The model learns to undo one round of the game.

### The forward process

Let $x_t \in \{1,\dots,K\}$ be a token, represented as a one-hot row vector. The forward step is a categorical distribution defined by a transition matrix $Q_t \in \mathbb{R}^{K\times K}$ with $[Q_t]_{ij} = q(x_t=j\mid x_{t-1}=i)$, rows summing to 1:

▸ $$q(x_t\mid x_{t-1}) = \mathrm{Cat}\big(x_t;\ p = x_{t-1}Q_t\big)$$

#### Unpacking the transition matrix

Everything here rests on one representational choice: **write a token as a one-hot row vector.**

If $K=5$ and the token is "cat" (index 3), then $x = (0,0,1,0,0)$. That looks wasteful — five numbers to store one integer — but it converts "which token is it?" into linear algebra, and linear algebra is what makes the closed form exist.

**Now $Q_t$.** It is a $K\times K$ grid where

$$[Q_t]_{ij} = q(x_t = j \mid x_{t-1} = i) = \text{"probability that token } i \text{ turns into token } j \text{ this step."}$$

**Rows sum to 1** because from any starting token *something* must happen: it either stays or becomes one of the others, and those possibilities exhaust the options. (A matrix with non-negative entries and rows summing to 1 is called **row-stochastic**; this is the standard object of Markov chain theory.)

**Why $x_{t-1}Q_t$ and not $Q_t x_{t-1}$.** Multiplying a *row* vector on the left of a matrix selects a weighted combination of the matrix's **rows** (Ch. 1 §1.1.1's second reading). And since $x_{t-1}$ is one-hot, the "weighted combination" is just **row $i$, extracted whole**:

$$\underbrace{(0,0,1,0,0)}_{\text{token 3}}\ \begin{pmatrix} \cdot&\cdot&\cdot&\cdot&\cdot \\ \cdot&\cdot&\cdot&\cdot&\cdot \\ 0.02&0.02&0.90&0.03&0.03 \\ \cdot&\cdot&\cdot&\cdot&\cdot \\ \cdot&\cdot&\cdot&\cdot&\cdot\end{pmatrix} \;=\; (0.02,\ 0.02,\ 0.90,\ 0.03,\ 0.03)$$

▸ **The matrix multiply is a table lookup wearing a costume.** In code nobody forms $Q_t$; you index. But writing it as a matrix product is what lets you *compose* steps by multiplying matrices, which is the whole point.

**$\mathrm{Cat}(x_t;\ p)$, decoded.** "Categorical distribution" — a weighted die with $K$ faces. Reading the row above: *"with probability 0.90 the token stays 'cat'; with probability 0.02 it becomes token 1; with probability 0.03 it becomes token 4"*, and so on. Sample from it and you have taken one forward step.

**The correspondence with Chapter 20, line by line:**

| Chapter 20 (continuous) | Chapter 21 (discrete) |
|---|---|
| $x_t$ is a real vector | $x_t$ is a one-hot row vector |
| Add Gaussian noise | Resample from a categorical distribution |
| Shrink by $\sqrt{1-\beta_t}$ | Multiply by $Q_t$ |
| $\bar\alpha_t = \prod \alpha_s$ (a scalar) | $\bar Q_t = \prod Q_s$ (a matrix) |
| Endpoint $\mathcal{N}(0,I)$ | Endpoint = the chain's stationary distribution |
| KL between Gaussians | KL between categoricals |

▸ **Scalars became matrices and products stayed products.** That is the entire translation, and once you see it the rest of Part A is bookkeeping.

#### Examples and non-examples: a valid transition matrix

Take $K=3$ throughout.

**✅  transition matrices (row-stochastic)**

| Matrix | Why it qualifies |
|---|---|
| $\begin{pmatrix}0.8&0.1&0.1\\0.1&0.8&0.1\\0.1&0.1&0.8\end{pmatrix}$ | Non-negative, every row sums to 1. "Mostly stay, sometimes swap" |
| $\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}$ | The identity. A perfectly valid chain that does nothing — $\beta_t = 0$ |
| $\begin{pmatrix}0.9&0&0.1\\0&0.9&0.1\\0&0&1\end{pmatrix}$ | The absorbing kernel with state 3 as `[MASK]`. Row 3 is $(0,0,1)$: once there, never leave |
| $\begin{pmatrix}0&1&0\\0&0&1\\1&0&0\end{pmatrix}$ | A deterministic cycle. Every row sums to 1; determinism is allowed |

**❌ Near-misses — look like transition matrices, and break the chain**

| Looks like it | Why it fails | What it actually is |
|---|---|---|
| $\begin{pmatrix}0.8&0.1&0.1\\0.2&0.7&0.2\\0.1&0.1&0.8\end{pmatrix}$ | Row 2 sums to $1.1$. From token 2, the "probabilities" of what happens next total more than certainty | A non-normalized array — the single most common D3PM implementation bug |
| $\begin{pmatrix}0.9&0.2&-0.1\\ \cdot&\cdot&\cdot\\ \cdot&\cdot&\cdot\end{pmatrix}$ | A negative entry. Row sums to 1, and there is no such thing as a $-10\%$ chance | Not a probability distribution at all |
| A matrix whose **columns** sum to 1 | Column-stochastic. Every formula in this chapter uses $x_{t-1}Q_t$ with $x$ a **row** vector; column-stochastic requires $Qx$ with $x$ a column | The transpose convention — correct mathematics, wrong side of the multiply |
| A symmetric similarity matrix from an embedding model | Symmetric and non-negative, but rows sum to arbitrary numbers | A kernel matrix, one `row_normalize` away from being usable |
| $\bar Q_t$ built as $Q_1 + Q_2 + \dots + Q_t$ | Composing steps is **multiplication**, not addition. Adding gives row sums of $t$ | Nothing meaningful |

▸ **The boundary:** a matrix defines a one-step corruption if and only if it is **non-negative with rows summing to exactly 1** — because "from token $i$, something must happen, and each possibility has a real probability." Every product of row-stochastic matrices is row-stochastic, which is precisely what makes $\bar Q_t = Q_1\cdots Q_t$ a legitimate object rather than an accident.

> **Common misconception.** *"The forward process needs a neural network too."* It has **no parameters at all.** $Q_t$ is chosen by you, in advance, from a schedule — exactly as $\beta_t$ was in Chapter 20. All the learning is in the reverse direction. The belief is tempting because in a variational autoencoder the encoder *is* learned, and diffusion is presented as a hierarchical VAE. **The defining move of diffusion, in both chapters, is freezing the encoder** — which is what makes the posterior available in closed form and the training objective decomposable.

> **Common misconception.** *"So a $50{,}000 \times 50{,}000$ matrix is multiplied at every step."* Never. $\bar Q_t$ for a 50k vocabulary is $2.5\times10^9$ entries **per timestep**, and nobody has ever materialized one. Both practical kernels are $(1-\bar\beta_t) I$ plus a rank-one term, so the entire object is described by one scalar per timestep and applied by indexing. **The matrix is a notation for thinking; a scalar and an `if` statement are the implementation.** The confusion is worth resolving early, because it makes people believe discrete diffusion is intractable at realistic vocabulary sizes when it is arguably cheaper than the continuous version.

> **Where this came from.** The mathematics of a state jumping between discrete possibilities with fixed probabilities is **Markov chain theory**, introduced by **Andrey Markov** in 1906. His motivation was a public argument, not an application: the mathematician Pavel Nekrasov had claimed that the law of large numbers required independent events, and had drawn theological conclusions from it about free will. Markov set out to construct **dependent** sequences that obey the law of large numbers anyway — and succeeded, inventing the chain in the process. In 1913 he demonstrated it on the first 20,000 letters of Pushkin's *Eugene Onegin*, counting by hand how often a vowel followed a consonant. **The first application of Markov chains was to the statistics of a novel in verse**, and it is a reasonable claim that the first statistical language model was built with pencil and paper in 1913.

### The closed form

Because the process is a Markov chain, the $t$-step transition is just the matrix product:

▸ $$\bar Q_t = Q_1Q_2\cdots Q_t,\qquad q(x_t\mid x_0) = \mathrm{Cat}\big(x_t;\ p = x_0\bar Q_t\big)$$

**This is the exact analogue of $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$** and serves the same essential purpose: one-shot corruption to any level, so training doesn't require simulating the chain.

#### Why "just multiply the matrices" is legitimate

$$\bar Q_t = Q_1Q_2\cdots Q_t$$

Read the bar the same way as in Chapter 20: **cumulative.** $\bar Q_t$ answers *"starting from token $i$, what is the probability I am at token $j$ after $t$ steps?"*

**Why it works.** Composing two probabilistic steps means summing over every intermediate possibility:

$$\Pr(i \to k \text{ in two steps}) = \sum_{j} \Pr(i\to j)\Pr(j\to k) = [Q_1Q_2]_{ik}$$

▸ **That sum over intermediate states is exactly the definition of matrix multiplication** (Ch. 0 §0.8). Matrix multiplication *is* "chain two random transitions together and account for every path." This is not an analogy or a convenience — it is why matrices were invented for this problem, and it is why Markov chains are studied with linear algebra.

**Work it small.** $K=2$ (tokens A and B), and each step flips with probability 0.1:

$$Q = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.9\end{pmatrix},\qquad Q^2 = \begin{pmatrix} 0.82 & 0.18 \\ 0.18 & 0.82\end{pmatrix},\qquad Q^{10} \approx \begin{pmatrix} 0.55 & 0.45 \\ 0.45 & 0.55\end{pmatrix}$$

After ten steps you have a 55% chance of still being A, barely better than a coin flip. After fifty steps the matrix is essentially $\begin{pmatrix}0.5&0.5\\0.5&0.5\end{pmatrix}$ — **all memory of the starting token is gone.** That limiting matrix is the **stationary distribution**, and it is the discrete counterpart of $\mathcal{N}(0,I)$: a distribution you can sample from without knowing anything about the data.

**Why it converges, and how fast.** $Q^t$ is a matrix raised to a power, so its behaviour is governed by eigenvalues (Ch. 1 §1.1.2). Every row-stochastic matrix has $\lambda_1 = 1$ (the stationary direction); the **second-largest** eigenvalue controls the rate. Here $\lambda_2 = 0.8$, and $0.8^{10}\approx 0.107$ — matching the $0.55 = 0.5 + 0.107/2$ we computed. ▸ **"How many steps until the text is destroyed?" is the question "how fast does $\lambda_2^t$ decay?", which is the identical mathematics as Chapter 20's $\bar\alpha_t = \prod\alpha_s$ and as vanishing gradients.** Three chapters, one exponential.

**The practical payoff is identical to Chapter 20's.** Precompute $\bar Q_1,\dots,\bar Q_T$ once, at startup. Then corrupting a training example to level 843 is a single lookup and a single categorical sample — no simulation of 843 steps, and every example in a batch can sit at a different $t$ with no coordination. **A billion training examples become a billion independent one-shot corruptions.**

**One honest caveat about memory.** $\bar Q_t$ is $K\times K$, and for a 50,000-token vocabulary that is $2.5\times10^9$ entries **per timestep** — completely impractical to store. This is why the kernels of the next section are chosen to have *structure*: for the uniform and absorbing kernels, $\bar Q_t$ has a closed form in terms of a single scalar, and you never materialize the matrix at all. ▸ **The matrix formulation is how you think about it; a scalar is how you compute it.**

### The exact posterior

By Bayes, with $\odot$ elementwise:

▸ $$q(x_{t-1}\mid x_t,x_0) = \mathrm{Cat}\!\left(x_{t-1};\ p = \frac{\big(x_tQ_t^\top\big)\odot\big(x_0\bar Q_{t-1}\big)}{x_0\bar Q_t x_t^\top}\right)$$

The denominator $x_0\bar Q_tx_t^\top$ is the scalar $q(x_t\mid x_0)$ — the normalizer. Everything is a $K$-vector operation, so the KL terms in the ELBO are **exact, closed-form categorical KLs.** No approximation anywhere.

#### Reading the exact posterior

$$q(x_{t-1}\mid x_t,x_0) = \mathrm{Cat}\!\left(x_{t-1};\ p = \frac{\big(x_tQ_t^\top\big)\odot\big(x_0\bar Q_{t-1}\big)}{x_0\bar Q_t x_t^\top}\right)$$

It is Bayes' rule, and it has exactly three moving parts.

| Piece | Shape | What it asks |
|---|---|---|
| $x_tQ_t^\top$ | $K$-vector | **"For each candidate previous token, how likely was it to produce what I see now?"** The likelihood |
| $x_0\bar Q_{t-1}$ | $K$-vector | **"For each candidate previous token, how likely was it to arise from $x_0$ in $t-1$ steps?"** The prior |
| $\odot$ | — | Multiply the two, entry by entry (Ch. 0 §0.8) |
| denominator | scalar | Divide so the $K$ numbers sum to 1 |

▸ **Posterior $\propto$ likelihood $\times$ prior, computed for all $K$ candidates at once by an elementwise product of two $K$-vectors.** Compare §20.3, where the same Bayes step required completing the square in a Gaussian exponent. **Here it is one line of vector arithmetic** — which is the recurring pleasure of the discrete formulation: everything you had to derive analytically in Chapter 20 you can now simply *enumerate*, because there are only $K$ possibilities.

**Why the transpose appears.** $Q_t$ answers "from $i$, where do I go?" But here we ask the reverse: "given that I arrived at $j$, where might I have come from?" Transposing reverses every arrow (Ch. 1 §1.2.2's rule 2, and for the identical reason). $x_tQ_t^\top$ picks out **column** $j$ of $Q_t$, which is the list of all the ways to *arrive* at $j$.

**Work it small.** $K=3$, absorbing kernel, $\beta_t = 0.1$, and suppose $x_t$ is `[MASK]` (index 3) while $x_0$ is token 1.

- **Likelihood** $x_tQ_t^\top$: which tokens could have become `[MASK]`? Token 1 with probability 0.1, token 2 with probability 0.1, and `[MASK]` itself with probability 1.0 → $(0.1,\ 0.1,\ 1.0)$.
- **Prior** $x_0\bar Q_{t-1}$: what could token 1 have become in $t-1$ steps? Say it is still token 1 with probability 0.4, or masked with probability 0.6 → $(0.4,\ 0,\ 0.6)$. (It can never become token 2 — the absorbing kernel only maps to `[MASK]`.)
- **Product**: $(0.04,\ 0,\ 0.60)$. **Normalize**: $(0.0625,\ 0,\ 0.9375)$.

Read the answer: *"given that this position is masked now and the original was token 1, there is a 6.25% chance it was still token 1 one step ago and a 93.75% chance it was already masked."* **Sensible, exact, and computed with six multiplications.**

▸ **"No approximation anywhere" is a stronger statement than it looks.** In Chapter 20 the Gaussian assumption on the reverse process is an *approximation* that only becomes exact in the limit of infinitely many steps — it is why $T$ has to be large. In the discrete case the posterior is a categorical distribution and the model's reverse step is a categorical distribution, so **the model family contains the truth exactly, at any $T$.** Discrete diffusion has no small-step requirement of that kind, which is part of why it can generate in 10–50 steps rather than 1000.

### The transition kernels

**Uniform / multinomial:**
▸ $$Q_t = (1-\beta_t)I + \frac{\beta_t}{K}\mathbf{1}\mathbf{1}^\top$$
Each token stays with probability $1-\beta_t+\beta_t/K$, or jumps to a uniformly random token. Stationary distribution: uniform. **Analogue of the Gaussian kernel.**

**Absorbing / masking:**
▸ $$Q_t = (1-\beta_t)I + \beta_t\,\mathbf{1}\,e_{[\text{MASK}]}^\top$$
Each token either stays or is permanently replaced by `[MASK]`. Stationary distribution: all mask.

▸ **The absorbing kernel is the one that won.** Reasons: (i) the model always knows *which* positions are corrupted, so it never wastes capacity deciding whether a token is real; (ii) it connects directly to BERT-style masked modelling, so the objective is familiar and well-conditioned; (iii) the posterior simplifies dramatically — an unmasked token stays unmasked with probability 1, so only masked positions need prediction.

**Structured kernels:** if the state space has metric structure (discretized pixel intensities, ordinal categories), use a **discretized Gaussian band** $Q_{ij}\propto\exp(-|i-j|^2/\sigma^2)$ so corruption respects locality. For molecules, kernels can be built from a **token similarity matrix** (e.g. chemically similar atom types more likely to interchange), or from the data's marginal distribution ($Q_t = (1-\beta_t)I+\beta_t\mathbf{1}\tilde p^\top$ with $\tilde p$ the empirical unigram), which makes the noise process match the data's own statistics.

#### The two kernels, decoded

Both formulas have the same shape — **"mostly stay put, occasionally do something else"** — and differ only in what the something-else is.

**Uniform kernel.** $Q_t = (1-\beta_t)I + \frac{\beta_t}{K}\mathbf{1}\mathbf{1}^\top$

- $I$ — the identity matrix: "stay exactly where you are."
- $\mathbf{1}\mathbf{1}^\top$ — the outer product of the all-ones column with the all-ones row, giving **a $K\times K$ matrix every entry of which is 1** (Ch. 0 §0.8). Divided by $K$, its rows are the uniform distribution.
- So: **with probability $1-\beta_t$ do nothing; with probability $\beta_t$ pick a token uniformly at random.**

Write it out for $K=3$, $\beta_t = 0.3$:

$$Q_t = 0.7\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix} + 0.1\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix} = \begin{pmatrix}0.8&0.1&0.1\\0.1&0.8&0.1\\0.1&0.1&0.8\end{pmatrix}$$

Rows sum to 1. ✓ The diagonal is $1-\beta_t+\beta_t/K = 0.7+0.1 = 0.8$ — slightly more than $1-\beta_t$, because "pick uniformly at random" sometimes picks the token you already had.

**Absorbing kernel.** $Q_t = (1-\beta_t)I + \beta_t\,\mathbf{1}\,e_{[\text{MASK}]}^\top$

- $\mathbf{1}\,e_{[\text{MASK}]}^\top$ — an outer product giving a matrix that is **all zeros except one column of ones**, the `[MASK]` column. Every row says "go to `[MASK]`."
- So: **with probability $1-\beta_t$ stay; with probability $\beta_t$ become `[MASK]` — permanently.**

"**Absorbing**" is standard Markov-chain vocabulary for a state you can enter but never leave: $[Q_t]_{\text{MASK},\text{MASK}} = 1$. Once masked, always masked.

> **Analogy for the two kernels.** Uniform corruption is a **photocopier with a fault** that occasionally substitutes a random glyph — you cannot tell corrupted characters from real ones, and you must decide, for every character, whether to trust it. Absorbing corruption is a **redaction marker** — the ink is unmistakable, and you know exactly which words need reconstructing and exactly which are safe. The redaction is strictly more informative, which is why it wins.

▸ **Reason (i) in the text deserves the emphasis it gets.** With the uniform kernel the network must jointly solve two problems: *is this token corrupted?* and *if so, what was it?* The first is a detection problem it cannot solve reliably — a plausible-looking word might be original or might be a lucky random substitution. With the absorbing kernel the detection problem is **free**, because corruption is self-announcing. **Deleting a subproblem entirely is a bigger win than solving it well.**

**Reason (iii), spelled out.** With absorbing noise, the exact posterior collapses to something almost trivial: an unmasked position stays unmasked with probability 1 (nothing can un-mask it going backwards), so the posterior over unmasked positions is a point mass, and **only the masked positions need any prediction at all.** Both the loss and the sampler simplify accordingly, which is what makes the MDLM result in §21.3 possible.

**The stationary distributions differ interestingly.** Uniform → the endpoint is a uniformly random token sequence; the "prior" you sample from at generation time is `randint(0, K)` at every position. Absorbing → the endpoint is a sequence of *all* `[MASK]`, which is a **single deterministic state.** ▸ **Discrete diffusion with an absorbing kernel starts generation from a blank page, not from noise** — and all the randomness enters through the sampling of the reverse steps rather than through the initial state. That is a real structural difference from Chapter 20 and worth being able to state.

**Structured kernels, in one line each.** The band kernel $Q_{ij}\propto\exp(-|i-j|^2/\sigma^2)$ says *"a corrupted value is likely to be near the true one"* — appropriate when the categories  sit on a line (pixel intensity 128 really is next to 129), which was the case the whole chapter opened by ruling out for text. The unigram kernel $Q_t = (1-\beta_t)I+\beta_t\mathbf{1}\tilde p^\top$ replaces "uniformly random token" with "a token drawn from the corpus's own frequency distribution," so corruption produces `the` far more often than `zygote` — **matching the noise to the data's marginal statistics, exactly as the discretized-Gaussian schedule matches noise to image statistics.**

#### Examples and non-examples: choosing the corruption kernel

**✅ Well-matched kernel choices**

| Data | Kernel | Why it fits |
|---|---|---|
| Natural-language tokens | **Absorbing / masking** | No metric between tokens; corruption should be self-announcing so the model never has to detect it |
| Molecular graphs (atom and bond types) | **Absorbing**, usually | Atom types are unordered labels; and the user often wants to fix a scaffold and fill the rest — which is masking by construction |
| Discretized pixel intensities 0–255 | **Band**, $Q_{ij}\propto\exp(-\lvert i-j\rvert^2/\sigma^2)$ | The categories  sit on a line. Corrupting 128 to 131 is a small error; to 7, a large one |
| Text where you want the noise to look like text | **Unigram**, $Q_t=(1-\beta_t)I+\beta_t\mathbf{1}\tilde p^\top$ | Corruption produces `the` far more often than `zygote`, so the noise distribution matches the data's marginal |

**❌ Near-misses — kernel choices that look reasonable and cost you**

| Looks like it | Why it fails | What it actually is |
|---|---|---|
| Uniform kernel on a 50k vocabulary, because it is "the analogue of Gaussian noise" | The model must now solve a **detection** problem it cannot solve: a plausible word might be original or a lucky random substitution. Capacity goes into "is this real?" instead of "what was it?" | The aesthetically parallel choice, not the informative one |
| Band kernel on token IDs | Token 4711 and 4712 are `" cat"` and `" catalog"`. The kernel encodes a locality that does not exist | Imposing a false metric — actively worse than uniform |
| Absorbing kernel on discretized pixel values | Throws away the one thing pixels have that tokens don't: order. The model must reconstruct from nothing rather than from "near 128" | A kernel that discards usable structure |
| Adding `[MASK]` to the vocabulary but letting it be re-emitted | If $[Q_t]_{\text{MASK},\text{MASK}} < 1$, the state is not absorbing, the posterior no longer collapses, and the "only masked positions need prediction" simplification is gone | A **leaky** absorbing kernel — a bug that silently doubles the work |
| A kernel where some token has zero probability of ever being reached | The chain never mixes to its stationary distribution, so $q(x_T\mid x_0)$ still depends on $x_0$ and the $L_T$ term never vanishes | A reducible chain — the discrete version of "the noise schedule doesn't reach pure noise" |

▸ **The boundary:** the right kernel is the one whose corruption **destroys exactly the information the model must learn to restore, and announces itself while doing so.** Absorbing wins on text because masking is maximally self-announcing; band wins on ordinal data because it destroys precision while preserving magnitude; uniform wins almost nowhere, and survives in the literature mainly because it is the most obvious translation of Gaussian noise.

> **Common misconception.** *"The absorbing kernel is a simplification, so it must be less expressive."* It is not a restriction of the model family — the reverse model still assigns a full distribution over all $K$ tokens at every masked position. What masking removes is a *subproblem the model would otherwise have to solve badly*: deciding which positions are corrupted. **Deleting a subproblem entirely is a bigger win than solving it well**, and the belief is tempting because "less to do" usually means "less capable." Here it means the same capacity is aimed at one question instead of two.

> **Common misconception.** *"`[MASK]` is just another word in the vocabulary."* It is a state with $[Q]_{\text{MASK},\text{MASK}}=1$ — enter it and you never leave, which is what "absorbing" means in Markov-chain vocabulary. Concretely: it expands the state space to $K+1$, it never appears in real data, the model must never *predict* it, and it is the entire terminal distribution of the forward chain. **Treating it as an ordinary token is how implementations end up with a model that fluently generates `[MASK]` at inference time.**

> **Where this came from.** **D3PM** — *Structured Denoising Diffusion Models in Discrete State-Spaces* — is by **Jacob Austin, Daniel Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg** at Google, 2021. Note that Jonathan Ho is a co-author of DDPM as well; D3PM is very deliberately the discrete rewrite of his own continuous paper, kernel by kernel. A closely related formulation, *multinomial diffusion*, was published independently at nearly the same time by **Emiel Hoogeboom and co-authors**. The absorbing kernel's dominance was not obvious at the time — D3PM presents it as one of several options, alongside uniform, band, and nearest-neighbour kernels. **It took another two to three years of empirical work for the field to conclude that masking was not merely one option but the right one.**

### The loss

The ELBO is the same structure as Chapter 20:

$$L_{\text{vb}} = \mathbb{E}_q\Big[\underbrace{\mathrm{KL}(q(x_T|x_0)\|p(x_T))}_{L_T} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Big]$$

**The parameterization that makes it work:** rather than predicting $p_\theta(x_{t-1}\mid x_t)$ directly, the network predicts a distribution over the **clean data** $\tilde p_\theta(x_0\mid x_t)$, and the reverse step is obtained by plugging that into the known posterior:

▸ $$p_\theta(x_{t-1}\mid x_t) \ \propto\ \sum_{\tilde x_0} q(x_{t-1}\mid x_t,\tilde x_0)\,\tilde p_\theta(\tilde x_0\mid x_t)$$

This is exactly the discrete analogue of $\epsilon$-prediction: **let the network solve the easy, well-posed problem (what was the original token?) and let the known posterior handle the rest.** It also means sampling can skip steps trivially.

#### The parameterization, decoded — why predict $x_0$ and not the reverse step

$$p_\theta(x_{t-1}\mid x_t) \ \propto\ \sum_{\tilde x_0} q(x_{t-1}\mid x_t,\tilde x_0)\,\tilde p_\theta(\tilde x_0\mid x_t)$$

Read the sum as a weighted vote:

> **"For every token the original *might* have been, ask the known posterior what the previous step would look like in that case, and average those answers weighted by how likely the network thinks each candidate is."**

- $\tilde p_\theta(\tilde x_0\mid x_t)$ — the network's output: **a softmax over the vocabulary**, exactly what any language model produces. It answers "what was the original token here?"
- $q(x_{t-1}\mid x_t,\tilde x_0)$ — the exact posterior from two sections ago, which requires no learning at all.
- $\sum_{\tilde x_0}$ — a loop over all $K$ candidates. Because the posterior is available in closed form, this whole sum is a matrix–vector product; **it does not cost $K$ network evaluations, it costs one.**

▸ **The division of labour is the point.** The network is given the one job it is good at — *"look at this partially destroyed sentence and guess the missing word"* — and every piece of probabilistic bookkeeping is handled by formulas we derived. Compare Chapter 20, where the network predicts $\epsilon$ and the closed-form relation to $\tilde\mu_t$ does the rest. **Same architecture of responsibility, different currency.**

> **Analogy — the restorer and the archivist.** A painting restorer looks at a damaged canvas and says "this patch was probably ultramarine." An archivist, who knows exactly how the damage process works, then computes what the canvas must have looked like one stage earlier, given that guess. Neither could do the other's job. **The network is the restorer; the exact posterior is the archivist.**

**Why "sampling can skip steps trivially" follows.** The reverse step is assembled from $\tilde p_\theta(x_0\mid x_t)$ and a posterior that is available for *any* pair of indices, not just adjacent ones. So $q(x_{t-10}\mid x_t, \tilde x_0)$ is just as computable as $q(x_{t-1}\mid x_t,\tilde x_0)$ — substitute $\bar Q_{t-10}$ for $\bar Q_{t-1}$ and every formula still holds. **This is the same argument that made DDIM's step-skipping legal in §20.7, arriving here for free rather than as a discovery.**

**What it looks like at sampling time with the absorbing kernel.** Start from all `[MASK]`. At each step the network predicts a distribution over the true token at every masked position; you then **unmask some fraction of positions** by sampling from those predictions, and leave the rest masked for later steps. Generation is progressive un-redaction, and the number of steps is simply how many rounds of unmasking you choose to do. ▸ **Every position is decoded in parallel within a step, which is exactly the "parallel generation" advantage of §21.3 — and the independence *within* a step is exactly its weakness.**

> **Common misconception.** *"If the network predicts $x_0$ directly, why run $T$ steps at all? Just take its answer."* You can, and the result is bad in a specific and instructive way. $\tilde p_\theta(x_0\mid x_t)$ is a distribution over each position **independently**. Committing to all of them at once samples every position from its own marginal and ignores every correlation between them — which is how you get *"New Delhi"* when the model's joint distribution strongly preferred *"New York"*. **The iterative procedure is not there to refine the network's guess; it is there to introduce the correlations that a per-position softmax structurally cannot express.** Each step unmasks a few positions, and the *next* step's prediction is conditioned on what was just committed. This is the whole reason step count trades against coherence, and it is the same reason a single-step diffusion sampler blurs images: one shot at a factorized distribution can only ever give you the product of the marginals.

> **Common misconception.** *"Predicting $x_0$ and predicting $x_{t-1}$ are two parameterizations of the same thing, so it's a matter of taste."* They describe the same family, and they are wildly different to *learn*. "What was the original token?" has a fixed, well-posed answer with a gradient of order 1 at every noise level. "What is the distribution one step back?" has an answer that at large $t$ is *almost independent of the data* — the posterior says "still masked" no matter what $x_0$ was — so the target carries almost no signal exactly where you most need the network to be learning something. **Reparameterizing does not change what is representable; it changes what is easy to descend**, and Chapter 20's $\epsilon$-prediction is the identical manoeuvre in the identical place.

### The auxiliary $x_0$ loss

D3PM's full objective adds a direct cross-entropy term on the clean-data prediction:

▸ $$\boxed{\ \mathcal{L} = \mathcal{L}_{\text{vb}} + \lambda\ \mathbb{E}_{x_0,t,x_t}\big[-\log\tilde p_\theta(x_0\mid x_t)\big]\ }$$

typically $\lambda = 0.001$–$0.01$.

▸ **Why it's there:** the VLB's per-timestep KL terms are small and noisy at large $t$ (where the posterior is nearly the stationary distribution regardless of $x_0$), giving a weak training signal. The direct cross-entropy on $x_0$ is a dense, well-scaled auxiliary target at every timestep. It plays the same role as $\mathcal{L}_{\text{simple}}$'s reweighting in continuous diffusion — trading likelihood optimality for sample quality and trainability.

**This auxiliary term is the quantity most implementations log as the primary training/validation metric**, since it is interpretable directly as a token-level cross-entropy (compare to $\log K$; see Ch. 1 §1.4.2). It is what Case Study A's `val_realCE` measures.

▸ **The evaluation subtlety that follows** (and the reason Case Study A appears throughout Part I): this metric is a **conditional expectation over $t$**, and its value varies enormously across $t$ — near $\log K$ at large $t$, near zero at small $t$. Sampling $t$ freshly per batch therefore injects a between-$t$ variance term into the estimate that divides only by the *number of batches*, not the number of examples (Ch. 3 §3.6). **Always report per-$t$-bucket cross-entropy alongside the aggregate**, and use a fixed stratified $t$ grid for validation. Aggregate CE can be flat while every bucket improves, purely from a shift in which $t$ values happened to be drawn.

#### The auxiliary loss, decoded

$$\mathcal{L} = \mathcal{L}_{\text{vb}} + \lambda\ \mathbb{E}_{x_0,t,x_t}\big[-\log\tilde p_\theta(x_0\mid x_t)\big]$$

- $\mathcal{L}_{\text{vb}}$ — the principled variational bound: a sum of KL terms.
- $-\log\tilde p_\theta(x_0\mid x_t)$ — **plain cross-entropy.** "How surprised was the network by the correct token?" Assign the right token 90% probability and pay $-\log 0.9 = 0.105$ nats; assign it 1% and pay $4.6$ nats (Ch. 1 §1.4.2).
- $\lambda \approx 0.001$–$0.01$ — a small weight, because the auxiliary term is a *guide*, not the objective.

▸ **The whole equation says: "optimize the principled thing, plus a small nudge from the practical thing."** And this is the same manoeuvre as Chapter 20's $\mathcal{L}_{\text{simple}}$ — knowingly deviate from the exact bound to get a better-conditioned training signal — arrived at by a different route. **Both chapters discovered that the exact variational objective is a bad thing to descend.**

**Why the VLB's signal goes weak at large $t$, concretely.** At $t=900$ with an absorbing kernel, nearly every position is `[MASK]`, and the exact posterior $q(x_{t-1}\mid x_t, x_0)$ says "almost certainly still masked" **regardless of what $x_0$ was.** The KL between that and the model's version is therefore tiny and barely depends on whether the model understood anything. The gradient is real but small, and it is swamped by the noise of a single sampled $t$.

The auxiliary cross-entropy has no such problem: **at every $t$, including $t=900$, "what was the original token?" is a well-posed question with a definite answer and a gradient of order 1.** It is a dense target where the VLB is sparse.

> **Analogy.** Grading a student only on the final exam gives you an accurate but noisy signal, especially early in the term when nothing they do changes the grade much. Adding weekly quizzes worth 1% each does not change what "doing well" means, but it gives them — and you — a usable readout at every point in the process. **$\lambda$ is how much the quizzes count.**

**Why this is the number everyone logs.** Cross-entropy in nats is directly interpretable, and it has a hard reference point: **a model that has learned nothing scores $\log K$.** With $K = 50{,}000$ that is $10.8$ nats; with a small molecular vocabulary of $K=40$ it is $3.7$. ▸ **A cross-entropy of 1.52 is meaningless until you know $\log K$** (Ch. 1 §1.4.2) — and it is doubly meaningless in diffusion until you know **which $t$ it was measured at**, since the same model scores near $\log K$ at $t=950$ and near zero at $t=5$.

#### Why per-$t$-bucket reporting is not optional

Work through the failure the text warns about, with numbers.

Suppose the metric behaves like this, and you sample $t$ freshly per validation batch:

| $t$ bucket | Cross-entropy | Fraction of the range |
|---|---|---|
| $1$–$200$ | $0.4$ | Nearly clean; easy |
| $201$–$500$ | $2.1$ | Real work |
| $501$–$800$ | $5.6$ | Hard |
| $801$–$1000$ | $9.9$ | Near $\log K = 10.8$ |

The aggregate is about $4.5$. Now note the **spread**: the standard deviation *across buckets* is roughly $3.7$ nats — far larger than any improvement you expect from a day of training. If a validation run happens to draw more large-$t$ examples than the previous one, the aggregate rises even though every bucket fell.

▸ **The between-$t$ variance divides by the number of *batches*, not the number of *examples*** (Ch. 3 §3.6). Ten thousand validation examples spread over 40 batches gives you $\sqrt{40}$ of averaging on the $t$-variation, not $\sqrt{10{,}000}$. **Fixing a stratified $t$ grid removes that term entirely and is completely free** — you are not estimating over $t$ any more, you are evaluating at chosen points. It is the single highest-return line of code in a discrete-diffusion project, and it routinely cuts metric noise by an order of magnitude.

**The practical rule, stated plainly:** report a *vector* of cross-entropies, one per $t$ bucket, and compare vectors. An aggregate that moves is uninformative; a vector where every entry moves the same way is conclusive.

---

## 21.3 The modern discrete-diffusion landscape

**SEDD (Score Entropy Discrete Diffusion).** Generalizes the score to discrete spaces as the **concrete score** — the ratio $\frac{p(y)}{p(x)}$ between states — and trains it with a denoising score entropy loss. Achieves perplexities competitive with autoregressive transformers of the same size, which was the first time discrete diffusion was  competitive on language.

**MDLM / MD4 (Masked Diffusion Language Models).** Show that with the absorbing kernel, a well-chosen parameterization, and careful attention to the continuous-time limit, the objective simplifies to a **weighted average of masked-language-modelling losses**:
$$\mathcal{L} \propto \mathbb{E}_{t}\left[\frac{1}{t}\,\mathbb{E}\big[-\log p_\theta(x_0^{\text{masked}}\mid x_t)\big]\right]$$
▸ **Discrete diffusion with an absorbing kernel is BERT with a random, continuously-varying mask rate and a principled weighting.** That is the cleanest possible statement of what these models are, and it explains both why they train stably and why they inherit BERT-like inductive biases.

**Why anyone wants this over autoregressive modelling:**
- **Parallel generation.** All positions decode simultaneously; a full sequence in 10–50 steps rather than $T$.
- **Bidirectional context** — every position sees every other at every step.
- **Controllable infilling** — conditioning on arbitrary known positions is trivial (just don't mask them), whereas autoregressive infilling requires special training.
- Natural fit for **non-sequential data**: molecular graphs, protein sequences with structural constraints, layouts.

**Why autoregressive still wins on general language:** likelihood is still somewhat worse at matched compute, KV caching has no clean analogue (each step recomputes everything), and the parallel-decoding independence assumption within a step hurts coherence.

#### The modern landscape, decoded

**SEDD and the "concrete score."** Chapter 20's score is $\nabla_x\log p(x)$ — a derivative, which requires $x$ to live somewhere you can take derivatives. Tokens don't. SEDD's move is to notice **what the score is actually *for*:** it tells you the relative preference between a point and its neighbours. In a discrete space you can ask that question directly, without calculus:

$$\text{concrete score}(x \to y) = \frac{p(y)}{p(x)}$$

Read: *"if I changed this token from $x$ to $y$, how many times more (or less) likely would the sequence become?"*

▸ **A ratio of probabilities is the discrete stand-in for a derivative of a log-probability** — and note that, exactly as in §20.6, **the normalizing constant cancels in the ratio.** SEDD gets the same "no partition function" benefit that made continuous score matching work, by the same mechanism, with the derivative replaced by a difference quotient. The result mattered because it was the first time discrete diffusion reached perplexities competitive with autoregressive transformers at matched size — before that, the approach was interesting but clearly behind.

**MDLM / MD4, and the sentence that explains everything.**

$$\mathcal{L} \propto \mathbb{E}_{t}\left[\frac{1}{t}\,\mathbb{E}\big[-\log p_\theta(x_0^{\text{masked}}\mid x_t)\big]\right]$$

Read the pieces:

- $-\log p_\theta(x_0^{\text{masked}}\mid x_t)$ — **the BERT objective.** Mask some tokens, predict them, score with cross-entropy. Nothing else.
- $\mathbb{E}_t$ — averaged over mask *rates*, rather than at a single fixed rate.
- $1/t$ — a principled weighting that falls out of the continuous-time limit; it up-weights the low-mask-rate regime.

▸ **"Discrete diffusion with an absorbing kernel is BERT with a random, continuously-varying mask rate and a principled weighting."** BERT masks 15% of tokens, always. Masked diffusion masks $r\%$ where $r$ is drawn uniformly from 0 to 100 — and, crucially, **it derives the weighting rather than choosing it**, which is what turns a representation-learning objective into a *generative* one. BERT could never generate text because it was only ever trained at one mask rate and had no principled way to go from 100% masked to 0%. **Sweep the rate and add the right weights, and the same architecture and the same loss become a generative model.**

> **Analogy.** BERT is a student drilled exclusively on cloze exercises with one word in seven removed. They become superb at that specific task and cannot write an essay from a blank page. Masked diffusion drills the same student at every difficulty — one word in a hundred missing, then half, then all of them. **After that training, "write an essay" is just the hardest cloze exercise**, and the student has a route from blank page to finished text that passes through every intermediate difficulty they have practised.

#### The trade-off with autoregressive models, made concrete

| | Autoregressive (Ch. 13) | Discrete diffusion |
|---|---|---|
| Steps to generate $n$ tokens | $n$ | 10–50, independent of $n$ |
| Work per step | One token, with KV cache | The whole sequence, no cache |
| Context each token sees | Only what came before | Everything, at every step |
| Infilling | Needs special training | Free — don't mask the known parts |
| Revising an earlier mistake | Impossible | Routine |
| Likelihood at matched compute | Better | Somewhat worse |

▸ **The KV-cache asymmetry is the underrated line in that table.** An autoregressive transformer generating token 1000 reuses all the work it did on tokens 1–999; only the new token's attention is computed. A diffusion step has **no such structure** — every position changes at every step, so nothing can be cached, and each of the 20 steps costs a full forward pass over the entire sequence. So "20 steps instead of 1000" is not a 50× win; measured in FLOPs it is much closer, and for short sequences autoregressive generation can be outright cheaper.

**The independence problem, stated precisely.** Within one reverse step, every masked position is sampled **independently** from its own predicted distribution. But the true joint distribution has correlations: if position 7 is going to be `"New"`, then position 8 should be `"York"`. Sampling both independently can produce `"New Delhi"` — each locally plausible, jointly wrong. ▸ **The fix is to unmask fewer positions per step, which restores coherence and costs steps** — and that trade is exactly why the practical step count is 10–50 rather than 2, and why quality degrades sharply if you push it lower.

**Where discrete diffusion  wins.** Not general text — the table above explains why. It wins where the data has **no natural left-to-right order**: molecular graphs (an atom's neighbours are a set, not a sequence), protein design under structural constraints, document layouts, and any setting where the *user* specifies scattered known positions and asks the model to fill the rest. In those cases autoregressive modelling has to invent an ordering, and the invented ordering is a modelling assumption that costs real accuracy.

#### Examples and non-examples: tasks that suit discrete diffusion

**✅  fits**

| Task | Why it fits |
|---|---|
| Generate a molecule containing a fixed benzene scaffold at specified positions | The known atoms are simply never masked. Infilling at arbitrary scattered positions is free |
| Design a protein sequence with three catalytic residues pinned | Same: constraints are unmasked positions, not a prefix |
| Fill in the blanks of a document layout given a header and footer box | The known content is not a prefix and not a suffix — it is scattered |
| Edit a generated sentence by re-masking two words and re-running | Revising an earlier commitment is routine; an autoregressive model would have to regenerate the suffix |
| Any graph where "node 1 then node 2 then node 3" is an arbitrary invention | There is no left-to-right order to impose, so no ordering assumption is paid for |

**❌ Near-misses — sound like discrete-diffusion wins, and aren't**

| Looks like it | Why it fails | What actually wins |
|---|---|---|
| "Generate a 2,000-token chat response faster, because 20 steps beats 2,000" | Each of the 20 steps is a **full forward pass over all 2,000 positions**, with no KV cache. In FLOPs the two are close, and for short outputs autoregression is cheaper | Autoregressive generation with a KV cache |
| "Fill in the middle of a code file" — assumed impossible for autoregressive models | Fill-in-the-middle training (reordering the prefix/suffix/middle at training time) gives autoregressive models this capability directly | An autoregressive model trained with FIM |
| "Better likelihood, because it sees bidirectional context" | Bidirectional context helps *representation*; likelihood at matched compute is still somewhat worse | Autoregression, on this metric |
| "Unmask everything in one step for a 2,000× speedup" | Positions are sampled independently within a step. One step gives you the product of the marginals, not the joint | 10–50 steps — coherence is bought with steps |
| "Non-autoregressive, so no exposure bias" | It has its own version: the model is trained on corruptions of *real* data and tested on its own partially-generated output | A real caveat for both families |

▸ **The boundary:** discrete diffusion wins where the data has **no canonical order** and where the user's constraints arrive as **scattered known positions** rather than as a prefix. Where the data is  sequential and the constraint is  a prefix, autoregression's KV cache and exact factorization are very hard to beat.

> **Common misconception.** *"Masked diffusion is just BERT, so BERT could already generate text."* The chapter's own line — masked diffusion *is* BERT with a random mask rate and a principled weighting — makes the difference easy to miss. BERT was trained at **one** mask rate, 15%, forever. It has no idea what to do with a 100%-masked input, and no principled route from "everything masked" to "nothing masked": the weighting that makes the sequence of mask rates a valid variational bound is exactly what BERT lacks. **Sweeping the rate and deriving the weights is what converts a representation-learning objective into a generative model**, and both halves are load-bearing — sweeping the rate without the weighting gives you a model that trains but does not correspond to any likelihood.

> **Common misconception.** *"Fewer steps means faster."* Step count and cost are different quantities. Autoregressive step $n$ costs one token's worth of attention because everything before it is cached; diffusion step $n$ costs a full pass over every position because everything changed. **"20 steps instead of 1,000" is a 50× reduction in steps and something far smaller — sometimes nothing, sometimes negative — in FLOPs.** The belief is tempting because step count is the number printed in papers and the number you set in a config; wall-clock is the number you actually pay.

---

## Part B — Conditional generation

## 21.4 The conditioning mechanisms

### The one-line idea

There are four ways to tell a network "generate *this* kind of thing," and they differ in how much interaction they allow between the condition and the content.

| Mechanism | Form | Best for |
|---|---|---|
| **Concatenation** | append $c$ to the input | spatially aligned conditions (segmentation maps, depth, inpainting masks) |
| **Cross-attention** | $Q$ from $x$, $K,V$ from $c$ | variable-length, token-structured conditions (text) |
| **FiLM / AdaGN** | $\gamma(c)\odot h + \beta(c)$ | global scalar or vector conditions (class, timestep) |
| **AdaLN-Zero** | FiLM on LayerNorm, gated, zero-init | **the DiT standard** |

### FiLM

▸ $$\mathrm{FiLM}(h\mid c) = \gamma(c)\odot h + \beta(c)$$

A cheap, powerful mechanism: a global condition modulates the *scale and shift* of every feature. It is how timestep information enters almost every diffusion model.

#### The four conditioning mechanisms, decoded

The table's organizing question is: **how much interaction do you need between the condition and the content?** Answer that and the choice is made for you.

**1. Concatenation.** Stack the condition onto the input as extra channels. If you are conditioning a $64\times64\times4$ latent on a $64\times64\times1$ depth map, you feed the network $64\times64\times5$. ▸ **Only sensible when the condition is *spatially aligned* with the output** — pixel $(i,j)$ of the condition corresponds to pixel $(i,j)$ of the result. Segmentation maps, depth maps, edge maps, inpainting masks: yes. A sentence: no, because a sentence has no pixel coordinates.

**2. Cross-attention.** $Q$ from the image, $K$ and $V$ from the condition (Ch. 11). Read it as: *"each image patch asks a question, and the text tokens answer."* This is the only mechanism on the list that handles a **variable-length, internally-structured** condition, and it lets different image regions attend to different words — which is exactly what "a red cube on a blue sphere" requires. Cost: an extra attention block per layer, with the sequence length of the condition.

**3. FiLM / AdaGN.** Squash the whole condition into one vector and use it to scale and shift every feature. ▸ **A global broadcast**: it can say "make everything more cat-like" but not "make the top-left corner more cat-like." Extremely cheap — one small MLP per block — and exactly right for a condition that  *is* global, which the timestep $t$ always is.

**4. AdaLN-Zero.** FiLM applied to the LayerNorm parameters, plus a gate, plus zero initialization. The DiT standard, and the subject of the next two sections.

| Mechanism | Cost per block | Condition can vary across positions? |
|---|---|---|
| Concatenation | Nearly zero | Yes, but only if spatially aligned |
| Cross-attention | An attention layer | Yes |
| FiLM / AdaGN | One small MLP | No — global only |
| AdaLN-Zero | One small MLP (6 outputs) | No — global only |

▸ **In practice, real systems use several at once.** Stable Diffusion conditions on the timestep with FiLM-style modulation *and* on text with cross-attention, because $t$ is a global scalar and a prompt is a sequence. Choosing "the" conditioning mechanism is usually the wrong framing; **choosing one per condition, matched to that condition's structure, is the right one.**

#### Examples and non-examples: a condition FiLM can carry

FiLM broadcasts one scale and one shift per feature to **every position**. That is the constraint, and it decides everything.

**✅ Conditions FiLM handles well**

| Condition | Why it fits |
|---|---|
| The diffusion timestep $t$ |  one number, and  global. Every position is at the same noise level |
| A class label from 1,000 ImageNet classes | One embedding vector, applying uniformly to the whole image |
| A style label ("Van Gogh", "Hokusai") | The original conditional-instance-norm result: the entire difference between two styles fits in a per-layer gain and bias |
| A guidance scale, or a scalar property target like "logP = 3.2" | One number, no spatial extent |
| A CLIP-pooled text embedding, when you only need coarse semantics | Deliberately squashed to one vector, and behaves accordingly |

**❌ Near-misses — conditions people push through FiLM, and shouldn't**

| Looks like it | Why it fails | What it actually needs |
|---|---|---|
| A full text prompt: *"a red cube to the left of a blue sphere"* | FiLM can say "make everything more cube-like." It cannot say "make the **left** region cube-like." All spatial and compositional structure is destroyed by pooling to one vector | **Cross-attention** — each patch queries the token sequence |
| A segmentation map | It is a full-resolution spatial signal; averaging it to one vector discards the only information it carries | **Concatenation** — it is pixel-aligned with the output |
| A reference image for style *and* layout | The layout half is spatial | Concatenation (layout) plus FiLM or cross-attention (style) |
| A variable-length list of constraints | FiLM's MLP takes a fixed-size input | Cross-attention, which is length-agnostic by construction |
| A depth map, pooled to its mean depth | The mean depth of a scene is one number and almost no information | Concatenation of the map itself |

▸ **The boundary:** FiLM can carry any condition that is **the same everywhere in the output.** The moment the condition needs to say something *different* about different positions, it must enter through a mechanism with positional structure — concatenation if it is pixel-aligned, cross-attention if it is a sequence.

> **Common misconception.** *"Cross-attention is the most powerful mechanism, so use it for everything."* It is the most powerful *and* it is the wrong tool for a scalar. Conditioning the timestep through cross-attention means building an attention layer over a sequence of length one, per block, to communicate a single number that every position needs identically — an attention layer's worth of parameters and FLOPs to do what $2d$ numbers already do. Worse, DiT's ablations found AdaLN-Zero beating cross-attention conditioning at every model size. **Expressive power you don't need is not free; it is parameters that must be trained, and an easy path the network can learn to ignore.**

> **Common misconception.** *"FiLM is a form of attention, since it decides what matters."* It has no attention in it — no queries, no keys, no softmax, no interaction between positions, and no data-dependence on $h$ at all. $\gamma$ and $\beta$ are computed from $c$ **alone**; the same scale is applied to feature 5 whether feature 5 is large or small, at every position. The confusion is tempting because both mechanisms are described as "the model deciding what to focus on." **Attention is content-dependent routing between positions; FiLM is a content-independent gain applied everywhere** — and that severe restriction is exactly why it costs $2d$ parameters rather than $4d^2$.

#### FiLM, decoded

$$\mathrm{FiLM}(h\mid c) = \gamma(c)\odot h + \beta(c)$$

| Piece | Read aloud | Meaning |
|---|---|---|
| $h$ | "h" | The hidden activations — a vector of features at this layer |
| $c$ | "c" | The condition, as a single vector |
| $\gamma(c)$ | "gamma of c" | A learned **scale**, one number per feature. Output of a small MLP fed $c$ |
| $\beta(c)$ | "beta of c" | A learned **shift**, one number per feature. Same MLP |
| $\odot$ | "elementwise" | Multiply feature $i$ by $\gamma_i$, keep them separate (Ch. 0 §0.8) |

▸ **Read it as: "the condition gets to turn each feature's volume knob up or down, and nudge its baseline."** It cannot mix features together, it cannot move information between positions, and it cannot add new content. **That severe restriction is why it is cheap and why it works** — the network already computes useful features; the condition only has to decide which ones matter right now.

**Work it small.** Let $h = (2.0,\ -1.0,\ 0.5)$ be three features, and suppose the condition "$t=900$, very noisy" produces $\gamma = (0.1,\ 1.8,\ 1.0)$ and $\beta = (0,\ 0.3,\ 0)$:

$$\mathrm{FiLM}(h) = (0.1\times2.0,\ \ 1.8\times(-1.0)+0.3,\ \ 1.0\times0.5) = (0.2,\ -1.5,\ 0.5)$$

**Feature 1 has been muted to a tenth; feature 2 has been amplified and biased.** If feature 1 detects fine texture and feature 2 detects coarse layout, the network has just learned "at high noise, ignore texture and attend to layout." That is precisely the timestep-dependent behaviour a diffusion model needs, expressed in six numbers.

> **Analogy — a graphic equalizer.** The music (the features) is fixed; the equalizer's sliders (the $\gamma$'s) decide which frequency bands you hear. Change the room and you change the sliders, not the recording. **The condition sets the sliders.**

**Why $\gamma$ and $\beta$ specifically, and not something richer?** Because a full learned linear map from $c$ to a transformation of $h$ would need $d^2$ parameters per block, and $d$ is typically 1024 or larger. FiLM needs $2d$. ▸ **It is the cheapest conditioning that is still *multiplicative***, and multiplicative is what matters: an additive-only condition can be ignored by a downstream layer, whereas scaling a feature to zero  removes it.

> **Where this came from.** **FiLM** — *Feature-wise Linear Modulation* — is by **Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville**, published at AAAI 2018. The problem they were solving was **visual question answering**: given an image and a question like *"is there a cube to the left of the red sphere?"*, condition the vision network on the question. The idea's direct ancestor is stranger still — **conditional instance normalization**, from Dumoulin, Shlens & Kudlur's 2017 work on artistic style transfer, where they discovered that a *single* convolutional network could paint in dozens of different artists' styles **with all weights shared, changing only the per-style gain and bias of the normalization layers.** That result is the surprising one: the entire difference between "in the style of Van Gogh" and "in the style of Hokusai" fits in two vectors per layer. **Diffusion models inherited timestep conditioning from style transfer**, and the mechanism is unchanged.

### AdaLN and AdaLN-Zero

Adaptive LayerNorm makes the norm's gain and bias functions of the condition:
$$\mathrm{AdaLN}(h\mid c) = \gamma(c)\odot\frac{h-\mu}{\sigma}+\beta(c)$$

**AdaLN-Zero** adds a third output — a per-block gate $\alpha(c)$ applied to the residual branch — and **initializes the MLP producing $\alpha$ to output zero**:

▸ $$x \leftarrow x + \alpha(c)\odot\mathrm{Attn}\big(\mathrm{AdaLN}(x\mid c)\big)$$

with $\alpha=0$ at initialization, so **every block starts as the exact identity.**

▸ **Why this matters so much:** it is the same zero-init-residual principle as Fixup, ReZero, and `zero_init_residual` in ResNets (Ch. 6 §6.4, Ch. 8 §8.2). The network begins as a shallow, perfectly-conditioned function and grows depth as training proceeds. In the DiT ablations, AdaLN-Zero beat in-context conditioning, cross-attention, and plain AdaLN **by a wide margin at every model size** — it was the single most important architectural finding in that paper. If you are debugging early instability in a conditional diffusion transformer, verify this zero-init is intact; it is easy to break during refactoring.

#### AdaLN, decoded

$$\mathrm{AdaLN}(h\mid c) = \gamma(c)\odot\frac{h-\mu}{\sigma}+\beta(c)$$

Start from ordinary LayerNorm (Ch. 7), which is

$$\mathrm{LN}(h) = \gamma\odot\frac{h-\mu}{\sigma}+\beta$$

where $\mu$ and $\sigma$ are **the mean and standard deviation of $h$ itself, computed across the feature dimension** — not learned, just measured, fresh for every input. The fraction $\frac{h-\mu}{\sigma}$ standardizes the activations to mean 0 and variance 1. Then $\gamma$ and $\beta$ are learned parameters that put the scale and shift back, under the network's control.

▸ **The only change in AdaLN is that $\gamma$ and $\beta$ stop being fixed parameters and become *functions of the condition*.** Instead of one gain per layer for all inputs, you get a different gain depending on what you're generating and at what noise level. **It is FiLM applied at exactly the point in the block where the scale is already being set anyway** — which is why it costs almost nothing: the multiply and add were already in the computation graph.

#### AdaLN-Zero, decoded — and why zero-initialization is one of the best tricks in deep learning

$$x \leftarrow x + \alpha(c)\odot\mathrm{Attn}\big(\mathrm{AdaLN}(x\mid c)\big)$$

- $\leftarrow$ — an **assignment**, a line of code, not an equation to solve (Ch. 0 §0.11).
- $\alpha(c)$ — a third output of the modulation MLP: **a per-block gate**, one number per feature.
- The MLP producing $\alpha$ has its final layer's weights **and bias initialized to exactly zero**, so $\alpha(c) = 0$ for every $c$ at step 0 of training.

Substitute $\alpha = 0$:

$$x \leftarrow x + 0\odot\mathrm{Attn}(\dots) = x$$

▸ **Every block is the identity function at initialization.** A 28-layer DiT starts life as a 0-layer network: whatever goes in comes out. Training then *grows* the depth, as each block's gate lifts off zero only to the extent that it earns its keep.

> **Analogy — a mixing desk with every fader down.** A recording studio with 28 channels all set to silence produces exactly the dry input signal. The engineer brings up one fader at a time and hears what each contributes. **Compare starting with all 28 faders at random positions:** you get an unusable wall of noise and you cannot tell which channel is causing what. Zero-init is starting the session with the faders down.

**Why this fixes the deepest problem in deep networks.** A randomly-initialized deep residual stack has a signal that grows with depth — each block adds a random perturbation, and $L$ independent perturbations accumulate as $\sqrt{L}$. Worse, the gradient must traverse all $L$ blocks, and each one multiplies it by something not quite 1, which compounds (Ch. 1 §1.1.2's $\lambda^k$, again). **With every gate at zero, the gradient path from the loss to the input is a clean straight line through the residual connections, with no multiplicative distortion at all.** The network is perfectly conditioned on the first step, and only becomes ill-conditioned to the extent that it has learned something worth the ill-conditioning.

**Put a number on it.** With $L=28$ blocks each applying a random transformation of gain $1.1$, the output at initialization is scaled by $1.1^{28}\approx 15$. With gates at zero, it is scaled by exactly $1$. That factor of 15 is the difference between a warmup schedule that must be babysat and one that just works.

▸ **"Verify this zero-init is intact" is  practical advice.** The failure mode is silent: the model still trains, just worse and less stably, and there is no error message. It is broken by innocuous refactors — a weight-init sweep that re-initializes all `nn.Linear` layers, a checkpoint-loading path that rebuilds modules, a `reset_parameters()` call inherited from a base class. **The check is one line: assert the modulation MLP's final weight and bias are all zero before the first optimizer step.**

> **Where this came from.** Zero-initialized residual branches have been rediscovered repeatedly under different names. Goyal and co-authors used a zero-initialized final BatchNorm gain per residual block in the 2017 "ImageNet in 1 hour" paper, as a stability measure at large batch sizes. **Fixup** (Zhang, Dauphin & Ma, 2019) showed you could train very deep residual networks with *no* normalization at all if you initialized the residual branches to zero and scaled carefully — a result whose point was that much of what normalization was credited with was really about initialization. **ReZero** (Bachlechner et al., 2020) reduced it to the minimal form: a single learnable scalar per block, initialized to zero. **AdaLN-Zero is ReZero with the scalar made a function of the condition.** The recurring lesson across all of them: *starting a deep network as a shallow one and letting it grow is worth more than almost any architectural cleverness.*

---

## 21.5 The DiT architecture

**Diffusion Transformer** — replaces the U-Net backbone with a plain transformer.

**The forward path:**
1. **Patchify.** A latent of shape $I\times I\times C$ becomes $T=(I/p)^2$ tokens of dimension $d$ via a linear projection of $p\times p$ patches. $p\in\{2,4,8\}$; smaller $p$ means more tokens, more FLOPs, better quality.
2. **Positional embeddings** — frozen sinusoidal (2-D) or RoPE.
3. **Condition embedding.** Timestep $t$ → sinusoidal → MLP. Class label → embedding table. Summed into a single conditioning vector $c$. (For text, a cross-attention sub-layer is added instead, as in PixArt/SD3.)
4. **$L$ DiT blocks:**
   ```
   shift, scale, gate  = MLP(c).chunk(6)      # 6 outputs: 3 for attn, 3 for ffn
   x = x + gate_1 * Attn(modulate(LN(x), shift_1, scale_1))
   x = x + gate_2 * FFN(modulate(LN(x), shift_2, scale_2))
   ```
   with the modulation MLP zero-initialized (AdaLN-Zero).
5. **Final layer:** AdaLN → linear → unpatchify to predict $\epsilon$ (and optionally $\Sigma$).

### The scaling result

▸ **DiT's central finding: sample quality (FID) improves monotonically and predictably with model FLOPs**, across patch sizes and model widths — the same clean scaling behaviour as language models, which U-Nets did not exhibit. That is why every major image and video model since 2023 (SD3, Flux, Sora, and successors) uses a transformer backbone.

**MM-DiT** (SD3) extends this with **separate weights for text and image token streams** that attend jointly — a two-stream transformer. It substantially improves text rendering and prompt adherence over a single shared stream.

#### The DiT forward path, decoded

**Step 1 — patchify, and the one number that dominates everything.** A latent of shape $I\times I\times C$ is cut into non-overlapping $p\times p$ squares, and each square is flattened and linearly projected to a $d$-dimensional token. The token count is

$$T = (I/p)^2$$

▸ **Note the square.** Halving $p$ **quadruples** the number of tokens, and since attention is $\mathcal{O}(T^2)$ (Ch. 0 §0.10), it multiplies attention cost by **sixteen.** Put numbers on it for a $32\times32$ latent:

| $p$ | $T = (32/p)^2$ | Attention pairs $T^2$ | Relative attention cost |
|---|---|---|---|
| 8 | 16 | 256 | $1\times$ |
| 4 | 64 | 4,096 | $16\times$ |
| 2 | 256 | 65,536 | $256\times$ |

**Patch size is the single most consequential hyperparameter in a DiT**, and the paper's finding is that smaller is reliably better for quality — you are simply paying for it in compute. This is the identical trade as the patch size in a Vision Transformer (Ch. 28), and it has the identical resolution.

**Step 2 — positional embeddings.** Patchifying throws away *where* each patch was; attention is permutation-invariant, so without positional information the model would treat the image as a bag of patches. **Frozen 2-D sinusoidal** embeddings (not learned — there is nothing to learn, and freezing them makes resolution changes easier) or **RoPE** (Ch. 12).

**Step 3 — condition embedding, and why everything gets summed into one vector.** The timestep $t$ goes through sinusoidal features and a small MLP; the class label goes through an embedding table; the two are **added** to form a single conditioning vector $c$. Summing two embeddings is the standard way to combine independent pieces of global information — the network can separate them again because they occupy near-orthogonal directions in high dimensions (Ch. 1 §1.1.5). ▸ **This works only for conditions that are  global. Text is not**, which is exactly why PixArt and SD3 add a cross-attention sub-layer rather than trying to squeeze a prompt into one summed vector.

**Step 4 — the block, line by line.**

```
shift, scale, gate  = MLP(c).chunk(6)      # 6 outputs: 3 for attn, 3 for ffn
x = x + gate_1 * Attn(modulate(LN(x), shift_1, scale_1))
x = x + gate_2 * FFN(modulate(LN(x), shift_2, scale_2))
```

- **`MLP(c).chunk(6)`** — one small MLP produces six vectors from the condition: shift/scale/gate for the attention branch and the same three for the feed-forward branch. With hidden width $d$, that MLP outputs $6d$ numbers per block.
- **`modulate(LN(x), shift, scale)`** — the AdaLN operation: normalize, then apply the condition's scale and shift.
- **`x = x + gate * (...)`** — the residual connection with the zero-initialized gate. At step 0, `gate` is 0 and both lines reduce to `x = x`.

▸ **Compare with a standard transformer block (Ch. 11): `x = x + Attn(LN(x))`. The DiT block is character-for-character the same, with two insertions — a modulation between the norm and the sublayer, and a gate on the residual.** That is the whole architecture. There is no U-Net, no skip connections across resolutions, no downsampling and upsampling path.

**Step 5 — unpatchify.** Reverse the patchify: project each token back to $p\times p\times C$ and reassemble the grid. The output is the predicted $\epsilon$ (and optionally a predicted variance).

#### Reading the scaling result

▸ **"FID improves monotonically and predictably with model FLOPs."** Three words carry the weight:

- **Monotonically** — more compute is never worse. No sweet spot to find, no regime where scaling up hurts.
- **Predictably** — you can *extrapolate*. Train three small models, fit the trend, and forecast what a model ten times larger will achieve before you spend the money.
- **With FLOPs** — not with parameters, and not with depth or width separately. Different combinations of width, depth, and patch size that reach the same FLOP count reach roughly the same FID.

**Why that is a bigger deal than a quality number.** U-Nets improved with scale too, but *unpredictably*: architecture choices interacted with scale in ways that made forecasting unreliable, so scaling up was a gamble. ▸ **A predictable scaling curve converts model development from research into engineering** — it turns "will a bigger model be better?" into "how much better, and is that worth the budget?" This is precisely the property that made large language models a fundable industrial programme (Ch. 15), imported wholesale into image generation. **That, and not the FID number, is why every major image and video model since 2023 uses a transformer backbone.**

> **Where this came from.** **DiT** — *Scalable Diffusion Models with Transformers* — is by **William Peebles and Saining Xie**, first posted in late 2022 and published at ICCV 2023. The paper was turned down on its first conference submission before being accepted, a fact its authors have discussed publicly; the objection was novelty, since "replace the U-Net with a transformer" is not, on its face, a new idea. What the paper actually contributed was the **ablation** — the systematic demonstration that AdaLN-Zero beats every alternative conditioning scheme at every scale, and that FID scales cleanly with FLOPs. **William Peebles subsequently co-led Sora at OpenAI, whose technical report describes a diffusion transformer**, and the architecture in this section is the backbone of essentially every frontier image and video generator. It is a useful case study in what reviewing rewards: the paper's contribution was rigour, and rigour reads as unoriginality.

### Conditioning strength diagnostics

Worth building into any conditional generative project:
- **Condition ablation.** Generate with $c$ and with $\varnothing$ and measure the divergence between output distributions. If they are similar, the model is ignoring the condition.
- **Condition swap.** Generate with a *mismatched* condition; the outputs should change substantially.
- **CFG sensitivity curve.** Sweep $w$ and plot fidelity vs diversity. A flat curve means the conditional and unconditional branches have not differentiated — usually a sign of too-high condition dropout, too-weak conditioning, or a broken AdaLN-Zero.
- **Per-condition validation loss.** Aggregate loss can hide the fact that the model is excellent on common conditions and useless on rare ones.

#### Examples and non-examples: evidence the model is actually using the condition

**✅  evidence**

| Evidence | Why it qualifies |
|---|---|
| Generating with $c$ and with $\varnothing$ produces measurably different output *distributions* | A direct test of whether $c$ enters the computation at all |
| Swapping in a mismatched condition changes the samples substantially | Rules out the model having learned one generic mode it emits regardless |
| The CFG sweep shows fidelity rising and diversity falling as $w$ grows | The conditional and unconditional branches have  differentiated — there is something to extrapolate between |
| Per-condition validation loss is low on rare classes, not just common ones | Distinguishes "uses the condition" from "matched the marginal" |
| Zeroing the modulation MLP's output at inference visibly degrades samples | The most direct possible ablation of the conditioning path |

**❌ Near-misses — look like evidence, and aren't**

| Looks like it | Why it fails | What it actually shows |
|---|---|---|
| The samples look good | A strong unconditional model produces good samples while ignoring $c$ entirely | Sample quality, which is not conditioning |
| Aggregate validation loss went down | Dominated by the common conditions; a model excellent on the top 5% and useless elsewhere scores well | Average performance, hiding the tail |
| The loss is lower with conditioning than without | Could be a small constant gain from the extra parameters rather than from using $c$ | Weak evidence at best — run the ablation |
| The CFG curve is flat as $w$ varies | This is a **failure** signal, not a null result: $\ell_c \approx \ell_\varnothing$ means the branches never differentiated | Broken AdaLN-Zero, too-high condition dropout, or a condition path that was never wired in |
| Two different prompts give two different images | Different random seeds also give different images. Compare distributions, at fixed seed | Nothing, without a controlled comparison |

▸ **The boundary:** conditioning is demonstrated only by a **controlled contrast** — the same model, the same noise, one thing changed, and a measured difference in the output distribution. Any observation that does not vary $c$ while holding everything else fixed cannot distinguish "conditioned" from "good."

---

## 21.6 Guidance in the discrete setting

Classifier-free guidance was derived for continuous scores (Ch. 20 §20.8). In the discrete case it is applied to **logits**:

▸ $$\ell_{\text{guided}} = \ell_\varnothing + w\,(\ell_c - \ell_\varnothing)$$

then softmax. This is extrapolation in logit space, which is equivalent to a *product-of-experts* reweighting of the categorical distribution:
$$p_{\text{guided}}(x)\ \propto\ p_\varnothing(x)\left(\frac{p_c(x)}{p_\varnothing(x)}\right)^{w}$$

Same trade-off as continuous CFG: fidelity up, diversity down, and at large $w$ the distribution collapses onto its mode.

**Practical notes for discrete guidance:**
- Guidance is applied at *every* denoising step, so effects compound; $w$ values are typically smaller than in image diffusion.
- With absorbing kernels, guidance only affects masked positions.
- **Condition dropout rate** (for training the $\varnothing$ branch) is typically 10%. Too low and the unconditional branch is undertrained; too high and the conditional branch suffers.

#### Discrete guidance, decoded

$$\ell_{\text{guided}} = \ell_\varnothing + w\,(\ell_c - \ell_\varnothing)$$

**Note that $\ell$ here means logits**, not a loss and not a layer index — one of the overloads Ch. 0 §0.4 warns about. Logits are the raw pre-softmax scores the network emits, one per vocabulary token.

The formula is **identical in shape to continuous CFG** (§20.8): interpolate from the unconditional prediction toward the conditional one, and keep going past it. Only the currency changed — noise vectors became logit vectors. **Extrapolate, then softmax.**

**Why extrapolating logits is a product of experts.** Softmax turns logits into probabilities via $p_i \propto e^{\ell_i}$. So adding logits multiplies probabilities, and scaling logits raises probabilities to a power:

$$e^{\ell_\varnothing + w(\ell_c-\ell_\varnothing)} \;=\; e^{\ell_\varnothing}\cdot\left(\frac{e^{\ell_c}}{e^{\ell_\varnothing}}\right)^{w} \;\propto\; p_\varnothing(x)\left(\frac{p_c(x)}{p_\varnothing(x)}\right)^{w}$$

▸ **Linear arithmetic in logit space is multiplicative arithmetic in probability space.** That is the single most useful fact about softmax, and it explains the whole formula: the guided distribution is the unconditional one, **multiplied by a "does this match the prompt?" factor raised to the power $w$.** A token that the prompt makes twice as likely gets boosted by $2^w$ — at $w=3$, an eightfold boost.

**A product of experts** (the term is Geoffrey Hinton's, from work on combining probabilistic models in the late 1990s) is a combination rule where **every factor can veto.** Adding distributions gives you a union — anything one expert likes survives. *Multiplying* them gives an intersection — a token needs support from **both** factors to survive. That is why guidance sharpens rather than blurs, and why at large $w$ everything but the single most-favoured token is annihilated.

**Work it small.** $K=3$, with $p_\varnothing = (0.5, 0.3, 0.2)$ and $p_c = (0.6, 0.3, 0.1)$ — the prompt mildly prefers token 1 and disfavours token 3.

| $w$ | Guided distribution | Comment |
|---|---|---|
| 1 | $(0.60,\ 0.30,\ 0.10)$ | Ordinary conditional |
| 3 | $(0.79,\ 0.19,\ 0.02)$ | Sharpened |
| 6 | $(0.92,\ 0.08,\ 0.004)$ | Nearly deterministic |

▸ **A mild preference of $0.6$ vs $0.5$ becomes overwhelming after six-fold extrapolation.** And this happens at every position, at every one of the denoising steps — hence the text's warning that **effects compound and discrete $w$ values are typically smaller than image-diffusion ones.** An image sampler applies CFG to a continuous quantity that is later averaged over a thousand small steps; a discrete sampler applies it to a *decision* that, once a token is unmasked, is final.

**"With absorbing kernels, guidance only affects masked positions."** Because an unmasked token has posterior probability 1 of staying put, there is no distribution left to sharpen there. Guidance operates exclusively on the positions still being decided — which is a small efficiency win and a useful sanity check: **if your guidance implementation changes an already-committed token, it has a bug.**

#### Examples and non-examples: what the guidance scale $w$ is

**✅ True statements about $w$**

| Statement | Why it holds |
|---|---|
| $w=0$ recovers the unconditional model | $\ell_\varnothing + 0\cdot(\ell_c-\ell_\varnothing) = \ell_\varnothing$. The condition is discarded entirely |
| $w=1$ recovers the ordinary conditional model | The bracket telescopes to $\ell_c$. No guidance at all |
| $w>1$ **extrapolates past** the conditional prediction | This is the entire mechanism, and the reason it is called guidance rather than interpolation |
| $w$ raises the likelihood ratio to a power | $p_\varnothing\,(p_c/p_\varnothing)^w$ — the product-of-experts form. At $w=3$, a $2\times$ preference becomes $8\times$ |
| Large $w$ collapses the distribution onto its mode | Each factor can veto; multiply hard enough and only the argmax survives |

**❌ Near-misses — things $w$ is often assumed to be**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "A dial for how well the output matches the prompt, with no downside" | Fidelity and diversity move in opposite directions. High $w$ produces prompt-adherent output that is also the *same* output every time | A trade, not an improvement |
| "At $w=5$ I am sampling from $p(x\mid c)$, just more confidently" | You are sampling from $p_\varnothing (p_c/p_\varnothing)^5$, which is the posterior of **nothing**. It is a deliberately distorted distribution chosen because it looks better | A tempered product of experts |
| "It works the same as in image diffusion, so reuse $w=7.5$" | Image CFG perturbs a continuous quantity that a thousand small steps re-average. Discrete guidance perturbs a **decision**, and once a token is unmasked it is final | Compounding, which is why discrete $w$ is typically much smaller |
| "A temperature" | Temperature scales all logits toward or away from uniform, using no condition. Guidance scales the *difference* between two conditioned predictions | A different operation that happens to also sharpen |
| "It needs no extra compute" | Every step requires **two** forward passes, one with $c$ and one with $\varnothing$ | Roughly a 2× inference cost, before any batching tricks |

▸ **The boundary:** $w$ interpolates along the line from $\ell_\varnothing$ to $\ell_c$ and then keeps going. Everything surprising about guidance — the sharpening, the diversity loss, the mode collapse at large $w$ — follows from that one word, **extrapolation**, plus the fact that softmax turns additive extrapolation into multiplicative reweighting.

> **Common misconception.** *"Condition dropout is a regularizer, like dropout in a hidden layer."* It shares a name and nothing else. Standard dropout randomly zeroes *activations* to prevent co-adaptation. Condition dropout replaces $c$ with the null token $\varnothing$ on roughly 10% of training examples for one purpose: **to train an unconditional branch inside the same network**, so that $\ell_\varnothing$ exists at inference time and CFG has something to extrapolate from. It is not fighting overfitting; it is manufacturing a second model for free. The consequences of getting it wrong are correspondingly specific — too low and $\ell_\varnothing$ is undertrained and noisy, so the extrapolation direction is garbage; too high and you have spent 30% of your training budget on a branch you only use as a reference point.

---

## 21.7 Practical guidance for discrete diffusion projects

1. **Prefer the absorbing/masking kernel** unless there is a specific reason for uniform or structured noise.
2. **Predict $x_0$**, not the reverse transition directly.
3. **Include the auxiliary $x_0$ cross-entropy** with $\lambda\approx0.01$; it is what makes training stable and the metric interpretable.
4. **Log per-$t$-bucket cross-entropy**, always. The aggregate is nearly uninterpretable on its own.
5. **Fix the validation $t$ grid** (stratified) and the validation subset. This is free and typically reduces metric noise by an order of magnitude (Ch. 3 §3.6).
6. **Maintain an EMA of the weights** ($\gamma\approx0.9999$) and evaluate the EMA.
7. **Compare CE to $\log K$**, and estimate the irreducible floor at each $t$ if possible, so you know how much headroom actually exists.
8. **Use AdaLN-Zero** for conditioning, and verify the zero-init survived refactoring.
9. **Evaluate generation, not just likelihood.** For molecules: validity, uniqueness, novelty, distributional property matches (a KL or Wasserstein distance on logP, QED, molecular weight), and scaffold diversity. Cross-entropy can improve while sample quality does not.
10. **Read the slope, not the minimum**, when judging progress (Ch. 3 §3.6).

#### The checklist, grouped by what it protects you from

Ten items is a lot to hold. They fall into four groups, and remembering the groups is easier than remembering the list.

| Group | Items | The failure it prevents |
|---|---|---|
| **Get the model right** | 1, 2, 3, 8 | Choosing a kernel or parameterization that makes training harder than it needs to be |
| **Get the measurement right** | 4, 5, 7, 10 | Believing a metric that is mostly noise, or that has no reference point |
| **Get the weights right** | 6 | Evaluating a checkpoint that bounces with every batch |
| **Measure the right thing** | 9 | Optimizing a proxy while the thing you care about stagnates |

▸ **Items 4, 5, 7, and 10 are all the same instruction in different clothes: never look at a number without knowing its noise floor and its reference point.** Item 7's "compare CE to $\log K$" gives you the ceiling (a model that has learned nothing). Item 5's fixed stratified grid removes the largest source of noise. Item 4's per-bucket reporting stops one $t$ range from masking another. Item 10's "read the slope, not the minimum" is Chapter 3's warning that the *best* value in a noisy series is biased optimistically — the minimum of a noisy curve is systematically below its true level, because you selected it for being low.

**Item 9 deserves the last word, because it is the one people skip.** Cross-entropy is a measure of how well you predict *held-out tokens from the data distribution*. It is not a measure of whether your generated molecules are chemically valid, or novel, or diverse. ▸ **A model can improve its cross-entropy by getting better at the common cases while its generated samples get more repetitive** — the loss rewards coverage of what's typical, and typicality and diversity are not the same thing. For molecules the standard battery is **validity** (does it parse as a molecule at all), **uniqueness** (are the samples distinct from each other), **novelty** (are they distinct from the training set), **property distribution match** (a KL or Wasserstein distance on logP, QED, and molecular weight), and **scaffold diversity** (are the core structures varied, or is everything a decoration of one skeleton). All five can move independently of the loss.

---

## Did you know?

- **Markov invented Markov chains to win a theological argument.** In 1906 the mathematician Pavel Nekrasov claimed that the law of large numbers required independent events, and drew conclusions from it about free will. Andrey Markov set out to construct *dependent* sequences that obey the law anyway — and did. The machinery underlying this chapter is a by-product of a dispute about determinism.

- **The first statistical language model was built by hand in 1913.** Markov demonstrated his chains on the first 20,000 letters of Pushkin's *Eugene Onegin*, counting vowel–consonant transitions with pencil and paper. It is a defensible claim that this was the first n-gram language model.

- **Discrete diffusion with a masking kernel is BERT with a randomized mask rate.** BERT masks 15% of tokens, always. Masked diffusion sweeps the rate from 0% to 100% and derives the correct weighting for each — and that single change converts a representation-learning objective into a generative model capable of writing from a blank page.

- **Generation with an absorbing kernel starts from a blank page, not from noise.** The chain's stationary state is a sequence of all `[MASK]` — a single deterministic state. Unlike continuous diffusion, where the randomness enters at the start, here all the randomness enters through the reverse sampling steps.

- **Timestep conditioning in diffusion models was inherited from artistic style transfer.** In 2017, Dumoulin, Shlens and Kudlur showed that one convolutional network could paint in dozens of artists' styles with *all* weights shared, changing only the gain and bias of its normalization layers. FiLM generalized that, and diffusion models adopted it wholesale for injecting $t$.

- **FiLM was invented for visual question answering, not for generation.** Perez and co-authors were trying to condition an image network on questions like "is there a cube to the left of the red sphere?" The mechanism turned out to be the natural way to tell a diffusion model how noisy its input is.

- **The DiT paper was rejected on its first conference submission.** Its authors have spoken publicly about it; the objection was novelty, since "use a transformer instead of a U-Net" sounds obvious. The paper's real contribution was the ablation showing AdaLN-Zero dominates every alternative at every scale, and that FID scales predictably with FLOPs — and that predictability is why every frontier image and video model now uses the architecture.

- **The single most important architectural finding in the DiT paper is a line of code that does nothing.** AdaLN-Zero initializes each block's gate to exactly zero, so at step 0 a 28-layer network computes the identity function. Depth is then grown by training rather than assumed at initialization.

- **Halving the DiT patch size multiplies attention cost by sixteen.** Token count scales as $(I/p)^2$ and attention as the square of token count, so $p$ from 4 to 2 takes 64 tokens to 256 and attention pairs from 4,096 to 65,536. Patch size is the most consequential single hyperparameter in the architecture.

- **Classifier-free guidance in discrete space is a product of experts.** Because softmax turns addition of logits into multiplication of probabilities, extrapolating logits raises the conditional/unconditional probability ratio to the power $w$. Multiplication means every factor can veto — which is exactly why guidance sharpens onto a single mode rather than blurring.

- **Discrete diffusion cannot use a KV cache, and that erases most of its apparent speed advantage.** An autoregressive model reuses all previous computation for each new token; a diffusion step changes every position at once and can cache nothing. "20 steps instead of 1000" is a much smaller win in FLOPs than in step count.

- **The independence assumption inside a single denoising step is why parallel text generation is hard.** Sampling every masked position independently can produce "New Delhi" when the model meant "New York" — each token locally plausible, the pair jointly wrong. The fix is to unmask fewer positions per step, which is to say: buy coherence with steps.

---

## Check for Understanding

**Discrete diffusion replaces Gaussian noise with a categorical transition matrix, keeping every structural feature of Chapter 20 — a closed-form cumulative corruption $\bar Q_t$, an exact posterior, a clean-data parameterization, and an auxiliary $x_0$ cross-entropy that carries most of the training signal — and with an absorbing kernel it reduces to BERT with a continuously varying mask rate; conditioning enters through AdaLN-Zero, whose zero-initialized gate makes every block start as the identity, which is the same trick that made deep residual networks trainable in the first place.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why can't you add a little Gaussian noise to a token?** Give the telephone-number analogy, and then say what the two legitimate replacements are.
2. **What is a transition matrix, and why does composing two steps mean multiplying two matrices?** The answer is one sentence about summing over intermediate possibilities.
3. **Why does the forward process need no neural network?** And what does freezing it buy you that a variational autoencoder's learned encoder does not?
4. **Why did masking beat uniform corruption?** (Correct answer: it deletes a subproblem — the model never has to *detect* which tokens are corrupted.)
5. **What does "absorbing state" mean, and why does generation therefore start from a blank page rather than from noise?**
6. **Why does the network predict the original token rather than the previous step?** Say what happens to the training signal at $t=900$ under each choice.
7. **Why run twenty steps if the network already outputs its best guess at the clean sequence?** The answer is "New Delhi," and you should be able to explain why.
8. **In what sense is masked diffusion "BERT with a random mask rate," and what exactly does BERT lack that stops it generating?**
9. **Why is "20 steps instead of 1,000" not a 50× speedup?** One phrase: KV cache.
10. **What is FiLM doing, in terms of a graphic equalizer?** And name one condition it structurally cannot carry.
11. **What does AdaLN-Zero's gate do on the very first training step, and why is starting a 28-layer network as a 0-layer network a good idea?**
12. **Why does halving the DiT patch size multiply attention cost by sixteen rather than four?**
13. **Why is guidance a product of experts rather than an average?** Say what "every factor can veto" means for the shape of the output distribution.
14. **Why must you report cross-entropy per timestep bucket instead of one aggregate number?** Explain what quantity the aggregate is secretly averaging over.

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 22 — Classical Supervised Learning](22-classical-supervised-learning.md)
