# Chapter 11 — Attention & the Transformer

> **Prerequisites:** Ch. 6, Ch. 7 (LayerNorm), Ch. 9 (§9.6), Ch. 10.
> **This is the most important chapter in the book.** Nearly every model deployed in 2026 is a transformer.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, $A^\top$, or $\mathcal{O}(n^2)$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. Chapter 0 §0.8 (the four ways to multiply) and §0.10 (big-O) are the two sections this chapter leans on hardest.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $T$ | "T" | **Sequence length** — how many tokens are in the input |
| $d$, $d_{\text{model}}$ | "d", "d-model" | The **width** of the model: how many numbers describe one token |
| $Q,K,V$ | "queries, keys, values" | Three different learned views of the same input — what each position **asks**, what it **advertises**, and what it **hands over** |
| $W_Q,W_K,W_V,W_O$ | "W-Q, W-K, W-V, W-O" | The weight matrices that produce $Q,K,V$, and the one that recombines the heads |
| $d_k$, $d_v$ | "d-k", "d-v" | Width of one head's keys/queries, and of its values |
| $\alpha_{ij}$ | "alpha i-j" | **Attention weight**: what fraction of position $i$'s update comes from position $j$ |
| $QK^\top$ | "Q K transpose" | The $T\times T$ grid of every query's match score against every key |
| $\mathrm{softmax}(z)$ | "softmax of z" | Turn a row of scores into positive numbers summing to 1 |
| $\sqrt{d_k}$ | "root d-k" | The divisor that keeps scores from saturating the softmax |
| $h$ | "h" | Number of **attention heads** |
| $\delta_{ij}$ | "Kronecker delta" | 1 if $i=j$, else 0 — an `if i == j` written as a symbol |
| $-\infty$ | "minus infinity" | A masked-out score, since $e^{-\infty}=0$ |
| $L$ | "L" | Number of **layers** in the stack |
| $d_{\text{ff}}$ | "d-f-f" | Width of the feed-forward hidden layer, usually $4d$ |
| $\phi(\cdot)$ | "phi" | A generic activation function (ReLU, GELU, SiLU …) |
| $\odot$ | "elementwise product" | Multiply matching entries and keep them separate (Ch. 0 §0.8) |
| $N$, $D$, $C$ | "N, D, C" | Parameter count, training-token count, total training FLOPs |
| $E$ | "E" | Number of **experts** in a mixture-of-experts layer |
| $\mathrm{TopK}(\cdot)$ | "top-k" | Keep the $k$ largest entries, discard the rest |
| $f_i$, $P_i$ | "f-i", "P-i" | Fraction of tokens routed to expert $i$; mean router probability for expert $i$ |
| $\mathsf{TC}^0$ | "T-C-zero" | A **complexity class**: what constant-depth threshold circuits can compute |

### Abbreviations used in this chapter

The book's complete glossary lives in [Chapter 0 §0.13](00-notation-and-math-primer.md). These are the ones you need here.

| Short | Full form |
|---|---|
| BOS | Beginning Of Sequence (token) |
| BERT | Bidirectional Encoder Representations from Transformers |
| FFN | Feed-Forward Network |
| FLOP | FLoating-point OPeration |
| GELU | Gaussian Error Linear Unit |
| GPT | Generative Pre-trained Transformer |
| KV | Key–Value (cache) |
| LLM | Large Language Model |
| LM | Language Model |
| LN | Layer Normalization |
| MEMIT | Mass-Editing Memory In a Transformer |
| MHA | Multi-Head Attention |
| MoE | Mixture of Experts |
| RMSNorm | Root Mean Square Normalization |
| ROME | Rank-One Model Editing |
| SiLU | Sigmoid Linear Unit |
| SwiGLU | Swish-Gated Linear Unit |
| ViT | Vision Transformer |

---

## 11.1 Attention from first principles

### The one-line idea

Every position emits a question, every position advertises what it has, and each position builds its next representation as a weighted average of what everyone has — weighted by how well the questions match the advertisements.

### The analogy

A conference poster session. You (a **query**) walk the hall with a specific question. Every poster has a title (**key**) and content (**value**). You glance at all the titles, decide how relevant each is, and then absorb content from each poster in proportion to that relevance — mostly from one or two, a little from several others. Then you update your understanding and walk the hall again with a *new* question. That second pass is the second layer.

### Building it up

Suppose we want position $i$'s new representation to be a mixture of all positions' information:

$$y_i = \sum_j \alpha_{ij}v_j,\qquad \sum_j\alpha_{ij}=1,\ \alpha_{ij}\ge0$$

We need $\alpha_{ij}$ to depend on how relevant $j$ is *to $i$*. Use a dot product as the relevance score and softmax to normalize. Give each position three different learned projections of its embedding so that "what I'm looking for" (query), "what I contain" (key), and "what I pass on" (value) can be different things:

$$Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V$$

▸ $$\boxed{\ \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\ }$$

Shapes: $X\in\mathbb{R}^{T\times d}$, $W_Q,W_K\in\mathbb{R}^{d\times d_k}$, $W_V\in\mathbb{R}^{d\times d_v}$, scores $\in\mathbb{R}^{T\times T}$, output $\in\mathbb{R}^{T\times d_v}$.

#### Reading the mixing equation first

Before the famous boxed formula, understand the line it is built from:

$$y_i = \sum_j \alpha_{ij}v_j,\qquad \sum_j\alpha_{ij}=1,\ \alpha_{ij}\ge0$$

| Piece | Read aloud | Meaning |
|---|---|---|
| $y_i$ | "y-i" | The **new** representation of position $i$ |
| $v_j$ | "v-j" | What position $j$ has to offer |
| $\alpha_{ij}$ | "alpha i-j" | How much of $j$'s offering goes into $i$'s update |
| $\sum_j \alpha_{ij}=1$ | "the alphas sum to one" | The weights are shares of a fixed budget |
| $\alpha_{ij}\ge 0$ | "alpha is non-negative" | You can take some of $j$, or none — never a negative amount |

▸ **Those two constraints together mean $y_i$ is a weighted average, and a weighted average is a *soft selection*.** If $\alpha_{i3}=1$ and everything else is 0, position $i$ has copied position 3 exactly. If the weights are spread evenly, it has blended everyone. Every intermediate case is available. **This is the whole reason attention is differentiable: hard lookup ("fetch item 3") has no gradient, but a soft weighted average does.**

> **Analogy.** A dictionary lookup returns exactly one entry: `d["cat"]` or a `KeyError`. Attention is a dictionary that returns 70% of "cat," 20% of "kitten," and 10% of "feline" — because the query didn't quite match anything and it would rather blend than fail. You can differentiate a blend. You cannot differentiate a `KeyError`.

#### Reading the attention formula in plain English

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Read the whole thing aloud once — *"softmax of Q K-transpose over root d-k, times V"* — then take it in four moves, inside out.

**Move 1 — $QK^\top$: every question meets every advertisement.** $Q$ is $T\times d_k$ (one query row per position) and $K^\top$ is $d_k\times T$. By the shape rule (Ch. 0 §0.8), $(T\times d_k)(d_k\times T) = T\times T$. Entry $(i,j)$ of that grid is $q_i^\top k_j$ — the dot product of position $i$'s query with position $j$'s key, i.e. **how well does what $i$ is looking for match what $j$ is offering?** One matrix multiply computes all $T^2$ of those comparisons at once, and that is where the $\mathcal{O}(T^2)$ cost of transformers comes from. It is not hidden anywhere subtle; it is this one product.

**Move 2 — divide by $\sqrt{d_k}$.** Keeps the numbers in a range where softmax is responsive. Derived properly in §11.2.

**Move 3 — softmax, applied along each row.** Each row of the $T\times T$ grid becomes a probability distribution: non-negative, summing to 1. **Row $i$ is now exactly the $\alpha_{ij}$ from the mixing equation.** The row-wise detail matters: position $i$'s weights over all $j$ sum to 1, but a column need not sum to anything in particular — attention is a *directed* relation.

**Move 4 — multiply by $V$.** $(T\times T)(T\times d_v) = T\times d_v$. Each output row is the weighted average of all the value rows. Shapes come out right, which is the check Chapter 0 §0.1 recommends doing first.

**Now work it with actual numbers.** Take $T=3$ tokens and $d_k=2$, and suppose the projections have produced:

$$q_1 = (1,\ 0),\qquad k_1 = (1,\ 0),\quad k_2 = (0.6,\ 0.8),\quad k_3 = (-1,\ 0)$$

with values $v_1 = (10, 0)$, $v_2 = (0, 10)$, $v_3 = (5,5)$.

| $j$ | $q_1^\top k_j$ | $\div\sqrt{2}$ | $\exp(\cdot)$ | $\alpha_{1j}$ |
|---|---|---|---|---|
| 1 | $1.0$ | $0.707$ | $2.028$ | $0.501$ |
| 2 | $0.6$ | $0.424$ | $1.529$ | $0.377$ |
| 3 | $-1.0$ | $-0.707$ | $0.493$ | $0.122$ |

Sum of exponentials: $4.050$. So position 1's output is

$$y_1 = 0.501(10,0) + 0.377(0,10) + 0.122(5,5) = (5.62,\ 4.38)$$

▸ **Read what happened: position 1 took about half of token 1, a bit over a third of token 2, and a token pointing the opposite way still contributed 12%.** Softmax never returns exactly zero. That is deliberate — a hard zero would kill the gradient to that position and it could never be recovered.

**Now the three projections, decoded.** $Q = XW_Q$ reads *"multiply the input matrix by a learned weight matrix"* — one matrix multiply producing all $T$ queries at once. Each position's embedding is asked three separate questions:

| Projection | The question it answers | Poster-session analogy |
|---|---|---|
| $W_Q$ | *"What am I looking for?"* | The question you walk in with |
| $W_K$ | *"What am I, for matching purposes?"* | The title on your poster |
| $W_V$ | *"What do I contribute if selected?"* | The content of the poster |

▸ **The single most important structural fact about attention: the weights $\alpha$ are computed from the data, not stored in the parameters.** A convolution's kernel is fixed after training and applies identically to every input. Attention's mixing pattern is recomputed for every sequence. **This is why one set of weights can handle "the cat sat on the mat" and a Python traceback** — the routing is a function of the input, so the same parameters implement different data flows.

> **Where this came from.** Attention was not invented for transformers. It was invented in 2014 by **Dzmitry Bahdanau**, then a visiting student in **Yoshua Bengio's** lab in Montreal, working with **Kyunghyun Cho**, to fix a specific defect in neural machine translation: sequence-to-sequence models compressed the entire source sentence into one fixed-length vector before decoding, and quality collapsed on long sentences because everything had to fit through that bottleneck. Their fix — let the decoder look back at all the encoder states and learn *where* to look — was called "soft alignment," and translation quality on long sentences stopped degrading. **Minh-Thang Luong** and colleagues at Stanford simplified the scoring function to a plain dot product in 2015. For three years attention was a **helper bolted onto a recurrent network**. The 2017 move was to delete the recurrent network and keep the helper.

> **The story behind "Attention Is All You Need."** The 2017 paper came from a Google team of eight, and it carries an unusual footnote stating that the authors are listed in **random order** with equal contribution — an explicit rejection of the convention that author position encodes credit. **Llion Jones** has said the title was his, a nod to the Beatles' "All You Need Is Love." The paper's stated goal was machine translation, and its abstract sells the architecture largely on **training speed** — recurrence forces you to process tokens one after another, while attention lets you process the whole sequence in parallel, and GPUs are built for exactly that. ▸ **The transformer won first as an engineering result about parallelism, and only afterwards as a modelling result.** By most accounts none of the eight authors expected it to become the substrate of an industry; all eight had left Google within a few years.

### Why three projections and not one?

If $Q=K$, the attention matrix is symmetric and every token attends maximally to itself (its own dot product is the largest). Separating $Q$ and $K$ allows **asymmetric** relations: "adjective looks for the noun it modifies" is not the same relation as "noun looks for its adjective."

If $V=K$, the thing used for *matching* is forced to equal the thing that gets *transmitted*. Separating them means a token can be found by one property and contribute a different one. (In circuit terms, Ch. 32: the QK circuit decides *where* to read, the OV circuit decides *what* to write.)

#### Why three projections, unpacked

Both arguments deserve concrete form, because "you need three matrices" sounds arbitrary until you see what collapses without them.

**Case 1: $Q = K$.** Then the score grid is $XW(XW)^\top$, which is **symmetric** — score$(i,j)$ = score$(j,i)$. Two consequences, both fatal:

- **Self-attention becomes self-obsession.** A vector's dot product with itself is $\lVert q_i\rVert^2$, which by the Cauchy–Schwarz inequality is at least as large as its dot product with any other vector of the same length. **Every token's highest score is itself**, so every token mostly copies itself, and the layer computes nothing.
- **All relations become mutual.** "Adjective looks for the noun it modifies" and "noun looks for its adjectives" would be forced to have the same strength. But in *"the red car,"* `red` badly needs to know what it modifies, while `car` barely needs `red` at all. **Language is full of one-way dependencies**, and a symmetric matrix cannot express one.

▸ **Separating $W_Q$ from $W_K$ is what makes the attention pattern a directed graph rather than an undirected one.**

**Case 2: $V = K$.** Then whatever makes a token *findable* is also everything it can *deliver*. Consider a pronoun-resolution head: it wants to find the antecedent by matching on grammatical features (singular, animate, subject position), but what it wants to *retrieve* is the semantic content — who that entity actually is. ▸ **Matching on one property and transmitting another is the entire point of a lookup table**, and merging $K$ with $V$ turns a dictionary into a list.

> **Analogy.** A library catalogue card carries a call number (the key: short, standardized, designed for matching) and describes a book (the value: long, idiosyncratic, the thing you actually want). Making the key equal the value means shelving books by their full text. Making the query equal the key means every book can only be found by someone holding an identical book.

**Where the terminology came from.** The words *query*, *key*, and *value* are lifted from database and information-retrieval vocabulary, and the borrowing is exact: you present a query, it is matched against keys, and you receive the associated values. ▸ **The only modification is that the match is soft and the retrieval returns a blend.** If you already know what a hash table is, you already know what attention is — you just have to allow the lookup to be fuzzy.

#### Examples and non-examples: what attention actually is

**✅  instances of the attention operation**

| Example | Why it qualifies |
|---|---|
| A GPT layer where token 40 draws 0.6 of its output from token 7 and 0.4 spread over the rest | Data-dependent, non-negative weights summing to 1, applied to *values* |
| Bahdanau's 2014 decoder looking back over encoder states | The original: weights computed from a learned score, then normalized |
| Cross-attention in an encoder–decoder, where queries come from the decoder and keys/values from the encoder | Same operation; only the source of $Q$ versus $K,V$ differs |
| A retrieval system that embeds a query, softmaxes similarities over 10,000 documents, and returns a weighted blend of their embeddings | Attention over a corpus instead of a sequence — literally the same three lines |
| Average pooling over $T$ tokens | The degenerate case $\alpha_{ij} = 1/T$ — a *constant* attention pattern, which is why it is a legitimate but uninteresting member of the family |

**❌ Near-misses — things that mix information across positions but are not attention**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A convolution | Its kernel is **fixed after training**; the same weights apply to every input | A learned but static local mixing operator |
| A fully-connected layer over the flattened sequence | Weights are parameters, not computed from the data, and the layer only works at one sequence length | A dense layer |
| Gating in an LSTM, $\sigma(\cdot) \odot c$ | Gates are elementwise, per-feature, and do not sum to 1 across positions | Multiplicative feature gating |
| A learned but **input-independent** $T\times T$ mixing matrix (as in some MLP-Mixer variants) | Fixed at inference; cannot route differently for different inputs | Token mixing with static weights |
| Squeeze-and-excitation channel weights | Data-dependent, but they reweight **channels**, not positions, and they do not sum to 1 | Channel attention by analogy of name only |
| $\mathrm{softmax}(QK^\top)$ with no $V$ | You have computed the routing and thrown away the payload | An attention *map*, half the operation |
| Hard top-$k$ selection of one position | Not differentiable at the selection boundary — the gradient to the unselected positions is exactly zero | Discrete retrieval (usable, but needs a gradient estimator) |

▸ **The boundary:** attention is a **weighted average whose weights are computed from the current input and sum to 1 along the source axis.** Drop "computed from the input" and you have a convolution or a dense layer; drop "sums to 1" and you have gating; drop "weighted average" and you have selection.

> **Common misconception.** *"Attention shows you which words the model thinks are important."* An attention weight says where information was *read from* on one head at one layer — it does not say the information mattered, and it does not say what was done with it. A head can place 0.9 of its weight on a token and then multiply the retrieved value by an OV circuit that outputs almost nothing. In a 32-layer, 32-head model there are over a thousand of these maps, and the one you plotted is a thousandth of the computation. **The tempting part is that the numbers are non-negative, sum to 1, and look exactly like a probability distribution over "importance"** — the format invites the interpretation. There is a substantial literature (Jain and Wallace's *Attention is not Explanation*, 2019, and Wiegreffe and Pinter's reply, *Attention is not not Explanation*, 2019) arguing over precisely how much of that interpretation survives; the honest summary is "less than the picture suggests, and the amount is contested."

> **Common misconception.** *"Softmax picks the best match."* Softmax is a *soft* argmax and it never returns exactly zero for anything. In the worked example above, a key pointing in the **opposite direction** to the query still received 12.2% of the weight. That is not a rounding artefact — it is the property that makes attention trainable, because a hard zero would send zero gradient to that position and the model could never learn to attend there later. **The leakage is the feature.**

> **Common misconception.** *"$Q$, $K$, and $V$ are three different kinds of thing."* At the start of the layer they are three different **linear projections of the same vectors**. In self-attention, $Q = XW_Q$, $K = XW_K$, and $V = XW_V$ all come from the identical input $X$. The three names describe three *roles* the same token plays simultaneously — what it wants, how it advertises itself, and what it hands over. The misconception is tempting because the database analogy implies queries and keys live in different tables, and in cross-attention they  do.

---

## 11.2 The $\sqrt{d_k}$ scaling — derive it, it's asked constantly

Assume $q,k\in\mathbb{R}^{d_k}$ have i.i.d. entries with mean 0 and variance 1. Then

$$\mathbb{E}[q^\top k] = \sum_{i=1}^{d_k}\mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$
$$\mathrm{Var}(q^\top k) = \sum_{i=1}^{d_k}\mathrm{Var}(q_ik_i) = \sum_{i=1}^{d_k}\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$$

▸ So the raw scores have standard deviation $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ restores unit variance.

#### Unpacking the $\sqrt{d_k}$ derivation

Two lines of algebra, each doing one job. Take them apart.

**The assumption.** "$q,k\in\mathbb{R}^{d_k}$ have i.i.d. entries with mean 0 and variance 1" says: each of the $d_k$ numbers in a query is drawn independently, averages to zero, and has typical size 1. **i.i.d.** is **independent and identically distributed** — the entries do not influence each other and all come from the same distribution. This is what you get at initialization, and it stays roughly true afterwards, which is enough for the argument.

**Line 1 — the mean is zero.** $\mathbb{E}[q^\top k] = \sum_i \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$. Each term factors into a product of expectations *because* $q_i$ and $k_i$ are independent, and each expectation is 0, so every term is 0. ▸ **Plain meaning: a random query and a random key have no reason to align, so on average they score zero.** Correct and unsurprising.

**Line 2 — the variance is $d_k$.** $\mathrm{Var}(q^\top k) = \sum_i \mathrm{Var}(q_ik_i) = \sum_i \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$. The first equality is *"variances of independent things add"*; the second uses $\mathrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$ with mean zero, so variance is just $\mathbb{E}[X^2]=1$. Each of the $d_k$ terms contributes $1\times1$, so the total is $d_k$.

▸ **Standard deviation is the square root of variance, so the scores have spread $\sqrt{d_k}$ — and that is the whole derivation.** Dividing by $\sqrt{d_k}$ brings it back to 1.

> **Analogy.** A hundred people each guess whether a coin lands heads, scoring $+1$ or $-1$. The average score is 0. But the *spread* is not 0 — it is $\sqrt{100}=10$, so a total of $\pm20$ is entirely ordinary. Random contributions do not cancel exactly; they accumulate like the square root of how many there are. This is the same $\sqrt{n}$ that governs standard error (Ch. 1 §1.3.1), random-walk distance, and near-orthogonality in high dimensions (Ch. 1 §1.1.5). **It is probably the most reused number in this book.**

**Sanity-check with the standard configuration.** $d_k = 64$, so unscaled scores have standard deviation $\sqrt{64} = 8$. Softmax cares about *differences* between scores, and among $T$ draws from a spread-8 distribution the top two commonly differ by several units. A gap of 10 means the larger term is $e^{10} \approx 22{,}000$ times the smaller — the softmax output is $0.99995$ on one entry. With scaling, the same gap becomes about 1.25, and $e^{1.25}\approx3.5$: a strong preference, not an absolute one.

▸ **The general rule worth carrying out of this section: whenever a sum of $n$ random terms feeds a saturating nonlinearity, ask whether it needs dividing by $\sqrt n$.** He initialization does this for layer widths; attention does it for head dimension; the arithmetic is identical.

### Why unit variance matters — the gradient argument

Recall (Ch. 1 §1.3.4) that for $p=\mathrm{softmax}(z)$,
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j)$$

If one score dominates, $p\to$ one-hot, and every entry of that Jacobian $\to0$. **The softmax saturates and its gradient vanishes.**

**Numbers.** $d_k=64$, unscaled scores have SD 8. The gap between the largest and second-largest of $T$ such scores is several units, so $e^{\Delta}$ with $\Delta\approx10$ gives a softmax that is $>0.9999$ on one entry. Gradient $\approx p(1-p)\approx10^{-4}$. With scaling, SD is 1, gaps are $O(1)$, and the softmax stays in its responsive range.

▸ **General principle worth extracting:** any time you feed a sum of $n$ terms into a saturating nonlinearity, check whether you need to divide by $\sqrt n$. This is the same variance-propagation argument as He initialization (Ch. 6 §6.4).

**Related failure at scale — logit explosion.** During training, $\|q\|$ and $\|k\|$ can grow, re-saturating attention even with the $\sqrt{d_k}$ factor. Fix: **QK-normalization** (apply LayerNorm/RMSNorm to $Q$ and $K$ before the dot product), now standard in large models (Ch. 14 §14.6).

#### What the softmax Jacobian says, and why saturation kills learning

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j)$$

**Decoding it.** $z$ is the vector of raw scores, $p = \mathrm{softmax}(z)$ the resulting probabilities. $\partial p_i/\partial z_j$ asks *"if I nudge score $j$, how much does probability $i$ move?"* And $\delta_{ij}$ is the **Kronecker delta** — 1 when $i=j$, 0 otherwise (Ch. 0 §0.6). It is an `if` statement written as a symbol, and it lets one formula cover two cases:

| Case | Formula becomes | Reading |
|---|---|---|
| $i = j$ | $p_i(1-p_i)$ | Raising a score raises its own probability |
| $i \ne j$ | $-p_ip_j$ | Raising one score **lowers** every other — probabilities are a fixed budget |

**Now watch it die.** Put in the numbers from the saturated case, $p_i = 0.99995$:

$$p_i(1-p_i) = 0.99995 \times 0.00005 = 5\times10^{-5}$$

Compare the healthy case $p_i = 0.5$: $0.5\times0.5 = 0.25$. ▸ **A factor of 5,000 difference in gradient magnitude, from a change that made the attention pattern "more confident."** The layer has effectively stopped learning where to look, and — this is the cruel part — it stopped while producing outputs that look perfectly reasonable. Saturation is silent.

> **Analogy.** A dimmer switch versus a light switch. Near the middle, a small turn of the dimmer visibly changes the brightness, so you can find the setting you want by feel. Pushed hard to one end, turning it further does nothing you can perceive — and if you are trying to *learn* the right setting by observing the effect of small adjustments, you have no information to learn from. Softmax with one dominant score is a dimmer jammed at the top.

**Why $\sqrt{d_k}$ is not sufficient on its own.** The derivation assumed entries with variance 1. During training, nothing enforces that: the weight matrices $W_Q$ and $W_K$ grow, so $\lVert q\rVert$ and $\lVert k\rVert$ grow with them, and the dot product grows as their product. Divide by a constant $\sqrt{d_k}$ and you have corrected for **dimension** but not for **magnitude drift**. In very large models this shows up as attention logits climbing into the hundreds and training loss spiking or diverging.

▸ **QK-normalization fixes it at the source: normalize $Q$ and $K$ to fixed length *before* the dot product, so scores are bounded no matter what the weights do.** The pattern is worth naming because it recurs — *a scaling constant chosen at initialization is a one-time correction; a normalization layer is a permanent one.* Chapter 7's whole argument for normalization is this sentence, and Chapter 14 §14.6 covers the large-scale-training version.

---

## 11.3 Multi-head attention

### The one-line idea

Run $h$ independent attention operations in parallel on lower-dimensional projections, so different heads can attend to different kinds of relationships simultaneously.

### The analogy

Reading a contract with a team: one person tracks defined terms, one tracks dates, one tracks dollar amounts, one tracks obligations. Each reads the whole document with a different question in mind. Then you combine their notes. A single reader with one question would miss most of it.

### The math

▸ $$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O,\qquad \mathrm{head}_i = \mathrm{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

with $d_k = d_v = d_{\text{model}}/h$, so the total parameter count and FLOP count match single-head attention at full width.

**Standard config:** $d_{\text{model}}=512$, $h=8$, $d_k=64$. Or $d=4096$, $h=32$, $d_k=128$.

#### Reading multi-head attention

$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O,\qquad \mathrm{head}_i = \mathrm{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

| Piece | Read aloud | What it does |
|---|---|---|
| $\mathrm{head}_i$ | "head i" | One complete attention operation, on its own narrow projections |
| $W_Q^i$ | "W-Q superscript i" | Head $i$'s **own** query matrix — the superscript indexes the head, it is not a power |
| $\mathrm{Concat}(\cdots)$ | "concatenate" | Lay the $h$ outputs side by side into one wide vector |
| $W_O$ | "W-O" | A learned matrix that mixes the concatenated heads back to width $d$ |

**Follow the shapes, which is where the elegance is.** With $d_{\text{model}} = 512$ and $h = 8$, each head uses $d_k = d_v = 512/8 = 64$:

1. Each head projects $X$ ($T\times512$) down to $T\times64$ — a **narrow** view.
2. Each head attends within its own 64 dimensions, producing $T\times64$.
3. Concatenating 8 of those gives $T\times512$ — back to full width.
4. $W_O$ ($512\times512$) mixes them.

▸ **The parameter count is identical to one head of full width.** $h$ heads at $d/h$ each is $h \times d \times (d/h) = d^2$ per projection — exactly what a single $d\times d$ projection costs. **Multi-head attention is free.** You are not buying extra capacity; you are *rearranging* the same capacity into several independent lookups instead of one.

> **Analogy.** You have eight hours of staff time to read a contract. Option A: one lawyer reads it for eight hours with one question in mind. Option B: eight lawyers read it for one hour each, one tracking dates, one tracking dollar figures, one tracking obligations. Same cost. Option B catches far more, because the binding constraint was never reading time — it was that a single reader can only hold one question at a time.

**Why "one softmax = one commitment" is the real argument.** A softmax row must sum to 1, so its mass is a fixed budget. If a token needs its syntactic head, its coreferent antecedent, *and* the previous token, a single distribution must divide 1.0 among three unrelated targets — and then the weighted average blends three incompatible things into mush. With three heads, each gets its own budget of 1.0, retrieves cleanly, and $W_O$ decides how to combine them afterwards. ▸ **Heads exist because attention's output is an average, and averaging unrelated things destroys them.**

**The rank argument, decoded.** $QK^\top$ is $(T\times d_k)(d_k\times T)$, so by the shape rule its rank is at most $d_k$ (Ch. 1 §1.1.3) — a $T\times T$ grid that only has $d_k$ independent directions in it. For $T = 2048$ and $d_k = 64$, that is a 2048×2048 table with at most 64 degrees of freedom per side. A single wide head with $d_k = 512$ raises the cap to 512, but it is still **one** low-rank structure. Eight heads give a **sum of eight** independently-shaped patterns, which spans strictly more than one pattern of the same total width. **More small pieces beat one large piece, for the same reason a sum of eight rank-64 matrices is more expressive than one rank-512 matrix constrained to a single softmax.**

**On pruning.** Michel and colleagues found most heads can be removed at inference with little loss, which sounds like it refutes everything above. It doesn't, and the resolution is worth stating: ▸ **the heads are redundant in the *trained* model but appear to be necessary during *training*.** Extra heads seem to function like extra lottery tickets (Ch. 31) — many paths to a good solution, of which the optimizer only needs to find one. You cannot train the pruned network from scratch and get the same result, which is the signature of an optimization benefit rather than a capacity one.

### Why a single wide head is worse

A single softmax produces **one** distribution per query. It must commit to one weighting. With $h$ heads you get $h$ distributions, so a token can simultaneously copy from its syntactic head, its coreferent antecedent, and the previous token.

There is also a rank argument: $\mathrm{softmax}(QK^\top/\sqrt{d_k})$ is a $T\times T$ matrix built from a rank-$d_k$ score matrix. Multiple heads give a *sum* of $h$ such low-rank-driven operations, which is strictly more expressive than one at the same total width.

**Diminishing returns:** Michel et al. (2019) showed most heads can be pruned at inference with little loss — often 1 head per layer suffices for many layers. Heads are redundant, but the redundancy appears to help *optimization*.

### What heads actually learn (empirically, in trained models)

- **Positional heads:** attend to $i-1$ or $i+1$.
- **Syntactic heads:** verb→subject, noun→determiner (measurably aligned with dependency parses).
- **Coreference heads:** pronoun→antecedent.
- **Duplicate-token / previous-token heads:** the components of induction circuits (Ch. 32).
- **"Attention sink" behaviour:** many heads dump most of their mass on the first token (or a BOS token) when they have nothing to do. This is a *no-op mechanism* — softmax must sum to 1, so a head with nothing to retrieve parks its mass somewhere harmless. Knowing this explains a real deployment issue: **evicting the first tokens from a KV cache destroys the model** (hence StreamingLLM keeps them).

#### Attention sinks, decoded — the clearest example of an architectural bug becoming a feature

This deserves unpacking because it is the single most useful thing in this chapter for anyone who deploys models, and because the reasoning is a small gem.

**The problem the model faces.** Softmax output must sum to exactly 1. A head therefore **cannot decline to attend**. There is no "none of the above" option, no null token, no way to output all zeros. Yet most heads, most of the time, have nothing to retrieve — a head that detects relative clauses has no work to do in a sentence without one.

**The model's solution, discovered without being designed.** Park the mass on a token whose *value* vector contributes almost nothing. The first token of the sequence is ideal: it is present in every sequence, it is at a fixed position, and — because it has no left context in a causal model — it is the least informative token available. **Attending to it is the closest thing to a no-op that the architecture permits.**

> **Analogy.** A voting system where abstention is forbidden and every ballot must name a candidate. A "none of the above" option appears on the ballot spontaneously, in the form of a joke candidate everyone votes for when they don't care. The joke candidate receives an enormous vote share that means nothing — and if you remove them from the ballot on the grounds that nobody really supports them, the votes scatter unpredictably across real candidates and the election result changes completely.

▸ **That last clause is exactly the deployment bug.** Long-context serving systems evict old tokens from the KV cache to bound memory, and the oldest tokens are the first ones. Evict the sink and every head that was parking there must redistribute its mass onto tokens that *do* carry information — injecting a large amount of content the model never intended to read. Perplexity does not degrade gracefully; it explodes. **StreamingLLM's fix is almost comically simple: keep the first four tokens forever, evict from the middle.** A few kilobytes of cache, and a broken system works.

Three practical corollaries fall out of the same reasoning, and they connect this section to the rest of the book:

- ▸ **A very large attention weight is not evidence of importance.** Attention maps are routinely presented as explanations; a head putting 90% of its mass on token 1 is usually saying nothing at all. Chapter 32 treats this carefully.
- **Quantization (Ch. 17) struggles with sink tokens**, because their value vectors carry outlier activations that dominate the numerical range.
- **Some architectures now add an explicit learned "null" slot** to the attention softmax — a dedicated place for unused mass. Giving the model the abstention option it was improvising is a cleaner fix than letting it commandeer a real token.

---

## 11.4 Masking

### Causal (decoder) masking

▸ $$\mathrm{scores}_{ij} \leftarrow \begin{cases}\mathrm{scores}_{ij} & j\le i\\ -\infty & j>i\end{cases}$$

$-\infty$ (in practice $-10^{9}$ or the dtype minimum) makes $e^{\text{score}}=0$ after softmax. This enforces autoregression: position $i$ cannot see the future.

▸ **The crucial consequence:** with causal masking you can compute the loss at *every* position in one forward pass, so a length-$T$ sequence yields $T$ training signals. This is why GPT-style pretraining is so sample-efficient per FLOP, and why it beat BERT-style masked LM (which only supervises 15% of positions) for generative scaling.

#### What the causal mask actually does

The case-bracket notation reads: *"keep the score if $j \le i$; replace it with minus infinity if $j > i$."* Since $i$ indexes the position doing the looking and $j$ the position being looked at, $j > i$ means **"$j$ is in the future."**

**Why $-\infty$ and not 0.** Setting a score to zero would not hide anything — $e^0 = 1$, so a zeroed score still receives a healthy share of the softmax. You need a value whose exponential is zero, and $e^{-\infty} = 0$. In practice you use the most negative representable number for the datatype, or $-10^9$; in bf16, $e^{-10^9}$ underflows to exactly zero, and the masked position drops out cleanly.

**Draw the grid for $T=4$.** Rows are queries, columns are keys, ✓ means visible:

| | $j{=}1$ | $j{=}2$ | $j{=}3$ | $j{=}4$ |
|---|---|---|---|---|
| $i{=}1$ | ✓ | — | — | — |
| $i{=}2$ | ✓ | ✓ | — | — |
| $i{=}3$ | ✓ | ✓ | ✓ | — |
| $i{=}4$ | ✓ | ✓ | ✓ | ✓ |

▸ **A lower-triangular matrix — and it is worth noticing that only about half the $T^2$ score grid is ever used.** The upper triangle is computed and thrown away in naive implementations; FlashAttention and its relatives skip those blocks entirely, which is a large part of where their speedup comes from.

**Now the consequence that decided the architecture of the industry, spelled out.** Feed the sentence *"the cat sat on the mat"* through a causally-masked transformer once. Position 1 sees `the` and must predict `cat`. Position 2 sees `the cat` and must predict `sat`. And so on. **One forward pass, five prediction problems, five gradient signals — all computed in parallel because the mask enforces the ordering rather than the computation doing so.**

Compare the two pretraining objectives on the same 1,000-token document:

| Objective | Supervised positions per pass | Cost per pass |
|---|---|---|
| Causal LM (GPT) | $1000$ | 1 forward + 1 backward |
| Masked LM at 15% (BERT) | $150$ | 1 forward + 1 backward |

▸ **A 6.7× difference in training signal for the same compute.** That number, compounded over trillions of tokens, is most of the answer to "why did decoder-only win?" — and it is a fact about **masks and loss functions**, not about anything deep concerning language or intelligence.

The counterpoint is real and worth holding: BERT's masked positions see context from *both* sides, which is strictly more information per prediction and produces better representations for classification and retrieval. **The causal model gets more, weaker signals; the masked model gets fewer, stronger ones.** For generation, where you must produce tokens left to right anyway, the trade is obviously worth taking. For encoding a sentence into a vector (Ch. 10 §10.6), it is not, which is why bidirectional encoders never went away in retrieval.

**Reading the other two masks.**

- **Padding masks** exist because batching requires rectangular tensors, so shorter sequences are filled out with a dummy token. That token is not data and must contribute nothing — masked in attention, excluded from the loss, and excluded from any mean-pooling. ▸ **Forgetting the third of those three is one of the most common silent bugs in embedding pipelines**: the vector is computed correctly and then averaged with fifty copies of the padding vector, and quality mysteriously depends on batch composition.
- **Block-causal masks** let you concatenate several short documents into one long sequence — "packing" — without letting a token attend across a document boundary. The mask becomes block-diagonal-plus-lower-triangular. This is pure throughput engineering: it keeps GPUs from processing padding, and at scale it is worth a substantial fraction of total training cost.

### Padding masking

Mask out padded positions so they contribute nothing. Must be applied in attention *and* excluded from the loss and from any mean-pooling.

### Prefix / block-causal masking

Bidirectional within a prompt, causal within the completion. Used by prefix-LMs (UL2, some multimodal models) and for packed-sequence training (block-diagonal masks so packed documents can't attend across boundaries).

---

## 11.5 The transformer block

### Pre-LN (the modern standard)

```
x = x + Attn(LN(x))          # communication: tokens exchange information
x = x + FFN(LN(x))           # computation: each token thinks independently
```

▸ **This two-phase decomposition is the cleanest mental model of a transformer.** Attention moves information *between* positions; the FFN transforms information *within* a position. Nothing else happens.

#### Reading the transformer block

Two lines of pseudocode contain the entire architecture, so read them slowly.

`x = x + Attn(LN(x))` decomposes into three separate decisions:

| Piece | What it is | Why it is there |
|---|---|---|
| `LN(x)` | **Layer normalization** (Ch. 7) | Rescale to a standard size before the sub-layer, so it always sees inputs in a familiar range |
| `Attn(...)` | The sub-layer's actual work | Compute an update |
| `x + ...` | The **residual connection** | Add the update to what was already there, rather than replacing it |

▸ **The `x + ` is the most important character in the architecture.** Without it, each layer must reconstruct everything it wants to keep — a hundred layers means a hundred opportunities to lose information, and the gradient must survive a hundred multiplications on the way back (Ch. 1 §1.1.2: $\lambda^{100}$ is either enormous or zero). With it, the default behaviour of a layer is **to do nothing**, and a layer must actively choose to contribute. Gradients flow to the first layer through a path of pure additions, whose derivative is exactly 1.

> **Analogy.** Two ways to run a document through a hundred reviewers. **Option A:** each reviewer rewrites the whole thing from memory and passes on their version. By reviewer 100 the original is unrecognisable, and nobody can tell which reviewer introduced an error. **Option B:** each reviewer adds tracked changes to the shared document. The original survives, every contribution is separable, and you can trace any change to its author. Residual connections are tracked changes — and they are why mechanistic interpretability (Ch. 32) is possible at all.

**"Communication then computation," made concrete.** Consider processing *"the trophy didn't fit in the suitcase because it was too big."*

- **The attention sub-layer** lets the position holding `it` reach back and gather information from `trophy` and `suitcase`. **No position can do this alone** — it is the only mechanism in the whole architecture that moves information sideways.
- **The FFN sub-layer** then takes each position's now-enriched vector and transforms it in place: recognizing patterns, retrieving stored facts, adjusting the prediction. **Every position runs the identical FFN with identical weights, independently, in parallel.** The FFN cannot see any other position and does not need to.

▸ **Everything a transformer does is one of those two things, repeated $L$ times.** A 96-layer model is 96 rounds of "look around, then think." There is no third mechanism, no memory, no control flow. Being able to say that sentence with confidence is most of what it means to understand the architecture.

**Why pre-LN rather than post-LN.** The alternative places the normalization *after* the addition: `x = LN(x + Attn(x))`. That puts a normalization layer directly on the residual path, so the clean additive highway is interrupted at every layer and the gradient is rescaled $2L$ times on its way back. Pre-LN keeps the highway unbroken — ▸ **the residual stream is never normalized, only read from and written to.** This is why pre-LN trains without a warmup schedule and post-LN needs one, and why post-LN becomes fragile past about a dozen layers (Ch. 7 §7.3).

### The feed-forward network

▸ $$\mathrm{FFN}(x) = W_2\,\phi(W_1x + b_1) + b_2,\qquad W_1\in\mathbb{R}^{d_{\text{ff}}\times d},\ d_{\text{ff}}=4d$$

**Why $4\times$?** Empirical, and remarkably stable across four generations of models. Narrower underperforms; wider gives little.

▸ **The FFN holds about $\frac{2}{3}$ of all non-embedding parameters** ($8d^2$ of $12d^2$ per layer — see §11.7). Any claim that "transformers are attention" is wrong on a parameter-count basis. Attention is the *routing*; the FFN is the *storage*.

#### Unpacking the feed-forward network

$$\mathrm{FFN}(x) = W_2\,\phi(W_1x + b_1) + b_2$$

**This is a two-layer neural network** — the plainest object in the entire book, and it sits inside the most celebrated architecture in machine learning. Read it right to left, since that is the order of operations:

| Step | Shape | What happens |
|---|---|---|
| $x$ | $d$ | One position's vector |
| $W_1x + b_1$ | $d_{\text{ff}} = 4d$ | **Expand** to four times the width |
| $\phi(\cdot)$ | $4d$ | Apply a nonlinearity elementwise ($\phi$ is "phi," a stand-in for ReLU/GELU/SiLU) |
| $W_2(\cdot) + b_2$ | $d$ | **Contract** back to the original width |

At $d = 4096$: a $4096$-vector expands to $16{,}384$, gets bent, and comes back to $4096$. ▸ **The expand-then-contract shape is doing real work. Without the widening, a two-layer network with a nonlinearity between could only represent a limited set of functions; the wide middle is where patterns get separated so a linear map can act on them.** It is the same instinct as a kernel method — project into a higher-dimensional space where the problem becomes easy, then project back.

**And notice the crucial constraint: no index $j$ appears anywhere.** The FFN sees one position's vector and nothing else. Every position is processed by identical weights with no knowledge of its neighbours. **All communication in a transformer happens in attention; the FFN is a per-token function applied $T$ times in parallel.**

> **Analogy.** Attention is the meeting; the FFN is what each person does back at their desk afterwards. The meeting is where information moves between people. The desk work is where it gets processed — and it is where the expertise lives, which is exactly why two-thirds of the parameters are there.

**"Why $4\times$?"** — the honest answer is that it was chosen in 2017 and has survived every attempt to improve it. It is worth being able to say that plainly rather than inventing a justification. What *is* known: narrower loses quality noticeably, wider gains little for the parameters spent, and the ratio has held from the original 512-wide transformer to models with $d$ above 12,000 — a 24× change in scale with no change in the ratio. Whether $4$ is optimal or merely a local optimum everyone copied is  open.

**The key–value memory interpretation** (Geva et al., 2021): write $W_1$'s rows as keys $k_i$ and $W_2$'s columns as values $v_i$. Then
$$\mathrm{FFN}(x) = \sum_{i=1}^{d_{\text{ff}}}\phi(k_i^\top x)\,v_i$$
Each hidden unit is a pattern detector that, when it fires, **adds a fixed vector to the residual stream**. Empirically these correspond to interpretable patterns (a specific $n$-gram, a topic, a syntactic construction), and the values shift the output distribution toward related tokens. **The FFN is an associative memory with $d_{\text{ff}}$ slots per layer.** This is the current best account of where factual knowledge lives in an LLM, and it is what model-editing methods (ROME, MEMIT) exploit.

#### The FFN as associative memory, decoded

$$\mathrm{FFN}(x) = \sum_{i=1}^{d_{\text{ff}}}\phi(k_i^\top x)\,v_i$$

This is the same formula as before, rewritten to make its meaning visible. The move is Chapter 1 §1.1.1's second reading of a matrix: **a matrix–vector product is a set of dot products against rows, and a matrix times a vector of coefficients is a weighted sum of columns.** So write $W_1$'s rows as $k_1,\dots,k_{d_{\text{ff}}}$ and $W_2$'s columns as $v_1,\dots,v_{d_{\text{ff}}}$, and the two-layer network becomes a sum over $d_{\text{ff}}$ terms.

Read one term, $\phi(k_i^\top x)\,v_i$:

| Piece | Meaning |
|---|---|
| $k_i^\top x$ | **How much does the input look like pattern $i$?** (a dot product, i.e. alignment) |
| $\phi(\cdot)$ | A threshold: near-zero unless the match is strong |
| $v_i$ | The **fixed vector this unit adds** when it fires |

▸ **So each hidden unit is a detector wired to a writer. If pattern $i$ is present, add vector $v_i$ to the residual stream; otherwise add nothing.** That is precisely the behaviour of an associative memory — content-addressed, not address-addressed. You do not ask for slot 4,712; you present a pattern and whatever matches responds.

**Notice how close this is to attention**, and why the paper naming the rows "keys" and the columns "values" was making a real claim rather than a pun:

| | Attention | FFN |
|---|---|---|
| Keys and values come from | **the current input sequence** | **the weights** |
| Number of slots | $T$ (varies per input) | $d_{\text{ff}}$ (fixed) |
| Contents | this context | everything learned in training |
| Normalization | softmax (weights sum to 1) | none — any number of units may fire |

▸ **Attention retrieves from the present; the FFN retrieves from the past.** Stated that way, "two-thirds of the parameters are in the FFN" stops being a piece of trivia: **the FFN is where the model keeps what it knows,** and attention is how it decides what to look up.

> **Analogy.** A doctor examining a patient. Attention is taking the history — gathering the facts of *this* case from *this* patient. The FFN is everything the doctor memorized in medical school: a very large set of "if you see this pattern, think of that condition" associations, identical for every patient, retrieved by pattern-match rather than by looking anything up alphabetically.

**Why this is more than an interpretation.** If a specific fact lives in a specific $(k_i, v_i)$ pair, you should be able to **edit** it — find where "the Eiffel Tower is in Paris" is stored and change the value vector so the model says Rome. Model-editing methods (**ROME**, "rank-one model editing," and **MEMIT**, "mass-editing memory in a transformer") do exactly this, with enough success to count as evidence for the account. ▸ **A theory of where knowledge lives that lets you surgically change a fact is doing more work than a metaphor.** The picture is incomplete — facts are distributed across units and layers rather than sitting in one slot, and edits have side effects — but it is the best account currently available.

**Modern variant — SwiGLU** (Ch. 6 §6.5): $W_3(\mathrm{SiLU}(W_1x)\odot W_2x)$ with $d_{\text{ff}}=\frac83 d$ to hold parameters constant. ~1% perplexity gain.

#### Reading SwiGLU

Three matrices instead of two, and one new operation. $\odot$ is the **elementwise product** (Ch. 0 §0.8): multiply matching entries and keep them separate, rather than summing them into a dot product.

- $W_1x$ goes through $\mathrm{SiLU}$ (the **sigmoid linear unit**, $\mathrm{SiLU}(z) = z\cdot\sigma(z)$) to become a **gate**.
- $W_2x$ is the **signal**, untouched.
- $\odot$ multiplies them entry by entry, so the gate can pass, attenuate, or block each channel independently.
- $W_3$ projects back down.

▸ **A gate lets the network make one part of the computation depend multiplicatively on another** — something a plain $\phi(Wx)$ cannot do, since its nonlinearity acts on each channel in isolation with no cross-channel control. This is the same mechanism as an LSTM's forget gate (Ch. 9), reappearing in a feed-forward setting.

**Why $\frac{8}{3}d$.** Three matrices of size $d\times d_{\text{ff}}$ instead of two means parameters would rise by 50% at fixed $d_{\text{ff}}$. Setting $d_{\text{ff}} = \frac83 d$ rather than $4d$ gives $3 \times \frac83 = 8$ — **exactly the $8d^2$ of the two-matrix version.** The comparison is therefore honest: same parameters, same FLOPs, roughly 1% better perplexity. ▸ **Always check whether an architectural "improvement" was compared at matched parameter count.** Most are not, and SwiGLU's careful $\frac83$ is a large part of why the result was believed.

### The residual stream

▸ Because every sub-layer *adds* to $x$, the residual stream is a **shared communication bus of dimension $d$** that all $2L$ sub-layers read from and write to. Consequences (developed fully in Ch. 32):

- Layers communicate by writing into *subspaces* of the stream; a layer can be read many layers later.
- The stream's dimension $d$ is a hard **bandwidth limit** — with far more features than $d$, the model must store them in superposition.
- The norm of the residual stream grows roughly monotonically with depth, since each layer adds.
- **Logit lens:** applying the final unembedding to an intermediate residual stream gives a readable (if noisy) distribution, showing the prediction refine layer by layer.

#### The residual stream, decoded

Start from the fact that generates everything else: **every sub-layer computes an update and *adds* it.** Nothing is ever overwritten. So for one position, the vector arriving at the end is

$$x_{\text{final}} = x_{\text{embed}} + \underbrace{\Delta_1 + \Delta_2 + \dots + \Delta_{2L}}_{\text{one contribution per sub-layer}}$$

▸ **A transformer's forward pass is one long sum.** That single observation is the foundation of modern interpretability, because a sum is *decomposable* — you can remove one term, or read one term, without disturbing the others. A product or a composition of nonlinear functions offers no such handle.

> **Analogy.** A shared whiteboard in a corridor, $d$ slots wide, that 192 people walk past in turn. Nobody erases; everyone adds. Anyone can read what earlier people wrote, and write something that only becomes relevant to someone twenty places down the line. **The whiteboard is the residual stream, and its width is the hard constraint** — with only $d$ slots, contributors must share, overlap, and encode compactly.

**"Layers communicate by writing into subspaces," decoded.** The stream is $d$-dimensional, so a sub-layer's update $\Delta$ is a vector in that space. If layer 3's update points along one set of directions and layer 4 reads along a different, near-perpendicular set, **the two do not interfere** — layer 4 can ignore what layer 3 wrote, and layer 20 can pick it up. In $d = 4096$ dimensions there are enormously many near-perpendicular directions available (Ch. 1 §1.1.5), which is what makes this workable. **The stream is a bus with many independent channels, and the channels are directions rather than slots.**

**"The dimension is a hard bandwidth limit," quantified.** A model wants to represent far more features than it has dimensions — plausibly hundreds of thousands of concepts in $d=4096$. Exactly orthogonal storage caps out at $d$ features. Near-orthogonal storage allows exponentially more, at the price of small interference between them. ▸ **That trade is called superposition, it is forced by the arithmetic rather than chosen, and Chapter 32 is largely about its consequences.** The reason a single neuron in a language model usually is not interpretable is right here: features are directions, and directions need not align with the coordinate axes.

**"The norm grows monotonically with depth," and why.** Each sub-layer adds a vector. If the additions were perfectly aligned, the norm would grow linearly in depth; if perfectly random, like $\sqrt{\text{depth}}$ (the same $\sqrt n$ as everywhere else in this book). Reality sits between. ▸ **The practical consequence is that later layers write into a stream that is already large, so their *relative* contribution shrinks unless they scale up their outputs — one reason deep transformers need careful initialization and normalization**, and why residual-scaling tricks like $1/\sqrt{2L}$ initialization of output projections show up in large-model training recipes.

**The logit lens, decoded.** "Unembedding" is the output projection: the matrix that turns a $d$-vector into a score per vocabulary token. Normally you apply it once, at the end. The logit lens applies it to the residual stream *partway through*, at layer 10 of 96, and reads off what the model would predict if it stopped there.

▸ **What you see is a prediction being refined rather than computed.** Early layers produce something generic; middle layers land on the right category; late layers pick the exact token. The picture is noisy — intermediate layers are not trained to be readable by the final unembedding, and results are much cleaner in some models than others — but the qualitative finding, that **the residual stream carries an interpretable running guess**, holds up and follows directly from the additive structure. If the forward pass were a composition rather than a sum, the intermediate values would be in no shared coordinate system and the lens would show nothing.

### Post-LN vs pre-LN
Covered in Ch. 7 §7.3. Summary: post-LN needs warmup and is fragile beyond ~12 layers without care; pre-LN is the default; DeepNorm/sandwich-LN recover post-LN's slight quality edge at depth.

---

## 11.6 Encoder, decoder, and encoder–decoder

**Encoder block:** bidirectional self-attention + FFN. Used for understanding (BERT, ViT, retrievers).

**Decoder block:** causal self-attention + FFN. Used for generation (GPT, LLaMA). Decoder-only is the dominant paradigm.

**Encoder–decoder:** the decoder additionally has a **cross-attention** sub-layer where $Q$ comes from the decoder and $K,V$ come from the encoder output:
```
x = x + CausalSelfAttn(LN(x))
x = x + CrossAttn(LN(x), enc_out, enc_out)
x = x + FFN(LN(x))
```
Used by T5, BART, Whisper, and most translation systems.

▸ **Why did decoder-only win for LLMs?** (A standard interview question.)
1. **Training efficiency:** every token is a prediction target.
2. **Simplicity:** one stack, one objective, trivially scalable.
3. **In-context learning** emerges naturally from a single sequence containing both instruction and content.
4. Encoder–decoder splits parameters into two pools; at fixed budget, one pool that does both is empirically better for open-ended generation.

Encoder–decoder remains better when there is a fixed, repeatedly-attended source (translation, speech, some multimodal settings), because the source is encoded once bidirectionally.

---

## 11.7 The complete accounting

### Parameters per layer (excluding biases and norms)

| Component | Parameters |
|---|---|
| $W_Q,W_K,W_V,W_O$ | $4d^2$ |
| FFN ($W_1,W_2$, $d_{\text{ff}}=4d$) | $8d^2$ |
| **Total per layer** | $\mathbf{12d^2}$ |

Plus $|V|d$ for embeddings (and again for the output head if untied).

▸ $$N_{\text{non-embed}} = 12Ld^2$$

**Check against GPT-3:** $L=96$, $d=12288$. $12\times96\times12288^2 = 1.74\times10^{11} = 174$B. Reported: 175B. ✓

### FLOPs

Per token, forward pass:
- Attention projections: $2\cdot4d^2 = 8d^2$
- Attention scores + weighted values: $2\cdot 2Td = 4Td$
- FFN: $2\cdot 8d^2 = 16d^2$

▸ $$\text{FLOPs/token/layer} \approx 24d^2 + 4Td$$

Total forward $\approx 2N$ FLOPs per token (where $N$ = parameter count), backward $\approx 4N$:

▸ $$\boxed{\ C \approx 6ND\ }$$

$C$ = total training FLOPs, $N$ = parameters, $D$ = training tokens. **Memorize this.** It underpins all of Chapter 15.

**When does attention dominate?** Attention's share is $\frac{4Td}{24d^2+4Td}$, which passes 50% at $T = 6d$. For $d=4096$: $T=24{,}576$. ▸ **Below ~25k context, a large transformer's cost is dominated by the FFN and projections, not by the $O(T^2)$ attention.** This surprises people and it is the correct answer to "isn't attention the bottleneck?"

### Memory
- Parameters: $N$ (× 2 bytes bf16)
- Gradients: $N$
- Adam states: $2N$ (fp32: $8N$ bytes)
- Activations: the dominant term; $O(BTLd)$ plus $O(BhT^2L)$ for materialized attention matrices — which FlashAttention eliminates (Ch. 12).

---

## 11.8 Mixture of Experts

### The one-line idea

Replace one big FFN with $E$ smaller ones and a router that sends each token to only $k$ of them, so parameter count grows without FLOPs growing.

### The analogy

A hospital. Rather than every patient seeing one enormous generalist who knows everything, a triage nurse routes each patient to 2 of 64 specialists. Total institutional knowledge is huge; the cost per patient is two consultations.

### The math

▸ $$y = \sum_{i\in\mathrm{TopK}(g(x))} g_i(x)\,E_i(x),\qquad g(x)=\mathrm{softmax}(W_gx)$$

**Total vs active parameters:** Mixtral 8×7B has 47B total, 13B active per token. That is the whole point: **quality tracks total parameters, cost tracks active parameters.**

### Load balancing

Left alone, the router collapses onto a few experts (rich-get-richer: a good expert gets more tokens, improves, gets more tokens). Fix with an auxiliary loss:

▸ $$\mathcal{L}_{\text{aux}} = \alpha E\sum_{i=1}^{E} f_i\,P_i$$

where $f_i$ = fraction of tokens routed to expert $i$ and $P_i$ = mean router probability for expert $i$. Minimized when both are uniform ($1/E$). Typically $\alpha=0.01$.

**Other mechanisms:** expert capacity limits with token dropping; **Expert Choice** routing (experts pick their top-$k$ tokens, guaranteeing perfect balance); auxiliary-loss-free load balancing via learned per-expert bias terms (DeepSeek-V3).

**Costs:** high memory (all experts must be resident), all-to-all communication in distributed training, training instability, and worse fine-tuning stability. **Benefit:** roughly 4–7× the effective capacity per unit of inference FLOPs.

---

## 11.9 What transformers cannot do

Worth knowing precisely, because it comes up:

- **A fixed-depth transformer is not Turing-complete** on a single forward pass. It's in a complexity class around $\mathsf{TC}^0$ (constant-depth threshold circuits) under standard assumptions — so it provably cannot solve certain problems (e.g. some state-tracking and composition tasks) in one pass at fixed depth, regardless of width.
- **Chain-of-thought changes this**: generating intermediate tokens gives the model an external serial scratchpad, lifting the effective complexity class. ▸ **This is the rigorous reason chain-of-thought works — it isn't "thinking," it's recurrence purchased with tokens.**
- **Length generalization** is poor: models trained on length $T$ often fail at $2T$, largely a positional-encoding problem (Ch. 12).
- **Exact copying and counting** are surprisingly hard without the right positional scheme.

---

## Did you know?

- **The paper that introduced the transformer is called "Attention Is All You Need," and the title was a deliberate provocation.** In 2017 the assumption was that sequence models *required* recurrence or convolution; attention was regarded as a helpful add-on bolted onto an RNN. The title asserts you can delete everything else. It worked: the paper is now among the most-cited in the history of computer science.

- **The transformer was built for machine translation, not for chatbots.** Its eight authors at Google were trying to speed up translation training, and the architecture's decisive advantage at the time was **parallelism** — an RNN must process tokens one after another, while attention processes them all at once, which suits a GPU far better. General-purpose language modelling was not the goal.

- **All eight authors of the transformer paper have since left Google**, and several founded AI companies of their own. The paper's unusual footnote lists contributions individually and states that the author order is random.

- **The $\sqrt{d_k}$ in scaled dot-product attention exists to stop the softmax from saturating.** A dot product of two $d_k$-dimensional random vectors has standard deviation $\sqrt{d_k}$, so without the division the logits grow with dimension, the softmax becomes nearly one-hot, and gradients vanish. It is a variance-control fix, and the paper explains it in a footnote.

- **The feed-forward network holds about two-thirds of a transformer's parameters**, despite attention getting essentially all the attention. For a standard block with $d_{\text{ff}} = 4d$, the FFN carries $8d^2$ parameters against attention's $4d^2$.

- **Attention costs scale with the *square* of sequence length.** Doubling context from 4,000 to 8,000 tokens does not double attention cost — it quadruples it. Chapters 12 and 17 are, to a large extent, a sustained engineering campaign against that one exponent.

- **Multi-head attention was originally motivated as an ensemble**, but heads turn out to specialize in interpretable ways — previous-token heads, syntactic heads, and the induction heads of Chapter 13 that underpin in-context learning. Some heads can be removed entirely with little loss; others are load-bearing.

- **The residual stream is a bandwidth-limited communication channel.** Every layer reads from and writes to the same $d$-dimensional vector, so $d$ is a hard ceiling on how much information can be in flight simultaneously — which is why Chapter 32's superposition arises, and why Johnson–Lindenstrauss (§1.1.5) is what makes it survivable.

- **Pre-LayerNorm versus post-LayerNorm was a  practical crisis.** The original transformer put normalization after the residual addition, which required a careful learning-rate warmup or training diverged. Moving the norm inside the residual branch made deep transformers trainable without warmup, and essentially every modern model uses pre-LN.

- **Mixture-of-Experts lets a model have far more parameters than it uses per token.** Only a couple of experts activate per token, so a model can carry hundreds of billions of parameters while spending the compute of a much smaller one — decoupling capacity from cost, which is otherwise nearly impossible.

- **A transformer has no built-in notion of order.** Self-attention is permutation-equivariant: shuffle the input tokens and the outputs shuffle identically. Everything the model knows about sequence order arrives through positional encodings (Chapter 12) — the architecture itself treats a sentence as a bag.

- **Transformers cannot reliably count or copy arbitrarily long sequences**, despite their power. A fixed number of layers means a fixed number of sequential computation steps, so problems needing unbounded sequential reasoning fall outside what a single forward pass can express — which is a large part of why chain-of-thought prompting helps.

---

## Check for Understanding

**A transformer alternates two operations — attention, which lets positions exchange information via a learned soft dictionary lookup, and a position-wise FFN, which is an associative memory holding two-thirds of the parameters — and both write into a shared residual stream whose dimension is a hard bandwidth limit on how much the model can represent at once.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What are queries, keys, and values**, in terms of a dictionary lookup? What makes the lookup "soft"?
2. **Why divide by $\sqrt{d_k}$?** What goes wrong at large $d_k$ if you don't?
3. **Why is attention permutation-equivariant**, and what does that force the architecture to add?
4. **What does a causal mask do**, and why is setting masked logits to $-\infty$ the natural way to do it?
5. **Why does multi-head attention beat one big head** of the same total width?
6. **Where do most of a transformer's parameters actually live**, and why does that surprise people?
7. **Why is attention $O(T^2)$**, and what does that mean concretely when you double the context?
8. **What is the residual stream, and why is its dimension a bandwidth limit?**
9. **Why did pre-LayerNorm replace post-LayerNorm?** What broke without it?
10. **How does a Mixture-of-Experts model have more parameters than it uses per token?**
11. **Why can't a transformer reliably count?** What does depth have to do with sequential reasoning?
12. **Why was "Attention Is All You Need" a provocative title in 2017?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 12 — Positional Information & Long Context](12-positional-encoding-long-context.md)
