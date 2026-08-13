# Chapter 6 — Neural Networks & Backpropagation

> **Prerequisites:** Ch. 1 (§1.2 matrix calculus).
> **Goal:** derive backprop from nothing, derive He/Xavier initialization from variance propagation, and be able to say *numerically* why a 50-layer plain network fails to train.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, $`A^\top`$, $`\odot`$, or $`\bar{y}`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. This chapter in particular leans hard on two things from Chapter 0: the **bar notation** ($`\bar y`$ means $`\partial\mathcal{L}/\partial y`$, §0.6) and the **four kinds of multiplication** (§0.8) — especially the difference between $`a^\top b`$ (a number), $`ab^\top`$ (a matrix), and $`a\odot b`$ (a vector).

### Symbols introduced in this chapter

Skim once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`h^{(\ell)}`$ | "h super ell" | The **activations** of layer $`\ell`$ — the list of numbers that layer hands forward |
| $`z^{(\ell)}`$ | "z super ell" | The **pre-activations** — layer $`\ell`$'s output *before* the nonlinearity |
| $`W^{(\ell)},\ b^{(\ell)}`$ | "W ell, b ell" | Layer $`\ell`$'s weight matrix and bias vector |
| $`\sigma(\cdot)`$ | "sigma of" | The **activation function** — in this chapter, *not* a standard deviation |
| $`\sigma'(z)`$ | "sigma prime of z" | Its slope: how much of a nudge the gate lets through |
| $`\delta^{(\ell)}`$ | "delta super ell" | The **error signal** $`\partial\mathcal{L}/\partial z^{(\ell)}`$ — the thing that flows backwards |
| $`\odot`$ | "elementwise times" (Hadamard) | Multiply matching entries and **keep them separate** |
| $`\bar y`$ | "y-bar" | Shorthand for $`\partial\mathcal{L}/\partial y`$; the gradient arriving at $`y`$ |
| $`\mathbb{1}[\,\cdot\,]`$ | "indicator" | 1 if the condition holds, 0 if not — an `if` written as a symbol |
| $`\mathrm{diag}(v)`$ | "diag of v" | A matrix with $`v`$ down the diagonal and zeros everywhere else |
| $`n_{\text{in}},\ n_{\text{out}}`$ | "fan-in, fan-out" | How many numbers flow into / out of a layer |
| $`\mathrm{Var}(X)`$ | "variance of X" | Average squared distance from the mean — the spread |
| $`\Phi(z),\ \phi(z)`$ | "big phi, little phi" | The Gaussian **CDF** (area to the left) and **PDF** (the bell curve itself) |
| $`\mathcal{U}[-a,a]`$ | "uniform on minus a to a" | Every value in that interval equally likely |
| $`\Omega(\cdot)`$ | "big omega" | "grows *at least* as fast as" — the lower-bound twin of $`\mathcal{O}`$ |
| $`L`$ vs $`\ell`$ | "big L, little ell" | Total number of layers vs. the index of one particular layer |

**Full forms for every abbreviation used in this chapter:**

| Short | Full form |
|---|---|
| AD | Automatic Differentiation |
| VJP | Vector–Jacobian Product |
| DAG | Directed Acyclic Graph |
| MLP | Multi-Layer Perceptron |
| ReLU | Rectified Linear Unit |
| PReLU | Parametric Rectified Linear Unit |
| ELU / SELU | Exponential Linear Unit / Scaled Exponential Linear Unit |
| GELU | Gaussian Error Linear Unit |
| SiLU | Sigmoid Linear Unit (the same function as Swish) |
| GLU / SwiGLU | Gated Linear Unit / Swish-Gated Linear Unit |
| MSE / CE | Mean Squared Error / Cross-Entropy |
| LR | Learning Rate |
| FLOP | FLoating-point OPeration |
| CDF / PDF | Cumulative Distribution Function / Probability Density Function |
| QR | (the QR decomposition — orthogonal matrix $`Q`$ times upper-triangular $`R`$) |
| DiT | Diffusion Transformer |
| AdaLN | Adaptive Layer Normalization |
| D3PM | Discrete Denoising Diffusion Probabilistic Model |

---

## 6.1 What a neural network is

### The one-line idea

A neural network is a composition of affine maps and elementwise nonlinearities, and the nonlinearity is the entire reason it isn't just one big matrix.

### The analogy

Think of a factory line. Each station takes a batch of parts (a vector), mixes and reweighs them (matrix multiply), then applies a simple pass/fail gate to each item (activation). The mixing lets stations combine information; the gate is what stops the whole line from collapsing into a single mixing step.

### The math

$$h^{(0)} = x,\qquad z^{(\ell)} = W^{(\ell)}h^{(\ell-1)} + b^{(\ell)},\qquad h^{(\ell)} = \sigma(z^{(\ell)}),\qquad \hat y = h^{(L)}$$

▸ **Without $`\sigma`$**, $`\hat y = W^{(L)}\cdots W^{(1)}x = \tilde Wx`$: an $`L`$-layer linear network computes exactly the same function class as a 1-layer one. (It optimizes differently — deep linear networks have interesting implicit bias — but it *expresses* nothing more.)

#### Reading the network equations in plain English

Four short equations, and they are the whole architecture. Take them one at a time.

- $`h^{(0)} = x`$ — **"the zeroth layer's output is just the input."** The superscript in parentheses is a *label*, not a power. $`h^{(3)}`$ means "the h belonging to layer 3," never "h cubed." The parentheses exist precisely to signal that.
- $`z^{(\ell)} = W^{(\ell)}h^{(\ell-1)} + b^{(\ell)}`$ — **"mix the previous layer's numbers, then shift them."** This is an **affine map**: a matrix multiply (which can rotate, stretch, and blend) plus an added offset $`b`$. Read aloud: *"z-ell equals W-ell times h-ell-minus-one, plus b-ell."*
- $`h^{(\ell)} = \sigma(z^{(\ell)})`$ — **"pass every entry through the same small gate function, independently."** $`\sigma`$ applied to a vector means applied to each entry separately: $`\sigma((3,-1)) = (\sigma(3), \sigma(-1))`$.
- $`\hat y = h^{(L)}`$ — **"the prediction is whatever the last layer produced."** The hat means "estimate," versus the true label $`y`$.

> **Analogy.** Each layer is a mixing desk followed by a row of gates. The matrix $`W`$ is the mixing desk: every output channel is a custom blend of every input channel, with the knob settings being the weights. The bias $`b`$ is a fixed offset added to each channel — a DC bias, in audio terms. Then $`\sigma`$ is a row of gates, one per channel, each deciding independently how much of its channel gets through. The desk lets channels *talk to each other*; the gates make the response *nonlinear*. Stack a hundred of these and you have a deep network.

**Now put numbers in, with the smallest network that shows anything.** Two inputs, two hidden units, one output, ReLU activation:

$$x = \begin{pmatrix}1\\2\end{pmatrix},\quad W^{(1)} = \begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix},\quad b^{(1)} = \begin{pmatrix}0\\0\end{pmatrix},\quad W^{(2)} = \begin{pmatrix}2 & 5\end{pmatrix},\quad b^{(2)} = 1$$

Forward pass, step by step:

| Step | Computation | Result |
|---|---|---|
| $`z^{(1)} = W^{(1)}x`$ | $`(1{\cdot}1 + 1{\cdot}2,\ \ 1{\cdot}1 - 1{\cdot}2)`$ | $`(3,\ -1)`$ |
| $`h^{(1)} = \mathrm{ReLU}(z^{(1)})`$ | $`(\max(0,3),\ \max(0,-1))`$ | $`(3,\ 0)`$ |
| $`z^{(2)} = W^{(2)}h^{(1)} + b^{(2)}`$ | $`2(3) + 5(0) + 1`$ | $`7`$ |

▸ **Look at what the ReLU did: it deleted the $`-1`$.** The second hidden unit contributed nothing. That single act of deletion is the difference between a neural network and a matrix.

**Why deleting the $`-1`$ matters — the collapse argument, concretely.** Rerun the same network with $`\sigma`$ removed. Now $`h^{(1)} = z^{(1)} = (3,-1)`$ and $`\hat y = 2(3) + 5(-1) + 1 = 2`$. But notice you could have skipped the middle entirely:

$$W^{(2)}W^{(1)} = \begin{pmatrix}2 & 5\end{pmatrix}\begin{pmatrix}1 & 1\\1 & -1\end{pmatrix} = \begin{pmatrix}7 & -3\end{pmatrix}, \qquad \begin{pmatrix}7 & -3\end{pmatrix}\begin{pmatrix}1\\2\end{pmatrix} + 1 = 7 - 6 + 1 = 2\ \checkmark$$

The two-layer linear network **is** the single row vector $`(7,\ -3)`$. Its "hidden layer" is a fiction you could compile away before ever seeing the data. That is what $`\tilde W`$ in the book's line means: the product of all the weight matrices, collapsed into one.

▸ **The nonlinearity is what makes the collapse illegal.** With ReLU in place you cannot pre-multiply the matrices, because *which entries get zeroed depends on the input*. For $`x = (1,2)`$ the second unit dies; for $`x = (2,1)`$ it lives ($`z^{(1)} = (3, 1)`$, nothing zeroed, $`\hat y = 6+5+1 = 12`$). The network is a *different* linear function in each region of input space, and the regions are carved out by the activations. **A ReLU network is a piecewise-linear function whose pieces it chooses for itself.**

> **Where this came from.** The **perceptron** — a single layer, a hard threshold instead of $`\sigma`$, and a learning rule — was built by **Frank Rosenblatt** at the Cornell Aeronautical Laboratory in 1958. The Mark I Perceptron was not a program but a physical machine: an array of 400 photocells wired to potentiometers whose settings were adjusted by electric motors, so that *training* was literally a set of knobs turning. Press coverage at the time, including in the *New York Times*, promised a machine that would soon walk, talk, see, and reproduce itself. In 1969, **Marvin Minsky and Seymour Papert** published *Perceptrons*, proving that a single-layer perceptron cannot compute XOR — a function of two bits. The usual telling is that this book single-handedly caused a decade-long funding winter for neural networks; historians of the field generally regard that as an oversimplification, since Minsky and Papert explicitly discussed multi-layer machines and interest had other reasons to wane. What is not disputed is that the obstacle they identified — that one layer is not enough, and nobody yet knew how to train more than one — stood for roughly seventeen years, until backpropagation was popularized in 1986. Rosenblatt did not see it resolved; he died in a boating accident in 1971, aged 43.

### Universal approximation, and why it's less impressive than it sounds

**Theorem (Cybenko '89, Hornik '91):** a one-hidden-layer network with a non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy, given enough width.

**Why it doesn't explain deep learning:**
- It's an *existence* result. It says nothing about whether SGD finds those weights.
- The required width can be exponential in the input dimension.
- It gives no reason to prefer depth.

**The depth results are more informative.** Telgarsky (2016): there are functions computable by a network of depth $`O(k^2)`$ and width $`O(1)`$ that require width $`\Omega(2^k)`$ at depth $`O(k)`$. Depth buys **exponential** expressivity for certain compositional functions. A ReLU network with $`L`$ layers of width $`d`$ can carve input space into $`O((d/n_{\text{in}})^{(L-1)n_{\text{in}}}d^{n_{\text{in}}})`$ linear regions — exponential in depth, polynomial in width.

▸ **The honest summary:** depth is efficient for functions that are *themselves* compositional — which real-world data (edges → textures → parts → objects; atoms → functional groups → scaffolds → molecules) tends to be. It's a match between architecture and the structure of reality, not a universal advantage.

#### What universal approximation actually claims — and what it quietly doesn't

Decode the theorem clause by clause, because every clause is load-bearing.

| Clause | What it means |
|---|---|
| "one hidden layer" | Exactly two weight matrices. $`x \to`$ hidden $`\to`$ output. As shallow as a network can be and still be one. |
| "non-polynomial activation" | $`\sigma`$ can be almost anything that bends — sigmoid, ReLU, tanh — **as long as it is not a polynomial.** Polynomials fail because a polynomial of a polynomial is still a polynomial of bounded degree, so stacking gains nothing. |
| "any continuous function" | No jumps. You may not approximate a step function perfectly, but you can get as close as you like everywhere except at the jump. |
| "on a compact set" | On a **closed, bounded region** — say, all inputs with every coordinate in $`[-1,1]`$. Not on all of $`\mathbb{R}^n`$. The theorem says nothing about what happens far outside your training data, which is exactly where deployed models fail. |
| "to arbitrary accuracy" | For any tolerance $`\epsilon > 0`$ you name, some network gets within $`\epsilon`$. |
| "given enough width" | And here is the catch. "Enough" is unbounded, and the theorem never says how much. |

> **Analogy.** Universal approximation is the statement *"any shape can be built out of enough Lego bricks."* True, useful to know, and almost entirely unhelpful when you want a specific model of a specific cathedral. It tells you the medium is expressive. It tells you nothing about how many bricks, how to find the arrangement, or whether the arrangement you find by trial and error will look like the cathedral or like a pile.

**The three failures of the theorem, stated plainly.**

1. **It is an existence proof.** It says a good set of weights *is out there*. Training is the problem of *finding* it with gradient descent, and the theorem is silent on that. Compare: "there exists a winning lottery ticket" versus "here is how to buy it."
2. **The width can be exponential in the input dimension.** For a $`d`$-dimensional input, the number of hidden units needed can scale like $`\epsilon^{-d}`$. Put a number on it: with $`d = 10`$ and $`\epsilon = 0.1`$, that is $`10^{10}`$ hidden units — a network with more parameters than any ever trained, to approximate one function on a ten-dimensional box.
3. **It gives no reason to prefer depth**, because it is a theorem *about* shallow networks. Every practical fact about deep learning is invisible to it.

**Reading the Telgarsky result, which is the informative one.** Recall from Chapter 0 that $`\mathcal{O}(\cdot)`$ means "grows no faster than" and $`\Omega(\cdot)`$ means "grows **at least** as fast as." So *"computable at depth $`O(k^2)`$ and width $`O(1)`$, but requires width $`\Omega(2^k)`$ at depth $`O(k)`$"* reads:

▸ **"There are functions a deep-and-thin network computes easily that a shallower network can only match by becoming exponentially fat."** Put $`k = 20`$: a deep network of constant width versus a shallower one needing width $`2^{20} \approx 10^6`$. That is not a constant-factor inconvenience; it is the difference between a laptop and an impossibility. **Depth is not a convenience — for some functions it is exponentially cheaper than width.**

**The linear-regions count, decoded.** A ReLU network is piecewise linear (we saw why above). The formula $`O((d/n_{\text{in}})^{(L-1)n_{\text{in}}}d^{n_{\text{in}}})`$ counts how many separate linear pieces it can carve input space into. Ignore the exact exponents and read the *shape* of the expression: **$`L`$ (depth) appears in the exponent; $`d`$ (width) appears in the base.** Adding a layer multiplies the number of regions; adding a neuron merely increases it polynomially. Concretely, with $`n_{\text{in}} = 2`$ and width $`d = 10`$: going from 3 layers to 4 raises the region count by roughly a factor of $`(d/n_{\text{in}})^{2} = 25`$, whereas adding one neuron to a layer changes it by a few percent. **Depth compounds; width accumulates.**

> **Where this came from.** **George Cybenko** proved the sigmoidal case in 1989, at the University of Illinois, using a functional-analysis argument (the Hahn–Banach theorem and properties of measures) rather than any constructive recipe — which is why the proof yields no algorithm. **Kurt Hornik**, an econometrician in Vienna, together with Maxwell Stinchcombe and Halbert White, proved a more general version essentially simultaneously, and Hornik's 1991 paper established the sharp condition: what matters is not that $`\sigma`$ is sigmoid-shaped but that it is **not a polynomial**. There is a persistent confusion worth clearing up: the **Kolmogorov–Arnold representation theorem** (Andrey Kolmogorov, 1957, answering a version of Hilbert's thirteenth problem) also says any continuous multivariate function can be written using only single-variable functions and addition — and it is *not* the universal approximation theorem. Kolmogorov's inner functions are wildly non-smooth and are not learned from data; the two results are often conflated in casual retellings. The  striking historical point is that a theorem the field cites constantly as its foundation was proved for a network architecture — one hidden layer — that essentially nobody uses.

#### Examples and non-examples: what depth actually buys

"Deep" is the field's name for itself, so it is worth being exact about what the depth is doing — because the honest answer is not "more layers means more power," it is "more **compositions of nonlinearities** means more power," and the two come apart in ways that are easy to demonstrate.

**✅  examples of depth buying something**

| Example | Why it qualifies |
|---|---|
| Telgarsky's construction: computable at depth $`O(k^2)`$ and width $`O(1)`$, but needing width $`\Omega(2^k)`$ at depth $`O(k)`$ | A **proved exponential separation.** At $`k=20`$ the shallow network needs width $`2^{20}\approx10^6`$ to match a constant-width deep one |
| Linear-region counting: at $`n_{\text{in}}=2`$, width $`10`$, one extra layer multiplies the region count by about $`(d/n_{\text{in}})^{n_{\text{in}}} = 25`$; one extra neuron changes it by a few percent | **Depth is in the exponent, width is in the base.** Depth compounds; width accumulates |
| A 50-layer ResNet beating a 50-layer plain network at identical parameter count | Depth was already present in the plain net and unusable. The residual path made the same depth trainable |
| Edges $`\to`$ textures $`\to`$ object parts $`\to`$ objects across layers of a vision model | Each layer's features are functions of the *previous layer's features*, which is only possible because a nonlinearity sits between them |

**❌ Near-misses — things that look like depth but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A 50-layer network with no activation functions: $`W_{50}\cdots W_2W_1x`$ | A product of matrices is a matrix. Fifty $`1024\times1024`$ layers ($`52`$M parameters) collapse to **one** $`1024\times1024`$ map ($`1`$M parameters' worth of behaviour) | A single linear layer wearing fifty hats |
| The same stack with one width-8 layer in the middle | Worse than collapse: $`\mathrm{rank}(W_{50}\cdots W_1) \le 8`$. The bottleneck caps the whole network's rank regardless of the other 49 layers | A rank-8 linear map. **This is what breaks when you narrow one layer in a linear stack** |
| 50 layers of $`f(z) = z^2`$ | A polynomial of a polynomial is a polynomial. The degree grows, but the *class* stays polynomial — which is exactly why the universal approximation theorem excludes polynomials by name | A high-degree polynomial regressor |
| "The universal approximation theorem proves deep networks work" | It is a theorem about networks with **one hidden layer**, and it is an existence proof. Width can scale like $`\epsilon^{-d}`$: at $`d=10`$, $`\epsilon=0.1`$, that is $`10^{10}`$ hidden units | An existence result about shallow networks, silent on depth and on training |
| "A deeper model has more parameters, so it's more expressive" | Depth and parameter count are separate dials. Halve every width and double the depth and you have fewer parameters and more compositions | Two axes being conflated |
| "More layers always makes optimization harder" | True for plain networks ($`\gamma^L`$, §6.3); false for residual ones, where the identity path contributes $`1^{L} = 1`$ exactly | A claim about the **architecture**, not about depth |

▸ **The boundary:** depth buys expressive power only through **composition of nonlinearities**. The dial that matters is the number of nonlinearities in the chain, not the number of weight matrices — and the two coincide only because every layer you write happens to come with an activation attached. Delete the activations and the depth evaporates; keep them and it compounds.

> **Common misconception.** *"Deeper networks are more powerful because they have more layers."* Only if the layers are separated by nonlinearities. This is worth testing rather than believing: take any trained multi-layer network, replace every activation with the identity, and the resulting function is *exactly* representable by a single matrix — you can compute that matrix by multiplying the weights together and check that it produces identical outputs. **The belief is tempting because the correlation is airtight in practice**: every deep network anyone has ever trained has activations, deeper ones really do work better, and there is no counterexample in the wild to break the association. The credit gets assigned to the visible, countable thing (layers) rather than to the interaction between it and the invisible thing (the kinks).

> **Common misconception.** *"Universal approximation explains why neural networks work."* It explains that they *could*. Three gaps: it is an existence proof with no route to the weights (compare "a winning lottery ticket exists" with "here is how to buy it"); the width it needs can be exponential in the input dimension; and it is a theorem about the one-hidden-layer architecture, so every fact about depth — the thing the entire field is named after — is invisible to it. **The belief is tempting because the theorem is stated so early in every course and sounds so complete** — "can approximate any continuous function" leaves nothing obvious to ask for. The question it does not answer is the only question that matters: not what is representable, but what is **findable by gradient descent from a random start, in a reasonable number of steps, from a reasonable amount of data.**

---

## 6.2 Backpropagation, derived properly

### The one-line idea

Compute the derivative of the loss with respect to every parameter by applying the chain rule once, backwards, reusing every intermediate result.

### The analogy

Blame assignment in an organization. The CEO knows the company lost a million pounds. Rather than each employee independently tracing their impact on the bottom line (forward mode: expensive, $`p`$ separate traces), the blame flows *down* the hierarchy: each manager receives their share of blame and splits it among reports in proportion to their contribution. One pass, everyone gets their number.

### Setup and the key quantity

Define the **error signal** at layer $`\ell`$:

▸ $$\delta^{(\ell)} := \frac{\partial\mathcal{L}}{\partial z^{(\ell)}} \in \mathbb{R}^{n_\ell}$$

This is the thing that propagates. Once you have $`\delta^{(\ell)}`$, the parameter gradients are immediate.

#### What the error signal $`\delta^{(\ell)}`$ actually is

Read the definition aloud: *"delta-super-ell is defined as the partial derivative of the loss with respect to z-super-ell, and it lives in R-to-the-n-ell."*

Three pieces:

- **$`:=`$ means "is defined as"** (Chapter 0 §0.11). This is not a fact to verify; it is a name being given to a quantity.
- **$`\partial\mathcal{L}/\partial z^{(\ell)}`$** answers: *"if I nudged this layer's pre-activation up by a hair, how much would the final loss move?"* One number per unit in the layer.
- **$`\in \mathbb{R}^{n_\ell}`$** is the shape declaration: a list of $`n_\ell`$ numbers, one per neuron in layer $`\ell`$. Same shape as $`z^{(\ell)}`$ itself, per the shape rule from Chapter 0 §0.7.

> **Analogy.** $`\delta^{(\ell)}_j`$ is **how much blame unit $`j`$ of layer $`\ell`$ is carrying.** Positive means "you pushed the answer too high — come down." Negative means "you were too timid." Zero means "you are not responsible for anything that happened, do not change." The whole of backpropagation is the bookkeeping of passing blame from the output backwards, layer by layer, and the fact that this can be done in one sweep instead of one sweep *per parameter* is the reason training a billion-parameter model is possible at all.

▸ **Why $`\delta`$ is defined on $`z`$ and not on $`h`$ or on $`W`$.** $`z^{(\ell)}`$ sits at the exact pinch point of the computation: everything before it (the weights, the bias, the previous layer) feeds *into* it, and everything after it (the activation, all later layers, the loss) flows *out* of it. Once you know how much the loss cares about $`z^{(\ell)}`$, the weight gradient and the bias gradient both fall out in one line each, and the gradient for the layer below falls out in one more. **Choosing the right intermediate quantity to propagate is the entire design of the algorithm.** Pick $`h`$ instead and you carry an extra $`\sigma'`$ around forever; pick $`W`$ and you have nothing to hand backwards.

### Derivation, index notation

**Base case** (output layer). For softmax + cross-entropy (Ch. 1 §1.3.4):
$$\delta^{(L)}_j = p_j - \mathbb{1}[j=y]$$

For MSE with linear output: $`\delta^{(L)} = 2(\hat y - y)`$.

**Recursion.** The loss depends on $`z^{(\ell)}`$ only through $`h^{(\ell)} = \sigma(z^{(\ell)})`$, which feeds $`z^{(\ell+1)} = W^{(\ell+1)}h^{(\ell)}+b^{(\ell+1)}`$. Chain rule:

$$
\delta^{(\ell)}_j = \frac{\partial\mathcal{L}}{\partial z^{(\ell)}_j} = \sum_k \frac{\partial\mathcal{L}}{\partial z^{(\ell+1)}_k}\cdot\frac{\partial z^{(\ell+1)}_k}{\partial h^{(\ell)}_j}\cdot\frac{\partial h^{(\ell)}_j}{\partial z^{(\ell)}_j}
= \sum_k \delta^{(\ell+1)}_k W^{(\ell+1)}_{kj}\,\sigma'(z^{(\ell)}_j)
$$

▸ In matrix form:
$$\boxed{\ \delta^{(\ell)} = \left(W^{(\ell+1)\top}\delta^{(\ell+1)}\right)\odot\sigma'(z^{(\ell)})\ }$$

#### Reading the backprop recursion in plain English

This one boxed line is the chapter. Say it out loud: *"delta-ell equals W-ell-plus-one transposed, times delta-ell-plus-one, elementwise-times sigma-prime of z-ell."*

In English, and in the order the computation happens:

1. **$`\delta^{(\ell+1)}`$** — the blame arriving from the layer above.
2. **$`W^{(\ell+1)\top}\delta^{(\ell+1)}`$** — *"spread that blame back down the wires that carried the signal up."* The transpose is not algebraic decoration: forward, $`W`$ carries information from layer $`\ell`$ to layer $`\ell+1`$; backward, you need the *same wires traversed the other way*, and transposing a matrix is exactly "reverse the direction of every arrow" (Chapter 1 §1.2.2).
3. **$`\odot\ \sigma'(z^{(\ell)})`$** — *"and then dim each unit's share of the blame by how much its own gate was open."* A unit whose gate was shut ($`\sigma' = 0`$) receives nothing, because nudging it would have changed nothing.

**The two-hop chain rule, spelled out.** The long index-notation formula above is just the chain rule taken through two links, once for each arrow in $`z^{(\ell)} \to h^{(\ell)} \to z^{(\ell+1)}`$:

| Factor | Reads as |
|---|---|
| $`\partial\mathcal{L}/\partial z^{(\ell+1)}_k = \delta^{(\ell+1)}_k`$ | "how much the loss cares about unit $`k`$ upstairs" |
| $`\partial z^{(\ell+1)}_k/\partial h^{(\ell)}_j = W^{(\ell+1)}_{kj}`$ | "how strongly unit $`j`$ down here feeds unit $`k`$ up there" |
| $`\partial h^{(\ell)}_j/\partial z^{(\ell)}_j = \sigma'(z^{(\ell)}_j)`$ | "how much of a nudge unit $`j`$'s own gate passes through" |
| $`\sum_k`$ | "add up over every upstairs unit that $`j`$ feeds" |

The $`\sum_k`$ appears because unit $`j`$ sends its output to *every* unit in the next layer, so its total responsibility is the sum of its responsibilities along each path. **Multiple paths add.** That single fact is why residual connections work (§6.2, VJP table) and why recurrent networks accumulate gradients over time.

> **Analogy.** A leaking roof. Water arriving in the ground-floor kitchen ($`\delta^{(L)}`$, the loss) came through the first-floor ceiling, which was fed by several separate cracks upstairs. To apportion blame among the cracks you (a) route the water back up through the pipes that carried it — that is $`W^\top`$ — and (b) discount any crack that was sealed at the time, because a sealed crack cannot have contributed no matter how big it looks — that is $`\odot\,\sigma'`$. Sum over all the paths water could have taken, and each crack gets its share in one pass through the building.

**Now the numbers.** Continue the tiny network from §6.1: $`x = (1,2)`$, $`z^{(1)} = (3,-1)`$, $`h^{(1)} = (3,0)`$, $`\hat y = 7`$. Say the true label is $`y = 5`$ and the loss is squared error, so (using the book's convention above) $`\delta^{(2)} = 2(\hat y - y) = 2(7-5) = 4`$.

| Step | Computation | Result |
|---|---|---|
| Transpose-multiply | $`W^{(2)\top}\delta^{(2)} = \begin{pmatrix}2\\5\end{pmatrix}\cdot 4`$ | $`(8,\ 20)`$ |
| Gate mask | $`\sigma'(z^{(1)}) = \mathbb{1}[z^{(1)}>0] = \mathbb{1}[(3,-1)>0]`$ | $`(1,\ 0)`$ |
| Elementwise product | $`(8,20)\odot(1,0)`$ | $`\delta^{(1)} = (8,\ 0)`$ |

▸ **The second unit receives a gradient of exactly zero.** Its pre-activation was $`-1`$, its ReLU was shut, it contributed nothing to the prediction — so it receives no blame and its incoming weights do not move on this example. This is the mechanism, in miniature, behind both **sparsity** (most units are off for any given input, which is fine and useful) and the **dying ReLU** problem (§6.5) (a unit that is off for *every* input never moves again).

**Sanity-check by hand, which is worth doing once in your life.** The network computes $`\hat y = 2\max(0, x_1{+}x_2) + 5\max(0,x_1{-}x_2) + 1`$. At $`x = (1,2)`$ the first branch is active and the second is not, so locally $`\hat y = 2(x_1 + x_2) + 1`$ and $`\partial\hat y/\partial z^{(1)}_1 = 2`$. With $`\partial\mathcal{L}/\partial\hat y = 4`$, the chain rule gives $`4 \times 2 = 8`$ — matching $`\delta^{(1)}_1 = 8`$ exactly. ✓

**Parameter gradients.** Since $`z^{(\ell)}_j = \sum_i W^{(\ell)}_{ji}h^{(\ell-1)}_i + b^{(\ell)}_j`$:

$$\frac{\partial\mathcal{L}}{\partial W^{(\ell)}_{ji}} = \delta^{(\ell)}_j h^{(\ell-1)}_i,\qquad \frac{\partial\mathcal{L}}{\partial b^{(\ell)}_j} = \delta^{(\ell)}_j$$

▸ $$\boxed{\ \nabla_{W^{(\ell)}}\mathcal{L} = \delta^{(\ell)}h^{(\ell-1)\top},\qquad \nabla_{b^{(\ell)}}\mathcal{L}=\delta^{(\ell)}\ }$$

That's the entirety of backpropagation. Three boxed equations.

#### Why the weight gradient is an outer product

$`\nabla_{W^{(\ell)}}\mathcal{L} = \delta^{(\ell)}h^{(\ell-1)\top}`$ is an **outer product** (Chapter 0 §0.8): a column vector times a row vector, producing a whole matrix. Entry by entry it says

$$\frac{\partial\mathcal{L}}{\partial W^{(\ell)}_{ji}} = \underbrace{\delta^{(\ell)}_j}_{\text{how wrong output } j \text{ was}} \times \underbrace{h^{(\ell-1)}_i}_{\text{how active input } i \text{ was}}$$

▸ **Blame is assigned in proportion to participation.** If input $`i`$ was silent, that connection did nothing, so it gets no correction — regardless of how wrong the output was. If input $`i`$ was loud *and* output $`j`$ was badly wrong, that connection gets a large correction. This is the entire learning rule of a neural network, and you can state it in a sentence with no symbols in it.

**Check the shapes, always.** $`\delta^{(\ell)}`$ is $`n_\ell \times 1`$; $`h^{(\ell-1)\top}`$ is $`1 \times n_{\ell-1}`$; the product is $`n_\ell \times n_{\ell-1}`$ — exactly the shape of $`W^{(\ell)}`$. ✓ (If it weren't, you'd have a bug, and this check finds it in two seconds.)

**And with numbers,** from the running example: $`\delta^{(1)} = (8,0)`$, $`h^{(0)} = x = (1,2)`$, so

$$\nabla_{W^{(1)}}\mathcal{L} = \begin{pmatrix}8\\0\end{pmatrix}\begin{pmatrix}1 & 2\end{pmatrix} = \begin{pmatrix}8 & 16\\ 0 & 0\end{pmatrix}$$

The bottom row is all zeros — the dead unit's weights do not move. The top row is $`(8, 16)`$: the second weight gets twice the update of the first, because its input was twice as large. **Twice the participation, twice the blame.**

**Why the bias gradient is just $`\delta`$.** The bias adds directly to $`z`$ with a coefficient of exactly 1 ($`\partial z_j/\partial b_j = 1`$), so the sensitivity passes through unchanged. Equivalently: a bias is a weight attached to an input that is permanently equal to 1, and the outer-product rule with $`h_i = 1`$ gives $`\delta_j \times 1 = \delta_j`$. Same rule, no exception.

> **Where this came from.** Chapter 1 tells the story of backpropagation's four independent inventions (Linnainmaa 1970, Werbos 1974, and the 1986 Rumelhart–Hinton–Williams paper that made it famous). Worth adding here is the **pre-history in aerospace**, which most machine-learning accounts skip entirely: the same reverse accumulation of derivatives through a chain of stages was developed in optimal-control theory before neural networks were a going concern. **Henry J. Kelley** published a gradient method for optimal flight paths in 1960, **Arthur Bryson** developed multistage gradient methods around the same period, and **Stuart Dreyfus** gave a clean chain-rule derivation in 1962. Their "layers" were successive time-steps of a rocket trajectory rather than layers of a network, and their "loss" was fuel or miss-distance — but the recursion is the same recursion. **The reason it kept being reinvented is that it is not really an algorithm about learning; it is an algorithm about the chain rule, and the chain rule is everywhere.**

### The batched version

With a batch of $`B`$ examples, $`H^{(\ell)}\in\mathbb{R}^{n_\ell\times B}`$, $`\Delta^{(\ell)}\in\mathbb{R}^{n_\ell\times B}`$:

$$\Delta^{(\ell)} = \left(W^{(\ell+1)\top}\Delta^{(\ell+1)}\right)\odot\sigma'(Z^{(\ell)}),\qquad \nabla_{W^{(\ell)}}\mathcal{L} = \frac1B\Delta^{(\ell)}H^{(\ell-1)\top}$$

The outer product becomes a matmul, summing over the batch. Note the $`1/B`$: **your gradient is a mean, not a sum**, which is why loss scales are batch-size-independent but gradient *noise* isn't.

#### The batched version, decoded

Nothing new happens here — the mathematics is identical, and only the bookkeeping changes. But the bookkeeping is where the speed comes from, so it deserves reading properly.

- **Capital letters mean "a whole batch's worth."** $`H^{(\ell)} \in \mathbb{R}^{n_\ell \times B}`$ is a grid: **one column per example**, each column being that example's activation vector. $`\Delta^{(\ell)}`$ is the same for the error signals. Lower-case $`h`$ and $`\delta`$ were single columns; capital $`H`$ and $`\Delta`$ are $`B`$ of them side by side.
- **The recursion line is unchanged**, character for character, except the letters got taller. That is the point: the same rule applied to $`B`$ examples at once is one big matrix multiply instead of $`B`$ small ones, and a GPU is roughly two orders of magnitude more efficient at the former.
- **$`\Delta^{(\ell)}H^{(\ell-1)\top}`$ is a matmul that has a sum hiding inside it.** For a single example the weight gradient was an outer product $`\delta h^\top`$. For $`B`$ examples you want the sum of $`B`$ such outer products — and $`(n_\ell \times B)(B \times n_{\ell-1})`$ *is* that sum, because matrix multiplication sums over the shared inner index. The batch dimension is contracted away, which is exactly what you want: the result is one $`n_\ell \times n_{\ell-1}`$ gradient, not $`B`$ of them.

▸ **The $`1/B`$ is doing more work than it looks like.** With a mean rather than a sum, the *magnitude* of the gradient is roughly independent of batch size, so a learning rate tuned at $`B = 32`$ is in the right ballpark at $`B = 256`$. What is **not** independent is the *noise*: averaging $`B`$ independent estimates cuts their standard deviation by $`\sqrt B`$ (the same $`1/\sqrt n`$ law as everywhere else in statistics). Put numbers on it: going from $`B = 32`$ to $`B = 512`$ is a 16× increase in compute per step, and buys exactly $`\sqrt{16} = 4\times`$ less gradient noise. **Doubling the batch buys only $`\sqrt2`$ less noise — which is the whole reason large-batch training has diminishing returns**, and the origin of the "square-root scaling" rule of thumb for learning rates (Ch. 4).

### Computational cost

Forward through layer $`\ell`$: $`O(n_\ell n_{\ell-1}B)`$ FLOPs (× 2 for multiply-add).
Backward: **two** matmuls of the same size ($`\delta`$ propagation and weight gradient).

▸ **Backward ≈ 2× forward.** Total training step ≈ 3× a forward pass. This is where the ubiquitous $`C \approx 6ND`$ FLOP formula comes from (Ch. 21): 2 FLOPs per parameter per token forward, 4 backward, 6 total.

**Memory** is the real constraint: you must store every $`h^{(\ell)}`$ (or $`z^{(\ell)}`$) from the forward pass to compute the backward pass. Activation memory $`\approx B\sum_\ell n_\ell`$ — which for a transformer at long sequence length dwarfs the parameters. Gradient checkpointing (Ch. 21) trades recompute for memory.

#### Where "backward ≈ 2× forward" and the $`6ND`$ rule come from

**The counting argument, in full.** A single weight $`W_{ji}`$ participates in exactly three multiply-accumulate operations per example:

| Pass | Operation | Cost |
|---|---|---|
| Forward | $`z_j \mathrel{+}= W_{ji}h_i`$ | 1 multiply-add |
| Backward, signal | $`\delta^{(\ell-1)}_i \mathrel{+}= W_{ji}\delta^{(\ell)}_j`$ | 1 multiply-add |
| Backward, weight | $`\nabla W_{ji} \mathrel{+}= \delta^{(\ell)}_j h^{(\ell-1)}_i`$ | 1 multiply-add |

▸ **One forward, two backward — hence backward $`\approx 2\times`$ forward, and a full training step $`\approx 3\times`$ a forward pass.** The backward pass costs double because it has *two jobs*: compute the gradient for this layer's weights, and hand a gradient to the layer below. Inference only ever does the first row of that table, which is why serving a model is roughly three times cheaper per token than training it.

**Now $`C \approx 6ND`$.** Read the letters: $`C`$ is total training compute in FLOPs (FLoating-point OPerations), $`N`$ is the number of parameters, $`D`$ is the number of training tokens. Each multiply-add is counted as **2 FLOPs** (one multiply, one add), so the table above gives $`2 + 2 + 2 = 6`$ FLOPs per parameter per token. Multiply by $`N`$ parameters and $`D`$ tokens:

$$C \approx 6 \times N \times D$$

**Put real numbers in.** A 7-billion-parameter model trained on 2 trillion tokens:

$$C \approx 6 \times (7\times10^9) \times (2\times10^{12}) = 8.4\times10^{22}\ \text{FLOPs}$$

A modern accelerator delivers on the order of $`10^{15}`$ FLOP/s on paper, and real training runs achieve roughly 40% of that, so call it $`4\times10^{14}`$ FLOP/s in practice. Then $`8.4\times10^{22}/4\times10^{14} \approx 2.1\times10^8`$ seconds — about **6.7 years on a single GPU, or roughly 2.4 days on 1024 of them.** That arithmetic, done on the back of an envelope before anything is launched, is how training runs are actually budgeted.

**Why memory, not compute, is what actually stops you.** The parameters of a model are a fixed cost. The *activations* are not: you must keep every $`h^{(\ell)}`$ alive from the moment it is computed in the forward pass until the backward pass consumes it, because the weight gradient $`\delta^{(\ell)}h^{(\ell-1)\top}`$ needs the input that produced it. Activation memory scales as $`B\sum_\ell n_\ell`$ — **linear in batch size and linear in depth** — so it grows in exactly the two directions you want to push.

> **Analogy.** Backpropagation is a hiker who must retrace their steps and can only do so by dropping a breadcrumb at every stop. Compute is the walking; memory is the bread. **Gradient checkpointing** is the strategy of dropping breadcrumbs only every tenth stop and re-walking the short stretches between them on the way back: roughly $`\sqrt{L}`$ times less bread for about one extra forward pass of walking. It is one of the very few places in systems engineering where a clean $`\sqrt{\cdot}`$ trade-off is available, and it is why models far larger than a GPU's memory can be trained on it at all.

### Reverse-mode AD in general

Backprop is reverse-mode automatic differentiation applied to a layered graph. The general algorithm:
1. Build a DAG of primitive operations during the forward pass (the "tape").
2. Each primitive registers a **VJP rule**: given $`\bar y`$, return $`\bar x = (\partial y/\partial x)^\top\bar y`$.
3. Traverse the DAG in reverse topological order, accumulating $`\bar{\cdot}`$ at each node (**summing** contributions when a node has multiple consumers — this is why residual connections add gradients from both paths).

Some VJP rules to have memorized:

| Forward | VJP |
|---|---|
| $`y=Wx`$ | $`\bar W = \bar yx^\top`$, $`\bar x = W^\top\bar y`$ |
| $`y=x_1+x_2`$ | $`\bar x_1=\bar x_2=\bar y`$ (gradient **copies**) |
| $`y=x_1\odot x_2`$ | $`\bar x_1 = \bar y\odot x_2`$ |
| $`y=\sigma(x)`$ | $`\bar x = \bar y\odot\sigma'(x)`$ |
| $`y=\mathrm{softmax}(x)`$ | $`\bar x = y\odot(\bar y - \langle\bar y,y\rangle)`$ |
| $`y = x/\|x\|`$ | $`\bar x = (\bar y - y\langle y,\bar y\rangle)/\|x\|`$ |

Note the addition rule: **a residual connection is a gradient highway** because $`+`$ passes $`\bar y`$ through unchanged. That's the whole trick of ResNets (Ch. 8).

#### The VJP table, decoded

First, the vocabulary, because three abbreviations arrive at once:

- **AD — Automatic Differentiation.** Not symbolic differentiation (which manipulates formulas and blows up in size) and not numerical differentiation (which perturbs inputs and loses precision). AD applies the chain rule to a *program*, one primitive operation at a time, and gets exact derivatives at roughly the cost of the program itself.
- **DAG — Directed Acyclic Graph.** The record of what your forward pass actually did: nodes are values, edges are operations, "acyclic" means nothing feeds itself so there is a well-defined order to walk it. Frameworks call it **the tape**, because it is literally a recording.
- **VJP — Vector–Jacobian Product.** The one rule each operation must supply: *"given the gradient arriving at my output, produce the gradient at my input."* The name is literal — $`\bar x = (\partial y/\partial x)^\top \bar y`$ is a vector $`\bar y`$ multiplied by a Jacobian, and the whole point is that you never build the Jacobian, you only ever compute the product.

▸ **This is the great reframing:** you do not need a differentiation algorithm per architecture. You need a VJP per *operation* — a few hundred of them, written once — and then any program built from those operations is differentiable for free. Every deep learning framework is, at its core, that table plus a loop that walks the tape backwards.

Now the table itself, row by row, said in words:

| Forward | What its VJP is really saying |
|---|---|
| $`y = Wx`$ | "Weights get blame proportional to participation; inputs get blame routed back through the same wires" (§6.2) |
| $`y = x_1 + x_2`$ | **Addition copies the gradient.** Both branches receive the *full* incoming gradient, undiminished |
| $`y = x_1\odot x_2`$ | **Multiplication swaps.** Each factor's gradient is the incoming gradient times *the other* factor — if $`x_2`$ was zero, $`x_1`$ gets nothing |
| $`y = \sigma(x)`$ | "Scale the incoming gradient by the local slope of the gate" |
| $`y = \mathrm{softmax}(x)`$ | "Subtract off the weighted average, then scale by the probabilities" — the subtraction is what enforces $`\sum_j \bar x_j = 0`$ |
| $`y = x/\|x\|`$ | "Keep only the part of the gradient perpendicular to $`y`$" — changing a unit vector's length is not a legal move, so that component is deleted |

**Why "$`+`$ copies" is the most consequential line in the table.** Consider a residual block, $`y = x + F(x)`$. The addition rule says the gradient at $`x`$ is the sum of two contributions: one that went through $`F`$ (and got multiplied by whatever $`F`$'s Jacobian is, possibly shrinking it) and **one that came straight through the $`+`$, completely untouched.** So even if $`F`$ annihilates the gradient entirely, $`x`$ still receives $`\bar y`$ at full strength.

▸ **Put a number on it.** In a 50-layer plain network with per-layer factor $`\gamma = 0.8`$, the gradient reaching layer 1 is $`0.8^{50} = 1.4\times10^{-5}`$ of what left the loss. In a 50-layer *residual* network, the identity path contributes a factor of exactly $`1^{50} = 1`$. The gradient cannot vanish, because there is a route from the loss to every layer that consists entirely of additions. **That is the whole of ResNet in one sentence, and it is a fact about the derivative of `+`.**

> **Analogy for the tape.** Reverse-mode AD is a delivery driver reconstructing their route. On the way out, they note every turn on a pad (the forward pass builds the DAG). To work out how a delay at the depot propagated to each stop, they read the pad **backwards**, and at each junction split the delay among the roads that fed it. Two facts make this work: the pad must be kept (memory cost — this is exactly the activation memory above), and each junction needs only a local rule for splitting (the VJP), never a map of the whole city.

**The "summing contributions when a node has multiple consumers" clause** is the same fact as the $`\sum_k`$ in the recursion. If a value is used in three places, three separate gradients come back to it and they **add**. Frameworks implement this by zero-initializing a gradient buffer per node and accumulating into it — which, incidentally, is why you have to call something like `zero_grad()` between steps: the accumulation is the mechanism, and it does not know where one step ends and the next begins.

---

## 6.3 Vanishing and exploding gradients, quantified

Unroll the recursion:

$$\delta^{(1)} = \left(\prod_{\ell=2}^{L}W^{(\ell)\top}D^{(\ell)}\right)\delta^{(L)},\qquad D^{(\ell)} = \mathrm{diag}(\sigma'(z^{(\ell)}))$$

▸ The gradient at layer 1 is a product of $`L-1`$ matrices. Products of matrices behave like products of scalars in log-space: if each contributes an average multiplicative factor $`\gamma`$, the total is $`\gamma^{L}`$.

- $`\gamma<1`$ ⇒ **vanishing**: $`\gamma=0.8`$, $`L=50`$ ⇒ $`0.8^{50}=1.4\times10^{-5}`$. Early layers receive essentially no signal.
- $`\gamma>1`$ ⇒ **exploding**: $`\gamma=1.2`$, $`L=50`$ ⇒ $`1.2^{50}=9100`$. NaNs.
- $`\gamma=1`$ is a knife-edge. **Everything about initialization and normalization is the project of engineering $`\gamma\approx1`$.**

**The sigmoid disaster.** $`\sigma(z)=1/(1+e^{-z})`$ has $`\sigma'(z)=\sigma(1-\sigma)\le 1/4`$. So even with perfectly scaled weights, $`\gamma \le 0.25`$ from the activation alone:
$$0.25^{10} = 9.5\times10^{-7}$$
▸ **A 10-layer sigmoid network cannot be trained by gradient descent.** This single number is why deep learning didn't work before ReLU, and why the 2006–2012 era needed layerwise pretraining.

#### Unpacking the product that ruins everything

**First the notation.** $`D^{(\ell)} = \mathrm{diag}(\sigma'(z^{(\ell)}))`$ is a square matrix with the activation slopes down its diagonal and zeros elsewhere. Multiplying by a diagonal matrix *is* elementwise multiplication — $`\mathrm{diag}(v)\,u = v\odot u`$ — so this is the same $`\odot\sigma'`$ from the recursion, rewritten as a matrix so that the whole backward pass becomes one clean product of matrices. The $`\prod_{\ell=2}^{L}`$ is a `for` loop that multiplies rather than adds (Chapter 0 §0.3).

▸ **Read the unrolled formula as a single sentence:** *"the gradient that reaches layer 1 is the gradient at the loss, pushed through $`L-1`$ matrix multiplications in a row."* Nothing else happens to it. Its fate is entirely determined by what a chain of $`L-1`$ matrices does to a vector.

**Why products of matrices behave like products of numbers.** From Chapter 1 §1.1.4, the most a matrix can stretch any vector is its largest singular value, $`\|A\|_2`$. Chain $`L`$ of them and the total stretch is at most the product $`\prod_\ell \|A_\ell\|_2`$. If each contributes a typical factor $`\gamma`$, the whole chain contributes about $`\gamma^{L}`$. **Depth turns a per-layer factor into an exponent, and exponents are unforgiving.**

> **Analogy.** A rumour passed down a line of 50 people. If each person exaggerates by 20% ($`\gamma = 1.2`$), the story arrives 9,100 times bigger than it started. If each person softens it by 20% ($`\gamma = 0.8`$), it arrives at $`1.4\times10^{-5}`$ of its original size — which is to say, it arrives as silence. **Nobody in the chain did anything unreasonable.** No single person distorted the story much at all. The catastrophe is entirely in the length of the line. This is why "each layer is only slightly off" is not a defence.

**Feel the knife-edge with numbers.** Here is $`\gamma^{50}`$ for values of $`\gamma`$ that all look equally innocuous:

| $`\gamma`$ | $`\gamma^{50}`$ | Verdict |
|---|---|---|
| $`0.8`$ | $`1.4\times10^{-5}`$ | Gradient is gone. Layer 1 never learns |
| $`0.95`$ | $`0.077`$ | 13× attenuation — trainable, but layer 1 crawls |
| $`1.0`$ | $`1`$ | Perfect |
| $`1.05`$ | $`11.5`$ | Manageable with gradient clipping |
| $`1.2`$ | $`9100`$ | Loss becomes NaN within a few steps |

▸ **The window of workable $`\gamma`$ narrows as depth grows, and it narrows exponentially.** At $`L = 10`$ you can be sloppy: $`1.2^{10} = 6.2`$, entirely survivable. At $`L = 50`$ the same sloppiness is fatal. **Every technique in this chapter and the next — He initialization, LayerNorm, residual connections, warmup — is a different engineering answer to the same question: how do we hold $`\gamma`$ at 1?**

**The sigmoid disaster, said carefully.** $`\sigma(z) = 1/(1+e^{-z})`$ squashes everything into $`(0,1)`$, and its derivative $`\sigma'(z) = \sigma(z)(1-\sigma(z))`$ is a bump that peaks at $`z = 0`$ with value $`\tfrac14`$ and falls off fast either side. So $`\sigma' \le 0.25`$ **always** — that is the best case, achieved only for units sitting exactly at zero. A unit with $`z = 4`$ has $`\sigma' \approx 0.018`$, fourteen times worse.

This means the activation *alone* contributes $`\gamma \le 0.25`$, before the weights have any say. There is no initialization scheme that repairs this, because the ceiling is a property of the function's shape, not of the weights. **The 0.25 is a hard cap, and $`0.25^{10} = 9.5\times10^{-7}`$ is the consequence.** Compare ReLU, whose derivative is exactly 1 wherever it is positive: a ReLU network's $`\gamma`$ is set by the weights, which you control, rather than by the activation, which you cannot.

> **Where this came from.** The vanishing-gradient problem was first analysed properly by **Sepp Hochreiter** in his 1991 diploma thesis at the Technical University of Munich, supervised by Jürgen Schmidhuber. It was written in German and went largely unread outside that group; it is the same body of work that led directly to the invention of the **LSTM** (Long Short-Term Memory) in 1997, whose entire design is an answer to the thesis's diagnosis. The result reached the English-speaking community through **Yoshua Bengio, Patrice Simard, and Paolo Frasconi** in 1994, whose paper carried the memorably blunt title *Learning Long-Term Dependencies with Gradient Descent Is Difficult*. For most of the following decade the field's response was not to fix the gradient but to route around it — **greedy layer-wise pretraining**, introduced by Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh in 2006, trained one layer at a time so that no long product of Jacobians was ever needed. It worked, it revived the field, and it was almost entirely abandoned within six years once ReLU and better initialization made the direct approach simply work. **A generation of technique was a workaround for a factor of four in a derivative.**

---

## 6.4 Initialization, derived

### The principle

Choose the initial weight distribution so that **the variance of activations is preserved going forward and the variance of gradients is preserved going backward.**

### Forward pass analysis

Let $`z_j = \sum_{i=1}^{n_{\text{in}}}W_{ji}h_i`$ with $`W_{ji}`$ i.i.d. mean-0 variance $`\sigma_W^2`$, independent of $`h`$, and $`h_i`$ i.i.d. with variance $`\sigma_h^2`$ and mean 0.

$$\mathrm{Var}(z_j) = \sum_{i=1}^{n_{\text{in}}}\mathrm{Var}(W_{ji}h_i) = n_{\text{in}}\,\sigma_W^2\,\sigma_h^2$$

(Using $`\mathrm{Var}(XY)=\mathbb{E}[X^2]\mathbb{E}[Y^2]-\mathbb{E}[X]^2\mathbb{E}[Y]^2 = \sigma_W^2\sigma_h^2`$ for independent zero-mean $`X,Y`$.)

▸ To preserve variance ($`\mathrm{Var}(z)=\sigma_h^2`$) we need $`\boxed{\sigma_W^2 = 1/n_{\text{in}}}`$.

#### Unpacking the variance calculation

**What "variance" is doing here.** $`\mathrm{Var}(X)`$ is the average squared distance of $`X`$ from its mean — the **spread**. In this section it is a stand-in for "the typical size of the numbers flowing through the network," and the whole of initialization theory is the project of keeping that typical size constant from layer to layer. If activations shrink by 30% per layer, then after 50 layers they are $`0.7^{50} = 1.8\times10^{-8}`$ of their original size: the network's output no longer depends on its input in any numerically meaningful way.

**Decoding the assumptions, each of which is doing real work:**

| Assumption | What it means | Why it's needed |
|---|---|---|
| $`W_{ji}`$ **i.i.d.** | Independent and identically distributed — every weight drawn separately from the same distribution | Lets you treat the sum as a sum of independent terms |
| **mean 0** | The weights are symmetric about zero | Kills the cross-terms; without it the mean would also drift layer to layer |
| **independent of $`h`$** | Weights aren't correlated with the data they multiply | True at initialization by construction; **false after training starts**, which is why this is an *initialization* theory |
| $`h_i`$ **mean 0, variance $`\sigma_h^2`$** | The incoming activations are centred and of known spread | This is what the previous layer is supposed to guarantee — the argument is inductive |

**Now the two-line derivation, slowly.**

*Step 1 — variance of a sum of independent things adds.* $`\mathrm{Var}(A + B) = \mathrm{Var}(A) + \mathrm{Var}(B)`$ when $`A`$ and $`B`$ are independent. So summing $`n_{\text{in}}`$ independent products gives $`n_{\text{in}}`$ times the variance of one of them. **This is the same "random things accumulate like $`\sqrt{n}`$, not $`n`$" law as the standard error** (Chapter 1 §1.3.1) — the terms partially cancel, so the *variance* grows like $`n`$ and the *magnitude* like $`\sqrt n`$.

*Step 2 — variance of a product of independent zero-mean things multiplies.* $`\mathrm{Var}(XY) = \sigma_X^2\sigma_Y^2`$. Put together: $`\mathrm{Var}(z_j) = n_{\text{in}}\sigma_W^2\sigma_h^2`$.

*Step 3 — demand that nothing changes.* Set $`\mathrm{Var}(z) = \sigma_h^2`$, cancel $`\sigma_h^2`$ from both sides, and $`n_{\text{in}}\sigma_W^2 = 1`$, so $`\sigma_W^2 = 1/n_{\text{in}}`$. **That is the whole result.**

> **Analogy.** You are pouring water through a funnel that has $`n_{\text{in}}`$ inlet pipes feeding one outlet. If each pipe carries a random amount, the outlet carries the sum — and more pipes means a bigger, more variable flow. To keep the outlet flow the same regardless of how many pipes feed it, you must narrow each pipe in proportion. $`\sigma_W^2 = 1/n_{\text{in}}`$ is exactly "narrow each pipe by the number of pipes." A layer with 1024 inputs needs weights $`\sqrt{1024/64} = 4\times`$ smaller than a layer with 64 inputs, or its output will be four times as loud.

▸ **The single most useful sentence in this section:** **initialization is not about picking small numbers, it is about picking numbers whose size depends on the layer's width.** A constant standard deviation like $`0.01`$ for every layer — which was common practice into the early 2010s — is right for exactly one width and wrong, exponentially, for all the others.

### Backward pass analysis

$`\delta^{(\ell)} = W^{(\ell+1)\top}\delta^{(\ell+1)}\odot\sigma'`$. The same argument with the transpose gives

▸ $$\sigma_W^2 = 1/n_{\text{out}}$$

### Xavier/Glorot: compromise

Can't satisfy both unless $`n_{\text{in}}=n_{\text{out}}`$. Take the harmonic-mean-flavoured compromise:

▸ $$\sigma_W^2 = \frac{2}{n_{\text{in}}+n_{\text{out}}}$$

Uniform version: $`W\sim\mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right]`$ (because $`\mathrm{Var}(\mathcal{U}[-a,a])=a^2/3`$).

**Assumes $`\sigma'\approx1`$ near zero** — true for tanh, false for ReLU.

#### Xavier, decoded — why you can't have both, and what the compromise is

**The conflict, stated plainly.** The forward analysis wants $`\sigma_W^2 = 1/n_{\text{in}}`$ so activations keep their size going up. The backward analysis wants $`\sigma_W^2 = 1/n_{\text{out}}`$ so gradients keep their size coming down. **These are the same requirement viewed from opposite ends of the same matrix**, and a matrix that isn't square cannot satisfy both. A layer mapping 784 inputs to 128 outputs would need $`\sigma_W^2`$ to be simultaneously $`1/784`$ and $`1/128`$ — a factor of six apart.

**Reading the compromise.** Rewrite it as

$$\sigma_W^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}} = \frac{1}{\ \tfrac12(n_{\text{in}} + n_{\text{out}})\ } = \frac{1}{\bar n}$$

▸ So Xavier is *"use $`1/n`$, where $`n`$ is the average of the two fan sizes."* For $`784 \to 128`$: $`\bar n = 456`$, giving $`\sigma_W^2 = 0.00219`$ and $`\sigma_W = 0.047`$. Compare the forward-optimal $`\sigma_W = \sqrt{1/784} = 0.036`$ and the backward-optimal $`\sqrt{1/128} = 0.088`$. **The compromise sits between them, so activations shrink a little and gradients shrink a little, rather than one of them being perfect and the other badly wrong.**

> **Analogy.** Two people sharing a thermostat, one who wants 18°C and one who wants 24°C. Nobody is happy at 21°C, but nobody is miserable either — and, crucially, the *product* of the two dissatisfactions over 50 layers stays bounded. Splitting the difference is not elegance; it is damage control, and it is the correct move.

**The uniform version, decoded.** A uniform distribution on $`[-a, a]`$ has variance $`a^2/3`$ (a standard fact — the spread of a flat distribution). Setting $`a^2/3 = 2/(n_{\text{in}}+n_{\text{out}})`$ and solving gives $`a = \sqrt{6/(n_{\text{in}}+n_{\text{out}})}`$. **The mysterious 6 is nothing but $`2 \times 3`$: a 2 from the compromise and a 3 from the geometry of a flat distribution.** Uniform and Gaussian initialization with matched variance behave almost identically in practice; the choice is a matter of taste and of what the framework defaults to.

**The assumption that kills it.** The whole derivation quietly used $`\sigma' \approx 1`$, i.e. *"the activation function does not change the variance."* Near zero, $`\tanh(z) \approx z`$, so this is fine for tanh — which is what Glorot and Bengio were using. For ReLU it is badly false: ReLU deletes half its input outright. That single mismatch is what the next section repairs, and it is worth exactly one factor of $`\sqrt2`$ — which, compounded thirty times, is the difference between a network that trains and one that does not.

> **Where this came from.** "Xavier initialization" is named after **Xavier Glorot**, then a PhD student with **Yoshua Bengio** at the University of Montreal; their paper *Understanding the Difficulty of Training Deep Feedforward Neural Networks* appeared at AISTATS in 2010. It is one of the very few pieces of standard machine-learning terminology named after an author's **first** name rather than their surname — which is why the same method is also correctly called *Glorot initialization*, and why the two names in a paper's related-work section sometimes refer to the same person. The paper's real contribution was methodological rather than mathematical: rather than proposing a trick, they instrumented networks and *plotted the activation and gradient variance at every layer*, showing empirically that the values collapsed with depth. The theory then explained the plots. Much of the practical progress in deep learning between 2010 and 2015 followed that template — measure what the network is actually doing at each layer, and the fix becomes obvious.

### He/Kaiming: the ReLU correction

ReLU zeroes half its inputs. For $`z`$ symmetric about 0:
$$\mathbb{E}[\mathrm{ReLU}(z)^2] = \mathbb{E}[z^2\mathbb{1}(z>0)] = \tfrac12\mathbb{E}[z^2]$$

▸ **ReLU halves the variance.** Compensate by doubling:

$$\boxed{\ \sigma_W^2 = \frac{2}{n_{\text{in}}}\ }\qquad W\sim\mathcal{N}\left(0,\tfrac{2}{n_{\text{in}}}\right)$$

**The empirical check that made this famous:** He et al. showed a 30-layer plain ReLU network with Xavier init **does not converge at all**, while the same network with He init trains fine. The difference is a factor of $`\sqrt2`$ per layer: $`(1/\sqrt2)^{30} = 3\times10^{-5}`$ — the activations die out by layer 30. **One factor of two, compounded 30 times, is the difference between working and not working.**

**Numbers.** Width 1024, He init: $`\sigma_W = \sqrt{2/1024} = 0.0442`$. A typical weight is $`\pm0.044`$. Your AdamW step of $`3\times10^{-4}`$ is thus $`0.7\%`$ of a typical weight per step. Useful calibration.

#### He initialization, decoded — one factor of two, thirty times over

**Reading $`\mathbb{E}[\mathrm{ReLU}(z)^2] = \mathbb{E}[z^2\mathbb{1}(z>0)] = \tfrac12\mathbb{E}[z^2]`$ aloud:** *"the average squared output of a ReLU equals the average of z-squared restricted to where z is positive, which is half the average of z-squared."*

Why the half: $`\mathbb{1}(z>0)`$ is the indicator — 1 where $`z`$ is positive, 0 where it isn't (Chapter 0 §0.6). If $`z`$ is symmetric about zero, then exactly half of its probability mass sits on each side, and the two sides contribute equally to $`\mathbb{E}[z^2]`$ because squaring erases the sign. ReLU keeps one side and discards the other. **So it keeps exactly half the squared magnitude.** Not approximately half — exactly, for any symmetric distribution.

▸ **The correction is therefore to double the variance:** $`\sigma_W^2 = 2/n_{\text{in}}`$. In standard deviation terms, $`\sqrt2 \approx 1.414`$ — the weights are 41% larger than Xavier would make them. **That is the entire content of He initialization: multiply by the square root of two.**

> **Analogy.** You are told to fill a bucket, but half of what you pour goes straight through a hole in the bottom. You pour twice as fast. Nothing subtle is happening; the only insight is *noticing the hole*. Xavier's derivation was correct for tanh, which has no hole; it was applied to ReLU, which does; and the field ran with a systematically 41%-too-small initialization for several years.

**Now sit with the compounding, because this is the point of the section.** Getting the scale wrong by a factor of $`\sqrt2`$ per layer means activations shrink by that factor at every layer:

| Depth $`L`$ | $`(1/\sqrt2)^{L}`$ | Consequence |
|---|---|---|
| 5 | $`0.18`$ | Barely noticeable |
| 10 | $`0.031`$ | Slow, but it trains |
| 20 | $`9.8\times10^{-4}`$ | Struggling |
| 30 | $`3.1\times10^{-5}`$ | **Does not converge at all** |

▸ **He et al. showed a 30-layer plain ReLU network with Xavier init simply fails, while the same network with He init trains.** No architectural change, no hyperparameter search, no new optimizer — **a single multiplication by $`\sqrt2`$ at step zero.** This is the cleanest demonstration in the book of the chapter's central lesson: in a deep network, a per-layer factor becomes an exponent, and there is no such thing as a small per-layer error.

**Reading the calibration numbers.** With width 1024 and He init, $`\sigma_W = \sqrt{2/1024} = 0.0442`$, so a typical weight is about $`\pm 0.044`$. An AdamW step of $`3\times10^{-4}`$ moves a weight by roughly $`3\times10^{-4}/0.0442 = 0.68\%`$ of its own size. **This is a  useful number to carry around:** it means a weight needs on the order of a hundred steps to change appreciably, and thousands to be substantially rewritten. When you see a training curve move sharply in the first ten steps, that is not the weights changing — it is the *output layer* and the biases finding the right offsets, which they can do quickly because the loss is very sensitive to them.

> **Where this came from.** **Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun** at Microsoft Research Asia published *Delving Deep into Rectifiers* in 2015. The paper's headline contribution was actually **PReLU** — the ReLU with a learnable negative slope — and the initialization scheme was the supporting analysis needed to make their deep rectifier networks trainable at all. It is the initialization that became universal; PReLU is rarely used today. The same paper reported the first result surpassing human-level top-5 accuracy on the ImageNet classification benchmark. The same four authors then published **ResNet** later that same year, which won the ImageNet competition and became one of the most-cited papers in the history of computer science. **Two of the three standard answers to "how do you train a deep network" — proper initialization and residual connections — came from one team in one year.**

### Other schemes you'll encounter

| Scheme | $`\sigma_W^2`$ | Use |
|---|---|---|
| LeCun | $`1/n_{\text{in}}`$ | SELU, self-normalizing nets |
| Xavier | $`2/(n_{\text{in}}+n_{\text{out}})`$ | tanh, sigmoid |
| He | $`2/n_{\text{in}}`$ | ReLU family |
| Orthogonal | $`Q`$ from QR of Gaussian, scaled by gain | RNNs; preserves norm *exactly*, not just in expectation |
| **Zero-init on residual branch** | last layer of each block $`=0`$ | **critical** — see below |

▸ **Zero-init of residual branches** (Fixup, ReZero, and the $`\gamma`$ in **AdaLN-Zero** used by DiT — Ch. 13): initialize the final projection of each residual block to zero, so the block initially computes the identity. The network starts as a shallow, well-conditioned function and *grows* depth as training proceeds. This removes the need for warmup in some settings and is the single most important initialization trick in modern generative architectures. **If you are training a DiT, you are already using this in AdaLN-Zero — and if you modified the architecture and broke it, that would show up as early instability.**

### Scaling by depth

For residual networks, the variance grows *additively* with depth: $`\mathrm{Var}(h^{(L)}) = \mathrm{Var}(h^{(0)}) + \sum_\ell\mathrm{Var}(F_\ell)`$, so it grows like $`L`$. Common fixes: scale residual branch outputs by $`1/\sqrt{L}`$ (used in GPT-2's init: final projections scaled by $`1/\sqrt{2L}`$), or use zero-init as above.

#### The initialization table and zero-init, decoded

**Reading the table.** Every row answers the same question — *"what variance should I draw the weights from?"* — under a different assumption about what the activation function does to the variance:

| Scheme | The assumption behind it |
|---|---|
| LeCun, $`1/n_{\text{in}}`$ | "The activation preserves variance." Correct for SELU by construction; SELU's constants were chosen so this is exactly true |
| Xavier, $`2/(n_{\text{in}}{+}n_{\text{out}})`$ | "The activation is roughly linear near zero, and I want to balance the forward and backward requirements." tanh, sigmoid |
| He, $`2/n_{\text{in}}`$ | "The activation halves the variance." ReLU and its relatives |
| Orthogonal | "Forget variance in expectation — make the map *exactly* norm-preserving." Every singular value is 1, so $`\gamma = 1`$ on the nose |

**Orthogonal initialization, explained.** Draw a random Gaussian matrix, run QR decomposition on it (factoring it as an orthogonal $`Q`$ times an upper-triangular $`R`$), and keep $`Q`$. From Chapter 1 §1.1.2, an orthogonal matrix is a **pure rotation**: it never changes any vector's length. So instead of "the variance is preserved *on average*," you get "the norm is preserved *exactly, for every input*." This matters most in RNNs, where the *same* matrix is applied at every timestep, so any deviation from 1 compounds not over $`L`$ layers but over $`T`$ timesteps — and $`T`$ can be thousands.

**Zero-init on the residual branch, decoded.** A residual block computes $`x_{\ell+1} = x_\ell + F(x_\ell)`$. Set the **final** projection inside $`F`$ to exactly zero and $`F`$ outputs zero, so $`x_{\ell+1} = x_\ell`$: **the block is the identity function.** Note that only the last layer of the block is zeroed — the layers before it are initialized normally, so they still receive meaningful gradients (the weight gradient depends on the *input* to a layer, not its output).

▸ **Why this is such a good idea.** At step 0 a 100-layer network is behaving exactly like a 0-layer network — a well-conditioned, shallow, trivially trainable function. As training proceeds, each block's final projection moves away from zero and the network *grows its own depth*, adding capacity only where the gradient asks for it. **You are not training a deep network from scratch; you are training a shallow one and letting it become deep.** This removes the need for learning-rate warmup in some settings, and it is the single most important initialization trick in modern generative architectures.

**Reading "the variance grows additively with depth."** Because a residual block *adds* rather than replaces, the variances add: $`\mathrm{Var}(h^{(L)}) = \mathrm{Var}(h^{(0)}) + \sum_\ell \mathrm{Var}(F_\ell)`$. If each branch contributes the same variance, the total grows like $`L`$ — so the *standard deviation* grows like $`\sqrt L`$. For $`L = 48`$ (GPT-2 scale), that is a factor of $`\sqrt{48} \approx 6.9`$ between the first layer's residual stream and the last. Scaling each branch's output by $`1/\sqrt{L}`$ cancels it exactly: $`L`$ branches each contributing $`1/L`$ of a variance sums back to 1. **The $`1/\sqrt{2L}`$ in GPT-2's initialization is this, with the 2 accounting for the fact that each transformer layer has two residual branches — attention and the feed-forward network — not one.**

> **Analogy.** Exploding variance in a residual network is a snowball rolling downhill: nothing is *wrong* at any point, each layer adds only a modest amount, but the total keeps accumulating because nothing ever subtracts. Dividing each contribution by $`\sqrt L`$ is agreeing in advance how many layers will contribute and rationing each one's share.

---

## 6.5 Activation functions

| Name | $`f(z)`$ | $`f'(z)`$ | Notes |
|---|---|---|---|
| Sigmoid | $`\frac{1}{1+e^{-z}}`$ | $`f(1-f)\le\frac14`$ | saturating, non-zero-centered. Avoid except as a gate. |
| Tanh | $`\frac{e^z-e^{-z}}{e^z+e^{-z}}`$ | $`1-f^2\le1`$ | zero-centered sigmoid. Fine in RNNs. |
| ReLU | $`\max(0,z)`$ | $`\mathbb{1}[z>0]`$ | fast, sparse, no saturation for $`z>0`$. **Dying ReLU.** |
| Leaky ReLU | $`\max(\alpha z,z)`$, $`\alpha=0.01`$ | $`\alpha`$ or 1 | fixes dying ReLU |
| PReLU | learnable $`\alpha`$ | | slight gain, extra params |
| ELU | $`z`$ or $`\alpha(e^z-1)`$ | | smooth, negative saturation, mean closer to 0 |
| GELU | $`z\Phi(z)`$ | $`\Phi(z)+z\phi(z)`$ | **transformer default** |
| SiLU/Swish | $`z\sigma(z)`$ | $`\sigma(z)(1+z(1-\sigma(z)))`$ | ≈GELU, cheaper |
| Mish | $`z\tanh(\mathrm{softplus}(z))`$ | | marginal gains, slower |
| GLU family | $` (Wx)\odot\sigma(Vx)`$ | | **SwiGLU** is the LLM standard |

#### The activation table, decoded

Every row is a different answer to one question: **what should a single number do on its way from one layer to the next?** The whole design space is "which shapes of bend are useful," and there are only a handful of considerations.

**The four properties that actually matter:**

| Property | Why it matters |
|---|---|
| **Does it saturate?** | A flat region means $`\sigma' \approx 0`$ there, which means gradients die. Sigmoid saturates on *both* sides; ReLU only on the left |
| **Is it zero-centred?** | If outputs are always positive (sigmoid, ReLU), the next layer's weight gradients all share a sign per unit, so updates zigzag rather than going straight |
| **Is $`\sigma'`$ bounded by 1?** | Anything below 1 contributes $`\gamma < 1`$ and compounds (§6.3). ReLU's derivative is exactly 1 where active — the best possible |
| **Is it smooth?** | ReLU has a kink at 0 where the second derivative is undefined. Adam builds a curvature estimate, and curvature is better behaved for smooth functions |

**Reading the derivative column with numbers.**

- **Sigmoid:** $`f' = f(1-f)`$. At $`f = 0.5`$ this is $`0.25`$ (the maximum); at $`f = 0.9`$ it is $`0.09`$; at $`f = 0.99`$ it is $`0.0099`$. **A confident sigmoid unit is a nearly dead one.**
- **Tanh:** $`f' = 1 - f^2`$. At $`f = 0`$ this is 1 — a full order of magnitude better than sigmoid's best case, and the reason tanh replaced sigmoid in hidden layers long before ReLU arrived. It still saturates: at $`f = 0.9`$, $`f' = 0.19`$.
- **ReLU:** $`f' = \mathbb{1}[z>0]`$ — literally 1 or 0, nothing in between. No saturation on the positive side at any magnitude, which is precisely the property that made deep networks trainable.
- **Leaky ReLU:** $`f' \in \{\alpha, 1\}`$ with $`\alpha = 0.01`$. A dead unit now gets $`1\%`$ of a gradient instead of $`0\%`$ — small, but the difference between "eventually recovers" and "never".

> **Analogy.** These are all valves. A **sigmoid** is a valve that jams shut at both extremes — push hard either way and it stops responding to the handle at all. **Tanh** is the same valve mounted so its neutral position is  neutral. **ReLU** is a one-way check valve: fully open in one direction, fully shut in the other, with no in-between and nothing to jam. **GELU and SiLU** are check valves with a slightly soft seat, so they close gradually rather than snapping. **GLU/SwiGLU** is not one valve but two — one carrying the flow and one controlling how much of it gets through — which is why it costs an extra matrix.

▸ **The honest summary of thirty years of activation-function research:** the difference between sigmoid and ReLU was worth *orders of magnitude* and made deep learning possible. Every difference since — ReLU to GELU to SiLU to Mish — is worth around half a percent. **The first bit of the design space contained nearly all the value, and the field has been mining the tail ever since.** It is worth knowing which era of the search a proposed improvement belongs to.

### GELU, properly

▸ $$\mathrm{GELU}(z) = z\cdot\Phi(z) = z\cdot\tfrac12\left[1+\mathrm{erf}\!\left(\tfrac{z}{\sqrt2}\right)\right]$$

**Interpretation:** it's a *stochastic regularizer made deterministic*. Consider multiplying $`z`$ by a Bernoulli mask with probability $`\Phi(z)`$ (so large inputs are kept more often — "adaptive dropout"). GELU is the expectation of that: $`\mathbb{E}[z\cdot m] = z\Phi(z)`$.

Tanh approximation (what most code uses):
$$\mathrm{GELU}(z)\approx 0.5z\left(1+\tanh\left[\sqrt{2/\pi}\left(z+0.044715z^3\right)\right]\right)$$

**Why it beats ReLU in transformers:** it's smooth (better-behaved second derivative for Adam's preconditioner), and it's non-monotone near zero, which lets it represent slightly richer local behaviour. The gain is small but consistent (~0.5% perplexity).

#### GELU, decoded

**The symbols first.** $`\Phi(z)`$ is the **CDF (cumulative distribution function)** of the standard Gaussian: *"the probability that a random draw from a bell curve lands below $`z`$."* It runs smoothly from 0 (far left) through $`0.5`$ (at $`z=0`$) to 1 (far right). It is a sigmoid-shaped curve, but a specific one with a probabilistic meaning. And $`\mathrm{erf}`$ is the **error function**, a rescaled version of the same integral — the two are related by $`\Phi(z) = \tfrac12[1 + \mathrm{erf}(z/\sqrt2)]`$, which is the second equality in the boxed formula. Nothing deep is happening there; it is a change of units, present because numerical libraries ship `erf` rather than `Phi`.

▸ **So $`\mathrm{GELU}(z) = z\cdot\Phi(z)`$ reads: "keep the input, scaled by the probability that a random Gaussian would fall below it."** Large positive $`z`$: $`\Phi \approx 1`$, so you keep essentially all of it — just like ReLU. Large negative $`z`$: $`\Phi \approx 0`$, so you keep essentially none — just like ReLU. Near zero: $`\Phi \approx 0.5`$, so you keep about half, whereas ReLU makes an abrupt decision. **GELU is a ReLU that is unsure near the origin and expresses its uncertainty by passing a fraction through.**

**Put numbers on the difference:**

| $`z`$ | $`\Phi(z)`$ | $`\mathrm{GELU}(z)`$ | $`\mathrm{ReLU}(z)`$ |
|---|---|---|---|
| $`-2`$ | $`0.023`$ | $`-0.046`$ | $`0`$ |
| $`-1`$ | $`0.159`$ | $`-0.159`$ | $`0`$ |
| $`-0.5`$ | $`0.309`$ | $`-0.154`$ | $`0`$ |
| $`0`$ | $`0.5`$ | $`0`$ | $`0`$ |
| $`1`$ | $`0.841`$ | $`0.841`$ | $`1`$ |
| $`2`$ | $`0.977`$ | $`1.954`$ | $`2`$ |

Notice something the algebra alone doesn't shout: GELU is **non-monotone**. Going from $`z=-1`$ to $`z=-0.5`$, the output goes *up* from $`-0.159`$ to $`-0.154`$; the minimum sits near $`z \approx -0.75`$. A small negative input produces a small negative output, and a *more* negative input eventually produces a less negative one. ReLU cannot do this; it is the extra expressive wrinkle the section refers to.

**The "stochastic regularizer made deterministic" reading, in full.** Imagine a version of dropout in which a unit's keep-probability depends on *how large its own pre-activation is* — big inputs kept often, small inputs dropped often. Formally, multiply $`z`$ by a random mask $`m \sim \mathrm{Bernoulli}(\Phi(z))`$. Its expected value is

$$\mathbb{E}[z\cdot m] = z\cdot\mathbb{P}(m=1) = z\,\Phi(z) = \mathrm{GELU}(z)$$

▸ **GELU is the average of an adaptive dropout you never actually run.** This is a recurring and  useful pattern in deep learning: define a stochastic procedure that would be expensive or noisy, then replace it with its expectation and compute that in closed form. Inference-time dropout (Ch. 7 §7.5) and the reparameterization trick are the same move.

**Why the tanh approximation exists.** `erf` is a special function that historically was slower than `tanh` on GPUs, so the approximation $`0.5z(1+\tanh[\sqrt{2/\pi}(z + 0.044715z^3)])`$ was written into the original reference implementations and propagated everywhere. It agrees with exact GELU to about $`10^{-3}`$. The two are close enough that models trained with one and evaluated with the other are essentially unaffected — but not so close that you should switch mid-training, and framework flags named `approximate='tanh'` exist precisely because a few reproducibility bugs have been traced to this.

> **Where this came from.** GELU was proposed by **Dan Hendrycks and Kevin Gimpel** in 2016, while Hendrycks was an undergraduate. The same paper also described the function $`z\sigma(z)`$, which was later independently rediscovered by a Google Brain team (Prajit Ramachandran, Barret Zoph, and Quoc Le) in 2017 through an **automated architecture search over activation functions** — they let a search algorithm propose compositions of primitive functions, trained networks with each, and the winner was named **Swish**. The authors acknowledged the earlier discovery once it was pointed out; the function is also called **SiLU**, from an independent 2017 reinforcement-learning paper by Stefan Elfwing, Eiji Uchibe, and Kenji Doya. **The same function has three names because it was found three times by people asking three different questions**, and modern frameworks now list `SiLU` and `Swish` as aliases. GELU became the default largely because BERT and GPT used it, which is a reminder that adoption in this field is driven at least as much by which reference implementation people copy as by which function is best.

### SwiGLU

$$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(W_1x)\big)\odot(W_2x),\quad \text{then } W_3(\cdot)$$

Three matrices instead of two, so to keep parameter count fixed you shrink the hidden dim by $`2/3`$: $`d_{\text{ff}} = \frac{8}{3}d_{\text{model}}`$ instead of $`4d_{\text{model}}`$. **Consistently ~1% better perplexity at matched params.** Used in LLaMA, PaLM, Mixtral. Noam Shazeer's paper famously concludes with "we attribute their success to divine benevolence" — which is honest, since nobody has a clean theory for why gating helps.

#### SwiGLU, decoded

**Read the formula as two parallel branches that then multiply.**

$$\underbrace{\mathrm{SiLU}(W_1x)}_{\text{the gate}} \ \odot\ \underbrace{(W_2x)}_{\text{the content}} \qquad\text{then}\qquad W_3(\cdot)$$

The same input $`x`$ is sent through **two different matrices at once**. One result is passed through a nonlinearity and used as a **gate** — a per-channel multiplier saying how much gets through. The other is left linear and carries the **content**. The $`\odot`$ multiplies them entry by entry, and $`W_3`$ projects the result back down to the model width.

> **Analogy.** A standard feed-forward layer is a single pipe with a valve at the end: the same signal decides both *what* flows and *how much*. A gated layer splits the job — one circuit measures the signal and sets the valve, a separate circuit carries the payload. **Separating "what to say" from "whether to say it" is the whole idea**, and it is the same idea as an LSTM's forget gate and as attention's softmax weights. Gating is arguably the most reused structural motif in deep learning.

**Where the $`\tfrac83`$ comes from — the arithmetic in full.** A standard transformer feed-forward block has two matrices, $`d \to 4d`$ and $`4d \to d`$, for $`2 \times 4d^2 = 8d^2`$ parameters. SwiGLU needs three: $`d \to d_{\text{ff}}`$ twice, and $`d_{\text{ff}} \to d`$ once, for $`3\,d\,d_{\text{ff}}`$ parameters. Setting them equal:

$$3\,d\,d_{\text{ff}} = 8d^2 \quad\Rightarrow\quad d_{\text{ff}} = \tfrac83 d \approx 2.67\,d$$

▸ **So SwiGLU is not "more parameters," it is "the same parameters spent differently"** — three narrower matrices instead of two wider ones. That is what makes the comparison fair, and it is why the reported ~1% perplexity gain is meaningful rather than a restatement of "bigger model does better." Concretely, at $`d = 4096`$: the classic block uses $`d_{\text{ff}} = 16384`$; SwiGLU uses $`d_{\text{ff}} = 10923`$ (usually rounded to a multiple of 256 for hardware reasons — this is why you see values like 11008 in real configurations, a number that otherwise looks arbitrary).

**On the "divine benevolence" line.** It is worth reading it as what it is: a researcher declining to invent a post-hoc story for an empirical result. The paper is short and consists largely of a table of variants and their scores; the honest finding is *"these work, we ran the experiments, we do not know why."* **In a field where every architectural choice arrives with a confident-sounding justification attached, that is a more useful posture than it first appears** — and one worth imitating when reporting your own results.

### Dying ReLU, quantified

If a unit's pre-activation is negative for every training example, its gradient is exactly zero forever — it's dead. With He init and no bias, roughly 50% of units are inactive *per example*, which is fine (that's sparsity). Permanent death happens when a large gradient step pushes the bias very negative. **Rate:** with a too-high LR, 20–40% of ReLU units can die in the first few hundred steps.

Fixes: leaky/ELU/GELU, lower LR, proper init, or normalization layers (which recenter pre-activations every step and make permanent death nearly impossible).

#### Dying ReLU, decoded — the difference between asleep and dead

**The mechanism, step by step.** A ReLU unit computes $`h = \max(0, w^\top x + b)`$. Its gradient with respect to $`w`$ is $`\delta \cdot x \cdot \mathbb{1}[z > 0]`$. So if $`z \le 0`$, the gradient is **exactly zero** — not small, zero. A zero gradient means no update. No update means $`w`$ and $`b`$ do not move. If $`z \le 0`$ for *every* example in the dataset, the unit is frozen for the rest of training, and no amount of further training will revive it.

▸ **This is a fixed point, and that is what makes it different from every other pathology in this chapter.** Vanishing gradients are a matter of degree — a tiny gradient still moves you, eventually. A dead ReLU is *absorbing*: the state that causes zero gradient is the state that zero gradient preserves. **There is no path out.**

**Two situations that look identical and are not:**

| | Inactive (fine, even desirable) | Dead (a bug) |
|---|---|---|
| Condition | $`z \le 0`$ for *this* example | $`z \le 0`$ for *every* example |
| Gradient | Zero on this example, nonzero on others | Zero, always |
| Effect | Sparsity — roughly 50% of units per example | Wasted capacity — the unit is a constant 0 |
| Reversible? | Yes, trivially | No |

**How a unit crosses from one to the other.** The usual cause is a single oversized step. The bias update is $`b \leftarrow b - \eta\,\delta`$, and if $`\eta\delta`$ is large enough, $`b`$ lands far enough below zero that $`w^\top x + b < 0`$ for the entire data distribution. From that moment forward, $`\delta = 0`$ and $`b`$ is stuck at whatever value the bad step left it. **The unit was killed by one step, and every step afterwards ratifies the outcome.** The quoted rate — 20–40% of units dying in the first few hundred steps at too high a learning rate — is why "the loss plateaued and never recovered" is such a common early-training failure report.

> **Analogy.** A light switch wired so that turning it off also cuts power to the switch itself. Perfectly functional while on; permanently off once off. The fix is never to be cleverer about flipping the switch — it is to rewire so the switch always has power. **Leaky ReLU is that rewiring:** by passing $`0.01z`$ instead of $`0`$ on the negative side, a "dead" unit still receives $`1\%`$ of a gradient, which is small but not absorbing, so it can crawl back.

**Why normalization layers almost eliminate the problem** (this is the bridge to Chapter 7). BatchNorm and LayerNorm **re-centre the pre-activations every single step**, forcing their mean to zero and their variance to one. A unit whose $`z`$ has drifted far negative is dragged straight back to a distribution straddling zero — so roughly half its inputs land positive by construction. **You cannot have a permanently-negative pre-activation when something is actively subtracting the mean.** This is one of several places where normalization quietly repairs a problem that had its own literature and its own set of proposed fixes; the fix that won was one that made the problem structurally impossible.

---

## 6.6 Debugging a network with numbers

The diagnostics that find 90% of bugs, in order:

1. **Overfit 1 batch.** Take 8 examples, train until loss $`\approx0`$. If you can't, you have a bug — not a hyperparameter problem. Do this before *anything* else.
2. **Check initial loss.** For $`K`$-way classification it should be $`\log K`$. If a D3PM starts at CE $`\ne \log K`$ at high $`t`$, your logits are miscalibrated at init or your loss is wrong.
3. **Activation statistics per layer.** Mean should be ~0 (or ~0.5·std for ReLU), std should be ~constant across depth. If std shrinks 10× per layer, your init is wrong.
4. **Gradient norms per layer.** Should be within ~1 order of magnitude of each other. A 4-order-of-magnitude spread means vanishing gradients.
5. **Update-to-weight ratio** $`\frac{\|\eta\Delta\theta_\ell\|}{\|\theta_\ell\|}`$. Target $`\approx10^{-3}`$. Above $`10^{-2}`$: LR too high. Below $`10^{-4}`$: that layer is frozen.
6. **Gradient check** (for custom ops): $`\frac{\mathcal{L}(\theta+\epsilon e_i)-\mathcal{L}(\theta-\epsilon e_i)}{2\epsilon}`$ vs analytic, $`\epsilon=10^{-4}`$ in **float64**. Relative error should be $`<10^{-6}`$.

#### Why each diagnostic works

These six are ordered by how much information they give per minute spent, and each one is a direct consequence of something derived earlier in the chapter. Knowing *why* each works is what lets you interpret a result that is neither clearly good nor clearly bad.

**1. Overfit one batch.** Take eight examples and drive the loss to near zero. This separates the two fundamentally different kinds of failure: *"the model cannot fit anything"* (a bug — wrong loss, detached gradient, wrong label alignment, a shape error that silently broadcast) versus *"the model fits but does not generalize"* (a modelling or data problem). A network with more parameters than eight examples have constraints **must** be able to memorize them. ▸ **If it can't, no hyperparameter will save you, and every hour spent tuning before this passes is wasted.** This is the highest-value five minutes in machine learning practice.

**2. Check the initial loss against $`\log K`$.** At initialization a well-set-up $`K`$-way classifier should be maximally uncertain, predicting $`1/K`$ for every class. Cross-entropy for that prediction is $`-\log(1/K) = \log K`$. Concretely:

| Task | $`K`$ | Expected initial loss |
|---|---|---|
| Binary | 2 | $`\log 2 = 0.69`$ |
| CIFAR-10 | 10 | $`\log 10 = 2.30`$ |
| ImageNet | 1000 | $`\log 1000 = 6.91`$ |
| LLM, typical vocabulary | 50,257 | $`\log 50257 = 10.82`$ |

▸ **This is the cheapest bug detector in existence.** An initial loss well *above* $`\log K`$ means your output layer is initialized with large logits and is confidently wrong — check for a missing $`1/\sqrt{n}`$ or a forgotten zero-init on the head. An initial loss *below* $`\log K`$ before any training has occurred means information is leaking from the labels into the inputs. Either way you know within one step, before spending a GPU-hour. (**D3PM** in the book's example stands for **Discrete Denoising Diffusion Probabilistic Model**; at a high noise level $`t`$ its input is nearly pure noise, so the same $`\log K`$ reasoning applies — a well-formed model should be maximally uncertain there.)

**3. Activation statistics per layer.** This is §6.4 turned into a measurement. You derived that the variance should be preserved layer to layer; now check that it is. A standard deviation shrinking $`10\times`$ per layer is exactly the failure mode He initialization exists to prevent, and you will see it as a straight line on a log-scale plot against depth. The "mean $`\approx 0.5 \times`$ std for ReLU" note is because ReLU outputs are non-negative, so their mean is pushed up — a mean near zero *after* a ReLU would itself be a symptom.

**4. Gradient norms per layer.** This is §6.3 turned into a measurement. The gradient reaching layer 1 is $`\gamma^{L}`$ times what left the loss. A spread of four orders of magnitude across layers corresponds to $`\gamma`$ meaningfully away from 1, so the early layers are effectively frozen while the late ones train. **One order of magnitude of spread is normal; four is a diagnosis.**

**5. Update-to-weight ratio.** $`\|\eta\Delta\theta_\ell\|/\|\theta_\ell\|`$ answers *"what fraction of its own size does this layer move per step?"* The target $`\approx 10^{-3}`$ says a weight changes by about $`0.1\%`$ per step, so it takes on the order of a thousand steps to be substantially rewritten — which matches the calibration computed in §6.4. Reading the extremes: above $`10^{-2}`$ a layer is rewriting itself every hundred steps and will oscillate rather than converge; below $`10^{-4}`$ it would need a hundred thousand steps to change at all, which for most runs means it is frozen. ▸ **This is a better learning-rate diagnostic than the loss curve, because it is per-layer.** A loss curve tells you the average is wrong; this tells you *which layer*.

**6. Gradient check, and why float64 is not optional.** The expression $`\frac{\mathcal{L}(\theta+\epsilon e_i) - \mathcal{L}(\theta - \epsilon e_i)}{2\epsilon}`$ is the **central difference**: nudge one parameter up, nudge it down, and take the slope between. ($`e_i`$ is the vector that is 1 in position $`i`$ and 0 everywhere else — "nudge only this one parameter.") It approximates the true derivative with error proportional to $`\epsilon^2`$.

There are two competing error sources, and this is the whole reason for the float64 requirement:

| Error source | Size | Wants $`\epsilon`$ to be |
|---|---|---|
| Truncation (the approximation itself) | $`\sim\epsilon^2`$ | small |
| Round-off (subtracting two nearly-equal numbers) | $`\sim`$ machine precision $`/\,\epsilon`$ | large |

In **float64**, machine precision is about $`10^{-16}`$, so at $`\epsilon = 10^{-4}`$ the round-off term is around $`10^{-12}`$ and the truncation term around $`10^{-8}`$ — comfortably below the $`10^{-6}`$ tolerance. In **float32**, machine precision is about $`10^{-7}`$, so the round-off term becomes $`10^{-7}/10^{-4} = 10^{-3}`$ — a thousand times larger than the tolerance you are trying to check against. ▸ **A gradient check in float32 fails for correct code and passes for nothing**, which is why the instruction is not stylistic. The subtraction of two nearly-equal floating-point numbers, known as **catastrophic cancellation**, is the specific culprit.

---

## Did you know?

- **The first neural network was a machine, not a program.** Rosenblatt's Mark I Perceptron (1958) was a room-sized device with 400 photocells wired to potentiometers. "Training" meant electric motors physically turning knobs, and the weights were voltages. There was no software; the learning rule was implemented in copper.

- **Backpropagation has at least four names in four fields, and they were coined independently.** Numerical analysts call it **reverse-mode automatic differentiation**; control theorists and physicists call it **the adjoint method**; meteorologists doing data assimilation call it **adjoint sensitivity analysis**; machine learning calls it backprop. All four compute the same thing: derivatives of one scalar with respect to many inputs, by walking a computation backwards.

- **Frank Rosenblatt never saw multi-layer networks vindicated.** He died in a boating accident in 1971 at the age of 43 — two years after Minsky and Papert's *Perceptrons* argued that his single-layer machines were fundamentally limited, and fifteen years before backpropagation showed that the limitation applied only to one layer.

- **The theorem most often cited as deep learning's mathematical foundation is about an architecture nobody uses.** The universal approximation theorem concerns networks with exactly **one** hidden layer. It says nothing about depth, nothing about training, and its width bounds can be exponential in the input dimension.

- **Xavier initialization is named after a first name.** Xavier Glorot was the first author; the convention in almost all other named methods is to use the surname, which is why the identical method is also called Glorot initialization and why the two occasionally appear in the same bibliography as if they were different techniques.

- **The difference between a 30-layer network that trains and one that does not was a single factor of $`\sqrt2`$.** Xavier initialization is correct for tanh and 41% too small for ReLU. Compounded over 30 layers, $`(1/\sqrt2)^{30} = 3\times10^{-5}`$ — the activations simply die out before reaching the end.

- **The same activation function was discovered three times and has three names.** $`z\,\sigma(z)`$ appeared in the 2016 GELU paper, was found independently in 2017 by a Google Brain team running an **automated search over candidate activation functions** (who named it Swish), and independently again in a 2017 reinforcement-learning paper (which named it SiLU). Modern frameworks list Swish and SiLU as aliases of one another.

- **The odd number 11008 in LLaMA-style configuration files is $`\tfrac83 \times 4096`$, rounded to a multiple of 256.** SwiGLU uses three weight matrices where a standard feed-forward block uses two, so the hidden width is scaled by $`2/3`$ to hold the parameter count fixed — and then nudged to a hardware-friendly multiple. A number that looks arbitrary is two decisions stacked.

- **The vanishing gradient problem was diagnosed in a German-language master's thesis that almost nobody read.** Sepp Hochreiter's 1991 diploma thesis at TU Munich, supervised by Jürgen Schmidhuber, analysed exactly why deep and recurrent networks stop learning. The LSTM, published six years later, is a direct architectural answer to that analysis.

- **An entire research era existed to route around a factor of four.** Sigmoid's derivative is capped at $`1/4`$, so a ten-layer sigmoid network attenuates gradients by $`0.25^{10} \approx 10^{-6}`$. Greedy layer-wise pretraining (2006) was invented to avoid ever computing a long product of such factors — and was largely abandoned within six years once ReLU removed the cap.

- **The 6 in the $`C \approx 6ND`$ training-compute formula is just $`2+2+2`$.** Each weight does one multiply-add in the forward pass and two in the backward pass, and a multiply-add is counted as two floating-point operations. The most quoted number in large-scale training economics is a three-row table.

- **ReLU is older than its fame by roughly four decades.** Rectifier-style nonlinearities appear in Kunihiko Fukushima's work on multilayered visual networks as early as 1969, long before the name "rectified linear unit" existed. It became standard only after 2010–2012, when Nair and Hinton, then Glorot and Bordes and Bengio, and then AlexNet demonstrated it — AlexNet reporting that a ReLU network reached a given training error several times faster than the equivalent tanh network.

---

## Check for Understanding

**Backpropagation is one recursion, $`\delta^{(\ell)} = (W^{(\ell+1)\top}\delta^{(\ell+1)})\odot\sigma'(z^{(\ell)})`$, applied backwards; everything difficult about training deep networks is that this recursion multiplies $`L`$ matrices together, so initialization, normalization, and residual connections all exist to keep that product's scale near one.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why does a network without activation functions collapse into a single layer, and what specifically does the nonlinearity prevent you from doing?** (Answer in terms of pre-multiplying the matrices, not in terms of "expressivity.")
2. **What is the error signal $`\delta`$, in terms of blame?** Why is it defined on the pre-activation $`z`$ rather than on the weights or on the activations?
3. **Why does the transpose appear in the backward pass?** (Correct answer: because backpropagation runs the network's wiring in reverse, and transposing reverses every arrow. It is not an algebraic trick.)
4. **Why is the weight gradient an outer product, and what does that mean in one sentence with no symbols?**
5. **Why is the backward pass about twice as expensive as the forward pass?** Which two jobs is it doing?
6. **Why does a residual connection stop gradients from vanishing?** (The answer is a fact about the derivative of the plus sign.)
7. **Why can a per-layer factor of 0.8 be fatal while a per-layer factor of 0.95 is merely annoying?** Put numbers on it for a 50-layer network.
8. **Why must initialization depend on the layer's width?** Explain it with pipes and a funnel rather than with variance algebra.
9. **Why does ReLU need twice the initialization variance that tanh does?**
10. **What is the difference between a ReLU unit that is inactive and one that is dead?** Why is the second irreversible, and why does normalization almost eliminate it?
11. **Why is universal approximation a weaker result than it sounds?** Name the three things it does not tell you.
12. **Why should the initial loss of a $`K`$-way classifier be $`\log K`$, and what does it mean if it isn't?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 07 — Normalization & Regularization](07-normalization-and-regularization.md)
