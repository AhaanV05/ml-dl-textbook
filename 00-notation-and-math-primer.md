# Chapter 0 — How to Read the Mathematics in This Book

> **Prerequisites:** none. This chapter exists so that the other thirty-four don't need any.
> **Why this chapter exists:** the rest of the book contains about 3,300 pieces of mathematical notation, but they are built from roughly **sixty symbols**. Learn the sixty and the 3,300 become readable. This is a decoder ring, not a course.

---

## 0.0 The one-line idea

**Mathematical notation is compressed English.** Every formula in this book can be read aloud as an ordinary sentence, and until you can read it aloud, you don't understand it yet.

### The analogy

Notation is like sheet music. A page of sheet music looks like an impenetrable wall of dots to someone who hasn't learned it, and yet it isn't *deep* — it's a compact way of writing "play this note, this loud, for this long." Nobody looks at sheet music and tries to intuit the melody from the shape of the page; they learn what five or six symbols mean, and then they can sight-read anything.

Mathematical notation is exactly this. $\sum$ is not a concept, it is an *abbreviation* for the word "add up." $\mathbb{E}$ is an abbreviation for "the average of." Once you stop treating the symbols as ideas and start treating them as **shorthand for words**, the wall becomes a sentence.

The single most damaging belief a beginner can have is that the difficulty lives in the symbols. It doesn't. The difficulty lives in the ideas, and the symbols are there to make the ideas *shorter*. Someone chose them to save writing.

---

## 0.1 The six habits that make any formula readable

Before any specific symbol, here is the *method*. When you hit a formula that looks like noise, do these six things in order. This works on every equation in this book, including the worst ones.

**Habit 1 — Find the equals sign and ask "what is being defined?"**
The left side is almost always the thing being named. The right side is the recipe for computing it. $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ is not a claim to verify, it's a *definition*: "the thing I'm calling variance is computed by this recipe."

**Habit 2 — Read every symbol as a word, out loud.**
$`\|x\|_2 = \sqrt{\sum_i x_i^2}`$ becomes *"the two-norm of x equals the square root of the sum over i of x-i squared."* Then translate to English: *"the length of the arrow x is the square root of the sum of its squared components."* Then to intuition: *"Pythagoras, in as many dimensions as you like."*

**Habit 3 — Identify what varies and what is fixed.**
In $`\sum_j e^{z_j}`$, the $j$ moves and everything else stands still. The index under a $\sum$ is a **loop counter** — literally a `for` loop. This one realization dissolves most of the fear of summation notation.

**Habit 4 — Check the shapes.**
Every quantity is a scalar (one number), a vector (a list), or a matrix (a grid). If you know the shape of every term, you can catch most errors without understanding the content at all. A formula where a $3\times4$ matrix is being added to a $7$-vector is wrong, and you know it's wrong without knowing what it means.

**Habit 5 — Set the numbers to something trivial.**
Set the dimension to 1. Set the batch size to 1. Set the matrix to the identity. Almost every intimidating formula collapses into something obvious when you make it small. This is the single most effective technique in this entire chapter.

**Habit 6 — Ask "what would make this big, and what would make it zero?"**
Understanding *when a quantity vanishes* and *when it blows up* is usually more useful than being able to compute it. It tells you what the formula is actually measuring.

▸ **If you do nothing else, do Habit 5.** Set $n=1$, set $d=1$, set the matrix to the identity, and see what the formula becomes. Generality is what makes notation look hard; specificity is the cure.

---

## 0.2 Sets and spaces — "what kind of thing is this?"

These symbols answer the question *"what type is this variable?"* They are the type annotations of mathematics.

| Symbol | Read aloud | What it actually means |
|---|---|---|
| $\in$ | "is in" / "belongs to" | Membership. $x \in S$ = "x is one of the things in S." |
| $\mathbb{R}$ | "the reals" | Any ordinary number: $3$, $-0.5$, $\pi$. Not complex, not restricted to integers. |
| $\mathbb{R}^n$ | "R-n" | A list of $n$ ordinary numbers. A **vector**. A point in $n$-dimensional space. |
| $\mathbb{R}^{m \times n}$ | "R m-by-n" | A grid of numbers with $m$ rows and $n$ columns. A **matrix**. |
| $x \in \mathbb{R}^n$ | "x in R-n" | **"x is a list of n numbers."** That is the whole content. |
| $A \in \mathbb{R}^{m\times n}$ | "A in R m-by-n" | "A is a table of numbers, m rows tall and n columns wide." |
| $\{a, b, c\}$ | "the set containing a, b, c" | An unordered collection, no duplicates. |
| $\varnothing$ | "the empty set" | Nothing in it. |
| $\subset$ | "is a subset of" | Everything in the first is also in the second. |
| $\to$ | "maps to" / "goes to" | $f: \mathbb{R}^n \to \mathbb{R}^m$ = "f eats an n-list and returns an m-list." |

**The most common line in this book** is something like $A \in \mathbb{R}^{m \times n}$. Beginners skim past it. Don't — it is the single most information-dense line in any derivation, because it tells you the **shape**, and shape is what lets you check everything else.

> **Analogy.** $x \in \mathbb{R}^{768}$ is the mathematical way of writing `x: float[768]` in a programming language. It is a type declaration. Nothing more mysterious than that.

### Superscripts that look scary but aren't

$\mathbb{R}^{n}$, $\mathbb{R}^{m\times n}$, $\mathbb{R}^{B \times T \times d}$ — the superscript is just **the shape**, exactly like a NumPy or PyTorch `.shape` tuple. $\mathbb{R}^{B\times T\times d}$ is a tensor of shape `(B, T, d)`: batch size $B$, sequence length $T$, feature width $d$. That's it.

---

## 0.3 The big operators — loops in disguise

These four symbols cause more beginner panic than everything else combined, and all four are `for` loops.

### $\sum$ — "add up"

$$\sum_{i=1}^{n} x_i \quad=\quad x_1 + x_2 + \dots + x_n$$

Read it as: **"start i at 1, go up to n, add each $`x_i`$ into a running total."** In code:

```python
total = 0
for i in range(1, n+1):
    total += x[i]
```

That is *all* it is. The three parts:
- **Under** the $\sum$: where the counter starts (and which letter is the counter).
- **Above** the $\sum$: where it stops.
- **After** the $\sum$: the thing being added each time — the loop body.

When you see $`\sum_i`$ or $`\sum_j`$ with nothing above or below, it means "over all of them" — the range is obvious from context.

**Double sums are nested loops.** $`\sum_{i}\sum_{j} A_{ij}`$ is:

```python
total = 0
for i in ...:
    for j in ...:
        total += A[i][j]
```

"Add up every entry in the grid." Nothing more.

### $\prod$ — "multiply up"

Identical to $\sum$, but multiplying instead of adding. $`\prod_{i=1}^{n} x_i = x_1 \cdot x_2 \cdots x_n`$.

▸ **Why products matter and why logs always follow them:** products of many numbers explode or vanish fast. $0.9^{100} \approx 0.000027$. This is why probability calculations are done in **log space** — $\log$ turns products into sums ($\log(ab) = \log a + \log b$), and sums don't underflow. Every time you see a $\log$ in front of a probability in this book, this is why.

### $\int$ — "add up, continuously"

$\int f(x)\,dx$ is a $\sum$ where the thing being summed over is continuous rather than discrete. If $\sum$ adds up a list, $\int$ adds up a curve. For intuition, **read $\int$ as $\sum$**; the distinction almost never matters for understanding what a formula is *for*.

The $dx$ at the end is not a multiplication — it is punctuation meaning "the variable we're sweeping over is $x$."

### $\max$, $\min$, $\arg\max$, $\arg\min$

This distinction trips up nearly everyone, and it matters.

| Notation | Returns | Example with $f(1){=}3,\ f(2){=}9,\ f(3){=}5$ |
|---|---|---|
| $`\max_x f(x)`$ | the **best value** | $9$ |
| $`\arg\max_x f(x)`$ | the **input that achieves it** | $2$ |

▸ **"arg" means "argument," i.e. the input.** $\max$ gives you the height of the mountain; $\arg\max$ gives you the GPS coordinates of the summit. In machine learning we nearly always want $`\arg\min_\theta \mathcal{L}(\theta)`$ — *"the parameter settings that make the loss smallest"* — because we want the model, not the loss value.

---

## 0.4 Greek letters — and the overloading traps

Greek letters are just variable names. There is no meaning inherent to $\theta$; it's a letter, chosen because Latin letters ran out. But **conventions** are strong, and knowing them lets you guess what a formula is about before reading it.

| Letter | Name | What it almost always means in this book |
|---|---|---|
| $\theta$ | theta | **The model's parameters.** All the weights, as one big vector. |
| $\phi,\ \varphi$ | phi | A *second* set of parameters (e.g. the encoder's, in a VAE). |
| $\eta$ | eta | **Learning rate** (step size). |
| $\alpha$ | alpha | Learning rate, or a mixing/step coefficient. |
| $\beta$ | beta | Momentum coefficient ($`\beta_1,\beta_2`$ in Adam), or an inverse temperature. |
| $\lambda$ | lambda | **Eigenvalue**, or **regularization strength**. Two very different jobs. |
| $\mu$ | mu | **Mean** (average). |
| $\sigma$ | sigma | See the trap below — **four different jobs**. |
| $\Sigma$ | capital sigma | **Covariance matrix** (when not the $\sum$ operator). |
| $\epsilon,\ \varepsilon$ | epsilon | A tiny number, or **random noise** (huge in diffusion models). |
| $\delta$ | delta | A small change, or the Kronecker delta (below). |
| $\Delta$ | capital delta | **A change in** something. $\Delta W$ = "the update to W." |
| $\gamma$ | gamma | Discount factor (RL), or a learned scale (normalization layers). |
| $\tau$ | tau | **Temperature**, or a time index, or a trajectory. |
| $\pi$ | pi | **Policy** (RL) — a strategy. Only rarely 3.14159 in this book. |
| $\rho$ | rho | Correlation, or a ratio. |
| $\kappa$ | kappa | **Condition number** (how badly-scaled a problem is). |
| $\omega$ | omega | A frequency. |
| $\nabla$ | nabla / "del" | **Gradient** — see §0.7. Not a Greek letter, but lives here in spirit. |

### ⚠ Trap 1: $\sigma$ means four different things

This is the single most confusing overload in machine learning, and this book — like every other — uses all four. **You must disambiguate from context:**

1. **Standard deviation** — $\sigma^2$ is variance. Context: probability, initialization.
2. **A singular value** — $`\sigma_i`$, $`\sigma_{\max}`$. Context: SVD, matrix norms, Chapter 1.
3. **The sigmoid function** — $\sigma(x) = 1/(1+e^{-x})$. Context: gates, binary classification.
4. **A generic activation function** — $\sigma(\cdot)$ meaning "whatever nonlinearity." Context: network definitions.

▸ **How to tell them apart instantly:** if it has a subscript index ($`\sigma_i`$, $`\sigma_{\max}`$) it's a singular value. If it's squared ($\sigma^2$) it's a standard deviation. If it has a function argument in parentheses ($\sigma(z)$) it's sigmoid-or-activation. That rule covers essentially every occurrence.

### ⚠ Trap 2: $\Sigma$ vs $\sum$

Capital sigma $\Sigma$ is a **covariance matrix**. The summation operator $\sum$ is a *taller, different glyph* that always has an index attached ($`\sum_i`$). If it has a loop index under it, it's "add up." If it stands alone next to a $\mu$, it's a covariance matrix.

### ⚠ Trap 3: $\ell$ vs $L$ vs $\mathcal{L}$

- $\ell$ (script-l) — a **per-example loss**, or a **layer index**, or the $`\ell_p`$ norm family.
- $\mathcal{L}$ (calligraphic L) — **the total loss** being minimized.
- $L$ — usually the **number of layers**.

---

## 0.5 Blackboard and calligraphic letters

Fancy-looking fonts are not decoration; the font *is* part of the meaning.

| Symbol | Read aloud | Means |
|---|---|---|
| $\mathbb{E}[X]$ | "expectation of X" | **The average value of X.** The single most important symbol in the book. |
| $\mathbb{P}(A)$ | "probability of A" | A number in $[0,1]$. |
| $\mathbb{R}$ | "the reals" | Ordinary numbers. |
| $\mathbb{1}[\text{condition}]$ | "indicator" | **1 if the condition is true, 0 if false.** A switch. |
| $\mathcal{L}$ | "script L" | The loss function. |
| $\mathcal{N}(\mu, \sigma^2)$ | "normal" | The Gaussian / bell curve with mean $\mu$, variance $\sigma^2$. |
| $\mathcal{D}$ | "script D" | The dataset, or the data distribution. |
| $\mathcal{O}(\cdot)$ | "big-O" | "grows no faster than" — see §0.10. |

### $\mathbb{E}$ deserves its own explanation

$\mathbb{E}[X]$ means **"the average of $X$, weighted by how likely each outcome is."**

$$\mathbb{E}[X] = \sum_x x \cdot p(x)$$

Read: "for each possible value $x$, multiply the value by its probability, and add them all up."

> **Analogy.** A lottery ticket pays nothing with probability $0.999$, and pays 1000 pounds with probability $0.001$. The expectation is $0(0.999) + 1000(0.001) = 1$ pound. That is the *fair price* — what you'd collect per ticket on average across a huge number of tickets. Note that you will never actually win one pound: the only possible outcomes are nothing and a thousand. **An expectation need not be a possible outcome.** It is a long-run average, not a prediction.

The subscript tells you *what is random*: $`\mathbb{E}_{x \sim p}[f(x)]`$ means "average $f(x)$ as $x$ is drawn from distribution $p$." When you see $`\mathbb{E}_{q_\phi}`$, it means "average over samples from $`q_\phi`$."

▸ **Why $\mathbb{E}$ is everywhere in ML:** you want your model to do well on *all possible data*, but you only have a finite sample. Every training objective in this book is secretly "minimize the *expected* loss," and every practical implementation replaces that expectation with an **average over a batch**. That swap — expectation → batch average — is the entire relationship between the theory and the code.

---

## 0.6 Decorations — hats, bars, tildes, and transposes

A mark placed *on* a symbol modifies its meaning. These are heavily used in this book and rarely defined.

| Decoration | Read aloud | Means |
|---|---|---|
| $\hat{y}$ | "y-hat" | **An estimate or a prediction.** The model's guess, versus the truth $y$. |
| $\bar{y}$ | "y-bar" | Usually an average — **but see the warning below.** |
| $\tilde{x}$ | "x-tilde" | A modified/perturbed/approximate version of $x$. |
| $A^\top$ | "A transpose" | Flip the matrix over its diagonal: rows become columns. |
| $A^{-1}$ | "A inverse" | The matrix that undoes $A$. |
| $`x^*`$ | "x-star" | The **optimal** value of $x$. |
| $\|x\|$ | "norm of x" | The **length** of the vector $x$. |
| $\langle a, b\rangle$ | "inner product" | Same as the dot product $a^\top b$. |

### ⚠ Trap 4: in this book, $\bar{y}$ usually means a *gradient*, not an average

This book uses the **adjoint (or "bar") notation** from automatic differentiation, in which

$$\bar{y} \;\equiv\; \frac{\partial \mathcal{L}}{\partial y}$$

read as *"the gradient of the loss with respect to $y$"* — often called the **upstream gradient** flowing backwards into $y$. So when Chapter 1 writes $\bar A = \bar y\, x^\top$, it means *"the gradient of the loss with respect to the weights equals the incoming gradient times the input, as an outer product."*

▸ This is a **space-saving convention**, not a new concept. Every time you see a bar in a backprop context, mentally expand it to $\partial \mathcal{L}/\partial(\text{that thing})$. The bar exists purely because writing the fraction 200 times is unbearable.

### The transpose $^\top$, concretely

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{pmatrix} \quad\Rightarrow\quad A^\top = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6\end{pmatrix}$$

Shape $2\times 3$ becomes $3 \times 2$. Its main job in this book is **making shapes line up so a multiplication is legal.** When you see a $^\top$ appear in a derivation, 90% of the time the reason is bookkeeping, not insight.

### The Kronecker delta $`\delta_{ij}`$

$$\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{otherwise}\end{cases}$$

It's an `if i == j` written as a symbol. It shows up in derivatives because $`\partial x_i/\partial x_j`$ is $1$ when they're the same variable and $0$ when they're different. That's all it is ever doing.

---

## 0.7 Calculus notation — derivatives and gradients

### The derivative, in one sentence

$\dfrac{df}{dx}$ means **"if I nudge $x$ up by a tiny amount, how much does $f$ move, per unit of nudge?"** It is a *sensitivity*, or a slope.

### $\partial$ — the partial derivative

$\dfrac{\partial f}{\partial x}$ is the same question when $f$ depends on several variables: **"nudge $x$, hold everything else fixed, how much does $f$ move?"**

The curly $\partial$ ("partial" or "der") signals *"there are other variables, and I'm freezing them."* That is the only difference from $d$.

### $\nabla$ — the gradient

$`\nabla_\theta \mathcal{L}`$ ("grad theta L," or "del theta L") is simply **all the partial derivatives collected into one vector, one per parameter**:

$$\nabla_\theta \mathcal{L} = \left(\frac{\partial \mathcal{L}}{\partial \theta_1},\ \frac{\partial \mathcal{L}}{\partial \theta_2},\ \dots,\ \frac{\partial \mathcal{L}}{\partial \theta_p}\right)$$

▸ **The gradient points in the direction of steepest increase.** So $`-\nabla_\theta\mathcal{L}`$ points in the direction of steepest *decrease*, which is why every learning algorithm in this book is some variation of $`\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}`$ — *"take a small step downhill."*

> **Analogy.** You're standing on a foggy hillside and can only feel the ground under your feet. The gradient is what your feet tell you: which way is uphill, and how steep. Gradient descent is the strategy "feel which way is downhill, take a step, repeat." The learning rate $\eta$ is how big a step you take. Too small and you're there all week; too big and you stride straight over the valley and up the other side.

**The subscript on $\nabla$ says what you're differentiating with respect to.** $`\nabla_\theta`$ = with respect to parameters (for learning). $`\nabla_x`$ = with respect to the input (for adversarial examples and saliency maps).

### The shape rule, which catches most bugs

▸ **A gradient always has the same shape as the thing you differentiated with respect to.** If $\theta$ is a $512\times 768$ matrix, then $`\nabla_\theta \mathcal{L}`$ is a $512 \times 768$ matrix. Always. No exceptions. This is worth more than any other single debugging heuristic in deep learning.

---

## 0.8 The four ways to multiply, which beginners constantly confuse

This section alone resolves an enormous fraction of early confusion. "Multiplication" means four different operations depending on the shapes, and the notation is nearly identical.

**Let $a, b \in \mathbb{R}^n$ (two vectors of the same length).**

### 1. Dot product / inner product — vectors in, **one number** out

$$a^\top b \;=\; \langle a, b\rangle \;=\; \sum_i a_i b_i$$

Multiply matching entries, add them all up. Result: **a single number.**

▸ **What it means:** a measure of **alignment**. Big and positive = the two vectors point the same way. Zero = perpendicular ("orthogonal," unrelated). Negative = opposing. *Every similarity score in machine learning is a dot product* — attention, embeddings, retrieval, all of it.

### 2. Outer product — vectors in, **a matrix** out

$$a b^\top \in \mathbb{R}^{n \times n}, \qquad (ab^\top)_{ij} = a_i b_j$$

Every entry of $a$ times every entry of $b$, arranged in a grid. Result: **a matrix.**

Note how close $a^\top b$ (a number) and $a b^\top$ (a matrix) look. **The transpose position is the entire difference.** Reading order: the "inner" dimensions must match, and the "outer" ones survive.

### 3. Elementwise (Hadamard) product $\odot$ — same shape in, **same shape** out

$$(a \odot b)_i = a_i b_i$$

Multiply matching entries and **keep them separate** (don't add). Result: same shape as the inputs. This is what `*` does in NumPy/PyTorch. It's how gates work in LSTMs: a gate vector of numbers in $[0,1]$ multiplies a signal elementwise to pass some parts and block others.

### 4. Matrix–vector product — matrix and vector in, **vector** out

$$(Ax)_i = \sum_j A_{ij}x_j$$

Result: a vector whose length is the number of *rows* of $A$. This is what a neural network layer does.

▸ **The shape rule for all matrix multiplication:** $(m \times n) \cdot (n \times k) = (m \times k)$. **The inner numbers must match and they disappear; the outer numbers survive.** If the inner numbers don't match, the operation is undefined — and that is the error message you will see most often in your career.

| Expression | Shapes | Result |
|---|---|---|
| $a^\top b$ | $(1{\times}n)(n{\times}1)$ | scalar — a number |
| $a b^\top$ | $(n{\times}1)(1{\times}n)$ | $n\times n$ matrix |
| $a \odot b$ | $(n)(n)$ | $n$-vector |
| $Ax$ | $(m{\times}n)(n{\times}1)$ | $m$-vector |

---

## 0.9 Probability notation

| Notation | Read aloud | Means |
|---|---|---|
| $p(x)$ | "p of x" | How likely $x$ is. |
| $p(x \mid y)$ | "p of x **given** y" | Likelihood of $x$ **once you already know** $y$. |
| $x \sim p$ | "x is **drawn from** p" | $x$ is a random sample from distribution $p$. |
| $p(x, y)$ | "joint" | Likelihood of $x$ and $y$ together. |
| $\mathcal{N}(\mu,\sigma^2)$ | "normal / Gaussian" | Bell curve, centred at $\mu$, spread $\sigma$. |
| $\propto$ | "is proportional to" | Equal up to a constant factor we don't care about. |
| i.i.d. | "eye-eye-dee" | **Independent and identically distributed** — each sample drawn separately from the same distribution, none influencing the others. |

### ⚠ Trap 5: the vertical bar $\mid$ means "given," not "divide" or "absolute value"

$p(x \mid y)$ is a **conditional probability**: *"the probability of $x$, in a world where we already know $y$ happened."*

> **Analogy.** $p(\text{rain})$ might be 0.2. But $p(\text{rain} \mid \text{dark clouds})$ is 0.7. The bar means "given that," and it re-weights the world to only the cases where the right-hand side is true. Knowing something changes the odds.

▸ **The tilde $\sim$ is not "approximately."** In this book $x \sim \mathcal{N}(0,1)$ means "$x$ is sampled from a standard normal," not "$x$ is roughly a normal." For "approximately equal," the book uses $\approx$.

### $\arg\max$ in probability, and the two big estimation ideas

- **MLE (maximum likelihood):** $`\arg\max_\theta p(\text{data}\mid\theta)`$ — *"pick the parameters that make the data I actually observed as unsurprising as possible."*
- **Posterior:** $p(\theta \mid \text{data})$ — *"having seen the data, what do I now believe about the parameters?"*

---

## 0.10 Asymptotics: big-O

$\mathcal{O}(f(n))$ means **"grows no faster than $f(n)$, ignoring constants."** It answers "how does the cost scale as the problem gets bigger?" and deliberately throws away everything else.

| Notation | Meaning | Attention example |
|---|---|---|
| $\mathcal{O}(n)$ | linear — double the input, double the cost | Sequence length vs. an RNN step |
| $\mathcal{O}(n^2)$ | quadratic — double the input, **4×** the cost | Self-attention over sequence length |
| $\mathcal{O}(\log n)$ | logarithmic — barely grows | Binary search |

▸ **Why $\mathcal{O}(T^2)$ is the single most consequential complexity in this book:** attention compares every token to every other token. Double the context from 4k to 8k and you don't double the attention cost — you **quadruple** it. Chapters 11, 12, and 17 are, to a large extent, a sustained engineering campaign against that one exponent.

Constants are dropped because they don't change the *shape* of the growth: $\mathcal{O}(3n^2 + 500n)$ is written $\mathcal{O}(n^2)$, since for large $n$ the squared term dominates everything else. (In practice, of course, a constant of 500 matters enormously — big-O tells you about scaling, not about speed.)

---

## 0.11 Relations and arrows

| Symbol | Read aloud | Means |
|---|---|---|
| $\approx$ | "approximately equals" | Close enough for the argument being made. |
| $\propto$ | "proportional to" | Equal after multiplying by some constant. |
| $\equiv$, $:=$ | "is defined as" | A definition, not a derived fact. |
| $\le,\ \ge$ | "at most / at least" | Bounds. Most theory results are inequalities, not equalities. |
| $\ll,\ \gg$ | "much less / much greater" | An informal but meaningful size gap. |
| $\to$ | "tends to" / "maps to" | A limit, or a function's direction. |
| $\leftarrow$ | "is assigned" | **An update.** $\theta \leftarrow \theta - \eta g$ is an assignment statement, like `theta = theta - eta*g`. |
| $\Rightarrow$ | "implies" | If the left is true, the right follows. |
| $\perp$ | "is orthogonal to" | Perpendicular; dot product zero; intuitively "unrelated." |
| $\forall$ | "for all" | Universally true. |
| $\exists$ | "there exists" | At least one exists. |
| $\iff$ | "if and only if" | Each implies the other; they're equivalent. |
| $\square$, $\blacksquare$, ∎ | "QED" | End of proof. Purely a full stop. |

▸ **$\leftarrow$ versus $=$ is worth internalizing.** $\theta \leftarrow \theta - \eta\nabla\mathcal{L}$ is not an equation to solve — it's a *line of code*. It says "compute the right side, then overwrite $\theta$ with it." If you read it as an equation you'd conclude $\eta \nabla \mathcal{L} = 0$, which is nonsense. Training loops are assignments, not equalities.

---

## 0.12 Worked example: reading a  intimidating formula

Here's one of the harder-looking equations in the book, the **Gaussian density** from §1.3.3. We'll apply the six habits.

$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\!\left(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\right)$$

**Habit 1 — what's being defined?** $p(x)$: a probability density. The right side is the recipe.

**Habit 2 — read the symbols as words.** "p of x equals one over (two pi to the d-over-two, times the square root of the determinant of Sigma), times e to the minus one-half times (x minus mu) transpose, Sigma inverse, (x minus mu)."

**Habit 3 — what varies?** Only $x$. Both $\mu$ (the centre) and $\Sigma$ (the spread) are fixed parameters.

**Habit 4 — shapes.** $x, \mu \in \mathbb{R}^d$. $\Sigma$ is $d\times d$. So $(x-\mu)^\top \Sigma^{-1}(x-\mu)$ is $(1{\times}d)(d{\times}d)(d{\times}1) = $ **a single number.** Good — it had better be, since $\exp$ eats a number.

**Habit 5 — make it small.** Set $d = 1$, $\mu = 0$, $\Sigma = 1$:

$$p(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$$

That's the familiar bell curve. **The scary version is the same thing in $d$ dimensions.**

**Habit 6 — what makes it big or zero?** The exponent is negative, so $p$ is largest when $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ is *smallest* — that is, when $x = \mu$. Density peaks at the mean and decays as you move away. Correct.

**Now the meaning of each piece:**

| Piece | Job |
|---|---|
| $(x - \mu)$ | "How far is this point from the centre?" |
| $\Sigma^{-1}$ | "Measure that distance in units of the distribution's own spread." Two units away is unremarkable if the data is wide, and extraordinary if it's narrow. |
| $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ | A **squared distance, scale-corrected** (the Mahalanobis distance). One number: "how many standard deviations out is this?" |
| $\exp(-\tfrac12 \cdot)$ | Turn "distance" into "likelihood": far away → exponentially less likely. |
| $\frac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}$ | A **normalizing constant**, whose only job is to make the total probability integrate to 1. It contains no insight. Ignore it on first reading. |

▸ **The whole formula in one sentence:** *"How likely is this point? Measure how far it is from the centre in units of the spread, and let likelihood fall off exponentially with that squared distance."*

**The transferable lesson:** most intimidating formulas are a **simple core** wrapped in a **normalizing constant** and **generalized to $d$ dimensions**. Strip the constant, set $d=1$, and read what's left.

---

## 0.12b Examples and non-examples of the core ideas

A definition tells you what something *is*. Examples tell you what it *looks like*. **Non-examples — the near-misses — tell you where the boundary is**, and that is where understanding actually lives. Anyone can recite a definition; only someone who understands can say why the lookalike fails.

This section does that for the ideas the rest of the book leans on hardest.

### Is it linear?

"Linear" is the most over-assumed word in machine learning. A function is linear if two things hold: scaling the input scales the output ($f(ax) = af(x)$), and adding inputs adds outputs ($f(a+b) = f(a)+f(b)$).

**✅  examples**

| Example | Why it qualifies |
|---|---|
| $f(x) = 3x$ | Double $x$, output doubles. Add inputs, outputs add. |
| $f(x) = Ax$ for a matrix $A$ | The definition of a linear map |
| Summing a batch of gradients | $`\sum(a_i + b_i) = \sum a_i + \sum b_i`$ |
| Expectation, $\mathbb{E}[aX+bY]$ | Linear **always**, even for dependent variables |

**❌ Near-misses — called "linear," but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| $f(x) = 3x + 5$ | $f(2x) = 6x+5 \ne 2f(x) = 6x+10$. The constant breaks it. | **Affine** — linear plus a shift |
| A "linear layer" $Wx + b$ | Same problem: the bias $b$ | Affine. The name is a universal abuse of terminology. |
| $f(x) = x^2$ | $f(2x) = 4x^2$, not $2x^2$ | Quadratic |
| Variance, $\mathrm{Var}(aX)$ | $= a^2\mathrm{Var}(X)$, not $a\mathrm{Var}(X)$ | Quadratic in the scaling |
| Linear regression on $x^2$ | The *model* is linear in its **parameters**, not in $x$ | Linear model, nonlinear features |

▸ **The boundary:** a straight line through the origin is linear; a straight line that misses the origin is affine. Nearly every "linear layer" in deep learning is affine, and the field simply doesn't care — but the distinction matters the moment you do algebra, because affine maps don't compose as cleanly.

> **Common misconception.** *"Stacking linear layers makes a deeper, more powerful model."* It does not. $`W_2(W_1x) = (W_2W_1)x`$, and $`W_2W_1`$ is just another matrix. **A hundred stacked linear layers have exactly the expressive power of one.** This is precisely why activation functions exist — without a nonlinearity between them, depth buys you nothing at all. It is the single most important reason ReLU is in your network.

### Is it a probability distribution?

**✅  examples**

| Example | Why it qualifies |
|---|---|
| $(0.7,\ 0.2,\ 0.1)$ | Non-negative, sums to exactly 1 |
| Output of a softmax | Guaranteed both properties by construction |
| A fair die: $(1/6)\times 6$ | Sums to 1 |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Logits $(2.0,\ 1.0,\ 0.0)$ | Doesn't sum to 1; can be negative | **Unnormalized scores.** Needs a softmax. |
| $(0.5,\ 0.6,\ 0.3)$ | Sums to 1.4 | Nothing — an error |
| Sigmoid outputs $(0.8,\ 0.7,\ 0.9)$ | Each is in $[0,1]$ but they sum to 2.4 | **Three independent probabilities**, correct for multi-label |
| A Gaussian density value $p(x) = 1.6$ | Densities can exceed 1 | A **density**, not a probability. Only its *integral* must be 1. |

▸ **The boundary:** a probability distribution needs non-negativity **and** a total of exactly 1. Individual numbers being in $[0,1]$ is not enough.

> **Common misconception.** *"A probability density can't be greater than 1."* It absolutely can. A narrow Gaussian concentrated in a tiny interval has density values far above 1 — what must equal 1 is the **area under the curve**, not the height. This confuses people because for *discrete* distributions the values  are probabilities and  are capped at 1.

### Independent versus uncorrelated

**✅  independent**

| Example | Why |
|---|---|
| Two separate coin flips | Neither tells you anything about the other |
| A random seed and the day of the week | No relationship whatsoever |

**❌ Near-misses**

| Looks independent | Why it isn't | Reality |
|---|---|---|
| $X$ uniform on $[-1,1]$ and $Y = X^2$ | Covariance is exactly **0** | Perfectly **dependent** — $Y$ is a *function* of $X$! |
| Two features with correlation 0.00 | Correlation only detects **straight-line** relationships | Could be perfectly related in a U-shape |

▸ **The boundary:** independence means *knowing one tells you nothing about the other*. Zero correlation means only that there's no **linear** trend. **Independence implies zero correlation; the reverse is false.** The $Y = X^2$ case is the standard counterexample and worth memorizing — as $X$ moves away from zero in either direction $Y$ rises, so the positive and negative sides cancel and covariance vanishes despite total dependence.

### Is that gradient descent step valid?

**✅  examples**

| Example | Why |
|---|---|
| $`\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}`$ | Move *against* the gradient — downhill |
| Adam, RMSProp, momentum | All of the above with a smarter step size |

**❌ Near-misses**

| Looks right | Why it's wrong |
|---|---|
| $`\theta \leftarrow \theta + \eta\nabla_\theta\mathcal{L}`$ | **Sign error** — this is gradient *ascent*. Your loss will climb. |
| $`\theta \leftarrow \theta - \eta\nabla_x\mathcal{L}`$ | Differentiating w.r.t. the **input**, not the parameters. This makes adversarial examples, not training. |
| $\theta = \theta - \eta\nabla\mathcal{L}$ read as an equation | It's an **assignment**, not an equality. Reading `=` as math gives nonsense. |

▸ **The boundary:** the gradient points **uphill**, so descent subtracts it. The subscript on $\nabla$ tells you what you're changing — parameters (learning) or inputs (attacking).

### Is that "average" an expectation?

**✅  examples**

| Example | Why |
|---|---|
| $`\mathbb{E}[X] = \sum_x x\,p(x)`$ | Weighted by probability |
| Mean loss over an i.i.d. batch | An **estimate** of the expectation |

**❌ Near-misses**

| Looks like an expectation | Why it isn't |
|---|---|
| Mean over a **biased** sample | Estimates the wrong distribution's expectation |
| Mean of the last 100 training losses | The model changed during those 100 steps — the distribution isn't fixed |
| A running average with momentum | An **exponentially weighted** average; recent points count more |

> **Common misconception.** *"The expected value is the value I should expect."* It frequently isn't a possible outcome at all. The expected number of children per family might be 1.8; no family has 1.8 children. A lottery ticket with an expectation of one pound pays either nothing or a thousand, never one. **Expectation is a long-run average, not a prediction.**

### Does a bigger number mean a better model?

**❌ Near-misses that fool practitioners constantly**

| Observation | Why it may mean nothing |
|---|---|
| Validation loss dropped 0.03 | If your standard error is 0.03, that's **noise** (§0.9 and Chapter 3) |
| Accuracy 99% | On a dataset that's 99% one class, predicting that class always scores 99% |
| Training loss near zero | Memorization. Says nothing about generalization. |
| Cross-entropy of 1.5 | Meaningless until compared to $\log K$ — excellent for 50,000 classes, terrible for 2 |
| Model B beat model A once | One run. Seed variance alone can produce that gap. |

▸ **The boundary:** a number is evidence only when you know its **baseline** and its **error bar**. This is the most common way practitioners fool themselves, and Chapter 3 exists entirely to address it.

---

## 0.13 Every abbreviation in this book, spelled out

Machine learning is dense with acronyms, and most texts assume you already know them. You don't have to. This is the complete list of every abbreviation used three or more times across these thirty-five chapters, with its full form.

**Read the full form aloud once.** Most acronyms stop being intimidating the moment you hear what they stand for — "GAN" is mysterious, "Generative Adversarial Network" describes itself.

### ⚠ First: the dangerous collisions

A handful of acronyms mean **two completely different things** in this book. Context always disambiguates, but knowing they collide prevents real confusion:

| Acronym | Meaning A | Meaning B |
|---|---|---|
| **ANN** | **A**pproximate **N**earest **N**eighbour (search) | **A**rtificial **N**eural **N**etwork |
| **GB** | **G**radient **B**oosting | **G**iga**b**yte |
| **PI** | **P**osition **I**nterpolation (context extension) | **P**rediction **I**nterval |
| **IS** | **I**nception **S**core | **I**mportance **S**ampling |
| **MI** | **M**utual **I**nformation | **M**embership **I**nference |
| **ID** | **I**n-**D**istribution | **I**ntrinsic **D**imension |
| **DP** | **D**ata **P**arallelism | **D**ynamic **P**rogramming |
| **ASR** | **A**utomatic **S**peech **R**ecognition | **A**ttack **S**uccess **R**ate |
| **SE** | **S**tandard **E**rror | **S**queeze-and-**E**xcitation |

### Mathematics and statistics

| Short | Full form |
|---|---|
| CE | Cross-Entropy |
| CV | Cross-Validation |
| ELBO | Evidence Lower BOund |
| EM | Expectation–Maximization |
| EMA | Exponential Moving Average |
| ERM | Empirical Risk Minimization |
| GGN | Generalized Gauss–Newton |
| i.i.d. | independent and identically distributed |
| JSD | Jensen–Shannon Divergence |
| KKT | Karush–Kuhn–Tucker (optimality conditions) |
| KL | Kullback–Leibler (divergence) |
| LOO | Leave-One-Out (cross-validation) |
| MAE | Mean Absolute Error |
| MAP | Maximum A Posteriori |
| MC | Monte Carlo |
| MCMC | Markov Chain Monte Carlo |
| MDL | Minimum Description Length |
| MI | Mutual Information |
| MLE | Maximum Likelihood Estimation |
| MSE | Mean Squared Error |
| NLL | Negative Log-Likelihood |
| ODE | Ordinary Differential Equation |
| OLS | Ordinary Least Squares |
| OOD | Out-Of-Distribution |
| PAC | Probably Approximately Correct |
| PCA | Principal Component Analysis |
| PMI | Pointwise Mutual Information |
| PSD | Positive Semi-Definite |
| SD | Standard Deviation |
| SDE | Stochastic Differential Equation |
| SE | Standard Error |
| SNR | Signal-to-Noise Ratio |
| SVD | Singular Value Decomposition |
| VC | Vapnik–Chervonenkis (dimension) |
| VLB | Variational Lower Bound |

### Optimization

| Short | Full form |
|---|---|
| AdamW | Adam with decoupled **W**eight decay |
| GD | Gradient Descent |
| K-FAC | Kronecker-Factored Approximate Curvature |
| L-BFGS | Limited-memory Broyden–Fletcher–Goldfarb–Shanno |
| LR | Learning Rate |
| RAdam | Rectified Adam |
| SAM | Sharpness-Aware Minimization |
| SGD | Stochastic Gradient Descent |
| WSD | Warmup–Stable–Decay (schedule) |

### Neural network architecture

| Short | Full form |
|---|---|
| BN | Batch Normalization |
| BPTT | Backpropagation Through Time |
| CNN | Convolutional Neural Network |
| CTC | Connectionist Temporal Classification |
| FC | Fully Connected (layer) |
| FFN | Feed-Forward Network |
| GELU | Gaussian Error Linear Unit |
| GNN | Graph Neural Network |
| GCN | Graph Convolutional Network |
| GIN | Graph Isomorphism Network |
| GRU | Gated Recurrent Unit |
| LN | Layer Normalization |
| LSTM | Long Short-Term Memory |
| MLP | Multi-Layer Perceptron |
| NN | Neural Network |
| ReLU | Rectified Linear Unit |
| RNN | Recurrent Neural Network |
| SwiGLU | Swish-Gated Linear Unit |
| WL | Weisfeiler–Lehman (graph isomorphism test) |

### Transformers and large language models

| Short | Full form |
|---|---|
| ALiBi | Attention with Linear Biases |
| BERT | Bidirectional Encoder Representations from Transformers |
| BPE | Byte-Pair Encoding |
| CLS | Classification (token) |
| GPT | Generative Pre-trained Transformer |
| GQA | Grouped-Query Attention |
| KV | Key–Value (cache) |
| LLM | Large Language Model |
| LM | Language Model |
| MHA | Multi-Head Attention |
| MLA | Multi-head Latent Attention |
| MoE | Mixture of Experts |
| MQA | Multi-Query Attention |
| NoPE | No Positional Encoding |
| OV | Output–Value (circuit) |
| QK | Query–Key (circuit) |
| RoPE | Rotary Position Embedding |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| YaRN | Yet another RoPE extensioN |

### Training at scale and systems

| Short | Full form |
|---|---|
| bf16 | brain floating point, 16-bit |
| DP | Data Parallelism |
| FLOP | FLoating-point OPeration |
| fp16 / fp32 | floating point, 16-/32-bit |
| FSDP | Fully Sharded Data Parallel |
| GPU | Graphics Processing Unit |
| HBM | High-Bandwidth Memory |
| MFU | Model FLOPs Utilization |
| PP | Pipeline Parallelism |
| SRAM | Static Random-Access Memory |
| TP | Tensor Parallelism |
| ZeRO | Zero Redundancy Optimizer |

### Post-training, alignment, and efficiency

| Short | Full form |
|---|---|
| AWQ | Activation-aware Weight Quantization |
| CFG | Classifier-Free Guidance |
| DoRA | Weight-**D**ecomposed L**oRA** |
| DPO | Direct Preference Optimization |
| GPTQ | Generative Pre-trained Transformer Quantization |
| GRPO | Group Relative Policy Optimization |
| KTO | Kahneman–Tversky Optimization |
| LoRA | Low-Rank Adaptation |
| ORPO | Odds Ratio Preference Optimization |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |
| QLoRA | Quantized Low-Rank Adaptation |
| RLAIF | Reinforcement Learning from AI Feedback |
| RLHF | Reinforcement Learning from Human Feedback |
| RLVR | Reinforcement Learning with Verifiable Rewards |
| RM | Reward Model |
| SFT | Supervised Fine-Tuning |
| SimPO | Simple Preference Optimization |

### Reinforcement learning

| Short | Full form |
|---|---|
| A2C / A3C | (Asynchronous) Advantage Actor–Critic |
| BC | Behaviour Cloning |
| CQL | Conservative Q-Learning |
| DDPG | Deep Deterministic Policy Gradient |
| DQN | Deep Q-Network |
| GAE | Generalized Advantage Estimation |
| IQL | Implicit Q-Learning |
| MDP | Markov Decision Process |
| PPO | Proximal Policy Optimization |
| SAC | Soft Actor–Critic |
| SARSA | State–Action–Reward–State–Action |
| TD | Temporal Difference |
| TRPO | Trust Region Policy Optimization |
| UCB | Upper Confidence Bound |

### Generative models

| Short | Full form |
|---|---|
| DDIM | Denoising Diffusion Implicit Model |
| DDPM | Denoising Diffusion Probabilistic Model |
| FID | Fréchet Inception Distance |
| GAN | Generative Adversarial Network |
| IS | Inception Score |
| NCSN | Noise Conditional Score Network |
| VAE | Variational AutoEncoder |
| VQ-VAE | Vector-Quantized Variational AutoEncoder |
| WGAN | Wasserstein Generative Adversarial Network |

### Classical machine learning

| Short | Full form |
|---|---|
| CART | Classification And Regression Trees |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| EFB | Exclusive Feature Bundling |
| GBDT | Gradient-Boosted Decision Trees |
| GMM | Gaussian Mixture Model |
| GOSS | Gradient-based One-Side Sampling |
| HDBSCAN | Hierarchical DBSCAN |
| ICA | Independent Component Analysis |
| LASSO | Least Absolute Shrinkage and Selection Operator |
| NMF | Non-negative Matrix Factorization |
| RBF | Radial Basis Function |
| SVM | Support Vector Machine |
| t-SNE | t-distributed Stochastic Neighbor Embedding |
| UMAP | Uniform Manifold Approximation and Projection |

### Representation learning, retrieval, and evaluation

| Short | Full form |
|---|---|
| ANN | Approximate Nearest Neighbour |
| BLEU | BiLingual Evaluation Understudy |
| BYOL | Bootstrap Your Own Latent |
| CLIP | Contrastive Language–Image Pre-training |
| DINO | self-**DI**stillation with **NO** labels |
| ECE | Expected Calibration Error |
| HNSW | Hierarchical Navigable Small World |
| InfoNCE | Information Noise-Contrastive Estimation |
| IVF-PQ | Inverted File index with Product Quantization |
| LSH | Locality-Sensitive Hashing |
| PR-AUC | Precision–Recall Area Under Curve |
| RAG | Retrieval-Augmented Generation |
| ROC-AUC | Receiver Operating Characteristic Area Under Curve |
| SAE | Sparse AutoEncoder |
| SHAP | SHapley Additive exPlanations |
| SSL | Self-Supervised Learning |

### Phenomena and theory

| Short | Full form |
|---|---|
| ETF | Equiangular Tight Frame |
| IRM | Invariant Risk Minimization |
| LTH | Lottery Ticket Hypothesis |
| NTK | Neural Tangent Kernel |
| PGD | Projected Gradient Descent |

### Datasets you'll see referenced

| Short | Full form |
|---|---|
| CIFAR-10 / -100 | Canadian Institute For Advanced Research (image datasets) |
| ImageNet | (not an acronym — a large labelled image database) |
| MNIST | Modified National Institute of Standards and Technology (handwritten digits) |
| SVHN | Street View House Numbers |

---

## 0.14 A note on why the notation is worth learning

It is fair to ask why any of this needs symbols at all. Two honest reasons:

1. **Precision.** "The average error is small" is ambiguous. $`\mathbb{E}_{x\sim\mathcal{D}}[\mathcal{L}(f_\theta(x), y)] < \epsilon`$ is not. Notation forces you to say *which* average, over *what* distribution, and *how* small.
2. **Compression.** The backprop rule $\bar{A} = \bar{y}x^\top$ is six characters and takes a paragraph to state in English. Once fluent, you read the six characters faster — and, crucially, you can *manipulate* them. You cannot do algebra on a paragraph.

But notation is a **tool for thinking**, not a badge. If a formula in this book ever feels impenetrable, the failure is in the exposition, not in you. Come back to this chapter, apply the six habits, set $d = 1$, and try again.

---

## Check for Understanding

**Every formula in this book is a sentence: $\sum$ and $\prod$ are loops, $\mathbb{E}$ is an average, $\nabla$ is "which way is uphill," a bar is a gradient flowing backwards, the vertical bar means "given," and any equation that looks impossible becomes obvious the moment you set the dimension to one.**

---

**Next:** [Chapter 01 — Mathematical Foundations](01-mathematical-foundations.md)
