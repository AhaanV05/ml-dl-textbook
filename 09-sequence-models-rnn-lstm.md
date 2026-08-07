# Chapter 9 — Sequence Models: RNNs, LSTMs & Seq2Seq

> **Prerequisites:** Ch. 6.
> **Why this chapter still matters:** attention was invented to fix a specific, well-defined failure of this architecture. You cannot understand *why* the transformer looks the way it does without knowing what it replaced. This is also where the vanishing-gradient problem is at its most extreme and most instructive.

> **New to the notation?** If symbols like $\in$, $\sum$, $\prod$, $\mathbb{E}$, $\nabla$, $\odot$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $t$ | "t" | The **timestep** — which word, which audio frame, which position in the sequence |
| $h_t$ | "h sub t" | The **hidden state** after step $t$ — everything the model remembers so far, in one fixed-size vector |
| $x_t$ | "x sub t" | The input arriving at step $t$ |
| $W_{hh},\ W_{xh},\ W_{hy}$ | "W h-h", "W x-h", "W h-y" | Weight matrices. **First letter = source, second = destination**: $W_{xh}$ maps input → hidden |
| $\sigma$ | "sigma" | ⚠ Here it means both a generic activation *and* the logistic sigmoid. Ch. 0, Trap 1 |
| $\prod_{j=k+1}^{t}$ | "product from j = k+1 to t" | A `for` loop that **multiplies** instead of adding |
| $\partial h_j/\partial h_{j-1}$ | "partial h-j by partial h-j-minus-1" | How much step $j$'s state moves if step $j{-}1$'s state is nudged |
| $\mathrm{diag}(v)$ | "diag of v" | A matrix with $v$ on the diagonal and zeros everywhere else |
| $\lambda_1$ | "lambda one" | The **largest singular value** of $W_{hh}$ — its biggest available stretch |
| $\gamma=\max\lvert\sigma'\rvert$ | "gamma equals max of the absolute value of sigma-prime" | The steepest slope the activation function ever has |
| $\odot$ | "elementwise product" / "Hadamard" | Multiply matching entries and keep them separate (Ch. 0 §0.8) |
| $c_t$ | "c sub t" | The LSTM's **cell state** — the long-term memory |
| $f_t,\ i_t,\ o_t$ | "f-t, i-t, o-t" | Forget, input and output **gates**: vectors of soft switches in $(0,1)$ |
| $\tilde c_t$ | "c-tilde sub t" | The **candidate** value — what would be written, if writing happens |
| $[h_{t-1},x_t]$ | "h-t-minus-1 concat x-t" | **Concatenation**, not multiplication: stack two vectors into one longer one |
| $z_t,\ r_t$ | "z-t, r-t" | The GRU's update and reset gates |
| $e_{tj},\ \alpha_{tj}$ | "e-t-j, alpha-t-j" | Attention **score** and attention **weight** for decoder step $t$, encoder position $j$ |
| $c_t$ (in §9.6) | "c sub t" | ⚠ Reused: here the attention **context vector**, not the LSTM cell state |
| $\varnothing$ | "blank" | CTC's blank symbol — "emit nothing at this frame" |
| $\mathcal{B}^{-1}(y)$ | "B-inverse of y" | The **pre-image**: every alignment that collapses to the label $y$ |
| $\mathcal{O}(T)$, $\mathcal{O}(1)$ | "big-O of T", "big-O of one" | How cost scales with sequence length (Ch. 0 §0.10) |

**Full forms for the abbreviations in this chapter:**

| Short | Full form |
|---|---|
| RNN | Recurrent Neural Network |
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| BPTT | Backpropagation Through Time |
| Seq2Seq | Sequence-to-Sequence |
| NMT | Neural Machine Translation |
| BLEU | BiLingual Evaluation Understudy (a translation-quality score) |
| CTC | Connectionist Temporal Classification |
| HMM | Hidden Markov Model |
| NER | Named Entity Recognition |
| ASR | Automatic Speech Recognition |
| KV cache | Key–Value cache (a transformer's stored per-token state) |
| RLHF | Reinforcement Learning from Human Feedback |
| SSM | State-Space Model |
| S4 | Structured State Space Sequence model |
| MLP | Multi-Layer Perceptron |
| EMA | Exponential Moving Average |

---

## 9.1 The recurrent neural network

### The one-line idea

Carry a hidden state forward through time, updating it with each new input, so the state summarizes everything seen so far.

### The analogy

Reading a novel while keeping only a single index card of notes. At every page you rewrite the card from scratch, using the old card plus the new page. The card's fixed size is the whole problem: by chapter 30, everything from chapter 1 has been overwritten many times.

### The math

▸ $$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b),\qquad y_t = W_{hy}h_t$$

Parameters are **shared across time** — the same $W_{hh}$ at every step. This is the temporal analogue of a convolution's weight sharing, and it gives the same benefit: the model can handle sequences of any length with a fixed parameter count.

#### Reading the RNN recurrence in plain English

$$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b),\qquad y_t = W_{hy}h_t$$

Read aloud: *"the new state is a squashed mixture of the old state and the new input; the output is a linear readout of the state."*

Every symbol:

| Symbol | Read aloud | What it is |
|---|---|---|
| $t$ | "t" | Which step you're on. Word 7 of a sentence, frame 250 of some audio |
| $h_t \in \mathbb{R}^d$ | "h sub t, in R-d" | A list of $d$ numbers. **The index card.** Typically $d = 256$ to $1024$ |
| $h_{t-1}$ | "h sub t minus 1" | The state you had a moment ago |
| $x_t$ | "x sub t" | This step's input — an embedded word, an audio frame |
| $W_{hh}$ | "W h-h" | **Square**, $d\times d$: how the old state is transformed on the way to becoming the new one |
| $W_{xh}$ | "W x-h" | How the new input is folded in |
| $W_{hy}$ | "W h-y" | How the state is read out into a prediction |
| $b$ | "b" | Bias |
| $\sigma$ | "sigma" | The activation — $\tanh$ in practice. **Not** a standard deviation here |

**The subscript convention is worth stating explicitly** because it is never stated: the **first letter is the source, the second is the destination**. $W_{xh}$ carries information *from* $x$ *to* $h$. Once you know that, you can read any RNN diagram in the literature.

> **Analogy (extending the book's index card).** You are reading a novel and allowed one index card. Every page, you must rewrite the card from scratch using only (a) the old card and (b) the page in front of you. $W_{hh}$ is your habit for rewriting the old notes; $W_{xh}$ is your habit for summarizing new material. The card never gets bigger. That fixed size is the whole story of this chapter.

**Now set $d=1$ and watch what happens.** Let $W_{hh}=0.5$, $W_{xh}=1$, $b=0$, $\sigma =$ identity, and feed a single pulse: $x = (1,0,0,0,\dots)$. Then:

| $t$ | $h_t$ |
|---|---|
| 1 | $0.5\cdot 0 + 1 = 1$ |
| 2 | $0.5\cdot 1 = 0.5$ |
| 3 | $0.25$ |
| 4 | $0.125$ |
| 10 | $0.5^{9} \approx 0.002$ |

The memory of that first input **halves every single step**. Now set $W_{hh} = 1.5$ instead: $h = 1,\ 1.5,\ 2.25,\ 3.375,\dots$, reaching $1.5^{9}\approx 38$ by step 10 and $1.5^{50}\approx 6\times10^{8}$ by step 50.

▸ **This one-dimensional toy contains the entire chapter.** The same number that decides *how long you remember* also decides *whether the state stays bounded*. Below 1 you forget exponentially. Above 1 you explode exponentially. Exactly 1 remembers perfectly but can never adapt — the state just accumulates unchanged. **There is no good setting, because one number is being asked to do two incompatible jobs.** Everything after this section — the LSTM, the GRU, attention, residual connections — is a way of splitting those two jobs apart.

**Why sharing $W_{hh}$ across time is not a compromise.** Like a convolution's kernel (Ch. 8), the same weights are reused at every position, so a fixed parameter count handles a sequence of any length, and every timestep in every training example contributes evidence about the *same* matrix. A model with per-timestep weights could not process a sentence longer than the longest one it was built for, and would have to learn "how to handle a verb" separately at position 3 and position 40.

> **Where this came from.** Recurrent networks in the modern shape arrived through two closely related designs in the late 1980s: **Michael Jordan's** networks (1986), which fed the *output* back as an extra input, and **Jeffrey Elman's** "simple recurrent network" (1990), which fed the *hidden layer* back — the version everyone means today. Elman's paper, "Finding Structure in Time," is worth knowing for its result rather than its architecture: he trained a network to predict the next word in simple generated sentences, then clustered the hidden states, and found the clusters had organized themselves into nouns, verbs, animates and inanimates. **Nobody had told it that grammatical categories existed.** That a next-token prediction objective causes linguistic structure to appear in a hidden representation, entirely as a side effect, is the founding observation of the line of work that ends at Chapter 13.

#### Examples and non-examples: is that a recurrent network?

"Recurrent" is not a synonym for "handles sequences." It names a specific commitment: **a fixed-size state is carried from one step to the next, and the new state is a function of the old state and the current input alone.** Two properties fall out immediately — the memory cost does not grow with sequence length, and the model cannot look at step 3 again once it has moved to step 4.

**✅  recurrences**

| Example | The update | Why it qualifies |
|---|---|---|
| `nn.RNN(input_size=100, hidden_size=256)` | $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$ | 256 numbers carried forward, whether the sequence is 10 steps or 10 million |
| `nn.LSTM` | $c_t = f_t\odot c_{t-1} + i_t\odot g_t$ | Two fixed-size states instead of one; still $O(1)$ per step |
| A GRU | $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t$ | Same commitment, one state, fewer gates |
| An exponential moving average $\mu_t = 0.9\mu_{t-1} + 0.1x_t$ | Literally a 1-D linear RNN with $W_{hh}=0.9$ | The minimal case. BatchNorm's running statistics (Ch. 7) are exactly this |
| A Kalman filter | $\hat s_t = A\hat s_{t-1} + K(y_t - CA\hat s_{t-1})$ | Fixed-size state, Markov update — recurrent, just not learned by gradient descent |
| Mamba / S4 at inference | $h_t = \bar A h_{t-1} + \bar B x_t$ | A *linear* recurrence, which is why it can also be evaluated in parallel during training (§9.8) |

**❌ Near-misses — process sequences, but are not recurrent**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A 1-D CNN over a token sequence | Every output depends on a **fixed window** of inputs. Nothing is carried; step 500 and step 5 are computed identically and independently | A **convolutional** sequence model. Finite, fixed memory by construction |
| Transformer self-attention | Position $t$ reads positions $1..t$ **directly**, not through a summary. There is no state variable at all | **Direct access.** This is why it has no vanishing-gradient problem and why it costs $O(T^2)$ |
| A transformer decoder generating token by token with a KV cache | It does carry something forward — but the cache **grows linearly with $t$**. At $t=4000$ it holds 4,000 key–value pairs, not a fixed 256 numbers | **Autoregressive generation with caching.** Autoregression is about the *output* dependency; recurrence is about *bounded state* |
| An $n$-gram model | Conditions on the previous 4 tokens, always exactly 4, never a summary of everything before | A **finite-context Markov model** |
| ALBERT / Universal Transformer sharing weights across layers | Weight sharing, yes — but the loop runs over **depth**, not over sequence positions | **Recurrence in depth.** Real recurrence, wrong axis |
| A sliding-window transformer with window 512 | Fixed window, no carried summary | A **local attention** model |
| An RNN unrolled and drawn as a $T$-layer feedforward network | This *is* the RNN — it is how backprop sees it | The **unrolled view**. Same model, drawn flat, which is exactly what §9.2 is about |

▸ **The boundary:** fixed-size state plus a Markov update. **If the memory the model carries between steps grows with $t$, it is not recurrent** — and that one property is simultaneously the recurrent network's great advantage (constant memory, $O(T)$ inference over any length) and its fatal limitation (everything about step 5 must survive inside 256 numbers by the time you reach step 5000).

> **Common misconception.** *"An RNN remembers the whole sequence."* It remembers a fixed-size summary of the whole sequence, and the size is $d$, chosen by you, regardless of $T$. A 256-unit RNN reading a 10,000-token document is compressing $10{,}000$ tokens into 256 floating-point numbers — a ratio the model has no way to escape. Contrast a transformer, whose KV cache after 10,000 tokens holds 10,000 key–value pairs and can attend back to any of them exactly. The belief is tempting because the recurrence formula *does* make $h_t$ depend on every earlier input — the dependency is  there in the algebra — and it is easy to read "depends on" as "retains." ▸ **Dependency is not retention.** Every number in $h_{5000}$ was influenced by $x_1$; the influence is $O(\lambda^{4999})$, and the capacity to store it was never there in the first place.

---

## 9.2 Backpropagation through time, and why it fails

Unroll the network into a $T$-layer feedforward net with tied weights, then apply Ch. 6's backprop. The gradient w.r.t. the shared weight sums over all timesteps:

$$\frac{\partial\mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T}\frac{\partial\mathcal{L}_t}{\partial W_{hh}},\qquad \frac{\partial\mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t}\frac{\partial\mathcal{L}_t}{\partial h_t}\underbrace{\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)}_{\text{the killer}}\frac{\partial h_k}{\partial W_{hh}}$$

#### Unpacking backpropagation through time

Take the second formula from the inside out, because that is the order it makes sense in.

- **$\mathcal{L}_t$** — the loss contributed by the prediction at step $t$. In language modelling there is one of these at *every* step, which is why the outer sum exists.
- **The outer $\sum_{t=1}^{T}$** — the total gradient is the sum of the gradients from every timestep's loss. Nothing subtle here; losses add, so their gradients add.
- **The inner $\sum_{k=1}^{t}$** — this is the one people skip past, and it is the important one. The **same matrix** $W_{hh}$ was used at step 1, at step 2, …, at step $t$. So $W_{hh}$ has $t$ separate *routes* by which it influenced $\mathcal{L}_t$. The multivariate chain rule says: when a variable affects an output through several paths, add up the contributions of all of them.
- **$\partial\mathcal{L}_t/\partial h_t$** — the gradient arriving at the state at the moment the loss is computed. The starting point of the backward journey.
- **$\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}$** — the chain of sensitivities linking $h_k$ to $h_t$. It has exactly $t-k$ factors: one per step you must travel back through.
- **$\partial h_k/\partial W_{hh}$** — how the state at step $k$ responded to the weight *at that step*, the final hop from state to parameter.

Read the whole thing right to left: *"the weight was used at step $k$; that use changed $h_k$; that change propagated forward through every state between $k$ and $t$; and that finally moved the loss at step $t$."*

**Why the underbrace says "the killer."** A product of $t-k$ matrices behaves like a number raised to the power $t-k$ — and $t-k$ is *the distance in time you are trying to learn across*. Wanting to learn a longer dependency means putting a bigger number in an exponent, which is never a good position to be in.

> **Analogy.** A message whispered down a line of people, each repeating it with 90% fidelity. After 100 people, $0.9^{100} = 0.003\%$ of the original survives — you are listening to noise. Now suppose each person instead *embellishes* by 10%: after 100 people the story has grown by a factor of 13,781. There is no fidelity setting that both preserves fine detail across a hundred hops and prevents runaway growth. **The person in the middle of the line has no way to know which job they are supposed to be doing.**

With $\frac{\partial h_j}{\partial h_{j-1}} = W_{hh}^\top\,\mathrm{diag}(\sigma'(\cdot))$:

▸ $$\left\|\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right\| \le \left(\|W_{hh}\|_2\cdot\max|\sigma'|\right)^{t-k}$$

#### Where that Jacobian comes from

$h_j = \sigma(W_{hh}h_{j-1} + \dots)$, so differentiating with respect to $h_{j-1}$ is one application of the chain rule: the derivative of the activation, then the derivative of the linear map.

- **$\sigma'(\cdot)$** — the slope of the activation function, evaluated at each unit's pre-activation value. A vector of $d$ numbers.
- **$\mathrm{diag}(\sigma'(\cdot))$** — those numbers arranged on the diagonal of a $d\times d$ matrix, zeros elsewhere. It is diagonal because **unit $i$'s activation depends only on unit $i$'s own pre-activation** — the nonlinearity does not mix units. Multiplying by a diagonal matrix is just elementwise scaling.
- **$W_{hh}^\top$** — the transpose is pure bookkeeping (Ch. 0 §0.6), making the shapes line up for the backward pass. It carries no extra meaning.

#### Reading the gradient-norm bound

$$\left\|\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right\| \le \left(\|W_{hh}\|_2\cdot\max\lvert\sigma'\rvert\right)^{t-k}$$

- **$\|\cdot\|$ around a matrix** is a norm — "how big is this thing" (Ch. 1 §1.1.4). $\|W\|_2$ specifically is the **spectral norm**: the largest factor by which $W$ can stretch any vector.
- **The bound follows from one fact** — the norm of a product is at most the product of the norms — applied $t-k$ times. Each hop can amplify by at most $\|W_{hh}\|_2$, and the nonlinearity can amplify by at most $\max\lvert\sigma'\rvert$.
- **$\max\lvert\sigma'\rvert$** is the steepest the activation's slope ever gets: $1$ for $\tanh$ (at zero), $1/4$ for the logistic sigmoid (at zero), $1$ for ReLU.
- **$t-k$ sits in the exponent.** That is the whole problem in one observation. Everything else is a constant; the *distance you want to remember* is the power.

**The condition (Pascanu et al., 2013):** let $\lambda_1$ be the largest singular value of $W_{hh}$ and $\gamma = \max|\sigma'|$ ($\gamma=1$ for tanh, $1/4$ for sigmoid).

- $\lambda_1\gamma < 1$ ⇒ **gradients vanish exponentially** — a *sufficient* condition for vanishing.
- $\lambda_1\gamma > 1$ ⇒ gradients *may* explode.

**Numbers.** tanh RNN, $\|W_{hh}\|_2 = 0.9$, sequence length 100: $0.9^{100} = 2.7\times10^{-5}$. The gradient from timestep 100 to timestep 1 is 27 parts per million. **The model cannot learn a dependency spanning 100 steps.** With $\|W_{hh}\|_2 = 1.1$: $1.1^{100} = 13{,}781$ — overflow.

▸ **The knife-edge is exact and unavoidable in a plain RNN.** There is no setting of $W_{hh}$ that both remembers for 100 steps and stays stable, because the same matrix does both jobs. Every fix — LSTM, GRU, attention, residual connections — is a way of *decoupling* memory from transformation.

#### The vanishing/exploding condition, decoded

**Read the two bullets as a single statement:** the quantity that governs everything is the product $\lambda_1\gamma$ — "biggest stretch the weight matrix can apply" times "steepest slope the nonlinearity can have." Raise it to the power of the time distance and you have the fate of your gradient.

Here is that number, tabulated. Read down a column to see what happens as you cross 1:

| $\lambda_1\gamma$ | over 10 steps | over 30 steps | over 100 steps |
|---|---|---|---|
| 0.5 | $9.8\times10^{-4}$ | $9.3\times10^{-10}$ | $7.9\times10^{-31}$ |
| 0.9 | $0.35$ | $0.042$ | $2.7\times10^{-5}$ |
| **0.99** | $0.90$ | $0.74$ | $\mathbf{0.37}$ |
| **1.00** | $1$ | $1$ | $\mathbf{1}$ |
| **1.01** | $1.10$ | $1.35$ | $\mathbf{2.70}$ |
| 1.1 | $2.59$ | $17.4$ | $13{,}781$ |

▸ **Look at the three bolded rows.** A value **one percent** below the critical point still destroys 63% of the gradient over 100 steps. A value one percent above nearly triples it. The window in which a plain RNN can carry gradient across a hundred steps is not narrow — it is essentially a single point, and gradient descent is actively moving $W_{hh}$ around during training, so you cannot even stay parked on it.

**Why the two conditions are stated asymmetrically.** This is a real distinction, not pedantry:

- $\lambda_1\gamma < 1$ is a **sufficient** condition for vanishing. If the biggest possible amplification is below 1, then the actual amplification is certainly below 1, and the gradient *must* die. No escape.
- $\lambda_1\gamma > 1$ only makes explosion **possible**. The formula is an *upper* bound, and upper bounds are not promises. In practice $\sigma'$ is nearly always well below its maximum — $\tanh'(z) = 1 - \tanh^2(z)$ hits 1 only exactly at $z=0$ and is small wherever the unit is saturated — so the realized product is usually smaller than the bound.

▸ **In practice, this asymmetry means vanishing is the common failure and exploding is the occasional one.** That matters because they need different treatment: exploding announces itself as a NaN in your loss and is fixed by clipping; vanishing is silent. Nothing crashes. The model simply trains to a mediocre loss and never learns anything requiring long-range memory, and you have no error message telling you why.

**Why tanh and not sigmoid, and why ReLU is awkward here.** With sigmoid's $\gamma = 1/4$, you would need $\|W_{hh}\|_2 > 4$ merely to break even — and a matrix with spectral norm 4 makes the forward pass unstable long before it helps the backward one. With $\tanh$'s $\gamma=1$, aiming for $\|W_{hh}\|_2\approx 1$ is at least coherent. That single factor of four is the reason $\tanh$ became the standard recurrent activation despite being a rescaled sigmoid in every other respect. ReLU has $\gamma = 1$ too but is unbounded, so a recurrent ReLU explodes readily — which is why the ReLU-based RNN that worked (the "IRNN" of Le, Jaitly and Hinton, 2015) required initializing $W_{hh}$ to the **identity matrix**, i.e. starting exactly on the knife-edge deliberately.

> **Where this came from.** The vanishing-gradient problem was diagnosed by **Sepp Hochreiter** in his 1991 diploma thesis at the Technical University of Munich, supervised by **Jürgen Schmidhuber**. It was written in German and not translated, and it circulated very little outside that group. An independent English analysis by **Yoshua Bengio, Patrice Simard and Paolo Frasconi** appeared in 1994 under the title "Learning Long-Term Dependencies with Gradient Descent is Difficult," which put the result in front of the wider field. The tightened singular-value treatment quoted here, along with gradient clipping as the practical remedy, comes from **Razvan Pascanu, Tomas Mikolov and Bengio** in 2013 — which means the problem was correctly identified more than twenty years before the field had architectures that fully answered it.

### Exploding gradients: solved
**Gradient clipping** (Ch. 4 §4.8) handles this completely. It is not a hack; it is the standard, principled fix.

### Vanishing gradients: needs architecture
Clipping cannot help — you can't amplify a zero. This required the LSTM.

### Practical BPTT

**Truncated BPTT:** carry the hidden state forward but only backpropagate $k$ steps. Standard $k$: 35–200. Biases the model against dependencies longer than $k$, which is usually acceptable and always necessary.

#### Truncated BPTT, decoded

Two things happen at every step of an RNN and they can be cut independently:

| | Forward | Backward |
|---|---|---|
| What flows | the hidden state $h_t$, a vector of numbers | the gradient, through the recorded computation graph |
| Truncated BPTT | **keeps flowing** across chunk boundaries | **stopped** at each chunk boundary |

So the model still *remembers* across the whole sequence — the numbers in $h$ carry over. What it stops doing is *assigning credit* across the boundary.

**Why you must truncate.** Full backpropagation through time requires storing every intermediate activation for the entire sequence, because the backward pass needs them. For $T = 10{,}000$ timesteps, that is 10,000 saved copies of every hidden state and every gate value. Memory grows linearly with sequence length and quickly becomes the binding constraint, long before compute does.

> **Analogy.** Reading a 600-page book while keeping running notes. You carry your notes forward across the whole book (the forward pass), but when you decide *"that sentence was misleading, I should read more carefully"*, you only revise your reading habits based on the last two pages — not by re-deriving what you should have done differently on page 1. You keep the accumulated knowledge and give up the long-range blame assignment.

**`h = h.detach()` is the whole implementation.** Detaching keeps the *values* in $h$ and discards the *history* of how they were computed. Without it, the computation graph grows without bound across chunks and you eventually run out of memory — a bug that presents as steadily increasing memory usage over the first few hundred training steps, and is one of the most common mistakes in hand-written RNN training loops.

▸ **The bias this introduces is real and worth naming.** A model truncated at $k=50$ receives no gradient signal at all about dependencies spanning 200 steps, so it cannot learn them even if the architecture could represent them. **Truncation converts an architectural capability into something the optimizer never gets told about.** This is one reason the transformer's full-sequence backward pass mattered — not just that gradients travel further, but that they travel at all.

#### Examples and non-examples: what backpropagation through time is

BPTT sounds like the name of an algorithm you would have to look up separately. It is not. It is **ordinary backpropagation, applied to the graph you get by unrolling the recurrence**, plus one bookkeeping consequence: a weight used at $T$ timesteps receives $T$ gradient contributions, which are summed.

**✅  BPTT**

| Example | Why it qualifies |
|---|---|
| Unroll 35 steps, compute the loss, call `loss.backward()` | The autograd graph already contains the 35 uses of $W_{hh}$; reverse-mode differentiation sums their contributions automatically. **You did not invoke anything called "BPTT"** |
| Truncated BPTT: `h = h.detach()` at each chunk boundary | Same operation on a shorter graph. The forward values cross the boundary, the gradient does not |
| Backprop through an LSTM over a 200-token sentence | Same thing; the graph is bushier but the algorithm is unchanged |
| Hand-deriving $\partial\mathcal{L}/\partial W_{hh}$ as the double sum in §9.2 | This *is* the derivation. The product $\prod_j \partial h_j/\partial h_{j-1}$ is the chain rule, written out |

**❌ Near-misses — associated with training RNNs, but not BPTT**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| **RTRL** (real-time recurrent learning) | A * different* algorithm: it propagates a $d\times d\times(\text{params})$ sensitivity tensor **forward** in time, needing no unrolled graph and no backward pass | Forward-mode differentiation of a recurrence. $O(d^4)$ per step, which is why nobody uses it |
| Teacher forcing | Concerns what you feed the model on the *forward* pass, not how gradients are computed | An **input-scheduling** choice (§9.7) |
| Gradient clipping at norm 1.0 | Runs *after* BPTT has already produced the gradient, and rescales the result | A **post-hoc** stabilizer |
| Detaching $h$ between chunks | This is the *truncation*, not the differentiation | The memory–bias trade-off knob |
| Backprop through a transformer over 2,048 positions | Nothing is unrolled and no weight is reused across positions in a temporal chain | Just **backprop**. The name BPTT only means something when there is a recurrence to unroll |
| Backprop through 50 layers of a ResNet | Same product-of-Jacobians structure, same vanishing risk — but the weights are **different at each layer** | **Backprop through depth.** The mathematics is analogous, which is exactly why residual connections and LSTM gates are the same idea (§8.2) |

▸ **The boundary:** BPTT is backprop with weight tying across the time axis. **The only thing "through time" adds to "backpropagation" is the inner sum $\sum_{k=1}^{t}$ — the acknowledgement that one matrix influenced the loss through many routes.** If you can differentiate a feedforward network with shared weights, you already know BPTT.

> **Common misconception.** *"BPTT is a separate algorithm you have to implement for RNNs."* It is the same reverse-mode automatic differentiation used for every other network, run on the unrolled computation graph. In PyTorch you never write a line of it: `loss.backward()` on an unrolled RNN performs BPTT, and the `+=` that accumulates $W_{hh}$'s gradient across the 35 timesteps is the same `+=` that accumulates a shared convolution kernel's gradient across spatial positions. The belief is tempting because BPTT has its own name, its own acronym, its own historical paper (Werbos, 1990; Rumelhart, Hinton and Williams had the ingredients in 1986), and appears in curricula as a distinct topic — the naming implies a distinct mechanism. There *is* a  different algorithm for the same problem, RTRL, which propagates sensitivities forward instead of backward — and its existence is the best evidence that BPTT's "through time" is a description of the graph, not of the method.

> **Common misconception.** *"Gradient clipping fixes the vanishing-gradient problem."* Clipping is `g ← g · min(1, c/‖g‖)`. If your gradient norm is $10^{-7}$ and your threshold is $c = 1$, then $\min(1, 10^7) = 1$ and clipping does **exactly nothing** — it only ever scales gradients *down*. Exploding gradients are a numerical accident (one bad batch, one large product) and clipping is a complete and cheap fix; vanishing gradients are a structural property of multiplying many contractive Jacobians, and no rescaling of the final answer can recover signal that was destroyed along the way. The belief is tempting because the two failures are always introduced together, as the pair of things that go wrong with $\lambda^{t-k}$, so a fix for one feels like a fix for both. ▸ **Exploding is a plumbing problem and has a plumbing fix. Vanishing is an architecture problem, and the architecture fix is the rest of this chapter.**

---

## 9.3 LSTM

### The one-line idea

Add a separate memory cell that is updated by **addition** rather than by matrix multiplication, and use learned gates to decide what to add, what to erase, and what to expose.

### The analogy

The index card becomes a filing cabinet with a librarian. Each page, the librarian decides: which folders to shred (forget gate), which new notes to file (input gate), and which folders to actually read aloud right now (output gate). Because filing is *adding a document to a drawer* rather than rewriting the whole cabinet, information from chapter 1 can survive to chapter 30 untouched.

### The equations

▸ $$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1},x_t]+b_f) &&\text{forget gate}\\
i_t &= \sigma(W_i[h_{t-1},x_t]+b_i) &&\text{input gate}\\
\tilde c_t &= \tanh(W_c[h_{t-1},x_t]+b_c) &&\text{candidate}\\
c_t &= f_t\odot c_{t-1} + i_t\odot\tilde c_t &&\textbf{cell state (the key line)}\\
o_t &= \sigma(W_o[h_{t-1},x_t]+b_o) &&\text{output gate}\\
h_t &= o_t\odot\tanh(c_t) &&\text{hidden state}
\end{aligned}
$$

#### Reading the LSTM equations line by line

**First, the name.** "Long Short-Term Memory" parses as *long* **short-term memory** — a short-term memory that lasts a long time. In a neural network, the *weights* are the long-term memory (they encode everything learned across the whole dataset) and the *activations* are the short-term memory (they hold what's happening right now). A plain RNN's short-term memory decays within a few dozen steps. The claim in the name is that this one doesn't.

**Machinery shared by every line:**

- **$[h_{t-1}, x_t]$ is concatenation, not multiplication.** Stack the previous hidden state on top of the current input to make one longer vector. If $h$ is 512-dimensional and $x$ is 300-dimensional, the bracket is an 812-vector, and $W_f$ is a $512\times 812$ matrix. Writing it this way is shorthand for $W_{f,h}h_{t-1} + W_{f,x}x_t$ — the same thing, fewer symbols.
- **Every gate has the identical form $\sigma(W[\cdot]+b)$**, where $\sigma$ here is specifically the **logistic sigmoid**, $\sigma(z) = 1/(1+e^{-z})$, which squashes any real number into $(0,1)$. A gate is therefore a vector of **soft switches** — one per memory slot, each somewhere between fully closed and fully open.
- **$\odot$ is the elementwise product** (Ch. 0 §0.8): multiply matching entries and keep them separate. Multiplying by a gate keeps the entries where the gate is near 1 and erases the entries where it is near 0. **This is what "gating" physically is** — nothing more exotic than a per-entry multiplication by a number between 0 and 1.

**Now each line, with the question it answers:**

| Line | The question it answers | Output range |
|---|---|---|
| $f_t = \sigma(W_f[h_{t-1},x_t]+b_f)$ | "For each memory slot, how much of what's there should **survive**?" | $(0,1)$ |
| $i_t = \sigma(W_i[h_{t-1},x_t]+b_i)$ | "For each slot, how much of the new candidate should be **written in**?" | $(0,1)$ |
| $\tilde c_t = \tanh(W_c[h_{t-1},x_t]+b_c)$ | "**What** would I write, if I were writing?" | $(-1,1)$ |
| $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ | "Keep this much of the old, add this much of the new." | unbounded |
| $o_t = \sigma(W_o[h_{t-1},x_t]+b_o)$ | "How much of each slot do I **expose** to the rest of the network right now?" | $(0,1)$ |
| $h_t = o_t\odot\tanh(c_t)$ | "The visible output is a squashed, gated **view** of the memory." | $(-1,1)$ |

▸ **The crucial separation is between $c$ and $h$: $c_t$ is what the network *remembers*; $h_t$ is what it *says*.** A plain RNN has a single vector doing both jobs, so every act of computation overwrites memory and every act of remembering constrains computation. The LSTM's real innovation is not the gates — it is that there are **two** state vectors, one protected and one exposed.

**Why sigmoid for gates and tanh for content.** Not arbitrary, and not interchangeable. A gate must be a **fraction** ("how much of this to keep"), so it needs a range of $(0,1)$ — a negative gate would mean "keep minus 30% of this memory," which is meaningless. Content must be **signed** ("add this, or subtract this"), so it needs $(-1,1)$. Swap the two activations and the architecture stops working for structural reasons, not tuning reasons.

> **Analogy.** A mixing desk in a recording studio. Each memory slot is one channel. The forget gate is that channel's fader on the "old material" bus; the input gate is its fader on the "new material" bus; the output gate is its fader to the monitor speakers. The network is a sound engineer with 512 channels, moving three faders on each of them, **every single timestep**, based on what it just heard. Nothing in the desk is doing anything clever — the cleverness is entirely in the fader positions.

**Put numbers on the memory.** Take one slot, starting at $c_0 = 0.8$, with $i_t = 0$ throughout (nothing new being written), and see what 100 steps do:

| Forget gate held at | $c_{100}$ | Verdict |
|---|---|---|
| $f = 1.00$ | $0.8$ exactly | **Perfect recall across 100 steps, at zero cost** |
| $f = 0.99$ | $0.8\times 0.366 = 0.29$ | Faded but clearly present |
| $f = 0.90$ | $0.8\times 2.7\times10^{-5} \approx 2\times10^{-5}$ | Gone |
| $f = 0.50$ | $\approx 6\times10^{-31}$ | Gone within about 30 steps |

Compare that table with the $\lambda_1\gamma$ table in §9.2. **They are the same arithmetic.** The difference is entirely in *who chooses the number*: in a plain RNN it is a global property of a shared weight matrix, fixed for all units and all timesteps, and it has to also serve as the transformation. In an LSTM it is chosen **per slot, per timestep, from the data**. Slot 3 can hold the subject of a sentence at $f\approx 1$ while slot 47 tracks the last two characters at $f\approx 0.2$, in the same network, at the same moment.

**Parameter count, so the cost is concrete.** Four gate matrices, each of shape $d\times(d+n)$ plus a $d$-vector of biases. For $d = 512$ hidden units and $n = 512$ input dimensions:

$$4\big(512\times 1024 + 512\big) = 4\times 524{,}800 = 2{,}099{,}200 \approx 2.1\text{M parameters per layer}$$

Four times what a plain RNN of the same width would use. That factor of four is the entire price of admission.

### The constant error carousel

The cell-state update's Jacobian is:
▸ $$\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)$$

**Diagonal, not a full matrix.** So the product over $T$ steps is $\prod_t f_t$ — an elementwise product of numbers in $(0,1)$, *chosen by the network per unit per timestep*.

- If $f_t\approx1$ for some unit, its gradient passes through **undiminished for arbitrarily many steps**.
- Different units can have different forget rates, giving multiple simultaneous timescales.

▸ **This is why the LSTM works, and it is the same trick as a residual connection:** replace a matrix product with an (approximately) additive path. `c_t = f·c_{t-1} + i·c̃_t` is a *gated* residual stream. Chapter 11's transformer residual stream is the ungated version of the same idea.

#### The constant error carousel, decoded

$$\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)$$

- **$\mathrm{diag}(f_t)$** — a $d\times d$ matrix with the forget gate's entries running down the diagonal and **zeros everywhere else**. Multiplying a vector by it scales each entry independently: entry 3 gets multiplied by $f_{t,3}$ and by nothing else.
- (A precise footnote worth having: this is the *direct* path. The gates themselves depend on $h_{t-1}$, which depends on $c_{t-1}$, so the full derivative has additional terms. Those terms are the ones that get squashed by sigmoids and small weights; the diagonal term is the one that carries gradient across long distances, and it is the term the architecture was designed around.)

**Why "diagonal, not a full matrix" is the whole point.** Compare the two:

| | Plain RNN: $W_{hh}^\top\mathrm{diag}(\sigma')$ | LSTM: $\mathrm{diag}(f_t)$ |
|---|---|---|
| Do the units mix? | **Yes** — every unit's gradient is a blend of all units' | **No** — slot 3 stays in slot 3 |
| What governs the repeated product? | The **largest singular value**, globally, for everything | Each slot's **own** $f$, independently |
| Who chooses it? | A learned matrix, fixed across time | The network, **per slot per timestep**, from the input |
| Can different timescales coexist? | No — one $\lambda_1$ rules them all | **Yes** — a slow slot and a fast slot in the same layer |

▸ **A full matrix raised to a high power collapses everything onto its dominant eigendirection.** That is not a bug in RNNs, it is what matrix powers *do* (Ch. 1 §1.1.2). A diagonal matrix has no dominant direction to collapse onto, because nothing mixes. That structural difference — not the gates, not the sigmoids — is why the LSTM can hold information for hundreds of steps.

> **Analogy.** Two ways to run a message-relay system. In the first, every message is poured into a communal tank at each stage and redistributed; after ten stages, whatever was loudest at the start is all that remains and everything else is diluted into it. In the second, each message travels down its own **sealed pipe with its own valve**, and the operator sets each valve independently at each stage. Message 3 arrives intact whether or not message 47 was shouting. The forget gate is that valve, and its diagonal-ness is the seal.

**Where the name comes from.** "Constant error carousel" is Hochreiter and Schmidhuber's own phrase, and it is descriptive: **error** (gradient) **circulates** around the cell **at constant magnitude** as long as the gate stays open. They named the mechanism before they named the architecture, which tells you what they thought the contribution was.

▸ **And now connect it to Chapter 8.** A residual connection is $x_{\ell+1} = x_\ell + F(x_\ell)$. An LSTM cell is $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$. Set $f = 1$ and $i = 1$ and they are the **same equation**. A residual connection is an LSTM cell with the gates welded permanently open. The transformer took the ungated version deliberately: across 100 stacked layers you do not need learned forgetting, you need a clean highway — the same conclusion ResNet-v2 reached about keeping the skip path bare.

**Practical:** initialize $b_f$ to **+1 or +2** so $\sigma(b_f)\approx0.73$–$0.88$ at the start — the network begins in "remember" mode and learns to forget, rather than starting in amnesia. Jozefowicz et al. showed this is worth more than most architectural search.

#### Why one line of initialization code matters this much

$\sigma(0) = 0.5$, $\sigma(1) = 0.731$, $\sigma(2) = 0.881$. Now propagate each for 20 steps:

| $b_f$ | $\sigma(b_f)$ | Memory surviving 20 steps |
|---|---|---|
| $0$ (default init) | $0.500$ | $9.5\times10^{-7}$ |
| $+1$ | $0.731$ | $0.002$ |
| $+2$ | $0.881$ | $0.079$ |

▸ **The default initialization creates a chicken-and-egg trap.** The network can only *learn* to hold information for 20 steps if gradient reaches back 20 steps. But gradient only reaches back 20 steps if the forget gates are already open. With $b_f = 0$, the untrained network forgets everything within about 20 steps, so it never receives a gradient signal about anything longer, so it never learns that longer structure exists. **It is not that the LSTM fails to learn long dependencies — it never gets told they are there.** Setting $b_f = +1$ breaks the loop by starting the gates open and letting the network learn to *close* them where forgetting is useful.

This pattern — **initialize in the state you want the network to be able to escape from, not the neutral one** — is the same principle as zero-initializing the residual branch in Chapter 8, and it recurs constantly. Neutral initialization is not the same as unbiased initialization.

**Peepholes** let the gates see $c_{t-1}$ directly; rarely used now.

> **The story behind the LSTM.** Hochreiter and Schmidhuber published "Long Short-Term Memory" in *Neural Computation* in 1997, six years after Hochreiter's thesis diagnosed the problem it solves. The detail worth knowing is this: **the forget gate was not in it.** The 1997 architecture had only an input gate and an output gate, and its memory cell had no mechanism for clearing itself — which works fine on short tasks and causes the cell state to grow without bound on long continuous streams. The $f_t\odot c_{t-1}$ term that everyone draws when they draw an LSTM was added three years later by **Felix Gers, Schmidhuber and Fred Cummins**, in a paper titled, with admirable directness, "Learning to Forget." **Peepholes** came from the same group shortly after.
>
> The architecture then sat in relative obscurity for roughly a decade. Its breakout was in speech recognition in the early 2010s, and by 2016 Google's translation system had switched from a decade of phrase-based statistical machine translation to a deep LSTM-based neural model — a change large enough to be visible in the output quality to ordinary users. That system was itself displaced by transformers within a few years. **The LSTM's entire reign as the dominant sequence architecture lasted about as long as the gap between its invention and its first success.**

#### Examples and non-examples: what the cell state is

The LSTM has **two** state vectors and they do different jobs. Conflating them is the single most common error in describing the architecture, and it makes the whole design look arbitrary — because the entire point of the design is that the thing that *remembers* and the thing that gets *read* are separate objects.

**✅  properties of the cell state $c_t$**

| Example | Why it qualifies |
|---|---|
| $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ | The only line in the LSTM where the previous state enters through **addition** rather than through a matrix |
| $\partial c_t/\partial c_{t-1} = \mathrm{diag}(f_t)$ | An elementwise multiplier, per unit, chosen per timestep — no $W_{hh}$ anywhere in the path |
| With $f_t = 1$ and $i_t = 0$ for 100 steps, $c_{100} = c_0$ **exactly** | Perfect, lossless retention is representable. This is the constant error carousel |
| A unit holding "the subject was plural" untouched across 40 words | $f = 1$ on that unit while the sentence continues; the information never passes through a nonlinearity |
| $c_t$ is **unbounded** — it can reach 15 or $-30$ | No squashing is applied on the carousel. The $\tanh$ appears only on the way *out* |
| `nn.LSTM` returns it separately, as `c_n` | The API keeps them apart because they are not the same object |

**❌ Near-misses — called "the cell state," but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| The `output` tensor from `output, (h_n, c_n) = lstm(x)` | `output` is the sequence of **$h_t$**, the gated view. The cell states are not in it at all | The hidden states — what the next layer sees |
| "$h_t$ is the LSTM's memory" | $h_t = o_t\odot\tanh(c_t)$. With $o_t \approx 0$, $h_t \approx 0$ **while $c_t$ is holding a full load of information** | The **exposed read** of memory. Memory and the decision to reveal it are deliberately decoupled |
| $\tanh(c_t)$ | Bounded to $(-1,1)$; the carousel value is not | The squashed intermediate, one step before the output gate |
| The GRU's $h_t$ | The GRU *merged* the two states and deleted the output gate. There is nothing hidden behind it | A single merged state — which is exactly the simplification the GRU is |
| A vanilla RNN's $h_t$ | Updated as $\sigma(W_{hh}h_{t-1} + \dots)$: matrix multiply, then a squashing nonlinearity, every step | The hidden state whose repeated multiplication §9.2 is about |
| The seq2seq context vector $c$ | Same letter, different object: the encoder's **final** state, handed to the decoder once | The **bottleneck** of §9.6. A  notation collision |
| The candidate $\tilde c_t$ | It is what the cell *might* absorb, before $i_t$ decides how much | A proposal, not a state |

▸ **The boundary:** the cell state is the variable updated by elementwise addition and **never handed out directly**; the hidden state is the gated, squashed view of it that every downstream consumer actually sees. **The LSTM's whole innovation is that these are two variables rather than one** — a vanilla RNN uses a single vector both as its memory and as its output, and cannot decide to remember something without also broadcasting it.

> **Common misconception.** *"The hidden state and the cell state are basically the same thing — $h$ is just $c$ after a squashing function."* They are related by $h_t = o_t\odot\tanh(c_t)$, and the $o_t$ is the whole point. A unit can hold a value in $c$ for fifty steps with $o_t = 0$ throughout, contributing nothing to any output, and then open $o$ at exactly the moment the information is needed. That is not a squashing function; it is a decision about *when to speak*, made independently of the decision about *what to keep*. The belief is tempting because most diagrams draw $h$ and $c$ as two parallel horizontal lines of equal thickness, because the GRU really does merge them successfully, and because `nn.LSTM`'s `output` gives you only $h$ — so $c$ is easy to treat as an implementation detail. ▸ **Three decisions, three gates: what to erase ($f$), what to write ($i$), what to reveal ($o$). Collapse any two and you have a different architecture.**

> **Common misconception.** *"LSTMs solve the vanishing-gradient problem."* They **mitigate** it, and the mitigation is real but bounded. The gradient along the carousel is $\prod_j \mathrm{diag}(f_j)$ — still a product, still one factor per timestep. The difference from a vanilla RNN is that each factor is a **learned, per-unit, per-timestep number in $(0,1)$** rather than a fixed matrix's spectrum, so the network can set $f \approx 1$ on the units that must remember and $f$ small on the units that must not. But a sigmoid never outputs exactly 1, so at $f = 0.99$ over 500 steps you retain $0.99^{500} = 0.0066$, and at $f = 0.9$ over 100 steps you retain $2.6\times10^{-5}$. Measurements agree with the algebra: Khandelwal and colleagues (2018) found LSTM language models make effective use of roughly 200 tokens of prior context, and are sharply sensitive only to the nearest 50 or so. The belief is tempting because the constant-error-carousel derivation  does produce a clean $\partial c_t/\partial c_{t-1} = f_t$ with no matrix in it, and "no matrix product" reads as "no vanishing" — but a product of scalars below 1 vanishes just as reliably as a product of matrices. **The LSTM moved the decay from something the architecture imposed to something the network controls. Control is not exemption.**

---

## 9.4 GRU

Merges the cell and hidden state, and ties the forget and input gates.

▸ $$
\begin{aligned}
z_t &= \sigma(W_z[h_{t-1},x_t]) &&\text{update gate}\\
r_t &= \sigma(W_r[h_{t-1},x_t]) &&\text{reset gate}\\
\tilde h_t &= \tanh(W[r_t\odot h_{t-1},x_t])\\
h_t &= (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t
\end{aligned}
$$

The **convex combination** $(1-z)h_{t-1} + z\tilde h$ is the leaky-integrator form. Three gate-matrices instead of four ⇒ ~25% fewer parameters and ~25% faster.

#### Reading the GRU equations

**GRU = Gated Recurrent Unit.** The design question it answers is: *how much of the LSTM's machinery is actually load-bearing?*

Two structural simplifications:

1. **One state instead of two.** The LSTM's separate $c$ (memory) and $h$ (exposed view) are merged back into a single $h$. The output gate disappears with them.
2. **One gate does two jobs.** The LSTM's $f_t$ and $i_t$ are independent; the GRU ties them, using $z_t$ for "write new" and $1-z_t$ for "keep old."

Line by line:

| Line | Reads as |
|---|---|
| $z_t = \sigma(W_z[h_{t-1},x_t])$ | **Update gate.** "For each slot: new material, or old?" |
| $r_t = \sigma(W_r[h_{t-1},x_t])$ | **Reset gate.** "How much of the past is the *candidate* even allowed to look at?" |
| $\tilde h_t = \tanh(W[r_t\odot h_{t-1},x_t])$ | The candidate, computed from a **reset-filtered** view of history |
| $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t$ | The blend |

**The reset gate is the one people misread.** It does not erase memory — $h_{t-1}$ passes into the final line untouched. It only controls what the *candidate* is allowed to condition on. Setting $r\approx 0$ lets a unit compute a fresh candidate **as though the sequence had just started**, which is exactly what you want at a sentence boundary or a scene change, without destroying what you were holding.

**Reading the convex combination.** $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t$ is a **convex combination**: the two coefficients are non-negative and add to exactly 1, so the result is a weighted average and always lies between its two inputs.

- $z_t = 0$: $h_t = h_{t-1}$ **exactly**. Perfect memory, no decay at all.
- $z_t = 1$: complete overwrite.
- $z_t = 0.1$: 90% old, 10% new — a slow-moving running average.

> **Analogy.** A thermostat's reading. Each measurement nudges the displayed temperature by a small fraction toward the new value: $\text{display} \leftarrow 0.9\,\text{display} + 0.1\,\text{measurement}$. A single spurious reading barely moves it; a sustained change moves it steadily. That is a **leaky integrator**, and it is the same three-symbol pattern as momentum in Chapter 5, as an exponential moving average of model weights, and as the running statistics inside BatchNorm. ▸ **Whenever you see $\text{new} = (1-\alpha)\cdot\text{old} + \alpha\cdot\text{fresh}$, you are looking at a low-pass filter with a learned time constant.** It shows up in every corner of this book.

**What the GRU gives up, precisely.** The LSTM's $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ has **two independent** coefficients, so it can do things a convex combination structurally cannot — set $f=1$ *and* $i=1$ and the cell **accumulates without bound**. That is a counter. A convex combination can never exceed its inputs, so it cannot count upward indefinitely. This is exactly the "unbounded counting" advantage noted below, and it is the clearest concrete difference between the two architectures.

**The cost saving, counted.** LSTM: four weight matrices. GRU: three ($W_z$, $W_r$, $W$). That is 25% fewer parameters and roughly 25% fewer floating-point operations per step — a real saving when the recurrence is your bottleneck.

**LSTM vs GRU:** empirically indistinguishable on most tasks (Chung et al., 2014; Greff et al.'s 5,400-run ablation). GRU is slightly better on small data; LSTM slightly better on very long sequences and on tasks requiring unbounded counting. **The honest answer in an interview is "they're equivalent in practice; pick GRU for speed, LSTM if your sequences are very long."**

> **Where this came from.** The GRU was introduced by **Kyunghyun Cho and colleagues in Yoshua Bengio's group at Montréal in 2014** — and not as the point of the paper. The paper was "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," and its headline system used a recurrent encoder–decoder to **rescore candidate translations** produced by a conventional phrase-based statistical machine translation system. The gating unit was a component, described in a subsection. The system it was built to support was obsolete within about two years; the gate design is still in production everywhere. ▸ **A recurring pattern in this field: the contribution that survives is frequently not the one the paper is about.**

#### Examples and non-examples: is that a gate?

A **gate** is an *input-dependent* multiplicative coefficient, elementwise, applied to a signal that would otherwise pass through unchanged. All three words are load-bearing, and dropping any one gives you something with a different name.

**✅  gates**

| Example | The multiplication | Why it qualifies |
|---|---|---|
| The LSTM forget gate | $f_t \odot c_{t-1}$, $f_t = \sigma(W_f[h_{t-1},x_t]+b_f)$ | Computed from the current input and state, elementwise, in $(0,1)$ |
| The GRU update gate | $(1-z_t)\odot h_{t-1}$ | Same, with the complement used for the new material |
| A Highway network's transform gate | $T(x)\odot F(x) + (1-T(x))\odot x$ | The ancestor of the residual connection, with the skip gated |
| GLU / SwiGLU in a modern transformer's feed-forward block | $(\,W_1x\,)\odot\sigma(W_2x)$ | One branch multiplies the other; the multiplier depends on the input |
| A soft mixture-of-experts router weighting expert outputs | $\sum_e g_e(x)\,E_e(x)$ | Input-dependent multiplicative coefficients on parallel paths |
| Squeeze-and-excitation in a CNN | Per-channel scale computed from a global pooled summary | Elementwise (per channel), input-dependent, in $(0,1)$ |

**❌ Near-misses — multiply, or look selective, but aren't gates**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| ReLU: $x\cdot\mathbf{1}[x>0]$ |  input-dependent and multiplicative — but the multiplier is **the signal's own sign**, taking only 0 or 1, with no learned parameters | A **nonlinearity**. The closest legitimate relative; GLU is essentially "what if the switch were learned and smooth" |
| Softmax attention weights | Mix *across positions* with weights summing to 1 — a redistribution, not an elementwise pass-or-block on one carried signal | **Content-based routing** (Ch. 11) |
| A dropout mask $m\odot h$ | Elementwise and multiplicative, but **random** and independent of the input | Injected **noise** (Ch. 7) |
| A causal or padding mask | Fixed, known before you see the data, and identical every forward pass | An **architectural constraint** |
| LayerNorm's learned gain $\gamma\odot\hat x$ | Learned and elementwise, but the **same vector for every input** — it does not react to anything | A learned **scale** |
| A sigmoid on the final output layer | Nothing is multiplied by it; it *is* the answer | A **probability** |
| Pruning that zeroes 90% of weights | Permanent, decided offline, applied to parameters rather than activations | **Sparsification** (Ch. 17) |
| $\mathbf{1}[\,\text{gate} > 0.5\,]$, a hard binary switch | Multiplicative and input-dependent, but its derivative is 0 almost everywhere — nothing can be learned through it | A **hard gate**, which needs straight-through estimators or REINFORCE to train |

▸ **The boundary:** input-dependent, multiplicative, elementwise, and differentiable. **The differentiability is why sigmoids are used instead of step functions** — the network must be able to feel that opening a gate slightly further would have helped, and a step function reports nothing at all.

> **Common misconception.** *"Gates are switches — they're basically on or off."* A trained LSTM's gates spend most of their time in the middle of the range. Values like $f = 0.94$ or $i = 0.31$ are the normal case, and they are doing something a switch could not: $f = 0.94$ specifies a **half-life** — the information decays to half its value after $\ln(0.5)/\ln(0.94) \approx 11$ steps — rather than a keep-or-discard verdict. The belief is tempting because the names ("gate," "forget," "input") are binary words, because every LSTM diagram draws them as valves, and because the extreme cases are the ones used to explain the mechanism ($f=1$ for the carousel, $f=0$ to reset). But a  binary gate would be untrainable: $\sigma$ was chosen over a step function precisely because gradient descent needs a slope to follow, and the intermediate values where that slope is largest are where the learning happens. ▸ **Read a gate value as a time constant, not as a decision.**

---

## 9.5 Architectural variations

- **Bidirectional:** run one RNN forward and one backward, concatenate. Doubles context; **impossible for autoregressive generation** (you'd need the future). Standard for tagging, NER, speech recognition.
- **Deep/stacked:** feed $h^{(1)}_t$ into a second RNN. 2–4 layers typical; deeper rarely helps without residual connections between layers.
- **Highway/Residual RNN:** add skip connections across layers, which is necessary beyond ~4 layers.

---

## 9.6 Seq2Seq and the bottleneck that created attention

### The architecture (Sutskever et al., 2014)

An encoder RNN consumes the source and produces a final state $c = h^{\text{enc}}_{T}$; a decoder RNN is initialized with $c$ and generates the target autoregressively.

▸ **The bottleneck: the entire source sentence must fit in one fixed-size vector.** A 512-dim state has to encode a 50-word sentence as faithfully as a 5-word one.

**The empirical signature:** BLEU score degrades sharply with source length beyond ~20 tokens. Sutskever's famous hack — **reversing the source sentence** — improved BLEU by ~5 points, purely because it shortened the path between the first source word and the first target word. That a trick this crude helped that much is the clearest possible evidence that the bottleneck was the binding constraint.

#### The seq2seq bottleneck, with numbers

**Vocabulary first.** **Seq2Seq** is "sequence to sequence." **BLEU** (BiLingual Evaluation Understudy) scores a machine translation from 0 to 100 by measuring n-gram overlap with human reference translations — the "understudy" is theatrical: a stand-in that performs when the human judge is unavailable. A one-point BLEU gain is a publishable result; five points is a different system.

**The architecture in one sentence.** An encoder RNN reads the whole source and produces $c = h^{\text{enc}}_T$ — its final hidden state. That single vector is handed to the decoder, which generates the translation from it. **$c$ is the only channel between the two networks.** Everything the source sentence contained must pass through it.

**Is it a capacity problem?** Count the bits and the answer is surprisingly no:

- A 1000-dimensional state in 32-bit floats is 4,000 bytes = 32,000 bits.
- A 50-token sentence from a 30,000-word vocabulary carries at most $50 \times \log_2 30{,}000 \approx 50\times 14.9 = 745$ bits of raw word-identity information.

On a bit-counting basis the vector has room to spare, by a factor of forty.

▸ **So the bottleneck is a *summarization* problem, not a *storage* problem.** The encoder must decide what to keep **before it knows what the decoder will ask for**. It writes one summary; the decoder needs fifty different summaries, one for each output word it produces. There is no single 1000-dimensional summary that is simultaneously optimal for translating word 1 and word 47.

> **Analogy.** You are allowed to photograph a document once, and then must answer fifty questions about it **from memory alone**, with the photograph taken before you heard any of the questions. Compare that with being allowed to glance back at the page before answering each question. The second arrangement does not need a better memory — it needs a different protocol. **Attention is that second protocol.**

**Why reversing the source works, and why it is such damning evidence.** Reversing puts source word 1 next to target word 1. For language pairs that are roughly word-aligned, the *average* distance between corresponding words is unchanged by reversal — reversing helps the beginning exactly as much as it hurts the end. But the *first few* decoder steps set the trajectory for everything that follows, and in an autoregressive decoder an early error propagates to every subsequent token.

▸ **A trick with no linguistic content, which does not change the model's capacity or its objective, and which merely rearranges the input order, was worth five BLEU points.** When a change that crude produces a gain that large, it is telling you the architecture has a structural defect and not a tuning problem. Attention was published within a year.

### Bahdanau attention (2015) — the invention

Instead of a single $c$, compute a **different context vector at every decoder step**, as a weighted average of *all* encoder states:

▸ $$
e_{tj} = a(s_{t-1}, h_j),\qquad \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k}\exp(e_{tk})},\qquad c_t = \sum_j \alpha_{tj}h_j
$$

with the "additive" score $a(s,h) = v^\top\tanh(W_1s + W_2h)$.

**Read this carefully — it is the transformer in embryo:**
- $s_{t-1}$ is the **query** (what the decoder wants now).
- $h_j$ are the **keys** (what each source position offers).
- $h_j$ are also the **values** (what gets retrieved).
- softmax over scores = a **differentiable soft dictionary lookup**.

**Luong attention (2015)** replaced the MLP score with a bilinear or dot product: $a(s,h) = s^\top W h$ or $s^\top h$. Cheaper and just as good — and that dot product is precisely what Vaswani et al. scaled up.

#### Reading Bahdanau attention in plain English

**Two clocks, and getting them straight is most of the battle.** The subscript $t$ indexes the **decoder** — which output word you are about to produce. The subscript $j$ indexes the **encoder** — which input word you are considering. They run over different sequences of different lengths. Every double-subscripted quantity here is a (decoder step, encoder position) pair.

| Symbol | Read aloud | What it is |
|---|---|---|
| $s_{t-1}$ | "s sub t minus 1" | The decoder's state just before producing output word $t$ — "what I need right now" |
| $h_j$ | "h sub j" | The encoder's state at source position $j$ — "what word $j$ has to offer" |
| $e_{tj}$ | "e sub t j" | A raw **compatibility score**. Any real number, positive or negative |
| $a(\cdot,\cdot)$ | "a of s, h" | The scoring function — a small learned network |
| $\alpha_{tj}$ | "alpha sub t j" | The **attention weight** after softmax. Non-negative, and $\sum_j\alpha_{tj}=1$ |
| $c_t$ | "c sub t" | The **context vector**: a weighted average of all encoder states |

The softmax denominator $\sum_k \exp(e_{tk})$ sums over $k$, which ranges over **encoder positions for a fixed decoder step $t$**. So each decoder step gets its own probability distribution over the source. Read $\alpha_{t\cdot}$ as the answer to *"where am I looking right now?"*

▸ **Note the shape of $c_t$: it is the same size as one $h_j$, regardless of whether the source had 5 words or 500.** That is what makes this work as a replacement for the fixed vector. The bottleneck is not widened — it is **recomputed**, freshly, for every output word.

**Work it with numbers.** A three-word source, and suppose the scores come out $e = (2.0,\ 1.0,\ 0.0)$:

| Step | Computation | Result |
|---|---|---|
| Exponentiate | $e^{2}, e^{1}, e^{0}$ | $7.389,\ 2.718,\ 1.000$ |
| Sum | | $11.107$ |
| Divide | | $\alpha = (0.665,\ 0.245,\ 0.090)$ |

With encoder states $h_1 = (1,0)$, $h_2 = (0,1)$, $h_3 = (-1,-1)$:

$$c = 0.665(1,0) + 0.245(0,1) + 0.090(-1,-1) = (0.575,\ 0.155)$$

Now raise the first score from 2.0 to 4.0, leaving the others alone. The weights become $(0.936,\ 0.047,\ 0.017)$ and the context vector is now almost exactly $h_1$.

▸ **That is softmax's sharpness doing the essential work: it interpolates continuously between "a blurry average of everything" and "a hard lookup of one item."** A blurry average would be useless (it is exactly the fixed-vector problem again), and a hard lookup would be non-differentiable. Softmax gives you a **differentiable soft dictionary lookup** — which is the single most useful one-sentence description of attention there is.

**Decoding the two scoring functions.**

- **Additive (Bahdanau):** $a(s,h) = v^\top\tanh(W_1s + W_2h)$. Project the query and the key into a shared space, add them, squash, and project down to one number with the vector $v$. It is a one-hidden-layer MLP that eats two vectors and returns a scalar. Two matrix multiplies per (query, key) pair.
- **Multiplicative (Luong):** $a(s,h) = s^\top W h$, or in the simplest case just $s^\top h$. One dot product. And a dot product measures **alignment** (Ch. 0 §0.8) — how much do these two vectors point the same way — which is a perfectly natural notion of "relevance."

▸ **The reason dot-product attention won is not accuracy — it is that it is a matrix multiply.** All $T\times T$ scores can be computed in **one** matmul, $QK^\top$, which is precisely the operation a GPU is built to do at peak throughput. Additive attention needs a small MLP evaluated per pair and cannot be folded into a single dense matrix product nearly as cleanly. This is the same lesson as the table in the next section, arriving a paragraph early.

**The query–key–value reading, made explicit.** The book's bullets are the transformer in embryo; here is the mapping laid out:

| Attention role | In Bahdanau's notation | Plain meaning |
|---|---|---|
| **Query** | $s_{t-1}$ | "What am I looking for at this moment?" |
| **Key** | $h_j$ | "What does position $j$ advertise itself as?" |
| **Value** | $h_j$ (the same vector) | "What does position $j$ hand over if selected?" |
| **Lookup** | softmax over scores | A differentiable, soft version of "fetch the matching entry" |

The transformer's contribution to this table is separating **key** from **value** — letting a position advertise itself with one vector and deliver a different one — plus letting the query come from the same sequence as the keys.

> **Where attention came from.** The mechanism was published by **Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio** at ICLR 2015, out of Montréal. Two details are worth knowing. First, **the paper is not called "attention."** Its title is "Neural Machine Translation by Jointly Learning to Align and Translate," and it describes the mechanism as a **soft search** over the source and as learning an **alignment** — the term the statistical machine translation literature had used for decades for exactly this "which source word produced which target word" question. The name "attention" settled in afterwards. Second, the framing was explicitly as a fix for a diagnosed defect: the paper argues that the fixed-length vector is the bottleneck, and proposes recomputing the summary per output word. It is a rare case of a field-defining mechanism arriving as a **targeted repair of a specific, well-understood failure** rather than as a general architectural proposal. **Minh-Thang Luong, Hieu Pham and Christopher Manning** at Stanford simplified the scoring function later the same year, and it is their dot product — not Bahdanau's MLP — that Vaswani et al. scaled up two years later.

▸ **The conceptual leap of the transformer was not attention.** It was: (a) using attention *within* a sequence to itself (self-attention) rather than only across two sequences, and (b) deleting the recurrence entirely, so all positions compute in parallel.

### Why removing recurrence mattered so much

| | RNN | Transformer |
|---|---|---|
| Sequential ops | $O(T)$ | $O(1)$ |
| Path length between any two positions | $O(T)$ | $O(1)$ |
| Compute per layer | $O(T d^2)$ | $O(T^2d + Td^2)$ |
| Parallelizable over $T$ during training | ✗ | ✓ |

▸ The **$O(1)$ path length** is the theoretical win (no vanishing gradient across distance at all), but the **parallelism** is the practical one: an RNN cannot use a modern GPU's throughput because step $t$ must wait for step $t-1$. The transformer's $O(T^2)$ compute is *worse* asymptotically and *vastly better* in wall-clock, because it is one big matmul. **This is the single most important lesson in modern ML systems design: hardware-friendly beats asymptotically-cheaper.**

#### Reading the RNN-vs-transformer table

Big-O notation means "how does this grow as the sequence gets longer, ignoring constants" (Ch. 0 §0.10). Each row measures something  different, and it is worth being precise about what:

**Row 1 — sequential operations.** The number of steps that must happen strictly **one after another**, because each needs the previous one's answer. An RNN over 1,000 tokens has 1,000 unavoidable serial steps; you cannot start step 500 until step 499 has finished, no matter how many processors you own. A transformer layer has a **constant** number of serial stages regardless of $T$ — every position's computation is independent given the layer's input.

**Row 2 — path length.** How many computational hops a signal must make to travel from position 1 to position 1,000. In an RNN that is 999 multiplications, and by §9.2 that means the gradient is multiplied by 999 Jacobians. In a transformer it is **one** attention operation: position 1,000's query dots directly with position 1's key. ▸ **There is no vanishing gradient across sequence distance in a transformer, because there is no distance.** Every position is one hop from every other.

**Row 3 — compute per layer, and the honest admission.** $O(Td^2)$ against $O(T^2d + Td^2)$. The transformer is *worse* whenever $T > d$:

| Sequence length $T$ | Model width $d$ | Which term dominates |
|---|---|---|
| $512$ | $1024$ | $Td^2$ — the feed-forward part; attention is a minor cost |
| $1024$ | $1024$ | roughly balanced |
| $100{,}000$ | $1024$ | $T^2d$ — attention utterly dominates, ~100× more arithmetic |

So on raw operation counts the transformer *lost*. It won anyway.

**Why it won: the number that isn't in the table.** A modern accelerator reaches its advertised throughput only when the work is arranged as a **large dense matrix multiplication** — thousands of independent multiply-accumulates that can be issued simultaneously and fed from fast on-chip memory. A chain of small operations that must each wait for the last one leaves the vast majority of that hardware idle, because the bottleneck is memory latency and dependency stalls rather than arithmetic.

> **Analogy.** A factory with ten thousand workers. Task A requires ten thousand *sequential* handovers — each worker must wait for the one before, so 9,999 workers stand idle at any moment and the job takes 10,000 shifts. Task B requires a hundred million operations but they are all independent, so all ten thousand workers go at once and the job takes 10,000 operations' worth of time. **Task B involves ten thousand times more work and finishes first.** Counting operations told you nothing useful; counting *idle workers* told you everything.

▸ **Wall-clock time is total operations divided by achieved utilization, and utilization varies by two orders of magnitude between computation shapes.** Asymptotic analysis silently assumes utilization is constant, which on parallel hardware it emphatically is not. This is why "hardware-friendly beats asymptotically-cheaper" is a real design principle and not a slogan — it also explains FlashAttention (Ch. 17), why quantization gives superlinear speedups, and why the field's efficient-attention literature is full of methods with better big-O that are slower in practice.

---

## 9.7 Training details that generalize beyond RNNs

### Teacher forcing and exposure bias

**Teacher forcing:** at training time, feed the *ground-truth* previous token to the decoder instead of its own prediction. Makes training parallel and stable.

▸ **Exposure bias:** at inference the model consumes its own outputs, so it encounters states it never saw in training, and errors compound. Mitigations: scheduled sampling (anneal from ground truth to model samples), sequence-level training (minimum risk training, RL with BLEU as reward).

**Modern status:** every LLM is trained with teacher forcing and exposure bias is real but empirically small at scale. RLHF (Ch. 16) is, among other things, a fix for it — it trains on the model's own sampled continuations.

#### Teacher forcing and exposure bias, decoded

**Teacher forcing, precisely.** When training a model to generate token $t$, you must give it tokens $1$ through $t-1$ as context. There are two choices: hand it the **ground-truth** prefix from the dataset, or let it generate its own prefix and hand it that. Teacher forcing is the first choice. The name is literal — a teacher who supplies the correct answer at every step.

Two reasons it is universal:

1. **Parallelism.** If every step's input is known in advance (it is just the dataset), all $T$ steps can be computed at once. If each step's input depends on the previous step's *output*, they must run sequentially — you would be paying an RNN's serial cost even inside a transformer.
2. **Gradient quality.** Without it, a model that is wrong at step 3 receives its own garbage as input for steps 4 through 50. The gradients from those steps carry almost no usable signal about the actual task, and early in training the model is wrong essentially always.

**Exposure bias, precisely.** At inference nobody hands you the truth. The model consumes its own outputs, so by step 20 it is conditioning on a prefix drawn from **its own distribution**, not from the data distribution. It has never been trained on those states. Errors compound: a slightly odd token at step 5 produces a slightly odder context at step 6, and so on.

> **Analogy.** Learning to drive with an instructor who silently corrects the wheel every second. You are never allowed to become slightly off-centre, so you never learn what being off-centre feels like or how to recover from it. Every hour of practice is spent in the narrow band of perfect road position. The first time you drive alone, a small drift takes you somewhere you have literally never been, and you have no trained response.

**Put a rough number on it.** Suppose each generated token has an independent 1% chance of putting the model into a context unlike anything it trained on. Over a 200-token generation, the probability of hitting at least one such state is $1 - 0.99^{200} = 87\%$. Under that model, drift is not an edge case — it is the norm.

**Why it turns out to be smaller than the argument suggests.** With enough data and capacity, the model's own samples are close enough to the data distribution that the "unseen" states are not really unseen — they are interpolations between things it has seen many times. The 1% figure above is far too pessimistic at scale. But the effect is real, and the mitigations are real:

| Mitigation | What it does |
|---|---|
| **Scheduled sampling** | Anneal from ground-truth inputs to the model's own samples during training |
| **Sequence-level training** | Optimize a whole-sequence metric (e.g. BLEU) rather than per-token likelihood |
| **RLHF** (Ch. 16) | Train on the model's **own** sampled continuations, scored by a reward model |

▸ **Reinforcement learning from human feedback is, among other things, exposure-bias correction.** It is the only stage of a modern language model's training in which the model is optimized on states it actually generated itself. Whatever else RLHF does about preferences and safety, it also closes this specific train/inference gap — which is worth remembering when people describe it as purely an alignment technique.

### CTC — alignment-free sequence loss

For speech and handwriting where input and output lengths differ and alignment is unknown. Introduce a blank symbol $\varnothing$; define a many-to-one collapse $\mathcal{B}$ (remove repeats, then remove blanks). Then

▸ $$p(y\mid x) = \sum_{\pi\in\mathcal{B}^{-1}(y)} \prod_{t=1}^{T}p(\pi_t\mid x)$$

The sum is over exponentially many alignments, computed in $O(T|y|)$ by a **forward–backward dynamic program** (identical in structure to HMM training). Assumes conditional independence of outputs given the input, which is CTC's main weakness — hence the RNN-Transducer, which adds an output-side language model.

#### Reading the CTC objective

**CTC = Connectionist Temporal Classification.** ("Connectionist" is the older name for neural networks; "temporal classification" means labelling a time series.)

**The problem it solves, concretely.** One second of speech is roughly 100 audio frames. The word it contains is maybe 5 letters. You have the transcript. You do **not** have any record of which frames correspond to which letter — and producing that by hand, for thousands of hours of audio, is not a project anyone will fund. Every ordinary sequence loss needs a per-step target. You have no per-step targets.

$$p(y\mid x) = \sum_{\pi\in\mathcal{B}^{-1}(y)} \prod_{t=1}^{T}p(\pi_t\mid x)$$

| Symbol | Read aloud | What it is |
|---|---|---|
| $y$ | "y" | The label you actually have — e.g. the 5 letters `hello` |
| $\pi$ | "pi" | An **alignment** (a "path"): one output symbol per input frame, so $\pi$ has length $T$. Nothing to do with 3.14159 |
| $\varnothing$ | "blank" | A special symbol meaning **"emit nothing at this frame"** |
| $\mathcal{B}$ | "script B" | The **collapse**: remove repeated symbols, then remove blanks |
| $\mathcal{B}^{-1}(y)$ | "B-inverse of y" | The **pre-image** — every path that collapses to $y$. Not an inverse function; the map is many-to-one |
| $p(\pi_t\mid x)$ | "p of pi-t given x" | The model's probability for symbol $\pi_t$ at frame $t$ |

**Watch the collapse work.** $\mathcal{B}(\texttt{h h}\ \varnothing\ \texttt{e l l}\ \varnothing\ \texttt{l o}) = \texttt{hello}$. Note the blank sitting between the two runs of `l`: without it, "remove repeats" would merge them into a single `l` and you would get `helo`. ▸ **The blank symbol's second job is to let you write  double letters** — it is not only "say nothing here," it is also a separator. This is the part of CTC that everyone has to be told twice.

**Read the formula aloud:** *"the probability of the label is the sum, over every way of stretching that label across the frames, of the probability of that particular stretching."*

- $\prod_t p(\pi_t\mid x)$ — the probability of one specific alignment, assuming each frame's output is independent of the others given the audio.
- $\sum_\pi$ — add over all valid alignments. ▸ **You never commit to an alignment. You sum over all of them, and let the gradient distribute credit across every explanation consistent with the label.** The alignment is a latent variable that is marginalized away, exactly as $z$ is in the ELBO (Ch. 1 §1.4.4).

**The numbers make the dynamic program look like magic.** With $T = 100$ frames and 28 output symbols, the total number of paths is $28^{100}\approx 10^{145}$ — more than the number of atoms in the observable universe raised to a considerable power. Even restricted to those collapsing to one specific 5-letter word, the count is astronomically large. The forward–backward algorithm computes the **exact** sum in $O(T\lvert y\rvert) = 100\times 5 = 500$ operations.

> **Analogy.** Counting the routes across a city grid from one corner to another. Enumerating them individually is hopeless. But the number of routes reaching any given intersection is simply the sum of the routes reaching the intersections you could have come from — so **one sweep across the grid counts every route without ever listing one.** That is dynamic programming, and this specific instance is the same algorithm as the forward pass of a hidden Markov model (HMM), which is where CTC borrowed it.

**The weakness, stated precisely.** "Conditional independence of outputs given the input" means the model computes $p(\pi_t\mid x)$ with **no dependence on what it already emitted**. It has no way to represent "after `q` comes `u`." A language model is therefore usually fused in at decoding time, and the **RNN-Transducer** fixes the problem architecturally by adding an output-side network that conditions on the emitted prefix.

> **Where this came from.** CTC was introduced by **Alex Graves** and colleagues in 2006, in Schmidhuber's group, for labelling unsegmented sequences — and it is what made end-to-end neural speech recognition possible. Before it, speech systems needed a separate alignment stage, usually a hidden Markov model, to say which frames belonged to which phoneme; the neural network was one component inside a hand-assembled pipeline. CTC deleted that stage by marginalizing over it. Graves published the **RNN-Transducer** extension in 2012, and variants of it are still what runs on-device speech recognition on phones today.

### Truncated BPTT state handling
Detach the hidden state between chunks (`h = h.detach()`), or the graph grows without bound.

---

## 9.8 Do RNNs still matter?

▸ Yes, in three places:

1. **Streaming/low-latency inference.** An RNN has $O(1)$ per-token state; a transformer's KV cache is $O(T)$ and grows without bound. On-device keyword spotting and real-time ASR still use recurrent models.
2. **Their descendants dominate the efficiency frontier.** State-space models (S4, **Mamba**) and linear attention are, mathematically, linear RNNs with structured state transitions — recast so they can be trained in parallel via associative scan (Ch. 12 §12.6). Mamba is a gated linear RNN with input-dependent transitions. The recurrent idea came back with the parallelism problem fixed.
3. **Conceptually.** The gating/additive-path insight is the direct ancestor of residual streams, and the query–key–value framing came from here.

#### Why the recurrent idea came back

**Reason 1, with numbers.** A transformer must keep the key and value vectors for **every** previous token, because attention at step $t$ dots against all of them. That store is the **KV cache** (key–value cache), and it grows linearly with the conversation. For a model with 32 layers, 8 key–value heads, head dimension 128, in 16-bit precision, at 100,000 tokens of context:

$$100{,}000 \times 32 \times 2 \times 8 \times 128 \times 2\ \text{bytes} \approx 13.1\ \text{GB}$$

(the factors, in order: tokens, layers, keys-and-values, heads, head dimension, bytes per number).

An RNN carrying a $d$-dimensional state through the same 32 layers stores $32\times d$ numbers **and that is the whole thing** — a few megabytes, and it does not grow by one byte as the conversation continues. ▸ **For a phone doing always-on keyword spotting, or a call-centre system transcribing an hour of speech, "memory that does not grow" is not an optimization — it is the difference between possible and impossible.**

**Reason 2, unpacked: what state-space models actually changed.** Recall from §9.2 that an RNN's cost is serial: step $t$ cannot begin until step $t-1$ finishes. The reason is the nonlinearity **inside the loop**. Take it out — make the recurrence linear, $h_t = Ah_{t-1} + Bx_t$ — and something changes categorically:

> **Analogy.** You have a list of a million numbers and want every running total. Left to right, that is a million sequential additions. But addition is **associative**, so you can instead build a tree: sum adjacent pairs, then adjacent pairs of pairs, and so on. Twenty rounds instead of a million steps, with each round fully parallel. The answer is identical.

That trick — a **parallel associative scan** — computes a linear recurrence in $\mathcal{O}(\log T)$ depth instead of $\mathcal{O}(T)$. It works because composing linear maps is associative. It does **not** work for $h_t = \tanh(Wh_{t-1} + \dots)$, because nonlinear composition is not associative and there is no tree to build.

▸ **So the RNN never lost on expressiveness. It lost on parallelism — and its descendants won that back by giving up the nonlinearity inside the loop and putting it between layers instead.** S4 (Structured State Space Sequence model) and **Mamba** are, mathematically, linear recurrences with carefully structured transition matrices, trained by scan and run recurrently at inference. Mamba's specific contribution is making $A$ and $B$ **input-dependent** — a selectivity that is, in spirit, exactly the LSTM's data-dependent forget gate, restored to a form that parallelizes.

**Reason 3, spelled out.** Three ideas from this chapter are load-bearing in every architecture that replaced it: the **additive/gated path** became the residual stream; **query–key–value** became attention; and the **hidden state as a compressed summary** is precisely what a KV cache is *not*, which is why the tension between the two designs is still live rather than settled.

---

## Did you know?

- **The vanishing gradient problem was diagnosed in a master's thesis written in German.** Sepp Hochreiter's 1991 diploma thesis at TU Munich, supervised by Jürgen Schmidhuber, identified the exact mechanism. It was never translated and circulated very little; an independent English analysis by Bengio, Simard and Frasconi in 1994 is what put the result in front of the field.

- **The LSTM's most famous equation was not in the LSTM paper.** The forget gate — the $f_t\odot c_{t-1}$ term that everyone draws — was added three years after the 1997 original by Felix Gers, Schmidhuber and Fred Cummins, in a paper titled "Learning to Forget." The original memory cell had input and output gates and **no way to clear itself**.

- **"Long Short-Term Memory" parses as long *short-term* memory.** The weights are a network's long-term memory; the activations are its short-term memory. The claim in the name is that this short-term memory lasts a long time — not that the model has two kinds of memory.

- **"Constant error carousel" is the authors' own phrase**, and they named the *mechanism* before the architecture. It tells you what they thought the contribution was: a path along which error circulates without being attenuated.

- **Backpropagation through time was named in 1990 but derived much earlier.** Paul Werbos, whose 1974 Harvard PhD thesis described reverse-mode differentiation for neural networks years before the field noticed, wrote the paper that gave BPTT its name.

- **The GRU was a subsection in a paper about statistical machine translation.** Cho and colleagues (2014) introduced it inside an encoder–decoder built to *rescore* candidates from a conventional phrase-based translation system. The system is long obsolete; the gate design is still everywhere.

- **Reversing the input sentence bought five BLEU points.** Sutskever and colleagues found that feeding the source backwards improved translation dramatically — a hint so blunt about the architecture's defect that attention was published within a year.

- **The attention paper does not have "attention" in its title.** Bahdanau, Cho and Bengio called it "Neural Machine Translation by Jointly Learning to Align and Translate," and described the mechanism as a **soft search** over the source. The name settled in afterwards.

- **BLEU's "U" stands for "understudy."** BiLingual Evaluation Understudy — the theatrical sense: a stand-in who performs when the human judge is unavailable.

- **A tanh RNN's gradient across 100 steps at $\|W_{hh}\|_2 = 0.9$ is 27 parts per million.** At $1.1$ it is $13{,}781\times$ too large. There is no third option available, because the same matrix sets both.

- **Elman's 1990 network discovered grammatical categories nobody told it about.** Trained only to predict the next word, its hidden states clustered into nouns, verbs, animates and inanimates. That next-token prediction produces linguistic structure as a side effect is the founding observation behind everything in Chapter 13.

- **CTC deleted an entire stage of the speech recognition pipeline by refusing to choose.** Rather than aligning audio frames to letters, it sums over *every possible* alignment — around $10^{145}$ of them for a one-second clip — and computes that sum exactly in about 500 operations by dynamic programming.

- **Recurrence came back by giving up its nonlinearity.** State-space models and Mamba are linear recurrences, which makes them computable by a parallel associative scan in $\mathcal{O}(\log T)$ depth. The same trick lets you sum a million numbers in twenty rounds instead of a million steps, and it exists only because addition is associative.

- **A transformer's memory grows with the conversation; an RNN's does not.** A 100,000-token KV cache for a mid-sized model is well over ten gigabytes. The same context in a recurrent model is a few megabytes, fixed — which is why on-device speech and keyword spotting never left recurrent architectures at all.

---

## Check for Understanding

**A plain RNN cannot both remember and transform, because the same matrix does both and its repeated product either vanishes or explodes; the LSTM fixes this by making memory additive and gated, and attention fixes the seq2seq bottleneck by replacing one fixed summary vector with a query-dependent soft lookup over all positions — which, once applied to a sequence attending to itself and stripped of recurrence, is the transformer.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why can't a plain RNN both remember for a hundred steps and stay numerically stable?** Answer in one dimension, with one number, before reaching for matrices.
2. **Why is vanishing a *sufficient* condition and exploding only a *possible* one — and why does that asymmetry change how you debug?**
3. **Why is exploding gradient the easy failure and vanishing gradient the dangerous one?** (Hint: what does each one look like on your screen?)
4. **What are the two state vectors in an LSTM, and what is each one for?** Why does a plain RNN's single vector cause a conflict?
5. **Why is a diagonal Jacobian fundamentally different from a full one when you raise it to the hundredth power?**
6. **Why does initializing the forget-gate bias to +1 matter so much?** Explain the chicken-and-egg problem it breaks.
7. **In what sense is a residual connection an LSTM cell with the gates welded open?**
8. **What can an LSTM do that a GRU structurally cannot, and why?** (Think about what a convex combination can never produce.)
9. **Why was the seq2seq bottleneck a summarization problem rather than a storage problem?** The bit-count says the vector had room.
10. **Why did reversing the source sentence buy five BLEU points, and why is that evidence rather than a trick?**
11. **What is attention doing, in terms of a photograph you took before you heard the questions?**
12. **Why did dot-product attention beat the additive version, given that they perform about the same?**
13. **Why is the transformer asymptotically worse and practically faster than an RNN?**
14. **What is exposure bias, and in what sense is RLHF a fix for it?**
15. **How does CTC train without knowing which audio frame corresponds to which letter?** Explain the summing-over-alignments idea without writing the formula.

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 10 — Tokenization, Embeddings & Vectorization](10-tokenization-embeddings-vectorization.md)
