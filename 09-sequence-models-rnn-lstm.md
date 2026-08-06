# Chapter 9 — Sequence Models: RNNs, LSTMs & Seq2Seq

> **Prerequisites:** Ch. 6.
> **Why this chapter still matters:** attention was invented to fix a specific, well-defined failure of this architecture. You cannot understand *why* the transformer looks the way it does without knowing what it replaced. This is also where the vanishing-gradient problem is at its most extreme and most instructive.

---

## 9.1 The recurrent neural network

### The one-line idea

Carry a hidden state forward through time, updating it with each new input, so the state summarizes everything seen so far.

### The analogy

Reading a novel while keeping only a single index card of notes. At every page you rewrite the card from scratch, using the old card plus the new page. The card's fixed size is the whole problem: by chapter 30, everything from chapter 1 has been overwritten many times.

### The math

▸ $$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b),\qquad y_t = W_{hy}h_t$$

Parameters are **shared across time** — the same $W_{hh}$ at every step. This is the temporal analogue of a convolution's weight sharing, and it gives the same benefit: the model can handle sequences of any length with a fixed parameter count.

---

## 9.2 Backpropagation through time, and why it fails

Unroll the network into a $T$-layer feedforward net with tied weights, then apply Ch. 6's backprop. The gradient w.r.t. the shared weight sums over all timesteps:

$$\frac{\partial\mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T}\frac{\partial\mathcal{L}_t}{\partial W_{hh}},\qquad \frac{\partial\mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t}\frac{\partial\mathcal{L}_t}{\partial h_t}\underbrace{\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)}_{\text{the killer}}\frac{\partial h_k}{\partial W_{hh}}$$

With $\frac{\partial h_j}{\partial h_{j-1}} = W_{hh}^\top\,\mathrm{diag}(\sigma'(\cdot))$:

▸ $$\left\|\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right\| \le \left(\|W_{hh}\|_2\cdot\max|\sigma'|\right)^{t-k}$$

**The condition (Pascanu et al., 2013):** let $\lambda_1$ be the largest singular value of $W_{hh}$ and $\gamma = \max|\sigma'|$ ($\gamma=1$ for tanh, $1/4$ for sigmoid).

- $\lambda_1\gamma < 1$ ⇒ **gradients vanish exponentially** — a *sufficient* condition for vanishing.
- $\lambda_1\gamma > 1$ ⇒ gradients *may* explode.

**Numbers.** tanh RNN, $\|W_{hh}\|_2 = 0.9$, sequence length 100: $0.9^{100} = 2.7\times10^{-5}$. The gradient from timestep 100 to timestep 1 is 27 parts per million. **The model cannot learn a dependency spanning 100 steps.** With $\|W_{hh}\|_2 = 1.1$: $1.1^{100} = 13{,}781$ — overflow.

▸ **The knife-edge is exact and unavoidable in a plain RNN.** There is no setting of $W_{hh}$ that both remembers for 100 steps and stays stable, because the same matrix does both jobs. Every fix — LSTM, GRU, attention, residual connections — is a way of *decoupling* memory from transformation.

### Exploding gradients: solved
**Gradient clipping** (Ch. 4 §4.8) handles this completely. It is not a hack; it is the standard, principled fix.

### Vanishing gradients: needs architecture
Clipping cannot help — you can't amplify a zero. This required the LSTM.

### Practical BPTT

**Truncated BPTT:** carry the hidden state forward but only backpropagate $k$ steps. Standard $k$: 35–200. Biases the model against dependencies longer than $k$, which is usually acceptable and always necessary.

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

### The constant error carousel

The cell-state update's Jacobian is:
▸ $$\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)$$

**Diagonal, not a full matrix.** So the product over $T$ steps is $\prod_t f_t$ — an elementwise product of numbers in $(0,1)$, *chosen by the network per unit per timestep*.

- If $f_t\approx1$ for some unit, its gradient passes through **undiminished for arbitrarily many steps**.
- Different units can have different forget rates, giving multiple simultaneous timescales.

▸ **This is why the LSTM works, and it is the same trick as a residual connection:** replace a matrix product with an (approximately) additive path. `c_t = f·c_{t-1} + i·c̃_t` is a *gated* residual stream. Chapter 11's transformer residual stream is the ungated version of the same idea.

**Practical:** initialize $b_f$ to **+1 or +2** so $\sigma(b_f)\approx0.73$–$0.88$ at the start — the network begins in "remember" mode and learns to forget, rather than starting in amnesia. Jozefowicz et al. showed this is worth more than most architectural search.

**Peepholes** let the gates see $c_{t-1}$ directly; rarely used now.

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

**LSTM vs GRU:** empirically indistinguishable on most tasks (Chung et al., 2014; Greff et al.'s 5,400-run ablation). GRU is slightly better on small data; LSTM slightly better on very long sequences and on tasks requiring unbounded counting. **The honest answer in an interview is "they're equivalent in practice; pick GRU for speed, LSTM if your sequences are very long."**

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

▸ **The conceptual leap of the transformer was not attention.** It was: (a) using attention *within* a sequence to itself (self-attention) rather than only across two sequences, and (b) deleting the recurrence entirely, so all positions compute in parallel.

### Why removing recurrence mattered so much

| | RNN | Transformer |
|---|---|---|
| Sequential ops | $O(T)$ | $O(1)$ |
| Path length between any two positions | $O(T)$ | $O(1)$ |
| Compute per layer | $O(T d^2)$ | $O(T^2d + Td^2)$ |
| Parallelizable over $T$ during training | ✗ | ✓ |

▸ The **$O(1)$ path length** is the theoretical win (no vanishing gradient across distance at all), but the **parallelism** is the practical one: an RNN cannot use a modern GPU's throughput because step $t$ must wait for step $t-1$. The transformer's $O(T^2)$ compute is *worse* asymptotically and *vastly better* in wall-clock, because it is one big matmul. **This is the single most important lesson in modern ML systems design: hardware-friendly beats asymptotically-cheaper.**

---

## 9.7 Training details that generalize beyond RNNs

### Teacher forcing and exposure bias

**Teacher forcing:** at training time, feed the *ground-truth* previous token to the decoder instead of its own prediction. Makes training parallel and stable.

▸ **Exposure bias:** at inference the model consumes its own outputs, so it encounters states it never saw in training, and errors compound. Mitigations: scheduled sampling (anneal from ground truth to model samples), sequence-level training (minimum risk training, RL with BLEU as reward).

**Modern status:** every LLM is trained with teacher forcing and exposure bias is real but empirically small at scale. RLHF (Ch. 16) is, among other things, a fix for it — it trains on the model's own sampled continuations.

### CTC — alignment-free sequence loss

For speech and handwriting where input and output lengths differ and alignment is unknown. Introduce a blank symbol $\varnothing$; define a many-to-one collapse $\mathcal{B}$ (remove repeats, then remove blanks). Then

▸ $$p(y\mid x) = \sum_{\pi\in\mathcal{B}^{-1}(y)} \prod_{t=1}^{T}p(\pi_t\mid x)$$

The sum is over exponentially many alignments, computed in $O(T|y|)$ by a **forward–backward dynamic program** (identical in structure to HMM training). Assumes conditional independence of outputs given the input, which is CTC's main weakness — hence the RNN-Transducer, which adds an output-side language model.

### Truncated BPTT state handling
Detach the hidden state between chunks (`h = h.detach()`), or the graph grows without bound.

---

## 9.8 Do RNNs still matter?

▸ Yes, in three places:

1. **Streaming/low-latency inference.** An RNN has $O(1)$ per-token state; a transformer's KV cache is $O(T)$ and grows without bound. On-device keyword spotting and real-time ASR still use recurrent models.
2. **Their descendants dominate the efficiency frontier.** State-space models (S4, **Mamba**) and linear attention are, mathematically, linear RNNs with structured state transitions — recast so they can be trained in parallel via associative scan (Ch. 12 §12.6). Mamba is a gated linear RNN with input-dependent transitions. The recurrent idea came back with the parallelism problem fixed.
3. **Conceptually.** The gating/additive-path insight is the direct ancestor of residual streams, and the query–key–value framing came from here.

---

## Check for Understanding

**A plain RNN cannot both remember and transform, because the same matrix does both and its repeated product either vanishes or explodes; the LSTM fixes this by making memory additive and gated, and attention fixes the seq2seq bottleneck by replacing one fixed summary vector with a query-dependent soft lookup over all positions — which, once applied to a sequence attending to itself and stripped of recurrence, is the transformer.**

---

**Next:** [Chapter 10 — Tokenization, Embeddings & Vectorization](10-tokenization-embeddings-vectorization.md)
