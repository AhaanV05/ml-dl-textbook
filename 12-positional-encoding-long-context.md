# Chapter 12 — Positional Information & Long Context

> **Prerequisites:** Ch. 11.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim once; each is unpacked properly where it first appears. Almost everything in this chapter is **an angle, a distance, or a byte count** — those three quantities carry the whole story.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`P`$ | "P" | A **permutation matrix** — a shuffler that reorders a list without changing its contents |
| $`pos`$, $`m`$, $`n`$ | "pos", "m", "n" | **Absolute position**: which slot in the sequence a token sits in (1st, 2nd, 57th…) |
| $`i - j`$ | "i minus j" | **Relative distance**: how many slots apart two tokens are |
| $`PE_{(pos,\,2i)}`$ | "P-E at pos, dimension 2i" | Entry number $`2i`$ of the positional code for slot $`pos`$ |
| $`\omega_i`$, $`\theta_i`$ | "omega-i", "theta-i" | The **angular frequency** of dimension-pair $`i`$ — how fast that pair spins as position advances |
| $`b_{i-j}`$ | "b sub i-minus-j" | A **learned number added to an attention score**, chosen purely by distance |
| $`m_h`$ | "m sub h" | ALiBi's per-head **slope** — how fast head $`h`$ loses interest with distance |
| $`R_m`$ | "R sub m" | A **rotation matrix**: turn a vector by an angle proportional to position $`m`$ |
| $`\langle a, b\rangle`$ | "inner product of a and b" | The dot product $`a^\top b`$ — one number measuring **alignment** |
| $`\mathcal{O}(T^2)`$ | "big-O of T squared" | Cost grows with the **square** of sequence length: double $`T`$, quadruple the bill |
| $`h_{kv}`$, $`d_{\text{head}}`$ | "h sub k-v", "d head" | Number of **key/value heads**, and the width of one head |
| $`\bar A,\ \bar B`$ | "A-bar, B-bar" | **Discretized** state-space matrices. ⚠ Here a bar means "discretized," *not* a gradient (contrast §0.6) |
| $`*`$ | "convolved with" | **Convolution** — slide a kernel along a sequence and take dot products |
| $`\phi(\cdot)`$ | "phi of" | A **feature map** applied to queries and keys, replacing softmax in linear attention |
| $`m,\ \ell`$ (FlashAttention) | "m", "ell" | Running softmax **max** and running **sum**. ⚠ $`m`$ is overloaded — position in §12.4, a maximum in §12.5 |

**Full forms for the abbreviations in this chapter.** Say each aloud once; most stop being intimidating the moment you hear what they stand for.

| Short | Full form |
|---|---|
| ALiBi | Attention with Linear Biases |
| bf16 | brain floating point, 16-bit |
| FFT | Fast Fourier Transform |
| FLOP | FLoating-point OPeration |
| GQA | Grouped-Query Attention |
| HBM | High-Bandwidth Memory (the big, slow-ish memory on a GPU board) |
| HiPPO | High-order Polynomial Projection Operators |
| KV | Key–Value (as in "KV cache") |
| LSH | Locality-Sensitive Hashing |
| MHA | Multi-Head Attention |
| MLA | Multi-head Latent Attention |
| MQA | Multi-Query Attention |
| NTK | Neural Tangent Kernel (borrowed here only as a name for a scaling trick) |
| PE | Positional Encoding |
| PI | Position Interpolation |
| RoPE | Rotary Position Embedding |
| S4 / S6 | Structured State Space Sequence model (S6 = the selective version inside Mamba) |
| SRAM | Static Random-Access Memory (the tiny, very fast memory on the GPU chip itself) |
| SSM | State-Space Model |
| YaRN | Yet another RoPE extensioN |

---

## 12.1 Why position needs encoding at all

### The one-line idea

Attention is a weighted average over a *set*, so without extra information a transformer literally cannot tell "dog bites man" from "man bites dog."

### The proof

Let $`P`$ be a permutation matrix. Then $`\mathrm{Attention}(PX) = P\,\mathrm{Attention}(X)`$ — self-attention is **permutation equivariant**. The FFN is applied position-wise, so it is too. Therefore the whole transformer is, and a token's output depends only on the *multiset* of tokens present.

▸ Position must be injected explicitly. There are two families: **absolute** (add position information to the token representation) and **relative** (make the attention score depend on $`i-j`$).

#### What permutation equivariance actually says

Every symbol first.

- A **permutation matrix** $`P`$ is a square grid of 0s and 1s with exactly one 1 in each row and each column. Multiplying by it **reorders a list**. Nothing else — no scaling, no mixing, no loss of information.
- $`X`$ is the stack of token vectors, one row per token: $`X \in \mathbb{R}^{T\times d}`$ for $`T`$ tokens each described by $`d`$ numbers.
- **Equivariant** means *"the operation and the shuffle commute."* Shuffle-then-attend gives literally the same answer as attend-then-shuffle. Read $`\mathrm{Attention}(PX) = P\,\mathrm{Attention}(X)`$ aloud as: **"attention doesn't care what order you hand it the tokens in — it just carries the order along."**
- A **multiset** is a bag of items where duplicates count but order does not. $`\{\text{dog}, \text{bites}, \text{man}\}`$ and $`\{\text{man}, \text{bites}, \text{dog}\}`$ are the same multiset.

**Work it with $`T = 2`$.** Take two tokens $`x_1`$ and $`x_2`$, and let $`P`$ be the swap:

$$P = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$$

Token 1's output is $`\alpha_{11}v_1 + \alpha_{12}v_2`$, where $`\alpha_{11}`$ is "how much token 1 attends to token 1." Those weights come from dot products $`q_1^\top k_1`$ and $`q_1^\top k_2`$ — and a dot product between two specific vectors is the same number no matter which row of the array either vector was sitting in. So swapping the rows just swaps which output row the same numbers land in. **Order never entered the arithmetic anywhere.**

> **Analogy.** A blender. Put in a banana, then strawberries, then milk; or milk, strawberries, then banana. The smoothie is identical, because blending is a set operation. Attention is a blender. If the recipe  depends on order — "add the eggs *after* the pan is hot" — a blender cannot express it, and you must attach a timestamp to each ingredient before it goes in. Positional encoding is that timestamp.

**Why the rest of the transformer doesn't rescue you.** The feed-forward network is applied **position-wise**: the same small MLP runs independently on each token's vector, with no communication between positions. So it is equivariant too. Layer normalization is per-token. Residual additions are per-token. **Attention is the only layer that mixes across positions, and it is blind to order** — therefore the entire stack is.

▸ **The consequence is not subtle: without positional information a transformer is a bag-of-words model with a very expensive feature extractor.** It could not distinguish "the cat sat on the mat" from "the mat sat on the cat," nor "$`3 - 5`$" from "$`5 - 3`$." Everything in this chapter exists to repair that one hole, and the different repairs differ mainly in how gracefully they behave at lengths the model never saw in training.

> **Where this came from.** Permutation invariance was studied as a *feature*, not a bug, before transformers made it a problem. **Deep Sets** (Zaheer, Kottur, Ravanbhakhsh, Poczos, Salakhutdinov and Smola, 2017) characterized exactly which functions can act on unordered sets, and was motivated by point clouds and population statistics — tasks where order  is meaningless. The transformer arrived the same year needing the opposite property. The same mathematical fact is a theorem in one paper and a defect to be patched in the other, which is a fair illustration of how much of architecture design is choosing which symmetries you want to keep.

---

## 12.2 Absolute positional encodings

### Sinusoidal (Vaswani et al.)

▸ $$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Added to the token embedding.

#### Reading the sinusoidal formula in plain English

This formula frightens people out of proportion to its content. It is **a bank of clocks**, and nothing more.

Decode each piece:

| Piece | Read aloud | What it is |
|---|---|---|
| $`pos`$ | "pos" | Which slot in the sentence: $`0, 1, 2, \dots`$ |
| $`i`$ | "i" | Which *pair* of dimensions we're filling: $`i = 0, 1, \dots, d/2 - 1`$ |
| $`2i`$, $`2i+1`$ | "two-i", "two-i plus one" | Even and odd slots in the vector. Each pair $`(2i,\,2i{+}1)`$ gets a $`\sin`$ and a $`\cos`$ — **one clock face, two hands' worth of information** |
| $`10000^{2i/d}`$ | "ten-thousand to the two-i over d" | The **period** of clock $`i`$. Small $`i`$ → small number → fast clock. Large $`i`$ → close to 10000 → very slow clock |
| $`d`$ | "d" | The model width, e.g. 512 |

▸ **The whole thing in one sentence: give every position a reading off $`d/2`$ clocks running at geometrically spaced speeds, and write down each clock's sine and cosine.** Position $`pos`$ becomes a $`d`$-number "timestamp," and that timestamp is added onto the token's meaning vector.

**Put real numbers in.** Take $`d = 4`$, so there are just two clocks: $`i=0`$ and $`i=1`$.

- Clock 0: $`10000^{0/4} = 1`$, so the angle is $`pos/1 = pos`$ radians. It completes a full turn every $`2\pi \approx 6.3`$ positions. **Fast.**
- Clock 1: $`10000^{2/4} = 100`$, so the angle is $`pos/100`$. It completes a full turn every $`628`$ positions. **Slow.**

So $`PE_{pos} = \big(\sin(pos),\ \cos(pos),\ \sin(pos/100),\ \cos(pos/100)\big)`$. At $`pos = 0`$ that's $`(0, 1, 0, 1)`$; at $`pos = 1`$ it's $`(0.841, 0.540, 0.010, 1.000)`$. The fast clock has moved a lot; the slow clock has barely twitched.

> **Analogy.** A mechanical odometer, or the hands of an analogue watch. The seconds hand tells you precisely *where in the minute* you are but is useless for telling you the hour; the hour hand is the reverse. Read all the hands together and you get a reading that is both precise and unambiguous over a huge range. The sinusoidal encoding is a watch with $`d/2`$ hands, geared in a geometric ratio, and every token is stamped with the time it arrived.

**Why sine *and* cosine, rather than just one?** Because $`\sin`$ alone is ambiguous: $`\sin(0.5) = \sin(\pi - 0.5)`$, so two different angles give the same reading. Pairing it with $`\cos`$ pins the angle down uniquely — the pair $`(\sin\alpha, \cos\alpha)`$ is a **point on the unit circle**, and a point on a circle names exactly one angle. Two numbers per clock, no ambiguity.

**Why sinusoids?** Because a shift is a **linear** transformation of the encoding. For a fixed offset $`k`$, using the angle-addition identities:

$$\begin{pmatrix}\sin(\omega_i(pos+k))\\\cos(\omega_i(pos+k))\end{pmatrix} = \begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\-\sin\omega_ik&\cos\omega_ik\end{pmatrix}\begin{pmatrix}\sin(\omega_i\,pos)\\\cos(\omega_i\,pos)\end{pmatrix}$$

▸ $`PE_{pos+k} = M_k\,PE_{pos}`$ with $`M_k`$ **independent of $`pos`$** — a rotation by a fixed angle in each 2-D subspace. So relative position is linearly recoverable, and the encoding extrapolates to positions never seen.

#### Unpacking "a shift is a rotation"

The matrix in the display above is the standard **2-D rotation matrix**. Read $`M_k`$ as *"turn the clock hand forward by angle $`\omega_i k`$."*

The claim being made has a precise and useful shape. Read it as three separate statements:

1. **Moving forward $`k`$ positions = turning every clock hand forward by a fixed angle.** Clock $`i`$ turns by $`\omega_i k`$. That's what a clock *does*.
2. **How much you turn depends on $`k`$ only — never on where you started.** $`M_k`$ contains no $`pos`$. Advancing from position 3 to position 8 is the same rotation as advancing from position 3003 to position 3008.
3. **Therefore the network can extract relative position with a linear layer.** Attention is built out of linear maps and dot products; if "5 positions apart" is a *fixed matrix*, a linear layer can learn to detect it. If it had been some tangled nonlinear function of $`pos`$, the network would have to learn a separate detector for every starting point.

**Do it with numbers.** Take clock $`i`$ with $`\omega_i = 1`$, currently at position $`pos`$ with reading $`(\sin pos, \cos pos)`$. Advance $`k = \pi/2`$ (a quarter turn). The identity says the new reading is

$$\begin{pmatrix}\cos(\pi/2) & \sin(\pi/2)\\ -\sin(\pi/2) & \cos(\pi/2)\end{pmatrix}\begin{pmatrix}\sin pos\\ \cos pos\end{pmatrix} = \begin{pmatrix}0 & 1\\ -1 & 0\end{pmatrix}\begin{pmatrix}\sin pos\\ \cos pos\end{pmatrix} = \begin{pmatrix}\cos pos\\ -\sin pos\end{pmatrix}$$

and indeed $`\sin(pos + \pi/2) = \cos pos`$ and $`\cos(pos + \pi/2) = -\sin pos`$. ✓ The rotation matrix is just the angle-addition formulas written as a grid.

▸ **This is the property that RoPE (§12.4) takes and pushes to its conclusion.** Sinusoidal encoding makes relative position *linearly recoverable in principle*; RoPE makes it *arithmetically automatic*, by rotating the query and key vectors themselves instead of adding a timestamp to the token. Everything in §12.4 is this same rotation identity, applied one step earlier in the pipeline.

**Wavelengths** range from $`2\pi`$ (fastest, $`i=0`$) to $`10000\cdot2\pi`$ (slowest). It is a **binary-like positional code in continuous form**: fast dimensions give fine local resolution, slow dimensions give coarse global position.

#### The binary-code analogy, made concrete

"Binary-like positional code" is the sentence that makes the whole design click, so it is worth spelling out.

Write the numbers 0 through 7 in binary:

| Number | bit 2 | bit 1 | bit 0 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 |

Look down the columns. **Bit 0 flips every step. Bit 1 flips every 2 steps. Bit 2 flips every 4 steps.** Each column is a square wave, and each is twice as slow as the one to its right. Three bits distinguish eight positions.

Sinusoidal encoding is exactly this, with two changes: the square waves become smooth sine waves (so the code is differentiable and generalizes between integers), and the ratio between consecutive frequencies is a gentler geometric factor rather than a hard 2×. **Fast dimensions are the low-order bits — they tell you exactly where you are locally but wrap around constantly. Slow dimensions are the high-order bits — they never wrap, but they can't resolve neighbours.**

▸ **Why a geometric spacing rather than, say, linear?** Because a geometric ladder covers an enormous range with few rungs. With $`d = 512`$ there are 256 clocks spanning periods from $`6.3`$ to $`62{,}800`$ — a range of four orders of magnitude — and adding more dimensions extends the range multiplicatively, not additively. Linear spacing would need thousands of dimensions to cover the same span. This is the same reason a slide rule, a piano keyboard, and the decibel scale are all logarithmic: **when the quantities you care about span orders of magnitude, you space your markers geometrically.**

**Weakness:** it's *added* to the embedding, so position and content compete for the same $`d`$ dimensions, and the model must learn to disentangle them. Extrapolation works in theory but is poor in practice.

#### Why "added to the embedding" is the problem

This weakness sounds mild and is not. Spell out the arithmetic: the vector that enters layer 1 is

$$h_{pos} = \underbrace{E[x_{pos}]}_{\text{what the word means}} + \underbrace{PE_{pos}}_{\text{where it sits}}$$

Two completely different kinds of information are **summed into the same $`d`$ numbers**. There is no separate "position channel"; there is one channel carrying a sum, and the network has to learn to pull the two apart from statistics alone.

> **Analogy.** Two people speaking into the same microphone at once. The recording contains both voices, and in principle you can separate them — they have different characteristic frequencies — but you have spent your recording budget twice and every downstream listener now has an extra job. A separate microphone per speaker would have been strictly better. Relative schemes (§12.3) and RoPE (§12.4) are, in effect, the second microphone: they inject position at the point where positions are *compared*, rather than smuggling it into the content.

▸ **And "extrapolation works in theory but is poor in practice" deserves a sharper statement.** The formula happily produces $`PE_{9999}`$ for a model trained only to position 512 — the sines and cosines are perfectly well-defined. But the *slow* clocks have only ever been observed over a tiny arc of their cycle during training. Clock $`i`$ with period 62,800 moves through less than 1% of a full turn across a 512-token training window, so every weight that reads it was fitted on a nearly-straight line segment. Ask it to interpret a reading from the other side of the circle and it has no basis for an answer. **The formula extrapolates; the learned weights that consume it do not.** This exact distinction returns in §12.4 as the "NTK" argument, and it is the single most reusable idea in the chapter.

### Learned absolute

$`PE\in\mathbb{R}^{T_{\max}\times d}`$, trained. Used by BERT, GPT-2, ViT. Slightly better in-distribution.

▸ **Fatal flaw: hard length limit.** Position $`T_{\max}+1`$ has no embedding, and untrained positions are garbage. This is the direct cause of GPT-2's 1024-token wall.

#### Learned absolute positions, decoded

Read $`PE \in \mathbb{R}^{T_{\max}\times d}`$ as *"a lookup table with $`T_{\max}`$ rows and $`d`$ columns"* — literally a second embedding table, but indexed by **slot number** instead of by word. Row 0 is "the vector meaning *first*," row 1 is "the vector meaning *second*," and so on. All of it is learned by gradient descent along with everything else.

This is appealingly simple: you stop guessing what a good positional code looks like and let the data decide. And in-distribution it wins slightly, because the model can encode whatever quirks of position actually matter in its corpus.

**But count the rows.** GPT-2 has $`T_{\max} = 1024`$. There is no row 1024. There is no row 1025. Not "a bad row" — **no row at all**, the way an array of length 1024 has no index 1024.

> **Analogy.** A theatre with 1024 numbered seats. Sinusoidal encoding is a tape measure laid down the aisle: ask where 3 metres past the back wall is and it will tell you, even though there is no floor there. A learned table is a printed seating chart. Ask for seat 1025 and you do not get a wrong answer, you get *nothing* — and if the software papers over the gap with a fresh random row, the model receives a vector it has never seen in its life, in the middle of a sentence.

▸ **Every scheme in the rest of this chapter can be read as an answer to one question: what happens at position $`T_{\max}+1`$?** Learned tables crash. Sinusoids produce numbers the weights can't interpret. Relative biases and ALiBi degrade gently because they only ever look at *differences*, which stay in a familiar range near the diagonal. RoPE keeps working until the slow clocks leave their trained arc — which is exactly what PI, NTK-scaling and YaRN are built to postpone.

> **Where this came from.** The sinusoidal scheme appeared in **"Attention Is All You Need"** (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin, Google, 2017) — the paper that introduced the transformer. The paper is unusually candid about it: the authors report that they *also* tried learned positional embeddings and found the two produced **nearly identical results**, and say they chose the sinusoidal version because they hypothesized it might let the model extrapolate to lengths longer than those seen in training. That hypothesis turned out to be largely wrong in practice, which is a nice reminder that the reasoning offered for a design choice and the reason it survives are often different things. The base constant 10000 is stated without derivation and has been inherited, essentially unquestioned, by an enormous amount of subsequent work — including RoPE, where it became the single most important tuning knob for long context.

#### Examples and non-examples: what counts as a positional encoding

**✅  examples**

| Example | Why it qualifies |
|---|---|
| Sinusoidal $`PE_{pos}`$ added to the token embedding | The vector entering layer 1 — and therefore every score computed from it — becomes a function of $`pos`$ |
| BERT's learned table $`PE\in\mathbb{R}^{512\times768}`$ | Row $`pos`$ is a distinct trained vector per slot; which row was added changes the score |
| T5's bias $`b_{i-j}`$ added to the logit | Injects a number that depends on nothing but the gap between the two tokens |
| ALiBi's $`-m_h(i-j)`$ | Same place, same job, with the number fixed in advance instead of learned |
| RoPE's rotation of $`q_i`$ and $`k_j`$ | After rotation the dot product is a function of $`i-j`$; position has entered the score itself |

**❌ Near-misses — look like they encode position, but don't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| The order of rows in the input tensor, `x[0], x[1], x[2], …` | This is the *problem*, not the fix. §12.1 showed that permuting those rows permutes the outputs and changes nothing else | Array layout — the thing positional encoding exists to compensate for |
| The causal mask | It controls *which* keys are visible, not how far away they are. Two visible tokens 3 apart and 3,000 apart are treated identically by the mask | A visibility constraint. It does leak a *count* — a token at slot $`i`$ sees exactly $`i+1`$ keys, which is why "NoPE" decoder-only models work at all — but it supplies no notion of distance |
| BERT's segment embedding $`E_A`$ / $`E_B`$ | Every token in sentence A gets the **same** vector, whether it is the 1st or the 40th | A sequence-membership tag |
| A `[CLS]` token parked at slot 0 | Marks a place to read an answer out of; says nothing about the gap between any two other tokens | A pooling convention |
| Rotating $`V`$ by position as well as $`Q`$ and $`K`$ | It *does* depend on position — that is exactly the trouble. A token's **content** would change with where it sits | A bug. Position belongs in the score, not the payload (§12.4) |

▸ **The boundary:** a positional encoding is anything that makes the attention score between token $`i`$ and token $`j`$ a function of $`i`$, $`j`$, or $`i-j`$. Machinery that changes *which* tokens are visible, or *what* they carry, is doing a different job — however much order-related information it happens to leak on the side.

> **Why summing doesn't wreck the vector.** A natural worry: adding $`PE_{pos}`$ to the embedding must destroy the word's meaning. It **degrades** the vector; it does not destroy it, and the reason is dimension. In $`d = 768`$ there is an enormous amount of room: two vectors drawn at random in 768 dimensions have an expected cosine of 0 with a spread of only about $`1/\sqrt{768}\approx0.036`$, so nearly everything is nearly perpendicular to nearly everything else. The positional codes occupy their own structured set of directions, and the trained $`W_Q`$ and $`W_K`$ can read around them. The real price is **capacity, not corruption** — some of the residual stream's budget now goes on bookkeeping. The belief is tempting because in the two or three dimensions where our geometric intuition lives, adding one vector to another  does wreck it. High-dimensional space is far roomier than it feels.

> **Common misconception.** *"Position is injected once at the bottom of the network, so by layer 40 the model has lost it."* Two things keep it alive, and they are different things. First, the residual stream is **additive** — whatever the embedding layer wrote is still present at layer 40 unless some layer deliberately subtracted it. Second, and more decisively, **RoPE is not injected once at all**: it is applied fresh at every layer, to that layer's own $`Q`$ and $`K`$, every single time attention is computed. This is a real advantage of relative schemes over absolute ones — an absolute code has to survive forty layers of residual traffic, while a relative one re-supplies the information at the exact moment it gets used. The misconception is tempting because the diagram in every tutorial draws the "+ positional encoding" box exactly once, at the very bottom.

---

## 12.3 Relative positional encodings

### Shaw et al. / T5 bias

Add a learned scalar to each attention score based on $`i-j`$:
$$e_{ij} = \frac{(x_iW_Q)(x_jW_K)^\top}{\sqrt{d_k}} + b_{i-j}$$

T5 **buckets** relative distances logarithmically (exact for small offsets, coarse for large) and shares $`b`$ across layers but not heads. Extrapolates gracefully; costs an extra $`T\times T`$ addition.

#### Unpacking the relative-bias score

Take the formula apart left to right.

| Piece | Read aloud | Job |
|---|---|---|
| $`e_{ij}`$ | "e sub i j" | The raw **attention score** from query token $`i`$ to key token $`j`$, before softmax |
| $`x_iW_Q`$ | "x-i W-Q" | Token $`i`$'s **query** vector: "what am I looking for?" |
| $`x_jW_K`$ | "x-j W-K" | Token $`j`$'s **key** vector: "what do I advertise?" |
| $`(\cdot)(\cdot)^\top`$ | "dot product" | How well the question matches the advertisement — one number |
| $`\sqrt{d_k}`$ | "root d-k" | The scaling from Ch. 11 that stops the dot products growing with width |
| $`b_{i-j}`$ | "b sub i minus j" | **A learned number that depends on nothing but the gap.** Added on top |

▸ **The entire idea is the last term.** The first term is ordinary attention — content matching content. The second term is a learned opinion about distance alone: *"all else equal, how interested should a token be in something 7 slots back?"* Content and position are now in **separate summands** instead of tangled into one vector, which is precisely the "second microphone" from §12.2.

**Numbers, with $`T=3`$.** There are 5 possible gaps: $`i-j \in \{-2,-1,0,1,2\}`$, so 5 learned scalars per head. A causal model only ever needs $`i - j \ge 0`$, so really 3. The bias matrix is *constant along each diagonal* — every pair 1 apart gets the same bonus:

$$b = \begin{pmatrix}b_0 & \cdot & \cdot\\ b_1 & b_0 & \cdot\\ b_2 & b_1 & b_0\end{pmatrix}$$

(A matrix constant along diagonals like this is called **Toeplitz**, and the name is worth knowing because it appears whenever a system cares only about differences.)

**What "buckets logarithmically" means.** With $`T = 4096`$ there are 4096 possible gaps and you do not want 4096 learned parameters per head. T5 groups them: gaps 0, 1, 2, …, 7 each get their own bucket, then 8–9 share one, 10–12 share one, and so on with geometrically widening bins, until everything past a few hundred lands in one final "far away" bucket. Roughly 32 buckets cover the whole range.

> **Analogy.** How you describe *when* something happened. "Three minutes ago" and "four minutes ago" are meaningfully different. "Three hundred and twelve days ago" and "three hundred and thirteen days ago" are not — you say "about a year ago." Human time vocabulary is logarithmically bucketed for exactly the reason T5's is: **resolution should scale with the thing being measured, not be uniform.**

▸ **This bucketing is also why it extrapolates gracefully.** At a length beyond training, new gaps simply fall into the existing "far" buckets, which are already trained. Nothing is undefined, nothing is out of range. The price is honest and stated: an extra $`T\times T`$ matrix to build and add — small compared to attention itself, but not free, and it must be materialized, which fights the FlashAttention trick in §12.5.

### ALiBi

▸ $$e_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} - m_h\,(i-j)$$

A **linear penalty** with a head-specific slope $`m_h`$, fixed as a geometric sequence: for $`h`$ heads, $`m_h = 2^{-8h'/h}`$ for $`h'=1..h`$.

**Intuition:** each head gets a different "attention span." Small slope = long-range head; large slope = local head. No learned parameters, and it extrapolates remarkably well (train at 1k, evaluate at 16k). Used by BLOOM, MPT.

**Limitation:** the monotone decay is a strong prior. It hurts on tasks requiring precise retrieval from far away.

#### Reading ALiBi in plain English

**ALiBi** stands for **Attention with Linear Biases**, and the formula is one of the simplest ideas in this book: *subtract a penalty proportional to how far away the token is.*

- $`q_i^\top k_j / \sqrt{d_k}`$ — ordinary attention, unchanged.
- $`i - j`$ — the gap. In a causal model $`j \le i`$, so this is $`\ge 0`$.
- $`m_h`$ — a **fixed** (not learned) positive slope belonging to head $`h`$.
- The minus sign — this is always a **penalty**, never a bonus. The further away, the worse the score, always.

▸ **In one sentence: every head is given a built-in impatience, and each head is given a different amount of it.**

**Put numbers on the slopes.** With $`h = 8`$ heads, $`m_{h'} = 2^{-8h'/8} = 2^{-h'}`$ for $`h' = 1..8`$: the slopes are $`\tfrac12, \tfrac14, \tfrac18, \dots, \tfrac1{256}`$.

Now watch what those slopes *do* to a token 50 positions back:

| Head | Slope $`m_h`$ | Penalty at gap 50 | Effect on the softmax weight |
|---|---|---|---|
| 1 | $`0.5`$ | $`-25`$ | $`e^{-25}\approx 10^{-11}`$ — annihilated |
| 4 | $`0.0625`$ | $`-3.1`$ | $`e^{-3.1}\approx 0.045`$ — heavily discounted |
| 8 | $`0.0039`$ | $`-0.2`$ | $`e^{-0.2}\approx 0.82`$ — barely touched |

Head 1 cannot see 50 tokens back at any price. Head 8 hardly notices the distance. **Head 1 is a syntax head; head 8 is a discourse head.** The geometric ladder of slopes hands the model a ready-made set of attention spans covering several orders of magnitude, and the model decides what to put in each.

> **Analogy.** A newsroom. One reporter covers the street outside and knows it in minute detail; one covers the city; one covers the country. Nobody assigned anyone a topic — they were assigned a *radius*, and the topics followed. ALiBi assigns radii.

**Why it extrapolates so well.** There is no table to run off the end of and no angle to leave its trained range. At length 16k the penalty $`-m_h(i-j)`$ is computed by the same multiplication as at length 1k; it just returns a bigger number, and a bigger number inside a softmax means a smaller weight, which is a perfectly sensible thing for the model to receive. **A formula with no learned parameters and no lookup cannot be out-of-distribution.** That is the whole trick, and it is why "train at 1k, evaluate at 16k" works.

▸ **And now the limitation, sharply.** The penalty is *monotone*: score always falls with distance, with no exceptions available. If the answer to a question sits 30,000 tokens back — a needle in a haystack, a variable defined at the top of a long file — ALiBi has hard-wired the belief that it cannot matter much. **You have traded the ability to extrapolate for the ability to look far away on purpose, and those are not the same thing.** RoPE's decay, by contrast, is a soft statistical tendency rather than an enforced rule, which is a large part of why RoPE won.

> **Where this came from.** ALiBi was introduced by **Ofir Press, Noah Smith and Mike Lewis** in 2021, in a paper whose title states the entire result: *"Train Short, Test Long."* The framing is worth noticing — they set out to attack **length extrapolation** as a problem in its own right, benchmarked the existing positional schemes on it, and found that the simplest possible thing (a straight-line penalty, no parameters at all) beat the more sophisticated ones. ALiBi went on to be used in BLOOM and MPT. It is a recurring pattern in this field: when the goal is generalizing beyond the training distribution, the method with the fewest free parameters often wins, because there is less that can be fitted to the wrong thing.

#### Examples and non-examples: relative position information

**✅  relative**

| Example | Why it qualifies |
|---|---|
| T5's $`b_{i-j}`$ | The added term is indexed by the gap and nothing else |
| ALiBi's $`-m_h(i-j)`$ | Same — a function of the gap, with a head-specific slope |
| Shaw et al.'s learned relative key vectors, clipped at $`\pm k`$ | Indexed by a clipped $`i-j`$; positions beyond the clip share a bucket |
| RoPE | $`\langle R_iq,\ R_jk\rangle = \langle R_{i-j}q,\ k\rangle`$ — the score is provably a function of the gap alone |

**❌ Near-misses — described as relative, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Sinusoidal absolute $`PE`$, "because a shift is a rotation" | $`PE_{pos+k}`$ really is a fixed linear map of $`PE_{pos}`$, so the gap is *recoverable in principle*. But nothing **forces** the score to be a function of $`i-j`$; it may depend on $`i`$ and $`j`$ separately, and empirically it does | An absolute scheme with relative structure lying around unused |
| A learned absolute table | The model could learn to compare row $`i`$ against row $`j`$ — but there is no constraint making it, and no reason position 900 behaves like position 100 | Absolute |
| Sliding-window attention with window $`w`$ | Everything inside the window is treated identically and everything outside is invisible. There is no gradation *within* the window | A relative **mask**, not a relative encoding |
| Normalizing position as $`i/T`$ | Shifting the whole sequence by one slot changes every value, and the same token gets a different code in a longer document | Rescaled absolute position |
| "RoPE is relative, so it never uses absolute position" | RoPE rotates $`q_i`$ by exactly $`i\theta`$ — an absolute angle, computed from an absolute index | Absolute rotations whose absoluteness *cancels* in the dot product |

▸ **The boundary:** a scheme is relative if and only if **adding the same constant to every token's index leaves every attention score unchanged.** That is a test you can run: slide the whole sequence one slot to the right and see whether any score moves. Under T5 bias, ALiBi and RoPE, nothing moves. Under sinusoidal or learned absolute encodings, everything does.

> **Common misconception.** *"ALiBi and RoPE both make attention fall off with distance, so they encode the same prior."* Plot the two decay curves and they look like cousins. The difference is whether the model is allowed to disagree. ALiBi **subtracts** $`m_h(i-j)`$ from the logit: with slope $`0.5`$ and a gap of 50 that is $`-25`$, and $`e^{-25}\approx10^{-11}`$, so no content match of any strength can recover that weight — the decay is a rule. RoPE subtracts nothing; its decay is a *statistical* tendency that emerges because many clocks drift out of phase and their contributions cancel by chance, so a strong enough query–key match still produces a large score across 50,000 tokens. **One is a constraint the model cannot argue with; the other is a prior it can.** That single difference is why ALiBi extrapolates beautifully and retrieves badly, and it is easy to miss because the two schemes are almost always introduced side by side under the same heading, "long-range decay."

---

## 12.4 RoPE — Rotary Position Embedding

**The dominant scheme in 2026.** Used by LLaMA, Mistral, Qwen, Gemma, DeepSeek, GPT-NeoX, PaLM.

### The one-line idea

Instead of adding position, **rotate** the query and key vectors by an angle proportional to their position — so their dot product automatically depends only on the *difference* in positions.

### The analogy

Two clock hands. If you rotate the hour hand of clock A by $`m`$ degrees and clock B by $`n`$ degrees, the angle *between* them depends only on $`m-n`$. Absolute rotations cancel in the comparison; relative rotation survives. RoPE does this in $`d/2`$ independent 2-D planes at once, each plane spinning at a different rate.

### The construction

Split the $`d`$-dimensional vector into $`d/2`$ consecutive pairs. For pair $`i`$ at position $`m`$, rotate by angle $`m\theta_i`$ where $`\theta_i = 10000^{-2i/d}`$:

▸ $$R_m^{(i)} = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}$$

Apply to queries and keys: $`\tilde q_m = R_mq_m`$, $`\tilde k_n = R_nk_n`$.

#### Reading the rotation matrix

Every symbol:

| Symbol | Read aloud | Meaning |
|---|---|---|
| $`m`$ | "m" | The token's **absolute position** — 0, 1, 2, … |
| $`i`$ | "i" | Which **pair of dimensions** we're rotating: $`i = 0,\dots,d/2-1`$ |
| $`\theta_i = 10000^{-2i/d}`$ | "theta-i" | Pair $`i`$'s **rotation speed**, in radians per position. Note the *minus* in the exponent: large $`i`$ → tiny $`\theta_i`$ → slow |
| $`m\theta_i`$ | "m theta-i" | The **total angle** this pair has turned by position $`m`$ |
| $`R_m^{(i)}`$ | "R-m superscript i" | The 2×2 matrix that performs that turn |
| $`\tilde q_m`$ | "q-tilde sub m" | The query **after** rotating |

▸ **The one-sentence version: chop the vector into 2-D pairs, treat each pair as an arrow on a little clock face, and spin arrow $`i`$ by $`m\theta_i`$ — a bigger spin for tokens later in the sequence, and a faster spin for earlier dimension-pairs.**

**What a 2×2 rotation matrix does, concretely.** Applied to a point $`(x,y)`$ it produces

$$\begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\cos\alpha - y\sin\alpha\\ x\sin\alpha + y\cos\alpha\end{pmatrix}$$

Take $`\alpha = 90° = \pi/2`$, so $`\cos\alpha = 0`$, $`\sin\alpha = 1`$, and the matrix is $`\begin{pmatrix}0&-1\\1&0\end{pmatrix}`$. Feed it $`(1, 0)`$ — an arrow pointing east — and you get $`(0,1)`$, pointing north. ✓ A quarter turn counter-clockwise, exactly as advertised. **Rotation matrices never change a vector's length**; they only change where it points. That will matter enormously in a moment.

**Real numbers for the speeds.** With $`d = 128`$ there are 64 pairs:

| Pair $`i`$ | $`\theta_i = 10000^{-2i/128}`$ | Full turn every… |
|---|---|---|
| $`0`$ | $`1.0`$ | $`6.3`$ positions |
| $`16`$ | $`0.1`$ | $`63`$ positions |
| $`32`$ | $`0.01`$ | $`628`$ positions |
| $`63`$ | $`\approx 0.00011`$ | $`\approx 57{,}000`$ positions |

▸ **Look hard at the last row.** In a model trained at $`T=4096`$, pair 63 turns through only $`4096 \times 0.00011 \approx 0.45`$ radians — **about 26°, less than a tenth of a full circle, across the entire training context.** It has literally never seen most of its own circle. Hold that fact; it is the entire explanation of why long-context extension is hard, and it returns as the "NTK argument" a few pages from now.

> **Analogy.** A combination lock with 64 dials, geared so dial 0 spins fast and dial 63 crawls. Each token turns all 64 dials by an amount proportional to its position. The fast dials distinguish neighbours; the slow dials distinguish chapters. The lock has been turned 4096 times during training — enough for dial 0 to have gone round 650 times, and not enough for dial 63 to complete even one revolution.

### The key property, derived

$$\tilde q_m^\top\tilde k_n = (R_mq)^\top(R_nk) = q^\top R_m^\top R_nk = q^\top R_{n-m}k$$

using $`R_m^\top R_n = R_{n-m}`$ (rotations form a group; $`R_m^{-1}=R_m^\top=R_{-m}`$).

▸ $$\boxed{\ \langle\mathrm{RoPE}(q,m),\ \mathrm{RoPE}(k,n)\rangle = g(q,k,\,n-m)\ }$$

**The attention score depends only on relative position, achieved through purely absolute operations.** That is the elegance: you rotate each vector once, independently, and relativity falls out of the inner product for free — no $`T\times T`$ bias matrix, no extra memory, fully compatible with the KV cache (cached keys are already rotated at their own absolute position and stay valid).

#### Why the absolute rotations cancel

This four-step derivation is the most important line of algebra in the chapter, and each step is a single, small move. Walk it:

$$\underbrace{\tilde q_m^\top\tilde k_n}_{\text{1}} = \underbrace{(R_mq)^\top(R_nk)}_{\text{2}} = \underbrace{q^\top R_m^\top R_nk}_{\text{3}} = \underbrace{q^\top R_{n-m}k}_{\text{4}}$$

1. **The thing we want**: the attention score between the rotated query at position $`m`$ and the rotated key at position $`n`$.
2. **Substitute the definitions.** Nothing has happened yet.
3. **Use $`(AB)^\top = B^\top A^\top`$.** Pure bookkeeping — the standard transpose rule from §0.6, moving the transpose inside and reversing the order. The $`R_m^\top R_n`$ has now been brought together in the middle.
4. **Use the group property $`R_m^\top R_n = R_{n-m}`$.** This is the payoff step, and it is a fact about rotations, not about transformers.

**Why $`R_m^\top R_n = R_{n-m}`$.** A rotation matrix is orthonormal, so its transpose *is* its inverse: $`R_m^\top = R_m^{-1} = R_{-m}`$. Turning back by $`m`$ and then forward by $`n`$ is turning forward by $`n - m`$. **Angles add.** That's it — that's the whole mechanism.

▸ **Now say it without symbols: token $`m`$'s query has been spun forward by $`m`$, token $`n`$'s key by $`n`$. When you compare them, the comparison only cares about the angle *between* them, and that angle is $`n - m`$. The two absolute spins subtract each other out automatically.**

> **Analogy — and it is exactly the book's clock-hands analogy, now earned.** Two people stand on a carousel and point at each other. As the carousel turns, both are rotated by the same amount, and the angle between their arms never changes. Their absolute headings are constantly changing; their relative heading is invariant. RoPE puts every token on a carousel whose rotation is proportional to its position, and attention only ever measures relative headings.

**Tiny worked example.** Take $`d = 2`$ (one pair), $`\theta = 1`$, $`q = k = (1, 0)`$.

- Positions $`m=3`$, $`n=5`$: $`\tilde q = (\cos 3, \sin 3)`$, $`\tilde k = (\cos 5, \sin 5)`$. Dot product $`= \cos 3\cos 5 + \sin 3 \sin 5 = \cos(5-3) = \cos 2 \approx -0.416`$.
- Positions $`m=1003`$, $`n=1005`$: dot product $`= \cos(1005 - 1003) = \cos 2 \approx -0.416`$. **Identical.**

Same gap, same score, a thousand tokens later. That is the promise, and it is exact, not approximate.

**Reading the boxed equation.** $`\langle\mathrm{RoPE}(q,m),\ \mathrm{RoPE}(k,n)\rangle = g(q,k,\,n-m)`$ says: *the score is some function $`g`$ of the query's content, the key's content, and the gap — and $`m`$ and $`n`$ appear nowhere else.* The word $`g`$ is deliberately vague; the content of the statement is entirely in **what is missing** from its argument list.

▸ **Why "compatible with the KV cache" is the sentence that won RoPE the field.** During generation, keys computed for earlier tokens are stored and reused. A T5-style bias depends on $`i - j`$, so when token 5000 arrives you must compute 5000 fresh bias values against the cache. RoPE's key for token 17 was rotated by $`17\theta`$ when it was first computed, and **that rotation is still correct forever** — token 5000's query is rotated by $`5000\theta`$, and the subtraction happens automatically in the dot product. *Nothing in the cache ever needs revisiting.* Position is baked in at write time, at zero read-time cost. For a scheme competing on long context, that is not a minor engineering nicety; it is the whole game.

> **Where this came from.** RoPE was invented by **Jianlin Su**, working with colleagues at the Chinese company Zhuiyi Technology, and its history is unusual. He first wrote it up in **2021 in Chinese, on his personal mathematics blog**, before the English paper *RoFormer: Enhanced Transformer with Rotary Position Embedding* appeared on arXiv. The paper attracted comparatively little attention at first. What carried it into the mainstream was open-source adoption: **EleutherAI** used it in GPT-J and GPT-NeoX, and from there it spread to essentially every major open model — LLaMA, Mistral, Qwen, Gemma, DeepSeek — until it became the default without ever having been the subject of a headline result. It is one of the clearest cases in recent machine learning of an idea propagating through code rather than through citations.

### Efficient implementation

Never build the rotation matrix. Using the "rotate-half" formulation:
```
q_rot = q * cos(mθ) + rotate_half(q) * sin(mθ)
rotate_half(q) = concat(-q[d/2:], q[:d/2])
```
Two elementwise multiplies and an add. $`O(d)`$, not $`O(d^2)`$.

#### Why you never build the matrix

A block-diagonal matrix of $`d/2`$ little rotations is, viewed as a $`d\times d`$ array, **almost entirely zeros** — only $`2d`$ of its $`d^2`$ entries are nonzero. For $`d = 128`$ that is 256 useful numbers hiding inside 16,384. Multiplying by it would cost $`O(d^2)`$ operations, of which over 98% multiply by zero.

The `rotate_half` trick performs the same arithmetic in $`O(d)`$. Read the code line by line:

- `cos(mθ)` and `sin(mθ)` are **vectors of length $`d`$**, precomputed once for every position and reused for every layer, every head, and every example in the batch. They are not recomputed per token.
- `rotate_half(q) = concat(-q[d/2:], q[:d/2])` takes the second half of the vector, negates it, and puts it in front. In NumPy/PyTorch slicing, `q[d/2:]` means "from the midpoint to the end."
- `q * cos + rotate_half(q) * sin` then reproduces $`x\cos\alpha - y\sin\alpha`$ in the first half and $`x\sin\alpha + y\cos\alpha`$ in the second — **exactly the 2×2 rotation, executed on the whole vector at once.**

▸ **Note the pairing this implies.** Rather than pairing dimension 0 with 1, and 2 with 3, this implementation pairs dimension 0 with $`d/2`$, dimension 1 with $`d/2+1`$, and so on. Mathematically it makes no difference — which dimensions you decide to call "a pair" is arbitrary, since the model learns whatever basis it likes — but it matters *enormously* when converting weights between codebases, and mismatched RoPE pairing conventions are a classic source of a model that loads without error and outputs fluent nonsense.

> **Analogy.** Multiplying by the explicit matrix is filling in a 128×128 spreadsheet to compute 128 numbers. `rotate_half` is noticing that the spreadsheet has one useful formula repeated 64 times and writing the formula once.

**Cost, concretely.** $`O(d)`$ versus $`O(d^2)`$ at $`d = 128`$ is a **64× reduction**, and at $`d=128`$ RoPE's cost is then a rounding error next to the attention it feeds — two elementwise multiplies and an add per vector. A positional scheme that changed the asymptotic cost of a forward pass would not have been adopted no matter how elegant.

### Properties

- **Long-range decay:** by a Cauchy–Schwarz-style bound, the expected score attenuates as positions separate — a soft locality prior, without ALiBi's hard monotone penalty.
- **Only applied to $`Q`$ and $`K`$**, never to $`V`$. Values carry content, not position.
- **The base $`\theta`$ (10000) is the key knob for context extension.**

#### The three properties, decoded

**"Long-range decay."** As two tokens separate, the many clocks spin to increasingly unrelated angles, and their contributions to the dot product increasingly cancel by chance rather than reinforcing. So the *typical* score falls off with distance. **Cauchy–Schwarz** is the inequality $`\lvert a^\top b\rvert \le \|a\|\|b\|`$ — "a dot product can never exceed the product of the lengths" — which is what lets you bound the sum of many oscillating terms.

▸ **The crucial word is "expected."** ALiBi *subtracts* a penalty, so distance is a hard rule the model cannot override. RoPE merely makes far-apart scores *tend* to be small on average — and a strong enough content match can still produce a large score across 50,000 tokens. **That is a prior the model can argue with, versus a constraint it cannot.** It is the main reason RoPE handles needle-in-a-haystack retrieval better than ALiBi.

**"Only applied to $`Q`$ and $`K`$, never to $`V`$."** Remember what each does (Ch. 11): the query and key produce the *score* — who should look at whom — while the value is the *content* that gets carried forward once the looking is settled. Position belongs in the first job and not the second. Rotating $`V`$ would mean a token's actual content changed depending on where it sat in the sentence, which is both wrong and destructive: the residual stream would receive spun-around versions of its own features.

> **Analogy.** At a conference, position tells you *who to talk to* — you find the people near you, or the people in your session. It does not change *what they know*. RoPE encodes seating; values encode expertise.

**"The base 10000 is the key knob."** Since $`\theta_i = \text{base}^{-2i/d}`$, raising the base makes the clocks slower — and it does so **unevenly**. Pair $`i`$'s wavelength is $`2\pi\cdot\text{base}^{2i/d}`$, so multiplying the base by 50 (from 10,000 to LLaMA-3's 500,000) leaves pair $`0`$ untouched, slows the middle pairs by $`\sqrt{50}\approx 7\times`$, and slows the slowest pair by the full $`50\times`$. Local resolution survives; global range multiplies. **That one constant is a large part of how LLaMA-3 reaches 128k context**, and it costs nothing at run time — it is a different number in a table of precomputed sines.

▸ **So the design has exactly one dial, and turning it trades local resolution for global range.** Everything in the next section is a more careful way of turning that dial — including turning it by different amounts for different dimensions.

### Context extension

The problem: a model trained at $`T=4096`$ has never seen rotation angles beyond $`4096\theta_i`$, and behaves badly past it.

| Method | Mechanism | Notes |
|---|---|---|
| **Position Interpolation (PI)** | scale positions: $`m\to m\cdot\frac{T_{\text{train}}}{T_{\text{new}}}`$ | all angles stay in-distribution, but *crowds* nearby positions and blurs local resolution. Needs a little fine-tuning. |
| **NTK-aware scaling** | increase the base: $`\theta_{\text{base}}: 10000\to10000\cdot s^{d/(d-2)}`$ | stretches low-frequency dims more, leaves high-frequency (local) dims alone. Often works with **no** fine-tuning. |
| **YaRN** | per-dimension: interpolate low-frequency dims, extrapolate high-frequency ones, plus a temperature correction on attention | current best; 10× extension with ~0.1% of original training tokens |
| **Train with a large base** | e.g. $`\theta_{\text{base}}=500{,}000`$ from scratch | LLaMA-3's approach; simplest if you control pretraining |

▸ **The unifying insight (the "NTK" argument):** RoPE dimensions form a spectrum of frequencies. High-frequency dimensions complete many full rotations within the training length, so they are well-trained and *can* extrapolate; low-frequency dimensions complete less than one rotation and have never seen large angles, so they *cannot*. The right fix therefore treats dimensions differently by frequency — which is exactly what YaRN does and what naive PI does not.

#### The context-extension table, decoded

First, restate the problem precisely, because it is easy to get backwards. A model trained at $`T = 4096`$ **does not crash** at position 8000 — RoPE will happily compute $`\cos(8000\theta_i)`$. It degrades because the *weights that consume those angles* were fitted on a range of angles that never included this one. **The formula generalizes; the learned function of it does not.** (Same distinction as in §12.2. It is the reusable one.)

Now each row, in plain terms. Let $`s = T_{\text{new}}/T_{\text{train}}`$ be the **stretch factor** — $`s = 4`$ means you want 4× the context.

**Position Interpolation (PI).** *Lie about the position.* Tell the model that token 8000 is at position 2000. Every angle is divided by $`s`$, so every angle lands back inside the trained range. It works, immediately, and it is one line of code.

> **Analogy.** Fitting a 30 cm ruler to a 120 cm plank by relabelling every centimetre as four. Every measurement you make is now in range. But your finest gradation is 4 cm wide — you have lost the ability to distinguish things less than 4 cm apart. That is exactly "crowds nearby positions and blurs local resolution": adjacent tokens, which used to sit $`\theta_0`$ apart on the fastest clock, now sit $`\theta_0/4`$ apart, and the model must resolve four times finer than it was ever trained to.

**NTK-aware scaling.** *Change the gearing instead of the ruler.* Rather than dividing all angles by $`s`$, raise the base so that **slow dimensions stretch a lot and fast dimensions barely change.** Fast dimensions did not need help — they had already completed hundreds of turns during training and are thoroughly exercised across their whole circle. Only the slow ones were starved. So fix only what is broken.

▸ **This is the key asymmetry, and once you see it the rest of the section is obvious.** The fast clocks are *well-trained and can extrapolate*; the slow clocks are *undertrained and cannot*. PI applies the same medicine to both and therefore damages the healthy one. That is why NTK-aware scaling frequently works with **no fine-tuning at all** while PI needs some — PI breaks something that was working, and the model has to re-learn it.

**YaRN.** *Do the sorting explicitly.* Classify each dimension by how many full turns it completed during training: many turns → **extrapolate** (leave it alone), less than one turn → **interpolate** (compress it, PI-style), in between → blend smoothly. Then add a temperature correction to the attention softmax, because spreading tokens over a longer context changes how peaked the attention distribution ends up being, and un-correcting for that costs accuracy. The claimed figure — 10× extension for ~0.1% of the original training tokens — is what makes it practical: a few hundred million tokens of fine-tuning rather than a re-run of pretraining.

**Train with a large base.** *Don't have the problem.* If you control pretraining, set the base to 500,000 on day one and every dimension is trained over the angle range you will actually use. Nothing to repair later. Strictly the best option and available to almost nobody, which is why the other three rows exist.

| Method | The one-line summary | What it costs |
|---|---|---|
| PI | Squash all positions to fit | Local resolution; needs fine-tuning |
| NTK-aware | Squash only the slow dimensions | Almost nothing; often no fine-tuning |
| YaRN | Squash, extrapolate, or blend, per dimension, plus a softmax temperature fix | A little fine-tuning; more moving parts |
| Large base from scratch | Never create the mismatch | You must own the pretraining run |

> **Where this came from — and it is one of the odder stories in modern machine learning.** **Position Interpolation** was published by a team at Meta (Chen and colleagues) in 2023. **NTK-aware scaling did not come from a paper at all**: it was posted to the r/LocalLLaMA subreddit in mid-2023 by a user going by *bloc97*, as a suggestion for extending LLaMA's context without fine-tuning. It was implemented in open-source inference stacks within days, well before any formal write-up existed. The **YaRN** paper (Bowen Peng, Jeffrey Quesnelle, Honglu Fan and Enrico Shippole, 2023) then consolidated the family of tricks into a single method with proper evaluation — with bloc97 among the authors. The name is a self-aware joke: *Yet another RoPE extensioN*. The borrowed label "NTK" refers to the Neural Tangent Kernel literature, invoked by loose analogy about high-frequency versus low-frequency learning; the connection is heuristic rather than derived, and the name has stuck largely because it was the one used in the original post.

#### Examples and non-examples:  context extension

**✅  examples**

| Example | Why it qualifies |
|---|---|
| Position Interpolation with a short fine-tune | Every angle is brought back inside the range the weights were fitted on, and the fine-tune repairs the lost local resolution |
| NTK-aware base scaling, $`10000 \to 10000\cdot s^{d/(d-2)}`$ | Fixes the undertrained slow dimensions and leaves the well-trained fast ones alone |
| YaRN | Sorts dimensions by how many turns they completed in training and applies the right treatment to each, plus a softmax temperature fix |
| LLaMA-3 pretraining with base $`500{,}000`$ | Never creates the angle mismatch in the first place |
| Continued pretraining on  long documents | Both the angles and the *behaviour* over long spans are now in-distribution |

**❌ Near-misses — look like context extension, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Editing `max_position_embeddings` to 128000 in the config | Changes a bound check. The rotation angles past $`T_{\text{train}}`$ are still ones the weights have never been fitted for | A configuration change. The model will accept the tokens and produce degraded output |
| Raising `--max-model-len` / allocating a bigger KV cache | Buys the *room* to hold 128k tokens; says nothing about whether the attention scores over them are meaningful | A memory allocation |
| Chunking a document into 4k pieces and processing each separately | No token in chunk 7 ever attends to a token in chunk 2. The cross-chunk dependencies are gone by construction | Sliding-window inference |
| Retrieval-augmented generation over a 500-page PDF | Reduces the input to a few thousand relevant tokens *before* the model sees it | Retrieval (Ch. 18) — a complement to long context, not a form of it |
| A 200k-token prompt that produces fluent, confident output | Fluency comes from the last few hundred tokens. It is entirely compatible with the middle 190k having been ignored | Local coherence |

▸ **The boundary:** context extension means the attention machinery at position 100,000 is operating in a regime the weights were actually fitted for. Anything that only changes what the runtime will *accept* — a config field, a buffer size, a chunking loop — moves the limit without moving the capability.

> **Common misconception.** *"RoPE is a learned positional embedding — it's in the name."* RoPE has **zero learned parameters.** The angles come from a closed-form table, $`\theta_i = \text{base}^{-2i/d}`$, computed once and cached; there is no gradient flowing into them and nothing to initialize. What *is* learned is $`W_Q`$ and $`W_K`$ and everything downstream, which is why changing the base is disruptive even though the base itself was never trained — you have changed the inputs those trained weights consume. The misconception is very tempting, because the word "embedding" almost everywhere else in deep learning means *a learned lookup table indexed by an integer*, and because RoPE's behaviour clearly does change when you retrain. Compare BERT's $`PE\in\mathbb{R}^{512\times768}`$, which  is a learned table, and the contrast is sharp: BERT has no row 512, while RoPE will happily compute an angle for position 10 million.

> **Common misconception.** *"We extended the model to 128k context, so now it can reason over 128k tokens."* Extension fixes exactly one thing: the mismatch between the angles the model sees and the angles its weights were fitted for. It does nothing whatsoever about the model's ability to *integrate* information spread across 128,000 tokens — that is a capability question, and §12.8 shows it lags the window size badly. A model can have perfectly in-distribution attention scores at position 120,000 and still be unable to answer a question that requires combining something at position 5,000 with something at position 95,000. The belief is tempting because the metrics used to validate an extension — perplexity on long documents, needle-in-a-haystack recall — are both dominated by *local* prediction and *lookup*, the two things that were never the hard part. **You measured the thing that was easy to measure and concluded something about the thing that wasn't.**

---

## 12.5 Efficient attention

### The cost

$`O(T^2)`$ time and, naively, $`O(T^2)`$ memory per head. At $`T=100`$k with $`h=32`$: the attention matrix alone is $`32\times10^{10}`$ entries = **640 GB in bf16 per layer**. Impossible without restructuring.

#### Where 640 GB comes from

Do the arithmetic yourself once; the number stops being abstract.

Attention builds a score for **every pair of tokens**. With $`T = 100{,}000`$ tokens that is $`T^2 = 10^{10}`$ scores — ten billion — **per head**. With $`h = 32`$ heads: $`32\times 10^{10}`$ numbers. In **bf16** (brain floating point, 16-bit) each number takes 2 bytes:

$$32 \times 10^{10} \times 2\ \text{bytes} = 6.4\times10^{11}\ \text{bytes} = 640\ \text{GB}$$

**Per layer.** A model with 80 layers would need 51 terabytes if it kept them all. The largest GPUs available carry 80–192 GB. **This is not a tight fit; it is off by four orders of magnitude.**

▸ **Feel the quadratic.** Go from 1k to 100k tokens — 100× longer — and the attention matrix grows **10,000×**. This is the $`\mathcal{O}(T^2)`$ from §0.10 doing its work: doubling the context does not double the cost, it quadruples it. Every technique in the rest of this chapter is a response to that one exponent.

> **Analogy.** A party where everyone must shake hands with everyone. Ten guests: 45 handshakes. A hundred guests: 4,950. A thousand: half a million. The room did not get ten times harder to manage, it got a hundred times harder. Attention is a room where every token shakes hands with every token, and someone naively decided to write down every handshake on a separate card before counting them.

**But notice what the number is measuring: storage, not arithmetic.** The FLOPs are large but a modern GPU eats them without complaint. The catastrophe is that a naive implementation *writes the cards down*. FlashAttention's insight is that **you never actually need the full matrix at once** — you need one row's worth of weighted average at a time. The next section is how to get that without ever materializing the sheet.

### FlashAttention — the one that actually mattered

▸ **The key realization: attention is memory-bound, not compute-bound.** GPU HBM bandwidth (~2–3 TB/s) is ~10× slower than SRAM. Standard attention writes the $`T\times T`$ matrix to HBM and reads it back — three round trips over an enormous array.

**The solution: tiling + online softmax + recomputation.** Never materialize the full matrix. Process $`K,V`$ in blocks, keeping a running output and running softmax statistics in SRAM.

**Online softmax** (the mathematical core). To compute a softmax-weighted sum in one pass, maintain running max $`m`$, running sum $`\ell`$, and running output $`O`$. On seeing a new block with max $`m^{\text{new}}`$:

▸ $$m' = \max(m, m^{\text{new}}),\qquad \ell' = e^{m-m'}\ell + e^{m^{\text{new}}-m'}\ell^{\text{new}},\qquad O' = \frac{e^{m-m'}\ell\,O + e^{m^{\text{new}}-m'}\ell^{\text{new}}O^{\text{new}}}{\ell'}$$

The rescaling factors correct the previously-accumulated result for the new global maximum. Numerically stable and exactly equal to standard softmax.

#### Reading online softmax in plain English

This is the mathematical core of FlashAttention and it looks worse than it is. Start with **why softmax needs a maximum at all.**

Softmax is $`p_i = e^{z_i}/\sum_j e^{z_j}`$. If some $`z_i = 800`$, then $`e^{800}`$ overflows to infinity in any float format and the whole computation becomes `NaN`. The standard fix — used in every library, everywhere — is to subtract the largest logit first:

$$p_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j$$

This changes nothing mathematically (the $`e^{-m}`$ cancels top and bottom) and everything numerically: the largest exponent is now $`e^0 = 1`$, so nothing can overflow.

**The problem this creates.** To subtract the maximum you must first *know* the maximum, which means seeing all $`T`$ scores — which means materializing the row you were trying to avoid materializing. Online softmax breaks that dependency.

Now the symbols:

| Symbol | Read aloud | What it holds |
|---|---|---|
| $`m`$ | "m" | The largest score **seen so far** (⚠ not position — $`m`$ is overloaded across this chapter) |
| $`\ell`$ | "ell" | The running **sum** of $`e^{\text{score} - m}`$, i.e. the softmax denominator so far |
| $`O`$ | "O" | The running weighted **output** so far |
| $`m^{\text{new}},\ \ell^{\text{new}},\ O^{\text{new}}`$ | — | The same three quantities computed for the **new block** alone |
| $`m'`$ | "m prime" | The updated maximum after merging |

**The whole algorithm in one sentence:** *keep a running answer computed against the best guess of the maximum so far; when a bigger maximum turns up, multiply the old answer by a correction factor and merge.*

The correction factor is $`e^{m - m'}`$. If the old running max was 3 and the new global max is 7, every previously accumulated term was divided by $`e^3`$ when it should have been divided by $`e^7`$, so it is too large by $`e^{4}`$ — multiply by $`e^{3-7} = e^{-4}`$ and it is exactly right again. **No approximation. No drift. Just rescaling.**

> **Analogy.** You are averaging exam marks that arrive in batches, expressing everything as a percentage of the highest mark so far. Ten scripts arrive, top mark 60, and you record everything relative to 60. Then a script scores 90. You do not go back and re-read the first ten — you multiply your running figures by $`60/90`$ and carry on. FlashAttention does this, with exponentials in place of ratios.

**Numbers, with two blocks.** Scores $`\{1, 2\}`$ then $`\{5\}`$, with values $`v = 10, 20, 30`$.

- Block 1: $`m = 2`$, $`\ell = e^{-1} + e^{0} = 1.368`$, $`O = (e^{-1}\cdot 10 + 1\cdot 20)/1.368 = 17.31`$.
- Block 2 arrives: $`m^{\text{new}} = 5`$, so $`m' = 5`$. Correction $`e^{2-5} = e^{-3} = 0.0498`$.
- $`\ell' = 0.0498(1.368) + 1(1) = 1.068`$.
- $`O' = \big(0.0498 \cdot 1.368 \cdot 17.31 + 1\cdot 1\cdot 30\big)/1.068 = (1.179 + 30)/1.068 = 29.19`$.

Check against computing it in one shot: weights $`\propto e^{-4}, e^{-3}, e^{0} = 0.0183, 0.0498, 1`$, sum $`1.068`$; output $`= (0.183 + 0.996 + 30)/1.068 = 29.19`$. ✓ **Bit-for-bit the same answer, computed without ever holding all three scores at once.**

▸ **This is why the book can claim the output is "numerically exact, not an approximation."** Every other row in the approximate-methods table below buys memory by giving up accuracy. FlashAttention buys memory by giving up *nothing* — it is a reorganization of the same arithmetic. That distinction is the reason it superseded a whole research literature rather than joining it.

**Backward pass:** recompute the attention block from $`Q,K,V`$ instead of storing it — extra FLOPs, far fewer memory accesses, net large win.

#### Why recomputing is faster than remembering

This inverts the instinct every programmer has, so it deserves a number.

An **A100 GPU** can perform roughly $`3\times10^{14}`$ floating-point operations per second, but its HBM (High-Bandwidth Memory) delivers only about $`2\times10^{12}`$ bytes per second. Divide: the chip can do on the order of **a hundred arithmetic operations in the time it takes to fetch one number from memory.**

▸ **So arithmetic is nearly free and memory traffic is the entire bill.** Recomputing an attention block during the backward pass costs some FLOPs from a budget that was never the constraint, and saves a round trip to HBM from a budget that was. Trading the abundant resource for the scarce one is not a clever hack; it is the only sensible move, and the reason it feels counterintuitive is that the ratio was much smaller when most programming intuitions were formed.

> **Analogy.** A chef deciding whether to walk to the cold store for a pre-chopped onion or chop a fresh one at the bench. If the cold store is a hundred paces away, you chop. The chopping was never the slow part.

**The vocabulary, since the book uses it without defining it.** **SRAM** is the small, very fast memory physically on the GPU die — tens of kilobytes to a few megabytes per streaming multiprocessor. **HBM** is the multi-gigabyte memory stacked next to the chip; roomy, and ~10× slower to reach. FlashAttention's tiling exists to keep the working set inside SRAM for as long as possible: load a block of $`K`$ and $`V`$, do all the work you can with it, and only then go back.

▸ **"Count memory movements, not FLOPs"** is the transferable lesson, and it generalizes well past attention. It is why fused kernels, operator fusion, and quantization (Ch. 17) deliver speedups that a FLOP count cannot explain.

▸ **Result:** memory drops from $`O(T^2)`$ to $`O(T)`$; speed improves 2–4×; the output is **numerically exact**, not an approximation. FlashAttention-2 improved work partitioning (~2× more); FlashAttention-3 exploits Hopper asynchrony and FP8.

**This is the single most important systems result in modern deep learning**, and it is a great example of the general lesson: *count memory movements, not FLOPs*.

> **Where this came from.** **FlashAttention** was published in 2022 by **Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra and Christopher Ré** at Stanford, with the subtitle *"Fast and Memory-Efficient Exact Attention with IO-Awareness."* The word **exact** in the title is doing deliberate work: by 2022 the literature was full of approximate attention mechanisms (the table below is a partial list), and the paper's argument was that the field had been optimizing the wrong quantity. Everyone was counting FLOPs; the bottleneck was memory traffic.
>
> Neither ingredient was new on its own. The **online softmax** recurrence was published by **Maxim Milakov and Natalia Gimelshein at NVIDIA in 2018**, purely as a way to compute a softmax in fewer passes over memory — no connection to attention. And **Markus Rabe and Charles Staats at Google** showed in 2021, in a paper titled *"Self-attention Does Not Need $`O(n^2)`$ Memory,"* that attention could be computed in linear memory by chunking. What FlashAttention added was the part that made it matter: a **CUDA implementation that actually kept the working set in SRAM**, so the theoretical saving became a real wall-clock speedup rather than a slower algorithm with a nicer memory bound. It is a good illustration of how much of deep learning progress is implementation rather than idea.

> **The story behind the tiling.** Blocking a computation so that its working set fits in fast memory is not a deep-learning invention — it is a technique from numerical linear algebra dating to the 1980s, and it is why the BLAS and LAPACK libraries are structured the way they are. The lesson had to be relearned because the transformer literature grew up in a Python/PyTorch idiom where memory hierarchy is invisible by design. The attention operation was written the way the equation is written, and nobody looked at the memory traffic for five years.

#### Examples and non-examples: exact attention

**✅  exact — same output as the textbook implementation**

| Example | Why it qualifies |
|---|---|
| FlashAttention 1, 2 and 3 | Every query–key pair is still scored; only the *order of memory accesses* changed |
| The online-softmax recurrence itself | The rescaling factors $`e^{m-m'}`$ are algebraically exact — they undo the old maximum and apply the new one |
| Rabe and Staats' chunked attention (2021) | Linear memory, identical arithmetic, no pairs dropped |
| Ring attention across devices | Blocks of $`K,V`$ are passed around a ring; the sum they contribute to is the same sum |
| Recomputing attention blocks in the backward pass | Recomputation reproduces a value rather than approximating it |

**❌ Near-misses — fast attention, but not the same attention**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Longformer / Mistral sliding-window attention | Pairs beyond the window are never scored at all. Their weight is not small — it is absent | An approximation via a sparsity pattern |
| Linformer | Projects the $`T`$ keys down to $`k`$ before scoring, so no individual key is scored | A low-rank approximation |
| Performer | Replaces the softmax kernel with a random-feature estimate; the answer is right *in expectation* | A stochastic kernel approximation |
| Sparse Transformer's strided pattern | A fixed subset of pairs is computed and the rest assumed to be zero | An approximation via a hand-designed pattern |
| KV eviction at inference (H2O, StreamingLLM) | Tokens are thrown out of the cache, so later queries cannot attend to them however much they would like to | Lossy cache compression |
| FlashAttention-3 in FP8 | The *algorithm* is exact — every pair is scored. The *arithmetic* is coarser than bf16, so the numbers differ | An exact algorithm run in reduced precision. A  different axis |

▸ **The boundary:** exact means *for the same inputs at the same precision, the output equals what the naive three-line implementation would produce* (up to floating-point reassociation, which reorders sums but drops nothing). Restructuring **when** you touch memory is exact. Changing **which pairs get scored** is not.

> **Common misconception.** *"FlashAttention is an approximation — that's how it gets the speedup."* It is not. It computes the identical softmax attention, to the last decimal that floating-point associativity permits; the subtitle of the paper is *"Fast and Memory-Efficient **Exact** Attention with IO-Awareness,"* and the emphasis is the authors' own. What it changes is the order of memory traffic: never write the $`T\times T`$ matrix to HBM, never read it back. Three things make the wrong belief almost irresistible. It is filed alongside Longformer and Performer in every survey; in 2019–2021, "faster attention" **always** meant approximate, so the prior is well-earned; and swapping the kernel in does change your loss curve at the seventh decimal place, because summing the same numbers in a different order gives a slightly different float — which looks exactly like the fingerprint of an approximation and is not.

> **A related trap.** *"FlashAttention removes the $`O(T^2)`$ cost of attention."* It removes the $`O(T^2)`$ **memory**, taking it to $`O(T)`$. The arithmetic is still quadratic: ten billion scores at $`T=100`$k is ten billion scores whether or not you write them down. This matters practically — going from 32k to 128k context still costs 16× the attention FLOPs, and no amount of tiling changes that. The confusion is understandable because the thing that was *stopping* people was the memory (640 GB per layer against an 80 GB card), so removing the memory wall felt like removing the whole wall. The quadratic was always there; it just stopped being the binding constraint.

### Approximate methods (largely superseded, still asked about)

| Method | Idea | Complexity |
|---|---|---|
| **Sparse Transformer** | strided + local patterns | $`O(T\sqrt T)`$ |
| **Longformer** | sliding window + global tokens | $`O(Tw)`$ |
| **BigBird** | window + global + random; proven Turing-complete | $`O(T)`$ |
| **Linformer** | project $`K,V`$ to length $`k`$ | $`O(Tk)`$ |
| **Performer** | random-feature kernel approximation of softmax | $`O(T)`$ |
| **Reformer** | LSH bucketing of similar queries | $`O(T\log T)`$ |

▸ **Why they mostly lost:** FlashAttention made exact attention fast enough that approximation's quality cost stopped being worth it. **Sliding-window attention survives** (Mistral, Gemma-2) because it composes: a window of $`w`$ over $`L`$ layers gives an effective receptive field of $`L\cdot w`$, and interleaving local with occasional global layers gets most of the benefit at a fraction of the cost.

#### The approximate-methods table, decoded

Every row is answering the same question — *which of the $`T^2`$ pairs can I skip?* — and each gives a different answer.

| Method | What it actually does | The bet it makes |
|---|---|---|
| **Sparse Transformer** | Each token attends to its neighbours plus every $`k`$-th token | Two hops of a sparse pattern reach everything |
| **Longformer** | A sliding window, plus a few designated tokens everyone can see | Most attention is local; a few hubs handle the rest |
| **BigBird** | Window + global tokens + a handful of *random* pairs | Random edges make a graph small-world, so any pair is a short path apart |
| **Linformer** | Squash the $`T`$ keys and values down to $`k`$ of them first | The attention matrix is approximately low-rank |
| **Performer** | Rewrite $`\exp(q^\top k)`$ as $`\phi(q)^\top\phi(k)`$ for a random feature map | Softmax is a kernel and kernels can be approximated by random features |
| **Reformer** | Hash queries and keys into buckets; only compare within a bucket | Only nearby-in-angle pairs get large scores anyway |

Decoding the complexity column:

- $`\mathcal{O}(T\sqrt T)`$ at $`T=100`$k is about $`3.2\times10^7`$ against $`10^{10}`$ — a **300× reduction on paper.**
- $`\mathcal{O}(Tw)`$ with window $`w=512`$ is $`5\times10^7`$ — a **200× reduction**, and it grows *linearly* with length rather than quadratically.
- $`\mathcal{O}(T)`$, the holy grail: doubling the context doubles the cost, full stop.

▸ **So why did essentially all of them lose?** Because the paper reductions were in **FLOPs**, and FLOPs were never the constraint (see the A100 numbers above). A method that cuts operations 300× but scatters its memory accesses can be *slower in wall-clock time* than dense attention running in a well-fused kernel. Meanwhile every one of them pays some quality tax. When FlashAttention made exact attention both fast and memory-light, the trade every row of this table offers — *lose a little quality, gain some speed* — simply stopped being a trade anyone wanted.

> **Analogy.** These are elaborate route-planning algorithms for avoiding traffic, published in a year when someone widened the motorway. The algorithms were not wrong. The problem they solved stopped being the binding one.

**Why sliding-window survived, in numbers.** With $`w = 4096`$ and $`L = 32`$ layers, the effective receptive field is $`L\cdot w \approx 131{,}000`$ tokens — because layer 1's output at position $`t`$ already contains information from $`t-4096`$, so layer 2 reading position $`t-4096`$ reaches back to $`t-8192`$, and so on. **Information propagates the way a rumour crosses a crowd: nobody talks to anyone far away, and the news still reaches the far side.** That is a  architectural saving, not a quality-for-speed trade, and it composes cleanly with FlashAttention rather than fighting it — which is exactly why Mistral and Gemma-2 use it and nobody uses Reformer.

> **Where this came from.** This whole family arrived in a burst between 2019 and 2021 — Sparse Transformer (OpenAI, 2019), Reformer, Longformer, Linformer, BigBird and Performer all within roughly eighteen months — and the period was known informally as the **"efficient transformer" or "x-former" era**. Google researchers eventually published a survey and a benchmark suite (*Long Range Arena*, 2020) specifically because there were too many methods to compare informally. BigBird's authors proved their sparse attention was Turing-complete and a universal approximator of sequence functions — a  strong theoretical result which nonetheless did not save the method, because the constant factors and the implementation complexity were what mattered in practice. It is a useful cautionary tale about what a proof does and does not buy you.

### Linear attention

Replace $`\mathrm{softmax}(QK^\top)V`$ with $`\phi(Q)(\phi(K)^\top V)`$ for a feature map $`\phi`$. By associativity, compute $`\phi(K)^\top V`$ first ($`d\times d`$), giving **$`O(Td^2)`$ instead of $`O(T^2d)`$.**

▸ In the causal case this is exactly a **linear RNN** with state $`S_t = S_{t-1} + \phi(k_t)v_t^\top`$, so inference is $`O(1)`$ memory per token — no KV cache at all. The cost is quality: linear attention cannot do sharp, selective retrieval, because a fixed $`d\times d`$ state must summarize everything.

#### Unpacking linear attention

The trick here is **associativity of matrix multiplication** — the fact that $`(AB)C = A(BC)`$ — and it is worth appreciating how much can hang on something that trivial.

Softmax is what blocks it. In $`\mathrm{softmax}(QK^\top)V`$ the softmax sits *between* $`QK^\top`$ and $`V`$, so you are forced to compute the $`T\times T`$ matrix first. **Remove the softmax and the parentheses are free to move.** Compare the two groupings:

| Grouping | Shapes multiplied | Cost |
|---|---|---|
| $`\big(\phi(Q)\phi(K)^\top\big)V`$ | $`(T{\times}d)(d{\times}T)`$ then $`(T{\times}T)(T{\times}d)`$ | $`\mathcal{O}(T^2 d)`$ |
| $`\phi(Q)\big(\phi(K)^\top V\big)`$ | $`(d{\times}T)(T{\times}d)`$ then $`(T{\times}d)(d{\times}d)`$ | $`\mathcal{O}(Td^2)`$ |

**Put numbers in.** $`T = 100{,}000`$, $`d = 128`$:

- $`T^2d = 10^{10}\times128 = 1.28\times10^{12}`$
- $`Td^2 = 10^5\times16384 = 1.6\times10^{9}`$

**An 800× reduction, from moving a bracket.** The intermediate object changes from a $`100{,}000\times100{,}000`$ matrix to a $`128\times128`$ one.

> **Analogy.** You want the total value of a shopping basket where each of 100,000 items has a price and a quantity. You can build the full 100,000 × 100,000 table of every item-pair, or you can total the quantities first and then multiply. Same answer; one of them fits on a napkin.

▸ **The crossover point is where $`T > d`$**, which for any real sequence is essentially always — so linear attention is asymptotically better from a few hundred tokens onward. Which raises the obvious question: why doesn't everyone use it?

**Reading the recurrence $`S_t = S_{t-1} + \phi(k_t)v_t^\top`$.** Here $`\phi(k_t)v_t^\top`$ is an **outer product** (§0.8) — a $`d\times d`$ matrix built from two vectors. Each new token adds its own outer product onto a running $`d\times d`$ accumulator. Then $`\phi(q_t)^\top S_t`$ reads out of it.

> **Analogy.** A single whiteboard that everyone writes on, in the same handwriting, without erasing. Any individual message is still technically present in the ink, but after ten thousand people have written over each other, extracting one specific sentence is hopeless. A transformer's KV cache is a filing cabinet: every message on its own sheet, indexed, retrievable exactly.

▸ **And that is the whole trade, stated honestly.** $`128\times128 = 16{,}384`$ numbers must summarize *everything* the model has read, whether that is 1,000 tokens or 1,000,000. The state size does not grow, so at some length information must be being destroyed — the only question is which information. **Attention's $`\mathcal{O}(T)`$ KV cache is not a design failure; it is what buys the ability to recall an exact string from 80,000 tokens back.** You cannot have both constant memory and perfect recall, and the rest of this chapter is the field negotiating where on that line to sit.

---

## 12.6 State-space models and the recurrence revival

### The idea

A continuous linear system $`h'(t)=Ah(t)+Bx(t)`$, $`y=Ch(t)`$, discretized to $`h_t = \bar Ah_{t-1}+\bar Bx_t`$, $`y_t = Ch_t`$.

**The two-mode trick:** because the recurrence is linear,
- **Training:** unroll into a convolution $`y = \bar K * x`$ with kernel $`\bar K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B,\dots)`$, computable in $`O(T\log T)`$ by FFT — fully parallel.
- **Inference:** run the recurrence, $`O(1)`$ state per token.

▸ **This solves the exact problem that killed RNNs in Ch. 9** (no training parallelism) while keeping their exact advantage (constant-size state).

#### Reading the state-space equations

Start with the continuous version, $`h'(t) = Ah(t) + Bx(t)`$, $`y = Ch(t)`$, and read it as three plain sentences.

| Symbol | Read aloud | Job |
|---|---|---|
| $`h(t)`$ | "h of t" | The **state** — everything the system currently remembers |
| $`h'(t)`$ | "h prime of t" | How fast the state is changing right now |
| $`x(t)`$ | "x of t" | The **input** arriving at time $`t`$ |
| $`y(t)`$ | "y of t" | The **output** read off the state |
| $`A`$ | "A" | How the state evolves **on its own**, with no input — the decay/mixing dynamics |
| $`B`$ | "B" | How strongly new input is **written into** the state |
| $`C`$ | "C" | How the state is **read out** |
| $`\bar A,\ \bar B`$ | "A-bar, B-bar" | The **discretized** versions — what $`A`$ and $`B`$ become when you step in whole time-steps rather than infinitesimally |

▸ **In one sentence: the state leaks and mixes according to $`A`$, new input pours in through $`B`$, and you read the state through $`C`$.**

> **Analogy — and this is literally where the equations come from.** A set of connected water tanks. $`h`$ is the water level in each tank. $`A`$ says how water flows between tanks and drains away. $`B`$ says which tanks the hose fills. $`C`$ says which gauge you read. This is not a metaphor invented for the textbook; these are the standard equations of **linear control theory**, used for aircraft autopilots and chemical plants since the 1960s, imported wholesale into deep learning.

**Discretization, plainly.** A neural network sees tokens at $`t = 1, 2, 3, \dots`$, not a continuous flow. Discretizing means asking: *given the water was at level $`h_{t-1}`$ and I let one time-step of duration $`\Delta`$ elapse, where is it now?* The answer for a linear system involves a matrix exponential, and the result is folded into the new matrices $`\bar A`$ and $`\bar B`$. **The bar means "converted to time-steps."** ⚠ This is a different use of the bar from Chapter 0's gradient notation — one of the  collisions in this book's symbol set.

#### Why the two-mode trick works, and why it is such a big deal

The recurrence $`h_t = \bar Ah_{t-1} + \bar Bx_t`$ looks fatally sequential — you cannot compute $`h_5`$ without $`h_4`$. But because there is **no nonlinearity anywhere**, you can unroll it and the mess telescopes:

$$h_1 = \bar Bx_1,\qquad h_2 = \bar A\bar Bx_1 + \bar Bx_2,\qquad h_3 = \bar A^2\bar Bx_1 + \bar A\bar Bx_2 + \bar Bx_3$$

Read out through $`C`$ and the pattern is clear: $`y_t`$ is a weighted sum of *all* past inputs, with weight $`C\bar A^{k}\bar B`$ on the input from $`k`$ steps ago. **That is a convolution** — the kernel $`\bar K = (C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\dots)`$ is the same for every $`t`$, so you can build it once and slide it along.

And a convolution of length $`T`$ can be done by **FFT** (Fast Fourier Transform) in $`\mathcal{O}(T\log T)`$ instead of $`\mathcal{O}(T^2)`$: at $`T=100{,}000`$ that is $`1.7\times10^6`$ against $`10^{10}`$, a factor of about 6,000.

▸ **Sit with what this buys.** An RNN's fatal flaw (Ch. 9) was that training could not be parallelized: 100,000 sequential steps means 100,000 round trips, and a GPU sits idle through almost all of them. The two-mode trick means you **train as a convolution** (fully parallel, all positions at once) and then **run as a recurrence** (constant state, one token at a time). *The same weights, two completely different execution strategies, chosen per phase.* You get the transformer's training parallelism and the RNN's inference economy.

> **Analogy.** A recipe that can be read as a sequence of steps by a single cook, or handed to twenty cooks working simultaneously on different parts, depending on which is convenient today. Very few computations admit both readings; linearity is what makes it possible, and it is exactly what the nonlinearity inside an LSTM cell destroys.

**And here is the price, stated plainly:** the trick requires the recurrence to be **linear and time-invariant** — the same $`\bar A`$ at every step, no matter what the input says. A model that cannot change its behaviour based on content is a weak model. That tension is what Mamba resolves.

**S4** makes long-range memory work by initializing $`A`$ with **HiPPO** structure — a matrix derived from optimal online polynomial approximation of the input history, so the state provably compresses the past in a principled basis.

#### What HiPPO is doing

**HiPPO** stands for **High-order Polynomial Projection Operators**, and the idea underneath the name is  elegant.

The question it answers: *if you have a fixed-size state and an ever-growing history, what is the mathematically best thing to store?* HiPPO's answer: **store the coefficients of the best polynomial approximation to the history-so-far.** Then it derives the matrix $`A`$ that keeps those coefficients correct as new data arrives, automatically.

> **Analogy.** Sketching a curve that keeps getting longer, on a fixed-size pad. You could record the last hundred points exactly and forget the rest. Or you could keep a smooth curve fitted through *everything* — a few coefficients that capture the overall shape, refined as new points arrive. HiPPO does the second, and proves the update rule that keeps the fit optimal at every instant.

▸ **Why the initialization matters so much.** A randomly initialized linear recurrence forgets exponentially: $`\bar A^k`$ shrinks to nothing within a few dozen steps (the same $`\lambda^k`$ argument as §1.1.2). It is not that the architecture *cannot* remember 10,000 steps back — it is that random weights place it in a region where it doesn't, and gradient descent will not find its way out. HiPPO starts it in the right region. **This is one of the clearest examples in deep learning of an initialization scheme being the entire result**, rather than a detail in the appendix.

**Mamba (S6)** makes $`B`$, $`C`$, and the discretization step $`\Delta`$ **input-dependent** — "selective" — so the model can choose to remember or forget based on content. This breaks linear-time-invariance and rules out the FFT trick, so Mamba uses a **hardware-aware parallel associative scan** instead, with the state kept in SRAM.

#### What "selective" means, and what it costs

**Linear time-invariant (LTI)** means the system's response to an input does not depend on *when* the input arrives or *what* it is — the same $`\bar A, \bar B, \bar C`$ at every step, forever. That is what allowed the convolution trick. It is also a crippling limitation for language.

▸ **The failing is easy to state.** An LTI model processes the word "the" and the word "Kazakhstan" with the identical update rule. It cannot decide *this one matters, hold onto it* or *this is filler, let it wash through*. It has a fixed forgetting rate applied to everything. **A language model that cannot choose what to remember based on what it is reading is a language model that cannot follow an argument.**

Mamba's fix is to make $`B`$, $`C`$ and the step size $`\Delta`$ **functions of the current token**. In particular $`\Delta`$ acts as a gate: a large $`\Delta`$ means "take a big step, overwrite the state with this input"; a small $`\Delta`$ means "barely move, keep what you have."

> **Analogy.** A note-taker in a lecture. An LTI note-taker writes down every seventh word regardless of content. A selective note-taker writes furiously when the lecturer says "this will be on the exam" and puts the pen down during the anecdote about the cat. Same pen, same paper, radically different notes.

**The cost is immediate and structural.** If $`\bar A`$ changes at every step, there is no single kernel $`\bar K`$ to slide along, so there is **no convolution and no FFT** — the entire trick that made §12.6 work has been forfeited.

**The repair: a parallel associative scan.** A **scan** computes all running totals of a sequence. Naively that is sequential, but if the combining operation is **associative** — $`(a\oplus b)\oplus c = a\oplus(b\oplus c)`$ — it can be done in $`\mathcal{O}(\log T)`$ parallel rounds by a tree.

> **Analogy.** Adding a thousand numbers. One at a time takes a thousand steps. Or pair them up and add simultaneously: 500 sums, then 250, then 125 — about ten rounds. Addition's associativity is what permits the pairing, and a linear recurrence's composition is associative for the same reason.

▸ **And "hardware-aware" is the other half of the answer, echoing FlashAttention exactly.** Mamba's state is much larger than a classical RNN's, so writing it to HBM at every step would dominate the run time. The implementation keeps it in **SRAM** and recomputes what it needs on the backward pass — the identical trade FlashAttention makes, for the identical reason. **Two of the most important architectures of the 2020s are both, at bottom, the same observation about GPU memory hierarchy.** (Tri Dao is an author of both, which is not a coincidence.)

### The honest comparison

| | Transformer | Mamba/SSM |
|---|---|---|
| Training | $`O(T^2)`$, perfectly parallel | $`O(T)`$, parallel via scan |
| Inference memory | $`O(T)`$ KV cache | $`O(1)`$ state |
| Selective recall from long context | **excellent** | weaker |
| In-context learning / induction | strong | weaker in pure form |

▸ **The empirical resolution is hybrids.** Jamba, Zamba, Samba and similar interleave a few full-attention layers among many Mamba layers, getting near-transformer recall at a fraction of the KV cache. This is the current practical answer, and a fair thing to say when asked "will transformers be replaced?": *not replaced, hybridized.*

#### Reading the comparison table

Every row of that table traces back to **one** structural fact: *the transformer keeps every past token; the SSM keeps a fixed-size summary.* Everything else follows.

| Row | Why it comes out that way |
|---|---|
| Training $`\mathcal{O}(T^2)`$ vs $`\mathcal{O}(T)`$ | The transformer compares all pairs; the SSM walks the sequence once |
| Inference memory $`\mathcal{O}(T)`$ vs $`\mathcal{O}(1)`$ | The KV cache grows with every token; the state does not |
| Selective recall | You cannot retrieve verbatim what you never stored verbatim |
| In-context learning | Induction heads (Ch. 13, Ch. 32) work by *matching a token against every earlier token* — which needs the earlier tokens |

▸ **The single most useful sentence about this trade: a transformer's memory cost is the price of a perfect transcript, and an SSM's constant memory is the price of only keeping notes.** Ask "what was the account number on line 400?" and the transcript wins every time. Ask "what is this document about?" and the notes are fine, and vastly cheaper.

**Why hybrids are the resolution rather than a fudge.** Most layers do not need perfect recall — they are doing local composition, syntax, feature construction. A few layers do the retrieval. So put attention in a handful of layers and Mamba in the rest, and the KV cache shrinks by the ratio. A model with attention in 1 layer out of 8 carries **one-eighth** the KV cache while retaining most of the recall, because a single attention layer that can see everything is enough to fetch the needle.

> **Analogy.** An office that keeps most of its paperwork as one-page summaries and a small archive of complete originals. You do not need the full original of every document; you need to be able to reach *some* originals when a question demands one. The design question is not "summaries or originals" but "what fraction of each," and the empirical answer so far is that a rather small fraction of originals suffices.

> **Where this came from.** State-space models entered deep learning through **Albert Gu** and **Christopher Ré** at Stanford: **HiPPO** in 2020, then **S4** in 2021, which produced a startling result on the *Long Range Arena* benchmark — a suite designed to be hard for transformers over very long sequences — and made the field take linear recurrence seriously again after roughly five years of assuming attention had settled the question. **Mamba** (Albert Gu and Tri Dao, 2023) added selectivity and the hardware-aware scan.
>
> Mamba's reception is worth knowing: despite very wide attention and rapid adoption in open-source work, **the paper was rejected from ICLR 2024**, a decision that became a minor public controversy about peer review in a field moving faster than its review cycles. The technical substance was unaffected — the architecture was already being built on by the time the reviews appeared. It sits alongside the more famous historical cases in this book (backpropagation ignored for sixteen years, Chapter 1) as a reminder that the review process and the field's actual attention are only loosely coupled.

---

## 12.7 KV-cache reduction

At inference, past keys and values are cached to avoid recomputation. Size:

▸ $$\text{KV cache bytes} = 2\times L\times T\times h_{kv}\times d_{\text{head}}\times \text{bytes/elem}$$

**Numbers.** A 70B-class model *with full MHA* — $`L=80`$, $`h_{kv}=64`$, $`d_{\text{head}}=128`$, $`T=4096`$, bf16:
$$2\times80\times4096\times64\times128\times2 = 10.7\ \text{GB per sequence.}$$
For a batch of 32, that is 343 GB — far more than the 140 GB of weights. **The KV cache, not the model, is usually the binding memory constraint in serving.**

▸ *This is exactly why no shipped 70B model uses MHA.* LLaMA-2-70B has the geometry above but uses **GQA with $`h_{kv}=8`$**, cutting the cache 8× to 1.34 GB per sequence. The MHA figure is what GQA was invented to avoid — keep both numbers in mind, since the contrast is the argument.

#### Unpacking the KV-cache formula

First, **why a cache exists at all.** Generating token 1000 requires attention over tokens 1–999. Their keys and values were already computed when those tokens were generated, and they never change — a key is a fixed function of a token and its position. Recomputing them for every new token would make generation $`\mathcal{O}(T^2)`$ *per token*. So you store them. That store is the KV cache.

Now every factor in the formula, and where it comes from:

| Factor | Read aloud | Why it's there |
|---|---|---|
| $`2`$ | "two" | You cache **both** a key and a value per token |
| $`L`$ | "L" | **Every layer** has its own attention and its own cache |
| $`T`$ | "T" | One entry per **token so far** — this is the one that grows |
| $`h_{kv}`$ | "h sub k-v" | One entry per **key/value head** |
| $`d_{\text{head}}`$ | "d head" | Each entry is a vector this wide |
| bytes/elem | — | 2 for bf16, 1 for int8 |

▸ **Say it aloud: "two, for keys and values, times layers, times tokens, times key-value heads, times head width, times bytes."** Six factors multiplied. There is no subtlety anywhere in it — which is precisely what makes it dangerous, because six moderate numbers multiply into an enormous one.

**Walk the 10.7 GB.** $`2 \times 80 \times 4096 \times 64 \times 128 \times 2`$:

- $`80 \times 4096 = 327{,}680`$ (layer-token pairs)
- $`\times\, 64 \times 128 = 8192`$ numbers per layer-token $`\;\Rightarrow\; 2.68\times10^9`$ numbers
- $`\times\, 2`$ (keys and values) $`\times\, 2`$ bytes $`= 1.07\times10^{10}`$ bytes $`=`$ **10.7 GB**

For **one** sequence, at a context of 4096 — which by 2026 standards is short.

**Now the comparison that matters.** A 70B model in bf16 weighs $`70\times10^9\times2 = 140`$ GB. That is fixed: it does not matter whether you serve one user or a thousand. But the KV cache is **per sequence**, so a batch of 32 costs $`32\times10.7 = 343`$ GB — nearly 2.5× the weights.

▸ **Read the consequence for a serving business.** Weights are a one-time cost you amortize over every user. The cache is a per-user cost you pay again for every concurrent request. **Batch size — which is to say, how many customers you serve per GPU, which is to say your cost per token — is set by the KV cache, not by the model.** This is why the fixes below are not micro-optimizations; each 8× cache reduction is roughly an 8× increase in how many users a given cluster can hold.

> **Analogy.** A restaurant. The weights are the kitchen: expensive, built once, shared by everyone. The KV cache is the table each diner occupies for the whole meal. You can install a bigger kitchen, but if every diner needs a table for eight, your covers-per-night is set by the floor space. GQA, MLA and quantized caches are all ways of seating people at smaller tables.

### The fixes

**MQA (Multi-Query Attention):** all query heads share **one** K/V head. $`h_{kv}=1`$ ⇒ 64× reduction. Noticeable quality loss.

**GQA (Grouped-Query Attention):** $`g`$ K/V heads shared among $`h`$ query heads. $`h_{kv}=8`$ with $`h=64`$ ⇒ 8× reduction at near-MHA quality. ▸ **The current default** (LLaMA-2/3 70B, Mistral, most production models). It is a clean interpolation: $`g=h`$ is MHA, $`g=1`$ is MQA.

**MLA (Multi-head Latent Attention, DeepSeek):** compress K and V jointly into a low-rank latent $`c_t = W_{DKV}x_t`$ and cache only $`c_t`$, reconstructing K/V on the fly. ~90%+ cache reduction with quality *better* than GQA at matched cache size.

**Quantized KV cache:** store in int8 or int4. 2–4× reduction, small quality cost, composes with the above.

**Eviction/streaming:** StreamingLLM keeps the first few "sink" tokens plus a recent window — enables unbounded streaming, at the cost of  forgetting the middle. H2O keeps "heavy hitter" tokens by accumulated attention mass.

#### The five fixes, decoded

Every one of them attacks a different factor in that six-term product. That is the clean way to organize them:

| Fix | Which factor it shrinks | Typical saving |
|---|---|---|
| MQA | $`h_{kv}: 64 \to 1`$ | 64× |
| GQA | $`h_{kv}: 64 \to 8`$ | 8× |
| MLA | $`d_{\text{head}}`$, effectively — cache a compressed latent instead | ~10× |
| Quantized cache | bytes/elem: $`2\to1`$ or $`0.5`$ | 2–4× |
| Eviction / streaming | $`T`$ — stop storing all of it | Unbounded |

**MQA and GQA, made concrete.** In standard **MHA (Multi-Head Attention)** every one of 64 query heads has its own key head and value head — 64 independent "what am I looking for" / "what do I advertise" pairs.

- **MQA (Multi-Query Attention)** keeps all 64 query heads but gives them **one shared** key/value head. All 64 heads look at the same advertisement board; only their questions differ.
- **GQA (Grouped-Query Attention)** splits the difference: 8 key/value heads, each shared by 8 query heads.

> **Analogy.** A newsroom with 64 reporters. MHA gives each reporter a personal filing cabinet. MQA gives all 64 one shared cabinet — cheap, but everyone's notes have been merged and the specialists lose their distinctive material. GQA gives eight desks of eight reporters each their own cabinet: the political desk and the sports desk keep separate files, while reporters *within* a desk share, which costs little because their interests overlap anyway.

▸ **Why GQA loses so little quality is the interesting part.** Attention heads turn out to be substantially redundant — many heads within a layer learn overlapping functions. Forcing groups of eight to share a key/value projection removes duplication rather than capability. MQA's quality loss appears when you compress past that point and heads that were doing  different jobs are made to share. **GQA sits at the knee of the curve, and $`g`$ is a clean dial**: $`g = h`$ recovers MHA exactly, $`g = 1`$ is MQA, and the interesting settings are in between.

**MLA, in one sentence.** Instead of caching $`K`$ and $`V`$ directly, project them jointly down to a small latent vector $`c_t`$, cache *that*, and reconstruct $`K`$ and $`V`$ on the fly with a matrix multiply. It is the low-rank idea from §1.1.3 applied to the cache: **store the coefficients, not the reconstruction.** It buys extra FLOPs at read time to save bytes at rest — which, given the memory-versus-arithmetic ratio established in §12.5, is a trade in the favourable direction.

**Quantized cache, in one sentence.** Store each cached number in 8 or 4 bits instead of 16. Keys and values are activations with fairly well-behaved ranges, so the precision loss is tolerable — and it **multiplies with** the others, since it changes a different factor. GQA (8×) and int8 (2×) together give 16×.

**Eviction and streaming, and the sink-token surprise.** StreamingLLM keeps a sliding window of recent tokens plus — the counterintuitive part — **the first handful of tokens of the sequence, permanently.** Drop those first few and quality collapses, even though they are usually something meaningless like a start-of-text marker.

▸ **The explanation is a  lovely piece of mechanistic detective work.** Softmax weights must sum to 1: attention is *forced* to distribute all of its mass somewhere, even when a head has nothing it wants to look at. Models learn to dump the surplus onto the first token — an **attention sink**, a designated place to throw away attention you don't need. Evict the sink and that mass is forced onto tokens that actually matter, corrupting their weights. **A model's most-attended token can be the one carrying the least information**, which should adjust how you read attention maps forever.

> **Where this came from.** **MQA** was proposed by **Noam Shazeer** in 2019, in a short single-author paper titled *"Fast Transformer Decoding: One Write-Head is All You Need"* — a play on the title of the transformer paper he had co-written two years earlier. It sat largely unused for several years, because in 2019 nobody was serving models where the KV cache was the binding constraint. **GQA** (Ainslie and colleagues at Google, 2023) arrived precisely when that changed, and was adopted almost immediately across the industry — LLaMA-2-70B, Mistral, and most production models since. It is a good example of an idea whose merit did not change at all, while the environment around it changed enough to make it essential. **MLA** came from **DeepSeek** in 2024. **StreamingLLM** and the attention-sink observation came from Guangxuan Xiao and colleagues (MIT and Meta) in 2023.

---

## 12.8 Does long context work?

**"Needle in a haystack"** tests retrieval of a planted fact. Most modern models pass it — but it is a weak test.

▸ **"Lost in the middle"** (Liu et al.): accuracy is high for information at the start and end of the context and **sags in the middle**, tracing a U-shape. Attributable partly to positional-encoding decay and partly to training-data position statistics.

Harder benchmarks (RULER, multi-hop over long context, aggregation queries) show that **effective context is typically much shorter than advertised context** — a model claiming 1M tokens may degrade materially past 100k on tasks requiring integration rather than lookup.

**Practical rule:** long context and retrieval are complements, not substitutes. Retrieval reduces the haystack; long context handles what's left. (Ch. 18.)

#### What "lost in the middle" actually says

**The needle-in-a-haystack test**, first. Take a long document, insert one sentence somewhere that does not belong ("the best thing to do in San Francisco is eat a sandwich in Dolores Park"), and ask the model to repeat it. Vary the length and the position of the insertion. It produces a nice heat map, it is easy to run, and **modern models pass it comfortably** — which is exactly why it is a weak test.

▸ **The weakness is that copying one distinctive sentence is a *lookup*, and lookup is the easiest thing attention does.** One induction-style head matching the question against the passage is sufficient. Nothing about it requires the model to have *read* the intervening 90,000 tokens, only to have indexed them.

**"Lost in the middle"** is the sharper finding. Place the needle at different depths and accuracy traces a **U-shape**: high at the start, high at the end, sagging in the middle. Two contributing causes:

1. **Positional-encoding decay.** RoPE's soft long-range attenuation (§12.4) makes distant tokens harder to reach — that explains the *recency* half of the U.
2. **Training-data position statistics.** Documents put important material at the top: titles, abstracts, opening sentences. The model learned that the beginning of a document matters, because in its training data it did. That explains the *primacy* half. **The model's positional prior is a compressed summary of how human documents are actually written**, which is either reassuring or alarming depending on your temperament.

> **Analogy.** Recalling a lecture you attended. You remember the opening — you were fresh and attentive — and you remember the closing summary. The forty minutes in between are a fog. This is not a defect specific to language models; it is the **serial position effect**, a robust finding in human memory research since the 19th century, and the fact that transformers reproduce it is a coincidence of mechanism rather than a shared cause. It is nonetheless a very handy way to remember which shape the curve has.

**Why "effective context is shorter than advertised context."** Distinguish three things a long context might be asked to do, in increasing order of difficulty:

| Task | What it needs | How models do |
|---|---|---|
| **Retrieval** — "what was the account number?" | Find one span | Well; this is needle-in-a-haystack |
| **Multi-hop** — "which client did the person who signed page 3 also sign for?" | Find a span, then find another *using* the first | Substantially worse |
| **Aggregation** — "how many times does the report contradict itself?" | Integrate across the *whole* context | Worst; often near-chance at long lengths |

▸ **The single number on the box — "1M token context" — describes what the model will *accept*, not what it will *use*.** Benchmarks like RULER exist to measure the second thing, and they routinely show a model advertised at 1M degrading materially past 100k on anything beyond lookup. When evaluating a long-context claim, the question to ask is never "how long?" but "how long, doing what?"

**Why retrieval and long context are complements.** Retrieval (Ch. 18) shrinks a million tokens to the ten thousand that could plausibly be relevant; long context then lets the model actually reason over those ten thousand rather than the 500 that older models could hold. **One reduces the haystack, the other raises the ceiling on what you can do once it's reduced.** Arguments about which "wins" usually turn out to be arguing about different stages of the same pipeline.

> **Where this came from.** *"Lost in the Middle: How Language Models Use Long Contexts"* was published by **Nelson Liu and colleagues at Stanford in 2023**, at a moment when context windows had just jumped from 4k to 32k and the field's marketing was well ahead of its measurement. The paper's method was almost aggressively simple — put the answer in different places, plot the accuracy — and its influence came from that simplicity: it was trivially reproducible, so it was reproduced everywhere within weeks. Some of the strongest results in empirical machine learning are of this kind. Not a new architecture, just a well-chosen thing to measure that nobody had bothered to measure.

#### Examples and non-examples: a model actually using its context

**✅ Evidence the context is  in use**

| Example | Why it qualifies |
|---|---|
| Answering "which client did the person who signed page 3 also sign for?" over a 90k-token contract | Two-hop: the second lookup is impossible without the result of the first, so both spans must be live at once |
| Counting how many times a 200-page report contradicts itself | The answer is a function of the *whole* context; no single span contains it |
| Using a variable defined on line 1 of a 40,000-line file correctly at line 39,000 | The dependency spans the full window and the wrong answer is silently plausible |
| A summary of a long transcript that includes a point made only in the middle third | Directly probes the sag in the U-curve |
| RULER-style tasks with multiple needles and distractors | Distractors defeat the single-match shortcut that plain needle tests reward |

**❌ Near-misses — look like proof of long-context use, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Passing a needle-in-a-haystack test at 1M tokens | Copying one distinctive sentence is a **lookup**, and lookup is the easiest thing attention does. One induction-style head suffices | Evidence of indexing, not of reading |
| Low perplexity on a 128k-token document | Token $`t`$ is overwhelmingly predicted from tokens $`t-50`$ to $`t-1`$. Perplexity barely moves whether the first 120k tokens were used or deleted | A measurement of local fluency |
| The model didn't crash and the output reads well | Local coherence comes from the recent window and is fully compatible with the middle 190k tokens being ignored | Fluency |
| Accurately restating the last paragraph of a huge prompt | Recency — the end of the U-curve that was never in doubt | The primacy/recency effect, not integration |
| A high score on a long-context benchmark whose answers all sit in the final 2k tokens | The benchmark has a short-context solution. The context length in the name is decoration | A short-context task with padding |
| "The API accepted 1,000,000 tokens" | That is the input validator agreeing with you | The advertised window |

▸ **The boundary:** a task tests long-context *use* only if its answer is impossible to produce without information from a position the model would otherwise skip. If the answer is recoverable from the first few thousand and last few thousand tokens, the benchmark measures the U-curve's peaks and tells you nothing about its trough.

> **Common misconception.** *"The model has a 1M-token context window, so it uses 1M tokens."* The number on the box is a statement about what the model will **accept**, not about what it will **use** — and the gap between the two is large and measurable. "Lost in the middle" (Liu et al., 2023) shows accuracy tracing a U-shape over the position of the needle: strong at the start, strong at the end, sagging badly in between. Harder suites like RULER routinely show a model advertised at 1M degrading materially past 100k on anything beyond lookup. The belief is tempting for three compounding reasons. The window length is a , checkable hard number, so it feels like a specification rather than a claim. The most-run test — needle in a haystack — passes, and it passes because it asks for the one operation that was never difficult. And nothing *fails loudly*: a model that ignored the middle of your document returns a confident, fluent, well-formed answer, with no error and no warning. **The right question is never "how long?" but "how long, doing what?"**

> **Common misconception.** *"The middle-of-context sag is a bug in the positional encoding, so a better encoding will fix it."* Only half of it is mechanical. RoPE's soft long-range attenuation explains the *recency* side of the U — distant tokens really are harder to reach. The *primacy* side comes from somewhere else entirely: **the training data**. Human documents front-load their important material into titles, abstracts and opening sentences, so a model that learned "the beginning of a document matters" learned something true about the corpus it was trained on. No amount of positional-encoding engineering removes a prior that the data itself installed; that half needs different data or different training, not different sinusoids. The belief is tempting because the two halves of the curve look symmetric and invite a single explanation — and because the mechanical half is the one you can fix with a config change.

---

## Did you know?

- **RoPE was first published in Chinese, on a personal blog.** Jianlin Su described rotary position embeddings on his own mathematics blog in 2021 before the English *RoFormer* paper appeared. It reached the mainstream not through citations but through code — EleutherAI put it in GPT-J and GPT-NeoX, LLaMA inherited it, and by 2026 it is in essentially every major open model.

- **The constant 10000 in every positional encoding has never really been justified.** It appears in "Attention Is All You Need" without derivation, was carried into RoPE unchanged, and sat unexamined for six years — until context extension made it the single most important tuning knob in the architecture. LLaMA-3 sets it to 500,000.

- **The transformer paper admits the sinusoidal encoding was a coin flip.** The authors report that learned positional embeddings performed *nearly identically*, and say they picked sinusoids because they might extrapolate to longer sequences. In practice they extrapolate poorly. The most-copied design decision in the architecture was chosen on a hypothesis that did not hold up.

- **NTK-aware context scaling was invented in a Reddit post.** The technique that let people extend LLaMA's context without any fine-tuning was posted to r/LocalLLaMA in mid-2023 by a user going by *bloc97*, was implemented in open-source inference stacks within days, and only later became part of a peer-reviewed paper — YaRN, on which bloc97 is a co-author. The "NTK" label is a loose analogy to the Neural Tangent Kernel literature rather than a derivation from it.

- **Language models dump their unwanted attention on the first token.** Softmax weights must sum to 1, so a head with nothing it wants to look at still has to put its mass somewhere. Models learn to designate the first token as a rubbish bin — an **attention sink**. Evict it from the cache to save memory and quality collapses, which is how the behaviour was discovered.

- **FlashAttention's core recurrence was published four years earlier, for an unrelated reason.** The online-softmax update came from Milakov and Gimelshein at NVIDIA in 2018, as a way to compute a softmax in fewer memory passes. Nobody connected it to attention until 2022.

- **The KV cache usually costs more memory than the model.** A 70B model in bf16 weighs 140 GB. Serving 32 concurrent sequences of 4096 tokens with full multi-head attention would cost 343 GB of cache. The thing being served is smaller than the bookkeeping for serving it.

- **Mamba was rejected from ICLR 2024.** By the time the reviews appeared, the architecture had already been widely adopted and built upon in open-source work. It joins backpropagation — ignored for sixteen years after its 1970 publication — on the list of results the review process failed to route correctly.

- **State-space models are borrowed from 1960s control theory.** The equations $`h'(t) = Ah(t) + Bx(t)`$, $`y = Ch(t)`$ are the standard linear-system description used for autopilots and industrial process control. The deep-learning contribution was not the model but the initialization (HiPPO) and the GPU kernel.

- **An "efficient transformer" boom produced a dozen methods in eighteen months, and almost none survive.** Between 2019 and 2021 the field published Sparse Transformer, Reformer, Longformer, Linformer, BigBird and Performer, among others — enough that Google built a benchmark suite (*Long Range Arena*) just to compare them. FlashAttention then made exact attention fast enough that the whole category lost its reason to exist. BigBird's authors had even proved their method was Turing-complete; the proof did not help.

- **Attention is a party where everyone shakes hands with everyone.** Ten guests: 45 handshakes. A hundred: 4,950. A hundred thousand tokens: ten billion pairs, per head. At 32 heads in bf16 that is 640 GB for a single layer's attention matrix — roughly four thousand times what fits on a top-end GPU.

- **Arithmetic is nearly free and memory is the whole bill.** An A100 can perform on the order of a hundred floating-point operations in the time it takes to fetch one number from its main memory. FlashAttention and Mamba are, at bottom, the same response to that one ratio — and Tri Dao is an author of both.

---

## Check for Understanding

**Attention is permutation-equivariant so position must be injected, and the winning scheme (RoPE) rotates queries and keys by angles proportional to their absolute positions so that their dot product depends only on the difference — while the practical limits on context come not from the $`O(T^2)`$ FLOPs, which FlashAttention made cheap by never materializing the attention matrix, but from the KV cache, which grows linearly with length and is what GQA, MLA, and state-space models exist to shrink.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why can't a transformer tell "dog bites man" from "man bites dog" without help?** What is it about attention specifically — not the feed-forward layer, not layer norm — that throws order away?
2. **Why does the sinusoidal encoding use many frequencies instead of one?** (The binary-counter answer is the good one: fast dimensions are the low-order bits, slow ones the high-order bits.)
3. **What does RoPE do, in terms of clock hands or a carousel?** Why does rotating both vectors by their *absolute* positions leave a score that depends only on the *difference*?
4. **Why is RoPE compatible with the KV cache when a T5-style relative bias isn't?** (Answer in terms of when the position gets baked in, not in terms of formulas.)
5. **Why is a context window hard to extend after training, when the rotation formula happily produces angles for any position?** What is actually undertrained?
6. **Why does NTK-aware scaling often need no fine-tuning while position interpolation does?** (Which dimensions were healthy, and which medicine got applied to them?)
7. **What does FlashAttention change, given that it computes exactly the same answer as before?** Why is recomputing something faster than remembering it?
8. **Why did a dozen clever approximate-attention methods lose to making the exact one fast?**
9. **Why is the KV cache, not the model's weights, what limits how many users you can serve per GPU?**
10. **What does GQA give up, and why does giving that up cost so little?**
11. **What can a transformer do that a Mamba-style model can't, and why is it a consequence of memory size rather than of cleverness?**
12. **Why does a model recall facts from the start and end of a long document better than from the middle?** Name both causes.

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 13 — GPT: Autoregressive Language Modelling](13-gpt-autoregressive-language-models.md)
