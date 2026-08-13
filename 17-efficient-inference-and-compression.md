# Chapter 17 — Efficient Inference & Compression

> **Prerequisites:** Ch. 11, 12, 14.
> **Why this matters:** a model is trained once and served billions of times. Inference is where almost all of the lifetime compute is spent, and where most ML engineering jobs actually live.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

This chapter is unusual: most of its difficulty is **vocabulary**, not mathematics. Skim the table once, then read on.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`\dfrac{\text{FLOPs}}{\text{bytes moved}}`$ | "arithmetic intensity" | How much arithmetic you do per byte you drag out of memory |
| $`\gamma`$ | "gamma" | How many tokens the small draft model guesses before the big one checks |
| $`\alpha`$ | "alpha" | **Three jobs in this chapter**: the speculative-decoding acceptance rate, the distillation mixing weight, and LoRA's scaling constant |
| $`\min\!\left(1,\ \frac{p(x)}{q(x)}\right)`$ | "min of one and p over q" | The accept-or-reject coin flip in speculative decoding |
| $`x_q,\ s,\ z`$ | "x-quantized, scale, zero-point" | The stored integer, the step size it represents, and the offset |
| $`\mathrm{round}(\cdot)`$ | "round" | Snap to the nearest whole number |
| $`b`$ | "b" | Bits per stored number ($`b=4`$ means 16 possible values) |
| W4A16 | "W-four, A-sixteen" | 4-bit **W**eights, 16-bit **A**ctivations |
| $`\lVert WX-\widehat WX\rVert_F^2`$ | "Frobenius norm squared of the difference" | Total squared gap between what the layer used to output and what it outputs now |
| $`H = 2XX^\top`$ | "H equals two X X-transpose" | The curvature of that squared error — which weights are dangerous to move |
| $`\tau`$ | "tau" | **Temperature** — divide the logits by it before softmax to flatten them |
| $`p_T^{(\tau)},\ p_S^{(\tau)}`$ | "p-T, p-S at temperature tau" | **T**eacher and **S**tudent probability distributions, softened |
| $`W_0 + \frac{\alpha}{r}BA`$ | "W-nought plus alpha-over-r, B A" | Frozen pretrained weights, plus a thin correction |
| $`r`$ | "r" | The **rank** of the LoRA correction — the width of the bottleneck |
| $`\tau_i`$ | "tau-i" | A **task vector** (fine-tuned minus pretrained). *Not* the temperature — same letter, different job |
| 2:4 | "two-of-four" | Exactly two zeros in every four consecutive weights |
| $`\theta^{\text{pre}}`$ | "theta pre" | The pretrained parameter vector, before any fine-tuning |

### Full forms for the abbreviations in this chapter

Every one of these is expanded on first use in the added commentary, but here they are in one place.

| Short | Full form |
|---|---|
| AWQ | Activation-aware Weight Quantization |
| bf16 | brain floating point, 16-bit |
| EOS | End Of Sequence (token) |
| FLOP | FLoating-point OPeration |
| GPTQ | Generative Pre-trained Transformer Quantization |
| HBM | High-Bandwidth Memory |
| ITL | Inter-Token Latency |
| KD | Knowledge Distillation |
| KV | Key–Value (cache) |
| LoRA | Low-Rank Adaptation |
| NF4 | Normal Float, 4-bit |
| PEFT | Parameter-Efficient Fine-Tuning |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |
| SLO | Service-Level Objective |
| STE | Straight-Through Estimator |
| TPOT | Time Per Output Token |
| TTFT | Time To First Token |
| W8A8 / W4A16 | Weight-bits / Activation-bits notation |

---

## 17.1 The two phases of inference

### The one-line idea

Generating text has two completely different computational profiles — a compute-bound phase that processes the prompt in parallel, and a memory-bound phase that emits one token at a time — and almost every optimization targets the second.

### The analogy

Cooking a meal versus serving it one spoonful at a time. Preparation (prefill) uses the whole kitchen at once and is limited by how fast you can chop. Serving (decode) is limited by how fast you can walk to the table and back — the work per trip is trivial, the *trip* is the cost.

### Arithmetic intensity

▸ $$\text{arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes moved}}$$

A GPU is compute-bound above its ridge point and memory-bound below it. For an H100 SXM at bf16 the dense ridge point is $`\frac{495\ \text{TFLOP/s}}{3.35\ \text{TB/s}}\approx 148`$ FLOP/byte (≈295 if you quote the with-sparsity peak — see Ch. 14 §14.7 on which to use).

| Phase | Work | Intensity | Bound by |
|---|---|---|---|
| **Prefill** | $`T`$ tokens through the whole model at once | high (matmuls) | **compute** |
| **Decode** | 1 token, but every weight must be read | $`\approx 2B`$ FLOP/byte | **memory bandwidth** |

▸ **At batch size 1, decode reads every parameter to produce one token.** A 70B bf16 model is 140 GB; at 3.35 TB/s that's a hard floor of 42 ms/token = **24 tokens/s, no matter how fast the GPU computes.** The arithmetic is trivial and the bandwidth is everything.

▸ **The direct consequence: batching is nearly free during decode.** Reading the weights once and applying them to 64 sequences costs almost the same wall-clock as applying them to 1. This single fact drives the entire design of modern serving systems.

#### Reading arithmetic intensity in plain English

Every symbol first.

- **FLOP** is **FL**oating-point **OP**eration — one multiply, or one add, on ordinary decimal numbers. "FLOPs" is the plural (the count); "FLOP/s" with the slash is the *rate*, operations per second. The two are constantly confused and the slash is the only thing distinguishing them.
- **Bytes moved** means bytes shifted between the GPU's main memory (**HBM**, High-Bandwidth Memory — the chips ringing the die) and the tiny fast scratchpad next to the arithmetic units. Nothing can be computed on until it has made that trip.
- **TFLOP/s** is tera-FLOP per second: $`10^{12}`$ operations a second. **TB/s** is terabytes per second.
- The **ridge point** is the ratio at which the two limits cross — the intensity at which a chip stops waiting for data and starts waiting for arithmetic.

So the fraction reads aloud as: *"how many arithmetic operations do I get out of each byte I bothered to fetch?"*

> **Analogy.** A carpenter's workshop where the timber is stored in a yard half a mile away. The saw is absurdly fast. What limits how many chairs you build is how quickly you can wheel timber in from the yard. **Arithmetic intensity is "how much cutting do I do per plank I fetch?"** If you fetch a plank and make one cut, the saw is idle 99% of the time and buying a faster saw changes nothing. If you fetch a plank and make two hundred cuts, the saw is the bottleneck and a faster one helps.
>
> The GPU's ridge point is the break-even: **148 cuts per plank** on an H100. Below that you are a person pushing a wheelbarrow; above it you are a saw.

**Put the numbers in.** The ridge point is one division:

$$\frac{495\ \text{TFLOP/s}}{3.35\ \text{TB/s}} = \frac{495\times10^{12}\ \text{FLOP/s}}{3.35\times10^{12}\ \text{byte/s}} \approx 148\ \frac{\text{FLOP}}{\text{byte}}$$

The seconds cancel — which is exactly why this is a *property of the chip* and not of any particular workload. It is a fixed line drawn on the floor, and every kernel you ever write lands on one side of it or the other.

**Now decode.** A single parameter, once fetched, gets used for one multiply and one add — **2 FLOPs** — for each sequence in the batch. With $`B`$ sequences in flight you therefore extract $`2B`$ FLOPs from that one fetch. That is the entire content of the table's $`\approx 2B`$ entry: **arithmetic intensity during decode is essentially the batch size.** At $`B=1`$ you are down near single digits, a hundred-fold below the ridge point of 148. At $`B=128`$ you have crossed it.

**Now the 24-tokens-per-second floor, worked out.** A 70-billion-parameter model in bf16 stores 2 bytes per parameter:

$$70\times10^{9}\ \text{params} \times 2\ \frac{\text{bytes}}{\text{param}} = 140\ \text{GB}$$

To emit *one* token, every one of those bytes must cross the memory bus at least once:

$$\frac{140\ \text{GB}}{3.35\ \text{TB/s}} = \frac{140}{3350}\ \text{s} \approx 0.042\ \text{s} = 42\ \text{ms} \quad\Longrightarrow\quad \frac{1}{0.042} \approx 24\ \text{tokens/s}$$

▸ **Notice what does not appear in that calculation: the GPU's speed.** No term for TFLOP/s, no term for clock rate, no term for how many tensor cores the chip has. **You could make the arithmetic units infinitely fast and get exactly 24 tokens per second.** This is the most important number in the chapter, because it reframes the whole subject: serving a large language model is not a computation problem, it is a **logistics** problem. Almost everything in this chapter — quantization, speculative decoding, batching, KV-cache tricks — is an attack on the numerator or the denominator of that one fraction, not on the arithmetic.

**Why this makes batching free.** Look at the fraction again. Serving 64 users means fetching the same 140 GB and doing 64× the arithmetic with it. The fetch cost is unchanged; only the arithmetic grows, and the arithmetic had 148× of headroom. **You get 64 tokens for the price of one.** This is why an API provider's cost per token collapses with traffic, and why a model that feels slow when you are the only user on the box is perfectly economical at scale.

> **Where this came from.** The picture of a "ridge point" separating memory-bound from compute-bound work is the **Roofline model**, published by **Samuel Williams, Andrew Waterman, and David Patterson** at Berkeley in 2009 in *Communications of the ACM*. Their subject was multicore CPUs; large language models did not exist and neither did the hardware anyone now runs them on. Patterson is one of the architects of RISC and a co-author of the standard computer-architecture textbook, and shared the 2017 Turing Award with John Hennessy. The model earned its name from the shape of its log–log plot: a diagonal line (bandwidth-limited) meeting a horizontal ceiling (compute-limited), which looks like the roof of a house. Fifteen years later it is the first diagram drawn on every whiteboard where someone is asked why their model is slow.

---

## 17.2 Serving systems

**Static batching** wastes enormous capacity: the batch is held until the longest sequence finishes.

▸ **Continuous (in-flight) batching:** as soon as one sequence emits EOS, evict it and admit a waiting request. Typically **10–20× throughput improvement** over static batching. This is the single highest-leverage serving optimization.

▸ **PagedAttention (vLLM):** allocate the KV cache in fixed-size **blocks** with a page table, exactly like virtual memory. Solves the fragmentation problem — you no longer need to reserve `max_length` per sequence, so memory utilization goes from ~30% to >90%. Also makes **prefix sharing** trivial: a shared system prompt occupies one copy of the blocks across all requests, with copy-on-write for divergence.

**Chunked prefill:** split long prompts into chunks and interleave them with decode steps, so a long prefill doesn't stall everyone else's token stream. Balances TTFT (time-to-first-token) against ITL (inter-token latency).

**Disaggregated serving:** run prefill and decode on *separate* hardware pools, since one is compute-bound and the other memory-bound. Increasingly standard at scale.

**The metrics to know:** TTFT, ITL/TPOT, end-to-end latency, throughput (tokens/s per GPU), goodput (requests meeting an SLO).

#### The serving vocabulary, decoded

That last line is five acronyms in a row, so here they are spelled out and explained.

| Short | Full form | What it measures | Who notices |
|---|---|---|---|
| **TTFT** | **T**ime **T**o **F**irst **T**oken | How long from pressing enter to the first word appearing | The user, immediately. Silence feels broken |
| **ITL** | **I**nter-**T**oken **L**atency | The gap between consecutive words once it starts | The user, as "reading speed" |
| **TPOT** | **T**ime **P**er **O**utput **T**oken | The same quantity as ITL, averaged over the response | Same |
| **Throughput** | — | Total tokens per second the whole GPU emits, summed over all users | The person paying the bill |
| **SLO** | **S**ervice-**L**evel **O**bjective | A promise, e.g. "95% of requests get TTFT under 500 ms" | The contract |
| **Goodput** | — | Throughput *counting only the requests that met the SLO* | The honest engineer |
| **EOS** | **E**nd **O**f **S**equence | The special token the model emits meaning "I'm finished" | The scheduler |

▸ **Throughput and latency are in direct opposition, and goodput is the honest referee.** Bigger batches raise throughput (more tokens per weight-read) and raise latency (each user waits behind more work). A server tuned purely for throughput can report a magnificent tokens-per-second figure while every single user times out. **Goodput exists because it is the only one of these numbers you cannot game** — it multiplies your throughput by the fraction of it that anyone actually wanted.

> **Analogy for TTFT versus ITL.** A restaurant. TTFT is how long before *anything* arrives at your table; ITL is the pace of the courses after that. Diners forgive a slow kitchen far less than a leisurely meal — a ten-minute wait for bread feels worse than a ten-minute gap between courses, even though the second costs more total time. Serving systems are tuned around exactly this asymmetry, which is why "stream the tokens as they come" became universal: it converts a 20-second wait into a 0.4-second wait followed by 20 seconds of visible progress.

#### Unpacking continuous batching and PagedAttention

**Static batching, and why it wastes so much.** Collect 32 requests, run them together, return all 32 when the *last* one finishes. If 31 requests want 20 tokens and one wants 2,000, then 31 slots sit idle computing padding for 1,980 steps. The GPU is busy and almost none of the work is useful.

**Continuous batching** — also called *in-flight* or *iteration-level* batching — changes the unit of scheduling from **a batch** to **a single decode step**. After every step, finished sequences are evicted and queued ones are admitted into the freed slots.

> **Analogy.** A bus versus a moving walkway. A bus leaves when it is full and everyone waits for the slowest passenger to reach their stop before the bus is free again. A moving walkway lets people step on and off continuously; the belt never stops and never runs empty. **Continuous batching is a moving walkway for tokens** — and the reported 10–20× improvement is not an optimization of the arithmetic at all. It is the removal of idleness.

**Now PagedAttention, and why "page table" is the right word.** The KV cache (Ch. 12) grows by one entry per token per layer, and you do not know in advance how long a response will be. The naive fix is to reserve `max_length` worth of memory per sequence up front. If the limit is 8,192 tokens and the average answer is 300, **you have reserved 27× more memory than you use**, and that reservation is what caps how many users fit on the card.

PagedAttention borrows the operating system's answer:

| Operating system | PagedAttention |
|---|---|
| A process thinks it has one long contiguous address space | A sequence thinks its KV cache is one long contiguous array |
| Physically it is scattered across fixed-size **pages** in RAM | Physically it is scattered across fixed-size **blocks** in HBM |
| A **page table** maps virtual → physical | A block table maps logical token position → physical block |
| Pages are allocated on demand, not reserved up front | Blocks are allocated as tokens are generated |
| Two processes can share a read-only page, copy-on-write if either writes | Two requests with the same system prompt share its blocks, copy-on-write when they diverge |

▸ **Note what the shared-page row buys you.** If ten thousand users hit the same 2,000-token system prompt, a naive server stores ten thousand copies of its KV cache. Under paging it stores **one**, and every request points at it. This is *prefix sharing*, and at scale it is often worth more than the fragmentation fix that motivated the design.

**Chunked prefill, in one sentence.** A 30,000-token prompt is one enormous compute-bound job; if you run it to completion, every other user's token stream freezes for its duration — the *head-of-line blocking* problem. Chopping it into pieces and interleaving those pieces with everyone's decode steps trades a slightly worse TTFT for that one request against a dramatically better ITL for everyone else.

**Disaggregated serving, in one sentence.** Prefill is compute-bound and decode is memory-bandwidth-bound (§17.1), so they want *different hardware*; running them on the same pool means one of the two phases is always using the wrong machine.

> **Where this came from.** **PagedAttention and vLLM** came out of the Sky Computing Lab at UC Berkeley — Woosuk Kwon, Zhuohan Li and colleagues — and were presented at SOSP in 2023, a *systems* conference rather than a machine-learning one. The paper is unusually explicit that the idea is a straight transplant: virtual memory and paging were invented for the **Atlas computer at the University of Manchester around 1962**, under Tom Kilburn, to let programs pretend they had more memory than the machine physically contained. Sixty years later, the same trick — pretend the array is contiguous, keep a table that says where the pieces really are — took GPU memory utilization on a language-model server from roughly 30% to over 90%. **Iteration-level scheduling**, the idea underneath continuous batching, was introduced a year earlier in the *Orca* serving system at OSDI 2022. Neither contribution involved changing a single thing about the model.

---

## 17.3 Speculative decoding

### The one-line idea

Have a small fast model guess several tokens, then have the big model check them all in a single forward pass — which costs the same as generating one token, because decode is memory-bound.

### The analogy

An assistant drafts a paragraph; the expert reads it and accepts everything up to the first thing they'd have written differently. Reading is one pass regardless of length; writing is one word at a time. So you get several words for the price of reading once.

### The algorithm

1. Draft model $`q`$ generates $`\gamma`$ tokens autoregressively.
2. Target model $`p`$ scores all $`\gamma+1`$ positions **in one forward pass** (parallel, because the tokens are already known).
3. Accept token $`i`$ with probability $`\min\left(1,\frac{p(x_i)}{q(x_i)}\right)`$.
4. On the first rejection, resample from the residual distribution
▸ $$p'(x) = \frac{\max(0,\ p(x)-q(x))}{\sum_{x'}\max(0,\ p(x')-q(x'))}$$
5. Continue.

▸ **The guarantee:** the accepted sequence is distributed **exactly** as if sampled from $`p`$. This is a modified-rejection-sampling argument, and it means speculative decoding is *lossless* — it is a pure latency optimization with no quality trade-off. Being able to state that is the point of the technique.

#### What the accept/reject rule actually says

**The symbols.** $`q`$ is the **draft** model — small, fast, roughly right. $`p`$ is the **target** model — the big one whose output you are contractually obliged to reproduce. $`q(x_i)`$ and $`p(x_i)`$ are the probabilities the two models assign to the same candidate token. $`\gamma`$ (gamma) is how many tokens the draft guesses per round.

**Step 3 read aloud:** *"accept this token with probability: the minimum of one, and the target's probability divided by the draft's probability."*

Unpack the two cases:

- **The target likes the token at least as much as the draft did** ($`p \ge q`$): the ratio is $`\ge 1`$, the minimum is 1, and you accept with certainty. Nothing to argue about — the draft guessed something the big model wanted anyway.
- **The draft was over-eager** ($`p < q`$): you accept only $`p/q`$ of the time. If the draft said 50% and the target says 40%, you keep it 80% of the time. **You are shaving off exactly the excess enthusiasm.**

**Step 4, the residual distribution.** When you reject, you cannot simply resample from $`p`$ — that would double-count the tokens the draft already got right. You must sample from *what is left over*:

$$p'(x) = \frac{\max\big(0,\ p(x)-q(x)\big)}{\sum_{x'}\max\big(0,\ p(x')-q(x')\big)}$$

Read: *"for every token, take how much more the target wants it than the draft did; clip negatives to zero; renormalize so it sums to one."* The $`\max(0,\cdot)`$ throws away tokens the draft *over*-supplied — those have already been accounted for. The denominator is just the normalizing constant that makes it a legal probability distribution.

**Now put real numbers in.** Take a three-token vocabulary $`\{A, B, C\}`$:

| Token | Draft $`q`$ | Target $`p`$ | Accept probability $`\min(1, p/q)`$ |
|---|---|---|---|
| A | 0.5 | 0.4 | $`0.4/0.5 = 0.8`$ |
| B | 0.3 | 0.4 | $`\min(1, 1.33) = 1`$ |
| C | 0.2 | 0.2 | $`1`$ |

The residual is $`\max(0, p-q) = (0,\ 0.1,\ 0)`$, which normalizes to $`p' = (0, 1, 0)`$: **on a rejection, always emit B.** Now trace where each token ends up:

- **A** is emitted only when drafted *and* accepted: $`0.5 \times 0.8 = \mathbf{0.4} = p(A)`$ ✓
- **C** is emitted when drafted and always accepted: $`0.2 \times 1 = \mathbf{0.2} = p(C)`$ ✓
- **B** has two routes. Drafted and accepted: $`0.3\times 1 = 0.3`$. *Or* the draft proposed A and it was rejected — which happens with probability $`0.5\times0.2 = 0.1`$ — and the residual then emits B for certain. Total: $`0.3 + 0.1 = \mathbf{0.4} = p(B)`$ ✓

▸ **Every column matches the target exactly.** That is the whole proof, and it is worth having done once by hand, because "lossless" is the word that makes this technique interesting. It is not "usually indistinguishable" or "within noise." **The output distribution is identical, token for token, to what the big model alone would have produced** — and the rejection rule plus the residual are precisely the machinery that makes it so.

**Why checking $`\gamma+1`$ tokens costs the same as generating one.** Two reasons stacked:

1. **The tokens are already known.** Scoring a known sequence needs one forward pass; the causal mask means position $`i`$ only looks leftward, so all $`\gamma+1`$ positions are scored simultaneously in that single pass. Generating them would have required $`\gamma+1`$ passes, because each one has to exist before the next can be conditioned on it.
2. **Decode is memory-bound** (§17.1). The 140 GB of weights get dragged across the bus once either way. Doing $`\gamma`$ times more arithmetic with them is free, because there was 148-fold headroom on the arithmetic.

▸ **Speculative decoding is a direct cash-out of the roofline gap.** It exists only because the arithmetic units were idle. On a hypothetical machine that was compute-bound during decode, it would buy you nothing at all.

> **Analogy.** A senior editor and a junior writer. The junior drafts a paragraph; the editor reads it — reading a paragraph and reading a sentence take roughly the same trip to the desk — and accepts everything up to the first word they would have chosen differently. Then they write *that* word themselves and hand it back. **The published text is exactly what the editor would have written alone**, because every retained word is one they endorsed and every disagreement was resolved in their favour. The junior's only contribution is speed.

> **Where this came from.** Speculative decoding was published twice, months apart, by two teams that did not know of each other: **Yaniv Leviathan, Matan Kalman and Yossi Matias at Google Research**, and **Charlie Chen and colleagues at DeepMind**, both in 2022–2023. (The book is full of these; see the SVD in Ch. 1.) The name is borrowed from **speculative execution** in CPU design, where a processor guesses which way a branch will go and executes ahead, discarding the work if it guessed wrong — a technique from the 1990s that is the reason modern chips are fast, and also the reason for the Spectre and Meltdown vulnerabilities of 2018. The accept/reject machinery itself is much older: **rejection sampling** is due to **John von Neumann**, described in a 1951 note on techniques for generating random digits. A method invented to make random numbers on a machine with no random-number generator now doubles the speed of every large language model in production.

### The speedup

With per-token acceptance rate $`\alpha`$, the expected number of tokens accepted per round is
▸ $$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

**Numbers:** $`\alpha=0.8`$, $`\gamma=4`$: $`\frac{1-0.8^5}{0.2} = \frac{0.672}{0.2}=3.36`$ tokens per target forward pass. Net speedup after draft cost is typically **2–3×**.

**Variants:** **Medusa** (extra prediction heads on the target model — no separate draft model); **EAGLE** (draft in feature space, higher acceptance); **Lookahead decoding** (Jacobi iteration, no draft model); **prompt lookup** (copy $`n`$-grams from the prompt — near-free and excellent for summarization and code editing).

#### Reading the expected-tokens formula

$$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

**The symbols.** $`\alpha`$ (alpha) is the **acceptance rate** — the probability that any one drafted token survives the check. $`\gamma`$ (gamma) is how many tokens you draft per round. $`\mathbb{E}[\cdot]`$ is the expectation: the long-run average (§0.5).

**Where the formula comes from, in three lines.** Treat each token's acceptance as an independent coin flip landing heads with probability $`\alpha`$. Then:

- The first draft token survives with probability $`\alpha`$.
- The first *two* both survive with probability $`\alpha^2`$ — you need two heads in a row, and one rejection ends the round.
- In general, at least $`i`$ tokens survive with probability $`\alpha^i`$.

Add these up, and include the **one token you always get for free** (on a rejection you emit the residual sample; if everything is accepted, the target's own last position hands you a bonus token):

$$\mathbb{E}[\text{tokens}] = \underbrace{1}_{\text{always}} + \alpha + \alpha^2 + \dots + \alpha^\gamma \;=\; \sum_{i=0}^{\gamma}\alpha^i \;=\; \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

▸ **That last step is just the finite geometric series** — the same identity as $`1+\tfrac12+\tfrac14+\dots`$. There is no deep content in the closed form; it is the sum of a run of heads, which is why the answer is the classic "expected length of a winning streak."

**Now watch what happens as you draft more.** With $`\alpha = 0.8`$:

| $`\gamma`$ (tokens drafted) | Expected tokens per round | Gain over previous |
|---|---|---|
| 1 | 1.80 | — |
| 2 | 2.44 | +0.64 |
| 4 | 3.36 | +0.51 (over $`\gamma{=}3`$) |
| 6 | 3.95 | +0.26 |
| 8 | 4.33 | +0.13 |
| $`\infty`$ | **5.00** | — |

▸ **There is a hard ceiling at $`\dfrac{1}{1-\alpha}`$, and you approach it geometrically.** At $`\alpha=0.8`$ that ceiling is 5 tokens per round no matter how much you draft, because a single rejection ends the round and rejections arrive on average every fifth token. **Drafting 50 tokens ahead is nearly pointless** — you would spend 50 draft-model passes to collect, on average, 5 useful tokens. This is why real systems use $`\gamma`$ between about 3 and 7 and stop.

**Why the *net* speedup is 2–3× and not 3.36×.** The draft model is not free. Say it costs one tenth of a target forward pass. Then a round costs $`1 + \gamma\times 0.1`$ target-passes and yields $`\mathbb{E}[\text{tokens}]`$ tokens:

$$\text{speedup} = \frac{\mathbb{E}[\text{tokens}]}{1+0.1\gamma} = \frac{3.36}{1.4} \approx 2.4\times$$

Sweep $`\gamma`$ and the answer is 2.40 at $`\gamma=4`$, 2.47 at $`\gamma=6`$, 2.45 at $`\gamma=7`$ — **a very flat optimum**, which is a mercy: you cannot really tune this wrong.

**What actually determines $`\alpha`$**, and why the variants exist:

| Lever | Effect on $`\alpha`$ |
|---|---|
| Bigger / better-matched draft model | Higher $`\alpha`$, but a more expensive draft — a direct trade |
| Draft trained by distillation *from the target* (§17.6) | Higher $`\alpha`$ for the same size, because you are matching the exact distribution being checked |
| Predictable text (code, boilerplate, quoting the prompt) | Very high $`\alpha`$ — this is why **prompt lookup**, which just copies $`n`$-grams from the input, works so well on summarization and code editing |
| Creative or high-temperature sampling | Lower $`\alpha`$ — the two models disagree more when both are being adventurous |

▸ **The one-sentence summary: acceptance rate is the only number that matters, and every variant in the list above is a different way to buy more of it.** Medusa buys it by predicting from the target model's own hidden state (so the "draft" is already almost the target); EAGLE buys it by drafting in feature space instead of token space; prompt lookup buys it for free by noticing that a lot of generated text is verbatim copied from the input.

---

## 17.4 Quantization

### The one-line idea

Store weights and activations in fewer bits. Since inference is memory-bandwidth-bound, halving the bits nearly halves the latency.

### Basic uniform quantization

▸ $$x_q = \mathrm{round}\left(\frac{x}{s}\right)+z,\qquad \hat x = s(x_q - z)$$

with scale $`s = \frac{\max-\min}{2^b-1}`$ (asymmetric) or $`s=\frac{\max|x|}{2^{b-1}-1}`$ (symmetric).

**Granularity** matters more than the formula: per-tensor (coarse, fast) → per-channel → per-group (e.g. 128 weights share a scale). **Group size 128 is the modern default** and recovers most of the accuracy.

#### Reading the quantization formula in plain English

$$x_q = \mathrm{round}\left(\frac{x}{s}\right)+z,\qquad \hat x = s(x_q - z)$$

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $`x`$ | "x" | The real number you actually have — a weight, say $`0.3142`$ |
| $`s`$ | "s", the **scale** | How much one integer step is worth. The size of one tick mark |
| $`z`$ | "z", the **zero-point** | An integer offset, so that real zero lands exactly on an integer |
| $`x_q`$ | "x-q" | The small integer you **store instead of** $`x`$ |
| $`\hat x`$ | "x-hat" | What you get back when you decode $`x_q`$. The hat means *estimate* (§0.6) |
| $`b`$ | "b" | Bits per stored number. $`b`$ bits give $`2^b`$ distinct values |

Read the first equation aloud: *"divide by the step size, round to the nearest whole number, add the offset."* The second: *"subtract the offset, multiply by the step size."* They are inverses, apart from the rounding — **and the rounding is where all the loss lives**.

> **Analogy.** A kitchen scale that only reads in whole grams. Your ingredient weighs 3.7 g; the display says 4. The scale's *resolution* — 1 g — is the scale factor $`s`$. Coarsen it to a scale that reads in 5 g steps and you can weigh a sack of flour without overflowing, but a teaspoon of yeast now reads as either 0 or 5. **Every quantization decision is this same trade: the range you can cover versus the detail you can see, with the number of tick marks fixed.**
>
> The zero-point $`z`$ is the tare button. Without it, an integer grid running $`0,1,2,\dots`$ cannot represent negative numbers at all; $`z`$ slides the grid so that some integer means exactly zero.

**Now put numbers in.** Suppose a group of weights ranges from $`-0.6`$ to $`+0.8`$, and $`b=4`$, so you have $`2^4 = 16`$ integers to work with.

**Symmetric** ($`s = \max\lvert x\rvert / (2^{b-1}-1)`$, $`z=0`$): the grid must cover $`\pm 0.8`$ using the range $`-7\ldots+7`$, so

$$s = \frac{0.8}{7} = 0.1143$$

Quantize $`x = 0.31`$: $`\mathrm{round}(0.31/0.1143) = \mathrm{round}(2.71) = 3`$, and decoding gives $`\hat x = 3 \times 0.1143 = 0.343`$. **Error: 0.033.**

**Asymmetric** ($`s = (\max-\min)/(2^b-1)`$): the grid covers exactly $`[-0.6, 0.8]`$ using all 16 integers, so

$$s = \frac{0.8 - (-0.6)}{15} = \frac{1.4}{15} = 0.0933,\qquad z = \mathrm{round}\!\left(\frac{0.6}{0.0933}\right) = 6$$

Quantize $`x=0.31`$: $`x_q = \mathrm{round}(3.32) + 6 = 9`$, decoding to $`\hat x = 0.0933\times(9-6) = 0.280`$. **Error: 0.030**, with a 18% finer step size.

▸ **Asymmetric wins here because the distribution isn't centred**, and symmetric quantization wastes the unused stretch from $`-0.8`$ to $`-0.6`$. Symmetric wins on *speed*, because $`z=0`$ removes an addition from the innermost loop of the matrix multiply. This is the entire debate, and it is decided by hardware, not by mathematics.

**Why granularity beats everything.** The formula is the same at every granularity; what changes is **how many numbers share one $`s`$**. Take a $`4096\times4096`$ weight matrix — 16.8 million weights:

| Granularity | Scales stored | Weights per scale | Effective bits/weight at $`b=4`$ |
|---|---|---|---|
| Per-tensor | 1 | 16,777,216 | 4.000 |
| Per-channel | 4,096 | 4,096 | 4.004 |
| **Per-group of 128** | **131,072** | **128** | **4.125** |

The last column is the honest accounting: each fp16 scale costs 16 bits, spread over 128 weights, which is $`16/128 = 0.125`$ extra bits each.

▸ **You buy an enormous accuracy improvement for 3% more storage.** A single scale for 16.8 million weights must be set by the largest weight in the entire matrix, so one freak value flattens everything else. A scale per 128 weights only has to cope with the largest value in its own small neighbourhood. **Group size 128 is the modern default because it is the point where the accuracy curve has flattened and the overhead has not yet started to hurt** — and that is an empirical finding, not a derivation.

> **Where this came from.** Quantization is not a deep-learning invention; it is the foundation of digital audio and telephony. **Alec Reeves** patented pulse-code modulation in the late 1930s — the idea of representing a continuous waveform by rounding it to a grid of levels many times a second — and **W. R. Bennett** at Bell Labs published the analysis of "quantization noise" in 1948, the same year and the same building as Shannon's information theory paper. The vocabulary you are reading here (scale, zero-point, quantization error, dynamic range) was worked out to make telephone calls fit down a copper wire, and arrived in machine learning essentially unmodified.

### The activation-outlier problem

▸ Weights quantize easily. **Activations do not**, because transformers develop *systematic outlier channels* — a handful of dimensions with magnitudes 10–100× the rest, appearing consistently in the same channels across all tokens. A per-tensor scale set by those outliers crushes everything else to a few levels.

**The fixes:**
- **LLM.int8():** decompose — keep outlier channels in fp16, quantize the rest to int8. Exact where it matters.
- **SmoothQuant:** migrate the difficulty from activations to weights with a per-channel rescaling $`X\mathrm{diag}(s)^{-1}\cdot\mathrm{diag}(s)W`$, which leaves the product unchanged but makes both factors easier to quantize.
- **AWQ (Activation-aware Weight Quantization):** protect the ~1% of weight channels with the largest *activation* magnitudes by scaling them up before quantization. Fast, calibration-light, very widely used.
- **GPTQ:** quantize weights column by column, using the inverse Hessian ($`H = 2XX^\top`$, from a calibration set) to compensate remaining columns for the error already introduced. Solves a layerwise reconstruction problem $`\min_{\widehat W}\|WX-\widehat WX\|_F^2`$. Excellent at 4-bit and even 3-bit.

#### The outlier problem, with numbers

The claim "weights quantize easily, activations do not" sounds like folklore until you put magnitudes on it.

**Weights** in a trained transformer are roughly bell-shaped and tightly clustered — nearly all of them within a few standard deviations of zero, no systematic freaks. That is a distribution a uniform grid handles gracefully.

**Activations** are not. Transformers reliably develop a handful of **outlier channels** — specific dimensions of the residual stream whose magnitude runs 10–100× everything else, *in the same dimensions for every token*. Suppose typical activations sit near $`1`$ and the outlier channel reaches $`100`$. Quantize the whole tensor to int8 with one symmetric scale:

$$s = \frac{\max\lvert x\rvert}{127} = \frac{100}{127} = 0.787$$

Now watch what happens to ordinary values:

| True activation | $`\mathrm{round}(x/s)`$ | Recovered $`\hat x`$ | Relative error |
|---|---|---|---|
| 100.0 | 127 | 99.9 | 0.1% |
| 1.0 | 1 | 0.79 | **21%** |
| 0.5 | 1 | 0.79 | **58%** |
| 0.3 | 0 | 0.00 | **100% — annihilated** |

▸ **You have 256 available levels and 99.9% of your numbers are using three of them.** That is the outlier problem stated exactly: one channel sets the scale for the whole tensor, and everything else is crushed into a handful of integers. The remaining 253 levels sit unused in a range that almost nothing occupies.

**Now each fix, in one sentence apiece:**

**LLM.int8() — separate the two populations.** Detect the outlier channels (typically well under 1% of dimensions), keep *those* in fp16, quantize the rest to int8, and add the two results. You pay for a small mixed-precision matmul and get exactness precisely where the damage was.

**SmoothQuant — move the difficulty into the weights.** The algebra is one line and worth reading slowly. For any positive vector $`s`$:

$$\big(X\,\mathrm{diag}(s)^{-1}\big)\big(\mathrm{diag}(s)\,W\big) \;=\; X\,\mathrm{diag}(s)^{-1}\mathrm{diag}(s)\,W \;=\; XW$$

The two $`\mathrm{diag}(s)`$ factors cancel, so **the product is untouched — the network computes exactly the same function.** But the two factors going *into* the product have changed: divide activation channel $`j`$ by 10, multiply weight row $`j`$ by 10.

> **Analogy.** Two people carrying a sofa, one at each end. The sofa's weight is fixed, but you can slide it so one person takes 90% and the other 10%. Neither the sofa nor the distance changes. SmoothQuant slides the "difficulty" along the sofa until **both ends are carryable**: an activation channel of magnitude 100 next to a weight row of magnitude 0.1 becomes an activation of 10 next to a weight of 1, and now a single scale serves both tensors well.

**AWQ (Activation-aware Weight Quantization) — protect the weights that meet big activations.** The insight in the name: what matters is not how large a *weight* is, it is how large the **product** $`w\cdot x`$ is. A tiny weight multiplied by an outlier activation contributes more to the output than a large weight multiplied by nothing. So AWQ picks the roughly 1% of weight channels whose *input activations* are largest and scales them up before quantizing, giving them proportionally finer resolution. **It never looks at the weight magnitudes at all** — that is the "activation-aware" part, and it is why the method needs only a small calibration set and no gradients.

#### Unpacking GPTQ's objective

$$\min_{\widehat W}\ \lVert WX-\widehat WX\rVert_F^2$$

**Read aloud:** *"choose the quantized weight matrix $`\widehat W`$ that minimizes the total squared difference between what this layer used to output and what it outputs now."*

- $`W`$ is the original weight matrix; $`\widehat W`$ (W-hat) is the quantized replacement.
- $`X`$ is a batch of **real activations**, harvested by running a small calibration set — a few hundred sequences — through the model.
- $`\lVert\cdot\rVert_F`$ is the Frobenius norm (Ch. 1 §1.1.4): flatten the matrix into one long vector and take its length. Squaring it means "sum of squares of every entry."

▸ **The crucial move is that $`X`$ appears at all.** The naive approach quantizes $`W`$ to be close to $`W`$. GPTQ quantizes $`W`$ so that **$`WX`$ is close to $`WX`$** — it does not care about weights the data never activates, and it cares enormously about weights that sit in the path of common inputs. Rounding is a budget; this decides where to spend it.

**Why a Hessian shows up.** The objective is quadratic in $`\widehat W`$, and the second derivative of $`\lVert(W-\widehat W)X\rVert_F^2`$ with respect to the weights is $`2XX^\top`$. That is the whole derivation. **This is not the Hessian of the training loss** — it is the curvature of *this layer's reconstruction error*, which is a far smaller and completely tractable object: for a layer of width 4096, it is a $`4096\times4096`$ matrix you can actually invert.

**What the algorithm does with it.** Quantize the columns one at a time, left to right. Each time you round a column, you have introduced a known error — so use the inverse Hessian to compute how to nudge the *not-yet-quantized* columns to cancel it out.

> **Analogy.** Tuning a piano where all the strings pull on the same frame. Tighten one string to pitch and every other string goes slightly flat. A naive tuner works left to right and finishes with an instrument that is wrong everywhere. A good tuner knows how much each string affects the others and compensates as they go. **The inverse Hessian is precisely the table of "how much does moving this one pull on that one."**

▸ **This is why GPTQ survives at 3 and 4 bits when round-to-nearest falls apart.** Round-to-nearest treats each weight as an independent decision. GPTQ treats them as a sequence of decisions where every one can be *repaired by the ones that follow* — and the last column, with nothing left to compensate it, is the only one that takes its error unmitigated.

> **Where this came from.** The lineage is unusually clean and unusually old. **Optimal Brain Damage** — Yann LeCun, John Denker and Sara Solla, 1989 — proposed removing weights by second-derivative saliency rather than by magnitude, arguing that the right question is not "which weight is small?" but "which weight, if removed, would change the loss least?" **Optimal Brain Surgeon** (Babak Hassibi and David Stork, 1993) added the crucial step: after deleting a weight, use the inverse Hessian to *update all the survivors* to compensate. That is exactly GPTQ's inner loop, thirty years early. **Elias Frantar and Dan Alistarh at IST Austria** revived the framework in 2022, first as Optimal Brain Quantization and then as GPTQ, with the engineering needed to run it on billion-parameter matrices; the same group produced **SparseGPT**. The 1989 and 1993 papers were written about networks with a few hundred weights. **The outlier-channel phenomenon itself** was characterized by **Tim Dettmers** and colleagues in the LLM.int8() work of 2022, which reported that these extreme channels emerge fairly abruptly as models grow past roughly 6–7 billion parameters — before that scale, naive int8 quantization simply works, which is why the problem took the field by surprise.

### QAT and the straight-through estimator

Quantization-aware training simulates quantization in the forward pass. But $`\mathrm{round}(\cdot)`$ has zero gradient almost everywhere.

▸ **Straight-through estimator (STE):** pretend the rounding is the identity in the backward pass.
$$\frac{\partial \hat x}{\partial x} := 1\ \ \text{(within the clipping range; 0 outside)}$$

Biased, and works extremely well. **The STE is a general tool** — it appears again in VQ-VAE (Ch. 19), binary networks, and any discrete bottleneck.

#### Why the straight-through estimator is allowed

First, the abbreviations: **QAT** is **Q**uantization-**A**ware **T**raining (simulate the rounding during training so the network learns to tolerate it), as opposed to **PTQ**, **P**ost-**T**raining **Q**uantization (take a finished model and squash it). **STE** is the **S**traight-**T**hrough **E**stimator.

**The problem, precisely.** $`\mathrm{round}(x)`$ is a staircase. Between the steps it is perfectly flat, so its derivative is exactly $`0`$. At the steps it jumps, so its derivative is undefined (infinite). There is no third region. Feed that into backpropagation and every gradient behind it is multiplied by zero: **the network receives the message "nothing you do to this weight makes any difference," which is locally true and globally catastrophic.**

**The fix, stated as the lie it is.** In the forward pass, round. In the backward pass, pretend you did not:

$$\frac{\partial \hat x}{\partial x} := 1$$

The $`:=`$ is "is *defined* to be" (§0.11) — not derived, not approximated. **Declared.** That is why it is called an *estimator* rather than a derivative.

> **Analogy.** A staircase and the ramp beside it. The staircase is what your feet actually do — discrete, step, step, step. But if someone asks "if I walk 3 metres further, how much higher will I be?", you do not answer "either zero or infinity, depending exactly where you stand." You answer using the **ramp**: the smooth surface that has the same average slope. **The STE walks the stairs and reasons about the ramp.** It is wrong at every individual point and right about the trend, and the trend is the only thing gradient descent was ever using.

**Why the clipping range matters.** The definition says the derivative is 1 *inside* the representable range and 0 outside it. That second half is not a lie at all — it is correct. If a weight has already been clamped to the maximum representable value, pushing it further  changes nothing about the forward pass, and a zero gradient is the honest answer. **The STE is dishonest only about rounding, not about saturation.**

▸ **Take the general lesson, because it recurs.** Any time a model contains a discrete decision — rounding, an $`\arg\max`$, a nearest-neighbour lookup, a hard threshold — the gradient dies there. The standard move throughout deep learning is to **keep the discrete operation in the forward pass and substitute a smooth surrogate in the backward pass.** You will meet exactly this again in VQ-VAE's codebook lookup (Ch. 19 §19.3), in binarized networks, and in the Gumbel-softmax. The technique is called "biased" because the gradient you compute is provably not the gradient of anything you are actually optimizing; it earns its place empirically, and it has earned it many times.

> **Where this came from.** The straight-through trick was described by **Geoffrey Hinton** in his 2012 Coursera lectures on neural networks, as a practical suggestion rather than a theorem. It was written up, named, and studied — alongside several alternatives — by **Yoshua Bengio, Nicholas Léonard and Aaron Courville** in a 2013 paper on propagating gradients through stochastic neurons. The paper is unusually candid that nobody could say why it worked; a decade later the theoretical account is still partial, and the empirical case is overwhelming.

### What actually degrades

| Bits | Typical quality | Note |
|---|---|---|
| bf16 | baseline | |
| int8 (W8A8) | ~lossless | with SmoothQuant |
| **int4 weights (W4A16)** | **~1% perplexity** | GPTQ/AWQ, group 128 — the standard deployment point |
| int3 | noticeable | needs care |
| int2 / ternary | large loss | needs QAT or special architectures |

▸ **The scaling insight:** at a *fixed total bit budget*, a larger model quantized to 4 bits generally beats a smaller model at 16 bits. Quantization is not just compression; it is a better point on the quality-per-byte frontier.

**KV-cache quantization** is separately valuable — Ch. 12 §12.7 showed the cache often exceeds the weights.

#### Reading the degradation table

**The W/A notation first.** **W8A8** means 8-bit **W**eights and 8-bit **A**ctivations — both tensors going into the matmul are integers, so the multiply itself runs on integer hardware. **W4A16** means 4-bit weights but 16-bit activations: the weights are stored small, then *decompressed back to 16 bits on the fly* and multiplied normally.

▸ **W4A16 sounds wasteful and is in fact the standard, because of §17.1.** You are not trying to make the arithmetic cheaper — the arithmetic units were idle anyway. You are trying to move fewer bytes. Weights are what you move; activations for a single token are negligible by comparison. **So you compress the thing that travels and leave the thing that computes alone.** That one sentence explains why the deployment sweet spot is an asymmetric format that looks, on paper, like a half-measure.

**"~1% perplexity" decoded.** Perplexity (Ch. 13) is the exponential of the cross-entropy — loosely, "how many options is the model effectively choosing between at each token." A 1% increase means going from, say, 8.00 to 8.08. **This is roughly the difference a few percent more training data would make, and it is invisible in blind side-by-side comparisons.** You are trading that for a 4× reduction in memory traffic.

**Now the memory arithmetic that makes the choice for you**, for a 70B model (using $`4.125`$ effective bits at group size 128, from the granularity table above):

| Format | Bytes/parameter | 70B model size | Decode floor at 3.35 TB/s | Tokens/s |
|---|---|---|---|---|
| bf16 | 2.000 | 140 GB | 42 ms | 24 |
| int8 | 1.000 | 70 GB | 21 ms | 48 |
| **int4, group 128** | **0.516** | **36 GB** | **11 ms** | **93** |
| int3, group 128 | 0.391 | 27 GB | 8 ms | 122 |

▸ **The tokens-per-second column is very nearly proportional to the bytes column**, because decode is bandwidth-bound and nothing else. This is what "quantization buys near-linear speedup" means, and it is why quantization is the highest-leverage single change in inference: it is the only optimization that improves latency, throughput, and the number of GPUs you need to rent, simultaneously and by the same factor.

**Reading the scaling insight, with numbers.** "At a fixed total bit budget, a larger model at 4 bits beats a smaller model at 16 bits" — compare three ways to spend roughly 35 GB:

| Option | Parameters | Bits each | Memory |
|---|---|---|---|
| A | 17B | 16 | 34 GB |
| B | 34B | 8 | 34 GB |
| **C** | **70B** | **4** | **36 GB** |

Option C wins on essentially every benchmark, and it is not close. **The reason is that capability scales with parameter count (Ch. 15) far more steeply than it degrades with bit width.** Four bits per weight is enough resolution for a weight to do its job; there is no comparable substitute for simply having four times as many of them.

▸ **So quantization is not really compression — it is a *reallocation*.** The right way to hold it: given a fixed number of gigabytes, your job is to buy the largest model those gigabytes can hold at the coarsest precision that still works. The empirical answer to "coarsest that still works" has settled at **4 bits**, and the sharp drop-off below 3 bits in the table above is why.

**And the reminder about the KV cache.** All of the above concerns *weights*. But Ch. 12 §12.7 showed that at long context and reasonable batch size, the **KV cache can exceed the weights entirely** — it grows with (batch × sequence length × layers), while the weights are fixed. Quantizing a 36 GB model to 4 bits and then filling the remaining memory with an fp16 KV cache is a common and self-defeating configuration. **Quantize both, or you have optimized the smaller of the two numbers.**

---

## 17.5 Pruning

**Unstructured (magnitude) pruning:** zero the smallest-magnitude weights. Can reach 90%+ sparsity with retraining — but **gives no speedup on standard hardware**, because sparse matmuls with irregular patterns are slower than dense ones.

**Structured pruning:** remove whole heads, channels, or layers. Real speedup; larger quality cost.

**Semi-structured (2:4):** exactly 2 of every 4 consecutive weights are zero. ▸ Supported natively by NVIDIA sparse tensor cores for **~2× matmul throughput**. The practical sweet spot.

**SparseGPT / Wanda:** one-shot pruning with no retraining. Wanda's criterion is elegantly simple — prune by $`|W_{ij}|\cdot\|X_j\|_2`$, i.e. weight magnitude times the input activation norm. Reaches 50% sparsity on large LLMs with minimal loss and no gradient computation at all.

▸ **The empirical regularity:** larger models are more prunable. Redundancy grows with scale, which is consistent with the lottery-ticket and superposition pictures (Ch. 31, 32).

#### Why 90% of your weights can be zero and nothing gets faster

This is the most counterintuitive sentence in the chapter, and it is entirely about hardware.

A GPU multiplies matrices with a fixed, rigid pipeline: fetch a tile of numbers, multiply, accumulate, move on. It does not ask what the numbers are. **A zero costs exactly as much to multiply as a 0.7.** So setting 90% of your weights to zero and storing them as zeros changes precisely nothing — same bytes moved, same operations performed.

To actually benefit you must *skip* the zeros, which means storing a list of where the survivors are. And that is where it falls apart:

| | Dense | Unstructured sparse (90%) |
|---|---|---|
| What you store | the values | the values **plus their coordinates** |
| Memory access | contiguous, predictable, prefetchable | scattered — every value needs a lookup |
| Hardware path | tensor cores, fully utilized | general-purpose units, poorly utilized |
| Typical outcome | baseline | **often slower than dense** |

> **Analogy.** A conveyor belt in a factory picks up every item in a fixed rhythm, whether or not the slot is full. Removing 90% of the items does not make the belt run faster — the belt has one speed. To go faster you would have to redesign the belt to skip empty slots, and the machinery for detecting and skipping costs more than the time it saves.

▸ **So unstructured pruning is a *storage* result, not a *speed* result** — and since decode is bandwidth-bound, storage that you still have to move is worth nothing. This is why the field converged on structure.

**2:4 semi-structured sparsity, concretely.** Take every run of four consecutive weights and force exactly two of them to zero. Store the two survivors plus a 2-bit index for each saying which of the four slots it came from:

$$4\ \text{weights}\ \longrightarrow\ 2\ \text{values} + 2\times2\ \text{bits of metadata}$$

**The pattern is now perfectly regular**, so the hardware can be built to exploit it — which NVIDIA did, in the Ampere generation (2020), with sparse tensor cores that deliver about 2× matmul throughput on 2:4 data. **The compromise is exactly the point:** unstructured sparsity is maximally flexible and unusable; fully structured sparsity (deleting whole heads or layers) is maximally usable and expensive in quality; 2:4 is regular enough to build silicon for and fine-grained enough that the network barely notices.

#### Wanda's criterion, decoded

$$\text{score}_{ij} = \lvert W_{ij}\rvert \cdot \lVert X_j\rVert_2$$

**Read aloud:** *"the importance of the weight in row $`i`$, column $`j`$ is its absolute value times the length of the vector of activations that arrive on input $`j`$."* The $`\lVert X_j\rVert_2`$ is the $`\ell_2`$ norm (Ch. 1 §1.1.4) of input feature $`j`$ measured across a calibration batch — "how loud is this input channel, typically?"

**Why the second factor changes everything.** Compare two weights:

| Weight | $`\lvert W_{ij}\rvert`$ | $`\lVert X_j\rVert_2`$ | Score | Magnitude pruning says | Wanda says |
|---|---|---|---|---|---|
| A | 0.50 | 0.1 | 0.05 | **keep** (it's big) | prune |
| B | 0.05 | 20.0 | 1.00 | **prune** (it's small) | keep |

Weight B is ten times smaller and contributes twenty times more to the output, because the input it multiplies is loud. **Magnitude pruning gets this exactly backwards.**

▸ **Notice this is the same insight as AWQ's**, arrived at independently for a different task: *what matters is the product, not the factor you happen to be storing.* Once you have seen it twice you will see it everywhere in this chapter — quantization, pruning, and distillation are all, in the end, arguments about where in the network the output is actually sensitive.

And Wanda needs **no gradients, no retraining, and no Hessian inversion** — one calibration pass to collect the norms, one sort, done. That it reaches 50% sparsity on large models with minimal loss, while being this simple, is the reason it is worth knowing.

> **Where this came from.** Pruning is one of the oldest ideas in neural networks: **Optimal Brain Damage** (LeCun, Denker and Solla) appeared in 1989, and its whole motivation was that the networks of the era overfit badly and pruning was a form of regularization — *making things smaller to make them better*, not to make them cheaper. The modern revival owes most to the **Lottery Ticket Hypothesis** of **Jonathan Frankle and Michael Carbin** (2018–2019), which showed that inside a large trained network there exists a small subnetwork that, *rewound to its original random initialization*, trains to the same accuracy alone. That result reframed pruning from "removing fat" to "finding the part that was doing the work all along," and it is the subject of Ch. 31.

---

## 17.6 Knowledge distillation

### The one-line idea

Train a small model to match a large model's full output *distribution*, not just its argmax — because the relative probabilities of the wrong answers carry most of the information.

### The analogy

A student who is told "the answer is B" learns one bit. A student told "it's B, but C was tempting and D was absurd" learns the shape of the problem. Hinton called the latter **dark knowledge**.

### The loss

▸ $$\mathcal{L} = \alpha\,\tau^2\,\mathrm{KL}\!\left(p_T^{(\tau)}\,\|\,p_S^{(\tau)}\right) + (1-\alpha)\,\mathrm{CE}(y, p_S)$$

with $`p^{(\tau)}=\mathrm{softmax}(z/\tau)`$, $`\tau\approx2`$–$`10`$.

**Why the $`\tau^2`$?** The gradient of the soft-target term scales as $`1/\tau^2`$ (differentiate $`\mathrm{softmax}(z/\tau)`$ w.r.t. $`z`$ and note the extra $`1/\tau`$ in the chain, appearing twice through the KL). Multiplying by $`\tau^2`$ keeps the relative weight of the two terms constant as you vary $`\tau`$. **This is a favourite interview follow-up.**

**Why high $`\tau`$ helps:** at $`\tau=1`$ a confident teacher's distribution is nearly one-hot and carries no more information than the label. Raising $`\tau`$ exposes the ratios among the small probabilities.

#### Reading the distillation loss

$$\mathcal{L} = \alpha\,\tau^2\,\mathrm{KL}\!\left(p_T^{(\tau)}\,\|\,p_S^{(\tau)}\right) + (1-\alpha)\,\mathrm{CE}(y, p_S)$$

**Every symbol:**

| Symbol | Read aloud | What it is |
|---|---|---|
| $`\mathcal{L}`$ | "script L" | The total loss being minimized |
| $`\alpha`$ | "alpha" | A mixing weight in $`[0,1]`$: how much you trust the teacher versus the label |
| $`\tau`$ | "tau" | **Temperature.** Divide the logits by it before softmax |
| $`\mathrm{KL}(a\|b)`$ | "KL of a from b" | Kullback–Leibler divergence (Ch. 1 §1.4) — how much you lose by believing $`b`$ when the truth is $`a`$ |
| $`p_T^{(\tau)}`$ | "p-T at tau" | The **T**eacher's softened probability distribution |
| $`p_S^{(\tau)}`$ | "p-S at tau" | The **S**tudent's, softened the same way |
| $`\mathrm{CE}(y, p_S)`$ | "cross-entropy" | The ordinary loss against the true label $`y`$ |

**Read the whole thing aloud:** *"Loss equals: some fraction of 'match the teacher's full opinion', plus the remaining fraction of 'get the right answer'."* It is a weighted average of two teachers — one that is right by definition (the label), and one that is merely excellent but far more informative.

**Now put numbers in, because "soft targets carry more information" is only convincing arithmetically.** Take four classes and a teacher whose logits are $`z = (8,\ 6,\ 2,\ 1)`$.

| Temperature | Teacher's probabilities | What the student sees |
|---|---|---|
| $`\tau = 1`$ | $`(0.878,\ 0.119,\ 0.0022,\ 0.0008)`$ | "It's class 1." The other three round to nothing |
| $`\tau = 4`$ | $`(0.499,\ 0.303,\ 0.111,\ 0.087)`$ | "It's class 1, but class 2 was a serious contender, class 3 plausible, class 4 nearly ruled out" |

At $`\tau = 1`$, the information about classes 3 and 4 is technically present — their ratio is a perfectly good $`2.7`$ — but they contribute about two thousandths of the loss, so **the gradient effectively ignores them**. At $`\tau=4`$ they carry a fifth of the mass and materially steer training.

▸ **This is what "dark knowledge" means, and the name is well chosen: the information was always there, it was just too faint to see.** Temperature is a telescope. A hard label says "the answer is B" — that is $`\log_2 4 = 2`$ bits. The soft distribution says "B, with C nearby and D absurd," which encodes something about the *geometry of the problem*: which classes are confusable, and therefore what features actually distinguish them. That is why a student trained on soft targets often beats an identical student trained on the ground-truth labels alone.

> **Analogy.** Two ways to be taught for a multiple-choice exam. Teacher one hands back your paper marked "the answer was B." Teacher two says "B — and I can see why you picked C, they differ only in the second clause; D was never in the running." **The second teacher is transmitting the structure of the subject, not just the answer key.** After a hundred questions the first student has memorized a hundred answers and the second has learned the subject.

#### Why the $`\tau^2`$, in slightly more detail

The book's explanation is exact; here is the same argument slowly.

**One factor of $`1/\tau`$ comes from the chain rule.** The student's softened distribution is $`\mathrm{softmax}(z_S/\tau)`$. Differentiating with respect to the *raw* logits $`z_S`$ means differentiating through the division by $`\tau`$, which drops a $`1/\tau`$ out front. Mechanical.

**The second factor of $`1/\tau`$ comes from the probabilities themselves.** At large $`\tau`$ every softened probability approaches uniform, $`1/K`$, and expanding to first order gives roughly

$$p_i^{(\tau)} \approx \frac{1}{K} + \frac{z_i - \bar z}{K\tau}$$

so the *differences* between teacher and student probabilities — which is what the gradient is proportional to — themselves shrink like $`1/\tau`$.

▸ **Multiply the two and the soft-target gradient decays as $`1/\tau^2`$.** Without the correction, raising the temperature from 2 to 10 would quietly shrink the teacher's influence 25-fold, and you would find yourself re-tuning $`\alpha`$ every time you touched $`\tau`$. **The $`\tau^2`$ makes the two hyperparameters independent** — that is its entire job, and it is a very good example of a factor that looks arbitrary until you ask what it is protecting.

> **Where this came from.** The idea predates the famous paper by nearly a decade: **Cristian Bucilă, Rich Caruana and Alexandru Niculescu-Mizil** published *Model Compression* in 2006, training a small neural network to imitate a large ensemble by labelling a mass of synthetic data with the ensemble's predictions. Their motivation was deployment on the modest hardware of the time. **Geoffrey Hinton, Oriol Vinyals and Jeff Dean** generalized it in 2014–2015 with the temperature machinery above, coining "dark knowledge" and framing the analogy that opens their paper: many insects have a **larval form** optimized for extracting nutrients from the environment and a completely different **adult form** optimized for travel and reproduction — and machine learning has been using the same organism for both jobs. Training is the larva; deployment is the butterfly. Their paper appeared as a **workshop** contribution at NeurIPS 2014 rather than in the main proceedings.
>
> The most striking experiment in it is easy to miss: they distilled a digit classifier onto a student whose training data **omitted every single example of one digit**, and the student still recognized that digit most of the time. It had never seen one. Everything it knew about that class arrived through the shadows the class cast on the teacher's probabilities for the other nine.

### Variants

- **Feature/hint distillation:** match intermediate activations (FitNets), attention maps, or relations between examples.
- **Sequence-level KD:** for generative models, train the student on the teacher's *generated sequences* — usually more effective than token-level KL, because it fixes exposure bias too.
- **On-policy / GKD:** compute the KL on the *student's own* samples, which fixes the train/inference distribution mismatch.
- **Self-distillation:** teacher and student are the same size; still improves accuracy, which is evidence that the effect is regularization rather than compression.
- **Reverse KL** $`\mathrm{KL}(p_S\|p_T)`$ is mode-seeking (Ch. 1 §1.4.1) and often preferable for generation, where you want the student to be sharp rather than to smear over everything the teacher considers possible.

---

## 17.7 Parameter-efficient fine-tuning

### LoRA

### The one-line idea

Freeze the pretrained weights and learn a low-rank correction, because the *update* needed for a downstream task turns out to have far lower rank than the weight matrix itself.

▸ $$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$

**Initialization:** $`A\sim\mathcal{N}(0,\sigma^2)`$, $`B=0`$. So $`\Delta W = 0`$ at the start and training begins exactly at the pretrained model — the same zero-init-residual idea as Ch. 6, Ch. 8, and AdaLN-Zero.

**Parameter count:** $`r(d+k)`$ instead of $`dk`$. For $`d=k=4096`$, $`r=8`$: $`65{,}536`$ vs $`16.8`$M — a **256× reduction**.

▸ **Why it works:** the intrinsic dimension of a fine-tuning task is small; Aghajanyan et al. showed you can fine-tune to within 90% of full performance in a few hundred random dimensions. LoRA exploits this directly.

#### Reading the LoRA equation

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA$$

**Every symbol, with shapes** — and shapes are the whole story here:

| Symbol | Shape | What it is |
|---|---|---|
| $`W_0`$ | $`d\times k`$ | The **frozen** pretrained weight matrix. Never updated. Read-only |
| $`\Delta W`$ | $`d\times k`$ | "Delta W" — the *change* you want to make (§0.4) |
| $`B`$ | $`d\times r`$ | A tall thin matrix. Trainable |
| $`A`$ | $`r\times k`$ | A short wide matrix. Trainable |
| $`r`$ | scalar | The **rank** — the width of the bottleneck. Typically 8–64 |
| $`\alpha`$ | scalar | A scaling constant. *Not* the distillation $`\alpha`$; same letter, unrelated job |

**Shape check** (Ch. 0 §0.8): $`(d\times r)(r\times k) = d\times k`$. The inner $`r`$'s match and vanish; $`d`$ and $`k`$ survive. So $`BA`$ has exactly the same shape as $`W_0`$ and can be added to it. ✓

▸ **The rank $`r`$ is a bottleneck the update must squeeze through**, and everything about LoRA follows from that. From Ch. 1 §1.1.3, a matrix of rank $`r`$ is a sum of $`r`$ rank-one pieces — $`r`$ simple patterns, each scaled. **So LoRA says: whatever fine-tuning needs to do to this layer, it can be written as eight patterns.** The claim is empirical and it holds remarkably well.

> **Analogy.** A printed map and a sheet of tracing paper laid over it. The map ($`W_0`$) is expensive to produce and you never redraw it. Everything you learn about the new city — one-way streets, closures, your favourite café — goes onto the tracing paper, which costs nothing and can be swapped for a different sheet when you fly somewhere else. And when you finally want a single clean map, you **merge**: press the two together and print once. That is the "zero inference cost" bullet — after training you compute $`W_0 + \frac{\alpha}{r}BA`$ once, store the result, and the model is indistinguishable from a fully fine-tuned one at runtime.

**Now the parameter count, worked.** With $`d = k = 4096`$ and $`r = 8`$:

$$\underbrace{4096\times 8}_{B} + \underbrace{8\times 4096}_{A} = 32{,}768 + 32{,}768 = 65{,}536 \quad\text{versus}\quad 4096\times4096 = 16{,}777{,}216$$

$$\frac{16{,}777{,}216}{65{,}536} = 256\times$$

The general formula: $`\dfrac{dk}{r(d+k)}`$, which for a square matrix is $`\dfrac{d}{2r}`$. **Halve the rank, double the saving.**

**Why $`B=0`$ and $`A`$ random — and why not both zero.** At initialization $`\Delta W = B A = 0\cdot A = 0`$, so the model starts *exactly* as the pretrained model: no shock, no degradation on step one. Same idea as zero-initialized residual branches in Ch. 6 and 8.

But why not set both to zero? Because the gradients would be

$$\frac{\partial\mathcal{L}}{\partial A} = B^\top\frac{\partial\mathcal{L}}{\partial \Delta W}, \qquad \frac{\partial\mathcal{L}}{\partial B} = \frac{\partial\mathcal{L}}{\partial \Delta W}A^\top$$

▸ **With both at zero, both gradients are zero — permanently.** The pair sits at a perfect saddle and nothing ever moves. Setting one of them random breaks the symmetry: $`B`$ receives a nonzero gradient immediately (because $`A^\top \ne 0`$), lifts off zero, and only then does $`A`$ start to learn. **You need exactly one of the two to be nonzero, and it must be the one that is *not* multiplied last.** This is the same symmetry-breaking argument as "don't initialize a neural network to all zeros," in a two-factor disguise.

**What $`\alpha/r`$ is doing.** Without it, doubling the rank would roughly double the size of the update, and every rank change would force you to re-tune the learning rate. Dividing by $`r`$ is meant to hold the effective magnitude fixed so that $`r`$ becomes a pure capacity knob.

▸ **But $`r`$ is arguably the wrong divisor**, and this is what **rsLoRA** ("rank-stabilized LoRA") fixes. $`BA`$ is a sum of $`r`$ roughly-independent rank-one terms, and independent random quantities accumulate like $`\sqrt r`$, not $`r`$ — the same $`\sqrt n`$ law as the standard error in Ch. 1 §1.3.1, and for the same reason (partial cancellation). Dividing by $`r`$ therefore **over**-corrects, quietly shrinking high-rank adapters and making $`r=64`$ underperform for reasons that have nothing to do with capacity. Dividing by $`\sqrt r`$ matches the actual growth. If you have ever raised the rank and found the model got *worse*, this is a leading suspect.

**Practical guidance:**
- Apply to **all** linear layers (Q, K, V, O, and the FFN), not just Q and V. This matters more than the rank.
- $`r=8`$–64 is typical; higher $`r`$ helps for tasks requiring new knowledge rather than new style.
- $`\alpha/r`$ is the effective scale; $`\alpha=2r`$ is a common default. Note that with this parameterization, the LR needed varies with $`r`$ — **rsLoRA** uses $`\alpha/\sqrt r`$ to fix this.
- **Zero inference cost:** merge $`W_0 + \frac\alpha r BA`$ into a single matrix after training.
- Many adapters can be hot-swapped on one base model — the basis of multi-tenant serving.

**QLoRA:** base model in **NF4** (a 4-bit format whose levels are the quantiles of a normal distribution — information-theoretically optimal for normally distributed weights), LoRA adapters in bf16, plus double quantization of the scales and paged optimizers. Enables fine-tuning a 65B model on a single 48 GB GPU.

**DoRA:** decompose $`W`$ into magnitude and direction, apply LoRA only to the direction. Closes much of the remaining gap to full fine-tuning.

#### What NF4 actually is, and why quantiles

**NF4** is **N**ormal **F**loat, 4-bit. Like any 4-bit format it has 16 levels; what makes it different is *where those levels are placed*.

A uniform int4 grid spreads its 16 levels evenly across the range, say $`-1.0, -0.867, \dots, +1.0`$. But trained weights are approximately **normally distributed** — densely bunched near zero, thinning out fast toward the extremes. So a uniform grid spends levels in the tails where almost nothing lives, and starves the crowded middle.

**NF4 places its 16 levels at the quantiles of a normal distribution**: the levels are positioned so that each of the 16 bins contains **the same number of weights**.

> **Analogy.** Sorting a school year into 16 groups. You could cut by score — $`0`$–$`6`$, $`6`$–$`12`$, and so on — and end up with two enormous middle groups and several empty ones at the ends. Or you could cut by **rank**: the top sixteenth, the next sixteenth, and so on. Every group is the same size, and each therefore does an equal share of the work of describing the class. **Quantile bins are the second scheme, and for a bell-shaped population they are provably the information-efficient choice.**

▸ **This is the difference between "4 bits" and "4 *useful* bits."** Under a uniform grid, several of the 16 codes are used by a vanishing fraction of weights and contribute almost nothing; under NF4 all 16 do equal work. The format is described as information-theoretically optimal *given the assumption that the weights are normally distributed* — which is the honest statement of the claim, and the assumption is a good but not perfect fit.

**Double quantization**, in one sentence: the group scales are themselves floating-point numbers, and at group size 64 there are a lot of them — so quantize the scales too. It saves roughly a third of a bit per parameter, which sounds negligible until you multiply by 65 billion.

**Paged optimizers**, in one sentence: the same virtual-memory trick as PagedAttention (§17.2), applied to optimizer state, so that a transient memory spike spills to CPU RAM instead of crashing the run.

> **Where this came from.** The intellectual precondition for LoRA was **Armen Aghajanyan and colleagues' 2020 result on intrinsic dimensionality**: they measured how few dimensions a fine-tuning update  needs by constraining it to a *random* low-dimensional subspace, and found a few hundred often sufficed to reach 90% of full performance. If a random subspace works, a learned one should work better — and **LoRA** followed from **Edward Hu and colleagues at Microsoft in 2021**, aimed at the problem of the day, which was adapting GPT-3 without storing a 175-billion-parameter copy per customer. It was published *before* the consumer language-model boom and found an audience nobody planned for: LoRA became a household word among people fine-tuning image-generation models for artistic styles, an application that did not exist when the paper was written. **QLoRA**, with NF4 and the paged optimizer, came from **Tim Dettmers and colleagues at the University of Washington in 2023** — the same researcher behind LLM.int8() and the outlier analysis in §17.4. Its practical effect was to move fine-tuning a 65-billion-parameter model from a cluster to a single desktop GPU, which is the sort of change that alters who gets to do research at all.

### The other PEFT methods

| Method | Mechanism |
|---|---|
| Adapters (Houlsby) | small bottleneck MLPs inserted in each block; adds inference latency |
| Prefix / P-tuning | learn virtual key/value vectors prepended to every layer's attention |
| Prompt tuning | learn soft embeddings prepended to the input only; matches full FT only at very large scale |
| **BitFit** | train only the biases (~0.1% of parameters); surprisingly competitive on small tasks |
| **IA³** | learn per-channel rescaling vectors for K, V, and FFN activations |

▸ **When full fine-tuning still wins:** learning  new knowledge or a new domain/language. PEFT excels at style, format, and task adaptation.

#### The PEFT table, decoded

**PEFT** is **P**arameter-**E**fficient **F**ine-**T**uning: change the model's behaviour while training a tiny fraction of its parameters. The methods differ along one axis that matters more than any other — **whether the change can be folded back into the original weights.**

| Method | Full form / where it acts | Trainable share | Adds inference latency? |
|---|---|---|---|
| LoRA | Low-Rank Adaptation — a correction to each weight matrix | ~0.1–1% | **No** — merges into $`W`$ |
| Adapters | small bottleneck MLPs inserted *between* the existing blocks | ~1–5% | **Yes** —  extra layers, run in sequence |
| Prefix / P-tuning | learned key/value vectors prepended inside every attention layer | ~0.1% | Slightly — they lengthen every sequence |
| Prompt tuning | learned embeddings prepended to the input only | ~0.01% | Slightly, same reason |
| **BitFit** | train only the **bi**as **t**erms | ~0.1% | **No** — biases already exist |
| **IA³** | **I**nfused **A**dapter by **I**nhibiting and **A**mplifying **I**nner **A**ctivations — one learned multiplier per channel for K, V and the FFN | ~0.01% | **No** — folds into the adjacent weights |

▸ **This column is why LoRA won.** Adapters came first and work well, but they insert new layers into the critical path, and §17.1 explained why that is expensive: extra sequential layers mean extra weight fetches, and decode is bandwidth-bound. **LoRA's arithmetic identity — that a low-rank correction can be *added into* the frozen matrix after training — means you pay nothing at serving time.** BitFit and IA³ share this property, which is why they remain interesting despite training even fewer parameters.

**And the multi-tenancy consequence.** Because an adapter is 65,536 numbers rather than 16.8 million, you can hold hundreds of them in memory next to *one* copy of the base model and switch per request. A single 70B model on one machine can serve a thousand differently-specialized customers. **That is a business model expressed as a matrix factorization**, and it is the main reason PEFT is treated as infrastructure rather than as a training trick.

**What "learning new knowledge" means in this context.** A rank-8 update to a layer can write eight patterns onto it (§17.7 above). That is ample for "always answer in JSON", "adopt this house style", "route this class of question to that behaviour" — reweightings of capabilities the model already has. It is not ample for "learn a language the base model never saw," where you need to change what the layer fundamentally computes. **The honest test is not how much data you have, it is whether the target behaviour is a recombination of existing abilities or a new one.**

---

## 17.8 Model merging

Combine multiple fine-tuned models **without any retraining**.

**Task arithmetic.** Define a task vector $`\tau_i = \theta_i^{\text{FT}} - \theta^{\text{pre}}`$. Then
▸ $$\theta_{\text{merged}} = \theta^{\text{pre}} + \sum_i\lambda_i\tau_i$$

Remarkably, this works: adding task vectors composes capabilities, and **negating** one ($`-\tau`$) reliably *removes* a behaviour (used for detoxification and unlearning). That such simple vector arithmetic works in a 10-billion-dimensional non-convex space is a  surprising empirical fact, and it's evidence for the linear-mode-connectivity picture in Ch. 31.

**TIES-Merging:** trim small-magnitude entries, elect a sign per parameter by summed magnitude, and average only the entries agreeing with the elected sign. Resolves the sign-conflict interference that plagues naive averaging.

**DARE:** randomly drop a large fraction of task-vector entries and rescale the rest by $`1/(1-p)`$ — most of a task vector is redundant.

**Model soups:** average many models fine-tuned with different hyperparameters from the same initialization. Reliably beats the best individual model with zero extra inference cost.

**SLERP:** spherical interpolation between two models, preserving norm.

▸ **The precondition for all of these:** the models must share an initialization and stay in the same loss basin. Merging models trained from different random seeds fails badly unless you first align their neurons by permutation (Ch. 31 §31.6).

#### Task arithmetic, decoded

$$\tau_i = \theta_i^{\text{FT}} - \theta^{\text{pre}}, \qquad \theta_{\text{merged}} = \theta^{\text{pre}} + \sum_i\lambda_i\tau_i$$

**The symbols.** $`\theta^{\text{pre}}`$ ("theta pre") is the pretrained parameter vector — every weight in the model, flattened into one enormous list. $`\theta_i^{\text{FT}}`$ is the same list after fine-tuning on task $`i`$. $`\tau_i`$ is their difference — **a direction in weight space**. $`\lambda_i`$ ("lambda") is how strongly to apply it, usually somewhere between 0.3 and 1.0. And $`\sum_i`$ is the ordinary loop from §0.3.

⚠ **Note the collision:** $`\tau`$ meant *temperature* in §17.6 and means *task vector* here. Same Greek letter, two unrelated jobs, three sections apart. This is normal and the book flags it because context is your only guide.

**Read the first equation aloud:** *"the task vector is fine-tuned minus pretrained"* — that is, **everything that fine-tuning did, and nothing else.** The second: *"start from the pretrained model and add some amount of each task's changes."*

> **Analogy.** Version control. $`\theta^{\text{pre}}`$ is the main branch, and $`\tau_i`$ is a **patch** — a diff describing exactly what one contributor changed. Adding two patches applies both sets of edits. Applying a patch with $`\lambda = 0.5`$ is a half-strength edit. And **negating a patch reverts the commit** — which is precisely how $`-\tau`$ is used to remove a behaviour a model learned. The remarkable claim of task arithmetic is that a 7-billion-dimensional patch, produced by thousands of steps of stochastic gradient descent through a wildly non-convex landscape, behaves like a well-mannered vector you can add, scale, and subtract.

▸ **Why " surprising" is not an overstatement.** Nothing about training guarantees this. Two fine-tuning runs explore different regions of a landscape with no linear structure; there is no theorem saying their displacements should compose. That subtracting a toxicity task vector reliably makes a model less toxic — without retraining, without data, by *arithmetic* — is an empirical discovery about the geometry of trained networks, and it is one of the main pieces of evidence for the linear-mode-connectivity picture of Ch. 31: fine-tuning from a shared starting point stays inside one broad, flat basin, and inside a basin, the landscape is locally linear enough for vector algebra to mean something.

**Why naive averaging fails, with numbers.** Two task vectors disagree on a parameter: task A wants $`+0.10`$, task B wants $`-0.09`$. Average them:

$$\frac{0.10 + (-0.09)}{2} = 0.005$$

**Both capabilities have been erased at that coordinate**, replaced by essentially zero. Do this across millions of parameters and the merged model is worse than either parent — the classic symptom, and the one TIES-Merging is built to fix.

**TIES, in three steps** (the name stands for **TrIm, Elect Sign**):
1. **Trim** — zero out the small-magnitude entries of each task vector. Most of a task vector is noise, and noise is what produces spurious conflicts.
2. **Elect** — for each parameter, sum the magnitudes on the positive side and on the negative side, and declare a winning sign.
3. **Merge** — average only the entries that agree with the elected sign, ignoring the dissenters.

The election converts a tug-of-war into a decision. **A parameter ends up firmly at $`+0.10`$ rather than limply at $`+0.005`$.**

**DARE's rescaling, and why $`1/(1-p)`$.** Drop a fraction $`p`$ of the entries at random and multiply the survivors by $`1/(1-p)`$. The expected value of any entry is then

$$\underbrace{(1-p)}_{\text{survives}}\times\underbrace{\frac{x}{1-p}}_{\text{rescaled}} \;+\; \underbrace{p\times 0}_{\text{dropped}} \;=\; x$$

▸ **Unchanged in expectation** — exactly the arithmetic behind inverted dropout in Ch. 7, reused here. That you can delete 90% of a task vector at random and lose nothing says the same thing the pruning section said: **most of what fine-tuning writes into a network is redundant.**

**Model soups versus ensembles — a distinction worth being precise about.** An ensemble averages the *predictions* of $`n`$ models and costs $`n`$ forward passes. A soup averages the *weights* and costs one. Averaging weights is normally a catastrophic idea — two networks that compute the same function can have wildly different weights, so their average computes neither — and it works here **only** because every ingredient started from the same initialization and never left the basin. **SLERP** (**S**pherical **L**inear int**ERP**olation) exists for the same reason: straight-line averaging of two vectors of similar length produces a shorter one, and weight norm matters, so you interpolate along the sphere instead of through it.

> **Where this came from.** **Model soups** (Mitchell Wortsman and colleagues, 2022) came out of the observation that when you sweep hyperparameters you normally keep the best run and discard the rest — and that discarding them is waste, because their weights average into something better than any individual. **Task arithmetic** (Gabriel Ilharco and colleagues, 2022) took the further step of treating the fine-tuning displacement as a first-class object that could be added and negated. Both rest on the **linear mode connectivity** results of the preceding years, which established that networks fine-tuned from a shared checkpoint stay connected by paths of low loss — and the *Git Re-Basin* line of work showed that models from *different* seeds can sometimes be brought into the same basin first, by permuting their neurons into alignment. That permutation step is the precondition the section above is warning you about, and it is why "merge two models you found on the internet" usually fails.

#### Examples and non-examples: what actually makes inference slow

Optimizing the wrong phase is the most common and most expensive mistake in serving.

**✅ Correctly diagnosed bottlenecks**

| Symptom | Real cause | The right fix |
|---|---|---|
| Slow token-by-token generation, low GPU utilization | **Decode is memory-bandwidth-bound** — you read every weight to produce one token | Bigger batches, quantization, speculative decoding |
| Slow time-to-first-token on a long prompt | **Prefill is compute-bound** — it's a big matrix multiply | More FLOPs, better kernels, chunked prefill |
| Throughput collapses with many users | KV cache exhausts memory | PagedAttention, GQA/MQA, shorter context |

**❌ Near-misses — plausible fixes that don't help**

| Attempted fix | Why it fails |
|---|---|
| Buying a GPU with more FLOPs to speed up decode | Decode is bandwidth-bound; the extra FLOPs sit idle |
| Unstructured pruning to 90% sparsity for speed | Dense hardware gets **no speedup** without structured patterns |
| Bigger batches to fix *latency* | Batching improves throughput and often makes individual latency **worse** |
| Distillation to cut memory | Distillation trains a smaller model — it's a training-time fix, not a serving knob |
| Assuming quantization degrades quality proportionally | 8-bit is often near-lossless; the cliff appears lower down, and it's abrupt rather than gradual |

▸ **The boundary:** prefill and decode are two different machines with two different bottlenecks sharing one model. **Prefill is compute-bound, decode is memory-bound**, and a fix aimed at the wrong one is money spent for nothing. The first question in any inference optimization is always *which phase am I actually in?*

> **Common misconception.** *"A bigger KV cache is basically free — it's just memory."* At long context and high concurrency the KV cache can exceed the size of the model weights, and every token of every sequence must be re-read on every decode step. It consumes the exact resource decode is starved of. Grouped-query attention, multi-query attention, and PagedAttention all exist to attack this one number.

> **Common misconception.** *"LoRA works because most weights don't matter."* It works because the *update* $`\Delta W`$ is empirically low-rank — the weights themselves are full-rank and matter enormously. You are not claiming the model is compressible; you are claiming that the *adjustment* to it lives in a small subspace. This distinction is what makes the Eckart–Young argument of §1.1.3 the right frame for it.

> **Common misconception.** *"Speculative decoding is a quality/speed trade-off."* It is **exactly lossless**. The rejection-sampling correction guarantees the output distribution is identical to the target model's — you're just guessing several tokens ahead with a cheap model and verifying them in one parallel pass. The gain comes from turning many memory-bound steps into one, not from accepting worse tokens.

---

## Did you know?

- **The most expensive GPU on the market spends over 99% of a single-user decode step doing nothing.** The 24-tokens-per-second floor for a 70B model in §17.1 contains no term for the chip's arithmetic speed at all. You could make the tensor cores infinitely fast and the number would not move. Serving a language model to one user is a memory-logistics problem wearing a supercomputer's clothes.

- **Pruning was invented to make networks *better*, not cheaper.** *Optimal Brain Damage* (LeCun, Denker and Solla, 1989) argued for deleting weights as a form of regularization, because the networks of the era overfit badly. The efficiency motivation that dominates today came decades later; the original paper was worried about generalization, not about inference bills.

- **Quantization arrived from the telephone network, essentially unmodified.** Alec Reeves patented pulse-code modulation in the late 1930s, and W. R. Bennett published the analysis of quantization noise at Bell Labs in 1948 — the same year and the same institution as Shannon's information theory. Scale factor, zero-point, dynamic range, quantization error: all of that vocabulary was worked out to fit a human voice down a copper wire.

- **The paper that launched knowledge distillation was a workshop paper, and it opens with an analogy about insects.** Hinton, Vinyals and Dean pointed out that many insects have a larval form optimized for eating and an adult form optimized for flying, and that machine learning insists on using the same organism for training and for deployment. It appeared at a NeurIPS workshop in 2014 rather than in the main proceedings.

- **A distilled student can learn to recognize a category it has literally never seen.** In the original distillation experiments, the transfer set omitted every example of one digit class, and the student still classified that digit correctly most of the time. Everything it knew about the missing class arrived indirectly, through the shape of the teacher's probabilities for the other nine.

- **Speculative decoding was discovered twice in the same year, at two labs, independently** — one team at Google Research, one at DeepMind. The name is borrowed from speculative execution in CPUs, the 1990s technique that makes modern processors fast and that turned out, in 2018, to be the root cause of the Spectre and Meltdown vulnerabilities.

- **The accept/reject rule at the heart of speculative decoding is due to John von Neumann, writing in 1951** about how to generate random numbers on machines that had no source of randomness. It is now the reason your chatbot replies twice as fast.

- **vLLM's central idea is sixty years old and came from a different discipline entirely.** Virtual memory and paging were built for the Atlas computer at the University of Manchester around 1962, so programs could pretend they had more memory than the machine contained. The same trick — a table saying where the pieces really are — raised GPU memory utilization on language-model servers from about 30% to over 90%.

- **The sparsity that hardware actually accelerates is 50%, and the sparsity researchers celebrate is 90%.** Unstructured 90%-sparse matrices are frequently *slower* than dense ones on a GPU, because you must store coordinates and chase them. The 2:4 pattern — exactly two zeros in every four — is regular enough for silicon and delivers about 2×.

- **LoRA's most famous application did not exist when LoRA was written.** The 2021 paper was aimed at adapting GPT-3 without keeping a 175-billion-parameter copy per customer. It became a household word among people training artistic styles into image generators, a community and a use case that arrived afterwards.

- **The activation outliers that make quantization hard do not exist in small models.** They emerge fairly abruptly as models grow past roughly 6–7 billion parameters. Below that scale, naive int8 quantization simply works — which is why the problem ambushed the field rather than being anticipated.

- **NF4, the format that put 65-billion-parameter fine-tuning on a single desktop GPU, contains no evenly spaced numbers.** Its sixteen levels sit at the quantiles of a normal distribution, so each level is used by the same number of weights. It is a number format designed around the statistics of its contents rather than around arithmetic convenience.

- **You can remove a behaviour from a model by subtracting a vector.** Compute (fine-tuned weights − pretrained weights) for a model deliberately trained to be toxic, then subtract it from a clean model. The clean model gets less toxic. No retraining, no data, no gradient — just arithmetic in a ten-billion-dimensional space where nothing guarantees this should work.

---

## Check for Understanding

**Decode is memory-bandwidth-bound, which means every parameter is read to produce one token — so batching is almost free, quantization buys near-linear speedup, speculative decoding gets multiple tokens per weight-read at provably zero quality cost, and the KV cache rather than the weights is usually what limits how many users you can serve.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why does a 70B model on the fastest available GPU emit only about 24 tokens per second for one user — and why does making the GPU faster not help?**
2. **Why is serving 64 users almost as cheap as serving one?** (Correct answer: because the expensive thing is reading the weights, and you read them once either way.)
3. **What is arithmetic intensity, and what is a ridge point?** Explain it with the workshop-and-timber-yard picture rather than the formula.
4. **Why is speculative decoding *lossless*?** Not "close enough" — exactly. What does the accept/reject rule do to guarantee it?
5. **Why does drafting 50 tokens ahead not help?** What is the ceiling, and what sets it?
6. **Why do activations quantize badly when weights quantize easily?** What does one outlier channel do to the other 253 available integer levels?
7. **Why does SmoothQuant not change what the network computes?** Explain the sofa.
8. **Why can you set 90% of a network's weights to zero and get no speedup at all?**
9. **Why is a soft teacher distribution more informative than a hard label, and why does temperature matter?** Where does the $`\tau^2`$ come from and what would break without it?
10. **Why does LoRA initialize one factor to zero and the other randomly — and what goes wrong if you zero both?**
11. **Why can a LoRA adapter be served at zero extra cost while an adapter layer cannot?**
12. **What is a task vector, and why is it surprising that adding and subtracting them works?**
13. **At a fixed memory budget, why is a big model at 4 bits better than a small model at 16?**

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 18 — Retrieval, RAG, Tools & Agents](18-retrieval-rag-agents.md)
