# Chapter 13 — GPT: Autoregressive Language Modelling

> **Prerequisites:** Ch. 10, 11, 12.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim once; each is unpacked properly where it first appears. This chapter has less mathematics than most and more **vocabulary** — the difficulty is in the terms of art, not the algebra.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`x_{1:T}`$ | "x one through T" | The **whole sequence** of $`T`$ tokens |
| $`x_{<t}`$ | "x less than t" | Everything **before** position $`t`$ — the prefix, the context so far |
| $`p_\theta(x_t \mid x_{<t})`$ | "p theta of x-t given x-before-t" | The probability the model assigns to the next token, **having seen** the prefix |
| $`\prod_{t=1}^{T}`$ | "product from t equals 1 to T" | **Multiply** over every position — a `for` loop that multiplies (§0.3) |
| $`\mathcal{L}`$ | "script L" | The **loss** — here, the average negative log-probability of the true next token |
| $`N,\ L,\ d,\ T`$ | — | **P**arameter count, **l**ayer count, model **width**, context **length** |
| $`\arg\max_x`$ | "arg max over x" | The **input** that achieves the maximum, not the maximum itself (§0.3) |
| $`\tau`$ | "tau" | **Temperature** — how flat or peaked the sampling distribution is |
| $`\propto`$ | "is proportional to" | Equal up to a normalizing constant we don't care about |
| $`\mathrm{PPL}`$ | "perplexity" | $`e^{\text{loss}}`$ — the **effective branching factor** |
| $`\mathcal{M}`$ | "script M" | The **set of masked positions** in BERT-style training |
| $`x_{\setminus\mathcal{M}}`$ | "x not M" | Every token **except** the masked ones |
| $`\lvert y\rvert^{\alpha}`$ | "length of y, to the alpha" | Sequence length raised to a power — beam search's **length normalization** |
| $`p_{\min}`$ | "p min" | Min-$`p`$'s relative cutoff |
| $`H`$ | "H" | **Entropy** — the model's average surprise (§1.4) |
| $`\mathcal{N}(0, 0.02^2)`$ | "normal, mean 0, sd 0.02" | The **initialization** distribution for weights |

**Full forms for the abbreviations in this chapter.** Say each aloud once.

| Short | Full form |
|---|---|
| AdamW | Adam with decoupled **W**eight decay |
| AIME | American Invitational Mathematics Examination (used as a benchmark) |
| ASR | Automatic Speech Recognition |
| BERT | Bidirectional Encoder Representations from Transformers |
| bf16 / fp32 | brain floating point 16-bit / floating point 32-bit |
| BPB / BPC | Bits Per Byte / Bits Per Character |
| C4 | Colossal Clean Crawled Corpus (T5's training set) |
| CFG | Context-Free Grammar (**not** classifier-free guidance, which is Ch. 20) |
| DeBERTa | Decoding-enhanced BERT with disentangled attention |
| ELECTRA | Efficiently Learning an Encoder that Classifies Token Replacements Accurately |
| EMA | Exponential Moving Average |
| EOS | End Of Sequence (token) |
| FFN | Feed-Forward Network |
| GPQA | Graduate-level Google-Proof Q&A benchmark |
| GPT | Generative Pre-trained Transformer |
| GQA | Grouped-Query Attention |
| GSM8K | Grade School Math, 8 thousand problems |
| LLM / LM | Large Language Model / Language Model |
| MFU | Model FLOPs Utilization |
| MLM | Masked Language Modelling |
| MMLU | Massive Multitask Language Understanding |
| MoE | Mixture of Experts |
| NER | Named Entity Recognition |
| NSP | Next Sentence Prediction (BERT's discarded second objective) |
| PPL | Perplexity |
| QA | Question Answering |
| RLHF | Reinforcement Learning from Human Feedback |
| RMSNorm | Root Mean Square Normalization |
| RoBERTa | Robustly optimized BERT approach |
| RoPE | Rotary Position Embedding |
| RULER | a long-context benchmark suite |
| SWE-bench | Software Engineering benchmark |
| SwiGLU | Swish-Gated Linear Unit |
| T5 | Text-To-Text Transfer Transformer (five T's) |
| WSD | Warmup–Stable–Decay (learning-rate schedule) |

---

## 13.1 The objective

### The one-line idea

Predict the next token. Do it on enough text and enough parameters, and essentially every language capability appears as a side effect.

### The analogy

Learning to play music by being handed millions of scores with the last note covered and being asked to guess it. To get good, you cannot just memorize — you have to internalize key signatures, chord progressions, phrasing, genre conventions, and the composer's habits. Prediction forces understanding because understanding is the cheapest way to predict.

### The math

Factorize the joint distribution by the chain rule of probability — **exactly, with no approximation**:

▸ $$p_\theta(x_{1:T}) = \prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\qquad \mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

#### Reading the chain rule of probability in plain English

Two formulas here. Take them one at a time.

**The first is a factorization, and it is an identity, not a modelling choice.**

$$p_\theta(x_{1:T}) = \prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

| Piece | Read aloud | Meaning |
|---|---|---|
| $`p_\theta(x_{1:T})`$ | "p theta of x-one-through-T" | How likely the model thinks **this entire document** is |
| $`\prod_{t=1}^{T}`$ | "product from t = 1 to T" | Multiply the following, once per position |
| $`p_\theta(x_t \mid x_{<t})`$ | "…of x-t **given** everything before it" | How likely **the next token** is, having read the prefix |
| $`\mid`$ | "given" | Conditional probability (§0.9, Trap 5) — **not** division, not absolute value |

▸ **In one sentence: the probability of a whole document equals the probability of its first word, times the probability of its second word given the first, times the probability of its third given the first two, and so on to the end.**

> **Analogy.** The chance of a particular hand of cards being dealt equals the chance of the first card, times the chance of the second *given* the first has left the deck, times the chance of the third given the first two. Nothing is being approximated — you are simply choosing to count the outcome one card at a time instead of all at once. **The word "exactly, with no approximation" in the text above is doing real work**, and §13.1's first bullet is why it matters.

**Numbers, with $`T=3`$.** The sentence "the cat sat," with the model's assigned probabilities:

$$p(\text{"the cat sat"}) = \underbrace{p(\text{the})}_{0.05}\times\underbrace{p(\text{cat}\mid\text{the})}_{0.01}\times\underbrace{p(\text{sat}\mid\text{the cat})}_{0.10} = 5\times10^{-5}$$

**The second formula is the loss:**

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

Read it right to left. Take the probability the model gave the token that *actually* came next. Take its log. Add up over all positions. Divide by $`T`$ to average. Negate, so that lower is better.

**Why the log?** Two reasons, both from §0.3. Products of thousands of numbers below 1 underflow to zero in any float format — $`0.1^{500}`$ is not representable. And $`\log`$ turns products into sums: $`\log\prod = \sum\log`$. **So minimizing $`\mathcal{L}`$ is exactly maximizing the probability of the training corpus** — it is the same objective, made numerically survivable.

**Numbers again.** Our three tokens: $`-\tfrac13(\log 0.05 + \log 0.01 + \log 0.10) = -\tfrac13(-3.00 - 4.61 - 2.30) = 3.30`$ nats. Perplexity $`= e^{3.30} = 27`$ — "as confused as if choosing uniformly among 27 options at each step." A good model on English prose sits nearer 10–20; a random model over a 50,000-token vocabulary sits at 50,000.

▸ **Why the negative sign, said properly.** $`\log`$ of a probability is always negative (probabilities are at most 1). Negating makes the loss positive, and it turns "make the probability large" into "make this number small," which is what gradient descent expects. **A loss of 0 would mean the model assigned probability exactly 1 to every token it saw** — perfect prediction, which is only possible if the text is deterministic.

> **Where this came from.** The idea of modelling language by the statistics of what follows what is older than computing hardware capable of doing it. **Claude Shannon**, in *A Mathematical Theory of Communication* (1948), estimated the statistics of English from books and used them to **generate text** — sampling letters, then letter-pairs, then words, then word-pairs, and showing that each step produced output that looked more like English than the last. It was a demonstration of an information-theoretic point, not an attempt to build anything, and he did the sampling by hand with a book, flipping to random pages to find the next occurrence of a phrase.
>
> Three years later, in *Prediction and Entropy of Printed English* (1951), he ran the experiment in reverse: he sat human subjects down and had them **guess the next letter of a text, one at a time**, using their error rate to estimate the entropy of English at roughly 1 bit per character. In other words, the first serious measurement of how predictable language is was made by using people as language models, and the number he obtained is still the reference point against which a modern model's bits-per-character is compared.

**Why this is such a good objective:**

1. **Exact likelihood.** No bound, no adversary, no partition function. Compare VAEs (a bound), GANs (no likelihood at all), EBMs (intractable $`Z`$) — Ch. 19.
2. **Dense supervision.** One forward pass over $`T`$ tokens yields $`T`$ prediction targets (thanks to causal masking, Ch. 11 §11.4).
3. **Self-supervised.** The labels are the data.
4. **Task-universal.** Any task expressible as text — translation, QA, summarization, code, reasoning — is a conditional distribution $`p(\text{answer}\mid\text{question})`$, which the model already represents.

#### The four advantages, decoded

These four bullets are the entire case for why this architecture won, so they are worth having at conversational speed.

**1. "Exact likelihood."** The model can tell you the actual probability it assigns to any text, computed exactly. Compare the alternatives in Ch. 19:

| Family | What you get | Why it hurts |
|---|---|---|
| Autoregressive LM | The **exact** number | — |
| VAE | A **lower bound** (the ELBO — evidence lower bound) | You never know how loose the bound is |
| GAN | **Nothing** — no likelihood at all | You cannot even measure whether the model improved |
| Energy-based | A number divided by an **intractable** constant $`Z`$ | The constant requires summing over all possible outputs |

> **Analogy.** A shop that prints the actual price on the tag, versus one that guarantees "no more than £50," versus one that will only tell you whether *this* item costs more than *that* one. The first is far easier to do business with, and the difference compounds through everything you build on top.

**2. "Dense supervision."** Feed in a 4096-token document and the causal mask (Ch. 11) means position 1 predicts token 2, position 2 predicts token 3, and so on — **4095 separate prediction problems from a single forward pass.** This is the fact that decides everything downstream.

▸ **Compare BERT (§13.6): 15% masking means roughly 614 learning signals from the same 4096-token document, against 4095.** That is a ~6.7× difference in supervision per unit of compute, and it is the single largest reason the field consolidated on decoder-only models. It is not a claim about which understands language better; it is arithmetic about training efficiency.

**3. "Self-supervised. The labels are the data."** There is no annotation step. Nobody labels anything. The correct answer to "what comes next" is *already sitting there in the text*, which is why the training set can be trillions of tokens scraped from the internet rather than the millions of hand-labelled examples that classical supervised learning is limited to.

**4. "Task-universal."** Translation is $`p(\text{French}\mid\text{English})`$. Summarization is $`p(\text{summary}\mid\text{article})`$. Question answering is $`p(\text{answer}\mid\text{question})`$. **Every one of these is already a conditional next-token distribution**, which the model represents by construction — so no new head, no new loss, no new training run is needed to attempt any of them.

▸ **Why "just predicting the next token" is not a limitation:** predicting the token after a long argument requires having tracked the argument. Predicting the last word of a murder mystery's reveal requires having solved it. The objective is shallow; the competence required to minimize it is not.

#### Why the shallow objective forces deep competence

This claim gets waved at more often than it gets argued, so here is the argument.

Consider the token the model must predict at the end of each of these:

| Prefix | To predict the next token you must have… |
|---|---|
| "The capital of Australia is ___" | stored a fact |
| "17 × 24 = ___" | performed arithmetic |
| "Alice put the key in her left pocket. She walked to the shop. The key was in her ___" | tracked an object through a narrative |
| "…and so the butler could not have been in the library, because ___" | followed a chain of inference |
| "def fib(n): if n <= 1: return n; return ___" | understood a recursive definition |

**Nothing in the loss function asks for facts, arithmetic, object permanence, deduction or recursion.** The loss asks for one thing: a good probability distribution over the next token. But the *only way* to be good at that on real text is to acquire all of those capabilities, because real text was produced by processes that had them.

> **Analogy.** An exam consisting entirely of fill-in-the-blank questions drawn from every book ever written. The format is trivial. Scoring well on it is not, because the blanks were made by deleting words from sentences written by people who knew things. **You cannot fill in "the murderer was the ___" without doing what the author did.**

▸ **The transferable framing: a prediction objective is a lower bound on the competence needed to satisfy it, and the bound is set by whoever generated the data, not by the objective.** This is why the same three-line loss function that produced a mediocre 2018 model produces a 2026 one — the loss never changed, the data and the capacity did.

#### Examples and non-examples: autoregressive language modelling

**✅  examples**

| Example | Why it qualifies |
|---|---|
| GPT predicting token $`t`$ from tokens $`1..t-1`$ behind a causal mask | The factorization $`p(x) = \prod_t p(x_t \mid x_{<t})`$ is exact, and no term peeks forward |
| A bigram model $`p(x_t \mid x_{t-1})`$ trained by counting | Same left-to-right factorization, with the conditioning set truncated to one token |
| WaveNet over raw audio samples, PixelCNN over pixels | Nothing in the objective is about language; it is about ordering the variables and conditioning on the prefix |
| Next-token prediction on Python source | Same loss, same mask, different corpus |
| An LSTM language model (Ch. 9) | Architecture is irrelevant to whether the objective is autoregressive |

**❌ Near-misses — look like next-token prediction, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| BERT's masked language modelling | Position $`t`$ is predicted using tokens on **both** sides. The per-position terms do not multiply into a valid joint distribution over the sequence | A denoising objective. It builds representations, not a generative model of $`p(x)`$ (§13.6) |
| T5 span corruption | Predicts deleted spans conditioned on the whole corrupted document, including what came after | A sequence-to-sequence denoising objective |
| A diffusion language model | Refines every position at once across denoising steps; there is no left-to-right order at all | Non-autoregressive generation (Ch. 21) |
| "The model wrote a paragraph" | One forward pass emits **one** distribution over the vocabulary. The paragraph comes from a `for` loop that calls the model 200 times, feeding each sampled token back in | A sampling loop wrapped around an autoregressive model |
| Training with teacher forcing and calling it self-generation | During training the model always conditions on the **true** prefix; it never sees its own output | Teacher forcing — and the mismatch is exactly the source of exposure bias |

▸ **The boundary:** autoregressive means the joint probability is written as a product of conditionals, each depending on **strictly earlier** positions and nothing later. If any prediction can see a token to its right, the chain rule no longer applies and you have a representation learner rather than a generator.

> **Common misconception.** *"The model plans out its whole answer before it starts writing."* Not in any architectural sense. At generation time there is exactly one forward pass per token; the model emits a distribution over the **next** token, something outside the model samples from it, and the sampled token is appended and fed back in. There is no buffer holding a draft, no lookahead search, no place a plan could be stored between steps — which is precisely why a model can paint itself into a corner mid-sentence and then produce a strained ending rather than backing up. But the honest version is more interesting than a flat denial: the *hidden state at the current position* can and does encode information about tokens several steps ahead, because doing so lowers next-token loss. Interpretability work has found representations that look like a target being held in mind before it is emitted — a model composing a rhyming couplet appears to select the rhyme word early and then steer toward it. So: **the loop has no lookahead; the representation can.** The misconception is tempting because the output reads like it was planned, and it reads that way because coherent text is what the objective selects for — not because a plan exists anywhere you could point to.

---

## 13.2 The architectural lineage

| Model | Year | $`N`$ | $`L`$ | $`d`$ | $`T`$ | The change that mattered |
|---|---|---|---|---|---|---|
| GPT-1 | 2018 | 117M | 12 | 768 | 512 | generative pretraining + discriminative fine-tuning |
| GPT-2 | 2019 | 1.5B | 48 | 1600 | 1024 | **pre-LN**; scale; zero-shot task transfer |
| GPT-3 | 2020 | 175B | 96 | 12288 | 2048 | scale; **in-context learning** |
| Chinchilla | 2022 | 70B | 80 | 8192 | 2048 | compute-optimal data scaling (Ch. 15) |
| LLaMA | 2023 | 7–65B | | | 2048 | RMSNorm + SwiGLU + RoPE; small models trained far past "optimal" |
| LLaMA-3 | 2024 | 8–405B | | | 8k–128k | 15T tokens; GQA; 128k vocab |
| Frontier (2025–26) | | sparse MoE | | | 128k–1M+ | MoE, long context, reasoning post-training |

▸ **The stable modern recipe**, worth being able to recite: decoder-only, pre-norm **RMSNorm**, **SwiGLU** FFN with $`d_{\text{ff}}=\frac83 d`$, **RoPE**, **GQA**, no biases anywhere, no dropout during pretraining, weight-tied or untied embeddings, AdamW with cosine or WSD schedule, bf16 with fp32 master weights.

#### Reading the lineage table

The four numeric columns are the entire specification of a transformer, so learn to read them as a shape.

| Column | Read aloud | What it controls |
|---|---|---|
| $`N`$ | "N" | **Total parameter count** — how many numbers the model has learned |
| $`L`$ | "L" | **Layers** — how many times the block is stacked; roughly, depth of reasoning per token |
| $`d`$ | "d" | **Width** of the residual stream — how many numbers describe each token at each layer |
| $`T`$ | "T" | **Context length** — how many tokens it can see at once |

▸ **A rough check you can do in your head: $`N \approx 12\,L\,d^2`$ for a standard transformer.** (Each layer holds roughly $`4d^2`$ of attention weights and $`8d^2`$ of feed-forward weights.) Try it on GPT-3: $`12\times96\times12288^2 = 1.74\times10^{11}`$ — **174B against the stated 175B.** The estimate is that good because the architecture really is just that block, repeated.

**Now feel the scale changes**, because the numbers do not read as dramatic on a page:

| From → To | Parameters | Context |
|---|---|---|
| GPT-1 → GPT-2 | 117M → 1.5B (**13×**) | 512 → 1024 (2×) |
| GPT-2 → GPT-3 | 1.5B → 175B (**117×**) | 1024 → 2048 (2×) |
| GPT-3 → LLaMA-3 | context 2048 → 128k (**62×**) | — |

> **Analogy for the width column.** $`d`$ is the width of a conveyor belt running the length of the factory, and every layer is a station that reads things off the belt and adds things back onto it. GPT-3's belt is 12,288 lanes wide. Ch. 32's superposition results are about how many distinct items can share those lanes without colliding — and §1.1.5 already told you the answer is far more than 12,288.

**Reading the "change that mattered" column as a story.** Each row is one lesson the field learned:

- **GPT-1 (2018)** — *pretrain first, then fine-tune.* Before this, most NLP models were trained from scratch per task.
- **GPT-2 (2019)** — *pre-LN and scale*, and the discovery that a big enough model does tasks **zero-shot**, with no fine-tuning at all.
- **GPT-3 (2020)** — *in-context learning.* Put examples in the prompt and the model adapts, with no weight update. Nobody designed this.
- **Chinchilla (2022)** — *you were training on too little data.* At the time it was standard to scale parameters far faster than tokens; Chinchilla showed a 70B model trained on more data beat a 280B model trained on less (Ch. 15).
- **LLaMA (2023)** — *train small models far past "optimal."* Chinchilla optimizes training compute; if you intend to *serve* the model to millions of people, inference cost dominates, so a smaller model over-trained is the better buy.
- **LLaMA-3 (2024)** — *15 trillion tokens.* Roughly 200× the Chinchilla-optimal amount for its size, and it kept improving.
- **Frontier (2025–26)** — *sparsity and post-training.* Mixture of Experts to grow parameters without growing per-token cost; reasoning post-training to spend compute at inference rather than training.

▸ **The one-line summary of eight years: the architecture barely changed and everything else did.** Strip away RMSNorm, SwiGLU, RoPE and GQA and a 2026 model is recognizably the 2017 decoder. What changed was scale, data quantity, data quality, and what happens *after* pretraining.

#### The modern recipe, decoded

Worth being able to recite, so worth understanding rather than memorizing. Each item, with why:

| Ingredient | What it is | Why it's there |
|---|---|---|
| **Decoder-only** | One stack, causal mask, no encoder | Dense supervision (§13.1) and one code path for everything |
| **Pre-norm** | Normalize *before* the block, not after | The residual path stays clean, so gradients reach layer 1 — this is what made 48+ layers trainable (Ch. 7) |
| **RMSNorm** | Root Mean Square Normalization: divide by the root-mean-square, **skip the mean subtraction** | Same benefit as LayerNorm, one fewer pass over the data, measurably faster |
| **SwiGLU** | A gated feed-forward network | Empirically a point or two better than plain ReLU at equal parameters |
| $`d_{\text{ff}} = \tfrac83 d`$ | The hidden width of the FFN | SwiGLU uses **three** weight matrices instead of two, so $`\tfrac83`$ keeps the parameter count matched to a conventional $`4d`$ FFN. **The odd fraction is a fair-comparison correction, not a magic number.** |
| **RoPE** | Rotary position embedding (Ch. 12) | Relative position, KV-cache compatible, extensible |
| **GQA** | Grouped-Query Attention (Ch. 12) | 8× smaller KV cache at near-MHA quality |
| **No biases** | Drop the $`+b`$ everywhere | They contribute almost nothing and cost memory traffic and stability |
| **No dropout** | None during pretraining | Dropout fights overfitting; on a trillion-token corpus seen roughly once, **there is nothing to overfit** |
| **Weight tying** | Share the input embedding and the output projection | Saves $`V\times d`$ parameters — at $`V=128`$k and $`d=8192`$ that is over a billion |
| **AdamW + cosine/WSD** | The optimizer and schedule (Ch. 5) | Decoupled weight decay; a schedule that anneals to a small final learning rate |
| **bf16 + fp32 master** | Compute in 16-bit, keep the authoritative copy in 32-bit | Half the memory traffic, without losing small updates to rounding (Ch. 14) |

▸ **Notice how many entries are removals**: no biases, no dropout, no mean subtraction in the norm, no encoder. **A large part of the last eight years of architecture research has consisted of deleting things and confirming nothing broke** — because at this scale, every component you keep costs memory bandwidth on every token of a trillion-token run.

> **Where this came from.** The lineage in this table is unusually well documented because most of it happened in public, and the individual stories are worth knowing:
>
> **GPT-1** (Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever at OpenAI, 2018) was never published at a peer-reviewed conference — it was released as a technical report and a blog post. **GPT-2** (2019) was the subject of a **staged release**: OpenAI initially withheld the full 1.5B-parameter model citing concerns about misuse for generating deceptive text, releasing progressively larger versions over several months and the full model only in November 2019. The decision was contentious at the time and is now unremarkable, both because the concerns proved overstated for a model of that size and because staged and gated releases became normal industry practice.
>
> **GPT-3**'s paper (2020) is titled *Language Models are Few-Shot Learners* — the title is the finding. The in-context learning result was not the stated goal of the project; scale was. And **Chinchilla** (Hoffmann and colleagues at DeepMind, 2022) is a rare case of a paper whose main contribution was showing that everyone, including its own authors' previous work, had been allocating compute wrongly — DeepMind's own 280B Gopher model was among the examples of the mistake.

---

## 13.3 In-context learning

### The phenomenon

Give the model examples in the prompt; it performs the task without any weight update.

```
sea otter => loutre de mer
peppermint => menthe poivrée
cheese =>              ← model completes "fromage"
```

▸ **Nothing about the training objective asks for this.** It is emergent, and it is arguably the most consequential empirical surprise in the field.

### Why it happens: the induction-head account

The best-supported mechanistic explanation (Olsson et al., 2022; expanded in Ch. 32):

An **induction head** implements the rule *"find where this token appeared before, and copy what followed it."* It requires two heads composing across layers:
1. A **previous-token head** in layer $`\ell`$ writes information about token $`t-1`$ into position $`t`$'s residual stream.
2. An **induction head** in layer $`\ell' > \ell`$ uses the current token as a query to match against that written information, locating earlier occurrences, and copies the token that followed.

**Minimum two attention layers.** One-layer attention-only transformers cannot form induction heads and cannot do in-context learning.

#### What makes in-context learning strange

State the strangeness precisely, because it is easy to under-appreciate.

**Learning, as defined everywhere else in this book, means changing the weights.** You compute a gradient, you take a step, $`\theta`$ becomes a different vector. In-context learning involves **none of that**. The weights are frozen. The model is in inference mode. Nothing is being optimized. And yet showing it three translated words makes it translate a fourth.

▸ **So whatever is happening is happening entirely in the activations** — in the residual stream, within a single forward pass, and it is discarded the moment the sequence ends. The model has, somehow, learned *an algorithm for adapting* rather than merely a set of answers, and that algorithm runs as a side effect of computing next-token probabilities.

> **Analogy.** A calculator that you never reprogram, which nevertheless does long division correctly the first time you type one in — because somewhere in its fixed circuitry there is a general division routine that the input activates. In-context learning says the pretrained weights contain general "figure out the pattern and continue it" machinery, and the prompt is what invokes it.

**And nothing in the training objective asked for this.** The loss said: *predict the next token in web text.* It never said *and also acquire the ability to pick up new tasks from examples.* The capability appeared because it happened to be useful for the stated goal — web text is full of lists, tables, translated pairs, and repeated formats, and a model that can spot "we're doing pairs now" predicts them better. **Emergence here is not mysterious in kind, only in degree.**

#### Unpacking induction heads

The mechanism is two heads working in sequence, and it is worth walking through slowly because it is the clearest known case of a specific circuit producing a specific behaviour.

**The rule being implemented:** *"Find where this token appeared before, and copy whatever came after it."* Formally, given a sequence containing `…A B … A`, predict `B`.

**Step 1 — the previous-token head (layer $`\ell`$).** A head whose attention pattern is trivial: every position attends to the position immediately before it. It copies information about token $`t-1`$ into position $`t`$'s residual stream. **After this layer, position $`t`$'s vector says, in effect, "I am `B`, and the thing before me was `A`."**

**Step 2 — the induction head (layer $`\ell' > \ell`$).** The current token is `A` again. This head forms a query meaning *"find positions whose predecessor was `A`."* Thanks to step 1, exactly those positions have that information written into them. The head matches, attends there, and copies out **what that position's token was** — namely `B`.

▸ **The two steps must be in different layers, and that is the whole content of "minimum two attention layers."** Step 2's query depends on information that step 1 wrote. A single layer's heads all read the *same* residual stream and write in parallel; none can read another's output. **The circuit needs a layer boundary to exist in, the way a two-stage pipeline needs a register between the stages.**

> **Analogy.** A library with no catalogue. Assistant 1 walks the shelves and writes on each book's spine the title of the book to its left. Assistant 2 is asked "what usually follows *Moby-Dick*?" and can now scan for spines annotated *Moby-Dick* and read off what those books are. Neither assistant could do the job alone, and assistant 2's work is only possible because assistant 1 went first.

**Why this explains in-context learning.** The translation prompt is `sea otter => loutre de mer / peppermint => menthe poivrée / cheese =>`. The model has seen `=>` followed by French twice. An induction-style circuit generalizing beyond exact token match — matching on *"the thing after an arrow"* rather than on a literal token — produces French. **Format-copying is induction. The task is inferred from the shape, not taught by the content.**

▸ **The decisive evidence:** during training there is a narrow window where in-context learning ability jumps sharply, and induction heads form *in that exact window*, visible as a bump in the loss curve. Ablating induction heads destroys in-context learning. This is one of the strongest mechanistic-to-behavioural links established in interpretability.

#### Why the "bump in the loss curve" is such strong evidence

Interpretability results usually establish **correlation**: this neuron lights up for dogs. That is compatible with the neuron being a cause, a side effect, or a coincidence. The induction-head result is stronger, on three counts:

1. **A timing coincidence too sharp to be chance.** Loss curves are otherwise smooth. During pretraining there is a brief window — a small **bump**, where loss briefly worsens or plateaus before dropping — and a behavioural capability appears in *that* window, and a specific circuit forms in *that* window. Three things co-located in training time.
2. **Ablation.** Delete the heads and the capability goes away. That converts correlation to causation, in the only way available.
3. **A predicted architectural constraint that holds.** The theory says you need two attention layers. One-layer attention-only transformers are checked, and they cannot do it. **A theory that forbids something, checked against a case where the thing is indeed absent, is worth far more than a theory that merely explains what was already observed.**

> **Analogy.** Suspecting a particular gene causes a trait. Noticing the trait appears exactly when the gene switches on is suggestive. Knocking the gene out and watching the trait vanish is decisive. Interpretability spent years on the first kind of evidence; this was one of the first clean instances of the second.

▸ **The reason this result is cited so heavily is what it promises rather than what it shows.** A single circuit explaining a single capability is a small result about a large system. But it is an **existence proof** that a capability which looks emergent and mysterious from the outside can turn out, on inspection, to be a specific and describable mechanism. Chapter 32 is the programme that follows from that.

> **Where this came from.** The induction-head account was developed at **Anthropic** by **Catherine Olsson and colleagues**, published in 2022 as *In-context Learning and Induction Heads*, building on the earlier *A Mathematical Framework for Transformer Circuits*. The framework work analyzed the smallest interesting cases — one- and two-layer attention-only transformers — precisely because they are simple enough to understand completely. **Induction heads were found there first, in toy models nobody would deploy**, and only then looked for (and found) in large ones. This is an unusual research strategy in a field that mostly studies its largest artifacts, and it is a large part of why the result is as clean as it is.

### Other framings

- **Implicit gradient descent:** a transformer layer can implement one step of gradient descent on a linear regression problem defined by the in-context examples. Provable in constructed settings; suggestive but not established for real models.
- **Bayesian task inference:** the pretraining distribution is a mixture over latent "tasks"; the prompt is evidence, and the model performs implicit posterior inference over which task it's in. Explains why format matters more than label correctness.

#### The two other framings, decoded

Both are attempts to answer *"what computation is the forward pass performing when it does in-context learning?"* They are not rival factions so much as different levels of description.

**Implicit gradient descent.** The claim: a transformer layer can, in principle, execute one step of gradient descent — and if the "training data" for that step is the examples in the prompt, then a stack of layers is running a small optimization loop inside the forward pass, on a problem defined by the prompt.

> **Analogy.** Discovering that a fixed mechanical calculator, when you feed it a certain input pattern, is internally performing Newton's method. The machine was never "given" an algorithm; the algorithm is what its gears amount to on that input.

The honest status is stated in the text and worth repeating: **provable in constructed settings, suggestive but not established for real models.** Someone has shown that weights *exist* which do this. That is not the same as showing that trained weights *do* it.

**Bayesian task inference.** The claim: pretraining data is a mixture over many latent "tasks" (writing recipes, translating, formatting tables, being a customer-service transcript). The prompt is **evidence** about which one you're in. The model computes something like a posterior over tasks and generates from the winner.

> **Analogy.** Overhearing a conversation. Three exchanges in, you have identified the register — a job interview, a doctor's appointment, a comedy routine — and you can predict the next line rather well. You did not *learn* anything; you *located* yourself in a space of situations you already knew.

▸ **The empirical finding that discriminates between accounts:** for many tasks, **randomizing the labels in the examples barely hurts performance** (Min et al., 2022). What the demonstrations mainly convey is the *format, label space, and input distribution* — not the input→output mapping. That is much more consistent with task-location than with in-context gradient descent. State this if asked; it's the kind of detail that signals real familiarity.

#### Why random labels are the killer experiment

Sit with what the experiment does. Take a sentiment classification prompt:

```
"loved every minute"  → positive
"waste of two hours"  → negative
"the pacing dragged"  → ???
```

Now **shuffle the labels so they are wrong**: `"loved every minute" → negative`, `"waste of two hours" → positive`. Keep everything else identical.

If in-context learning were fitting the input→output mapping — gradient descent, or anything like it — this should be **catastrophic**. You have handed it a training set where the labels are noise. Accuracy should collapse to chance.

▸ **It barely moves.** For many tasks, performance with randomized labels is close to performance with correct ones — and far above performance with no examples at all. **So the examples are doing something, but that something is not teaching the mapping.**

What the examples *do* convey, and what breaks when you remove it:

| Signal in the prompt | Effect of corrupting it |
|---|---|
| **The label space** (that answers are "positive"/"negative", not paragraphs) | Large degradation |
| **The format** (input, arrow, single word) | Large degradation |
| **The input distribution** (these are film reviews) | Noticeable degradation |
| **The input→output mapping** (which review gets which label) | **Small degradation** |

> **Analogy.** Asking someone to fill in a form. Showing them a completed example helps enormously — they learn which box takes a date, that names go in capitals, that the answer is one word. Whether the *specific* details in your example were true barely matters. The example is teaching them the genre, not the content.

**The caveats an expert would add**, since this result gets over-stated: the effect varies by task and by model, and larger and more capable models show more sensitivity to label correctness than smaller ones — the mapping matters more as models get better. **The finding is "format dominates," not "content is irrelevant."**

#### Examples and non-examples: in-context learning

**✅  examples**

| Example | Why it qualifies |
|---|---|
| GPT-3 shown 32 English→French pairs, then translating a 33rd | Behaviour changed within a single forward pass; the parameter file is byte-identical before and after |
| Five demonstrations of an invented format — `⟨word⟩ ⇒ ⟨word spelled backwards⟩` — and the model continuing it | The mapping was specified entirely by the prompt |
| Chain-of-thought prompting lifting GSM8K accuracy | The only thing that changed was the text in front of the question |
| An induction head completing `[A][B] … [A] → [B]` | The mechanistic substrate: match the earlier occurrence, copy what followed it |
| A model adopting your JSON field names after one example | Format transfer, the signal that in-context learning conveys most strongly |

**❌ Near-misses — look like in-context learning, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Fine-tuning on the same 32 pairs | Weights change. Run the model on an unrelated prompt afterwards and it behaves differently | Supervised fine-tuning |
| LoRA on a handful of examples | An adapter's weights change; the effect persists after the prompt is gone | Parameter-efficient fine-tuning (Ch. 17) |
| Retrieval-augmented generation pasting three documents into the prompt | The retrieved text supplies **facts**, not a demonstration of a task pattern. Remove the documents and the model still knows how to answer questions | Retrieval augmentation (Ch. 18) |
| A chatbot "remembering" what you said ten turns ago | The whole transcript is re-sent as the prompt on every single request. Nothing whatsoever persists inside the model between calls | Context re-submission |
| The model getting better at a task you have prompted it on many times across separate sessions | There is no channel by which that could happen — unless the provider is training on your data, which is a different mechanism entirely | Either nothing, or offline fine-tuning |

▸ **The boundary:** in-context learning is behaviour that changes as a function of the prompt **while every parameter stays frozen and identical**. The test is brutally simple: hash the weights before and after. If they match, whatever happened was in-context.

> **Common misconception.** *"In-context learning means the model is learning from the examples — it's fitting the input→output mapping in its head."* The random-label experiment is decisive against the strong form of this. Shuffle the labels so every demonstration is **wrong** and performance barely moves; corrupt the *format* or the *label space* instead and it collapses. So the examples are carrying enormous signal, but the signal is "here is the shape of the answer I want," not "here is the function to fit." There is a real research thread showing that a transformer *can* implement something equivalent to a gradient-descent step in its forward pass — but that result is constructed in tidy linear-regression settings, and it is a statement about what the architecture can express, not a demonstration of what a pretrained LLM does. The belief is tempting for two very good reasons: the word **learning** is right there in the name, and a few-shot prompt looks exactly like handing someone a small training set. The honest caveat runs the other way too — larger and more capable models are measurably *more* sensitive to label correctness than small ones, so the finding is **"format dominates," not "content is irrelevant."**

---

## 13.4 Decoding strategies

The model gives you $`p(x_t\mid x_{<t})`$. Turning it into text is a separate, consequential design choice.

### Deterministic

**Greedy:** $`x_t=\arg\max p`$. Fast; repetitive and dull; **not** the highest-probability *sequence* (a locally-best token can foreclose a better continuation).

**Beam search:** keep $`k`$ partial sequences, expand all, retain the top $`k`$ by cumulative log-probability.
- Needs **length normalization**: $`\frac{1}{|y|^\alpha}\sum\log p`$, $`\alpha\approx0.6`$–1.0. Without it, shorter sequences always win because every added token multiplies by a probability $`<1`$.
#### Greedy and beam search, decoded

**Greedy**, first. $`x_t = \arg\max p`$ reads *"pick the single most likely next token, every time."* Remember §0.3: $`\arg\max`$ returns the **token**, not the probability.

▸ **The subtle and important part is "not the highest-probability *sequence*."** Greedy is locally optimal and globally not. Work a two-step example:

| First token | $`p`$ | Best second token given it | $`p`$ | Sequence total |
|---|---|---|---|---|
| "The" | **0.6** | "end" | 0.10 | $`0.6\times0.10 = 0.06`$ |
| "A" | 0.4 | "remarkable" | 0.50 | $`0.4\times0.50 = \mathbf{0.20}`$ |

Greedy takes "The" because 0.6 > 0.4, and ends up on a path worth 0.06. The path through "A" was worth **more than three times as much**. *A locally-best token can foreclose a better continuation* — and by the time you find out, you cannot go back, because generation is one-way.

> **Analogy.** Driving by always taking the widest road at each junction. Perfectly sensible at every individual junction, and it will still route you into a cul-de-sac, because the widest road at junction 3 was the one that fed the motorway you needed at junction 2.

**Beam search** is the fix: keep $`k`$ candidate sequences alive instead of 1, extend all of them, keep the best $`k`$ by total log-probability, repeat. With $`k = 4`$ you are hedging across four futures at every step.

**Why length normalization is not optional.** Sequence probability is a **product** of numbers less than 1, so it can only shrink. In log space, $`\log p`$ of a whole sequence is a sum of negatives:

| Sequence length | Typical $`\log p`$ per token | Total |
|---|---|---|
| 5 tokens | $`-2`$ | $`-10`$ |
| 30 tokens | $`-2`$ | $`-60`$ |

The 30-token sequence is not worse — it is **longer**, and every additional token subtracts more. Without correction, beam search will always prefer the shortest thing it can say, which in practice means outputs that stop almost immediately.

The fix $`\frac{1}{\lvert y\rvert^{\alpha}}\sum\log p`$ divides by length raised to a power. Read $`\lvert y\rvert`$ as "the number of tokens in $`y`$." $`\alpha = 1`$ divides by length exactly, giving average log-probability per token; $`\alpha = 0`$ turns the correction off. **The tuned value of $`\alpha \approx 0.6`$–1.0 tells you that dividing by full length slightly over-corrects and starts favouring rambling** — the fractional exponent is an empirical compromise between two failure modes.

- ▸ **Beam search is right for translation and summarization, and wrong for open-ended generation.** In open-ended text, higher sequence probability correlates with *worse* human-judged quality past a point — the "likelihood trap." The highest-probability continuation of most prompts is degenerate repetition.

#### The likelihood trap, and why it is not a bug

This is the most counterintuitive claim in the chapter: **searching harder for the most probable text makes the text worse.** Two things must both be true for that to make sense, and both are.

**First: repetition really is high-probability.** Once a model has produced "the cat sat on the mat. the cat sat on the mat." the strongest available evidence about what comes next is *what came before* — and induction heads (§13.3) are extremely good at continuing an established pattern. The loop is self-reinforcing: each repetition makes the next repetition more likely still. Beam search, whose entire job is to find high-probability sequences, walks straight into it.

**Second, and this is the real insight: human text is not high-probability text.** Real writing is *locally surprising*. People choose the unexpected word, change subject, make a joke. Measure the per-token probability of human-written prose under a good language model and it **fluctuates wildly**. Measure beam-search output and it is unnaturally smooth and high — flat in a way no human text ever is.

> **Analogy.** A weather forecast that maximizes accuracy by predicting "same as yesterday" every day. It scores well and it is useless, because what makes weather worth forecasting is precisely the days it changes. **Maximizing likelihood optimizes for the most typical continuation, and interesting text is by definition not the most typical continuation.**

▸ **So why does beam search work for translation?** Because translation is **not open-ended**. There is a right answer, determined by the source sentence, and the model's job is to find it. High probability means "this is what the source says," and searching harder  helps. Summarization is similar. **The rule is: search when the answer is constrained by the input, sample when the output is  open.** Everything in the "Choosing" table below follows from that one distinction.

### Stochastic

**Temperature:** $`p_i\propto \exp(z_i/\tau)`$. $`\tau<1`$ sharpens, $`\tau>1`$ flattens, $`\tau\to0`$ is greedy.
▸ Note $`\tau`$ acts on **logits**, so it is not a linear reweighting of probabilities: $`p_i^{(\tau)} \propto p_i^{1/\tau}`$.

**Top-$`k`$:** sample from the $`k`$ most probable tokens. Problem: $`k`$ is fixed, but the appropriate number of plausible tokens varies enormously by context (after "the capital of France is" there is one; after "she opened the door and saw" there are thousands).

**Top-$`p`$ / nucleus (Holtzman et al.):** sample from the smallest set $`S`$ with $`\sum_{i\in S}p_i\ge p`$. ▸ **Adaptive to the distribution's entropy** — this is precisely the fix for top-$`k`$'s flaw, and it is the reason nucleus sampling became the default. Typical $`p=0.9`$–0.95.

**Min-$`p`$:** keep tokens with $`p_i \ge p_{\min}\cdot\max_j p_j`$. A relative threshold; more robust at high temperature.

**Typical sampling:** keep tokens whose surprisal $`-\log p_i`$ is close to the distribution's entropy $`H`$ — an information-theoretic criterion motivated by the observation that natural human text has locally near-uniform information density.

**Contrastive decoding / search:** penalize tokens that a smaller "amateur" model also finds likely, or penalize similarity to already-generated context. Reduces degeneration without sacrificing coherence.

#### Temperature, decoded

$`p_i \propto \exp(z_i/\tau)`$. The $`z_i`$ are **logits** — the raw scores the final layer produces, before softmax. $`\tau`$ divides them before exponentiating.

▸ **The crucial and frequently-missed point is the one the book flags: $`\tau`$ acts on the logits, so $`p_i^{(\tau)} \propto p_i^{1/\tau}`$ — it raises probabilities to a power, it does not scale them.** Powers reshape a distribution in a way multiplication cannot.

**Put numbers in.** Three tokens with probabilities $`0.6, 0.3, 0.1`$:

| $`\tau`$ | Computation | Resulting distribution |
|---|---|---|
| $`2.0`$ | $`p^{0.5}`$: $`0.775, 0.548, 0.316`$ → normalize | $`0.47,\ 0.34,\ 0.19`$ — **flattened** |
| $`1.0`$ | unchanged | $`0.60,\ 0.30,\ 0.10`$ |
| $`0.5`$ | $`p^{2}`$: $`0.36, 0.09, 0.01`$ → normalize | $`0.78,\ 0.20,\ 0.02`$ — **sharpened** |
| $`\to 0`$ | $`p^{\infty}`$ | $`1,\ 0,\ 0`$ — **greedy** |

Notice what happens to the third token. It started at one-sixth of the leader's probability ($`0.1`$ against $`0.6`$). At $`\tau = 2`$ it is up to $`0.19`$, about 40% of the leader. At $`\tau = 0.5`$ it is at $`0.02`$ against $`0.78`$ — **one thirty-sixth**. Low temperature does not merely favour the leader; it **annihilates the tail**, and the tail is where both the creativity and the mistakes live.

> **Analogy — and it is the literal origin.** This is the Boltzmann distribution from 19th-century thermodynamics, where $`\tau`$ is *actual temperature* (see Ch. 1's Did-you-know). Hot gas: molecules everywhere, high-energy states occupied, disorder. Cold gas: everything settles into the lowest-energy state. "Turning up the temperature to make the model more creative" is applying a physics equation with its original meaning nearly intact.

#### Top-$`k`$, top-$`p`$, and min-$`p`$, decoded

All three answer the same question — **which tokens are even allowed to be sampled?** — and they differ only in how they draw the line. This is called *truncation*, and it exists because the tail of a 128,000-token vocabulary contains a great deal of garbage whose probabilities are individually tiny but collectively not.

▸ **Do that arithmetic once, because it explains why truncation is mandatory.** Suppose 100,000 implausible tokens each carry probability $`10^{-6}`$. Individually negligible — one in a million. Together: $`100{,}000\times10^{-6} = 0.1`$, **a tenth of the total mass**. Generate 200 tokens and you will sample from that tail about **20 times**. And a single absurd token is not a local blemish: the model then conditions on its own mistake, and everything after it is a continuation of nonsense. **Truncation is not about polish. It is about not stepping off the path.**

**Top-$`k`$**: keep the $`k`$ best, renormalize, sample. Simple, and its flaw is that $`k`$ is a constant while the right answer is not:

| Context | Number of  plausible next tokens |
|---|---|
| "the capital of France is" | **1** |
| "she opened the door and saw" | **thousands** |

With $`k = 50`$, the first case admits 49 wrong answers into the pool. With $`k=50`$ in the second case, you have amputated a  open distribution down to 50 options and lost most of the available variety.

**Top-$`p`$ / nucleus**: keep adding tokens, most-probable first, until their probabilities **sum to $`p`$**. Then sample from that set.

| Context | Sorted probabilities | Set at $`p = 0.9`$ |
|---|---|---|
| "capital of France is" | $`0.95, 0.01, \dots`$ | **1 token** (0.95 already ≥ 0.9) |
| "she opened the door and saw" | $`0.03, 0.02, 0.02, \dots`$ | **hundreds of tokens** |

▸ **Same parameter, wildly different set sizes — that is the whole point.** The set is small when the model is confident and large when it isn't, because the *size of the set is determined by the distribution* rather than fixed in advance. This is exactly the fix for top-$`k`$'s flaw, and it is why nucleus sampling became the default.

> **Analogy.** Top-$`k`$ is "always interview the top 5 candidates." Top-$`p`$ is "interview candidates until you have covered 90% of the qualified pool" — which is 1 person when there's an obvious hire and 40 when the field is even.

**Min-$`p`$**: keep tokens with $`p_i \ge p_{\min}\cdot\max_j p_j`$ — a threshold **relative to the leader** rather than an absolute mass. With $`p_{\min}=0.1`$ and a top token at $`0.5`$, everything above $`0.05`$ survives; with a top token at $`0.02`$ (a flat distribution), everything above $`0.002`$ survives. It adapts the same way top-$`p`$ does, and it degrades more gracefully at high temperature, where flattening can otherwise let top-$`p`$'s cumulative sum sweep in a long tail of near-equal junk.

**Typical sampling**, decoded. **Surprisal** is $`-\log p_i`$ — how startled you are by a token, in nats. **Entropy** $`H`$ is the *average* surprisal (§1.4). Typical sampling keeps tokens whose surprisal is close to $`H`$ — that is, it **discards tokens that are too predictable as well as tokens that are too surprising.**

> **Analogy.** Conversation. Say only the obvious and you are boring; say only the bizarre and you are incoherent. Fluent speech runs at a roughly steady rate of new information — that is the "uniform information density" observation from psycholinguistics, and typical sampling is its direct implementation.

**Contrastive decoding**, in one sentence: ask a small weak model what *it* would say, and penalize those tokens — on the grounds that anything a 125M-parameter model finds obvious is generic, and what makes the large model worth having is where the two disagree. **It is a differencing operation: subtract the generic to isolate the specific.**

#### Examples and non-examples: deterministic generation

**✅  reproducible**

| Example | Why it qualifies |
|---|---|
| Greedy decoding, same weights, same batch composition, same kernel build, same GPU, run twice | Identical logits every step, and $`\arg\max`$ of identical logits is identical |
| Sampling with a fixed pseudo-random seed and everything else held constant | The RNG replays the same draws in the same order |
| Constrained decoding to a grammar that admits exactly one continuation | Only one token is legal, so the scores cannot matter |
| A cached response returned from a key–value store | Not generation at all, which is precisely why it reproduces perfectly |

**❌ Near-misses — look deterministic, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| `temperature=0` on a hosted API | The *selection rule* is deterministic. The *logits* are not: they depend on which other requests shared your batch, the tensor-parallel degree, the kernel version and the GPU model | Deterministic selection from non-reproducible scores |
| `temperature=0` on a Mixture-of-Experts model | Routing and expert capacity depend on the whole batch. A different neighbouring request changes which experts your tokens reach, and so changes your logits | Batch-dependent computation |
| Same seed, same prompt, different batch size | Floating-point addition is not associative: $`(a+b)+c \ne a+(b+c)`$ in general. Reducing a sum in a different order changes the last bits | Identical mathematics, different rounding |
| "Two logits differ by $`10^{-7}`$, so the output is the same" | $`\arg\max`$ has no tolerance. When the top two tokens are near-tied a last-bit difference flips the choice — and once one token differs, everything after it is a different sequence | A chaotic system with a discrete trigger |
| `temperature=0` meaning "the model is certain" | Temperature is applied to the logits *after* the model has finished thinking. It changes what you sample from, not what the model believes | A decoding setting mistaken for a model property |

▸ **The boundary:** determinism needs **two** things — a deterministic selection rule *and* bit-identical logits to select from. `temperature=0` buys you the first only, and in a shared serving environment you almost never get the second for free.

> **Common misconception.** *"Set temperature to 0 and you get the same output every time."* You get the same output every time *the logits come out the same*, and in production they routinely do not. Continuous batching (Ch. 17) means your request is computed alongside whatever else arrived in that millisecond; matrix reductions sum in whatever order the kernel picked for that batch shape; an MoE router's decisions depend on the other sequences present. Any of these perturbs a logit in the seventh decimal place — harmless, until two candidate tokens are near-tied, at which point $`\arg\max`$ amplifies $`10^{-7}`$ into a completely different paragraph. The belief is tempting because it is **locally true and easily checked**: run it twice on your laptop at batch size 1 and you get identical text, which feels like proof. It is proof about your laptop. The practical consequence is that "temperature 0" is a variance-reduction setting rather than a reproducibility guarantee, and evaluation harnesses that assume otherwise will report run-to-run noise as a difference between models.

> **Common misconception.** *"Beam search finds better text than sampling, so use it whenever quality matters."* Beam search finds **higher-probability** text, and past a point higher probability is *worse* text for open-ended generation — that is the likelihood trap above. Widening the beam, which by every intuition about search ought to help, reliably makes open-ended output blander and more repetitive; the effect is documented well enough to have a name, the *beam search curse*. There is a second, quieter failure: beams share prefixes, so asking for $`k = 8`$ candidate story openings usually returns eight near-identical strings rather than eight ideas. The belief is tempting because beam search  **is** the right answer for translation, grammatical error correction and speech-recognition rescoring — and those were the tasks it was built for and evaluated on for two decades. The field's decoding intuitions were formed on problems where the input pins down the output. **Search when the answer is determined by the input; sample when the output is  open.**

### Repetition control

- **Repetition penalty:** divide the logit of already-seen tokens by $`r\approx1.1`$.
- **Frequency / presence penalty** (OpenAI-style): subtract $`\alpha\cdot\text{count}`$ or a flat $`\beta`$ for any prior appearance.
- **No-repeat n-gram blocking:** hard ban on repeating any $`n`$-gram. Effective but can block legitimate repetition (names, code).

### Constrained decoding

#### Repetition control and constrained decoding, decoded

**The three repetition penalties differ in a way worth getting right**, because they behave differently and are frequently confused:

| Penalty | Mechanism | Behaviour |
|---|---|---|
| **Repetition penalty** | Divide the logit of any already-seen token by $`r\approx1.1`$ | **Multiplicative on the logit.** ⚠ Note the asymmetry: dividing a *negative* logit by 1.1 makes it *less* negative, i.e. more likely — so implementations must special-case the sign |
| **Frequency penalty** | Subtract $`\alpha\times(\text{times used so far})`$ | **Escalates.** The fifth use is penalized five times as hard as the first |
| **Presence penalty** | Subtract a flat $`\beta`$ for any prior appearance | **Binary.** Used once or used fifty times, same penalty |

▸ **Frequency penalty discourages overuse; presence penalty encourages topic change.** They are different instruments and setting them from the same slider is a common misconfiguration.

**No-repeat $`n`$-gram blocking** is the blunt one: maintain a set of every $`n`$-token sequence produced so far and set the logit of any token that would complete a repeat to $`-\infty`$. It is absolute — the repeat becomes impossible rather than unlikely.

> **Analogy.** A word game where you may not say any three-word phrase twice. It reliably stops you droning, and it also stops you saying "on the other hand" a second time when you legitimately needed to. That is the failure mode named in the text: names, code identifiers, chemical formulae, and legal boilerplate are *supposed* to repeat.

**Constrained decoding**, decoded. **CFG** here means **Context-Free Grammar** — a formal specification of which strings are valid (⚠ not classifier-free guidance, which is Ch. 20's CFG). The mechanism:

1. Compile the grammar (a JSON schema, a regular expression, a full grammar) into an **automaton** — a state machine that knows, from any point, which characters may legally come next.
2. At each generation step, look at the automaton's current state and compute the set of tokens that could continue a valid string.
3. Set every other logit to $`-\infty`$, so softmax gives them probability exactly zero.
4. Sample from what remains, and advance the automaton.

▸ **The guarantee this provides is categorically different from prompting.** "Please respond in valid JSON" makes malformed output *unlikely*; masking makes it **impossible** — there is no sequence of sampling outcomes that produces an unclosed brace, because the token was never on the menu. **When you need a contract rather than a tendency, you change the sampler, not the prompt.** The cost is real but modest: you must maintain the automaton and map grammar symbols onto the tokenizer's vocabulary, which is fiddly precisely because tokens do not respect character boundaries (Ch. 10).

Mask logits to enforce a grammar (JSON schema, regex, a CFG). Implemented by compiling the grammar to an automaton and, at each step, zeroing the logits of tokens that cannot continue a valid string. **This is how reliable structured output is actually achieved** — far more robust than asking nicely in the prompt.

> **Where this came from.** **Beam search** is not a language-model idea at all — it comes from **1970s speech recognition**, and the name is associated with the **Harpy** system built at Carnegie Mellon under Raj Reddy, where the "beam" was the narrow band of hypotheses kept alive as the search moved forward through an utterance. It arrived in neural sequence modelling through machine translation, where it fits the task, and was then applied to open-ended generation by inheritance rather than by argument — which is how the field spent several years generating repetitive text and treating it as a modelling problem.
>
> The diagnosis came from **"The Curious Case of Neural Text Degeneration"** (Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes and Yejin Choi, 2020), which introduced nucleus sampling. Its most persuasive piece of evidence was a plot rather than a theorem: the per-token probability of **human-written** text under a language model fluctuates constantly, while beam-search output is unnaturally smooth and uniformly high. The two curves look nothing alike. **The paper's real contribution was reframing the problem — the model was not broken, the decoding objective was wrong** — and it is a good example of how much can turn on measuring the right thing.

### Choosing

| Task | Setting |
|---|---|
| Factual QA, code, math | greedy or $`\tau\approx0.1`$ |
| Translation, summarization | beam 4–5 with length norm |
| Creative writing | $`\tau=0.8`$–1.0, top-$`p`$ 0.9–0.95 |
| Structured output | greedy + grammar constraint |
| Self-consistency / majority vote | $`\tau\approx0.7`$, many samples, then vote (Ch. 16) |

#### Reading the choosing table

Every row follows from one question: **is there a right answer, or a space of acceptable ones?**

| Task | Right answer exists? | Therefore |
|---|---|---|
| Factual QA, code, math | **Yes, exactly one** | Do not sample. Randomness can only introduce errors — there is nothing to gain from variety in `x = 7` |
| Translation, summarization | **Roughly one**, determined by the source | Search for it. Beam search, with length normalization |
| Creative writing | **No** | Sample, and sample warm. Variety is the product |
| Structured output | Yes, and it must **parse** | Greedy plus a hard grammar constraint |
| Self-consistency | Yes, but the *path* to it varies | Sample many warm paths, then vote on the destination |

▸ **The last row is the subtle one and worth understanding.** Self-consistency (Ch. 16) samples the same maths problem 40 times at $`\tau\approx0.7`$ and takes the majority answer. Why deliberately introduce randomness into a task with one right answer? Because **the reasoning path is where the variety belongs.** Different chains of thought reach the same correct answer by different routes, while errors are idiosyncratic and scatter. So the correct answer accumulates votes and the mistakes do not. **Temperature is being used to explore the space of derivations, not the space of answers** — and then the vote collapses it back down.

> **Analogy.** Asking forty people to independently work out a sum. Their arithmetic slips are all different; their correct answers are all the same. The mode of the answers is far more reliable than any single person, and the *disagreement* itself tells you how hard the problem was.

---

## 13.5 Evaluating language models

### Perplexity

▸ $$\mathrm{PPL} = \exp\left(-\frac1T\sum_t\log p_\theta(x_t\mid x_{<t})\right)$$

Interpretation: the effective branching factor — the model is as uncertain as if choosing uniformly among PPL options.

#### Reading perplexity in plain English

The formula is **the loss from §13.1, exponentiated**. That is the entire relationship: $`\mathrm{PPL} = e^{\mathcal{L}}`$.

So why bother? Because a loss of 2.3 is not an interpretable number, and $`e^{2.3} = 10`$ is. **"Effective branching factor" means: the model is exactly as uncertain as a fair die with PPL faces.**

**Check the intuition against a case you can verify.** Suppose the model has no idea, and spreads probability uniformly over a vocabulary of $`K`$ tokens. Then $`p = 1/K`$ at every step, so

$$\mathcal{L} = -\log(1/K) = \log K, \qquad \mathrm{PPL} = e^{\log K} = K$$

**A model that knows nothing has perplexity equal to its vocabulary size.** ✓ That is the ceiling, and it is why perplexity is meaningful in a way raw loss is not — you can immediately place a number on the scale.

| Perplexity | Reading |
|---|---|
| $`1`$ | Perfect. Assigns probability 1 to the true token every time |
| $`10`$–$`20`$ | A strong modern model on ordinary English prose |
| $`50{,}000`$ | Knows nothing; uniform over a 50k vocabulary |

> **Analogy.** A game of Twenty Questions. A perplexity of 16 means that at each word, the model is in the position of someone who has narrowed it to 16 equally-plausible candidates — four yes/no questions from certainty. A perplexity of 50,000 means they have not started.

▸ **The unit matters and the book uses nats.** $`\log`$ here is natural log, so the loss is in **nats**; exponentiating with $`e`$ recovers a count. If you use $`\log_2`$ the loss is in **bits** and you exponentiate with 2. The perplexity is the same number either way — $`e^{\ln K} = 2^{\log_2 K} = K`$ — but mixing the two is a classic and embarrassing error, and it is why a loss figure without a stated base is not a figure.

▸ **Pitfalls that make cross-model perplexity comparisons meaningless unless controlled:**
1. **Tokenizer-dependent.** Fewer, larger tokens ⇒ each carries more information ⇒ higher per-token perplexity, for an identical model. **Compare bits-per-byte or bits-per-character instead:** $`\mathrm{BPB} = \frac{\text{total nats}}{\ln 2 \cdot \text{total bytes}}`$.
2. **Data-dependent.** Perplexity on Wikipedia is not perplexity on code.
3. **Contamination.** If the eval text was in pretraining, perplexity is meaningless.
4. **Weakly correlated with usefulness after post-training.** RLHF typically *raises* perplexity while improving human preference scores.

#### Why tokenizer-dependence breaks the comparison

This is the pitfall that trips up the most people, so work it concretely.

Two models read the same sentence. Model A's tokenizer splits it into **10 tokens**; model B's into **20**. Both assign the same total probability to the sentence — they are, let us say, equally good models. Now compute per-token loss:

| | Tokens | Total nats | Loss per token | Perplexity |
|---|---|---|---|---|
| Model A (big tokens) | 10 | 30 | $`3.0`$ | $`20.1`$ |
| Model B (small tokens) | 20 | 30 | $`1.5`$ | $`4.5`$ |

**Model B looks four times better and the two models are identical.** All that changed is how the same total surprise was divided up. Larger tokens each carry more information, so each is harder to predict, so per-token perplexity is higher — **for an identical model.**

> **Analogy.** Comparing two delivery firms on "cost per parcel" when one ships in crates and the other in envelopes. The crate firm looks extortionate. Weigh the goods instead.

**Bits per byte (BPB) is the fix**, and it is the "weigh the goods" move: normalize by something the tokenizer cannot change.

$$\mathrm{BPB} = \frac{\text{total nats}}{\ln 2 \cdot \text{total bytes}}$$

- **Total nats** — the sum of $`-\log p`$ over the whole text. Tokenizer-independent, since it's the total probability of the same string.
- **Total bytes** — the length of the raw text in bytes. Also tokenizer-independent.
- $`\ln 2 \approx 0.693`$ — converts nats to bits, since $`1`$ bit $`= \ln 2`$ nats.

▸ **Both numerator and denominator are properties of the text and the model's distribution over it, not of the tokenizer.** So BPB compares two models fairly no matter how they chop up words — and it is directly comparable to Shannon's 1951 estimate of about 1 bit per character for English, which makes it interpretable in absolute terms rather than only relative ones.

**The other three pitfalls, briefly.** *Data-dependent*: perplexity on Python is not perplexity on Shakespeare, and a model tuned toward one will look worse on the other regardless of quality. *Contamination*: if the eval text was in pretraining, you are measuring memorization, and the number can be arbitrarily good while meaning nothing.

▸ **And the fourth is the one that changes how you should think about the metric entirely: RLHF typically *raises* perplexity while improving human preference scores.** That is not a paradox once you see what each measures. Perplexity asks *"how well does this model imitate the distribution of internet text?"* Post-training asks it to stop doing that — to be helpful, direct, and consistent, which is a *narrower* and less internet-like distribution. **The model gets worse at the thing perplexity measures because you asked it to.** Which means perplexity is a good instrument for pretraining and a poor one for anything after it.

> **Where this came from.** **Perplexity** entered language modelling from speech recognition at **IBM in the 1970s**, in the group led by **Frederick Jelinek** — the term appears in work by Jelinek, Robert Mercer, Lalit Bahl and James Baker, who needed a way to say how *hard* a recognition task was, independent of which recognizer you used. "Effective vocabulary size" was the concept; perplexity was the name.
>
> That group is also the origin of the field's most-repeated anecdote: Jelinek is said to have remarked that every time he fired a linguist, his speech recognizer's performance improved. **The exact wording is disputed and the story has been retold in several forms** — Jelinek himself later wrote about it and was uncertain of the phrasing — but the underlying dispute was real and consequential. The IBM group's statistical approach was deeply unfashionable against the rule-based, linguistically-grounded methods that dominated at the time, and the entire subsequent history of language modelling, this chapter included, is the statistical side of that argument having won.

#### Examples and non-examples: a meaningful perplexity comparison

**✅ Comparisons that mean something**

| Example | Why it qualifies |
|---|---|
| Step 10,000 versus step 50,000 of the same training run, same eval set | Tokenizer, data and provenance are all held fixed by construction. Only the weights moved |
| Two architectures trained with the **same** tokenizer on the same corpus, evaluated on the same held-out split | The one thing that differs is the thing you are asking about |
| Two models compared in **bits per byte** on the same text | The denominator is raw bytes, which no tokenizer can change |
| Perplexity on a held-out set with a documented decontamination pass | Removes the memorization confound rather than assuming it away |

**❌ Near-misses — look like a comparison, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "Model A reports perplexity 5.2, model B reports 18" from two different papers | Almost certainly different tokenizers **and** different eval corpora | Two unrelated numbers printed in the same font |
| A 32k-vocab model scoring lower per-token perplexity than a 128k-vocab one | Smaller tokens split the same total surprise across more predictions, so per-token loss falls. The 10-versus-20-token worked example above shows a factor of $`4.5`$ from tokenization alone | An artifact of how the text was chopped up |
| Perplexity on Wikipedia versus perplexity on Python | Different distributions; a model tuned toward one looks worse on the other regardless of quality | A statement about the eval set |
| A spectacular perplexity on a public benchmark | If the text was in pretraining you are measuring recall of memorized strings, and the number can be arbitrarily good while meaning nothing | Contamination |
| "Perplexity went up after RLHF, so we broke the model" | Post-training deliberately moves the model **off** the internet-text distribution that perplexity measures | Two metrics pointing at two different objectives |

▸ **The boundary:** a perplexity comparison is meaningful only when the tokenizer, the evaluation text and the training-data provenance are all held fixed. Change any one of the three and the number moves for reasons that have nothing to do with how good the model is.

> **Common misconception.** *"Lower perplexity means a better model."* Lower perplexity means the model assigns higher probability to **this particular text, chopped up this particular way**. That is a  excellent signal during pretraining, where the tokenizer and corpus are pinned down and the number is the cleanest single indicator of progress you will get. It is a poor signal for almost everything after: RLHF *raises* perplexity while raising human preference scores, because you have asked the model to stop imitating the average of the internet and start being helpful, direct and consistent — a narrower and much less internet-like distribution. **The model gets worse at what perplexity measures because you told it to.** The misconception is tempting because perplexity has every property a metric is supposed to have: it is cheap, continuous, principled, derived from the training loss itself, and it decreases monotonically with everything you know to be good during pretraining. It earns trust honestly and then keeps it past the point where it deserves it.

### Benchmarks and their problems

Knowledge (MMLU, GPQA), reasoning (GSM8K, MATH, AIME), code (HumanEval, SWE-bench), long context (RULER), agentic tasks, and human/model preference arenas.

▸ **The systematic issues to be able to name:**
- **Contamination.** Benchmarks leak into web-scraped pretraining data. Detect with n-gram overlap, canary strings, or performance gaps between original and perturbed variants.
- **Multiple-choice artifacts.** Models exhibit position and letter biases; scores shift when options are permuted.
- **Saturation.** Once a benchmark is near ceiling it stops discriminating.
- **Prompt sensitivity.** Reported scores vary by several points with formatting alone; few-shot vs zero-shot, chain-of-thought or not, all change the number.
- **Judge bias.** LLM-as-judge favours longer, more confident, and self-generated responses.

#### The benchmark names, and what each is actually testing

The book lists these as bare acronyms. Here is what they are, because you cannot reason about a score without knowing what produced it.

| Benchmark | Full form / origin | What it measures |
|---|---|---|
| **MMLU** | Massive Multitask Language Understanding | Multiple-choice across 57 subjects — law, medicine, history. Broad **knowledge** |
| **GPQA** | Graduate-level **G**oogle-**P**roof **Q**&**A** | Questions written by PhDs to be hard to look up. The name states the design goal |
| **GSM8K** | **G**rade **S**chool **M**ath, 8 thousand problems | Multi-step arithmetic word problems |
| **MATH** | — | Competition mathematics; much harder than GSM8K |
| **AIME** | American Invitational Mathematics Examination | A real human olympiad-qualifier exam, used as a benchmark |
| **HumanEval** | — | Write a Python function from a docstring; graded by **running the tests** |
| **SWE-bench** | Software Engineering benchmark | Resolve real GitHub issues in real repositories. Graded by whether the repo's tests pass |
| **RULER** | — | Long-context evaluation beyond needle-in-a-haystack (Ch. 12) |

▸ **Notice the split: HumanEval and SWE-bench are graded by *execution*, the rest by *string or letter match*.** Execution-graded benchmarks are far harder to game and far harder to contaminate usefully, because you cannot fake a passing test suite. **When comparing benchmark claims, ask first how the grading works** — it predicts how much the number is worth.

#### The five systematic issues, decoded

**Contamination.** Benchmarks are published on the web. Pretraining scrapes the web. So the answers may be in the training data, and the model may be recalling rather than reasoning. The three detection methods named:

- **$`n`$-gram overlap** — search the training corpus for verbatim spans of the test set.
- **Canary strings** — benchmark authors embed a unique random string in the files, so anyone can grep their corpus for it. An honour system, and one that only works if the corpus is inspectable.
- **Perturbed variants** — rewrite the question so the answer is unchanged but the surface form is not. **A model that memorized drops sharply; a model that understood does not.** This is the strongest of the three, because it needs no access to the training data at all.

**Multiple-choice artifacts.** Models have measurable preferences for particular answer positions and letters. **Permute the options and the score moves** — which means part of what is being measured is not knowledge. The standard defence is to evaluate all permutations, or at least to report that you checked.

**Saturation.** Once every serious model scores 88–91%, the benchmark has stopped being an instrument. The remaining gap is mostly mislabelled items and ambiguous questions, so you are measuring agreement with annotation errors. **This is why benchmarks have a shelf life measured in a couple of years**, and why the list above keeps being replaced.

**Prompt sensitivity.** The same model on the same benchmark scores several points differently depending on formatting, few-shot count, and whether chain-of-thought was requested. **So a comparison between two models evaluated under different harnesses is not a comparison.**

**Judge bias.** LLM-as-judge — using a strong model to score outputs — is cheap and scales, and it has known, measurable biases: toward **longer** answers, toward **confident** phrasing, and toward outputs from **itself or its own family**. All three reward things that are not correctness.

> **Analogy.** All five are versions of the same failure: **the measurement has become the target.** This is Goodhart's law — *when a measure becomes a target, it ceases to be a good measure* — named after the economist Charles Goodhart, who formulated it in the mid-1970s about monetary policy. Benchmark scores are optimized against by everyone in the field simultaneously, which is about as adversarial a pressure as a metric can face.

The right posture: **treat any single benchmark number as a noisy, gameable proxy**, and demand error bars (Ch. 3) and multiple independent evaluations.

▸ **Why the error-bar demand is not pedantry.** A benchmark of 1,000 items scored at 85% has a standard error of about $`\sqrt{0.85\times0.15/1000} \approx 1.1`$ **percentage points** (§1.3.1) — so a two-standard-error interval is roughly ±2.3 points before you account for prompt sensitivity, seed variation, or judge noise. **A headline "84.1 vs 82.9" is, quite straightforwardly, a tie.** A large fraction of published model comparisons are differences of this size reported as results.

---

## 13.6 The alternatives, and why decoder-only won

### BERT: masked language modelling

▸ $$\mathcal{L} = -\sum_{i\in\mathcal{M}}\log p(x_i\mid x_{\setminus\mathcal{M}})$$

Mask 15% of tokens (of those: 80% → `[MASK]`, 10% → random, 10% → unchanged — the last two exist because `[MASK]` never appears at fine-tuning time, so the model must not rely on seeing it).

**Advantage:** bidirectional context, which is strictly better for *understanding* tasks (classification, NER, retrieval).
▸ **Disadvantage:** only 15% of positions produce a learning signal, so it is ~6× less sample-efficient per FLOP, and it cannot generate autoregressively.

#### Reading the masked-language-modelling loss

$$\mathcal{L} = -\sum_{i\in\mathcal{M}}\log p(x_i\mid x_{\setminus\mathcal{M}})$$

| Piece | Read aloud | Meaning |
|---|---|---|
| $`\mathcal{M}`$ | "script M" | The **set of positions we chose to mask** — about 15% of them |
| $`i \in \mathcal{M}`$ | "i in M" | Sum only over the masked positions (§0.2) |
| $`x_{\setminus\mathcal{M}}`$ | "x, set-minus M" | **All the other tokens** — the visible ones. The backslash means "excluding" |
| $`p(x_i \mid x_{\setminus\mathcal{M}})`$ | "…given everything unmasked" | Recover the hidden token from **both** sides of it |

▸ **The one-sentence version: hide one word in seven and train the model to guess it from everything around it — before *and* after.** Compare §13.1, where the model may only look left.

> **Analogy.** Autoregressive training is reading a sentence one word at a time with the rest covered by a card, guessing what's next. Masked training is a **cloze test** — the exercise you did at school, where a passage has blanks and you fill them from context on both sides. Cloze tests were invented for exactly the reason BERT uses them: they measure comprehension of the whole passage rather than of its prefix.

**The 80/10/10 rule, decoded**, because the reason for it is  clever. Of the 15% chosen for masking: 80% become `[MASK]`, 10% become a random token, 10% are left alone.

Why not simply mask all 15%? Because **`[MASK]` never appears at fine-tuning time.** If the model only ever needs to produce a good representation at positions marked `[MASK]`, it will learn to do its work only there — and then in deployment, where nothing is masked, that machinery never fires.

- The **10% random** tokens force the model to distrust the token at a position: it might be corrupt, so build a representation from context regardless.
- The **10% unchanged** tokens mean the model can never be sure a position is *safe* either. It must model every position, all the time.

▸ **The whole scheme exists to close a train/test gap** — the model must not be allowed to condition on a symbol that only exists during training. **This is a general pattern worth carrying: whenever training introduces an artifact absent at deployment, some fraction of training must be run without it.** Dropout at test time, scheduled sampling, and exposure bias in sequence models are all the same problem in different clothes.

#### Why 15% masking costs a factor of six

The disadvantage in the text is a straight arithmetic consequence, and it is the number that decided the field.

Feed a 4096-token document into each model:

| Model | Learning signals per forward pass | Cost of that pass |
|---|---|---|
| Decoder-only (GPT) | **4095** — every position predicts the next | ~1 unit |
| Encoder-only (BERT) | **~614** — only the 15% masked | ~1 unit |

**Same compute, 6.7× fewer gradients.** The forward and backward passes cost essentially the same either way — you push all 4096 tokens through the whole stack in both cases — but BERT throws away 85% of the possible supervision.

> **Analogy.** Two students working through the same textbook. One does every exercise; the other does every seventh and skips the rest. Same hours at the desk, same pages turned, one-seventh the practice.

▸ **This — not any claim about which paradigm "understands" language better — is why the field consolidated on decoder-only.** When you are spending tens of millions of dollars on a single training run, a 6× difference in supervision per FLOP is not a preference, it is a decision. Bidirectional context is  better for understanding tasks; it is simply not worth 85% of your gradient signal when you are trying to train the largest model you can afford.

**Descendants:** RoBERTa (more data, no NSP, dynamic masking), ELECTRA (replaced-token detection — supervision on **100%** of tokens, far more efficient), DeBERTa (disentangled attention), ModernBERT.

#### The descendants, decoded

Each fixes something specific about BERT, and ELECTRA's fix is the interesting one.

- **RoBERTa** — *Robustly optimized BERT approach.* Same architecture, better training: more data, longer, and **no NSP** (Next Sentence Prediction, BERT's second objective of predicting whether two segments were adjacent — which turned out to contribute little). **Dynamic masking** re-chooses which tokens to mask each time a document is seen, rather than fixing the choice once in preprocessing. The headline result was that BERT had been **undertrained**, which is the same lesson Chinchilla delivered to the other side of the family three years later.
- **ELECTRA** — a  different objective. A small generator model replaces some tokens with plausible alternatives; the main model then classifies **every position** as original-or-replaced. ▸ **Because the judgement is made at all $`T`$ positions rather than 15%, the sample-efficiency gap with autoregressive training closes.** It is the same insight as the 6× arithmetic above, applied from the encoder side. (The name is a backronym: *Efficiently Learning an Encoder that Classifies Token Replacements Accurately.*)
- **DeBERTa** — *disentangled attention*: represent content and position as **separate vectors** and compute attention from both, rather than summing them into one. This is the "second microphone" fix from Ch. 12 §12.2, arrived at independently on the encoder side.
- **ModernBERT** — a contemporary re-training of the encoder recipe with the architectural improvements the decoder side accumulated (RoPE, longer context, better data), demonstrating that most of BERT's apparent obsolescence was a lack of maintenance rather than a limit of the design.

▸ **Encoder models are not obsolete.** For classification and especially for retrieval embeddings, a 300M-parameter encoder is often better *and* 100× cheaper than an LLM. Knowing when *not* to use a generative model is a mark of judgement.

#### Why a 300M encoder beats an LLM at retrieval

Both the "better" and the "cheaper" claims deserve support, because "use the small model" sounds like a compromise and here it isn't.

**Cheaper** is straightforward: 300M parameters against 70B is roughly 230× fewer FLOPs per token, and an embedding model runs **once per document** at index time and **once per query** — there is no generation loop, no KV cache, no sampling. You can embed a hundred million documents on hardware that could not serve a single LLM conversation.

**Better** is the interesting part, and it comes down to bidirectionality. A retrieval embedding must summarize an entire passage into one vector.

- A **decoder** builds each token's representation from the **left context only**. The final token has seen everything, but every earlier token was built while blind to what followed.
- An **encoder** builds every token's representation from the **whole passage at once**.

> **Analogy.** Summarizing a document while reading it once, forwards, versus summarizing it after having read the whole thing. The second is obviously better, and the first is what a causal model is structurally required to do.

▸ **The trade in §13.1 runs in both directions, and this is the other direction.** Causal masking is what buys dense supervision; the price is that every representation is one-sided. When your task is *generation*, you have no choice — the future is not available. When your task is *understanding a passage that already exists*, giving up the future is pure loss. **Match the mask to the task.**

### T5: span corruption, encoder–decoder

Mask contiguous spans, replace with sentinel tokens, decode the missing spans. Casts every task as text-to-text.

#### T5's span corruption, decoded

**T5** is **Text-To-Text Transfer Transformer** — five T's, hence the name. Its objective sits between BERT's and GPT's.

BERT masks *individual tokens*. T5 masks **contiguous runs** and replaces each entire run with a single **sentinel** token — a placeholder like `<X>`, `<Y>`. The decoder then produces the missing spans, in order:

```
Input:   "Thank you <X> me to your party <Y> week."
Target:  "<X> for inviting <Y> last"
```

▸ **Masking a whole span is harder than masking a token, and that is the point.** Given "the capital of <MASK> is Paris," the single missing word is nearly forced. Given "the capital <X> Paris," the model must produce *"of France is"* — deciding the length, the syntax and the content together. **Span corruption forces the model to generate, not merely to classify.**

**"Casts every task as text-to-text"** is T5's other contribution and it was the more influential one. Every task — classification, similarity scoring, translation, question answering — is expressed as a string in and a string out, with the task named in a prefix (`"translate English to German: …"`, `"cola sentence: …"`). No task-specific heads, no task-specific losses. **One model, one loss, one interface.** That framing is now so standard that it is invisible; in 2019 it was a deliberate argument.

> **Where this came from.** **BERT** (Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova at Google, 2018) is an acronym constructed to land on a name: **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. It follows **ELMo** — Embeddings from Language Models (Peters and colleagues at the Allen Institute, 2018) — and the Sesame Street theme was then taken up deliberately across the field, producing ERNIE, Grover, Big Bird and KERMIT within about two years. The naming became enough of a running joke that later papers commented on it, and it eventually stopped, which is roughly the life cycle of every naming convention in this field.
>
> **T5** (Colin Raffel and colleagues at Google, 2019) came with a dataset that arguably mattered more than the model: **C4**, the Colossal Clean Crawled Corpus, a filtered version of Common Crawl released publicly. A great deal of subsequent open language-model work was trained on C4 or on things built in its image, which is a recurring pattern worth noticing — **the artifact from a paper that outlives it is often the dataset, not the method.**

### The comparison

| | Decoder-only | Encoder-only | Encoder–decoder |
|---|---|---|---|
| Generation | ✓ | ✗ | ✓ |
| Bidirectional context | ✗ | ✓ | ✓ (source only) |
| Training signal density | 100% | 15% | ~15% of source |
| In-context learning | strong | none | weak |
| Best for | general LLM | classification, retrieval | translation, ASR |

#### Reading the comparison table

Every row of this table traces to **one** decision: *which tokens is each position allowed to look at?* Everything else is downstream.

| Architecture | The mask | Consequence |
|---|---|---|
| Decoder-only | Each position sees **only the past** | Can generate; every position is a training target; representations are one-sided |
| Encoder-only | Each position sees **everything** | Cannot generate (nothing is hidden, so there is nothing to predict); must artificially mask to create targets |
| Encoder–decoder | Source is fully visible; target is causal | Best of both for translation; two stacks to train and serve |

▸ **Note the deep symmetry: you cannot have both full context and dense supervision.** If every position sees everything, there is nothing left to predict — the answer is already visible. Hiding tokens is what creates a prediction problem, and hiding *the future specifically* is the one choice that creates a prediction problem at **every** position simultaneously. **Causal masking is not one arbitrary way to hide things; it is the only one that leaves no position wasted.**

**Why "in-context learning: none" for encoders.** Induction heads (§13.3) work by matching the current token against earlier ones and copying what followed. That is a *sequential*, *causal* operation — "what came after" is only a meaningful question if there is a direction. A bidirectional model has no fixed direction of flow, and no mechanism that corresponds to it.

**Why encoder–decoder is "best for translation, ASR."** Both are **transduction** tasks: a complete input exists before you start, and an output must be produced from it. The source deserves full bidirectional attention (it's all there — why look at only half of it?), the target must be causal (you're generating). The architecture matches the task shape exactly.

> **Analogy.** A translator working from a printed page versus a simultaneous interpreter in a booth. The one with the page reads the whole sentence, including the verb German puts at the end, before saying anything. The interpreter must start speaking before the sentence is over. An encoder–decoder is the first; a decoder-only model doing translation is the second, and does surprisingly well anyway — which is most of the reason encoder–decoders lost.

▸ **The one-sentence answer to "why did decoder-only win?"** — not because it is better at understanding, which it isn't, but because **one stack, one mask, one loss, and a training target at every single position** is a simpler and more compute-efficient thing to scale, and at the scales that turned out to matter, scaling efficiency dominated architectural fit.

#### Examples and non-examples: tasks that need a decoder-only LLM

**✅  examples**

| Example | Why it qualifies |
|---|---|
| Drafting an email, continuing a code file, holding a conversation | The output is new text of unbounded length, produced one token at a time |
| Adapting to a new task from three examples in the prompt | In-context learning requires a generative model over the whole prompt |
| Agentic tool use — decide, call, read the result, decide again | The number of steps is not known in advance, so the output cannot be a fixed-size head |
| Chain-of-thought reasoning | The intermediate tokens *are* the computation; there is nowhere else for them to live |
| Rewriting a paragraph in a different register | The target is text, and it is open-ended |

**❌ Near-misses — reach for an LLM, and shouldn't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Dense retrieval over 50 million documents | You need one vector per document, computed **once, offline**. Fifty million 70B forward passes is not a budget, it is a rounding error away from the pretraining bill | A bidirectional encoder — 100–300M parameters, one pass per document, and better embeddings |
| Reranking the top 50 candidates for a query | A cross-encoder reads query and document **together with bidirectional attention** and scores the pair; that is strictly more information than a causal model gets | A cross-encoder reranker |
| Named-entity recognition against a fixed schema | Deciding whether "Washington" is a person or a place needs the words on **both** sides. Causal masking forbids the right-hand ones | An encoder with a token-classification head |
| Sentiment classification when you have 100k labelled examples | A fine-tuned 100M-parameter encoder matches or beats a prompted frontier model here, at a thousandth of the cost and a millisecond of latency | Supervised fine-tuning of an encoder |
| "Use the LLM for embeddings, it's bigger so it must be better" | Under a causal mask **no token has seen its future**, so a document's representation is structurally one-sided. Bidirectionality is not a size you can scale into | An architecture mismatch, not a capacity shortfall |

▸ **The boundary:** decoder-only wins when the output is **generated** — open-ended, variable length, one token at a time. Encoders win when the output is a **fixed-size judgement about a complete input you already hold**, because then there is no reason to forbid a token from looking rightward, and forbidding it costs you.

> **Common misconception.** *"A bigger model is better at everything, so a 70B LLM must beat a 300M encoder at search."* It loses, and for two independent reasons that both survive any amount of scaling. First, **economics**: a retrieval index needs one embedding per document, computed once over the whole corpus, and the cost gap between a 300M encoder and a 70B decoder is roughly two hundred–fold on a job you must run over every document you own. Second, and more fundamental, **the causal mask**: a decoder-only model is trained so that no position ever attends to a later one, which means the representation of a document is built without the second half informing the first. An encoder trained with bidirectional attention was built for exactly this job. **Scale does not fix an architecture that has been forbidden from looking rightward.** The misconception is tempting because scaling laws are real, dramatic and correctly famous — and because for *generation* the intuition is simply true, so it gets carried across a boundary it does not hold at. This is why BERT-family encoders, declared obsolete somewhere around 2021, still quietly run most of the world's retrieval, reranking and classification.

---

## 13.7 The practical checklist for training a GPT

1. **Data** — dedup aggressively, filter for quality, mix domains deliberately (Ch. 14).
2. **Tokenizer** — train it on your actual data distribution; check compression rate.
3. **Packing** — concatenate documents to fill the context, separated by EOS, with block-diagonal attention masks so documents can't attend across boundaries. Wasting 40% of every batch on padding is a common and expensive mistake.
4. **Architecture** — the recipe in §13.2.
5. **Init** — $`\mathcal{N}(0,0.02^2)`$, with residual projections scaled by $`1/\sqrt{2L}`$.
6. **Optimizer** — AdamW, $`\beta=(0.9,0.95)`$, $`\lambda=0.1`$, grad clip 1.0, warmup then cosine or WSD.
7. **Batch size** — scale up during training (small batches early are more sample-efficient; large batches later are more compute-efficient).
8. **Precision** — bf16 with fp32 master weights (Ch. 14).
9. **Monitor** — loss, grad norm, LR, the fraction of tokens contributing, MFU, and per-domain validation loss.
10. **Checkpoint often**, and keep an EMA.

#### The checklist, decoded

Three of these carry more content than their one-line form suggests.

**Item 3 — packing, and why 40% waste is easy to hit.** Documents have different lengths. A naive batch pads everything to the longest, so a batch containing one 4096-token document and thirty-one 500-token documents spends most of its compute multiplying zeros. **Packing** concatenates documents end to end until the context is full, separated by an **EOS** (End Of Sequence) token.

▸ **But concatenation creates a new bug**, and this is the part people miss: document B now sits *after* document A in the same context, so attention lets B's tokens read A. The model learns from a context that will never occur at deployment, and can even learn to expect it. The fix is a **block-diagonal attention mask** — each document may attend only within itself. Without it, packing trades a compute waste for a correctness bug.

> **Analogy.** Filling a shipping container efficiently by removing the boxes and pouring everything in loose. You have solved the empty-space problem and created a sorting problem. The block-diagonal mask is the dividers.

**Item 5 — why residual projections are scaled by $`1/\sqrt{2L}`$.** Every layer adds its output onto the residual stream. Add $`L`$ independent contributions of variance $`\sigma^2`$ each and the total variance is $`L\sigma^2`$ — so the stream's magnitude **grows like $`\sqrt{L}`$** with depth. At $`L=96`$ that is a factor of about 10 accumulating from top to bottom, purely from stacking.

Scale each layer's output projection by $`1/\sqrt{2L}`$ at initialization (the 2 is because a transformer block writes to the stream twice: once from attention, once from the feed-forward network) and the contributions shrink at exactly the rate the count grows. **The residual stream then starts at roughly unit scale no matter how deep the model is.** This is the same $`\sqrt{n}`$ law as §1.3.1 and §1.1.5 — random things accumulate like $`\sqrt{n}`$, not $`n`$, because they partially cancel.

**Item 7 — why batch size should grow during training.** Early on, the gradient signal is enormous compared with the noise: almost any batch points roughly downhill, so a small batch is fine and you get many more updates per unit of compute. Late in training, the model is near a minimum and the true gradient is small — now the batch noise dominates, and you need a large batch to average it away and see the real direction.

> **Analogy.** Navigating to a distant mountain. At the start you barely need a compass; any rough bearing works, so take fast steps. Near the summit you are trying to detect a slight upward slope, and you must average many careful readings to tell it from noise.

**Item 10 — EMA** is **Exponential Moving Average**: keep a slowly-updated running average of the weights alongside the live ones, $`\theta_{\text{EMA}} \leftarrow \beta\theta_{\text{EMA}} + (1-\beta)\theta`$. The averaged weights are usually slightly better than any individual checkpoint, because averaging along the trajectory cancels the last-step noise and lands nearer the centre of the basin rather than on its wall.

---

## Did you know?

- **Shannon built the first language model in 1948, by hand, with a book.** To generate word-pair-statistical English he would open a book at random, find a word, then flip to another random page and scan for the next occurrence of it, and write down what followed. The output looked more like English at every order of approximation — the demonstration that started the field, produced with a paperback and a pencil.

- **The first serious measurement of English's predictability used humans as the language model.** In *Prediction and Entropy of Printed English* (1951), Shannon had people guess the next letter of a text one at a time and used their error rates to estimate the entropy of English at roughly 1 bit per character. Bits-per-character scores for modern models are still compared against that number.

- **GPT-1 was never peer-reviewed.** It was released as a technical report and a blog post, and it is among the most consequential documents in the field's history. GPT-2 followed with a **staged release** — OpenAI initially withheld the full 1.5B model over misuse concerns, publishing it only nine months later.

- **BERT is a Sesame Street joke that got out of hand.** It followed ELMo (Embeddings from Language Models), and the theme was then taken up deliberately: ERNIE, Grover, Big Bird and KERMIT all appeared within about two years, before the field quietly gave it up.

- **Beam search comes from 1970s speech recognition, not from language modelling.** The name is associated with the Harpy system at Carnegie Mellon, where the "beam" was the narrow band of hypotheses kept alive while scanning an utterance. It was inherited by neural machine translation, where it fits, and then applied to open-ended text generation by habit, where it does not.

- **Searching harder for the most likely text makes the text worse.** The highest-probability continuation of most prompts is degenerate repetition. Human writing is *locally surprising* — per-token probability under a language model fluctuates constantly — while beam-search output is unnaturally smooth. The plot showing those two curves side by side did more to change decoding practice than any theorem.

- **Randomizing the labels in a few-shot prompt barely hurts performance.** Show a model examples with deliberately wrong answers and it still does the task far better than with no examples at all. What the demonstrations convey is the format, the label space and the input distribution — not the mapping.

- **In-context learning appears in a narrow window during training, and you can see it happen.** Induction heads form in that same window, visible as a small bump in an otherwise smooth loss curve. Ablate the heads and the capability disappears. It is one of the few clean causal links from a specific circuit to a specific behaviour.

- **Temperature is literally temperature.** $`p_i \propto \exp(z_i/\tau)`$ is the Boltzmann distribution from 1868 statistical mechanics, where $`\tau`$ is the temperature of a gas. Turning up an LLM's temperature to make it more creative applies a 19th-century physics equation with its original meaning nearly intact: more heat, more disorder.

- **Perplexity was invented at IBM to describe how hard a speech-recognition task was**, in Frederick Jelinek's group in the 1970s. That group is also the source of the field's most-repeated anecdote, about performance improving each time a linguist was fired — the exact wording is disputed and has been retold in several forms, but the underlying methodological argument was real, and the statistical side won it decisively.

- **RLHF makes perplexity worse and models better.** Post-training pushes the model away from imitating internet text toward being helpful and direct — a narrower, less internet-like distribution. The metric gets worse because you asked it to. Any evaluation that treats perplexity as a quality score after post-training is measuring the wrong thing.

- **A 300M-parameter encoder often beats a 70B LLM at retrieval, and costs ~100× less.** A causal model builds every token's representation from the left only; an encoder sees the whole passage. When the task is understanding text that already exists, giving up the future is pure loss — and the small model wins on both axes at once.

---

## Check for Understanding

**Next-token prediction is an exact factorization of the joint distribution that supervises every position, which makes it both the most sample-efficient and the most general objective available — and the capabilities that seem to exceed it, like in-context learning, are mechanistically explicable (induction heads) rather than mysterious, while the quality of what you actually see depends as much on the decoding strategy as on the model.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is "the probability of a document = the product of next-token probabilities" an exact identity rather than an approximation?** Use the card-dealing framing, not the formula.
2. **Why does predicting the next token force a model to acquire facts, arithmetic and deduction, when the loss never mentions any of them?**
3. **Why does causal masking give you a training signal at every position, and why is that worth roughly 6× over BERT's 15% masking?**
4. **Why does BERT replace 10% of its chosen tokens with random ones and leave another 10% alone?** (Answer in terms of what exists at training time but not at deployment.)
5. **What is in-context learning doing, given that no weight is ever updated?**
6. **What is an induction head, in terms of two library assistants?** Why does it need two layers rather than one?
7. **Why does randomizing the labels in a few-shot prompt barely hurt, and what does that tell you the examples are for?**
8. **Why is greedy decoding not the same as finding the most likely sentence?** Give the two-step example.
9. **Why is beam search right for translation and wrong for storytelling?**
10. **What does top-$`p`$ fix about top-$`k`$?** Say it in terms of how many plausible next words there are after "the capital of France is" versus "she opened the door and saw."
11. **Why can two identical-quality models have perplexities that differ by 4×?** What do you measure instead?
12. **Why does RLHF make perplexity worse while making the model better?**
13. **Why would you sample a maths problem forty times at temperature 0.7 when there is exactly one right answer?**
14. **Why did decoder-only win, given that bidirectional models  understand text better?**

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 14 — Training LLMs at Scale](14-training-llms-at-scale.md)
