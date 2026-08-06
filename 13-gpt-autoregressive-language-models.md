# Chapter 13 — GPT: Autoregressive Language Modelling

> **Prerequisites:** Ch. 10, 11, 12.

---

## 13.1 The objective

### The one-line idea

Predict the next token. Do it on enough text and enough parameters, and essentially every language capability appears as a side effect.

### The analogy

Learning to play music by being handed millions of scores with the last note covered and being asked to guess it. To get good, you cannot just memorize — you have to internalize key signatures, chord progressions, phrasing, genre conventions, and the composer's habits. Prediction forces understanding because understanding is the cheapest way to predict.

### The math

Factorize the joint distribution by the chain rule of probability — **exactly, with no approximation**:

▸ $$p_\theta(x_{1:T}) = \prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\qquad \mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

**Why this is such a good objective:**

1. **Exact likelihood.** No bound, no adversary, no partition function. Compare VAEs (a bound), GANs (no likelihood at all), EBMs (intractable $Z$) — Ch. 19.
2. **Dense supervision.** One forward pass over $T$ tokens yields $T$ prediction targets (thanks to causal masking, Ch. 11 §11.4).
3. **Self-supervised.** The labels are the data.
4. **Task-universal.** Any task expressible as text — translation, QA, summarization, code, reasoning — is a conditional distribution $p(\text{answer}\mid\text{question})$, which the model already represents.

▸ **Why "just predicting the next token" is not a limitation:** predicting the token after a long argument requires having tracked the argument. Predicting the last word of a murder mystery's reveal requires having solved it. The objective is shallow; the competence required to minimize it is not.

---

## 13.2 The architectural lineage

| Model | Year | $N$ | $L$ | $d$ | $T$ | The change that mattered |
|---|---|---|---|---|---|---|
| GPT-1 | 2018 | 117M | 12 | 768 | 512 | generative pretraining + discriminative fine-tuning |
| GPT-2 | 2019 | 1.5B | 48 | 1600 | 1024 | **pre-LN**; scale; zero-shot task transfer |
| GPT-3 | 2020 | 175B | 96 | 12288 | 2048 | scale; **in-context learning** |
| Chinchilla | 2022 | 70B | 80 | 8192 | 2048 | compute-optimal data scaling (Ch. 15) |
| LLaMA | 2023 | 7–65B | | | 2048 | RMSNorm + SwiGLU + RoPE; small models trained far past "optimal" |
| LLaMA-3 | 2024 | 8–405B | | | 8k–128k | 15T tokens; GQA; 128k vocab |
| Frontier (2025–26) | | sparse MoE | | | 128k–1M+ | MoE, long context, reasoning post-training |

▸ **The stable modern recipe**, worth being able to recite: decoder-only, pre-norm **RMSNorm**, **SwiGLU** FFN with $d_{\text{ff}}=\frac83 d$, **RoPE**, **GQA**, no biases anywhere, no dropout during pretraining, weight-tied or untied embeddings, AdamW with cosine or WSD schedule, bf16 with fp32 master weights.

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
1. A **previous-token head** in layer $\ell$ writes information about token $t-1$ into position $t$'s residual stream.
2. An **induction head** in layer $\ell' > \ell$ uses the current token as a query to match against that written information, locating earlier occurrences, and copies the token that followed.

**Minimum two attention layers.** One-layer attention-only transformers cannot form induction heads and cannot do in-context learning.

▸ **The decisive evidence:** during training there is a narrow window where in-context learning ability jumps sharply, and induction heads form *in that exact window*, visible as a bump in the loss curve. Ablating induction heads destroys in-context learning. This is one of the strongest mechanistic-to-behavioural links established in interpretability.

### Other framings

- **Implicit gradient descent:** a transformer layer can implement one step of gradient descent on a linear regression problem defined by the in-context examples. Provable in constructed settings; suggestive but not established for real models.
- **Bayesian task inference:** the pretraining distribution is a mixture over latent "tasks"; the prompt is evidence, and the model performs implicit posterior inference over which task it's in. Explains why format matters more than label correctness.

▸ **The empirical finding that discriminates between accounts:** for many tasks, **randomizing the labels in the examples barely hurts performance** (Min et al., 2022). What the demonstrations mainly convey is the *format, label space, and input distribution* — not the input→output mapping. That is much more consistent with task-location than with in-context gradient descent. State this if asked; it's the kind of detail that signals real familiarity.

---

## 13.4 Decoding strategies

The model gives you $p(x_t\mid x_{<t})$. Turning it into text is a separate, consequential design choice.

### Deterministic

**Greedy:** $x_t=\arg\max p$. Fast; repetitive and dull; **not** the highest-probability *sequence* (a locally-best token can foreclose a better continuation).

**Beam search:** keep $k$ partial sequences, expand all, retain the top $k$ by cumulative log-probability.
- Needs **length normalization**: $\frac{1}{|y|^\alpha}\sum\log p$, $\alpha\approx0.6$–1.0. Without it, shorter sequences always win because every added token multiplies by a probability $<1$.
- ▸ **Beam search is right for translation and summarization, and wrong for open-ended generation.** In open-ended text, higher sequence probability correlates with *worse* human-judged quality past a point — the "likelihood trap." The highest-probability continuation of most prompts is degenerate repetition.

### Stochastic

**Temperature:** $p_i\propto \exp(z_i/\tau)$. $\tau<1$ sharpens, $\tau>1$ flattens, $\tau\to0$ is greedy.
▸ Note $\tau$ acts on **logits**, so it is not a linear reweighting of probabilities: $p_i^{(\tau)} \propto p_i^{1/\tau}$.

**Top-$k$:** sample from the $k$ most probable tokens. Problem: $k$ is fixed, but the appropriate number of plausible tokens varies enormously by context (after "the capital of France is" there is one; after "she opened the door and saw" there are thousands).

**Top-$p$ / nucleus (Holtzman et al.):** sample from the smallest set $S$ with $\sum_{i\in S}p_i\ge p$. ▸ **Adaptive to the distribution's entropy** — this is precisely the fix for top-$k$'s flaw, and it is the reason nucleus sampling became the default. Typical $p=0.9$–0.95.

**Min-$p$:** keep tokens with $p_i \ge p_{\min}\cdot\max_j p_j$. A relative threshold; more robust at high temperature.

**Typical sampling:** keep tokens whose surprisal $-\log p_i$ is close to the distribution's entropy $H$ — an information-theoretic criterion motivated by the observation that natural human text has locally near-uniform information density.

**Contrastive decoding / search:** penalize tokens that a smaller "amateur" model also finds likely, or penalize similarity to already-generated context. Reduces degeneration without sacrificing coherence.

### Repetition control

- **Repetition penalty:** divide the logit of already-seen tokens by $r\approx1.1$.
- **Frequency / presence penalty** (OpenAI-style): subtract $\alpha\cdot\text{count}$ or a flat $\beta$ for any prior appearance.
- **No-repeat n-gram blocking:** hard ban on repeating any $n$-gram. Effective but can block legitimate repetition (names, code).

### Constrained decoding

Mask logits to enforce a grammar (JSON schema, regex, a CFG). Implemented by compiling the grammar to an automaton and, at each step, zeroing the logits of tokens that cannot continue a valid string. **This is how reliable structured output is actually achieved** — far more robust than asking nicely in the prompt.

### Choosing

| Task | Setting |
|---|---|
| Factual QA, code, math | greedy or $\tau\approx0.1$ |
| Translation, summarization | beam 4–5 with length norm |
| Creative writing | $\tau=0.8$–1.0, top-$p$ 0.9–0.95 |
| Structured output | greedy + grammar constraint |
| Self-consistency / majority vote | $\tau\approx0.7$, many samples, then vote (Ch. 16) |

---

## 13.5 Evaluating language models

### Perplexity

▸ $$\mathrm{PPL} = \exp\left(-\frac1T\sum_t\log p_\theta(x_t\mid x_{<t})\right)$$

Interpretation: the effective branching factor — the model is as uncertain as if choosing uniformly among PPL options.

▸ **Pitfalls that make cross-model perplexity comparisons meaningless unless controlled:**
1. **Tokenizer-dependent.** Fewer, larger tokens ⇒ each carries more information ⇒ higher per-token perplexity, for an identical model. **Compare bits-per-byte or bits-per-character instead:** $\mathrm{BPB} = \frac{\text{total nats}}{\ln 2 \cdot \text{total bytes}}$.
2. **Data-dependent.** Perplexity on Wikipedia is not perplexity on code.
3. **Contamination.** If the eval text was in pretraining, perplexity is meaningless.
4. **Weakly correlated with usefulness after post-training.** RLHF typically *raises* perplexity while improving human preference scores.

### Benchmarks and their problems

Knowledge (MMLU, GPQA), reasoning (GSM8K, MATH, AIME), code (HumanEval, SWE-bench), long context (RULER), agentic tasks, and human/model preference arenas.

▸ **The systematic issues to be able to name:**
- **Contamination.** Benchmarks leak into web-scraped pretraining data. Detect with n-gram overlap, canary strings, or performance gaps between original and perturbed variants.
- **Multiple-choice artifacts.** Models exhibit position and letter biases; scores shift when options are permuted.
- **Saturation.** Once a benchmark is near ceiling it stops discriminating.
- **Prompt sensitivity.** Reported scores vary by several points with formatting alone; few-shot vs zero-shot, chain-of-thought or not, all change the number.
- **Judge bias.** LLM-as-judge favours longer, more confident, and self-generated responses.

The right posture: **treat any single benchmark number as a noisy, gameable proxy**, and demand error bars (Ch. 3) and multiple independent evaluations.

---

## 13.6 The alternatives, and why decoder-only won

### BERT: masked language modelling

▸ $$\mathcal{L} = -\sum_{i\in\mathcal{M}}\log p(x_i\mid x_{\setminus\mathcal{M}})$$

Mask 15% of tokens (of those: 80% → `[MASK]`, 10% → random, 10% → unchanged — the last two exist because `[MASK]` never appears at fine-tuning time, so the model must not rely on seeing it).

**Advantage:** bidirectional context, which is strictly better for *understanding* tasks (classification, NER, retrieval).
▸ **Disadvantage:** only 15% of positions produce a learning signal, so it is ~6× less sample-efficient per FLOP, and it cannot generate autoregressively.

**Descendants:** RoBERTa (more data, no NSP, dynamic masking), ELECTRA (replaced-token detection — supervision on **100%** of tokens, far more efficient), DeBERTa (disentangled attention), ModernBERT.

▸ **Encoder models are not obsolete.** For classification and especially for retrieval embeddings, a 300M-parameter encoder is often better *and* 100× cheaper than an LLM. Knowing when *not* to use a generative model is a mark of judgement.

### T5: span corruption, encoder–decoder

Mask contiguous spans, replace with sentinel tokens, decode the missing spans. Casts every task as text-to-text.

### The comparison

| | Decoder-only | Encoder-only | Encoder–decoder |
|---|---|---|---|
| Generation | ✓ | ✗ | ✓ |
| Bidirectional context | ✗ | ✓ | ✓ (source only) |
| Training signal density | 100% | 15% | ~15% of source |
| In-context learning | strong | none | weak |
| Best for | general LLM | classification, retrieval | translation, ASR |

---

## 13.7 The practical checklist for training a GPT

1. **Data** — dedup aggressively, filter for quality, mix domains deliberately (Ch. 14).
2. **Tokenizer** — train it on your actual data distribution; check compression rate.
3. **Packing** — concatenate documents to fill the context, separated by EOS, with block-diagonal attention masks so documents can't attend across boundaries. Wasting 40% of every batch on padding is a common and expensive mistake.
4. **Architecture** — the recipe in §13.2.
5. **Init** — $\mathcal{N}(0,0.02^2)$, with residual projections scaled by $1/\sqrt{2L}$.
6. **Optimizer** — AdamW, $\beta=(0.9,0.95)$, $\lambda=0.1$, grad clip 1.0, warmup then cosine or WSD.
7. **Batch size** — scale up during training (small batches early are more sample-efficient; large batches later are more compute-efficient).
8. **Precision** — bf16 with fp32 master weights (Ch. 14).
9. **Monitor** — loss, grad norm, LR, the fraction of tokens contributing, MFU, and per-domain validation loss.
10. **Checkpoint often**, and keep an EMA.

---

## Check for Understanding

**Next-token prediction is an exact factorization of the joint distribution that supervises every position, which makes it both the most sample-efficient and the most general objective available — and the capabilities that seem to exceed it, like in-context learning, are mechanistically explicable (induction heads) rather than mysterious, while the quality of what you actually see depends as much on the decoding strategy as on the model.**

---

**Next:** [Chapter 14 — Training LLMs at Scale](14-training-llms-at-scale.md)
