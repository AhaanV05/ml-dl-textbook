# Chapter 18 — Retrieval, RAG, Tools & Agents

> **Prerequisites:** Ch. 10 (embeddings, BM25, ANN search), Ch. 13.

---

## 18.1 Why retrieval

### The one-line idea

Parameters are an expensive, slow-to-update, unverifiable place to store facts; a database is a cheap, instantly-updatable, citable one. Retrieval separates knowledge from reasoning.

### The analogy

A doctor who has memorized every drug interaction versus one who reasons well and consults the formulary. The second is more reliable, easier to update when a new drug is approved, and can show you the page. The first is faster only when they happen to be right.

### What retrieval actually fixes

| Problem | How retrieval helps |
|---|---|
| Knowledge cutoff | fetch current documents |
| Hallucination | ground generation in retrieved text; enable citation |
| Private/proprietary data | never needs to enter training |
| Attribution | the source is in the context |
| Update cost | re-index instead of re-train |
| Long-tail facts | parameters are a poor store for rare facts |

▸ **What it does not fix:** reasoning, synthesis across many documents, or anything the retriever fails to find. **A RAG system's ceiling is its retriever's recall.** Most "RAG failures" are retrieval failures, and diagnosing which stage failed is the core operational skill.

---

## 18.2 Retrieval methods

### Sparse: BM25

Covered in Ch. 10 §10.2. Still the correct baseline. **Strengths:** exact term matching, rare entities, product codes, names, out-of-domain robustness, no training, trivially updatable, interpretable. **Weakness:** vocabulary mismatch — "car" doesn't match "automobile."

### Dense: bi-encoders

Encode query and document independently; retrieve by nearest neighbour in embedding space.

▸ **Training objective** (contrastive, Ch. 25):
$$\mathcal{L} = -\log\frac{\exp(\mathrm{sim}(q,d^+)/\tau)}{\exp(\mathrm{sim}(q,d^+)/\tau) + \sum_{d^-}\exp(\mathrm{sim}(q,d^-)/\tau)}$$

**The three things that determine quality:**
1. **Hard negatives.** Random negatives are trivially separable and teach almost nothing. Mine negatives with a first-stage retriever (top-ranked but non-relevant documents). This is worth more than any architectural choice.
   - *Caution:* mined negatives include false negatives (actually-relevant unlabelled documents). Filter with a cross-encoder, or use a margin threshold.
2. **In-batch negatives + large batch.** Every other document in the batch is a free negative. Batch 1024 ≫ batch 32. Use gradient caching (GradCache) if memory-limited.
3. **Scale of weak supervision.** Mined pairs (title↔body, question↔answer, anchor↔citation) in the hundreds of millions.

**Asymmetry:** queries and documents have different distributions. Either use separate encoders, or one encoder with instruction prefixes (`"query: "` / `"passage: "`).

### Cross-encoders (rerankers)

Concatenate query and document, run them through a transformer **jointly**, output a relevance score.

▸ Full token-level interaction ⇒ substantially more accurate. But scoring requires one forward pass **per (query, document) pair**, so it cannot search a corpus — $O(N)$ per query.

**The standard architecture is therefore a cascade:** BM25/dense retrieve top-100 → cross-encoder reranks to top-10 → generator sees top-5. Cheap and accurate.

### Late interaction: ColBERT

Store a vector **per token**; score by the sum of maximum similarities:
▸ $$s(q,d)=\sum_{i\in q}\max_{j\in d}\ \langle E_{q_i}, E_{d_j}\rangle$$

Most of the cross-encoder's expressiveness with pre-computable document representations. Cost: storage grows by ~100× (mitigated by PLAID/quantization).

### Hybrid search

Combine sparse and dense. The scores are on incomparable scales, so combine **ranks**:

▸ $$\mathrm{RRF}(d) = \sum_{r\in\text{retrievers}}\frac{1}{k + \mathrm{rank}_r(d)},\qquad k=60$$

Reciprocal Rank Fusion is parameter-light, scale-free, and consistently beats either retriever alone. **Use it by default.**

---

## 18.3 Building a RAG system

### The pipeline

```
Documents → chunk → embed → index
Query → (rewrite) → retrieve (hybrid) → rerank → assemble context → generate → cite
```

### Chunking — the most underrated decision

- **Fixed size** (e.g. 512 tokens, 10–20% overlap): simple, ignores structure.
- **Structural** (by heading, paragraph, function, section): far better when the documents have structure. Usually the right answer.
- **Semantic:** split where consecutive-sentence embedding similarity drops.
- **Late chunking:** embed the *whole* document with a long-context encoder, then pool token embeddings per chunk. Each chunk's vector then carries document-level context — this fixes the pronoun/reference problem where a chunk says "it costs $40" with no idea what "it" is.
- **Contextual retrieval:** prepend an LLM-generated one-sentence summary of the document to each chunk before embedding. Expensive at index time, substantially better recall.

▸ **Small chunks retrieve precisely but lack context; large chunks have context but retrieve imprecisely.** The standard resolution: **index small, retrieve small, then expand** to the parent chunk or surrounding window before passing to the generator.

### Query processing

- **Rewriting:** turn a conversational follow-up ("what about the other one?") into a standalone query. Essential in multi-turn.
- **Decomposition:** split a multi-hop question into sub-queries.
- **HyDE:** have the LLM write a *hypothetical answer*, embed that, and retrieve with it. Answer-to-answer similarity beats question-to-answer similarity when the corpus is prose.
- **Multi-query:** generate several paraphrases, retrieve for each, fuse with RRF.

### Context assembly

- Order matters: put the most relevant material at the **start and end** (the "lost in the middle" U-curve, Ch. 12 §12.8).
- Deduplicate near-identical chunks.
- Include source metadata so the model can cite.
- Leave headroom — filling the context to the limit degrades quality.

### Generation

Instruct explicitly: answer only from the provided context, cite chunk IDs, and say "I don't know" when the context is insufficient. **Train or few-shot this behaviour** — models default to answering from parametric knowledge, which silently defeats the whole system.

---

## 18.4 Evaluating RAG

▸ **Evaluate the stages separately, always.** An end-to-end score cannot tell you whether the retriever missed the document or the generator ignored it.

**Retrieval:** Recall@k (the important one — the ceiling on everything downstream), Precision@k, MRR $=\frac1{|Q|}\sum\frac{1}{\mathrm{rank}_1}$, and **nDCG**:
$$\mathrm{DCG@k}=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},\qquad \mathrm{nDCG@k}=\frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}$$

**Generation:**
- **Faithfulness / groundedness:** is every claim supported by the retrieved context? (Decompose into atomic claims, verify each — via NLI model or LLM judge.)
- **Answer relevance:** does it address the question?
- **Context precision/recall:** were the retrieved chunks the right ones?

**Build a golden set.** 100–500 (query, relevant-docs, reference-answer) triples from your actual domain. Everything else is guesswork. And apply Chapter 3 — report confidence intervals on these numbers, because with 200 queries the standard error on a 0.8 recall is $\sqrt{0.8\cdot0.2/200}=0.028$, so a "3-point improvement" is noise.

### Common failure modes

| Symptom | Likely cause |
|---|---|
| Right document never retrieved | chunking, embedding domain mismatch, vocabulary mismatch → add BM25 |
| Retrieved but ignored | context too long, buried in the middle, weak prompt |
| Contradicts the context | model trusting parametric knowledge → stronger instruction, fine-tune |
| Correct but uncited | prompt/format issue |
| Multi-hop failure | needs decomposition or iterative retrieval |
| Aggregation failure ("how many X?") | retrieval is the wrong tool; use structured query / SQL |

---

## 18.5 Long context vs retrieval

**Long context:** simpler, no infrastructure, better cross-document reasoning.
**Retrieval:** cheaper (cost is linear in context length, often quadratic in attention), scales to arbitrary corpora, updatable, attributable, and empirically more accurate on precise lookup.

▸ **Not substitutes.** The consensus architecture is retrieval to reduce the corpus to a few tens of thousands of tokens, then long context to reason over what's left. Feeding 1M tokens when 5k suffice is a 200× cost increase for a quality *decrease* (Ch. 12 §12.8).

**Caching changes the arithmetic:** if the same large corpus is queried repeatedly, prompt caching makes stuffing it into context cheaper than it looks. Worth checking the numbers rather than assuming.

---

## 18.6 Tool use and function calling

### The mechanism

1. Tools are described to the model in a schema (name, description, JSON parameter spec).
2. The model emits a structured call.
3. The runtime executes it and returns the result as a message.
4. The model continues, possibly calling more tools.

**How models learn this:** fine-tuning on trajectories containing tool calls, with special tokens delimiting the call and result. Reliability comes from **constrained decoding** against the JSON schema (Ch. 13 §13.4) — not from asking politely.

### What makes it work in practice

- **Tool descriptions are prompts.** Write them for a reader who cannot see your code. This is where most of the reliability comes from.
- **Fewer tools, better described.** Accuracy degrades noticeably past ~20 tools; use hierarchical routing or retrieval over tool descriptions beyond that.
- **Return errors as messages**, so the model can retry with corrections. Silent failure is the worst outcome.
- **Idempotency and least privilege.** The model will call things twice and will call things it shouldn't.

---

## 18.7 Agents

### The definition

▸ An agent is an LLM in a **loop** with tools, where the model — not a fixed program — decides what to do next. The defining property is *control flow determined by the model*.

### ReAct

Interleave reasoning and acting:
```
Thought: I need the 2024 revenue figure.
Action: search("Acme 2024 annual revenue")
Observation: ...
Thought: That's the 2023 figure. Let me refine.
Action: ...
```
The explicit thought step measurably improves tool selection and lets the model recover from bad observations.

### The components

- **Planning:** decompose upfront (brittle, cheap) or plan-and-revise (robust, expensive). Reflexion adds a self-critique step after failures.
- **Memory:** short-term (context window), long-term (vector store of past episodes), and working memory (a scratchpad file). Summarize or evict as the context fills.
- **Tools:** search, code execution, file I/O, APIs, other agents.
- **Termination:** step limits, budget limits, and a self-assessed done condition. **Always have a hard cap** — unbounded loops are the default failure mode.

### Multi-agent patterns

Supervisor/worker, debate, pipeline, blackboard.

▸ **The honest assessment:** multi-agent systems are frequently worse than a single well-prompted agent. Each hand-off loses context, errors compound multiplicatively (10 steps at 95% each = 60% end-to-end), and cost scales with the number of agents. Use multiple agents when there is genuine parallelism or a genuine need for isolated context — not because the diagram looks impressive.

### Why agents fail

▸ **Compounding error is the fundamental issue.** With per-step reliability $p$ and $n$ steps, success is $p^n$. At $p=0.95$: 20 steps gives $0.36$. **Getting from 95% to 99% per-step reliability matters more than any other improvement**, because $0.99^{20}=0.82$.

Other modes: context rot over long trajectories, tool-output overflow, looping without progress, over-confidence in bad observations, and **prompt injection from tool outputs** — a retrieved web page containing "ignore previous instructions" is an attack surface that does not exist in a non-agentic system.

### Evaluating agents

Task success rate is necessary but insufficient. Also measure: steps taken, cost, tool-call accuracy, recovery rate after an error, and safety violations. Use held-out task suites with executable verification (SWE-bench, WebArena, τ-bench style) rather than judged prose.

---

## Check for Understanding

**Retrieval separates knowledge from reasoning, and every RAG system is bounded by its retriever's recall — so evaluate the stages separately, use hybrid search with rank fusion and a cross-encoder reranker, and remember that agents multiply per-step reliability, which is why a 4-point improvement in step accuracy matters more than any architectural cleverness.**

---

**Next:** [Chapter 19 — Generative Models: The Taxonomy](19-generative-models-taxonomy.md)
