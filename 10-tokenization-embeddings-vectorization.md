# Chapter 10 — Tokenization, Embeddings & Vectorization

> **Prerequisites:** Ch. 1 (§1.1, §1.4), Ch. 6.
> **Scope:** how discrete symbols become vectors. This is the input layer of every language model and the foundation of all retrieval systems.

---

## 10.1 The problem

### The one-line idea

Neural networks consume real-valued vectors; language is a sequence of discrete symbols. Something has to bridge that, and the choice of bridge determines what the model can and cannot represent.

### The analogy

You are building a library catalogue. First you must decide what a "unit" is — whole books? chapters? sentences? individual words? Too coarse and you can't find anything specific; too fine and the catalogue is enormous and every lookup is fragmented. Then you must decide where on the shelves each unit goes, so that related units sit near each other. Tokenization is the first decision; embedding is the second.

---

## 10.2 The pre-neural lineage (still relevant, still interviewed)

### One-hot encoding

$x\in\{0,1\}^{|V|}$, exactly one 1.

**Problems:** dimension $=|V|$ (50k–250k); every pair of distinct words is **exactly equidistant** ($\|e_i-e_j\|=\sqrt2$ for all $i\ne j$); no notion of similarity; no generalization to unseen words.

▸ **Key insight:** $We_i$ = the $i$-th **column** of $W$ (Ch. 1 §1.1.1). So an embedding layer is not a matrix multiply — it is a row lookup. `nn.Embedding` is `W[idx]`, $O(1)$, not $O(|V|d)$.

### Bag of words and TF-IDF

Count vector $c_i$ = frequency of term $i$ in the document. Discards order entirely.

▸ $$\mathrm{tf\text{-}idf}(t,d) = \underbrace{\mathrm{tf}(t,d)}_{\text{count, often } \log(1+f)}\times\underbrace{\log\frac{N}{1+\mathrm{df}(t)}}_{\text{inverse document frequency}}$$

The IDF term down-weights terms appearing in many documents. **Information-theoretic reading:** $\log(N/\mathrm{df})$ is the self-information of "this document contains term $t$" under a unigram model — rare terms carry more bits, so they discriminate better.

**BM25** (the actual retrieval standard, still competitive in 2026):
▸ $$\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}$$
with $k_1\approx1.2$ (term-frequency saturation) and $b\approx0.75$ (length normalization). The saturation is the improvement over TF-IDF: the 20th occurrence of a word says almost nothing more than the 10th. **BM25 beats many dense retrievers on out-of-domain and rare-entity queries and should always be in your baseline** (Ch. 18).

### Latent Semantic Analysis

Truncated SVD of the term–document matrix: $X\approx U_k\Sigma_kV_k^\top$. Rows of $U_k\Sigma_k$ are word vectors. This is PCA on co-occurrence counts, and it is the direct ancestor of everything below.

---

## 10.3 Word2vec

### The one-line idea

A word's meaning can be learned from the company it keeps — so train a model to predict a word's neighbours, and take its internal representation as the meaning.

*The distributional hypothesis (Firth, 1957): "You shall know a word by the company it keeps."*

### Skip-gram

Maximize the probability of context words given a centre word:

$$\mathcal{L} = -\frac1T\sum_{t=1}^{T}\sum_{-m\le j\le m, j\ne0}\log p(w_{t+j}\mid w_t),\qquad p(o\mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}$$

▸ Two separate embedding matrices: $v$ for "as centre", $u$ for "as context." Final vectors are usually $v$, or $v+u$.

**The problem:** the softmax denominator sums over $|V|=10^6$ words for every training pair. Intractable.

### Negative sampling (SGNS) — derive this, it's asked constantly

Replace the $|V|$-way softmax with $k+1$ independent binary logistic problems: "is this (centre, context) pair real or fake?"

▸ $$\mathcal{L} = -\log\sigma(u_o^\top v_c) - \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\big[\log\sigma(-u_{w_i}^\top v_c)\big]$$

- First term: push real pairs' dot products up.
- Second: push $k$ sampled fake pairs' dot products down. $k=5$–20 for small corpora, 2–5 for large.
- Noise distribution: $P_n(w)\propto U(w)^{3/4}$. **The 3/4 exponent flattens the Zipf distribution** — it samples rare words more than unigram frequency would, and common words less. Purely empirical, but a robust finding.

**Hierarchical softmax** is the alternative: arrange $V$ as a Huffman tree, predict $\log_2|V|$ binary decisions. $O(\log|V|)$ instead of $O(|V|)$. Better for rare words; negative sampling is better for frequent words and simpler.

**Subsampling frequent words:** discard word $w$ with probability $P(w) = 1-\sqrt{t/f(w)}$, $t\approx10^{-5}$. Removes "the", "of" from most windows, which both speeds training and improves rare-word vectors.

### What word2vec is actually doing

▸ **Levy & Goldberg (2014):** SGNS with $k$ negatives is implicitly factorizing the shifted PMI matrix:
$$u_w^\top v_c \approx \mathrm{PMI}(w,c) - \log k,\qquad \mathrm{PMI}(w,c) = \log\frac{p(w,c)}{p(w)p(c)}$$

So word2vec is **matrix factorization of pointwise mutual information** — i.e. LSA with a better weighting. That result demystified the whole thing and is a great thing to know.

### CBOW

Predict the centre from the averaged context. Faster, better for frequent words; skip-gram is better for rare words and small corpora.

### GloVe

Fit log co-occurrence counts directly with a weighted least-squares objective:
▸ $$J = \sum_{i,j} f(X_{ij})\left(w_i^\top\tilde w_j + b_i+\tilde b_j - \log X_{ij}\right)^2,\quad f(x)=\min\left(1,(x/x_{\max})^{0.75}\right)$$
Global statistics in one pass rather than local windows. Comparable quality; the two approaches are theoretically closely related.

### The analogy property, and its honest caveats

$$\mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man}) + \mathrm{vec}(\text{woman}) \approx \mathrm{vec}(\text{queen})$$

**Why it works:** if $u_w^\top v_c\approx\mathrm{PMI}$, then differences of vectors correspond to *ratios* of co-occurrence probabilities, and a relation like gender is a roughly constant ratio-shift across pairs.

▸ **The caveats you should state if asked:** standard evaluations *exclude the three input words* from the answer candidates, and without that exclusion the nearest neighbour is very often "king" itself. Analogy accuracy is also heavily driven by the offsets of the most frequent word pairs. The property is real but weaker than the famous demo suggests.

### Bias

Word embeddings encode the statistical regularities of the corpus, including social bias (`doctor - man + woman ≈ nurse`). Debiasing by projecting out a "gender direction" (Bolukbasi et al.) **reduces the measurement of bias more than the bias itself** — Gonen & Goldberg showed clustering structure survives the projection. This is worth knowing as an example of a general lesson: *removing a linear direction is not removing a concept*.

### The fundamental limitation

▸ **One vector per word type.** "bank" (river) and "bank" (finance) get the same vector — necessarily the average of both senses. This is what contextual embeddings fixed: ELMo (bidirectional LSTM states), then BERT, then every LLM. In a transformer, the *same* token gets a different residual-stream vector at every layer and in every context.

**fastText** partially fixes OOV by representing a word as the sum of its character $n$-grams, which also helps morphologically rich languages.

---

## 10.4 Subword tokenization

### The one-line idea

Don't choose between characters (too long, no semantics) and words (huge vocabulary, out-of-vocabulary failures) — learn a vocabulary of frequent character sequences that lands in between.

### The analogy

Lego. Characters are 1×1 studs: universal but you need thousands to build anything. Words are pre-moulded castle walls: fast when they fit, useless otherwise. Subwords are the standard brick set — a few thousand shapes that compose into anything, with common structures available as single pieces.

### Byte-Pair Encoding (BPE)

**Training:**
1. Initialize the vocabulary with all individual characters (or all 256 bytes).
2. Count all adjacent symbol pairs in the corpus.
3. Merge the most frequent pair into a new symbol; record the merge.
4. Repeat until $|V|$ reaches the target.

**Encoding:** apply the recorded merges in the order they were learned.

**Worked example** on `{low:5, lower:2, newest:6, widest:3}`:
```
start:  l o w </w> ×5 | l o w e r </w> ×2 | n e w e s t </w> ×6 | w i d e s t </w> ×3
pair counts: (e,s)=9  (s,t)=9  (l,o)=7  (o,w)=7 ...
merge (e,s) → es   : n e w es t ×6 | w i d es t ×3
merge (es,t) → est : n e w est ×6 | w i d est ×3
merge (l,o) → lo   : lo w ×5 | lo w e r ×2
merge (lo,w) → low : low ×5 | low e r ×2
```
The algorithm has discovered the morpheme `est` and the stem `low` from frequency alone.

▸ **Byte-level BPE** (GPT-2 onward) starts from the 256 raw bytes, so **every possible string is encodable and there is no `<UNK>` token, ever.** This is why modern LLMs never fail on emoji or unusual scripts. Cost: non-Latin scripts consume more tokens per character.

### WordPiece (BERT)

Same merge loop, different criterion. Instead of raw frequency, merge the pair that maximizes the corpus likelihood under a unigram model:

▸ $$\text{score}(a,b) = \frac{\mathrm{count}(ab)}{\mathrm{count}(a)\cdot\mathrm{count}(b)}$$

This is (a monotone transform of) pointwise mutual information — it prefers pairs that co-occur *more than chance*, rather than merely often. `un`+`##able` beats `th`+`##e` even if the latter is more frequent, because `th` and `e` are individually very common.

Marks continuations with `##`.

### Unigram LM / SentencePiece

Runs **backwards**: start with a large candidate vocabulary, then iteratively *remove* the pieces whose deletion least increases the corpus negative log-likelihood, under a unigram model where
$$p(x) = \prod_i p(x_i)\quad\text{over the segmentation } x = (x_1,\dots,x_n)$$
optimized with EM, and where the probability of a sentence marginalizes over all segmentations (computed by a Viterbi/forward pass).

▸ Because it is probabilistic, it supports **subword regularization**: sample a different segmentation each epoch. This is data augmentation for tokenization and reliably improves translation robustness by 1–2 BLEU.

**SentencePiece** is the *implementation* (supporting BPE and Unigram) that treats input as a raw Unicode stream, encoding spaces as `▁`, so it is language-agnostic and detokenization is lossless. Used by T5, LLaMA, Gemma, Mistral.

### Vocabulary size: the actual trade-off

| Larger $|V|$ | Smaller $|V|$ |
|---|---|
| Shorter sequences → less attention compute ($O(T^2)$!) | Longer sequences |
| More parameters in embedding + output ($2|V|d$) | Fewer parameters |
| Rarer tokens → worse-estimated embeddings | Better-estimated embeddings |
| Better compression of common words | More compositional generalization |

▸ **Empirical guidance:** 32k was the BERT/LLaMA-1 standard; 100k–256k is now common (GPT-4, LLaMA-3, Gemma) because longer sequences are the more expensive resource and larger vocabularies improve multilingual fairness. The output softmax at $|V|=256$k and $d=8192$ is $2.1$B parameters — often the single largest matrix in the model.

**Compression rate matters directly:** if tokenizer A produces 10% fewer tokens than B for the same text, it is 10% cheaper to train *and* serve, and it fits 10% more into a context window.

### The failure modes tokenization causes

▸ These are excellent interview material because they show you understand the input layer:

- **Arithmetic.** `1234` may tokenize as `12|34` or `1|234` depending on frequency, so digit position is inconsistent. Fix: force single-digit tokenization (LLaMA does this) or right-to-left digit grouping.
- **Character-level tasks.** "How many r's in strawberry?" is hard because the model sees ~3 tokens, not 10 characters. It has to have memorized the spelling, not observe it.
- **Rhyming, anagrams, reversal** — same cause.
- **Glitch tokens.** `SolidGoldMagikarp` and friends: tokens that appeared in the tokenizer's training corpus but almost never in the LM's, so their embeddings stayed near initialization and produce bizarre behaviour.
- **Multilingual inequity.** Some languages cost 3–5× more tokens per unit of meaning, which is a direct cost and context-length penalty.
- **Prompt-boundary effects.** Trailing whitespace changes tokenization and can measurably change outputs.

**Tokenizer-free alternatives:** ByT5 (raw bytes, 4× slower), CANINE, and **Byte Latent Transformer** (dynamic entropy-based patching). None has displaced BPE yet, but this is an active area.

---

## 10.5 The embedding layer in practice

### Shape and initialization

$E\in\mathbb{R}^{|V|\times d}$. Init $\mathcal{N}(0, 0.02^2)$ (GPT convention) or $\mathcal{N}(0,1/d)$.

### Weight tying

▸ Set the output projection $W_{\text{out}} = E^\top$. Saves $|V|d$ parameters (for $|V|=128$k, $d=4096$: **524M parameters**), and consistently improves perplexity in small/medium models by regularizing both matrices toward a shared space.

Argument: the input embedding maps token→vector, the output maps vector→token-logit. If the representation space is consistent, these should be transposes. Note some large models *untie* them, since at scale the extra capacity is worth more than the regularization.

### Factorized embeddings (ALBERT)

$|V|\times d$ becomes $|V|\times e$ then $e\times d$ with $e\ll d$. Decouples vocabulary embedding size from hidden size. Reduces parameters from $O(|V|d)$ to $O(|V|e+ed)$.

### The scaling detail

The original Transformer multiplies embeddings by $\sqrt{d_{\text{model}}}$ before adding positional encodings. Reason: embeddings initialized with variance $1/d$ have norm $\approx1$, while sinusoidal positional encodings have entries of magnitude $\approx1$ and norm $\approx\sqrt{d}$. Without the scaling the positional signal would drown the token signal.

---

## 10.6 Sentence and document embeddings

### Why pooling BERT naively fails

Averaging BERT's token outputs gives sentence vectors that underperform averaged GloVe on similarity tasks. The reason is **anisotropy**: BERT's representation space occupies a narrow cone, so the average cosine similarity between *random* sentences is ~0.6, leaving little dynamic range.

### Sentence-BERT and contrastive training

Fine-tune with a siamese architecture and a contrastive or triplet objective so that cosine similarity becomes meaningful. The modern recipe (E5, BGE, GTE, Nomic) is:

1. Large-scale weakly-supervised contrastive pretraining on mined pairs (title↔body, question↔answer, citation pairs).
2. Fine-tune on labelled pairs with **hard negatives** mined from a first-stage retriever.
3. **In-batch negatives** with large batch (see InfoNCE, Ch. 25):
▸ $$\mathcal{L} = -\log\frac{\exp(\cos(q,d^+)/\tau)}{\sum_{d\in\mathcal{B}}\exp(\cos(q,d)/\tau)},\quad \tau\approx0.02\text{–}0.05$$
4. Instruction prefixes (`"query: "` / `"passage: "`) so one model serves asymmetric retrieval.

**Matryoshka representation learning:** train with the loss applied to nested prefixes of the embedding (first 64, 128, 256, … dims), so a single model yields usable embeddings at many dimensionalities. Lets you store 768-d and search at 64-d.

### Similarity measures

| Measure | Formula | Note |
|---|---|---|
| Cosine | $\frac{a^\top b}{\|a\|\|b\|}$ | magnitude-invariant; the default |
| Dot product | $a^\top b$ | magnitude carries information (e.g. document "importance") |
| Euclidean | $\|a-b\|$ | for normalized vectors, $\|a-b\|^2 = 2-2\cos$ — **monotonically equivalent to cosine** |

▸ That last identity is worth remembering: **on unit-normalized vectors, ranking by cosine, dot product, and Euclidean distance give identical orderings.**

---

## 10.7 Vector search

Exact search over $N$ vectors is $O(Nd)$ per query — fine to ~1M, hopeless at 1B.

### HNSW (graph-based)

Build a multi-layer proximity graph; upper layers are sparse "highways," lower layers dense. Search greedily descends from a top-layer entry point.

- Query: $O(\log N)$, very high recall (98%+).
- Memory: $O(N(d + M))$ with $M\approx16$–64 neighbours per node — the graph itself is significant.
- Parameters: `M` (connectivity), `efConstruction` (build quality), `efSearch` (query-time recall/latency trade).

### IVF + Product Quantization (compression-based)

**IVF:** k-means the corpus into $n_{\text{list}}$ cells; search only the $n_{\text{probe}}$ nearest cells.

**PQ:** split each $d$-dim vector into $m$ sub-vectors, k-means each sub-space into 256 centroids, store $m$ bytes per vector.
▸ Compression: $d\times4$ bytes → $m$ bytes. For $d=768$, $m=96$: $3072\to96$ bytes, a **32× reduction**. Distances are computed by table lookup and summation, which is extremely fast.

**Choosing:** HNSW for ≤10M vectors with a memory budget; IVF-PQ beyond that; ScaNN (anisotropic quantization) when maximum-inner-product recall matters most.

### The metric that matters
**Recall@k vs latency**, at fixed memory. Every ANN method is a point on that trade-off surface; there is no free lunch.

---

## Check for Understanding

**Tokenization decides what the model can perceive at all — which is why LLMs are bad at spelling and arithmetic — and embedding turns those tokens into a geometry where similarity is distance, learned either from co-occurrence statistics (word2vec, which is implicitly factorizing PMI) or from context inside a transformer, where a token's vector finally depends on the sentence it appears in.**

---

**Next:** [Chapter 11 — Attention & the Transformer](11-attention-and-transformers.md)
