# Chapter 10 — Tokenization, Embeddings & Vectorization

> **Prerequisites:** Ch. 1 (§1.1, §1.4), Ch. 6.
> **Scope:** how discrete symbols become vectors. This is the input layer of every language model and the foundation of all retrieval systems.

> **New to the notation?** If symbols like $\in$, $\sum$, $\prod$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar — or if you have ever wondered why $\sigma$ seems to mean four different things — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\lvert V\rvert$ | "size of V" / "vocab size" | How many distinct tokens exist. Typically 32,000–256,000 |
| $`e_i`$ | "e-i" | A **one-hot vector**: all zeros except a single $1$ at position $i$ |
| $d$ | "d" | The **width** of a vector — how many numbers describe one token |
| $E \in \mathbb{R}^{\lvert V\rvert \times d}$ | "E in R, V-by-d" | The **embedding table**: one row of $d$ numbers per token |
| $\mathrm{tf}(t,d)$ | "t-f of t, d" | How many times term $t$ appears inside document $d$ |
| $\mathrm{df}(t)$ | "d-f of t" | In **how many documents** term $t$ appears at all |
| $\overline{\lvert d\rvert}$ | "average doc length" | The mean document length across the whole corpus |
| $`u_o^\top v_c`$ | "u-o transpose v-c" | Dot product of two word vectors — a **similarity score** |
| $\sigma(z)$ | "sigmoid of z" | $1/(1+e^{-z})$; squashes any number into $(0,1)$. In this chapter $\sigma$ is **always** the sigmoid, never a standard deviation |
| $`P_n(w)`$ | "P-n of w" | The **noise distribution** — how fake ("negative") words get drawn |
| $\mathrm{PMI}(w,c)$ | "P-M-I of w, c" | **Pointwise mutual information**: how much more often $w$ and $c$ co-occur than chance predicts |
| $`X_{ij}`$ | "X-i-j" | How many times word $j$ appeared in word $i$'s context window |
| $\propto$ | "is proportional to" | Equal after multiplying by a constant nobody cares about |
| $\tau$ | "tau" | **Temperature** — divides scores before a softmax; small $\tau$ sharpens |
| $\cos(a,b)$ | "cosine similarity" | $a^\top b / (\lVert a\rVert \lVert b\rVert)$ — alignment, with length ignored |
| $\mathcal{B}$ | "script B" | The current **batch** of examples |
| ▁ | "the SentencePiece space" | A printable stand-in for a space character |

### Abbreviations used in this chapter

The book's complete glossary lives in [Chapter 0 §0.13](00-notation-and-math-primer.md). These are the ones you need here.

| Short | Full form |
|---|---|
| ALBERT | A Lite BERT |
| ANN | Approximate Nearest Neighbour |
| BERT | Bidirectional Encoder Representations from Transformers |
| BLEU | BiLingual Evaluation Understudy |
| BM25 | Best Match 25 |
| BPE | Byte-Pair Encoding |
| CBOW | Continuous Bag Of Words |
| ELMo | Embeddings from Language Models |
| EM | Expectation–Maximization |
| GloVe | Global Vectors (for word representation) |
| HNSW | Hierarchical Navigable Small World |
| IDF | Inverse Document Frequency |
| InfoNCE | Information Noise-Contrastive Estimation |
| IVF | Inverted File (index) |
| LSA | Latent Semantic Analysis |
| OOV | Out-Of-Vocabulary |
| PCA | Principal Component Analysis |
| PMI | Pointwise Mutual Information |
| PQ | Product Quantization |
| ScaNN | Scalable Nearest Neighbors |
| SGNS | Skip-Gram with Negative Sampling |
| SVD | Singular Value Decomposition |
| TF-IDF | Term Frequency–Inverse Document Frequency |

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

**Problems:** dimension $=|V|$ (50k–250k); every pair of distinct words is **exactly equidistant** ($`\|e_i-e_j\|=\sqrt2`$ for all $i\ne j$); no notion of similarity; no generalization to unseen words.

▸ **Key insight:** $`We_i`$ = the $i$-th **column** of $W$ (Ch. 1 §1.1.1). So an embedding layer is not a matrix multiply — it is a row lookup. `nn.Embedding` is `W[idx]`, $O(1)$, not $O(|V|d)$.

#### Unpacking one-hot encoding

Read the line $x\in\{0,1\}^{\lvert V\rvert}$ aloud: *"x is a list of $\lvert V\rvert$ numbers, each of which is either 0 or 1."* The braces $\{0,1\}$ name the only two allowed values; the superscript is the length, exactly as $\mathbb{R}^n$ means "a list of $n$ real numbers." And $\lvert V\rvert$ — vertical bars around a set — means **"how many things are in $V$"**, the *size* of the vocabulary. (Same bars, different job from absolute value and from "given"; Chapter 0 §0.9 catalogues the collision.)

So with a 5-word vocabulary $V = \{\texttt{cat}, \texttt{dog}, \texttt{run}, \texttt{the}, \texttt{sat}\}$, the word `dog` is

$$e_2 = (0,\ 1,\ 0,\ 0,\ 0)$$

**a light switch panel with exactly one switch flipped on.** That is the whole idea, and it is where the name comes from: in digital circuit design, a "one-hot" register is one where exactly one bit is high.

**Now the damning property, with numbers.** Take `cat` $`= e_1 = (1,0,0,0,0)`$ and `dog` $`= e_2 = (0,1,0,0,0)`$. Their difference is $(1,-1,0,0,0)$, whose length is $\sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1.414$. Now take `cat` and `sat` $`= e_5`$: the difference is $(1,0,0,0,-1)$, length $\sqrt2$ again. **Every** pair gives $\sqrt2$, because any two distinct one-hot vectors disagree in exactly two places, each by exactly 1.

▸ **This is the fatal flaw stated precisely: in one-hot space, `cat` is exactly as far from `dog` as it is from `Tuesday`.** The representation contains no information beyond identity. A model given one-hot inputs must learn everything about every word from scratch, with no ability to transfer what it learned about `cat` to `kitten`. Every idea in the rest of this chapter exists to replace that flat, uninformative geometry with one where distance means something.

> **Analogy.** One-hot encoding is a filing cabinet where every folder is in its own separate room, and all the rooms are the same distance apart down an infinitely long corridor. You can find anything if you know its room number, but the layout tells you nothing — "cookbooks" is not near "recipes." An embedding is what you get when you finally rearrange the building so related folders share a shelf.

**"Not a matrix multiply — a lookup," decoded.** Mathematically, multiplying the embedding matrix by a one-hot vector *selects* a single column (Ch. 1 §1.1.1: a matrix–vector product is a weighted sum of columns, and here all the weights are zero except one). Doing it literally would cost $\lvert V\rvert \times d$ multiply-adds — with $\lvert V\rvert = 128{,}000$ and $d = 4096$ that is **524 million operations to retrieve 4096 numbers**, virtually all of them multiplications by zero. So frameworks index instead: $O(1)$.

Two orientations are floating around here and they trip people up. If you write the table as $W \in \mathbb{R}^{d\times\lvert V\rvert}$ (tokens as *columns*), the lookup is a column selection $`We_i`$. If you write it as $E \in \mathbb{R}^{\lvert V\rvert\times d}$ (tokens as *rows*, which is what `nn.Embedding` stores and what §10.5 uses), the lookup is a row selection `E[idx]`. **Same operation, transposed bookkeeping.** Check which convention a piece of code or a paper is using before you trust a shape.

### Bag of words and TF-IDF

Count vector $`c_i`$ = frequency of term $i$ in the document. Discards order entirely.

▸ $$\mathrm{tf\text{-}idf}(t,d) = \underbrace{\mathrm{tf}(t,d)}_{\text{count, often } \log(1+f)}\times\underbrace{\log\frac{N}{1+\mathrm{df}(t)}}_{\text{inverse document frequency}}$$

The IDF term down-weights terms appearing in many documents. **Information-theoretic reading:** $\log(N/\mathrm{df})$ is the self-information of "this document contains term $t$" under a unigram model — rare terms carry more bits, so they discriminate better.

#### Reading TF-IDF in plain English

**TF-IDF** is short for **term frequency–inverse document frequency**. The formula is a product of exactly two ideas, and the underbraces in the equation are labels telling you which is which.

| Piece | Read aloud | What it asks |
|---|---|---|
| $\mathrm{tf}(t,d)$ | "term frequency of t in d" | *"How often does this word appear in **this** document?"* |
| $N$ | "N" | The total number of documents in the collection |
| $\mathrm{df}(t)$ | "document frequency of t" | *"How many documents contain this word **at all**?"* |
| $\log\frac{N}{1+\mathrm{df}(t)}$ | "inverse document frequency" | *"How rare is this word across the whole collection?"* |
| $\log(1+f)$ | "log of one plus f" | A softener: the 50th occurrence should not count 50× the first |

The $1+$ in the denominator is defensive plumbing — it stops you dividing by zero for a term that appears in no document at all.

**Put real numbers in.** Suppose you have $N = 1{,}000{,}000$ web pages.

| Term | $\mathrm{df}(t)$ | $\log\frac{N}{1+\mathrm{df}}$ | Verdict |
|---|---|---|---|
| "the" | $1{,}000{,}000$ | $\log(1) \approx 0$ | Appears everywhere → **weight zero.** Useless for telling documents apart |
| "network" | $50{,}000$ | $\log(20) \approx 3.0$ | Moderately informative |
| "Beltrami" | $200$ | $\log(4975) \approx 8.5$ | Very rare → **a strong signal** |

So a document mentioning "Beltrami" three times scores $3 \times 8.5 = 25.5$ on that term, while a document mentioning "the" three hundred times scores $300 \times 0 = 0$.

▸ **The one sentence to keep: a word is useful for finding a document in proportion to how surprising it is that the word appeared.** "The" is not surprising anywhere, so it carries no evidence. "Beltrami" is surprising almost everywhere, so seeing it narrows the field enormously.

> **Analogy.** You are trying to identify a stranger from a description. "Has two eyes" is true of everyone and eliminates nobody. "Wears a red beret" eliminates 99.99% of the population. The value of a clue is not how loudly it is stated, but how many candidates it removes — and IDF is exactly the logarithm of how many candidates it removes.

**Where the $\log$ comes from — and why it's the *self-information*.** If a fraction $\mathrm{df}/N$ of documents contain $t$, then the probability that a randomly-picked document contains $t$ is $p = \mathrm{df}/N$. Information theory (Ch. 1 §1.4) says an event of probability $p$ carries $-\log p$ nats of information. And $-\log(\mathrm{df}/N) = \log(N/\mathrm{df})$ — **the IDF term *is* the surprisal, exactly.** A weighting scheme invented by intuition in the early 1970s turned out to be a quantity Shannon had defined in 1948; the connection was noticed afterwards.

> **Where this came from.** The inverse-document-frequency idea is due to **Karen Spärck Jones**, working at Cambridge, in a 1972 paper in the *Journal of Documentation* titled "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." The term-frequency half is older, going back to **Hans Peter Luhn** at IBM in the late 1950s, who was trying to build automatic abstracts of technical articles. Spärck Jones's contribution — arguably the single most-used formula in the history of search engines — was regarded for years as a modest empirical result, and she worked in relative obscurity for much of her career. She is widely quoted as having said that computing is too important to be left to men. The awards came late: she received the ACM Athena Lecturer award and the BCS Lovelace Medal in the 2000s, shortly before her death in 2007.

**BM25** (the actual retrieval standard, still competitive in 2026):
▸ $$\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}$$
with $`k_1\approx1.2`$ (term-frequency saturation) and $b\approx0.75$ (length normalization). The saturation is the improvement over TF-IDF: the 20th occurrence of a word says almost nothing more than the 10th. **BM25 beats many dense retrievers on out-of-domain and rare-entity queries and should always be in your baseline** (Ch. 18).

#### Unpacking BM25

This formula looks forbidding and is actually three simple corrections bolted onto TF-IDF. Read the outer structure first and ignore the fraction:

$$\mathrm{BM25}(q,d) = \sum_{t \in q} \mathrm{IDF}(t)\cdot(\text{something about how often } t \text{ appears in } d)$$

*"For each term $t$ in the query $q$, take how rare that term is, multiply by how much this document delivers on it, and add up over all query terms."* The $`\sum_{t\in q}`$ is a loop over query words; $t \in q$ reads *"t is in q"*. That's the whole skeleton.

Now the pieces inside:

| Symbol | Read aloud | What it is |
|---|---|---|
| $f(t,d)$ | "f of t, d" | Raw count of term $t$ in document $d$ |
| $`k_1`$ | "k-one" | **Saturation knob.** How fast extra occurrences stop mattering |
| $b$ | "b" | **Length-normalization knob**, between 0 and 1 |
| $\lvert d\rvert$ | "length of d" | Number of words in this document |
| $\overline{\lvert d\rvert}$ | "d-bar" | The **average** document length in the corpus |

**Correction 1 — saturation.** Set $b=0$ for a moment to kill length normalization, and the fraction becomes $`\frac{f(k_1+1)}{f + k_1}`$. Watch what it does with $`k_1 = 1.2`$:

| $f$ (occurrences) | $\frac{f\cdot 2.2}{f + 1.2}$ | Marginal gain over previous row |
|---|---|---|
| 1 | 1.00 | — |
| 2 | 1.38 | $+0.38$ |
| 5 | 1.77 | $+0.13$ per occurrence |
| 10 | 1.96 | $+0.04$ per occurrence |
| 100 | 2.17 | $+0.002$ per occurrence |
| $\infty$ | 2.20 | nothing |

▸ **The count is squeezed into a bounded range — no matter how many times a word appears, the contribution can never exceed $`k_1+1 = 2.2`$.** Plain TF-IDF is linear in the count, so a spam page repeating "insurance" ten thousand times outranks a  article that mentions it four times. BM25 makes that attack worthless. The 20th mention scores $2.076$ against the 19th's $2.069$ — a gain of **0.3%**.

> **Analogy.** Saturation is the law of diminishing returns on evidence. Hearing one witness say the defendant was at the scene is a big update. The second witness is a decent update. The four-hundredth witness saying the same thing changes essentially nothing — and if all four hundred are the same person shouting, it should change nothing at all. $`k_1`$ sets how quickly you stop listening.

**Correction 2 — length normalization.** The factor $\left(1 - b + b\frac{\lvert d\rvert}{\overline{\lvert d\rvert}}\right)$ sits in the denominator, so a **larger** value makes the score **smaller**. Read it as a dial:

- $b = 0$ → the factor is $1$ always → document length ignored entirely.
- $b = 1$ → the factor is $\lvert d\rvert / \overline{\lvert d\rvert}$ → full normalization by relative length.
- $b = 0.75$ (the standard) → three-quarters of the way to full normalization.

Concretely, a document of 4000 words in a corpus averaging 1000 words has $\lvert d\rvert/\overline{\lvert d\rvert} = 4$, so with $b=0.75$ the denominator factor is $1 - 0.75 + 0.75(4) = 3.25$. Its term counts are effectively divided by more than three. **The reason:** a long document mentions everything a few times by accident. Without this correction, search results are dominated by whichever page is longest.

**Correction 3 — IDF, inherited.** Same rarity weighting as §10.2's TF-IDF, though BM25's usual variant is $\log\frac{N - \mathrm{df} + 0.5}{\mathrm{df} + 0.5}$, a smoothed form derived from a probabilistic model of relevance rather than from Shannon information.

▸ **Why a 1990s formula with two hand-tuned constants still beats neural retrievers on some queries:** BM25 does **exact lexical matching**, so it can never fail to find a document containing a rare string. Ask a dense embedding model for the part number `ABX-4471-QT` and it will return things that *feel* similar; ask BM25 and it returns the document with that exact string or nothing. Neural retrieval generalizes; lexical retrieval is literal. Real systems run both and fuse the rankings — which is why Chapter 18 treats hybrid retrieval as the default, not a compromise.

> **Where this came from.** BM25 is more properly **Okapi BM25**, and both halves of that name are accidents. *Okapi* was the name of an experimental information-retrieval system built at City University, London, in the 1980s and used in the early TREC evaluations. *BM* stands for **"Best Match"** — and the **25** is a version number: it was roughly the twenty-fifth weighting function that **Stephen Robertson**, **Karen Spärck Jones**, and colleagues tried. The thing that has powered Elasticsearch, Lucene, and a large fraction of the world's search bars is, by its own name, experiment number twenty-five.

### Latent Semantic Analysis

Truncated SVD of the term–document matrix: $`X\approx U_k\Sigma_kV_k^\top`$. Rows of $`U_k\Sigma_k`$ are word vectors. This is PCA on co-occurrence counts, and it is the direct ancestor of everything below.

#### What Latent Semantic Analysis actually does

**LSA** stands for **latent semantic analysis** — "latent" meaning *hidden*, the structure you did not write down but that is implied by the counts.

Start with the **term–document matrix** $X$: one row per word, one column per document, and entry $`X_{ij}`$ = how many times word $i$ appears in document $j$ (often TF-IDF-weighted rather than raw). For a corpus of 50,000 words and 100,000 documents, $X$ is a $50{,}000 \times 100{,}000$ grid — five billion entries, of which perhaps 0.1% are non-zero.

The **SVD (singular value decomposition, Ch. 1 §1.1.3)** rewrites any matrix as "rotate → stretch → rotate": $X = U\Sigma V^\top$. **Truncated** means you keep only the $k$ largest stretch factors and throw the rest away — the subscript in $`U_k\Sigma_kV_k^\top`$ is that $k$. Eckart–Young guarantees this is the best possible rank-$k$ approximation, so nothing cleverer exists.

▸ **The payoff in one sentence: throwing away the small singular values forces words that behave alike to collapse onto the same few directions.** "Car" and "automobile" almost never appear in the same sentence — writers pick one and stick with it — so in the raw counts they look unrelated. But they appear alongside the *same other words*: "engine," "driver," "highway." Compressing to $k=300$ dimensions leaves no room to keep them apart, and they end up nearly on top of each other. **The compression is not a cost; it is the entire mechanism.**

> **Analogy.** Photograph a crowded market from a great distance. Detail is destroyed, but *groupings* become visible: the fruit sellers are over there, the fabric stalls over here. Zoomed all the way in you see individual faces and no structure; zoomed all the way out you see one blur. There is an intermediate resolution — a value of $k$ — at which the market's organization is most legible. Choosing $k \approx 300$ is choosing that altitude.

**Why "rows of $`U_k\Sigma_k`$ are word vectors."** $U$ has one row per *word* (the rows of $X$ were words), and $`\Sigma_k`$ scales each of the $k$ directions by its importance. So row $i$ of $`U_k\Sigma_k`$ is a $k$-number summary of word $i$'s behaviour across all documents — a **word embedding**, twenty years before anyone used that phrase.

> **Where this came from.** LSA was introduced in 1990 by **Scott Deerwester, Susan Dumais, George Furnas, Thomas Landauer, and Richard Harshman**, most of them at Bellcore (the research arm spun out of the Bell System breakup), and a patent was filed in 1988. The problem they were attacking was **synonymy in search**: a user types "car" and misses every document that says "automobile." Their idea was that the low-rank structure of the co-occurrence matrix already encodes synonymy, if you simply stop insisting on representing every word separately. Landauer went further and argued LSA modelled something about human vocabulary acquisition — trained on enough text, it passed the synonym section of the TOEFL exam at roughly the level of a non-native applicant. Whether that says something about cognition or about the TOEFL was debated at the time and never fully settled.

---

## 10.3 Word2vec

### The one-line idea

A word's meaning can be learned from the company it keeps — so train a model to predict a word's neighbours, and take its internal representation as the meaning.

*The distributional hypothesis (Firth, 1957): "You shall know a word by the company it keeps."*

#### The distributional hypothesis, decoded

The claim is stronger and stranger than it sounds. It says: **meaning is not a property a word carries around; it is a shadow cast by the word's distribution over contexts.** If two words appear in the same neighbourhoods, they mean similar things — and nothing else about them needs to be known.

Test it on yourself. You have never seen the word *tesgüino*, but read these four sentences: *"A bottle of tesgüino is on the table." / "Everybody likes tesgüino." / "Tesgüino makes you drunk." / "We make tesgüino out of corn."* You now know it is a fermented alcoholic drink made from maize. **You inferred that entirely from the company it keeps** — no definition required. (This example is Nida's, popularised through Jurafsky and Martin's textbook.)

▸ **Everything in this chapter after this point is a machine that does that inference at scale.** Word2vec, GloVe, BERT, and modern embedding models differ in how they count contexts and how they compress the counts, but the hypothesis underneath is identical and is stated in that one sentence from 1957.

> **Where this came from.** **John Rupert Firth** was Britain's first Professor of General Linguistics, at SOAS in London; the line appears in his 1957 essay collection *Papers in Linguistics* (in the paper "A Synopsis of Linguistic Theory, 1930–1955"). The same idea was formulated independently and more formally by **Zellig Harris** in his 1954 paper "Distributional Structure," which argued that linguistic structure could be recovered from patterns of co-occurrence alone. Harris also happens to be the man who supervised Noam Chomsky — whose entire research programme was a rejection of exactly this view, holding that distribution is not enough and that innate grammatical structure is required. The distributional side of that argument now runs every language model in the world, which is a striking outcome for a debate that ran for fifty years mostly the other way.

### Skip-gram

Maximize the probability of context words given a centre word:

$$\mathcal{L} = -\frac1T\sum_{t=1}^{T}\sum_{-m\le j\le m, j\ne0}\log p(w_{t+j}\mid w_t),\qquad p(o\mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}$$

▸ Two separate embedding matrices: $v$ for "as centre", $u$ for "as context." Final vectors are usually $v$, or $v+u$.

**The problem:** the softmax denominator sums over $|V|=10^6$ words for every training pair. Intractable.

#### Reading the skip-gram objective in plain English

Two formulas, taken one at a time.

**The loss.** $`\mathcal{L} = -\frac1T\sum_{t=1}^{T}\sum_{-m\le j\le m,\ j\ne0}\log p(w_{t+j}\mid w_t)`$ decomposes into:

| Piece | Read aloud | Meaning |
|---|---|---|
| $T$ | "T" | The number of words in the training corpus |
| $`\sum_{t=1}^{T}`$ | "sum over t from 1 to T" | Slide a window along every position in the corpus |
| $`w_t`$ | "w-t" | The **centre** word at position $t$ |
| $m$ | "m" | The **window radius** — how far left and right you look |
| $`\sum_{-m \le j \le m,\ j\ne 0}`$ | "sum over j, skipping zero" | Loop over the neighbours; $j\ne0$ excludes the centre word itself |
| $`w_{t+j}`$ | "w t-plus-j" | A **context** word $j$ steps away |
| $p(\cdot\mid\cdot)$ | "p of, given" | Conditional probability (Ch. 0 §0.9) |
| $-\frac1T$ | "minus one over T" | Average, and flip the sign so that *minimizing* means *maximizing probability* |

▸ **In one sentence: walk through the corpus, and at every word, ask the model to predict its neighbours; penalize it by how surprised it is.** With $m = 5$, every position generates up to 10 training pairs, so a 10-billion-word corpus produces on the order of $10^{11}$ prediction problems. That volume is the whole reason the method works — it is an enormous number of very weak signals.

**The probability.** $`p(o\mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}`$ is a softmax (Ch. 1 §1.4). Read it as:

- $`v_c`$ — the vector for word $c$ **acting as the centre**. Think of it as the question being asked.
- $`u_o`$ — the vector for word $o$ **acting as context**. Think of it as an answer being advertised.
- $`u_o^\top v_c`$ — their dot product: **how well does this answer match this question?** (Ch. 0 §0.8: a dot product measures alignment.)
- $\exp(\cdot)$ — make it positive, and exaggerate differences.
- $`\sum_{w\in V}`$ — divide by the same quantity computed for **every word in the vocabulary**, so the numbers become probabilities that sum to 1.

**Work a two-word example.** Suppose $d=2$, and the centre word `dog` has $`v_{\text{dog}} = (2, 0)`$. Three candidate context words have vectors $`u_{\text{bark}} = (3,0)`$, $`u_{\text{cat}} = (1,1)`$, $`u_{\text{algebra}} = (-2,0)`$. Then the scores are $6$, $2$, and $-4$. Exponentiate: $403.4$, $7.39$, $0.018$. Normalize by their sum $410.8$: **0.982, 0.018, 0.00004.** The model puts nearly all its probability on `bark`. Training nudges $`v_{\text{dog}}`$ and $`u_{\text{bark}}`$ toward each other whenever they actually co-occur, and away otherwise.

▸ **Why two matrices instead of one.** If a word used the *same* vector as centre and as context, then its score against itself would be $v^\top v = \lVert v\rVert^2$ — the largest score it can possibly produce. **Every word would predict itself as its own most likely neighbour**, which is both useless and false ("dog dog dog" is not English). Two matrices break that symmetry. It is the same argument that makes transformers use separate $`W_Q`$ and $`W_K`$ (Ch. 11 §11.1) — and it is worth noticing that the two fields arrived at it independently.

**Why the denominator is fatal.** With $\lvert V\rvert = 10^6$ and $d = 300$, one softmax denominator costs $10^6 \times 300 = 3\times10^8$ multiply-adds. You need one per training pair, and there are $\sim10^{11}$ pairs. That is $3\times10^{19}$ operations for a single epoch — at the roughly $10^{9}$ useful operations per second a single 2013-era CPU core delivered, about **$3\times10^{10}$ seconds, or nine hundred years.** The next section is not an optimization; it is the difference between possible and impossible.

> **Where this came from.** Word2vec came out of Google in 2013, from a team led by **Tomáš Mikolov** with Kai Chen, Greg Corrado, and Jeffrey Dean. Mikolov had encountered the vector-arithmetic behaviour earlier while working on recurrent neural network language models for his doctoral work in Brno and during a stint at Microsoft Research — the analogy property was a *byproduct* noticed in a model built for something else. Word2vec's real contribution was less the representation than the **speed**: by stripping the model down to a single projection layer with no hidden nonlinearity, and replacing the full softmax, the released C implementation could train on a billion words in hours on one machine. The paper that changed the field was, in engineering terms, a paper about making an old idea cheap enough to use.

### Negative sampling (SGNS) — derive this, it's asked constantly

Replace the $|V|$-way softmax with $k+1$ independent binary logistic problems: "is this (centre, context) pair real or fake?"

▸ $$\mathcal{L} = -\log\sigma(u_o^\top v_c) - \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\big[\log\sigma(-u_{w_i}^\top v_c)\big]$$

- First term: push real pairs' dot products up.
- Second: push $k$ sampled fake pairs' dot products down. $k=5$–20 for small corpora, 2–5 for large.
- Noise distribution: $`P_n(w)\propto U(w)^{3/4}`$. **The 3/4 exponent flattens the Zipf distribution** — it samples rare words more than unigram frequency would, and common words less. Purely empirical, but a robust finding.

#### Unpacking negative sampling

**SGNS** is **skip-gram with negative sampling**. The trick is a change of question, and the change is worth stating before any algebra.

> **The old question:** *"Out of a million words, which one comes next?"* — a one-in-a-million multiple-choice exam, and grading it requires reading all million options.
> **The new question:** *"Here is a pair of words. Real, or did I make it up?"* — a yes/no question, graded in one step.

You cannot ask the second question once and be done, so you ask it $k+1$ times: once about a  pair from the corpus, and $k$ times about pairs assembled from random words. **Answering "real or fake" well requires the same knowledge as ranking a million candidates, and costs $k+1$ operations instead of a million.**

Now the formula, symbol by symbol:

$$\mathcal{L} = -\log\sigma(u_o^\top v_c) - \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\big[\log\sigma(-u_{w_i}^\top v_c)\big]$$

| Piece | Read aloud | Job |
|---|---|---|
| $\sigma(z)$ | "sigmoid of z" | $1/(1+e^{-z})$ — turns any score into a probability in $(0,1)$ |
| $`u_o^\top v_c`$ | "u-o transpose v-c" | Score for the **real** (centre, context) pair |
| $-\log\sigma(\cdot)$ | "minus log sigmoid" | Binary cross-entropy for the answer "yes, real" |
| $`w_i \sim P_n`$ | "w-i drawn from P-n" | Sample a **fake** context word from the noise distribution |
| $`\mathbb{E}_{w_i\sim P_n}[\cdot]`$ | "expectation over the noise" | The average over such draws — in code, just draw one and use it |
| $`\sigma(-u_{w_i}^\top v_c)`$ | "sigmoid of minus the score" | Probability the pair is **fake**; note $\sigma(-z) = 1-\sigma(z)$ |
| $k$ | "k" | How many fakes per real pair |

**Put numbers on it.** Suppose the real pair (`dog`, `bark`) currently scores $u^\top v = 2.0$. Then $\sigma(2.0) = 0.881$, and its loss contribution is $-\log(0.881) = 0.127$ — small, because the model already believes it. Now a fake pair (`dog`, `algebra`) scores $-1.0$. Its contribution is $-\log\sigma(1.0) = -\log(0.731) = 0.313$. If instead the model had wrongly scored the fake pair at $+3.0$, its contribution would be $-\log\sigma(-3.0) = -\log(0.047) = 3.05$ — **ten times the penalty of the fake it already handles correctly, and twenty-four times the penalty of the real pair.** Gradient descent therefore spends nearly all its effort pushing apart the pairs it currently, wrongly, believes.

▸ **The one takeaway: negative sampling replaces a $\lvert V\rvert$-way softmax with $k+1$ independent yes/no questions, turning $10^6$ operations per training pair into about $6$.** That is a factor of roughly 150,000. This same manoeuvre — *don't normalize over everything, just contrast against a handful of impostors* — reappears as noise-contrastive estimation, as InfoNCE in §10.6 and Chapter 25, and as the core of every modern embedding model. Learning it once here pays out repeatedly.

> **Analogy.** You are teaching someone to recognize  banknotes. The expensive method is to have them memorize every possible forgery in the world. The cheap method is to hand them a real note and five fakes and ask them to sort. Do it a million times with different fakes and they end up knowing what "real" looks like — without ever having enumerated the forgeries.

**Why $U(w)^{3/4}$ and not $U(w)$.** $U(w)$ is the **unigram frequency** — how often word $w$ occurs, as a fraction of the corpus. Word frequencies follow **Zipf's law**: the $n$-th most common word appears roughly proportionally to $1/n$, so "the" alone is around 5–7% of English text while the 10,000th word is a rounding error. Raising to the power $3/4$ compresses that range, because the exponent is less than 1:

| Word | Frequency $U(w)$ | $U(w)^{3/4}$ | Effect after renormalizing |
|---|---|---|---|
| "the" | $0.05$ | $0.105$ | Ratio to "aardvark" drops from $50{,}000{:}1$ to about $1{,}800{:}1$ |
| a mid-frequency word | $10^{-4}$ | $10^{-3}$ | Sampled more often |
| "aardvark" | $10^{-6}$ | $5.6\times10^{-5}$ | Sampled far more often than raw frequency implies |

▸ **What flattening buys you:** if you sampled negatives at true frequency, "the" would be the fake context word in most comparisons, and the model would learn one thing extremely well — *that word X does not go with "the"* — while learning nothing about rare words. The $3/4$ exponent trades some realism for **coverage of the vocabulary**. The value was found by experiment; nobody has derived it, and the paper says so.

**Hierarchical softmax** is the alternative: arrange $V$ as a Huffman tree, predict $`\log_2|V|`$ binary decisions. $O(\log|V|)$ instead of $O(|V|)$. Better for rare words; negative sampling is better for frequent words and simpler.

**Subsampling frequent words:** discard word $w$ with probability $P(w) = 1-\sqrt{t/f(w)}$, $t\approx10^{-5}$. Removes "the", "of" from most windows, which both speeds training and improves rare-word vectors.

#### The two escape hatches, decoded

**Hierarchical softmax, in plain terms.** Instead of scoring a million words, arrange the vocabulary as the leaves of a **binary tree** and score a path down it. At each internal node you make one yes/no choice — left or right — and the probability of a word is the product of the probabilities along its path.

$$p(\text{word}) = \prod_{\text{nodes on the path}} p(\text{correct turn at that node})$$

A balanced tree over $\lvert V\rvert = 10^6$ words is about $`\log_2(10^6) \approx 20`$ levels deep. ▸ **So you evaluate 20 binary decisions instead of $10^6$ scores — a 50,000× reduction, and it is exact, not an approximation.**

A **Huffman tree** goes further: it gives *frequent* words short paths and rare words long ones. If "the" sits three levels down and "aardvark" thirty, then the common case — which is most of your training time — is three operations, not twenty.

> **Analogy.** Twenty questions. You cannot name the object I am thinking of out of a million possibilities, but you can find it with twenty well-chosen yes/no questions, because each halves the field. Hierarchical softmax is twenty questions with the questions learned rather than asked.

> **Where this came from.** **David Huffman** devised his coding scheme in 1951 as a graduate student at MIT, in Robert Fano's information theory class. Fano — who with Shannon had been attacking the same problem — offered the students a choice: sit the final exam, or write a term paper finding the most efficient way to represent symbols by bits. Huffman took the paper, worked for months without success, and by his own later account was about to give up and throw his notes away when the key idea arrived: build the tree **bottom-up** from the two rarest symbols rather than top-down. He did not know at the time that Fano and Shannon had both failed on the problem. The algorithm is provably optimal for symbol-by-symbol coding and is still in JPEG, MP3, and ZIP today.

**Subsampling, with numbers.** The rule discards word $w$ with probability $P(w) = 1 - \sqrt{t/f(w)}$, where $f(w)$ is the word's frequency in the corpus and $t \approx 10^{-5}$ is a threshold. Reading it: **if a word is rarer than the threshold, the square root exceeds 1, the discard probability goes negative, and nothing is thrown away.** Only words above the threshold get thinned.

| Word | $f(w)$ | $\sqrt{t/f(w)}$ | Discard probability |
|---|---|---|---|
| "the" | $0.05$ | $\sqrt{2\times10^{-4}} = 0.0141$ | **98.6% discarded** |
| "network" | $10^{-4}$ | $\sqrt{0.1} = 0.316$ | 68% discarded |
| "tesgüino" | $10^{-7}$ | $\sqrt{100} = 10$ | 0% — kept always |

▸ **Why deleting most instances of "the" *improves* the vectors rather than degrading them.** Two reasons, both worth being able to say aloud. First, "the" co-occurs with everything, so it carries almost no information about its neighbours — its presence in a window is pure noise. Second, and less obvious: **deleting words shortens the distance between the survivors.** With "the" removed, a window of radius 5 around "sat" now reaches "cat" and "mat" instead of stopping at function words. Subsampling widens the *effective* context for free.

### What word2vec is actually doing

▸ **Levy & Goldberg (2014):** SGNS with $k$ negatives is implicitly factorizing the shifted PMI matrix:
$$u_w^\top v_c \approx \mathrm{PMI}(w,c) - \log k,\qquad \mathrm{PMI}(w,c) = \log\frac{p(w,c)}{p(w)p(c)}$$

So word2vec is **matrix factorization of pointwise mutual information** — i.e. LSA with a better weighting. That result demystified the whole thing and is a great thing to know.

#### What "implicitly factorizing PMI" actually says

Start with **PMI — pointwise mutual information**:

$$\mathrm{PMI}(w,c) = \log\frac{p(w,c)}{p(w)\,p(c)}$$

Read the fraction first, and ignore the log. The numerator $p(w,c)$ is **how often $w$ and $c$ actually appear together**. The denominator $p(w)p(c)$ is **how often they would appear together if they were completely unrelated** — because for independent events, the joint probability is the product of the separate ones. So the ratio asks:

> *"Do these two words show up together more than coincidence would explain, and by what factor?"*

Then $\log$ converts the factor into a signed quantity centred on zero:

| Situation | Ratio | PMI | Meaning |
|---|---|---|---|
| Co-occur 10× more than chance | $10$ | $+2.3$ | Strong association ("New"/"York") |
| Exactly chance | $1$ | $0$ | Independent — knowing one tells you nothing |
| Co-occur 10× less than chance | $0.1$ | $-2.3$ | Actively avoid each other |

**A worked example.** In a corpus of $10^8$ words, suppose "York" appears $10^4$ times ($p = 10^{-4}$), "New" appears $10^6$ times ($p = 10^{-2}$), and they appear adjacent $9\times10^3$ times ($p = 9\times10^{-5}$). Chance would predict $10^{-4}\times10^{-2} = 10^{-6}$. The ratio is $9\times10^{-5}/10^{-6} = 90$, so $\mathrm{PMI} = \log 90 \approx 4.5$. That number — **4.5** — is what the model is being asked to reproduce as a dot product.

**Now the surprising claim.** Levy and Goldberg showed that when SGNS converges, its two learned vectors satisfy

$$u_w^\top v_c \approx \mathrm{PMI}(w,c) - \log k$$

In words: **the dot product of the learned vectors is the PMI of the word pair, shifted down by a constant.** The $-\log k$ is bookkeeping from the $k$ negatives — with $k=5$ it is a uniform $-1.6$ applied to every entry, which shifts everything and changes no comparison.

▸ **This is the deflation that made word2vec comprehensible.** For a year the field treated these vectors as a mysterious emergent property of neural training. The truth is that SGNS is a **matrix factorization** in disguise: it is finding a low-rank approximation of a $\lvert V\rvert\times\lvert V\rvert$ table of PMI values, by stochastic gradient descent, without ever building the table. LSA (§10.2) factorizes a table of co-occurrence counts; word2vec factorizes a table of PMIs. **Same machine, better-chosen matrix.** The "neural" part was never doing the work.

> **Analogy.** Imagine someone builds a machine that predicts house prices remarkably well, and everyone marvels at what it has learned about architecture. Then a researcher opens it up and finds it is computing price-per-square-foot times square footage. Nothing is *wrong* — the machine works — but the mystery evaporates, and now you can reason about when it will fail. Levy and Goldberg opened the box.

**Why it matters practically, not just philosophically.** Once you know the target is PMI, you can construct the PMI matrix directly and factor it with a plain SVD, and the resulting vectors are competitive. You can also see the failure mode immediately: for a pair that never co-occurs, $p(w,c) = 0$ and $\mathrm{PMI} = -\infty$. Since most word pairs never co-occur, **the matrix being factorized is mostly negative infinity**, which is why practical variants clip at zero (positive PMI, or PPMI). Word2vec sidesteps the problem by never building the matrix — it only ever samples entries.

> **Where this came from.** **PMI** predates all of this: it comes from **Kenneth Church and Patrick Hanks**, in a 1990 paper on word association norms and lexicography — they were building tools for *dictionary-makers*, who needed to know which words  belong together in an entry. **Omer Levy and Yoav Goldberg's** analysis appeared at NIPS 2014, roughly a year after word2vec's release, and they followed it with a paper showing that much of the reported advantage of neural embeddings over older count-based methods came down to **hyperparameter choices that had simply never been applied to the older methods** — window weighting, context distribution smoothing, negative-sample shifting. Applied fairly, the gap mostly closed. It is one of the cleaner examples in modern machine learning of a result that made the field more modest.

### CBOW

Predict the centre from the averaged context. Faster, better for frequent words; skip-gram is better for rare words and small corpora.

### GloVe

Fit log co-occurrence counts directly with a weighted least-squares objective:
▸ $$J = \sum_{i,j} f(X_{ij})\left(w_i^\top\tilde w_j + b_i+\tilde b_j - \log X_{ij}\right)^2,\quad f(x)=\min\left(1,(x/x_{\max})^{0.75}\right)$$
Global statistics in one pass rather than local windows. Comparable quality; the two approaches are theoretically closely related.

#### Reading the GloVe objective

**GloVe** stands for **Global Vectors**. The name is the argument: word2vec looks at one window at a time and never sees the corpus as a whole, whereas GloVe first counts *everything*, then fits vectors to the counts.

The objective is a **weighted least squares** problem — the same shape as fitting a line through points, with two twists. Take it apart:

| Piece | Read aloud | Job |
|---|---|---|
| $`X_{ij}`$ | "X-i-j" | How many times word $j$ appeared in word $i$'s context, over the whole corpus |
| $`w_i`$, $`\tilde w_j`$ | "w-i", "w-j-tilde" | The two vectors for a word — centre role and context role, as in word2vec |
| $`b_i`$, $`\tilde b_j`$ | "b-i", "b-j-tilde" | **Bias terms** — one scalar per word per role |
| $`\log X_{ij}`$ | "log X-i-j" | The **target** being fitted |
| $(\cdots)^2$ | "squared" | Squared error: penalize being off in either direction |
| $`f(X_{ij})`$ | "f of X-i-j" | **How much this pair's error should count** |

So the core of it is: *"make the dot product of two word vectors, plus two bias terms, equal the log of how often they co-occur."*

▸ **Why fit $`\log X_{ij}`$ and not $`X_{ij}`$?** Because a dot product is unbounded in both directions while counts are non-negative and Zipf-distributed — "the/of" might co-occur $10^7$ times and "aardvark/theorem" once. Fitting raw counts, the objective would care $10^7$ times more about the first pair. Taking logs collapses a range spanning $1$ to $10^7$ into a range spanning $0$ to $16.1$, which a bounded dot product can actually cover. **And there is a deeper reason: subtracting logs gives ratios, and it is the *ratios* of co-occurrence probabilities that carry meaning** — the point developed under the analogy property below.

**The biases are doing real work.** Some words are common in every context; that is a property of the word, not of the pair. The bias $`b_i`$ absorbs that per-word baseline so the dot product $`w_i^\top\tilde w_j`$ can be spent on the *interaction*. Without them, every vector would have to encode its own frequency, wasting dimensions. (This is the same reason a linear regression has an intercept.)

**The weighting function, with numbers.** $`f(x) = \min\!\left(1, (x/x_{\max})^{0.75}\right)`$ with $`x_{\max}`$ typically 100:

| $`X_{ij}`$ | $`(X_{ij}/100)^{0.75}`$ | $f$ | Effect |
|---|---|---|---|
| 1 | $0.032$ | $0.032$ | Counted at 3% weight — a single co-occurrence is mostly noise |
| 10 | $0.178$ | $0.178$ | 18% weight |
| 100 | $1.0$ | $1.0$ | Full weight |
| $10^6$ | $1000$ | $1.0$ | **Capped** — "of the" does not get to dominate the fit |
| 0 | — | not summed | Zero-count pairs are skipped entirely, which is why $\log 0$ never arises |

▸ **$f$ solves both tails at once.** Rare pairs are unreliable and get down-weighted; ultra-common pairs are reliable but uninformative and get capped. Note the exponent: **$0.75$ again**, the same number as word2vec's noise distribution, arrived at independently by a different group solving a different sub-problem. Nobody has explained the coincidence, and it is worth flagging as an honestly open loose end rather than a deep truth.

**The efficiency argument.** Word2vec must revisit a frequent pair every single time it occurs in the corpus. GloVe counts each pair once during a single pass, then trains on the table of non-zero entries — for typical corpora that is a few hundred million entries rather than tens of billions of window samples. The counting pass is the expensive part and it is embarrassingly parallel.

> **Where this came from.** GloVe came from **Jeffrey Pennington, Richard Socher, and Christopher Manning** at Stanford in 2014, explicitly framed as a response to word2vec: their paper argues that the count-based tradition (LSA and its descendants) and the prediction-based tradition (word2vec) were two ways of doing the same thing, and sets out to build the model that makes the connection explicit. Within a year, Levy and Goldberg's PMI result (above) had shown the same equivalence from the other direction. **Two groups, arguing from opposite ends, met in the middle** — and the field settled on the conclusion that the choice between them is a matter of engineering, not of principle.

### The analogy property, and its honest caveats

$$\mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man}) + \mathrm{vec}(\text{woman}) \approx \mathrm{vec}(\text{queen})$$

**Why it works:** if $`u_w^\top v_c\approx\mathrm{PMI}`$, then differences of vectors correspond to *ratios* of co-occurrence probabilities, and a relation like gender is a roughly constant ratio-shift across pairs.

▸ **The caveats you should state if asked:** standard evaluations *exclude the three input words* from the answer candidates, and without that exclusion the nearest neighbour is very often "king" itself. Analogy accuracy is also heavily driven by the offsets of the most frequent word pairs. The property is real but weaker than the famous demo suggests.

#### The analogy arithmetic, decoded

$\mathrm{vec}(\cdot)$ just means "the vector for this word." So the line reads: *"take king's vector, subtract man's, add woman's, and the result lands near queen's."*

**Why subtraction is the right operation.** A vector difference is a **direction of travel**. If $\mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man})$ is the arrow that points from "man" to "king," then that arrow means something like *"become royal."* Adding it to "woman" applies the same journey from a different starting point. The claim of the analogy property is that **the same semantic relation is the same arrow, wherever in the space you attach it** — the space is not merely clustered, it has consistent directions.

> **Analogy.** On a map, the arrow from London to Paris and the arrow from Edinburgh to Amsterdam are different journeys, but the arrow "travel 200 miles south-east" is one arrow you can start from anywhere. Word embeddings appear to encode "make it feminine" or "make it plural" or "make it past tense" as arrows of that second kind.

**Where the arrows come from, given PMI.** Since $u^\top v \approx \mathrm{PMI}$ (above), a *difference* of vectors corresponds to a *difference of logs*, and a difference of logs is the log of a **ratio**:

$$(\mathrm{vec}(a) - \mathrm{vec}(b))^\top v_c \;\approx\; \log\frac{p(c\mid a)}{p(c\mid b)}$$

▸ **So the analogy vector is asking: "which context words are more likely near $a$ than near $b$, and by how much?"** For king-minus-man, the answer is words about thrones, reigns, and crowns — and that same ratio-shift is roughly what separates "queen" from "woman." The relation lives in ratios of co-occurrence probabilities, and vector subtraction is exactly how you compute a log-ratio. GloVe's paper builds its entire objective backwards from this observation.

**Now be honest about the caveats, with the mechanics spelled out.** The standard evaluation computes $`\arg\max_{x}\ \cos(x,\ \mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man}) + \mathrm{vec}(\text{woman}))`$ **over a candidate set with king, man, and woman removed**. Without the removal, the top hit is usually "king" — for a mechanical reason: the result vector is a small perturbation of $\mathrm{vec}(\text{king})$, and small perturbations of a vector are closest to that vector. **The demo works partly because the answer you would have given is deleted from the multiple-choice options.**

▸ This is a  useful lesson beyond embeddings: **when a benchmark excludes the trivial answer, ask how much of the reported performance is the exclusion doing.** The analogy property is real — the arrows do exist, and they are not artefacts — but "king − man + woman = queen" as a slogan overstates it substantially.

> **The story behind the analogy demo.** The vector-arithmetic behaviour was not the goal of word2vec and was not first seen there. Mikolov and colleagues reported it in 2013 in work on recurrent-network language models — they were checking what the models had learned and found the offsets by inspection. The demo then travelled far beyond its evidence: within two years it appeared in textbooks, keynotes, and popular-science articles as *the* illustration of what neural networks understand. The corrective work — showing the dependence on the exclusion rule, on frequency, and on the exact similarity measure used — came from several groups over the following few years and is much less widely known than the original example. It is a clean case study in how a vivid demonstration outruns its caveats.

### Bias

Word embeddings encode the statistical regularities of the corpus, including social bias (`doctor - man + woman ≈ nurse`). Debiasing by projecting out a "gender direction" (Bolukbasi et al.) **reduces the measurement of bias more than the bias itself** — Gonen & Goldberg showed clustering structure survives the projection. This is worth knowing as an example of a general lesson: *removing a linear direction is not removing a concept*.

### The fundamental limitation

▸ **One vector per word type.** "bank" (river) and "bank" (finance) get the same vector — necessarily the average of both senses. This is what contextual embeddings fixed: ELMo (bidirectional LSTM states), then BERT, then every LLM. In a transformer, the *same* token gets a different residual-stream vector at every layer and in every context.

**fastText** partially fixes OOV by representing a word as the sum of its character $n$-grams, which also helps morphologically rich languages.

#### One vector per word type, decoded

**"Word type" versus "word token"** is the distinction doing the work here, and it is standard linguistics vocabulary worth owning. In the sentence *"the cat sat on the mat"* there are **six tokens** but **five types**, because "the" occurs twice. Word2vec learns one vector per *type*. Every occurrence of "bank" in every sentence ever written shares a single row of the embedding table.

**What that forces mathematically.** The training signal for "bank" is the average of all its contexts. Half those contexts are about rivers, half about money, so the learned vector settles somewhere between — a **weighted average of two meanings that is a good representation of neither.** In embedding space it sits in an odd place: near "money" and near "river" simultaneously, and therefore not particularly near either.

> **Analogy.** You are asked for one paint colour to represent a company whose two divisions use red and blue. You mix them and hand over purple. Purple is the correct average and it is wrong for both divisions. Static embeddings hand you purple for every ambiguous word in the language.

▸ **What contextual embeddings changed, stated precisely: the vector stops being a property of the word and becomes a property of the *occurrence*.** In a transformer, "bank" in *"river bank"* and "bank" in *"bank loan"* start from the same embedding-table row, and then attention (Ch. 11) mixes in information from the surrounding tokens, so by layer 3 the two residual-stream vectors have separated. There is no lookup table of senses anywhere — the disambiguation is computed, per sentence, every time.

**How fastText's fix works.** Represent a word as the sum of its character $n$-grams. For $n = 3$, `where` becomes `<wh`, `whe`, `her`, `ere`, `re>` plus the whole word, where `<` and `>` mark boundaries. Then:

- **OOV (out-of-vocabulary) words become representable.** A word you have never seen still has $n$-grams you have seen, so it gets a vector rather than an error.
- **Morphology comes free.** `walking`, `walked`, and `walker` share the `walk` $n$-grams, so their vectors are related by construction rather than by luck. This matters enormously for Turkish, Finnish, or Hungarian, where a single stem can generate hundreds of surface forms and no corpus contains them all.

Notice that this is the same instinct that produces subword tokenization in §10.4: **stop treating a word as an atom.** fastText applies it to embeddings; BPE applies it to the vocabulary itself.

> **Where this came from.** **ELMo** — "Embeddings from Language Models" — came from the Allen Institute for AI in 2018 (Peters et al.) and was the first widely-adopted contextual embedding: run a bidirectional LSTM over the sentence, and use its hidden states as the word's vector. The acronym is a Sesame Street reference, and it started a naming convention that produced BERT ("Bidirectional Encoder Representations from Transformers"), ERNIE, Big Bird, and Grover — a run of increasingly strained backronyms that the field eventually abandoned out of embarrassment. **fastText** came from Facebook AI Research in 2016, from a team that included Tomáš Mikolov again; having built the canonical one-vector-per-word method, he co-authored its successor.

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

#### Reading the BPE trace

The worked example above is the whole algorithm executing, so it is worth walking through line by line.

**The setup.** `{low:5, lower:2, newest:6, widest:3}` means the corpus contains "low" five times, "lower" twice, and so on. The `</w>` marker is an **end-of-word symbol** — without it, the algorithm could merge across word boundaries and learn that "the" plus "cat" is one unit. Every word starts fully split into characters, so the initial vocabulary is just the alphabet.

**Step by step:**

1. **Count adjacent pairs.** `(e,s)` appears in `newest` (6 times) and `widest` (3) = **9**. `(s,t)` likewise 9. `(l,o)` appears in `low` (5) and `lower` (2) = **7**. Ties are broken arbitrarily.
2. **Merge the winner.** `(e,s) → es`. Every occurrence in the corpus is rewritten: `n e w e s t` becomes `n e w es t`. The symbol `es` is now in the vocabulary and can itself be merged further.
3. **Repeat.** Now `(es,t)` has count 9 and wins, producing `est`. Then `(l,o)`, then `(lo,w)`.

▸ **The single most important property: merges are recursive.** `est` was built from `es`, which was built from `e` and `s`. This is why BPE builds up long units efficiently — every merge doubles the potential length of what the next merge can capture, so $k$ merges can produce pieces far longer than $k$ characters.

▸ **And notice what it found.** `est` is the English superlative suffix; `low` is a stem. **The algorithm has no notion of morphology, grammar, or meaning — it counts adjacent symbol pairs.** Yet it recovered a morpheme, because morphemes are, definitionally, the pieces that recur. Compression and linguistic structure turn out to point the same way, which is one of the quietly remarkable facts in this chapter.

> **Analogy.** Shorthand invents itself. Give a stenographer no rules and a thousand hours of transcription, and they will independently start abbreviating "-tion," "-ing," and "the" — not because they studied morphology but because those sequences keep coming back and writing them out is expensive. BPE is that process run by a computer, with "expensive" measured in vocabulary slots.

**Encoding is deterministic replay.** Training produces an ordered *list of merges*. Encoding a new string applies that list in the same order — merge 1, then merge 2, and so on. This is why the same tokenizer always produces the same tokens for the same string, and why tokenizer files ship as a merge list plus a vocabulary rather than as a model.

**Byte-level, unpacked.** Starting from the 256 possible byte values rather than from characters means the base vocabulary is finite, small, and **complete**: every file on earth is a sequence of bytes, so every possible input is expressible. There is no `<UNK>` ("unknown token") because there is nothing left to be unknown about.

The cost is real and unevenly distributed. UTF-8 encodes ASCII in **1 byte per character**, most European accented letters and Greek/Cyrillic in **2**, and most CJK and Devanagari characters in **3**. Before any merges, a page of Hindi is three times as many symbols as a page of English — and since merges are learned from a corpus that is mostly English, English gets the good merges too. ▸ **The inequity compounds twice: once in the byte encoding, once in whose text the merges were counted from.**

> **Where this came from.** BPE is not a natural language processing algorithm at all. **Philip Gage** published it in 1994 in *The C Users Journal* as a **general-purpose data compression** technique, under the title "A New Algorithm for Data Compression" — the idea being to replace the most common byte pair with an unused byte, and repeat. It was a modest, practical compression trick and it went nowhere in particular for twenty years. In 2015, **Rico Sennrich, Barry Haddow, and Alexandra Birch** at Edinburgh repurposed it for neural machine translation, to solve the rare-word problem: translation systems of the day had fixed vocabularies and simply failed on unseen words. **The algorithm running inside every large language model today was designed to make files smaller on 1990s hard drives.**

### WordPiece (BERT)

Same merge loop, different criterion. Instead of raw frequency, merge the pair that maximizes the corpus likelihood under a unigram model:

▸ $$\text{score}(a,b) = \frac{\mathrm{count}(ab)}{\mathrm{count}(a)\cdot\mathrm{count}(b)}$$

This is (a monotone transform of) pointwise mutual information — it prefers pairs that co-occur *more than chance*, rather than merely often. `un`+`##able` beats `th`+`##e` even if the latter is more frequent, because `th` and `e` are individually very common.

Marks continuations with `##`.

#### What the WordPiece score actually says

$$\text{score}(a,b) = \frac{\mathrm{count}(ab)}{\mathrm{count}(a)\cdot\mathrm{count}(b)}$$

Read the numerator as *"how often do $a$ and $b$ appear glued together?"* and the denominator as *"how often does each appear at all?"* The ratio therefore asks the better question: **"do these two go together more than their individual popularity already explains?"**

▸ **This is PMI without the logarithm** (§10.3). $\log\frac{p(ab)}{p(a)p(b)}$ is exactly pointwise mutual information; dropping the log and the corpus-size constants leaves this score, and since $\log$ is increasing, ranking by one is ranking by the other. Same quantity, cheaper to compute, and it is the third time PMI has appeared in this chapter under a different name.

**Numbers make the difference obvious.** Suppose a corpus of $10^8$ tokens, and consider merging `th` with `e` versus `un` with `able`:

| Pair | $\mathrm{count}(ab)$ | $\mathrm{count}(a)$ | $\mathrm{count}(b)$ | Frequency rank | Score $\propto$ |
|---|---|---|---|---|---|
| `th` + `e` | $6\times10^6$ | $8\times10^6$ | $2\times10^7$ | wins on raw count | $3.8\times10^{-14}$ |
| `un` + `able` | $4\times10^4$ | $3\times10^5$ | $6\times10^4$ | far rarer | $2.2\times10^{-12}$ |

▸ **`un`+`able` scores about 60× higher despite occurring 150× less often.** Raw-frequency BPE would merge `th`+`e` first, spending a vocabulary slot on a sequence you could have predicted from the fact that `th` and `e` are each individually everywhere. WordPiece spends the slot on `unable`, which is  a unit.

> **Analogy.** Two people are photographed together twenty times. If both are A-list celebrities photographed constantly, twenty joint appearances is unremarkable. If both are private citizens, twenty joint appearances means they are a couple. **The evidence in a co-occurrence is what it tells you *beyond* the base rates** — which is the same idea as IDF in §10.2, as PMI in §10.3, and as this score.

**What `##` is for.** WordPiece needs to distinguish a piece that *starts* a word from the same characters *continuing* one. `un` at the start of `unable` is a different token from `un` inside `run`, so continuations are written `##un`. This makes detokenization unambiguous: strip `##` and join, otherwise insert a space. (BPE solves the same problem from the other side, marking word-*starts* with a space character or with SentencePiece's ▁.)

### Unigram LM / SentencePiece

Runs **backwards**: start with a large candidate vocabulary, then iteratively *remove* the pieces whose deletion least increases the corpus negative log-likelihood, under a unigram model where
$$p(x) = \prod_i p(x_i)\quad\text{over the segmentation } x = (x_1,\dots,x_n)$$
optimized with EM, and where the probability of a sentence marginalizes over all segmentations (computed by a Viterbi/forward pass).

▸ Because it is probabilistic, it supports **subword regularization**: sample a different segmentation each epoch. This is data augmentation for tokenization and reliably improves translation robustness by 1–2 BLEU.

**SentencePiece** is the *implementation* (supporting BPE and Unigram) that treats input as a raw Unicode stream, encoding spaces as `▁`, so it is language-agnostic and detokenization is lossless. Used by T5, LLaMA, Gemma, Mistral.

#### Unigram tokenization, decoded

The three methods differ in **direction** more than in spirit:

| Method | Direction | Criterion |
|---|---|---|
| BPE | Bottom-up: start from characters, **add** merges | Highest raw pair frequency |
| WordPiece | Bottom-up: start from characters, **add** merges | Highest PMI-style score |
| Unigram LM | Top-down: start from a huge candidate set, **remove** pieces | Smallest damage to corpus likelihood |

**"Unigram LM" spelled out** is *unigram language model*: a model in which every piece's probability is independent of every other piece. That is what the formula says:

$$p(x) = \prod_i p(x_i)\quad\text{over the segmentation } x = (x_1,\dots,x_n)$$

Read it: *"the probability of this way of cutting the string is the product of the probabilities of the pieces you cut it into."* The $\prod$ is a `for` loop that multiplies (Ch. 0 §0.3). It is a deliberately naive model — it does not care what order the pieces come in — but you are not using it to generate text, only to score cuts.

**Work an example.** Take `unable`, with piece probabilities $p(\texttt{un}) = 0.01$, $p(\texttt{able}) = 0.005$, $p(\texttt{unable}) = 0.0002$, $p(\texttt{u}) = 0.02$, $p(\texttt{n}) = 0.03$, $p(\texttt{able}) = 0.005$.

| Segmentation | Probability | Log-probability |
|---|---|---|
| `unable` | $0.0002$ | $-8.52$ |
| `un` + `able` | $0.01\times0.005 = 5\times10^{-5}$ | $-9.90$ |
| `u` + `n` + `able` | $0.02\times0.03\times0.005 = 3\times10^{-6}$ | $-12.72$ |

The single piece `unable` wins. ▸ **Notice the built-in bias: every extra cut multiplies in another probability below 1, so more pieces always means lower probability.** The model automatically prefers the fewest, longest pieces it can justify — which is exactly what you want from a tokenizer, and you got it for free rather than by adding a length penalty.

**"Marginalizes over all segmentations," decoded.** A string of length $n$ has up to $2^{n-1}$ possible cuttings — for a 10-character word, 512. You do not enumerate them: a **Viterbi / forward pass** (dynamic programming, the same machinery as in hidden Markov models) computes either the best segmentation or the total probability over all of them in $O(n \times \text{max piece length})$ time, by solving the problem for every prefix once and reusing the answers.

**"Optimized with EM," decoded.** **EM** is **expectation–maximization**, and it handles the chicken-and-egg problem here: you need piece probabilities to find segmentations, and segmentations to estimate piece probabilities. EM alternates — **E-step:** given current probabilities, work out how each piece is being used; **M-step:** given that usage, re-estimate the probabilities. Repeat until it stops moving. The pruning loop then sits on top: estimate probabilities, compute how much the corpus likelihood would fall if each piece were deleted, drop the least useful 10–20%, and re-run.

▸ **Why probabilistic segmentation enables subword regularization.** BPE gives one segmentation per string, always. Unigram gives a *distribution*, so you can sample: `unable` this epoch, `un`+`able` next. The model consequently never sees `unable` as an indivisible atom, and learns that the pieces mean something on their own. **It is data augmentation applied to the input encoding rather than to the input** — the same idea as random crops in vision, aimed at a layer nobody previously thought of as augmentable.

**What SentencePiece actually contributes.** It is easy to conflate SentencePiece with Unigram; they are different things. Unigram is an *algorithm*; SentencePiece is an *implementation* that supports both BPE and Unigram. Its distinctive move is treating input as a raw Unicode stream with **no pre-tokenization step** — most earlier tokenizers split on whitespace first, which silently assumes the language uses spaces. Japanese, Chinese, and Thai do not. By encoding the space itself as a visible character (▁), SentencePiece makes detokenization exactly `text.replace("▁", " ")`: **lossless, reversible, and language-agnostic.**

> **Where this came from.** Both the Unigram LM method (2018) and SentencePiece (2018, with John Richardson) came from **Taku Kudo** at Google. Kudo's earlier career was in Japanese morphological analysis — he wrote MeCab, a widely used Japanese tokenizer — and the design of SentencePiece reads as someone who had spent years on a language where the whitespace assumption baked into Western NLP tooling simply does not hold. The "no pre-tokenization" decision, which looks like a small engineering choice, is the reason the same tokenizer library can be pointed at any of the world's writing systems.

### Vocabulary size: the actual trade-off

| Larger $\lvert V \rvert$ | Smaller $\lvert V \rvert$ |
|---|---|
| Shorter sequences → less attention compute ($O(T^2)$!) | Longer sequences |
| More parameters in embedding + output ($2\lvert V \rvert d$) | Fewer parameters |
| Rarer tokens → worse-estimated embeddings | Better-estimated embeddings |
| Better compression of common words | More compositional generalization |

▸ **Empirical guidance:** 32k was the BERT/LLaMA-1 standard; 100k–256k is now common (GPT-4, LLaMA-3, Gemma) because longer sequences are the more expensive resource and larger vocabularies improve multilingual fairness. The output softmax at $|V|=256$k and $d=8192$ is $2.1$B parameters — often the single largest matrix in the model.

**Compression rate matters directly:** if tokenizer A produces 10% fewer tokens than B for the same text, it is 10% cheaper to train *and* serve, and it fits 10% more into a context window.

#### Reading the vocabulary-size trade-off

Every row of that table is the same lever pulled in a different direction, so here is each one with a number attached.

**Row 1 — sequence length and the $O(T^2)$ term.** $T$ is the sequence length in tokens, and attention compares every token to every other, so its cost grows with $T^2$ (Ch. 0 §0.10, Ch. 11). Suppose a document is 10,000 tokens under a 32k vocabulary and 8,000 tokens under a 128k vocabulary — a 20% reduction. The attention cost falls by $1 - 0.8^2 = 36\%$. ▸ **A 20% better compression rate buys a 36% saving on the quadratic term.** Squaring is unforgiving in both directions, which is why vocabulary sizes have marched upward.

**Row 2 — embedding parameters.** The embedding table is $\lvert V\rvert \times d$ and, if untied, the output head is another $\lvert V\rvert \times d$, giving $2\lvert V\rvert d$:

| $\lvert V\rvert$ | $d$ | $2\lvert V\rvert d$ |
|---|---|---|
| $32{,}000$ | $4096$ | $262$M |
| $128{,}000$ | $4096$ | $1.05$B |
| $256{,}000$ | $8192$ | $4.2$B |

These are not free parameters in the useful sense — they do no computation, they are storage. In a 7B model, a 256k vocabulary would be a third of the entire parameter budget spent on a lookup table.

**Row 3 — estimation quality, which is the subtle one.** Every token's embedding row is trained only when that token appears. Split a fixed training corpus of $D$ tokens across a bigger vocabulary and each entry gets fewer updates. Under Zipf's law the tail is brutal: with a 256k vocabulary and a 1-trillion-token corpus, the most common token might appear $10^{10}$ times while the 250,000th appears a few thousand times. ▸ **Rare-token embeddings stay close to their random initialization, which is precisely the mechanism behind glitch tokens (§10.4).**

**Row 4 — compositional generalization.** If `unhappiness` is one token, the model must learn its meaning as a fresh fact. If it is `un`+`happi`+`ness`, the model can compose meaning from pieces it has seen thousands of times each. Bigger vocabularies memorize; smaller vocabularies compose.

> **Analogy.** Vocabulary size is the choice between a phrasebook and a grammar. The phrasebook gets you through the twenty situations it covers, quickly and idiomatically, and abandons you in the twenty-first. The grammar is slower for everything and never abandons you. Real systems want both, which is why the answer is neither 256 nor 256,000 but somewhere in between.

**The multilingual fairness point, quantified.** The same sentence costs very different numbers of tokens depending on the language, because the merges were counted from a corpus that was mostly English. Since commercial models are priced per token and context windows are measured in tokens, ▸ **a speaker of an under-represented language pays more money for less context to say the same thing.** Enlarging the vocabulary and training the tokenizer on more balanced data is the direct fix, and it is a large part of why the field moved from 32k to 128k–256k.

### The failure modes tokenization causes

▸ These are excellent interview material because they show you understand the input layer:

- **Arithmetic.** `1234` may tokenize as `12|34` or `1|234` depending on frequency, so digit position is inconsistent. Fix: force single-digit tokenization (LLaMA does this) or right-to-left digit grouping.
- **Character-level tasks.** "How many r's in strawberry?" is hard because the model sees ~3 tokens, not 10 characters. It has to have memorized the spelling, not observe it.
- **Rhyming, anagrams, reversal** — same cause.
- **Glitch tokens.** `SolidGoldMagikarp` and friends: tokens that appeared in the tokenizer's training corpus but almost never in the LM's, so their embeddings stayed near initialization and produce bizarre behaviour.
- **Multilingual inequity.** Some languages cost 3–5× more tokens per unit of meaning, which is a direct cost and context-length penalty.
- **Prompt-boundary effects.** Trailing whitespace changes tokenization and can measurably change outputs.

**Tokenizer-free alternatives:** ByT5 (raw bytes, 4× slower), CANINE, and **Byte Latent Transformer** (dynamic entropy-based patching). None has displaced BPE yet, but this is an active area.

#### Why these failures happen, and what they are not

Every item on that list has the same root cause, and being able to state it in one sentence is worth more than memorizing the list.

▸ **A language model does not see text. It sees a sequence of integers, and the mapping from text to integers was fixed by a frequency-counting procedure before training began.** Anything that mapping destroys is unrecoverable — not hard, not unlikely, *unrecoverable in a single forward pass*.

**Arithmetic, concretely.** Consider adding 1234 and 5678. If the tokenizer splits them as `12|34` and `56|78`, the model must learn that the token `34` in position 2 contributes $34 \times 10^0$ while `12` contributes $12\times10^2$ — and that the place value depends on how many tokens follow, which varies with the number. Now change the numbers to `1|234` and `567|8` and the whole scheme is different. **The model is being asked to do arithmetic in a positional system whose place values shift depending on which digit strings happened to be frequent in a web crawl.** Forcing one digit per token makes place value a simple function of position, and this is why several model families now do exactly that.

**"How many r's in strawberry," concretely.** Under a typical BPE vocabulary, `strawberry` is roughly `st|raw|berry` — three integers. The model receives no characters at all. To answer, it must have memorized the spelling as a *fact* about those three integers, learned from text that discussed the spelling. ▸ **This is not a reasoning failure and it is not a sign of shallow understanding; it is a question about information the model was never given.** Asking it is like asking someone to count the letters in a word they have only ever heard spoken.

Rhyming, anagrams, reversal, acrostics, and syllable counting all fail for the identical reason, which is why they cluster together in benchmarks.

**Prompt-boundary effects, concretely.** In byte-level BPE, the leading space is usually part of the token: `" hello"` and `"hello"` are *different integers*. So a prompt ending in a trailing space forces the model to continue from a token that, in training, almost always began a word — an unusual state. **A trailing space you cannot see can measurably change the output**, and this is a real source of irreproducible behaviour in production systems.

> **The story behind glitch tokens.** In early 2023, **Jessica Rumbelow and Matthew Watkins** were probing GPT-2 and GPT-3's embedding space — clustering token embeddings to see what structure existed — and found a cluster of tokens sitting near the centroid, far from everything meaningful. Prompting the models with them produced bizarre, evasive, and occasionally hostile output; asking a model to repeat `SolidGoldMagikarp` back would yield an entirely different word, or a refusal. The explanation turned out to be mundane and instructive: these strings appeared often enough in the **tokenizer's** training data to earn a vocabulary slot, but that data included sources — notably Reddit usernames, including from a subreddit devoted to users counting upward one comment at a time — that were filtered out of the **language model's** training data. The tokens therefore existed but were never trained. Their embeddings stayed at random initialization, and feeding a random vector into a trained network produces exactly what you would expect. ▸ **The lesson generalizes past this curiosity: whenever the tokenizer and the model are trained on different data, you have created inputs the model has never seen but cannot refuse.**

---

## 10.5 The embedding layer in practice

### Shape and initialization

$E\in\mathbb{R}^{|V|\times d}$. Init $\mathcal{N}(0, 0.02^2)$ (GPT convention) or $\mathcal{N}(0,1/d)$.

#### The embedding table, decoded

$E\in\mathbb{R}^{\lvert V\rvert\times d}$ reads *"E is a grid of numbers with $\lvert V\rvert$ rows and $d$ columns"* — **one row per token in the vocabulary, one column per dimension of the vector.** For $\lvert V\rvert = 128{,}000$ and $d = 4096$ that is a table with 128,000 rows and 4096 columns: 524 million numbers, about 1 GB in bf16. It is the largest single object in many models and it does no arithmetic at all; it is a phone book.

$\mathcal{N}(0, 0.02^2)$ reads *"draw each number independently from a normal distribution with mean 0 and standard deviation 0.02"* — so essentially every entry lands within $\pm0.06$ of zero (three standard deviations). The alternative $\mathcal{N}(0, 1/d)$ specifies the **variance** as $1/d$, so the standard deviation is $1/\sqrt{d}$; at $d = 4096$ that is $0.0156$, which is why the two conventions give similar numbers in practice despite looking unrelated.

▸ **Why the variance is tied to $d$.** A vector of $d$ entries each with variance $\sigma^2$ has expected squared length $d\sigma^2$. Setting $\sigma^2 = 1/d$ makes the expected squared length exactly $1$, so **every token starts as a unit-length vector pointing in a random direction**, regardless of how wide the model is. That is the same variance-preservation argument as He and Xavier initialization (Ch. 6), applied to the input layer. Get it wrong and the first LayerNorm either has almost nothing to normalize or is fed values that saturate downstream nonlinearities.

**And the geometry at initialization is not arbitrary.** By Johnson–Lindenstrauss (Ch. 1 §1.1.5), 128,000 random unit vectors in 4096 dimensions have pairwise cosine similarities of about $1/\sqrt{4096} = 0.0156$ — **effectively mutually perpendicular.** So training does not begin from a blank slate; it begins from something already better than one-hot: every token is distinct, no token is accidentally near another, and there is room for training to *pull together* the ones that belong together. Initialization hands you the separation for free and lets learning spend its effort on the structure.

### Weight tying

▸ Set the output projection $`W_{\text{out}} = E^\top`$. Saves $|V|d$ parameters (for $|V|=128$k, $d=4096$: **524M parameters**), and consistently improves perplexity in small/medium models by regularizing both matrices toward a shared space.

Argument: the input embedding maps token→vector, the output maps vector→token-logit. If the representation space is consistent, these should be transposes. Note some large models *untie* them, since at scale the extra capacity is worth more than the regularization.

#### Weight tying, decoded

A language model has **two** places where tokens meet vectors, at opposite ends of the network:

| Where | Direction | Shape |
|---|---|---|
| Input embedding $E$ | token ID → vector | $\lvert V\rvert \times d$ |
| Output projection $`W_{\text{out}}`$ | vector → one logit per token | $d \times \lvert V\rvert$ |

**Weight tying says: use the same numbers for both.** Setting $`W_{\text{out}} = E^\top`$ means the output layer is the input table, transposed. Since transposing is free (Ch. 0 §0.6), you store one matrix and use it twice.

**Why it is not merely a memory hack.** Look at what the output layer computes: the logit for token $i$ is $`E_i^\top h`$ — the dot product of token $i$'s embedding row with the final hidden state $h$. ▸ **So predicting the next token becomes "find the token whose embedding points most nearly the same way as my current hidden state."** Prediction is reduced to a nearest-neighbour search in the very space the input layer defined, which forces both ends of the network to agree on what each direction means. Untied, the model can maintain two unrelated vocabularies for the same tokens and nothing stops the two from drifting apart.

> **Analogy.** Tying is insisting that the dictionary you read with and the dictionary you write with are the same book. You save shelf space, certainly — but the real benefit is that a word cannot come to mean one thing when you read it and another when you write it.

**The numbers.** At $\lvert V\rvert = 128{,}000$ and $d = 4096$: $128{,}000 \times 4096 = 524$M parameters saved. In a 1.5B-parameter model, that is **a third of the model** — which is why tying is close to mandatory below a few billion parameters, and why the effect on perplexity is largest exactly there.

**And why large models untie.** The regularization argument cuts both ways. Tying constrains the model, and a constraint that helps when you are short of capacity hurts when you are not. Above roughly ten billion parameters the 524M is a rounding error, while the freedom to let the output head specialize — for instance, to handle the fact that *predicting* a token and *representing* one are  different jobs — is worth real perplexity. ▸ **The general shape of this argument recurs throughout the book: parameter-sharing schemes are regularizers, and regularizers stop paying once data and capacity are abundant.**

### Factorized embeddings (ALBERT)

$|V|\times d$ becomes $|V|\times e$ then $e\times d$ with $e\ll d$. Decouples vocabulary embedding size from hidden size. Reduces parameters from $O(|V|d)$ to $O(|V|e+ed)$.

#### Factorized embeddings, with numbers

**ALBERT** is **A Lite BERT**. The trick is low-rank factorization (Ch. 1 §1.1.3) applied to the embedding table: instead of one fat matrix, use two thin ones in series. A token ID looks up a small $e$-dimensional vector, which a shared $e\times d$ matrix then expands to the model width.

| Design | Parameters | At $\lvert V\rvert=128{,}000$, $d=4096$, $e=128$ |
|---|---|---|
| Direct | $\lvert V\rvert d$ | $524$M |
| Factorized | $\lvert V\rvert e + ed$ | $16.4\text{M} + 0.5\text{M} = 16.9$M |

▸ **A 31× reduction.** The symbol $\ll$ reads "much less than," and it is carrying the whole argument: the saving only exists because $e \ll d$.

**The conceptual justification, which is the interesting part.** The input embedding is *context-free* — it is a property of the token alone. The hidden state is *context-dependent* — it is a property of the token in this sentence. ▸ **There is no reason those two things need the same number of dimensions.** Tying them together, as standard transformers do, means that widening the model to give it more room to think also forces you to buy a proportionally larger lookup table you never asked for. Factorization severs that link.

**The cost.** The rank of the composed map is at most $e$, so all $\lvert V\rvert$ token embeddings are confined to an $e$-dimensional subspace of the $d$-dimensional space. With $e=128$ and 128,000 tokens, you are asking 128 dimensions to hold the distinctions among 128,000 words. Johnson–Lindenstrauss says that is geometrically possible; whether it is *enough* is empirical, and the answer has turned out to be "yes for BERT-scale models, less clearly for large generative ones," which is why the technique is common in encoders and rare in frontier language models.

### The scaling detail

The original Transformer multiplies embeddings by $`\sqrt{d_{\text{model}}}`$ before adding positional encodings. Reason: embeddings initialized with variance $1/d$ have norm $\approx1$, while sinusoidal positional encodings have entries of magnitude $\approx1$ and norm $\approx\sqrt{d}$. Without the scaling the positional signal would drown the token signal.

#### Why the $`\sqrt{d_{\text{model}}}`$ factor is there

This is a one-line detail in the original paper that people either skip or misremember, and the argument is entirely about **matching magnitudes before adding two things together.**

Follow the lengths (Ch. 0 §0.6: $\lVert\cdot\rVert$ is the length of a vector). For a vector of $d$ entries each with variance $\sigma^2$, the expected squared length is $d\sigma^2$.

| Quantity | Entry magnitude | Vector length | At $d=512$ |
|---|---|---|---|
| Token embedding, init $\mathcal{N}(0,1/d)$ | $\approx 1/\sqrt{d}$ | $\sqrt{d \cdot 1/d} = 1$ | $1$ |
| Sinusoidal positional encoding | $\approx 1$ (sines and cosines) | $\approx\sqrt{d}$ | $22.6$ |

▸ **Add those two and the token identity is a 4% perturbation of a vector that is 96% position.** The model would know exactly where every token is and almost nothing about what it is. Multiplying the embedding by $\sqrt{d} = 22.6$ puts both signals at length $\approx 22.6$, and the sum  carries both.

> **Analogy.** You are mixing two audio tracks — a voice recorded at conversational volume and a click track recorded at full scale. Summing them raw gives you a click track with a faint voice underneath. The fix is not clever engineering; it is turning the voice up before you sum. $`\sqrt{d_{\text{model}}}`$ is the gain knob.

**Two things worth knowing beyond the derivation.** First, the factor depends on the initialization convention: if embeddings are initialized at $\mathcal{N}(0, 1)$ instead of $\mathcal{N}(0,1/d)$, they already have length $\sqrt d$ and the multiplier is wrong. Implementations differ on this and it is a classic source of silent reproduction failures. Second, ▸ **the whole issue mostly evaporates in modern models**, which use learned or rotary positional encodings (Ch. 12) rather than fixed sinusoids, and which put a normalization layer immediately after the embedding — and a normalization layer's job is precisely to erase whatever scale you handed it. The factor survives in codebases largely as inheritance.

---

## 10.6 Sentence and document embeddings

### Why pooling BERT naively fails

Averaging BERT's token outputs gives sentence vectors that underperform averaged GloVe on similarity tasks. The reason is **anisotropy**: BERT's representation space occupies a narrow cone, so the average cosine similarity between *random* sentences is ~0.6, leaving little dynamic range.

#### Anisotropy, decoded

**Isotropic** means "the same in all directions" (from Greek *isos*, equal, and *tropos*, turn). **Anisotropic** is the negation: the vectors are not spread evenly over the sphere, they bunch into a narrow cone all pointing broadly the same way.

**Why that ruins cosine similarity — with numbers.** Cosine similarity ranges over $[-1, 1]$, and you would like unrelated sentences near $0$ and related ones near $1$. In a healthy high-dimensional space that is what you get: two random unit vectors in 768 dimensions have expected cosine $0$ with standard deviation $1/\sqrt{768} \approx 0.036$ (Ch. 1 §1.1.5). In BERT's raw space, two *random, unrelated* sentences score about $0.6$.

| Space | Unrelated pair | Related pair | Usable range |
|---|---|---|---|
| Isotropic | $\approx 0.00$ | $\approx 0.7$ | $0.7$ |
| BERT, raw | $\approx 0.60$ | $\approx 0.75$ | $0.15$ |

▸ **The signal has not vanished — it has been compressed into a fifth of the dial.** Every measurement error, every quantization step, every threshold you pick now costs five times as much. A retrieval system built on this does not fail loudly; it degrades, and the degradation looks like "embeddings just aren't very good," which sends people looking in the wrong place.

> **Analogy.** A thermometer that reads between 98°F and 102°F over the entire range from freezing to boiling. Nothing is broken and the ordering is preserved — but you cannot read it, because the interesting variation is smaller than the width of the needle. Contrastive fine-tuning is re-graduating the scale.

**Why it happens.** The leading explanation is the training objective. A language model's output layer must push the vectors of rare tokens somewhere, and since rare tokens receive few gradient updates and are penalized whenever they are *not* the answer — which is nearly always — they get driven together into a shared region, dragging the geometry with them. The effect was documented under the name "representation degeneration" and is a property of the softmax-over-vocabulary objective, not of transformers as such.

▸ **The practical consequence, which is the thing to remember: a model trained to predict tokens is not thereby trained to place sentences usefully in space.** Those are different objectives and they produce different geometries. This is exactly why §10.6's recipe exists, and why "just use the last hidden state as an embedding" is bad advice that sounds sensible.

### Sentence-BERT and contrastive training

Fine-tune with a siamese architecture and a contrastive or triplet objective so that cosine similarity becomes meaningful. The modern recipe (E5, BGE, GTE, Nomic) is:

1. Large-scale weakly-supervised contrastive pretraining on mined pairs (title↔body, question↔answer, citation pairs).
2. Fine-tune on labelled pairs with **hard negatives** mined from a first-stage retriever.
3. **In-batch negatives** with large batch (see InfoNCE, Ch. 25):
▸ $$\mathcal{L} = -\log\frac{\exp(\cos(q,d^+)/\tau)}{\sum_{d\in\mathcal{B}}\exp(\cos(q,d)/\tau)},\quad \tau\approx0.02\text{–}0.05$$
4. Instruction prefixes (`"query: "` / `"passage: "`) so one model serves asymmetric retrieval.

**Matryoshka representation learning:** train with the loss applied to nested prefixes of the embedding (first 64, 128, 256, … dims), so a single model yields usable embeddings at many dimensionalities. Lets you store 768-d and search at 64-d.

#### Reading the contrastive objective

$$\mathcal{L} = -\log\frac{\exp(\cos(q,d^+)/\tau)}{\sum_{d\in\mathcal{B}}\exp(\cos(q,d)/\tau)}$$

This is a softmax with a specific interpretation, so decode the pieces and then read the whole.

| Piece | Read aloud | Meaning |
|---|---|---|
| $q$ | "q" | The **query** vector |
| $d^+$ | "d-plus" | The **positive**: the document that  answers $q$ |
| $\mathcal{B}$ | "script B" | The **batch** — every document in this training step |
| $\cos(q,d)$ | "cosine of q and d" | Similarity in $[-1,1]$ |
| $\tau$ | "tau" | **Temperature** — divides every score before exponentiating |
| $-\log(\cdot)$ | "minus log" | Cross-entropy: zero loss if the fraction is 1 |

▸ **In one sentence: this is a multiple-choice exam where the choices are the other documents in the batch, and the model must pick the right one.** If the correct document gets all the probability, the fraction is 1 and the loss is 0. If it is buried among the alternatives, the loss is large.

**In-batch negatives, decoded — and why batch size is the whole game.** Nobody assembles a set of wrong answers. With $B$ query–document pairs in a batch, each query treats **the other $B-1$ documents as its negatives**, for free. So a batch of 1024 gives every query 1023 wrong answers at no additional cost. ▸ **This is why contrastive embedding training is one of the few places in deep learning where large batch size is not an efficiency choice but a *quality* choice** — it directly sets how hard the exam is, and models trained at batch 8192 measurably beat the same architecture trained at batch 256. Half the engineering in this literature (gradient caching, cross-device negative gathering) exists to make batches bigger.

**Temperature, with numbers.** Cosine similarities live in $[-1,1]$, a very narrow range to feed a softmax — untouched, the best and worst possible scores differ by a factor of only $e^2 \approx 7.4$, so even a perfect answer would receive modest probability and the gradient would be limp. Dividing by $\tau = 0.02$ multiplies every score by 50:

| Pair | $\cos$ | $\cos/\tau$ at $\tau{=}0.02$ | $\exp(\cdot)$ |
|---|---|---|---|
| Correct document | $0.90$ | $45$ | $3.5\times10^{19}$ |
| A plausible distractor | $0.80$ | $40$ | $2.4\times10^{17}$ |
| An unrelated document | $0.10$ | $5$ | $148$ |

The correct document now takes about 99.3% of the probability against that distractor. ▸ **Small $\tau$ makes the model intolerant of near-misses** — it forces the positive to beat the *hardest* negative, not merely the average one. Too small and training becomes unstable, chasing individual mislabelled pairs; $0.02$–$0.05$ is where the field has settled.

**Hard negatives, decoded.** A random negative is easy — the model separates "how do I reset my password" from a recipe for risotto on day one, and learns nothing further from that pair. A **hard negative** is a document a first-stage retriever already ranks highly but which is wrong: a passage about resetting a *router*. ▸ **Training signal comes only from mistakes you are close to making**, so the pipeline in §10.6 runs a cheap retriever first purely to manufacture difficulty.

**Instruction prefixes, decoded.** Retrieval is **asymmetric**: a short question and a long passage are different kinds of object, and the same encoder must place them in the same space. Prefixing `"query: "` and `"passage: "` tells the model which role it is encoding, letting one set of weights serve both. It is a remarkably cheap fix for what looks like it should require two models.

**Matryoshka, decoded.** Named for Russian nesting dolls. Ordinary training makes all $d$ dimensions jointly meaningful, so truncating to the first 64 gives noise. Matryoshka training applies the loss **separately** to the first 64 dimensions, the first 128, the first 256, and so on, which forces the early dimensions to be independently sufficient. The result is a **coarse-to-fine ordering**: dimension 1 carries the most, dimension 768 the least.

▸ **What it buys, in system terms:** store the full 768-dimensional vector, but run first-pass search over the first 64 dimensions — 12× less memory traffic and 12× less arithmetic — then re-rank the top few hundred candidates at full width. One model, one index, and a quality/latency dial you can move at query time rather than at training time.

> **Where this came from.** **Sentence-BERT** (Nils Reimers and Iryna Gurevych, UKP Lab, TU Darmstadt, 2019) is the paper that made this a field. Their motivating observation was brutally practical: BERT could score a sentence pair for similarity very accurately, but only by running both sentences through the network *together*. Finding the most similar pair among 10,000 sentences therefore required about 50 million forward passes — roughly 65 hours on the hardware of the day. Encoding each sentence *once* into a vector and comparing with cosine similarity reduced the same task to about five seconds. **The contribution was not accuracy; it was turning an $O(n^2)$ neural comparison into an $O(n)$ encoding plus cheap arithmetic** — the identical restructuring that makes retrieval systems possible at all.

### Similarity measures

| Measure | Formula | Note |
|---|---|---|
| Cosine | $\frac{a^\top b}{\|a\|\|b\|}$ | magnitude-invariant; the default |
| Dot product | $a^\top b$ | magnitude carries information (e.g. document "importance") |
| Euclidean | $\|a-b\|$ | for normalized vectors, $\|a-b\|^2 = 2-2\cos$ — **monotonically equivalent to cosine** |

▸ That last identity is worth remembering: **on unit-normalized vectors, ranking by cosine, dot product, and Euclidean distance give identical orderings.**

#### Why cosine, dot product, and Euclidean agree on unit vectors

The identity is three lines of algebra and it is worth doing once, because it explains why vector databases can be indifferent to which metric you name.

Expand the squared distance, using $\lVert a - b\rVert^2 = (a-b)^\top(a-b)$:

$$\lVert a-b\rVert^2 = a^\top a - 2a^\top b + b^\top b = \lVert a\rVert^2 + \lVert b\rVert^2 - 2\,a^\top b$$

Now impose **unit normalization**, $\lVert a\rVert = \lVert b\rVert = 1$. The first two terms become 1 each, and $a^\top b$ becomes exactly the cosine (since the denominator $\lVert a\rVert\lVert b\rVert$ is 1):

$$\lVert a-b\rVert^2 = 2 - 2\cos(a,b)$$

▸ **Read the relationship off the formula: as cosine goes up, distance goes down, monotonically and with no exceptions.** So sorting ascending by Euclidean distance and sorting descending by cosine produce the *same list in the same order*. Nothing is approximated.

**Check it with numbers.**

| $\cos(a,b)$ | $2-2\cos$ | $\lVert a-b\rVert$ | Interpretation |
|---|---|---|---|
| $1.0$ | $0$ | $0$ | Identical direction |
| $0.5$ | $1$ | $1$ | 60° apart |
| $0.0$ | $2$ | $1.414$ | Perpendicular — and note this is the $\sqrt2$ from one-hot vectors in §10.2, which are unit vectors and mutually perpendicular |
| $-1.0$ | $4$ | $2$ | Opposite |

**When the choice does matter.** Everything above assumed normalization. Without it:

- **Cosine** throws length away entirely. Two documents on the same topic score identically whether one is a tweet or a textbook.
- **Dot product** keeps length, so $a^\top b = \lVert a\rVert\lVert b\rVert\cos$ rewards *both* alignment and magnitude. Systems that want a notion of document quality, authority, or confidence baked into the vector use this deliberately — a longer vector wins ties.

▸ **The practical rule: normalize, and then say whichever word you like; or do not normalize, and then be very clear about which you meant.** The commonest bug in retrieval systems is a mismatch between what the index was built with and what the query uses — and because both metrics rank *plausibly*, the failure is quiet. Recall drops a few points and nothing throws an error.

---

## 10.7 Vector search

Exact search over $N$ vectors is $O(Nd)$ per query — fine to ~1M, hopeless at 1B.

### HNSW (graph-based)

Build a multi-layer proximity graph; upper layers are sparse "highways," lower layers dense. Search greedily descends from a top-layer entry point.

- Query: $O(\log N)$, very high recall (98%+).
- Memory: $O(N(d + M))$ with $M\approx16$–64 neighbours per node — the graph itself is significant.
- Parameters: `M` (connectivity), `efConstruction` (build quality), `efSearch` (query-time recall/latency trade).

#### Why exact search fails, and how HNSW escapes

**First, why $O(Nd)$ is fatal.** Exact search means computing the distance from the query to **every** stored vector: $N$ dot products of length $d$, so $Nd$ multiply-adds. Put numbers on it:

| $N$ | $d$ | Operations per query | At $10^{10}$ ops/sec |
|---|---|---|---|
| $10^6$ | $768$ | $7.7\times10^8$ | $0.08$ s |
| $10^8$ | $768$ | $7.7\times10^{10}$ | $7.7$ s |
| $10^9$ | $768$ | $7.7\times10^{11}$ | $77$ s |

A billion-vector index answering one query per minute is not a system. And note that the older trick — a k-d tree or similar space-partitioning structure — **does not work here**, because in high dimensions the partitions stop pruning anything. That is the curse of dimensionality: with everything roughly equidistant from everything else (Ch. 1 §1.1.5), a query's nearest neighbour is barely nearer than its thousandth, so a tree cannot rule out branches.

▸ **This is why every method in this section is *approximate*. ANN stands for approximate nearest neighbour, and the "approximate" is not laziness — exact nearest-neighbour search in high dimensions is believed to require essentially linear scanning.** You buy speed by accepting that you will occasionally miss a true neighbour.

**How HNSW searches, concretely.** **HNSW** is **hierarchical navigable small world**. Build a graph where each vector is a node connected to $M$ of its near neighbours, then stack several such graphs: the bottom layer contains everything, and each layer up contains a random sample of maybe a tenth of the layer below, with longer-range links.

A query then:
1. Enters at a single node in the sparse **top** layer.
2. Repeatedly hops to whichever neighbour is closer to the query, until no neighbour improves.
3. Drops to the next layer down and repeats, now with finer-grained links.
4. At the bottom layer, keeps a candidate list of size `efSearch` and returns the best $k$.

> **Analogy.** Getting to a specific house in a foreign country. You do not consult a list of every address. You fly to the nearest international airport (top layer, huge hops), take a train to the region (middle layer), a bus to the town, and then walk street by street (bottom layer, one house at a time). **Each layer is the right granularity for one stage of the journey**, and the total number of moves is logarithmic in the number of houses rather than linear.

**Reading the parameters.**

- **`M`** — how many neighbours each node keeps. Higher means better recall and more memory; the graph itself costs $O(NM)$ on top of the $O(Nd)$ vectors. At $M=32$ and 4-byte integers, that is 128 bytes per vector of pure bookkeeping — comparable to a compressed vector, which is why "the graph itself is significant" in the text above.
- **`efConstruction`** — how hard the builder searches when deciding each node's neighbours. Paid once, at index build.
- **`efSearch`** — how many candidates to keep alive during a query. ▸ **This is the recall dial, and it is adjustable at query time without rebuilding anything** — the single most useful operational property of HNSW.

> **Where this came from.** HNSW is due to **Yury Malkov and Dmitry Yashunin**, published around 2016, and it builds directly on the **"small world" network** idea from graph theory: **Duncan Watts and Steven Strogatz** showed in a 1998 *Nature* paper that adding a small number of random long-range links to an otherwise local network collapses the average path length to logarithmic while keeping local clustering intact. That paper was itself formalizing **Stanley Milgram's** 1960s "six degrees of separation" letter-forwarding experiments. ▸ **The insight that makes billion-scale vector search possible is the same one that explains why a chain of acquaintances connects any two people on earth: you need mostly local links plus a few long ones.**

### IVF + Product Quantization (compression-based)

**IVF:** k-means the corpus into $`n_{\text{list}}`$ cells; search only the $`n_{\text{probe}}`$ nearest cells.

**PQ:** split each $d$-dim vector into $m$ sub-vectors, k-means each sub-space into 256 centroids, store $m$ bytes per vector.
▸ Compression: $d\times4$ bytes → $m$ bytes. For $d=768$, $m=96$: $3072\to96$ bytes, a **32× reduction**. Distances are computed by table lookup and summation, which is extremely fast.

**Choosing:** HNSW for ≤10M vectors with a memory budget; IVF-PQ beyond that; ScaNN (anisotropic quantization) when maximum-inner-product recall matters most.

#### Unpacking IVF and product quantization

These are two independent ideas that ship together because they solve different halves of the problem: **IVF reduces how many vectors you look at; PQ reduces how much each vector costs to store and compare.**

**IVF — inverted file index.** The name is borrowed from text search, where an "inverted index" maps each word to the documents containing it. Here, k-means clustering (Ch. 24) partitions the corpus into $`n_{\text{list}}`$ cells, each with a centroid. At query time you compare the query to the $`n_{\text{list}}`$ **centroids only**, then search exhaustively inside the $`n_{\text{probe}}`$ nearest cells.

Numbers, for $N = 10^8$ vectors and $`n_{\text{list}} = 65{,}536`$:

| Step | Comparisons |
|---|---|
| Query vs. centroids | $65{,}536$ |
| Vectors inside $`n_{\text{probe}}=32`$ cells | $32 \times \frac{10^8}{65{,}536} \approx 48{,}800$ |
| **Total** | $\approx 114{,}000$ instead of $10^8$ |

▸ **A 900× reduction in vectors examined.** And the failure mode is visible in the arithmetic: if the true nearest neighbour sits in cell number 33, you never see it. That is the approximation, and $`n_{\text{probe}}`$ is the dial that trades recall against latency — exactly parallel to `efSearch` in HNSW.

> **Analogy.** Finding a book in a library. You do not walk every shelf; you go to the right section, then the right few shelves. If the book has been mis-shelved into the neighbouring section, you will not find it. Raising $`n_{\text{probe}}`$ is checking the adjacent sections too.

**PQ — product quantization.** "Quantization" means replacing a continuous value with the nearest entry from a small fixed set. "Product" refers to a Cartesian product: the vector is chopped into $m$ chunks and each chunk is quantized independently, so the effective codebook is the product of $m$ small codebooks.

Walk it through with the text's numbers, $d = 768$ and $m = 96$:

1. **Split.** Cut each 768-dimensional vector into 96 sub-vectors of 8 dimensions each.
2. **Learn codebooks.** For each of the 96 slots, run k-means over the whole corpus with $k = 256$ centroids. You now have 96 little codebooks of 256 entries.
3. **Encode.** Replace each sub-vector with the ID of its nearest centroid — a number from 0 to 255, i.e. **exactly one byte**. A vector becomes 96 bytes.

| Representation | Size per vector | For $10^9$ vectors |
|---|---|---|
| Raw fp32 | $768\times4 = 3072$ bytes | $3.1$ TB |
| PQ, $m=96$ | $96$ bytes | $96$ GB |

▸ **32× compression — the difference between an index that needs a rack of machines and one that fits on a single large server.**

**Why the effective codebook is astronomically large.** Each vector is described by 96 independent choices out of 256, so the number of representable vectors is $256^{96} = 2^{768}$. Quantizing all 768 dimensions jointly with a single codebook of that size is obviously impossible; splitting into chunks makes it trivial. **That factorization is the entire trick, and it is the same "decompose the problem into independent sub-problems" move that makes multi-head attention and separable convolutions work.**

**Why comparisons get *faster*, not just smaller.** This is the part people miss. To compare a query against all encoded vectors, first compute — once per query — a table of the distance from each of the query's 96 sub-vectors to each of that slot's 256 centroids. That is $96\times256 = 24{,}576$ distance computations, done once. Then the approximate distance to **any** stored vector is 96 table lookups and 95 additions: ▸ **no multiplications at all.** This is called asymmetric distance computation, and it is why PQ is fast on ordinary CPUs rather than merely compact.

> **Where this came from.** Product quantization was introduced by **Hervé Jégou, Matthijs Douze, and Cordelia Schmid** at INRIA in 2011, for searching large collections of image descriptors — the problem of the day was finding near-duplicate photographs among millions, long before text embeddings mattered. The same group later built **FAISS** at Facebook AI Research, which is why the library that most vector databases are built on top of has its roots in computer vision. The clustering underneath it all is **Lloyd's algorithm**, written down by **Stuart Lloyd** at Bell Labs in 1957 as an internal memo about pulse-code modulation — how to pick the quantization levels for digitizing an analogue signal. It went unpublished until 1982. The name "k-means" was coined separately by James MacQueen in 1967. ▸ **Compressing a billion neural embeddings uses, essentially unmodified, a method invented to digitize telephone calls.**

### The metric that matters
**Recall@k vs latency**, at fixed memory. Every ANN method is a point on that trade-off surface; there is no free lunch.

#### Recall@k, decoded

**Recall@k** answers: *"of the $k$ items a perfect exhaustive search would have returned, how many did my approximate index actually return?"* Read the `@` aloud as "at."

If the true 10 nearest neighbours are $\{A,B,C,D,E,F,G,H,I,J\}$ and your index returns $\{A,B,C,D,E,F,G,H,I,Z\}$, then Recall@10 is $9/10 = 0.90$. **Note what it does not measure: order.** Getting all ten in the wrong sequence still scores 1.0, which is why retrieval systems report ranking metrics too, and why a re-ranking stage almost always follows the ANN stage.

**Why the trade-off is a surface and not a curve.** There are three quantities, and you may fix any two:

| Fix | Then you are trading |
|---|---|
| Memory and latency | recall — how many neighbours you miss |
| Memory and recall | latency — how long each query takes |
| Recall and latency | memory — how many machines you rent |

▸ **"There is no free lunch" is the precise statement that no method dominates on all three.** HNSW buys speed and recall with memory (the graph). IVF-PQ buys memory with recall (quantization error). Exact search buys recall with everything else. **When someone claims a new index is simply better, the first question is which of the three axes they held fixed** — and the second is at what scale, because the ranking changes between a million vectors and a billion.

#### Examples and non-examples: what is a token?

Almost every beginner assumes tokens are words. They are not, and the gap causes real bugs.

**✅ Things that  are single tokens** (typical byte-pair-encoding vocabulary)

| Example | Why |
|---|---|
| `" the"` — with the leading space | Space-prefixed common words are single tokens |
| `" understanding"` | Frequent enough to earn its own entry |
| `"!"` | Punctuation is usually its own token |

**❌ Near-misses — look like one token, but aren't**

| Looks like one token | Reality | Why |
|---|---|---|
| A rare surname | Often 4–6 tokens | Rare strings get split into fragments |
| `"1234567"` | Several tokens, split unpredictably | Digit grouping is an artifact of the merge order, not a rule |
| A word in a low-resource language | Frequently one token **per byte** | The vocabulary was fitted mostly on English |
| An emoji | Often 2–4 tokens | Multi-byte UTF-8 sequences get split |
| `"the"` vs `" the"` | **Two different tokens** | The leading space is part of the token |

▸ **The boundary:** a token is whatever the tokenizer's learned merge table says it is — a frequency artifact of the training corpus, not a linguistic unit. **This is why the same sentence costs more tokens in Hindi than in English, why models are bad at arithmetic and at counting letters in a word, and why "how many tokens is this?" has no answer without naming the tokenizer.**

> **Common misconception.** *"Tokens are words, roughly."* For English prose the ratio is around 0.75 words per token, which makes the approximation feel safe. It fails exactly where it matters: code, numbers, names, URLs, and non-English text. A model that cannot see that `"strawberry"` contains three `r` characters isn't failing at reasoning — it never received the letters as separate units in the first place.

> **Common misconception.** *"An embedding layer is a matrix multiplication, so it's expensive."* Mathematically it multiplies a one-hot vector by the embedding matrix; computationally it is an array index. Doing it literally at $\lvert V\rvert = 128{,}000$ and $d = 4096$ would be 524 million multiply-adds — nearly all by zero — to fetch 4,096 numbers. **The math and the implementation deliberately diverge**, which is why `nn.Embedding` exists as its own layer rather than being a linear layer.

> **Common misconception.** *"Lower perplexity means a better language model."* Perplexity is per **token**, and tokens differ between tokenizers. A model with a larger vocabulary emits fewer, more informative tokens per sentence, which changes per-token perplexity without changing quality at all. **Comparing perplexity across tokenizers is meaningless** — a mistake that has appeared in published papers.

#### Examples and non-examples: what cosine similarity actually tells you

**✅ Reliable uses**

| Use | Why it works |
|---|---|
| Ranking candidates against one fixed query | Relative order within one query is meaningful |
| Retrieval with an embedding model trained contrastively | Trained *specifically* so cosine means relevance |
| Nearest-neighbour lookup in a well-fitted space | The metric matches the training objective |

**❌ Near-misses**

| Tempting use | Why it fails |
|---|---|
| Comparing scores across *different* queries | Similarity scores aren't calibrated between queries |
| Using raw BERT embeddings for similarity | Untuned language-model spaces are **anisotropic** — everything looks similar |
| Treating 0.85 as "85% related" | The number has no absolute interpretation |
| Assuming a threshold transfers between models | Each model's scale differs |
| Expecting antonyms to be far apart | "hot" and "cold" appear in near-identical contexts, so they embed *close together* |

▸ **The boundary:** cosine similarity measures **distributional similarity — do these appear in similar contexts?** — not semantic equivalence. That's why antonyms cluster, and it's the single most surprising property of word embeddings for newcomers.

---

## Did you know?

- **BPE was invented to compress files, not to tokenize language.** Philip Gage published it in 1994 in *The C Users Journal* as a general-purpose data compression algorithm. It sat unused by the language community for twenty-one years until Sennrich, Haddow, and Birch repurposed it for machine translation in 2015.

- **TF-IDF's rarity term is Shannon's surprisal, though nobody planned it that way.** $\log(N/\mathrm{df})$ is exactly the self-information of the event "this document contains term $t$." Karen Spärck Jones derived it from empirical retrieval experiments in 1972; the information-theoretic reading was recognized afterwards.

- **BM25's name is an experiment number.** *BM* stands for "Best Match," and the *25* is the twenty-fifth weighting function the Okapi project at City University, London, tried. The formula behind Elasticsearch and Lucene is, by its own name, attempt number twenty-five.

- **"One-hot" is borrowed from digital circuit design**, where a one-hot register is one in which exactly one bit is high at a time. The term arrived in machine learning fully formed from hardware engineering.

- **Word2vec's famous analogy demo works partly because the obvious answer is deleted from the options.** Standard evaluations exclude the three input words from the candidate set. Without that exclusion, the nearest vector to king − man + woman is very often "king."

- **The 3/4 exponent appears twice, independently, in this chapter.** Word2vec raises unigram frequencies to the power 3/4 to pick negative samples; GloVe raises co-occurrence counts to the power 3/4 to weight its least-squares terms. Different teams, different sub-problems, same magic number, and no theory explaining it.

- **Word2vec turned out to be matrix factorization in disguise.** Levy and Goldberg showed in 2014 that skip-gram with negative sampling implicitly factorizes a shifted pointwise-mutual-information matrix — making it a better-weighted cousin of Latent Semantic Analysis from 1990, not a new kind of object.

- **Huffman coding, which powers hierarchical softmax, was a homework assignment.** David Huffman devised it in 1951 as an MIT graduate student, after Robert Fano offered his class a choice between a final exam and a term paper on optimal symbol coding. Huffman did not know that Fano and Shannon had both already tried and failed to solve it.

- **Glitch tokens exist because the tokenizer and the model were trained on different data.** Strings like `SolidGoldMagikarp` earned vocabulary slots from a corpus that included Reddit usernames, but that corpus was filtered before language-model training. The tokens were never trained, so their embeddings remained at random initialization — and feeding a random vector into a trained network produces exactly the incoherence you would expect.

- **The algorithm compressing billion-vector search indexes was invented to digitize phone calls.** Lloyd's algorithm — the clustering step inside product quantization — was written as a Bell Labs internal memo in 1957 about choosing quantization levels for pulse-code modulation. It went unpublished for twenty-five years.

- **The insight behind billion-scale vector search is the same one behind "six degrees of separation."** HNSW rests on small-world graph theory: Watts and Strogatz showed in 1998 that a handful of long-range links collapses path lengths to logarithmic, formalizing Milgram's 1960s letter-forwarding experiments.

- **Sentence-BERT's contribution was a change of asymptotics, not of accuracy.** Finding the most similar pair among 10,000 sentences required about 50 million paired forward passes through BERT — roughly 65 hours. Encoding each sentence once and comparing vectors cut the same task to about five seconds.

- **Latent Semantic Analysis, at 300 dimensions, once passed the synonym section of the TOEFL** at roughly the level of a non-native applicant — in 1997, using nothing but a truncated SVD of word counts. Whether that revealed something about cognition or about the TOEFL was argued at the time and never settled.

- **In UTF-8, a page of Hindi is three times as many bytes as a page of English before tokenization even begins**, and the merges are then learned from a corpus that is mostly English. The multilingual token-cost inequity compounds twice, and it is charged to users as money and as context length.

---

## Check for Understanding

**Tokenization decides what the model can perceive at all — which is why LLMs are bad at spelling and arithmetic — and embedding turns those tokens into a geometry where similarity is distance, learned either from co-occurrence statistics (word2vec, which is implicitly factorizing PMI) or from context inside a transformer, where a token's vector finally depends on the sentence it appears in.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is every pair of one-hot vectors exactly the same distance apart, and why does that make one-hot encoding useless as a representation of meaning?**
2. **Why does the IDF term make rare words count more?** Answer it twice — once as "clues that eliminate more candidates are worth more," and once in terms of information.
3. **What does BM25's saturation term prevent, and why did search engines need it?** (Hint: think about who benefits when scores are linear in word count.)
4. **State the distributional hypothesis without using the word "vector."** Then explain how you learned what *tesgüino* means.
5. **Why is the softmax denominator in skip-gram intractable, and what question does negative sampling ask instead?**
6. **What does it mean that word2vec is "implicitly factorizing a PMI matrix"?** Why did that result make the field more modest rather than less?
7. **Why does an algorithm that only counts adjacent character pairs end up discovering the suffix "-est"?**
8. **Explain why a language model cannot count the r's in "strawberry"** — without saying that it is bad at reasoning.
9. **Why does weight tying help small models and hurt very large ones?** What general principle about regularizers does that illustrate?
10. **Why do BERT's raw sentence vectors give an average cosine similarity of 0.6 between unrelated sentences, and why does that break retrieval?**
11. **Why is large batch size a quality decision in contrastive training, rather than an efficiency one?**
12. **Why is exact nearest-neighbour search hopeless at a billion vectors, and what does an approximate index give up in exchange for speed?**
13. **Explain product quantization to someone who knows what a codebook is** — and say why splitting the vector into chunks is the whole trick.

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 11 — Attention & the Transformer](11-attention-and-transformers.md)
