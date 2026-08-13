# Chapter 14 — Training LLMs at Scale

> **Prerequisites:** Ch. 11, 13.
> **Scope:** everything between "I have an architecture" and "I have a trained model." This is where most real-world ML engineering time actually goes.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`J(A,B)`$ | "Jaccard of A and B" | The **overlap fraction** of two documents: shared pieces divided by total distinct pieces |
| $`\lvert A \cap B\rvert`$ | "size of A intersect B" | How many things are in **both** sets. The bars mean "count," not absolute value |
| $`\lvert A \cup B\rvert`$ | "size of A union B" | How many distinct things are in **either** set |
| $`\pi`$ (in §14.1) | "pi" | A **random shuffle** of a list. Not 3.14159, and not a policy — see the trap below |
| $`k,\ b,\ r`$ | "k, b, r" | Number of hash functions, number of bands, rows per band. They satisfy $`k = br`$ |
| $`N`$ | "N" | The number of **parameters** in the model. "7B" means $`N = 7\times10^9`$ |
| $`D`$ | "D" | The number of **training tokens** the model will see |
| $`C = 6ND`$ | "C equals six N D" | Total training cost, in **floating-point operations** (FLOPs) |
| $`P`$ | "P" | The number of **devices** (GPUs) you are training across |
| $`B,\ T,\ L`$ | "B, T, L" | Batch size, sequence length (tokens per example), number of layers |
| $`S`$ | "S" | The **loss scale** — a large constant you multiply the loss by, in fp16 only |
| $`\hat m,\ \hat v`$ | "m-hat, v-hat" | Adam's running average of the gradient, and of the squared gradient |
| $`\mathrm{tr}(\Sigma)`$ | "trace of Sigma" | Total gradient **noise**, summed over every parameter |
| $`\mathcal{L}_z`$ | "script L sub z" | The **z-loss** — a small penalty that stops output logits drifting upward together |
| $`\mathcal{O}(\sqrt{L})`$ | "big-O of root L" | "Grows like the square root of the depth" |

▸ **The $`\pi`$ trap, worth naming now.** In §14.1, $`\pi`$ is a *random permutation* — a shuffle of a list. In Chapter 16 the identical letter means a *policy*. In geometry it means 3.14159. Greek letters are recycled shamelessly across machine learning, and **the meaning always comes from the sentence, never from the glyph.** The same warning applies to $`\Sigma`$ here: in §14.5 it is the **covariance matrix of the gradient noise**, not the summation sign.

### Abbreviations used in this chapter

| Short | Full form |
|---|---|
| bf16 | brain floating point, 16-bit |
| DP | Data Parallelism |
| DRO | Distributionally Robust Optimization (the "group-DRO" inside DoReMi) |
| E4M3 / E5M2 | fp8 layouts: 4 exponent bits and 3 mantissa bits, or 5 and 2 |
| FLOP | FLoating-point OPeration |
| fp8 / fp16 / fp32 | floating point, 8-/16-/32-bit |
| FSDP | Fully Sharded Data Parallel |
| HBM | High-Bandwidth Memory (the GPU's own on-package RAM) |
| LSH | Locality-Sensitive Hashing |
| LR | Learning Rate |
| MFU | Model FLOPs Utilization |
| MoE | Mixture of Experts |
| NVLink | NVIDIA's high-bandwidth direct GPU-to-GPU interconnect |
| PII | Personally Identifiable Information |
| PP | Pipeline Parallelism |
| SXM | NVIDIA's socketed GPU module (more power and bandwidth than a PCIe card) |
| TP | Tensor Parallelism |
| WSD | Warmup–Stable–Decay (learning-rate schedule) |
| ZeRO | Zero Redundancy Optimizer |

---

## 14.1 Data

### The one-line idea

At scale, data quality and composition determine model quality more than architecture does — and almost all of the work is deduplication and filtering, not collection.

### The pipeline

**1. Sourcing.** Common Crawl (raw web), refined derivatives (C4, RefinedWeb, FineWeb, DCLM), code (GitHub, The Stack), books, papers (arXiv, PubMed), Wikipedia, Q&A forums, and increasingly **synthetic data**.

**2. Language identification and extraction.** HTML → text (trafilatura, resiliparse). Boilerplate removal matters enormously.

**3. Quality filtering.**
- *Heuristic*: mean word length, symbol-to-word ratio, fraction of lines ending in punctuation, stopword presence, repeated-line fraction. (Gopher's rules are the canonical set.)
- *Model-based*: a classifier trained to distinguish a high-quality reference corpus from raw web, applied as a filter or a sampling weight. **This is now the dominant approach** (DCLM, FineWeb-Edu) and is worth more than any heuristic set.
- *Perplexity filtering* against a reference model — but beware: it selects for text resembling the reference model's training data, which narrows diversity.

**4. Deduplication.** The highest-value single step.

▸ **Exact duplicates** are found by hashing. **Near-duplicates** require MinHash + LSH:
- Represent a document by its set of $`n`$-grams (typically $`n=5`$ words).
- **Jaccard similarity**: $`J(A,B)=\frac{|A\cap B|}{|A\cup B|}`$.
- **MinHash:** for a random permutation $`\pi`$ of the universe, $`\Pr[\min\pi(A)=\min\pi(B)] = J(A,B)`$ **exactly**. Use $`k`$ hash functions to get a $`k`$-dim signature; the fraction of matching entries estimates $`J`$ with standard error $`\approx\sqrt{J(1-J)/k}`$.
- **LSH banding:** split the $`k`$ signature entries into $`b`$ bands of $`r`$ rows ($`k=br`$). Two documents become candidates if any band matches exactly. Probability of becoming a candidate:
▸ $$P(\text{candidate}) = 1-(1-J^r)^b$$
This is an S-curve with threshold near $`J^*\approx(1/b)^{1/r}`$. With $`k=128`$, $`b=16`$, $`r=8`$: $`J^*=16^{-1/8}\approx0.71`$ — tune $`b,r`$ to place the cutoff where you want it.

#### Unpacking MinHash and LSH

That block stacks four ideas on top of one another, and each is simple alone. Here they are slowly.

**First, the problem, with a real number attached.** Suppose you hold $`10^8`$ documents and want to find every near-duplicate pair. Comparing every pair means about $`\tfrac12(10^8)^2 = 5\times10^{15}`$ comparisons. At a billion comparisons per second that is **160 years**. Any method that so much as *glances* at every pair is dead on arrival. Everything below exists to avoid looking at almost all of them.

**Step 1 — turn a document into a set.** An **$`n`$-gram** is a window of $`n`$ consecutive words slid along the text. With $`n=3`$, the sentence *"the cat sat on the mat"* becomes

$$A = \{\text{"the cat sat"},\ \text{"cat sat on"},\ \text{"sat on the"},\ \text{"on the mat"}\}$$

The document is now a **set**: order thrown away, repeats thrown away. Two pages that differ only by a changed headline or an injected advertisement still share nearly all their $`n`$-grams, which is exactly the robustness you want.

**Step 2 — Jaccard similarity is "shared over total."**

$$J(A,B) = \frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$$

Read aloud: *"the size of A-intersect-B, over the size of A-union-B."* Decoding each symbol: $`\cap`$ ("intersect") means the things in **both**; $`\cup`$ ("union") means the things in **either**; and vertical bars around a set mean **how many things are in it** — the same bars that mean absolute value around a number, doing a completely different job. Context tells you which.

Tiny worked example. Let $`A=\{1,2,3,4\}`$ and $`B=\{3,4,5\}`$:

$$A\cap B=\{3,4\}\ \Rightarrow\ \lvert A\cap B\rvert=2, \qquad A\cup B=\{1,2,3,4,5\}\ \Rightarrow\ \lvert A\cup B\rvert=5$$

$$J(A,B) = \frac{2}{5} = 0.4$$

$`J=1`$ means identical, $`J=0`$ means nothing shared. Web-scale pipelines typically treat anything above $`J\approx0.8`$ as a duplicate.

**Step 3 — MinHash estimates $`J`$ without ever computing $`J`$.** This is the clever part, and it has a one-sentence proof.

> **Analogy — the shuffled queue.** Line up every $`n`$-gram in the world in a random order; that shuffle is what the permutation $`\pi`$ is. Now ask document $`A`$: *"of your $`n`$-grams, which one stands nearest the front of the queue?"* Ask $`B`$ the same. **They name the same $`n`$-gram exactly when the front-most member of the combined crowd $`A\cup B`$ happens to belong to both of them.** Because the shuffle was uniform, every member of $`A\cup B`$ is equally likely to be standing at the front, so the chance it is one of the shared ones is $`\lvert A\cap B\rvert / \lvert A\cup B\rvert`$ — the Jaccard similarity, exactly.

That paragraph *is* the proof of $`\Pr[\min\pi(A)=\min\pi(B)] = J(A,B)`$. Nothing is approximated; the equality is exact for a uniformly random shuffle. In practice nobody shuffles the universe — you apply a hash function to each $`n`$-gram and take the smallest hash value, which behaves like a shuffle and costs one pass over the document.

**Step 4 — repeat $`k`$ times to get an error bar.** One shuffle gives you a single coin flip that lands heads with probability $`J`$. That tells you almost nothing. Do it with $`k`$ independent hash functions and you get a **signature**: a list of $`k`$ numbers per document. The fraction of positions where two signatures agree is an unbiased estimate of $`J`$, and because it is a mean of $`k`$ independent coin flips, its standard error is the familiar binomial $`\sqrt{J(1-J)/k}`$.

▸ **Put numbers on it.** With $`k=128`$ and a true $`J=0.8`$: standard error $`=\sqrt{0.8\times0.2/128} = \sqrt{0.00125}\approx 0.035`$. So a 128-number signature pins down the similarity of two documents to about $`\pm0.04`$ — and a document of 10,000 words has been compressed into 128 integers with a known error bar. **That compression is the entire reason the method scales.**

**Step 5 — LSH banding turns the estimate into a lookup.** Even with signatures, comparing all $`5\times10^{15}`$ pairs is still impossible. Banding removes the comparison step altogether. Chop the $`k=128`$ signature into $`b=16`$ bands of $`r=8`$ rows each ($`k=br`$, which is all that equation is saying). Hash each band to a bucket. **Two documents are ever compared only if at least one of their 16 bands lands in the same bucket.**

A band matches only if all $`r=8`$ of its entries agree, which happens with probability $`J^r`$. It fails with probability $`1-J^r`$. All $`b`$ bands fail with probability $`(1-J^r)^b`$. So

$$P(\text{candidate}) = 1-(1-J^r)^b$$

is just *"one minus the chance that every band missed."*

| True $`J`$ | $`J^8`$ | $`P(\text{candidate}) = 1-(1-J^8)^{16}`$ |
|---|---|---|
| 0.3 | 0.000066 | 0.001 |
| 0.5 | 0.0039 | 0.061 |
| 0.7 | 0.058 | 0.61 |
| 0.8 | 0.168 | 0.95 |
| 0.9 | 0.43 | 0.9999 |

▸ **Read that column as a switch, not a curve.**  near-duplicates ($`J\ge0.8`$) are caught essentially always; unrelated documents ($`J\le0.5`$) are examined about six times in a hundred; and documents at $`J=0.3`$ are never looked at at all. You get a *sharp threshold out of a smooth quantity*, and you get it in one pass over the corpus instead of 160 years of pairwise work.

**What $`b`$ and $`r`$ each control.** $`r`$ (rows per band) sets **where** the cliff sits — a longer band is a stricter test, pushing the threshold up. $`b`$ (number of bands) sets **how many lottery tickets** each pair holds — more bands means more chances to be noticed, pulling the threshold down and sharpening the curve. The approximation $`J^*\approx(1/b)^{1/r}`$ locates the steep region; at $`J^*=0.71`$ in the table above the true probability is around $`0.65`$, not exactly $`0.5`$, so treat it as a dial setting rather than a guarantee.

> **Where this came from.** MinHash was invented by Andrei Broder at Digital Equipment Corporation's research lab in the late 1990s, for the AltaVista search engine — the problem was that the web was full of mirrored and near-mirrored pages and the index was drowning in them. His 1997 paper *On the Resemblance and Containment of Documents* introduced the sketching trick above. Locality-sensitive hashing as a general framework came shortly after, from Piotr Indyk and Rajeev Motwani in 1998, aimed at approximate nearest-neighbour search rather than duplicate detection. **The pleasing symmetry is that a technique built to stop a search engine from indexing the same web page twice is now used to stop a language model from memorizing the same web page twice.** The problem never changed; only what we do with the corpus afterwards did.

> **The story behind the Jaccard index.** Paul Jaccard was a Swiss botanist, and he devised his coefficient around 1901 to compare the plant species found on different patches of the Alps: two meadows are similar if they share many species relative to the total number of species between them. There is no probability, no hashing, and no computer anywhere in the original idea — it is a way of counting flowers. The formula reached web-scale text deduplication a full century later, entirely unchanged.

**Why dedup matters so much:** duplicated text is memorized rather than generalized; it inflates evaluation via contamination; and Lee et al. showed dedup lets models reach the same loss with substantially less compute while emitting ~10× less verbatim training data.

#### Why one duplicated paragraph is worse than one missing paragraph

The asymmetry here is not obvious and is worth stating plainly.

**Duplication is silently a change of epoch count.** A document that appears 1,000 times in your corpus is not a thousand documents. It is *one* document that the model does a thousand passes over, while every other document gets one. You did not choose that; the crawler did. Training loss will fall beautifully on that paragraph and the model will have learned to recite rather than to generalize — the classic overfitting picture from Chapter 2, applied to a sliver of the data you never inspected.

> **Analogy.** A student revises from a textbook in which one page was photocopied a thousand times and bound into the middle. They will be able to recite that page from memory and will be no better at anything else. Worse, their practice-exam score will look excellent, because the practice exam was printed on that page.

**Contamination is the practice-exam problem.** If a benchmark question sits somewhere in the training corpus, the benchmark stops measuring reasoning and starts measuring recall. This is why step 5 exists.

▸ **Why $`n=13`$ for decontamination.** A 5-word overlap is meaningless — *"on the other hand it"* appears in millions of unrelated documents. A **13-word exact match** essentially never happens by chance in natural prose, so it is strong evidence that one text was copied from the other. The choice of $`n`$ is a false-positive/false-negative dial: small $`n`$ deletes half your corpus for no reason, large $`n`$ misses paraphrased leakage. Thirteen is the community's settled compromise, not a derived constant.

**The verbatim-emission finding is a third, separate benefit.** A model trained on deduplicated data reproduces its training data word-for-word roughly ten times less often. That is simultaneously a privacy property, a legal property, and a sign that the model spent its capacity on structure instead of storage. **Three unrelated problems collapse into one fix, which is why deduplication is the highest-value step in the entire pipeline.**

**5. Decontamination.** Remove documents overlapping evaluation sets ($`n`$-gram matching, $`n=13`$ is a common choice).

**6. PII removal and safety filtering.**

**7. Mixture weights.** Domain proportions are a first-class hyperparameter. Methods: manual (LLaMA's mix), **DoReMi** (train a small proxy model with group-DRO to *learn* the weights), or online adjustment. Multi-epoching: up to ~4 epochs on high-quality data is roughly as good as fresh data; beyond that returns collapse (Muennighoff et al.).

**8. Curriculum.** Increasingly standard: general web early, then upweight code/math/high-quality/long-context data in a final "mid-training" or annealing phase. The WSD schedule (Ch. 5) is designed around this — the sharp decay phase is where the highest-quality data goes.

#### The pipeline, decoded as a funnel

Those eight steps look like a checklist. They are better understood as a **funnel**, and the numbers only make sense once you see how much is thrown away.

> **Analogy — a water treatment plant.** Common Crawl is the reservoir intake. Nobody drinks reservoir water. Steps 2–6 are screening (remove the branches and the shopping trolleys), settling (let the sludge sink), filtration (remove what you cannot see), and chlorination (remove what is actively harmful). Step 7 is deciding the blend — how much river, how much aquifer. Step 8 is what you put in the pipe last, closest to the tap.

**What each step is actually doing:**

| Step | The question it answers | What it removes |
|---|---|---|
| Extraction | "Where is the prose?" | Navigation menus, cookie banners, footers, script tags — often most of the bytes |
| Heuristic filtering | "Does this look like written language at all?" | Link farms, keyword-stuffed spam, tables of numbers, truncated pages |
| Model-based filtering | "Would an educated person find this worth reading?" | Low-value text that passes every heuristic |
| Deduplication | "Have I already seen this?" | Mirrors, syndicated news, boilerplate licence text |
| Decontamination | "Is this secretly my exam paper?" | Benchmark items and their near-copies |
| PII and safety | "Should this exist in a model at all?" | Personally identifiable information, targeted harm |

▸ **The headline number: published web pipelines keep on the order of a tenth of the documents they start with**, once filtering and deduplication are both applied. The exact retention depends entirely on how aggressive the thresholds are, and that aggressiveness is itself one of the most consequential decisions in the whole project. **You are not collecting data; you are deciding what to destroy.**

**Decoding "a classifier trained to distinguish a high-quality reference corpus from raw web."** Take a set of documents you trust (textbook prose, well-edited reference material), label them $`1`$; take random crawl pages, label them $`0`$; train a cheap classifier. Its output on a new page is a **quality score**, and you either threshold it (keep the top $`x\%`$) or use it as a sampling weight (keep everything, but show the good pages more often). This is now worth more than every hand-written heuristic combined, because a classifier can learn "reads like it was written by someone who knew the subject" — a property nobody can express as a rule about punctuation counts.

**Decoding "group-DRO" in DoReMi.** DRO is **distributionally robust optimization**. Ordinary training minimizes the *average* loss across all your domains, which lets a huge domain drown out a small one. Group-DRO instead minimizes the loss of the **worst-performing group**, which automatically pushes weight toward the domains the model is currently bad at. DoReMi runs that on a small proxy model to *learn* the mixture proportions, then reuses those proportions to train the real one. **The mixture stops being a guess and becomes a fitted quantity.**

**Why the end of training gets the good data.** In the WSD schedule the learning rate is high and flat for most of the run, then decays sharply at the end. During the flat phase the parameters are still moving a long way each step, so any individual document's influence gets overwritten. During the decay phase the steps are tiny and the model is settling into a basin — whatever it sees there sticks.

▸ **The final coat of paint determines the finish.** That single sentence is why "mid-training," "annealing," and "high-quality decay phase" all describe the same practice: put your most trustworthy tokens where the learning rate is smallest, because that is where they leave a permanent mark.

**Decoding the multi-epoch result.** "Up to ~4 epochs is roughly as good as fresh data" means: if you are out of unique text, showing the model the same corpus a second, third, and fourth time buys you nearly what new text would have bought. Past that the curve bends over hard, and repeated data starts to behave like no data at all. This is a **budget constraint on the whole field**, not a tuning detail, and Chapter 15 §15.2 returns to it.

> **Where this came from.** Common Crawl is a nonprofit, founded by Gil Elbaz in 2007, that crawls the web and publishes the result for anyone to download. Its founding purpose was explicitly egalitarian: search-engine-scale data should not be the private property of the handful of companies that could afford to crawl. It succeeded in a way nobody predicted — that free archive became the raw substrate of essentially every large language model. **C4**, one of its most-used refinements, stands for *Colossal Clean Crawled Corpus* and was built for Google's T5 work in 2019; the canonical heuristic filtering rules come from DeepMind's Gopher report in 2021, which is why practitioners still say "the Gopher rules."

---

## 14.2 Numerical precision

### The formats

| Format | Bits (S/E/M) | Max | Min normal | Relative precision |
|---|---|---|---|---|
| fp32 | 1/8/23 | $`3.4\times10^{38}`$ | $`1.2\times10^{-38}`$ | $`\sim10^{-7}`$ |
| **fp16** | 1/5/10 | $`65{,}504`$ | $`6.1\times10^{-5}`$ | $`\sim10^{-3}`$ |
| **bf16** | 1/8/7 | $`3.4\times10^{38}`$ | $`1.2\times10^{-38}`$ | $`\sim10^{-2}`$ |
| fp8 (E4M3) | 1/4/3 | 448 | | $`\sim10^{-1}`$ |
| fp8 (E5M2) | 1/5/2 | 57344 | | |

▸ **bf16 has fp32's exponent range with fewer mantissa bits.** That trade is exactly right for deep learning: gradients span many orders of magnitude (range matters) but need few significant digits (precision doesn't). **bf16 needs no loss scaling; fp16 does.**

#### Reading the floating-point table

That table has five columns and none of them are labelled in English. Here is what each one is.

**A floating-point number is scientific notation, in binary.** You already know the format: $`6.022\times10^{23}`$. There are three pieces — a sign, the digits $`6.022`$, and the exponent $`23`$. A computer stores exactly those three pieces:

$$\text{value} = (-1)^{\text{sign}} \times 1.\underbrace{\text{mantissa}}_{\text{the digits}} \times 2^{\,\text{exponent}}$$

"S/E/M" in the table's header is **sign bits / exponent bits / mantissa bits**, and it always adds up to the format's total width: $`1+8+23 = 32`$ for fp32, $`1+5+10=16`$ for fp16, $`1+8+7=16`$ for bf16.

▸ **The two groups of bits do two completely different jobs, and this is the whole chapter's point in one line: exponent bits buy you *range*, mantissa bits buy you *precision*.**

> **Analogy — a tape measure.** The exponent bits are **how long the tape is**; the mantissa bits are **how finely it is graduated**. fp32 gives you a tape that reaches from the diameter of a proton to the width of the galaxy, marked in seven significant figures. fp16 keeps decent markings but cuts the tape down to a few hundred metres. bf16 keeps the galaxy-length tape and marks it only every few percent. **Deep learning needs a very long tape with coarse markings**, which is the one combination that classical numerical computing never wanted and therefore nobody had built.

**Now the columns, one at a time:**

| Column | What it is | How to read it |
|---|---|---|
| Bits (S/E/M) | The layout | How the 16 or 32 bits are divided up |
| Max | The largest finite number | Anything bigger becomes $`\infty`$ — an **overflow** |
| Min normal | The smallest number stored at full precision | Below this, digits start falling off the bottom; eventually you hit zero — an **underflow** |
| Relative precision | The gap between neighbouring representable numbers, as a fraction | "How many significant digits you get" |

**Where fp16's odd-looking maximum comes from.** With 5 exponent bits the largest exponent available is $`15`$, and the mantissa can reach just under $`2`$, so the largest fp16 number is $`(2 - 2^{-10})\times 2^{15} = 65{,}504`$. That is not a round number and it is not arbitrary — and it is a number a great many engineers have had cause to memorize, because it is the exact value at which a training run starts producing `inf`.

**Where bf16's precision comes from.** Seven stored mantissa bits plus one implied leading bit gives eight significant binary digits, so neighbouring bf16 values differ by about $`2^{-8} = 0.0039`$ — roughly **0.4%**. (The table's $`\sim10^{-2}`$ is the order of magnitude; $`0.004`$ is the exact figure.) In decimal terms bf16 gives you between two and three significant digits. That sounds catastrophic and mostly isn't, for reasons §14.2's next subsection makes concrete.

▸ **The fact that makes bf16 pleasant to work with: a bf16 number is literally the top half of an fp32 number.** Same sign bit, same eight exponent bits, and the mantissa is fp32's mantissa with the bottom 16 bits deleted. Converting fp32 → bf16 is a *truncation*, and it can never overflow, because the exponent field is untouched. Converting fp32 → fp16, by contrast, has to re-encode the exponent into five bits, and anything outside $`[6\times10^{-5},\ 65504]`$ is lost. **That single structural difference is the reason one format needs loss scaling and the other does not.**

**A worked example — one gradient, three formats.** Take a gradient component of $`10^{-8}`$, an entirely ordinary value deep in a large network:

| Format | What happens | Why |
|---|---|---|
| fp32 | Stored fine, $`\sim7`$ digits | Min normal is $`1.2\times10^{-38}`$ |
| fp16 | **Becomes exactly zero** | Min normal is $`6.1\times10^{-5}`$; even fp16's subnormal range bottoms out near $`6\times10^{-8}`$ |
| bf16 | Stored fine, $`\sim2`$ digits | Min normal is $`1.2\times10^{-38}`$ — same as fp32 |

The fp16 row is the entire problem. The gradient was not *inaccurate*; it was **annihilated**. No amount of extra mantissa would have helped, and no amount of careful algorithm design detects it, because zero is a perfectly plausible-looking gradient.

**Decoding the fp8 names.** "E4M3" means 4 **e**xponent bits and 3 **m**antissa bits; "E5M2" means 5 and 2. Together with the sign bit that is 8 bits either way — the two formats spend the same budget differently. Notice which is used where: **E4M3 (more precision, less range) for the forward pass, E5M2 (more range, less precision) for gradients.** That is exactly the bf16-versus-fp16 argument again, one level further down: activations are bounded and need digits; gradients span orders of magnitude and need room.

> **Where this came from.** bf16 was designed at Google for its Tensor Processing Units — the "b" stands for *brain*, after Google Brain. It is one of the very few numeric formats in computing history invented specifically for machine learning rather than inherited from scientific computing or graphics. fp16, by contrast, arrived from the opposite direction: half precision was standardized in IEEE 754-2008 but had already been in use in computer graphics, where colours and coordinates live in a narrow, well-behaved range and 65,504 is an enormous number. Deep learning then adopted the graphics format, discovered that gradients are not colours, and had to invent loss scaling to cope. **bf16 is what you design when you know in advance that the quantity you are storing spans forty orders of magnitude and you only need two digits of it.**

### Loss scaling (fp16 only)

fp16's minimum normal is $`6\times10^{-5}`$; gradients routinely fall below it and flush to zero. Fix:
1. Multiply the loss by $`S`$ (e.g. $`2^{16}`$) before backward.
2. Gradients are scaled by $`S`$, landing in representable range.
3. Unscale by $`1/S`$ before the optimizer step.
4. **Dynamic scaling:** if any gradient is inf/NaN, skip the step and halve $`S`$; after $`N`$ successful steps, double $`S`$.

#### Loss scaling, decoded

*(Reading note: the $`N`$ in step 4 is a count of successful steps — a few thousand, typically. It is not the parameter count $`N`$ used everywhere else in this chapter. The letter is doing double duty.)*

**Why this is allowed at all.** Gradients are *linear* in the loss:

$$\frac{\partial (S\cdot\mathcal{L})}{\partial\theta} = S\cdot\frac{\partial\mathcal{L}}{\partial\theta}$$

Multiplying the loss by a constant multiplies **every** gradient by that same constant. Not approximately — exactly, and identically for every parameter. So the *direction* of the gradient is untouched and only its exponent moves. Divide by $`S`$ at the end and you are back where you started, having spent the intervening arithmetic in a range fp16 can actually represent.

▸ **Put numbers on it.** Take $`S = 2^{16} = 65{,}536`$ and the doomed gradient from the previous subsection, $`10^{-8}`$. Scaled: $`10^{-8}\times65{,}536 = 6.55\times10^{-4}`$ — comfortably above fp16's minimum normal of $`6.1\times10^{-5}`$. It survives the backward pass, gets divided by $`65{,}536`$ before the optimizer sees it, and arrives at the update step as $`10^{-8}`$ again. **Nothing was approximated. The number simply took a detour through a neighbourhood where fp16 can count.**

> **Analogy.** You want the mass of one feather, and your kitchen scale reads to the nearest gram. One feather reads $`0`$ g — not "approximately zero," *exactly* zero, and no amount of squinting recovers it. So you put 65,536 feathers on the scale, read 20 g, and divide. You did not make the scale more precise. You moved the quantity into the range the scale can see, and then undid the move.

**Where the unscaling must happen — this is the part people get wrong.** Unscale *before* gradient clipping and *before* Adam updates its moments. If you unscale after clipping, you clip at the wrong threshold; if you let Adam see scaled gradients, $`\hat m/\sqrt{\hat v}`$ mostly cancels the scale but the $`\epsilon`$ in the denominator does not, and your effective learning rate quietly changes. **Step 3 says "before the optimizer step" and it means it literally.**

**Why dynamic scaling works like a thermostat.** The algorithm has no way to know the right $`S`$ in advance — it depends on the model, the data, and the current point in training, and it drifts as training proceeds. So it hunts: creep upward until something overflows, back off by half, creep upward again. The steady state is a value of $`S`$ that sits just below the overflow cliff, and the cost is an occasional discarded step.

▸ **The design principle behind that asymmetry is worth extracting.** Overflow is **loud** — it produces `inf` or `NaN`, which you can detect in one comparison. Underflow is **silent** — it produces zero, which looks exactly like a legitimately tiny gradient. Given a choice between a failure mode you can detect and one you cannot, you deliberately bias toward the detectable one. That is why the schedule pushes $`S`$ up until it breaks rather than picking something safely small.

> **Where this came from.** Loss scaling was introduced in NVIDIA's mixed-precision training work in 2017, alongside the fp32 master-weight recipe below. Both were engineering responses to a hardware fact rather than a mathematical discovery: the new tensor cores were far faster in fp16, and fp16 turned out to be the wrong shape for gradients. The whole of §14.2 is, in a sense, the field negotiating with a numeric format it did not choose.

### Mixed precision, properly

▸ Keep **fp32 master weights**. Compute forward/backward in bf16. Update the fp32 master copy, then cast down.

**Why:** the update $`\eta\cdot\hat m/\sqrt{\hat v}`$ can be $`10^{-7}`$ relative to a weight of $`10^{-2}`$ — a ratio of $`10^{-5}`$, well below bf16's $`10^{-2}`$ resolution. **Adding it to a bf16 weight is a no-op; the update is silently discarded.** This is a real and subtle failure mode, and "why do we keep fp32 master weights" is a good interview question.

**Keep in fp32 regardless:** loss computation and softmax, normalization statistics, and optimizer moments (or use stochastic rounding for bf16 moments).

#### Why fp32 master weights are not paranoia

The paragraph above contains the most important number in §14.2, so let us make it fully concrete.

**Set up the arithmetic.** Take a weight $`w = 0.01`$ and an Adam update of $`10^{-7}`$ — both entirely typical values partway through a run. Now ask: **what is the gap between $`0.01`$ and the next number bf16 can represent?**

A bf16 value near $`0.01`$ sits between $`2^{-7} = 0.0078`$ and $`2^{-6}=0.0156`$. Within that band, bf16's seven stored mantissa bits space the representable values $`2^{-7}\times 2^{-7} = 2^{-14} \approx 6.1\times10^{-5}`$ apart.

$$\frac{\text{update}}{\text{gap to the next representable number}} = \frac{10^{-7}}{6.1\times10^{-5}} \approx \frac{1}{600}$$

▸ **The update is six hundred times smaller than the smallest change bf16 can express.** So `w + update` rounds straight back to `w`. The addition executes, reports success, and changes nothing. **Do that ten million times and the weight has still never moved.**

> **Analogy.** You are adding paperclips to a lorry parked on a weighbridge that reads to the nearest 100 kg. Every individual paperclip is free — the reading never budges, and that seems fine, because a paperclip really does weigh almost nothing. But you add a million of them, which is half a tonne, and the reading has still never budged. **The bug is not that any one step is wrong. It is that every step is discarded, permanently, and the only symptom is a loss curve that flattens for no visible reason.**

**Why fp32 fixes it.** fp32 has 23 stored mantissa bits, so near $`0.01`$ its spacing is $`2^{-7}\times2^{-23} = 2^{-30}\approx 9\times10^{-10}`$ — about a hundred times *finer* than the $`10^{-7}`$ update rather than six hundred times coarser. The update lands. So you keep the authoritative copy of every weight in fp32, apply the update there, and cast a fresh bf16 copy for the matrix multiplications, which  do not care about the missing digits.

**That fp32 copy is where 4 of the 16 bytes in §14.3's $`16N`$ come from.** You are paying an extra 4 bytes per parameter — 28 GB on a 7B model — to prevent an error that produces no error message.

**Decoding "stochastic rounding."** Ordinary rounding is biased when the thing you are adding is always smaller than half a step: $`0.4`$ rounds to $`0`$ every single time, so a million additions of $`0.4`$ give $`0`$ instead of $`400{,}000`$. **Stochastic rounding** instead rounds $`0.4`$ up to $`1`$ with probability $`0.4`$ and down to $`0`$ with probability $`0.6`$. Any individual answer is wrong, but the *expected* answer is exactly right, and a million additions land near $`400{,}000`$ with a small random error rather than a total loss.

▸ **Stochastic rounding converts a systematic bias into zero-mean noise** — and neural network training, which is stochastic anyway, tolerates noise far better than it tolerates bias. That is why it is the one technique that lets you store Adam's moments in 16 bits without the silent-discard failure above.

**Why softmax, the loss, and normalization statistics stay in fp32.** All three involve **accumulating many terms**, and accumulation is where low precision actually bites. A softmax over a 128,000-token vocabulary sums 128,000 exponentials; a LayerNorm mean sums thousands of activations. Each individual addition loses a fraction of a percent in bf16, and those losses compound in one direction.

▸ **The general rule, which transfers everywhere: the multiplication is not the problem, the addition of many terms is.** This is precisely why GPU tensor cores multiply in low precision but *accumulate* in fp32 internally — the hardware makes the same distinction the software does.

**FP8** (Hopper/Blackwell): per-tensor or per-block scaling factors, E4M3 for forward, E5M2 for gradients. ~2× throughput over bf16; requires careful scaling management.

---

## 14.3 The memory budget

For $`N`$ parameters with AdamW in mixed precision:

| Item | Bytes |
|---|---|
| bf16 weights | $`2N`$ |
| bf16 gradients | $`2N`$ |
| fp32 master weights | $`4N`$ |
| fp32 Adam $`m`$ | $`4N`$ |
| fp32 Adam $`v`$ | $`4N`$ |
| **Total (states)** | $`\mathbf{16N}`$ |

▸ **A 7B model needs 112 GB of optimizer/weight state alone** — more than a single 80 GB GPU, before a single activation. This is why distributed training is not optional above ~1.5B parameters.

**Activation memory** per layer per token $`\approx \mathcal{O}(sd)`$ with $`s\approx 10`$–20 depending on what is stored; times $`B\times T\times L`$.

#### Unpacking the $`16N`$ rule

**Where each row comes from.** $`N`$ is the number of parameters. A bf16 number occupies 2 bytes, an fp32 number 4. So each row of that table is just "how many copies of every parameter am I keeping, and how wide is each copy?"

| Item | Copies × width | For $`N = 7\times10^9`$ |
|---|---|---|
| bf16 weights | $`1\times2`$ bytes | 14 GB |
| bf16 gradients | $`1\times2`$ bytes | 14 GB |
| fp32 master weights | $`1\times4`$ bytes | 28 GB |
| fp32 Adam $`m`$ | $`1\times4`$ bytes | 28 GB |
| fp32 Adam $`v`$ | $`1\times4`$ bytes | 28 GB |
| **Total** | $`\mathbf{16}`$ **bytes per parameter** | **112 GB** |

▸ **The model itself is 14 GB. The bookkeeping *about* the model is 98 GB.** You are carrying seven times more paperwork than product, and every byte of it must sit in GPU memory simultaneously.

**Why Adam needs two extra full-size copies.** Adam keeps, for each parameter separately, a running average of the gradient ($`m`$, the momentum) and a running average of the *squared* gradient ($`v`$, used to give each parameter its own step size). "For each parameter separately" is the operative phrase: both are exactly $`N`$ numbers, the same size as the model. **This is the price of adaptivity** — plain SGD with momentum would need only one such buffer, and plain SGD none at all, which is exactly why you occasionally see very large models trained with memory-frugal optimizers.

**Why $`m`$ and $`v`$ are the fp32 rows.** Look at Adam's second-moment update with the usual $`\beta_2 = 0.999`$:

$$v \leftarrow 0.999\,v + 0.001\,g^2$$

Each step changes $`v`$ by about **0.1%** of its current value. bf16's resolution near any number is about **0.4%**. So in bf16 that update is smaller than the gap to the nearest representable value, and $`v`$ can freeze in place entirely — the same silent-discard failure as the master weights, one level down. Either keep the moments in fp32, or use stochastic rounding so the discarded fraction becomes zero-mean noise instead of a permanent loss.

**Now the activation term, with real numbers.** Take a 7B-class model: hidden width $`d = 4096`$, $`L=32`$ layers, sequence length $`T = 8192`$, one sequence at a time ($`B=1`$), and $`s = 15`$ stored tensors per layer per token, in bf16:

$$B\times T\times L\times d\times s\times 2\ \text{bytes} = 1\times 8192\times 32\times 4096\times 15\times 2 \approx 3.2\times10^{10} = \mathbf{32\ GB}$$

▸ **One sequence of 8,192 tokens costs 32 GB of activations, on top of the 112 GB of state — and that is before you have processed a *second* sequence.** Activation memory grows linearly in both batch size and sequence length, which means the obvious lever (bigger batches for better hardware efficiency) runs directly into the wall. Everything in §14.3 and §14.4 is a response to this collision.

> **Analogy.** Training is a workshop with a very small floor. The model is the machine you are building (14 GB). The optimizer state is the jigs, templates, and measurement records you cannot throw away (98 GB). The activations are the offcuts and part-finished pieces from the job currently on the bench (32 GB and rising with every extra job you take on at once). **An 80 GB GPU is a workshop where the jigs alone do not fit through the door.**

### Gradient checkpointing (activation recomputation)

Store only a subset of activations; recompute the rest during backward.

▸ Checkpointing every $`\sqrt L`$ layers gives memory $`O(\sqrt L)`$ instead of $`O(L)`$, at the cost of one extra forward pass ($`\approx+33\%`$ compute, since forward:backward is 1:2).

**Selective checkpointing** is better: recompute only the cheap-but-memory-hungry ops (norms, activations, dropout) and keep the expensive matmul outputs. Gets most of the memory saving for ~5% compute.

#### Why $`\sqrt{L}`$ is the sweet spot, derived

The $`\sqrt{L}`$ in that bullet is not a heuristic. It falls out of one line of calculus, and deriving it takes thirty seconds.

**Set up the trade.** Suppose you keep a saved activation ("a checkpoint") every $`c`$ layers out of $`L`$. Two things then sit in memory at once:

- The checkpoints themselves: there are $`L/c`$ of them, kept for the whole backward pass.
- One segment being replayed: while backpropagating through a segment you must re-run its forward pass and hold its $`c`$ layers of activations.

$$\text{peak memory} \ \propto\ \underbrace{\frac{L}{c}}_{\text{checkpoints}} + \underbrace{c}_{\text{the segment in flight}}$$

**Minimize it.** Differentiate with respect to $`c`$ and set to zero:

$$\frac{d}{dc}\left(\frac{L}{c} + c\right) = -\frac{L}{c^2} + 1 = 0 \quad\Longrightarrow\quad c = \sqrt{L}$$

Substituting back gives peak memory $`\propto 2\sqrt{L}`$ instead of $`L`$. **That is the whole result.**

▸ **Put numbers on it.** With $`L = 64`$ layers: storing everything costs 64 units, checkpointing every $`\sqrt{64}=8`$ layers costs $`2\times 8 = 16`$ units — a **4× reduction**. With $`L=256`$: 256 units becomes $`2\times16 = 32`$, a **8× reduction**. **The deeper the network, the better the deal**, which is a rare and welcome direction for a scaling argument to point.

Applied to the 32 GB of activations computed above ($`L=32`$): $`2\sqrt{32}\approx 11.3`$ units out of 32, so roughly **11 GB instead of 32 GB.**

**Where the $`+33\%`$ comes from.** A training step is one forward pass plus one backward pass, and backward costs about twice forward (it computes gradients with respect to both the inputs and the weights). So a normal step costs $`1 + 2 = 3`$ units of compute. Checkpointing adds one extra forward pass: $`1 + 1 + 2 = 4`$ units. $`4/3 = 1.33`$. **You are buying a 4× memory reduction for a 33% compute surcharge**, which is why essentially every large training run uses it.

> **Analogy.** You are reading a 600-page novel and someone will ask you to recall, in reverse order, the context of every plot point. Option one: keep all 600 pages spread out on an enormous desk. Option two: keep a bookmark every 25 pages, and when asked about page 340, flip to the bookmark at 325 and re-read fifteen pages. The second option needs a desk a fraction of the size, at the cost of re-reading each chapter once. **Memory and computation are exchangeable, and checkpointing sets the exchange rate.**

**Why *selective* checkpointing beats it.** The derivation above quietly assumes every layer costs the same to recompute. It does not. A GELU, a LayerNorm, and a dropout mask each occupy as much memory as a matrix multiply's output, but cost perhaps a thousandth as much to recompute — they are elementwise operations, limited by memory bandwidth rather than arithmetic.

▸ **The insight is that an operation's memory footprint and its compute cost are almost uncorrelated.** So you recompute the cheap-but-bulky operations and *keep* the expensive ones. You capture most of the memory saving for a few percent of compute instead of 33%. This is a general engineering pattern worth carrying beyond this chapter: whenever you are trading two resources, check whether the exchange rate is the same for every item, because it usually is not.

> **Where this came from.** The memory-versus-recomputation trade in reverse-mode differentiation was worked out in the *automatic differentiation* literature long before deep learning existed. Andreas Griewank analysed checkpointing schemes for reverse-mode AD in the early 1990s, proving logarithmic and square-root trade-offs — motivated by problems in scientific computing and optimal control, where the "network" is a time-stepping simulation. Chen, Xu, Zhang and Guestrin reintroduced it to neural networks in 2016 under the name *sublinear memory cost*. **This is the same story as backpropagation itself, which Chapter 1 notes was published in 1970 in a Finnish master's thesis about rounding errors: the AD community solves a problem, and the machine learning community rediscovers it a couple of decades later.**

---

## 14.4 Parallelism

#### The five parallelisms, on one page

"Parallelism" here covers several  different ideas that happen to share a word. Read this map before the details; the rest of §14.4 is each row expanded.

| Name | What gets **split** | What gets **copied** | The communication it forces |
|---|---|---|---|
| **Data (DP)** | The batch | The whole model, on every device | All-reduce the gradients, once per step |
| **Tensor (TP)** | Individual weight matrices | The batch | An all-reduce *inside* every layer — very frequent |
| **Pipeline (PP)** | The layers | Nothing | Hand activations to the next stage — small messages, but creates idle time |
| **Sequence / context** | The tokens of a single example | The whole model | Pass key/value blocks around a ring |
| **Expert (MoE)** | The experts | Everything else | All-to-all, to send each token to its chosen expert |

▸ **These are orthogonal cuts through the same computation, which is why a real run uses several at once.** Splitting the batch, splitting the matrices, splitting the layers, and splitting the sequence are four independent decisions. The engineering question is never "which one" but **"how many devices along each axis, and which axis gets the fastest wires."**

> **Analogy — a restaurant kitchen with too many orders.** One cook cannot keep up, and there are four unrelated fixes. Hire more cooks and give each a share of the tickets (**data parallelism**). Put several cooks on the *same* dish, one chopping while another sears (**tensor parallelism**). Build stations so each dish moves prep → grill → plating (**pipeline parallelism**). Or split one enormous banquet order by course (**sequence parallelism**). Real kitchens do all four simultaneously, and the binding constraint is always the same: **how much shouting across the room each arrangement requires.** Tensor parallelism is two cooks sharing one chopping board — it only works if they are standing next to each other.

### Data parallelism (DP)

Replicate the model, split the batch, **all-reduce** the gradients.
- Communication: $`2N`$ bytes per step (ring all-reduce is bandwidth-optimal).
- Scales until the batch per device is too small, or the global batch exceeds the critical batch size (Ch. 15).

#### All-reduce, decoded

**What the word means.** A **reduce** combines one value from every participant into a single result (sum, mean, max). An **all-**reduce does that *and hands the result back to everyone*. So after an all-reduce of gradients, all $`P`$ devices hold the identical summed gradient.

**Why it is mandatory rather than merely nice.** Each device computed a gradient from *its own slice of the batch*. Those gradients differ. If each device then applied its own gradient, the $`P`$ copies of the model would drift apart, and after a few thousand steps you would own $`P`$ mediocre models instead of one good one. **The all-reduce is what keeps the replicas bit-identical**, and it is the only synchronization point in the whole step.

**How the ring version works, in two halves.** Arrange the $`P`$ devices in a circle and cut the gradient buffer into $`P`$ chunks.

1. **Reduce-scatter** ($`P-1`$ passes): each device sends one chunk to its right-hand neighbour and adds the chunk arriving from its left. After $`P-1`$ passes, each device holds *one* chunk that is the complete sum over all devices — but a different chunk each.
2. **All-gather** ($`P-1`$ passes): pass those finished chunks around the ring until everyone has all of them.

▸ **The property that matters: each device sends about $`2\times\frac{P-1}{P}`$ buffer-fulls of data, which is at most twice the buffer no matter how many devices you add.** Communication per device does not grow with $`P`$. That is precisely what "bandwidth-optimal" means, and it is why data parallelism scales to thousands of GPUs when a naive "everyone sends everything to device 0" design would collapse at eight.

*(The $`2N`$ bytes in the bullet above is the size of the object being reduced — $`N`$ parameters at 2 bytes each in bf16. A ring implementation moves roughly twice that per device; the load-bearing fact is that neither figure grows with $`P`$.)*

**Put numbers on it.** A 7B model has a 14 GB bf16 gradient buffer, so each device pushes roughly 28 GB per step. On a 400 Gb/s interconnect — 50 GB/s — that is about **0.56 seconds of communication per step**. If the step's compute takes 2 seconds, you have just added 28% to your step time and thrown away a quarter of your cluster.

▸ **Which is why the real implementation never waits.** Gradients become available *layer by layer* during the backward pass, from the last layer to the first. So you launch the all-reduce for layer $`L`$'s gradients the instant they exist, while layers $`L-1, L-2,\dots`$ are still computing theirs. Done well, essentially all the communication hides underneath compute that was going to happen anyway. **"Overlap communication with computation" is the single most repeated sentence in distributed training, and this is what it means.**

> **Analogy.** Ten accountants each audit a tenth of a company's transactions and must all end up with the same final ledger. The bad design is for all ten to post their findings to one person, who combines them and posts ten copies back — that person is a bottleneck who does ten times the work of anyone else. The ring design is for them to sit in a circle, each take responsibility for one section of the ledger, and pass their sheets around twice. **Everybody works, nobody is a bottleneck, and adding an eleventh accountant does not slow the others down.**

> **Where this came from.** Ring all-reduce is not a deep learning invention — it is a standard collective-communication algorithm from high-performance computing, where the same "sum a vector across every node" problem shows up in every distributed physics simulation. It reached deep learning around 2017, when researchers at Baidu published a ring all-reduce implementation for neural network training, and Uber's *Horovod* library packaged the idea for TensorFlow shortly after. **The pattern recurs throughout this chapter: the numerical problems of large-scale training had mostly been solved by other communities, and the field's contribution was recognizing which shelf to take them off.**

### ZeRO / FSDP — shard the states

Standard DP replicates all $`16N`$ bytes on every device. ZeRO shards them:

| Stage | Shards | Memory per device |
|---|---|---|
| ZeRO-1 | optimizer states | $`4N + \frac{12N}{P}`$ |
| ZeRO-2 | + gradients | $`2N+\frac{14N}{P}`$ |
| **ZeRO-3 / FSDP** | + parameters | $`\frac{16N}{P}`$ |

▸ ZeRO-3 gives **linear memory scaling in the number of devices** at the cost of an all-gather of each layer's parameters just before it is used (and again in backward). With overlapping of communication and compute, the throughput cost is typically 10–20%.

#### Reading the ZeRO table

**Each formula is "what is still replicated, plus what is now shared."** The $`16N`$ splits into $`2N`$ weights, $`2N`$ gradients, and $`12N`$ of optimizer state (the $`4N`$ fp32 master copy plus $`4N`$ for $`m`$ plus $`4N`$ for $`v`$). Dividing a term by $`P`$ means "that item is cut into $`P`$ pieces and each device keeps one."

| Stage | Still replicated | Now sharded | Formula |
|---|---|---|---|
| ZeRO-1 | weights + gradients ($`4N`$) | optimizer state ($`12N`$) | $`4N + \frac{12N}{P}`$ |
| ZeRO-2 | weights ($`2N`$) | gradients + optimizer ($`14N`$) | $`2N + \frac{14N}{P}`$ |
| ZeRO-3 | nothing | everything ($`16N`$) | $`\frac{16N}{P}`$ |

▸ **Put numbers on it.** A 7B model on $`P=64`$ GPUs:

| Setup | Memory per device |
|---|---|
| Plain data parallelism | 112 GB — does not fit on an 80 GB card |
| ZeRO-1 | $`28 + 1.3 = 29.3`$ GB |
| ZeRO-2 | $`14 + 1.5 = 15.5`$ GB |
| ZeRO-3 / FSDP | $`112/64 = \mathbf{1.75}`$ **GB** |

**112 GB down to 1.75 GB, with no change to the mathematics whatsoever.** Every device computes exactly the same update it would have computed before. The only thing that changed is who is holding which bytes.

**What you give up.** Under ZeRO-3 a device no longer owns the weights it needs. Before computing layer $`\ell`$ it must **all-gather** that layer's parameters from the other 63 devices, use them, and immediately discard them — then do it again during the backward pass. So ZeRO-3 converts *one* gradient all-reduce per step into *a stream* of weight all-gathers, two per layer.

That sounds ruinous and is not, because of the same trick as before: while layer $`\ell`$ is computing, you are already fetching layer $`\ell+1`$'s weights. Done properly the fetches hide underneath the arithmetic, leaving the 10–20% overhead the text quotes.

> **Analogy.** Sixty-four mechanics share a workshop. In plain data parallelism each keeps a complete duplicate toolset at their own bench — enormously wasteful, since at any moment each is holding one spanner. Under ZeRO-3, each keeps one sixty-fourth of the tools, and before every job shouts for what they need and hands it straight back. **It works beautifully provided somebody is already fetching the next job's tools while you are using these.** The moment the fetching falls behind the work, everybody stands idle holding nothing.

**The name says exactly what it does.** *Zero Redundancy Optimizer* — the redundancy being eliminated is the embarrassing fact that standard data parallelism stores 64 bit-identical copies of the same numbers and calls it a design. **FSDP** (Fully Sharded Data Parallel) is PyTorch's implementation of the same idea, which is why the two names appear together.

> **Where this came from.** ZeRO was published in 2020 by the DeepSpeed team at Microsoft Research. The observation behind it is almost embarrassingly simple in retrospect — nobody needs 64 copies of the same optimizer state — but it changed what was trainable on a given cluster more than any architectural change of the same period. **The general lesson is that at scale, the binding constraint is usually not "what can we compute" but "what can we fit," and the biggest wins come from noticing something is stored more times than it is used.**

### Tensor parallelism (TP)

Split individual matrices across devices *within* a layer.
- **Column-parallel** then **row-parallel** for the FFN: $`Y = \mathrm{GeLU}(XA)`$, split $`A`$ by columns; $`Z=YB`$, split $`B`$ by rows. One all-reduce per block instead of two.
- Attention: split by heads — each device owns a subset of heads.

▸ **Requires very high bandwidth (NVLink).** Keep TP **within a node** (typically TP ≤ 8); across nodes it is disastrous.

#### Column-parallel then row-parallel, decoded

That two-line recipe hides the single prettiest trick in distributed training, and it is entirely about **where the nonlinearity sits**.

**The shapes.** A transformer feed-forward block is two matrices back to back: $`A`$ of shape $`4096\times16384`$ (widen), then $`B`$ of shape $`16384\times4096`$ (narrow again), with a GeLU in between. Split across 8 devices.

**Half one — split $`A`$ by columns.** Device $`i`$ gets $`A_i`$, a $`4096\times2048`$ slice. It computes $`XA_i`$ and obtains a $`2048`$-wide slice of $`Y`$'s **columns**. Now apply the GeLU. Because GeLU is **elementwise**, and each device owns whole columns of complete numbers, **each device can apply it alone. No communication.**

**Half two — split $`B`$ by rows.** Device $`i`$ gets $`B_i`$, a $`2048\times4096`$ slice, matching the columns it already holds. It computes $`Y_iB_i`$, which is a **full-size** $`4096`$-wide output — but only a *partial sum*. Add the eight partial sums together and you have $`Z`$. **That addition is one all-reduce, and it is the only communication in the whole block.**

▸ **Now the point: run it the other way and it breaks.** If you had split $`A`$ by rows, each device would hold a *partial sum* of $`Y`$, and $`\mathrm{GeLU}(a + b) \ne \mathrm{GeLU}(a) + \mathrm{GeLU}(b)`$ — nonlinear functions do not distribute over sums. You would have to all-reduce *before* the activation and again after: two communications per block instead of one. **The split is arranged so that the nonlinearity only ever sees complete numbers.** Everything in the column-then-row ordering follows from that one requirement.

**Attention splits by heads for the same reason.** Each attention head does its own projections, its own softmax, and its own weighted sum — heads never interact until the output projection concatenates them. So heads are a natural seam with no nonlinearity crossing it. With 32 heads on 8 devices, each device owns 4 heads end-to-end, and one all-reduce after the output projection stitches the result together.

**Why the bandwidth requirement is so severe — count the messages.** Data parallelism all-reduces **once per step**. Tensor parallelism all-reduces **twice per layer** — once in attention, once in the FFN. With 32 layers that is 64 all-reduces per forward pass and more again in backward, versus one for the whole step.

▸ **Roughly two orders of magnitude more communication events, each of them sitting on the critical path with nothing to hide behind.** Inside a node, NVLink gives hundreds of gigabytes per second directly between GPUs. Between nodes you are on a network an order of magnitude slower. **That gap is the entire reason for the rule "TP ≤ 8": eight is how many GPUs are wired together inside one chassis.** Cross a node boundary with tensor parallelism and your all-reduces stop hiding and start dominating.

> **Analogy.** Tensor parallelism is two cooks sharing one chopping board — extraordinarily efficient if they are elbow to elbow, and unworkable if one of them is in a different building. Data parallelism is two cooks in different buildings each cooking a whole meal and comparing notes once at the end; the distance barely matters. **Give the shortest wires to the technique that talks the most.** That is the whole content of "order the dimensions so the highest-communication one uses the fastest links."

> **Where this came from.** This particular column-then-row arrangement was introduced in NVIDIA's *Megatron-LM* work in 2019, which is why practitioners often say "Megatron-style tensor parallelism" rather than naming the operation. The insight it contributed was not that matrices can be split — that is obvious — but that **the split should be chosen by looking at where the nonlinearities are**, so that the number of synchronization points falls from two per block to one.

### Pipeline parallelism (PP)

Split layers across devices; micro-batch to keep everyone busy.

▸ **Bubble fraction** $`= \frac{P-1}{m + P - 1}`$ for $`P`$ stages and $`m`$ micro-batches. With $`P=8`$, $`m=8`$: 47% idle. With $`m=64`$: 10%. **Use many micro-batches.** Interleaved (virtual-stage) schedules and zero-bubble variants reduce it further.

#### Where the bubble formula comes from

"Bubble" is the pipeline word for **an idle device**, and the formula is a counting argument you can do on your fingers.

> **Analogy — a car wash with 8 bays.** Each car passes through bay 1, then bay 2, and so on. When the first car is in bay 1, **the other seven bays have nothing to do** — they are waiting for work to reach them. That is the *fill*. At the end, when the last car reaches bay 8, bays 1 through 7 have run out of cars. That is the *drain*. In between, everybody is busy. **The fill and the drain are fixed costs; the only way to make them small is to send more cars.**

**The count.** Push $`m`$ micro-batches through $`P`$ stages. It takes $`P-1`$ time slots to fill the pipeline and the final micro-batch needs $`m`$ slots of useful work, so the run occupies $`m + P - 1`$ slots while only $`m`$ of them are fully productive. The wasted fraction is therefore

$$\frac{(m + P - 1) - m}{m+P-1} = \frac{P-1}{m+P-1}$$

**Check the book's numbers.** $`P=8`$, $`m=8`$: $`7/15 = 0.467`$ — **47% of your cluster is idle.** $`P=8`$, $`m=64`$: $`7/71 = 0.099`$ — under 10%. $`m=256`$: $`7/263 = 0.027`$.

▸ **Notice what does and does not appear in that formula.** The bubble depends only on $`P`$ and $`m`$ — not on model size, not on bandwidth, not on how fast your kernels are. **It is a pure scheduling loss, and you fix it by scheduling, not by buying faster hardware.**

**So why not set $`m`$ to a million?** Because every micro-batch that is still in flight has activations that must be kept alive until its backward pass runs. More micro-batches means more concurrent activations means more memory — the same wall as §14.3. The bubble is bought back with memory, not with compute, which is why pipeline schedules have names: **1F1B** ("one forward, one backward") interleaves the two directions so that at most $`P`$ micro-batches are ever in flight, capping the memory while keeping $`m`$ large.

> **Where this came from.** Micro-batched pipeline parallelism, and the "bubble" vocabulary, entered deep learning through Google's **GPipe** in 2018. The underlying idea is much older — it is the instruction pipeline of a CPU, where the same fill-and-drain penalty is why a branch misprediction is expensive. **A transformer's layers being fed micro-batches and a processor's stages being fed instructions are, at this level of abstraction, the same machine.**

### Sequence / context parallelism

Split the sequence dimension. **Ring Attention** passes K/V blocks around a ring so each device attends over the full sequence while holding only a slice — this is what makes million-token training feasible.

### Expert parallelism (MoE)
Distribute experts across devices; routing becomes an all-to-all.

### 3D/4D parallelism — the composition

▸ Typical frontier configuration: **TP within a node (8), PP across a few nodes, DP/FSDP across everything else, plus context parallelism for long sequences.** Order the dimensions so that the highest-communication one (TP) uses the fastest links.

### Measuring efficiency

▸ $$\mathrm{MFU} = \frac{6ND / t}{\text{peak FLOP/s}\times\text{devices}}$$

Model FLOPs Utilization. **40–55% is good** for large transformers; below 30% means something is wrong (usually communication, small batch, or bad kernel choice). Report MFU, not "GPU utilization," which only measures whether kernels are running, not whether they're doing useful work.

#### Where the 6 in $`6ND`$ comes from

This constant appears in every compute estimate in this book and the next chapter, and it is worth being able to reconstruct rather than recall.

**Count the arithmetic for one parameter, one token.**

| Pass | Cost per parameter per token | Why |
|---|---|---|
| Forward | 2 FLOPs | Each weight is used in one multiply and one add ("multiply–accumulate") |
| Backward w.r.t. the **input** | 2 FLOPs | To keep propagating the gradient to earlier layers |
| Backward w.r.t. the **weight** | 2 FLOPs | To get the update for this weight |
| **Total** | **6 FLOPs** | |

▸ **So read $`C = 6ND`$ aloud as "six floating-point operations, per parameter, per token."** Multiply by $`N`$ parameters and $`D`$ tokens and you have the whole training run. It also explains the 1:2 forward:backward ratio used in the checkpointing argument above — backward does two of the three jobs.

*(The formula counts the multiplications against the **weights** and ignores the attention score computation, which has no weights of its own. That omission is small when the sequence length is modest relative to the model width, and grows into a real correction at very long contexts — one more reason Chapter 12's long-context work matters.)*

#### Reading the MFU formula

$$\mathrm{MFU} = \frac{6ND/t}{\text{peak FLOP/s}\times\text{devices}}$$

**Numerator:** the useful work the model required, divided by how long you took — the FLOP/s you actually delivered. **Denominator:** the FLOP/s the hardware could in principle deliver. The ratio is the fraction of your cluster that did something the model needed.

> **Analogy.** A factory rated at 1,000 widgets an hour ships 450. Its efficiency is 45%. Now consider the *other* metric: "were the machines switched on?" A factory whose machines run flat out twenty-four hours a day producing scrap and then re-melting it scores **100% on machine utilization and 0% on widgets**. `nvidia-smi` reports the first number. **MFU reports the widgets.**

▸ **The subtlety that makes MFU the honest metric — and the reason it is called *Model* FLOPs Utilization.** The numerator counts only the FLOPs the *model* mathematically requires, not the FLOPs the *hardware* actually executed. Gradient checkpointing runs an extra forward pass, so it executes 33% more FLOPs than $`6ND`$ — and it therefore *lowers* your MFU even though it was the right decision. That is deliberate. A metric that rewarded executed FLOPs would rate recomputation, redundant work, and inefficient kernels as achievements. **MFU asks "how quickly did you train the model," not "how busy did you look."**

> **Where this came from.** The term *Model FLOPs Utilization* was introduced in Google's PaLM report in 2022, precisely to replace the ambiguous "hardware FLOPs utilization" figures that were being quoted at the time and were not comparable between labs. It is a small contribution and an unusually valuable one: **a great deal of engineering progress consists of agreeing on a number that cannot be gamed.**

---

## 14.5 Batch size

**Critical batch size** (McCandlish et al.): the gradient-noise scale
▸ $$B_{\text{crit}} \approx \frac{\mathrm{tr}(\Sigma)}{\|\nabla\mathcal{L}\|^2}$$

Below $`B_{\text{crit}}`$, doubling the batch roughly halves the steps needed (near-perfect scaling). Above it, you buy almost nothing.

▸ $`B_{\text{crit}}`$ **grows during training** — as the loss falls, $`\|\nabla\mathcal{L}\|`$ shrinks faster than the noise, so the ratio grows. Hence **batch-size ramping**: start small (sample-efficient), grow large (compute-efficient). Frontier runs commonly ramp from ~1M to ~60M tokens per batch.

Pair with the LR rules from Ch. 4 §4.6: linear scaling for SGD, square-root scaling for Adam is the safer default.

#### The gradient noise scale, decoded

$$B_{\text{crit}} \approx \frac{\mathrm{tr}(\Sigma)}{\|\nabla\mathcal{L}\|^2}$$

**Every symbol, out loud.** $`\Sigma`$ ("capital sigma") is the **covariance matrix of the per-example gradient** — not the summation sign, and not a standard deviation. It records how much individual training examples *disagree* about which way to step. $`\mathrm{tr}(\Sigma)`$ ("trace of Sigma") is the sum of its diagonal entries, which is the **total variance summed over every parameter**: one number for "how noisy is a single-example gradient." And $`\|\nabla\mathcal{L}\|^2`$ is the squared length of the true, full-dataset gradient: one number for "how strong is the actual signal."

▸ **So $`B_{\text{crit}}`$ is a noise-to-signal ratio, and it comes out in units of *examples*.**

**Where the formula comes from, in two lines.** Averaging $`B`$ independent examples divides the noise variance by $`B`$ — Chapter 1's $`\sigma/\sqrt{n}`$, squared. So a batch of size $`B`$ has noise $`\mathrm{tr}(\Sigma)/B`$ and signal $`\|\nabla\mathcal{L}\|^2`$. Ask when they are comparable:

$$\frac{\mathrm{tr}(\Sigma)}{B} = \|\nabla\mathcal{L}\|^2 \quad\Longrightarrow\quad B = \frac{\mathrm{tr}(\Sigma)}{\|\nabla\mathcal{L}\|^2}$$

**That is the whole derivation.** $`B_{\text{crit}}`$ is the batch size at which you have averaged away just enough noise to see the signal. Below it, extra examples are still buying you real information. Above it, you are re-measuring something you already know.

▸ **Put numbers on it.** If $`\mathrm{tr}(\Sigma) = 100`$ and $`\|\nabla\mathcal{L}\|^2 = 0.01`$, then $`B_{\text{crit}} = 10{,}000`$ examples. Use a batch of 1,000 and every step is  informative. Use a batch of 100,000 and nine tenths of your compute confirmed a direction the first ten thousand examples had already established.

> **Analogy — polling.** To call an election that one side is winning 70–30, fifty respondents will do; the signal dwarfs the noise. To call a race at 50.1–49.9, fifty thousand are not enough. **The number of people you must ask is set by how close the race is, not by how many people you can afford to phone.** Batch size is the same quantity: how many examples must agree before you trust the direction.

**Why $`B_{\text{crit}}`$ grows during training — the honest reason.** Early on, every example wants the same thing: *stop predicting nonsense*. The signal is enormous and the disagreement is small. Late in training, the easy agreement is used up; example A wants the model nudged one way and example B the other, so the **mean** gradient becomes small while the **spread** stays substantial. The numerator holds roughly steady, the denominator collapses, and the ratio climbs.

▸ **Hence batch ramping, and hence why it is not a hack.** Start with a small batch, when small batches are informative and each step is cheap; grow it as the race tightens and you need more voters per call. Frontier runs commonly ramp from around 1M to around 60M tokens per batch over the course of training. **The batch size is not a memory decision or a hardware decision — it is a statistics decision that the hardware then has to accommodate.**

> **Where this came from.** The gradient noise scale was formalized in a 2018 paper from OpenAI, by a group that needed a principled answer to a very practical question: how many machines can we usefully throw at a single training run before we are just burning electricity? The striking finding was that the answer is **predictable from a quantity you can measure during training**, and that it differs by orders of magnitude between domains — simple supervised tasks saturate at small batches, while hard reinforcement-learning problems tolerate enormous ones, because their gradients are so much noisier.

---

## 14.6 Instabilities and how to fix them

Large runs fail in characteristic ways. Knowing the taxonomy is  valuable.

**Loss spikes.** Sudden jumps, sometimes recovering, sometimes not.
- *Causes:* attention-logit growth; a bad data batch; fp16 overflow; too-high LR interacting with a sharpness increase.
- *Fixes:* **skip the batch** and continue (standard practice — PaLM restarted from a checkpoint ~100 steps back and skipped ~250 batches); lower LR; gradient clipping; the structural fixes below.

> **Where this came from — and the detail that makes it interesting.** The PaLM team reported something counterintuitive about those spikes: when they replayed *the same batches* starting from a checkpoint before the spike, **the spike did not recur.** So the offending data was not sufficient on its own to cause the failure. What caused it was the *combination* of that data with the exact parameter state at that exact moment. This is why "skip the batch" works as a remedy despite sounding like superstition — you are not removing poison from the corpus, you are stepping around a single unlucky coincidence of data and geometry. It also explains why loss spikes are so hard to reproduce in a debugging session, and why the standard operational response is to restart from a checkpoint rather than to hunt for the bad document.

▸ **Attention-logit explosion.** $`q^\top k`$ grows without bound, softmax saturates, gradients vanish, then a large step destabilizes everything. **Fix: QK-normalization** — RMSNorm on $`Q`$ and $`K`$ before the dot product. This has become standard and is one of the highest-value stability interventions known.

#### Attention-logit explosion, with numbers

**Why the dot product is so eager to grow.** An attention score is $`q^\top k = \sum_{i=1}^{d_k} q_i k_i`$ — a sum of $`d_k`$ products. Suppose $`d_k = 128`$ and the entries of $`q`$ and $`k`$ are around unit scale. Each product has variance about 1, and 128 of them add, so the sum has standard deviation $`\sqrt{128}\approx 11.3`$. Divide by $`\sqrt{d_k} = 11.3`$ (Chapter 11's scaling) and the scores have standard deviation **1**. That is the regime the architecture was designed for.

Now let training drift the entries of $`q`$ and $`k`$ up to a typical size of 3. Each product's standard deviation goes up by $`3\times3 = 9`$, so the scaled scores now have standard deviation **9** instead of 1.

**What a softmax does to logits with standard deviation 9.** The gaps between competing scores are now tens of units, and $`e^{20}`$ is roughly $`5\times10^{8}`$. The softmax output is, for practical purposes, **one-hot**: one token gets probability $`0.9999\ldots`$ and everything else gets nothing.

▸ **And a saturated softmax has no gradient.** The derivative of a softmax output involves $`p(1-p)`$, which is $`\approx 0`$ when $`p`$ is $`0`$ or $`1`$. So the attention head stops receiving gradient — **it is frozen, and it cannot learn its way back out, because escaping requires exactly the gradient that saturation has removed.** Meanwhile the rest of the network keeps moving, and eventually one step lands somewhere the frozen head cannot cope with. That is the spike.

> **Analogy — a microphone with no limiter.** As the singer gets louder the preamp saturates, and past a point the output is a flat clipped square wave. It no longer carries any information about how loud they actually were, and no amount of mixing afterwards recovers it. **QK-normalization is a limiter placed *before* the preamp**: it pins $`\|q\|`$ and $`\|k\|`$ to a fixed scale so the dot product cannot drift into clipping, whatever the weights do.

**What the fix costs.** RMSNorm on $`Q`$ and $`K`$ keeps a learned gain, so the model can still choose how sharp its attention is — but it now does so **deliberately, through one parameter, instead of accidentally through 128 of them.**

▸ **The intervention adds no capability. It removes a degree of freedom the model was using to hurt itself.** That is the general shape of almost every stability fix in this section, and it is worth recognizing: you are not making the model better, you are closing a door it kept walking through.

> **Where this came from.** Normalizing queries and keys before the dot product was proposed for transformers around 2020, but the diagnosis that made it standard came from scaling experience. Google's ViT-22B work in 2023 described the failure explicitly — attention logits growing during training until the softmax collapsed to one-hot and training destabilized — and reported QK-normalization as the fix. **The lesson generalizes: a great many "large model instabilities" turn out on inspection to be a saturating nonlinearity somewhere, silently cutting the gradient path.**

▸ **Output-logit divergence.** The final logits grow, driving overconfidence and fp16 overflow. **Fix: z-loss**, an auxiliary penalty
$$\mathcal{L}_z = \alpha\left(\log\sum_j e^{z_j}\right)^2,\qquad \alpha\approx10^{-4}$$
which pulls the log-partition function toward 0 and keeps logits bounded without constraining their differences. Used by PaLM and many since.

#### What the z-loss actually says

**Decoding the expression.** $`z_j`$ is the raw output score ("logit") for vocabulary item $`j`$. The quantity $`\log\sum_j e^{z_j}`$ has a name — the **log-partition function**, or `logsumexp` — and it is exactly the denominator of the softmax, written in logs. The z-loss squares it and adds a tiny multiple of the result to the loss.

**Now the fact that makes it necessary.** Softmax is **invariant to adding the same constant to every logit**:

$$\mathrm{softmax}(z + c) = \mathrm{softmax}(z)\quad\text{for any constant } c$$

Check it with numbers. Logits $`(2,1,0)`$ give probabilities $`(0.665,\ 0.245,\ 0.090)`$. Logits $`(1002,1001,1000)`$ give **exactly the same probabilities.** The model's predictions depend only on the *differences* between logits, never on their overall level.

▸ **Which means there is no gradient pressure of any kind against the level drifting upward.** It is a completely free direction in parameter space — and free directions in a high-dimensional optimization *drift*, because nothing is holding them still. The model is not doing anything wrong. It simply has no reason to care, and so it wanders.

**Why the wandering is dangerous.** The two logit vectors above have log-partition functions of $`2.41`$ and $`1002.41`$. The predictions are identical, but $`e^{1002}`$ overflows every floating-point format in this chapter, the intermediate exponentials become `inf`, and the loss becomes `NaN`. **A quantity the model is indifferent to has destroyed the run.**

**What the penalty does about it.** $`\mathcal{L}_z = \alpha(\log\sum_j e^{z_j})^2`$ pushes the log-partition function toward zero — that is, toward $`\sum_j e^{z_j} = 1`$ — while leaving every *difference* between logits untouched. It disciplines the free direction and only the free direction.

▸ **And the quadratic is the design, not an accident.** With $`\alpha = 10^{-4}`$ and a healthy log-partition function of $`2.4`$, the penalty is $`10^{-4}\times 5.8 = 0.0006`$ — invisible next to a cross-entropy of around 2. At a log-partition function of $`1000`$ the penalty is $`10^{-4}\times 10^6 = 100`$, which dwarfs everything else. **The term is silent when things are fine and overwhelming when they are not.** That is precisely what you want from a safety mechanism.

> **Analogy.** A ratchet strap over cargo that is not going anywhere. It costs you thirty seconds and you never think about it again — right up until the one journey where it is the only thing between the load and the road.

**Divergence after an LR warmup ends** — usually the peak LR is too high for the current sharpness (Ch. 5 §5.5).

**Silent degradation:**
- Dead/underflowing gradients in fp16.
- A tokenizer/data bug producing repeated content.
- Expert collapse in MoE (check routing entropy).
- Weight-norm blowup from missing weight decay.

**The monitoring set worth having by default:** loss, grad norm (global and per-layer), LR, weight norm per layer, update-to-weight ratio, attention-logit max, router entropy (MoE), throughput and MFU, and per-domain validation loss.

#### The monitoring set, decoded — what each number is for

That list is a cockpit, and every instrument on it exists because somebody once lost a week without it.

| Metric | What it is | What "bad" looks like |
|---|---|---|
| Loss | The headline number | A spike, or a plateau that arrives earlier than your scaling law predicted |
| Global grad norm | $`\lVert\nabla\mathcal{L}\rVert`$ over all parameters | A sudden climb — this often moves **a step or two before** the loss does, making it your earliest warning |
| Per-layer grad norm | The same, split by layer | One layer diverging while the rest are calm — this tells you *where*, not just *that* |
| Learning rate | The scheduled value, logged | Sounds trivial; a schedule bug that silently resets or misreads the step count is a  common way to lose a run |
| Weight norm per layer | $`\lVert W\rVert`$ per layer | Unbounded growth, which almost always means weight decay is not reaching that parameter group |
| Update-to-weight ratio | $`\lVert\Delta W\rVert / \lVert W\rVert`$ per layer | See below — the most diagnostic number in the list |
| Attention-logit max | The largest score entering a softmax | Creeping into the tens: the §14.6 explosion, caught early |
| Router entropy (MoE) | How spread out the expert routing is | Falling toward zero: all tokens funnelling into a few experts while you pay for the rest |
| Throughput / MFU | Tokens per second, and the ratio from §14.4 | A step down, which usually means a degraded node or a network problem, visible long before the loss notices |
| Per-domain val loss | Validation loss split by data source | Aggregate loss flat while code improves and prose quietly rots |

▸ **Why the update-to-weight ratio deserves special attention.** $`\|\Delta W\|/\|W\|`$ is **dimensionless**, so unlike a raw gradient norm it is directly comparable across layers, across model sizes, and across runs. A healthy value is on the order of $`10^{-3}`$ per step. Much smaller and that layer is effectively frozen — and note that §14.2's silent-discard failure produces exactly this signature, which is how you catch it. Much larger and the layer is being rewritten from scratch every step. **One number, no units, and it detects both of the failure modes that produce no error message.**

> **Analogy.** An altimeter tells you that you are descending. It does not tell you whether that is the pilot, the wind, or an engine. **The point of the other nine instruments is to answer "why," and none of them help if you install them after the incident** — which is the whole argument for logging all of this by default. Each costs a negligible amount of compute, and a large training run is far too expensive to repeat merely to add a logging line.

---

## 14.7 A realistic budget

Training a 7B model on 2T tokens:

$$C = 6ND = 6\times7\times10^9\times2\times10^{12} = 8.4\times10^{22}\ \text{FLOPs}$$

On 512 H100 SXM at **dense** bf16 peak ($`\approx4.95\times10^{14}`$ FLOP/s — note vendor sheets quote $`\approx9.9\times10^{14}`$, but that figure assumes 2:4 structured sparsity and must not be used here), at 40% MFU:
$$\text{effective} = 512\times4.95\times10^{14}\times0.40 = 1.01\times10^{17}\ \text{FLOP/s}$$
$$t = \frac{8.4\times10^{22}}{1.01\times10^{17}} = 8.3\times10^{5}\ \text{s} \approx \mathbf{9.6\ days}$$

▸ **Always state which peak you used.** Dense-vs-sparse peak is a factor of 2, and quoting the sparsity number against a dense workload silently halves your reported MFU.

▸ This calculation — $`C=6ND`$, divide by (devices × peak × MFU) — is the single most useful back-of-envelope in the field, and it comes up constantly in interviews. Practice it until it's automatic.

#### The back-of-envelope, one step at a time

There are only three steps and one piece of arithmetic you may be rusty on: $`10^a\times10^b = 10^{a+b}`$, and $`10^a/10^b = 10^{a-b}`$. Everything else is multiplication of small numbers.

**Step 1 — how much work is there?** $`C = 6ND`$ with $`N = 7\times10^9`$ and $`D = 2\times10^{12}`$. Handle the digits and the powers separately:

$$7\times 2 = 14,\qquad 10^{9}\times10^{12} = 10^{21}\quad\Rightarrow\quad ND = 1.4\times10^{22}$$
$$C = 6\times 1.4\times10^{22} = 8.4\times10^{22}\ \text{FLOPs}$$

> **How big is $`8.4\times10^{22}`$?** If every human being alive performed one multiplication per second, without sleeping, the species would finish this training run in about **330,000 years**. A single well-fed cluster does it in a week and a half. That gap — roughly ten orders of magnitude — is what the last two decades of parallel hardware actually bought.

**Step 2 — how fast can the cluster go?** Multiply devices by per-device peak by the fraction you expect to achieve:

$$512 \times 4.95\times10^{14} = 2.53\times10^{17}\ \text{FLOP/s (theoretical)}$$
$$2.53\times10^{17}\times 0.40 = 1.01\times10^{17}\ \text{FLOP/s (realistic)}$$

**Step 3 — divide, then convert to human units.**

$$t = \frac{8.4\times10^{22}}{1.01\times10^{17}} = 8.3\times10^{5}\ \text{s}$$

A day is $`86{,}400 \approx 8.64\times10^4`$ seconds, so $`8.3\times10^5 / 8.64\times10^4 \approx \mathbf{9.6\ days}`$. In the currency clusters are actually billed in: $`512 \times 9.6 \times 24 \approx \mathbf{118{,}000}`$ **GPU-hours.**

**Now the part that makes it useful — the sensitivity.** Every input enters linearly, so you can re-run the estimate in your head:

| Change | New time | Why |
|---|---|---|
| MFU 40% → 30% | 12.8 days | $`9.6\times\frac{0.40}{0.30}`$ |
| MFU 40% → 55% | 7.0 days | $`9.6\times\frac{0.40}{0.55}`$ |
| 512 → 1024 GPUs | 4.8 days | Twice the machines |
| 2T → 3T tokens | 14.4 days | $`C`$ is linear in $`D`$ |
| 7B → 13B params | 17.8 days | $`C`$ is linear in $`N`$ |

▸ **And the sparsity trap, made concrete.** Substituting the vendor's $`9.9\times10^{14}`$ headline figure doubles the denominator of the MFU definition, so **a run  achieving 40% would be reported as 20%** — or, if you use it to forecast, you would promise 4.8 days and deliver 9.6. The factor of two lives in a footnote about 2:4 structured sparsity, and dense transformer training does not use it. **This is the most common way a compute estimate is quietly wrong by 2×, and it is entirely a units problem.**

**Why this specific calculation is an interview staple.** It checks four things at once: that you know $`C=6ND`$ and can say where the 6 comes from; that you know what MFU means and roughly what value is realistic; that you know real hardware numbers rather than vague impressions; and that you can do exponent arithmetic without a calculator. **It is also, unglamorously, the arithmetic that actually gets done on the job — before every run, to decide whether the run is affordable at all.**

#### Examples and non-examples: the four kinds of parallelism

These are constantly conflated, and choosing wrong wastes an entire cluster.

**✅ What each one actually splits**

| Kind | What gets split across devices | When to reach for it |
|---|---|---|
| **Data parallelism** | The *batch* — each device holds a full model copy | Default; model fits on one device |
| **Tensor parallelism** | Individual *matrices*, within a layer | A single layer is too big for one device |
| **Pipeline parallelism** | *Layers* — device 1 holds layers 1–10, etc. | Model too deep for one device |
| **Sequence parallelism** | The *sequence length* | Activations dominate at long context |

**❌ Near-misses — common confusions**

| Belief | Why it's wrong |
|---|---|
| "ZeRO is a fifth kind of parallelism" | ZeRO is **data** parallelism that shards optimizer state, gradients, and parameters to remove redundancy |
| "Tensor parallelism scales across nodes" | It needs an all-reduce *per layer* — keep it inside one node's fast interconnect |
| "Pipeline parallelism uses all devices at once" | It creates a **bubble**; devices idle while waiting for the pipeline to fill and drain |
| "More GPUs is always faster" | Communication grows too; past a point you add hardware and lose throughput |
| "Gradient checkpointing speeds things up" | It **costs** roughly 30% more compute to save memory — a deliberate trade |

▸ **The boundary:** ask *"what dimension am I running out of?"* Out of memory for parameters → shard them (ZeRO / tensor / pipeline). Out of memory for activations → checkpointing or sequence parallelism. Out of *time* with memory to spare → data parallelism. **Each strategy targets a different scarce resource, and applying the wrong one adds complexity without relieving the actual constraint.**

> **Common misconception.** *"Gradient checkpointing is an optimization."* It makes training **slower** — you discard activations on the forward pass and recompute them during the backward pass, paying roughly a third more compute to cut activation memory dramatically. It is a trade you make when memory is the binding constraint, not a free win. The name misleads because most things called "checkpointing" are about safety, not memory.

> **Common misconception.** *"Model FLOPs Utilization is the same as GPU utilization."* `nvidia-smi` showing 100% means the GPU is *busy*, not that it is doing useful matrix work — it can be busy waiting on memory or communication. MFU compares achieved useful FLOPs against the hardware peak, and a good large-scale training run lands around 40–50%. **A cluster at 100% "utilization" and 15% MFU is mostly burning money.**

> **Common misconception.** *"bf16 and fp16 are interchangeable 16-bit formats."* They allocate their bits differently: bf16 keeps fp32's exponent range with less precision, fp16 keeps more precision over a much narrower range. That's why fp16 training needs loss scaling to avoid gradient underflow and bf16 generally does not. **The failure mode isn't inaccuracy, it's silent underflow to zero.**

---

## Did you know?

- **MinHash was invented to stop a search engine indexing the same page twice.** Andrei Broder built it at Digital Equipment Corporation in the late 1990s for AltaVista, which was drowning in mirrored web pages. Twenty-five years later the identical algorithm is used to stop language models *memorizing* the same page twice. The corpus never changed; only what we do with it did.

- **The Jaccard index comes from counting alpine flowers.** Paul Jaccard, a Swiss botanist, devised it around 1901 to compare which plant species grew on which mountain meadows. There is no probability and no computer anywhere in the original idea, and the formula reached web-scale text deduplication a century later completely unchanged.

- **A bf16 number is literally the top half of an fp32 number.** Same sign bit, same eight exponent bits, and the mantissa is fp32's with the bottom sixteen bits deleted. Converting fp32 to bf16 is a truncation and can never overflow — which is the entire reason it needs no loss scaling. The "b" stands for *brain*, after Google Brain, where it was designed for TPUs.

- **fp16's maximum is 65,504, and a surprising number of engineers know that number by heart.** It comes out of the format's five exponent bits as $`(2-2^{-10})\times2^{15}`$, and it is the exact value at which a training run starts emitting `inf`. fp16 was inherited from computer graphics, where colours and coordinates never get anywhere near it.

- **The $`\sqrt{L}`$ checkpointing trick predates deep learning by about twenty-five years.** Andreas Griewank analysed memory-versus-recomputation trade-offs for reverse-mode automatic differentiation in the early 1990s, motivated by scientific computing and optimal control. Neural networks rediscovered it in 2016 — the same pattern as backpropagation itself, which appeared in a Finnish master's thesis about rounding errors in 1970.

- **Ring all-reduce is borrowed from supercomputing.** "Sum a vector across every node" is the oldest problem in distributed physics simulation, and the ring algorithm was standard there long before machine learning existed. It reached deep learning around 2017, through work at Baidu and Uber's Horovod library.

- **PaLM's loss spikes could not be reproduced from the same data.** When the team replayed the exact batches that preceded a spike, starting from an earlier checkpoint, the spike did not happen again. The culprit was never a poisoned document — it was the coincidence of particular data meeting a particular parameter state. This is why "skip the batch and continue" is real engineering practice rather than superstition.

- **"GPU utilization" is close to meaningless, and MFU exists because of it.** A GPU computing pure garbage at full speed reports 100% utilization. Google's PaLM report introduced *Model FLOPs Utilization* in 2022 specifically to have a number that cannot be inflated by doing more work than the model requires — which is why gradient checkpointing, a good decision, correctly *lowers* your MFU.

- **On a 7B model, the model is 14 GB and the bookkeeping around it is 98 GB.** Weights, gradients, an fp32 master copy, and Adam's two moments come to sixteen bytes per parameter. You are carrying seven times more paperwork than product, and all of it must be resident in GPU memory at once.

- **Common Crawl is a nonprofit, and it exists for egalitarian reasons.** Founded by Gil Elbaz in 2007, its purpose was to ensure that web-scale data would not be the private property of the few companies able to crawl it. It succeeded in a way nobody anticipated: that free archive is the raw substrate of essentially every large language model in existence.

- **Half of every GPU spec sheet is a footnote about sparsity.** Vendors commonly quote a peak throughput that assumes 2:4 structured sparsity — a factor of two that dense transformer training does not get. Quote it against a dense workload and your reported efficiency is silently halved, which is the single most common way a compute estimate goes wrong by exactly 2×.

- **Meta published a day-by-day logbook of the OPT-175B training run.** It chronicles hardware failures, restarts, diverging losses, and the ordinary misery of keeping hundreds of GPUs in agreement for months. Publishing it was unusual, and it remains one of the few honest public records of what a frontier-scale run actually looks like from the inside rather than in the results table.

- **The pipeline "bubble" is the same phenomenon as a CPU pipeline stall.** A transformer's layers being fed micro-batches and a processor's stages being fed instructions are, at the level of the fill-and-drain arithmetic, the same machine — which is why a mispredicted branch and a badly configured pipeline schedule cost you idle silicon for identical reasons.

---

## Check for Understanding

**Training at scale is a memory and communication problem, not a mathematics problem: the $`16N`$ bytes of optimizer state force sharding, the $`O(T^2)`$ attention forces FlashAttention and context parallelism, bf16 works only because gradients need range rather than precision, and the quality of the result is determined mostly by how aggressively you deduplicated and filtered the data.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is deduplication the single highest-value step in a data pipeline?** Name the three separate problems it fixes at once.
2. **Explain MinHash to someone who knows no probability**, using the shuffled queue. Why is the identity exact rather than approximate?
3. **Why does bf16 work where fp16 needs loss scaling?** (A correct answer is about range versus precision. "It's newer" is not an answer.)
4. **Why do we keep fp32 master weights?** What exactly goes wrong without them, and why does the failure produce no error message?
5. **Where does the 16 in "$`16N`$ bytes" come from?** Count the five copies out loud.
6. **Why is $`\sqrt{L}`$ the right checkpointing interval, and what does the extra 33% of compute buy you?**
7. **What is an all-reduce, and why does the ring version's cost per device not grow as you add devices?**
8. **Why must tensor parallelism stay inside one node when data parallelism happily spans a datacentre?** Answer in terms of how often each one talks.
9. **In the feed-forward block, why split the first matrix by columns and the second by rows?** (The answer is one word, and it is the nonlinearity.)
10. **Why does the pipeline bubble shrink when you add micro-batches but not when you buy faster GPUs?**
11. **Why does the critical batch size grow as training proceeds?** Explain it as an election that keeps getting closer.
12. **Why is the model completely indifferent to how large its logits are — and why is that indifference exactly the problem?**
13. **Someone tells you their GPU utilization is 100%. What do you ask next, and why?**
14. **Estimate the training time for a 7B model on 2T tokens across 512 GPUs, out loud, in powers of ten, without a calculator.**

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 15 — Scaling Laws & Emergence](15-scaling-laws-and-emergence.md)
