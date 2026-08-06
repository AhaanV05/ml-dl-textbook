# Chapter 32 — Mechanistic Interpretability & Superposition

> **Prerequisites:** Ch. 11 (residual stream), Ch. 13 (§13.3 induction heads), Ch. 30.

---

## 32.1 The programme

### The one-line idea

Treat a trained network as a compiled program and reverse-engineer it into human-understandable algorithms — not "which input pixels mattered," but "what computation is this circuit performing, and can I prove it."

### The analogy

Interpretability by saliency map is like determining what a program does by watching which memory addresses it touches. Mechanistic interpretability is decompiling the binary. It is far harder, far slower, and gives you something qualitatively different: a claim you can *test* by editing the code and predicting the change.

### The three foundational hypotheses

▸ **1. Features.** Networks represent human-meaningful properties as **directions in activation space**. A feature is a direction, not a neuron.

▸ **2. Circuits.** Features are computed from earlier features by identifiable, reusable subgraphs of weights.

▸ **3. Universality.** The same features and circuits recur across models trained on similar data (curve detectors in every vision model; induction heads in every language model).

All three are empirical claims with substantial support and known exceptions. **The linear representation hypothesis is the load-bearing one**: it is what makes probing, steering, and sparse autoencoders possible at all. Evidence: linear probes recover an enormous range of properties; **steering vectors** (add a direction to the residual stream and the behaviour changes predictably) work; and arithmetic on representations behaves sensibly. Known limits: some features appear to be represented in circular or multi-dimensional structures (days of the week, months) rather than as single directions.

---

## 32.2 Superposition

### The problem

▸ A model has $d$ dimensions in its residual stream but needs to represent far more than $d$ concepts. GPT-3 has $d=12{,}288$ and plainly knows millions of things.

### The one-line idea

If features are **sparse** — only a few active at once — a network can pack many more than $d$ of them into $d$ dimensions using *almost*-orthogonal directions, accepting a little interference in exchange for a lot of capacity.

### The analogy

A radio spectrum. You have limited bandwidth, but if each station only broadcasts occasionally and you accept faint crosstalk, you can fit far more stations than "one per clean channel" would allow. The crosstalk is tolerable precisely because two stations are rarely loud at once.

### Why it's possible: almost-orthogonality

In $d$ dimensions you can have at most $d$ mutually orthogonal vectors. But by the Johnson–Lindenstrauss lemma (Ch. 1 §1.2), you can have $\exp(O(\epsilon^2 d))$ vectors with pairwise $|\cos\vartheta|<\epsilon$.

▸ **Numbers:** in $d=12{,}288$, the number of directions with pairwise cosine similarity below 0.1 is astronomically larger than $d$. **Exponentially many nearly-orthogonal directions exist.** That is the mathematical fact superposition exploits.

### The toy model

Anthropic's setup: sparse features $x\in\mathbb{R}^{n}$ (each active with probability $S$, with importance weights $I_i$), a bottleneck $h = Wx$ with $W\in\mathbb{R}^{d\times n}$, $d\ll n$, and reconstruction $\hat x = \mathrm{ReLU}(W^\top h + b)$. Train on weighted MSE.

▸ **The phase diagram** — the central result:

| Sparsity | Behaviour |
|---|---|
| **Dense** (all features active) | learn the top $d$ features, one per dimension, **orthogonally**; drop the rest. Classical PCA-like behaviour. |
| **Moderately sparse** | superposition begins; features organize into **antipodal pairs** (two features sharing a direction with opposite signs, since they rarely co-occur) |
| **Very sparse** | rich geometric structures — pentagons, **tetrahedra, triangular bipyramids** — corresponding to optimal sphere-packing configurations |

▸ **Two conditions are jointly necessary:** feature sparsity (so interference is rarely realized) and **a nonlinearity** (ReLU suppresses small interference terms, cleaning up the crosstalk). Without either, superposition doesn't form.

**The trade-off the model is solving:** representing more features reduces the error from *missing* features but increases the error from *interference*. The optimum depends on sparsity and on the relative importance of the features — and the network finds it.

### Polysemanticity

▸ **The observable consequence:** a single neuron fires for apparently unrelated inputs — academic citations, English dialogue, HTTP requests, and Korean text (a real documented example). Not a bug and not noise: the neuron is a *coordinate*, and multiple feature directions have components along it.

▸ **This is why "look at what neuron 1,432 responds to" fails as an interpretability method.** The neuron basis is not the feature basis. That realization is what motivated everything in §32.3.

---

## 32.3 Sparse autoencoders

### The one-line idea

If the model packed many sparse features into few dimensions, train an overcomplete autoencoder with a sparsity penalty to unpack them.

### The construction

▸ $$f(x) = \mathrm{ReLU}\big(W_{\text{enc}}(x-b_{\text{dec}}) + b_{\text{enc}}\big)\in\mathbb{R}^{m},\qquad m = 8d\ \text{to}\ 256d$$
▸ $$\hat x = W_{\text{dec}}f(x)+b_{\text{dec}}$$
▸ $$\mathcal{L} = \underbrace{\|x-\hat x\|_2^2}_{\text{reconstruction}} + \lambda\underbrace{\|f(x)\|_1}_{\text{sparsity}}$$

**Overcomplete** ($m\gg d$) because there are more features than dimensions — that is the entire premise. The $\ell_1$ penalty forces most of them off for any given input (typically 10–100 active out of 16k–16M).

**Decoder columns are unit-normalized**, otherwise the model can shrink $f$ and grow $W_{\text{dec}}$ to cheat the $\ell_1$ term.

### What it found

Anthropic's Claude 3 Sonnet SAE work recovered millions of interpretable features:
- Concrete entities (the Golden Gate Bridge, specific people, code functions).
- Abstract concepts (deception, sycophancy, inner conflict, security vulnerabilities).
- Multilingual and multimodal features — the same feature fires for a concept in English text, Chinese text, and an image.

▸ **The features are causal, not merely correlational.** Clamping the Golden Gate Bridge feature high made the model identify *as* the bridge across every context. **This is the crucial evidence**: an interpretability method that only produced correlations would be a curiosity, but steering demonstrates the direction is used by the computation.

### The known problems

- ▸ **Shrinkage.** The $\ell_1$ penalty biases *all* activations toward zero, systematically underestimating magnitudes even for correctly identified features. **JumpReLU** and **TopK** SAEs address this — TopK enforces exactly $k$ active features and drops $\ell_1$ entirely, removing the bias.
- **Dead features:** a large fraction never activate. Mitigations: resampling, auxiliary losses on dead features, careful init.
- **Feature splitting:** widen the SAE and one feature splits into many finer ones. ▸ **There is no canonical granularity** — this is arguably the deepest conceptual problem with SAEs, because it suggests "the" feature set may not be well-defined.
- **Evaluation is genuinely hard.** Reconstruction loss and $L_0$ are proxies; automated interpretability scores use an LLM to name and predict feature activations, which is circular in an uncomfortable way. Recent work has found SAEs underperform simpler baselines on some downstream tasks, and the field is actively debating how much they have delivered.

**Related methods:** transcoders (approximate an MLP layer sparsely, giving circuits directly), crosscoders (share features across layers or models), and attribution-based dictionary learning.

---

## 32.4 The transformer as a circuit

### The residual stream

▸ **The core framing (Elhage et al., "A Mathematical Framework for Transformer Circuits"): the residual stream is a communication channel that every layer reads from and writes to.**

- It is **linear** — every sub-layer *adds*. So contributions decompose additively and can be attributed.
- It has **finite bandwidth** $d$, shared by all $2L$ sub-layers. Hence superposition.
- Layers communicate across arbitrary distance by writing to a subspace and having a much later layer read it.
- **Layers do not compose sequentially** in the naive sense; layer 30 can read directly from layer 2's output.

**The logit lens:** apply the final unembedding to an intermediate residual stream to see the model's "current best guess." Predictions refine layer by layer, often converging several layers before the end. The **tuned lens** fits a per-layer affine correction and is substantially more faithful.

### QK and OV circuits

▸ An attention head factorizes into two independent, low-rank operations:

$$\text{attention pattern} = \mathrm{softmax}\!\left(\frac{x^\top W_Q^\top W_K x}{\sqrt{d_k}}\right)\quad\Rightarrow\quad W_{QK}=W_Q^\top W_K$$
$$\text{output} = \big(\text{pattern}\big)\cdot x\,W_V^\top W_O^\top\quad\Rightarrow\quad W_{OV}=W_OW_V$$

▸ **The QK circuit decides *where* to read; the OV circuit decides *what* to write.** They are separately analyzable, and each is a single low-rank matrix in the residual-stream basis. This factorization is what makes attention heads tractable objects of study at all.

### Induction heads — the fully worked example

**The behaviour:** given `... [A][B] ... [A]`, predict `[B]`. In-context copying.

**The circuit** requires two heads composing across layers:
1. **Previous-token head** (early layer): attends from position $t$ to $t-1$ and writes information about token $t-1$ into position $t$'s residual stream.
2. **Induction head** (later layer): its QK circuit uses the *current* token as query and matches against that written information — locating earlier positions whose *predecessor* was the current token. Its OV circuit then copies the token at that position to the output.

▸ **Why two layers are required:** the induction head's key must depend on the *previous* token, and no single attention layer can produce that. This is **K-composition** — one head writing into the subspace another head reads as keys.

**The evidence chain, which is the strongest in the field:**
- Induction heads form in a narrow window during training, visible as a bump in the loss curve.
- In-context learning ability jumps in the *same* window.
- Ablating induction heads destroys in-context learning.
- They appear in every transformer examined, at every scale — a strong case for universality.

### Other identified circuits

**Indirect Object Identification** ("John and Mary went to the store; John gave a drink to ___" → "Mary"): 26 heads in GPT-2 small in 7 functional classes, including **name-mover heads** (copy the name), **S-inhibition heads** (suppress the *subject* name), **duplicate-token heads**, and **backup name-movers** that take over when the primary is ablated. ▸ **That last finding — self-repair — is important and inconvenient: ablation understates a component's importance, because the network compensates.**

**Docstring, greater-than, modular arithmetic** (Ch. 30 §30.3 — completely reverse-engineered as a Fourier-transform circuit), and factual recall circuits localized to mid-layer MLPs, which is what ROME and MEMIT edit.

---

## 32.5 The causal toolkit

▸ **The core principle: correlation is not mechanism. Intervene.**

**Activation patching (causal tracing).** Run the model on a clean input and a corrupted one; copy an activation from one run into the other; measure the effect on the output. If patching component $X$ restores the correct behaviour, $X$ carries the relevant information.

- **Denoising** (corrupted → patch in clean): finds components *sufficient* to restore behaviour.
- **Noising** (clean → patch in corrupted): finds components *necessary*.
- ▸ These give different answers, and reporting only one is a common error.

**Path patching** restricts the intervention to a specific *edge* — patch the effect of head A on head B's queries only, leaving all other paths intact. This is what isolates a circuit rather than a set of components.

**Attribution patching** approximates patching with a first-order Taylor expansion, making it $O(1)$ forward+backward passes instead of $O(\text{components})$. Necessary for scaling to large models; less accurate for large effects.

**Causal scrubbing.** The strongest available validation: given a hypothesized circuit, replace every activation the hypothesis claims is irrelevant with a resampled value from a different input. ▸ **If the hypothesis is complete and correct, performance should be preserved.** Any drop measures what your explanation is missing. This turns interpretability claims into falsifiable predictions.

**Ablation** (zero, mean, or resample). ▸ **Mean-ablation is usually better than zero-ablation**, because zero is off-distribution and its effects conflate "this component mattered" with "the model was pushed somewhere weird."

---

## 32.6 Probing and its confounds

Train a classifier on internal activations to detect a property.

▸ **The confound that invalidates naive probing: a sufficiently expressive probe can find structure the model does not use.** A probe achieving 90% on syntax proves the information is *present*, not that the model *reads* it.

**Controls that make probing meaningful:**
- **Selectivity** (Hewitt & Liang): compare against a probe trained on random control labels. High accuracy with low selectivity means the probe learned the task itself.
- **Keep probes weak** — linear, low-capacity.
- **Amnesic probing:** remove the property (project out its direction) and check whether behaviour changes. This is the causal version.
- ▸ **Always follow a probe with an intervention.** A probe is a hypothesis; steering or patching is the test.

**Steering vectors.** Compute $v = \bar h_{\text{positive}} - \bar h_{\text{negative}}$ over contrastive pairs, add $\alpha v$ to the residual stream at inference. Works for sentiment, refusal, truthfulness, sycophancy, and style. ▸ **That such a crude method works at all is strong evidence for the linear representation hypothesis** — and SAE features give cleaner, more targeted steering directions.

---

## 32.7 The honest status

### What is established

- Features are largely linear directions; superposition is real and its toy model is well-understood.
- Specific circuits have been reverse-engineered end to end in small models (modular arithmetic, IOI, induction).
- Causal interventions work and predict behaviour.
- SAEs recover large numbers of interpretable, causally-relevant features.
- **Universality holds for at least some circuits.**

### What is not

- **No frontier model has been fully explained.** Circuit-level understanding covers a small fraction of behaviour.
- **Faithfulness of chain-of-thought is not established.** Models sometimes produce reasoning that demonstrably does not reflect the computation that produced the answer (they can be steered to a conclusion by a hint they never mention). ▸ **This matters directly for safety and for evaluation: a plausible explanation is not evidence of the underlying process.**
- **Self-repair** means ablation systematically understates importance.
- **SAE feature granularity is not canonical**, and SAE evaluation remains contested.
- **Scaling is the binding constraint.** IOI took months for one behaviour in a 117M-parameter model.

### Why it matters anyway

▸ **Safety:** detecting deception, backdoors, or dangerous capabilities requires looking at the mechanism, not the output — a model that behaves well under evaluation and badly in deployment is undetectable behaviourally by construction.
▸ **Debugging:** knowing *why* a model fails is the difference between fixing it and guessing.
▸ **Editing:** targeted knowledge editing and behaviour steering require knowing where things live.
▸ **Science:** it is the only route from "deep learning works" to "here is why."

---

## Check for Understanding

**Networks represent more features than they have dimensions by placing them in nearly-orthogonal directions and relying on sparsity to keep interference rare — which is why individual neurons are polysemantic and why sparse autoencoders can unpack them — and the transformer's linear residual stream makes this analyzable, since every head decomposes into a QK circuit that decides where to read and an OV circuit that decides what to write, with causal interventions rather than correlations being what turns any of it into evidence.**

---

**Next:** [Chapter 33 — Calibration, Uncertainty & Robustness](33-calibration-uncertainty-robustness.md)
