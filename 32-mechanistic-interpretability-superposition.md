# Chapter 32 — Mechanistic Interpretability & Superposition

> **Prerequisites:** Ch. 11 (residual stream), Ch. 13 (§13.3 induction heads), Ch. 30.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

> **Why this chapter exists:** every other chapter tells you how to *build* a network. This one asks what is actually inside the thing you built — and supplies the only tools anyone has for answering that with evidence instead of a plausible story. The single most important idea in it, **superposition**, is a direct consequence of the Johnson–Lindenstrauss lemma you met in **Chapter 1 §1.1.5**; if that section is hazy, re-read it before §32.2.

### Symbols introduced in this chapter

Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`d`$ | "d" | Width of the **residual stream** — how many numbers describe one token inside the model |
| $`n`$ | "n" | How many *features* (concepts) the model wants to represent. The whole chapter assumes $`n \gg d`$ |
| $`m`$ | "m" | How many entries a sparse autoencoder's **dictionary** has. Also $`m \gg d`$ |
| $`x \in \mathbb{R}^n`$ | "x in R-n" | A feature vector that is **mostly zeros** — "which concepts are present right now" |
| $`h = Wx`$ | "h equals W x" | The squeezed $`d`$-number representation the model actually stores |
| $`\hat x`$ | "x-hat" | A **reconstruction** — an attempt to recover $`x`$ from $`h`$ |
| $`\mathrm{ReLU}(z)`$ | "rel-you of z" | $`\max(0,z)`$ — keep positive numbers, flatten negative ones to zero |
| $`\cos\vartheta`$ | "cosine theta" | **Alignment** of two directions: $`1`$ = identical, $`0`$ = perpendicular, $`-1`$ = opposite |
| $`\epsilon`$ | "epsilon" | A small tolerance — here, how far from perpendicular we are willing to accept |
| $`\|f(x)\|_1`$ | "the $`\ell_1`$ norm of f" | Add up the sizes of all the entries — the **sparsity penalty** |
| $`\lambda`$ | "lambda" | How hard that sparsity penalty pushes |
| $`L_0`$ | "L-zero" | **How many entries are nonzero.** A count, not a length |
| $`W_{QK}=W_Q^\top W_K`$ | "the Q-K circuit" | The one matrix that decides *where* an attention head reads from |
| $`W_{OV}=W_OW_V`$ | "the O-V circuit" | The one matrix that decides *what* that head writes back |
| $`L`$ | "L" | Number of transformer layers. All $`2L`$ sub-layers share one stream |
| $`\alpha v`$ | "alpha times v" | A **steering vector**, scaled by $`\alpha`$, added into the residual stream |

**Full forms for the abbreviations in this chapter** (the complete book-wide list is in §0.13):

| Short | Full form |
|---|---|
| SAE | Sparse AutoEncoder |
| MLP | Multi-Layer Perceptron (the feed-forward block in a transformer) |
| QK / OV | Query–Key / Output–Value (circuits) |
| IOI | Indirect Object Identification |
| ROME | Rank-One Model Editing |
| MEMIT | Mass-Editing Memory In a Transformer |
| PCA | Principal Component Analysis |
| MSE | Mean Squared Error |
| ReLU | Rectified Linear Unit |
| J–L | Johnson–Lindenstrauss (lemma) |
| CoT | Chain-of-Thought |
| TopK | "top-$`k`$" — keep the $`k`$ largest, zero the rest |

---

## 32.1 The programme

### The one-line idea

Treat a trained network as a compiled program and reverse-engineer it into human-understandable algorithms — not "which input pixels mattered," but "what computation is this circuit performing, and can I prove it."

### The analogy

Interpretability by saliency map is like determining what a program does by watching which memory addresses it touches. Mechanistic interpretability is decompiling the binary. It is far harder, far slower, and gives you something qualitatively different: a claim you can *test* by editing the code and predicting the change.

#### What a "saliency map" is, and why the analogy is unkind on purpose

A **saliency map** answers: *which input pixels, if nudged, would change the output most?* Mechanically it is $`\nabla_x \mathcal{L}`$ — the gradient of the loss with respect to the **input** rather than the parameters (§0.7: the subscript on $`\nabla`$ says what you differentiate with respect to). Take the absolute value, reshape it back to the image grid, and colour it in. Bright pixels "mattered."

The trouble is that this tells you about **sensitivity at one point**, not about **computation**. A thermostat's output is highly sensitive to the room temperature; that fact tells you nothing about whether it contains a bimetallic strip or a microcontroller.

▸ **The result that settled the argument:** in *Sanity Checks for Saliency Maps* (Adebayo et al., 2018), several popular saliency methods produced nearly the **same picture** for a fully trained network and for a network with **randomly initialized weights**. A method whose output is unchanged when you delete everything the model learned is not explaining the model — it is mostly re-drawing edges in the image.

**Why the "decompiling" framing is the right one.** A decompiled program supports a *prediction*: "if I change this line, the output changes like so." That is falsifiable. Every technique in §32.5 exists to turn interpretability claims into exactly that shape — you say what a component does, then you break it and check whether the model fails in the way you predicted.

> **Where this came from.** The modern programme grew out of feature-visualization work at Google. In 2015, **Alexander Mordvintsev**, **Christopher Olah**, and **Mike Tyka** published *Inceptionism* — images made by running gradient ascent on a vision network's own activations to see what excited them. Nicknamed **DeepDream**, it escaped the lab as internet art before anyone treated it as science; Mordvintsev built the first version while investigating what an image classifier had actually learned. Olah and collaborators then turned the technique into a research agenda: the *Circuits* thread in the journal *Distill*, opening in 2020 with *Zoom In: An Introduction to Circuits*, which stated the three hypotheses below as **claims to be tested** rather than assumptions. The programme later moved to Anthropic and was re-aimed from vision models at transformers. The adjective *mechanistic* was adopted precisely to separate this from the then-dominant style of interpretability — the saliency-map style — which explains *which inputs mattered* rather than *what the model computes*.

### The three foundational hypotheses

▸ **1. Features.** Networks represent human-meaningful properties as **directions in activation space**. A feature is a direction, not a neuron.

▸ **2. Circuits.** Features are computed from earlier features by identifiable, reusable subgraphs of weights.

▸ **3. Universality.** The same features and circuits recur across models trained on similar data (curve detectors in every vision model; induction heads in every language model).

All three are empirical claims with substantial support and known exceptions. **The linear representation hypothesis is the load-bearing one**: it is what makes probing, steering, and sparse autoencoders possible at all. Evidence: linear probes recover an enormous range of properties; **steering vectors** (add a direction to the residual stream and the behaviour changes predictably) work; and arithmetic on representations behaves sensibly. Known limits: some features appear to be represented in circular or multi-dimensional structures (days of the week, months) rather than as single directions.

#### What "a feature is a direction, not a neuron" actually says

Three words carry all the weight here. Take them one at a time.

- **Activation space.** At each layer, the model's state for *one token* is a list of $`d`$ numbers. Read that list as a point in $`d`$-dimensional space. For GPT-2 small $`d = 768`$; for GPT-3, $`d = 12{,}288`$. "Activation space" is nothing more than the space those lists live in — exactly the $`\mathbb{R}^d`$ of §0.2.
- **A direction.** Pick a unit vector $`v\in\mathbb{R}^d`$ — an arrow of length 1. To ask "how much of $`v`$ is in the current state $`h`$?" compute the dot product $`v^\top h`$, which returns a single number (§0.8, reading 1). That number is the **feature's activation**.
- **A neuron.** Neuron 1,432 *is* a direction — the one pointing along coordinate 1,432, namely $`(0,0,\dots,1,\dots,0)`$. It is a perfectly good direction. There is simply no law saying it is a **meaningful** one.

▸ **So the hypothesis is: meaning lives in arbitrary directions, and the neurons are just the arbitrary coordinate system the code happened to be written in.**

> **Analogy.** A city's streets run north–south and east–west, so every address is naturally two coordinates. But the *river* runs diagonally across the grid. "Distance from the river" is a real and useful quantity — and it is a blend of both coordinates, belonging cleanly to neither. Ask "what does the north–south axis mean?" and you get a mumble; ask "what does the river direction mean?" and you get a crisp answer. **The neuron basis is the street grid. Features are rivers.** Nobody laid the streets to line up with the water.

**Put real numbers on it.** Set $`d = 2`$ so you can see the whole thing. Suppose the feature "this text is formal" lives along the unit direction $`v = (0.6,\ 0.8)`$ — check: $`0.6^2+0.8^2 = 0.36+0.64 = 1`$. Suppose the current state is $`h = (3,\ 4)`$. Then

$$v^\top h = 0.6(3) + 0.8(4) = 1.8 + 3.2 = 5.0$$

The feature is strongly on. Now look at the neurons: neuron 1 reads $`3`$, neuron 2 reads $`4`$. **Neither number is the feature**, and neither on its own tells you it fired. You needed the mixture $`0.6`$-of-one-plus-$`0.8`$-of-the-other. Ask "what does neuron 1 mean?" and the honest answer is *"it's 0.6 of formality plus whatever else happens to lean this way."*

**Why the three hypotheses are stated in that order.** They stack:

| Hypothesis | If true, you can... | If false... |
|---|---|---|
| **Features** are directions | detect one with a dot product, add one with vector addition, and decompose a state into a sum of features | nothing else in this chapter works |
| **Circuits** are reusable subgraphs | name a mechanism once and expect to find it again in the same model | every behaviour is a special case; no generalization |
| **Universality** across models | study a 117M-parameter model and learn something about a 400B one | interpretability must be redone per model, and does not scale |

▸ **The linear representation hypothesis is load-bearing because it is what converts "understanding a network" from a philosophy problem into a linear algebra problem.** Detecting a feature becomes a dot product; inserting one becomes an addition; separating them becomes a sparse decomposition. All three are things we know how to do.

**The evidence, made concrete.** The oldest piece predates the field's current vocabulary: word embeddings satisfy $`\text{king} - \text{man} + \text{woman} \approx \text{queen}`$ (Mikolov and colleagues, 2013). That only makes sense if "royalty" and "gender" are *directions* that can be added and subtracted. The modern version is a steering vector: average the residual stream over a few dozen happy sentences, subtract the average over a few dozen sad ones, add the difference back during generation, and the model's tone shifts. Crude arithmetic on activations should not work at all if meaning were tangled up nonlinearly. It does.

**And the honest exception.** Recent work (2024) found that some concepts are *not* single directions: days of the week and months of the year are represented on **circles**, where the model rotates through a cycle rather than sliding along a line. That is still linear-algebraic — a circle lives in a 2-D plane — but it is a *two*-dimensional feature, not a one-dimensional one. The hypothesis survives in the form "features live in low-dimensional subspaces," which is weaker than "features are single directions."

---

## 32.2 Superposition

### The problem

▸ A model has $`d`$ dimensions in its residual stream but needs to represent far more than $`d`$ concepts. GPT-3 has $`d=12{,}288`$ and plainly knows millions of things.

#### The counting problem, stated bluntly

Here is the whole difficulty in one arithmetic comparison.

| Quantity | GPT-3 |
|---|---|
| Residual stream width $`d`$ | $`12{,}288`$ numbers per token |
| Exactly perpendicular directions available in $`\mathbb{R}^d`$ | exactly $`12{,}288`$ — that **is** the definition of dimension |
| Distinct things the model demonstrably knows | millions: every person, place, idiom, API, chemical, chess opening... |

If each concept demanded its own private, non-interfering axis, the model could hold **twelve thousand concepts and not one more**. It plainly holds vastly more. So either the linear representation hypothesis is wrong, or concepts are *sharing* axes.

▸ **Superposition is the second answer, and it turns out to be correct.**

> **Analogy.** A hotel with 12,288 rooms and ten million guests. If every guest needs a private room, you are finished before you start. But suppose almost every guest is out all day and only a handful are ever in the building at once. Now you can assign several guests to each room and collisions will be rare — and when two do collide, they mostly ignore each other. **Sparsity is what makes overbooking safe.** The whole of §32.2 is an argument about how badly you can overbook and what happens when you push too far.

### The one-line idea

If features are **sparse** — only a few active at once — a network can pack many more than $`d`$ of them into $`d`$ dimensions using *almost*-orthogonal directions, accepting a little interference in exchange for a lot of capacity.

### The analogy

A radio spectrum. You have limited bandwidth, but if each station only broadcasts occasionally and you accept faint crosstalk, you can fit far more stations than "one per clean channel" would allow. The crosstalk is tolerable precisely because two stations are rarely loud at once.

### Why it's possible: almost-orthogonality

In $`d`$ dimensions you can have at most $`d`$ mutually orthogonal vectors. But by the Johnson–Lindenstrauss lemma (Ch. 1 §1.2), you can have $`\exp(O(\epsilon^2 d))`$ vectors with pairwise $`|\cos\vartheta|<\epsilon`$.

▸ **Numbers:** in $`d=12{,}288`$, the number of directions with pairwise cosine similarity below 0.1 is astronomically larger than $`d`$. **Exponentially many nearly-orthogonal directions exist.** That is the mathematical fact superposition exploits.

#### Reading "almost-orthogonal" with real numbers

This is the load-bearing paragraph of the chapter and it is written very compressed. Unpack it.

**Every symbol.**

- $`\vartheta`$ ("theta") is the **angle between two directions**. $`\cos\vartheta`$ is their alignment: $`1`$ if identical, $`0`$ if perpendicular, $`-1`$ if opposite. For unit vectors it *is* the dot product, $`\cos\vartheta = u^\top v`$.
- $`|\cos\vartheta| < \epsilon`$ reads *"the absolute alignment is smaller than epsilon"* — the bars are absolute value, so it rules out strongly opposed as well as strongly aligned. With $`\epsilon = 0.1`$: **no two directions overlap by more than 10%.**
- $`\exp(O(\epsilon^2 d))`$ reads *"e raised to something proportional to epsilon-squared times d."* The $`O(\cdot)`$ (§0.10) hides a constant we don't know exactly; what matters is that $`d`$ sits **in the exponent**.
- **Orthogonal** means perpendicular, dot product zero, and — the intuition that matters — **non-interfering**. If I write along $`u`$ and you read along $`v`$ with $`u^\top v = 0`$, you see exactly nothing of what I wrote.

**The two counts, side by side.** Fix $`d = 12{,}288`$ and $`\epsilon = 0.1`$:

| Requirement | How many directions fit |
|---|---|
| $`\cos\vartheta`$ exactly $`0`$ (true orthogonality) | $`12{,}288`$ — linear in $`d`$ |
| $`\lvert\cos\vartheta\rvert < 0.1`$ (almost orthogonality) | $`\exp(c\,\epsilon^2 d)`$ — **exponential in $`d`$** |

Put arithmetic in the exponent. $`\epsilon^2 d = (0.1)^2 \times 12{,}288 = 122.88`$, and $`e^{122.88}\approx 2\times 10^{53}`$. Even if the hidden constant $`c`$ is a punishing $`\tfrac14`$, you get $`e^{30.7}\approx 2\times10^{13}`$ — **twenty trillion** directions, versus twelve thousand. ▸ **Relaxing "perpendicular" to "almost perpendicular" changes the count from linear in $`d`$ to exponential in $`d`$. That is not a small improvement; it is a different universe.**

**Where the smallness comes from.** Two *random* unit vectors in $`\mathbb{R}^d`$ have expected cosine similarity $`0`$ with standard deviation $`1/\sqrt{d}`$ — the same $`1/\sqrt{n}`$ law as the standard error in §1.3.1, and for the same reason: a dot product is a sum of $`d`$ random terms that partially cancel. So:

$$d = 12{,}288 \quad\Rightarrow\quad \tfrac{1}{\sqrt{d}} = \tfrac{1}{110.85} \approx 0.009$$

▸ **Two directions picked at random out of GPT-3's residual stream overlap by about 0.9%.** You do not have to *design* near-orthogonal directions in high dimensions. You get them by accident. Reaching $`|\cos\vartheta| = 0.1`$ would take an eleven-standard-deviation coincidence.

> **Analogy.** In a room, two people can face  different directions but you can only fit so many "completely different" orientations before someone repeats. On a sphere in ten thousand dimensions, you can throw darts blindfolded all day and every pair will land nearly perpendicular. **High-dimensional space is almost entirely made of right angles.** The intuition that fails you here is the 3-D intuition you were born with, in which "different directions" run out fast.

**The correct cross-reference.** The theorem underneath this is the **Johnson–Lindenstrauss lemma**, stated and unpacked in **Chapter 1 §1.1.5** — read there, in particular, the paragraph "Now read it backwards," which does exactly the inversion used here: J–L normally answers *"given $`N`$ points, how few dimensions do I need?"* ($`d = O(\log N/\epsilon^2)`$), and superposition needs the same statement read the other way, *"given $`d`$ dimensions, how many near-orthogonal directions fit?"* ($`N \approx e^{\epsilon^2 d}`$). Chapter 1 calls J–L "the mathematical permission slip for superposition"; this section is where the permission gets spent.

**And the honest caveat.** Almost-orthogonal is not orthogonal. If features $`i`$ and $`j`$ sit at $`\cos\vartheta = 0.09`$ and feature $`i`$ fires with strength $`10`$, the read-out for feature $`j`$ picks up a phantom $`0.9`$. Superposition does not eliminate interference — **it makes interference rare and small enough to be worth it.** Everything in the phase diagram below is the network deciding exactly how much of that bargain to take.

### The toy model

Anthropic's setup: sparse features $`x\in\mathbb{R}^{n}`$ (each active with probability $`S`$, with importance weights $`I_i`$), a bottleneck $`h = Wx`$ with $`W\in\mathbb{R}^{d\times n}`$, $`d\ll n`$, and reconstruction $`\hat x = \mathrm{ReLU}(W^\top h + b)`$. Train on weighted MSE.

#### Unpacking the toy model

This is a one-sentence description of an experiment you could code in twenty lines, so let us actually build it.

**Every symbol, read aloud.**

| Symbol | Read aloud | What it is |
|---|---|---|
| $`x\in\mathbb{R}^n`$ | "x in R-n" | The **ground truth**: a list of $`n`$ feature strengths, almost all exactly $`0`$ |
| $`S`$ | "S" | Probability that any one feature is active on any one sample. Small |
| $`I_i`$ | "I-i" | **Importance weight** of feature $`i`$ — how much you are penalized for getting it wrong |
| $`W\in\mathbb{R}^{d\times n}`$ | "W in R d-by-n" | The squeeze. Its $`n`$ **columns** are the directions assigned to the $`n`$ features |
| $`h = Wx`$ | "h equals W x" | The stored state: $`d`$ numbers, the bottleneck |
| $`W^\top`$ | "W transpose" | Rows become columns (§0.6). Reading with the *same* directions used for writing |
| $`b`$ | "b" | A learned per-feature offset. It will turn out to be the noise gate |
| $`\hat x`$ | "x-hat" | The recovered guess at $`x`$ |

**What the model is doing, in English.** *"I have $`n`$ concepts but only $`d`$ slots. Give each concept a direction. To store a set of concepts, add up their directions with the right strengths. To read a concept back out, dot the stored vector against that concept's direction, subtract a threshold, and clip anything negative to zero."*

Note something important: $`h = Wx`$ with the columns reading $`W_{:,1},\dots,W_{:,n}`$ is exactly **Reading 2 from §1.1.1** — a matrix-vector product is a *weighted sum of columns*. Storing three features means adding three arrows.

**Now make it small enough to hold in your head: $`n = 5`$, $`d = 2`$.** Five concepts, two numbers of storage. Set $`S = 0.05`$, so each feature is on 5% of the time.

- Expected number of features active per sample: $`nS = 5(0.05) = 0.25`$. **Most samples have nothing on at all.**
- Probability two *specific* features collide: $`S^2 = 0.0025`$ — one sample in four hundred.
- Probability *any* two collide: $`1 - (0.95)^5 - 5(0.05)(0.95)^4 = 1 - 0.774 - 0.204 = 0.023`$, about **2.3% of samples**.

▸ **That last number is the entire bargain.** The network is deciding whether to accept 2.3% of samples being slightly corrupted in exchange for storing five concepts in two dimensions instead of two concepts in two dimensions.

**The solution it finds: a pentagon.** Place the five columns of $`W`$ as unit arrows in the plane, evenly spaced $`72°`$ apart. Then for any two features, $`\cos\vartheta`$ is either $`\cos 72° = 0.309`$ (the two neighbours) or $`\cos 144° = -0.809`$ (the two non-neighbours).

**Watch one feature go in and come out.** Set $`x_1 = 1`$ and everything else $`0`$. Then $`h = W_{:,1}`$, and reading back with $`W^\top`$ gives the five alignments

$$W^\top h = (\underbrace{1.000}_{\text{feature }1},\ \underbrace{0.309}_{2},\ \underbrace{-0.809}_{3},\ \underbrace{-0.809}_{4},\ \underbrace{0.309}_{5})$$

Now add a bias of $`b = -0.31`$ everywhere and apply ReLU (§0's "keep positives, zero negatives"):

$$\hat x = \mathrm{ReLU}(1.000-0.31,\ -0.001,\ -1.119,\ -1.119,\ -0.001) = (0.69,\ 0,\ 0,\ 0,\ 0)$$

▸ **Feature 1 comes back; the four phantoms are gone.** Read what each piece did:
- The two **negative** overlaps ($`-0.809`$) were killed by ReLU **for free** — negative interference costs nothing, which is why the geometry likes spreading features into opposing configurations.
- The two **positive** overlaps ($`+0.309`$) were killed by the bias, which acts as a **noise gate**: anything quieter than $`0.31`$ is treated as crosstalk and silenced.
- The price is **shrinkage**: feature 1 comes back as $`0.69`$, not $`1.0`$. The same threshold that removes the noise also shaves the signal. (Hold that thought — the identical trade reappears in §32.3 as the SAE's biggest known flaw.)

**And the failure, honestly.** Fire features $`1`$ and $`3`$ together — the pair at $`-0.809`$. Then $`(W^\top h)_1 = 1 - 0.809 = 0.191`$, which is below the $`0.31`$ gate, so ReLU zeroes it. **Both features vanish completely.** That happens on $`S^2 = 0.25\%`$ of samples, and the network judges it a fair price. This is what "accepting a little interference in exchange for a lot of capacity" means arithmetically.

> **Analogy.** A noise gate on a microphone. Set the threshold above the hum of the air conditioning and the hum disappears — but a whispered word disappears with it, and if two speakers cancel each other you get silence where there should have been two voices. **ReLU plus a negative bias is a noise gate, and superposition is only survivable because there is one.**

▸ **The phase diagram** — the central result:

| Sparsity | Behaviour |
|---|---|
| **Dense** (all features active) | learn the top $`d`$ features, one per dimension, **orthogonally**; drop the rest. Classical PCA-like behaviour. |
| **Moderately sparse** | superposition begins; features organize into **antipodal pairs** (two features sharing a direction with opposite signs, since they rarely co-occur) |
| **Very sparse** | rich geometric structures — pentagons, **tetrahedra, triangular bipyramids** — corresponding to optimal sphere-packing configurations |

▸ **Two conditions are jointly necessary:** feature sparsity (so interference is rarely realized) and **a nonlinearity** (ReLU suppresses small interference terms, cleaning up the crosstalk). Without either, superposition doesn't form.

**The trade-off the model is solving:** representing more features reduces the error from *missing* features but increases the error from *interference*. The optimum depends on sparsity and on the relative importance of the features — and the network finds it.

#### The phase diagram, decoded

The word **phase** is not decoration. As you slide the sparsity dial, the solution does not deform smoothly — it **jumps**, the way water does not gradually become ice. Here is what each regime means with numbers attached.

A useful single statistic is **dimensions per feature**, $`D^* = d/n`$ — how much of a dimension each stored feature gets. Below $`1`$, the model is overbooking.

| Regime | Sparsity | Geometry it picks | $`D^*`$ | Why |
|---|---|---|---|---|
| **Dense** | $`S \approx 1`$ — everything on at once | top $`d`$ features on orthogonal axes, rest discarded | $`1.0`$ | Interference would be *constant*, so it is never worth paying |
| **Moderately sparse** | $`S \approx 0.1`$ | **antipodal pairs** — two features on one line, opposite signs | $`0.5`$ | $`\cos\vartheta = -1`$, and ReLU deletes negative crosstalk for free |
| **Very sparse** | $`S \approx 0.01`$ or below | pentagons, **tetrahedra**, **triangular bipyramids** | $`0.4`$ and below | Spread as many arrows as possible while keeping every overlap small or negative |

**Dense, decoded.** If every feature is on in every sample, then every pair collides in every sample. Interference stops being an occasional accident and becomes a permanent tax. The optimal move is to give up: keep the $`d`$ most important features on clean perpendicular axes and throw the other $`n-d`$ away. ▸ **Notice what this is — it is PCA.** Keep the top $`d`$ components, discard the tail, exactly the Eckart–Young result from §1.1.3. **A dense network is a linear compressor. Sparsity is what makes it something more interesting.**

**Antipodal pairs, decoded.** Two features $`i`$ and $`j`$ share one line: $`W_{:,j} = -W_{:,i}`$, so $`\cos\vartheta = -1`$. Store feature $`i`$ at strength $`2`$ and read out feature $`j`$: you get $`-2`$, and ReLU turns that into $`0`$. **A perfectly hostile overlap is a perfectly harmless one**, provided the readout can only see positive numbers. The pair only breaks when both fire at once — probability $`S^2 = 0.01`$ at $`S = 0.1`$ — and then the larger one survives and the smaller is erased.

**The exotic geometries, decoded.** With $`n = 4`$ features and $`d = 3`$ dimensions, the arrangement that keeps every pairwise overlap as negative as possible is the **regular tetrahedron**: four unit arrows with $`\cos\vartheta = -\tfrac13 \approx -0.333`$ between *every* pair. Every overlap is negative, so ReLU absorbs all of it for free. With $`n = 5`$ and $`d = 3`$, the answer is the **triangular bipyramid** — two poles plus a triangle round the equator.

▸ **Those are not arbitrary shapes; they are the answers to a 120-year-old physics problem.** "Place $`N`$ points on a sphere so that the repulsion between them is minimized" is the **Thomson problem**, posed by J. J. Thomson in 1904 while modelling how electrons might arrange themselves inside an atom. Its known solutions are: $`N=4`$ → tetrahedron, $`N=5`$ → triangular bipyramid, $`N=6`$ → octahedron. A neural network trained on synthetic sparse data, with no knowledge of any of this, rediscovers the same configurations — because "spread directions apart so they interfere least" and "spread charges apart so they repel least" are the same optimization wearing different clothes.

#### Why both conditions are needed — a two-line proof

**Drop the nonlinearity.** Suppose $`\hat x = W^\top W x`$ with no ReLU and no bias. Then the reconstruction of feature $`j`$ is $`\hat x_j = x_j + \sum_{i\ne j}(\cos\vartheta_{ij})\,x_i`$ — the true value **plus every crosstalk term, permanently, with no way to remove any of it.** The best you can do is minimize the total squared error, and the answer to that is PCA again. ▸ **Without a nonlinearity there is no gate, and without a gate superposition has no upside.** ReLU is not a detail here; it is the mechanism.

**Drop the sparsity.** Set $`S = 1`$ so every feature fires on every sample. Then the crosstalk sum above has $`n-1`$ nonzero terms *every single time*. With $`n = 100`$ features at typical overlap $`0.1`$, the noise on each readout is roughly $`\sqrt{99}\times 0.1 \approx 1.0`$ — the same size as the signal you are trying to read. ▸ **Sparsity is what keeps that sum nearly empty.** At $`S = 0.01`$ the expected number of interfering terms drops to about $`1`$, and the noise with it.

> **Where this came from.** The toy model is from *Toy Models of Superposition* (**Elhage, Hume, Olsson and colleagues at Anthropic, 2022**) — a paper whose method is deliberately unfashionable: instead of probing a real language model, they built the smallest possible synthetic system that could exhibit the phenomenon and studied it exhaustively. The word **superposition** is borrowed from physics, where it means a state that is a sum of several basis states at the same time. The underlying mathematics is older and has another name: **compressed sensing**, the result of Emmanuel Candès, Justin Romberg, Terence Tao and (independently) David Donoho around 2006, which says a *sparse* signal can be recovered exactly from far fewer measurements than its dimension would suggest. Compressed sensing was developed for magnetic-resonance imaging and signal acquisition — the practical goal was shorter MRI scans. The theory that lets a hospital scanner finish in a third of the time is the theory that explains why a neuron responds to Korean text and HTTP requests at once.

### Polysemanticity

▸ **The observable consequence:** a single neuron fires for apparently unrelated inputs — academic citations, English dialogue, HTTP requests, and Korean text (a real documented example). Not a bug and not noise: the neuron is a *coordinate*, and multiple feature directions have components along it.

▸ **This is why "look at what neuron 1,432 responds to" fails as an interpretability method.** The neuron basis is not the feature basis. That realization is what motivated everything in §32.3.

#### Polysemanticity, decoded

**The word.** Greek *poly* (many) + *sēma* (sign, meaning): "many meanings." Its opposite, **monosemantic**, is the thing interpretability wants and rarely gets — one neuron, one concept. Both terms are borrowed from linguistics, where "bank" is the standard polysemous word.

**The mechanism, in one line of algebra.** Neuron $`j`$ reports the $`j`$-th coordinate of the state, which by $`h = Wx`$ is

$$h_j = \sum_{i=1}^{n} W_{ji}\,x_i$$

▸ Read that aloud: *"neuron $`j`$'s activation is a weighted sum over **every** feature, weighted by how much of feature $`i`$'s direction happens to point along axis $`j`$."* Because $`x`$ is sparse, only one or two terms are nonzero at a time — so neuron $`j`$ lights up for **whichever of its features happens to be active right now**, and to an observer scrolling through examples that looks like a neuron with no coherent theme.

**Go back to the pentagon.** $`n=5`$, $`d=2`$, features at $`0°, 72°, 144°, 216°, 288°`$. Neuron 1 *is* the horizontal axis, $`(1,0)`$. Its component on each feature is the cosine of that feature's angle:

| Feature | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Component on neuron 1 | $`1.000`$ | $`0.309`$ | $`-0.809`$ | $`-0.809`$ | $`0.309`$ |

Ask "what does neuron 1 detect?" and the only truthful answer is: *"feature 1 strongly, features 2 and 5 weakly, features 3 and 4 in reverse."* **That is not a concept. It is a linear combination of five concepts, and no English sentence describes it** — not because the network is doing something mysterious, but because you asked about the wrong basis. In the documented real example, "academic citations, English dialogue, HTTP requests, Korean text" is what that mixture looks like when the five features are those four things plus noise.

> **Analogy.** A microphone at the back of a busy restaurant. Ask "what conversation is this microphone recording?" and there is no answer — it is picking up table 3 loudly, tables 7 and 12 faintly, and the kitchen in the background. The microphone is not confused and it is not broken. It is a *location*, and conversations are not organized by location. **Neurons are microphones. Features are conversations. Sparse autoencoders are the un-mixing.**

▸ **The uncomfortable corollary:** a decade of "we found the cat neuron" results were reading the mixture, not the source. Some of those neurons really were near-monosemantic — that happens when a feature is important enough to earn a private axis — but there was never a reason to expect it in general, and the field spent years surprised by its own noise.

---

## 32.3 Sparse autoencoders

### The one-line idea

If the model packed many sparse features into few dimensions, train an overcomplete autoencoder with a sparsity penalty to unpack them.

### The construction

▸ $$f(x) = \mathrm{ReLU}\big(W_{\text{enc}}(x-b_{\text{dec}}) + b_{\text{enc}}\big)\in\mathbb{R}^{m},\qquad m = 8d\ \text{to}\ 256d$$
▸ $$\hat x = W_{\text{dec}}f(x)+b_{\text{dec}}$$
▸ $$\mathcal{L} = \underbrace{\|x-\hat x\|_2^2}_{\text{reconstruction}} + \lambda\underbrace{\|f(x)\|_1}_{\text{sparsity}}$$

**Overcomplete** ($`m\gg d`$) because there are more features than dimensions — that is the entire premise. The $`\ell_1`$ penalty forces most of them off for any given input (typically 10–100 active out of 16k–16M).

**Decoder columns are unit-normalized**, otherwise the model can shrink $`f`$ and grow $`W_{\text{dec}}`$ to cheat the $`\ell_1`$ term.

#### Reading the sparse autoencoder in plain English

Three equations, and they are the same three equations any autoencoder has — encode, decode, penalize. What is strange is only the *shape*.

**Every symbol.**

| Symbol | Read aloud | What it is | Shape |
|---|---|---|---|
| $`x`$ | "x" | An activation vector pulled out of the real model at some layer | $`\mathbb{R}^d`$ |
| $`W_{\text{enc}}`$ | "W-encode" | Rows are the **detectors**: row $`i`$ asks "how much of feature $`i`$ is here?" | $`\mathbb{R}^{m\times d}`$ |
| $`b_{\text{dec}}`$ | "b-decode" | A learned **centre** — subtracted going in, added coming out | $`\mathbb{R}^d`$ |
| $`b_{\text{enc}}`$ | "b-encode" | A learned **threshold** per feature. The noise gate again | $`\mathbb{R}^m`$ |
| $`f(x)`$ | "f of x" | The **feature activations**: $`m`$ numbers, almost all exactly zero | $`\mathbb{R}^m`$ |
| $`W_{\text{dec}}`$ | "W-decode" | Columns are the **dictionary**: column $`i`$ is feature $`i`$'s direction | $`\mathbb{R}^{d\times m}`$ |
| $`\hat x`$ | "x-hat" | The reconstruction, rebuilt from the few active features | $`\mathbb{R}^d`$ |
| $`\lambda`$ | "lambda" | The dial trading reconstruction quality against sparsity | scalar |

**The whole thing in one sentence.** *"Given a $`d`$-number activation, decide which of $`m`$ possible concepts are present and how strongly (that's $`f`$), then rebuild the activation by adding up just those concepts' directions (that's $`\hat x`$) — and pay a fine for every concept you claim is present."*

▸ **The shape is the point, and it is backwards from every other autoencoder you have seen.** A normal autoencoder squeezes $`d`$ down to something smaller to compress. This one **expands** $`d`$ up to $`m = 8d`$ or $`256d`$. It is not compressing; it is **undoing a compression the model already performed.** With $`d = 512`$ and $`m = 8d`$, the dictionary has $`4{,}096`$ entries; at $`m = 256d`$ it has $`131{,}072`$.

> **Analogy.** A chord played on a piano arrives at your ear as a single pressure waveform — one number per instant, everything summed together. A Fourier analysis expands that one number into hundreds of frequency bins, of which only three or four are actually ringing. **The expansion is not adding information; it is separating information that was already there but added up.** The $`\ell_1`$ penalty is the assertion that a real chord has a handful of notes, not four hundred quiet ones.

**Line 1, decoded.** $`f(x) = \mathrm{ReLU}(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}})`$:
1. **Subtract $`b_{\text{dec}}`$** — recentre. Residual-stream activations have a large, boring mean; the dictionary should describe *deviations from it*, not spend an entry on "the usual."
2. **Multiply by $`W_{\text{enc}}`$** — take $`m`$ dot products at once, one per candidate feature (§1.1.1, Reading 3: each row of a matrix is a question asked of the input).
3. **Add $`b_{\text{enc}}`$, then ReLU** — the noise gate from the toy model, feature by feature. Anything below its own threshold is declared crosstalk and set to exactly zero.

**Line 2, decoded.** $`\hat x = W_{\text{dec}} f(x) + b_{\text{dec}}`$ is a **weighted sum of dictionary columns** — Reading 2 of §1.1.1 once more. Make it concrete with $`d = 3`$, $`m = 5`$ and

$$f(x) = (0,\ 2.3,\ 0,\ 0.7,\ 0)$$

Then $`\hat x = 2.3\,W_{\text{dec},2} + 0.7\,W_{\text{dec},4} + b_{\text{dec}}`$. **Only two of the five terms exist.** Read it as a claim: *"this activation is 2.3 units of concept 2 plus 0.7 units of concept 4, and nothing else."* That claim is the interpretability output. Everything else is machinery for producing it.

**$`L_0`$, the number everyone quotes.** $`L_0`$ is how many entries of $`f(x)`$ are nonzero — a *count*, not a length. With $`m = 16{,}384`$ and $`L_0 = 50`$, the code is $`50/16{,}384 = 0.3\%`$ dense. ▸ **A paper reporting SAE quality reports two numbers and you need both: reconstruction error (how much of $`x`$ came back) and $`L_0`$ (how many features it took).** Either alone is meaningless — you can drive error to zero with $`L_0 = m`$, and $`L_0`$ to one with garbage reconstruction.

**Why $`\ell_1`$ produces exact zeros.** $`\|f\|_1 = \sum_i \lvert f_i\rvert`$ (§1.1.4). Its gradient has constant magnitude $`\lambda`$ no matter how small $`f_i`$ gets, so the penalty keeps pushing with full force right up to zero and pins it there. Compare $`\ell_2`$: the gradient of $`f_i^2`$ is $`2f_i`$, which fades to nothing as $`f_i`$ shrinks, so $`\ell_2`$ produces many small values and no zeros. ▸ **$`\ell_1`$ selects; $`\ell_2`$ shrinks.** That distinction is why the sparsity term is $`\ell_1`$ and not $`\ell_2`$.

**The unit-norm rule, and the exact cheat it blocks.** Suppose you multiply dictionary column $`i`$ by $`c > 1`$ and divide $`f_i`$ by $`c`$. The reconstruction $`f_i W_{\text{dec},i}`$ is **completely unchanged** — but the penalty term $`\lambda\lvert f_i\rvert`$ becomes $`\lambda\lvert f_i\rvert / c`$. Take $`c = 100`$ and the sparsity penalty falls by $`100\times`$ for free. Let $`c\to\infty`$ and the penalty vanishes entirely while the model reconstructs exactly as before. ▸ **Constraining every decoder column to length 1 removes the loophole.** This is not fussiness; without it the $`\ell_1`$ term is not a constraint at all.

> **Where this came from.** The method is **sparse dictionary learning**, and it was invented to explain the brain. In 1996, **Bruno Olshausen and David Field** published *Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images* in *Nature*. Their question was neuroscience, not engineering: why does the primary visual cortex contain cells tuned to oriented edges at particular positions and scales? Their answer was that if you take natural photographs and learn an overcomplete dictionary under a sparsity penalty — exactly the loss above — the dictionary entries that emerge **are** oriented edge detectors, matching what physiologists had measured in cat and monkey V1. Sparsity alone predicted the architecture of a piece of the visual system. Nearly thirty years later the same loss was pointed at transformer activations, by two groups within weeks of each other in late 2023: **Cunningham, Ewart, Riggs, Huben and Sharkey**, and **Anthropic's** *Towards Monosemanticity*. The technique that explained why you have edge detectors is now the technique for finding out what a language model is thinking.

### What it found

Anthropic's Claude 3 Sonnet SAE work recovered millions of interpretable features:
- Concrete entities (the Golden Gate Bridge, specific people, code functions).
- Abstract concepts (deception, sycophancy, inner conflict, security vulnerabilities).
- Multilingual and multimodal features — the same feature fires for a concept in English text, Chinese text, and an image.

▸ **The features are causal, not merely correlational.** Clamping the Golden Gate Bridge feature high made the model identify *as* the bridge across every context. **This is the crucial evidence**: an interpretability method that only produced correlations would be a curiosity, but steering demonstrates the direction is used by the computation.

#### Why "causal, not correlational" is the whole argument

**"Clamping" means:** run the model normally, but at one layer force $`f_i`$ — one entry of the sparse code — to a fixed large value, rebuild $`\hat x`$ from the modified code, and put it back into the residual stream. You have not retrained anything and you have not touched the prompt. You reached in and turned one dial.

The distinction being drawn is the oldest one in empirical science, and it is worth stating precisely:

| Claim | Evidence needed | What it would look like here |
|---|---|---|
| **Correlational** | "when the model reads about bridges, direction $`v`$ lights up" | $`v`$ might be a side-effect the model never consults — a warm exhaust pipe, not the engine |
| **Causal** | "when I *set* direction $`v`$, the behaviour changes as predicted" | the model starts calling itself a bridge |

▸ **Only the second kind of evidence distinguishes an explanation from a coincidence,** and it is available here only because features are directions you can *add to* a stream, not opaque nonlinear tangles.

> **Analogy.** You notice a wire in a car that gets hot whenever the headlights are on. That is a correlation, and it is consistent with the wire *powering* the lights, *reporting* on them, or merely running alongside the wire that does. Cut it. If the headlights go out, you know. **Interpretability without intervention is looking at wires. Interpretability with intervention is cutting them.**

> **The story behind Golden Gate Claude.** In May 2024, Anthropic published *Scaling Monosemanticity*, which trained sparse autoencoders on a production model — Claude 3 Sonnet — with dictionaries of roughly one million, four million, and thirty-four million features. To demonstrate that the recovered features were causal rather than decorative, they clamped a single feature, the one that fired on mentions and images of the Golden Gate Bridge, to a high value and put the resulting model on the public internet for a few days. **Golden Gate Claude** would steer any conversation — a recipe, a coding question, a request for relationship advice — back to the bridge, and when asked about itself would answer that it *was* the bridge. The demo was widely treated as a joke, and it was funny, but the scientific content is serious and rare: a public, reproducible causal intervention on a single interpretable direction inside a deployed model. **The joke is the experiment.**

### The known problems

- ▸ **Shrinkage.** The $`\ell_1`$ penalty biases *all* activations toward zero, systematically underestimating magnitudes even for correctly identified features. **JumpReLU** and **TopK** SAEs address this — TopK enforces exactly $`k`$ active features and drops $`\ell_1`$ entirely, removing the bias.
- **Dead features:** a large fraction never activate. Mitigations: resampling, auxiliary losses on dead features, careful init.
- **Feature splitting:** widen the SAE and one feature splits into many finer ones. ▸ **There is no canonical granularity** — this is arguably the deepest conceptual problem with SAEs, because it suggests "the" feature set may not be well-defined.
- **Evaluation is  hard.** Reconstruction loss and $`L_0`$ are proxies; automated interpretability scores use an LLM to name and predict feature activations, which is circular in an uncomfortable way. Recent work has found SAEs underperform simpler baselines on some downstream tasks, and the field is actively debating how much they have delivered.

**Related methods:** transcoders (approximate an MLP layer sparsely, giving circuits directly), crosscoders (share features across layers or models), and attribution-based dictionary learning.

#### Shrinkage, derived in four lines

Shrinkage sounds like a vague complaint. It is not — you can compute exactly how much magnitude the $`\ell_1`$ penalty steals, and the answer is embarrassing.

Suppose feature $`i`$ is  present with true strength $`1`$, and the dictionary column is a perfect unit vector, so the reconstruction would be exact at $`f_i = 1`$. Now let the model output $`f_i = 1 - \delta`$ instead and see what the loss does.

- **Reconstruction term.** You are short by $`\delta`$ along a unit direction, so the squared error grows by $`\delta^2`$.
- **Sparsity term.** You claimed $`\delta`$ less, so the $`\ell_1`$ penalty falls by $`\lambda\delta`$.
- **Net change.** $`\Delta\mathcal{L}(\delta) = \delta^2 - \lambda\delta`$.
- **Minimize.** $`\dfrac{d}{d\delta}(\delta^2 - \lambda\delta) = 2\delta - \lambda = 0 \;\Rightarrow\; \boxed{\delta = \lambda/2}`$

▸ **The optimal answer is to understate every feature by $`\lambda/2`$, always.** With $`\lambda = 0.2`$, every feature magnitude the SAE reports is low by $`0.1`$ — not sometimes, not for hard cases, but as the *definition* of the optimum. A feature whose true strength is $`0.08`$ is reported as zero and never seen at all.

**Why this is more than an accounting error.** You cannot fix it by turning $`\lambda`$ down, because $`\lambda`$ is what produces the sparsity you came for. You are stuck choosing between "sparse but systematically wrong magnitudes" and "accurate magnitudes but not sparse." ▸ **The penalty that finds the features is the same penalty that mismeasures them.**

**How the fixes work.** **TopK** SAEs delete $`\lambda`$ entirely: keep the $`k`$ largest pre-activations, zero everything else, and impose no penalty on the survivors — so there is no force pulling magnitudes down, and $`L_0 = k`$ exactly, by construction rather than by tuning. **JumpReLU** keeps a penalty but replaces the smooth ReLU with a step: below a learned threshold the output is $`0`$, above it the output is the *unshrunk* value, so crossing the gate costs you a fixed price rather than a per-unit tax.

**A note on the deepest problem.** Feature splitting is worse than it sounds. Widen the dictionary from $`4{,}096`$ to $`16{,}384`$ and a single "legal language" feature can resolve into contract clauses, court citations, statutory language, and legalese-in-fiction. Neither answer is wrong. ▸ **There is no fact of the matter about which is "the" feature set, which means SAE output is a *choice of resolution*, not a discovery of ground truth** — closer to picking a magnification on a microscope than to reading off a parts list.

---

## 32.4 The transformer as a circuit

### The residual stream

▸ **The core framing (Elhage et al., "A Mathematical Framework for Transformer Circuits"): the residual stream is a communication channel that every layer reads from and writes to.**

- It is **linear** — every sub-layer *adds*. So contributions decompose additively and can be attributed.
- It has **finite bandwidth** $`d`$, shared by all $`2L`$ sub-layers. Hence superposition.
- Layers communicate across arbitrary distance by writing to a subspace and having a much later layer read it.
- **Layers do not compose sequentially** in the naive sense; layer 30 can read directly from layer 2's output.

**The logit lens:** apply the final unembedding to an intermediate residual stream to see the model's "current best guess." Predictions refine layer by layer, often converging several layers before the end. The **tuned lens** fits a per-layer affine correction and is substantially more faithful.

#### The residual stream, decoded

**What it is, mechanically.** A transformer layer does not *replace* its input. It writes:

$$x_{\ell+1} = x_\ell + \text{sublayer}_\ell(x_\ell)$$

Unroll that all the way from the embedding to the output and the plus signs never go away:

$$x_L = x_0 + \sum_{\ell=1}^{2L} \text{sublayer}_\ell(\cdot)$$

▸ **Read that aloud: "the final state is the embedding plus the sum of every single thing every sub-layer decided to add."** Nothing is overwritten, nothing is destroyed. That is what "the residual stream is linear" means, and it is the property everything in §32.5 depends on — you can ask "how much of the final answer came from head 7 in layer 3?" and get a definite answer, because addition is decomposable and function composition is not.

> **Analogy.** A shared whiteboard in a long corridor of offices. Nobody may erase; everyone may add. Each office reads whatever is currently on the board, thinks, and writes its contribution in some corner. At the end you read the whole board. **Two consequences follow immediately.** First, office 30 can read what office 2 wrote directly — there is no chain of custody through offices 3 to 29, which is why "layers do not compose sequentially." Second, the board is only so big, so if everyone writes freely they write over each other. **That is superposition, arriving as an engineering constraint rather than a curiosity.**

**Put numbers on the bandwidth.**

| Model | $`d`$ (board width) | Sub-layers sharing it ($`2L`$) | Dimensions per sub-layer if split evenly |
|---|---|---|---|
| GPT-2 small | $`768`$ | $`24`$ | $`32`$ |
| GPT-3 | $`12{,}288`$ | $`192`$ | $`64`$ |

▸ **Sixty-four dimensions each, for every attention block and every MLP in a 96-layer model.** They do not actually partition it — they overlap, in near-orthogonal subspaces, which is exactly the trick of §32.2. **Superposition is not only how features are stored; it is how sub-layers avoid shouting over one another.**

#### What the logit lens is doing

**The unembedding** $`W_U`$ is the final matrix that turns a $`d`$-vector into one score per vocabulary token. Normally it is applied once, at the end. The logit lens applies it **early** — to $`x_5`$, say, in a 12-layer model — and asks: *if the model had to answer right now, what would it say?*

Because of the additive stream, this is a legitimate question rather than a violation. $`x_5`$ lives in the same space as $`x_{12}`$; it is simply an unfinished sum.

▸ **What you see is a prediction sharpening.** Early layers put mass on generically common tokens; middle layers commit to a syntactic category; the correct answer often locks in several layers before the end, after which the remaining layers change little. That last observation is directly useful — it is the empirical basis for early-exit and layer-skipping inference (Ch. 17).

**Why the tuned lens is better.** The lens assumes intermediate states are directly readable by the *final* unembedding, but each layer works in its own slightly rotated and rescaled version of the space. The tuned lens fits a small per-layer affine map $`A_\ell x_\ell + c_\ell`$ first — a learned translator — and produces far more faithful readings, especially in the early layers where the raw lens often outputs nonsense.

> **Where this came from.** The residual-stream framing is from *A Mathematical Framework for Transformer Circuits* (**Elhage, Nanda, Olsson and colleagues, 2021**), which took the deliberate step of studying transformers with **zero or one or two attention layers and no MLPs** — small enough to be solved completely — and only then asking what carries over. The **logit lens** did not come from a paper at all: it was described in a 2020 blog post on the forum LessWrong by a writer under the pseudonym **nostalgebraist**, who noticed that GPT-2's intermediate activations were already readable by the output matrix. It was adopted as standard tooling on the strength of the observation alone. The **tuned lens** (Belrose and colleagues, 2023) supplied the per-layer correction and the evidence that the raw version is unreliable early. It is a small illustration of how young this field is: one of its most-used instruments started as an internet post.

### QK and OV circuits

▸ An attention head factorizes into two independent, low-rank operations:

$$\text{attention pattern} = \mathrm{softmax}\!\left(\frac{x^\top W_Q^\top W_K x}{\sqrt{d_k}}\right)\quad\Rightarrow\quad W_{QK}=W_Q^\top W_K$$
$$\text{output} = \big(\text{pattern}\big)\cdot x\,W_V^\top W_O^\top\quad\Rightarrow\quad W_{OV}=W_OW_V$$

▸ **The QK circuit decides *where* to read; the OV circuit decides *what* to write.** They are separately analyzable, and each is a single low-rank matrix in the residual-stream basis. This factorization is what makes attention heads tractable objects of study at all.

#### The QK/OV factorization, unpacked

An attention head is usually taught as four matrices — $`W_Q`$, $`W_K`$, $`W_V`$, $`W_O`$ — and that framing makes it look like a four-part machine. It is a **two**-part machine, and the two parts never interact.

**Every symbol.**

| Symbol | Read aloud | Job |
|---|---|---|
| $`W_Q`$ | "W-query" | Turns a token's state into a *question* |
| $`W_K`$ | "W-key" | Turns a token's state into an *advertisement* |
| $`W_V`$ | "W-value" | Extracts the *payload* a token offers |
| $`W_O`$ | "W-output" | Decides where in the stream that payload gets written |
| $`d_k`$ | "d-k" | Width of the query/key space per head — $`64`$ in GPT-2 small |
| $`\sqrt{d_k}`$ | "root d-k" | The scaling that stops dot products from growing with $`d_k`$ and saturating the softmax |
| $`\mathrm{softmax}`$ | "soft-max" | Turns raw scores into weights that are positive and sum to 1 (§1.3.4) |

**Why $`W_Q`$ and $`W_K`$ collapse into one matrix.** The attention score between token $`i`$ and token $`j`$ is $`(W_Q x_i)^\top (W_K x_j)`$, which regroups as $`x_i^\top (W_Q^\top W_K)\, x_j`$ — the queries and keys never appear except *through their product*. Define $`W_{QK} = W_Q^\top W_K`$ and the head's routing behaviour is one matrix.

▸ **Consequence, and it is stronger than it first sounds: $`W_Q`$ and $`W_K`$ are individually meaningless.** Replace $`W_Q`$ by $`MW_Q`$ and $`W_K`$ by $`M^{-\top}W_K`$ for any invertible $`M`$ and the product is unchanged, so the model behaves identically. **Any story you tell about "what the query matrix represents" is a story about an arbitrary choice of basis.** Only $`W_{QK}`$ is real. The same argument applies to $`W_{OV} = W_O W_V`$.

**Put shapes and counts on it** (GPT-2 small: $`d = 768`$, 12 heads, $`d_k = 64`$):

| Object | Shape | Rank | Free parameters |
|---|---|---|---|
| $`W_Q`$, $`W_K`$ separately | $`64\times 768`$ each | — | $`2\times 49{,}152 = 98{,}304`$ |
| $`W_{QK} = W_Q^\top W_K`$ | $`768\times 768`$ | **at most 64** | still $`98{,}304`$ |
| A general $`768\times768`$ matrix | $`768\times 768`$ | up to $`768`$ | $`589{,}824`$ |

▸ **A head's routing rule is a $`768\times768`$ table described by only 98,304 numbers, and it can only "see" a 64-dimensional slice of the stream.** That $`64/768 \approx 8\%`$ is a severe restriction — and it is a gift to the interpreter, because a rank-64 object in a 768-dimensional space is a small, analyzable thing. **Low rank is why attention heads can be understood at all;** a full-rank head would be as opaque as an MLP.

> **Analogy.** A hotel switchboard. The **QK circuit** is the routing table: given who is calling, which room does the call go to? The **OV circuit** is the message: whatever gets delivered to that room, in what form. Change the routing table and calls go elsewhere; change the message and the same rooms hear something different. The two are wired independently, which is why you can hold one fixed and vary the other — and that is precisely how induction heads are diagnosed below.

**Reading each formula aloud, once.**

- $`\mathrm{softmax}\!\big(x^\top W_Q^\top W_K x/\sqrt{d_k}\big)`$: *"score every earlier position by how well its advertisement answers my question, divide by root-$`d_k`$ so the scores don't blow up, and normalize into a set of weights that sum to one."*
- $`(\text{pattern})\cdot x\,W_V^\top W_O^\top`$: *"take a weighted average of the payloads at those positions, and add it into my own stream."*

**A worked micro-case.** Set $`d_k = 4`$ and suppose $`x_i^\top W_{QK} x_j`$ evaluates to $`(2.0,\ 8.0,\ 1.0)`$ for three candidate positions. Divide by $`\sqrt{4} = 2`$: $`(1.0,\ 4.0,\ 0.5)`$. Exponentiate: $`(2.72,\ 54.60,\ 1.65)`$, sum $`58.97`$. Softmax: $`(0.046,\ 0.926,\ 0.028)`$. ▸ **92.6% of the head's attention goes to position 2** — an attention pattern is not a soft blur but usually a near-decision, which is why "this head attends to the previous token" is a statement you can check by eye.

### Induction heads — the fully worked example

**The behaviour:** given `... [A][B] ... [A]`, predict `[B]`. In-context copying.

**The circuit** requires two heads composing across layers:
1. **Previous-token head** (early layer): attends from position $`t`$ to $`t-1`$ and writes information about token $`t-1`$ into position $`t`$'s residual stream.
2. **Induction head** (later layer): its QK circuit uses the *current* token as query and matches against that written information — locating earlier positions whose *predecessor* was the current token. Its OV circuit then copies the token at that position to the output.

▸ **Why two layers are required:** the induction head's key must depend on the *previous* token, and no single attention layer can produce that. This is **K-composition** — one head writing into the subspace another head reads as keys.

#### The induction circuit, traced position by position

The notation `... [A][B] ... [A]` is compact to the point of hiding the mechanism. Run a real string through it.

> **Sentence:** `Mr Dursley was proud to say that he was normal . Mr ___`

Number the positions: $`1 = `$ `Mr`, $`2 = `$ `Dursley`, …, $`12 = `$ `Mr`. The model must produce `Dursley` at position 13. It has never been trained on this name in particular; it must copy it from earlier in the *context*.

**Step 1 — the previous-token head, layer 0.** At every position $`t`$, this head attends almost entirely to $`t-1`$ and copies information about that token into position $`t`$'s stream. At position 2, the residual stream now says, roughly:

> *"I am the token `Dursley`, and the token before me was `Mr`."*

That second clause is new information. Position 2's embedding never contained it; a head put it there.

**Step 2 — the induction head, layer 1, at position 12.** Its **QK circuit** builds a query from the current token, `Mr`, meaning approximately *"find me positions whose predecessor was `Mr`."* It compares that query against every earlier position's key. Position 2's key was built from the tag the layer-0 head wrote, so it advertises *"my predecessor was `Mr`"* — a match. Nothing else in the sentence matches. Attention concentrates on position 2.

**Step 3 — the same head's OV circuit.** It reads position 2's value, which encodes the token `Dursley`, and writes it into position 12's residual stream in a direction that the unembedding reads as *"the next token is `Dursley`."* Logit for `Dursley`: up.

▸ **In one sentence: find where this token appeared before, look at what came next, and say that.** It is a two-instruction program, and it is the mechanical origin of a large part of what gets called in-context learning.

#### Why one layer provably cannot do it

Attention keys at position $`j`$ are computed as $`W_K x_j`$, and before any attention has run, $`x_j`$ contains **only** token $`j`$'s embedding plus its positional encoding. So a first-layer key can advertise *"I am `Dursley`"* or *"I am at position 2"* — but there is no way for it to advertise *"the token before me was `Mr`"*, because that information is not in $`x_j`$ yet.

▸ **The fix is that a previous head has to write it there first.** Naming the three ways one head can feed another is worth memorizing, because circuit papers use these terms constantly:

| Name | Head A writes into the subspace head B reads as its... | Effect |
|---|---|---|
| **Q-composition** | queries | changes *what B is asking for* |
| **K-composition** | keys | changes *what earlier positions advertise* |
| **V-composition** | values | changes *what gets copied* |

The induction circuit is **K-composition**: the previous-token head edits the advertisements, not the questions. That is why two layers are the minimum and why a one-layer transformer, however wide, cannot do in-context copying.

> **Analogy.** A library with no catalogue. Ask "which book follows *Dune* on the shelf?" and you cannot answer from the spine of any single book — the information is not printed anywhere. Send an assistant down the aisle first to write, on each book's cover, the title of the book to its left. **Now** the question is answerable by one pass of scanning covers. The assistant is the previous-token head; the scan is the induction head; the covers are the residual stream.

**The evidence chain, which is the strongest in the field:**
- Induction heads form in a narrow window during training, visible as a bump in the loss curve.
- In-context learning ability jumps in the *same* window.
- Ablating induction heads destroys in-context learning.
- They appear in every transformer examined, at every scale — a strong case for universality.

#### Why that evidence chain is the strongest in the field

Each bullet on its own would be weak. Stacked, they close off the alternatives — which is what an argument has to do to count as evidence.

| Observation | Alternative it kills |
|---|---|
| Heads form in a narrow training window | "induction heads are an analyst's construct" — no, they have a birthday |
| In-context learning jumps in the **same** window | "the timing is coincidence" — two independent measurements line up |
| Ablating them destroys in-context learning | "they correlate with it but aren't used" — the causal test from §32.5 |
| They appear at every scale examined | "it's an artefact of small models" |

▸ **Notice the shape: a *timing* coincidence plus a *causal* intervention plus a *replication*.** That is the standard structure of evidence in an experimental science, and it is rare in interpretability because the causal step is expensive and the replication step requires the phenomenon to be universal.

> **Where this came from.** Induction heads were identified and named in *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021), where they fell out of exhaustively solving two-layer attention-only models — the smallest architecture in which K-composition is possible at all. The follow-up, *In-Context Learning and Induction Heads* (**Olsson and colleagues at Anthropic, 2022**), supplied the timing argument: training curves for these models show a small, visible **bump** early on, a brief window in which the loss deviates from its otherwise smooth descent, and that bump is when induction heads form. In-context learning ability jumps at the same moment. ▸ **A kink in a loss curve that had been dismissed as noise for years turned out to be the moment a specific algorithm assembled itself.** It is the clearest example anyone has of an emergent capability being traced to a concrete mechanism.

### Other identified circuits

**Indirect Object Identification** ("John and Mary went to the store; John gave a drink to ___" → "Mary"): 26 heads in GPT-2 small in 7 functional classes, including **name-mover heads** (copy the name), **S-inhibition heads** (suppress the *subject* name), **duplicate-token heads**, and **backup name-movers** that take over when the primary is ablated. ▸ **That last finding — self-repair — is important and inconvenient: ablation understates a component's importance, because the network compensates.**

**Docstring, greater-than, modular arithmetic** (Ch. 30 §30.3 — completely reverse-engineered as a Fourier-transform circuit), and factual recall circuits localized to mid-layer MLPs, which is what ROME and MEMIT edit.

#### The IOI circuit, and why self-repair is bad news

**Walk the sentence.** `John and Mary went to the store; John gave a drink to ___`. The right answer is `Mary`. Getting it requires three sub-tasks, and the 26 heads sort into classes that do exactly them:

| Head class | What it does | In this sentence |
|---|---|---|
| **Duplicate-token heads** | notice a name has appeared twice | flag that `John` occurs at two positions |
| **S-inhibition heads** | write a *suppression* signal against the repeated name | "whatever you output, not `John`" |
| **Name-mover heads** | copy a name from earlier in the context to the output | copy a name — but `John` is now suppressed, so `Mary` wins |
| **Backup name-movers** | sit idle, and activate only if the primary name-movers are removed | nothing, normally |

▸ **Read the middle two rows again: the circuit does not find the right answer. It finds *both* names and then deletes the wrong one.** Suppression, not selection. That is a  unobvious algorithm, and no amount of staring at attention maps would have suggested it — it took path patching (§32.5) to isolate.

**Now the inconvenient part.** Ablate the name-mover heads and performance should collapse. It partly does not: the **backup** name-movers, which contributed nothing in the intact model, step in and recover much of the behaviour.

▸ **This breaks the standard inference from ablation.** The usual argument is *"I removed component $`X`$ and performance dropped by $`\Delta`$, therefore $`X`$ contributed $`\Delta`$."* Self-repair makes that false in a specific direction: **ablation measures how much a component contributes *given that the network is compensating*, which understates its role.** A head could be doing all the work and show an ablation effect near zero, because a backup silently absorbs its job.

> **Analogy.** Measure an employee's importance by sending them on holiday and seeing whether the company notices. If a colleague quietly covers for them, you conclude they were unnecessary — and you are wrong, and you will find out the hard way when you fire them and the colleague leaves too. **Redundancy defeats the leave-one-out experiment,** which is why §32.5 insists that ablation is the weakest tool in the kit and causal scrubbing is the strongest.

**Where the redundancy comes from is not settled.** Dropout, the general pressure of training on many tasks, and the sheer abundance of heads are all plausible; none is established. It is worth noting it is not unique to IOI — self-repair has since been observed broadly, including in models where nothing like it was expected.

> **Where this came from.** The IOI circuit was mapped in *Interpretability in the Wild* (**Kevin Wang, Jacob Steinhardt and colleagues, 2022**) — the first end-to-end reverse-engineering of a *natural-language* behaviour in a real pretrained model rather than a toy or synthetic task, and the paper that put path patching into general use. The **factual-recall** results came from a different direction: **Kevin Meng, David Bau and colleagues (2022)** used causal tracing to localize where a fact like "the Eiffel Tower is in Paris" is stored, found it concentrated in mid-layer MLP blocks, and then *edited* it with a rank-one weight update — **ROME**, Rank-One Model Editing. Their demonstration was to move the Eiffel Tower to Rome, after which the model would describe the walk from it to the Vatican. **MEMIT** scaled the same operation to thousands of simultaneous edits. ▸ **Model editing is the cleanest possible test of a localization claim: if you say the fact lives *there*, change it *there* and see whether the model changes its mind.**

---

## 32.5 The causal toolkit

▸ **The core principle: correlation is not mechanism. Intervene.**

**Activation patching (causal tracing).** Run the model on a clean input and a corrupted one; copy an activation from one run into the other; measure the effect on the output. If patching component $`X`$ restores the correct behaviour, $`X`$ carries the relevant information.

- **Denoising** (corrupted → patch in clean): finds components *sufficient* to restore behaviour.
- **Noising** (clean → patch in corrupted): finds components *necessary*.
- ▸ These give different answers, and reporting only one is a common error.

**Path patching** restricts the intervention to a specific *edge* — patch the effect of head A on head B's queries only, leaving all other paths intact. This is what isolates a circuit rather than a set of components.

**Attribution patching** approximates patching with a first-order Taylor expansion, making it $`O(1)`$ forward+backward passes instead of $`O(\text{components})`$. Necessary for scaling to large models; less accurate for large effects.

**Causal scrubbing.** The strongest available validation: given a hypothesized circuit, replace every activation the hypothesis claims is irrelevant with a resampled value from a different input. ▸ **If the hypothesis is complete and correct, performance should be preserved.** Any drop measures what your explanation is missing. This turns interpretability claims into falsifiable predictions.

**Ablation** (zero, mean, or resample). ▸ **Mean-ablation is usually better than zero-ablation**, because zero is off-distribution and its effects conflate "this component mattered" with "the model was pushed somewhere weird."

#### Activation patching, worked with numbers

The description is abstract. Do it once, concretely, on the IOI sentence from §32.4.

**Set up two runs.**

| Run | Prompt | Correct answer |
|---|---|---|
| **Clean** | `John and Mary went to the store; John gave a drink to ___` | `Mary` |
| **Corrupted** | `John and Mary went to the store; Mary gave a drink to ___` | `John` |

They differ in **one token**. That is the design principle: change the minimum that flips the answer, so anything you measure is attributable to that change.

**Pick a metric.** Use the **logit difference**, $`\text{logit}(\texttt{Mary}) - \text{logit}(\texttt{John})`$ — a single number, positive when the model prefers `Mary`. Say it reads $`+3.5`$ on the clean run and $`-3.5`$ on the corrupted run.

**Patch.** Run the corrupted prompt, but at one chosen point — layer 9, head 9, at the final token position — **overwrite** that activation with the value it took in the clean run. Let everything else proceed normally. Suppose the metric now reads $`+2.8`$.

**Score it.** Recovery is the fraction of the gap closed:

$$\frac{2.8 - (-3.5)}{3.5 - (-3.5)} = \frac{6.3}{7.0} = 0.90$$

▸ **Ninety per cent of the model's ability to answer correctly was restored by copying one head's output at one position.** That head is carrying the relevant information, and you now have a number rather than an impression.

**Denoising versus noising, and why the answers differ.**

| Direction | You run... | You patch in... | You learn |
|---|---|---|---|
| **Denoising** | the corrupted prompt | clean activations | which components are **sufficient** to restore the behaviour |
| **Noising** | the clean prompt | corrupted activations | which components are **necessary** for it |

▸ **These come apart exactly when there is redundancy.** If two heads both carry the answer, patching either one in is *sufficient* (denoising lights up both), but breaking either one alone is not *necessary* (noising lights up neither). Report only denoising and you overcount; report only noising and you miss redundant components entirely — which, given self-repair, is the more dangerous error.

> **Analogy.** Two ways to test whether a fuse matters. **Denoising:** the appliance is dead; swap in one known-good fuse at a time and see what revives it. **Noising:** the appliance works; pull one fuse at a time and see what dies. In a circuit with a backup fuse, the first test finds both and the second finds neither. **They are different questions, and the wire diagram is only settled when both agree.**

#### Path patching versus component patching

Activation patching asks *"does component $`X`$ matter?"* Path patching asks *"does the connection **from** $`X`$ **to** $`Y`$ matter?"* — you patch head A's contribution **only where it enters head B's queries**, leaving A's effect on everything else at its original value.

▸ **That is the difference between a list of parts and a wiring diagram.** IOI's S-inhibition heads are a case in point: they matter *only* through their effect on the name-movers' queries. Component patching says "these heads matter"; path patching says "these heads matter *by telling those heads what not to look at*," which is the actual algorithm.

#### The cost problem, in numbers

| Model | Heads | Passes to patch every head at every one of ~15 positions | Passes to path-patch every head **pair** |
|---|---|---|---|
| GPT-2 small (117M) | $`12\times12 = 144`$ | $`\approx 2{,}160`$ | $`144^2 = 20{,}736`$ edges, before positions |
| A 70B model | $`80\times 64 = 5{,}120`$ | $`\approx 77{,}000`$ | $`\approx 26`$ million edges |

▸ **This is why attribution patching exists.** Instead of actually running each patch, approximate the effect with a first-order Taylor expansion: the change in the metric $`\approx`$ (gradient of the metric with respect to that activation) $`\cdot`$ (how much the activation would change). One forward pass and one backward pass give you an estimate for **every component at once**, turning $`O(\text{components})`$ into $`O(1)`$.

The catch is the one every linearization has: a first-order approximation is accurate for *small* effects and unreliable for large ones — precisely the effects you care most about. ▸ **The standard practice is therefore two-stage: attribution-patch everything to find candidates cheaply, then actually patch the survivors to confirm.** Screen wide and cheap, verify narrow and exact.

#### Causal scrubbing, and why it is the strongest test

The others test a component. Causal scrubbing tests **your whole explanation at once**, and it inverts the usual burden of proof.

1. Write down your hypothesis as a claim about which activations carry which information, and which are irrelevant.
2. For every activation the hypothesis says is **irrelevant**, replace it with the value it took on a *different* input — one your hypothesis says should be interchangeable.
3. Measure performance.

▸ **If the explanation is complete, the model should still work.** You have replaced everything you claimed did not matter, so if you were right, nothing that mattered changed. Any drop in performance is a direct measurement of **what your explanation left out.**

> **Analogy.** You claim a recipe's flavour comes from the garlic and the lemon, and that the parsley, the pan, and the brand of salt are irrelevant. The strong test is not to remove the garlic — it is to swap out *everything you called irrelevant*, all at once, for other versions, and see whether the dish still tastes right. If it does not, your account of the recipe was incomplete, and the size of the difference tells you by how much.

**Why zero-ablation is the weak option.** Setting an activation to $`0`$ produces a state the model has never encountered — the residual stream at that layer has some typical magnitude and direction, and $`0`$ is nowhere near it. The performance drop then mixes two causes: *"this component carried information"* and *"you handed the model an impossible input."* **Mean-ablation** (substitute the average over the dataset) and **resample-ablation** (substitute the value from another real input) both stay on-distribution, so the drop measures only the first thing. ▸ **Interventions should move the model to a place it could plausibly have been, not off the map.**

> **Where this came from.** The intellectual ancestor is **Judea Pearl's** causal framework, developed from the 1980s onward, whose central move is the **do-operator**: $`p(y \mid \text{do}(x))`$, the distribution of $`y`$ when you *set* $`x`$ by intervention, as opposed to $`p(y\mid x)`$, the distribution when you merely *observe* it. The two differ whenever anything else influences both. Pearl received the Turing Award in 2011 for this work, decades before anyone applied it to network internals. **Activation patching is a do-operation on a computation graph** — the unusual luxury being that in a neural network you can intervene on any node you like, exactly, for free, which is a privilege no epidemiologist or economist has ever had. The specific technique entered interpretability as **causal tracing** in the ROME work (Meng, Bau and colleagues, 2022), and **causal scrubbing** was developed at **Redwood Research** in 2022 as an attempt to answer the question the field kept dodging: *what would it even mean for an interpretability hypothesis to be wrong?*

---

## 32.6 Probing and its confounds

Train a classifier on internal activations to detect a property.

▸ **The confound that invalidates naive probing: a sufficiently expressive probe can find structure the model does not use.** A probe achieving 90% on syntax proves the information is *present*, not that the model *reads* it.

**Controls that make probing meaningful:**
- **Selectivity** (Hewitt & Liang): compare against a probe trained on random control labels. High accuracy with low selectivity means the probe learned the task itself.
- **Keep probes weak** — linear, low-capacity.
- **Amnesic probing:** remove the property (project out its direction) and check whether behaviour changes. This is the causal version.
- ▸ **Always follow a probe with an intervention.** A probe is a hypothesis; steering or patching is the test.

**Steering vectors.** Compute $`v = \bar h_{\text{positive}} - \bar h_{\text{negative}}`$ over contrastive pairs, add $`\alpha v`$ to the residual stream at inference. Works for sentiment, refusal, truthfulness, sycophancy, and style. ▸ **That such a crude method works at all is strong evidence for the linear representation hypothesis** — and SAE features give cleaner, more targeted steering directions.

#### The probing confound, made concrete

**What a linear probe is.** Freeze the model. Collect its internal state $`h\in\mathbb{R}^d`$ for many inputs. Fit a logistic regression $`\hat y = \sigma(w^\top h + c)`$ predicting some property you care about — is this word a noun, is this sentence about medicine, is the speaker angry. If it predicts well, the property is **linearly readable** from $`h`$.

**Why "linearly readable" is weaker than it sounds.** The residual stream at layer 8 has $`d = 768`$ numbers. A probe fitting $`768`$ free parameters on, say, $`2{,}000`$ labelled examples has an enormous amount of freedom. And crucially, the *inputs* to the probe are extremely rich — they encode the token, its position, its neighbours, its syntax, its topic. ▸ **A probe can succeed by assembling the answer out of ingredients the model never combines itself.** Presence is not use.

> **Analogy.** A restaurant's bins contain everything the kitchen touched. A determined investigator can reconstruct the menu from them with high accuracy. That proves the ingredients passed through the building. It does **not** prove the chef cooked the dish you reconstructed — you might have assembled it yourself out of scraps. **A probe reads the bins.**

**Selectivity, with numbers.** Hewitt and Liang's fix is to run the identical probe on a **control task**: same inputs, same probe architecture, but labels assigned *randomly and fixed per word type*. Any accuracy above chance on that is pure memorization capacity, because there is nothing to learn.

Suppose you are probing for part of speech:

| Probe | Real-task accuracy | Control-task accuracy | Selectivity | Verdict |
|---|---|---|---|---|
| Linear, low capacity | $`95\%`$ | $`60\%`$ | $`35`$ | The **model** encodes it |
| Two-layer MLP, high capacity | $`97\%`$ | $`94\%`$ | $`3`$ | The **probe** learned it |

▸ **The MLP has the higher accuracy and the worthless result.** This is the one place in machine learning where you deliberately want the weaker model — a probe's job is to be too feeble to cheat. (These numbers are illustrative; the pattern is the paper's finding.)

**Amnesic probing, with arithmetic.** To remove a direction $`v`$ (unit length) from a state $`h`$, project it out:

$$h' = h - (v^\top h)\,v$$

Take $`d = 2`$, $`v = (0.6,\ 0.8)`$, $`h = (3,\ 1)`$. Then $`v^\top h = 1.8 + 0.8 = 2.6`$, so

$$h' = (3,\ 1) - 2.6(0.6,\ 0.8) = (1.44,\ -1.08)$$

Check: $`v^\top h' = 0.864 - 0.864 = 0`$. **The information along $`v`$ is gone and everything perpendicular to $`v`$ is untouched.** Now run the model forward with $`h'`$ in place of $`h`$: if the behaviour you cared about is unaffected, the model was never using that direction, whatever the probe said. (In practice one iterates — remove a direction, retrain the probe, find it has learned another, remove that too — because a property is rarely confined to a single direction.)

▸ **Always follow a probe with an intervention** is the same principle as §32.5's: a probe is an observation, and observations do not distinguish "used" from "present."

#### Steering vectors, worked

$`\bar h_{\text{positive}}`$ reads *"the average residual stream over the positive examples"* — the bar is an ordinary average here, not the gradient bar of §0.6.

Build one for sentiment. Collect thirty sentences with positive tone and thirty with negative tone, matched as closely as possible. Average each set's activations at one layer, subtract:

$$v = \bar h_{\text{positive}} - \bar h_{\text{negative}}$$

**Everything shared between the two sets cancels.** Both sets are English, both are the same length, both are about similar topics — those components appear in both averages and vanish in the difference. What survives is the axis along which they differ. Now at generation time set $`h \leftarrow h + \alpha v`$.

**A miniature.** With $`d = 2`$, $`v = (0.6,\ 0.8)`$, $`h = (3,\ 1)`$ and $`\alpha = 5`$: the new state is $`(6,\ 5)`$, and the read-out along $`v`$ goes from $`2.6`$ to $`7.6`$. The state has been pushed hard in the "positive" direction and left alone in every direction perpendicular to it.

▸ **$`\alpha`$ is a  dial with a  failure mode at both ends.** Too small and nothing changes. Too large and the state leaves the region the model was trained on: output degrades into repetition or gibberish, the same off-distribution problem that makes zero-ablation unreliable. **The useful range is narrow and found by hand,** which is one of the main practical arguments for SAE-derived directions — they come with a natural scale, since you know what activation values that feature normally takes.

> **Where this came from.** Linear probing entered deep learning through **Guillaume Alain and Yoshua Bengio's** 2016 note *Understanding Intermediate Layers Using Linear Classifier Probes*, whose framing was diagnostic — they wanted to see how information about the label became progressively more linearly available with depth. **Been Kim and colleagues** generalized the idea to arbitrary human concepts with **TCAV** (Testing with Concept Activation Vectors, 2018), whose selling point was that a domain expert could define a concept by supplying examples rather than by writing code. **John Hewitt and Percy Liang's** control-task paper (2019) was the corrective, and it landed hard: a large body of "BERT knows syntax" results had to be re-examined, because nobody had asked how much of the syntax the *probe* was supplying. **Projecting a concept out** of a representation was developed partly for a different purpose — removing gender information from embeddings for fairness reasons (iterative nullspace projection, 2020) — and was then repurposed as a causal test. ▸ **This is the chapter's recurring pattern: a method built to *change* a model gets adopted as a method to *understand* one, because intervention is the only thing that settles an argument.**

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

#### Unfaithful chain-of-thought, and what the experiment actually was

**"Faithful" means:** the stated reasoning is a description of the computation that produced the answer. **"Unfaithful" means:** it is a plausible story generated afterwards, which may or may not correspond to anything.

The cleanest demonstration (Turpin, Michael, Perez and Bowman, 2023) is worth knowing in detail because the design is so simple:

1. Give a model a multiple-choice task with few-shot examples, and **rig the examples so the correct answer is always option (A).**
2. Ask for step-by-step reasoning before the answer.
3. Measure how much the model's answers shift toward (A), and read what the reasoning says.

▸ **The answers shifted. The reasoning never mentioned the pattern.** Instead it produced fluent, on-topic justifications for whichever option the bias had pushed it toward. The model was influenced by a feature of the prompt and then explained itself with reference to something else entirely.

**Why this is a mechanistic-interpretability problem and not a prompting problem.** The obvious hope for oversight is: ask the model to explain itself, then read the explanation. That hope requires the explanation to be *causally downstream of the computation*. This experiment shows it need not be. ▸ **A model's self-report is another output, subject to the same training pressures as every other output — and what training rewards is explanations that look good, which is not the same as explanations that are true.**

> **The unsettling precedent.** This is a very old finding about people. In 1977, **Richard Nisbett and Timothy Wilson** published *Telling More Than We Can Know*, reviewing experiments in which subjects' choices were demonstrably driven by factors they were unaware of — in one, shoppers evaluating identical pairs of stockings strongly favoured the right-most pair, a pure position effect, and when asked why, confidently cited knit, sheerness, and elasticity. Told that position might have mattered, they denied it. The authors' conclusion was that introspective reports are often not read off from the process at all; they are *generated* by the same machinery that generates any other plausible account. ▸ **The parallel is exact and it cuts in an uncomfortable direction: verbal self-report was never reliable evidence about mechanism, in humans or in models, and mechanistic interpretability exists precisely because there is no substitute for looking inside.**

### Why it matters anyway

▸ **Safety:** detecting deception, backdoors, or dangerous capabilities requires looking at the mechanism, not the output — a model that behaves well under evaluation and badly in deployment is undetectable behaviourally by construction.
▸ **Debugging:** knowing *why* a model fails is the difference between fixing it and guessing.
▸ **Editing:** targeted knowledge editing and behaviour steering require knowing where things live.
▸ **Science:** it is the only route from "deep learning works" to "here is why."

#### Examples and non-examples: what counts as evidence about a mechanism

Interpretability's central difficulty is that suggestive-looking evidence is usually not evidence at all. This distinction is the discipline.

**✅  causal evidence**

| Method | Why it establishes something |
|---|---|
| **Activation patching** — swap a component's activation from a corrupted run into a clean one | An **intervention**: you changed one thing and measured the effect |
| **Ablation** with a behaviour change | Removing it breaks the behaviour |
| **Path patching** — isolating which route carries the effect | Localizes the causal pathway, not just the correlate |
| A steering vector that reliably changes outputs | The representation is doing work, not merely present |

**❌ Near-misses — look like explanations, but aren't**

| Looks explanatory | Why it isn't | What it actually shows |
|---|---|---|
| "Attention weights show the model looks at the subject" | Attention weights are **not** explanations — high weight ≠ causal importance | Where information *could* flow |
| A probe reaching 95% accuracy on a concept | A strong probe can extract features the model never **uses** | The information is *present*, not that it's *used* |
| A neuron that fires on dogs | Polysemanticity — the same neuron also fires on three unrelated things | A correlation with one of several features |
| A pretty t-SNE of activations by class | Structure exists; nothing causal is established | Local neighbourhood geometry |
| A sparse-autoencoder feature with a clean label | The dictionary is one of many that reconstruct equally well | *A* decomposition, not *the* decomposition |

▸ **The boundary: correlation tells you information is present; only intervention tells you it is used.** A probe that reads sentiment off layer 8 proves the information is *there*, which is compatible with the model completely ignoring it. **The whole causal toolkit of §32.5 exists because the correlational version of every one of these methods was tried first and turned out to prove less than it appeared to.**

> **Common misconception.** *"One neuron equals one concept."* This is the assumption superposition demolishes. A network represents far more features than it has neurons by placing them on nearly-orthogonal directions (Chapter 1 §1.1.5 is the geometry that permits it), so a typical neuron is **polysemantic** — it participates in many unrelated features. Finding a neuron that fires on dogs does not mean you found the dog neuron; you found one direction that dogs partly load onto. The belief is tempting because  monosemantic neurons *do* exist and make memorable examples, which biases what gets published and shown.

> **Common misconception.** *"Attention weights explain what the model is doing."* They are seductive because they're free, visual, and look like an explanation. But high attention weight does not imply high causal influence — a head can attend strongly to a token and route none of its information onward, since what matters is the value vector and the downstream circuit, not the attention probability. Several papers around 2019 argued "attention is not explanation," and the argument largely held.

> **Common misconception.** *"Sparse autoencoders recover the model's true features."* An SAE finds *a* sparse overcomplete dictionary that reconstructs the activations. Nothing guarantees uniqueness, and different SAEs — different widths, sparsity penalties, or seeds — yield different feature sets that all reconstruct comparably well. Add dead features and the difficulty of evaluating whether a "feature" is real, and the honest position is that SAEs are a promising decomposition tool whose outputs require independent causal validation.

> **Common misconception.** *"Interpretability has explained how large language models work."* It has produced , verified results — induction heads, the indirect-object-identification circuit, superposition as a real phenomenon — almost entirely on small models and narrow tasks. Scaling those methods to frontier models remains largely unsolved. **The field's own honest self-assessment (§32.7) is considerably more modest than its press coverage**, and being able to state both the achievements and the limits is what distinguishes understanding it from having read about it.

---

## Did you know?

- **DeepDream was a debugging tool that escaped the lab.** Alexander Mordvintsev built the first version at Google in 2015 to see what an image classifier had actually learned. The psychedelic dog-slug images went viral as art, spawned phone apps, and were treated as a novelty for years before the underlying technique — feature visualization — became the foundation of a research programme.

- **The mathematics behind superposition was developed to shorten MRI scans.** Compressed sensing (Candès, Romberg and Tao; independently Donoho; around 2006) proved that a *sparse* signal can be reconstructed from far fewer measurements than its dimension suggests. The practical goal was getting patients out of the scanner faster. The same theorem explains why a language model can store millions of concepts in twelve thousand numbers.

- **A neural network trained on synthetic data rediscovers a physics problem from 1904.** When features are very sparse, the toy model arranges them as tetrahedra and triangular bipyramids — which are exactly the known optimal solutions to the **Thomson problem**, J. J. Thomson's question of how $`N`$ mutually repelling electrons arrange themselves on a sphere. "Spread directions so they interfere least" and "spread charges so they repel least" turn out to be the same optimization.

- **Sparse autoencoders were invented to explain the visual cortex, not to inspect transformers.** Bruno Olshausen and David Field showed in *Nature* in 1996 that learning an overcomplete, sparse dictionary of natural images produces oriented edge detectors matching the cells physiologists had measured in V1. The interpretability tool is a 1996 neuroscience result pointed at a new target.

- **Three completely independent routes arrive at the same filters.** Hubel and Wiesel found orientation-selective cells in cat visual cortex in the late 1950s and 1960s (Nobel Prize, 1981); Olshausen and Field derived them from a sparsity penalty in 1996; and the first layer of AlexNet learned them from ImageNet in 2012 with nobody asking. Biology, optimization theory, and gradient descent converged on oriented edge detectors.

- **One of the field's standard instruments started as a pseudonymous blog post.** The **logit lens** was described in 2020 on the forum LessWrong by a writer using the name *nostalgebraist*, who noticed that GPT-2's intermediate activations were already readable by the final output matrix. It was adopted as tooling on the strength of the observation, with no paper behind it.

- **Anthropic deliberately broke a production model and put it on the internet.** In May 2024, clamping a single sparse-autoencoder feature high produced **Golden Gate Claude**, which redirected every conversation to the bridge and, asked about itself, said it *was* the bridge. It read as a joke and was a public causal experiment on one interpretable direction inside a deployed model.

- **In GPT-3's residual stream, two directions picked at random overlap by about 0.9%.** Random unit vectors in $`\mathbb{R}^d`$ have expected cosine similarity $`0`$ with standard deviation $`1/\sqrt{d}`$, and $`1/\sqrt{12{,}288}\approx 0.009`$. You do not have to design near-perpendicular directions in high dimensions; you get them by accident.

- **The query matrix has no meaning on its own.** Replace $`W_Q`$ with $`MW_Q`$ and $`W_K`$ with $`M^{-\top}W_K`$ for any invertible $`M`$ and the model's behaviour is bit-for-bit identical, because only the product $`W_Q^\top W_K`$ ever appears. Any interpretation of "what the queries represent" is an interpretation of an arbitrary basis choice.

- **A kink in a loss curve turned out to be an algorithm assembling itself.** Small transformers show a brief bump early in training where the loss departs from its smooth descent. It had been treated as noise. It is the window in which induction heads form — and in-context learning ability jumps in the same window.

- **The best-understood language circuit answers by elimination, not by selection.** GPT-2's indirect-object circuit does not find the right name. It finds *both* names, then runs a separate set of heads whose job is to suppress the wrong one. Nobody predicted that architecture from the outside.

- **The unreliability of chain-of-thought was documented in humans in 1977.** Richard Nisbett and Timothy Wilson found shoppers choosing the right-most of four identical pairs of stockings and then confidently explaining the choice in terms of knit and sheerness — denying position mattered when it was suggested. Verbal self-report has never been evidence about mechanism, in people or in models.

- **Model editing's calling card was relocating the Eiffel Tower.** The ROME paper's demonstration was a rank-one weight edit that placed the tower in Rome, after which the model would happily describe the walk from it to the Vatican. Editing a fact where you claim it lives is the sharpest available test of a localization claim.

---

## Check for Understanding

**Networks represent more features than they have dimensions by placing them in nearly-orthogonal directions and relying on sparsity to keep interference rare — which is why individual neurons are polysemantic and why sparse autoencoders can unpack them — and the transformer's linear residual stream makes this analyzable, since every head decomposes into a QK circuit that decides where to read and an OV circuit that decides what to write, with causal interventions rather than correlations being what turns any of it into evidence.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is "a feature is a direction, not a neuron" the load-bearing claim of the whole field?** What breaks if it is false?
2. **Why can a model with 12,288 dimensions represent millions of concepts?** (If your answer uses the word "exponential," say *what* is exponential in *what*.)
3. **Why are both sparsity and a nonlinearity required for superposition, and what exactly does each one do?** Answer in terms of a noise gate.
4. **Why is a polysemantic neuron not a bug?** Explain it as a microphone in a restaurant, without algebra.
5. **Why does a sparse autoencoder make the representation *wider* rather than narrower?** Every other autoencoder you have met does the opposite.
6. **Why does the $`\ell_1`$ penalty systematically understate every feature's magnitude, and why can't you fix it by lowering $`\lambda`$?**
7. **Why is "feature splitting" the deepest problem with sparse autoencoders?** What would it mean for there to be no canonical feature set?
8. **Why does the residual stream being *additive* make attribution possible at all?**
9. **Why does an induction head need two layers?** State the reason in terms of what a first-layer key can possibly know.
10. **What is the difference between denoising and noising, and construct a case where they disagree.**
11. **Why does self-repair make ablation results systematically misleading, and in which direction?**
12. **Why is a probe with 97% accuracy sometimes worse evidence than one with 95%?**
13. **What would it take for an interpretability hypothesis to be *wrong*?** (This is the question causal scrubbing was built to answer.)

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 33 — Calibration, Uncertainty & Robustness](33-calibration-uncertainty-robustness.md)
