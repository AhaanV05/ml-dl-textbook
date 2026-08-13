# Chapter 25 — Self-Supervised & Representation Learning

> **Prerequisites:** Ch. 1 (§1.4 mutual information), Ch. 7, Ch. 10.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, $`\propto`$, or $`\leftarrow`$ are unfamiliar — or if you have ever wondered why $`\sigma`$ seems to mean four different things — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`x^+`$ | "x-plus" | A **positive** — a second view of the *same* thing |
| $`x_j`$ | "x-j" | A **negative** — a view of a *different* thing |
| $`z`$ | "z" | The **embedding**: the vector a network produces for an input |
| $`\mathrm{sim}(a,b)`$ | "similarity" | Usually **cosine similarity** — alignment, ignoring length |
| $`\tau`$ | "tau" | **Temperature** — divides scores before the softmax; small $`\tau`$ = sharper |
| $`N`$ | "N" | How many candidates the model chooses among (batch size, roughly) |
| $`I(x;x^+)`$ | "mutual information" | How much knowing one view tells you about the other |
| $`f`$ | "f" | The **encoder** — the network you actually keep |
| $`g`$ | "g" | The **projection head** — a small network thrown away after training |
| $`\theta_k,\ \theta_q`$ | "theta-k, theta-q" | Weights of the **key** (target) and **query** (online) encoders |
| $`m`$ | "m" | **Momentum** for the EMA update, typically 0.999 |
| $`\leftarrow`$ | "is assigned" | An **update**, i.e. a line of code — not an equation |
| $`\lvert P(i)\rvert`$ | "size of P of i" | How many positives example $`i`$ has |

### Abbreviations used in this chapter

Full glossary in [Chapter 0 §0.13](00-notation-and-math-primer.md).

| Short | Full form |
|---|---|
| BN | Batch Normalization |
| BYOL | Bootstrap Your Own Latent |
| CLIP | Contrastive Language–Image Pre-training |
| DINO | self-**DI**stillation with **NO** labels |
| EMA | Exponential Moving Average |
| InfoNCE | Information Noise-Contrastive Estimation |
| MAE | Masked AutoEncoder |
| MI | Mutual Information |
| MLM / masked LM | Masked Language Model |
| MLP | Multi-Layer Perceptron |
| MoCo | Momentum Contrast |
| SimCLR | Simple framework for Contrastive Learning of visual Representations |
| SSL | Self-Supervised Learning |
| VICReg | Variance–Invariance–Covariance Regularization |
| ViT | Vision Transformer |

---

## 25.1 The premise

### The one-line idea

Labels are scarce and expensive; raw data is abundant and free. Construct a supervised problem *out of the data itself*, solve it, and keep the representation rather than the solution.

### The analogy

Learning a city by being made to give directions between random pairs of places, rather than by memorizing a list of landmark names. Nobody cares about the directions. But you cannot produce them without building an internal map — and the map is what you were actually after.

### The three families

| Family | Signal | Examples |
|---|---|---|
| **Pretext / generative** | reconstruct hidden content | masked LM, MAE, colourization, inpainting |
| **Contrastive** | pull positives together, push negatives apart | SimCLR, MoCo, CLIP |
| **Non-contrastive / distillation** | match a target network's output; avoid collapse structurally | BYOL, SimSiam, DINO, Barlow Twins, VICReg |

▸ **The unifying question every method must answer: what stops the model from outputting a constant?** A constant representation trivially satisfies "similar things are similar." Contrastive methods answer with negatives; non-contrastive methods answer with architectural or statistical constraints. **This is the axis along which to organize the whole field.**

#### What "self-supervised" actually means

The terms in this area are used loosely, so let's pin them down.

| Term | Where the training signal comes from |
|---|---|
| **Supervised** | A human wrote down the answer |
| **Unsupervised** | No target at all — you're finding structure (clustering, density) |
| **Self-supervised** | The target is **generated automatically from the data itself** |

▸ **Self-supervised learning is supervised learning where the labels are free.** You hide part of the data and make the model predict it, or you show it two views of the same thing and make it recognize they match. Nobody labelled anything, but there is still a correct answer to check against — which means ordinary cross-entropy and backpropagation work unchanged. **That's the whole trick, and it is why the field scaled: the bottleneck stopped being human annotation.**

> **Analogy.** Cloze exercises in language class. "The capital of France is ____." Nobody wrote a labelled dataset — the answer was already in the sentence, and the teacher just covered it up. Do this a few billion times over the internet and you have a language model.

**Why the "constant output" question dominates everything.** Suppose the objective is only *"two views of the same image should have similar embeddings."* There is a perfect, useless solution: **output the same vector for every input.** Every pair is then maximally similar, the loss is zero, and the representation contains nothing. This is called **representational collapse.**

▸ **Every method in this chapter is essentially a different answer to "how do I forbid the constant solution?"** Once you see that, the zoo of acronyms organizes itself:

| Method | How it forbids collapse |
|---|---|
| SimCLR, MoCo, CLIP | **Negatives** — things must be *pushed apart*, so they can't all coincide |
| BYOL, SimSiam | **Asymmetry** — a predictor and a stop-gradient break the degenerate fixed point |
| DINO | **Centering + sharpening** of the teacher's output |
| Barlow Twins, VICReg | **Statistical constraints** — force variance and decorrelation explicitly |

#### Examples and non-examples: is it self-supervised?

**✅  self-supervised**

| Example | The free label |
|---|---|
| Next-token prediction | The next token, already in the text |
| Masked language modelling | The word you covered up |
| Masked autoencoder on images | The patches you deleted |
| SimCLR on two crops of one photo | "These came from the same image" |
| CLIP on image–caption pairs | The pairing already present on the web |

**❌ Near-misses — often called self-supervised, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Training on ImageNet labels | Humans wrote those labels | **Supervised** |
| k-means clustering | No prediction target at all | **Unsupervised** |
| Fine-tuning on user thumbs-up data | Humans supplied the signal | Supervised (with human feedback) |
| PCA | Finds structure, predicts nothing | Unsupervised dimensionality reduction |
| Reinforcement learning from a reward function | An external signal, not recovered from data | **Reinforcement learning** |
| Supervised contrastive learning (§25.3) | Uses labels to choose positives | **Supervised**, borrowing a contrastive loss |

▸ **The boundary:** self-supervision requires the target to be **recoverable from the input itself**, with no external annotator. The CLIP case is the interesting edge — captions were written by humans, but not *for the purpose of training*, so the supervision was harvested rather than commissioned. That distinction is economic more than mathematical, and it is exactly why CLIP could scale to 400 million pairs.

> **Common misconception.** *"Self-supervised means the model learns without any supervision."* It is fully supervised in the mechanical sense — there are targets, a loss, and gradients. What is absent is the **human annotator**, not the supervision. The name misleads a great many newcomers into thinking something exotic is happening; nothing is, except where the labels came from.

> **Where the term came from.** The idea long predates the name. Autoencoders date to the 1980s, and Yann LeCun has argued for years that most learning must be self-supervised, using his "cake" analogy — self-supervised learning is the cake, supervised learning the icing, reinforcement learning the cherry, in proportion to how much information each provides per sample. He has also said he regrets the phrase "predictive learning" that he tried earlier, and that "self-supervised" won because it was clearer about the mechanism. The approach became dominant in language first (word2vec in 2013, BERT and GPT later) and took until roughly 2020 to work comparably well in vision, because images have no natural discrete units to mask.

---

## 25.2 Pretext tasks

**Early vision attempts:** predict a patch's relative position, solve a jigsaw of shuffled patches, predict rotation (0/90/180/270°), colourize, inpaint.

▸ **Why they were superseded:** each encodes a narrow, hand-designed invariance, and the model can often shortcut it. Rotation prediction is solved by finding the sky; jigsaw is solved by matching JPEG compression artifacts and edge continuity at patch boundaries. **The model solves the pretext task without learning the semantics you wanted.** This is the general hazard of hand-designed pretext tasks, and it's a good example of specification gaming.

**In language, the pretext task won outright.** Next-token prediction (Ch. 13) and masked LM (§13.6) are pretext tasks that happen to require  understanding, because there is no shortcut to predicting the next word.

---

## 25.3 Contrastive learning and InfoNCE

### The objective

Given an anchor $`x`$, a positive $`x^+`$, and $`N-1`$ negatives:

▸ $$\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp\big(\mathrm{sim}(z,z^+)/\tau\big)}{\sum_{j=1}^{N}\exp\big(\mathrm{sim}(z,z_j)/\tau\big)}$$

**This is cross-entropy over an $`N`$-way classification problem: "which of these $`N`$ items is the true partner?"** Recognizing that immediately demystifies it.

### The mutual-information bound — derive it

**Claim:** $`I(x;x^+)\ \ge\ \log N - \mathcal{L}_{\text{InfoNCE}}`$.

Sketch: the optimal critic for the $`N`$-way classification task is $`f(x,x^+)\propto\frac{p(x^+\mid x)}{p(x^+)}`$ (the density ratio). Substituting the optimal critic into the loss and taking expectations yields
$$\mathcal{L}^{\text{opt}} = -\mathbb{E}\left[\log\frac{\frac{p(x^+|x)}{p(x^+)}}{\frac{p(x^+|x)}{p(x^+)} + \sum_{j\ne +}\frac{p(x_j|x)}{p(x_j)}}\right] \approx \log N - I(x;x^+)$$
where the approximation uses $`\sum_{j\ne+}\frac{p(x_j|x)}{p(x_j)}\approx (N-1)\,\mathbb{E}_{x_j}\!\left[\frac{p(x_j|x)}{p(x_j)}\right] = N-1`$. Rearranging gives the bound. ∎

▸ **Two consequences that are frequently asked about:**
1. **More negatives ⇒ a tighter bound**, which is the theoretical case for large batches.
2. **The bound saturates at $`\log N`$.** With $`N=256`$, $`\log N = 5.5`$ nats — so InfoNCE *cannot certify* more than 5.5 nats of mutual information no matter how good the representation is.

▸ **The honest caveat you should state:** the empirical success of contrastive learning is **not** well explained by the MI bound. Tighter MI estimators give *worse* representations, and the bound is loose exactly where performance is best. The better explanation is §25.6's alignment/uniformity decomposition, plus the specific inductive bias of the augmentations. Saying this distinguishes someone who read the paper from someone who read the abstract.

#### Reading InfoNCE as what it actually is

The formula looks forbidding but the text gives the key away: **it is cross-entropy over a multiple-choice quiz.**

$$\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}{\sum_{j=1}^{N}\exp(\mathrm{sim}(z,z_j)/\tau)}$$

Compare with softmax cross-entropy from §1.3.4: $`-\log\frac{e^{z_y}}{\sum_j e^{z_j}}`$. **They are the same formula.** The only difference is what plays the role of the logit — here it's a similarity score divided by temperature.

So the model is being asked, for each anchor:

> *"Here are $`N`$ candidates. Exactly one is another view of you. Which is it?"*

- **Numerator** — the score for the true partner. Push it up.
- **Denominator** — scores for all candidates including the true one. Push the rest down.
- **$`\tau`$** — the temperature, exactly as in §1.3.4. Small $`\tau`$ makes the softmax sharper.

▸ **Recognizing this collapses the mystery.** There is no exotic new objective — it's the classifier loss you already know, applied to a quiz the data generates for free. A batch of 256 images is a 256-way classification problem constructed at zero labelling cost.

> **Analogy.** A police lineup. The witness (the anchor) must pick their acquaintance from $`N`$ people. Getting it right requires actually knowing what the person looks like — not memorizing a name. And crucially, **the difficulty depends on the lineup**: if the other $`N-1`$ people look nothing alike, the task is trivial and teaches nothing. This is exactly why hard negatives and large $`N`$ matter.

**Reading the mutual-information bound.** The claim $`I(x;x^+)\ \ge\ \log N - \mathcal{L}`$ says: *"solve this quiz well and you have proved your representation captures at least this much shared information between the two views."*

Put numbers on the two consequences:

- More negatives → tighter bound. With $`N=256`$, $`\log 256 = 5.55`$ nats. With $`N=65{,}536`$ (MoCo's queue), $`\log N = 11.1`$ nats.
- **The ceiling is $`\log N`$.** Even a perfect model ($`\mathcal{L}=0`$) certifies only $`\log N`$ nats. With a batch of 256 you *cannot* demonstrate more than 5.55 nats of mutual information, however good the representation truly is.

▸ **This ceiling is the crux of the honest caveat.** Real images share far more than 5.55 nats of information, so the bound is enormously loose in exactly the regime where these methods work best. If mutual information were the mechanism, tighter estimators should give better representations — and experiments found the opposite. **The bound is a valid theorem that turns out not to be the explanation.** Being able to say that cleanly is  the difference between having read the paper and having read the abstract.

#### Examples and non-examples: what makes a good positive pair?

The augmentations *are* the method — they define what the representation is told to ignore.

**✅ Good positive pairs**

| Pair | What invariance it teaches |
|---|---|
| Two random crops of one photo | "Location and scale don't change identity" |
| Original + colour-jittered version | "Lighting and hue don't change identity" |
| An image and its caption (CLIP) | "These two modalities describe one thing" |
| Two augmented views of one audio clip | "Noise and pitch shift preserve content" |

**❌ Near-misses — pairs that look reasonable but break the method**

| Pair | Why it fails |
|---|---|
| Crop alone, no colour jitter | Model cheats by **matching colour histograms** — no semantics learned |
| Two different photos of the same class | Not what the loss assumes; these are the **false negatives** problem |
| Rotation as the only augmentation | Solved by "find the sky." Specification gaming. |
| Views so aggressive they share no content | No signal — the task becomes unanswerable |
| Grayscale + colour, when colour *is* the task | You just taught the model to discard the label |

▸ **The boundary:** a positive pair must preserve what you care about and vary what you don't. **Choosing augmentations is choosing what the representation will be blind to** — which is why "random crop + colour jitter" is not a tuning detail but the central design decision of SimCLR. Use it on a task where colour matters (say, distinguishing ripe from unripe fruit) and you have systematically destroyed the signal.

> **Common misconception.** *"The projection head is just an extra layer for capacity."* It is a **sacrificial layer**. The contrastive loss demands invariance to the augmentations, which means throwing away colour and orientation — but downstream tasks may need those. The head absorbs the invariance requirement so that the encoder beneath it can stay richer, and you **discard the head after training**. Keeping it and using its output costs 10–15% linear-probe accuracy. This is one of the most transferable and least intuitive findings in the field.

> **Common misconception.** *"Bigger batches help because of better gradient estimates."* In contrastive learning the batch **is the task** — every other item in the batch is a negative, so batch size sets the number of choices in the quiz and hence $`\log N`$, the difficulty. That is a qualitatively different reason from ordinary supervised training, and it's why MoCo's queue (decoupling negative count from batch size) was such a useful idea for people without large GPU clusters.

> **Where this came from.** **InfoNCE** was introduced by Aaron van den Oord, Yazhe Li, and Oriol Vinyals in the 2018 paper *Representation Learning with Contrastive Predictive Coding*; "NCE" refers to noise-contrastive estimation, a technique Michael Gutmann and Aapo Hyvärinen developed in 2010 for estimating unnormalized models — the same partition-function dodge that motivated score matching (Ch. 19 §19.6). **SimCLR** came from Ting Chen and colleagues at Google Brain in 2020, and its contribution was largely negative-result clearing: it showed that no exotic architecture was needed, only the right augmentations, a big batch, a projection head, and long training. **MoCo**, from Kaiming He's group at Facebook AI Research, appeared at nearly the same time with a different answer to the batch-size problem. He is also the first author on ResNet (Ch. 8), one of the most-cited papers in all of science.

### SimCLR

The recipe, stripped to essentials:
1. Two augmented views of each image → positives. All other images in the batch → negatives.
2. Encoder $`f`$ (ResNet/ViT) → **projection head** $`g`$ (2-layer MLP) → normalize → InfoNCE on $`z=g(f(x))`$.
3. **Discard $`g`$ after training; use $`f`$'s output.**

▸ **Why the projection head matters so much (a +10–15% linear-probe effect):** the contrastive loss forces $`z`$ to be invariant to the augmentations, which means *discarding* colour, orientation, and crop information. But that information is useful downstream. The projection head absorbs the invariance requirement, leaving $`f`$'s representation richer. **The head is a sacrificial layer.** This is one of the most transferable findings in SSL.

**What actually drives performance, in order:**
1. **Augmentation composition.** Random crop + colour jitter is the critical pair. Crop alone is solved by matching colour histograms — the model cheats. **The augmentations define what the representation is invariant to, which means they define the representation.**
2. Large batch (4096+) for enough negatives.
3. Long training (800+ epochs).
4. The projection head.
5. Temperature.

### Temperature

▸ $`\tau`$ controls the sharpness of the negative-weighting. The gradient with respect to a negative is proportional to its softmax weight, so:
- **Small $`\tau`$ (0.05–0.1):** almost all the gradient goes to the *hardest* negatives. Learns fine-grained separation, but is sensitive to false negatives.
- **Large $`\tau`$ (0.5+):** uniform treatment; tolerant of noise; blurrier representation.

There is a real tension here (Wang & Liu, 2021): hard negatives are the most informative *and* the most likely to be false negatives (a different photo of the same class). Standard $`\tau\approx0.1`$ for images, $`\approx0.02`$–0.05 for text retrieval.

### MoCo

The batch-size problem, solved differently: maintain a **queue** of past encoded keys as negatives (65,536 of them), decoupling the negative count from the batch size.

Since old keys came from an older encoder, they'd be inconsistent — so update the key encoder as a **momentum average**:
▸ $$\theta_k \leftarrow m\theta_k + (1-m)\theta_q,\qquad m=0.999$$

The key encoder changes slowly enough that the queue stays coherent. ▸ **This slow-moving target encoder is the idea that turned out to matter most**, reappearing in BYOL, DINO, and EMA teachers generally.

*(MoCo also needed **ShuffleBN** — shuffling samples across GPUs before BatchNorm — because BN leaks batch statistics that let the model identify the positive without learning anything. A concrete instance of Ch. 7's warning about BN's cross-sample dependency.)*

### Supervised contrastive learning

If labels are available, use all same-class examples as positives:
$$\mathcal{L}=\sum_{i}\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\cdot z_p/\tau)}{\sum_{a\ne i}\exp(z_i\cdot z_a/\tau)}$$
Outperforms cross-entropy on ImageNet and is notably more robust to label noise and corruptions.

---

## 25.4 Non-contrastive methods

### BYOL

**Two networks:** an online network (encoder → projector → **predictor**) and a target network (encoder → projector), where the target's weights are an EMA of the online network's. Loss: minimize the distance between the online *prediction* and the target *projection*.

▸ **No negatives at all.** The obvious question — why doesn't it collapse to a constant? — has a subtle answer with three ingredients:
1. **The predictor asymmetry.** Only the online branch has the extra predictor head, so the two branches are not symmetric and the trivial solution isn't a joint fixed point.
2. **The stop-gradient on the target.** Gradients never flow into the target branch.
3. **The EMA.** The target moves slowly, so the online network is always chasing a lagged version of itself rather than co-collapsing.

Analysis (Tian et al.) shows the predictor plus stop-gradient makes the dynamics approximate an **eigenspace alignment**: the predictor learns to match the correlation structure of the representation, and collapse is an unstable equilibrium rather than an attractor.

*(A famous early claim was that BatchNorm was secretly providing implicit negatives via batch statistics. Follow-up work showed BYOL still works with GroupNorm + weight standardization, so BN is not the mechanism — but the episode is a nice illustration of how hard it is to identify what's actually doing the work.)*

### SimSiam

Removes the EMA entirely: the target is just the online encoder with a stop-gradient.

▸ **The ablation is decisive: remove the stop-gradient and it collapses immediately; remove the predictor and it collapses. Keep both, and it works with batch size 256 and no negatives.** So the essential ingredient is the **predictor + stop-gradient** pair, not the momentum encoder and not the negatives.

### Barlow Twins

▸ Make the **cross-correlation matrix** between the two views' embeddings equal the identity:
$$\mathcal{L} = \underbrace{\sum_i(1-C_{ii})^2}_{\text{invariance}} + \lambda\underbrace{\sum_i\sum_{j\ne i}C_{ij}^2}_{\text{redundancy reduction}},\qquad C_{ij}=\frac{\sum_b z^A_{b,i}z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2}\sqrt{\sum_b (z^B_{b,j})^2}}$$

Diagonal → 1 forces the two views to agree. Off-diagonal → 0 forces **different dimensions to encode different things**, which is what prevents collapse. Information-theoretically motivated by the redundancy-reduction principle from neuroscience.

### VICReg

Three explicit terms: **V**ariance (hinge loss keeping each dimension's std above 1 — a direct anti-collapse term), **I**nvariance (MSE between views), **C**ovariance (off-diagonal decorrelation).

▸ **VICReg is the most pedagogically clear method**, because it names the three requirements explicitly instead of achieving them implicitly. Any SSL method must satisfy all three; VICReg just writes them down. It also needs no batch-level negatives, no EMA, and no predictor — and the two branches need not share weights.

### DINO

Self-**di**stillation with **no** labels: a student matches a momentum teacher's output distribution over $`K`$ prototypes, with the teacher's outputs **centred** (subtract an EMA of the mean, preventing one dimension dominating) and **sharpened** (low temperature, preventing uniform collapse). ▸ **Centering and sharpening are opposing forces, and their balance is what avoids both collapse modes.**

**The famous result:** DINO's ViT attention maps segment objects without any segmentation supervision — the `[CLS]` token's attention delineates object boundaries. This was the first strong evidence that self-supervised ViTs learn semantic structure that supervised ViTs do not.

**DINOv2/v3** scale this with curated data and produce general-purpose visual features that are competitive with or better than supervised ones across dense and global tasks.

### Masked image modelling: MAE

Mask **75%** of patches, encode only the visible 25%, and decode to reconstruct the raw pixels of the masked ones.

▸ **Why the mask ratio is so much higher than BERT's 15%:** images are enormously redundant. At 15% masking, a patch is recoverable by local interpolation — no semantics needed. At 75%, the model must reason about object structure. **Density of information determines mask ratio**, and this is the crisp answer if asked why MAE differs from BERT.

The asymmetric design (encoder sees only visible patches) makes pretraining ~3× faster, since the encoder processes a quarter of the sequence.

▸ **MAE vs contrastive, a useful contrast:** MAE gives weaker *linear probes* but equal or better *fine-tuned* performance. Reconstruction preserves low-level detail that a linear probe cannot exploit but a fine-tuned network can. **This is why comparing SSL methods by linear probe alone is misleading.**

---

## 25.5 Alignment and uniformity — the best explanatory frame

Wang & Isola (2020) decompose the contrastive objective on the unit hypersphere into two asymptotic terms:

▸ $$\mathcal{L}_{\text{align}} = \mathbb{E}_{(x,x^+)}\big[\|f(x)-f(x^+)\|^\alpha\big]$$
▸ $$\mathcal{L}_{\text{uniform}} = \log\ \mathbb{E}_{x,y}\big[e^{-t\|f(x)-f(y)\|^2}\big]$$

**Alignment:** positives should be close. **Uniformity:** the representations should be spread as evenly as possible over the sphere — which maximizes the information they preserve.

▸ **These two metrics predict downstream performance better than the contrastive loss value itself**, and directly optimizing them matches InfoNCE. This is the cleanest available account of what contrastive learning does, and it makes the roles obvious: **negatives exist to enforce uniformity; augmentations define alignment.** Collapse is total failure of uniformity.

It also explains anisotropy in language models (Ch. 10 §10.6): pretrained BERT has good alignment and terrible uniformity, so cosine similarities are compressed — and contrastive fine-tuning fixes exactly the uniformity term.

---

## 25.6 Evaluating representations

| Protocol | What it measures | Caveat |
|---|---|---|
| **Linear probe** | linear separability | favours methods that discard nuisance info; not what you'll actually do |
| **k-NN** | local metric structure | no training, no hyperparameters — a good sanity check |
| **Fine-tuning** | practical value | expensive; conflates representation with optimizability |
| **Few-shot / low-data** | data efficiency | the most informative for real use |
| **Transfer to many tasks** | generality | the right evaluation, and the most expensive |
| **Dense tasks** (segmentation, depth) | spatial structure | global methods can look good and be useless here |

▸ **Rankings disagree across protocols**, and this is not noise — it's the point. Contrastive methods top linear probes; masked modelling tops fine-tuning; DINO-style methods top dense tasks. **Always state which protocol produced a number.**

#### Why BYOL was a shock, and how the collapse is avoided

BYOL removed negatives entirely, which by the reasoning of §25.1 should collapse instantly — nothing is pushing anything apart. It didn't. Understanding why is the most instructive puzzle in this chapter.

**The three ingredients, and what each does:**

| Ingredient | Role |
|---|---|
| **Predictor** (extra network on the online branch only) | Breaks the symmetry between the two branches |
| **Stop-gradient** on the target branch | The target is a *fixed* objective, not something to optimize against |
| **EMA target** ($`\theta_k \leftarrow m\theta_k + (1-m)\theta_q`$) | The target moves slowly, so it never chases the online net into the constant solution |

▸ **The intuition:** the constant solution *is* a fixed point, but the asymmetric architecture makes it **unstable** rather than attractive. The online network is always chasing a slightly stale copy of itself, and the predictor means it must model *how the target differs from itself* — a task with no content if everything is constant. Gradient descent doesn't find the collapse because the path there isn't downhill.

**Reading the EMA update.** $`\theta_k \leftarrow m\theta_k + (1-m)\theta_q`$ with $`m = 0.999`$ means: *"keep 99.9% of the old target, mix in 0.1% of the current online weights."* The arrow is an **assignment**, not an equation (§0.11).

Put a number on it: with $`m=0.999`$, the target's effective memory is about $`1/(1-m) = 1000`$ steps. It reflects roughly where the online network was a thousand steps ago — recent enough to be relevant, stale enough to be a stable target.

> **Analogy.** Learning to sing in tune by matching a recording of yourself from last week. If you matched yourself *live*, you'd drift anywhere and still "match" perfectly — that's collapse. The lag is what makes it a real target.

#### Examples and non-examples: evaluating a representation

**✅ Sound evaluation practice**

| Practice | Why |
|---|---|
| Reporting the protocol alongside the number | Rankings  reverse between protocols |
| Using k-NN as a sanity check | No training, no hyperparameters to fudge |
| Testing on dense tasks if you need spatial structure | Global methods can score well and be useless for segmentation |
| Few-shot evaluation | Closest to why you wanted a representation |

**❌ Near-misses — evaluations that mislead**

| Practice | Why it misleads |
|---|---|
| Linear probe alone | Favours methods that **discard** nuisance information; MAE loses badly and fine-tunes better |
| Comparing across papers with different protocols | The numbers aren't measuring the same thing |
| ImageNet linear probe as a proxy for everything | Doesn't predict dense-task or out-of-domain performance |
| Fine-tuned accuracy alone | Conflates representation quality with how easy the net is to optimize |
| Judging by the SSL loss value | Loss is not comparable across methods with different objectives |

▸ **The boundary:** a representation has no single quality score — it has a **profile** across tasks. The MAE-versus-contrastive case is the clean demonstration: contrastive wins the linear probe, masked modelling wins fine-tuning, and **both facts are real.** Asking "which representation is better?" without naming the downstream use is a malformed question.

> **Common misconception.** *"A higher linear-probe score means a better representation."* Linear probing rewards representations that have already thrown away everything except linearly-separable class information. That is  useful if you plan to freeze the encoder and train a linear layer — and actively misleading if you plan to fine-tune, where the discarded detail turns out to help. **The metric encodes an assumption about how you'll use the model.**

> **The story behind BYOL.** *Bootstrap Your Own Latent* came from DeepMind in 2020 and was greeted with widespread disbelief — removing negatives was supposed to guarantee collapse. A well-known follow-up blog post from a group at Untitled AI reported that BYOL's performance dropped dramatically when batch normalization was removed, suggesting BN was smuggling in an implicit contrastive effect through batch statistics. The DeepMind authors then showed BYOL still worked without batch normalization given appropriate initialization and normalization changes, which weakened that explanation. **SimSiam**, from Kaiming He's group, stripped things down further — no EMA at all, just a predictor and a stop-gradient — and demonstrated the stop-gradient was the essential piece. The episode is a good illustration of how empirical deep learning actually proceeds: a surprising result, a plausible debunking, a rebuttal, and a simplification that isolates the real mechanism.

---

## Did you know?

- **Self-supervised learning is supervised learning where nobody had to write the labels.** Mechanically there are targets, a loss, and gradients — exactly as in supervised learning. What's missing is the human annotator, not the supervision. The name misleads a great many newcomers.

- **The projection head in SimCLR is thrown away after training**, and keeping it costs 10–15% accuracy. It exists to absorb the "be invariant to augmentations" demand so the encoder underneath doesn't have to obey it — one of the least intuitive and most transferable findings in the field.

- **MAE masks 75% of an image; BERT masks 15% of a sentence.** The difference is information density. At 15% masking an image patch can be recovered by interpolating its neighbours, so no understanding is required. Words carry far more information each, so 15% is already hard.

- **Early vision pretext tasks were defeated by shortcuts.** Rotation prediction was solved by locating the sky. Jigsaw puzzles were solved by matching JPEG compression artifacts along patch edges. The model solved the stated task without learning anything about objects — a textbook case of specification gaming.

- **DINO's attention maps segment objects with no segmentation labels at all.** The `[CLS]` token's attention traces object boundaries. This was the first strong evidence that self-supervised vision transformers learn semantic structure that *supervised* ones do not.

- **MoCo needed "ShuffleBN" because batch normalization was leaking the answer.** BatchNorm's cross-sample statistics let the model identify which item was the positive without learning any semantics — a concrete instance of the cross-sample dependency warned about in Chapter 7.

- **BYOL was widely believed to be impossible when it appeared**, since removing negatives should permit instant collapse. The subsequent debate over whether batch normalization was secretly providing the contrast, and SimSiam's demonstration that the stop-gradient was the key, is one of the better-documented mechanism hunts in recent deep learning.

- **VICReg simply writes down the three requirements every SSL method must satisfy** — variance, invariance, covariance — instead of achieving them implicitly through architecture tricks. This makes it the clearest method to learn from even where it isn't the strongest.

- **The InfoNCE mutual-information bound caps out at $`\log N`$.** With a batch of 256 you cannot certify more than 5.55 nats of mutual information no matter how good your representation is — and real image pairs share far more than that. The bound is a valid theorem that turns out not to explain why the method works.

- **Tighter mutual-information estimators produce *worse* representations.** If maximizing mutual information were the mechanism, this should be impossible. It is one of the clearer cases in modern machine learning of a beautiful theoretical justification being, empirically, the wrong explanation.

- **Yann LeCun's "cake" analogy** casts self-supervised learning as the cake, supervised learning as the icing, and reinforcement learning as the cherry — ranked by how many bits of information each provides per sample.

- **Choosing augmentations is choosing what your model will be blind to.** SimCLR's crop-plus-colour-jitter recipe teaches invariance to colour — which is exactly wrong if your task is distinguishing ripe fruit from unripe. The augmentation pipeline is not a preprocessing detail; it is the core modelling decision.

---

## Check for Understanding

---

## 25.7 Where this ended up

- **Language:** self-supervision *is* the paradigm; there is no supervised alternative at scale.
- **Vision:** DINOv2/v3-class features are used off-the-shelf for retrieval, depth, segmentation, and as VLM encoders (Ch. 28). MAE-style pretraining is standard for fine-tuning.
- **Multimodal:** CLIP-style contrastive learning across modalities (Ch. 28 §28.3) is the dominant approach — the "augmentation" is simply the other modality, which is a far more semantic and less hand-designed invariance than any crop or jitter.
- **Audio, video, graphs, molecules, time series, tabular:** the same three families reappear with domain-specific augmentations, and choosing the augmentations remains the hard part.

---

## Check for Understanding

**Every self-supervised method must simultaneously make augmented views agree and prevent the representation from collapsing to a constant — contrastive methods use negatives to enforce uniformity on the hypersphere, non-contrastive methods use a predictor with a stop-gradient or explicit variance and decorrelation terms instead — and because the invariances are imposed by the augmentations, choosing the augmentations is the real modelling decision.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What makes learning "self-supervised" rather than supervised or unsupervised?** Where do the labels come from?
2. **What is representational collapse, and why is "output a constant" a perfect score on a naive objective?**
3. **How do contrastive and non-contrastive methods each prevent collapse?** (One sentence per family.)
4. **Why is InfoNCE just cross-entropy?** What is the multiple-choice quiz being posed? (The police lineup.)
5. **Why does the mutual-information bound cap at $`\log N`$, and why is that a problem for the theory?**
6. **Why do tighter mutual-information estimators give worse representations, and what does that tell you about the MI explanation?**
7. **Why is the projection head thrown away, and why does keeping it hurt?**
8. **Why does SimCLR need colour jitter as well as cropping?** What does the model do if you only crop?
9. **Why does MAE mask 75% when BERT masks 15%?**
10. **Why was BYOL surprising, and what stops it from collapsing without negatives?** (The singing-along-to-last-week's-recording analogy.)
11. **Why can a method win on linear probing and lose on fine-tuning?** Which would you trust for your use case?
12. **Why is "choosing the augmentations" the same as "choosing the representation"?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 26 — Reinforcement Learning Foundations](26-rl-foundations.md)
