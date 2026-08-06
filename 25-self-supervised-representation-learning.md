# Chapter 25 — Self-Supervised & Representation Learning

> **Prerequisites:** Ch. 1 (§1.4 mutual information), Ch. 7, Ch. 10.

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

---

## 25.2 Pretext tasks

**Early vision attempts:** predict a patch's relative position, solve a jigsaw of shuffled patches, predict rotation (0/90/180/270°), colourize, inpaint.

▸ **Why they were superseded:** each encodes a narrow, hand-designed invariance, and the model can often shortcut it. Rotation prediction is solved by finding the sky; jigsaw is solved by matching JPEG compression artifacts and edge continuity at patch boundaries. **The model solves the pretext task without learning the semantics you wanted.** This is the general hazard of hand-designed pretext tasks, and it's a good example of specification gaming.

**In language, the pretext task won outright.** Next-token prediction (Ch. 13) and masked LM (§13.6) are pretext tasks that happen to require genuine understanding, because there is no shortcut to predicting the next word.

---

## 25.3 Contrastive learning and InfoNCE

### The objective

Given an anchor $x$, a positive $x^+$, and $N-1$ negatives:

▸ $$\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp\big(\mathrm{sim}(z,z^+)/\tau\big)}{\sum_{j=1}^{N}\exp\big(\mathrm{sim}(z,z_j)/\tau\big)}$$

**This is cross-entropy over an $N$-way classification problem: "which of these $N$ items is the true partner?"** Recognizing that immediately demystifies it.

### The mutual-information bound — derive it

**Claim:** $I(x;x^+)\ \ge\ \log N - \mathcal{L}_{\text{InfoNCE}}$.

Sketch: the optimal critic for the $N$-way classification task is $f(x,x^+)\propto\frac{p(x^+\mid x)}{p(x^+)}$ (the density ratio). Substituting the optimal critic into the loss and taking expectations yields
$$\mathcal{L}^{\text{opt}} = -\mathbb{E}\left[\log\frac{\frac{p(x^+|x)}{p(x^+)}}{\frac{p(x^+|x)}{p(x^+)} + \sum_{j\ne +}\frac{p(x_j|x)}{p(x_j)}}\right] \approx \log N - I(x;x^+)$$
where the approximation uses $\sum_{j\ne+}\frac{p(x_j|x)}{p(x_j)}\approx (N-1)\,\mathbb{E}_{x_j}\!\left[\frac{p(x_j|x)}{p(x_j)}\right] = N-1$. Rearranging gives the bound. ∎

▸ **Two consequences that are frequently asked about:**
1. **More negatives ⇒ a tighter bound**, which is the theoretical case for large batches.
2. **The bound saturates at $\log N$.** With $N=256$, $\log N = 5.5$ nats — so InfoNCE *cannot certify* more than 5.5 nats of mutual information no matter how good the representation is.

▸ **The honest caveat you should state:** the empirical success of contrastive learning is **not** well explained by the MI bound. Tighter MI estimators give *worse* representations, and the bound is loose exactly where performance is best. The better explanation is §25.6's alignment/uniformity decomposition, plus the specific inductive bias of the augmentations. Saying this distinguishes someone who read the paper from someone who read the abstract.

### SimCLR

The recipe, stripped to essentials:
1. Two augmented views of each image → positives. All other images in the batch → negatives.
2. Encoder $f$ (ResNet/ViT) → **projection head** $g$ (2-layer MLP) → normalize → InfoNCE on $z=g(f(x))$.
3. **Discard $g$ after training; use $f$'s output.**

▸ **Why the projection head matters so much (a +10–15% linear-probe effect):** the contrastive loss forces $z$ to be invariant to the augmentations, which means *discarding* colour, orientation, and crop information. But that information is useful downstream. The projection head absorbs the invariance requirement, leaving $f$'s representation richer. **The head is a sacrificial layer.** This is one of the most transferable findings in SSL.

**What actually drives performance, in order:**
1. **Augmentation composition.** Random crop + colour jitter is the critical pair. Crop alone is solved by matching colour histograms — the model cheats. **The augmentations define what the representation is invariant to, which means they define the representation.**
2. Large batch (4096+) for enough negatives.
3. Long training (800+ epochs).
4. The projection head.
5. Temperature.

### Temperature

▸ $\tau$ controls the sharpness of the negative-weighting. The gradient with respect to a negative is proportional to its softmax weight, so:
- **Small $\tau$ (0.05–0.1):** almost all the gradient goes to the *hardest* negatives. Learns fine-grained separation, but is sensitive to false negatives.
- **Large $\tau$ (0.5+):** uniform treatment; tolerant of noise; blurrier representation.

There is a real tension here (Wang & Liu, 2021): hard negatives are the most informative *and* the most likely to be false negatives (a different photo of the same class). Standard $\tau\approx0.1$ for images, $\approx0.02$–0.05 for text retrieval.

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

Self-**di**stillation with **no** labels: a student matches a momentum teacher's output distribution over $K$ prototypes, with the teacher's outputs **centred** (subtract an EMA of the mean, preventing one dimension dominating) and **sharpened** (low temperature, preventing uniform collapse). ▸ **Centering and sharpening are opposing forces, and their balance is what avoids both collapse modes.**

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

---

## 25.7 Where this ended up

- **Language:** self-supervision *is* the paradigm; there is no supervised alternative at scale.
- **Vision:** DINOv2/v3-class features are used off-the-shelf for retrieval, depth, segmentation, and as VLM encoders (Ch. 28). MAE-style pretraining is standard for fine-tuning.
- **Multimodal:** CLIP-style contrastive learning across modalities (Ch. 28 §28.3) is the dominant approach — the "augmentation" is simply the other modality, which is a far more semantic and less hand-designed invariance than any crop or jitter.
- **Audio, video, graphs, molecules, time series, tabular:** the same three families reappear with domain-specific augmentations, and choosing the augmentations remains the hard part.

---

## Check for Understanding

**Every self-supervised method must simultaneously make augmented views agree and prevent the representation from collapsing to a constant — contrastive methods use negatives to enforce uniformity on the hypersphere, non-contrastive methods use a predictor with a stop-gradient or explicit variance and decorrelation terms instead — and because the invariances are imposed by the augmentations, choosing the augmentations is the real modelling decision.**

---

**Next:** [Chapter 26 — Reinforcement Learning Foundations](26-rl-foundations.md)
