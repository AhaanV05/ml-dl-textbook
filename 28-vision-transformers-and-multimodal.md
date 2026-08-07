# Chapter 28 — Vision Transformers & Multimodal Models

> **Prerequisites:** Ch. 8, Ch. 11, Ch. 25.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $224\times224\times3$ | "224 by 224 by 3" | An image's shape: height, width, colour channels (red, green, blue) |
| $P$ | "the patch size" | The side length of one square tile, usually 14 or 16 pixels |
| $N = HW/P^2$ | "N equals H W over P squared" | **How many patches an image becomes.** For $224/16$: $14\times14 = 196$ |
| `[CLS]` | "the class token" | An extra, learned, content-free token whose output vector stands for the whole image |
| $`I_i`$ | "I-sub-i" | The embedding vector of the $i$-th **image** in the batch |
| $`T_j`$ | "T-sub-j" | The embedding vector of the $j$-th **text** (caption) in the batch |
| $`I_iT_j^\top`$ | "I-i times T-j transpose" | A **dot product** — one number saying how aligned image $i$ is with caption $j$ |
| $\tau$ | "tau" | The **temperature** — a divisor that sharpens or flattens the similarity scores |
| $\mathrm{diag}$ | "the diagonal" | The labels $(1,2,\dots,N)$: image $i$'s correct caption is caption $i$ |
| $\sigma(\cdot)$ | "sigmoid" | $1/(1+e^{-x})$ — squashes any number into $(0,1)$ |
| $`z_{ij}`$ | "z-i-j" | A $\pm1$ label: $+1$ if image $i$ and caption $j$ are a true pair, $-1$ otherwise |
| $O(T^2)$ | "order T squared" | Cost grows with the **square** of the number of tokens |
| $T\times H\times W$ | "T by H by W" | A video's shape: frames × height × width |
| $`2595\log_{10}(1+f/700)`$ | "the mel formula" | Converts a frequency in hertz to a *perceptual* pitch scale |

**Full forms of the abbreviations used in this chapter**

| Short | Full form |
|---|---|
| ViT | Vision Transformer |
| CNN | convolutional neural network |
| CLS | classification (token) |
| CLIP | Contrastive Language–Image Pre-training |
| VLM | vision–language model |
| LLM | large language model |
| MLP | multi-layer perceptron |
| LN / pre-LN | layer normalization / normalization applied *before* each sublayer |
| InfoNCE | information noise-contrastive estimation |
| CE | cross-entropy |
| ASR | automatic speech recognition |
| TTS | text-to-speech |
| STFT | short-time Fourier transform |
| RVQ | residual vector quantization |
| VQ | vector quantization |
| VAE | variational autoencoder |
| BPE | byte-pair encoding |
| DiT | diffusion transformer |
| MoE | mixture of experts |
| MAE | masked autoencoder |
| OCR | optical character recognition |
| RoPE | rotary position embedding |
| POPE | Polling-based Object Probing Evaluation |
| ARO | Attribution, Relation, and Order (benchmark) |

---

## 28.1 Vision Transformers

### The one-line idea

Cut the image into patches, treat each patch as a token, and run a standard transformer — discarding almost every inductive bias that made CNNs work.

### The analogy

Reading a painting the way you read a sentence. A CNN examines the canvas through a small sliding magnifier, building understanding from local texture upward. A ViT chops the canvas into 196 squares, lays them out in a row, and lets every square talk to every other square immediately. It has no idea which squares were adjacent — you have to tell it (positional embeddings) — but in exchange, nothing is far away.

### The architecture

1. **Patchify.** $224\times224\times3$ → $14\times14=196$ patches of $16\times16\times3=768$ values. A linear projection maps each to $d$. ▸ *This projection is exactly a $16\times16$ convolution with stride 16* — the only convolution in the model, and it is why ViT is sometimes described as "a transformer with a single conv stem."
2. **Prepend a `[CLS]` token** whose final representation is the image embedding. (Global average pooling over patch tokens works about as well.)
3. **Add positional embeddings** — learned 1-D is standard; 2-D-aware and RoPE variants are used in newer models.
4. $L$ standard pre-LN transformer blocks.
5. Linear head on the `[CLS]` output.

#### Unpacking the five steps, with real numbers

Take one ordinary photograph and follow it all the way through. Nothing here is more complicated than arithmetic.

**Step 1 — patchify.** The image is a block of numbers of shape $224\times224\times3$: 224 rows of pixels, 224 columns, and 3 colour channels (red, green, blue). Multiply those out and the image is **150,528 numbers**.

Now lay a grid of $16\times16$ squares over it. Along the height, $224/16 = 14$ squares fit. Along the width, another 14. So the grid has $14 \times 14 = 196$ cells. Each cell contains $16 \times 16 = 256$ pixels, each with 3 channels, so each cell is $16\times16\times3 = 768$ numbers.

▸ **Check the bookkeeping:** $196 \times 768 = 150{,}528$ — exactly the number of pixels you started with. **Patchification throws nothing away.** It is a reshape, not a summary. Every pixel is still present; it has only been re-grouped.

Then the linear projection: a single matrix $E \in \mathbb{R}^{768 \times d}$ turns each patch's 768 raw numbers into a $d$-dimensional token. With $d = 768$ that matrix holds $768 \times 768 = 589{,}824$ weights, and it is **shared across all 196 patches** — the same projection is applied to the top-left corner and the bottom-right corner alike.

**Why this "is exactly a $16\times16$ convolution with stride 16."** A convolution slides a kernel across the image; stride 16 means it jumps 16 pixels each time. With a $16\times16$ kernel and a stride of 16, consecutive kernel placements **never overlap and never leave gaps** — they tile the image perfectly. In code this is not an analogy, it is the literal implementation:

```python
# The entire "patchify + project" stage of a ViT
patch_embed = nn.Conv2d(in_channels=3, out_channels=768,
                        kernel_size=16, stride=16)
tokens = patch_embed(image)        # (B, 768, 14, 14)
tokens = tokens.flatten(2).transpose(1, 2)   # (B, 196, 768)
```

> **Analogy.** A mosaic artist takes a photograph and reproduces it in 196 square tiles. Patchification is cutting the photo into those 196 squares; the linear projection is the artist deciding, for each square, which single tile from her workshop's palette best stands for it. She uses the same palette for every square. What she has *not* done is record where each square sat in the original — that comes next.

**Step 2 — the `[CLS]` token.** This is a single learned vector, the same one for every image, glued onto the front of the sequence. It carries no image content whatsoever at input. Its job is to be a **scratchpad**: as it passes through the attention layers it reads from the 196 patch tokens and accumulates a summary. Its final vector is what the classifier sees.

The sequence length is therefore $196 + 1 = 197$.

Why would a content-free token work? Because attention lets it *choose* what to read. A token that starts as a blank slate has no preconceptions to overcome — it is a query with no answer yet. The alternative, global average pooling (add up all 196 patch vectors and divide by 196), works about as well, which tells you the `[CLS]` token is a convenience rather than a deep necessity.

**Step 3 — positional embeddings.** Here is the crucial point, and the one beginners skip. Self-attention is **permutation-equivariant**: shuffle the input tokens and the outputs shuffle identically, but nothing else changes. To the raw attention mechanism, the 196 patches are a *bag*, not a grid. Sky patches and grass patches are unordered.

So you add a learned vector $`p_k \in \mathbb{R}^{d}`$ to patch $k$'s embedding — 197 such vectors, learned by gradient descent like any other parameter. That is 197 × 768 ≈ 151,000 extra numbers, and it is the model's entire sense of geography.

▸ **What would break without them.** Take a photo of a person standing on grass under sky, shuffle the 196 patches into a random order, and a ViT *without* positional embeddings gives **bit-for-bit the same output**. It cannot tell a portrait from a jigsaw puzzle poured out on a table. Empirically, removing positional embeddings costs several points of ImageNet accuracy — not catastrophic, because texture alone carries a lot of signal, but the model is  blind to layout.

**Step 4 — the transformer blocks.** These are unchanged from Chapter 11. Nothing is vision-specific. "Pre-LN" means layer normalization is applied *before* each sublayer rather than after, which is what makes deep stacks trainable without a warmup-dependent knife-edge (Ch. 11, Ch. 14).

Attention cost: every one of the 197 tokens compares itself to every other, so the attention matrix has $197^2 = 38{,}809$ entries **per head, per layer**. With 12 heads and 12 layers that is 5.6 million pairwise scores for one image.

**Step 5 — the head.** One matrix from $d$ to the number of classes. For ImageNet-1k that is $768 \times 1000$.

#### Examples and non-examples: what counts as a "token" in a vision transformer

**✅  examples**

| Example | Why it qualifies |
|---|---|
| One $16\times16$ pixel patch, projected to a 768-vector | A fixed-length vector that attention can treat as an atomic unit |
| The `[CLS]` token | A learned vector in the same space, participating in the same attention |
| A "tubelet" in a video model — a $2\times16\times16$ block spanning two frames | Same idea, one dimension richer |
| A 25 ms slice of a mel spectrogram, projected | Audio's version of the same move (§28.4) |

**❌ Near-misses — called tokens, but aren't playing the same role**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A single **pixel** | Attention over $224^2 = 50{,}176$ pixels needs 2.5 **billion** pairwise scores per head — computationally impossible at scale | The reason patches exist at all. iGPT tried pixel-level sequences and paid enormous compute for $32\times32$ images |
| A **detected object box** ("there is a dog here") | Requires a detector — an extra trained model with its own failure modes and a fixed vocabulary | A region proposal; the pre-ViT approach used by early vision–language models |
| A CNN **feature map cell** after 5 downsampling stages | It is a token *now*, but it arrived through a hierarchy of learned convolutions, so most of the modelling already happened | A hybrid stem (used by CvT, CoAtNet, and detection heads) |
| A **superpixel** or segment | Variable in size and count, so it breaks the fixed-shape batching every transformer relies on | A segmentation primitive |

▸ **The boundary:** a token is any fixed-size vector that the attention mechanism is willing to treat as an indivisible unit. **The choice of what to make a token is the only  vision-specific decision in a ViT** — everything after step 1 is the same code that runs on text.

> **Common misconception.** *"A ViT understands the image as a grid, like a CNN does."* It does not, and this is the whole point. A CNN's convolution physically cannot look at a pixel far away — its receptive field grows only through depth, so locality is baked into the hardware of the operation. A ViT's attention has **no such structure at all**; layer 1 head 1 is free to compare the top-left patch to the bottom-right patch, and the only reason it knows they are far apart is that someone handed it a learned vector saying so. The misconception is tempting because ViTs *do* often end up behaving locally — but that is a learned outcome, not an architectural guarantee, and the difference is exactly what the data-hunger result below is about.

### The data-hunger result

▸ The original ViT paper's central finding: on ImageNet-1k (1.3M images) ViT **underperforms** a comparable ResNet. On ImageNet-21k (14M) it matches. On JFT-300M it **wins decisively**.

**The interpretation is the generalizable lesson:** a CNN's locality and translation-equivariance priors are *correct but restrictive*. When data is scarce, a correct prior substitutes for data. When data is abundant, the same prior becomes a ceiling, and a model that can learn its own structure surpasses it. **This is Chapter 2's approximation/estimation trade-off appearing as an architectural choice.**

▸ Probing confirms the mechanism: trained ViTs learn convolution-like local attention in early layers when data is plentiful — they *rediscover* the prior — while retaining global attention where it helps.

#### What the data-hunger result actually says

Put the three datasets on one line so the scale gap is visible:

| Dataset | Images | Roughly | ViT vs. a comparable ResNet |
|---|---|---|---|
| ImageNet-1k | 1.3M | one photo per person in a small city | ViT **loses** |
| ImageNet-21k | 14M | ten times more | ViT **ties** |
| JFT-300M | 300M | ~230× ImageNet-1k, noisily labelled, Google-internal | ViT **wins clearly** |

**"Inductive bias" decoded.** An inductive bias is any assumption a model makes *before it sees data* about which explanations are plausible. A CNN makes two of them, and they are excellent assumptions about photographs:

- **Locality** — pixels near each other are related; pixels far apart are probably not, at least not directly.
- **Translation equivariance** — a cat in the top-left and the same cat in the bottom-right should be processed identically. A convolution enforces this by *reusing the same kernel everywhere*.

> **Analogy.** A prior is a pair of guide rails. Rails are wonderful when you have a beginner driver and a short trip: they keep you on the road with almost no skill required. They are a liability on a rally course, where the fastest line is off the marked path. **Data is driving skill.** With very little of it, the rails are all that keeps you from the ditch. With a great deal of it, the rails become the thing stopping you from going faster.

**Why "a correct prior substitutes for data" is literally true.** A prior removes hypotheses from consideration before you look at anything. In Chapter 2's language, it shrinks the hypothesis class $\mathcal{H}$. A smaller $\mathcal{H}$ means smaller **estimation error** (less overfitting, you need fewer samples to pick the right member) but potentially larger **approximation error** (the truth might not be in $\mathcal{H}$ at all). A convolution's weight sharing is a hard constraint: the same 9 numbers are used at every position. A ViT can express that solution but is not forced into it, so it must *spend data* discovering it.

▸ **Put numbers on the trade.** A $3\times3$ conv layer with 64 input and 64 output channels has $3\cdot3\cdot64\cdot64 = 36{,}864$ weights, and those same weights are applied at every one of tens of thousands of spatial positions. A fully-connected layer over a $56\times56\times64$ feature map would need $(56\cdot56\cdot64)^2 \approx 4\times10^{10}$ weights. **The convolutional prior is a roughly billion-fold reduction in what must be learned** — and that reduction is exactly the thing that becomes a ceiling when it turns out some of the discarded hypotheses were the good ones.

**What "rediscover the prior" means concretely.** Take a trained ViT, feed it an image, and plot for each attention head the *average distance* (in patches) between a query patch and the patches it attends to. In early layers of a ViT trained on plenty of data, many heads show a mean attention distance of only one or two patches — they are looking at their immediate neighbours. That is a convolution, learned rather than imposed. Other heads in the same layer show distances spanning the whole image. **The model built itself a hybrid** that no human would have specified.

> **Where this came from.** The ViT paper — *An Image is Worth 16×16 Words*, by Alexey Dosovitskiy and colleagues at Google Research — appeared on arXiv in October 2020 and at ICLR 2021. Its notable feature is what it *doesn't* contain: the authors deliberately kept the architecture as close as possible to the original 2017 text transformer, changing essentially nothing except how the input is chopped up. Transformers on images had been attempted before — OpenAI's iGPT (2020) ran a GPT directly over sequences of pixels and needed enormous compute to handle $32\times32$ images — and the patch idea was what made the cost tractable. There was also a strikingly relevant theoretical result the year before: Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi at EPFL proved in 2020 that a multi-head self-attention layer *can express any convolutional layer*, and their experiments already used small patches. So the expressiveness question was settled before the engineering question was.

> **Common misconception.** *"ViT beat CNNs, so attention is a better operation than convolution."* The result says something narrower and more interesting: attention is a **more general** operation, and generality is only an advantage when you can pay for it in data. On ImageNet-1k alone the CNN wins. ConvNeXt later showed that a plain CNN given ViT-style *training* (heavy augmentation, AdamW, long schedules, large kernels) matches Swin — which means a good share of the original gap was the recipe, not the operation. The misconception is tempting because the headline number is a single scalar and the caveat is a paragraph.

> **Common misconception.** *"More data always favours the model with fewer assumptions."* Only when the assumptions were wrong or restrictive in a way that matters. Translation equivariance for photographs is very nearly *true*, which is why the crossover needs hundreds of millions of images rather than a few million. For a domain where the prior is badly wrong — say, satellite imagery where "up" is meaningless, or medical scans where absolute position is diagnostic — the crossover arrives far sooner.

#### Examples and non-examples: inductive bias

**✅  inductive biases**

| Example | The assumption it encodes |
|---|---|
| Convolution's weight sharing | "The same feature is worth detecting everywhere" |
| Max-pooling | "Small translations shouldn't change the answer" |
| A recurrent network's hidden state | "The past matters only through a fixed-size summary" |
| Causal masking in a language model | "Token $t$ cannot depend on token $t+1$" |
| $`\ell_2`$ weight decay | "Small weights are more plausible than large ones" |
| A graph neural network's message passing | "Only connected nodes influence each other" |

**❌ Near-misses — often called inductive biases, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| **Data augmentation** (random crops, flips) | It doesn't restrict what the model *can* represent; it changes the data distribution the model sees | A **prior expressed through data** — same intent, different mechanism, and it can be overridden by contrary evidence |
| **Dropout** | Applies at training only, and constrains nothing at inference | A stochastic regularizer |
| A **learned** positional embedding | It is learned from data, not assumed in advance | A learned parameter. (A *sinusoidal* or 2-D-aware one, fixed in advance,  is a bias) |
| **Batch size 32,768** | A resource decision | An optimization/systems choice |
| Choosing **ReLU over tanh** | Both are universal approximators | An optimization-friendliness choice, with a mild sparsity effect |

▸ **The boundary:** an inductive bias is a restriction on the hypothesis space that exists **before any data arrives** and that data cannot undo. Augmentation and regularization *nudge*; architecture *forbids*. A CNN cannot represent a function that treats the top-left corner differently from the bottom-right at the same layer — it is not that it prefers not to.

### The variants

| Model | Contribution |
|---|---|
| **DeiT** | matched ViT performance on ImageNet-1k alone using heavy augmentation (RandAugment, mixup, CutMix), stochastic depth, and **distillation from a CNN teacher via a dedicated distillation token**. Showed the data-hunger was largely a *recipe* problem. |
| **Swin** | hierarchical, with **shifted windows**: attention within local windows, and windows shifted between layers to allow cross-window information flow. Restores $O(T)$ complexity and multi-scale features — necessary for detection and segmentation. |
| **PVT / CvT / CoAtNet** | reintroduce pyramids and convolutions in various proportions |
| **MAE / BEiT** | self-supervised pretraining (Ch. 25 §25.4) |
| **DINOv2/v3** | self-supervised features usable off the shelf for dense and global tasks |
| **ConvNeXt** | the counter-argument — a CNN with ViT-style training matches Swin (Ch. 8 §8.3) |

#### Unpacking the variants: what each one is actually fixing

Every row in that table is a different answer to the same question — *which piece of the CNN was worth keeping?*

**DeiT — "the data hunger was a recipe problem."** DeiT keeps the ViT architecture exactly and changes only the training. The interesting piece is the **distillation token**: alongside `[CLS]`, add a *second* extra token whose output is trained to match a CNN teacher's prediction. So the model has two summary tokens with two different jobs — one predicting the true label, one predicting what a ResNet would have said.

Why distil from a *CNN* specifically, when the CNN is worse? Because the CNN's mistakes encode its priors. Learning to imitate a convolutional network is a soft, differentiable way of absorbing the locality bias without hard-coding it — a prior delivered through the loss instead of the architecture. DeiT trained to competitive ImageNet-1k accuracy on a single eight-GPU machine in around three days, which at the time was the difference between "Google can do this" and "anyone can do this."

**Swin — "restore the pyramid."** The problem Swin solves is arithmetic. Global attention over $N$ tokens costs $O(N^2)$. At $224\times224$ with patch 4 (Swin's stem is finer than ViT's) you would have $56\times56 = 3{,}136$ tokens, and $3{,}136^2 \approx 9.8$ million pairs per head per layer. Confine attention to non-overlapping $7\times7$ windows and each window has 49 tokens, costing $49^2 = 2{,}401$ pairs; with $64$ such windows the total is $64 \times 2{,}401 = 153{,}664$ pairs.

▸ **That is a 64× reduction, and more importantly the cost is now linear in image area rather than quadratic.** Double the image size and windowed attention doubles in cost; global attention quadruples.

The catch: information cannot cross a window boundary. Swin's fix is the **shift** — in every other layer, slide the whole window grid by half a window ($7/2 \to 3$ pixels). A patch that was at the right edge of window A is now in the middle of a window that also contains what used to be the left edge of window B. Two layers, and everything can reach its neighbours.

> **Analogy.** Seat 3,136 people at 64 tables of 49 and let each table talk. Nobody learns anything from another table. Now, between courses, shift every table's boundaries by half a table's width so each person sits with a partly new set of neighbours. After a few courses, gossip has crossed the entire room — and you never had to run a 3,136-person conversation.

**Why detection and segmentation *need* the hierarchy.** These tasks require predictions at multiple scales: a bounding box may be 20 pixels or 400. Standard ViT produces one resolution ($14\times14$) at every layer. Swin halves the resolution at each stage ($56 \to 28 \to 14 \to 7$) exactly like a ResNet, so a detection head can attach feature maps at four scales.

**ConvNeXt — the control experiment.** It contains no new module. The authors took a ResNet-50 and applied ViT-era training and design choices one at a time — AdamW, 300-epoch schedules, heavy augmentation, larger $7\times7$ depthwise kernels, fewer normalization layers, GELU instead of ReLU, an inverted bottleneck — measuring after each change. **It is a paper structured as an ablation of a rumour**, and it landed at Swin-level accuracy. Read it as the answer to "how much of ViT's win was attention, and how much was 2020s training?"

#### Examples and non-examples: "a vision transformer"

**✅  vision transformers**

| Example | Why it qualifies |
|---|---|
| ViT-B/16 | Patches → tokens → plain global self-attention |
| DeiT | Identical architecture; only the training recipe differs |
| Swin | Attention is windowed rather than global, but the token-mixing operation is still attention |
| MAE / BEiT | ViT backbones with a self-supervised pretraining objective bolted on top |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| **ConvNeXt** | Contains no attention anywhere. Token mixing is done by depthwise convolution | A modernized CNN — the deliberate counter-example |
| **MLP-Mixer** | Mixes across tokens with a fixed learned MLP, not a data-dependent attention matrix | An all-MLP architecture. The mixing weights don't depend on the input |
| **A CNN with an attention block at the end** (e.g. squeeze-and-excitation, or a non-local block) | The overwhelming majority of computation is still convolutional | A CNN with attention *augmentation* |
| **DETR** | Uses a transformer decoder for the *detection head*, but a ResNet does the seeing | A transformer detection head on a CNN backbone |
| **CoAtNet / CvT** | Convolutions in early stages, attention later | A  hybrid — and often the best small-data option |

▸ **The boundary:** it is a vision transformer if the *primary* mechanism mixing information between spatial locations is **input-dependent attention** — a matrix computed from the data itself, different for every image. Convolution's mixing weights are the same for every image ever seen; attention's are recomputed each forward pass. That data-dependence is the whole difference, and it is both the source of the flexibility and the reason the model needs so much data.

> **Common misconception.** *"Swin's windows make it a CNN in disguise."* The window restricts *where* a token may look, exactly as a convolution's kernel does — that part of the intuition is right. But inside the window, the mixing weights are still computed from the content of this particular image. A $7\times7$ convolution applies the same 49 numbers to a photo of a beach and a photo of a circuit board; a $7\times7$ attention window computes 49 fresh weights for each. **Restricted scope, not fixed weights.**

▸ **The synthesis to state if asked "CNN or transformer for vision?":** at small scale with limited data, a modern CNN is at least as good and cheaper. At large scale with pretraining, transformers win and — more importantly — they unify vision with language, which is the actual reason they took over.

---

## 28.2 Contrastive vision–language: CLIP

### The one-line idea

Train an image encoder and a text encoder so that matching image–caption pairs land near each other in a shared embedding space — and get open-vocabulary classification for free.

### The analogy

Teaching two people, one who only sees and one who only reads, to point at the same spot on a shared map. Once they agree on the map, you can describe something in words and the seeing person can find it — even for things neither was explicitly trained on, because the map has structure.

### The objective

For a batch of $N$ pairs, compute image embeddings $`I_i`$ and text embeddings $`T_j`$, both L2-normalized. The logits are $`\frac{I_iT_j^\top}{\tau}`$, and the loss is **symmetric cross-entropy** over both axes:

▸ $$\mathcal{L}=\frac12\left[\underbrace{\mathrm{CE}\big(\tfrac{IT^\top}{\tau},\,\mathrm{diag}\big)}_{\text{image}\to\text{text}} + \underbrace{\mathrm{CE}\big(\tfrac{TI^\top}{\tau},\,\mathrm{diag}\big)}_{\text{text}\to\text{image}}\right]$$

**It is an $N$-way classification: "which of these $N$ captions belongs to this image?"** — InfoNCE (Ch. 25 §25.3) with the other modality as the positive.

▸ **The temperature $\tau$ is learned**, parameterized as $\exp(t)$ and clipped to prevent instability. A learned temperature was a small but  important detail.

**Scale:** 400M pairs, batch size **32,768**. The large batch matters — the number of negatives directly determines the difficulty and the tightness of the bound.

#### Reading the CLIP loss in plain English

Every symbol, then a worked example with actual arithmetic.

| Piece | Read aloud | What it is |
|---|---|---|
| $N$ | "N" | The batch size — how many image–caption pairs are in flight at once |
| $`I_i`$ | "I-sub-i" | Image $i$'s embedding, a vector of length $d$ (say 512) |
| $`T_j`$ | "T-sub-j" | Caption $j$'s embedding, same length |
| "L2-normalized" | — | Every vector has been divided by its own length, so $`\lVert I_i\rVert = 1`$ |
| $`I_iT_j^\top`$ | "I-i T-j transpose" | Their **dot product** — a single number |
| $\tau$ | "tau" | Temperature. Divide by it before the softmax |
| $\mathrm{CE}(\cdot,\ \mathrm{diag})$ | "cross-entropy against the diagonal" | Standard classification loss where the correct answer for row $i$ is column $i$ |
| $\tfrac12[\cdots+\cdots]$ | — | Average the two directions so neither modality is privileged |

**Why normalization turns a dot product into a cosine.** For unit vectors, $I \cdot T = \cos\theta$, where $\theta$ is the angle between them. So every score lives in $[-1, 1]$: $1$ means "pointing the same way," $0$ means "unrelated / perpendicular," $-1$ means "opposite." **This is the whole reason for the normalization** — without it, the model could win by making some vectors enormously long instead of making them point in the right direction.

**The matrix picture.** With $N$ images and $N$ captions you compute all $N^2$ dot products and lay them out in a grid:

$$S = \frac{IT^\top}{\tau} \in \mathbb{R}^{N\times N}, \qquad S_{ij} = \frac{I_i \cdot T_j}{\tau}$$

Row $i$ is "how well does image $i$ match each of the $N$ captions?" Column $j$ is "how well does caption $j$ match each of the $N$ images." **The correct answers are all on the diagonal**, because the data was assembled as pairs: image 1 came with caption 1. That is what $\mathrm{diag}$ means as a label — the target for row $i$ is the integer $i$.

The two terms are the two directions of reading the grid: softmax across each **row** (image finds its caption), softmax down each **column** (caption finds its image). Both are needed, because a model can be excellent at one and mediocre at the other.

```python
# The entire CLIP loss, in five lines
logits = (img_emb @ txt_emb.T) * logit_scale   # (N, N)
labels = torch.arange(N)                       # [0, 1, 2, ..., N-1]
loss_i = F.cross_entropy(logits,   labels)     # across rows
loss_t = F.cross_entropy(logits.T, labels)     # across columns
loss   = (loss_i + loss_t) / 2
```

**Now put real numbers in it.** Set $N = 4$ and $\tau = 0.07$ (so $1/\tau \approx 14.3$). Suppose image 1's cosine similarities to the four captions are:

$$(\,0.30,\ 0.10,\ 0.05,\ 0.20\,)$$

with $0.30$ being the true pair. Divide by $\tau$:

$$(\,4.29,\ 1.43,\ 0.71,\ 2.86\,)$$

Exponentiate: $(72.7,\ 4.17,\ 2.04,\ 17.4)$, which sums to $96.3$. So

$$p(\text{correct}) = \frac{72.7}{96.3} = 0.755, \qquad \mathcal{L}_{\text{row 1}} = -\ln(0.755) = 0.28$$

▸ **Now run the same numbers with $\tau = 1$** — that is, with no temperature at all. The logits stay $(0.30, 0.10, 0.05, 0.20)$; exponentiating gives $(1.350,\ 1.105,\ 1.051,\ 1.221)$, summing to $4.728$. So $p(\text{correct}) = 0.286$ and the loss is $1.25$ — against a pure-chance loss of $\ln 4 = 1.39$. **The model has nearly all the information it needs and the loss barely registers it.**

That is what temperature does: cosine similarities live in a cramped range around zero, and the softmax of a cramped range is nearly uniform. Dividing by $0.07$ stretches a $0.25$ gap into a $3.6$ gap, which the softmax can act on. Without it the gradient signal is so weak that training essentially fails to start.

> **Analogy.** Temperature is the zoom on a microscope. Two cells differing by a hair look identical at 1×; at 100× the difference is obvious. Zoom too far and you see only one cell and lose all context — which is why $\tau$ is **clipped**: as training progresses the model wants $\tau$ smaller and smaller (sharper and sharper), and left unchecked it drives itself to a numerically unstable regime where a single hard negative dominates the entire loss. CLIP parameterizes the scale as $\exp(t)$ so it stays positive, learns $t$ by gradient descent like any other parameter, and caps how large the scale may grow.

**Why the batch size of 32,768 is a scientific choice, not just a resource one.** The loss asks "which of these $N$ captions?" With $N = 4$, guessing at random succeeds $25\%$ of the time and the task is trivial. With $N = 32{,}768$, chance is $0.003\%$ and the chance-level loss is $\ln(32{,}768) = 10.4$ nats. **Each in-batch example serves as a negative for every other**, so a batch of 32,768 supplies 32,767 negatives per image at no extra encoding cost — the hardest possible bargain in machine learning.

▸ And the negatives get *harder* as $N$ grows. In a batch of 4, the other three captions are almost certainly about completely different things. In a batch of 32,768 scraped from the internet, several of the "negatives" will be other photographs of dogs — so the model is forced to learn the difference between a golden retriever and a labrador rather than the difference between a dog and a bridge.

#### Examples and non-examples: what is a "negative" in contrastive learning

**✅  negatives**

| Example | Why it qualifies |
|---|---|
| The other 32,767 captions in the batch, for a given image | Each is a plausible-looking alternative that is nonetheless wrong |
| Another photo of a different dog breed | A **hard** negative — most informative, because separating it requires real discrimination |
| A caption describing the same scene at a different time of day | Hard negative; teaches temporal/lighting sensitivity where it matters |

**❌ Near-misses — treated as negatives, but they poison the loss**

| Looks like a negative | Why it isn't | What it actually is |
|---|---|---|
| A different photo of **the same object with the same caption** ("a photo of a dog" appears 8,000 times in the batch) | It is a **true positive** the loss is punishing the model for matching | A **false negative** — the known central pathology of in-batch contrastive learning |
| The image paired with itself | The loss is asked to push a vector away from itself; the gradient is degenerate | Nothing — an implementation bug |
| A caption in a different language describing the same image | Semantically a positive | A cross-lingual positive, and the reason multilingual CLIP variants need care |

▸ **The boundary:** a negative must be **wrong**, not merely **different**. In-batch contrastive learning assumes that any non-paired combination is wrong, which is false at a rate that grows with batch size and with how repetitive your caption distribution is. It works anyway because the false-negative rate is low relative to 32,767, but it is the mechanism behind several of CLIP's known weaknesses.

> **Where this came from.** CLIP — *Learning Transferable Visual Models From Natural Language Supervision*, Alec Radford and colleagues at OpenAI — was released on 5 January 2021, the same day as the original DALL·E. Learning images from captions was not new: a 2017 paper from Facebook, *Learning Visual N-Grams from Web Data*, had already done zero-shot ImageNet classification from web captions and reached about **11.5%** top-1. CLIP reached **76.2%**, matching a fully supervised ResNet-50 — while never seeing a single ImageNet training label. The CLIP paper describes itself as a scaled-up, simplified relative of ConVIRT, a 2020 method by Yuhao Zhang and co-authors that had applied contrastive image–text learning to **chest X-rays and radiology reports**. The technique that gave the internet open-vocabulary image search was prototyped on medical imaging, where paired image–text data happens to exist naturally because radiologists write reports.

> **Where this came from.** The loss itself descends from **noise-contrastive estimation**, introduced by Michael Gutmann and Aapo Hyvärinen in 2010 for an entirely different purpose: estimating probability models whose normalizing constant is intractable to compute. Their trick was to turn density estimation into a classification problem — "is this sample from the data or from noise?" — because classification does not require the normalizer. Aäron van den Oord, Yazhe Li, and Oriol Vinyals adapted it as **InfoNCE** in the 2018 Contrastive Predictive Coding paper, showing it lower-bounds the mutual information between the two views. So the objective at the heart of CLIP was invented to dodge an integral, not to align modalities.

### Zero-shot classification

Embed the class names as prompts ("a photo of a {class}"), embed the image, take the nearest. ▸ **This is retrieval, not classification**, and it means the label set can be changed at inference time with no retraining. That is what made CLIP transformative.

**Prompt ensembling** — averaging embeddings over 80 prompt templates — gives ~+3.5% ImageNet top-1 for free.

▸ **Why CLIP generalizes so well:** its "augmentation" is a natural-language description, which is a far more semantic and less arbitrary invariance than any crop or colour jitter (Ch. 25 §25.3). Two images sharing a caption share *meaning*, not pixel statistics.

**Known weaknesses, worth being precise about:**
- **Compositionality and word order.** CLIP behaves substantially like a bag of concepts: "a horse riding an astronaut" embeds close to "an astronaut riding a horse" (the ARO benchmark quantifies this).
- **Counting, spatial relations, and negation** are poor.
- **Fine-grained distinctions** (species, aircraft variants) are weak.
- **Rendered text shortcut:** it can read text in the image and use it, which is a shortcut rather than visual understanding.
- Performance tracks the concept's frequency in the pretraining data — "zero-shot" is doing less work than it sounds.

#### Zero-shot classification, step by step with numbers

Suppose you want CLIP to distinguish **cat**, **dog**, and **fire hydrant** — three classes it was never explicitly trained on as classes.

1. **Write the prompts.** `"a photo of a cat"`, `"a photo of a dog"`, `"a photo of a fire hydrant"`. Feed each through the *text* encoder. You now have three 512-dimensional unit vectors. **Do this once** and cache them; they are your classifier weights.
2. **Encode the image.** One forward pass through the *image* encoder → one 512-dimensional unit vector.
3. **Three dot products.** Say you get $(0.31,\ 0.22,\ 0.04)$.
4. **Argmax.** "cat."

▸ **Notice what step 1 is.** Three text embeddings stacked into a $3\times512$ matrix, multiplied against the image vector, is *exactly* a linear classification head — except the weights were written in English rather than learned by gradient descent. **A prompt is a row of a weight matrix that you can type.**

**Why "this is retrieval, not classification."** A trained classifier has a fixed output dimension baked into its final matrix; adding a fourth class means adding a row and training it. CLIP's "classifier" is a nearest-neighbour lookup against a set of text vectors you can regenerate at any moment. Adding "traffic cone" costs one text forward pass and zero gradient steps.

**Prompt ensembling, concretely.** Instead of one template, encode 80 of them for the same class — `"a photo of a {}"`, `"a blurry photo of a {}"`, `"a photo of the large {}"`, `"art of a {}"`, and so on — then **average the 80 vectors and re-normalize**. This is the same variance-reduction move as ensembling models (Ch. 3), applied in embedding space instead of prediction space, and it costs nothing at inference because the averaging happens once. It buys about $+3.5\%$ ImageNet top-1.

**Why the bare class name is a bad prompt.** The caption distribution CLIP trained on is full of sentences, not nouns. A photo captioned with the single word `"boxer"` on the internet is roughly as likely to be a fighter as a dog. `"a photo of a boxer, a type of pet"` resolves it. **The prompt is disambiguating which sense of the word you mean**, using the same mechanism a human would.

#### Examples and non-examples: "zero-shot"

**✅  zero-shot**

| Example | Why it qualifies |
|---|---|
| Classifying satellite images into land-use categories by typing the category names | No labelled example of any category was used to fit anything |
| Retrieving "a red car parked under a tree" from an unlabelled photo library | The query never existed as a training class |
| Classifying a newly named product category the week it launches | Nothing was retrained |

**❌ Near-misses — described as zero-shot, but something was fitted**

| Looks zero-shot | Why it isn't | What it actually is |
|---|---|---|
| Trying 30 prompt templates and keeping the one with the best validation accuracy | You used labelled validation data to select a hyperparameter | **Prompt tuning** — a form of few-shot learning with a very small parameter count |
| Fitting a logistic-regression head on frozen CLIP features using 16 labelled images per class | Sixteen labels per class were used | **Linear probing**, 16-shot |
| Classifying dog breeds when the pretraining set contained millions of breed-captioned dog photos | The task's concepts were densely present in training | Zero-shot *transfer* of a well-covered concept. , but far less impressive than the phrase implies |
| Evaluating on a benchmark whose images were scraped from the same web crawl as the pretraining data | Test data leakage | Contamination |

▸ **The boundary:** zero-shot means **no labelled example of the target task was used, including for tuning and model selection**. The moment a validation set touches your choice of prompt, threshold, or checkpoint, you are few-shot with an unusually efficient parameterization. This is not a criticism of the method — it is a criticism of the reporting.

> **Common misconception.** *"Zero-shot means the model has never seen anything like this."* It means the model has never seen this **task**. It has almost certainly seen the concept, thousands of times, described in words. CLIP's zero-shot accuracy on a class tracks that class's frequency in the pretraining corpus closely enough that the correlation is one of the paper's own findings. **The knowledge was acquired during pretraining; only the task specification is new at test time.** The misconception is tempting because "zero" sounds absolute, and because the demo is  astonishing the first time you see it.

> **Common misconception.** *"CLIP understands the sentence, so it must understand word order."* It substantially does not. Its text encoder was trained only to make the caption vector land near the image vector, and that objective is fully satisfied by a bag of concepts — nothing in the loss ever rewards distinguishing "a horse riding an astronaut" from "an astronaut riding a horse," because in the training data those two orderings essentially never appear as competing captions for different images. The ARO (Attribution, Relation, and Order) benchmark was built specifically to measure this, by shuffling words in true captions and checking whether the model notices. It largely doesn't. **The training objective never asked for compositionality, so the model never bought any.**

#### What would break: the failure modes, mechanically

| Symptom | The mechanism |
|---|---|
| Can't count ("three cats" vs "two cats") | Both captions have near-identical bags of concepts; the contrastive loss almost never had a batch containing both a two-cat and a three-cat photo with those exact captions |
| Negation fails ("a photo with no dog" retrieves dogs) | The word "dog" contributes its embedding regardless of the surrounding "no." A bag has no operators |
| Reads text in the image and uses it | An apple with the word "iPod" written on it classifies as an iPod. The caption-image correlation for rendered text is extremely strong and extremely easy to exploit — it is a shortcut in the exact sense of Chapter 2 |
| Fine-grained species confusion | Web captions say "a bird," not "a black-throated blue warbler." The supervision signal for the distinction was never there |

### SigLIP

Replace the softmax cross-entropy with a **pairwise sigmoid** loss:
▸ $$\mathcal{L}=-\frac1N\sum_{i}\sum_j\log\sigma\big(z_{ij}(t\,I_i\cdot T_j + b)\big),\qquad z_{ij}=+1 \text{ if } i=j \text{ else } -1$$

▸ **The softmax requires a global normalization over the batch, forcing an expensive all-gather across devices. The sigmoid loss is fully decomposable per pair**, so it works at small batch sizes and scales without the communication cost. SigLIP matches or beats CLIP at much smaller batches, and is the default image encoder in many current VLMs.

#### Unpacking the sigmoid loss

Read the formula aloud first: *"minus one over N, sum over i, sum over j, of the log of sigmoid of z-i-j times the quantity t times I-i dot T-j plus b."*

| Piece | What it is |
|---|---|
| $\sigma(x) = 1/(1+e^{-x})$ | The sigmoid. Turns any real number into a probability in $(0,1)$ |
| $t$ | A learned **scale**, playing the role $1/\tau$ played in CLIP |
| $b$ | A learned **bias** — new, and load-bearing (see below) |
| $`z_{ij} = \pm1`$ | $+1$ for a true pair, $-1$ otherwise |
| $`\sum_i\sum_j`$ | Over **all $N^2$ cells**, not just each row |

**The one structural change.** CLIP asks, for each row, *"which one of these $N$ captions is correct?"* — a single question with $N$ competing answers, which is why a softmax is needed and why every cell in the row must be known before any cell can be scored. SigLIP instead asks, for each of the $N^2$ cells independently, *"is this particular pair a match — yes or no?"*

▸ **One $N$-way question becomes $N^2$ independent yes/no questions.** That is the entire idea, and everything else follows from it.

**Why $`z_{ij}`$ multiplies the logit.** It is a compact way to write "for a positive pair, push the score up; for a negative pair, push it down." If $z = +1$, you are maximizing $\log\sigma(\text{score})$, which grows as the score grows. If $z = -1$, you are maximizing $\log\sigma(-\text{score})$, which grows as the score *shrinks*. One line covers both cases — the same trick used in the hinge loss of a support vector machine.

**Put numbers in it.** Take a partly-trained model with $t = 10$ and $b = -5$:

| Pair | Cosine | $t\cdot\cos + b$ | $z$ | Argument to $\sigma$ | $\sigma$ | Loss $=-\ln\sigma$ |
|---|---|---|---|---|---|---|
| True pair | $0.60$ | $1.0$ | $+1$ | $1.0$ | $0.731$ | $0.313$ |
| Easy negative | $0.10$ | $-4.0$ | $-1$ | $4.0$ | $0.982$ | $0.018$ |
| **Hard** negative | $0.45$ | $-0.5$ | $-1$ | $0.5$ | $0.622$ | $0.474$ |

▸ The easy negative contributes almost nothing; the hard negative contributes more than the true pair does. **The loss automatically spends its gradient on the confusable cases** — the same self-focusing property that makes the softmax version work, but obtained per-pair.

**Why the bias $b$ exists, and why it must start very negative.** Count the cells: $N^2$ total, of which exactly $N$ are positive. At $N = 32{,}768$ that is one positive per 32,768 cells — a $0.003\%$ positive rate. A binary classifier facing that imbalance and initialized neutrally will be swamped by the negatives on step one and simply learn to output "no" for everything. Initializing $b$ to a large negative value means the model *starts* predicting "no" everywhere, so the early gradient is dominated by the positives it is getting wrong, which is exactly the signal you want. **The bias is not a detail; without it, training at large $N$ is badly behaved from the first step.**

**Why "the softmax requires a global normalization" is a systems problem.** To compute $\mathrm{softmax}$ across row $i$ you need $`\sum_{j=1}^{N}e^{S_{ij}}`$ — every entry in the row. Split a batch of 32,768 across 32 devices and each device holds only 1,024 text embeddings, so **no device can compute a single row of the loss on its own**. Every device must receive every other device's embeddings: an all-gather, on every step, with a synchronization barrier.

▸ **And the matrix itself is large.** $32{,}768^2 = 1.07$ billion similarity scores; in 32-bit floats that is **4.3 GB** for one intermediate tensor in one loss function.

The sigmoid loss needs no row sums. Device 3 holding image chunk 3 and text chunk 7 can compute those $1024 \times 1024$ cells, add their contribution to the running loss, and discard them. In practice implementations pass text chunks around the devices in a ring, so each device only ever materializes a small tile. **The loss became embarrassingly parallel because the normalization went away.**

#### Examples and non-examples: losses that need a global normalization

**✅ Need the whole batch (or the whole vocabulary)**

| Example | What it must sum over |
|---|---|
| Softmax cross-entropy over a vocabulary | All $\lvert V\rvert$ logits, to get the denominator |
| CLIP's InfoNCE | All $N$ captions in the row |
| Batch normalization | The batch mean and variance — every example in the batch |

**❌ Look like they need it, but don't**

| Looks global | Why it's actually local | Consequence |
|---|---|---|
| SigLIP's pairwise sigmoid | Each cell is scored on its own | Scales to any batch size; no all-gather |
| Binary cross-entropy for multi-label classification | Each label is an independent yes/no | You can add a label without retraining the others |
| Layer normalization | Normalizes across **features within one example** | Works at batch size 1 — the reason transformers use it |
| Noise-contrastive estimation with fixed noise samples | Compares against a sampled set, not the full space | The original 2010 motivation: dodging an intractable normalizer |

▸ **The boundary:** a loss needs a global normalization exactly when its output is a **distribution over mutually exclusive alternatives**, because probabilities must sum to 1 and you cannot sum without seeing everything. The moment you reframe the task as independent binary decisions, the constraint evaporates — and with it, the communication cost.

> **Where this came from.** SigLIP came from Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer at Google DeepMind in 2023, in a paper whose entire contribution is replacing one loss function with another. The most quoted result is not the accuracy number but the practicality one: they showed the batch-size advantage of contrastive training **saturates** far earlier than people assumed — performance plateaus in the low tens of thousands rather than continuing to climb — which retired a widely held belief that image–text pretraining was permanently the province of labs with very large clusters. The same group's shape-optimized ViT-SO400M, produced by searching over width/depth/MLP ratios instead of scaling them uniformly, became one of the most widely reused vision encoders in open vision–language models.

---

## 28.3 Vision–language models

### The architectural patterns

| Pattern | Mechanism | Examples |
|---|---|---|
| **Frozen encoder + projector** | image → ViT → linear/MLP → tokens prepended to the LLM's input | **LLaVA**, most open VLMs |
| **Cross-attention** | LLM layers cross-attend to visual features via inserted gated layers | Flamingo, Idefics |
| **Q-Former / resampler** | learned queries compress $N$ patch tokens to $k$ tokens | BLIP-2, Qwen-VL |
| **Early fusion** | image and text tokens interleaved from layer 1, trained jointly | Chameleon, Fuyu |

#### The four patterns, decoded

All four are answers to one question: **an LLM eats a sequence of $d$-dimensional vectors; how do you get an image in there?**

**1. Frozen encoder + projector — "translate the image into fake words."**

The vision encoder emits, say, 576 vectors of dimension 1024. The LLM expects vectors of dimension 4096. A projector — literally one or two matrices — maps $1024 \to 4096$. Then you *concatenate*: the LLM's input becomes `[576 image-derived vectors] + [the tokenized user question]`, and the LLM has no idea that the first 576 entries didn't come from its embedding table.

```python
# The whole of the LLaVA-style interface
feats  = vision_encoder(image)        # (576, 1024), frozen
visual = projector(feats)             # (576, 4096), trained
text   = llm.embed(tokenizer(prompt)) # (L, 4096)
out    = llm(torch.cat([visual, text], dim=0))
```

▸ **The projector is a two-matrix adapter — often under 20 million parameters — sitting between two frozen models totalling tens of billions.** That ratio is why this pattern dominates: the expensive parts are downloaded, not trained.

**2. Cross-attention — "let the LLM ask the image questions."**

Instead of putting image vectors in the input sequence, insert new layers into the LLM whose queries come from the text stream and whose keys and values come from the visual features. The text sequence length never grows.

▸ **The trade:** cross-attention costs no context-window budget — a 100-image document does not consume 57,600 tokens — but it requires **modifying the LLM's internals**, which means you cannot swap in a new base model by changing a config line.

Flamingo's version has a detail worth knowing: each inserted cross-attention block ends with a $\tanh$ gate whose scalar is **initialized to zero**. Since $\tanh(0) = 0$, the block contributes literally nothing at initialization, so the model at step 0 is *bit-for-bit the original language model*. The visual pathway then fades in as the gate learns. This is the same idea as a residual branch initialized to zero, and it is why the language ability isn't destroyed on the first few thousand steps.

**3. Q-Former / resampler — "compress before you speak."**

Create $k$ learned query vectors (BLIP-2 uses 32). Run a small transformer where those queries cross-attend to the $N$ patch features. Output: exactly $k$ vectors, regardless of $N$.

▸ $576 \to 32$ is an **18× reduction in context consumed**. Since LLM attention is quadratic, that is roughly a $324\times$ reduction in the attention cost the image imposes on the language model. The price is that 32 vectors cannot carry what 576 carried — which is precisely why resampler-based models struggle with dense tasks like reading a spreadsheet.

**4. Early fusion — "there is only one kind of token."**

Quantize the image into discrete codes with a VQ tokenizer, add those codes to the text vocabulary, and train one transformer on interleaved sequences from scratch. There is no encoder and no projector because there is no boundary. The model can *generate* image tokens as naturally as it generates words.

▸ The cost is that you cannot start from a strong pretrained text model — the vocabulary and the whole training distribution changed — so you pay full pretraining cost and typically end up with weaker text reasoning at equal budget.

#### Examples and non-examples: "multimodal model"

**✅  multimodal**

| Example | Why it qualifies |
|---|---|
| LLaVA answering "what is written on the sign?" | A single forward pass reasons jointly over pixels and words |
| CLIP | One shared embedding space that both modalities are trained into |
| Flamingo doing few-shot learning from interleaved image–text prompts | Text conditions on images and vice versa within one model |
| Whisper (audio + text) | Audio encoder and text decoder trained end to end together |

**❌ Near-misses — pipelines that look multimodal**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| **Run OCR, paste the extracted text into GPT** | The language model never sees a pixel. Anything OCR dropped is gone forever | A **cascade**. Errors compound and cannot be recovered downstream |
| **A captioning model feeding an LLM** | Same failure: the caption is a lossy bottleneck chosen without knowledge of the question | A cascade with a natural-language interface |
| **A CNN classifier plus a template sentence** | No joint reasoning; the text is generated by string formatting | A classifier with a print statement |
| **Two separate models with a router** ("if there's an image, use the vision model") | Neither model sees both modalities at once | An ensemble / dispatcher |
| **Text-to-image generation via a frozen text encoder** (e.g. a diffusion model conditioned on T5) |  multimodal at the system level, but the text encoder is never updated by image gradients | Conditional generation — a real and important case, but the alignment is one-directional |

▸ **The boundary:** a model is multimodal if **gradients flowed between the modalities during training**, so that one modality's representation was shaped by the other. A cascade can be built out of two unimodal models in an afternoon and will beat a bad VLM on easy tasks; it fails exactly where the answer depends on something the intermediate representation didn't preserve — layout, colour, spatial relation, a smudge, the thing the captioner didn't consider important.

> **Common misconception.** *"The vision encoder tells the LLM what's in the image."* It hands over a set of vectors, and those vectors are whatever CLIP-style training found useful for matching captions. If the pretraining captions never described the number of windows on a building, that information may simply not survive the encoder — and no amount of LLM cleverness recovers it. **The LLM cannot perceive what the encoder discarded**, which is why "the vision encoder is a hard bottleneck" appears in the failure-mode list below rather than as an afterthought. The misconception is tempting because the failures *look* like reasoning failures: the model produces a fluent, confident, wrong sentence, which is the signature of a language model doing its best with missing input.

### The LLaVA recipe — the one to know

1. **Stage 1 (alignment):** freeze both the vision encoder and the LLM; train **only the projector** on image–caption pairs. Cheap, and it teaches the projector to speak the LLM's embedding language.
2. **Stage 2 (instruction tuning):** unfreeze the LLM (and sometimes the encoder); train on multimodal instruction data.

▸ **Why a frozen encoder plus a trained projector works at all:** CLIP's image embedding already contains language-aligned semantics, so the projector's job is a change of basis, not a change of content. This is why VLMs are so much cheaper to build than their capability suggests.

#### "A change of basis, not a change of content," unpacked

This sentence carries the whole section, so here it is slowly.

A **basis** is a choice of coordinate axes. The vector describing a point in a room is different depending on whether you measure from the north-west corner in metres or the door frame in feet — but **it is the same point**. A change of basis is a matrix that converts one description into the other. It moves no information; it renames it.

The claim is that the semantic content the LLM needs — "dog," "outdoors," "wearing a hat" — is *already present* in CLIP's 1024-dimensional image vector, because CLIP was trained specifically to make that vector align with a sentence describing exactly those things. The LLM writes its concepts in a different 4096-dimensional coordinate system. **The projector is a phrasebook between two languages that already contain the same words.**

> **Analogy.** You have hired a brilliant analyst who only reads French and a brilliant researcher who only writes German. You do not need to re-educate either of them. You need a translator — and if the two are discussing the same subject with the same concepts, that translator can be far less capable than either specialist and the collaboration still works. Stage 1 of LLaVA trains the translator; stage 2 lets the analyst adjust her habits now that she is being read.

▸ **This is also a precise prediction, and it holds.** If the theory is right, a *randomly initialized* vision encoder should not work with a small projector, because the content wouldn't be there to re-express. It doesn't. And a **non-language-aligned** encoder like DINOv2 — excellent features, but never trained against text — needs more projector capacity and more alignment data than CLIP does. The theory earns its keep by making the difference predictable.

**Put costs on the two stages.**

| Stage | Trained | Rough parameter count in play | Data |
|---|---|---|---|
| 1 — alignment | Projector only | $\sim10$–$20$M | Hundreds of thousands of image–caption pairs |
| 2 — instruction tuning | LLM (+ sometimes encoder) | $7$–$70$B | $\sim10^5$ multimodal instruction examples |

▸ **Compare to CLIP's own pretraining: 400 million pairs.** A capable open VLM is assembled from two downloaded checkpoints plus a couple of days of training on a data budget four orders of magnitude smaller. **The capability was paid for by someone else, twice, and the projector is the receipt.**

> **Where this came from.** LLaVA — *Visual Instruction Tuning*, by Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee (University of Wisconsin–Madison and Microsoft Research) in 2023 — is best known for its architecture, but its more consequential trick was how it got training data. Multimodal instruction data barely existed. So the authors took COCO images, which come with human-written captions and object bounding boxes, and fed **only the text** — the captions and the box coordinates as numbers — to a **text-only GPT-4**, asking it to invent conversations, detailed descriptions, and reasoning questions about the scene. GPT-4 never saw a single image. Roughly 158,000 examples were produced this way, and that synthetic set was enough to turn a frozen CLIP encoder and a chat LLM into a working visual assistant. It is one of the cleanest demonstrations that **a description can stand in for perception when you are generating supervision rather than performing the task.**

#### Examples and non-examples: what a "frozen" encoder means

**✅  frozen**

| Example | What it implies |
|---|---|
| `for p in vision_encoder.parameters(): p.requires_grad = False` | No gradient is ever applied to those weights |
| Precomputing and caching all image features to disk before training | The strongest form — the encoder isn't even in the training loop |

**❌ Near-misses**

| Looks frozen | Why it isn't | What it actually is |
|---|---|---|
| Encoder in `eval()` mode but gradients still flowing | `eval()` only changes dropout and batch-norm behaviour; it does **not** stop learning | A very common bug |
| Freezing the encoder but training its final LayerNorm | Those are parameters, and normalization parameters are unusually influential | Partial unfreezing (sometimes deliberate and effective) |
| Freezing the encoder while training a LoRA adapter on it | The effective function changed | Parameter-efficient fine-tuning |
| Freezing during stage 1 and unfreezing in stage 2 | Frozen for part of training only | Staged unfreezing — the standard recipe |

▸ **The boundary:** frozen means **the weights at the end are bit-for-bit the weights at the start**. If the function computed by the module changed at all, it wasn't frozen, and any claim about training cost or about "reusing the pretrained representation" needs re-checking.

### The design questions that matter

**How many visual tokens?** A $336\times336$ image at patch 14 gives 576 tokens — a large fraction of the context, and attention is quadratic. Resamplers compress to 32–256. **High resolution is essential for OCR, charts, and documents**, so the standard trick is **dynamic tiling** (AnyRes): split a high-resolution image into tiles, encode each, plus a downscaled global view.

**Which encoder?** CLIP/SigLIP ViT-L or ViT-SO400M. Some models ensemble a CLIP encoder (semantics) with a DINOv2 encoder (spatial detail).

**Native multimodal vs bolted-on.** Training on interleaved image–text from scratch gives better integration; adapting a strong text LLM is far cheaper and currently gives better text reasoning. Most open models take the second path.

#### The visual-token budget, in arithmetic

This is the design constraint that shapes every VLM, and it is pure counting.

**One image, standard resolution.** $336 \times 336$ pixels, patch size 14. Along each side: $336/14 = 24$. So $24 \times 24 = 576$ tokens.

**What that costs in a 4,096-token context:** 576 tokens is $14\%$ of the window — gone before the user has typed a word.

**Now go high-resolution, which you must for documents.** A screenshot of a spreadsheet downsampled to $336\times336$ has text roughly 3 pixels tall. It is not "hard to read"; the information is *destroyed by the resize*, before the model runs.

**Dynamic tiling (AnyRes), concretely.** Take a $672 \times 672$ image. Cut it into four $336\times336$ tiles, encode each ($4 \times 576 = 2{,}304$ tokens), and additionally downscale the whole image to $336\times336$ and encode that too as a "global view" (another 576). Total: **2,880 tokens for one image.**

| Configuration | Visual tokens | Fraction of a 4,096 context |
|---|---|---|
| $224^2$, patch 16 | 196 | 5% |
| $336^2$, patch 14 | 576 | 14% |
| AnyRes, $2\times2$ tiles + global | 2,880 | **70%** |
| AnyRes, $3\times3$ tiles + global | 5,760 | **Exceeds it** |
| Q-Former, 32 queries | 32 | 0.8% |

▸ **Why the global view is not optional.** Four tiles encoded independently have no idea they are neighbours — each is processed as if it were a standalone photograph. A table spanning the seam between tiles 1 and 2 appears to the model as two unrelated fragments. The downscaled global view is a low-resolution map that lets the language model stitch the pieces back together.

**Why the quadratic cost is worse than it sounds.** Attention over 2,880 visual tokens plus 1,000 text tokens is $3{,}880^2 \approx 15.1$ million pairwise scores per head per layer. The same query with a 32-token resampler is $1{,}032^2 \approx 1.1$ million — a **14× reduction in attention work**, at the cost of throwing away almost everything that made the high resolution worth having.

▸ **This is the central, unresolved tension in VLM design:** resolution and context length trade directly against each other, and the right point on that curve depends entirely on the task. Document understanding wants every token it can get; captioning a photograph of a beach does not.

> **Common misconception.** *"Higher input resolution means the model sees more detail."* Only if the token count rises with it. Feeding a $1024\times1024$ image to an encoder with fixed $224\times224$ input just means a bigger downscale happens inside the preprocessing pipeline — the extra pixels are discarded before layer 1. **Resolution is only real if it survives to become tokens.** The misconception is tempting because every other part of the stack (cameras, displays, file sizes) treats resolution as a property of the image rather than of the model's interface.

#### Examples and non-examples: tasks that need high resolution

**✅  resolution-bound**

| Task | Why |
|---|---|
| Reading a receipt total | The digits occupy a few dozen pixels in the original |
| Reading axis labels on a chart | Small text, and getting it wrong changes the answer completely |
| Counting cells in a microscopy image | The objects are near the resolution limit by construction |
| Detecting a manufacturing defect | The defect is, definitionally, a small local anomaly |

**❌ Near-misses — feel like they need resolution, but don't**

| Task | Why it isn't resolution-bound | What it's actually bound by |
|---|---|---|
| "Is this photo indoors or outdoors?" | Solvable from a $32\times32$ thumbnail | Nothing — it is easy |
| "How many people are in this crowd of 200?" | Doubling resolution doesn't help; the model has no counting mechanism | **Counting ability** — an architectural and training-data gap |
| "Is the cup to the left of the laptop?" | Both objects are large and clearly visible | **Spatial reasoning**, which position embeddings and training data govern |
| "What year was this building built?" | The information isn't in the pixels at any resolution | **World knowledge** |

▸ **The boundary:** a task is resolution-bound when the deciding evidence occupies few enough pixels that downsampling removes it. Everything else that fails at high resolution was going to fail anyway — and adding tiles will cost you 2,880 tokens to learn that.

### Failure modes

- **Object hallucination** — describing objects that aren't present, driven by language priors ("a kitchen usually has a sink"). Measured by POPE. Mitigations: contrastive decoding against a blank image, and grounding-focused training data.
- **Fine-grained spatial reasoning** and precise counting.
- **Text-only degradation** after multimodal tuning; mix in text-only data.
- **The vision encoder is a hard bottleneck** — the LLM cannot perceive what the encoder discarded.

#### Object hallucination, decoded

**What POPE actually does.** POPE (Polling-based Object Probing Evaluation) sidesteps the difficulty of grading free-form captions by turning the question into a yes/no poll: given an image, ask *"Is there a sink in the image?"* for objects that are present and for objects that are not. Accuracy on the "not present" questions is the hallucination rate, measured without a human in the loop.

The clever part is **how the absent objects are chosen**. Picking them at random is too easy — "is there a submarine in this kitchen?" is answerable from priors alone. So POPE also samples absent objects that are **statistically likely to co-occur** with what *is* in the image. In a kitchen, it asks about a sink. That is the adversarial version, and models do dramatically worse on it.

▸ **Which tells you the mechanism precisely.** The model is not misperceiving; it is **completing a scene from language statistics**. The LLM has read enough text to know kitchens have sinks, and when the visual evidence is weak or absent, the language prior fills the gap. Hallucination here is the language model doing exactly what it was trained to do — predict plausible continuations — applied where perception should have overruled it.

**Why contrastive decoding against a blank image works.** Run the model twice: once with the real image, once with a black or noise image. The second run's output distribution is *pure language prior* — whatever the model would say with no visual evidence at all. Subtract it (in log space) from the first, and what remains is the part of the prediction that the image actually caused:

$$\text{score} = \log p(\text{token}\mid \text{image}) - \alpha\log p(\text{token}\mid \text{blank})$$

▸ A token like "sink," which scores high both with and without the image, gets its advantage cancelled. A token that scores high **only** when the real image is present survives. **It is a differencing experiment run at inference time**, and it needs no retraining — just a second forward pass.

#### Examples and non-examples: hallucination

**✅  hallucination**

| Example | Why it qualifies |
|---|---|
| Describing a sink in a kitchen photo that has no sink | Confident assertion of a specific absent object |
| Reading "$47.99" off a receipt that says "$41.99" | Plausible completion overriding the actual pixels |
| Inventing a plausible caption for a blank image | No evidence at all, output produced anyway |

**❌ Near-misses — errors that get called hallucination but have different causes and different fixes**

| Looks like hallucination | Why it isn't | What it actually is |
|---|---|---|
| Model says "I see a small dark shape, possibly a cat" for a  ambiguous blur | It expressed uncertainty and the uncertainty was warranted | **Correct calibration** (Ch. 33) |
| Model can't read 4-pixel-tall text and guesses | The information was destroyed before layer 1 | A **resolution / encoder bottleneck** — no decoding trick fixes it |
| Model states a wrong fact about a correctly identified landmark | Perception succeeded; the world knowledge was wrong | A **factual error in the LLM**, identical to a text-only error |
| Model refuses to answer a visible question | Over-conservative alignment | A **refusal**, tuned in during post-training |
| Model describes an object that *is* present but that the benchmark's annotator failed to label | The model is right | A **label error in the evaluation set** |

▸ **The boundary:** hallucination is a **confident assertion whose evidence is absent from the input**. The diagnostic question is: *would a human with the same input have known?* If the pixels  never contained the answer, the failure belongs to the encoder or the resolution, not to the generator — and the fix is a different one entirely.

> **Common misconception.** *"Hallucination means the model is lying or making things up."* There is no separate "making things up" mode. A language model always produces the most probable continuation given its context; when the context includes strong visual evidence, that evidence dominates, and when it doesn't, the linguistic priors do. **Hallucination and correct answering are the same computation with different amounts of evidence.** The misconception is tempting because the output *sounds* like a claim, complete with the confident register of a factual statement — but that register is a property of the training corpus, not a signal about the model's internal certainty.

---

## 28.4 Audio and speech

### Representations

Raw waveform (16 kHz) → **mel spectrogram**: STFT with a ~25 ms window and 10 ms hop, magnitude, then a mel filterbank (spacing ~$`2595\log_{10}(1+f/700)`$, approximating human pitch perception), then log. ▸ **A spectrogram is an image, so every vision architecture applies** — which is exactly how the field developed.

#### Building a mel spectrogram, one step at a time with numbers

Every term in that sentence, in order, with arithmetic.

**"Raw waveform (16 kHz)."** Sound is air pressure changing over time. A microphone measures that pressure 16,000 times per second. So **one second of audio is a list of 16,000 numbers**, and nothing more. Ten seconds is 160,000 numbers.

▸ That is already the problem. A transformer over 160,000 tokens is hopeless — $160{,}000^2 = 2.6\times10^{10}$ pairwise scores — and worse, individual pressure samples carry almost no meaning on their own. A single sample says nothing about whether someone said "cat."

**"STFT with a ~25 ms window and 10 ms hop."** STFT is the **short-time Fourier transform**. The Fourier transform answers "which frequencies is this signal made of?" — it converts a wiggle over time into a list of how much of each pitch is present. But a whole sentence's Fourier transform is useless, because it tells you the frequencies present *somewhere* with no indication of when.

So you do it in slices:

| Quantity | Value | In samples at 16 kHz |
|---|---|---|
| Window length | 25 ms | $0.025 \times 16{,}000 = 400$ samples |
| Hop (step between windows) | 10 ms | $0.010 \times 16{,}000 = 160$ samples |
| Overlap | 15 ms | 240 samples — consecutive windows **share 60% of their content** |
| Frames per second | — | $1/0.010 = \mathbf{100}$ |

▸ **The compression is immediate: 16,000 samples per second become 100 frames per second — a 160× reduction along the time axis.**

**Why 25 ms specifically?** It is a compromise forced by physics. Too short and you cannot resolve low frequencies at all — a 100 Hz tone completes one cycle in 10 ms, so a 5 ms window doesn't contain a whole cycle of it and the transform cannot see it. Too long and the sound changes within the window, smearing distinct phonemes together. Speech is roughly stationary over 20–30 ms, so that is where the window sits. **This is the uncertainty principle in an engineering costume: you cannot have arbitrarily fine time resolution and arbitrarily fine frequency resolution simultaneously.**

**Why overlapping windows?** Without overlap, a consonant burst landing on a window boundary is split across two frames and half-destroyed in each. Overlap ensures every moment of the signal sits near the centre of at least one window.

**"Magnitude."** The Fourier transform returns complex numbers, each carrying an amplitude *and* a phase. Taking the magnitude throws the phase away. This is a real, deliberate information loss — it is why converting a spectrogram back to audio requires a phase-reconstruction algorithm or a learned vocoder — and it is done because phase is perceptually near-irrelevant for recognition while being extremely hard to model.

**"Then a mel filterbank."** You now have maybe 201 frequency bins per frame, linearly spaced from 0 to 8,000 Hz. Human hearing is not linear in frequency, so you sum groups of bins into ~80 **mel** bands whose widths follow perception. Put actual numbers through the formula $`m = 2595\log_{10}(1 + f/700)`$:

| Frequency $f$ (Hz) | Mel value $m$ |
|---|---|
| 100 | 150 |
| 1,000 | 1,000 |
| 2,000 | 1,522 |
| 4,000 | 2,146 |
| 8,000 | 2,840 |

▸ **Read the gaps, not the values.** Going from 100 Hz to 1,000 Hz — a span of 900 Hz — covers 850 mel. Going from 4,000 Hz to 8,000 Hz — a span of 4,000 Hz, over four times wider — covers only 694 mel. **The mel scale says that 900 Hz down low matters more perceptually than 4,000 Hz up high**, which is exactly why you can hear the difference between a bass note and the note a semitone above it, but two high whistles a hundred hertz apart sound nearly identical. (The scale is calibrated so that 1,000 Hz maps to 1,000 mel exactly — the anchor point.)

**"Then log."** Human loudness perception is logarithmic: the step from a whisper to a conversation *sounds* like the step from a conversation to a shout, though the energy ratio is enormous. Taking the log compresses a dynamic range spanning several orders of magnitude into a range a neural network can handle without the loud frames dominating every gradient.

**The final shape.** Ten seconds of audio → $1{,}000$ frames × $80$ mel bands = an $80 \times 1000$ grid of numbers. **That is an image**: one axis is time, one is frequency, and each cell is a brightness. Feed it to a CNN or patchify it for a ViT — the code does not need to know it came from a microphone.

> **Analogy.** A spectrogram is sheet music that a machine wrote by listening. The horizontal axis is time, the vertical axis is pitch, and darkness is loudness. Speech looks like horizontal stripes (the resonances of a vocal tract) that bend as the mouth changes shape; a hand clap looks like a single vertical line; a violin looks like an evenly spaced stack of horizontal lines. **Once you can read one, you can identify a great deal of audio with your eyes.**

> **Where this came from.** The mel scale came out of a psychology experiment, not an engineering one. Stanley Smith Stevens, John Volkmann, and Edwin Newman at Harvard's Psycho-Acoustic Laboratory published it in 1937: they played listeners a reference tone and asked them to adjust a second tone until it sounded **half as high**, then built a scale from the answers. The name is short for *melody*. The particular formula $`2595\log_{10}(1+f/700)`$ is a later analytic fit to that curve rather than the original authors' equation, and several slightly different versions circulate — which is worth knowing, because two libraries can produce different mel spectrograms from identical audio and both be correct. Separately, the **sound spectrograph** — the machine that first drew audio as a picture — was developed at Bell Labs during the Second World War and declassified afterwards; one of its stated motivations was teaching deaf people to read speech visually, published as *Visible Speech* in 1947. The representation that every modern speech model consumes was designed to be looked at by human eyes.

#### Examples and non-examples: "a spectrogram is an image"

**✅ Where the analogy holds**

| Example | Why |
|---|---|
| Running a ResNet or ViT over the mel grid | The operation is well-defined and works extremely well |
| Using 2-D convolutions to find local time–frequency patterns | Formants and transients  are local 2-D structures |
| Applying SpecAugment — masking rectangles of time and frequency | The augmentation analogy transfers directly, and it is one of the most effective ASR augmentations known |

**❌ Where the analogy breaks**

| Looks like image intuition applies | Why it doesn't | Consequence |
|---|---|---|
| **Translation invariance in the vertical (frequency) axis** | Shifting a photo of a dog up 20 pixels leaves a dog. Shifting a spectrogram up 20 bands changes the speaker's voice, and can change the phoneme | Frequency is a **semantic** axis, not a spatial one. Vertical translation invariance is wrong here |
| **Flipping horizontally as augmentation** | Reversed speech is not speech | Time has a direction; space doesn't |
| **The two axes have the same units** | One is milliseconds, one is mel | A square patch is not a square region of anything real |
| **Resizing to a fixed square** | Rescaling time changes speaking rate; rescaling frequency changes pitch | Both are audible distortions, not benign resamplings |
| **A pixel value is what it seems** | It's log-energy in a perceptually warped band | Interpreting brightness as "amount of sound" is only roughly true |

▸ **The boundary:** a spectrogram supports every image *operation* but only some image *assumptions*. **The machinery transfers; the invariances do not.** This is why speech models that copy vision architectures wholesale still tend to alter the augmentation and the positional handling.

### The models

**wav2vec 2.0** — self-supervised speech. A CNN feature extractor, a transformer context network, masked spans, and a **contrastive loss against quantized latent targets** (via Gumbel-softmax over a product codebook). Result: state-of-the-art ASR with 10 minutes of labelled audio, from 53k hours of unlabelled. **HuBERT** is the same idea with offline k-means clustering providing discrete targets, iteratively refined — simpler and usually better.

**Whisper** — the opposite bet: plain encoder–decoder transformer on log-mel input, trained *supervised* on 680k hours of weakly-labelled multilingual web audio, with special tokens for the task (transcribe/translate), language, and timestamps. ▸ **The lesson is about data, not architecture: massive weak supervision beat elaborate self-supervision for robustness.** Whisper generalizes across accents, noise, and domains far better than models trained on clean curated corpora.

**Neural audio codecs (SoundStream, EnCodec, DAC)** — residual vector quantization produces discrete audio tokens at ~1.5–6 kbps. ▸ **This is the enabling technology for audio language models**: once audio is tokens, an autoregressive transformer generates speech and music the same way it generates text. AudioLM, VALL-E, MusicGen, and modern TTS all rest on it.

**Speech LLMs** now do end-to-end speech-to-speech with no intermediate text, which preserves prosody, emotion, and speaker identity and removes ASR/TTS latency.

#### The three bets, side by side

These models are not variations on a theme; they are three different answers to *where does the supervision come from?*

| | wav2vec 2.0 | HuBERT | Whisper |
|---|---|---|---|
| Supervision | Self-supervised (contrastive) | Self-supervised (masked prediction of cluster IDs) | **Supervised**, weakly |
| Training audio | ~53k hours unlabelled | Similar | **680k hours** weakly labelled |
| Targets | Quantized latents of the audio itself | Offline k-means cluster assignments | Actual transcripts scraped from the web |
| Labelled data needed downstream | As little as 10 minutes | Similar | **None** — it transcribes out of the box |
| Where it shines | Low-resource languages; fine-tuning | Same, usually a bit better | Robustness to accent, noise, domain |

▸ **Scale check:** 53,000 hours is about 6 years of continuous audio. 680,000 hours is about **78 years** — more than a human lifetime of listening, assembled from the web.

**"Masked spans" and why they're spans, not single frames.** Masking one 10 ms frame is trivially easy to fill in: the frames on either side overlap it by 60%, so the answer is essentially copied from the neighbours. Speech is smooth over short intervals; a task that can be solved by interpolation teaches nothing. So wav2vec 2.0 masks *contiguous runs* — long enough that the model must use phonetic and lexical structure rather than local continuity.

> **This is the exact same lesson as masking in vision.** BERT masks 15% of tokens because words are discrete and individually informative; MAE masks **75%** of image patches because patches are so redundant that masking a few is solvable by copying. **The optimal mask difficulty is set by the redundancy of the medium**, and audio, like images, is highly redundant.

**Why quantized targets at all?** You cannot ask a model to predict a raw continuous audio slice — the target has infinite detail, most of it irrelevant (microphone noise, room acoustics, exact phase), and a regression loss would spend all its capacity there. Quantizing to a discrete codebook forces a decision about *which category of sound* this is, throwing away the nuisance detail. **HuBERT's insight is that you do not need anything sophisticated to produce those categories**: run k-means on the features, use the cluster IDs as targets, retrain, re-cluster on the better features, repeat. Simpler than a learned quantizer and usually better.

#### Examples and non-examples: self-supervised vs. weakly supervised

**✅  self-supervised**

| Example | Why it qualifies |
|---|---|
| wav2vec 2.0 predicting masked spans of its own audio | The target is derived entirely from the input |
| HuBERT predicting k-means cluster IDs computed from its own features | No human ever labelled anything |
| MAE reconstructing masked image patches | The pixels are their own labels |
| Next-token prediction on raw text | The next token is the label |

**❌ Near-misses**

| Looks self-supervised | Why it isn't | What it actually is |
|---|---|---|
| **Whisper** on web audio with subtitles | A human wrote those subtitles, however carelessly | **Weakly supervised** — real labels, noisy and free |
| **CLIP** on image–caption pairs | A human wrote the caption | Weak supervision from **naturally occurring** annotation |
| Fine-tuning a self-supervised model on 10 minutes of transcripts | Those 10 minutes are labelled | Self-supervised pretraining **plus** supervised fine-tuning |
| Training on outputs of another model (distillation) | The teacher's knowledge came from labels | **Distillation** — supervision laundered through a model |
| Contrastive learning with in-batch negatives | The pairing itself is the label | Self-supervised **only if** the pairing is automatic (two crops of one image); weakly supervised if a human paired them |

▸ **The boundary:** it is self-supervised if you could run it on a hard drive of raw, completely unannotated data. **The test is whether a human ever wrote anything down.** Web subtitles were written by humans — they are just free, plentiful, and bad, which turns out to be a very good trade.

> **Common misconception.** *"Whisper beat the self-supervised models, so self-supervision doesn't work."* Whisper won on **robustness out of the box** by using thirteen times more audio with real transcripts attached. That option only exists for languages and domains where the web is full of subtitled audio. For a language with 200,000 speakers and no subtitle corpus, wav2vec-style pretraining on unlabelled recordings plus ten minutes of transcription is still the only method that works at all. **The lesson is that weak supervision at extreme scale beats clever self-supervision at modest scale — not that self-supervision is obsolete.** The misconception is tempting because benchmark leaderboards are dominated by English, where the data asymmetry is at its most extreme.

#### Neural audio codecs and residual vector quantization

**The problem.** You want audio as *tokens* so a transformer can generate it. A single codebook would need to be astronomically large: to represent 20 ms of audio faithfully with one lookup, you would need a codebook covering every distinguishable 20 ms sound.

**The trick — quantize the leftovers.** Encode the audio to a vector. Find the nearest entry in codebook 1 and emit its index. Now compute the **residual**: what codebook 1 failed to capture. Quantize *that* with codebook 2. Emit that index. Repeat.

▸ **With 8 codebooks of 1,024 entries each, you store $8 \times 1{,}024 = 8{,}192$ vectors but can express $1{,}024^8 \approx 1.2\times10^{24}$ distinct combinations.** That is the entire reason residual quantization exists: exponential expressiveness from linear storage.

> **Analogy.** Making change. To pay £8.73 you don't need a coin for every amount between 0 and 10 pounds — you take the largest coin that fits (£5), then the largest that fits the remainder (£2), then £1, then 50p, and so on. Eight "codebooks" of coins covers every amount to the penny. Each stage handles what the previous stage couldn't.

**Put the rates together.** A typical configuration runs at 50 frames per second with 8 codebooks — that is **400 tokens per second of audio**. At 10 bits per index (1,024 entries) the bitrate is $400 \times 10 = 4{,}000$ bits/s, or 4 kbps. Compare raw 16 kHz 16-bit audio at 256 kbps: a **64× compression**, and the result still sounds like speech.

▸ **Now the consequence for language models.** A 3-minute song is $180 \times 400 = \mathbf{72{,}000}$ tokens. A 10-second reply is 4,000 tokens. **Audio is dramatically more token-hungry than text** — a 3-minute song costs more context than a short novel chapter — which is why audio language models lean on hierarchical schemes: a coarse model that predicts only codebook 1 (semantic content, 50 tokens/s) and a separate stage that fills in codebooks 2–8 (acoustic detail).

> **Where this came from.** Vector quantization for speech is old engineering, not a deep-learning invention: the Linde–Buzo–Gray algorithm for designing codebooks dates to 1980, and multi-stage (residual) quantization was standard in telephony codecs decades before neural networks were involved, for precisely the reason above — you cannot afford a single large codebook in a device that has to fit in a handset. What changed is that the encoder and decoder around the quantizer became learned neural networks (SoundStream at Google in 2021, EnCodec at Meta in 2022), and that the resulting indices turned out to be usable as **tokens for a language model** — a use nobody designing telephone codecs in 1985 had any reason to anticipate.

---

## 28.5 Video

The extra dimension is expensive: $T\times H\times W$ patches makes attention $O((THW)^2)$.

**Approaches:** factorized space-time attention (ViViT — attend spatially, then temporally); tubelet embedding (3-D patches); sparse causal attention; and, for generation, **spatiotemporal latent diffusion** on video tokens from a 3-D VAE.

#### Why "the extra dimension is expensive" — the arithmetic

Attention over $T$ tokens costs $O(T^2)$. Video does not add a little to $T$; it multiplies it.

| Input | Tokens | Pairwise scores per head per layer |
|---|---|---|
| One image, $224^2$, patch 16 | 196 | 38,416 |
| 16 frames of the same | 3,136 | 9,834,496 |
| 10 seconds at 24 fps (240 frames) | 47,040 | **2,212,761,600** |

▸ **Sixteen frames costs $16^2 = 256$ times a single image, not 16 times.** Doubling the clip length quadruples the cost. This single fact is why every video architecture in the list above exists.

**Factorized space-time attention, with numbers.** ViViT's move is to refuse to run one attention over all 47,040 tokens, and instead run two smaller ones:

- **Spatial:** within each frame, 196 tokens attend to each other. Cost: $240 \times 196^2 = 9{,}219{,}840$.
- **Temporal:** for each spatial position, the 240 frames attend to each other. Cost: $196 \times 240^2 = 11{,}289{,}600$.
- **Total: ~20.5 million** versus 2.21 billion.

▸ **A 108× reduction, and the information can still travel anywhere** — a patch reaches a distant patch in a different frame by moving spatially in one layer and temporally in the next. **The path exists; it just takes two hops instead of one.** This is the same architectural bargain as Swin's shifted windows, and the same as separable convolutions: factor an expensive joint operation into two cheap marginal ones.

> **Analogy.** To get a message to everyone in a 240-storey building with 196 offices per floor, you do not hold one meeting of 47,040 people. You hold a meeting on each floor, then a meeting of one representative from each floor per office-column. Two rounds, and everyone is informed — at a fraction of the coordination cost.

**Tubelet embedding, concretely.** Instead of one token per $16\times16$ patch per frame, make one token per $2\times16\times16$ *block* spanning two consecutive frames. Frame count effectively halves: 240 → 120, so tokens drop from 47,040 to 23,520 and the attention cost falls **4×**. It is patchification applied to the time axis, and it works for the same reason: consecutive video frames are enormously redundant.

**The 3-D VAE, and where the real compression happens.** Before any transformer runs, a learned autoencoder compresses the video into a latent grid — say 8× in each spatial dimension and 4× in time. That is $8\times8\times4 = \mathbf{256\times}$ fewer elements to model, and the diffusion transformer never touches a pixel. ▸ **Almost all of video generation's tractability comes from this step, not from the attention pattern** — the same lesson as latent diffusion for images (Ch. 20).

**The hard problems** are temporal consistency (objects must persist and deform plausibly), physical plausibility, and cost.

#### Examples and non-examples: temporal consistency

**✅  temporal consistency**

| Example | Why it qualifies |
|---|---|
| A person walks behind a pillar and emerges wearing the same jacket | Object identity survived an occlusion |
| A poured liquid's level rises monotonically | The state has a coherent history |
| A shadow moves consistently with a single light source across 10 seconds | A global scene property is maintained |

**❌ Near-misses — look consistent frame by frame, but aren't**

| Looks consistent | Why it isn't | What it actually is |
|---|---|---|
| Every individual frame is a beautiful, sharp image | Frame quality says nothing about frame-to-frame identity | **Per-frame realism**. A model can produce 240 excellent photos of 240 slightly different dogs |
| The video is smooth with no flicker | Smoothness is a low-frequency property; identity is a semantic one | **Temporal smoothness** — necessary, nowhere near sufficient |
| An object stays in the same place | Nothing was tested; a static scene is trivially consistent | **A static shot** |
| A person's face is stable but their hands gain a finger mid-clip | Consistency held for the well-modelled region only | **Partial consistency** — the standard failure signature |
| The clip loops seamlessly | Loop closure is a boundary condition | **Seamless looping**, orthogonal to identity |

▸ **The boundary:** temporal consistency is about **persistent identity of objects and scene state through time**, including through occlusion. Smoothness and per-frame sharpness are both easier to achieve and easier to measure, which is why early video-generation metrics rewarded them and why models optimized against those metrics looked good in stills and wrong in motion.

> **Common misconception.** *"Video generation is image generation repeated."* If it were, you could generate 240 independent images and concatenate them — and the result is a strobing nightmare, because each sample lands somewhere different in a very high-dimensional distribution. The model must generate a *joint* sample over all frames, which is why the denoising happens over the whole spatiotemporal latent at once rather than frame by frame. **The unit of generation is the clip, not the frame.** The misconception is tempting because the architecture  is the image architecture with one more axis — the change is in what is being sampled, not in what computes it.

▸ The current dominant recipe for video generation — a **3-D causal VAE** compressing to a spatiotemporal latent, plus a **DiT with full spatiotemporal attention**, trained with **flow matching** (Ch. 20 §20.10) — is a direct composition of Chapters 19–21. Video generation introduced remarkably few  new ideas; it is mostly scale and engineering applied to the image recipe.

---

## 28.6 The convergence

▸ **The structural observation worth carrying:** every modality is now converted to a sequence of tokens — text by BPE, images by patchification or VQ, audio by RVQ codecs, video by 3-D tokenization, actions by discretization — and processed by the same transformer. The modality-specific work has collapsed into the **tokenizer**, and everything after it is shared.

This is why "any-to-any" models are feasible at all: once modality differences live only in the tokenizer and the embedding table, a single autoregressive or diffusion backbone can be trained on the union.

#### The convergence, made concrete

Line the modalities up and the claim stops being abstract:

| Modality | Raw form | Tokenizer | What one token is | Rough rate |
|---|---|---|---|---|
| Text | Characters | BPE | A word fragment | ~1.3 tokens per English word |
| Image | $H\times W\times3$ pixels | Patchify (+ linear projection) or VQ | A $16\times16$ tile | 576 tokens per $336^2$ image |
| Audio | 16,000 samples/s | RVQ codec | A 20 ms sound, 8 codes deep | ~400 tokens/s |
| Video | $T\times H\times W$ | 3-D VAE + patchify | A spatiotemporal block | Thousands per second, before compression |
| Robot actions | Continuous joint angles | Uniform binning | One discretized command | One token per joint per step |

▸ **Every row does the same three things:** cut the input into pieces of fixed size, map each piece to a vector, and hand the sequence to a transformer that has no idea where it came from. **The transformer is modality-blind by construction** — it sees a list of vectors, and self-attention has no notion of "pixel" or "phoneme."

**Why this matters more than it sounds.** It means the enormous engineering investment in transformers — FlashAttention, tensor parallelism, KV caching, quantization, speculative decoding, every optimizer trick in Chapters 5 and 14 — is a **shared asset**. An improvement to attention kernels speeds up speech recognition, video generation, and protein folding on the same afternoon. The field consolidated its effort onto one primitive, and that consolidation is arguably a larger practical effect than any individual architectural result in this chapter.

#### Examples and non-examples: "everything is tokens"

**✅  tokenizations**

| Example | Why it qualifies |
|---|---|
| BPE merging `"token"` + `"ization"` | Discrete units, fixed vocabulary, invertible |
| VQ-VAE image codes | Each patch maps to one of $K$ codebook entries; the decoder inverts it |
| RVQ audio codes | Same, with a residual hierarchy |
| Discretized robot joint angles (256 bins per joint) | A continuous quantity forced onto a finite alphabet |
| Amino acid letters in a protein | Discrete by nature — no tokenizer needed at all |

**❌ Near-misses**

| Looks like tokenization | Why it isn't | What it actually is |
|---|---|---|
| ViT's **patch embedding** | The output is a continuous vector, not an index into a finite vocabulary | A **continuous** embedding. You can feed it to a transformer, but you cannot *generate* it with a softmax, which is why ViT can classify but not generate |
| Feeding raw float features to an LLM's input | No discretization, no vocabulary | Soft prompting / prefix tuning |
| Word-level splitting on spaces | Breaks on unseen words, agglutinative languages, and code | A whitespace **split** — the thing BPE was invented to replace |
| A hash of an image | Not invertible, no local structure, no semantics | A fingerprint |
| One-hot encoding of a category | Correct as far as it goes, but there is no compression or structure learned | A trivial tokenization — fine for 10 categories, hopeless for audio |

▸ **The boundary:** a tokenizer maps arbitrary input onto a **finite, fixed alphabet of discrete symbols, invertibly enough that a decoder can reconstruct the input**. The discreteness is what lets you put a softmax on the output and *generate*; the invertibility is what makes the generated symbols mean something. ViT's patch embedding satisfies neither, which is exactly why generative image models use VQ or diffusion in latent space rather than the ViT stem.

> **Common misconception.** *"If everything is tokens, the tokenizer doesn't matter."* It matters more than anything else in the pipeline, because **it is the only lossy step you can never recover from**. A codec that discards high frequencies caps the audio quality of every model built on it, permanently. An image tokenizer that cannot represent small text means no amount of language-model scale will let the system read a receipt. The chapter's own closing question — "whether the tokenizer's information loss is a fundamental ceiling" — is asking exactly this. The misconception is tempting because the tokenizer is the least glamorous component, is usually downloaded rather than trained, and appears in one line of the config file.

**The open questions:** whether one shared backbone or modality-specific experts (an MoE over modalities) is better; how to weight modalities in the loss; whether autoregressive or diffusion generation wins for images and audio; and whether the tokenizer's information loss is a fundamental ceiling.

---

## Did you know?

- **The Vision Transformer paper's title is a pun that is also a specification.** *An Image is Worth 16×16 Words* plays on "a picture is worth a thousand words" — but $16\times16$ is the literal patch size, and at $224\times224$ resolution an image turns out to be worth exactly **196 words**, not a thousand.

- **A ViT's patchification is a convolution, in the source code as well as in theory.** Reference implementations create it with `nn.Conv2d(3, 768, kernel_size=16, stride=16)`. The architecture famous for abandoning convolutions opens with one, and it is the only one in the model.

- **Patchification loses nothing.** A $224\times224\times3$ image is 150,528 numbers, and $196 \times 768$ is also 150,528. The step that looks like a drastic simplification is a pure reshape — every pixel is still there, just regrouped.

- **Attention was proved capable of expressing convolution before anyone built a good vision transformer.** Cordonnier, Loukas, and Jaggi at EPFL showed in 2020 that a multi-head self-attention layer can represent any convolutional layer. The expressiveness question was answered before the engineering question was even asked.

- **CLIP and DALL·E were announced on the same day**, 5 January 2021. One made images searchable by arbitrary language and the other made them generatable from it; they were built on the same insight about caption supervision at scale.

- **Zero-shot ImageNet accuracy went from 11.5% to 76.2% in four years.** A 2017 paper had already done open-vocabulary classification from web captions and reached 11.5%. CLIP reached 76.2% — matching a fully supervised ResNet-50 without seeing a single ImageNet label.

- **The technique behind CLIP was prototyped on chest X-rays.** The CLIP paper describes its method as a scaled-up, simplified relative of ConVIRT, a 2020 approach to learning from paired medical images and radiology reports. Radiology had the naturally paired image–text data before the web did.

- **You can fool CLIP with a sticky note.** OpenAI's own researchers demonstrated that taping a handwritten label reading "iPod" onto a Granny Smith apple makes CLIP confidently classify the apple as an iPod. The model learned to read, and reading is a shortcut.

- **"Mel" is short for "melody," and the scale comes from a 1937 psychology experiment.** Stevens, Volkmann, and Newman at Harvard played listeners a tone and asked them to tune a second tone until it sounded *half as high*. The scale is anchored so that 1,000 Hz equals exactly 1,000 mel — every other value in every speech model on earth is measured from that peg.

- **The spectrogram was invented so deaf people could read speech with their eyes.** Bell Labs built the sound spectrograph during the Second World War; after declassification, the 1947 book presenting it was titled *Visible Speech*, and visual speech training was among its stated aims. Every modern speech model consumes a representation designed for human vision.

- **SigLIP's entire contribution is deleting a softmax.** Replacing one loss function with another removed the need for a cross-device all-gather, shrank a 4.3 GB intermediate tensor, and showed that the supposed permanent advantage of enormous batch sizes saturates far earlier than the field believed.

- **Flamingo's visual pathway is initialized to do nothing at all.** Each inserted cross-attention block ends in a $\tanh$ gate whose scalar starts at zero, so at step 0 the multimodal model computes bit-for-bit what the original text-only language model computed. Vision fades in as the gate learns.

- **LLaVA's training data was written by a model that could not see.** The 158,000 multimodal instruction examples were generated by a **text-only** GPT-4, given nothing but COCO captions and lists of bounding-box coordinates. It described images it never saw, and that synthetic supervision was enough to build a working visual assistant.

- **Swin Transformer won ICCV 2021's Marr Prize** for best paper, on the strength of an idea that fits in one sentence: do attention inside small windows, and slide the windows half a window over between layers.

- **ConvNeXt is a paper with no new component in it.** It is a ResNet modified one design decision at a time toward ViT-era practice, with accuracy measured after each step — an ablation study of a rumour, which arrived at Swin-level performance and reopened a question everyone considered closed.

- **Ten seconds of video, tokenized naively, costs 2.2 billion attention scores per head per layer.** 240 frames at 196 patches each is 47,040 tokens, and $47{,}040^2 = 2{,}212{,}761{,}600$. This one number is the reason every video architecture in this chapter exists.

---

## Check for Understanding

**A vision transformer works by treating patches as tokens and abandoning the convolutional prior, which loses at small data scale and wins at large scale; CLIP aligns images and text with a symmetric InfoNCE loss and thereby converts classification into retrieval with an open vocabulary; and every modality now reduces to a token sequence, which is why one transformer architecture can serve all of them.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why does a vision transformer need positional embeddings when a CNN doesn't?** (If your answer doesn't include the word "shuffle," try again.)

2. **Why does the same architecture lose to a ResNet on 1.3 million images and beat it on 300 million?** Say it in terms of guide rails and driving skill, not in terms of parameters.

3. **What does "inductive bias" mean, and why is data augmentation not one?**

4. **Why is a ViT's patch embedding literally a convolution?** What does stride 16 with a $16\times16$ kernel do that stride 1 doesn't?

5. **Why does CLIP need a temperature at all?** Work the numbers: what happens to the loss if you set $\tau = 1$ and the similarities sit between 0.05 and 0.30?

6. **Why is CLIP's zero-shot classification "retrieval, not classification"?** What can you do at 3 p.m. that a supervised classifier can't?

7. **Why does replacing a softmax with a sigmoid remove a communication cost across GPUs?** The answer is one word about what a softmax needs before it can produce any number at all.

8. **Why can CLIP not tell "a horse riding an astronaut" from "an astronaut riding a horse"?** Explain it in terms of what the training objective ever rewarded.

9. **Why does a frozen CLIP encoder plus a small trained projector produce a working vision–language model?** Use the phrase "change of basis" and then say what that means without it.

10. **Why does a high-resolution image cost 2,880 tokens instead of 576, and what breaks if you skip the downscaled global view?**

11. **Why is a spectrogram an image, and which image intuitions become wrong the moment you treat it as one?** (Start with what happens if you flip it horizontally.)

12. **Why do residual vector quantizers use eight small codebooks instead of one big one?** Explain it with coins.

13. **Why does adding one time axis make attention 256 times more expensive for a 16-frame clip, and how does factorizing it recover most of that?**

14. **What is the one property that makes something a tokenizer, and why does a ViT's patch embedding fail it?**

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 29 — Graph Neural Networks & Geometric Deep Learning](29-graph-neural-networks.md)
