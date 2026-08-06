# Chapter 28 — Vision Transformers & Multimodal Models

> **Prerequisites:** Ch. 8, Ch. 11, Ch. 25.

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

### The data-hunger result

▸ The original ViT paper's central finding: on ImageNet-1k (1.3M images) ViT **underperforms** a comparable ResNet. On ImageNet-21k (14M) it matches. On JFT-300M it **wins decisively**.

**The interpretation is the generalizable lesson:** a CNN's locality and translation-equivariance priors are *correct but restrictive*. When data is scarce, a correct prior substitutes for data. When data is abundant, the same prior becomes a ceiling, and a model that can learn its own structure surpasses it. **This is Chapter 2's approximation/estimation trade-off appearing as an architectural choice.**

▸ Probing confirms the mechanism: trained ViTs learn convolution-like local attention in early layers when data is plentiful — they *rediscover* the prior — while retaining global attention where it helps.

### The variants

| Model | Contribution |
|---|---|
| **DeiT** | matched ViT performance on ImageNet-1k alone using heavy augmentation (RandAugment, mixup, CutMix), stochastic depth, and **distillation from a CNN teacher via a dedicated distillation token**. Showed the data-hunger was largely a *recipe* problem. |
| **Swin** | hierarchical, with **shifted windows**: attention within local windows, and windows shifted between layers to allow cross-window information flow. Restores $O(T)$ complexity and multi-scale features — necessary for detection and segmentation. |
| **PVT / CvT / CoAtNet** | reintroduce pyramids and convolutions in various proportions |
| **MAE / BEiT** | self-supervised pretraining (Ch. 25 §25.4) |
| **DINOv2/v3** | self-supervised features usable off the shelf for dense and global tasks |
| **ConvNeXt** | the counter-argument — a CNN with ViT-style training matches Swin (Ch. 8 §8.3) |

▸ **The synthesis to state if asked "CNN or transformer for vision?":** at small scale with limited data, a modern CNN is at least as good and cheaper. At large scale with pretraining, transformers win and — more importantly — they unify vision with language, which is the actual reason they took over.

---

## 28.2 Contrastive vision–language: CLIP

### The one-line idea

Train an image encoder and a text encoder so that matching image–caption pairs land near each other in a shared embedding space — and get open-vocabulary classification for free.

### The analogy

Teaching two people, one who only sees and one who only reads, to point at the same spot on a shared map. Once they agree on the map, you can describe something in words and the seeing person can find it — even for things neither was explicitly trained on, because the map has structure.

### The objective

For a batch of $N$ pairs, compute image embeddings $I_i$ and text embeddings $T_j$, both L2-normalized. The logits are $\frac{I_iT_j^\top}{\tau}$, and the loss is **symmetric cross-entropy** over both axes:

▸ $$\mathcal{L}=\frac12\left[\underbrace{\mathrm{CE}\big(\tfrac{IT^\top}{\tau},\,\mathrm{diag}\big)}_{\text{image}\to\text{text}} + \underbrace{\mathrm{CE}\big(\tfrac{TI^\top}{\tau},\,\mathrm{diag}\big)}_{\text{text}\to\text{image}}\right]$$

**It is an $N$-way classification: "which of these $N$ captions belongs to this image?"** — InfoNCE (Ch. 25 §25.3) with the other modality as the positive.

▸ **The temperature $\tau$ is learned**, parameterized as $\exp(t)$ and clipped to prevent instability. A learned temperature was a small but genuinely important detail.

**Scale:** 400M pairs, batch size **32,768**. The large batch matters — the number of negatives directly determines the difficulty and the tightness of the bound.

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

### SigLIP

Replace the softmax cross-entropy with a **pairwise sigmoid** loss:
▸ $$\mathcal{L}=-\frac1N\sum_{i}\sum_j\log\sigma\big(z_{ij}(t\,I_i\cdot T_j + b)\big),\qquad z_{ij}=+1 \text{ if } i=j \text{ else } -1$$

▸ **The softmax requires a global normalization over the batch, forcing an expensive all-gather across devices. The sigmoid loss is fully decomposable per pair**, so it works at small batch sizes and scales without the communication cost. SigLIP matches or beats CLIP at much smaller batches, and is the default image encoder in many current VLMs.

---

## 28.3 Vision–language models

### The architectural patterns

| Pattern | Mechanism | Examples |
|---|---|---|
| **Frozen encoder + projector** | image → ViT → linear/MLP → tokens prepended to the LLM's input | **LLaVA**, most open VLMs |
| **Cross-attention** | LLM layers cross-attend to visual features via inserted gated layers | Flamingo, Idefics |
| **Q-Former / resampler** | learned queries compress $N$ patch tokens to $k$ tokens | BLIP-2, Qwen-VL |
| **Early fusion** | image and text tokens interleaved from layer 1, trained jointly | Chameleon, Fuyu |

### The LLaVA recipe — the one to know

1. **Stage 1 (alignment):** freeze both the vision encoder and the LLM; train **only the projector** on image–caption pairs. Cheap, and it teaches the projector to speak the LLM's embedding language.
2. **Stage 2 (instruction tuning):** unfreeze the LLM (and sometimes the encoder); train on multimodal instruction data.

▸ **Why a frozen encoder plus a trained projector works at all:** CLIP's image embedding already contains language-aligned semantics, so the projector's job is a change of basis, not a change of content. This is why VLMs are so much cheaper to build than their capability suggests.

### The design questions that matter

**How many visual tokens?** A $336\times336$ image at patch 14 gives 576 tokens — a large fraction of the context, and attention is quadratic. Resamplers compress to 32–256. **High resolution is essential for OCR, charts, and documents**, so the standard trick is **dynamic tiling** (AnyRes): split a high-resolution image into tiles, encode each, plus a downscaled global view.

**Which encoder?** CLIP/SigLIP ViT-L or ViT-SO400M. Some models ensemble a CLIP encoder (semantics) with a DINOv2 encoder (spatial detail).

**Native multimodal vs bolted-on.** Training on interleaved image–text from scratch gives better integration; adapting a strong text LLM is far cheaper and currently gives better text reasoning. Most open models take the second path.

### Failure modes

- **Object hallucination** — describing objects that aren't present, driven by language priors ("a kitchen usually has a sink"). Measured by POPE. Mitigations: contrastive decoding against a blank image, and grounding-focused training data.
- **Fine-grained spatial reasoning** and precise counting.
- **Text-only degradation** after multimodal tuning; mix in text-only data.
- **The vision encoder is a hard bottleneck** — the LLM cannot perceive what the encoder discarded.

---

## 28.4 Audio and speech

### Representations

Raw waveform (16 kHz) → **mel spectrogram**: STFT with a ~25 ms window and 10 ms hop, magnitude, then a mel filterbank (spacing ~$2595\log_{10}(1+f/700)$, approximating human pitch perception), then log. ▸ **A spectrogram is an image, so every vision architecture applies** — which is exactly how the field developed.

### The models

**wav2vec 2.0** — self-supervised speech. A CNN feature extractor, a transformer context network, masked spans, and a **contrastive loss against quantized latent targets** (via Gumbel-softmax over a product codebook). Result: state-of-the-art ASR with 10 minutes of labelled audio, from 53k hours of unlabelled. **HuBERT** is the same idea with offline k-means clustering providing discrete targets, iteratively refined — simpler and usually better.

**Whisper** — the opposite bet: plain encoder–decoder transformer on log-mel input, trained *supervised* on 680k hours of weakly-labelled multilingual web audio, with special tokens for the task (transcribe/translate), language, and timestamps. ▸ **The lesson is about data, not architecture: massive weak supervision beat elaborate self-supervision for robustness.** Whisper generalizes across accents, noise, and domains far better than models trained on clean curated corpora.

**Neural audio codecs (SoundStream, EnCodec, DAC)** — residual vector quantization produces discrete audio tokens at ~1.5–6 kbps. ▸ **This is the enabling technology for audio language models**: once audio is tokens, an autoregressive transformer generates speech and music the same way it generates text. AudioLM, VALL-E, MusicGen, and modern TTS all rest on it.

**Speech LLMs** now do end-to-end speech-to-speech with no intermediate text, which preserves prosody, emotion, and speaker identity and removes ASR/TTS latency.

---

## 28.5 Video

The extra dimension is expensive: $T\times H\times W$ patches makes attention $O((THW)^2)$.

**Approaches:** factorized space-time attention (ViViT — attend spatially, then temporally); tubelet embedding (3-D patches); sparse causal attention; and, for generation, **spatiotemporal latent diffusion** on video tokens from a 3-D VAE.

**The hard problems** are temporal consistency (objects must persist and deform plausibly), physical plausibility, and cost.

▸ The current dominant recipe for video generation — a **3-D causal VAE** compressing to a spatiotemporal latent, plus a **DiT with full spatiotemporal attention**, trained with **flow matching** (Ch. 20 §20.10) — is a direct composition of Chapters 19–21. Video generation introduced remarkably few genuinely new ideas; it is mostly scale and engineering applied to the image recipe.

---

## 28.6 The convergence

▸ **The structural observation worth carrying:** every modality is now converted to a sequence of tokens — text by BPE, images by patchification or VQ, audio by RVQ codecs, video by 3-D tokenization, actions by discretization — and processed by the same transformer. The modality-specific work has collapsed into the **tokenizer**, and everything after it is shared.

This is why "any-to-any" models are feasible at all: once modality differences live only in the tokenizer and the embedding table, a single autoregressive or diffusion backbone can be trained on the union.

**The open questions:** whether one shared backbone or modality-specific experts (an MoE over modalities) is better; how to weight modalities in the loss; whether autoregressive or diffusion generation wins for images and audio; and whether the tokenizer's information loss is a fundamental ceiling.

---

## Check for Understanding

**A vision transformer works by treating patches as tokens and abandoning the convolutional prior, which loses at small data scale and wins at large scale; CLIP aligns images and text with a symmetric InfoNCE loss and thereby converts classification into retrieval with an open vocabulary; and every modality now reduces to a token sequence, which is why one transformer architecture can serve all of them.**

---

**Next:** [Chapter 29 — Graph Neural Networks & Geometric Deep Learning](29-graph-neural-networks.md)
