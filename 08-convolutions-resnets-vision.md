# Chapter 8 — Convolutions, ResNets & Vision Architectures

> **Prerequisites:** Ch. 6, Ch. 7.

---

## 8.1 Convolution

### The one-line idea

A convolution is a fully-connected layer with two constraints bolted on: the weights are shared across positions, and each output only looks at a small local window.

### The analogy

A single rubber stamp dragged across a page, versus hand-drawing every mark individually. The stamp (kernel) encodes one pattern; sliding it means "this pattern is worth detecting anywhere." That is **translation equivariance**, and it is a prior about the world: a cat is a cat in the top-left or bottom-right.

### The math

For input $X\in\mathbb{R}^{C_{\text{in}}\times H\times W}$ and kernel $K\in\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times k\times k}$:

▸ $$Y_{o,i,j} = \sum_{c=1}^{C_{\text{in}}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K_{o,c,u,v}\,X_{c,\,i s+u-p,\,j s+v-p} + b_o$$

(Technically cross-correlation, not convolution — no kernel flip. Nobody cares; the kernel is learned.)

**Output size:**
▸ $$H_{\text{out}} = \left\lfloor\frac{H+2p-d(k-1)-1}{s}\right\rfloor + 1$$
with padding $p$, stride $s$, dilation $d$. Memorize this; it appears in every shape-debugging session.

**Parameters:** $C_{\text{out}}\cdot C_{\text{in}}\cdot k^2 + C_{\text{out}}$.
**FLOPs:** $2\cdot C_{\text{out}}C_{\text{in}}k^2 H_{\text{out}}W_{\text{out}}$.

**Compared to a dense layer** on a $224\times224\times3$ image with 64 outputs at every position: dense would be $150{,}528\times(224\cdot224\cdot64)=4.8\times10^{11}$ parameters. A $3\times3$ conv is $3\cdot64\cdot9=1{,}728$. **Eight orders of magnitude**, purchased entirely by the two structural assumptions.

### Equivariance vs invariance

▸ Convolution is **equivariant** to translation: $\mathrm{Conv}(\mathrm{shift}(x)) = \mathrm{shift}(\mathrm{Conv}(x))$.
Global pooling at the end converts equivariance into **invariance**: $\mathrm{Pool}(\mathrm{shift}(f)) = \mathrm{Pool}(f)$.

Convolution is *not* equivariant to rotation or scale — hence rotation augmentation, and hence group-equivariant CNNs (Ch. 29).

### Receptive field

For a stack of layers with kernel $k_\ell$, stride $s_\ell$, dilation $d_\ell$:

▸ $$r_L = 1 + \sum_{\ell=1}^{L} \big(d_\ell(k_\ell-1)\big)\prod_{j<\ell}s_j$$

**Numbers:** ten $3\times3$ convs, stride 1: $r = 1 + 10\cdot2 = 21$ pixels. Add a stride-2 pool after every two convs and it grows geometrically instead.

▸ The **effective** receptive field is much smaller than the theoretical one and is approximately **Gaussian**, with effective radius $\propto\sqrt{L}$ rather than $L$ (Luo et al., 2016) — because the number of paths from a centre pixel to the output vastly exceeds the number from the edge, the influence is a random-walk convolution. **Always assume your effective receptive field is roughly the square root of what the formula says.** This is the practical justification for dilated convolutions and for attention.

### Variants

| Variant | Definition | Use |
|---|---|---|
| **1×1 conv** | $k=1$ | channel mixing; a per-position dense layer; the bottleneck in ResNet |
| **Dilated/atrous** | insert $d-1$ holes | exponential receptive field growth at constant cost; segmentation, WaveNet |
| **Depthwise** | one kernel per input channel, $C_{\text{out}}=C_{\text{in}}$ | $k^2C$ params instead of $k^2C^2$ |
| **Depthwise separable** | depthwise then $1\times1$ | $\frac{k^2C + C^2}{k^2C^2}\approx\frac1{k^2}$ of the cost; MobileNet, Xception |
| **Grouped** | split channels into $g$ groups | AlexNet (GPU memory), ResNeXt (cardinality) |
| **Transposed ("deconv")** | fractionally-strided | upsampling; **causes checkerboard artifacts** — prefer `upsample + conv` |
| **Deformable** | learned sampling offsets | adaptive geometry |

**Depthwise separable arithmetic:** standard $3\times3$ with $C=256$: $9\cdot256^2 = 590$k params. Separable: $9\cdot256 + 256^2 = 67.9$k. **8.7× fewer**, at ~1% accuracy cost. This one substitution is the entire MobileNet family.

### Pooling

Max pooling: local invariance to small translations, gradient routes only to the max. Average pooling: smoother. **Global average pooling** replaced the giant FC head in modern CNNs — it removes ~90% of AlexNet-era parameters and enforces a channel↔class correspondence.

Strided convolution has largely replaced pooling (it is a *learned* downsample).

---

## 8.2 Residual networks

### The one-line idea

Make each block compute a *correction* to its input rather than a replacement, so the network can trivially represent the identity and gradients can flow around every block.

### The analogy

Editing a document by tracked changes rather than retyping it from scratch. If a chapter needs no edit, "no change" is the cheapest possible edit. Deep plain networks have to retype the whole document at every layer, and errors compound.

### The problem it solved

▸ **Degradation, not overfitting.** He et al. (2015) observed a 56-layer plain CNN with *higher training error* than a 20-layer one. That cannot be overfitting — the deeper network *contains* the shallower one as a special case (set extra layers to identity). It is an **optimization** failure: SGD cannot find the identity mapping in a stack of convs.

### The fix

▸ $$x_{\ell+1} = x_\ell + F(x_\ell;\theta_\ell)$$

Now identity is $F=0$, which is the easiest function for a zero-initialized layer to represent.

### Why it works: three explanations, all partly right

**1. Gradient highway.** Unroll:
$$x_L = x_\ell + \sum_{i=\ell}^{L-1}F(x_i)\quad\Rightarrow\quad \frac{\partial\mathcal{L}}{\partial x_\ell} = \frac{\partial\mathcal{L}}{\partial x_L}\left(1 + \sum_{i=\ell}^{L-1}\frac{\partial F(x_i)}{\partial x_\ell}\right)$$

▸ The leading **1** guarantees the gradient reaches every layer regardless of what the residual branches do. Contrast Ch. 6: plain nets multiply $L$ Jacobians and die; residual nets *add* corrections to an identity. This is the strongest of the three explanations.

**2. Ensemble of shallow paths** (Veit et al.). A ResNet with $L$ blocks has $2^L$ paths from input to output (take or skip each block). Deleting one block from a trained ResNet barely hurts — unlike a plain net, where deleting a layer is catastrophic. The measured gradient contribution is dominated by paths of length ~10–30, so a 110-layer ResNet behaves like an ensemble of relatively shallow networks.

**3. Loss-landscape smoothing** (Li et al., 2018). Visualizations of the loss surface along random 2-D slices show plain deep nets have chaotic, fractal-looking landscapes and ResNets have smooth, convex-looking basins. Cause and effect are hard to separate here, but the correlation is striking.

### Block designs

**Basic (ResNet-18/34):** conv3×3 → BN → ReLU → conv3×3 → BN → (+x) → ReLU.

**Bottleneck (ResNet-50+):** conv1×1 (reduce $C\to C/4$) → conv3×3 → conv1×1 (expand). Costs $\sim\frac{1}{4}$ the FLOPs of two 3×3s at the same width. **ResNet-50 is 4× deeper than ResNet-34 at similar FLOPs.**

**Pre-activation (ResNet-v2):** BN → ReLU → conv → BN → ReLU → conv → (+x), with *nothing* on the skip path. This is the residual analogue of pre-LN, and it's what allows 1000-layer ResNets.

**Shortcut when shapes change:** either a $1\times1$ stride-2 conv (option B, standard) or zero-padding (option A). Option B is better but adds parameters.

### The BatchNorm–residual interaction

At initialization, $\mathrm{Var}(x_{\ell+1}) = \mathrm{Var}(x_\ell)+\mathrm{Var}(F)$, so variance grows **linearly with depth** and the signal-to-residual ratio degrades. BatchNorm inside the block rescales $F$ so this stays controlled.

▸ **Zero-initializing the last BN's $\gamma$ in each block** makes every block start as exact identity. This is standard (`zero_init_residual=True`) and gives ~0.5% top-1 for free. It is the same idea as **Fixup**, **ReZero**, and **AdaLN-Zero** (Ch. 21). Learn it once; it recurs everywhere.

---

## 8.3 The architecture lineage, compressed

| Model | Year | Key idea | Top-1 (IN-1k) | Params |
|---|---|---|---|---|
| LeNet-5 | 1998 | conv+pool+FC | — | 60k |
| AlexNet | 2012 | ReLU, dropout, GPU | 63.3% | 60M |
| VGG-16 | 2014 | stacks of 3×3 (two 3×3 = one 5×5 receptive field, fewer params, more nonlinearity) | 71.6% | 138M |
| **GoogLeNet/Inception** | 2014 | parallel multi-scale branches; 1×1 bottlenecks; factorized $n\times n \to n\times1,1\times n$ | 74.8% | 6.8M |
| **ResNet-50** | 2015 | residual connections | 76.1% | 25.6M |
| ResNeXt | 2016 | grouped conv, "cardinality" | 77.8% | 25M |
| DenseNet | 2016 | concatenate all previous features: $x_\ell = H([x_0,\dots,x_{\ell-1}])$; feature reuse, $O(L^2)$ connections | 77.7% | 8M |
| SENet | 2017 | **channel attention**: squeeze (GAP) → excite (2 FC + sigmoid) → rescale channels. The first widely-used attention in vision. | 82.7% | 146M |
| MobileNetV2 | 2018 | inverted residual + linear bottleneck | 72.0% | 3.4M |
| **EfficientNet** | 2019 | **compound scaling**: $d=\alpha^\phi, w=\beta^\phi, r=\gamma^\phi$ s.t. $\alpha\beta^2\gamma^2\approx2$ — scale depth, width, resolution together | 84.3% (B7) | 66M |
| ConvNeXt | 2022 | a ResNet modernized with ViT design choices (7×7 depthwise, LN, GELU, fewer norms, inverted bottleneck) — matches Swin | 87.8% | 198M |

▸ **The ConvNeXt lesson, which is the one worth carrying:** most of the ViT-over-CNN gap in 2021 was *training recipe and design detail*, not attention. When a ResNet is given AdamW, 300 epochs, RandAugment, Mixup, stochastic depth, LayerNorm, GELU, and large depthwise kernels, it matches transformers on ImageNet. **Architecture comparisons at unequal training recipes are worthless.**

---

## 8.4 U-Net

The dominant architecture for dense prediction, and the backbone of most diffusion models before DiT.

**Structure:** an encoder that downsamples ($H\to H/2\to H/4\dots$) while widening channels, a bottleneck, and a decoder that upsamples symmetrically — with **skip connections concatenating each encoder feature map to the decoder at matching resolution.**

▸ **Why the skips matter:** the encoder path destroys spatial precision (that's what downsampling does) while building semantic abstraction. The skips restore high-frequency spatial detail that the bottleneck cannot carry. Without them, segmentation boundaries are mush.

**In diffusion (Ch. 20):** the U-Net is augmented with (i) timestep embedding injected via FiLM/AdaGN into every block, (ii) self-attention at the lower resolutions (typically 16×16 and 8×8 — full attention at 64×64 is too expensive), (iii) cross-attention for text conditioning, (iv) GroupNorm (batch-independent), (v) SiLU activations.

**Why DiT replaced it:** the U-Net's multi-resolution inductive bias is valuable at low compute but becomes a constraint at scale; a plain transformer over latent patches scales more predictably and follows the same scaling laws as language models (Ch. 21).

---

## 8.5 Detection and segmentation, in brief

**Two-stage (R-CNN → Fast → Faster):** a Region Proposal Network emits candidate boxes; RoIAlign crops features; heads classify and regress. Higher accuracy, slower.

**One-stage (YOLO, SSD, RetinaNet):** dense prediction over anchors in one pass. **Focal loss** solved the extreme foreground/background imbalance:
▸ $$\mathrm{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log p_t,\qquad \gamma=2$$
The $(1-p_t)^\gamma$ factor down-weights easy examples by up to $100\times$, letting the rare hard positives dominate the gradient. **This is the canonical answer to "how do you handle class imbalance in dense prediction."**

**DETR:** treats detection as set prediction with a transformer; uses **Hungarian matching** for a bipartite assignment between predictions and ground truth, removing NMS and anchors entirely.

**Segmentation:** FCN → U-Net → DeepLab (atrous spatial pyramid pooling) → Mask R-CNN (instance) → SAM (promptable, foundation-model style).

**IoU and NMS.** $\mathrm{IoU} = \frac{|A\cap B|}{|A\cup B|}$. Non-max suppression greedily keeps the highest-scoring box and deletes overlapping ones above an IoU threshold. Losses: GIoU/DIoU/CIoU fix the zero-gradient problem when boxes don't overlap at all.

---

## 8.6 Interview-grade facts

- **Two 3×3 convs vs one 5×5:** same receptive field; $2\cdot9C^2=18C^2$ vs $25C^2$ params; two nonlinearities instead of one. This is the entire VGG argument.
- **1×1 convolutions** do three jobs: channel dimensionality reduction, cross-channel mixing, and adding nonlinearity cheaply.
- **Why do CNNs need less data than MLPs?** Weight sharing and locality are a hard prior encoding translation equivariance — it's the same argument as a smaller hypothesis class in Ch. 2, but the constraint is *correct* for images, so approximation error barely rises while estimation error falls sharply.
- **Why did attention beat convolution in vision at scale?** Not expressivity — a ViT with enough data learns convolution-like early layers. The gap is that conv's prior *helps* at 1M images and *constrains* at 300M. Inductive bias is a data-efficiency/ceiling trade.
- **Checkerboard artifacts** come from transposed convolution when `stride` doesn't divide `kernel_size`, producing uneven overlap. Fix: nearest/bilinear upsample followed by a normal conv.

---

## Check for Understanding

**Convolution is a dense layer restricted by locality and weight sharing — a correct prior for images that buys eight orders of magnitude in parameters — and residual connections fix the resulting depth problem by making every block compute a correction to an identity, so gradients add rather than multiply.**

---

**Next:** [Chapter 09 — Sequence Models: RNNs, LSTMs, Seq2Seq](09-sequence-models-rnn-lstm.md)
