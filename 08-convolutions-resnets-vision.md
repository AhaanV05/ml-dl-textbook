# Chapter 8 — Convolutions, ResNets & Vision Architectures

> **Prerequisites:** Ch. 6, Ch. 7.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`C_{\text{in}},\ C_{\text{out}}`$ | "C-in, C-out" | How many channels (feature maps) go in, how many come out |
| $H,\ W$ | "H, W" | Height and width of the image, in pixels |
| $k$ | "k" | **Kernel size** — the sliding stamp is $k\times k$ pixels |
| $s$ | "s" | **Stride** — how many pixels the stamp jumps between placements |
| $p$ | "p" | **Padding** — rows and columns of zeros glued round the border |
| $d$ | "d" | **Dilation** — how many gaps to leave *inside* the stamp |
| $`X_{c,i,j}`$ | "X sub c, i, j" | The input value at channel $c$, row $i$, column $j$ |
| $`K_{o,c,u,v}`$ | "K sub o, c, u, v" | One weight: output channel $o$, input channel $c$, stamp position $(u,v)$ |
| $\lfloor z\rfloor$ | "floor of z" | Round down to the nearest whole number |
| $`r_L`$ | "r sub L" | **Receptive field** — how wide a patch of input one output can see |
| $`F(x_\ell;\theta_\ell)`$ | "F of x-ell, with theta-ell" | The **residual branch** — a small stack of layers with its own weights |
| $`x_{\ell+1}=x_\ell+F(x_\ell)`$ | — | "New activations = old activations **plus a correction**" |
| $`\mathrm{Var}(x_\ell)`$ | "variance of x-ell" | How spread out the activations at layer $\ell$ are |
| $\mathrm{IoU}$ | "eye-oh-you" | **Intersection over Union** — how much two boxes overlap, from 0 to 1 |
| $`p_t`$ | "p sub t" | The probability the model assigned to the **correct** class |
| $\gamma$ | "gamma" | ⚠ Two jobs here: the focal loss's focusing power, *and* BatchNorm's learned scale |
| $\alpha,\beta,\phi$ | "alpha, beta, phi" | EfficientNet's depth/width/resolution scaling exponents |

**Full forms for the abbreviations in this chapter:**

| Short | Full form |
|---|---|
| CNN | Convolutional Neural Network |
| FC | Fully Connected (layer) |
| MLP | Multi-Layer Perceptron |
| BN | Batch Normalization |
| LN | Layer Normalization |
| GAP | Global Average Pooling |
| FLOP | FLoating-point OPeration |
| IN-1k | ImageNet-1k (the 1,000-class ImageNet benchmark) |
| ILSVRC | ImageNet Large Scale Visual Recognition Challenge |
| ResNet | Residual Network |
| SE / SENet | Squeeze-and-Excitation (Network) |
| ViT | Vision Transformer |
| GELU / SiLU | Gaussian Error Linear Unit / Sigmoid Linear Unit |
| FiLM | Feature-wise Linear Modulation |
| AdaGN / AdaLN | Adaptive Group / Layer Normalization |
| DiT | Diffusion Transformer |
| RPN | Region Proposal Network |
| RoI | Region of Interest |
| NMS | Non-Maximum Suppression |
| IoU / GIoU / DIoU / CIoU | Intersection over Union, and its Generalized / Distance / Complete variants |
| DETR | DEtection TRansformer |
| FCN | Fully Convolutional Network |
| ASPP | Atrous Spatial Pyramid Pooling |
| SAM | Segment Anything Model |
| VGG | Visual Geometry Group (the Oxford lab that built it) |

---

## 8.1 Convolution

### The one-line idea

A convolution is a fully-connected layer with two constraints bolted on: the weights are shared across positions, and each output only looks at a small local window.

### The analogy

A single rubber stamp dragged across a page, versus hand-drawing every mark individually. The stamp (kernel) encodes one pattern; sliding it means "this pattern is worth detecting anywhere." That is **translation equivariance**, and it is a prior about the world: a cat is a cat in the top-left or bottom-right.

### The math

For input $`X\in\mathbb{R}^{C_{\text{in}}\times H\times W}`$ and kernel $`K\in\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times k\times k}`$:

▸ $$Y_{o,i,j} = \sum_{c=1}^{C_{\text{in}}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K_{o,c,u,v}\,X_{c,\,i s+u-p,\,j s+v-p} + b_o$$

(Technically cross-correlation, not convolution — no kernel flip. Nobody cares; the kernel is learned.)

#### Reading the convolution formula in plain English

Read it aloud first, symbol by symbol:

*"Y at output-channel o, row i, column j equals — the sum over every input channel c, and over every row u and every column v of the kernel — of the kernel weight times the input pixel sitting underneath it, plus a bias."*

Now every piece:

| Piece | Read aloud | What it is |
|---|---|---|
| $`Y_{o,i,j}`$ | "Y sub o, i, j" | **One single number** in the output. Three subscripts: which feature map, which row, which column |
| $`\sum_{c}\sum_{u}\sum_{v}`$ | "sum over c, sum over u, sum over v" | Three nested `for` loops (Ch. 0 §0.3) |
| $`C_{\text{in}}`$ | "C-in" | How many channels the input has — 3 for a colour photo, 256 deep in a network |
| $`K_{o,c,u,v}`$ | "K sub o, c, u, v" | **One weight.** The kernel is a 4-dimensional array of them |
| $`X_{c,\,is+u-p,\,js+v-p}`$ | "X at channel c, row i-s-plus-u-minus-p, …" | The input pixel currently under the stamp |
| $`b_o`$ | "b sub o" | One bias per output channel — added once, after the sum |

**The index arithmetic is the entire trick,** so slow down on $is + u - p$:

- $is$ — where the stamp's corner sits. Bump the output column by one and the stamp slides $s$ pixels. That is what stride *is*.
- $+u$ — walk around **inside** the stamp, from $0$ to $k-1$.
- $-p$ — shift back, because padding glued $p$ columns of zeros onto the left edge and shifted every real pixel to the right by $p$.

> **Analogy.** Print a $3\times3$ grid of numbers on a transparent sheet. Lay it over a sheet of graph paper filled with numbers. Multiply each printed number by the number showing through underneath it, add up all nine products, and write the total into a fresh grid. Slide the transparency $s$ squares right and repeat. The transparency never changes — **one sheet, dragged across the whole page.** That is a convolution, and the fact that the sheet is the *same everywhere* is the whole idea.

**Put numbers in it.** Collapse to one dimension, one input channel, one output channel: $`C_{\text{in}}=C_{\text{out}}=1`$, $k=3$, $s=1$, $p=1$, bias $0$. Let the kernel be $K = (-1, 0, +1)$ and the input be a signal that steps up halfway through:

$$X = (2,\,2,\,2,\,9,\,9,\,9)$$

With padding, the formula reduces to $`Y_i = -X_{i-1} + X_{i+1}`$ (treating out-of-range as $0$):

| $i$ | computation | $`Y_i`$ |
|---|---|---|
| 1 | $-0 + 2$ | $2$ |
| 2 | $-2 + 2$ | $0$ |
| 3 | $-2 + 9$ | $\mathbf{7}$ |
| 4 | $-2 + 9$ | $\mathbf{7}$ |
| 5 | $-9 + 9$ | $0$ |
| 6 | $-9 + 0$ | $-9$ |

$Y = (2,0,7,7,0,-9)$. The two large values sit **exactly at the boundary** between the flat region of 2s and the flat region of 9s. Flat regions produce zero. Those three numbers $(-1,0,+1)$ are an **edge detector**, and they are the single most-drawn kernel in the history of image processing.

▸ **Now the payoff.** The same three numbers would have found that edge at position 3, at position 47, or at position 900. One kernel, every position, one gradient shared across all of them. That is what "weight sharing" buys: **every pixel in every training image contributes evidence about the same 9 numbers.** A dense layer would have to learn "edge detection at position 47" separately from "edge detection at position 48," and would need enough data to see edges at every position independently.

#### Why "convolution" is the wrong word (and why it doesn't matter)

A *true* mathematical convolution flips the kernel before sliding it: $`\sum_u K_u X_{i-u}`$, with a **minus**. What the formula above computes is $`\sum_u K_u X_{i+u}`$, with a **plus** — which mathematicians call **cross-correlation**.

Why nobody cares: the kernel is *learned*. If the optimal true-convolution kernel is $(-1,0,+1)$, gradient descent on a cross-correlation layer simply learns $(+1,0,-1)$ instead and produces identical behaviour. The flip is absorbed into the weights.

Why it occasionally *does* matter: true convolution is commutative and associative ($`a * b = b * a`$), which is what makes the **convolution theorem** — convolution in space equals multiplication in the frequency domain — hold. That theorem is why some convolution kernels are implemented with fast Fourier transforms rather than by sliding.

> **Where this came from.** The convolutional network descends directly from neurophysiology. In experiments beginning in the late 1950s at Johns Hopkins, **David Hubel and Torsten Wiesel** recorded from single neurons in the cat's primary visual cortex and found cells that fired only for a bar of light at a *particular orientation* ("simple cells"), and cells that fired for that orientation *anywhere in a small region* ("complex cells"). The discovery has a famous accidental element: they were projecting dot patterns on glass slides and, by their own account, a cell fired not for any dot but as the **edge of the glass slide swept across the screen** while they were changing it. They shared the 1981 Nobel Prize in Physiology or Medicine.
>
> **Kunihiko Fukushima** turned that anatomy into an architecture with the **Neocognitron** (1980), built at NHK's broadcasting research labs in Japan: alternating layers of "S-cells" (simple, feature-detecting, weight-shared) and "C-cells" (complex, pooling). It had the structure of a modern CNN but no backpropagation — it was trained by unsupervised competitive learning. **Yann LeCun** supplied the missing piece at Bell Labs in 1989, training a convolutional network with backpropagation to read handwritten ZIP codes for the US Postal Service. The lineage runs cat visual cortex → Neocognitron → LeNet → everything in this chapter.

#### Examples and non-examples: is that a convolution?

A convolutional layer is defined by exactly two properties: **locality** (each output depends on a small neighbourhood of the input) and **weight sharing** (the *same* small weight set is used at every position). Drop either one and you have a different layer with a different parameter count and different inductive bias.

**✅  convolutions**

| Example | Why it qualifies |
|---|---|
| `nn.Conv2d(64, 128, kernel_size=3, padding=1)` | 3$\times$3 window (local), one set of $3\cdot3\cdot64$ weights per output channel reused at all $H\cdot W$ positions (shared) |
| `nn.Conv2d(256, 64, kernel_size=1)` | Window of size 1 is still a window; the same $256\to64$ mixing matrix is applied at every pixel |
| Depthwise conv, `groups=C` | Local and shared; it simply refuses to mix channels |
| Dilated conv with $d=8$ | The window has gaps, but it is still a fixed-size window with shared weights |
| A hand-coded Sobel edge filter slid across an image | Local, shared — the classical special case, with the weights chosen rather than learned |
| A 1-D conv over a token sequence in a TextCNN | Same definition, one spatial axis |

**❌ Near-misses — slide, filter, or look local, but aren't convolutions**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| True mathematical convolution $`\sum_u K_u X_{i-u}`$ | It flips the kernel first; the layer in your framework does not | **Cross-correlation.** The flip is absorbed into the learned weights, which is why nobody cares |
| A **locally connected** layer with a $3\times3$ window per position | Local, but each position gets its **own** 9 weights. On a $224\times224$ map that is 50,176 separate kernels | An **unshared local** layer. Used in early face-recognition work; enormous parameter count, no translation equivariance |
| `nn.Linear(224*224*3, 1000)` on a flattened image | Every output touches every input — no locality, no sharing, 150M parameters | A **dense layer.** It *can* represent any convolution; it simply has no reason to |
| Self-attention | The mixing weights are **computed from the data** and change per input; a conv's weights are fixed after training | **Dynamic, global, content-based mixing** (Ch. 11) |
| Transposed convolution / "deconvolution" | It does not invert anything. It is a convolution on a zero-stuffed input | **Fractionally-strided convolution** — an upsampler. The name "deconvolution" is a well-known misnomer |
| Max pooling with a $2\times2$ window | Local, and shared in the sense of "same rule everywhere" — but there are **no weights at all**, and $\max$ is not a weighted sum | A fixed **nonlinear downsampler** |
| Multiplying two images' Fourier transforms elementwise | This *is* a true convolution — but of the circular kind, wrapping around the image boundary | **Circular convolution.** Correct only if your data is  periodic |

▸ **The boundary:** locality plus weight sharing. A dense layer drops both, a locally-connected layer drops sharing, attention replaces fixed weights with computed ones, and pooling drops the weights entirely. **Weight sharing is the load-bearing property** — it is where translation equivariance, the parameter savings, and the entire inductive bias come from.

> **Common misconception.** *"A convolutional layer computes a convolution."* It computes a **cross-correlation**: $`\sum_u K_u X_{i+u}`$, with a plus. The  convolution has a minus — the kernel is flipped end-for-end before sliding. PyTorch, TensorFlow, JAX, and cuDNN all implement the plus version and all call it convolution. This costs you nothing in practice, because if the ideal flipped kernel is $(-1, 0, +1)$, gradient descent on a cross-correlation layer simply learns $(+1, 0, -1)$ and the layer behaves identically. The belief is tempting because the name is right there in the API and because the two operations  coincide for any symmetric kernel — a Gaussian blur, for instance, is its own flip, so the distinction is invisible in the most commonly drawn example. It stops being harmless in exactly one place: **the convolution theorem** ($`\mathcal{F}\{a * b\} = \mathcal{F}\{a\}\cdot\mathcal{F}\{b\}`$) holds for the flipped version, so an FFT-based implementation must flip, and mixing the conventions gives you a silently mirrored kernel.

> **Common misconception.** *"A $3\times3$ convolution has 9 parameters."* It has $`3 \times 3 \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}`$. A $3\times3$ conv from 256 channels to 256 channels holds $9 \cdot 256 \cdot 256 = 589{,}824$ weights — sixty-five thousand times the naive count, and this single layer is larger than all of LeNet-5. The belief is tempting because every diagram of a convolution shows a small square sliding over a flat grey image, which is the $`C_{\text{in}} = C_{\text{out}} = 1`$ case, and that picture never gets updated when the discussion moves to real networks. ▸ **The kernel is not a square, it is a $`3\times3\times C_{\text{in}}`$ box, and you have $`C_{\text{out}}`$ of them.** Getting this wrong makes every FLOP and memory estimate you produce wrong by three to five orders of magnitude.

**Output size:**
▸ $$H_{\text{out}} = \left\lfloor\frac{H+2p-d(k-1)-1}{s}\right\rfloor + 1$$
with padding $p$, stride $s$, dilation $d$. Memorize this; it appears in every shape-debugging session.

#### Unpacking the output-size formula

Read aloud: *"take the height, add the padding on both sides, subtract how far the kernel reaches, divide by the stride, round down, add one."*

Four pieces, each with a job:

- **$H + 2p$** — padding adds $p$ rows on top *and* $p$ on the bottom. That is where the 2 comes from.
- **$d(k-1)$** — the **extra span** of the kernel beyond its first tap. An ordinary $3\times3$ kernel ($d=1$) covers 3 pixels, so it reaches 2 beyond its starting point. A dilated one with $d=2$ still has 9 weights but physically straddles 5 pixels, reaching 4 beyond. So the *effective* kernel width is $d(k-1)+1$.
- **$\lfloor\ \cdot/s\ \rfloor$** — the floor bracket (Ch. 0). If the stamp cannot complete one more full placement before running off the edge, that partial placement is silently **dropped**. This is why a stride that doesn't divide evenly loses a pixel or two, and why shapes mysteriously fail to match up in a decoder.
- **$-1$ then $+1$** — the classic **fencepost correction**. Lay a 3-wide stamp on a 5-wide row with stride 1 and there are 3 valid positions, not 2: you count *posts*, not *gaps*. Subtracting 1 counts gaps; adding 1 converts back to posts.

**Now real numbers.** Take $H = 224$ (the standard ImageNet crop) throughout:

| $H$ | $k$ | $s$ | $p$ | $d$ | $`H_{\text{out}}`$ | What it's for |
|---|---|---|---|---|---|---|
| 224 | 3 | 1 | 1 | 1 | **224** | "same" padding — resolution preserved |
| 224 | 3 | 1 | 0 | 1 | 222 | "valid" padding — you lose the border |
| 224 | 3 | 2 | 1 | 1 | **112** | halve the resolution |
| 224 | 7 | 2 | 3 | 1 | **112** | ResNet's opening "stem" layer |
| 224 | 3 | 1 | 2 | 2 | **224** | dilation 2 with padding 2 — still "same" |

Work the third row by hand: $\lfloor(224 + 2 - 2 - 1)/2\rfloor + 1 = \lfloor 223/2\rfloor + 1 = 111 + 1 = 112$. ✓

▸ **The rule worth memorizing instead of the formula: for stride 1, "same" padding is $p = d(k-1)/2$.** So $k=3\Rightarrow p=1$; $k=5\Rightarrow p=2$; $k=7\Rightarrow p=3$. Notice this only produces a whole number when $k$ is **odd** — which is precisely why essentially every kernel you will ever see is $3\times3$, $5\times5$ or $7\times7$. An even kernel has no centre pixel, so it cannot be placed symmetrically, so it shifts the image by half a pixel every layer.

**Parameters:** $`C_{\text{out}}\cdot C_{\text{in}}\cdot k^2 + C_{\text{out}}`$.
**FLOPs:** $`2\cdot C_{\text{out}}C_{\text{in}}k^2 H_{\text{out}}W_{\text{out}}`$.

**Compared to a dense layer** on a $224\times224\times3$ image with 64 outputs at every position: dense would be $150{,}528\times(224\cdot224\cdot64)=4.8\times10^{11}$ parameters. A $3\times3$ conv is $3\cdot64\cdot9=1{,}728$. **Eight orders of magnitude**, purchased entirely by the two structural assumptions.

#### Where the eight orders of magnitude come from

**The parameter count first.** $`C_{\text{out}}C_{\text{in}}k^2 + C_{\text{out}}`$ is "one weight for every (output channel, input channel, kernel row, kernel column) combination, plus one bias per output channel." Notice what is **missing** from that expression: $H$ and $W$. The image size appears nowhere.

▸ **A convolution's parameter count is completely independent of the image size.** This is why you can pretrain at $224\times224$ and fine-tune at $384\times384$ with the very same weights, and why in vision the thing that runs out is *memory for activations*, not memory for parameters.

**The FLOP count.** For each of the $`H_{\text{out}}W_{\text{out}}`$ output positions and each of the $`C_{\text{out}}`$ channels, you perform $`C_{\text{in}}k^2`$ multiply-accumulate operations. The leading $2$ is because one multiply-accumulate is counted as **two** floating-point operations (FLOP = FLoating-point OPeration): one multiply, one add.

Concretely, a $3\times3$ conv with $`C_{\text{in}}=C_{\text{out}}=64`$ on a $56\times56$ feature map:

- Parameters: $64\cdot64\cdot9 + 64 = 36{,}928$ — about 148 kilobytes in fp32.
- FLOPs: $2\cdot 64\cdot 64\cdot 9\cdot 56\cdot 56 = 2\times 36{,}864 \times 3{,}136 \approx 2.3\times 10^{8}$ — 0.23 GFLOP, for **one layer, one image**.

Double the input resolution to $112\times112$: parameters stay at 36,928, FLOPs **quadruple** to $9.2\times10^8$. Parameters and compute scale on completely different axes.

**Now the eight orders of magnitude, unpacked.** The dense comparison is worth doing slowly because the numbers are so extreme they stop feeling real:

- An input image is $224\times224\times3 = 150{,}528$ numbers.
- Producing 64 numbers at every one of $224\times224$ positions means $224\cdot224\cdot64 = 3{,}211{,}264$ outputs.
- A dense layer connects **every** input to **every** output: $150{,}528 \times 3{,}211{,}264 \approx 4.83\times10^{11}$ weights.
- At 4 bytes each, that is **1.9 terabytes for a single layer.** It does not fit on any machine that exists.

The convolution doing the "same" job needs $3\times64\times9 = 1{,}728$ weights — about **7 kilobytes**. The ratio is $4.83\times10^{11}/1728 \approx 2.8\times10^{8}$.

That factor splits cleanly into the two assumptions:

| Assumption | What it removes | Factor |
|---|---|---|
| **Locality** — an output looks at $k^2=9$ pixels, not all $224^2 = 50{,}176$ | the "look everywhere" cost | $\approx 5{,}600\times$ |
| **Weight sharing** — one kernel reused at all $50{,}176$ positions, not a private one per position | the "learn it again here" cost | $\approx 50{,}176\times$ |

$5{,}600 \times 50{,}176 \approx 2.8\times10^{8}$. The two factors multiply, and together they are the whole saving.

> **Analogy.** Suppose you must write a proofreading manual for a 50,000-page book. The dense approach writes a separate instruction for every position on every page: "on page 12, line 4, character 7, check for a comma." The convolutional approach writes **one** rule — "check for a missing comma" — and says "apply it everywhere." The second manual is eight orders of magnitude shorter, and it also *generalizes*: it works on page 50,001, which the first manual never mentioned.

▸ **The saving is not free — you paid for it with an assumption.** You asserted that a useful feature at one location is useful at every location, and that nearby pixels matter more than distant ones. For natural images those assumptions are close to true, so the approximation error barely rises while the estimation error (Ch. 2) collapses. For a table of unrelated tabular features, the same assumptions are simply false, and a convolution would be actively harmful. **Inductive bias is only a gift when it happens to be correct.**

### Equivariance vs invariance

▸ Convolution is **equivariant** to translation: $\mathrm{Conv}(\mathrm{shift}(x)) = \mathrm{shift}(\mathrm{Conv}(x))$.
Global pooling at the end converts equivariance into **invariance**: $\mathrm{Pool}(\mathrm{shift}(f)) = \mathrm{Pool}(f)$.

Convolution is *not* equivariant to rotation or scale — hence rotation augmentation, and hence group-equivariant CNNs (Ch. 29).

#### Equivariance and invariance, decoded

Two words that sound alike and mean opposite things. Both are claims about **what happens to the output when you move the input.**

| Word | Latin sense | The claim | One-line test |
|---|---|---|---|
| **Equivariant** | "varies together with" | Move the input, the output moves **the same way** | The answer changes, predictably |
| **Invariant** | "does not vary" | Move the input, the output is **unchanged** | The answer doesn't change at all |

Reading $\mathrm{Conv}(\mathrm{shift}(x)) = \mathrm{shift}(\mathrm{Conv}(x))$ aloud: *"stamping a shifted page gives you the same marks, shifted."* It does not matter whether you move first and stamp, or stamp first and move. The two operations **commute**.

> **Analogy.** Equivariance is a **shadow**: move the object and the shadow moves with it, same shape, same size, just relocated. Invariance is a **bathroom scale**: put the object anywhere on the platform and you read the same weight. A convolution is a shadow. A global pool is a scale.

**Watch it happen with real numbers.** One dimension, kernel $(1,1,1)$, $p=1$, input $X = (0,0,1,0,0,0)$ — a single bright pixel at position 3. Then $`Y_i = X_{i-1}+X_i+X_{i+1}`$:

$$Y = (0,\ 1,\ 1,\ 1,\ 0,\ 0)$$

Now shift the input one step right, $X' = (0,0,0,1,0,0)$:

$$Y' = (0,\ 0,\ 1,\ 1,\ 1,\ 0)$$

**The identical bump, moved by exactly one.** That is equivariance, and it holds for any kernel and any shift.

Now apply global max pooling: $\max(Y) = 1$ and $\max(Y') = 1$. **The same number.** That is invariance — and note *where* it came from: not from the convolution, but from the pooling step that threw away position.

▸ **You want equivariance in the middle of the network and invariance at the end.** Equivariance keeps the spatial information alive so later layers can build on it — you cannot detect "wheel below a car body" if you've already forgotten where things are. Invariance is what you finally want from a classifier, because "cat" is the answer regardless of where the cat sits. **Convolution preserves the geometry; pooling discards it, on purpose, at the last possible moment.**

**Why rotation doesn't work the same way.** The nine weights of a $3\times3$ kernel sit at nine *fixed grid positions*. Rotate the image by 90° and the pixel that used to be above the centre is now to the left of it — where a **different weight** is waiting. Nothing in the architecture ties those two weights together, so the network has no reason to respond identically. Two fixes exist: **augmentation** (show the network rotated copies and let it learn the invariance from data, spending capacity to do so) or **group-equivariant CNNs** (Ch. 29, which build the weight-tying into the architecture so the invariance is free and exact). The same choice — learn it or build it in — recurs throughout this book.

### Receptive field

For a stack of layers with kernel $`k_\ell`$, stride $`s_\ell`$, dilation $`d_\ell`$:

▸ $$r_L = 1 + \sum_{\ell=1}^{L} \big(d_\ell(k_\ell-1)\big)\prod_{j<\ell}s_j$$

**Numbers:** ten $3\times3$ convs, stride 1: $r = 1 + 10\cdot2 = 21$ pixels. Add a stride-2 pool after every two convs and it grows geometrically instead.

▸ The **effective** receptive field is much smaller than the theoretical one and is approximately **Gaussian**, with effective radius $\propto\sqrt{L}$ rather than $L$ (Luo et al., 2016) — because the number of paths from a centre pixel to the output vastly exceeds the number from the edge, the influence is a random-walk convolution. **Always assume your effective receptive field is roughly the square root of what the formula says.** This is the practical justification for dilated convolutions and for attention.

#### Reading the receptive-field formula

$$r_L = 1 + \sum_{\ell=1}^{L} \big(d_\ell(k_\ell-1)\big)\prod_{j<\ell}s_j$$

Read aloud: *"start from a single pixel; for each layer, add the extra reach that layer contributes, magnified by the product of every stride that came before it."*

- **$`r_L`$** — the side length, in **original input pixels**, of the square patch that can influence one output value at layer $L$. It is a question about *lineage*: which pixels are this output's ancestors?
- **The leading $1$** — you always see at least yourself.
- **$`d_\ell(k_\ell-1)`$** — layer $\ell$'s own contribution. A $3\times3$ kernel reaches one pixel either side, so it adds $1\cdot 2 = 2$. Dilating it by $d$ multiplies that reach.
- **$`\prod_{j<\ell}s_j`$** — the **magnification factor**, and the reason the formula isn't just a sum. If two stride-2 layers came earlier, then one pixel at layer $\ell$'s input is a $4\times4$ block of the original image, so every step layer $\ell$ takes is worth 4 original pixels. **Strides compound multiplicatively; kernels accumulate additively.**

**Numbers, three ways.** All using $3\times3$ kernels:

| Stack | Receptive field | Cost |
|---|---|---|
| 4 plain convs ($s=1$, $d=1$) | $1 + 2+2+2+2 = 9$ | 4 layers |
| 10 plain convs | $1 + 10\cdot 2 = 21$ | 10 layers |
| 4 **dilated** convs, $d = 1,2,4,8$ | $1 + 2 + 4 + 8 + 16 = \mathbf{31}$ | 4 layers, *identical* parameter count |

▸ **Dilation buys receptive field for free.** Four dilated layers see further than ten plain ones, with 60% fewer parameters and 60% less compute. That is why dilated convolutions dominate segmentation (where you need global context at full resolution) and why WaveNet could model raw audio at 16,000 samples per second — its dilations double all the way to 512, giving a receptive field of thousands of samples in a handful of layers.

**Now the part that catches people out: theoretical vs effective.** The formula answers "which pixels *can* influence the output." It does not answer "which pixels *do*." Those are wildly different questions.

> **Analogy.** A rumour started in one office can, in principle, reach everyone in a city — the social graph is connected. In practice it barely leaves the building, because the number of *routes* from you to your neighbour is enormous and the number of routes from you to a stranger across town is tiny. Influence follows path count, not reachability.

Same thing here. A centre pixel has an overwhelming number of paths to the output; a corner pixel of the theoretical window has almost none. Each layer is a small local blur, so stacking $L$ of them is a **random walk of $L$ steps** — and the spread of a sum of $L$ independent small steps grows like $\sqrt{L}$, not $L$ (the same Central Limit Theorem argument that makes diffusion spread as the square root of time).

**Put a number on the gap.** Stack 100 plain $3\times3$ stride-1 convs. The formula promises $r = 1 + 200 = 201$ pixels. The effective radius grows like $\sqrt{100} = 10$, so the window where a perturbation measurably changes the output is closer to **20 pixels wide**. You built a network that in theory sees the whole image and in practice sees a postage stamp.

▸ **This single fact explains three design decisions at once.** It is why segmentation networks use dilation rather than more depth; why U-Nets downsample aggressively (halving the resolution doubles the reach of every subsequent kernel *for free*); and why attention — where every position reaches every other position in exactly one hop, with no random walk to dilute it — was such a decisive change. **Depth is an expensive and unreliable way to buy receptive field.**

#### Examples and non-examples: when a bigger receptive field actually helps

Receptive field is a *budget item*, not a virtue. It costs depth, parameters, or resolution, and it pays back only when the task  requires evidence from far away. Ask what the label depends on before you buy more of it.

**✅ Tasks where enlarging the receptive field  helps**

| Example | Why it qualifies |
|---|---|
| Segmenting a road that runs diagonally across a $1024\times1024$ street scene | The label at one pixel depends on structure hundreds of pixels away — you cannot tell road from rooftop from a $20\times20$ patch |
| Classifying "beach" vs. "desert" | Both are sand up close. The discriminating evidence (water, sky, vegetation) is elsewhere in the frame |
| WaveNet modelling raw audio at 16 kHz | A phoneme spans thousands of samples; dilations doubling to 512 buy the reach that stacking would not |
| Counting how many people are in a crowded photograph | Requires integrating over the whole image, not deciding locally |
| Detecting a small object *by its context* — a tennis ball identified partly by the court around it | The object is 15 pixels; the disambiguating evidence is not |

**❌ Near-misses — problems that look like they need more receptive field, but don't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Texture classification (fabric, wood grain, cancerous vs. healthy tissue) | Texture *is* a local statistic. A $32\times32$ patch already contains the full signal; averaging more of it adds nothing | A **capacity** or **resolution** problem, not a reach problem |
| Denoising or super-resolving an image | Each output pixel's answer lives in its immediate neighbourhood. Very deep denoisers work through capacity, not reach | A **capacity** problem |
| Segmenting a 20-pixel lesion in a high-resolution scan | The evidence is inside 20 pixels. Aggressive downsampling to grow the receptive field will *destroy* it | A **resolution** problem — you need to downsample *less*, not more |
| "My theoretical receptive field already exceeds the image, so context is covered" | Effective radius grows like $\sqrt L$. ResNet-50's final conv layer has a theoretical receptive field larger than its own $224$-pixel input, and still behaves as if it sees a much smaller window | A **theoretical** receptive field. The number that matters is the effective one |
| Adding 20 more stride-1 $3\times3$ layers to fix a context failure | Adds 40 pixels of theoretical reach and roughly $\sqrt{20}$-ish of effective reach, at full cost in parameters and latency | An expensive way to buy almost nothing. **Dilate or downsample instead** |
| Replacing convs with global attention on a $512\times512$ feature map | Reach becomes global in one hop — and cost becomes $O((512^2)^2)$, i.e. $6.9\times10^{10}$ pairwise terms per head | A different **cost regime**, not a free upgrade |

▸ **The boundary:** more receptive field helps if and only if the label depends on information outside the current window. **Ask "could a human answer this from the crop the network can see?"** If yes, the failure is capacity, resolution, or data — and widening the window will cost you and buy nothing.

> **Common misconception.** *"A bigger receptive field is always better — you want the network to see as much as possible."* Every pixel added to the window is a pixel whose variation the network must now be robust to, and reach is bought with real currency: depth (latency and vanishing signal), stride (destroyed spatial detail), or dilation (gridding artifacts where the sampled positions never interact). A segmentation network that downsamples 32$\times$ to gain context has thrown away the precision needed to place a boundary within a pixel — which is exactly why U-Net's skip connections exist, to hand the fine detail back. The belief is tempting because the receptive field is easy to compute and satisfying to make bigger, and because the field's headline architectural shift — convolution to attention — really was a reach story. But that shift bought *global* reach at quadratic cost for tasks where long-range dependency was the bottleneck; it is not a general instruction to widen every window. **Reach and precision trade against each other, and most vision failures are precision failures.**

> **Common misconception.** *"CNNs are translation invariant."* Convolution is translation **equivariant** — shift the input and the feature map shifts identically, as the worked example above shows. Invariance arrives only when something discards position: global pooling, or a flatten-then-dense layer that has learned to ignore it. And in real networks even the equivariance is imperfect: strided layers and pooling with stride 2 mean a one-pixel shift of the input does not map to a clean shift of the output (Zhang, 2019, showed ImageNet classifiers changing their prediction under a single-pixel shift). The belief is tempting because "the same filter is applied everywhere" *sounds* like "position doesn't matter," and because the end-to-end classifier really does behave roughly invariantly — but that final invariance is contributed by the pooling at the end, not by the convolutions in the middle, and confusing the two makes it impossible to reason about detection and segmentation, where you need position preserved all the way through.

### Variants

| Variant | Definition | Use |
|---|---|---|
| **1×1 conv** | $k=1$ | channel mixing; a per-position dense layer; the bottleneck in ResNet |
| **Dilated/atrous** | insert $d-1$ holes | exponential receptive field growth at constant cost; segmentation, WaveNet |
| **Depthwise** | one kernel per input channel, $`C_{\text{out}}=C_{\text{in}}`$ | $k^2C$ params instead of $k^2C^2$ |
| **Depthwise separable** | depthwise then $1\times1$ | $\frac{k^2C + C^2}{k^2C^2}\approx\frac1{k^2}$ of the cost; MobileNet, Xception |
| **Grouped** | split channels into $g$ groups | AlexNet (GPU memory), ResNeXt (cardinality) |
| **Transposed ("deconv")** | fractionally-strided | upsampling; **causes checkerboard artifacts** — prefer `upsample + conv` |
| **Deformable** | learned sampling offsets | adaptive geometry |

**Depthwise separable arithmetic:** standard $3\times3$ with $C=256$: $9\cdot256^2 = 590$k params. Separable: $9\cdot256 + 256^2 = 67.9$k. **8.7× fewer**, at ~1% accuracy cost. This one substitution is the entire MobileNet family.

#### The variants table, decoded

Every row of that table is a different answer to one question: **a standard convolution mixes across space *and* across channels at the same time — which of those two jobs can we do more cheaply, or skip?**

**$1\times1$ convolution.** Set $k=1$ and the sliding window is a single pixel, so there is **no spatial mixing at all**. What survives is a matrix multiply across the channel axis, applied independently and identically at every position — an ordinary dense layer, run once per pixel. Cost drops from $`C_{\text{in}}C_{\text{out}}k^2`$ to $`C_{\text{in}}C_{\text{out}}`$. Its three jobs: shrink the channel count before an expensive operation, mix information across channels, and (with an activation after it) add a nonlinearity for almost nothing.

**Dilated / atrous.** Insert $d-1$ empty positions between the kernel's taps. The word *atrous* is French — **à trous**, "with holes" — borrowed from the wavelet literature's *algorithme à trous*. A $3\times3$ kernel with $d=4$ still holds exactly 9 weights but straddles a $9\times9$ region. Stack dilations $1,2,4,8,\dots$ and the receptive field doubles per layer at flat cost.

**Depthwise.** Give each input channel its own private $k\times k$ kernel and **never mix channels at all**. Parameters fall from $k^2C^2$ to $k^2C$ — with $C=256, k=3$ that is $589{,}824 \to 2{,}304$, a factor of 256. On its own this is useless (the channels never talk), which is why it always appears paired with a $1\times1$.

**Depthwise separable = depthwise, then $1\times1$.** The insight is that spatial mixing and channel mixing are **separable jobs**, and doing them in sequence costs far less than doing them jointly. Unpack the ratio in the table:

$$\frac{k^2C + C^2}{k^2C^2} \;=\; \frac{1}{C} \;+\; \frac{1}{k^2}$$

For $k=3$ that second term is $1/9 = 0.111$, and $1/C$ vanishes as layers get wide. So the saving converges to exactly $k^2 = 9\times$. Check the book's numbers: $\tfrac1{256} + \tfrac19 = 0.0039 + 0.1111 = 0.115$, and $1/0.115 = 8.7$. ✓

> **Analogy.** To season a hundred dishes with a hundred spices you could taste every combination — or you could season along one axis, then blend along the other. Doing the two operations one after the other rather than all-at-once costs $k^2 + C$ instead of $k^2C$, and for most recipes the difference in the result is imperceptible. That is the whole MobileNet bargain: about **9× cheaper for about 1% accuracy**.

**Grouped.** Split the channels into $g$ groups and let each output group see only its own input group; parameters drop by exactly $g$. Depthwise is the extreme case $g = C$; a standard conv is $g=1$. The history here is worth knowing: AlexNet used $g=2$ **not as an idea but as a workaround** — the model would not fit in the 3 GB of memory on a single GTX 580, so it was split across two GPUs that only communicated at certain layers. Fifteen years later, grouping is a deliberate design axis (ResNeXt calls the group count "cardinality" and treats it as a third scaling dimension alongside depth and width).

**Transposed ("deconvolution").** It **does not** invert a convolution — the name is a misnomer that stuck. What it does is the *transpose of the matrix* that a convolution would apply, which has the effect of scattering each input value into a $k\times k$ patch of a larger output. **Checkerboard artifacts** appear when `stride` does not divide `kernel_size`: some output pixels receive contributions from two overlapping kernel placements and their neighbours receive only one, so the output acquires a periodic bright/dark grid. Fix by decoupling the two jobs — nearest-neighbour or bilinear upsample (which is uniform by construction), then a normal convolution.

**Deformable.** Ordinary convolution samples a fixed grid of offsets $\{(u,v)\}$. Deformable convolution *learns* an additional offset per tap per position, so the sampling pattern can bend to follow an object's shape. It buys geometric adaptivity at the cost of irregular, cache-unfriendly memory access.

### Pooling

Max pooling: local invariance to small translations, gradient routes only to the max. Average pooling: smoother. **Global average pooling** replaced the giant FC head in modern CNNs — it removes ~90% of AlexNet-era parameters and enforces a channel↔class correspondence.

Strided convolution has largely replaced pooling (it is a *learned* downsample).

#### What pooling actually does

Pooling takes a small window — almost always $2\times2$ with stride 2 — and replaces the four numbers in it with one. The output is half as tall and half as wide, so **three quarters of the spatial data is discarded on purpose.**

**Max pooling** keeps the largest of the four. Read it as the question *"was this feature present **anywhere** in this little neighbourhood?"* — which is exactly the small-translation invariance you want: a vertical edge two pixels left of where it was still produces the same answer.

The backward pass is the part worth understanding. **The gradient routes only to the max**: the winning input receives the entire incoming gradient, and the other three receive exactly zero.

> **Analogy.** A winner-take-all election. Only the winner's district gets the budget; the runner-up gets nothing, not even a proportional share. Do this at every layer and the network trains a small number of strongly-selected pathways rather than nudging everything a little.

**Average pooling** takes the mean instead. Every input receives a quarter of the gradient. Smoother, no winner-take-all dynamics, and it preserves low-frequency information that max pooling throws away — which is why average pooling is standard *at the end* of a network and max pooling in the middle.

**Global average pooling (GAP)** is average pooling with a window equal to the entire feature map: it collapses a $C\times H\times W$ tensor to $C$ numbers, one per channel. The parameter arithmetic explains why it took over:

- AlexNet's classifier head was three fully-connected layers on a flattened $6\times6\times256 = 9{,}216$-vector. The first alone is $9{,}216\times 4{,}096 \approx 37.7$M weights; the three together are about **58M of AlexNet's 60M parameters**.
- Global average pooling produces those same 256 (or 2,048) numbers with **zero parameters**, and then a single small classifier layer maps them to classes.

▸ **Global average pooling deleted roughly 97% of a classic CNN's parameters and improved generalization at the same time.** It also creates a direct channel↔class correspondence — each channel becomes "how much of concept $c$ is present in this image," which is what makes class activation maps (a standard interpretability tool) work at all.

**Strided convolution as learned downsampling.** Max pooling hard-codes the rule "keep the biggest." A stride-2 convolution *learns* what to keep, at the cost of parameters. Modern architectures overwhelmingly use the learned version, except for the single max-pool in ResNet's stem, which survives mostly by tradition.

#### Examples and non-examples: downsampling, and the stride/pooling confusion

**Stride** and **pooling** both halve a feature map, and they are constantly used as synonyms. They are not synonyms — they are two different answers to *which* values survive, and one of them has parameters.

**✅  downsampling operations**

| Example | How it halves $56\times56 \to 28\times28$ | Parameters |
|---|---|---|
| `nn.MaxPool2d(2, stride=2)` | Of each $2\times2$ block, keep the largest value | 0 |
| `nn.AvgPool2d(2, stride=2)` | Of each $2\times2$ block, keep the mean | 0 |
| `nn.Conv2d(64, 128, 3, stride=2, padding=1)` | Evaluate the $3\times3$ kernel at every *other* position | $3\cdot3\cdot64\cdot128 = 73{,}728$ |
| `nn.AdaptiveAvgPool2d(1)` at the end of a ResNet | Average all $7\times7$ positions into one number per channel | 0 |
| Patchifying with `nn.Conv2d(3, 768, 16, stride=16)` (ViT stem) | Non-overlapping $16\times16$ windows, each projected | $3\cdot16\cdot16\cdot768$ |

**❌ Near-misses — commonly said to be the same thing, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "Stride 2 *is* pooling" | Stride is a **sampling rate** on a weighted-sum operation with learned weights. Pooling is a **fixed nonlinear reduction** with none | Two mechanisms with the same output shape and different function classes |
| "Pooling is what makes the network smaller" | A stride-2 conv reduces the map identically, and modern nets (ResNet-50 onward, ConvNeXt, ViT) do almost all their downsampling this way | Downsampling. Pooling is one implementation of it, and no longer the dominant one |
| Stride 2 with a $1\times1$ kernel | It reaches only one pixel per output, so it *literally discards* three quarters of the input values, unseen | **Subsampling** — and a known source of aliasing. This is why ResNet-D replaces it with average-pool-then-$1\times1$ |
| Dilation $d=2$ | Samples a wider window but produces the **same output resolution** | A **reach** change, not a resolution change. Dilation and stride are frequently swapped in conversation and do opposite things to the output size |
| Global average pooling | Collapses $7\times7$ to $1\times1$ — it does not halve anything, it annihilates the spatial axes entirely | An **aggregation** step that converts equivariance into invariance |
| A $2\times2$ max-pool with stride 1 | Same window, but the output is the same size as the input | A local **max filter** (a morphological dilation), not a downsampler |

**A concrete comparison.** Take the $2\times2$ block $\begin{pmatrix}1 & 8\\ 2 & 3\end{pmatrix}$:

| Operation | Output | What it kept |
|---|---|---|
| Max pool | $8$ | The strongest response. Robust to small shifts, blind to the other three |
| Average pool | $3.5$ | The total energy. Sensitive to all four, blurs the peak |
| Stride-2 conv with weights $`(w_1,w_2,w_3,w_4)`$ | $`w_1 + 8w_2 + 2w_3 + 3w_4`$ | Whatever the data said was worth keeping |

▸ **The boundary:** pooling chooses *by a fixed rule you wrote down*; stride chooses *by a rate*, and the accompanying convolution's learned weights decide what survives. **Stride costs nothing extra to compute (it computes fewer positions); pooling costs nothing to store (it has no weights). They are not substitutes for each other, and the modern default is stride precisely because "keep the biggest" is a strong assumption nobody actually verified.**

> **Common misconception.** *"Stride and pooling are interchangeable — both just downsample."* They produce the same shape and different functions. A stride-2 convolution is a linear map with $`9 C_{\text{in}} C_{\text{out}}`$ learnable weights that can implement average pooling, learn a low-pass filter, or learn something else entirely; max pooling is a fixed nonlinearity that cannot be adjusted and, being a $\max$, is not even differentiable everywhere. The belief is tempting because in shape-debugging — which is where most people meet both — they  are interchangeable, and the output-size formula is the same. The distinction shows up the moment you care about *what information survived*: a stride-2 $1\times1$ conv silently drops 75% of its input pixels without ever looking at them, which is an aliasing bug that pooling would not have had, and which several ResNet variants exist specifically to repair.

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

#### Reading the residual update, and why "degradation" is the right word

First the diagnosis, because it is the more surprising half. A 56-layer plain network had **higher training error** than a 20-layer one. Sit with why that is impossible on paper: take the trained 20-layer network, then add 36 layers that each compute the identity. That construction has *exactly* the 20-layer network's training error. So the 56-layer network's optimum is at least as good — the hypothesis class strictly contains the smaller one.

▸ **The deeper network could represent the better solution and gradient descent could not find it.** This is why the word is *degradation*, not *overfitting*. Overfitting is a statistics problem (Ch. 2): too much capacity, not enough data, training error goes down while test error goes up. Degradation is an **optimization** problem: training error itself gets worse. Those require completely different fixes, and confusing them wastes months.

**Now the equation.** $`x_{\ell+1} = x_\ell + F(x_\ell;\theta_\ell)`$

| Piece | Read aloud | What it is |
|---|---|---|
| $`x_\ell`$ | "x sub ell" | The whole activation tensor entering block $\ell$ |
| $\ell$ | "ell" | A **layer index**, not a loss (Ch. 0, Trap 3) |
| $`F(\cdot\,;\theta_\ell)`$ | "F of dot, given theta-ell" | The **residual branch** — typically two or three convs with BN and ReLU, holding their own weights $`\theta_\ell`$ |
| $`\theta_\ell`$ | "theta sub ell" | This block's parameters |
| $+$ | "plus" | Ordinary **elementwise** addition — which is why the shapes must match, and why you need a $1\times1$ projection when they don't |

**Where the name comes from.** Suppose the block ought to compute some function $\mathcal{H}(x)$. Rearranging, $F(x) = \mathcal{H}(x) - x$: the branch learns the **residual**, the leftover, the difference between what you already have and what you want. If the answer is "you already have it," the residual is zero.

**Why zero is easy and identity is hard.** For a plain block $`x_{\ell+1} = \sigma(Wx_\ell)`$ to compute the identity, $W$ must be *exactly* the identity matrix and the activation must be linear over the relevant range — a single measure-zero point in a space of millions of dimensions, which gradient descent has no particular reason to visit. For a residual block, "do nothing" means $F = 0$, and $F=0$ happens whenever the branch's final weights are near zero — which is **where you initialize** and where weight decay pulls you back.

> **Analogy.** A relay of 56 scribes copying a manuscript. In the plain design, each scribe reads the previous copy and rewrites the whole page from scratch; small errors accumulate and by scribe 56 the text is unrecognisable, and there is no way for a scribe to "do nothing" — copying *is* the job. In the residual design, each scribe receives the page itself and writes only **corrections in the margin**. A scribe with nothing to add adds nothing, which costs no effort and introduces no error. After 56 scribes the original is still legible underneath.

▸ **Residual connections changed the default from "transform" to "pass through," and made transforming the thing you have to earn.** Everything else in this section — the gradient highway, the ensemble view, the smooth landscape — is a consequence of that one reversal of defaults.

> **Where this came from.** Residual networks were introduced by **Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun** at Microsoft Research Asia, published at the end of 2015. The 152-layer version won the ImageNet classification challenge that year with a 3.57% top-5 error — below the roughly 5% often quoted for careful human annotators on the same task. The paper also trained a **1,202-layer** network on CIFAR-10, which optimized perfectly well and merely overfit: the point was to demonstrate that optimization had stopped being the barrier.
>
> A gated ancestor arrived a few months earlier the same year: **Highway Networks**, by Rupesh Srivastava, Klaus Greff and Jürgen Schmidhuber, which used a learned gate $T(x)$ to interpolate between the transformed and the untransformed path — $y = T(x)\odot F(x) + (1 - T(x))\odot x$ — and drew explicitly on the LSTM's gating (Ch. 9). ResNet is the **ungated simplification**: delete the gate, hardwire it open, and the thing works better and trains faster. It is one of the cleanest cases in the field of a good idea improved by removing machinery rather than adding it. "Deep Residual Learning for Image Recognition" has since become one of the most-cited scientific papers ever written, in any discipline.

### Why it works: three explanations, all partly right

**1. Gradient highway.** Unroll:
$$x_L = x_\ell + \sum_{i=\ell}^{L-1}F(x_i)\quad\Rightarrow\quad \frac{\partial\mathcal{L}}{\partial x_\ell} = \frac{\partial\mathcal{L}}{\partial x_L}\left(1 + \sum_{i=\ell}^{L-1}\frac{\partial F(x_i)}{\partial x_\ell}\right)$$

▸ The leading **1** guarantees the gradient reaches every layer regardless of what the residual branches do. Contrast Ch. 6: plain nets multiply $L$ Jacobians and die; residual nets *add* corrections to an identity. This is the strongest of the three explanations.

**2. Ensemble of shallow paths** (Veit et al.). A ResNet with $L$ blocks has $2^L$ paths from input to output (take or skip each block). Deleting one block from a trained ResNet barely hurts — unlike a plain net, where deleting a layer is catastrophic. The measured gradient contribution is dominated by paths of length ~10–30, so a 110-layer ResNet behaves like an ensemble of relatively shallow networks.

**3. Loss-landscape smoothing** (Li et al., 2018). Visualizations of the loss surface along random 2-D slices show plain deep nets have chaotic, fractal-looking landscapes and ResNets have smooth, convex-looking basins. Cause and effect are hard to separate here, but the correlation is striking.

#### Unpacking the gradient highway

**Where the unrolling comes from.** Apply $`x_{i+1} = x_i + F(x_i)`$ repeatedly and watch the terms accumulate:

$$x_{\ell+1} = x_\ell + F(x_\ell)$$
$$x_{\ell+2} = x_{\ell+1} + F(x_{\ell+1}) = x_\ell + F(x_\ell) + F(x_{\ell+1})$$

Keep going and you get $`x_L = x_\ell + \sum_{i=\ell}^{L-1}F(x_i)`$: **the activations at the top are the activations at layer $\ell$ plus a pile of corrections.** The original signal is never multiplied by anything. It is *literally still there*, additively, at the top of a 152-layer network.

Differentiate that with respect to $`x_\ell`$ and read the result aloud:

$$\frac{\partial\mathcal{L}}{\partial x_\ell} = \frac{\partial\mathcal{L}}{\partial x_L}\left(1 + \sum_{i=\ell}^{L-1}\frac{\partial F(x_i)}{\partial x_\ell}\right)$$

*"The gradient arriving at layer $\ell$ equals the gradient at the top, times one-plus-a-correction."* The $1$ comes from differentiating the $`x_\ell`$ term — the skip path — and it does not depend on any weight anywhere in the network.

**Put numbers on the contrast.** Suppose each layer's Jacobian typically scales gradients by 0.8, and there are 50 layers:

| Architecture | Gradient reaching layer 1 |
|---|---|
| **Plain**: multiply 50 Jacobians | $0.8^{50} = 1.4\times10^{-5}$ |
| **Residual**: the bracket is $1 + (\text{stuff})$ | $\approx 1$, whatever "stuff" is |

Even if every single $\partial F/\partial x$ term were $10^{-9}$ — a catastrophically dead residual branch — the bracket would be $1.000000001$ and the gradient would still arrive intact.

▸ **The whole difference is between multiplying by 0.8 fifty times and adding fifty small numbers to 1.** Multiplication compounds and annihilates; addition does not. That sentence is also the answer to "why do transformers have residual connections," "why does the LSTM's cell state work," and "why do diffusion models predict noise rather than images."

**One important caveat the formula hides.** The leading $1$ only survives if the skip path is a **clean identity**. Put a BatchNorm on the skip, or a $1\times1$ convolution, or a gate, and that $1$ becomes a matrix — and matrices multiplied 50 times behave exactly like $0.8^{50}$ again. This is precisely why **pre-activation ResNet-v2** moves every normalization and activation *inside* the branch and leaves the skip path completely bare, and it is why 1000-layer ResNets became possible only after that change.

#### The ensemble view, with numbers

Every block is a fork: the signal either goes through $F$ or around it. With $L$ blocks that is $2^L$ distinct routes from input to output. For a 110-block ResNet, $2^{110} \approx 1.3\times 10^{33}$ paths — more routes than there are atoms in a human body.

Not all paths matter equally. A path that passes through $m$ residual branches has its gradient multiplied by $m$ Jacobians, so its contribution decays roughly geometrically in $m$. Meanwhile the *number* of paths of length $m$ is $\binom{L}{m}$, which is astronomically larger for middling $m$. Multiply "how many" by "how strong" and the product peaks somewhere in the middle — Veit et al. measured the peak at paths of length **10 to 30**.

> **Analogy.** A city with a thousand parallel bypass roads between two points versus a single-lane highway. Close one bypass and traffic barely notices; the other 999 absorb it. Close one lane of the highway and everything stops. That difference is not a metaphor for the ResNet — it is the mechanism.

▸ **This view made a falsifiable prediction, and the prediction held.** Delete a whole block from a *trained* ResNet and accuracy drops by a percent or two. Delete a layer from a trained VGG and the network is destroyed. A plain network is a chain, where every link is load-bearing; a residual network is a mesh, where no single link is.

**How the three explanations relate.** They are not competitors so much as three views of one fact. The identity path (1) is what creates the many parallel routes (2), and having many routes of varied length is a plausible mechanism for why no single direction in weight space is catastrophic, which is what a smooth landscape (3) looks like. The gradient-highway argument is the one you should reach for first, because it is exact algebra rather than measurement.

#### Examples and non-examples: is that a residual connection?

The definition is narrow and the narrowness is the point: **the block's own, unmodified input is added to the block's output.** Everything the previous two subsections proved depends on the skip path being an untouched identity — the leading $1$ in the gradient, the $2^L$ routes, the fact that the block can become a no-op by driving $F$ to zero.

**✅  residual connections**

| Example | The equation | Why it qualifies |
|---|---|---|
| A ResNet-v2 basic block | $`x_{\ell+1} = x_\ell + F(x_\ell)`$ | Bare skip path; setting $F = 0$ recovers the identity exactly |
| A transformer sublayer | $x + \mathrm{Attn}(\mathrm{LN}(x))$ | The normalization sits *inside* the branch; the skip is untouched |
| Diffusion models predicting $\epsilon$ rather than $`x_0`$ | $`\hat x_0 = (x_t - \sigma\hat\epsilon)/\alpha`$ | The network outputs a *correction* to what it was handed |
| A ResNet stage with 3 blocks at constant width | $`x + F_1`$, then $`+F_2`$, then $`+F_3`$ | Three clean additions in a row — the mesh of §8.2 |
| Zero-initializing the last BN's $\gamma$ in each branch | $`x_{\ell+1} = x_\ell + 0`$ at step 0 | Starts the network *as* the identity and lets it earn its depth |

**❌ Near-misses — skip something, but aren't the residual connection the algebra assumes**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| DenseNet's $`[x_\ell, F(x_\ell)]`$ | **Concatenation**, not addition. The channel count grows every block; there is no "$1 +$" to differentiate | **Dense connectivity.** Feature reuse rather than a gradient identity |
| U-Net's encoder→decoder skips | Span the whole network across a resolution change, and are concatenated | **Long-range skips** carrying spatial detail past the bottleneck (§8.4) |
| Highway networks: $T(x)\odot F(x) + (1-T(x))\odot x$ | The skip is multiplied by a **learned gate**. Differentiating gives $(1-T)$ where ResNet gives $1$ — and 50 such factors multiply | A **gated** skip. The direct ancestor, and the gate is exactly what ResNet removed |
| The shortcut $1\times1$ stride-2 conv used when shapes change | The "identity" is now a matrix $`W_s x`$; the leading term is $`W_s`$, not $1$ | A **projection shortcut** (option B). Necessary, but each one is a break in the highway — ResNet-50 has only four |
| A BatchNorm placed **on** the skip path | Same problem: $1$ becomes a Jacobian, and 50 Jacobians multiply back down to $0.8^{50}$ territory | The original post-activation ResNet-v1 arrangement — which is *why* v2 exists |
| Feeding the raw input directly to the final classifier | A single long jump, not a per-block addition; the intermediate layers get no identity path | A **skip to output** / wide-and-deep pattern |
| "Residuals" in a regression diagnostic plot | $y - \hat y$: a measured error, not a piece of architecture | The **statistical** sense of the word. Same etymology, unrelated object |
| $`x_{\ell+1} = 0.5x_\ell + 0.5F(x_\ell)`$ | The coefficient on the skip is $0.5$, and $0.5^{50} = 8.9\times10^{-16}$ | A **convex-combination** update — and a good illustration of why the coefficient must be exactly 1 |

▸ **The boundary:** the skip path must be the identity — no weights, no normalization, no gate, no coefficient. **Anything you put on the skip converts the constant $1$ into something that gets multiplied $L$ times, and that is precisely the disease residual connections were invented to cure.**

> **Common misconception.** *"ResNets work because they let gradients flow — they solve the vanishing-gradient problem."* Gradient flow is real and is the cleanest piece of algebra in the chapter, but it is not the problem the paper set out to solve, and by 2015 it was largely already handled. He et al. state directly that the plain networks they compared against were trained *with BatchNorm*, that they verified the backward-propagated gradients had healthy norms, and that vanishing gradients were therefore unlikely to be the cause. What they observed was **degradation**: a 56-layer plain network with *higher training error* than a 20-layer one — a network failing to fit data it demonstrably had the capacity to fit, since it contains the shallower net as a special case. The disease was that SGD could not *find* the identity mapping inside a stack of convolutions; the cure was to make the identity the default rather than something to be discovered. The belief is tempting because "vanishing gradients" is the memorable phrase attached to every depth problem in deep learning, and because the gradient-highway derivation is  correct — it just answers a question that was not the bottleneck. ▸ **Say "degradation" rather than "vanishing gradients" and you will be right about both the history and the mechanism.**

### Block designs

**Basic (ResNet-18/34):** conv3×3 → BN → ReLU → conv3×3 → BN → (+x) → ReLU.

**Bottleneck (ResNet-50+):** conv1×1 (reduce $C\to C/4$) → conv3×3 → conv1×1 (expand). Costs $\sim\frac{1}{4}$ the FLOPs of two 3×3s at the same width. **ResNet-50 is 4× deeper than ResNet-34 at similar FLOPs.**

**Pre-activation (ResNet-v2):** BN → ReLU → conv → BN → ReLU → conv → (+x), with *nothing* on the skip path. This is the residual analogue of pre-LN, and it's what allows 1000-layer ResNets.

**Shortcut when shapes change:** either a $1\times1$ stride-2 conv (option B, standard) or zero-padding (option A). Option B is better but adds parameters.

#### The three block designs, decoded

**Basic block.** Two $3\times3$ convolutions, each followed by BatchNorm, with a ReLU between them and another after the addition. Parameter cost at width $C$: $2\cdot 9C^2 = 18C^2$.

**Bottleneck block** — the design that makes deep ResNets affordable. Read the three convolutions as *squeeze, work, expand*:

1. $1\times1$ conv taking $C \to C/4$ — **squeeze** the channels down cheaply.
2. $3\times3$ conv at $C/4 \to C/4$ — do the expensive spatial work in the **narrow** space.
3. $1\times1$ conv taking $C/4\to C$ — **expand** back so the addition's shapes match.

Count it with $C = 256$ (so $C/4 = 64$):

| Step | Parameters |
|---|---|
| $1\times1$, $256\to64$ | $16{,}384$ |
| $3\times3$, $64\to64$ | $36{,}864$ |
| $1\times1$, $64\to256$ | $16{,}384$ |
| **Total** | $\mathbf{69{,}632}$ |
| Two $3\times3$ at $256\to256$ (basic block) | $1{,}179{,}648$ |

**Seventeen times cheaper**, and the $3\times3$ — the only part that costs $k^2$ — now runs at a quarter of the width, where it costs sixteen times less.

> **Analogy.** A narrow doorway between two large rooms. Everything must funnel through it, so whatever expensive processing happens at the doorway only has to handle a quarter of the traffic. You pay two cheap conversions at the ends to make the expensive middle small.

▸ **This is why ResNet-50 is four times deeper than ResNet-34 at comparable total compute.** The bottleneck is the reason "deeper" stopped being expensive, and the identical squeeze–work–expand pattern reappears as the transformer's feed-forward network, MobileNetV2's *inverted* residual (which expands then squeezes, because depthwise convolutions are cheap in the wide space), and every adapter and LoRA module in Chapter 17.

**Pre-activation (ResNet-v2).** Move BatchNorm and ReLU to the *front* of each convolution, so the block is BN → ReLU → conv → BN → ReLU → conv, and the addition is the last thing that happens. The skip path then carries a completely unmodified $`x_\ell`$: **no normalization, no activation, nothing.** By the caveat in the gradient-highway section, that is exactly the condition under which the leading $1$ survives all the way down, and it is what allowed 1000-layer networks to train.

This is the same rearrangement as **pre-LN** versus **post-LN** in transformers (Ch. 7 and Ch. 11), for exactly the same reason, discovered independently in the two literatures. Learn the principle once: **keep the residual highway clean and push everything else into the branch.**

**Shortcut projections.** When a block changes resolution or channel count, $`x_\ell`$ and $`F(x_\ell)`$ no longer have matching shapes, so the $+$ is undefined. Option A pads the extra channels with zeros (free, no parameters). Option B inserts a $1\times1$ stride-2 convolution on the skip (better accuracy, but note that it puts a matrix back on the highway — which is why these projections appear only at the three or four resolution changes in the network, never in every block).

### The BatchNorm–residual interaction

At initialization, $`\mathrm{Var}(x_{\ell+1}) = \mathrm{Var}(x_\ell)+\mathrm{Var}(F)`$, so variance grows **linearly with depth** and the signal-to-residual ratio degrades. BatchNorm inside the block rescales $F$ so this stays controlled.

▸ **Zero-initializing the last BN's $\gamma$ in each block** makes every block start as exact identity. This is standard (`zero_init_residual=True`) and gives ~0.5% top-1 for free. It is the same idea as **Fixup**, **ReZero**, and **AdaLN-Zero** (Ch. 21). Learn it once; it recurs everywhere.

#### Why the variance grows, and what zero-init fixes

**Reading $`\mathrm{Var}(x_{\ell+1}) = \mathrm{Var}(x_\ell)+\mathrm{Var}(F)`$.** $\mathrm{Var}$ is variance — the average squared distance from the mean, i.e. how *spread out* the numbers are (Ch. 1). The identity holds because variances of **independent** quantities add, and at initialization the branch's random output is essentially uncorrelated with its input. Two things being added and neither being subtracted means the spread only ever grows.

**The consequence, with numbers.** Start with $`\mathrm{Var}(x_0) = 1`$ and let every block contribute variance $1$. Then $`\mathrm{Var}(x_L) = L+1`$:

| After $L$ blocks | Variance | Typical activation magnitude | One block's share of the signal |
|---|---|---|---|
| 1 | 2 | $1.41$ | 50% |
| 10 | 11 | $3.32$ | 9% |
| 50 | 51 | $7.14$ | 2% |
| 100 | 101 | $10.05$ | 1% |

Two separate problems live in that table. **Activations grow like $\sqrt{L}$**, which pushes them into the saturating or numerically awkward parts of later operations. And the **relative contribution of each new block shrinks like $1/L$** — block 100 is whispering into a shout, so its gradient signal is proportionally tiny and it learns slowly.

BatchNorm inside the branch fixes the first problem by rescaling $F$'s output to unit variance regardless of what came before.

**Now the zero-init trick.** BatchNorm's output is $\gamma\hat{x} + \beta$, where $\gamma$ is a *learned* per-channel scale. Set $\gamma = 0$ in the **last** BatchNorm of each residual branch and that branch outputs exactly zero, so $`x_{\ell+1} = x_\ell`$ **exactly**. Every block is a perfect identity at step zero.

▸ **A network initialized this way starts out effectively shallow and grows its own depth.** At step 0 it is a linear stem plus a classifier — trivially easy to optimize. As each $\gamma$ drifts off zero, that block switches itself on. You are not training a 152-layer network from scratch; you are training a 1-layer network that recruits 151 more as it needs them. Half a percent of top-1 accuracy for one line of initialization code is one of the best returns in the entire field.

**The same idea under four names**, which is why it is worth learning once:

| Name | Where | What it zeroes |
|---|---|---|
| `zero_init_residual` | ResNets | the last BatchNorm's $\gamma$ in each block |
| **Fixup** | norm-free ResNets | scales the branch weights down by a depth-dependent factor, no normalization needed |
| **ReZero** | any residual architecture | a single learned scalar on the branch, initialized to 0 |
| **AdaLN-Zero** | diffusion transformers (Ch. 21) | the conditioning modulation, so each block starts as identity |

They all say: **start every block as a no-op and let gradient descent decide which ones to turn on.**

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
| DenseNet | 2016 | concatenate all previous features: $`x_\ell = H([x_0,\dots,x_{\ell-1}])`$; feature reuse, $O(L^2)$ connections | 77.7% | 8M |
| SENet | 2017 | **channel attention**: squeeze (GAP) → excite (2 FC + sigmoid) → rescale channels. The first widely-used attention in vision. | 82.7% | 146M |
| MobileNetV2 | 2018 | inverted residual + linear bottleneck | 72.0% | 3.4M |
| **EfficientNet** | 2019 | **compound scaling**: $d=\alpha^\phi, w=\beta^\phi, r=\gamma^\phi$ s.t. $\alpha\beta^2\gamma^2\approx2$ — scale depth, width, resolution together | 84.3% (B7) | 66M |
| ConvNeXt | 2022 | a ResNet modernized with ViT design choices (7×7 depthwise, LN, GELU, fewer norms, inverted bottleneck) — matches Swin | 87.8% | 198M |

▸ **The ConvNeXt lesson, which is the one worth carrying:** most of the ViT-over-CNN gap in 2021 was *training recipe and design detail*, not attention. When a ResNet is given AdamW, 300 epochs, RandAugment, Mixup, stochastic depth, LayerNorm, GELU, and large depthwise kernels, it matches transformers on ImageNet. **Architecture comparisons at unequal training recipes are worthless.**

#### Reading the lineage table

**What the columns mean.** "Top-1 (IN-1k)" is the percentage of ImageNet-1k validation images for which the model's single highest-scoring guess is the correct one out of 1,000 classes. Random guessing scores 0.1%. "Params" is the parameter count — and the most instructive thing in the whole table is that **it does not move monotonically with accuracy**. GoogLeNet is competitive with VGG-16 using roughly **twenty times fewer** parameters, and ResNet-50 beats both with a fifth of VGG's weights. (Treat all published accuracy figures as approximate: they depend on input resolution, test-time cropping, and which variant of a family is being reported, so numbers from different sources rarely line up exactly.)

> **A caution about the Inception row specifically.** That row compresses an entire *family* into one line, and the pieces come from different members of it. **GoogLeNet — Inception-v1, the 2014 ILSVRC winner — is the model with roughly 6.8M parameters**, and its commonly-reported single-crop top-1 accuracy is in the high 60s, not 74.8%. The **$n\times n \to n\times1,\ 1\times n$ factorization listed as its key idea belongs to Inception-v3** (Szegedy et al., *Rethinking the Inception Architecture*, 2015), not to v1. So the row pairs v1's parameter count with a later variant's accuracy and a later variant's headline technique.
>
> **Nothing in the argument depends on the discrepancy** — the point being made is that parameter count and accuracy are decoupled, and that holds with either figure (6.8M against VGG-16's 138M is a 20× gap whether the accuracy is 69.8% or 74.8%). But it is a good illustration of the caveat in the parenthesis above, and of a broader hazard: **"Inception" names four architectures spanning 2014–2016, and comparison tables routinely blur them.** When you see an Inception number quoted, ask which version.

#### Examples and non-examples: reading an architecture comparison table

**✅ Conclusions this table genuinely supports**

| Claim | Why it holds |
|---|---|
| Parameter count and accuracy are decoupled | GoogLeNet beats VGG-16's efficiency by ~20× on params |
| Architectural ideas compound over time | Each row inherits the previous rows' tricks |
| Efficiency was a design target, not an accident | MobileNetV2 at 3.4M params is deliberately small |

**❌ Conclusions it does *not* support**

| Tempting reading | Why it's wrong |
|---|---|
| "ConvNeXt is better than ResNet-50 by 11.7 points" | Different eras, different training recipes, different resolutions. Not a controlled comparison |
| "SENet is worse than EfficientNet per parameter" | Rows differ in input resolution and augmentation, not just architecture |
| "Accuracy improved steadily each year" | Much of the gain came from **training recipes**, not architecture — which is the ConvNeXt lesson below |
| "These numbers are directly comparable" | They come from different papers with different evaluation protocols |
| "2014's Inception got 74.8%" | See the caution above — that's a later variant |

▸ **The boundary:** a lineage table shows *what people tried and roughly where it landed*, not a controlled experiment. **Architecture comparisons at unequal training recipes are worthless** — which is precisely the ConvNeXt finding stated immediately below, and it retroactively undermines most cross-era comparisons anyone draws from tables like this one, including this one.

**Where the names come from,** since half of them are jokes or acronyms:

| Name | Where the name comes from |
|---|---|
| LeNet | Yann **Le**Cun's network |
| AlexNet | **Alex** Krizhevsky, its first author |
| VGG | Oxford's **V**isual **G**eometry **G**roup |
| GoogLeNet | Google + a deliberate tribute to Le**Net**, with the capitalization to make it obvious |
| Inception | the film — the paper explicitly cites the internet meme "we need to go deeper" |
| ResNet | **Res**idual **Net**work |
| ResNeXt | ResNet + the **next** dimension (cardinality) |
| SENet | **S**queeze-and-**E**xcitation Network |
| Xception | "**Extreme** Inception" |

**Three rows worth working through.**

**VGG's argument, in arithmetic.** Two stacked $3\times3$ convolutions see a $5\times5$ patch (each adds a reach of 2, so $1+2+2 = 5$). Their cost is $2\times 9C^2 = 18C^2$; a single $5\times5$ costs $25C^2$. So you get the same receptive field for **28% fewer parameters** *and* you get two ReLUs instead of one, which is strictly more expressive. Three stacked $3\times3$s match a $7\times7$ at $27C^2$ against $49C^2$ — a 45% saving. ▸ **This single observation is why the $3\times3$ kernel became the near-universal default for a decade**, and it is a textbook example of a result that is obvious once stated and was not obvious before.

**Inception's argument.** Rather than choosing a kernel size, run $1\times1$, $3\times3$, $5\times5$ and pooling **in parallel** and concatenate the results — let the network weight them itself. The $1\times1$ bottlenecks placed before the expensive branches are what keep this affordable. Later versions add *factorization*: an $n\times n$ convolution is replaced by an $n\times 1$ followed by a $1\times n$, costing $2n$ weights per channel pair instead of $n^2$ — for $n=7$ that is 14 instead of 49.

**EfficientNet's compound scaling, decoded.** You can make a network bigger along three axes: **depth** $d$ (more layers), **width** $w$ (more channels), and **resolution** $r$ (bigger images). The proposal is to scale all three together, governed by one knob $\phi$:

$$d = \alpha^\phi,\qquad w = \beta^\phi,\qquad r = \gamma^\phi,\qquad \text{subject to}\quad \alpha\beta^2\gamma^2 \approx 2$$

- $\phi$ ("phi") is the single dial the practitioner turns — "how much compute do I have?"
- $\alpha,\beta,\gamma$ are fixed ratios found once by a small grid search, then never changed.
- The constraint $\alpha\beta^2\gamma^2\approx 2$ exists because **FLOPs scale linearly with depth but quadratically with both width and resolution.** (Doubling the channels quadruples the work in a conv, since both $`C_{\text{in}}`$ and $`C_{\text{out}}`$ double; doubling the resolution quadruples the number of output positions.) So the constraint says "one step of $\phi$ doubles the compute" — and total cost is then simply $2^\phi$.

▸ **The insight is that the three axes are complementary, not interchangeable.** A very deep network on tiny images has receptive field it cannot use; a very wide network with few layers has capacity but no compositional depth; high resolution with a shallow network has detail nothing can integrate. Scaling one axis alone saturates quickly. Scaling all three in a fixed ratio does not. **The same logic, with different axes, is exactly what Chapter 15's scaling laws do for language models** — there too the finding is that parameters and data must grow together, and that growing one alone wastes the other.

> **Where this came from.** The row that changed everything is **AlexNet, 2012**. Krizhevsky, Sutskever and Hinton cut the ImageNet top-5 error from about 26% to about 15% in a single year — a margin so far outside the usual annual increment that the field reorganized around it within months. It trained for five to six days on **two consumer GTX 580 gaming cards**, which is where the grouped convolutions came from: 3 GB of memory each was not enough to hold the model, so it was split. Nothing in AlexNet was conceptually new relative to LeCun's 1989 work; what was new was ReLU, dropout, enough labelled data, and GPUs.
>
> That data is its own story. **ImageNet** was assembled beginning in 2007 by **Fei-Fei Li** and collaborators, organized by the WordNet noun hierarchy and labelled through Amazon Mechanical Turk on a scale that was widely regarded at the time as an odd use of effort — the prevailing view was that better algorithms, not more data, were the bottleneck. The competition built on it ran from 2010 to 2017.
>
> **ConvNeXt (2022)** deserves reading as a *methodology* paper rather than an architecture paper. Its authors took a ResNet-50 and changed one thing at a time — the training recipe, the stage compute ratios, the stem, depthwise convolutions, an inverted bottleneck, a $7\times7$ kernel, fewer activations, fewer normalizations, LayerNorm instead of BatchNorm — reporting the accuracy after each single change. The conclusion was that a convolutional network given a transformer's training recipe matches a transformer. ▸ **The transferable lesson is about experimental hygiene, not about convolutions: if you compare two architectures under different recipes, you have measured the recipes.**

---

## 8.4 U-Net

The dominant architecture for dense prediction, and the backbone of most diffusion models before DiT.

**Structure:** an encoder that downsamples ($H\to H/2\to H/4\dots$) while widening channels, a bottleneck, and a decoder that upsamples symmetrically — with **skip connections concatenating each encoder feature map to the decoder at matching resolution.**

▸ **Why the skips matter:** the encoder path destroys spatial precision (that's what downsampling does) while building semantic abstraction. The skips restore high-frequency spatial detail that the bottleneck cannot carry. Without them, segmentation boundaries are mush.

**In diffusion (Ch. 20):** the U-Net is augmented with (i) timestep embedding injected via FiLM/AdaGN into every block, (ii) self-attention at the lower resolutions (typically 16×16 and 8×8 — full attention at 64×64 is too expensive), (iii) cross-attention for text conditioning, (iv) GroupNorm (batch-independent), (v) SiLU activations.

**Why DiT replaced it:** the U-Net's multi-resolution inductive bias is valuable at low compute but becomes a constraint at scale; a plain transformer over latent patches scales more predictably and follows the same scaling laws as language models (Ch. 21).

#### The U-Net, decoded

**Why it is called a U.** Draw the layers left to right and put resolution on the vertical axis: the encoder descends ($224 \to 112 \to 56 \to 28$), the bottleneck sits at the bottom, the decoder climbs back up ($28\to 56\to 112\to 224$). The picture is a U. The skip connections are the horizontal rungs joining the two arms at matching heights.

**"Dense prediction"** means the output has one value *per pixel* — a segmentation mask, a depth map, a denoised image — rather than one value per image. Classification throws away spatial resolution deliberately; dense prediction has to get it back.

**The tension the U-Net resolves.** Downsampling does two things simultaneously, one wanted and one not:

| Downsampling gives you | Downsampling costs you |
|---|---|
| Larger effective receptive field per layer (a $3\times3$ at $\tfrac14$ resolution sees a $12\times12$ patch of the original) | Spatial precision — after three halvings, a pixel's location is known only to within 8 original pixels |
| Cheaper computation (quarter the positions per halving) | High-frequency detail: edges, thin structures, texture |

You need both. Semantic understanding ("this region is liver") requires context, which requires downsampling. Precise boundaries ("the edge is *here*, not three pixels left") require full resolution, which downsampling destroys.

▸ **The skip connections are the resolution of that conflict: send the semantics down the U and the geometry across it.** The decoder receives, at every resolution, both the abstract low-resolution answer coming up from below *and* the sharp high-resolution features from the matching encoder layer. Without the skips, the network knows *what* but not *where*, and segmentation boundaries come out as mush — this is not a subtle degradation, it is immediately visible.

> **Analogy.** Tracing a map. To decide "this is a river, not a road" you need to step back and see the whole region — but stepping back means you can no longer see exactly where the bank is. So you do both: step back to identify it, then put your finger back on the detailed map to trace it precisely. The skip connection is putting your finger back on the detailed map.

**Note the difference from a residual connection.** A ResNet skip **adds** ($x + F(x)$, shapes must match). A U-Net skip **concatenates** ($`[\,x_{\text{enc}},\, x_{\text{dec}}\,]`$, channel counts add up and the next convolution decides how to combine them). Addition forces the two signals into the same representational space; concatenation keeps them separate and lets the network learn the mixture. Both are "skip connections" and they are not the same operation.

**The diffusion additions, in plain terms** (all of them arrive properly in Ch. 20):

- **Timestep embedding via FiLM/AdaGN** — the network must behave differently at noise level 900 than at noise level 10, so the timestep is turned into a vector and used to produce a per-channel scale and shift applied inside every block. FiLM = **Feature-wise Linear Modulation**: literally $\gamma(t)\cdot h + \beta(t)$.
- **Self-attention at low resolutions only** — attention costs $O(N^2)$ in the number of positions. At $64\times64$ that is $4096^2 \approx 1.7\times10^7$ pairs per head; at $16\times16$ it is $256^2 = 65{,}536$, about 256 times cheaper. So attention is affordable exactly where the U-Net has already downsampled.
- **Cross-attention for text** — the same mechanism, but the keys and values come from the text encoder instead of the image. This is the entire mechanism by which a prompt steers an image.
- **GroupNorm rather than BatchNorm** — because BatchNorm's statistics depend on the other images in the batch, which makes generation depend on what else happened to be sampled alongside it (Ch. 7). GroupNorm is per-sample and therefore batch-independent.

> **Where this came from.** The U-Net was published in 2015 by **Olaf Ronneberger, Philipp Fischer and Thomas Brox** at the University of Freiburg, for **biomedical** image segmentation — tracking cells in electron microscopy images. The striking detail is the data scale: they won the ISBI cell-tracking challenge training on a few dozen annotated images, leaning on elastic deformation augmentation to squeeze generalization out of almost nothing. The architecture was designed under the constraint that annotated biological images are agonizingly expensive to produce. Less than a decade later, essentially the same U-shape with attention bolted on was generating photorealistic images from text prompts in Stable Diffusion — an architecture built for thirty microscopy slides scaled to billions of images without changing its skeleton.

---

## 8.5 Detection and segmentation, in brief

**Two-stage (R-CNN → Fast → Faster):** a Region Proposal Network emits candidate boxes; RoIAlign crops features; heads classify and regress. Higher accuracy, slower.

**One-stage (YOLO, SSD, RetinaNet):** dense prediction over anchors in one pass. **Focal loss** solved the extreme foreground/background imbalance:
▸ $$\mathrm{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log p_t,\qquad \gamma=2$$
The $`(1-p_t)^\gamma`$ factor down-weights easy examples by up to $100\times$, letting the rare hard positives dominate the gradient. **This is the canonical answer to "how do you handle class imbalance in dense prediction."**

**DETR:** treats detection as set prediction with a transformer; uses **Hungarian matching** for a bipartite assignment between predictions and ground truth, removing NMS and anchors entirely.

**Segmentation:** FCN → U-Net → DeepLab (atrous spatial pyramid pooling) → Mask R-CNN (instance) → SAM (promptable, foundation-model style).

**IoU and NMS.** $\mathrm{IoU} = \frac{|A\cap B|}{|A\cup B|}$. Non-max suppression greedily keeps the highest-scoring box and deletes overlapping ones above an IoU threshold. Losses: GIoU/DIoU/CIoU fix the zero-gradient problem when boxes don't overlap at all.

#### Focal loss, decoded

$$\mathrm{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log p_t,\qquad \gamma=2$$

Every symbol:

- **$`p_t`$** — the probability the model assigned to the **correct** answer for this example. If the true label is "background" and the model says 99% background, then $`p_t = 0.99`$. The subscript $t$ stands for "true," not for time.
- **$`-\log p_t`$** — this is just ordinary cross-entropy (Ch. 1). Read it as "your surprise at the truth."
- **$`(1-p_t)^\gamma`$** — the **modulating factor**, the entire contribution of this paper. It is near zero when the model is already confident and correct, and near one when the model is badly wrong.
- **$\gamma$** ("gamma", the *focusing parameter*, $\gamma=2$ in practice) — how aggressively to discount easy examples. $\gamma=0$ recovers plain cross-entropy exactly.
- **$`\alpha_t`$** — a fixed per-class weight (typically 0.25 for the foreground class) handling raw class *frequency*, a separate job from difficulty.

**The numbers are the argument.** With $\gamma = 2$:

| $`p_t`$ | Plain CE $`=-\log p_t`$ | $`(1-p_t)^2`$ | Focal loss | Down-weighted by |
|---|---|---|---|---|
| 0.99 (easy, confident) | 0.0101 | 0.0001 | $1.0\times10^{-6}$ | $10{,}000\times$ |
| 0.9 (easy) | 0.105 | 0.01 | 0.00105 | $100\times$ |
| 0.5 (uncertain) | 0.693 | 0.25 | 0.173 | $4\times$ |
| 0.1 (hard, wrong) | 2.303 | 0.81 | 1.865 | $1.2\times$ |

**Now why any of this matters.** A one-stage detector scores something like 100,000 candidate boxes per image, of which perhaps 10 contain an object. Under plain cross-entropy:

- 99,990 easy background boxes $\times\ 0.01$ each $\approx\ \mathbf{1{,}000}$ of total loss.
- 10 hard foreground boxes $\times\ 2.3$ each $\approx\ \mathbf{23}$ of total loss.

▸ **The background outvotes the foreground roughly 40 to 1, so gradient descent learns the only strategy that satisfies that majority: predict "background" everywhere.** This is not the model failing to learn; it is the model correctly optimizing the objective you wrote.

Apply focal loss with $\gamma=2$ and the same tally becomes $99{,}990\times 10^{-6}\approx 0.1$ against $10\times 1.87 \approx 18.7$. The ratio flips from 40:1 against the objects to nearly 200:1 in favour of them.

> **Analogy.** A teacher with thirty students and one hour. Giving everyone equal time means most of the hour is spent with students who already understand the material. Focal loss is the rule "spend time in proportion to how much each student is struggling" — and crucially, it **re-measures who is struggling every single week**, using the model's own current confidence as the measure of difficulty. It is not a fixed class weighting; it is an adaptive one.

▸ **The general principle, well beyond detection: when your loss is dominated by examples you already get right, you are spending nearly all your gradient on confirming what you know.** Hard-negative mining, curriculum learning, and importance sampling are all attacking this same problem with different machinery.

#### IoU and non-max suppression, decoded

$$\mathrm{IoU} = \frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$$

- $A$ and $B$ are two boxes, treated as **sets of pixels**.
- $\lvert\cdot\rvert$ is **cardinality** — how many pixels, i.e. the area. (Not absolute value, and not "given". Ch. 0 §0.9 covers the vertical-bar collisions.)
- $\cap$ is intersection: the overlapping region. $\cup$ is union: everything either box covers.
- The ratio runs from **0** (no overlap at all) to **1** (identical boxes).

**Work one.** Two $10\times10$ boxes, offset by 5 pixels horizontally and aligned vertically:

- Intersection $= 5\times 10 = 50$.
- Union $= 100 + 100 - 50 = 150$ (subtract the overlap once, or you'd double-count it).
- $\mathrm{IoU} = 50/150 = 0.333$.

The conventional threshold for "this counts as a correct detection" is 0.5, which is a **much looser standard than it sounds** — two boxes at IoU 0.5 can be visibly misaligned.

**The zero-gradient problem, and why GIoU exists.** If two boxes do not overlap *at all*, the IoU is 0 — and it is *still* 0 whether the prediction is 1 pixel away or 1,000 pixels away. The loss surface is perfectly flat in that whole regime, so the gradient is zero and there is nothing to descend. **GIoU (Generalized IoU)** repairs this by subtracting a penalty based on the smallest box that encloses both: as the boxes separate, that enclosing box grows, so the penalty keeps increasing and the gradient keeps pointing home.

**Non-maximum suppression, as a procedure.** Sort every predicted box by confidence. Take the highest. Delete every remaining box whose IoU with it exceeds a threshold. Repeat with the next survivor.

> **Analogy.** Twenty people in a lecture hall raise their hand about the same point. Call on the most insistent, then tell everyone sitting near them to put their hands down, then move to the next raised hand elsewhere in the room. It is greedy, it is not differentiable, and it has a well-known failure mode: two  different objects that overlap heavily (one person standing in front of another) — one of them gets suppressed. **DETR's contribution was making this step unnecessary**: by forcing a one-to-one assignment between predictions and ground-truth objects via Hungarian matching, the model has no incentive to emit duplicates in the first place, so there is nothing to suppress.

---

## 8.6 Interview-grade facts

- **Two 3×3 convs vs one 5×5:** same receptive field; $2\cdot9C^2=18C^2$ vs $25C^2$ params; two nonlinearities instead of one. This is the entire VGG argument.
- **1×1 convolutions** do three jobs: channel dimensionality reduction, cross-channel mixing, and adding nonlinearity cheaply.
- **Why do CNNs need less data than MLPs?** Weight sharing and locality are a hard prior encoding translation equivariance — it's the same argument as a smaller hypothesis class in Ch. 2, but the constraint is *correct* for images, so approximation error barely rises while estimation error falls sharply.
- **Why did attention beat convolution in vision at scale?** Not expressivity — a ViT with enough data learns convolution-like early layers. The gap is that conv's prior *helps* at 1M images and *constrains* at 300M. Inductive bias is a data-efficiency/ceiling trade.
- **Checkerboard artifacts** come from transposed convolution when `stride` doesn't divide `kernel_size`, producing uneven overlap. Fix: nearest/bilinear upsample followed by a normal conv.

#### Unpacking the interview facts

These are compressed to the point of being telegrams. Here is each one at conversational speed.

**"Two $3\times3$ convs versus one $5\times5$."** Stack two $3\times3$ layers and the second one's window covers a $5\times5$ patch of the original — each layer adds a reach of 2, so $1+2+2=5$. Parameters: $2\times 9C^2 = 18C^2$ against $25C^2$, a 28% saving. And you get **two** nonlinearities where the $5\times5$ gives you one, so the two-layer version is strictly more expressive as well as cheaper. Both things improve at once, which is rare and is why the argument was decisive.

**"$1\times1$ convolutions do three jobs."** They cannot mix across space (the window is one pixel), so everything they do is across channels. (1) **Dimensionality reduction** — drop 256 channels to 64 before an expensive $3\times3$, which is the bottleneck block. (2) **Cross-channel mixing** — recombine what different filters found at the same location. (3) **Cheap nonlinearity** — an activation after a $1\times1$ adds depth for $C^2$ parameters instead of $9C^2$.

**"Why do CNNs need less data than MLPs?"** Restate it in Chapter 2's vocabulary. Total error decomposes into **approximation error** (the best function your architecture can represent is still not the truth) plus **estimation error** (you can't find that best function from finite data). Constraining the hypothesis class always *raises* approximation error and *lowers* estimation error. The convolution's constraints — locality and weight sharing — happen to be nearly true of natural images, so approximation error barely moves while estimation error falls off a cliff. ▸ **A prior is only free when it is right; the identical constraint applied to tabular data would be a disaster.**

**"Why did attention beat convolution at scale?"** Not expressivity: a vision transformer trained on enough data learns convolution-like local attention patterns in its early layers all by itself. The trade is that a *correct* prior is a substitute for data, and therefore also a **ceiling** once you have plenty of data. At one million images the convolution's assumptions are worth more than the model could learn on its own; at 300 million they are worth less, and now they are only a restriction. ▸ **Inductive bias trades ceiling for sample efficiency, and which side of that trade you want depends entirely on how much data you have.**

**"Checkerboard artifacts."** A transposed convolution scatters each input value into a $k\times k$ output patch. If the stride does not divide the kernel size, adjacent output pixels receive different *numbers* of overlapping contributions — say 2, then 1, then 2, then 1 — so the output picks up a regular grid of brighter and darker pixels. It is a counting problem, not a learning problem, and no amount of training removes it cleanly. Upsample-then-convolve avoids it because the upsampling step is uniform by construction.

---

## Did you know?

- **Convolutional networks began with a cat and a slide projector.** Hubel and Wiesel found orientation-selective neurons in the cat visual cortex partly by accident: their cells stayed silent for the dot patterns they were projecting, and then one fired as the **edge of the glass slide** swept across the screen while they were changing it. The mechanism they went on to characterize — simple cells detecting oriented edges, complex cells pooling them over a small region — is the architecture of the first two layers of every CNN in this chapter. They shared the 1981 Nobel Prize.

- **"Atrous" is just French for "with holes."** Dilated convolution is called *à trous* in the literature, borrowed from the *algorithme à trous* in wavelet analysis. Two names for the same operation exist purely because the idea arrived in vision from signal processing.

- **AlexNet's grouped convolutions were a memory hack, not an insight.** The model did not fit in the 3 GB on a single GTX 580, so it was split across two cards that only exchanged information at certain layers. Fifteen years later, grouped convolution is a deliberate design axis — ResNeXt treats the group count as a third scaling dimension, and depthwise convolution (the entire MobileNet family) is the extreme case where every channel is its own group.

- **The layer everyone calls a "deconvolution" is not one.** It does not invert a convolution and never did; it applies the transpose of the convolution's matrix. The accurate name is *transposed convolution*. The misnomer entered the field through a 2010 paper title and has resisted every attempt to dislodge it since.

- **GoogLeNet's name is two jokes at once.** The capitalization is a deliberate tribute to LeNet, and the "Inception" module is named after the film — the paper explicitly cites the internet meme "we need to go deeper." It also beat VGG-16 by three points of top-1 accuracy with **twenty times fewer parameters**.

- **The ResNet paper is one of the most-cited scientific papers ever written.** "Deep Residual Learning for Image Recognition" is routinely reported as the single most-cited paper of the 21st century across all disciplines, with citation counts in the hundreds of thousands — ahead of most Nobel-winning work in physics and chemistry.

- **The original ResNet paper trained a 1,202-layer network** on CIFAR-10. It optimized without difficulty and merely overfit. That was the point: the authors were demonstrating that depth had stopped being an *optimization* problem, so the only remaining barrier was statistical.

- **Global average pooling deleted about 97% of a classic CNN's parameters.** AlexNet's three fully-connected head layers hold roughly 58 million of its 60 million weights. Replacing them with an average over the spatial map costs **zero** parameters and generalizes better.

- **Both the $1\times1$ convolution and global average pooling come from one short 2013 paper**, Lin, Chen and Yan's "Network In Network" — two ideas now present in essentially every vision architecture ever since, introduced together in a paper whose own headline proposal is largely forgotten.

- **The U-Net was trained on a few dozen images.** Ronneberger and colleagues won the 2015 ISBI cell-tracking challenge with a training set of a handful of annotated microscopy slides, relying on elastic deformation augmentation. Under a decade later, the same U-shape was the backbone of Stable Diffusion, generating photorealistic images from text.

- **IoU is a 120-year-old botany statistic.** It is the **Jaccard index**, published by the Swiss botanist Paul Jaccard around 1901 to compare which plant species occurred in different alpine regions. Every object detector in production evaluates itself with a measure invented for comparing flower meadows.

- **The Hungarian algorithm inside DETR is named after two mathematicians who never saw it.** Harold Kuhn called his 1955 method "Hungarian" in tribute to Dénes Kőnig and Jenő Egerváry, whose earlier results it builds on. In the 2000s it emerged that Carl Gustav Jacobi had solved essentially the same assignment problem before 1851; his work was published posthumously, in Latin, and went unnoticed for over 150 years.

- **SENet won the last ImageNet competition.** ILSVRC ran from 2010 to 2017, and Squeeze-and-Excitation networks took the final year with **channel attention** — attention arriving in vision four years before the vision transformer, in a form nobody at the time called attention.

- **ConvNeXt is a controlled experiment dressed as an architecture.** Its contribution was changing one thing at a time in a ResNet-50 — recipe, stem, kernel size, normalization, activation count — and reporting the accuracy after each single change, until it matched a Swin transformer. The finding was that most of the "attention wins" gap had been the training recipe all along.

---

## Check for Understanding

**Convolution is a dense layer restricted by locality and weight sharing — a correct prior for images that buys eight orders of magnitude in parameters — and residual connections fix the resulting depth problem by making every block compute a correction to an identity, so gradients add rather than multiply.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What are the two assumptions a convolution makes, and roughly how many parameters does each one save?** (One answer is about looking at 9 pixels instead of 50,000; the other is about reusing the same 9 numbers everywhere.)
2. **What is the difference between equivariance and invariance, and why do you want one in the middle of a network and the other at the end?**
3. **Why is your effective receptive field much smaller than the formula says?** Explain it in terms of the number of paths from a pixel, not in terms of variance.
4. **Why was a 56-layer network worse than a 20-layer one on the *training* set, and why does that rule out overfitting as the explanation?**
5. **Why is "do nothing" easy for a residual block and hard for a plain one?** Answer in terms of where the weights start, not in terms of gradients.
6. **In the gradient-highway equation, what is the leading $1$, and what happens if you put a BatchNorm on the skip path?**
7. **Why does a bottleneck block let you make a network four times deeper for the same compute?**
8. **What does zero-initializing the last BatchNorm scale in every block actually do to the network at step 0?**
9. **Why does a U-Net need skip connections at all — what specifically goes wrong without them?**
10. **Why does a detector trained with plain cross-entropy learn to predict "background" everywhere, and what does the $`(1-p_t)^\gamma`$ factor change about that?**
11. **Why is the gradient of an IoU loss zero when two boxes don't overlap, and why is that a problem rather than a curiosity?**
12. **Why did attention overtake convolution in vision at large scale, given that it isn't more expressive?**

If any of these produce a formula rather than a sentence, re-read that section.

---

**Next:** [Chapter 09 — Sequence Models: RNNs, LSTMs, Seq2Seq](09-sequence-models-rnn-lstm.md)
