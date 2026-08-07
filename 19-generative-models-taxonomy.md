# Chapter 19 — Generative Models: The Taxonomy

> **Prerequisites:** Ch. 1 (§1.4 ELBO, KL), Ch. 6.

> **New to the notation?** If symbols like $\in$, $\prod$, $\mathbb{E}$, $\nabla$, $\odot$, or $\sim$ are unfamiliar — or if you have ever wondered why $\sigma$ seems to mean four different things — read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $p_\theta(x)$ | "p-theta of x" | The **model's** guess at how likely data point $x$ is |
| $p_{\text{data}}$ | "p-data" | The **true** distribution the world draws from. Never known directly |
| $z$ | "z" | A **latent** variable — a hidden, compact code describing $x$ |
| $p(z)$ | "the prior" | The distribution latents are *supposed* to follow, usually $\mathcal{N}(0,I)$ |
| $q_\phi(z\mid x)$ | "q-phi of z given x" | The **encoder**: given data, guess its code |
| $p_\theta(x\mid z)$ | "p-theta of x given z" | The **decoder**: given a code, rebuild the data |
| $x_{<i}$ | "x less-than-i" | **Everything before position $i$** — all previous tokens |
| $\mathcal{N}(0,I)$ | "standard normal" | Bell curve, centred at zero, spread 1, no correlations |
| $D$ | "D" | The **discriminator** in a GAN — a learned fake-detector |
| $G$ | "G" | The **generator** in a GAN — makes the fakes |
| $\mathrm{JSD}$ | "Jensen–Shannon divergence" | A *symmetric* cousin of KL divergence |
| $W(p,q)$ | "Wasserstein distance" | "Earth mover's distance" — cost to reshape one pile into another |
| $\left\lvert\det \frac{\partial f}{\partial x}\right\rvert$ | "absolute determinant of the Jacobian" | **The volume-change factor** of a transformation |
| $E_\theta(x)$ | "energy of x" | A learned score where **low = likely**. Backwards from probability |
| $\beta$ | "beta" | A knob weighting the KL term against reconstruction |

### Abbreviations used in this chapter

Full glossary in [Chapter 0 §0.13](00-notation-and-math-primer.md).

| Short | Full form |
|---|---|
| AR | Autoregressive |
| EBM | Energy-Based Model |
| ELBO | Evidence Lower BOund |
| FID | Fréchet Inception Distance |
| GAN | Generative Adversarial Network |
| IS | Inception Score |
| JSD | Jensen–Shannon Divergence |
| KL | Kullback–Leibler (divergence) |
| MSE | Mean Squared Error |
| VAE | Variational AutoEncoder |
| VQ-VAE | Vector-Quantized Variational AutoEncoder |
| WGAN | Wasserstein Generative Adversarial Network |

---

## 19.1 The problem and the trilemma

### The one-line idea

Learn a distribution $p_\theta(x)$ that matches $p_{\text{data}}$ well enough that you can sample new data from it — which is hard because $x$ is high-dimensional and the true density concentrates on a thin, curved manifold.

### The analogy

You've been shown ten thousand photographs of faces and asked to produce a new one. You cannot store them all; you must infer the *rules* — two eyes, symmetric, lighting consistent, skin tone continuous — and then generate something new obeying them. The rules live in a space of a few hundred dimensions; the pixels live in a space of a million. Finding that low-dimensional structure is the entire game.

### The generative trilemma

▸ Every generative model trades off three properties, and until recently none had all three:

```
            High sample quality
                   /\
                  /  \
                 /    \
       GANs ----/------\---- Diffusion
               /        \
              /          \
   Fast sampling ------- Mode coverage / likelihood
                  VAEs, Flows, AR
```

- **GANs:** fast, sharp, poor coverage, no likelihood.
- **VAEs:** fast, good coverage, blurry.
- **Flows:** exact likelihood, fast, architecturally constrained.
- **Autoregressive:** exact likelihood, excellent quality, slow ($O(d)$ sequential steps).
- **Diffusion:** excellent quality and coverage, slow (though distillation has largely fixed this — Ch. 20).

### Two ways to specify a distribution

- **Explicit density:** write down $p_\theta(x)$. Tractable (AR, flows) or approximate (VAE, EBM).
- **Implicit:** define only a sampling procedure (GANs). No likelihood, hence no principled evaluation.

#### What "learn a distribution" actually means

This phrase does a lot of unexplained work, so let's be concrete. A **distribution** $p_\theta(x)$ is a function that takes a candidate image (or sentence, or molecule) and returns a number: *how plausible is this?*

A generative model must be able to do one or both of:

1. **Score** — given an $x$, say how likely it is. ("Is this a plausible face?")
2. **Sample** — produce a fresh $x$ out of thin air that looks like it came from the real distribution.

▸ **These are  different abilities, and that is the hidden theme of the entire chapter.** GANs can sample beautifully but cannot score at all. Flows can do both exactly but are architecturally hobbled. Understanding which of the two a model gives you explains nearly every design decision that follows.

> **Analogy.** Scoring is being a **restaurant critic** — taste a dish, rate it. Sampling is being a **chef** — produce a new dish. These are different skills. A critic who cannot cook is a GAN discriminator; a chef who cannot taste is a GAN generator. Training a GAN is putting the two in a room and making them compete.

**"The true density concentrates on a thin, curved manifold," decoded.** A **manifold** is a surface that is lower-dimensional than the space containing it — a sheet of paper (2-D) crumpled inside a room (3-D).

Make it concrete: a $256\times256$ colour image has $256\times256\times3 = 196{,}608$ numbers. So the space of *all possible* such images is 196,608-dimensional. But almost every point in it is **pure static**. Real faces occupy a vanishingly thin sliver of that space.

▸ **This is why generative modelling is hard and why it is possible at all.** Hard, because you must find an unimaginably thin surface in a vast space. Possible, because that surface has maybe a few hundred real degrees of freedom (pose, lighting, identity, expression), not 196,608. **Every model in this chapter is a different strategy for finding that sliver.**

> **Common misconception.** *"A generative model memorizes the training data and recombines it."* If it did, it would be useless — it must produce images that were never in the training set. What it learns are the *constraints* the data obeys (faces are symmetric, lighting is consistent, skin tone is continuous). Memorization does happen and is a  problem, but it is a **failure mode**, not the mechanism.

#### Examples and non-examples: is it a generative model?

**✅  generative models**

| Example | What it models | Can it sample? |
|---|---|---|
| A language model predicting next tokens | $p(\text{text})$ | Yes — sample tokens one at a time |
| A diffusion image model | $p(\text{image})$ | Yes — denoise from pure noise |
| A Gaussian fitted to a dataset | $p(x)$, crudely | Yes — it's a distribution |

**❌ Near-misses — commonly called generative, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A classifier outputting $p(y\mid x)$ | Models the **label** given the image, not the image | **Discriminative.** It can't produce an image. |
| A plain autoencoder | Reconstructs inputs; its latent space has holes | A **compressor.** Sample a random code and you get garbage. |
| An image upscaler | Needs an input image to start | A conditional **transformation** |
| An embedding model | Maps data to vectors, no way back | A **representation** |
| A GAN's likelihood score | GANs have no likelihood | Nonexistent — this is why FID exists |

▸ **The boundary:** a generative model must be able to produce a *new* sample from nothing but noise. If it requires a real input to operate on, it's transforming, not generating. **The plain-autoencoder case is the instructive one** — it has an encoder and a decoder and looks exactly like a VAE, but because nothing forces its codes to fill the latent space, decoding a random code produces noise. The KL term in a VAE exists precisely to fix this, which is the whole reason a VAE is generative and an autoencoder is not.

> **Where this came from.** The trilemma framing — quality, coverage, speed, pick two — was articulated in a 2021 paper by Zhisheng Xiao, Karsten Kreis, and Arash Vahdat that named it the "generative learning trilemma." It captured a frustration the field had felt for years: GANs made gorgeous images that silently omitted whole categories of the data; VAEs covered everything but produced blurs; autoregressive models were excellent and unbearably slow. Diffusion models, plus the distillation techniques of Chapter 20, are the closest anyone has come to escaping the trilemma — which is a large part of why they took over so completely after 2020.

---

## 19.2 Autoregressive models

Covered in Ch. 13. The essentials for comparison:

▸ $$p_\theta(x) = \prod_{i=1}^{d}p_\theta(x_i\mid x_{<i})$$

**Exact likelihood, stable maximum-likelihood training, best-in-class quality for discrete sequences.** The cost is $d$ sequential sampling steps and an imposed ordering (natural for text, arbitrary for images).

**PixelCNN/PixelRNN** applied this to images with masked convolutions; image AR models are now mostly used over *discrete latent tokens* (VQ codes) rather than raw pixels, which cuts $d$ by ~100×.

---

## 19.3 Variational Autoencoders

### The one-line idea

Learn an encoder that maps data to a distribution over a compact latent space, and a decoder that maps back — trained to maximize a tractable lower bound on the data likelihood.

### The analogy

A lossy compressor with a twist. A normal compressor maps an image to a specific code. A VAE maps it to a *fuzzy region* of code space, and the decoder must produce a good image from anywhere in that region. The fuzziness forces nearby codes to decode to similar images, which is what makes the space smooth enough to sample from.

### Derivation

Latent variable model: $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$ with prior $p(z)=\mathcal{N}(0,I)$.

The integral is intractable, so use the ELBO (Ch. 1 §1.4.4) with an amortized approximate posterior $q_\phi(z\mid x)$:

▸ $$\log p_\theta(x)\ \ge\ \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularization}}$$

with the gap exactly $\mathrm{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x))$.

**Reading the two terms:** reconstruction wants $q$ to be a precise, informative code. The KL wants $q$ to be the prior — i.e. to carry no information at all. The equilibrium is a code that keeps only what's worth its bits. **The ELBO is a rate–distortion objective.**

### The reparameterization trick

We need $\nabla_\phi\mathbb{E}_{q_\phi(z|x)}[f(z)]$, but the distribution we're sampling from depends on $\phi$.

▸ Write $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$. Now
$$\nabla_\phi\mathbb{E}_{\epsilon}\big[f(\mu_\phi+\sigma_\phi\epsilon)\big] = \mathbb{E}_\epsilon\big[\nabla_\phi f(\mu_\phi+\sigma_\phi\epsilon)\big]$$
The expectation is over a *fixed* distribution, so the gradient passes inside. **Randomness moved off the computational path.**

▸ **Why this matters:** the alternative, the score-function/REINFORCE estimator $\nabla_\phi\mathbb{E}[f] = \mathbb{E}[f(z)\nabla_\phi\log q_\phi(z)]$, is unbiased but has **variance orders of magnitude higher**. Reparameterization is what made VAEs trainable, and the same trick reappears in diffusion (Ch. 20) and in continuous-action RL (Ch. 27, SAC).

### The closed-form KL

For $q=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))$ and $p=\mathcal{N}(0,I)$:

▸ $$\mathrm{KL} = \frac12\sum_{j=1}^{J}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

Note it is zero iff $\mu=0,\sigma=1$, and each term is non-negative. Networks output $\log\sigma^2$ for numerical stability.

#### The VAE, decoded piece by piece

**What the latent variable $z$ is.** A short list of numbers — typically 32 to 512 — that summarizes an image. If $x$ is 196,608 pixel values, $z$ might be 128 numbers meaning things like "face angle," "hair colour," "lighting direction." **$z$ is the *description*; $x$ is the *thing*.**

**The two networks, in plain terms:**

| Network | Symbol | Job | Analogy |
|---|---|---|---|
| Encoder | $q_\phi(z\mid x)$ | Look at an image, produce its description | A police sketch artist working backwards from a photo |
| Decoder | $p_\theta(x\mid z)$ | Read a description, draw the image | The sketch artist drawing from the description |

**Why the encoder outputs a *fuzzy region* instead of a point.** This is the single strangest design choice in the VAE and the one worth understanding properly. The encoder does not output a code; it outputs a **mean $\mu$ and a spread $\sigma$**, and the actual code is sampled from that little cloud.

▸ **Why deliberately add noise?** Because without it, the model assigns each training image its own isolated point and leaves the space between points meaningless. Forcing each image to decode correctly from *anywhere in a small cloud* makes nearby codes produce similar images — and **that continuity is exactly what lets you sample a random code later and get a real-looking face rather than static.** The noise is not a nuisance to tolerate; it is the mechanism.

> **Analogy.** Imagine assigning postcodes to houses. If every house gets a unique random number, the postcode tells you nothing about location. If instead each house gets a *fuzzy region* and neighbouring houses' regions overlap, then similar postcodes mean nearby houses — and you can invent a plausible new address by picking a number in between. The VAE's noise is what makes latent space a *map* rather than a *lookup table*.

**Reading the closed-form KL term by term.** For each latent dimension $j$, the penalty $\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1$:

- $\mu_j^2$ — punishes codes drifting away from centre.
- $\sigma_j^2 - \log\sigma_j^2 - 1$ — punishes spread that is either too small **or** too large. It is exactly zero at $\sigma_j = 1$ and positive everywhere else.

Sanity-check with numbers: at $\mu=0,\sigma=1$ the penalty is $0 + 1 - 0 - 1 = 0$. ✓ At $\sigma = 0.1$ (an overconfident, nearly point-like code): $0.01 - \log(0.01) - 1 = 0.01 + 4.6 - 1 = 3.61$. **Heavily penalized** — the model is forbidden from quietly collapsing its noise to zero and becoming a plain autoencoder.

▸ **This is the whole enforcement mechanism.** Left alone, reconstruction would drive $\sigma\to 0$, because a precise code reconstructs better. The KL term makes that expensive. The equilibrium — some noise, but not so much that information is destroyed — is what produces a smooth, samplable latent space.

**"The ELBO is a rate–distortion objective," unpacked.** From data compression: **rate** = how many bits your code spends; **distortion** = how badly the reconstruction differs from the original. Spend more bits, get better reconstruction. The KL term *is* the rate (bits used by the code), and the reconstruction term *is* the distortion. **A VAE is a lossy compressor that gets to choose its own compression level**, and $\beta$ in the $\beta$-VAE is literally the dial.

#### Examples and non-examples: posterior collapse

Posterior collapse is the VAE's signature failure and is widely misdiagnosed.

**✅  posterior collapse**

| Symptom | What it means |
|---|---|
| KL term $\to 0$ during training | The encoder is outputting the prior — the code carries no information |
| Changing $z$ doesn't change the output | The decoder is ignoring the latent entirely |
| Reconstruction is fine, but latent interpolation is meaningless | Model became an autoregressive decoder with a vestigial latent |

**❌ Near-misses — look like collapse, but aren't**

| Looks like it | Why it isn't | What's actually happening |
|---|---|---|
| KL is small but nonzero and stable | A small rate can be correct for simple data | Healthy — the model found an efficient code |
| Blurry samples | Blur comes from the Gaussian likelihood and forward KL | A **different** problem (see the three reasons above) |
| Some latent dimensions have KL $\approx 0$ | Unused dimensions in an over-provisioned latent | Normal and often benign — the model is pruning |
| Reconstruction loss plateaus | Could be capacity, learning rate, anything | Not diagnostic of collapse on its own |

▸ **The boundary:** collapse is specifically *the decoder ignoring $z$*. The definitive test is an **intervention**: hold $x$ fixed, resample $z$, and see whether the output changes. If it doesn't, the latent is dead. This test is worth more than any loss curve, because collapse is about a *causal* relationship, not about a number's magnitude.

> **Common misconception.** *"Posterior collapse means the model failed to train."* Reconstruction can look excellent during collapse — a strong autoregressive decoder models the data perfectly well by itself. **The model succeeded at the loss and failed at the goal**, which is why the loss curve won't tell you and the intervention test will. This is a good general lesson: when a model can win the objective through a degenerate route, it will.

> **Where this came from.** The VAE was introduced by **Diederik Kingma and Max Welling** in *Auto-Encoding Variational Bayes*, submitted in December 2013, and independently by **Danilo Rezende, Shakir Mohamed, and Daan Wierstra** in 2014. Neither invented variational inference or the ELBO — both date to the 1990s — and the reparameterization idea has older roots in simulation. The contribution was recognizing that these could be combined so that a *neural network* could be trained through a sampling step by ordinary backpropagation. Kingma was a PhD student at the time; the following year he co-authored Adam. **The name is slightly misleading:** a VAE is not really an autoencoder that happens to be variational, it is a latent-variable probabilistic model whose inference network happens to look like an encoder.

### Why VAE samples are blurry

Three contributing reasons, all worth knowing:

1. **The Gaussian likelihood.** $p_\theta(x\mid z)=\mathcal{N}(f_\theta(z),\sigma^2I)$ makes the reconstruction term an MSE, and **the MSE-optimal prediction under uncertainty is the mean** — an average of all plausible images, which is a blur.
2. **Forward KL / mode covering.** Maximum likelihood penalizes assigning low density where data exists, so the model spreads mass (Ch. 1 §1.4.1).
3. **The bound is loose.** Maximizing a lower bound doesn't maximize the likelihood, and the slack is $\mathrm{KL}(q\|p(z|x))$, which is large when the true posterior is complicated.

### Posterior collapse

▸ If the decoder is powerful enough to model $x$ without $z$ (e.g. an autoregressive decoder), the optimum is $q_\phi(z\mid x)=p(z)$, KL $=0$, and **the latent is ignored entirely.** The model becomes a plain autoregressive model with an unused latent.

**Fixes:** KL annealing (warm up the KL weight from 0), **free bits** (don't penalize KL below a floor $\lambda$ per dimension), weakening the decoder, or $\delta$-VAE (constrain $q$ to stay away from the prior).

### $\beta$-VAE and the rate–distortion view

$$\mathcal{L} = \mathbb{E}_q[\log p_\theta(x\mid z)] - \beta\,\mathrm{KL}(q_\phi\|p)$$

$\beta>1$ increases disentanglement (each latent dimension captures one factor) at the cost of reconstruction. The mechanism: a tighter information budget forces the model to use its channel efficiently, and axis-aligned factorized codes are efficient when the data has independent factors.

### VQ-VAE

Replace the continuous latent with a **discrete** one via nearest-neighbour lookup in a learned codebook $\{e_k\}$:

▸ $$z_q = e_k,\qquad k = \arg\min_j\|z_e(x)-e_j\|_2$$

$$\mathcal{L} = \underbrace{\log p(x\mid z_q)}_{\text{reconstruction}} + \underbrace{\|\mathrm{sg}[z_e]-e\|^2}_{\text{codebook}} + \underbrace{\beta\|z_e-\mathrm{sg}[e]\|^2}_{\text{commitment}}$$

where $\mathrm{sg}$ is stop-gradient. The argmin has no gradient, so the **straight-through estimator** (Ch. 17 §17.4) copies the decoder gradient directly to the encoder.

**Why this matters enormously:** it converts images/audio into **discrete token sequences**, which can then be modelled by any autoregressive transformer. This is the foundation of DALL·E 1, Parti, MusicLM, most neural audio codecs, and image tokenizers generally. **Codebook collapse** (most codes unused) is the main failure; fixes include EMA codebook updates, code resetting, and low-dimensional codes (as in FSQ, which drops the codebook entirely in favour of scalar quantization).

---

## 19.4 GANs

### The one-line idea

Train a generator by pitting it against a discriminator that tries to tell real from fake — so the loss is itself a learned, adaptive model of what "realistic" means.

### The analogy

A forger and a detective who improve together. The detective learns whatever tell currently exposes the forgeries; the forger learns to eliminate exactly that tell. Neither has a fixed target — which is why it works spectacularly and why it is unstable.

### The objective

▸ $$\min_G\max_D\ V(D,G) = \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

### The optimal discriminator — derive this

For fixed $G$ with induced distribution $p_g$, we maximize pointwise:

$$V = \int_x \big[p_{\text{data}}(x)\log D(x) + p_g(x)\log(1-D(x))\big]dx$$

The integrand $a\log y + b\log(1-y)$ is maximized at $y=\frac{a}{a+b}$ (set the derivative $\frac ay - \frac{b}{1-y}=0$):

▸ $$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$$

Substitute back:

$$V(D^*,G) = \mathbb{E}_{p_{\text{data}}}\!\left[\log\frac{p_{\text{data}}}{p_{\text{data}}+p_g}\right] + \mathbb{E}_{p_g}\!\left[\log\frac{p_g}{p_{\text{data}}+p_g}\right]$$

Insert $\log 2$ terms to form the mixture $m = \frac{p_{\text{data}}+p_g}{2}$:

$$= -2\log 2 + \mathrm{KL}\!\left(p_{\text{data}}\|m\right) + \mathrm{KL}\!\left(p_g\|m\right) = -2\log2 + 2\,\mathrm{JSD}(p_{\text{data}}\|p_g)$$

▸ $$\boxed{\ \min_G V(D^*,G) = -\log4 + 2\,\mathrm{JSD}(p_{\text{data}}\,\|\,p_g)\ }$$

Since JSD $\ge0$ with equality iff the distributions match, **the global optimum is $p_g = p_{\text{data}}$, with value $-\log 4$.** ∎

### Where the theory breaks

▸ When $p_{\text{data}}$ and $p_g$ have **disjoint support** — which is essentially always, since both are concentrated on low-dimensional manifolds in a high-dimensional space — JSD is constant at $\log 2$, so **its gradient is zero.** A perfect discriminator provides no learning signal. This is the fundamental instability, and it is a theorem, not bad luck.

**Practical fix 1 — non-saturating loss.** Instead of minimizing $\log(1-D(G(z)))$ (which saturates when $D$ is confident), maximize $\log D(G(z))$. Same fixed point, much stronger gradients early.

**Practical fix 2 — Wasserstein GAN.** Replace JSD with the Earth-Mover distance, which is finite and has useful gradients even for disjoint supports. By Kantorovich–Rubinstein duality:

▸ $$W(p,q) = \sup_{\|f\|_L\le1}\ \mathbb{E}_{x\sim p}[f(x)] - \mathbb{E}_{x\sim q}[f(x)]$$

so the "critic" $f$ must be 1-Lipschitz. Enforced by weight clipping (original, crude), **gradient penalty** $\lambda(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2$ on interpolates (WGAN-GP), or **spectral normalization** $W\leftarrow W/\sigma_{\max}(W)$ (cheapest and most widely used).

### Mode collapse

The generator maps many $z$ to a few outputs. **Why it happens:** the objective rewards fooling $D$, and there is no term rewarding coverage — producing one perfect sample forever is a valid local optimum of the generator's objective.

**Mitigations:** minibatch discrimination (let $D$ see batch statistics), unrolled GANs, WGAN, PacGAN, and simply using a diffusion model instead.

#### Reading the GAN objective, slowly

$$\min_G\max_D\ V(D,G) = \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

Decode it left to right:

- $D(x)$ — the discriminator's estimate that $x$ is **real**, a number in $(0,1)$.
- $G(z)$ — the generator turning random noise $z$ into a fake sample.
- $\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]$ — *"on real data, $D$ should output near 1."* ($\log 1 = 0$, the maximum; $\log$ of anything smaller is negative.)
- $\mathbb{E}_{z}[\log(1 - D(G(z)))]$ — *"on fakes, $D$ should output near 0."*
- $\max_D$ — the discriminator maximizes both, i.e. gets good at telling them apart.
- $\min_G$ — the generator **minimizes** the same quantity, i.e. makes $D$ fail.

▸ **The unusual part is $\min\max$ — two networks optimizing the *same* number in opposite directions.** Ordinary training descends toward a minimum. This descends in one variable while ascending in another, seeking a **saddle point**, not a minimum. That is why GAN training is unstable: a saddle point is a balance, and balances can be lost. Nothing guarantees the two players converge rather than circling each other forever.

> **Analogy.** Tennis, not golf. In golf you improve against a fixed course, and progress is monotone. In tennis your opponent adapts, so "getting better" is relative and the rally can continue indefinitely with neither winning. GAN training is a rally — and a common failure is that one player becomes so strong the other stops learning entirely.

**Why the optimal discriminator formula makes sense.** $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$ reads: *"of all the probability mass at this point, what fraction came from real data?"* Check the extremes: where only real data lives, $p_g = 0$, so $D^* = 1$. Where only fakes live, $D^* = 0$. Where they overlap equally, $D^* = 1/2$ — **the discriminator is reduced to guessing, which is exactly what a perfectly-trained generator achieves.**

**Why "disjoint support" kills training — the most important idea in this section.** *Support* means the region where a distribution has any mass. Two distributions have **disjoint support** when they never overlap.

Real images and generated images both live on thin manifolds in a huge space (§19.1). Two thin surfaces in a 196,608-dimensional space essentially never intersect — think of two random threads in a warehouse.

▸ When they don't overlap, the discriminator can separate them **perfectly**, and $\mathrm{JSD}$ is pinned at the constant $\log 2$. A constant has **zero gradient.** The generator is told "you're wrong" but receives no information about *which direction* would be less wrong. **This is a theorem, not an engineering bug** — and it explains why every practical GAN fix (non-saturating loss, Wasserstein distance, gradient penalty, spectral normalization) is aimed at manufacturing a usable gradient where the theory provides none.

> **Analogy for Wasserstein distance.** JSD asks "do these two piles of earth overlap?" — a yes/no question that gives no guidance when the answer is no. The **earth mover's distance** asks "how much work to shovel one pile into the other?" — which still gives a sensible, decreasing number as the piles get closer, even while they remain disjoint. **That's the entire motivation for WGAN**: replace a question that goes flat with one that keeps pointing downhill.

#### Examples and non-examples: mode collapse

**✅  mode collapse**

| Symptom | Diagnosis |
|---|---|
| Trained on 10 digit classes, model only ever emits 3s and 8s | Whole modes missing |
| Different random $z$ produce near-identical images | The generator ignores its input |
| High precision, low **recall** in the precision/recall metric | Fidelity fine, coverage broken |

**❌ Near-misses — look like mode collapse, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Samples look similar to each other | Could be a low-diversity *dataset* | Faithful modelling of narrow data |
| Truncation trick producing similar faces | A deliberate quality/diversity trade at sampling time | Working as designed |
| Blurry averaged outputs | Blur is the VAE/MSE failure, not the GAN one | Mode **averaging**, the opposite failure |
| Low FID but poor variety | FID conflates fidelity and coverage | The reason precision/recall exists |

▸ **The boundary:** mode collapse is *missing regions of the data distribution*, not *low variance in outputs*. **Mode collapse and mode averaging are opposite failures** and are constantly confused: a GAN drops modes (produces some things perfectly, others never); a VAE averages modes (produces a blurry blend of everything). This is the forward-versus-reverse KL asymmetry of Chapter 1 §1.4.1 showing up as two different visual pathologies.

> **Common misconception.** *"Mode collapse means the GAN is undertrained."* It is often a **stable optimum** of the generator's objective. Nothing in the loss rewards coverage — producing one flawless image forever  fools the discriminator. Training longer can entrench it rather than fix it. This is a recurring theme: *when the objective doesn't measure what you care about, more optimization makes things worse, not better.*

> **The story behind GANs.** **Ian Goodfellow** conceived the idea in 2014, reportedly during an argument in a Montreal bar about how to make a generative model, and — as the widely-repeated account goes — implemented a working version that same night. The paper was published in 2014 with Yoshua Bengio among the co-authors. Yann LeCun later called adversarial training "the most interesting idea in the last 10 years in machine learning." The bar-and-one-night story is Goodfellow's own retelling and has been repeated often enough to become folklore; the essential point that survives scrutiny is that the first implementation came together remarkably fast, because the idea requires no new mathematical machinery — only two ordinary networks pointed at each other.

### The lineage

DCGAN (conv architecture rules) → Progressive GAN (grow resolution) → **StyleGAN** (mapping network to a disentangled $\mathcal{W}$ space, AdaIN style injection, per-layer noise — still the state of the art for controllable face synthesis and latent-space editing) → BigGAN (scale + truncation trick) → StyleGAN2/3 (fix artifacts, alias-free).

▸ **Why GANs lost to diffusion**: not sample quality at the top end (StyleGAN faces remain superb) but **training stability, mode coverage, controllability, and scalability to open-domain text-conditional generation.** GANs survive where speed is paramount — one-step super-resolution, real-time synthesis, and as decoders in codec models.

---

## 19.5 Normalizing flows

### The one-line idea

Build an *invertible* network, so you can compute the exact likelihood by tracking how much the transformation stretches or compresses volume.

### The analogy

Kneading dough. A simple ball (the Gaussian) is stretched and folded into a complicated shape (the data). Because you recorded every fold, you can run it backwards exactly, and you know how much each region was stretched — which is exactly what the density calculation needs.

### The change of variables formula

For an invertible $f$ with $z = f(x)$:

▸ $$p_X(x) = p_Z(f(x))\left|\det\frac{\partial f}{\partial x}\right| \quad\Longrightarrow\quad \log p_X(x) = \log p_Z(f(x)) + \log\left|\det J_f(x)\right|$$

**Exact likelihood, exact inference, exact sampling.** The catch: you need $f$ invertible *and* $\det J$ cheap. A general $d\times d$ determinant is $O(d^3)$.

### Coupling layers (RealNVP) — the key construction

Split $x = (x_{1:k}, x_{k+1:d})$ and transform only the second half, conditioned on the first:

▸ $$y_{1:k}=x_{1:k},\qquad y_{k+1:d} = x_{k+1:d}\odot\exp\big(s(x_{1:k})\big) + t(x_{1:k})$$

The Jacobian is **triangular**, so
$$\log|\det J| = \sum_{j>k}s_j(x_{1:k})$$

▸ **$O(d)$ instead of $O(d^3)$, and $s,t$ can be arbitrarily complex networks** because they are never inverted — inversion only requires solving for $x_{k+1:d}$, which is elementwise. Alternate which half is transformed between layers.

**Glow** adds invertible $1\times1$ convolutions (a learned channel permutation with $\log|\det| = h\cdot w\cdot\log|\det W|$) and actnorm.

**Autoregressive flows:** MAF (fast density, slow sampling) and IAF (slow density, fast sampling) — the two are transposes of each other, and this asymmetry is the central design tension in flows.

**Continuous normalizing flows** define $\frac{dz}{dt}=f_\theta(z,t)$ and use the instantaneous change of variables $\frac{\partial\log p}{\partial t} = -\mathrm{tr}\left(\frac{\partial f}{\partial z}\right)$ — the trace, not the determinant. This is the bridge to **flow matching** (Ch. 20 §20.10), which is where flows became competitive again.

### Why classical flows underperformed

The invertibility constraint is expensive: no dimensionality reduction (the latent must be the same size as the data), and coupling layers are less expressive per parameter than an unconstrained network. Flows need many more layers for the same quality.

---

## 19.6 Energy-based models

▸ $$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)},\qquad Z(\theta)=\int\exp(-E_\theta(x))\,dx$$

**Maximum flexibility** — any function can be an energy. **The problem is $Z$**, which is intractable.

The gradient of the log-likelihood is:
▸ $$\nabla_\theta\log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{x'\sim p_\theta}\big[\nabla_\theta E_\theta(x')\big]$$

**Push energy down on the data, up on the model's own samples.** Beautiful, and it requires sampling from $p_\theta$, which requires MCMC.

**Contrastive divergence:** run a short MCMC chain (often Langevin dynamics, $x_{t+1}=x_t-\frac\epsilon2\nabla_xE(x_t)+\sqrt\epsilon\,\eta$) from the data instead of to convergence. Biased, but workable.

**Score matching** avoids $Z$ entirely by matching $\nabla_x\log p$ (the score), which is independent of $Z$ since $\nabla_x \log Z = 0$. ▸ **This is the direct ancestor of diffusion models** (Ch. 20 §20.6) and the reason that chapter's derivations look the way they do.

#### Flows and energy models, decoded

**The change of variables formula, in one idea.** Squeeze a distribution into a smaller region and it must get **taller** — the total probability is always 1, so compressing horizontally forces vertical growth.

> **Analogy.** A fixed amount of dough. Roll it into a narrow strip and it becomes thick; spread it wide and it becomes thin. Density is dough per unit area, and it must adjust exactly to compensate for how much you stretched.

The determinant term $\left\lvert\det\frac{\partial f}{\partial x}\right\rvert$ is precisely **the local volume-change factor** — how much the transformation stretches or squeezes space at that point (§1.1.2: the determinant is a volume scale factor). So the formula says: *"the density at $x$ is the density of its image under $f$, corrected for how much $f$ stretched the neighbourhood."*

▸ **Why the determinant is the whole problem.** A general $d\times d$ determinant costs $O(d^3)$. For a $256\times256$ image, $d \approx 200{,}000$, so $d^3 \approx 8\times10^{15}$ — per image, per step. **Every architectural constraint in the flow literature exists to make that determinant cheap**, and coupling layers are the winning trick: transform only half the variables using the other half, and the Jacobian becomes triangular. A triangular matrix's determinant is just the product of its diagonal — $O(d)$ instead of $O(d^3)$.

**Energy-based models, and why the minus sign.** $p_\theta(x) \propto \exp(-E_\theta(x))$ means **low energy = high probability**, borrowed directly from physics, where systems settle into low-energy states. A ball rolls to the bottom of a valley; the valley floor is where you're most likely to find it.

$Z(\theta)$, the **partition function**, is the normalizer that makes probabilities sum to 1 — and computing it requires integrating over *every possible image*, which is hopeless. That single intractable integral is why energy-based models, despite being the most flexible family here, are the hardest to train.

▸ **The gradient rule is worth reading as a sentence:** *"push energy **down** on real data, **up** on the samples my model currently produces."* The model is being told to prefer reality over its own imagination — and the awkward part is that it must sample from itself to know what to push up against, which is what drags MCMC into the picture.

> **Where this came from.** Energy-based models descend from **statistical mechanics** and specifically the Boltzmann distribution (the same 1868 formula behind softmax, Chapter 1 §1.3.4). The **Boltzmann machine** was introduced by Geoffrey Hinton and Terry Sejnowski in the 1980s, with the *restricted* Boltzmann machine and contrastive divergence following — and it was stacked RBMs that powered the 2006 "deep learning" revival, before backpropagation with better initialization made them unnecessary. Hinton shared the 2024 Nobel Prize in Physics with John Hopfield, an award that recognised precisely this physics-to-networks lineage. **Score matching** was introduced by Aapo Hyvärinen in 2005 as a way to dodge the partition function, and sat relatively quiet until Yang Song and Stefano Ermon connected it to iterative denoising in 2019 — the bridge that leads directly into Chapter 20.

---

## 19.7 Evaluation

### Inception Score
$$\mathrm{IS} = \exp\big(\mathbb{E}_x\,\mathrm{KL}(p(y\mid x)\,\|\,p(y))\big)$$
Rewards confident per-image classification and diverse marginal classes. ▸ **Badly flawed:** it never looks at the real data at all, so a model that memorizes one image per class scores perfectly.

### FID
▸ $$\mathrm{FID} = \|\mu_r-\mu_g\|^2 + \mathrm{Tr}\left(\Sigma_r+\Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)$$
The Fréchet (2-Wasserstein) distance between Gaussians fitted to Inception-V3 pool features.

**Flaws to be able to state:** assumes Gaussianity of features (false); depends on an ImageNet classifier's biases; **strongly biased by sample count** (FID at $n=10$k and $n=50$k are not comparable — always report $n$); insensitive to some artifacts, oversensitive to others; and it collapses fidelity and diversity into one number.

### Precision and Recall for generative models
Estimate the two manifolds via k-NN and measure: **precision** = fraction of generated samples inside the real manifold (fidelity), **recall** = fraction of real samples inside the generated manifold (coverage). ▸ This decomposition is what FID hides, and it is the right diagnostic for mode collapse.

**Others:** CLIP score (text–image alignment), KID (unbiased kernel variant), human evaluation (the only ground truth), and **negative log-likelihood in bits/dim** where available.

#### Why evaluating generative models is  hard

For a classifier, evaluation is easy: it was right or it wasn't. For a generative model there is **no correct answer** — the model produces an image that has never existed, so there's nothing to compare it against. You must instead compare *distributions*, using only finite samples from each.

Two things must be measured, and they trade off:

| Property | Question | Failure if absent |
|---|---|---|
| **Fidelity** | Do the samples look real? | Blurry or broken outputs |
| **Diversity** | Do they cover the full range of real data? | Mode collapse |

▸ **A model can max out either one trivially.** Memorize a single perfect training image → flawless fidelity, zero diversity. Output uniform noise → maximal diversity, zero fidelity. **Any single-number metric can be gamed by sacrificing the axis it under-weights**, which is exactly why FID's collapse of both into one number is its central weakness.

**Reading FID.** The two terms answer two questions about Inception features:

- $\|\mu_r - \mu_g\|^2$ — *"are the average features in the same place?"*
- $\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ — *"is the spread and correlation structure the same?"*

Lower is better; 0 means the two Gaussian summaries match exactly.

#### Examples and non-examples: valid FID comparisons

**✅ Legitimate comparisons**

| Comparison | Why it's valid |
|---|---|
| Two models, same 50k samples, same reference set, same Inception weights | Everything held constant but the model |
| One model across training checkpoints, fixed protocol | Consistent measurement |

**❌ Invalid comparisons that appear constantly in practice**

| Comparison | Why it's broken |
|---|---|
| FID at $n{=}10$k vs FID at $n{=}50$k | FID is **biased by sample count** — smaller $n$ inflates it |
| FID across different reference datasets | It's a distance *to a reference*; change the reference, change the meaning |
| FID from different codebases | Implementations differ in resizing, interpolation, Inception weights |
| Your FID vs a number quoted in a paper | Almost never the same protocol |
| FID on non-natural images (medical, molecules) | Inception features are trained on ImageNet and don't transfer |

▸ **The boundary:** FID is a **relative** measure under a fixed protocol, not an absolute score. "FID 12" means nothing on its own. **A reported FID without its sample count and reference set is uninterpretable**, and comparing across papers without matching protocols is the single most common error in generative-model evaluation.

> **Common misconception.** *"Lower FID means better images."* FID measures distance between *feature distributions*, so a model can lower FID by better matching statistics while producing images humans find worse — and it is insensitive to some artifacts while overreacting to others. This is why human evaluation remains the ground truth and why precision/recall exists: it splits the single number back into the two things you actually care about.

> **Where this came from.** The **Inception Score** was proposed in 2016 by Tim Salimans and colleagues at OpenAI; **FID** followed in 2017 from Martin Heusel and co-authors at Johannes Kepler University Linz, explicitly to fix IS's blindness to the real data. Both are named for the **Inception** network, itself named after the 2014 film *Inception* — the architecture's paper contains the "we need to go deeper" meme reference, and the joke has now propagated into the standard evaluation metric for an entire field. The **Fréchet** distance honours Maurice Fréchet, who developed the underlying metric-space mathematics in the early 20th century, decades before there was anything to measure with it.

---

## 19.8 The comparison table

| | Likelihood | Sampling | Quality | Coverage | Training |
|---|---|---|---|---|---|
| Autoregressive | exact | slow $O(d)$ | high | high | stable |
| VAE | lower bound | fast $O(1)$ | blurry | high | stable |
| GAN | none | fast $O(1)$ | very high | **poor** | unstable |
| Flow | exact | fast | medium | high | stable |
| EBM | unnormalized | slow (MCMC) | medium | high | hard |
| **Diffusion** | lower bound | slow → fast* | **very high** | **high** | **stable** |

*after distillation (Ch. 20 §20.9)

▸ **Why diffusion won:** it is the only family that is simultaneously stable to train, mode-covering, and capable of the highest sample quality. Its one weakness — slow sampling — turned out to be fixable by distillation, while GANs' instability and mode collapse did not turn out to be fixable.

#### Reading the comparison table as one argument

The table is really a single claim: **each family sacrifices something specific to make an intractable density tractable, and the value of a family is decided by whether its particular sacrifice can later be undone.**

| Family | What it sacrifices | Was it fixable? |
|---|---|---|
| Autoregressive | Speed — $d$ sequential steps | Partly (speculative decoding, Ch. 17) |
| VAE | Exactness — a bound, not the likelihood | No; the blur is structural |
| GAN | The likelihood entirely | No; instability and collapse proved stubborn |
| Flow | Architectural freedom | Partly (continuous flows, flow matching) |
| EBM | The normalizer $Z$ | Partly (score matching — which became diffusion) |
| **Diffusion** | Speed — many denoising steps | **Yes — distillation fixed it** |

▸ **That last row is the whole story of 2020–2024.** Diffusion did not win by being better on every axis at the start; in 2020 it was slower than everything. It won because **its weakness was an engineering problem and its rivals' weaknesses were mathematical ones.** Sampling speed is a cost you can attack with better solvers and distillation. Mode collapse is a property of an objective that never rewarded coverage — no amount of engineering repairs an objective that isn't measuring the thing you want.

> **Common misconception.** *"GANs are obsolete."* They remain the tool of choice where a single forward pass is mandatory — real-time synthesis, super-resolution, and the decoders inside neural audio codecs. StyleGAN's latent space is still exceptionally good for controllable editing. The correct statement is that GANs lost the *open-domain text-to-image* competition, not that adversarial training stopped working.

---

## Did you know?

- **The generative adversarial network was reportedly designed and implemented in a single night.** Ian Goodfellow has recounted conceiving the idea during a 2014 argument in a Montreal bar and having a working version running before morning. It required no new mathematics — only two ordinary networks pointed at each other, which is why it came together so fast.

- **The Inception network is named after the film.** The 2014 architecture paper references the "we need to go deeper" meme from Christopher Nolan's *Inception*. That joke is now embedded in the standard evaluation metric of an entire research field, since both the Inception Score and the Fréchet Inception Distance are named after it.

- **Both the VAE's key ingredients predate it by two decades.** Variational inference and the evidence lower bound come from the 1990s. The 2013 contribution was making them work with a neural network trained by ordinary backpropagation through a sampling step.

- **Energy-based models won a Nobel Prize in Physics.** Geoffrey Hinton and John Hopfield shared the 2024 prize for foundational work on neural networks that traces directly to statistical mechanics — the Boltzmann machine is a neural network built out of a thermodynamics equation.

- **Stacked restricted Boltzmann machines started the deep learning revival in 2006**, then were almost entirely abandoned once better initialization and activation functions made plain backpropagation work on deep networks. An idea can be historically decisive and technically superseded at the same time.

- **Score matching sat quietly for fourteen years.** Aapo Hyvärinen introduced it in 2005 to sidestep the partition function. It became one of the most important ideas in generative modelling only after Yang Song and Stefano Ermon connected it to iterative denoising in 2019 — the step that leads directly to modern diffusion models.

- **A $256\times256$ colour image lives in a space of 196,608 dimensions**, and essentially every point in that space is static. The set of images a human would call "a face" is an unimaginably thin sliver. Generative modelling is the problem of finding that sliver.

- **VQ-VAE turned images into text.** By quantizing latents into discrete codes, it let ordinary autoregressive transformers — built for language — model images and audio directly. DALL·E 1, Parti, MusicLM, and most neural audio codecs rest on this one move.

- **FID is biased by how many samples you use**, which means the number you compute at 10,000 samples is not comparable to one computed at 50,000. Papers that omit the sample count are reporting an uninterpretable number, and this happens often.

- **The Inception Score never looks at the real data at all.** It only examines the generated images through a classifier. A model that memorized exactly one training image per class would score perfectly.

- **"Mode collapse" and "mode averaging" are opposite failures**, constantly confused with each other. GANs drop modes entirely; VAEs blend them into a blur. They are the two faces of the forward-versus-reverse KL asymmetry from Chapter 1.

- **Normalizing flows can't compress.** Because invertibility requires the latent to have exactly as many dimensions as the data, a flow's "code" for a megapixel image is a megapixel-sized vector — no bottleneck, no dimensionality reduction, which is a large part of why they underperformed per parameter.

---

## Check for Understanding

**Every generative model is a different answer to "how do I make a high-dimensional density tractable" — autoregressive models factorize it, VAEs bound it, flows constrain the architecture so the Jacobian determinant is cheap, GANs abandon the density entirely in favour of a learned adversarial loss, and energy-based models keep the density but give up the normalizer — and diffusion won because it inherited the stability of maximum likelihood and the sample quality of the adversarial approach without the failure mode of either.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **What is the difference between *scoring* and *sampling*, and which can a GAN do?** (The restaurant critic versus the chef.)
2. **Why is the data said to live on a "thin manifold," and why does that make generation both hard and possible?**
3. **Why does a VAE's encoder output a fuzzy cloud instead of a single point?** What breaks if you let the noise go to zero?
4. **Why is a plain autoencoder not a generative model, given that it has an encoder and a decoder just like a VAE?**
5. **What is posterior collapse, and why won't the loss curve tell you it's happening?** (What test would?)
6. **Why is GAN training described as a saddle point rather than a minimum, and why does that make it unstable?**
7. **Why does disjoint support kill GAN training?** Why is the Wasserstein distance a fix? (The two piles of earth.)
8. **What is the difference between mode collapse and mode averaging, and which model family suffers which?**
9. **Why is the Jacobian determinant the central obstacle in normalizing flows, and how do coupling layers dodge it?**
10. **Why can't you just train an energy-based model directly?** What is $Z$ and why is it hopeless?
11. **Why is a reported FID number meaningless without its sample count and reference set?**
12. **Why did diffusion beat GANs**, given that GANs were faster and, at the top end, produced comparable images?

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 20 — Diffusion Models](20-diffusion-models.md)
