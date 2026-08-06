# Chapter 19 — Generative Models: The Taxonomy

> **Prerequisites:** Ch. 1 (§1.4 ELBO, KL), Ch. 6.

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

---

## Check for Understanding

**Every generative model is a different answer to "how do I make a high-dimensional density tractable" — autoregressive models factorize it, VAEs bound it, flows constrain the architecture so the Jacobian determinant is cheap, GANs abandon the density entirely in favour of a learned adversarial loss, and energy-based models keep the density but give up the normalizer — and diffusion won because it inherited the stability of maximum likelihood and the sample quality of the adversarial approach without the failure mode of either.**

---

**Next:** [Chapter 20 — Diffusion Models](20-diffusion-models.md)
