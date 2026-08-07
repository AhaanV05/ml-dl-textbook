# Chapter 20 — Diffusion Models

> **Prerequisites:** Ch. 1 (§1.3.3 Gaussian closure, §1.4.4 ELBO), Ch. 19.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, $\prod$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

This is the most notation-dense chapter in the book. That is not because the ideas are hard — the central idea fits in one sentence, *"train a network to remove a little noise, then run it a thousand times"* — but because writing that sentence precisely requires a lot of subscripts. **Every formula below is a sentence.** Where the text gives you the compressed form, the inserted `####` sections give you the sentence.

### Symbols introduced in this chapter

Skim this once; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $x_0$ | "x-zero" | A **clean** data point — a real image from the dataset |
| $x_t$ | "x-t" | The same image after $t$ rounds of noise have been added |
| $x_T$ | "x-capital-T" | The fully-destroyed endpoint — pure static |
| $t$ | "t" | The **noise level**, an integer from 1 to $T$ (usually $T=1000$). Not wall-clock time |
| $\beta_t$ | "beta-t" | How much fresh noise is injected at step $t$. Tiny, e.g. $0.0001$ to $0.02$ |
| $\alpha_t$ | "alpha-t" | $1-\beta_t$ — the fraction of signal that **survives** one step |
| $\bar\alpha_t$ | "alpha-bar-t" | $\prod_{s\le t}\alpha_s$ — the fraction of signal surviving **all $t$ steps** |
| $q(\cdot)$ | "q" | The **forward** (noising) process. Fixed by us; nothing is learned |
| $p_\theta(\cdot)$ | "p-theta" | The **reverse** (denoising) process. This is the neural network |
| $\epsilon$ | "epsilon" | The random noise that was actually added. The training target |
| $\epsilon_\theta(x_t,t)$ | "epsilon-theta" | The network's **guess** at that noise |
| $\mathcal{N}(x;\ \mu,\ \Sigma)$ | "normal, x, mu, Sigma" | Bell curve **for the variable $x$**, centred at $\mu$, spread $\Sigma$ |
| $\tilde\mu_t,\ \tilde\beta_t$ | "mu-tilde", "beta-tilde" | The mean and variance of the *exact* one-step-back distribution |
| $\mathrm{KL}(p\|q)$ | "KL of p from q" | How much you lose by believing $q$ when reality is $p$ |
| $s(x) = \nabla_x\log p(x)$ | "the score" | Which way to move $x$ to make it **more probable** |
| $\mathrm{SNR}(t)$ | "signal-to-noise ratio" | $\bar\alpha_t/(1-\bar\alpha_t)$ — how much signal is left relative to noise |
| $w$ | "the guidance scale" | How hard to push a sample toward the prompt |
| $\varnothing$ | "null" / "the empty condition" | The placeholder meaning "no prompt given" |
| $dw$, $d\bar w$ | "dee-w", "dee-w-bar" | An infinitesimal random kick (forward / backward in time) |
| $v$ | "vee" | The **velocity** parameterization, $\sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$ |

### Abbreviations used in this chapter, spelled out

| Short | Full form |
|---|---|
| DDPM | Denoising Diffusion Probabilistic Model |
| DDIM | Denoising Diffusion Implicit Model |
| NCSN | Noise Conditional Score Network |
| ELBO / VLB | Evidence Lower BOund / Variational Lower Bound |
| KL | Kullback–Leibler (divergence) |
| MSE | Mean Squared Error |
| SDE / ODE | Stochastic / Ordinary Differential Equation |
| VP / VE | Variance Preserving / Variance Exploding |
| SNR | Signal-to-Noise Ratio |
| CFG | Classifier-Free Guidance |
| EMA | Exponential Moving Average |
| VAE | Variational AutoEncoder |
| FID | Fréchet Inception Distance |
| LPIPS | Learned Perceptual Image Patch Similarity |
| LCM | Latent Consistency Model |
| LoRA | Low-Rank Adaptation |
| ADD | Adversarial Diffusion Distillation |
| DPM-Solver | Diffusion Probabilistic Model Solver |
| SD / SDXL / SD3 | Stable Diffusion (and its XL / version-3 variants) |
| RK4 | Runge–Kutta, 4th order (a classical ODE solver) |

---

## 20.1 The idea

### The one-line idea

Gradually destroy data by adding noise until it becomes pure Gaussian noise, then train a network to reverse one step of that destruction — and generate by running the reversal from noise.

### The analogy

A drop of ink diffusing in water. Running it forward is easy and requires no intelligence: physics does it. Running it *backward* — reassembling the drop — is the hard part. But here's the trick that makes it tractable: **reversing one tiny instant of diffusion is easy**, because over an infinitesimal step the ink has barely moved and you only need to guess a small correction. Chain a thousand easy corrections and you have reassembled the drop.

▸ **The deep reason diffusion works:** it decomposes an intractable generative problem into a thousand tractable denoising problems. Each is a simple supervised regression. The difficulty is amortized across steps rather than concentrated in one.

#### Why "decompose into a thousand easy problems" is the whole trick

Generative modelling asks an absurd question: *"produce a $512\times512\times3$ array of numbers — 786,432 of them — such that the result looks like a photograph."* Nothing in that array can be chosen independently; the pixel at position (100, 200) has to agree with the one at (101, 200) about which object they belong to. Asking a network to get all 786,432 numbers jointly right in one shot is asking it to solve the entire problem in a single forward pass, and it is why earlier generative models were so temperamental.

Diffusion's answer is to **refuse to solve the hard problem at all.**

> **Analogy — sculpture versus sanding.** Carving a statue from a block of marble in one motion is impossible. Sanding a nearly-finished statue so it looks very slightly more finished is easy — you barely have to know what the statue is *of*. Diffusion takes a block of noise and applies a thousand rounds of sanding. Each round is a task so mild that a network can be trained on it with ordinary supervised regression. The statue emerges because a thousand mild improvements compose into one enormous one.

Put numbers on the mildness. With $T = 1000$ steps and $\beta_t \approx 0.01$, each forward step replaces roughly **1%** of the signal with noise. So each reverse step only has to recover about 1% of the picture. **A network that is 99% right about a 1% correction produces a very good image after a thousand steps** — errors do not accumulate the way you would fear, because every step re-anchors on the actual current state $x_t$ rather than on its own previous guess.

▸ **This is the structural difference from an autoregressive model.** A language model (Ch. 13) also decomposes generation into many small decisions, but its decomposition is *spatial* — one token at a time, left to right, and every token is final the moment it is emitted. Diffusion's decomposition is by **level of detail**: every pixel is revised at every step, coarse structure first, fine texture last. Nothing is ever final until the last step. That is why diffusion can fix a mistake it made and an autoregressive sampler cannot.

**What "amortized" means here.** To amortize a cost is to spread it out — the word comes from finance, where a large debt is repaid in small scheduled instalments. The generative difficulty is the debt. Diffusion pays it in a thousand instalments at sampling time, which is exactly why sampling is slow (§20.5) and why half the field's engineering effort (§20.9) is spent negotiating a shorter repayment schedule.

> **Where this came from.** Diffusion models were introduced in 2015 by **Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli** at Stanford, in a paper titled *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. The framing was explicitly borrowed from statistical physics: in non-equilibrium thermodynamics there are results (the Jarzynski equality, and the annealed importance sampling methods built on them) about relating a complicated distribution to a simple one through a slow, gradual process — and the authors realized that "data distribution" and "Gaussian noise" could play those two roles. The paper worked. It generated small, blurry, obviously-correct samples. And then **it was largely ignored for five years**, while the field spent that time on generative adversarial networks. It took **Jonathan Ho, Ajay Jain, and Pieter Abbeel** at UC Berkeley in 2020 — with the paper *Denoising Diffusion Probabilistic Models* — to find the parameterization and the training recipe that made the same idea produce photorealistic images. The mathematics barely changed between 2015 and 2020. What changed was $\epsilon$-prediction, a better noise schedule, and a U-Net big enough to matter.

> **The story behind the word "diffusion."** It is Latin — *diffundere*, "to pour out" or "to spread." The physical phenomenon was described quantitatively by **Adolf Fick** in 1855 (Fick's laws), by analogy with Fourier's law of heat conduction. The random microscopic motion underlying it was the thing Robert Brown observed in 1827 watching pollen grains jitter in water, and that Einstein explained in 1905 as the visible consequence of invisible molecular collisions — one of the papers that finally convinced physicists that atoms were real. **Every piece of vocabulary in this chapter comes from 19th-century physics**, and it is not decoration: the forward process really is a discretized Brownian motion, and the reverse-time result in §20.7 really is a theorem about stochastic processes.

---

## 20.2 The forward process

Fix a variance schedule $\beta_1,\dots,\beta_T$ with $0<\beta_t\ll1$.

▸ $$q(x_t\mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$$

**Why $\sqrt{1-\beta_t}$ and not 1?** Variance preservation. If $\mathrm{Var}(x_{t-1})=1$ then $\mathrm{Var}(x_t) = (1-\beta_t)\cdot1+\beta_t = 1$. Without the shrinkage the variance would grow without bound. (The alternative, "variance exploding," is used by score-based SDE formulations.)

#### Reading the forward step in plain English

Take the formula one symbol at a time.

$$q(x_t\mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$$

| Piece | Read aloud | What it means |
|---|---|---|
| $q$ | "q" | The **forward** process. A fixed recipe we choose; it has **no parameters and is never trained** |
| $x_t \mid x_{t-1}$ | "x-t **given** x-t-minus-one" | "The distribution of the next image, once you know the current one" (Ch. 0 §0.9) |
| $\mathcal{N}(x_t;\ \mu,\ \Sigma)$ | "normal, for x-t, with mean mu and covariance Sigma" | A Gaussian. **The semicolon matters:** the thing before it is the *variable*, the things after are the *parameters* |
| $\sqrt{1-\beta_t}\,x_{t-1}$ | "root one-minus-beta-t times x-t-minus-one" | The mean: **the old image, faded slightly** |
| $\beta_t I$ | "beta-t times the identity" | The covariance: independent noise of variance $\beta_t$ **added to every pixel separately** |

▸ **In one sentence: "shrink the picture a little toward zero, then sprinkle a little independent noise on every pixel."** That is the entire forward process. It is three lines of code and there is nothing to learn.

**The identity matrix $I$ is doing real work.** Writing the covariance as $\beta_t I$ rather than some general matrix $\Sigma$ says the noise on pixel 1 is **statistically unrelated** to the noise on pixel 2. This is why the whole chapter stays computable: a general covariance over 786,432 pixels would be a matrix with $6\times10^{11}$ entries. Diagonal-and-equal means you can generate the noise with one call to a random number generator and never form a matrix at all.

#### Why the square root, worked with numbers

The shrink factor is $\sqrt{1-\beta_t}$, not $1-\beta_t$, and the reason is that **variances add, but standard deviations do not.** When you scale a random variable by a constant $c$, its variance scales by $c^2$ (Ch. 1 §1.3.1). So:

$$\mathrm{Var}(x_t) \;=\; \underbrace{\left(\sqrt{1-\beta_t}\right)^2}_{=\,1-\beta_t}\cdot\underbrace{\mathrm{Var}(x_{t-1})}_{=\,1} \;+\; \underbrace{\beta_t}_{\text{new noise}} \;=\; (1-\beta_t) + \beta_t \;=\; 1$$

The square root exists precisely so that the squaring cancels it. **Put a number in:** with $\beta_t = 0.01$, the shrink factor is $\sqrt{0.99} = 0.99499$, so we keep 99.5% of the amplitude and add noise with standard deviation $\sqrt{0.01} = 0.1$. Total variance after the step: $0.99 \times 1 + 0.01 = 1$. Exactly preserved. ✓

> **Analogy — the topped-up glass.** You have a full glass of orange juice. Each step you pour out 1% of the liquid and top the glass back up to the brim with water. The *level never changes* — that is variance preservation. What changes is the concentration: after enough rounds the glass is full of water with a memory of juice. **The clean image is the juice, the noise is the water, and $\bar\alpha_t$ (next section) is the concentration.**

▸ **Why anyone bothers to preserve variance.** Neural networks are sensitive to input scale — every initialization scheme, every normalization layer, and every learning rate in Chapters 6 and 7 assumes activations are order-1. If the forward process let variance grow, then $x_{500}$ and $x_{900}$ would differ by orders of magnitude in typical size, and one network could not serve both. Fixing $\mathrm{Var}(x_t)=1$ for every $t$ means **the network sees inputs of the same scale at every noise level**, and only has to condition on $t$ to know *how much* signal is hiding in them.

**The alternative mentioned in parentheses.** "Variance exploding" (VE) keeps the signal at full strength and lets the noise grow: $x_t = x_0 + \sigma_t\epsilon$ with $\sigma_t$ climbing to something enormous like 50. Nothing is wrong with it mathematically — §20.7 shows VP and VE are two coordinate systems for the same object — but the inputs then span five orders of magnitude, so implementations must rescale the network's input by hand. VP hides that bookkeeping inside the schedule.

### The closed form — the key computational fact

Define $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$.

▸ $$\boxed{\ q(x_t\mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)\ }$$

**Proof by induction.** Assume it holds at $t-1$: $x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon'$. Then
$$x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon'' = \sqrt{\alpha_t\bar\alpha_{t-1}}x_0 + \sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\epsilon' + \sqrt{1-\alpha_t}\,\epsilon''$$
The last two terms are independent zero-mean Gaussians, so they combine (Ch. 1 §1.3.3) into a single Gaussian with variance
$$\alpha_t(1-\bar\alpha_{t-1}) + 1-\alpha_t = 1-\alpha_t\bar\alpha_{t-1} = 1-\bar\alpha_t$$
Since $\alpha_t\bar\alpha_{t-1}=\bar\alpha_t$, we get $x_t = \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$. ∎

▸ **Why this is everything:** you can jump to any noise level in **one** operation. Training samples a random $t$, corrupts $x_0$ directly, and asks the network to denoise. Without this closed form you would have to simulate $t$ steps per training example and diffusion would be computationally hopeless.

As $t\to T$, $\bar\alpha_T\to0$ and $q(x_T)\to\mathcal{N}(0,I)$ — a tractable prior.

#### Unpacking $\bar\alpha_t$ — the one number that runs this chapter

Two definitions, both trivial once said aloud:

- $\alpha_t = 1 - \beta_t$ — **"the fraction of variance that survives one step."** If $\beta_t = 0.01$ then $\alpha_t = 0.99$: 99% survives.
- $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ — **"multiply all the survival rates together."** The bar means "cumulative"; $\prod$ is a `for` loop that multiplies instead of adds (Ch. 0 §0.3).

$$\bar\alpha_t \;=\; \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$$

▸ **$\bar\alpha_t$ is the answer to "how much of the original picture is left after $t$ steps?"** — and because it is a *product* of numbers slightly below 1, it decays exponentially. This is the $\lambda^k$ phenomenon from Ch. 1 §1.1.2 wearing a different hat.

**Now the closed form, read as a sentence:**

$$q(x_t\mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
\quad\Longleftrightarrow\quad
x_t = \underbrace{\sqrt{\bar\alpha_t}}_{\text{signal}}\,x_0 \;+\; \underbrace{\sqrt{1-\bar\alpha_t}}_{\text{noise}}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

*"The noisy image at level $t$ is a fixed blend of the clean image and one draw of pure noise, with blend weights that always square to one."* The right-hand form (the "reparameterization") is the one implemented in code, and it is the one to memorize.

**Check the weights.** $(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = \bar\alpha_t + 1 - \bar\alpha_t = 1$. The signal and noise coefficients live on a unit circle; as $t$ increases you rotate from "all signal" toward "all noise." Nothing is ever lost or gained in total variance — precisely the variance-preservation property, now visible in one line.

#### Real numbers: what a schedule actually looks like

Take the original DDPM linear schedule — $\beta_t$ rising linearly from $10^{-4}$ at $t=1$ to $0.02$ at $t=1000$:

| $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ (signal) | $\sqrt{1-\bar\alpha_t}$ (noise) | What the image looks like |
|---|---|---|---|---|---|
| 1 | $0.0001$ | $\approx 0.9999$ | $0.99995$ | $0.01$ | Indistinguishable from the original |
| 100 | $0.002$ | $\approx 0.91$ | $0.95$ | $0.30$ | Slightly grainy photo |
| 400 | $0.008$ | $\approx 0.21$ | $0.46$ | $0.89$ | Recognizable shapes in heavy static |
| 700 | $0.014$ | $\approx 0.007$ | $0.084$ | $0.996$ | Faint blobs of colour; content unguessable |
| 1000 | $0.02$ | $\approx 4\times10^{-5}$ | $0.006$ | $0.99998$ | Pure noise |

▸ **Read the fourth column as "what fraction of the original amplitude is still present."** At $t=700$ it is 8%. At $t=1000$ it is six parts in a thousand — below the quantization step of an 8-bit image, so the original is , irrecoverably gone. That is the point: $q(x_T)$ must be a distribution we can *sample from without knowing anything*, and $\mathcal{N}(0,I)$ is that distribution.

**Why it collapses so fast.** $\bar\alpha_{1000}$ is a product of a thousand numbers averaging about $0.99$. And $0.99^{1000} \approx 4.3\times10^{-5}$. **Compound decay is brutal**, which is why $\beta_t$ has to be so small: a schedule with $\beta_t = 0.1$ throughout would give $0.9^{1000} \approx 10^{-46}$ and would have destroyed the image by step 200, wasting 800 steps on pure noise.

#### Reading the induction proof, if you want to

You do not have to follow the algebra to use the result, but the proof is short and rests on exactly one fact, so here it is in words.

**The one fact (Ch. 1 §1.3.3, "Gaussian closure"):** if you add two *independent* Gaussians, you get a Gaussian, and **the variances add.** $\mathcal{N}(0,a) + \mathcal{N}(0,b) = \mathcal{N}(0, a+b)$. Not the standard deviations — the variances. This is the only non-obvious step in the derivation.

The proof then says: *suppose the formula is right at step $t-1$; apply one more forward step; you now have two separate noise terms — the accumulated one ($\epsilon'$) and the fresh one ($\epsilon''$) — and since they are independent you may merge them into a single noise draw whose variance is the sum.* Adding the two variances gives $\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t)$, and the $\alpha_t\bar\alpha_{t-1}$ terms telescope into $\bar\alpha_t$. The symbol ∎ at the end just means "proof over" (Ch. 0 §0.11).

▸ **The load-bearing sentence is "the last two terms are independent zero-mean Gaussians, so they combine."** If they were correlated — if the noise added at step $t$ knew anything about the noise added at step $t-1$ — the merge would be illegal and there would be no closed form. **Independence of the noise is what buys the whole chapter.**

#### Why the closed form is the difference between "possible" and "impossible"

Without it, generating a single training example at $t=700$ would require running 700 sequential Gaussian samples, and each is a dependency on the last, so nothing parallelizes.

Put the numbers together. A typical run sees $10^9$ training examples. If each required a random $t$ averaging 500 sequential noise steps, that is $5\times10^{11}$ sequential operations **of pure data preparation**, before a single gradient is computed. With the closed form it is **one** operation: draw $t$, draw $\epsilon$, compute $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$. The table of $\bar\alpha_t$ values is precomputed once, at startup, and is a thousand floats.

> **Analogy.** Compound interest. To find the balance after 30 years you do not simulate 360 monthly payments — you use the closed-form formula $P(1+r)^n$. $\bar\alpha_t$ is exactly that formula for noise instead of money, and it is available for exactly the same reason: each step multiplies by a constant, and products of constants collapse.

▸ **This is also the reason training is embarrassingly parallel.** Every example in a batch can be at a *different* noise level — one at $t=13$, the next at $t=871$ — with no coordination whatsoever. A diffusion training step looks, to the hardware, exactly like any other supervised regression on independent examples. There is no sequential structure to fight, which is why diffusion training scales as cleanly as it does.

#### Examples and non-examples: the forward process

**✅  forward-diffusion steps**

| Example | Why it qualifies |
|---|---|
| $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$ | Fixed, parameter-free, and it lands on exactly the marginal that $t$ sequential steps would have produced |
| Drawing $t=713$ for one image in a batch and $t=12$ for the next | The corruption levels are independent across examples — nothing couples them |
| Shrinking the signal by $\sqrt{\bar\alpha_t}$ **and** adding noise of size $\sqrt{1-\bar\alpha_t}$ | Both halves are required: the shrink is what holds total variance at 1 |
| A schedule that reaches $\bar\alpha_T \approx 0$ | The endpoint must be a distribution you can sample from knowing nothing |

**❌ Near-misses — look like the forward process, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| $x_t = x_0 + \sigma_t\epsilon$ — add noise, leave the image at full scale | Variance grows without bound, so $x_T$ is not $\mathcal{N}(0,I)$ and you cannot start sampling from a standard normal | The **variance-exploding** formulation (NCSN/SMLD), a legitimate but different process — §20.6 |
| Gaussian-noise data augmentation | One fixed noise level, and the training target is a label, not the noise | Regularization, not a corruption *process* |
| Dropout | Multiplicative and Bernoulli, not additive and Gaussian; no $t$, no schedule | Stochastic regularization (Ch. 7) |
| A variational autoencoder's encoder | It is **learned**, and it maps into a compressed space rather than to a noisier copy | Amortized inference (Ch. 19) |
| Progressively blurring the image | Deterministic and (in principle) information-preserving, so there is no noise to predict | "Cold diffusion" — needs an entirely different reverse rule |

▸ **The boundary:** the forward process is **fixed, parameter-free, Gaussian, variance-preserving, and indexed by $t$**. Nothing in it is learned, and its endpoint must be a distribution you can draw from cold.

> **Common misconception.** *"Diffusion adds noise and then removes it — the two directions are mirror images of each other."* They are not remotely symmetric. **The forward direction has zero parameters and a closed form; the reverse direction is a neural network and has no closed form at all.** Going forward, a photograph becomes static, and there is essentially one way for that to happen. Going backward, static becomes *a* photograph, and there are astronomically many valid answers — which is exactly why each reverse step must be **sampled** rather than computed. The belief is tempting because every paper draws the two directions as arrows of equal length, and because $q$ and $p_\theta$ are written in matching notation, which makes them look like inverse operations. **A better mental image: shredding a document takes a machine; reconstructing it takes someone who knows what documents say.**

---

## 20.3 The reverse process and the exact posterior

We want $q(x_{t-1}\mid x_t)$, which is intractable (it requires the marginal $q(x_t)$). But **conditioned on $x_0$ it is a tractable Gaussian.**

▸ $$q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\right)$$

### Derivation

By Bayes, $q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$. Take logs and keep terms in $x_{t-1}$:

$$-\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{2\beta_t} - \frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0)^2}{2(1-\bar\alpha_{t-1})}$$

Collect the quadratic coefficient:
$$\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}} = \frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})} = \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}$$

so the variance is its reciprocal:

▸ $$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$$

The linear coefficient gives the mean:

▸ $$\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t$$

**A weighted average of the clean data and the current noisy sample.** At large $t$, $\bar\alpha_{t-1}\approx\bar\alpha_t\approx0$ so the weights are $\approx(\sqrt{\bar\alpha_{t-1}}\beta_t, \sqrt{\alpha_t})$ — mostly $x_t$. At small $t$ the $x_0$ term dominates.

#### What "intractable" means here, and why conditioning on $x_0$ rescues it

The section opens with a sentence that decides the shape of the whole chapter: *"We want $q(x_{t-1}\mid x_t)$, which is intractable (it requires the marginal $q(x_t)$)."* Unpack it.

- $q(x_{t-1}\mid x_t)$ — **"given a noisy image, what did it look like one step less noisy?"** This is the thing we actually want at sampling time, when all we have is $x_t$.
- $q(x_t)$ — the **marginal**: "how likely is this particular noisy image, averaged over every possible clean image in the universe?" Formally $q(x_t) = \int q(x_t\mid x_0)\,q(x_0)\,dx_0$ — an integral over the entire data distribution.
- **Intractable** means: not that it's hard, but that computing it exactly would require knowing $q(x_0)$, the distribution of all real images. **That is the very thing we are trying to learn.** Requiring it would be circular.

> **Analogy.** Someone shows you a blurred photograph and asks "what did this look like one notch sharper?" To answer *properly* you would have to know every photograph that exists, weigh which of them could have blurred into this one, and average. Nobody can do that. **But if they also tell you the original photograph, the question becomes arithmetic** — you know both endpoints, so the intermediate step is pinned down.

▸ **That is the entire move.** During training we *do* have $x_0$ — it is the training example sitting right there. So we can compute $q(x_{t-1}\mid x_t, x_0)$ exactly, in closed form, and use it as the regression target. At sampling time we don't have $x_0$, but by then the network has learned to supply an estimate of it. **The intractable quantity is only needed during training, and during training it is not intractable.**

#### Reading the Bayes derivation without doing the algebra

The derivation starts from

$$q(x_{t-1}\mid x_t,x_0)\;\propto\;q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$$

which is **Bayes' rule** (Ch. 1 §1.3): *"the probability that the previous state was $x_{t-1}$ is proportional to (how likely that state was to produce what we see) × (how likely that state was in the first place)."* The symbol $\propto$ means "equal up to a constant we will fix later by making the probabilities sum to 1" (Ch. 0 §0.11) — dropping that constant is what makes the algebra tolerable.

Then: *"take logs and keep terms in $x_{t-1}$."* Two habits at work.

1. **Take logs** because both factors are Gaussians, and $\log$ of $e^{-\text{something}}$ is just $-\text{something}$. Multiplication becomes addition; exponentials vanish. This is the same reason every probability computation in this book happens in log space (Ch. 0 §0.3).
2. **Keep only terms in $x_{t-1}$** because everything else is part of that constant we agreed to ignore.

What is left is a sum of two negative squared terms in $x_{t-1}$ — and *any* expression of the form $-a x^2 + bx + c$ is the log of a Gaussian. So the answer is guaranteed to be Gaussian before you compute anything. **You are only extracting two numbers: the coefficient of $x_{t-1}^2$ (which gives the variance) and the coefficient of $x_{t-1}$ (which gives the mean).**

▸ **A Gaussian is fully described by two numbers, so "completing the square" is fully described by two coefficients.** For a log-density $-\frac{1}{2\tilde\beta}x^2 + \frac{\tilde\mu}{\tilde\beta}x + \text{const}$, the variance is the reciprocal of (twice) the quadratic coefficient and the mean falls out of the linear one. That is why the text can say "so the variance is its reciprocal" and move on — it is not a trick, it is pattern-matching against the shape of a Gaussian.

#### $\tilde\beta_t$ and $\tilde\mu_t$, decoded

**The tilde ($\tilde{\ }$) marks "the exact posterior version"** — $\tilde\beta_t$ is *not* $\beta_t$, and $\tilde\mu_t$ is *not* the forward mean. Read them "beta-tilde" and "mu-tilde." Ch. 0 §0.6 lists the tilde as "a modified version of"; here the modification is "conditioned on knowing $x_0$."

$$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$$

**Put numbers in.** At $t = 400$ with the linear schedule, $\bar\alpha_{399}\approx 0.213$, $\bar\alpha_{400}\approx 0.211$, $\beta_{400}\approx 0.008$:

$$\tilde\beta_{400} = \frac{1-0.213}{1-0.211}\times 0.008 = \frac{0.787}{0.789}\times0.008 \approx 0.00798$$

Just under $\beta_t$ — and it is *always* just under, because $\bar\alpha_{t-1} > \bar\alpha_t$ makes the fraction less than 1.

▸ **$\tilde\beta_t < \beta_t$ says: "knowing the clean image reduces your uncertainty about the intermediate step."** Information narrows the distribution. That is not a coincidence of the algebra; it is the definition of information. The shrinkage is small here because at $t=400$ the clean image barely constrains anything, but at $t=5$ the ratio is dramatic.

Now the mean:

$$\tilde\mu_t(x_t,x_0) = \underbrace{\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}}_{\text{weight on }x_0}\,x_0 + \underbrace{\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}_{\text{weight on }x_t}\,x_t$$

**Two coefficients, one on the clean image and one on the current noisy image, and they behave exactly as intuition demands.** Compute them at three noise levels:

| $t$ | weight on $x_0$ | weight on $x_t$ | Interpretation |
|---|---|---|---|
| $t=900$ (very noisy) | $\approx 0.0013$ | $\approx 0.993$ | *"I have almost no idea what the clean image is; mostly just take a tiny step from where I am."* |
| $t=400$ (medium) | $\approx 0.0047$ | $\approx 0.994$ | Still dominated by $x_t$ — one step is one step |
| $t=5$ (nearly clean) | $\approx 0.0009$ | $\approx 0.999$ | The step is minuscule; $x_t$ is already almost $x_0$ |

▸ **The weight on $x_t$ is near 1 at every $t$, and that is the correct and reassuring answer.** A single reverse step is supposed to be a *small correction*, not a leap to the answer. The $x_0$ term is the nudge; the $x_t$ term is the anchor. If the weight on $x_0$ were large, one step would jump straight to a clean image and there would be no need for a thousand of them — which is exactly the regime that distillation (§20.9) tries to engineer.

> **Analogy.** You are lost in fog and someone on a radio knows where the car park is. Each time you check in, they don't teleport you — they say "take one step, slightly left of where you're facing." Your position ($x_t$) determines almost all of where you end up next; their knowledge ($x_0$) determines the small course correction. A thousand small corrections get you to the car park.

**Why the exact posterior matters so much.** It converts the reverse process from "something we must invent and hope is expressive enough" into **"a known Gaussian whose mean we merely have to estimate."** All the network has to supply is one vector — and §20.4 shows it doesn't even have to supply that, only the noise.

---

## 20.4 The variational bound

Model the reverse as $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta)$. Apply the ELBO:

$$-\log p_\theta(x_0) \le \mathbb{E}_q\left[\underbrace{\mathrm{KL}(q(x_T|x_0)\|p(x_T))}_{L_T\text{, no params}} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{-\log p_\theta(x_0|x_1)}_{L_0}\right]$$

▸ The whole objective is a sum of **KLs between Gaussians**, each of which has a closed form. Since both are Gaussian with the same (fixed) variance:

$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\big\|^2\right] + C$$

**The model is doing regression on the posterior mean.**

#### The variational bound, decoded — read this slowly

This is the single densest display in the chapter, so take it in four passes: what the symbols mean, what the *shape* of the expression is, what each of the three groups of terms does, and why any of it is necessary.

**Pass 1 — the symbols.**

| Piece | Read aloud | Meaning |
|---|---|---|
| $-\log p_\theta(x_0)$ | "minus log p-theta of x-zero" | **The negative log-likelihood.** "How surprised is my model to see this real image?" Low = good |
| $\le$ | "is at most" | We are not computing the thing we want; we are computing a **ceiling** on it |
| $\mathbb{E}_q[\cdot]$ | "expectation under q" | Average over random draws of the forward process (Ch. 0 §0.5) |
| $\mathrm{KL}(a\|b)$ | "KL of a from b" | How far distribution $a$ is from distribution $b$. Zero iff identical, never negative |
| $q(x_{t-1}\mid x_t,x_0)$ | — | The **exact** one-step-back distribution from §20.3 — the answer key |
| $p_\theta(x_{t-1}\mid x_t)$ | — | The **network's** one-step-back distribution — the student's answer |
| $L_T,\ L_{t-1},\ L_0$ | "L-T", "L-t-minus-one", "L-zero" | Names for the three kinds of term, so we can discuss them separately |
| $\underbrace{\ \ }_{\text{label}}$ | — | Pure typography. A brace labelling the term above it. It changes nothing |

**Pass 2 — the shape.** Strip everything and the display says:

$$\text{(what we want to minimize)} \;\le\; \text{(a boundary term)} \;+\; \text{(a sum of }T-1\text{ comparison terms)} \;+\; \text{(a final reconstruction term)}$$

▸ **Everything on the right is computable; the thing on the left is not.** That asymmetry is the entire reason the bound exists. We cannot evaluate $\log p_\theta(x_0)$ because it would require integrating over all $10^{3}$-step paths from noise to this image. We *can* evaluate every term on the right in closed form. So we minimize the ceiling and the floor comes down with it.

> **Analogy — the mountain in cloud (the same picture Ch. 1 §1.4.4 uses for the ELBO).** You want to know the height of a peak hidden in cloud. You cannot see the summit, but you can see a cloud layer you know sits *above* it. Push the cloud layer down and the peak must come down too — and when the layer is resting on the summit, its height *is* the summit's height. **The ELBO is that cloud layer. "Variational" means "we optimize over a family of candidate bounds and pick the tightest."**

**Pass 3 — the three groups.**

**$L_T = \mathrm{KL}(q(x_T\mid x_0)\ \|\ p(x_T))$ — "did the forward process actually end in noise?"** It compares where the corruption *really* left the image after $T$ steps against $\mathcal{N}(0,I)$, the distribution we will sample from at generation time. It contains **no learnable parameters** — both sides are fixed — so it contributes nothing to the gradient and is usually dropped from the code entirely. Its practical role is diagnostic: if $L_T$ is not near zero, your schedule doesn't fully destroy the image, and the mismatch shows up as an artifact at the *start* of sampling. **This is exactly the "zero terminal SNR" bug in §20.12**, and it is why standard Stable Diffusion could never produce a truly black image.

**$L_{t-1} = \mathrm{KL}(q(x_{t-1}\mid x_t,x_0)\ \|\ p_\theta(x_{t-1}\mid x_t))$ — the $T-1$ terms that do all the work.** Read it as: *"at noise level $t$, how different is the network's guess about the previous step from the exact answer we can compute because we know $x_0$?"* This is **a supervised comparison against a known target**, repeated at every noise level. There is nothing adversarial, nothing sampled from the model, nothing unstable.

**$L_0 = -\log p_\theta(x_0\mid x_1)$ — the last step out.** Going from $x_1$ (barely noisy) to $x_0$ (an actual image) has to land on **discrete** pixel values in $\{0,\dots,255\}$, so it is handled as a discretized likelihood rather than a KL. It matters for reporting bits-per-dimension and matters not at all for image quality.

**Pass 4 — why the sum of KLs is a gift.** The KL divergence between two arbitrary distributions requires an integral. But the KL between **two Gaussians with the same covariance** collapses to a formula a schoolchild could evaluate:

$$\mathrm{KL}\big(\mathcal{N}(\mu_1,\sigma^2 I)\ \big\|\ \mathcal{N}(\mu_2,\sigma^2 I)\big) \;=\; \frac{1}{2\sigma^2}\|\mu_1-\mu_2\|^2$$

▸ **A KL between equal-variance Gaussians is just a scaled squared distance between their means.** That single fact converts the entire variational objective from an intractable information-theoretic quantity into $\|\text{target} - \text{prediction}\|^2$ — **plain mean squared error**. It is the reason the boxed $\mathcal{L}_{\text{simple}}$ below is legitimate rather than a heuristic, and it is why §20.3's derivation of the exact Gaussian posterior was worth the effort.

**Working it small.** Set $d=1$, $\sigma = 1$, target mean $\tilde\mu = 3$, network output $\mu_\theta = 2.5$. Then $L_{t-1} = \tfrac12(3-2.5)^2 = 0.125$. Not a divergence between distributions in any felt sense — an arithmetic difference of two numbers, squared and halved.

**Where the constant $C$ went.** The line ends with "$+\ C$", meaning **a term that does not depend on $\theta$**. Since training only ever uses $\nabla_\theta$, and the gradient of a constant is zero (Ch. 0 §0.7), $C$ can be discarded without changing a single update. Textbooks keep it because the *value* of the bound matters when you report likelihoods; implementations drop it because only the *gradient* matters when you train.

#### Why a bound at all — the honest version

A fair objection: why not just maximize the likelihood directly?

Because $p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}$ — an integral over **every possible trajectory** through $T$ intermediate images. With $T=1000$ and 786,432 dimensions per step, that is an integral over a space of $7.9\times10^{8}$ dimensions. There is no closed form and no usable Monte Carlo estimate: a random trajectory from noise almost never lands on your specific image, so the estimator's variance is astronomical.

The variational trick sidesteps this by **importing a proposal distribution** — here, the forward process $q$, which we already know and which by construction *does* connect $x_0$ to noise. Jensen's inequality (Ch. 1 §1.4.4) then converts "log of an average" into "average of a log," which is where the inequality sign comes from, and everything becomes an expectation we can estimate from one sample.

▸ **The variational bound is the price of admission, not the product.** Once you have it, the derivation's whole purpose is to show that a completely ordinary MSE denoising objective is *secretly* a valid likelihood bound. **You will never implement this equation.** You implement four lines of code, and this equation is why those four lines are principled.

> **Where this came from.** The variational lower bound is far older than diffusion. Its modern machine-learning form was assembled in the 1990s — the influential synthesis is the 1999 review by **Michael Jordan, Zoubin Ghahramani, Tommi Jaakkola, and Lawrence Saul** — as a way to make Bayesian inference tractable in graphical models by replacing an intractable posterior with the closest member of a simple family. It reached deep learning through the **variational autoencoder** of Kingma & Welling and, independently and near-simultaneously, Rezende, Mohamed & Wierstra, both in 2013–14. Diffusion inherits the machinery wholesale: a diffusion model is, structurally, a VAE with $T$ stacked latent layers, a *fixed* (untrained) encoder, and latents the same shape as the data. Seeing it that way explains why the objective looks the way it does — it is the VAE bound, written out $T$ times.

### The $\epsilon$-parameterization

From the closed form, $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon\right)$. Substitute into $\tilde\mu_t$ and simplify:

▸ $$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

So parameterize $\mu_\theta$ with the same form and a network $\epsilon_\theta$:

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)$$

Then the loss becomes:

▸ $$L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t)\big\|^2\right]$$

#### What the $\epsilon$-parameterization actually says

The section performs a change of variable, and change of variable is one of those moves that looks like magic and is really just bookkeeping. Here is the bookkeeping.

**Start from the forward closed form and solve it backwards for $x_0$.** We know $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$. Treat that as an equation in $x_0$ and rearrange:

$$x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\right)$$

*"The clean image equals the noisy image minus the noise, rescaled back up."* No probability theory involved — this is the same algebra as solving $y = 3x + 2$ for $x$.

▸ **The consequence is the key insight of the whole section: $x_0$ and $\epsilon$ carry exactly the same information, given $x_t$.** If you know the noisy image and you know the noise, you know the clean image, and vice versa. They are two coordinate systems on one unknown. **So a network that predicts $\epsilon$ has *implicitly* predicted $x_0$, and one that predicts $x_0$ has implicitly predicted $\epsilon$.** Nothing is gained or lost in principle; everything is gained in practice, for reasons of scaling.

**Substituting turns the target into something beautifully simple:**

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

Read it: *"take where you are, subtract a small multiple of the noise, and rescale slightly."* Compare it to the two-term weighted average from §20.3 — same object, but now **$x_t$ appears once instead of twice, and the only unknown is $\epsilon$.**

**Now the trick that makes it trainable.** Write the network's mean in *the identical algebraic form*, with $\epsilon_\theta(x_t,t)$ wherever the truth had $\epsilon$. Subtract the two. Every term that doesn't involve $\epsilon$ cancels exactly, and what survives is

$$\|\tilde\mu_t - \mu_\theta\|^2 \;=\; \underbrace{\frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}}_{\text{a number}}\ \big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2$$

▸ **A KL between distributions became a squared error between two mean vectors, which became a squared error between the true noise and the network's guess at it.** Three layers of abstraction collapse because the two parameterizations were chosen to share a form. **This is the pivotal derivation of the chapter** — everything before it is setup, everything after it is engineering.

#### Reading the weighted loss with real numbers

$$L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\underbrace{\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}}_{\text{weight }w(t)}\big\|\epsilon-\epsilon_\theta(x_t,\ t)\big\|^2\right]$$

- $\mathbb{E}_{x_0,\epsilon}$ — "average over training images **and** over noise draws." In code this is "average over the batch," since each example carries its own fresh $\epsilon$ (Ch. 0 §0.5).
- $\|\cdot\|^2$ — squared $\ell_2$ norm: subtract elementwise, square, add up. For an image this is summed over all 786,432 numbers.
- The fraction out front — a **fixed scalar depending only on $t$**, computable before training starts.

**Compute the weight $w(t)$ at two ends of the schedule** (taking $\sigma_t^2 = \beta_t$, so $w(t) = \beta_t/(2\alpha_t(1-\bar\alpha_t))$):

| $t$ | $\beta_t$ | $1-\bar\alpha_t$ | $w(t)$ | Effect |
|---|---|---|---|---|
| 1 | $0.0001$ | $0.0001$ | $\approx 0.5$ | Enormous relative emphasis |
| 100 | $0.002$ | $0.09$ | $\approx 0.011$ | Moderate |
| 500 | $0.010$ | $0.90$ | $\approx 0.0056$ | Small |
| 1000 | $0.020$ | $\approx 1.0$ | $\approx 0.010$ | Small |

▸ **The true bound cares about $t=1$ roughly fifty times more than about $t=500$.** And $t=1$ is the step that decides whether a pixel is `#3A7F2B` or `#3A7F2C`. The variational objective is telling you, correctly, that most of the *likelihood* lives in imperceptible high-frequency detail — and that is a fact about likelihood, not about pictures. Hence the next section.

### $L_\text{simple}$ — the loss everyone actually uses

Ho et al. found that **dropping the weight** works better:

▸ $$\boxed{\ \mathcal{L}_{\text{simple}} = \mathbb{E}_{t\sim\mathcal{U}[1,T],\,x_0,\,\epsilon\sim\mathcal{N}(0,I)}\left[\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2\right]\ }$$

▸ **The entire training algorithm is four lines:**
```
x0 ~ data;  t ~ Uniform{1..T};  eps ~ N(0,I)
xt = sqrt(abar[t])*x0 + sqrt(1-abar[t])*eps
loss = || eps - eps_theta(xt, t) ||^2
backprop
```
**A diffusion model is a denoiser trained with MSE.** All the machinery above exists to justify that this simple objective is a bound on the negative log-likelihood.

**Why the reweighting helps:** the true VLB weights heavily toward small $t$ (nearly-clean images, where the KL is large but the perceptual content is minor). $\mathcal{L}_{\text{simple}}$ up-weights larger $t$, where the model must decide global structure. It is a worse *likelihood* objective and a better *perceptual* one.

#### $\mathcal{L}_{\text{simple}}$, decoded

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t\sim\mathcal{U}[1,T],\,x_0,\,\epsilon\sim\mathcal{N}(0,I)}\left[\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2\right]$$

Every symbol under the $\mathbb{E}$ names one thing that gets randomly drawn:

| Under the E | Read aloud | In code |
|---|---|---|
| $t\sim\mathcal{U}[1,T]$ | "t drawn uniformly from 1 to T" | `t = randint(1, 1001)` — every noise level equally often |
| $x_0$ | "x-zero" | The next image out of the data loader |
| $\epsilon\sim\mathcal{N}(0,I)$ | "epsilon drawn from a standard normal" | `eps = randn_like(x0)` |

And the body $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ is `((eps - model(xt, t))**2).mean()`.

▸ **Compare it to the weighted version: the only difference is that the scalar $w(t)$ has been deleted.** That's it. Someone tried setting all the weights to 1, the pictures got better, and it became the standard. **The most influential objective in generative modelling is a principled bound with one factor crossed out.**

#### The four-line training algorithm, line by line

```
x0 ~ data;  t ~ Uniform{1..T};  eps ~ N(0,I)
xt = sqrt(abar[t])*x0 + sqrt(1-abar[t])*eps
loss = || eps - eps_theta(xt, t) ||^2
backprop
```

1. **Draw three independent random things.** An image, a noise level, and a noise vector. None depends on the others; all three are drawn fresh per example.
2. **Corrupt in one shot** using §20.2's closed form. `abar` is a precomputed lookup table of 1000 floats.
3. **Ask the network to name the noise it sees**, and score it with plain squared error. The network is *told* $t$ — that's the second argument — so it knows how much noise it should expect.
4. **Backpropagate.** Ordinary supervised learning from here on.

▸ **There is no sampling from the model during training, no discriminator, no reinforcement, no adversarial game, no sequential rollout.** Compare with GANs (Ch. 19), where two networks chase each other and either can collapse. **A diffusion model's training loop is indistinguishable from training a regressor on synthetic labels — and that boring stability is the single biggest reason diffusion displaced GANs.** You can train one for three weeks on a large cluster and it will not diverge.

> **Analogy — the noise-spotting exam.** Imagine a classroom exercise: an examiner takes a photograph, adds a known amount of static, and asks the student to point at exactly where the static is and how strong. Sometimes the static is barely there ($t=3$); sometimes the photograph is entirely gone ($t=980$) and the honest answer is "the whole thing is static." **The student is graded on squared error and told the difficulty level in advance.** Do this a billion times and the student has, without ever being asked to, learned what photographs look like — because you cannot separate signal from noise unless you know what signal looks like.

#### Why deleting the weight is a *perceptual* choice, and how to say why

The text says $\mathcal{L}_{\text{simple}}$ is "a worse likelihood objective and a better perceptual one." That trade deserves spelling out, because it is the clearest example in the book of a mismatch between what likelihood measures and what humans see.

- **Likelihood is dominated by fine detail.** An image is roughly 786,000 numbers, and the overwhelming majority of the *bits* needed to specify it exactly go into high-frequency texture — the precise value of each pixel relative to its neighbours. Getting the global composition right buys you almost no bits.
- **Human perception is dominated by coarse structure.** Whether the picture contains a dog at all is decided at $t\approx 600$–$900$. Whether the fur texture is exactly right is decided at $t\approx 10$–$50$, and viewers barely register it.
- The true VLB weight $w(t)$ therefore spends the model's capacity on precisely the part nobody looks at.

▸ **Deleting $w(t)$ redistributes the model's attention from "the bits that cost the most" to "the decisions that matter the most."** This is why FID improves while bits-per-dimension gets worse — the two metrics are measuring different things, and $\mathcal{L}_{\text{simple}}$ knowingly sacrifices one. **If someone asks you why diffusion models report poor likelihoods compared to autoregressive models, this is the answer: they are not trained to optimize likelihood, on purpose.**

> **Where this came from.** The 2015 Sohl-Dickstein paper trained the full weighted variational bound and produced correct but blurry samples. The 2020 DDPM paper of **Ho, Jain, and Abbeel** made three changes: predict $\epsilon$ rather than $\tilde\mu$ or $x_0$; drop the per-timestep weight; and use a substantially larger U-Net with attention at low resolutions. Those changes moved image quality from "clearly a research prototype" to "competitive with the best GANs," and the paper is explicit that the reweighting was found empirically and justified after the fact. **The single most consequential line in modern generative modelling is a simplification that the authors could not fully explain at the time.**

#### Examples and non-examples: what $\epsilon_\theta$ is actually predicting

**✅ What the network  does**

| Example | Why it qualifies |
|---|---|
| Given $x_t$ at $t=500$, emit a tensor the same shape as the image that best matches $\mathbb{E}[\epsilon \mid x_t, t]$ | Squared error's minimizer is the **conditional mean** — an average over every clean image consistent with $x_t$ |
| Receiving $t$ as an explicit second input | The same tensor $x_t$ means different things at different noise levels; the network is told which |
| Producing an output for an $x_t$ that no real image ever generated | It is a function of $(x_t, t)$ alone — at sampling time it has never seen an $x_0$ |
| $v$-prediction, $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | An invertible reparameterization of the same regression; you can recover $\epsilon$ from it exactly |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "It recovers the exact $\epsilon$ that was drawn" | Impossible — millions of $(x_0,\epsilon)$ pairs produce the same $x_t$ at $t=800$ | It predicts the **average** of those $\epsilon$'s, which is why a one-shot $\hat x_0$ is blurry |
| "It's a denoiser, so its output is the clean image" | The output is the *noise*; $\hat x_0$ is derived from it by algebra | The $\epsilon$-parameterization. $x_0$-prediction is a different head for the same information |
| "It estimates how noisy the input is" | $t$ is an **input**, not an output | Timestep conditioning |
| A classical denoiser — BM3D, a Gaussian filter, a bilateral filter | One fixed noise level, no learned prior over images, no $t$ | Image restoration |
| The discriminator in a GAN | Outputs a scalar judgement, trained adversarially against a moving opponent | A critic, not a regressor (Ch. 19) |

▸ **The boundary:** $\epsilon_\theta(x_t,t)$ is a plain supervised regression onto $\mathbb{E}[\epsilon \mid x_t, t]$ — a **conditional average**, not a recovery, and every characteristic behaviour of diffusion follows from that one word.

> **Common misconception.** *"The model learns to undo the specific noise that was added — training is a matching game."* It cannot, and it doesn't. At $t=800$ the same noisy tensor is reachable from millions of different clean images paired with millions of different noise draws, so squared error drives the network to the **average of all compatible answers**, never to the particular draw. This is precisely why jumping in one shot from $x_{800}$ to $\hat x_0$ gives a washed-out, over-smoothed blob: you are looking at an average of possibilities, not a photograph. **The thousand small steps exist so that each step only has to average over a *small* set of possibilities**, and the noise injected at each step commits you to one branch before the next average is taken. The misconception is tempting because the loss literally compares against the $\epsilon$ that was drawn — but a loss that compares against a random sample still has its minimum at that sample's mean.

> **Common misconception.** *"The three parameterizations are three different models, so pick the best one."* They carry identical information: given $x_t$, $t$, and any one of $\{\hat x_0, \hat\epsilon, \hat v\}$ you can compute the other two with a two-term linear formula. **What differs is the numerical conditioning of the regression target, not the content.** At $t=950$ the target $x_0$ is almost unconstrained by the input, so predicting it is a high-variance job; the target $\epsilon$ is nearly the input itself, so predicting it is easy and well-scaled. At $t=5$ the situation reverses. The belief is tempting because the three heads are drawn as three architectures in most diagrams, when the only real difference is which linear combination you ask a finite-precision network to output.

### The three parameterizations

| Predict | Relation | Best at |
|---|---|---|
| $x_0$ | direct | small $t$ (target is well-determined) |
| $\epsilon$ | $x_0 = \frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$ | large $t$; **the standard** |
| $v$ | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | **all $t$**; best for distillation and high resolution |

▸ **Why $\epsilon$ beats $x_0$:** at large $t$, $x_t$ is nearly pure noise, so $\epsilon\approx x_t$ and the target is well-scaled; but $x_0$ is almost unconstrained, so predicting it is a high-variance target. Conversely at small $t$, predicting $\epsilon$ is hard because tiny errors get amplified by $1/\sqrt{\bar\alpha_t}$. **$v$-prediction interpolates smoothly between them**, which is why it dominates for distillation and very high SNR ranges.

**SNR framing:** $\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$. Different parameterizations and loss weightings are all just different functions $w(\mathrm{SNR})$ multiplying the same underlying regression. **Min-SNR-$\gamma$ weighting** ($w = \min(\mathrm{SNR},\gamma)/\mathrm{SNR}$ for $\epsilon$-pred) resolves the conflicting gradients between timesteps and speeds convergence substantially.

#### The three parameterizations, decoded

All three networks see the same input ($x_t$ and $t$) and are mathematically interchangeable — given any one of them you can compute the other two. **What differs is the numerical conditioning of the target**, and in a finite-precision world trained by finite-variance gradients, that decides everything.

| Parameterization | The network is asked | Target magnitude |
|---|---|---|
| $x_0$-prediction | "What was the original picture?" | Always order 1 (an image) |
| $\epsilon$-prediction | "What noise was added?" | Always order 1 (a standard normal draw) |
| $v$-prediction | "What is the *velocity* along the noising path?" | Order 1 by construction, at every $t$ |

**Why $\epsilon$ wins at large $t$, with numbers.** At $t=1000$, $\sqrt{\bar\alpha_t} = 0.006$. The recovery formula is

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}} \;=\; \frac{x_t - 0.99998\,\hat\epsilon}{0.006}$$

▸ **Dividing by $0.006$ multiplies any error by 167.** So at high noise, asking the network for $x_0$ directly means asking it to name a quantity that the input barely determines — the target has enormous conditional variance, and the network's best possible answer is a blurry average of every image that could have produced this static. Asking for $\epsilon$ instead is asking for something $x_t$ almost *is*, so the target is well-determined and the regression is well-posed.

**Why $\epsilon$ loses at small $t$.** Run the same formula at $t=1$: $\sqrt{\bar\alpha_t} = 0.99995$, $\sqrt{1-\bar\alpha_t} = 0.01$. Now $\hat x_0 \approx x_t - 0.01\hat\epsilon$ — the noise term is multiplied by $0.01$, so **the network's $\epsilon$ prediction is almost irrelevant to the answer, and its gradient signal is correspondingly weak.** Worse, the errors that *do* matter get amplified in the other direction when you convert back. The parameterization has become a magnifying glass pointed at a tiny quantity.

> **Analogy.** You are asked to state a person's exact height. Two ways to answer: name the height directly, or name their deviation from the average. When you can see them clearly, naming the height is easy and naming a deviation is fussy. When they are a silhouette a mile away, naming the height is a wild guess but "about average" is a defensible answer. **$x_0$-prediction is naming the height; $\epsilon$-prediction is naming the deviation. Which is better depends entirely on how much you can see — which is exactly what $t$ controls.**

**$v$-prediction, decoded.** $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ looks arbitrary until you notice the coefficients: they are the *same two numbers* as in the forward process, but **swapped and with a sign flip.** Geometrically, if you write $\sqrt{\bar\alpha_t} = \cos\phi$ and $\sqrt{1-\bar\alpha_t} = \sin\phi$ (legal, since they square to 1), then

$$x_t = \cos\phi\, x_0 + \sin\phi\,\epsilon, \qquad v = \cos\phi\,\epsilon - \sin\phi\, x_0$$

▸ **$v$ is $x_t$ rotated by 90°.** It is the *tangent* to the circular path from data to noise — hence "velocity." Because a rotation never changes a vector's length, **$v$ has order-1 magnitude at every single $t$, with no bad regime at either end.** That is the whole argument for $v$-prediction, and it is why distillation (§20.9), which must be accurate at very few, very widely-spaced timesteps, essentially requires it.

> **Where this came from.** $v$-prediction was introduced by **Tim Salimans and Jonathan Ho** in their 2022 paper on **progressive distillation**, and it was introduced out of necessity rather than elegance: when you distil a 1000-step sampler down to 4 steps, the student is evaluated at timesteps where $\epsilon$-prediction is numerically hopeless, and training simply failed. The fix — rotate the target so its scale is constant — has since been adopted far beyond distillation, and is standard in high-resolution and video models.

#### Reading the SNR framing — the unifying idea

$$\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t} = \frac{\text{signal variance}}{\text{noise variance}}$$

**Signal-to-noise ratio** is a term borrowed straight from electrical engineering, where it has meant exactly this since the era of radio: how loud is the message compared to the hiss. Here the "message" is the clean image, scaled by $\bar\alpha_t$, and the "hiss" is the accumulated noise, of variance $1-\bar\alpha_t$.

| $t$ | $\bar\alpha_t$ | $\mathrm{SNR}(t)$ | In decibels | Analogy |
|---|---|---|---|---|
| 1 | $0.9999$ | $\approx 10{,}000$ | $+40$ dB | A studio recording |
| 100 | $0.91$ | $\approx 10$ | $+10$ dB | A phone call |
| 400 | $0.21$ | $\approx 0.27$ | $-6$ dB | A voice in a noisy bar |
| 900 | $0.0004$ | $\approx 0.0004$ | $-34$ dB | Static, with a rumour of a voice |

▸ **SNR is the honest $x$-axis for everything in this chapter.** The timestep index $t$ is an arbitrary label — nothing forces it to be an integer from 1 to 1000, and continuous-time formulations (§20.7) discard it entirely. SNR is the physically meaningful coordinate, and once you plot in it, the noise schedule, the parameterization, and the loss weighting stop being three separate design choices and become **one function: how much weight do you place on each SNR level?**

**Min-SNR-$\gamma$, unpacked.** The weight $w = \min(\mathrm{SNR},\gamma)/\mathrm{SNR}$ reads: *"weight each timestep by 1, except when the SNR exceeds $\gamma$, in which case cap it."* With $\gamma = 5$: at $\mathrm{SNR} = 2$ the weight is 1; at $\mathrm{SNR}=10{,}000$ the weight is $5/10{,}000 = 0.0005$. **It is a clamp on how much the nearly-clean timesteps are allowed to shout.**

The problem it solves is worth naming, because it recurs everywhere in multi-task learning: **conflicting gradients.** One network serves a thousand tasks (one per $t$). The tasks disagree — the update that best sharpens fine texture is not the update that best decides global layout — and if a handful of tasks produce gradients hundreds of times larger than the rest, they dominate the average and the others make no progress. Capping the weight equalizes the tasks' voices. Reported speedups on convergence are substantial, typically several-fold in training steps to a given FID.

---

## 20.5 Sampling

**Ancestral (DDPM):**
▸ $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z,\qquad z\sim\mathcal{N}(0,I),\ z=0 \text{ at } t=1$$

$\sigma_t^2 = \beta_t$ or $\tilde\beta_t$ both work; the choice matters little.

$T=1000$ steps. Slow — this is diffusion's one real weakness, addressed in §20.9.

#### The sampling step, decoded

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)}_{\text{the deterministic estimate }\mu_\theta} + \underbrace{\sigma_t z}_{\text{a fresh random kick}}$$

Read it as a three-step recipe, executed a thousand times:

1. **Ask the network how much noise it sees**: $\epsilon_\theta(x_t,t)$.
2. **Remove a fraction of it.** Not all of it — only $\beta_t/\sqrt{1-\bar\alpha_t}$ of it, which at $t=500$ is about $0.0105$, roughly 1%. Then divide by $\sqrt{\alpha_t}\approx0.995$ to restore the scale.
3. **Add a small fresh random kick** $\sigma_t z$, with $z$ a new standard normal draw.

**"$z=0$ at $t=1$"** means the last step is deterministic — you do not sprinkle noise onto the final image, since it is supposed to be the answer.

▸ **Why add noise back in on the way *down*? This is the question that trips up everyone.** It seems perverse: we are trying to remove noise, and each step deliberately adds some. The reason is that **we are sampling from a distribution, not solving for a point.** $q(x_{t-1}\mid x_t,x_0)$ has a  spread $\tilde\beta_t$; if you always take its mean you are not drawing from it, you are taking its mode a thousand times in a row, and the result collapses toward the average image. **The added noise is what makes the output a sample rather than a summary.**

> **Analogy.** Sanding the statue again — but this time the sander vibrates slightly. If the tool were perfectly rigid you would converge on the single most "average" statue the material allows. The vibration is what lets you land on *a* statue rather than *the mean* statue. §20.7's probability-flow ODE is the remarkable result that you can remove the vibration entirely and still get the same distribution of statues, by changing the sanding rule.

**"$\sigma_t^2=\beta_t$ or $\tilde\beta_t$ both work; the choice matters little."** These are the two natural guesses: $\beta_t$ is the forward step's variance, $\tilde\beta_t$ is the exact posterior variance from §20.3. They are the **upper and lower bounds** on the correct value, and since $\tilde\beta_t/\beta_t = (1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$ is within a percent of 1 for almost all $t$, the two choices differ by almost nothing. Learning $\Sigma_\theta$ instead (as Nichol & Dhariwal's *Improved DDPM* did) improves likelihood measurably and image quality barely.

#### Why 1000 steps is diffusion's one real weakness

Put a cost on it. A 1000-step sample requires **1000 forward passes of a large network**. With classifier-free guidance (§20.8) that becomes **2000**. Generating one image therefore costs about as much compute as classifying two thousand images.

| Model class | Network evaluations per sample |
|---|---|
| GAN | 1 |
| VAE | 1 |
| Normalizing flow | 1 |
| Autoregressive image model | one per pixel (thousands, but each is cheap) |
| **DDPM** | **1000, each a full U-Net** |
| DDIM / DPM-Solver++ | 10–50 |
| Consistency / distilled | 1–4 |

▸ **Everything in §20.7 and §20.9 exists to attack that one number.** The path from 1000 to 4 evaluations, over roughly three years, is one of the fastest and most consequential efficiency campaigns in the field — and the interesting part is that almost none of it required retraining the base model. The 1000 was never inherent to the model; it was inherent to the *sampler*.

#### Examples and non-examples: the noise schedule versus the sampler

Two words that sound interchangeable, are swapped constantly in forum posts and product interfaces, and refer to entirely different objects — decided at different times, by different people, with different costs.

**✅ Noise schedule — a property of the *trained model***

| Example | Why it qualifies |
|---|---|
| Linear $\beta_t$ from $10^{-4}$ to $0.02$ | Defines $\bar\alpha_t$, and therefore defines what the index "$t=700$" *means* |
| The cosine schedule with $s=0.008$ | Same role, gentler decay — it changes the data the network is trained on |
| A resolution-shifted schedule for 1024px | Changes the corruption itself, so it requires retraining |
| Zero-terminal-SNR rescaling | Alters $\bar\alpha_T$, hence the training distribution at the top of the chain |

**✅ Sampler — a property of *how you run* the trained model**

| Example | Why it qualifies |
|---|---|
| DDPM ancestral sampling, 1000 steps | A rule for turning $\epsilon_\theta$ into $x_{t-1}$, applied to fixed weights |
| DDIM with $\eta=0$, 50 steps | A different rule, the **same checkpoint**, no retraining |
| DPM-Solver++, 15 steps | A higher-order numerical integrator for the same trajectory |
| Euler, Heun, LMS, UniPC | All solvers of the probability-flow ODE of §20.7 |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "I switched to the Karras *schedule* in the interface" | It changes which timesteps the sampler visits, not the corruption used in training | A **timestep discretization** — a sampler-side choice that inherited the word "schedule" |
| "Fewer steps means it was trained with fewer steps" | Training $T$ and the number of sampling evaluations are independent numbers | Step **subsampling** |
| "DDIM is a faster way to train" | DDIM never touches training at all | An inference-time solver |
| Changing the guidance scale $w$ | Neither the corruption nor the integrator — it modifies the vector field being integrated | A guidance weight (§20.8) |
| Distillation to 4 steps | This *does* retrain — a student learns to take several teacher steps at once | A new set of weights, not a new sampler |

▸ **The boundary:** change the **schedule** and you must retrain; change the **sampler** and you may swap it after the fact, on the same checkpoint, halfway through a project, for free.

> **Common misconception.** *"More sampling steps always means better images."* Quality climbs steeply from 5 steps to roughly 30 and then flattens; past about 50 the improvement is often invisible in a blind comparison, and with stochastic samplers it can reverse — every ancestral step injects fresh noise, so a thousand tiny steps deliver a thousand tiny doses of randomness that can sand away fine detail. Meanwhile the cost is strictly linear: 100 steps is exactly twice the money of 50. The belief is tempting because everywhere else in numerical analysis a finer discretization is strictly more accurate, and here it  is — for the *solver*. **But "integrating the model's trajectory more accurately" and "producing a picture a person prefers" are different targets, and after roughly 30 steps the binding error is the model's, not the solver's.** Refining your integration of a slightly wrong vector field converges beautifully to a slightly wrong answer.

> **Common misconception.** *"$T = 1000$ is baked into the model, so it needs 1000 evaluations."* The $T$ in the training procedure fixes the *meaning of the timestep index*; it does not obligate you to visit every index. DDIM samples the identical weights in 50 steps by skipping indices, and DPM-Solver++ does it in 15 by treating the whole trajectory as an ordinary differential equation and using a better integrator on it. **None of that required retraining a single parameter** — which is why the 1000→50 speedup arrived as a paper about sampling rather than as a new model. The belief is tempting because $T=1000$ appears prominently in the training algorithm and in every diagram of the forward chain, so it reads like an architectural constant rather than a coordinate system.

---

## 20.6 The score-based view

### Score matching

Define the **score** as $s(x) = \nabla_x\log p(x)$ — the direction of steepest increase in log-density. Note it is independent of the normalizing constant (Ch. 19 §19.6), which is what makes it tractable.

**Langevin dynamics** samples from $p$ using only the score:
$$x_{k+1} = x_k + \frac{\delta}{2}\nabla_x\log p(x_k) + \sqrt{\delta}\,z_k$$

#### The score, decoded — and why it dodges the hardest problem in generative modelling

$$s(x) = \nabla_x\log p(x)$$

| Piece | Read aloud | Meaning |
|---|---|---|
| $\nabla_x$ | "grad x" / "del x" | The gradient **with respect to the input $x$**, not the parameters (Ch. 0 §0.7) |
| $\log p(x)$ | "log p of x" | The log-probability of the point $x$ |
| $s(x)$ | "the score at x" | A vector, **the same shape as $x$**, pointing toward higher probability |

▸ **In one sentence: the score is an arrow, drawn at every point in image space, pointing in the direction that makes the image more plausible.** For a picture, it is literally a $512\times512\times3$ array saying "brighten this pixel a bit, darken that one, and the whole thing becomes more photograph-like."

> **Analogy.** Picture the probability distribution as a landscape, with mountains where data is common and flat plains where it is not. **The score is the compass reading at your feet: which way is uphill, and how steep.** Chapter 0's foggy-hillside picture, transplanted from parameter space to data space. Gradient descent on parameters climbs down a loss; Langevin dynamics on data climbs up a density.

**Now the crucial property — why the score is *computable* when the density is not.** Every probability density has the form

$$p(x) = \frac{\tilde p(x)}{Z}, \qquad Z = \int \tilde p(x)\,dx$$

where $\tilde p$ is an unnormalized score of plausibility your model can output freely, and $Z$ — the **partition function** — is the number that makes everything integrate to 1. **$Z$ is the great villain of classical generative modelling.** It is an integral over the entire space of images, it has no closed form, and estimating it is harder than the original problem. Energy-based models, Boltzmann machines, and Markov-chain Monte Carlo all spent decades wrestling with it.

Watch it vanish:

$$\nabla_x \log p(x) = \nabla_x\big[\log \tilde p(x) - \log Z\big] = \nabla_x\log\tilde p(x) - \underbrace{\nabla_x \log Z}_{=\ 0}$$

▸ **$Z$ is a constant — it does not depend on $x$ — so its gradient with respect to $x$ is exactly zero.** The score of a distribution is completely unaffected by the normalizing constant. **You can model the shape of a distribution without ever knowing its total mass.** This one line is why score-based methods exist, and it is the same "the normalizing constant contains no insight" observation Ch. 0 §0.12 makes about the Gaussian density — here promoted from a reading tip to the foundation of a research programme.

#### Langevin dynamics, decoded

$$x_{k+1} = x_k + \frac{\delta}{2}\underbrace{\nabla_x\log p(x_k)}_{\text{climb toward likely}} + \underbrace{\sqrt{\delta}\,z_k}_{\text{random jitter}},\qquad z_k\sim\mathcal{N}(0,I)$$

- $\delta$ — "delta," a small step size, playing exactly the role of a learning rate.
- The first term is **gradient ascent on log-probability**: walk uphill toward more plausible images.
- The second term is **noise deliberately injected**, with standard deviation $\sqrt{\delta}$.

**Why the noise, and why $\sqrt{\delta}$ specifically?** Without the noise this is plain gradient ascent, which converges to the single highest peak — the *most likely* image, over and over. That is a mode-finder, not a sampler. The noise lets the walker wander, and the specific pairing of $\delta/2$ on the gradient with $\sqrt{\delta}$ on the noise is what makes the walk's long-run distribution equal to $p$ itself rather than to some sharpened or flattened version of it. **Note the mismatch in powers: the drift scales as $\delta$, the noise as $\sqrt{\delta}$.** That is the $\sqrt{\ }$ signature of Brownian motion — random walks accumulate as the square root of time, the same $1/\sqrt{n}$ law as the standard error in Ch. 1 §1.3.1, for the same reason.

> **Analogy — a marble in a bowl on a shaking table.** Gravity (the score) pulls the marble toward the bottom. The shaking (the noise) keeps knocking it around. If you shake at exactly the right intensity relative to the slope, the marble's long-run position histogram traces out the shape of the bowl. Too little shaking and it sits at the bottom forever; too much and it flies out and the histogram is uniform. **Langevin dynamics is the recipe for shaking at exactly the right intensity.**

▸ **Why plain Langevin dynamics on real data fails, and why diffusion fixes it.** The score of a natural-image distribution is meaningless almost everywhere: real images occupy a vanishingly thin manifold in pixel space, so at a random starting point the density is effectively zero and the "uphill direction" is undefined — there is no signal to follow. **Diffusion's answer is to blur the landscape.** At $t=900$ the noised distribution is nearly a wide Gaussian with a well-defined score everywhere; as $t$ decreases the landscape sharpens back toward the true one. **Running Langevin at a sequence of decreasing noise levels — starting where the score is easy and ending where it is meaningful — *is* the diffusion sampler.** This is precisely the argument of Song & Ermon's noise-conditional score networks, and it is why they and DDPM converged on the same algorithm from opposite directions.

> **Where this came from.** **Paul Langevin** wrote his equation in 1908, in a three-page note giving what he called "an infinitely simpler" derivation of Einstein's 1905 results on Brownian motion. His equation splits the force on a suspended particle into a smooth drag term and a rapidly fluctuating random term — exactly the drift-plus-noise structure above. Langevin was a student of Pierre Curie, and during the First World War he led French work on piezoelectric ultrasonic echo-ranging, the direct ancestor of sonar. **Score matching** as an estimation method is much more recent: **Aapo Hyvärinen** introduced it in 2005, precisely to fit unnormalized models without touching $Z$. Its practical form here — *denoising* score matching, where you estimate the score by learning to denoise — was proved equivalent by **Pascal Vincent in 2011**, in a paper connecting score matching to denoising autoencoders. Vincent's result is the mathematical bridge that makes "predict the noise" and "estimate the score" literally the same training objective, and it predates DDPM by nine years.

### The connection

For $q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$:

$$\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

▸ $$\boxed{\ s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}\ }$$

**Predicting the noise and estimating the score are the same task up to a scale factor.** DDPM (Ho et al.) and NCSN (Song & Ermon) were developed independently and turned out to be the same algorithm. This is one of the more satisfying convergences in modern ML, and knowing it is a strong signal in an interview.

**Tweedie's formula** makes the third connection explicit: for $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma\epsilon$,
$$\mathbb{E}[x_0\mid x_t] = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log q(x_t)\right)$$
▸ **The optimal denoiser, the score, and the posterior mean are the same object.** Three literatures, one function.

#### Deriving the noise–score identity in two lines

The claim is $s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$, and it follows from differentiating a Gaussian, which is the easiest calculus in the chapter.

For $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)I)$, the log-density is

$$\log q(x_t\mid x_0) = -\frac{\|x_t - \sqrt{\bar\alpha_t}x_0\|^2}{2(1-\bar\alpha_t)} + \text{const}$$

Differentiate with respect to $x_t$. The derivative of $-\|x_t - c\|^2/(2\sigma^2)$ is $-(x_t-c)/\sigma^2$ (Ch. 1 §1.2.2's $\nabla_x\|x\|^2 = 2x$, with the chain rule). So

$$\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}$$

Now substitute the forward process: $x_t - \sqrt{\bar\alpha_t}x_0$ **is** $\sqrt{1-\bar\alpha_t}\,\epsilon$, by definition. Cancel one factor:

$$= -\frac{\sqrt{1-\bar\alpha_t}\,\epsilon}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

▸ **The score *is* the noise, negated and rescaled.** Nothing deeper is happening: for a Gaussian, "which way is uphill" is "back toward the centre," and "back toward the centre" is "minus the noise that took you away from it." The minus sign is the whole geometric content — **the noise points away from plausibility, so the score points opposite it.**

**Why the scale factor matters in practice.** $1/\sqrt{1-\bar\alpha_t}$ ranges from $1/0.01 = 100$ at $t=1$ to $\approx 1$ at $t=1000$. So an $\epsilon$-network and a score-network trained identically will have wildly different output scales at low noise — which is why score-based implementations (NCSN) had to introduce careful output scaling by hand, while DDPM's $\epsilon$-parameterization got it for free. **Same function, better units.**

#### Why "DDPM and NCSN turned out to be the same algorithm" is worth pausing on

Two groups, working simultaneously and independently, with completely different motivations:

| | **DDPM** (Ho, Jain, Abbeel, 2020) | **NCSN** (Song & Ermon, 2019) |
|---|---|---|
| Starting point | Non-equilibrium thermodynamics; a variational bound | Score matching; how to sample without a partition function |
| The stated goal | Maximize a likelihood lower bound | Estimate $\nabla_x\log p$ at many noise levels |
| The network predicts | The noise $\epsilon$ | The score $s$ |
| The sampler | Ancestral sampling from a learned reverse Markov chain | Annealed Langevin dynamics |
| **What it turned out to be** | **The same thing, up to $s = -\epsilon/\sqrt{1-\bar\alpha_t}$** | |

▸ **Convergences like this are the strongest available evidence that you have found something real rather than something invented.** When two derivations that share no vocabulary land on the same algorithm, the algorithm is not an artifact of either derivation. Compare Ch. 1's SVD (discovered twice in consecutive years) and backpropagation (four times). **In an interview, being able to state the identity $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ *and* explain that it unified two literatures is a strong signal**, because it demonstrates you have read past the implementation.

#### Tweedie's formula, decoded

$$\mathbb{E}[x_0\mid x_t] = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log q(x_t)\right)$$

- $\mathbb{E}[x_0\mid x_t]$ — "the **average clean image**, over all clean images that could have produced this noisy one." Note it is an average, and Ch. 0 §0.5 warns that an average need not be a possible outcome: at high noise this is a blur, not a photograph, and correctly so.
- The right-hand side says: **take where you are, push a little way along the score, and rescale.**

**Read it as a statement about optimal denoising.** The best possible denoiser — best in the squared-error sense, which is what $\mathcal{L}_{\text{simple}}$ measures — outputs the conditional mean. Tweedie says that conditional mean is computable from the score alone. So:

$$\underbrace{\text{the MSE-optimal denoiser}}_{\text{a regression problem}} \;=\; \underbrace{\text{the posterior mean}}_{\text{a Bayesian object}} \;=\; \underbrace{\text{a function of the score}}_{\text{a geometric object}}$$

▸ **Three fields asked three different questions and got one answer.** Signal processing asked "how do I clean up a noisy measurement?" Bayesian statistics asked "what should I believe about the truth given the observation?" Score-based modelling asked "which way is uphill in probability?" **The optimal denoiser, the posterior mean, and the score are one function wearing three costumes** — and it is the function your U-Net computes.

**The practical payoff.** Because of Tweedie, any trained diffusion model gives you, for free and at any noise level, an estimate of the clean image: $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$. Every sampler in §20.7 uses this quantity, every progress visualization plots it, and clamping it to the valid pixel range (`x0_clip`) is one of the most common and most effective practical hacks in diffusion codebases.

> **Where this came from.** The formula carries the name of **Maurice Tweedie**, though the first appearance in print is a 1956 paper by **Herbert Robbins** on empirical Bayes, where Robbins credits Tweedie for it. **Bradley Efron** brought the name back into wide circulation with a 2011 paper, *Tweedie's Formula and Selection Bias*, where the application was correcting for the fact that the most extreme measurements in a study are the ones most likely to have been inflated by luck — the statistical phenomenon behind regression to the mean and the "winner's curse." **A result developed to stop scientists from over-trusting their most spectacular results is now the reason your image generator works.** It is a  strange lineage, and a good illustration of how narrow the toolkit of applied mathematics actually is.

---

## 20.7 The SDE / ODE formulation

Song et al. (2021) showed diffusion is the discretization of a stochastic differential equation.

**Forward SDE:**
$$dx = f(x,t)\,dt + g(t)\,dw$$
- **VP-SDE** (= DDPM): $f=-\frac12\beta(t)x$, $g=\sqrt{\beta(t)}$.
- **VE-SDE** (= NCSN): $f=0$, $g=\sqrt{\frac{d\sigma^2(t)}{dt}}$.

**Reverse SDE** (Anderson, 1982):
▸ $$dx = \left[f(x,t) - g(t)^2\nabla_x\log p_t(x)\right]dt + g(t)\,d\bar w$$

**Probability-flow ODE** — the crucial one:
▸ $$\frac{dx}{dt} = f(x,t) - \frac12 g(t)^2\nabla_x\log p_t(x)$$

**This deterministic ODE has exactly the same marginal distributions $p_t(x)$ as the SDE at every $t$.**

Consequences:
1. **Deterministic sampling** — same noise always gives the same image; latents are meaningful and interpolable.
2. **Any ODE solver applies.** Heun, DPM-Solver, RK4 — decades of numerical analysis become available, and this is what cut sampling from 1000 steps to 20.
3. **Exact likelihoods** via the instantaneous change of variables (it is a continuous normalizing flow, Ch. 19 §19.5).

#### Reading a stochastic differential equation without a course in stochastic calculus

$$dx = f(x,t)\,dt + g(t)\,dw$$

You do not need Itô calculus to read this. You need one translation: **an SDE is a recipe for a very small step, and $d$ means "a very small amount of."**

| Piece | Read aloud | Meaning |
|---|---|---|
| $dx$ | "dee-x" | The tiny change in $x$ over a tiny interval of time |
| $dt$ | "dee-t" | That tiny interval of time |
| $f(x,t)$ | "the drift" | The **predictable** part: where the system is being pushed |
| $g(t)$ | "the diffusion coefficient" | How **loud** the randomness is |
| $dw$ | "dee-w" | A tiny random kick — an increment of a **Wiener process** (Brownian motion) |

▸ **In one sentence: "over the next instant, move a bit in the direction $f$ says, and also get shoved by a random amount of size $g$."** Drift plus noise. That is all any SDE ever says. The ordinary differential equations you met in calculus are the special case $g = 0$.

**Compare it to the discrete step you already understand.** Chapter 20's forward process was

$$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon \;\approx\; x_{t-1} - \tfrac{1}{2}\beta_t x_{t-1} + \sqrt{\beta_t}\,\epsilon$$

using $\sqrt{1-\beta}\approx 1-\beta/2$ for small $\beta$. Rearranged: *"the change in $x$ is $-\frac{1}{2}\beta x$ plus noise of size $\sqrt{\beta}$."* Now read the VP-SDE line: $f = -\frac12\beta(t)x$, $g = \sqrt{\beta(t)}$. **They are character-for-character the same statement.** The SDE is not a new model. It is DDPM with the step size taken to zero.

▸ **Why $\sqrt{\beta}$ and not $\beta$ on the noise term — the one  non-obvious fact about Brownian motion.** Over a time interval $\Delta t$, drift accumulates proportionally to $\Delta t$ but *randomness accumulates proportionally to $\sqrt{\Delta t}$*, because independent random kicks partially cancel (Ch. 1 §1.3.1's $\sqrt{n}$ law again). Halve the step size and the drift per step halves, but the noise per step only drops by 30%. **This mismatch of exponents is the single reason stochastic calculus is different from ordinary calculus**, and it is why the two forms VP and VE, and the extra $\frac12$ in the probability-flow ODE below, all exist.

**VP versus VE, decoded.** These are two coordinate systems for the same geometry:

| | VP-SDE (= DDPM) | VE-SDE (= NCSN) |
|---|---|---|
| Drift $f$ | $-\frac12\beta(t)x$ — actively shrinks the signal | $0$ — leaves the signal alone |
| Diffusion $g$ | $\sqrt{\beta(t)}$ | $\sqrt{d\sigma^2(t)/dt}$ |
| Total variance | Pinned at 1 forever | Grows to something large ($\sigma_{\max}\sim 50$–$80$) |
| Endpoint | $\mathcal{N}(0, I)$ | $\mathcal{N}(x_0, \sigma_{\max}^2 I)\approx\mathcal{N}(0,\sigma_{\max}^2I)$ |

You can convert between them by rescaling $x$. **Neither is more correct**; VP keeps the network's inputs at a fixed scale, VE keeps the *data* untouched and lets the noise dwarf it. The 2022 EDM paper of Karras et al. made this explicit by writing down the general family and treating the choice as a hyperparameter like any other.

#### The reverse SDE, and why it is a  theorem

$$dx = \left[f(x,t) - g(t)^2\nabla_x\log p_t(x)\right]dt + g(t)\,d\bar w$$

Read the differences from the forward SDE:

- The drift gains a new term, $-g(t)^2\nabla_x\log p_t(x)$: **the score, scaled by the noise power.** Loud noise ⇒ strong correction needed.
- $d\bar w$ is a Brownian increment **running backwards in time**. The bar marks time-reversal, not a gradient (Ch. 0 §0.6's warning about bars applies to backprop, not here).
- Time flows from $T$ down to $0$, so $dt$ is negative.

▸ **The theorem says: the time-reversal of a diffusion process is itself a diffusion process, and the only extra thing you need to run it backwards is the score of the marginal at each time.** This is not obvious and it is not free — it is a real result about stochastic processes, and it is the mathematical permission slip for the entire chapter. Without it, "run diffusion backwards" would be a hope rather than a construction.

> **Where this came from.** The reverse-time result is due to **Brian D. O. Anderson**, an Australian control theorist, in a 1982 paper in *Stochastic Processes and their Applications* titled *Reverse-time diffusion equation models*. Anderson's field was systems and control theory — filtering, stability, and estimation — and the motivation was smoothing problems, where you want to infer a system's past state from later observations. It sat in the stochastic-processes literature for **thirty-nine years** before Yang Song and co-authors identified it, in 2021, as exactly the missing piece linking score estimation to generative sampling. **The reason diffusion models can be run in reverse at all is a control-theory theorem from 1982 about estimating the past.**

#### The probability-flow ODE — the most consequential equation in the chapter

$$\frac{dx}{dt} = f(x,t) - \frac12 g(t)^2\nabla_x\log p_t(x)$$

Two differences from the reverse SDE, and they are everything:

1. **The random term is gone.** No $d\bar w$. This is an ordinary differential equation — completely deterministic.
2. **The score coefficient is halved**: $\frac12 g^2$ rather than $g^2$.

▸ **The claim is that this deterministic equation has *exactly* the same distribution $p_t(x)$ at every time $t$ as the noisy one.** Not approximately. Exactly. Individual trajectories differ completely — the SDE's path is a jittery random walk, the ODE's is a smooth curve — but if you release a cloud of particles under either rule, the shape of the cloud at every instant is identical.

> **Analogy — smoke and traffic.** Watch smoke spread from a chimney. Each particle jitters unpredictably (the SDE). But the *plume* — the shape of the visible cloud — is smooth and predictable, and you could reproduce it exactly by moving particles along smooth streamlines with no randomness at all (the ODE), as long as you got the streamlines right. **The probability-flow ODE is the streamline description of the same cloud.** The halved coefficient is precisely the correction that compensates for removing the jitter: half the spreading was coming from the random kicks, so the deterministic drift must supply it instead.

**The three consequences, spelled out.**

**1. Deterministic sampling makes latents meaningful.** With the SDE, the same starting noise produces a different image every time — the randomness is injected all the way down. With the ODE, the map from $x_T$ to $x_0$ is a **function**: same input, same output, every time. That gives you an invertible correspondence between noise and images, which is what makes latent interpolation, image editing by inversion, and reproducible seeds possible. **Every "same seed, same image" guarantee in every image-generation UI rests on this.**

**2. Any ODE solver applies — and this is where the 50× speedup came from.** Once sampling is "integrate an ODE," you inherit a mature field. Euler's method (1768) is the crude one that DDIM reduces to; Heun's method, Runge–Kutta, and the exponential integrators behind DPM-Solver are all higher-order, meaning their error shrinks faster than linearly as you take more steps. **Numerical analysis had spent two centuries on this exact problem, and the reward for noticing was cutting sampling from 1000 evaluations to 20.** It is worth being explicit about the size of that win: nothing about the trained network changed. The same weights, sampled better, got fifty times faster.

**3. Exact likelihoods.** A deterministic invertible map lets you track how probability density is squeezed or stretched, via the instantaneous change-of-variables formula — so a diffusion model can report an exact log-likelihood after all, despite having been trained on a bound. It is a continuous normalizing flow (Ch. 19 §19.5) that was never trained as one.

### DDIM

The discrete non-Markovian family interpolating between stochastic and deterministic:

▸ $$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\underbrace{\left(\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted }x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta}_{\text{direction pointing to }x_t} + \sigma_t z$$

$\sigma_t=0$ gives **deterministic DDIM** — a first-order discretization of the probability-flow ODE. Crucially, it uses **the same trained network** and allows skipping timesteps: 20–50 steps for near-DDPM quality.

**Modern solvers:** DPM-Solver++ (exploits the semi-linear structure analytically; 10–20 steps), UniPC, Heun/Euler-ancestral (the "karras" samplers). ▸ **The insight enabling all of them:** the ODE is *semi-linear* — the linear part can be solved exactly, so the numerical error only comes from the nonlinear network term.

#### The DDIM update, decoded term by term

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\underbrace{\left(\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted }x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta}_{\text{direction pointing to }x_t} + \sigma_t z$$

It looks intimidating and it is doing something very simple. **Read it as: "guess the finished image, then re-noise it to the previous noise level."**

1. **Predict $x_0$.** The bracketed term is Tweedie's estimate from §20.6 — the network's best guess at the clean image, computed by algebra from its noise prediction.
2. **Re-apply exactly the amount of noise appropriate to level $t-1$.** Multiply the guess by $\sqrt{\bar\alpha_{t-1}}$ and add $\sqrt{1-\bar\alpha_{t-1}}$ worth of noise direction. Compare with §20.2's $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ — **it is the identical formula, at index $t-1$, with the network's guesses substituted for the unknowns.**
3. **Choose how much of the re-applied noise is fresh** ($\sigma_t z$) versus reused from the current step ($\epsilon_\theta$). That split is the $\sigma_t^2$ under the square root, and it is the one free parameter.

▸ **Set $\sigma_t = 0$ and the update is entirely deterministic** — the noise it re-applies is the noise the network just identified, not a new draw. That is deterministic DDIM.

**Why "non-Markovian" is the technically important word.** DDPM's forward process is a Markov chain: $x_t$ depends only on $x_{t-1}$. Song, Meng & Ermon's observation was that **the training objective $\mathcal{L}_{\text{simple}}$ only ever references the marginals $q(x_t\mid x_0)$ — it never uses the chain structure at all.** So you are free to invent a *different* forward process, one that isn't a Markov chain, as long as it has the same marginals. Every such process gives a valid sampler, and they all work with the network you already trained.

▸ **This is why DDIM required no retraining and could be applied retroactively to every diffusion model in existence.** It is not a new model; it is the discovery that the trained network was compatible with a whole family of samplers nobody had noticed. **The 1000 steps were never in the weights.**

**Why skipping steps becomes legal.** Because the update only references $\bar\alpha_t$ and $\bar\alpha_{t-1}$, and both are just table lookups, nothing stops you from jumping from $t=1000$ straight to $t=950$: substitute $\bar\alpha_{950}$ for $\bar\alpha_{t-1}$ and the formula is still exactly the same shape. A 50-step DDIM sampler is the 1000-step schedule evaluated at every 20th index. With the stochastic DDPM sampler you cannot do this, because each step's added noise is calibrated to a single-step gap.

**"Semi-linear," decoded.** The probability-flow ODE has the form $\frac{dx}{dt} = \underbrace{a(t)x}_{\text{linear in }x} + \underbrace{b(t)\,\epsilon_\theta(x,t)}_{\text{the network}}$. The first term is an ordinary linear ODE and has an exact closed-form solution (an exponential). Only the second is intractable. **DPM-Solver's insight is to solve the easy half exactly and only approximate the hard half** — so the discretization error comes from one term instead of two.

> **Analogy.** You are integrating "constant headwind plus unpredictable gusts." Naively you'd approximate both numerically. But the headwind's effect is available in closed form, so approximate only the gusts. Your error is now the gust error alone, which at 15 steps is smaller than the combined error at 100.

> **Where this came from.** **DDIM** — *Denoising Diffusion Implicit Models* — is by **Jiaming Song, Chenlin Meng, and Stefano Ermon** at Stanford, published at ICLR 2021, less than a year after DDPM. The paper's contribution was almost entirely conceptual: it did not improve the model, it improved the *question*, by noticing that the objective constrains only the marginals. **DPM-Solver** (Cheng Lu and co-authors, 2022) then supplied the numerical-analysis machinery. The sequence from 1000 steps (2020) to 20 steps (2022) to 1–4 steps (2023, via distillation) happened almost entirely at sampling time, which is a useful lesson: **when a method is slow, check whether the slowness is in the model or merely in how you are querying it.**

---

## 20.8 Guidance

### Classifier guidance

Bayes: $\nabla_x\log p(x\mid y) = \nabla_x\log p(x) + \nabla_x\log p(y\mid x)$. Add a scaled classifier gradient:

$$\hat\epsilon = \epsilon_\theta(x_t,t) - w\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)$$

**Requires a separate classifier trained on noisy inputs.** Awkward.

#### Classifier guidance, decoded

Start with the Bayes identity, which is the load-bearing line:

$$\nabla_x\log p(x\mid y) = \nabla_x\log p(x) + \nabla_x\log p(y\mid x)$$

**Where it comes from.** Bayes' rule says $p(x\mid y) = p(x)p(y\mid x)/p(y)$. Take logs: $\log p(x\mid y) = \log p(x) + \log p(y\mid x) - \log p(y)$. Now differentiate **with respect to $x$** — and $\log p(y)$ doesn't contain $x$, so it disappears. Same disappearing act as the partition function in §20.6, same reason.

Read the surviving equation in English:

> **"The direction that makes an image more plausible *given the prompt* = the direction that makes it more plausible *at all* + the direction that makes the prompt more likely to describe it."**

| Term | What it knows | Who supplies it |
|---|---|---|
| $\nabla_x\log p(x)$ | "What do real images look like?" | The unconditional diffusion model |
| $\nabla_x\log p(y\mid x)$ | "Does this image look like a *dog*?" | A separate classifier |
| $\nabla_x\log p(x\mid y)$ | "What do real images **of dogs** look like?" | The sum |

▸ **This decomposition is the whole idea of guidance: realism and relevance are separable, and you can turn up one without retraining the other.** The classifier's gradient is the answer to "which pixels would I change to make this classifier more confident it's a dog?" — the same computation as an adversarial attack (Ch. 1 §1.1.4), pointed at generation rather than at breaking things.

**The scale $w$ is where it stops being Bayes.** The formula that gets implemented multiplies the classifier term by $w$:

$$\hat\epsilon = \epsilon_\theta(x_t,t) - w\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)$$

The $\sqrt{1-\bar\alpha_t}$ is the units conversion from §20.6 — scores and $\epsilon$'s differ by exactly that factor — and the minus sign is the same one. With $w=1$ this is exactly Bayes. With $w>1$ it is **not sampling from any conditional distribution you asked for**; it is sampling from something proportional to $p(x)p(y\mid x)^w$, a sharpened version. That is a deliberate distortion, and §20.8's trade-off discussion is the accounting for it.

**Why "requires a classifier trained on noisy inputs" is  awkward.** Your classifier must handle $x_t$ at every noise level — an off-the-shelf ImageNet model is useless, since it has never seen an image that is 90% static. So you must train a second network, on noisy data, at every $t$, and keep it in sync with the diffusion model. Worse, classifier gradients at high noise are notoriously unreliable: the classifier finds adversarial directions that increase its confidence without making the image more dog-like at all.

> **Where this came from.** Classifier guidance was introduced by **Prafulla Dhariwal and Alex Nichol** at OpenAI in 2021, in the paper whose title was itself the claim: *Diffusion Models Beat GANs on Image Synthesis*. That paper is the moment the field's centre of gravity moved. It combined architectural improvements to the U-Net with classifier guidance and, for the first time, beat the best GANs on ImageNet generation by FID. **Classifier-free guidance, published within months, then made the classifier unnecessary — but the ordering matters: the awkward version had to work first, to show that the fidelity/diversity dial existed at all.**

### Classifier-free guidance — the one that matters

Train **one** network on both conditional and unconditional objectives by randomly dropping the condition (replacing $y$ with a null token) ~10% of the time. Then at sampling:

▸ $$\boxed{\ \tilde\epsilon_\theta(x_t,t,y) = \epsilon_\theta(x_t,t,\varnothing) + w\big[\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)\big]\ }$$

**Derivation of why this is the right form:** an implicit classifier satisfies $\nabla\log p(y|x)=\nabla\log p(x|y)-\nabla\log p(x)$. Substituting into the classifier-guidance formula and converting scores to $\epsilon$ gives exactly the above. So CFG is classifier guidance with the classifier expressed through the generative model itself.

- $w=1$: ordinary conditional sampling.
- $w>1$: **extrapolation beyond the conditional distribution** — pushes toward regions where the condition is *unusually* likely.

▸ **The trade-off, stated precisely:** higher $w$ increases prompt adherence and per-sample fidelity while **reducing diversity** and eventually producing oversaturated, over-contrasted artifacts. It is a fidelity/diversity dial, and it is the reason a text-to-image model's outputs at $w=15$ look "AI-generated." Typical: $w=7.5$ for images, $2$–$4$ for video, lower for models trained with CFG-distillation.

**Cost:** two forward passes per step. **CFG distillation** trains a student to match the guided output in one pass.

**Refinements:** dynamic thresholding (Imagen — rescale to avoid saturation at high $w$), interval guidance (apply CFG only in the middle of the trajectory, where it matters), and per-timestep $w$ schedules.

#### Classifier-free guidance, decoded — the most important formula you will actually use

$$\tilde\epsilon_\theta(x_t,t,y) = \epsilon_\theta(x_t,t,\varnothing) + w\big[\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)\big]$$

**Symbols first.**

| Piece | Read aloud | Meaning |
|---|---|---|
| $\varnothing$ | "null" / "the empty set" | The **absence of a prompt.** In code, a special learned embedding, or the empty string |
| $\epsilon_\theta(x_t,t,\varnothing)$ | "epsilon-theta, unconditional" | What the network predicts **with no prompt** — "make it look like *any* image" |
| $\epsilon_\theta(x_t,t,y)$ | "epsilon-theta, conditional" | What it predicts **given the prompt** — "make it look like *this*" |
| $\tilde\epsilon_\theta$ | "epsilon-theta-tilde" | The **guided** prediction actually used by the sampler |
| $w$ | "the guidance scale" | How hard to push. Typically 7.5 for images |

**Now the algebra, which is a single familiar operation.** Set $a = \epsilon_\varnothing$ and $b = \epsilon_y$. The formula is

$$\tilde\epsilon = a + w(b-a)$$

▸ **That is linear interpolation — the `lerp` you have written a hundred times — except that $w$ is allowed to exceed 1, which makes it *extra*polation.** At $w=0$ you get $a$ (ignore the prompt). At $w=1$ you get $b$ (ordinary conditional generation). **At $w=7.5$ you get a point 7.5 times further along the arrow from "generic image" to "image matching the prompt" — past the endpoint, out beyond where the model's own conditional distribution lives.**

> **Analogy — the caricature.** An artist draws an average face and a portrait of a specific person. Subtract one from the other and you get the direction "what makes *this* person distinctive": a longer nose, a wider jaw. Now add that difference back **seven times**. You get a caricature — unmistakably that person, more identifiable than the real portrait, and visibly not a photograph. **CFG is caricature applied to a noise prediction.** It explains the whole trade-off in one image: caricatures are *more* recognizable and *less* real, and every caricature of a given person looks somewhat like every other one.

**Working it with real numbers.** Set $d=1$ so all quantities are scalars. Suppose at some step the unconditional prediction is $\epsilon_\varnothing = 0.30$ and the conditional is $\epsilon_y = 0.34$ — the prompt shifts the noise estimate by $0.04$.

| $w$ | $\tilde\epsilon = 0.30 + w(0.04)$ | Effect |
|---|---|---|
| $0$ | $0.30$ | Prompt ignored entirely |
| $1$ | $0.34$ | Honest conditional sampling |
| $3$ | $0.42$ | Noticeably stronger adherence |
| $7.5$ | $0.60$ | Standard for image models — **twice the unconditional value** |
| $15$ | $0.90$ | Oversaturated, high-contrast, "AI-looking" |

▸ **At $w = 7.5$, a difference of $0.04$ has become a difference of $0.30$ — the prompt's influence has been amplified 7.5-fold, and it is applied at every one of the sampling steps.** Effects compound down the trajectory, which is why guidance is far more powerful than the single-step arithmetic suggests, and why pushing $w$ too high produces the characteristic blown-out look: the pixel values are being pushed past the valid range and clipped, step after step.

#### Why the derivation works — the implicit classifier

The text's justification is three lines; here it is slowly.

**Step 1.** Rearrange the Bayes identity to isolate the classifier's contribution:

$$\nabla\log p(y\mid x) = \nabla\log p(x\mid y) - \nabla\log p(x)$$

*"How much this image looks like a dog, as a direction, equals (what dog-images look like) minus (what images look like)."* **The classifier has been eliminated.** Both terms on the right are things a conditional diffusion model can produce — the first by feeding it the prompt, the second by feeding it $\varnothing$.

**Step 2.** Substitute that into the classifier-guidance formula and convert scores to $\epsilon$'s using $s = -\epsilon/\sqrt{1-\bar\alpha_t}$. The square roots cancel throughout, and out drops exactly the boxed CFG formula.

▸ **So CFG is classifier guidance with the classifier replaced by a difference of two generative predictions.** No separate network, no noisy-classifier training, no adversarial gradients. **The model classifies itself, implicitly, by the difference between what it does with and without the prompt.**

**Why the training procedure is so cheap.** To get the $\varnothing$ branch you do not train a second model: during training, **replace the prompt with the null token about 10% of the time.** That is a one-line change to the data pipeline. The same weights learn both jobs, and because they share everything except which conditioning vector arrives, the difference $\epsilon_y - \epsilon_\varnothing$ isolates exactly the prompt's effect and nothing else.

**Why 10%?** It is a bias/variance trade on the two branches. Too low (say 1%) and the unconditional branch is undertrained, so $\epsilon_\varnothing$ is a poor estimate and the difference is noisy. Too high (say 50%) and half your compute goes to a branch that only appears as a *reference point* in the final formula. **10–20% is the standard range, and it is one of the most robust hyperparameters in the field** — models are not very sensitive to it, which is itself worth knowing.

#### The trade-off, made concrete

$w>1$ samples from something proportional to $p(x)\,p(y\mid x)^w$ — the conditional distribution raised to a power, which **sharpens** it. Raising a probability distribution to a power $w>1$ makes likely things much likelier and unlikely things vanish; it is the same operation as lowering the temperature of a softmax (Ch. 1 §1.3.4), and it has the same consequences.

| As $w$ increases | Goes up | Goes down |
|---|---|---|
| Prompt adherence | ✓ | |
| Per-sample fidelity / "polish" | ✓ | |
| Contrast and saturation | ✓ (eventually too far) | |
| Diversity across seeds | | ✓ |
| Coverage of the true conditional distribution | | ✓ |
| FID (which penalizes lost diversity) | | ✓ then ✗ — it improves, then degrades |

▸ **This is a fidelity/diversity dial, and there is no setting that is best on all metrics.** FID, which measures distributional match, typically prefers $w\approx 2$–$3$. Human raters, who look at one picture at a time and cannot see the diversity they are missing, typically prefer $w\approx 7$–$10$. **The gap between those two numbers is the gap between "matches the data distribution" and "looks good," and it is the cleanest example in this book of a metric and a preference disagreeing.**

**Why the artifacts appear.** At high $w$, $\tilde\epsilon$ becomes large, so the predicted $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\tilde\epsilon)/\sqrt{\bar\alpha_t}$ swings outside the valid pixel range $[-1,1]$. Clipping it flattens the extremes, and repeating that a thousand times bakes in the flat, over-contrasted look. **Imagen's dynamic thresholding** is the fix: instead of clipping at a fixed value, compute a high percentile (say the 99.5th) of $|\hat x_0|$ at each step and rescale the whole image by it — so the extremes are pulled in *proportionally* rather than chopped off.

**Why interval guidance works.** Guidance matters most in the middle of the trajectory, where global composition is being decided. At very high $t$ everything is noise and the conditional and unconditional predictions barely differ; at very low $t$ the image is essentially determined and guidance only adds contrast artifacts. **Applying CFG only for $t$ in the middle band buys most of the adherence at a fraction of the cost and with fewer artifacts** — and since CFG is what doubles the sampling cost, skipping it at either end is nearly free speed.

> **Where this came from.** Classifier-free guidance is by **Jonathan Ho and Tim Salimans**, and it first appeared as a **workshop paper** at NeurIPS 2021 — not a main-conference paper, a short workshop note. It is now used in essentially every text-to-image, text-to-video, and text-to-audio diffusion system deployed anywhere, and the parameter it introduced is the single most-adjusted knob in generative AI: the "CFG scale" or "guidance" slider in every image-generation interface is $w$. **One of the highest-impact ideas in modern machine learning was published in four pages at a workshop.** It is a useful corrective to the assumption that impact tracks venue.

#### Examples and non-examples: classifier-free guidance

**✅  classifier-free guidance**

| Example | Why it qualifies |
|---|---|
| $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_y - \epsilon_\varnothing)$ with $w=7.5$ | Extrapolates *past* the conditional prediction, along the direction the prompt is responsible for |
| Dropping the prompt to $\varnothing$ on 10% of training examples | The one change that lets a single network serve both branches |
| Two U-Net evaluations per sampling step, usually batched into one call | The literal arithmetic cost of needing both $\epsilon_y$ and $\epsilon_\varnothing$ |
| Applying guidance only for $t$ in a middle band | Still CFG — the formula is unchanged, just not applied everywhere |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| $w = 1$ | The formula collapses to $\tilde\epsilon = \epsilon_y$; the correction term is exactly zero | Ordinary **conditional** sampling, with no guidance at all |
| A negative prompt | $\epsilon_\varnothing$ is replaced by $\epsilon_{y^-}$, a second *conditional* prediction | Guidance **away from** a rival prompt — a generalization, not the same object |
| Classifier guidance (earlier in §20.8) | Needs a separate classifier trained on **noisy** images and its input gradients | The 2021 predecessor that CFG displaced |
| Reweighting tokens in the prompt | Changes the conditioning vector $y$, never the score | Prompt-side manipulation |
| Turning $w$ up to make a model obey a concept it never learned | Sharpening a wrong conditional produces confident wrongness | A training-data problem wearing a sampling costume |
| Lowering the softmax temperature of a text encoder | Operates on the encoder's output distribution, not on $\epsilon_\theta$ | Conditioning-side sharpening |

▸ **The boundary:** classifier-free guidance is an **extrapolation between two predictions from the same weights**. No unconditional branch, or no subtraction, means it is not CFG.

> **Common misconception.** *"Classifier-free guidance is free — it's in the name."* The word "free" refers to the **classifier**: you no longer train a separate noisy-image classifier the way classifier guidance required. The guidance itself is the opposite of free. Every sampling step now evaluates $\epsilon_\theta(x_t,t,y)$ **and** $\epsilon_\theta(x_t,t,\varnothing)$ — two full forward passes of a large U-Net, so **sampling costs exactly $2\times$.** A 50-step generation with CFG is 100 network evaluations, not 50. That factor of two is why interval guidance (skipping CFG where it barely matters, at very high and very low $t$) is such attractive, nearly-free speed, and why this one word has caused more confusion than anything else in the chapter. The misconception is tempting because in almost every other context in this field, "free" is a claim about compute.

> **Common misconception.** *"Higher guidance means the model understands the prompt better."* Guidance adds no understanding whatsoever; it **sharpens whatever conditional the model already has**, sampling from something proportional to $p(x)\,p(y\mid x)^w$. If the model never learned what a sextant looks like, moving $w$ from 7 to 20 yields a more saturated, more confident, less diverse picture of the wrong object, plus the clipping artifacts of §20.8. The belief is tempting because for prompts the model *does* know, raising $w$ visibly improves adherence — and it is natural, having watched that work three times, to assume the knob supplies comprehension rather than merely amplifying it.

---

## 20.9 Making it fast

| Method | Steps | Mechanism |
|---|---|---|
| DDPM | 1000 | ancestral sampling |
| DDIM | 50 | deterministic, skip steps |
| DPM-Solver++ | 10–20 | high-order ODE solver |
| **Progressive distillation** | 4–8 | student learns to take 2 teacher steps at once; halve repeatedly |
| **Consistency models** | **1–4** | train $f_\theta(x_t,t)\to x_0$ to be *constant along an ODE trajectory*: $f(x_t,t)=f(x_{t'},t')$ |
| **LCM / LCM-LoRA** | 2–8 | consistency distillation in latent space, as a LoRA adapter |
| **Adversarial distillation** (SDXL-Turbo, ADD) | 1–4 | add a GAN loss to the distillation objective |

▸ **Consistency models are the conceptually cleanest:** enforce the *self-consistency* property that every point on the same ODE trajectory maps to the same origin. Trained either by distillation from a diffusion model or from scratch. Single-step generation with quality approaching multi-step diffusion.

#### The acceleration table, decoded

The table's three columns are "how many network evaluations," "which trick," and implicitly "what did it cost you." Read it as a progression of increasingly aggressive bargains.

**Progressive distillation, unpacked.** Train a **student** network to reproduce, in **one** step, what the **teacher** does in **two**. Then treat the student as the new teacher and repeat.

$$1000 \to 500 \to 250 \to 125 \to 62 \to 31 \to 16 \to 8 \to 4$$

▸ **Eight rounds of halving takes 1000 steps to 4.** Each round is a full training run, but each round's task is easy — matching a teacher you can query as often as you like, on inputs you generate yourself. There is no data required beyond noise. **This is the same halving trick as binary search or fast exponentiation: an exponential reduction bought with a logarithmic number of rounds.**

> **Analogy — memorizing a route.** The first time you drive somewhere you follow a thousand turn-by-turn instructions. After a few trips you have chunked them: "get on the motorway, exit 14, second left." You have not learned a different route; you have compressed the same route into fewer decisions. **Distillation is chunking.** And like chunking, it works because the route was largely predictable — most of the thousand instructions were "keep going straight."

**Consistency models, unpacked.** The condition is $f_\theta(x_t,t) = f_\theta(x_{t'},t')$ whenever $x_t$ and $x_{t'}$ lie on the same probability-flow ODE trajectory. In English:

> **"Every point along a single noise-to-image path must map to the same final image."**

The ODE trajectory is a curve through space that starts at pure noise and ends at one specific picture. A diffusion sampler *walks* that curve. A consistency model **learns the curve's destination as a function of any point on it** — so you can jump straight from the start to the end.

> **Analogy — the river.** A diffusion sampler is a leaf drifting down a river, taking a thousand small movements to reach the sea. A consistency model is a map that, given your position anywhere on the river, tells you which river mouth you will come out of. Consult it once at the source and you are done. **The multi-step sampler is still available if you want it** — you can jump partway, add noise back, and jump again, which is where the "1–4 steps" range comes from: a few jumps beat one, because each correction re-anchors on real geometry.

▸ **Why the self-consistency condition is enough to train on.** It is a *constraint*, not a target — you never need to know the right answer, only that two things must agree. Train by picking adjacent points on a trajectory and penalizing disagreement, anchored at $t=0$ where the answer is trivially the image itself. **That is why consistency models can be trained from scratch, without a teacher diffusion model at all**, which was the surprising part of the result.

**What every acceleration method costs.** No entry in that table is free, and it is worth naming the currency:

| Method | Price paid |
|---|---|
| DDIM / DPM-Solver | Discretization error — small at 20 steps, visible below 10 |
| Progressive distillation | Many training runs; each halving loses a little quality |
| Consistency / LCM | Reduced diversity; slightly softer detail |
| Adversarial distillation (ADD / SDXL-Turbo) | GAN instability returns, and mode coverage narrows measurably |

▸ **The pattern across all of them: you trade *distributional coverage* for *speed*.** A 1000-step sampler explores the conditional distribution; a 1-step sampler produces its most confident answer. This is the same axis as the CFG dial in §20.8, and for a related reason — both are ways of spending diversity to buy something you can see in a single sample.

> **Where this came from.** **Progressive distillation** is Salimans & Ho (2022), the same paper that introduced $v$-prediction — the two ideas are linked, because distillation is what exposed $\epsilon$-prediction's numerical failure at widely-spaced timesteps. **Consistency models** are Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever (OpenAI, 2023). **Latent Consistency Models and LCM-LoRA** then made the idea practical for the open-source ecosystem by packaging consistency distillation as a small LoRA adapter (Ch. 17), so a user could add few-step sampling to a model they already had, downloading a few dozen megabytes rather than retraining anything. **That packaging decision — shipping a capability as an adapter rather than a model — did more for adoption than the underlying research.**

---

## 20.10 Flow matching — the modern reformulation

### The one-line idea

Skip the noise-schedule machinery entirely: define a straight path from noise to data and regress the velocity along it.

**Construction.** Take the linear interpolation
$$x_t = (1-t)x_0^{\text{noise}} + t\,x_1^{\text{data}},\qquad t\in[0,1]$$
The velocity along this path is simply $u_t = x_1 - x_0$. Train:

▸ $$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,x_0,x_1}\big\|v_\theta(x_t,t) - (x_1-x_0)\big\|^2$$

Sample by integrating $\frac{dx}{dt}=v_\theta(x,t)$ from $t=0$ to $1$.

▸ **Why it's better in practice:**
- **Straight paths.** Diffusion's probability-flow trajectories are curved, so a large ODE step incurs large discretization error. Linear paths need fewer steps — often 10–20 without distillation.
- **No noise schedule to tune.** The interpolation *is* the schedule.
- Simpler theory: it is a continuous normalizing flow (Ch. 19 §19.5) trained by regression instead of by likelihood.

**Rectified flow** iterates: generate pairs with the trained model, retrain on those pairs, and the paths straighten further. "Reflow" can reach  one-step generation.

**Status:** Stable Diffusion 3, Flux, and most 2024+ image and video models use rectified flow / flow matching rather than DDPM. ▸ **Conceptually, flow matching and diffusion are the same family** — different paths through the same space of interpolations between noise and data — and the DDPM schedule is one particular (curved) choice.

#### Flow matching, decoded

$$x_t = (1-t)\,x_0^{\text{noise}} + t\,x_1^{\text{data}},\qquad t\in[0,1]$$

**Note the reversed convention** — this is a real trap when reading code. In flow-matching papers $t=0$ is **noise** and $t=1$ is **data**, the opposite of DDPM's convention where $t=0$ is the clean image. Both appear in the literature and libraries mix them. Always check which end is which before trusting a snippet.

The formula itself is the most ordinary thing in the chapter: **a straight-line blend between two points.** At $t=0.5$ you are exactly halfway between a noise sample and an image. That's it. Compare with DDPM's blend, $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, whose coefficients trace a *quarter-circle* (they square to one) rather than a straight line.

**The velocity, decoded.** Differentiate the path with respect to $t$:

$$u_t = \frac{d x_t}{dt} = \frac{d}{dt}\big[(1-t)x_0 + t x_1\big] = -x_0 + x_1 = x_1 - x_0$$

▸ **The velocity along a straight line is constant — it does not depend on $t$ at all.** That is the entire appeal. The training target is a fixed vector, "the arrow from this noise sample to this image," and the network's job is to predict it from an intermediate point.

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,x_0,x_1}\big\|v_\theta(x_t,t) - (x_1-x_0)\big\|^2$$

Read: *"draw a random noise vector and a random image, pick a random point on the line between them, and ask the network which way the line was going."* Four lines of code, same as $\mathcal{L}_{\text{simple}}$, no $\bar\alpha$ table.

> **Analogy — flight paths.** Diffusion's trajectory from noise to data is a curved flight path; to follow it accurately with a small number of waypoints you need many waypoints, because a straight hop between two distant points on a curve cuts a lot of corner. Flow matching **builds the runway in a straight line**, so two waypoints suffice in principle. **The curvature *is* the discretization error**, and rectified flow is the procedure for ironing it out.

#### Why straightness buys steps — with a number

Take an ODE trajectory and approximate it by $N$ straight segments (Euler's method). The error of that approximation scales with the path's **curvature** and with the square of the step size. For a perfectly straight path, curvature is zero and **one step is exact**.

DDPM's probability-flow trajectories are noticeably curved because the noise schedule's coefficients trace a circular arc. Flow matching's are straight *for each individual pair* $(x_0, x_1)$ — though the *learned* average velocity field is not perfectly straight, because many different noise vectors must be routed to many different images and the paths cross. **Rectified flow's "reflow" procedure fixes exactly this:** generate pairs (noise, image) using the trained model, then retrain on those pairs. Now each noise vector is paired with *the image it actually produces*, the paths stop crossing, and the field straightens. Iterate and you approach  one-step generation.

▸ **The lesson is that "the model is slow" was, once again, not a fact about the model but about the geometry of the path it was asked to follow.** Chapter 20's whole efficiency story — DDIM, solvers, distillation, flow matching — is variations on that one realization.

**"No noise schedule to tune."** In DDPM you must choose $\beta_1,\dots,\beta_T$; the choice matters (linear versus cosine is a visible quality difference), it interacts with resolution (§20.12), and it has a subtle bug at the endpoint (zero terminal SNR). In flow matching the schedule is $t$ itself, uniform on $[0,1]$. **An entire category of hyperparameter disappears** — although a $t$-sampling distribution (logit-normal weighting, as in SD3) reappears in its place, so the honest statement is that the tuning became simpler, not that it vanished.

> **Where this came from.** Flow matching arrived from **three independent groups within about a month of each other in late 2022**: **Yaron Lipman and co-authors** at Meta AI (*Flow Matching for Generative Modeling*), **Xingchao Liu, Chengyue Gong, and Qiang Liu** at UT Austin (*Rectified Flow*), and **Michael Albergo and Eric Vanden-Eijnden** at NYU (*stochastic interpolants*). Three vocabularies — optimal transport, ODE straightening, and statistical mechanics — one algorithm. **This is the third independent-rediscovery story in this chapter alone** (after SVD-style convergences in Ch. 1, and DDPM/NCSN in §20.6), and by now the pattern should be readable as a signal rather than a coincidence: when a formulation is the natural one, several people find it at once.

---

## 20.11 Latent diffusion

▸ Run diffusion in the latent space of a pretrained autoencoder rather than in pixel space.

- VAE encoder: $512\times512\times3 \to 64\times64\times4$. **A 48× reduction in elements**, so $\sim$48× less compute per step, and attention becomes affordable.
- The autoencoder handles imperceptible high-frequency detail (a *perceptual compression* stage); the diffusion model handles semantics (the *semantic compression* stage). Splitting these is the whole insight.
- Trained with a combination of $L_1$/perceptual (LPIPS) loss, a small KL or VQ regularizer, and a patch-GAN discriminator to keep decoded images sharp.

**This is Stable Diffusion**, and it is why consumer-GPU image generation exists.

**Caveat:** the autoencoder is a hard ceiling. Fine text and small faces are lost at encode time and no amount of diffusion training recovers them. Hence the trend toward higher-channel latents (4→16) in SD3/Flux.

#### Latent diffusion, decoded — where the 48× comes from

Count the numbers.

$$\underbrace{512\times512\times3}_{786{,}432\ \text{pixel values}} \quad\longrightarrow\quad \underbrace{64\times64\times4}_{16{,}384\ \text{latent values}}$$

$$\frac{786{,}432}{16{,}384} = 48$$

▸ **Forty-eight times fewer numbers to denoise, at every one of a thousand steps.** Diffusion's cost is roughly linear in the number of elements for the convolutional parts — so that alone is a 48× saving — but the win is far larger for attention, whose cost is **quadratic** in the number of positions (Ch. 0 §0.10). Attention over $512^2 = 262{,}144$ spatial positions is $6.9\times10^{10}$ pairs and simply cannot be run. Attention over $64^2 = 4{,}096$ positions is $1.7\times10^{7}$ pairs — ordinary. **Latent diffusion is what made attention affordable inside an image generator, and attention is what made text conditioning work.**

#### The two-stage compression argument

This is the conceptual core, and it is worth stating carefully because it generalizes far beyond images.

**Not all bits in an image are equally meaningful.** The exact value of one pixel of grass texture is nearly random and carries essentially no information a viewer would notice. The presence of a face carries an enormous amount. Yet a pixel-space diffusion model spends its capacity on both equally — indeed disproportionately on the texture, since there is so much more of it (§20.4).

So split the job in two:

| Stage | Handled by | Job | Trained with |
|---|---|---|---|
| **Perceptual compression** | The autoencoder | Throw away detail humans cannot see | $L_1$ + LPIPS + a small KL or VQ term + patch-GAN |
| **Semantic compression** | The diffusion model | Decide *what is in the picture* | $\mathcal{L}_{\text{simple}}$ in latent space |

> **Analogy — JPEG, then composition.** A photographer does not think about the individual bytes of a JPEG file; the codec handles the imperceptible parts and the photographer handles what's in the frame. **Latent diffusion hires a codec so the generative model can be a photographer.** And the analogy is closer than it sounds: the autoencoder's job description — discard what the eye won't miss — is exactly JPEG's, just learned rather than hand-designed with a discrete cosine transform.

**Why that grab-bag of losses.** Each term prevents a specific failure of the others:

- **$L_1$** (rather than $L_2$) — plain squared error on pixels produces blurry reconstructions, because it prefers averaging over committing. $L_1$ is a bit sharper.
- **LPIPS** (Learned Perceptual Image Patch Similarity) — compares deep features rather than pixels, so a texture that is *statistically* right but not pixel-aligned is not punished.
- **A small KL or VQ regularizer** — keeps the latent space from becoming an arbitrary high-variance code. The diffusion model has to add Gaussian noise to these latents, so they must be roughly standardized. **This is the term that makes the two stages compatible.**
- **A patch-GAN discriminator** — the only one of the four that can enforce "this looks like a real photograph at the texture level." Without it, decoded images are subtly soft.

▸ **The caveat deserves emphasis because it explains a failure everyone has seen.** The autoencoder is a lossy codec applied *before* the diffusion model ever runs, so **anything it destroys is gone permanently** — no amount of diffusion training, guidance, or step count brings it back. Small text becomes illegible squiggles, distant faces become smears, and fine repeated patterns (chain-link fence, brick, text on signage) alias. The fix is more latent channels: 4 → 16 in SD3 and Flux, which quadruples the information budget per latent position and visibly improves text rendering, at the cost of a larger and slower diffusion model.

> **Where this came from.** Latent diffusion is by **Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer** — the CompVis group at LMU Munich — published in 2022. The paper's argument is exactly the two-stage one above. Its consequence was that a state-of-the-art text-to-image model became runnable on a consumer graphics card with a few gigabytes of memory rather than on a cluster, and when the weights were released publicly later that year as **Stable Diffusion**, the effect on the field was immediate: an entire ecosystem of fine-tunes, LoRA adapters, ControlNets, and interfaces appeared within months. **The research contribution was a compute reduction; the historical contribution was that the reduction crossed the threshold of what an individual could run.**

#### Examples and non-examples: latent diffusion

**✅  latent diffusion**

| Example | Why it qualifies |
|---|---|
| Stable Diffusion: a $512\times512\times3$ image becomes a $64\times64\times4$ code, and all 50 denoising steps run on that code | The diffusion process never touches a pixel |
| The decoder running **once**, after the last sampling step | Decoding is not part of the trajectory; it is the exit door |
| A latent kept near unit variance by the KL or VQ term | Adding $\mathcal{N}(0,I)$ noise is only meaningful on a standardized code |
| Latent consistency models (LCM) | Distillation applied inside the same code space |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Imagen / DALL·E 2 style cascades: 64px diffusion, then a 256px diffusion upsampler | Every stage runs in **pixel** space; the small stage is a small *image*, not a code | **Cascaded** pixel diffusion |
| Bicubically downscaling to 64px and diffusing that | A resize is not a learned perceptual codec — it discards by frequency, not by what the eye misses | Low-resolution pixel diffusion |
| Decoding to pixels each step to watch progress | A visualization hook, not part of the algorithm | A preview |
| "The latent is a tiny thumbnail" | Its four channels are not red, green, blue and a spare; they are an arbitrary learned basis | A code only the decoder can read |
| ControlNet, LoRA, textual inversion | They modify conditioning or weights, never the space the process lives in | Adapters *on top of* latent diffusion |
| A VQ-VAE plus an autoregressive prior (DALL·E 1) | Same two-stage idea, but the second stage predicts tokens left to right | Autoregressive modelling in a latent space |

▸ **The boundary:** in latent diffusion the noise, the schedule, the network, and every sampling step all live in the autoencoder's code space — pixels appear exactly twice in the whole pipeline, at encode time during training and at decode time at the very end.

> **Common misconception.** *"Latent diffusion is just diffusion on a smaller image."* The latent is not an image. Plot its channels directly and you get false-colour mush, because they are a learned basis with no colour interpretation. More importantly the compression is **perceptual, not spatial**: the autoencoder was trained with LPIPS and a patch discriminator specifically to throw away what a viewer will not miss, which a bicubic downscale does not do and cannot do. **And the $48\times$ reduction comes with a hard, permanent information ceiling** — whatever the encoder discards is gone before the diffusion model runs, which is why small text and distant faces degrade in ways that no step count, guidance scale, or extra training ever repairs. The belief is tempting because the latent really is spatially arranged at $64\times64$, and plotting its first three channels really does produce something that looks like a smudged version of the picture.

> **Common misconception.** *"The autoencoder is just an efficiency trick — a bigger diffusion model in pixel space would do strictly better."* On quality-per-unit-of-compute, no. A pixel-space model spends the overwhelming majority of its capacity on high-frequency texture, because that is where almost all the *bits* in an image live (§20.4) — and that is the part viewers barely register. **The autoencoder does not merely shrink the problem; it removes the part of the problem the diffusion model is worst at spending capacity on**, which is why the two-stage split improves the compute/quality frontier rather than just sliding along it. The belief is tempting because compression is lossy, and lossy usually means worse — here the loss is deliberately aimed at the bits nobody looks at.

---

## 20.12 Practical notes

- **Noise schedule.** Linear $\beta$ from $10^{-4}$ to $0.02$ (original); **cosine** $\bar\alpha_t = \frac{\cos^2\left(\frac{t/T+s}{1+s}\cdot\frac\pi2\right)}{\cos^2\left(\frac{s}{1+s}\cdot\frac\pi2\right)}$ with $s=0.008$ destroys information more gradually and is better. At higher resolutions the schedule must be **shifted** toward more noise, because neighbouring pixels are more redundant so a given $\beta$ destroys less information.
- ▸ **EMA of the weights is mandatory**, not optional. $\gamma\approx0.9999$. Sample quality with and without EMA is not close.
- **Timestep embedding:** sinusoidal features of $t$ through a small MLP, injected into every block (via AdaGN/AdaLN — Ch. 21).
- **Offset noise / zero-terminal-SNR:** standard schedules never reach $\bar\alpha_T=0$, so the model never sees pure noise and cannot generate very dark or very bright images. Fixing the schedule to reach zero terminal SNR resolves it.
- **Evaluation:** FID, CLIP score, and human preference — with Chapter 3's warnings about sample-size dependence and Chapter 19's about FID's flaws.

#### The practical notes, decoded

Each of these is terse and each hides a real story. In order.

**The noise schedule, and why cosine beat linear.** The schedule is the function $t\mapsto\bar\alpha_t$ — "how much signal survives by step $t$." The original linear-$\beta$ schedule destroys information *too fast in the middle*: by $t=600$ the image is already unrecognizable, so the last 400 steps are spent denoising something that carries almost no information. **The cosine schedule spends its budget more evenly**, keeping recognizable content alive further into the trajectory. The messy-looking formula

$$\bar\alpha_t = \frac{\cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac\pi2\right)}{\cos^2\!\left(\frac{s}{1+s}\cdot\frac\pi2\right)}$$

is doing two mundane things: $\cos^2$ falls smoothly from 1 to 0 over a quarter period (so it is a natural "fraction remaining" curve), and the small offset $s=0.008$ plus the denominator prevent $\bar\alpha_t$ from being exactly 1 at $t=0$, which would make $\sqrt{1-\bar\alpha_t}=0$ and produce a division by zero. **The denominator is a normalizing constant. Ignore it on first reading**, exactly as Ch. 0 §0.12 advises.

**Why higher resolution needs a shifted schedule.** Neighbouring pixels in a natural image are highly correlated — a $1024\times1024$ photo is not four times as much *information* as a $512\times512$ one, just four times as many numbers. So a given amount of noise destroys proportionally *less* information at high resolution: you can still see what the picture is, because averaging over neighbours recovers it. ▸ **The same $\beta$ schedule is therefore effectively "less destructive" at higher resolution, and the fix is to shift the whole schedule toward more noise.** Skipping this is one of the standard reasons a model that trains beautifully at 256px produces mush at 1024px.

**EMA, and why "mandatory" is not an exaggeration.** An exponential moving average of the weights, $\theta_{\text{EMA}} \leftarrow \gamma\theta_{\text{EMA}} + (1-\gamma)\theta$, with $\gamma = 0.9999$. Put a number on the memory length: the average has a time constant of about $1/(1-\gamma) = 10{,}000$ steps, so it is a smoothed version of the last ten thousand updates. **Because each training step sees a random $t$ and a random $\epsilon$, consecutive weight vectors bounce around a great deal; the EMA averages that bounce away.** Sample quality with and without is not a marginal difference — the raw weights produce visibly worse images. Always evaluate the EMA copy, always checkpoint both.

**Timestep embedding.** The network must know how much noise to expect, and $t$ is a single integer, so it is expanded into a rich vector using the same sinusoidal features as transformer positional encodings (Ch. 12) and injected into every block. **The reason it goes into every block rather than only the input is that "how much noise is present" changes what *every* layer should do**, not just the first — this is a global mode switch, not a piece of input data.

**Zero terminal SNR, the bug that lived in production.** With the standard linear schedule, $\bar\alpha_T \approx 4\times10^{-5}$, not $0$. So $x_T$ during training always retains a whisper of the original image — in particular, **its mean brightness.** At sampling time you start from pure $\mathcal{N}(0,I)$, which has mean zero, so the model has been trained to assume "the average brightness is already roughly correct" and never learns to change it. ▸ **The consequence is famous: standard Stable Diffusion cannot generate a  black image, or a  white one.** Ask for "a solid black background" and you get dark grey. It is a train/test mismatch of four parts in a hundred thousand, and it produced a visible, widely-noticed limitation in a deployed product — a good reminder that in generative modelling, tiny distributional gaps do not stay tiny.

---

## Did you know?

- **Diffusion models were published in 2015 and then ignored for five years.** Sohl-Dickstein, Weiss, Maheswaranathan and Ganguli's *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* had the forward process, the reverse process, and the variational bound. The samples were small and blurry, generative adversarial networks were producing better pictures, and the field moved on. The mathematics that made 2022's image models work was sitting in a 2015 paper the whole time.

- **The entire framework was imported from statistical physics.** The 2015 paper's argument comes from non-equilibrium thermodynamics — results about relating a complicated distribution to a simple one by a slow, gradual process. "Diffusion," "Langevin dynamics," "annealing," and "the probability-flow equation" are all borrowed vocabulary, and in every case the borrowing is literal rather than metaphorical.

- **Classifier-free guidance — the slider in every image-generation interface — was a four-page workshop paper.** Ho and Salimans presented it at a NeurIPS 2021 workshop, not the main conference. It is now used in essentially every text-conditioned diffusion system deployed anywhere.

- **The theorem that lets you run diffusion backwards is from 1982 control theory.** Brian D. O. Anderson, an Australian systems and control theorist, proved that the time-reversal of a diffusion process is itself a diffusion process, in a paper about smoothing problems. It sat unused by machine learning for thirty-nine years.

- **DDPM and score-based models were developed independently and turned out to be the same algorithm.** Ho, Jain and Abbeel arrived from a variational bound; Song and Ermon arrived from score matching and Langevin sampling. The two are related by $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ — a rescaling and a minus sign.

- **Tweedie's formula was popularized as a correction for the winner's curse.** Bradley Efron's 2011 paper applied it to selection bias: the most extreme measurements in a study are the ones most likely to have been inflated by luck, and Tweedie's formula tells you how far to shrink them back. The same equation is why a diffusion model's noise prediction is also an optimal denoiser.

- **The U-Net at the heart of every image diffusion model was invented for biomedical microscopy.** Ronneberger, Fischer and Brox designed it in 2015 at Freiburg to segment cells in electron-microscopy images, with a training set of thirty annotated images. It won a cell-tracking challenge. Nothing about it was designed for generation.

- **Flow matching was discovered three times within about a month.** In late 2022, Lipman et al. at Meta (flow matching), Liu, Gong and Liu at UT Austin (rectified flow), and Albergo and Vanden-Eijnden at NYU (stochastic interpolants) independently arrived at the same construction from optimal transport, ODE straightening, and statistical mechanics respectively.

- **Standard Stable Diffusion cannot generate a truly black image.** Its noise schedule leaves $\bar\alpha_T\approx 4\times10^{-5}$ rather than zero, so a whisper of the original image's mean brightness survives to the final step during training — and the model consequently never learns to change the overall brightness at sampling time. The fix ("zero terminal SNR") is a two-line schedule change.

- **A single 1000-step sample with guidance costs 2000 full network evaluations.** That is roughly the compute of classifying two thousand images, to produce one. The three-year campaign from 1000 steps to 4 was fought almost entirely at sampling time — the trained weights barely changed.

- **The paper that made diffusion famous was titled after its own claim.** Dhariwal and Nichol's 2021 OpenAI paper is called *Diffusion Models Beat GANs on Image Synthesis*. Six years earlier the same family of models had been producing blurry thumbnails.

- **Langevin's 1908 paper was three pages long and described itself as offering an "infinitely simpler" derivation than Einstein's.** Paul Langevin was a student of Pierre Curie, and during the First World War he led French work on piezoelectric echo-ranging — the direct ancestor of sonar. The equation named after him is the drift-plus-noise structure that every diffusion sampler implements.

- **After a thousand steps of the original schedule, $0.6\%$ of the image's amplitude remains.** $\sqrt{\bar\alpha_{1000}}\approx 0.006$ — below the size of one quantization level in an 8-bit image. The destruction is not metaphorical; the original is  unrecoverable, which is exactly what makes $\mathcal{N}(0,I)$ a legitimate starting point.

---

## Check for Understanding

**A diffusion model is a denoiser trained with plain MSE, and everything else — the closed-form forward process that makes training one-shot, the Gaussian posterior that makes the ELBO a sum of tractable KLs, the equivalence between noise prediction and the score, and the probability-flow ODE that lets a numerical solver replace a thousand ancestral steps — is the machinery that makes that one simple regression a principled and fast generative model.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is generating an image by removing noise a thousand times easier than generating it in one shot?** (Answer in terms of what each individual step is being asked to do, not in terms of the objective.)
2. **What does $\bar\alpha_t$ mean in words, and why does it collapse so fast?** Why does that single number make training possible?
3. **Why can you compute the exact one-step-back distribution during training but not during sampling?** What changes between the two?
4. **What is the variational bound for, given that nobody ever implements it?** Use the mountain-in-cloud picture, not the algebra.
5. **Why does the network predict the noise rather than the image?** Say what goes wrong at high noise if you ask for the image, and what goes wrong at low noise if you ask for the noise.
6. **Why did dropping the per-timestep weight from the loss make the pictures better and the likelihood worse?** (Correct answer: likelihood and perception care about different frequencies.)
7. **What is the score, and why does it let you model a distribution without ever computing its normalizing constant?**
8. **Why must the sampler add fresh noise on the way down, when the whole point is to remove noise?**
9. **What does the probability-flow ODE claim, and why did it cut sampling from a thousand steps to twenty?** (Note that nothing about the trained network changed.)
10. **Explain classifier-free guidance using the caricature analogy.** Why does turning it up improve prompt adherence and hurt diversity at the same time?
11. **Why is the guidance scale that human raters prefer higher than the one that minimizes FID?** What does that gap tell you about both metrics?
12. **Why does running diffusion in a VAE's latent space rather than in pixels make attention affordable?** Put a number on the reduction.
13. **What is the hard ceiling that latent diffusion imposes, and why can no amount of extra sampling steps get past it?**

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 21 — Discrete Diffusion & Conditional Generation](21-discrete-diffusion-and-conditioning.md)
