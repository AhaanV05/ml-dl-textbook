# Chapter 20 — Diffusion Models

> **Prerequisites:** Ch. 1 (§1.3.3 Gaussian closure, §1.4.4 ELBO), Ch. 19.

---

## 20.1 The idea

### The one-line idea

Gradually destroy data by adding noise until it becomes pure Gaussian noise, then train a network to reverse one step of that destruction — and generate by running the reversal from noise.

### The analogy

A drop of ink diffusing in water. Running it forward is easy and requires no intelligence: physics does it. Running it *backward* — reassembling the drop — is the hard part. But here's the trick that makes it tractable: **reversing one tiny instant of diffusion is easy**, because over an infinitesimal step the ink has barely moved and you only need to guess a small correction. Chain a thousand easy corrections and you have reassembled the drop.

▸ **The deep reason diffusion works:** it decomposes an intractable generative problem into a thousand tractable denoising problems. Each is a simple supervised regression. The difficulty is amortized across steps rather than concentrated in one.

---

## 20.2 The forward process

Fix a variance schedule $\beta_1,\dots,\beta_T$ with $0<\beta_t\ll1$.

▸ $$q(x_t\mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$$

**Why $\sqrt{1-\beta_t}$ and not 1?** Variance preservation. If $\mathrm{Var}(x_{t-1})=1$ then $\mathrm{Var}(x_t) = (1-\beta_t)\cdot1+\beta_t = 1$. Without the shrinkage the variance would grow without bound. (The alternative, "variance exploding," is used by score-based SDE formulations.)

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

---

## 20.4 The variational bound

Model the reverse as $p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta)$. Apply the ELBO:

$$-\log p_\theta(x_0) \le \mathbb{E}_q\left[\underbrace{\mathrm{KL}(q(x_T|x_0)\|p(x_T))}_{L_T\text{, no params}} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{-\log p_\theta(x_0|x_1)}_{L_0}\right]$$

▸ The whole objective is a sum of **KLs between Gaussians**, each of which has a closed form. Since both are Gaussian with the same (fixed) variance:

$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\big\|^2\right] + C$$

**The model is doing regression on the posterior mean.**

### The $\epsilon$-parameterization

From the closed form, $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon\right)$. Substitute into $\tilde\mu_t$ and simplify:

▸ $$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

So parameterize $\mu_\theta$ with the same form and a network $\epsilon_\theta$:

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)$$

Then the loss becomes:

▸ $$L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t)\big\|^2\right]$$

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

### The three parameterizations

| Predict | Relation | Best at |
|---|---|---|
| $x_0$ | direct | small $t$ (target is well-determined) |
| $\epsilon$ | $x_0 = \frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$ | large $t$; **the standard** |
| $v$ | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | **all $t$**; best for distillation and high resolution |

▸ **Why $\epsilon$ beats $x_0$:** at large $t$, $x_t$ is nearly pure noise, so $\epsilon\approx x_t$ and the target is well-scaled; but $x_0$ is almost unconstrained, so predicting it is a high-variance target. Conversely at small $t$, predicting $\epsilon$ is hard because tiny errors get amplified by $1/\sqrt{\bar\alpha_t}$. **$v$-prediction interpolates smoothly between them**, which is why it dominates for distillation and very high SNR ranges.

**SNR framing:** $\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$. Different parameterizations and loss weightings are all just different functions $w(\mathrm{SNR})$ multiplying the same underlying regression. **Min-SNR-$\gamma$ weighting** ($w = \min(\mathrm{SNR},\gamma)/\mathrm{SNR}$ for $\epsilon$-pred) resolves the conflicting gradients between timesteps and speeds convergence substantially.

---

## 20.5 Sampling

**Ancestral (DDPM):**
▸ $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z,\qquad z\sim\mathcal{N}(0,I),\ z=0 \text{ at } t=1$$

$\sigma_t^2 = \beta_t$ or $\tilde\beta_t$ both work; the choice matters little.

$T=1000$ steps. Slow — this is diffusion's one real weakness, addressed in §20.9.

---

## 20.6 The score-based view

### Score matching

Define the **score** as $s(x) = \nabla_x\log p(x)$ — the direction of steepest increase in log-density. Note it is independent of the normalizing constant (Ch. 19 §19.6), which is what makes it tractable.

**Langevin dynamics** samples from $p$ using only the score:
$$x_{k+1} = x_k + \frac{\delta}{2}\nabla_x\log p(x_k) + \sqrt{\delta}\,z_k$$

### The connection

For $q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$:

$$\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

▸ $$\boxed{\ s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}\ }$$

**Predicting the noise and estimating the score are the same task up to a scale factor.** DDPM (Ho et al.) and NCSN (Song & Ermon) were developed independently and turned out to be the same algorithm. This is one of the more satisfying convergences in modern ML, and knowing it is a strong signal in an interview.

**Tweedie's formula** makes the third connection explicit: for $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma\epsilon$,
$$\mathbb{E}[x_0\mid x_t] = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log q(x_t)\right)$$
▸ **The optimal denoiser, the score, and the posterior mean are the same object.** Three literatures, one function.

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

### DDIM

The discrete non-Markovian family interpolating between stochastic and deterministic:

▸ $$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\underbrace{\left(\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted }x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta}_{\text{direction pointing to }x_t} + \sigma_t z$$

$\sigma_t=0$ gives **deterministic DDIM** — a first-order discretization of the probability-flow ODE. Crucially, it uses **the same trained network** and allows skipping timesteps: 20–50 steps for near-DDPM quality.

**Modern solvers:** DPM-Solver++ (exploits the semi-linear structure analytically; 10–20 steps), UniPC, Heun/Euler-ancestral (the "karras" samplers). ▸ **The insight enabling all of them:** the ODE is *semi-linear* — the linear part can be solved exactly, so the numerical error only comes from the nonlinear network term.

---

## 20.8 Guidance

### Classifier guidance

Bayes: $\nabla_x\log p(x\mid y) = \nabla_x\log p(x) + \nabla_x\log p(y\mid x)$. Add a scaled classifier gradient:

$$\hat\epsilon = \epsilon_\theta(x_t,t) - w\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)$$

**Requires a separate classifier trained on noisy inputs.** Awkward.

### Classifier-free guidance — the one that matters

Train **one** network on both conditional and unconditional objectives by randomly dropping the condition (replacing $y$ with a null token) ~10% of the time. Then at sampling:

▸ $$\boxed{\ \tilde\epsilon_\theta(x_t,t,y) = \epsilon_\theta(x_t,t,\varnothing) + w\big[\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)\big]\ }$$

**Derivation of why this is the right form:** an implicit classifier satisfies $\nabla\log p(y|x)=\nabla\log p(x|y)-\nabla\log p(x)$. Substituting into the classifier-guidance formula and converting scores to $\epsilon$ gives exactly the above. So CFG is classifier guidance with the classifier expressed through the generative model itself.

- $w=1$: ordinary conditional sampling.
- $w>1$: **extrapolation beyond the conditional distribution** — pushes toward regions where the condition is *unusually* likely.

▸ **The trade-off, stated precisely:** higher $w$ increases prompt adherence and per-sample fidelity while **reducing diversity** and eventually producing oversaturated, over-contrasted artifacts. It is a fidelity/diversity dial, and it is the reason a text-to-image model's outputs at $w=15$ look "AI-generated." Typical: $w=7.5$ for images, $2$–$4$ for video, lower for models trained with CFG-distillation.

**Cost:** two forward passes per step. **CFG distillation** trains a student to match the guided output in one pass.

**Refinements:** dynamic thresholding (Imagen — rescale to avoid saturation at high $w$), interval guidance (apply CFG only in the middle of the trajectory, where it matters), and per-timestep $w$ schedules.

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

**Rectified flow** iterates: generate pairs with the trained model, retrain on those pairs, and the paths straighten further. "Reflow" can reach genuinely one-step generation.

**Status:** Stable Diffusion 3, Flux, and most 2024+ image and video models use rectified flow / flow matching rather than DDPM. ▸ **Conceptually, flow matching and diffusion are the same family** — different paths through the same space of interpolations between noise and data — and the DDPM schedule is one particular (curved) choice.

---

## 20.11 Latent diffusion

▸ Run diffusion in the latent space of a pretrained autoencoder rather than in pixel space.

- VAE encoder: $512\times512\times3 \to 64\times64\times4$. **A 48× reduction in elements**, so $\sim$48× less compute per step, and attention becomes affordable.
- The autoencoder handles imperceptible high-frequency detail (a *perceptual compression* stage); the diffusion model handles semantics (the *semantic compression* stage). Splitting these is the whole insight.
- Trained with a combination of $L_1$/perceptual (LPIPS) loss, a small KL or VQ regularizer, and a patch-GAN discriminator to keep decoded images sharp.

**This is Stable Diffusion**, and it is why consumer-GPU image generation exists.

**Caveat:** the autoencoder is a hard ceiling. Fine text and small faces are lost at encode time and no amount of diffusion training recovers them. Hence the trend toward higher-channel latents (4→16) in SD3/Flux.

---

## 20.12 Practical notes

- **Noise schedule.** Linear $\beta$ from $10^{-4}$ to $0.02$ (original); **cosine** $\bar\alpha_t = \frac{\cos^2\left(\frac{t/T+s}{1+s}\cdot\frac\pi2\right)}{\cos^2\left(\frac{s}{1+s}\cdot\frac\pi2\right)}$ with $s=0.008$ destroys information more gradually and is better. At higher resolutions the schedule must be **shifted** toward more noise, because neighbouring pixels are more redundant so a given $\beta$ destroys less information.
- ▸ **EMA of the weights is mandatory**, not optional. $\gamma\approx0.9999$. Sample quality with and without EMA is not close.
- **Timestep embedding:** sinusoidal features of $t$ through a small MLP, injected into every block (via AdaGN/AdaLN — Ch. 21).
- **Offset noise / zero-terminal-SNR:** standard schedules never reach $\bar\alpha_T=0$, so the model never sees pure noise and cannot generate very dark or very bright images. Fixing the schedule to reach zero terminal SNR resolves it.
- **Evaluation:** FID, CLIP score, and human preference — with Chapter 3's warnings about sample-size dependence and Chapter 19's about FID's flaws.

---

## Check for Understanding

**A diffusion model is a denoiser trained with plain MSE, and everything else — the closed-form forward process that makes training one-shot, the Gaussian posterior that makes the ELBO a sum of tractable KLs, the equivalence between noise prediction and the score, and the probability-flow ODE that lets a numerical solver replace a thousand ancestral steps — is the machinery that makes that one simple regression a principled and fast generative model.**

---

**Next:** [Chapter 21 — Discrete Diffusion & Conditional Generation](21-discrete-diffusion-and-conditioning.md)
