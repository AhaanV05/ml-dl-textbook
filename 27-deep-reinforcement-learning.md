# Chapter 27 — Deep Reinforcement Learning

> **Prerequisites:** Ch. 26, Ch. 6.

---

## 27.1 The problem with function approximation

Tabular methods store $Q(s,a)$ for every pair. Atari from pixels has $\sim256^{84\times84\times4}$ states. So approximate: $Q_\theta(s,a)$ with a neural network.

▸ **This immediately activates the deadly triad** (Ch. 26 §26.7). Naive Q-learning with a neural network diverges. The history of deep RL is largely the history of stabilizing tricks.

The two families:
- **Value-based** — learn $Q$, act greedily. Sample-efficient, off-policy, discrete actions.
- **Policy-based** — learn $\pi_\theta$ directly. Handles continuous actions and stochastic policies, but is on-policy and sample-hungry.
- **Actor–critic** — both. This is where nearly everything modern lives.

---

## 27.2 DQN

### The architecture and the two crucial tricks

Network: pixels → conv stack → FC → $|\mathcal{A}|$ outputs (one forward pass gives all action values).

**Loss:**
▸ $$\mathcal{L}(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\Big(r+\gamma\max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a)\Big)^2\right]$$

**Trick 1 — Experience replay.** Store transitions in a buffer (1M capacity) and sample minibatches uniformly.
- Breaks the temporal correlation that violates i.i.d.
- Each transition is reused many times (**sample efficiency**).
- Averages over many past behaviour distributions, smoothing the moving-target problem.

**Trick 2 — Target network.** A frozen copy $\theta^-$, updated every $C\approx10^4$ steps (or by a slow Polyak average $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$).

▸ **Why the target network is essential:** without it, the regression target $r+\gamma\max_{a'}Q_\theta(s',a')$ depends on the same parameters being updated. Increasing $Q(s,a)$ increases the target for $Q(s',a')$, which feeds back — a positive feedback loop that diverges. **Freezing the target turns RL back into a sequence of ordinary supervised regression problems.** This is the single most important idea in DQN.

**Other essentials:** reward clipping to $[-1,1]$ (one learning rate across games — but it destroys reward magnitude information), frame stacking (4 frames, to restore the Markov property), frame skipping, and **Huber loss** (quadratic near zero, linear far away — bounds the gradient from large TD errors).

---

## 27.3 The DQN improvements

**Double DQN.** Decouple selection from evaluation (Ch. 26 §26.6):
▸ $$y = r+\gamma\,Q_{\theta^-}\!\Big(s',\ \arg\max_{a'}Q_\theta(s',a')\Big)$$
The online network chooses, the target network evaluates. Removes most of the maximization bias with a one-line change.

**Dueling DQN.** Split the head into value and advantage streams:
▸ $$Q(s,a) = V(s) + \Big(A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\Big)$$
The mean subtraction resolves the identifiability problem (otherwise $V$ and $A$ are only determined up to a constant). ▸ **Why it helps:** in many states the action doesn't matter much, and this architecture learns $V(s)$ from *every* transition rather than only from the action taken.

**Prioritized Experience Replay.** Sample transitions with probability $\propto|\delta_i|^\alpha$. Corrects the resulting bias with importance weights $w_i=\left(\frac{1}{N}\cdot\frac{1}{P(i)}\right)^\beta$, annealing $\beta\to1$. Learns most from surprising transitions.

**Multi-step returns:** $n=3$ typically. Faster reward propagation.

**Distributional RL (C51, QR-DQN).** Learn the **distribution** of returns $Z(s,a)$, not just its mean, via a distributional Bellman equation $Z(s,a)\stackrel{D}{=}R+\gamma Z(S',A')$.
▸ **Why it works better even when you only need the mean:** predicting a full distribution is a richer auxiliary task that shapes the representation, and it makes learning robust to the multimodality of returns. This was one of the largest single improvements in the Rainbow ablation.

**NoisyNets.** Learnable parametric noise on the weights: $y=(\mu^w+\sigma^w\odot\epsilon^w)x+\dots$. State-dependent exploration that anneals itself, replacing $\epsilon$-greedy.

**Rainbow** combines all six. The ablation is instructive: **prioritized replay and multi-step returns contribute most**; every component contributes something.

---

## 27.4 The policy gradient theorem

### The setup

Parameterize the policy directly: $\pi_\theta(a\mid s)$. Objective $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$.

**The difficulty:** the distribution being sampled from depends on $\theta$, so we can't just move the gradient inside.

### The log-derivative trick

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)$$

Therefore
$$\nabla_\theta J = \nabla_\theta\int p_\theta(\tau)R(\tau)d\tau = \int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)R(\tau)d\tau = \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\big]$$

Now expand the trajectory probability:
$$\log p_\theta(\tau)=\log\rho(s_0)+\sum_{t}\Big[\log\pi_\theta(a_t\mid s_t) + \log P(s_{t+1}\mid s_t,a_t)\Big]$$

▸ **The environment dynamics $P$ and the initial distribution $\rho$ do not depend on $\theta$, so they vanish under $\nabla_\theta$.** This is the crucial step, and it is why policy gradients are **model-free** — you never need to know the transition probabilities.

▸ $$\boxed{\ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\ \Psi_t\right]\ }$$

**Interpretation:** increase the log-probability of actions that led to high $\Psi$, decrease it for low $\Psi$. **It is weighted maximum likelihood, where the weights are the returns.**

### The choices of $\Psi_t$ — this is the whole design space

| $\Psi_t$ | Name | Bias | Variance |
|---|---|---|---|
| $R(\tau)$ | REINFORCE | none | enormous |
| $\sum_{t'\ge t}r_{t'}$ | reward-to-go | none | very high |
| $\sum_{t'\ge t}r_{t'} - b(s_t)$ | with baseline | none | high |
| $Q^\pi(s_t,a_t)$ | Q actor–critic | low | medium |
| $A^\pi(s_t,a_t)$ | **advantage AC** | low | **lowest** |
| $r_t+\gamma V(s_{t+1})-V(s_t)$ | TD residual | higher | very low |
| GAE($\lambda$) | §27.6 | tunable | tunable |

**Reward-to-go** uses causality: an action at time $t$ cannot affect rewards before $t$, so including them adds only variance.

### Why a baseline is free — prove it

For any $b(s)$ not depending on $a$:
$$\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big] = b(s)\sum_a\pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)} = b(s)\,\nabla_\theta\underbrace{\sum_a\pi_\theta(a\mid s)}_{=1} = 0$$

▸ **Subtracting any state-dependent baseline leaves the gradient unbiased while reducing variance.** The variance-minimizing baseline is close to $V^\pi(s)$, which is why the advantage $A=Q-V$ is the canonical choice.

▸ **This is also exactly why GRPO works** (Ch. 16 §16.6): the mean reward over a group of completions for the same prompt is a valid, action-independent baseline — a Monte-Carlo estimate of $V(s)$ that replaces a learned value network.

---

## 27.5 Actor–critic

**Actor:** $\pi_\theta$, updated by the policy gradient.
**Critic:** $V_\phi$ or $Q_\phi$, updated by TD regression.

```
δ = r + γ V_φ(s') − V_φ(s)          # TD error, an estimate of the advantage
θ ← θ + α_θ ∇_θ log π_θ(a|s) · δ    # actor
φ ← φ + α_φ δ ∇_φ V_φ(s)            # critic
```

**A2C/A3C:** run many parallel environments to decorrelate data (A3C did it asynchronously; A2C synchronously and works just as well on a GPU). Add an **entropy bonus** $+\beta H(\pi_\theta(\cdot\mid s))$ to the objective — a direct, cheap, and effective exploration mechanism that prevents premature determinism.

---

## 27.6 Generalized Advantage Estimation

We need an advantage estimate that trades bias against variance smoothly.

Define the TD residual $\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$. The $k$-step advantage estimator is
$$\hat A_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$$

($k=1$: low variance, high bias from the imperfect $V$. $k=\infty$: unbiased, huge variance.)

**GAE** is the exponentially-weighted average of all of them:

▸ $$\boxed{\ \hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}\ }$$

- $\lambda=0$: $\hat A_t=\delta_t$ — maximum bias, minimum variance.
- $\lambda=1$: $\hat A_t=\sum_l\gamma^l r_{t+l} - V(s_t)$ — unbiased Monte Carlo.
- **$\lambda=0.95$ is the near-universal default.**

▸ **GAE is TD($\lambda$) applied to advantages** (Ch. 26 §26.5) — the same geometric-averaging idea, and it computes in one backward pass over the trajectory:
```
adv = 0
for t in reversed(range(T)):
    delta = r[t] + gamma*V[t+1]*(1-done[t]) - V[t]
    adv = delta + gamma*lam*(1-done[t])*adv
    A[t] = adv
```
▸ Note that $\gamma$ and $\lambda$ do different jobs: **$\gamma$ defines the objective** (how farsighted you want to be); **$\lambda$ is purely an estimator hyperparameter.** Conflating them is a common error.

---

## 27.7 TRPO and PPO

### The problem

Policy gradient steps in *parameter* space can cause arbitrarily large changes in *policy* space. Since the data distribution depends on the policy, one bad step collapses performance and there is no fixed dataset to recover from. **Unlike supervised learning, RL has no ground truth to fall back on.**

### TRPO

Constrain the step in **KL divergence**, which measures distance between policies rather than parameters:

▸ $$\max_\theta\ \mathbb{E}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}\hat A\right]\quad\text{s.t.}\quad \mathbb{E}_s\big[\mathrm{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\big]\le\delta$$

Solved by a second-order approximation: the KL constraint's Hessian is the **Fisher information matrix**, so the step direction is the **natural gradient** $F^{-1}g$ (Ch. 4 §4.7), computed by conjugate gradient with Hessian-vector products, followed by a line search to enforce the constraint exactly.

Monotonic-improvement guarantee, and it works — but it is complex, second-order, and awkward to combine with shared actor–critic parameters or with dropout/BatchNorm.

### PPO — the workhorse

▸ $$\boxed{\ \mathcal{L}^{\text{CLIP}}(\theta)=\mathbb{E}_t\left[\min\Big(\rho_t\hat A_t,\ \ \mathrm{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,\hat A_t\Big)\right]\ },\qquad \rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

**Read the $\min$ carefully — this is the part people get wrong.**

- **If $\hat A_t>0$** (a good action): the unclipped term grows with $\rho$; the clipped term saturates at $(1+\epsilon)\hat A$. The $\min$ picks the smaller, so **the objective stops rewarding you past $\rho=1+\epsilon$.** No incentive to move further.
- **If $\hat A_t<0$** (a bad action): the terms are negative; the $\min$ picks the *more negative*, which is the unclipped term when $\rho>1$. So **there is no ceiling on pushing a bad action's probability down**, but once $\rho<1-\epsilon$ the clip kicks in and the gradient vanishes.

▸ **The $\min$ makes the objective a pessimistic lower bound on the unclipped surrogate**: it removes the incentive to move far, without ever preventing a correction back toward the old policy. That asymmetry is the design, and $\epsilon=0.2$ is the standard.

**The full PPO loss:**
$$\mathcal{L} = \mathcal{L}^{\text{CLIP}} - c_1\underbrace{(V_\phi(s_t)-V^{\text{targ}}_t)^2}_{\text{critic}} + c_2\underbrace{H[\pi_\theta](s_t)}_{\text{entropy bonus}}$$

**Algorithm:** collect $T$ steps from $N$ parallel actors → compute GAE → do $K$ (typically 4–10) epochs of minibatch SGD on the same data → discard → repeat.

▸ **Reusing the data for several epochs is what makes PPO sample-efficient, and the clipping is what makes that reuse safe.** That is the one-sentence summary of the algorithm.

**The implementation details that actually determine whether PPO works** (documented in "The 37 Implementation Details of PPO" — worth knowing that these matter as much as the algorithm):
1. **Advantage normalization** per minibatch.
2. **Observation normalization** with a running mean/std.
3. **Reward scaling** by a running estimate of the return's std.
4. **Orthogonal initialization**, with the policy head's gain set to 0.01 so the initial policy is near-uniform.
5. **Value-function loss clipping.**
6. **Gradient clipping** at 0.5.
7. **LR annealing.**
8. Adam $\epsilon=10^{-5}$, not the default $10^{-8}$.

---

## 27.8 Continuous control

### DDPG

Deterministic policy $\mu_\theta(s)$; the critic's gradient flows through the actor via the chain rule:
▸ $$\nabla_\theta J = \mathbb{E}\Big[\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}\ \nabla_\theta\mu_\theta(s)\Big]$$

Off-policy with replay and target networks. Exploration by adding noise to the action. **Notoriously brittle** and sensitive to hyperparameters.

### TD3 — three fixes to DDPG

1. **Clipped double Q:** learn two critics, use $\min(Q_1,Q_2)$ in the target. Directly counters overestimation bias — a deliberately *pessimistic* estimate.
2. **Delayed policy updates:** update the actor once per two critic updates, so the actor chases a more converged critic.
3. **Target policy smoothing:** add clipped noise to the target action, $a'=\mu_{\theta^-}(s')+\mathrm{clip}(\epsilon,-c,c)$. This smooths the value estimate over nearby actions and prevents the actor from exploiting sharp, spurious peaks in $Q$.

### SAC — maximum entropy RL

▸ $$J(\pi)=\sum_t\mathbb{E}\big[r(s_t,a_t)+\alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big]$$

**Maximize reward *and* entropy.** Consequences: better exploration (the policy stays stochastic), robustness (it learns multiple ways to succeed rather than one brittle way), and much less hyperparameter sensitivity.

The soft value functions become
$$Q(s,a)=r+\gamma\,\mathbb{E}_{s'}\big[V(s')\big],\qquad V(s)=\mathbb{E}_{a\sim\pi}\big[Q(s,a)-\alpha\log\pi(a\mid s)\big]$$

and the optimal policy is a **Boltzmann distribution** $\pi^*(a\mid s)\propto\exp\!\big(\tfrac1\alpha Q(s,a)\big)$.

▸ **That is exactly the KL-regularized optimum from Ch. 16 §16.5 with a uniform reference policy.** RLHF's KL-to-reference and SAC's entropy bonus are the same mathematical object; recognizing this is a strong cross-domain connection to be able to draw.

**Implementation:** twin critics with the min; a squashed Gaussian policy $a=\tanh(\mu+\sigma\epsilon)$ trained via the reparameterization trick (Ch. 19 §19.3); and **automatic temperature tuning** — adjust $\alpha$ to hold the entropy at a target $\bar{\mathcal H}$ (typically $-\dim(\mathcal A)$) by minimizing $\mathbb{E}[-\alpha(\log\pi + \bar{\mathcal H})]$.

▸ **SAC is the default choice for continuous control**: off-policy (sample-efficient), stable, and largely tune-free.

---

## 27.9 Model-based RL

Learn $\hat P(s'\mid s,a)$ and plan with it. ▸ **Sample efficiency improves by 10–100×**, because the model can be queried without touching the environment.

- **Dyna:** interleave real experience, model learning, and planning on imagined transitions.
- **PETS:** an ensemble of probabilistic dynamics models + model-predictive control. The ensemble captures epistemic uncertainty and prevents the planner from exploiting model errors.
- **MuZero:** learns a *latent* dynamics model trained only to predict reward, value, and policy — **not to reconstruct observations.** It models what matters for decisions and nothing else, and plans with MCTS in latent space. Mastered Go, chess, shogi, and Atari with no rules given.
- **Dreamer:** learns a recurrent latent world model and trains an actor–critic entirely inside imagined rollouts. DreamerV3 solved Minecraft diamond collection from scratch with a single hyperparameter setting.

▸ **The central failure mode is model exploitation:** the planner finds and drives toward states where the model is wrong and optimistic. Countermeasures: ensembles, short rollout horizons, and pessimistic value estimates.

---

## 27.10 Offline RL

Learn from a fixed dataset with no environment interaction. This is what makes RL applicable to healthcare, robotics with expensive hardware, and recommendation systems.

▸ **The core problem is distributional shift on *actions*, not states.** The learned policy proposes actions absent from the data; $Q$ is unconstrained there and, because of the $\max$, systematically **overestimates** them. The policy then chases these hallucinated values, and there is no environment feedback to correct the error. **More gradient steps make it worse.**

**The families of solutions:**
- **Policy constraint (BCQ, BEAR, TD3+BC):** keep $\pi$ close to the behaviour policy. TD3+BC simply adds a behaviour-cloning term to the actor loss — trivially simple and a strong baseline.
- **Conservative value estimation (CQL):** add a penalty that pushes down $Q$ on out-of-distribution actions and up on dataset actions, yielding a provable lower bound on the true value.
- **In-sample learning (IQL):** never evaluate $Q$ at an unseen action at all. Use expectile regression to approximate the max over *in-dataset* actions only, then extract the policy by advantage-weighted regression. **Elegant, because it sidesteps the problem rather than penalizing it.**
- **Sequence modelling (Decision Transformer):** condition on a desired return and treat the trajectory as a sequence to be autoregressively modelled. No Bellman backup, no TD error, no divergence — supervised learning applied to RL data. Strong on many benchmarks, weaker where **trajectory stitching** (combining parts of different suboptimal trajectories into a better one) is required, which is precisely what dynamic programming does and sequence modelling does not.

---

## 27.11 Multi-agent RL and the practical realities

**MARL.** Non-stationarity is the core issue: from any one agent's perspective, the environment changes as the others learn, so the Markov property fails at the level of the individual agent. Standard framework: **centralized training with decentralized execution** (MADDPG, QMIX — where QMIX enforces monotonic value factorization so that a global argmax decomposes into local argmaxes). Self-play drove AlphaGo, OpenAI Five, and AlphaStar.

**Sim-to-real.** **Domain randomization** — randomize physics parameters, textures, lighting, and latencies during training so the real world looks like just another sample from the training distribution. This is the workhorse technique for robotics.

**Why deep RL is hard in practice, stated plainly:**
- Sample efficiency: often $10^7$–$10^9$ environment steps.
- **Reproducibility:** results vary enormously across random seeds; always report a distribution over ≥5 seeds, not a single curve (this is Chapter 3's lesson, and RL violates it more than any other subfield).
- Hyperparameter sensitivity, especially for DDPG-family methods.
- Reward specification, which is usually the real bottleneck (Ch. 26 §26.9).
- **A well-tuned behaviour-cloning or scripted baseline frequently beats RL** on real problems. Check first.

▸ **Where deep RL is unambiguously winning today:** games and simulated control; RLHF and RLVR for language models (Ch. 16) — currently the single largest deployment of policy-gradient methods in the world; chip floorplanning; data-centre cooling; recommender systems; and increasingly, robotic manipulation via sim-to-real.

---

## Check for Understanding

**Deep RL is the Bellman equation plus a neural network plus a large collection of devices for keeping the resulting feedback loop stable — target networks and replay for value methods, baselines and GAE for variance, and trust regions or clipping for policy methods — and PPO dominates because its clipped ratio makes it safe to take several gradient epochs on the same batch of on-policy data.**

---

**Next:** [Chapter 28 — Vision Transformers & Multimodal Models](28-vision-transformers-and-multimodal.md)
