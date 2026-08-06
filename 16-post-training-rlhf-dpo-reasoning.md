# Chapter 16 — Post-Training: SFT, RLHF, DPO & Reasoning

> **Prerequisites:** Ch. 13. Chapter 27 (Deep RL) covers PPO in general; this chapter covers its language-specific form and can be read either before or after.

---

## 16.1 Why post-training exists

### The one-line idea

A pretrained model predicts what text is *likely*; a useful assistant produces text that is *helpful, honest, and harmless*. Those are different objectives, and the second cannot be expressed as a likelihood over web text.

### The analogy

Pretraining is reading the entire internet — you absorb every register, including the unhelpful, the wrong, and the malicious, and you have no idea which one you're supposed to be. Post-training is an apprenticeship: someone shows you how the job is done (SFT), then critiques your attempts until your instincts change (RLHF). The knowledge was already there. What changes is which of your many capabilities gets expressed.

▸ **Evidence for this framing:** LIMA achieved strong instruction-following with **1,000** carefully curated examples. Post-training is largely *elicitation*, not teaching. (Caveat: reasoning-focused RL does appear to teach genuinely new behaviour, §16.7.)

---

## 16.2 Supervised fine-tuning

Train on (prompt, response) pairs with the standard cross-entropy objective, **masking the loss on prompt tokens** so only response tokens are supervised.

**Data is everything.** Sources: human-written demonstrations (expensive, high quality), distillation from a stronger model (cheap, effective, licence-encumbered), and templated conversion of existing NLP datasets (FLAN — good for benchmark scores, poor for open-ended chat).

**Chat templates.** Special tokens delimit roles:
```
<|im_start|>system ... <|im_end|>
<|im_start|>user ... <|im_end|>
<|im_start|>assistant ... <|im_end|>
```
▸ These delimiters are also the **security boundary** for prompt injection, and they are only as strong as training made them. A model that has never seen adversarial content inside a user turn will not respect the boundary. This is the root cause of most prompt-injection vulnerabilities, and it is a good answer to "how would you defend an agent against injection?" — training-time role separation plus input sanitization plus least-privilege tooling, not prompt instructions.

**Hyperparameters:** 1–3 epochs (more overfits fast), LR $10^{-5}$–$10^{-6}$ (10–100× below pretraining), cosine decay, small batch. Watch for **catastrophic forgetting** — mixing in a small fraction of pretraining data helps.

---

## 16.3 Reward modelling

### The problem

We can't write down a reward function for "helpful." But humans can *compare* two responses reliably, even when they can't score one in isolation.

### Bradley–Terry

Assume a latent scalar reward $r^*(x,y)$ such that the probability a human prefers $y_w$ over $y_l$ given prompt $x$ is:

▸ $$p^*(y_w\succ y_l\mid x) = \frac{\exp r^*(x,y_w)}{\exp r^*(x,y_w)+\exp r^*(x,y_l)} = \sigma\big(r^*(x,y_w)-r^*(x,y_l)\big)$$

**This is exactly logistic regression on the reward difference.** Fit by maximum likelihood:

▸ $$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$

**Implementation:** initialize from the SFT model, replace the unembedding with a scalar head, and read the reward off the final token.

▸ **Note the identifiability gap:** Bradley–Terry determines $r$ only up to an additive function of $x$ (a per-prompt constant), since only differences within a prompt are observed. This is harmless for policy optimization — which we'll see is invariant to it — but it means reward magnitudes are not comparable across prompts.

### The failure modes

- **Length bias.** Humans prefer longer answers; the RM learns "longer = better"; the policy learns to ramble. Fixes: length-controlled evaluation, explicit length penalties, or debiasing the RM.
- **Sycophancy.** Humans prefer agreement, so the model learns to agree.
- **Distribution shift.** The RM is trained on SFT-model outputs but evaluated on the *optimized* policy's outputs, which drift out of distribution. This is why the KL penalty (§16.4) is not optional.
- **Reward hacking.** Optimizing hard against an imperfect proxy finds its flaws. **Goodhart's law is not a metaphor here; it is the central engineering problem.**

**Mitigations:** ensembles of reward models; conservative/pessimistic aggregation; periodic RM retraining on fresh policy samples; and generative or rubric-based reward models that produce a critique before a score.

---

## 16.4 RLHF with PPO

### The objective

▸ $$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big] - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

**Why the KL term is essential, not decorative:**
1. It keeps the policy inside the region where the reward model is valid.
2. It preserves fluency and knowledge from pretraining.
3. Without it, the policy collapses onto a small set of reward-maximizing degenerate outputs — the classic failure is a model that emits the same flattering paragraph for every prompt.

Typical $\beta = 0.01$–$0.1$. Implemented as a per-token penalty added to the reward: $r_t = -\beta\big(\log\pi_\theta(y_t\mid\cdot) - \log\pi_{\text{ref}}(y_t\mid\cdot)\big)$, with the RM's scalar score added at the final token.

### The RL formulation

- **State:** the prompt plus tokens generated so far.
- **Action:** the next token.
- **Reward:** the KL penalty every step, plus $r_\phi$ at the end. **Extremely sparse.**
- **Episode:** one completion.

PPO's clipped surrogate (full derivation in Ch. 27 §27.7):
▸ $$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\left[\min\left(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],\qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$

**The practical burden:** four models in memory — policy, reference, reward model, and value model. Plus generation in the loop. This is why RLHF is expensive and why DPO was such a welcome result.

---

## 16.5 DPO — derive it, this is the key result

### The one-line idea

The optimal RLHF policy has a closed form. Invert it to express the reward in terms of the policy, substitute into the Bradley–Terry likelihood, and the reward model disappears — leaving a simple classification loss on the policy itself.

### Step 1: solve the KL-regularized objective exactly

$$\max_\pi\ \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$$

Write it as a single expectation:
$$= \max_\pi\ \mathbb{E}_{y\sim\pi}\left[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\right] = \max_\pi -\beta\,\mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}}\right]$$

Define $Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$ and $\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$. Then

$$= \max_\pi\ -\beta\left[\mathrm{KL}\big(\pi\,\|\,\pi^*\big) - \log Z(x)\right]$$

$Z(x)$ doesn't depend on $\pi$, and KL is minimized at zero when $\pi=\pi^*$. Therefore:

▸ $$\boxed{\ \pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)\ }$$

**This is a Gibbs/Boltzmann distribution** — the reference policy exponentially tilted by reward. It is the exact solution, not an approximation.

### Step 2: invert

$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

### Step 3: substitute into Bradley–Terry

The BT likelihood depends only on the **difference** $r(x,y_w)-r(x,y_l)$, and $\beta\log Z(x)$ is the same for both — **it cancels.** This is the crucial step, and it's why the intractable partition function never has to be computed.

▸ $$\boxed{\ \mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]\ }$$

∎

### What the gradient does

$$\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l - \hat r_w)}_{\text{weight: high when the model is wrong}}\big(\underbrace{\nabla\log\pi_\theta(y_w)}_{\text{increase}} - \underbrace{\nabla\log\pi_\theta(y_l)}_{\text{decrease}}\big)\Big]$$

▸ Raise the winner's likelihood, lower the loser's, **weighted by how badly the implicit reward model currently has it backwards.** Pairs the model already scores correctly contribute almost nothing — automatic hard-example mining.

### The trade-offs

| DPO | PPO-RLHF |
|---|---|
| 2 models in memory | 4 models |
| No sampling loop | Online generation |
| Simple, stable, cheap | Complex, finicky |
| **Offline** — fixed preference data | **Online** — learns from its own samples |
| Can over-optimize on OOD responses | KL constraint is enforced on-policy |

▸ **Where DPO is weaker, and why:** its constraint is only enforced on the responses in the dataset. For $y$ far from the data, $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ is unconstrained, so DPO can push probability mass onto unseen outputs — the observed failure is that DPO often *decreases* the likelihood of both $y_w$ and $y_l$ while increasing the likelihood of neither. Fixes: **iterative/online DPO** (regenerate preferences from the current policy each round), and adding an SFT term on $y_w$.

Careful head-to-head studies find well-tuned online PPO still edges out DPO on the hardest tasks; DPO wins decisively on cost.

### The DPO family

| Method | Change |
|---|---|
| **IPO** | replaces the log-sigmoid with a squared loss, avoiding the overfitting that occurs when preferences are deterministic |
| **KTO** | uses *unpaired* binary feedback (good/bad), with a prospect-theory-inspired value function — much easier data collection |
| **ORPO** | folds preference optimization into SFT via an odds-ratio penalty; **no reference model at all** |
| **SimPO** | uses length-normalized average log-probability as the implicit reward; no reference model; directly addresses length bias |
| **RRHF / RSO** | rank-based and rejection-sampling variants |

---

## 16.6 GRPO and verifiable rewards

### The motivation

PPO needs a **value network** — an extra model of the same size, trained on a noisy regression target, that is hard to fit for language. GRPO removes it.

### The mechanism

▸ Sample a **group** of $G$ completions for the same prompt, and use the group's own reward statistics as the baseline:

$$\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac1G\sum_{i=1}^G \min\left(\rho_i\hat A_i,\ \mathrm{clip}(\rho_i,1\pm\epsilon)\hat A_i\right)\right] + \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$$

**Why this is valid:** any baseline that does not depend on the action leaves the policy gradient unbiased (Ch. 27 §27.5). The group mean is a Monte-Carlo estimate of $V(x)$ computed from samples rather than from a learned network. **You trade a value model for $G\times$ more sampling** — which is a good trade when generation is cheap relative to training a second large model.

### RLVR — reinforcement learning from verifiable rewards

▸ For math, code, and formal tasks, **skip the reward model entirely**: check the answer.

$$r(x,y) = \mathbb{1}[\text{answer is correct}] \quad\text{or}\quad \mathbb{1}[\text{unit tests pass}]$$

**This eliminates reward hacking at the source** — you cannot fool a compiler. It is the single most important recent development in post-training, and it is why reasoning models improved so abruptly.

**Limits:** only applies to verifiable domains; the binary signal is sparse; and models still learn to exploit *test* weaknesses (special-casing to pass unit tests) — so verification quality becomes the new bottleneck.

---

## 16.7 Reasoning models

### Chain of thought

Prompting the model to produce intermediate steps improves accuracy dramatically on multi-step problems.

▸ **The rigorous reason (Ch. 11 §11.9):** a fixed-depth transformer has bounded serial computation per forward pass. Emitting intermediate tokens gives it an external, unbounded scratchpad — each generated token is another forward pass conditioned on the previous ones. **Chain of thought converts a parallel-compute-limited model into a serial one.** It is buying depth with tokens.

- **Zero-shot CoT:** "Let's think step by step."
- **Self-consistency:** sample $k$ chains, majority-vote the final answer. Accuracy rises roughly logarithmically in $k$.
- **Tree of thoughts / search:** explore and prune branches with a value estimate.

### Trained long reasoning

Rather than prompting for CoT, **train** it with RL on verifiable rewards. The recipe (as publicly described for R1-style models):

1. Optional cold-start SFT on a small set of long, well-formatted reasoning traces.
2. Large-scale GRPO/RLVR on math and code with correctness rewards, plus a format reward for using the thinking delimiters.
3. Rejection-sample the best traces, SFT on them, and repeat.

▸ **The striking empirical result:** with pure RL on correctness, models spontaneously develop backtracking, self-verification, and explicit "wait, let me reconsider" behaviour, and **average response length grows steadily throughout training** without anyone rewarding length. The model discovers that thinking longer raises its success probability. That is genuine capability acquisition, not elicitation — which is a real qualification to the framing in §16.1.

**The remaining problems:** faithfulness (the stated chain need not be the actual computation — Ch. 32), overthinking on easy problems, and poor transfer of verifiable-domain gains to open-ended domains.

---

## 16.8 Constitutional AI and RLAIF

Replace (or supplement) human labels with model-generated ones:

1. **Supervised phase:** the model generates a response, critiques it against a written list of principles ("the constitution"), and revises. Fine-tune on the revisions.
2. **RL phase:** the model itself judges pairs of responses against the constitution, producing preference data. Train a reward model on those.

▸ **Why it matters practically:** it scales oversight (labels become cheap), makes the value specification **explicit and auditable** (a written document rather than an implicit aggregate of annotator judgements), and improves consistency. The obvious limitation is that it inherits and can amplify the judge model's own blind spots, so human data remains necessary as an anchor.

---

## 16.9 Evaluating post-trained models

- **Human preference arenas** (pairwise, Elo-rated). The best available signal; slow, expensive, and susceptible to style-over-substance effects.
- **LLM-as-judge.** Fast and correlates ~0.8 with humans, but biased toward longer, more confident, and self-authored responses. Mitigate with position-swapping, reference answers, explicit rubrics, and calibration against a human-labelled subset.
- **Capability benchmarks** — see Ch. 13 §13.5.
- **Safety evaluations:** refusal rates, jailbreak robustness, red-teaming.
- ▸ **The alignment tax:** post-training usually *worsens* raw benchmark scores and perplexity while improving usefulness. Always report both, and never compare a base model and a chat model on the same benchmark without saying so.

---

## Check for Understanding

**Post-training converts a text predictor into an assistant by learning a reward from human comparisons (Bradley–Terry) and then maximizing it under a KL leash — and DPO's contribution was noticing that the KL-regularized optimum is a closed-form Boltzmann tilt of the reference policy, so the reward can be substituted out and the whole procedure collapses into one classification loss.**

---

**Next:** [Chapter 17 — Efficient Inference & Compression](17-efficient-inference-and-compression.md)
