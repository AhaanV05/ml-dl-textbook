# Chapter 27 — Deep Reinforcement Learning

> **Prerequisites:** Ch. 26, Ch. 6.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

This chapter fuses Chapter 26's vocabulary with Chapter 6's, so a symbol may be wearing either hat. Skim this once; each entry is unpacked where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`Q_\theta(s,a)`$ | "Q-theta of s, a" | A **neural network** standing in for the Q-table. $\theta$ is its weights |
| $\theta^-$ | "theta-minus" | The **target network** — a frozen, out-of-date copy of $\theta$ |
| $\mathcal{D}$ | "script D" | The **replay buffer** — a big box of past transitions to sample from |
| $(s,a,r,s')$ | "s, a, r, s-prime" | One **transition**: where you were, what you did, what you got, where you ended up |
| $\tau$ | "tau" | Either a **trajectory** (a whole episode) or the **Polyak rate**. Context tells you which |
| $J(\theta)$ | "J of theta" | The **objective being maximized** — expected total reward. RL climbs; it does not descend |
| $`\Psi_t`$ | "psi-t" | A **placeholder** for whatever weight multiplies the log-probability. §27.4 is a menu of choices |
| $`\nabla_\theta\log\pi_\theta(a\mid s)`$ | "grad log pi" | The **score function** — which way to nudge $\theta$ to make this action more likely |
| $`\hat A_t`$ | "A-hat-t" | An **estimate** of the advantage. The hat means estimate, not truth |
| $`\delta_t`$ | "delta-t" | The **TD residual** $`r_t+\gamma V(s_{t+1})-V(s_t)`$ — one step's worth of surprise |
| $`\rho_t`$ | "rho-t" | The **probability ratio** $`\pi_{\text{new}}/\pi_{\text{old}}`$ — how much the policy changed on this action |
| $\mathrm{clip}(x,\,l,\,u)$ | "clip x between l and u" | Squash $x$ into the interval: return $l$ if too small, $u$ if too big, else $x$ |
| $\mathrm{KL}(p\,\lVert\,q)$ | "KL of p from q" | How far apart two distributions are (Ch. 1 §1.4) |
| $F$ | "the Fisher matrix" | The curvature of KL — the local geometry of *policy* space rather than parameter space |
| $H[\pi]$, $\mathcal{H}(\pi)$ | "the entropy of pi" | How undecided the policy is. High entropy = spread out; zero = fully committed |
| $`\mu_\theta(s)`$ | "mu-theta of s" | A **deterministic** policy — outputs the action itself, not a distribution over actions |
| $Z(s,a)$ | "Z of s, a" | The **whole distribution** of returns, not just its mean $Q(s,a)$ |
| $\stackrel{D}{=}$ | "equals in distribution" | The two sides have the same distribution, not the same value |
| $\odot$ | "elementwise product" | Multiply matching entries and keep them separate (§0.8) |
| $b(s)$ | "b of s" | A **baseline** — anything subtracted that depends on the state but not the action |
| $\epsilon$ | "epsilon" | The **clip width** in PPO (typically $0.2$). Not the same $\epsilon$ as $\epsilon$-greedy |

**Every abbreviation in this chapter, spelled out.** Read the full form aloud once — most of these are descriptions in disguise.

| Short | Full form |
|---|---|
| DQN | Deep Q-Network |
| PER | Prioritized Experience Replay |
| C51 | Categorical 51-atom distributional DQN |
| QR-DQN | Quantile Regression Deep Q-Network |
| A2C / A3C | (Asynchronous) Advantage Actor–Critic |
| GAE | Generalized Advantage Estimation |
| TRPO | Trust Region Policy Optimization |
| PPO | Proximal Policy Optimization |
| GRPO | Group Relative Policy Optimization |
| DDPG | Deep Deterministic Policy Gradient |
| TD3 | Twin Delayed Deep Deterministic policy gradient |
| SAC | Soft Actor–Critic |
| MPC | Model-Predictive Control |
| MCTS | Monte Carlo Tree Search |
| PETS | Probabilistic Ensembles with Trajectory Sampling |
| BC | Behaviour Cloning |
| BCQ / BEAR | Batch-Constrained Q-learning / Bootstrapping Error Accumulation Reduction |
| CQL | Conservative Q-Learning |
| IQL | Implicit Q-Learning |
| MARL | Multi-Agent Reinforcement Learning |
| MADDPG | Multi-Agent DDPG |
| QMIX | Q-value MIXing network |
| RLHF / RLVR | RL from Human Feedback / RL with Verifiable Rewards |
| SGD | Stochastic Gradient Descent |
| LR | Learning Rate |

---

## 27.1 The problem with function approximation

Tabular methods store $Q(s,a)$ for every pair. Atari from pixels has $\sim256^{84\times84\times4}$ states. So approximate: $`Q_\theta(s,a)`$ with a neural network.

▸ **This immediately activates the deadly triad** (Ch. 26 §26.7). Naive Q-learning with a neural network diverges. The history of deep RL is largely the history of stabilizing tricks.

The two families:
- **Value-based** — learn $Q$, act greedily. Sample-efficient, off-policy, discrete actions.
- **Policy-based** — learn $`\pi_\theta`$ directly. Handles continuous actions and stochastic policies, but is on-policy and sample-hungry.
- **Actor–critic** — both. This is where nearly everything modern lives.

#### Unpacking "so approximate"

$\sim256^{84\times84\times4}$ deserves to be read out loud, because the number is the argument. An Atari frame is $84\times84$ greyscale pixels after preprocessing; each pixel takes one of $256$ values; four frames are stacked to restore the Markov property (Ch. 26 §26.2). So the state is $84\times84\times4 = 28{,}224$ numbers, each with 256 possibilities:

$$256^{28{,}224} \approx 10^{68{,}000}$$

**There are about $10^{80}$ atoms in the observable universe.** The number of Atari states exceeds it by a factor of $10^{67{,}920}$. A table is not merely impractical here — you could not store one entry per atom and get anywhere near.

▸ **The move that saves you is not compression, it is *generalization*.** A table treats every state as unrelated to every other; a network treats similar states similarly, so experience in one state teaches you about states you have never seen. **That is the entire reason function approximation works, and it is also exactly why it is dangerous**: the same sharing that lets a useful update spread also lets a bad update spread.

$`Q_\theta(s,a)`$ read aloud is *"Q, parameterized by theta, of s and a."* The subscript $\theta$ is not an index — it means "this function's behaviour is determined by the weight vector $\theta$." Learning changes $\theta$; $\theta$ changes the function; the function changes the policy.

> **Analogy.** A tabular value function is a phone book: one entry per person, and knowing Smith's number tells you nothing about Smyth's. A neural value function is a *rule* for guessing numbers from names. The rule generalizes — which is wonderful when names carry signal, and catastrophic when correcting one entry silently corrupts a thousand others you never checked.

**The three families, decoded** (the book says "two families" and lists three, because actor–critic is the marriage of the first two rather than a third parent):

| Family | What it learns | What it can't easily do |
|---|---|---|
| **Value-based** | $`Q_\theta(s,a)`$; the policy is $`\arg\max_a Q`$ | Continuous actions — you cannot enumerate a $\max$ over $\mathbb{R}^7$ |
| **Policy-based** | $`\pi_\theta(a\mid s)`$ directly | Reuse old data — the gradient is only valid for the current policy |
| **Actor–critic** | Both, each helping the other | Nothing in particular; hence its dominance |

▸ **The dividing line is the $\arg\max$.** With four Atari buttons, scanning every action is free. With a robot arm's seven continuous joints there are infinitely many actions and no scan is possible — so you must represent the policy explicitly instead. **Every architectural choice in this chapter descends from whether that $\arg\max$ is computable.**

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

▸ **Why the target network is essential:** without it, the regression target $`r+\gamma\max_{a'}Q_\theta(s',a')`$ depends on the same parameters being updated. Increasing $Q(s,a)$ increases the target for $Q(s',a')$, which feeds back — a positive feedback loop that diverges. **Freezing the target turns RL back into a sequence of ordinary supervised regression problems.** This is the single most important idea in DQN.

**Other essentials:** reward clipping to $[-1,1]$ (one learning rate across games — but it destroys reward magnitude information), frame stacking (4 frames, to restore the Markov property), frame skipping, and **Huber loss** (quadratic near zero, linear far away — bounds the gradient from large TD errors).

#### Reading the DQN loss in plain English

$$\mathcal{L}(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\Big(r+\gamma\max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a)\Big)^2\right]$$

Read aloud: *"L of theta equals the expectation, over transitions s-a-r-s-prime drawn from D, of: r plus gamma times the max over a-prime of Q-theta-minus of s-prime a-prime, minus Q-theta of s a, all squared."* Now in English:

▸ **"Sample an old memory. Compute what you now think that moment was worth. Compare it to what you thought at the time. Square the difference and shrink it."**

The pieces, one at a time:

| Piece | Read aloud | Job |
|---|---|---|
| $(s,a,r,s')\sim\mathcal{D}$ | "sampled from script D" | Pull a random past transition out of the replay buffer |
| $`r+\gamma\max_{a'}Q_{\theta^-}(s',a')`$ | "the target" | The Bellman optimality right-hand side (Ch. 26 §26.3), estimated |
| $\theta^-$ | "theta-minus" | The **frozen** weights. The minus is a label meaning "old", not a negation |
| $`Q_\theta(s,a)`$ | "the prediction" | What the live network currently believes |
| $(\cdot)^2$ | "squared" | Ordinary squared error — this is a regression problem |

**Notice what this equation is.** Strip the RL vocabulary and it is $\mathbb{E}[(y - \hat y)^2]$: mean squared error, the most ordinary loss in the book. **DQN's entire design is an effort to make Q-learning look like supervised regression**, because supervised regression is a thing we know how to make stable.

**Where the two tricks fit into that sentence.** The target $`y = r + \gamma\max_{a'}Q_{\theta^-}(s',a')`$ has two properties that ordinary regression targets have and RL targets do not: it must be **fixed** while you fit it (that is the target network), and the training examples must be **shuffled** rather than arriving in temporal order (that is the replay buffer). The tricks are not clever heuristics; each restores one assumption of supervised learning that RL had broken.

#### Trick 1, decoded: why a buffer of memories

Without replay, consecutive training examples are consecutive game frames. Frame 1000 and frame 1001 are nearly the same picture with nearly the same label. **A batch of 32 consecutive frames carries roughly one frame's worth of independent information**, and the network spends its capacity chasing whatever is on screen right now.

> **Analogy.** Learning to drive by practising the same roundabout for an hour, then the same motorway slip road for an hour. You will become excellent at whichever one you did most recently and quietly forget the other. **Experience replay is a driving instructor who shuffles the day's lessons and re-runs old ones at random**, so nothing you learned in the morning is overwritten by the afternoon.

Three benefits, and they are  distinct:

1. **Decorrelation** — restores something close to the i.i.d. assumption every convergence result depends on.
2. **Sample efficiency** — a transition that took a real second to collect can be learned from dozens of times. With a 1M buffer and a batch of 32 sampled every 4 steps, each transition is replayed about 8 times on average.
3. **Distribution smoothing** — the buffer holds experience from many past policies, so the training distribution changes slowly even when the policy changes quickly.

#### Trick 2, decoded: why freezing the target is the crucial idea

Suppose you drop the target network and use $\theta$ on both sides. Now watch what a single gradient step does. You raise $`Q_\theta(s,a)`$ because the target said to. But $s$ and $s'$ are similar states — that is the whole point of generalization — so raising $`Q_\theta(s,a)`$ **also raises $`Q_\theta(s',a')`$**. Which raises the target. Which tells you to raise $`Q_\theta(s,a)`$ further.

▸ **You are chasing a target that runs away from you at a speed proportional to how fast you chase it.** That is a positive feedback loop, and positive feedback loops do not settle; they diverge. This is the deadly triad of Ch. 26 §26.7 made mechanical.

> **Analogy.** Try to catch up to a car whose accelerator is wired to your own. Every time you speed up, so does it. The fix is embarrassingly simple: **make the car stop for ten seconds at a time.** Freeze $\theta^-$ for $C \approx 10^4$ steps, chase a stationary target, then move the target to where you are and repeat. Between refreshes you are doing plain supervised regression against fixed labels — a problem with no feedback loop at all.

The **Polyak** alternative $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$ (with $\tau\approx0.005$) does the same job continuously instead of in jumps: the target creeps toward the online network at half a percent per step, so it lags by roughly $1/\tau = 200$ steps at all times. Same idea, smoother.

#### The "other essentials," decoded

**Reward clipping to $[-1,1]$.** Different Atari games pay in wildly different currencies — one awards a point per pellet, another thousands per level. A single learning rate cannot serve both, because the gradient scales with the reward. Clipping every reward to $-1$, $0$, or $+1$ makes all games numerically comparable. ▸ **The cost is real and often forgotten: after clipping, collecting a 1-point item and a 1000-point item are worth exactly the same to the agent.** You have made the problem trainable by making it a different problem.

**Frame stacking and frame skipping.** Stacking four frames restores the Markov property (velocity is visible in differences). Skipping — repeating each chosen action for 4 frames and only observing every 4th — cuts compute roughly fourfold and costs almost nothing, because Atari physics do not change meaningfully in 1/60th of a second.

**Huber loss, with numbers.** Squared error's gradient is proportional to the error itself: a TD error of $100$ produces a gradient of $200$. Huber loss is quadratic while $\lvert\delta\rvert \le 1$ and switches to linear beyond, so its gradient **saturates at 1**:

| TD error $\delta$ | Squared-loss gradient | Huber gradient |
|---|---|---|
| $0.5$ | $1.0$ | $1.0$ |
| $1$ | $2$ | $1$ |
| $10$ | $20$ | $1$ |
| $100$ | $200$ | $1$ |

▸ **This is the same bounded-gradient safety property that makes cross-entropy the right classification loss (Ch. 1 §1.5).** In RL you cannot avoid occasional enormous TD errors — a rare event, a newly discovered reward — and without a bound one such transition can destroy a run that took a day to train.

> **Where this came from.** **Experience replay is not a DQN invention.** It was introduced by **Long-Ji Lin** in his 1992 CMU PhD thesis on reinforcement learning for robots, more than twenty years earlier, for exactly the reason given above: real robot experience is expensive, so reuse it. DQN's contribution was to recognize that the technique also *stabilizes* learning, which was not Lin's motivation.
>
> DQN itself came from **DeepMind**, first as a short workshop paper in 2013 and then as the February 2015 *Nature* article *Human-level control through deep reinforcement learning*, by Volodymyr Mnih and colleagues. The result that made it famous was not any single game score but the **uniformity**: the same network architecture, the same hyperparameters and the same learning algorithm were applied to 49 different Atari games with no per-game tuning, and matched or beat a professional human games tester on a large fraction of them. **Nobody had previously shown one agent learning many different tasks from raw pixels.**
>
> A footnote worth knowing: the 2013 workshop version circulated before Google acquired DeepMind in early 2014, and the Atari demonstration is widely reported to have been influential in that acquisition. The reported sums vary between accounts, so treat the specific figure with caution — but the sequence (Atari result, then acquisition, then *Nature*) is not in dispute.

---

## 27.3 The DQN improvements

**Double DQN.** Decouple selection from evaluation (Ch. 26 §26.6):
▸ $$y = r+\gamma\,Q_{\theta^-}\!\Big(s',\ \arg\max_{a'}Q_\theta(s',a')\Big)$$
The online network chooses, the target network evaluates. Removes most of the maximization bias with a one-line change.

**Dueling DQN.** Split the head into value and advantage streams:
▸ $$Q(s,a) = V(s) + \Big(A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\Big)$$
The mean subtraction resolves the identifiability problem (otherwise $V$ and $A$ are only determined up to a constant). ▸ **Why it helps:** in many states the action doesn't matter much, and this architecture learns $V(s)$ from *every* transition rather than only from the action taken.

**Prioritized Experience Replay.** Sample transitions with probability $`\propto|\delta_i|^\alpha`$. Corrects the resulting bias with importance weights $`w_i=\left(\frac{1}{N}\cdot\frac{1}{P(i)}\right)^\beta`$, annealing $\beta\to1$. Learns most from surprising transitions.

**Multi-step returns:** $n=3$ typically. Faster reward propagation.

**Distributional RL (C51, QR-DQN).** Learn the **distribution** of returns $Z(s,a)$, not just its mean, via a distributional Bellman equation $Z(s,a)\stackrel{D}{=}R+\gamma Z(S',A')$.
▸ **Why it works better even when you only need the mean:** predicting a full distribution is a richer auxiliary task that shapes the representation, and it makes learning robust to the multimodality of returns. This was one of the largest single improvements in the Rainbow ablation.

**NoisyNets.** Learnable parametric noise on the weights: $y=(\mu^w+\sigma^w\odot\epsilon^w)x+\dots$. State-dependent exploration that anneals itself, replacing $\epsilon$-greedy.

**Rainbow** combines all six. The ablation is instructive: **prioritized replay and multi-step returns contribute most**; every component contributes something.

#### Unpacking the six improvements

Each of these is one small idea. Take them one at a time.

**Double DQN, decoded.** Compare the two targets:

$$\underbrace{r+\gamma\max_{a'}Q_{\theta^-}(s',a')}_{\text{DQN}} \qquad\text{versus}\qquad \underbrace{r+\gamma\,Q_{\theta^-}\!\big(s',\ \arg\max_{a'}Q_\theta(s',a')\big)}_{\text{Double DQN}}$$

In the first, one network both **picks** the best action and **reports** its value — so whichever action got a lucky noise draw wins the selection *and* supplies the inflated number (Ch. 26 §26.6). In the second, the online network $\theta$ picks and the frozen network $\theta^-$ scores. **Two networks already exist in DQN; Double DQN simply notices that and gives them different jobs.** It is  a one-line change, and it removes most of the overestimation.

**Dueling DQN, decoded.**

$$Q(s,a) = V(s) + \Big(A(s,a)-\tfrac{1}{\lvert\mathcal{A}\rvert}\sum_{a'}A(s,a')\Big)$$

The **identifiability problem** it fixes: if you only ever observe $Q$, you cannot recover $V$ and $A$ separately, because $V+5$ paired with $A-5$ gives exactly the same $Q$. An infinite family of $(V, A)$ pairs explains the same data, so gradient descent has no reason to settle on any of them — a recipe for drifting, unstable heads.

Subtracting the mean of $A$ pins it down. Average both sides over actions: the two $A$ terms cancel and you get $`V(s) = \frac{1}{\lvert\mathcal{A}\rvert}\sum_a Q(s,a)`$. ▸ **The mean-subtraction is a *definition*: it decrees that $V$ shall be the average action value and $A$ shall be the deviation from it.** No ambiguity remains.

**Put numbers on why this helps.** Suppose a state has action values $(102, 100, 98)$. The dueling decomposition gives $V = 100$ and $A = (+2, 0, -2)$. A single transition in which you took the middle action teaches the $V$ head that this state is worth about 100 — and **that lesson applies to all three actions**, because $V$ is shared. Plain DQN would have updated only the entry for the action taken. In states where the choice barely matters (most states, in most games), you have tripled the effective data for the part of the estimate that carries almost all the magnitude.

**Prioritized Experience Replay, decoded.** Sample transitions with probability $`\propto\lvert\delta_i\rvert^\alpha`$ — *"the bigger the surprise, the more often you revisit it."* The exponent $\alpha$ (typically $0.6$) controls how aggressive that is: $\alpha=0$ is uniform sampling, $\alpha=1$ is fully proportional to surprise.

But non-uniform sampling **biases the expectation** — you are no longer averaging over the distribution you meant to. The importance weight $`w_i = \big(\frac{1}{N}\cdot\frac{1}{P(i)}\big)^\beta`$ is the correction from Ch. 26 §26.7, applied to the buffer instead of to the policy. Annealing $\beta$ from about $0.4$ to $1$ means: **correct the bias only partially at first (when the estimates are all wrong anyway and speed matters more than exactness) and fully at the end (when you need the right answer).**

> **Analogy.** Revising for an exam by re-reading the topics you keep getting wrong, rather than working front-to-back through the textbook. Obviously better — but if you *only* ever study your weak spots, your sense of which topics are common on the exam becomes distorted. The importance weight is the correction that keeps your priorities honest.

**Multi-step returns, decoded.** Replace the 1-step target with the $n$-step return of Ch. 26 §26.5 (typically $n=3$). ▸ **The point is propagation speed.** With 1-step targets, information about a reward crawls backwards one state per update — a reward 100 steps from the start needs 100 rounds of propagation. With $n=3$ it moves three times as fast. In a system where each update is a full network gradient step, that is a threefold saving in the slowest part of learning.

**Distributional RL, decoded.** $Z(s,a)\stackrel{D}{=}R+\gamma Z(S',A')$ — the symbol $\stackrel{D}{=}$ is read *"equals in distribution."* It does **not** say the two sides are the same number; it says they are the same *random variable*, statistically. The whole Bellman equation has been lifted from means to distributions.

Why bother, when you only need the mean to act? Because a mean can be a lie. **If an action gives $+100$ half the time and $-100$ the other half, its mean is $0$ — indistinguishable from an action that always gives exactly $0$.** Those are wildly different situations, and a mean-only network is forced to represent them identically.

▸ **The deeper reason it helps, though, is representational.** Predicting an entire distribution is a much richer training signal than predicting one number, so the network's internal features must encode far more about the situation — the same auxiliary-task effect that makes self-supervised pretraining work (Ch. 25). **C51** (the name comes from its 51 discrete "atoms" of return value) was among the single largest contributors in the Rainbow ablation, despite the acting rule being unchanged.

**NoisyNets, decoded.** Instead of exploring by occasionally choosing at random ($\epsilon$-greedy), inject learnable noise into the weights themselves: $y=(\mu^w+\sigma^w\odot\epsilon^w)x+\dots$, where $\odot$ is the elementwise product (§0.8) and $\epsilon^w$ is fresh random noise each forward pass. The network learns $\sigma^w$ — **how uncertain to be** — as an ordinary parameter.

▸ **Why this is better than $\epsilon$-greedy: the noise is state-dependent and self-annealing.** $\epsilon$-greedy explores uniformly everywhere and needs a hand-designed decay schedule. NoisyNets can stay exploratory in parts of the state space it still finds confusing while acting decisively where it is confident, and it shrinks $\sigma$ by itself when noise stops paying. **No schedule to tune is a real advantage in a field where schedules are half the work.**

> **Where this came from.** The distributional perspective is due to **Marc Bellemare, Will Dabney and Rémi Munos** at DeepMind in 2017; it was a  surprising result, because the theory says the extra information should be irrelevant to a mean-maximizing agent, and the experiments said otherwise. **Rainbow** (Matteo Hessel and colleagues, 2018) then did the unglamorous and valuable work of combining six independently-published improvements and ablating each one — a paper whose main contribution is careful bookkeeping, and which is cited constantly because of it.

---

## 27.4 The policy gradient theorem

### The setup

Parameterize the policy directly: $`\pi_\theta(a\mid s)`$. Objective $`J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]`$.

**The difficulty:** the distribution being sampled from depends on $\theta$, so we can't just move the gradient inside.

### The log-derivative trick

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)$$

Therefore
$$\nabla_\theta J = \nabla_\theta\int p_\theta(\tau)R(\tau)d\tau = \int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)R(\tau)d\tau = \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\big]$$

Now expand the trajectory probability:
$$\log p_\theta(\tau)=\log\rho(s_0)+\sum_{t}\Big[\log\pi_\theta(a_t\mid s_t) + \log P(s_{t+1}\mid s_t,a_t)\Big]$$

▸ **The environment dynamics $P$ and the initial distribution $\rho$ do not depend on $\theta$, so they vanish under $`\nabla_\theta`$.** This is the crucial step, and it is why policy gradients are **model-free** — you never need to know the transition probabilities.

▸ $$\boxed{\ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\ \Psi_t\right]\ }$$

**Interpretation:** increase the log-probability of actions that led to high $\Psi$, decrease it for low $\Psi$. **It is weighted maximum likelihood, where the weights are the returns.**

#### Unpacking the policy gradient derivation, line by line

This is the most important derivation in the chapter, and it is four steps long. Take them slowly.

**Step 0 — what we want and why it's awkward.** $`J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]`$ reads: *"J of theta equals the expected total reward of a trajectory tau, when trajectories are drawn from the policy pi-theta."* Note that $J$ is a thing to **maximize** — RL climbs hills while supervised learning descends them, so expect $+\eta\nabla$ rather than $-\eta\nabla$.

The difficulty in one line: **the thing we are differentiating is hiding inside the thing we are averaging over.** In supervised learning, $\theta$ affects the *value* of the loss but not which data you see. Here, $\theta$ affects **which trajectories exist at all**. You cannot pull $`\nabla_\theta`$ through an expectation whose distribution depends on $\theta$.

> **Analogy.** You run a restaurant and want to know how a menu change affects average customer satisfaction. Easy if the same customers come regardless. But changing the menu **changes who walks in**. The change to the menu and the change to the clientele are entangled, and you cannot measure one while holding the other fixed.

**Step 1 — the log-derivative trick.**

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)$$

The first equality is **multiplying and dividing by $`p_\theta(\tau)`$**, which changes nothing. The second uses the chain rule backwards: $\nabla \log f = \frac{\nabla f}{f}$, so $\frac{\nabla f}{f} = \nabla\log f$.

▸ **That is the whole trick, and it looks like it accomplishes nothing.** What it accomplishes is enormous: the right-hand side has the form "$`p_\theta(\tau)`$ times something," and *a probability times something is an expectation*. **The trick converts a derivative-of-a-distribution into an expectation you can estimate by sampling.**

**Step 2 — swap the integral for an average.** Once you have $`\int p_\theta(\tau)\,[\cdot]\,d\tau`$, that *is* $`\mathbb{E}_{\tau\sim\pi_\theta}[\cdot]`$ by definition (§0.5). And an expectation can be estimated by running the policy and averaging. You have gone from an unsampleable object to a sampleable one without approximating anything.

**Step 3 — the environment disappears.** Expand the log-probability of a whole trajectory:

$$\log p_\theta(\tau)=\log\rho(s_0)+\sum_{t}\Big[\log\pi_\theta(a_t\mid s_t) + \log P(s_{t+1}\mid s_t,a_t)\Big]$$

Read it: *"the log-probability of an entire episode is the log-probability of the start state, plus, for each step, the log-probability that the policy chose that action and the log-probability that the world produced that next state."* The sum appears because $\log$ turns products into sums (§0.3) — trajectories are chains of independent-given-the-past events, so their probabilities multiply.

Now differentiate with respect to $\theta$. **$`\rho(s_0)`$ has no $\theta$ in it. $`P(s_{t+1}\mid s_t,a_t)`$ has no $\theta$ in it.** Anything without $\theta$ differentiates to zero.

$$\nabla_\theta\log p_\theta(\tau) = \sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)$$

▸ **The laws of physics have vanished from the gradient.** This is the single most consequential line in the chapter: you can compute the exact gradient of expected reward **without knowing, estimating, or even being able to write down the environment's dynamics**. That is what "model-free" means, and it is why policy gradients apply to any simulator, any game, any physical robot.

**The result.**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\ \Psi_t\right]$$

$`\nabla_\theta\log\pi_\theta(a\mid s)`$ is called the **score function**. Read it as: *"the direction in weight space that makes this exact action, in this exact state, more likely."* It is entirely about the policy and says nothing about whether the action was any good — that judgement is $`\Psi_t`$'s job.

**Put real numbers in.** Say a softmax policy over three actions currently assigns $(0.5, 0.3, 0.2)$. You sample action 2 and the episode scores $\Psi = +4$. The update pushes $\theta$ along $+4\times$ the direction that raises $`\log\pi(a_2)`$ — so next time the distribution might be $(0.47, 0.35, 0.18)$. If instead $\Psi = -4$, it moves the other way. If $\Psi = 0$, nothing happens at all.

▸ **"Weighted maximum likelihood, where the weights are the returns" is the sentence to memorize.** Ordinary supervised learning does maximum likelihood on a fixed dataset of correct answers. Policy gradient does maximum likelihood on **its own past behaviour, with each example weighted by how well it turned out.** Good episodes become training data to imitate; bad episodes become training data to avoid. That reframing makes the whole family intuitive, and it is exactly the mental model to carry into RLHF (Ch. 16).

> **Where this came from.** The algorithm in its simplest form is **REINFORCE**, published by **Ronald J. Williams** in 1992 in the journal *Machine Learning*. The name is one of the more heroic acronyms in the field: it stands for **RE**ward **I**ncrement = **N**onnegative **F**actor $\times$ **O**ffset **R**einforcement $\times$ **C**haracteristic **E**ligibility — a compressed description of the update rule itself, backronymed into a word that also happens to mean what the algorithm does.
>
> The log-derivative trick is much older than RL and much more widely used than its RL name suggests. In operations research and simulation it is known as the **likelihood-ratio method** or the **score function estimator**, and it was developed for sensitivity analysis of stochastic simulations — asking how a queueing system's throughput responds to a parameter change. **The same estimator reappears in variational inference for discrete latent variables (Ch. 19)**, where its notorious variance is precisely why the reparameterization trick was invented as an alternative. Whenever you need a gradient through a sampling step, these are the only two options, and RL cannot use reparameterization because the environment is not differentiable.
>
> The **policy gradient theorem** in its general, discounted, function-approximation form is due to **Richard Sutton, David McAllester, Satinder Singh and Yishay Mansour** in 1999 — the result that established policy-based methods as a principled family rather than a heuristic.

### The choices of $`\Psi_t`$ — this is the whole design space

| $`\Psi_t`$ | Name | Bias | Variance |
|---|---|---|---|
| $R(\tau)$ | REINFORCE | none | enormous |
| $`\sum_{t'\ge t}r_{t'}`$ | reward-to-go | none | very high |
| $`\sum_{t'\ge t}r_{t'} - b(s_t)`$ | with baseline | none | high |
| $`Q^\pi(s_t,a_t)`$ | Q actor–critic | low | medium |
| $`A^\pi(s_t,a_t)`$ | **advantage AC** | low | **lowest** |
| $`r_t+\gamma V(s_{t+1})-V(s_t)`$ | TD residual | higher | very low |
| GAE($\lambda$) | §27.6 | tunable | tunable |

**Reward-to-go** uses causality: an action at time $t$ cannot affect rewards before $t$, so including them adds only variance.

#### Reading the $`\Psi_t`$ table

Every row is the same algorithm with a different answer to one question: **"how do I score the action I just took?"** The rows are ordered from "most honest, most useless" to "slightly biased, actually works."

| $`\Psi_t`$ | What it says | Why it's bad or good |
|---|---|---|
| $R(\tau)$ | "The whole episode scored 400, so every action in it was worth 400" | Blames the opening move for a mistake on turn 90. Unbiased and nearly unusable |
| $`\sum_{t'\ge t}r_{t'}`$ | "Only count rewards that came *after* this action" | Free improvement — the past is not caused by the present |
| $`\sum_{t'\ge t}r_{t'} - b(s_t)`$ | "…and compare it to how well this state usually goes" | Free again, and the biggest single win |
| $`Q^\pi(s_t,a_t)`$ | "Use a learned estimate instead of a noisy sample" | Trades sampling noise for approximation error |
| $`A^\pi(s_t,a_t)`$ | "Use how much better than average this action is" | The canonical choice, for the reasons in Ch. 26 §26.2 |
| $`r_t+\gamma V(s_{t+1})-V(s_t)`$ | "One step of reality, then trust the critic" | Lowest variance, most bias — TD taken to its extreme |
| GAE($\lambda$) | "Blend all of the above" | The dial. §27.6 |

▸ **Reward-to-go, made concrete.** An episode pays $(1, 0, 0, 5)$ and the action in question was taken at $t=2$. Plain REINFORCE credits it with the full $6$. Reward-to-go credits it with $5$ — dropping the $1$ that was collected *before* the action existed. **That $1$ was never affected by the action, so including it contributes exactly zero signal and a full share of noise.** Removing terms that cannot possibly depend on your decision is variance reduction with no cost whatsoever, which is a rare thing.

**The pattern down the table is one trade, made repeatedly:** replace something you *sampled* (honest, noisy) with something you *estimated* (biased, smooth). Every row after the third is buying variance reduction with a learned function's imperfection.

### Why a baseline is free — prove it

For any $b(s)$ not depending on $a$:
$$\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big] = b(s)\sum_a\pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)} = b(s)\,\nabla_\theta\underbrace{\sum_a\pi_\theta(a\mid s)}_{=1} = 0$$

▸ **Subtracting any state-dependent baseline leaves the gradient unbiased while reducing variance.** The variance-minimizing baseline is close to $V^\pi(s)$, which is why the advantage $A=Q-V$ is the canonical choice.

▸ **This is also exactly why GRPO works** (Ch. 16 §16.6): the mean reward over a group of completions for the same prompt is a valid, action-independent baseline — a Monte-Carlo estimate of $V(s)$ that replaces a learned value network.

#### Unpacking the baseline proof

The proof is three equalities and each one is a single move. Read it right to left, which is how it was constructed.

$$\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big] = b(s)\sum_a\pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)} = b(s)\,\nabla_\theta\underbrace{\sum_a\pi_\theta(a\mid s)}_{=1} = 0$$

1. **Pull $b(s)$ out.** It doesn't depend on $a$, and we are averaging over $a$, so it is a constant with respect to the averaging. Then write the expectation longhand as $`\sum_a \pi_\theta(a\mid s)[\cdot]`$, and expand $\nabla\log\pi$ back into $\frac{\nabla\pi}{\pi}$ — the log-derivative trick, run in reverse.
2. **The $`\pi_\theta(a\mid s)`$ cancels** top and bottom, leaving $`\sum_a \nabla_\theta \pi_\theta(a\mid s) = \nabla_\theta\sum_a\pi_\theta(a\mid s)`$.
3. **Probabilities sum to 1**, always, for every $\theta$. And the derivative of a constant is zero.

▸ **The whole proof rests on the single fact that a probability distribution sums to one no matter what the parameters are.** Nothing about rewards, dynamics, or neural networks enters. That is why the result is so robust: you can subtract *any* function of the state — a learned value network, a running average, the mean over a batch of samples, the number 7 — and the gradient stays exactly correct in expectation.

#### Why the variance reduction is so large — with numbers

Here is the situation the baseline is fixing. Suppose you are in a state where all three available actions lead to returns of about $1000$: specifically $1002$, $1000$, and $998$.

**Without a baseline**, the gradient sample is (score function) $\times$ (roughly $1000$). Every sampled action gets pushed *up* hard, and the only thing distinguishing a good action from a bad one is a 0.2% difference in the size of the push. The gradient's mean is correct — the normalization of the softmax means "push everything up" is a no-op on average — but each individual sample is dominated by a huge common term that carries no information about the choice.

**With baseline $b = 1000$**, the weights become $+2$, $0$, and $-2$. Now the good action is pushed up, the bad one is pushed down, and the magnitude of every update is $500\times$ smaller.

▸ **Variance scales with the square of the weight, so shrinking the typical weight from $1000$ to $2$ cuts the gradient's variance by a factor on the order of $(1000/2)^2 = 250{,}000$.** And the estimator is *still exactly unbiased*. This is not a marginal tuning improvement; without it, policy gradient methods essentially do not work on any problem where returns are large and differences are small — which is most of them.

> **Analogy.** Judging a diving competition by each diver's altitude above sea level. All the numbers are near 10 metres, all of them are technically correct, and none of them tells you who dived well. **Subtracting the height of the board is not a trick — it is the act of measuring the thing you actually care about.** The baseline does exactly that: it removes the part of the score that is about the *situation* rather than the *decision*.

▸ **And now GRPO makes sense in one sentence.** Ask a language model for eight answers to the same prompt, score them all, and use the **mean of those eight scores** as the baseline. It depends on the state (the prompt) and not on the action (which answer), so the proof above applies verbatim. **You have replaced an entire learned value network — a second model, with its own parameters, optimizer, and failure modes — with an average over samples you were computing anyway.**

---

## 27.5 Actor–critic

**Actor:** $`\pi_\theta`$, updated by the policy gradient.
**Critic:** $`V_\phi`$ or $`Q_\phi`$, updated by TD regression.

```
δ = r + γ V_φ(s') − V_φ(s)          # TD error, an estimate of the advantage
θ ← θ + α_θ ∇_θ log π_θ(a|s) · δ    # actor
φ ← φ + α_φ δ ∇_φ V_φ(s)            # critic
```

**A2C/A3C:** run many parallel environments to decorrelate data (A3C did it asynchronously; A2C synchronously and works just as well on a GPU). Add an **entropy bonus** $`+\beta H(\pi_\theta(\cdot\mid s))`$ to the objective — a direct, cheap, and effective exploration mechanism that prevents premature determinism.

#### Reading the actor–critic loop

Three lines, and they are worth reading as a conversation between two networks.

```
δ = r + γ V_φ(s') − V_φ(s)          # TD error, an estimate of the advantage
θ ← θ + α_θ ∇_θ log π_θ(a|s) · δ    # actor
φ ← φ + α_φ δ ∇_φ V_φ(s)            # critic
```

**Line 1 — the critic renders a verdict.** $\delta$ is the TD error of Ch. 26 §26.5: what actually happened minus what the critic predicted. Note the subscripts: **$\theta$ is the actor's weights, $\phi$ is the critic's weights.** Two separate parameter vectors, two separate networks (or two heads on one trunk).

▸ **Why $\delta$ is a legitimate advantage estimate.** Recall $A(s,a) = Q(s,a) - V(s)$, and $r + \gamma V(s')$ is a one-sample estimate of $Q(s,a)$. Substitute and you get exactly $\delta$. **The TD error *is* the advantage, sampled once** — a fact that looks like a coincidence and is actually the reason actor–critic exists as a coherent method.

**Line 2 — the actor takes the note.** If $\delta > 0$ the action beat expectations, so make it more likely; if $\delta < 0$, less likely. This is the policy gradient of §27.4 with $`\Psi_t = \delta_t`$.

**Line 3 — the critic corrects itself.** Ordinary TD regression: move $`V_\phi(s)`$ toward the target by $`\alpha_\phi \delta`$.

> **Analogy.** A footballer and a coach. The **coach** (critic) watches and says "that pass was better than I expected from that position" — one number, immediately, without waiting for the final score. The **player** (actor) adjusts their instincts accordingly. And the coach, watching how things actually unfold, keeps refining their own sense of what to expect. **Neither is trustworthy alone: an unrehearsed coach gives nonsense feedback, and a player with no feedback learns only from final scores, which arrive far too rarely.** They bootstrap each other, which is the appeal and also, as always, the risk.

▸ **Actor–critic is the exact point where the two families of §27.1 merge.** The actor supplies what value methods cannot (continuous actions, stochastic policies); the critic supplies what policy methods cannot (low-variance per-step feedback instead of one number per episode). PPO, SAC, TD3 and the algorithm behind RLHF are all instances of these three lines.

#### Parallel environments and the entropy bonus, decoded

**Why parallel environments do replay's job.** Policy gradients are **on-policy**: the derivation in §27.4 is only valid for data generated by the *current* $`\pi_\theta`$, so a replay buffer full of old policies' experience is not usable. But you still need decorrelated batches. The fix is to step 16 (or 64, or 1024) independent copies of the environment simultaneously and take one transition from each. **Sixteen environments at different points in different episodes give you sixteen  unrelated samples — the same decorrelation replay provides, obtained through space instead of through time.**

A3C did this **asynchronously**: separate workers each with their own copy, pushing gradients to a shared parameter server whenever they finished. A2C does it **synchronously**: step all environments, batch, one update. Synchronous turned out to work just as well and vectorizes cleanly on a GPU, so the asynchrony — the "A" that gave A3C its name — quietly turned out not to be the important part.

**The entropy bonus, with numbers.** $H(\pi)$ is the entropy of the action distribution (Ch. 1 §1.4) — read it as *"how undecided is the policy?"*

| Policy over 3 actions | Entropy (nats) |
|---|---|
| $(1/3, 1/3, 1/3)$ — maximally unsure | $\ln 3 = 1.099$ |
| $(0.9, 0.05, 0.05)$ — fairly confident | $0.394$ |
| $(1, 0, 0)$ — fully committed | $0$ |

Adding $+\beta H$ to the objective (typically $\beta \approx 0.01$) pays the agent a small bribe for staying undecided.

▸ **Why this matters more than it sounds: policy collapse is an absorbing state.** If $`\pi_\theta(a\mid s)`$ drifts to $0.0001$, you will sample that action roughly once in ten thousand visits — so you almost never get a gradient for it, so its probability almost never recovers. **An action whose probability reaches zero is deleted from the agent's universe, permanently, regardless of how good it was.** The entropy bonus is a floor that prevents that deletion. It costs one term in the loss and prevents an entire class of silent, unrecoverable failure.

> **Where this came from.** **A3C** (Volodymyr Mnih and colleagues at DeepMind, 2016) was startling for a reason unrelated to its algorithm: it matched or beat DQN on Atari while training on **a single multi-core CPU**, in less wall-clock time than DQN needed on a GPU. At a moment when the field's assumption was that deep RL required expensive accelerators, a paper showing that parallel CPU workers sufficed changed what people believed was possible on a modest budget — and the synchronous simplification (A2C) that followed made even the asynchrony unnecessary.

---

## 27.6 Generalized Advantage Estimation

We need an advantage estimate that trades bias against variance smoothly.

Define the TD residual $`\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)`$. The $k$-step advantage estimator is
$$\hat A_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$$

($k=1$: low variance, high bias from the imperfect $V$. $k=\infty$: unbiased, huge variance.)

**GAE** is the exponentially-weighted average of all of them:

▸ $$\boxed{\ \hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}\ }$$

- $\lambda=0$: $`\hat A_t=\delta_t`$ — maximum bias, minimum variance.
- $\lambda=1$: $`\hat A_t=\sum_l\gamma^l r_{t+l} - V(s_t)`$ — unbiased Monte Carlo.
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

#### Reading GAE in plain English

$$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}$$

Read aloud: *"A-hat-t equals the sum over l of gamma-lambda to the l, times delta t-plus-l."* In English: **"add up all the future surprises, discounting each one a little more the further ahead it is."**

That is the whole formula. One symbol at a time:

- $`\delta_{t+l} = r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l})`$ — the surprise at step $t+l$. Positive means "better than the critic expected."
- $(\gamma\lambda)^l$ — the fading weight. Note it is the **product** of the two, so both parameters shrink the window.
- $`\hat A_t`$ — the hat means **estimate**. You are estimating the advantage, not computing it.

▸ **The core intuition: how good was your action at time $t$? Look at every surprise that followed it, and give more weight to the surprises that came soon after.** A pleasant surprise three steps later is probably something you caused. A pleasant surprise three hundred steps later probably is not.

**Put numbers in.** With the standard $\gamma = 0.99$ and $\lambda = 0.95$, the decay per step is $\gamma\lambda = 0.9405$:

| Steps ahead $l$ | Weight $(\gamma\lambda)^l$ |
|---|---|
| 0 | $1.000$ |
| 1 | $0.941$ |
| 5 | $0.735$ |
| 20 | $0.293$ |
| 50 | $0.047$ |

The effective window is $\frac{1}{1-\gamma\lambda} \approx 17$ steps. **Your action is credited with roughly the next seventeen steps' worth of surprise, tapering out.** Beyond about fifty steps its influence is negligible, which is a sensible statement about causality in most environments.

#### Why $\lambda=1$ really is Monte Carlo — the telescoping

The claim "$\lambda=1$ gives $`\hat A_t=\sum_l\gamma^l r_{t+l} - V(s_t)`$" looks like it needs work, and it does not. Substitute the definition of $\delta$ and watch the value terms cancel:

$$\sum_l \gamma^l\big[r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\big] = \sum_l \gamma^l r_{t+l} + \sum_l\big[\gamma^{l+1} V(s_{t+l+1}) - \gamma^{l} V(s_{t+l})\big]$$

**Every value term destroys its neighbour** — the $`+\gamma^1 V(s_{t+1})`$ from $l=0$ is cancelled by the $`-\gamma^1 V(s_{t+1})`$ from $l=1$, and so on — leaving only $`-V(s_t)`$ from the very first term. What survives is "actual discounted rewards, minus the critic's opening prediction," which is precisely the Monte Carlo advantage.

▸ **This is the same telescoping trick as potential-based reward shaping (Ch. 26 §26.9), and the same structure as the driving-home story.** A chain of one-step revisions to a forecast, summed, equals the total revision from start to finish. It is worth recognising because it appears repeatedly once you know the shape.

#### Reading the backward recursion

```
adv = 0
for t in reversed(range(T)):
    delta = r[t] + gamma*V[t+1]*(1-done[t]) - V[t]
    adv = delta + gamma*lam*(1-done[t])*adv
    A[t] = adv
```

**Why backwards?** Because $`\hat A_t`$ depends on everything *after* $t$. Written as a recursion, $`\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}`$ — *"my advantage is my own surprise, plus a discounted copy of the advantage of the step after me."* Once you have $`\hat A_{t+1}`$, computing $`\hat A_t`$ is one multiply and one add. Walking the trajectory from the end gives you all $T$ advantages in $O(T)$ time.

▸ **A naive forward implementation would be $O(T^2)$** — for each of $T$ timesteps, sum over all subsequent steps. For $T=2048$ (the standard PPO rollout length) that is 4 million operations instead of 2 thousand. **The recursion is not a micro-optimization; it is what makes GAE free.**

**The `(1-done[t])` factor is the episode-boundary guard.** When an episode ends, `done` is 1 and the factor becomes 0, which does two things at once: it zeroes $V[t+1]$ (there is no next state, so its value is not $V$ but $0$) and it zeroes the carried-forward `adv` (credit must not leak from one episode into the previous one). **Getting this wrong is one of the most common and hardest-to-spot bugs in a PPO implementation**, because the code still runs and the agent still improves — just more slowly and less reliably than it should.

#### $\gamma$ and $\lambda$ are not the same kind of thing

This is the distinction the book flags, and it is worth stating twice.

| | $\gamma$ | $\lambda$ |
|---|---|---|
| Belongs to | **The problem** | **The estimator** |
| Changing it changes | What the optimal policy *is* | Only how accurately you measure |
| Analogous to | Choosing your objective | Choosing your measuring instrument |

▸ **Change $\gamma$ and you have changed the task**: a $\gamma$ of $0.9$  prefers a different policy than a $\gamma$ of $0.999$, because it  values the future differently. **Change $\lambda$ and the task is identical** — you are only trading bias against variance in your estimate of a quantity that was there all along. Reporting a result at "$\gamma=0.99$" is reporting what you optimized; reporting "$\lambda=0.95$" is reporting how you measured it.

> **Where this came from.** GAE is due to **John Schulman** and colleagues at Berkeley in 2015–16, and it is the estimator half of the same research programme that produced TRPO and then PPO in the following two years. The idea is explicitly TD($\lambda$) (Ch. 26 §26.5) transplanted from values to advantages — **the paper's contribution is largely the recognition that a 1988 idea about prediction applies directly to a 2015 problem about control**, plus the careful empirical work showing $\lambda \approx 0.95$ to be a remarkably robust default.

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

- **If $`\hat A_t>0`$** (a good action): the unclipped term grows with $\rho$; the clipped term saturates at $(1+\epsilon)\hat A$. The $\min$ picks the smaller, so **the objective stops rewarding you past $\rho=1+\epsilon$.** No incentive to move further.
- **If $`\hat A_t<0`$** (a bad action): the terms are negative; the $\min$ picks the *more negative*, which is the unclipped term when $\rho>1$. So **there is no ceiling on pushing a bad action's probability down**, but once $\rho<1-\epsilon$ the clip kicks in and the gradient vanishes.

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

#### Why "distance in parameter space" is the wrong distance

TRPO's whole design rests on one observation, and it is easiest to see with numbers.

Take a softmax policy over two actions with logits $`(z_1, z_2)`$, and add exactly $1$ to the first logit:

| Starting logits | Policy before | Policy after $`z_1 \mathbin{+}= 1`$ | How much changed |
|---|---|---|---|
| $(0,\,0)$ | $(0.500,\ 0.500)$ | $(0.731,\ 0.269)$ | **Enormously** |
| $(10,\,0)$ | $(0.99995,\ 0.00005)$ | $(0.99998,\ 0.00002)$ | **Almost nothing** |

▸ **The same step in parameter space produced a dramatic behaviour change in one case and no behaviour change at all in the other.** A learning rate is a promise about $\lVert\Delta\theta\rVert$, and $\lVert\Delta\theta\rVert$ tells you nothing about how much the agent's actual conduct changed. In supervised learning that is a nuisance. In RL it is fatal, because a policy that changes drastically starts generating a different data distribution, and **there is no fixed dataset to fall back on when it goes wrong.**

> **Analogy.** A steering wheel with variable ratio. Ten degrees of wheel might be a gentle lane change at motorway speed and a full spin in a car park. If your instruction is "turn the wheel ten degrees per second," you will be fine most of the time and occasionally destroy the car. **The fix is to specify the manoeuvre, not the wheel angle** — constrain how much the *trajectory* changes, and let the wheel do whatever it needs to.

#### Reading the TRPO objective

$$\max_\theta\ \mathbb{E}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}\hat A\right]\quad\text{s.t.}\quad \mathbb{E}_s\big[\mathrm{KL}(\pi_{\theta_{\text{old}}}\,\lVert\,\pi_\theta)\big]\le\delta$$

Read aloud: *"maximize the expected ratio times advantage, subject to the expected KL divergence between old and new policies being at most delta."* "s.t." means **"subject to"** — everything after it is a constraint the answer must satisfy, not part of the thing being maximized.

- **The ratio** $`\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}`$ is the importance-sampling correction of Ch. 26 §26.7, over a *single* action rather than a whole trajectory. It lets you evaluate a proposed new policy using data the old one collected.
- **The constraint** measures distance in **policy space** using KL divergence (Ch. 1 §1.4), which is scale-free and does not care how the parameters happen to be arranged. $\delta$ is typically about $0.01$ — a very small permitted change per update.

▸ **"Trust region" means exactly what it says: a neighbourhood around the current policy inside which your approximation is trustworthy.** Optimize freely inside it; refuse to leave it.

**The natural gradient, in one paragraph.** To enforce a KL constraint efficiently you approximate KL locally as a quadratic form, and its curvature matrix is the **Fisher information matrix** $F$. The resulting step direction is $F^{-1}g$ instead of plain $g$ — the **natural gradient** of Ch. 4 §4.7. Read it as: *"the gradient, re-expressed in units of how much the policy actually changes rather than how much the numbers change."* Computing $F^{-1}$ explicitly is impossible for a large network, so TRPO uses conjugate gradient with Hessian-vector products, which never forms the matrix at all — then a line search to make sure the constraint holds exactly rather than approximately.

**And why nobody uses it.** Second-order machinery, conjugate gradient loops, a line search, and awkward interactions with shared actor–critic trunks, dropout and BatchNorm (which make the "same policy" produce different outputs on different forward passes, breaking the KL computation). **TRPO is a correct answer that is unpleasant to implement.** PPO is what happens when you ask for 95% of the benefit with 5% of the machinery.

#### Reading the PPO clip — the part people get wrong

$$\mathcal{L}^{\text{CLIP}}(\theta)=\mathbb{E}_t\left[\min\Big(\rho_t\hat A_t,\ \ \mathrm{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,\hat A_t\Big)\right]$$

$\mathrm{clip}(x, l, u)$ is read *"clip x to between l and u"* — return $l$ if $x < l$, $u$ if $x > u$, otherwise $x$ unchanged. With $\epsilon=0.2$ the window is $[0.8,\ 1.2]$: the new policy may make an action at most 20% more or 20% less likely than the old one before the brakes engage.

**Here is the table that makes the $\min$ obvious.** Set $\epsilon = 0.2$ and work out both terms.

**Case $`\hat A_t = +1`$ (the action was good — we want $\rho$ to go up):**

| $\rho$ | Unclipped $\rho\hat A$ | Clipped term | $\min$ | Gradient? |
|---|---|---|---|---|
| $0.5$ | $0.50$ | $0.80$ | $0.50$ | **Yes** — pushes $\rho$ up |
| $1.0$ | $1.00$ | $1.00$ | $1.00$ | Yes |
| $1.2$ | $1.20$ | $1.20$ | $1.20$ | Yes |
| $1.5$ | $1.50$ | $1.20$ | $1.20$ | **No** — flat, gradient is zero |

**Case $`\hat A_t = -1`$ (the action was bad — we want $\rho$ to go down):**

| $\rho$ | Unclipped $\rho\hat A$ | Clipped term | $\min$ | Gradient? |
|---|---|---|---|---|
| $0.5$ | $-0.50$ | $-0.80$ | $-0.80$ | **No** — flat, gradient is zero |
| $1.0$ | $-1.00$ | $-1.00$ | $-1.00$ | Yes |
| $1.5$ | $-1.50$ | $-1.20$ | $-1.50$ | **Yes** — keeps pushing $\rho$ down |

▸ **Read the two tables together and the asymmetry appears.** Going *too far* in the rewarded direction gets you cut off (row 4 of the first table, row 1 of the second). Coming *back* toward the old policy is never cut off (row 1 of the first table, row 3 of the second). **The clip is a one-way valve: it removes the incentive to run away, and never removes the ability to return.** That is what "pessimistic lower bound" means in practice, and it is the reason the $\min$ is there rather than a plain clip.

> **Analogy.** A speed limiter that stops the accelerator working past 70 mph but leaves the brakes fully functional at every speed. You cannot be rewarded for going faster than the limit, but if you find yourself at 90 you can still slow down as hard as you like. **A symmetric clip — one that also disabled the brakes — would trap the policy wherever a bad update had left it.**

#### Why the whole algorithm hangs together

The loop is: collect $T$ steps from $N$ parallel actors → compute GAE → run $K$ epochs of minibatch SGD on that same data → throw it away → repeat.

**The tension it resolves:** policy gradients are on-policy, so strictly speaking each batch of data is valid for exactly one gradient step, and one gradient step per rollout is agonizingly wasteful. **The clip is what buys you permission to take more.** As long as the policy stays inside $[1-\epsilon, 1+\epsilon]$ of where the data came from, the importance-sampled objective remains a reasonable approximation; the moment it leaves, the gradient switches off. **PPO reuses on-policy data by making "too far off-policy" a condition the loss detects and neutralizes by itself.**

▸ **That is the one-sentence summary, and it is also the design principle: instead of *preventing* a bad step with a constraint (TRPO), make the objective *stop rewarding* the bad step (PPO).** A first-order method with a cleverly shaped loss replaced a second-order method with a hard constraint, at almost no cost in quality. That trade — constraint into loss shape — is worth recognising, because it recurs throughout modern machine learning.

#### The eight implementation details, decoded

The book is right that these matter as much as the algorithm. Briefly, each one and why:

| Detail | What it fixes |
|---|---|
| **Advantage normalization** per minibatch | Subtract mean, divide by std, so the gradient scale is independent of reward magnitude and the effective learning rate stops drifting during training |
| **Observation normalization** (running mean/std) | The same problem on the input side — a network fed pixel values in $[0,255]$ and joint angles in $[-0.1, 0.1]$ cannot use one learning rate for both |
| **Reward scaling** by the return's running std | Keeps the critic's regression targets a sane size as the agent gets better and returns grow |
| **Orthogonal init**, policy head gain $0.01$ | Tiny initial logits mean a near-uniform initial policy — maximum entropy, no accidental early commitment (§27.5) |
| **Value-loss clipping** | The same trust-region idea applied to the critic, so the critic cannot lurch either |
| **Gradient clipping** at $0.5$ | A hard ceiling on step size for the rare enormous batch |
| **LR annealing** | Large steps while everything is wrong, small steps while polishing |
| **Adam $\epsilon = 10^{-5}$** instead of $10^{-8}$ | Adam divides by $\sqrt{v}+\epsilon$; when gradients are tiny — routine in RL, where most advantages are near zero — a minuscule $\epsilon$ lets that denominator vanish and produces enormous effective steps. Raising $\epsilon$ puts a floor under it |

▸ **Notice that seven of the eight are about *scale*.** Deep RL has no fixed dataset and no fixed target, so every quantity in the system — observations, rewards, returns, advantages, gradients — drifts in magnitude over training. **Most of the "37 implementation details" are, in aggregate, one idea: normalize everything, constantly, because nothing holds still.**

> **Where this came from.** **TRPO (2015) and PPO (2017) are both John Schulman's**, the second explicitly a simplification of the first — a rare and admirable case of an author publishing the easier method that makes their own harder one obsolete. PPO went on to become the default policy-gradient algorithm essentially everywhere: OpenAI Five's Dota 2 agent, countless robotics results, and — most consequentially — the reinforcement-learning stage of **RLHF for large language models** (Ch. 16). **A 2017 algorithm designed for simulated robots and Atari is now, by a wide margin, the most heavily deployed reinforcement-learning method in the world**, because it happened to be the one running when instruction-tuned language models arrived.
>
> The list of implementation details comes from a body of reproducibility work — most visibly a widely-read blog-track paper cataloguing 37 of them — motivated by a  embarrassing discovery: independent reimplementations of PPO, all faithful to the published equations, produced wildly different results. **The equations were not the algorithm. The codebase was.** That finding did a great deal to make the field take reproducibility seriously (Ch. 3).

---

## 27.8 Continuous control

### DDPG

Deterministic policy $`\mu_\theta(s)`$; the critic's gradient flows through the actor via the chain rule:
▸ $$\nabla_\theta J = \mathbb{E}\Big[\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}\ \nabla_\theta\mu_\theta(s)\Big]$$

Off-policy with replay and target networks. Exploration by adding noise to the action. **Notoriously brittle** and sensitive to hyperparameters.

### TD3 — three fixes to DDPG

1. **Clipped double Q:** learn two critics, use $`\min(Q_1,Q_2)`$ in the target. Directly counters overestimation bias — a deliberately *pessimistic* estimate.
2. **Delayed policy updates:** update the actor once per two critic updates, so the actor chases a more converged critic.
3. **Target policy smoothing:** add clipped noise to the target action, $`a'=\mu_{\theta^-}(s')+\mathrm{clip}(\epsilon,-c,c)`$. This smooths the value estimate over nearby actions and prevents the actor from exploiting sharp, spurious peaks in $Q$.

### SAC — maximum entropy RL

▸ $$J(\pi)=\sum_t\mathbb{E}\big[r(s_t,a_t)+\alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big]$$

**Maximize reward *and* entropy.** Consequences: better exploration (the policy stays stochastic), robustness (it learns multiple ways to succeed rather than one brittle way), and much less hyperparameter sensitivity.

The soft value functions become
$$Q(s,a)=r+\gamma\,\mathbb{E}_{s'}\big[V(s')\big],\qquad V(s)=\mathbb{E}_{a\sim\pi}\big[Q(s,a)-\alpha\log\pi(a\mid s)\big]$$

and the optimal policy is a **Boltzmann distribution** $`\pi^*(a\mid s)\propto\exp\!\big(\tfrac1\alpha Q(s,a)\big)`$.

▸ **That is exactly the KL-regularized optimum from Ch. 16 §16.5 with a uniform reference policy.** RLHF's KL-to-reference and SAC's entropy bonus are the same mathematical object; recognizing this is a strong cross-domain connection to be able to draw.

**Implementation:** twin critics with the min; a squashed Gaussian policy $a=\tanh(\mu+\sigma\epsilon)$ trained via the reparameterization trick (Ch. 19 §19.3); and **automatic temperature tuning** — adjust $\alpha$ to hold the entropy at a target $\bar{\mathcal H}$ (typically $-\dim(\mathcal A)$) by minimizing $\mathbb{E}[-\alpha(\log\pi + \bar{\mathcal H})]$.

▸ **SAC is the default choice for continuous control**: off-policy (sample-efficient), stable, and largely tune-free.

#### Why continuous actions need a different idea entirely

Everything in §§27.2–27.3 ended with $`\arg\max_a Q(s,a)`$: scan every action, pick the best. **A robot arm with seven joints, each taking any real value, has infinitely many actions.** You cannot scan them. So value-based methods, as written, simply do not apply.

DDPG's answer is one of the more elegant moves in the chapter: **if you cannot search for the best action, differentiate your way toward it.**

$$\nabla_\theta J = \mathbb{E}\Big[\underbrace{\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}}_{\text{"which way should the action move?"}}\ \underbrace{\nabla_\theta\mu_\theta(s)}_{\text{"which way should }\theta\text{ move to do that?"}}\Big]$$

Read aloud: *"the gradient of J with respect to theta equals the expectation of: the gradient of Q with respect to a, evaluated at a equals mu-theta of s, times the gradient of mu-theta with respect to theta."* The vertical bar with a subscript, $`\big|_{a=\mu_\theta(s)}`$, means **"evaluated at"** — compute the derivative first, then plug in this particular action.

This is nothing but the **chain rule**, applied across the seam between two networks.

> **Analogy.** The critic is a hillside drawn over the space of possible actions, and the actor is a knob that positions you on that hillside. Ask the hill which way is up ($`\nabla_a Q`$), then ask the knob how to turn to move you that way ($`\nabla_\theta\mu`$). **The gradient physically flows out of the critic, across the action, and into the actor's weights.**

▸ **Notice what is absent: there is no $\log\pi$ and no log-derivative trick.** Those existed in §27.4 only because the policy was random and you cannot differentiate through a coin flip. **A deterministic policy has no coin flip, so the gradient passes straight through** — which is far lower variance, and is exactly why DDPG can be dramatically more sample-efficient than REINFORCE. The price is that a deterministic policy explores nothing by itself, so exploration has to be bolted on as additive noise, and a policy that must be told how to explore is a policy that is easy to mistune. Hence "notoriously brittle."

#### TD3's three fixes, decoded

**TD3** stands for **T**win **D**elayed **D**DPG, and the name lists two of the three fixes.

**1. Clipped double Q.** Learn two critics and use $`\min(Q_1, Q_2)`$ in the target. This is Double Q-learning (Ch. 26 §26.6) pushed one step further: rather than merely decoupling selection from evaluation, **take the more pessimistic of two independent opinions.**

▸ **Why deliberate underestimation is safe when overestimation is not.** The $\max$ in the Bellman target actively *seeks out* whichever estimate is too high, so overestimation is amplified and propagated backwards through bootstrapping. Underestimation gets no such amplification — the $\max$ does not hunt for it. **The asymmetry of the error is what makes a biased-low estimator the safer choice**, and this reasoning reappears in offline RL (§27.10) as conservatism.

**2. Delayed policy updates.** Update the actor once per two critic updates. The actor is climbing a hill that the critic is still drawing; if the actor moves as fast as the map is being redrawn, it chases artefacts. **Slowing one side of a two-sided feedback loop is the same medicine as the target network in §27.2.**

**3. Target policy smoothing.** Add clipped noise to the *target* action: $`a'=\mu_{\theta^-}(s')+\mathrm{clip}(\epsilon,-c,c)`$.

▸ **This is the most interesting of the three, because it is a regularizer disguised as noise.** A neural critic fitted to finite data will have sharp spurious peaks — narrow spikes where $Q$ happens to be large for no good reason. A deterministic actor doing gradient ascent on $Q$ will find those spikes and sit on them, because finding maxima is precisely its job. Averaging the target over a small neighbourhood of actions **smooths those spikes away before the actor can exploit them.**

> **Analogy.** Do not trust a single unusually high reading from a noisy instrument; average over a few nearby measurements first. A deterministic actor with no smoothing is a machine specifically built to seek out and believe the noisiest reading available.

#### SAC and maximum-entropy RL, decoded

$$J(\pi)=\sum_t\mathbb{E}\big[r(s_t,a_t)+\alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big]$$

Read: *"the objective is the sum over time of expected reward, plus alpha times the entropy of the policy."* **The agent is paid for collecting reward and paid, separately, for remaining undecided.** $\alpha$ sets the exchange rate, and is called the **temperature**.

This changes the optimal policy. The solution is no longer "always take the best action" but a **Boltzmann distribution**:

$$\pi^*(a\mid s)\propto\exp\!\Big(\tfrac1\alpha Q(s,a)\Big)$$

Read: *"the probability of an action is proportional to e-to-the Q-over-alpha."* $\propto$ means "proportional to" — equal after dividing by whatever makes it sum to 1 (§0.11). **This is a softmax over Q-values with temperature $\alpha$**, and it is the same Boltzmann distribution from 19th-century thermodynamics that gives softmax its name (Ch. 1).

**Put numbers on the temperature.** Two actions with $Q = 10$ and $Q = 8$:

| $\alpha$ | $`\pi^*`$ | Behaviour |
|---|---|---|
| $0.1$ | $(0.9999,\ 0.0001)$ | Effectively greedy |
| $1$ | $(0.88,\ 0.12)$ | Prefers the better one, still tries the other |
| $10$ | $(0.55,\ 0.45)$ | Nearly indifferent |

▸ **As $\alpha\to0$ the Boltzmann policy becomes the $\arg\max$ and SAC reduces to ordinary RL.** The entropy term is a continuous dial between "commit" and "keep your options open," and unlike an $\epsilon$-greedy schedule, it is part of the objective rather than bolted on — so the agent's *optimal* behaviour includes exploration, instead of exploration being a deviation from optimal behaviour.

**Why maximizing entropy buys robustness.** A policy pushed toward high entropy must find **several** ways to succeed, not one — because a single narrow solution has low entropy and is therefore penalized. When the world shifts slightly (a joint stiffens, a surface changes friction) an agent that knows three ways to do the task degrades gracefully, and an agent that knows one does not.

**The implementation details, decoded:**

- **Twin critics with the min** — TD3's pessimism, imported wholesale.
- **A squashed Gaussian policy $a=\tanh(\mu+\sigma\epsilon)$** — sample $\epsilon$ from a standard normal, scale and shift it by the network's outputs, then squash through $\tanh$ into the valid action range $[-1,1]$. Crucially, **the randomness enters as an input $\epsilon$ rather than as a sampling step**, so the gradient flows straight through — this is the **reparameterization trick** of Ch. 19 §19.3, and it is why SAC gets low-variance gradients through a *stochastic* policy, which DDPG could only manage by giving up stochasticity entirely.
- **Automatic temperature tuning** — rather than hand-tuning $\alpha$, set a target entropy (usually $-\dim(\mathcal{A})$, so $-6$ for a six-joint arm) and adjust $\alpha$ up whenever the policy is too decisive and down when it is too vague. ▸ **This converts the single most sensitive hyperparameter into a thermostat**, and it is the main reason SAC has a reputation for working out of the box.

▸ **And the cross-domain connection the book flags is worth stating in full.** SAC maximizes $r + \alpha\mathcal{H}(\pi)$; RLHF maximizes $`r - \beta\,\mathrm{KL}(\pi\,\lVert\,\pi_{\text{ref}})`$ (Ch. 16 §16.5). Since $\mathrm{KL}(\pi\,\lVert\,\text{uniform}) = -\mathcal{H}(\pi) + \text{const}$, **entropy regularization is KL-regularization against a uniform reference.** They are the same objective with a different choice of what to stay close to: SAC stays close to "no opinion," RLHF stays close to "the pretrained model." **Recognizing that one identity connects continuous robot control to language-model alignment**, and it is exactly the kind of link worth being able to draw on demand.

> **Where this came from.** **DDPG** (Timothy Lillicrap and colleagues at DeepMind, 2016) built directly on the **deterministic policy gradient theorem** proved by David Silver and colleagues in 2014 — the result establishing that a deterministic policy has a well-defined gradient at all, which was not obvious. **TD3** came from Scott Fujimoto, Herke van Hoof and David Meger in 2018, and its lasting contribution is as much diagnostic as algorithmic: it *demonstrated* the overestimation in DDPG's critics rather than assuming it, then fixed it. **SAC** came from Tuomas Haarnoja and colleagues at Berkeley in 2018, out of a research line on maximum-entropy control that had been running for years before deep RL existed. **All three appeared within about three years of each other, and the field's default for continuous control moved twice in that window** — a good illustration of how fast this particular corner was moving.

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

#### Examples and non-examples: what the reward actually is

Reinforcement learning attracts more conceptual confusion than any other topic in this book, and most of it concentrates here.

**✅ Correct statements about reward**

| Statement | Why it's right |
|---|---|
| The reward defines what the agent optimizes | It is the *entire* specification of the objective |
| The agent maximizes **expected cumulative discounted** reward | Not immediate reward, and not a single episode |
| A reward function can be satisfied in ways you didn't intend | Specification gaming is the default, not the exception |

**❌ Near-misses — near-universal misconceptions**

| Common belief | Why it's wrong |
|---|---|
| "The reward is the agent's goal" | The agent has no goals. It has a **gradient**. It optimizes the reward you wrote, not the one you meant |
| "Reward tells the agent what to do" | It tells it how well it did, *after* the fact. That's why credit assignment is hard |
| "Off-policy means offline" | **Off-policy** = learning from a different policy's data. **Offline** = no environment interaction at all. DQN is off-policy and online |
| "The value function is the reward" | Value is *expected future return from here*. Reward is the immediate signal |
| "Higher discount $\gamma$ is always better" | Larger $\gamma$ means longer horizons *and* higher variance and slower convergence |
| "The policy gradient is the gradient of the true objective" | It's an *estimator* — unbiased but famously high-variance, which is why baselines and GAE exist |
| "More exploration is always safer" | Exploration has a real cost; the trade-off is the point |

▸ **The boundary:** the reward is a **scalar score, delivered after the fact, that the agent will maximize by any route available.** It is not instruction, not intention, and not a goal. **Every reward-hacking story — the boat spinning in circles to collect checkpoint pickups instead of finishing the race — is this distinction being learned the hard way.**

> **Common misconception.** *"Deep RL failed to reproduce because the paper was wrong."* Deep RL results are notoriously seed-sensitive: the *same* algorithm on the *same* task with different random seeds can produce wildly different learning curves, sometimes bimodal between "solved" and "never learned." Reporting a single run, or the best of several, was common practice for years. **If you see an RL curve without a seed count and a variance band, you cannot interpret it** — which is Chapter 3's argument arriving in a new setting.

> **Common misconception.** *"The target network is a minor implementation detail."* Without it, the network is chasing a target computed from its own rapidly-changing weights — you are regressing toward a moving goalpost that moves *because* you moved. Removing target networks doesn't degrade DQN gracefully; it frequently diverges outright. The same is true of replay: the "deadly triad" of function approximation, bootstrapping, and off-policy learning has no general convergence guarantee, and these devices are what make it work in practice anyway.

---

## Did you know?

- **The "deadly triad" has no general convergence guarantee, and deep RL uses all three anyway.** Function approximation, bootstrapping, and off-policy learning combined can provably diverge. Target networks, replay buffers, and careful tuning are engineering that makes an unguaranteed method work in practice — which is why deep RL feels more fragile than supervised learning. It  is.

- **DQN learned to play Atari from raw pixels with the same network and hyperparameters across 49 games.** The 2015 *Nature* paper from DeepMind was striking not because it played well but because *one* configuration generalized across wildly different games with no per-game tuning.

- **The famous CoastRunners boat is the canonical reward-hacking story.** Asked to maximize score in a boat race, an agent discovered it could circle a lagoon collecting respawning pickups forever, scoring higher than any competitor while never finishing the race — and repeatedly catching fire. The reward was maximized exactly as written.

- **Q-learning was proved to converge in 1989, in Chris Watkins's PhD thesis** — but only for tabular representations. Every convergence guarantee you learn in the foundations chapter evaporates the moment you replace the table with a neural network.

- **Experience replay was inspired by biology.** The idea of storing and re-playing past transitions echoes hippocampal replay during sleep, in which sequences of place-cell firing are re-run. The analogy motivated the technique, though the mechanisms are only loosely related.

- **PPO's dominance is a story about practicality, not optimality.** TRPO had the stronger theory — a  trust region with monotonic-improvement guarantees — but required second-order machinery. PPO replaced it with a clipped ratio that approximates the same effect in a few lines of first-order code, and won the field almost entirely. It is also the algorithm behind RLHF in Chapter 16.

- **Policy gradients are unbiased but so high-variance that they're unusable raw.** Nearly everything in an actor–critic method — baselines, advantage estimation, GAE's $\lambda$ — exists to trade a little bias for a large reduction in variance. The bias/variance decomposition of Chapter 2 shows up here as an engineering discipline rather than a theoretical result.

- **Subtracting a baseline from the reward does not change the gradient in expectation**, but can reduce its variance enormously. This is one of the most elegant results in RL: you get a free variance reduction because the baseline term has zero expected gradient contribution.

- **AlphaGo's move 37 against Lee Sedol** was assessed by commentators as a mistake, and its estimated probability of being played by a human was about one in ten thousand. It was, on later analysis, decisive. It became the standard reference point for machine-discovered strategies outside human convention.

- **Offline RL fails in a specific and instructive way.** Trained on a fixed dataset, the value function extrapolates confidently to actions never present in the data, assigns them high values, and the policy learns to prefer exactly those unverifiable actions. Conservative Q-Learning and Implicit Q-Learning exist to penalize that extrapolation — the whole subfield is organized around one failure mode.

- **A discount factor of $\gamma = 0.99$ corresponds to an effective horizon of about 100 steps** ($1/(1-\gamma)$). Practitioners tune $\gamma$ as if it were a learning-rate-like knob, but it is really a statement about how far ahead the agent is being asked to care — and it changes the problem being solved, not merely how fast it's solved.

- **Reinforcement learning was shaped by animal psychology.** Richard Sutton and Andrew Barto drew explicitly on the behaviourist literature — temporal difference learning has direct antecedents in models of classical conditioning, and TD error turned out to correspond remarkably well to dopamine neuron firing patterns observed by Wolfram Schultz and colleagues. The theory predicted the neuroscience.

---

## Check for Understanding

**Deep RL is the Bellman equation plus a neural network plus a large collection of devices for keeping the resulting feedback loop stable — target networks and replay for value methods, baselines and GAE for variance, and trust regions or clipping for policy methods — and PPO dominates because its clipped ratio makes it safe to take several gradient epochs on the same batch of on-policy data.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is "the reward is the agent's goal" wrong**, and what is the reward actually?
2. **What is the deadly triad**, and why does deep RL use all three parts despite the lack of guarantees?
3. **Why does a target network help?** (What goes wrong when the target moves because you moved?)
4. **What does experience replay fix**, and why is correlated sequential data a problem?
5. **Distinguish off-policy from offline.** Which is DQN?
6. **Why are policy gradients unbiased but nearly unusable in raw form?**
7. **Why does subtracting a baseline reduce variance without introducing bias?**
8. **What does GAE's $\lambda$ trade off?** Where have you seen this trade before?
9. **What does PPO's clipping actually prevent**, and why does that let you reuse a batch for several epochs?
10. **Why did PPO beat TRPO despite TRPO having better theory?**
11. **Why does offline RL fail on actions absent from the dataset**, and what do CQL and IQL do about it?
12. **What does $\gamma = 0.99$ mean in units of time steps**, and why is changing it changing the problem?

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 28 — Vision Transformers & Multimodal Models](28-vision-transformers-and-multimodal.md)
