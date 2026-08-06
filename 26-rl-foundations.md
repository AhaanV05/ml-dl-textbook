# Chapter 26 — Reinforcement Learning Foundations

> **Prerequisites:** Ch. 1 (§1.3 probability), Ch. 4.

---

## 26.1 The setting

### The one-line idea

An agent acts in an environment, receives rewards, and must learn a behaviour that maximizes cumulative reward — without being told what the right actions are, and with its own actions determining what data it ever sees.

### The analogy

Learning to cook without a recipe or a teacher, judged only by whether the meal tastes good at the end. Nobody tells you "you should have added salt at minute six." You get one number after twenty minutes of work, and you must figure out which of your fifty decisions deserves the credit. That is the **credit assignment problem**, and it is what makes RL fundamentally harder than supervised learning.

### What makes RL different from supervised learning

| | Supervised | RL |
|---|---|---|
| Supervision | correct answer given | scalar reward only |
| Feedback timing | immediate | delayed, sometimes by thousands of steps |
| Data distribution | fixed | **depends on the policy — it moves as you learn** |
| i.i.d.? | yes | no; strongly correlated in time |
| Exploration | not needed | **essential** — you only learn about what you try |

▸ The third row is the deepest difference. In supervised learning the data is a fixed object. In RL, improving your policy changes the distribution of the data you collect, which changes what you learn next. **This feedback loop is the source of most of RL's instability.**

---

## 26.2 Markov Decision Processes

▸ An MDP is the tuple $\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$: states, actions, transition kernel $P(s'\mid s,a)$, reward $R(s,a)$, discount $\gamma\in[0,1)$.

**The Markov property:** $P(s_{t+1}\mid s_t,a_t) = P(s_{t+1}\mid s_1,a_1,\dots,s_t,a_t)$. The current state contains everything relevant about the past.

▸ **This is an assumption about the state representation, not about the world.** A single video frame is not Markov (you can't tell velocity); a stack of four frames nearly is. When it fails you have a **POMDP**, and the standard fix is to make the state a function of history — a frame stack, an RNN hidden state, or a transformer context.

### Returns and discounting

▸ $$G_t = \sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$

**Why $\gamma$ exists — three independent reasons:**
1. **Convergence.** With bounded rewards $|R|\le R_{\max}$, $|G_t|\le\frac{R_{\max}}{1-\gamma}<\infty$. Undiscounted infinite sums may diverge.
2. **Uncertainty.** $\gamma$ is equivalent to a per-step $1-\gamma$ probability of episode termination.
3. **Variance reduction.** Discounting truncates the effective horizon, and long-horizon returns have enormous variance.

▸ **The effective horizon is $\frac{1}{1-\gamma}$.** $\gamma=0.99$ → 100 steps; $\gamma=0.999$ → 1000. This is the number to reason with when choosing $\gamma$, not $\gamma$ itself. Higher $\gamma$ is more farsighted *and* harder to learn.

### Value functions

▸ $$V^\pi(s) = \mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a) = \mathbb{E}_\pi[G_t\mid S_t=s, A_t=a]$$

**Advantage:** $A^\pi(s,a) = Q^\pi(s,a)-V^\pi(s)$ — how much better than average this action is. ▸ Note $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)]=0$ by construction, which is why the advantage is the right quantity for policy gradients (Ch. 27 §27.5).

---

## 26.3 The Bellman equations

### The expectation equation — derived

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k\ge0}\gamma^kR_{t+k+1}\ \Big|\ S_t=s\right] = \mathbb{E}_\pi\left[R_{t+1}+\gamma\sum_{k\ge0}\gamma^kR_{t+k+2}\ \Big|\ S_t=s\right]$$

The inner sum is $G_{t+1}$, and by the Markov property its expectation given $S_{t+1}=s'$ is $V^\pi(s')$:

▸ $$\boxed{\ V^\pi(s) = \sum_a\pi(a\mid s)\sum_{s',r}P(s',r\mid s,a)\big[r+\gamma V^\pi(s')\big]\ }$$

▸ $$Q^\pi(s,a)=\sum_{s',r}P(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\Big]$$

**This is the single most important idea in RL: the value of now equals the immediate reward plus the discounted value of next.** Every algorithm in Chapters 26 and 27 is a way of solving or approximating this recursion.

### The optimality equations

▸ $$V^*(s)=\max_a\sum_{s',r}P(s',r\mid s,a)\big[r+\gamma V^*(s')\big]$$
▸ $$Q^*(s,a)=\sum_{s',r}P(s',r\mid s,a)\Big[r+\gamma\max_{a'}Q^*(s',a')\Big]$$

The $\max$ replaces the policy average. **These are nonlinear** (because of the max), so unlike the expectation equations they cannot be solved by linear algebra.

Given $Q^*$, the optimal policy is trivial: ▸ $\pi^*(s)=\arg\max_aQ^*(s,a)$. **This is why value-based methods are appealing — learn the value and the policy comes free.**

---

## 26.4 Dynamic programming and the convergence proof

Assume $P$ and $R$ are known.

**Policy evaluation:** iterate $V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s'}P[r+\gamma V_k(s')]$.
**Policy improvement:** $\pi'(s)=\arg\max_aQ^\pi(s,a)$.
**Policy iteration:** alternate the two. Converges in finitely many iterations, since there are finitely many deterministic policies and each step strictly improves.
**Value iteration:** apply the optimality operator directly — one sweep of evaluation fused with improvement.

### Why it converges: the contraction argument

Define the Bellman optimality operator $\mathcal{T}$:
$$(\mathcal{T}V)(s) = \max_a\sum_{s'}P(s'|s,a)\big[R(s,a)+\gamma V(s')\big]$$

▸ **Claim: $\mathcal{T}$ is a $\gamma$-contraction in the sup norm.**

$$\|\mathcal{T}V-\mathcal{T}W\|_\infty = \max_s\left|\max_a\sum_{s'}P[R+\gamma V(s')] - \max_a\sum_{s'}P[R+\gamma W(s')]\right|$$

Using $|\max_af(a)-\max_ag(a)|\le\max_a|f(a)-g(a)|$:

$$\le\max_{s,a}\ \gamma\sum_{s'}P(s'|s,a)\,|V(s')-W(s')| \ \le\ \gamma\max_{s'}|V(s')-W(s')| = \gamma\|V-W\|_\infty$$

using $\sum_{s'}P=1$. ∎

▸ **By the Banach fixed-point theorem, $\mathcal{T}$ has a unique fixed point $V^*$ and value iteration converges to it geometrically: $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty$.**

**This is the theoretical foundation of all of RL, and it is why $\gamma<1$ matters mathematically rather than just practically.** It also shows the convergence rate degrades as $\gamma\to1$ — long horizons are provably harder.

**Complexity:** $O(|\mathcal{S}|^2|\mathcal{A}|)$ per sweep. Tabular DP is infeasible for real problems (backgammon has $10^{20}$ states, Go has $10^{170}$) — hence function approximation, and hence everything in Chapter 27.

---

## 26.5 Learning without a model

### Monte Carlo

Run a full episode, then set $V(s)\leftarrow V(s)+\alpha[G_t - V(s)]$.

Unbiased. **Zero bias, high variance.** Requires episodes to terminate. Cannot learn online.

### Temporal difference

▸ $$V(S_t)\leftarrow V(S_t)+\alpha\underbrace{\big[\,\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\,\big]}_{\text{TD error }\delta_t}$$

**Bootstrapping:** the target uses the current estimate $V(S_{t+1})$ rather than the actual return.

▸ **Biased (the target depends on a wrong estimate), but far lower variance** — one reward and one value estimate, instead of a sum of hundreds of random rewards. Learns online, from incomplete episodes, and empirically converges much faster.

**MC vs TD, the deepest distinction:** MC converges to the value estimate that minimizes squared error on the observed returns. **TD converges to the value function of the maximum-likelihood MDP implied by the data.** So TD exploits the Markov structure and MC does not — which is why TD wins when the Markov assumption holds and MC can be more robust when it doesn't.

### $n$-step and TD($\lambda$)

$$G_t^{(n)} = R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n})$$

$n=1$ is TD, $n=\infty$ is MC. Intermediate $n$ (3–10) is usually best — an explicit bias–variance dial.

**TD($\lambda$)** takes a geometrically weighted average of all $n$-step returns:
▸ $$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

Implemented online with **eligibility traces**: $e_t(s)=\gamma\lambda e_{t-1}(s)+\mathbb{1}[S_t=s]$, then update every state by $\alpha\delta_te_t(s)$.

▸ **The trace is a short-term memory of which states were recently visited, so one TD error can be assigned to all of them at once.** It is the elegant solution to credit assignment, and it reappears as GAE in Chapter 27 §27.6 — GAE *is* TD($\lambda$) applied to advantages.

---

## 26.6 Control: SARSA and Q-learning

**SARSA (on-policy):**
▸ $$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\big]$$
The update uses the action actually taken next, so it learns the value of **the policy being followed, exploration included.**

**Q-learning (off-policy):**
▸ $$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\big]$$
The $\max$ means it learns the value of the **greedy** policy regardless of how it behaves.

### The Cliff Walking example — the canonical illustration

A grid with a cliff along the bottom edge; the shortest path runs right along it. With $\epsilon$-greedy exploration:
- **Q-learning** learns the optimal path along the cliff edge — but because it *behaves* $\epsilon$-greedily, it occasionally steps off and gets a large negative reward. **Better policy, worse online performance.**
- **SARSA** learns a safer path one row further from the edge, because its updates account for the fact that it will sometimes explore. **Worse asymptotic policy, better online performance.**

▸ **This is the whole on-policy/off-policy distinction in one picture, and it's the answer to give if asked.** SARSA optimizes what you will actually do; Q-learning optimizes what you would do if you stopped exploring.

**Expected SARSA:** replace $Q(S_{t+1},A_{t+1})$ with $\sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a')$. Removes the variance from sampling $A_{t+1}$; strictly better than SARSA at the same cost.

**Convergence guarantee (tabular):** Q-learning converges to $Q^*$ with probability 1 provided all state–action pairs are visited infinitely often and the step sizes satisfy the Robbins–Monro conditions $\sum_t\alpha_t=\infty$, $\sum_t\alpha_t^2<\infty$.

### Maximization bias

▸ $\mathbb{E}[\max_a\hat Q(a)] \ge \max_a\mathbb{E}[\hat Q(a)]$ by Jensen — **the max of noisy estimates is systematically too large.** Q-learning uses the same values to *select* and to *evaluate* the best action, so it is biased upward.

**Double Q-learning** fixes it by maintaining two estimates and decoupling the roles:
$$Q_1(s,a)\leftarrow Q_1(s,a)+\alpha\Big[r+\gamma\,Q_2\big(s',\ \arg\max_{a'}Q_1(s',a')\big)-Q_1(s,a)\Big]$$
One network picks the action, the other scores it. **This is the direct ancestor of Double DQN** (Ch. 27 §27.3).

---

## 26.7 Off-policy learning and importance sampling

To evaluate a target policy $\pi$ using data from a behaviour policy $b$:

▸ $$\rho_{t:T} = \prod_{k=t}^{T}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},\qquad V^\pi(s)=\mathbb{E}_b[\rho_{t:T}G_t\mid S_t=s]$$

▸ **The variance of a product of $T$ ratios explodes exponentially in $T$.** A single trajectory where $\pi$ and $b$ disagree early can have $\rho=10^6$ or $\rho=0$. This is why naive off-policy Monte Carlo is unusable beyond short horizons.

**Mitigations:** weighted (self-normalized) importance sampling — biased but dramatically lower variance and consistent; per-decision IS; and clipping the ratio, which is exactly what PPO does (Ch. 27 §27.7).

▸ **Q-learning avoids importance sampling entirely** because its target uses $\max_{a'}$, which doesn't depend on which policy generated the data. That is the real reason it is the workhorse of off-policy RL.

### The deadly triad

▸ **Function approximation + bootstrapping + off-policy training** — any two are safe; **all three together can diverge**, even in tiny linear problems (Baird's counterexample).

**Why:** bootstrapping means the target moves with the parameters; off-policy means the distribution of updates doesn't match the distribution the value function is being fit under; function approximation means an update at one state changes others. The composition need not be a contraction in any norm.

**This is the fundamental instability of deep RL**, and every trick in Chapter 27 — target networks, replay buffers, trust regions, gradient clipping, conservative updates — is a partial countermeasure.

---

## 26.8 Exploration

### The dilemma
Exploit what you know, or explore to learn more. **The information value of an action is not in its reward.**

### Multi-armed bandits — the clean case

**$\epsilon$-greedy.** Simple; **linear regret** $\Theta(\epsilon T)$ if $\epsilon$ is fixed, since you keep exploring forever. Decay $\epsilon_t\propto1/t$ for logarithmic regret.

**UCB1:**
▸ $$a_t=\arg\max_a\left[\hat\mu_a + c\sqrt{\frac{\ln t}{N_a}}\right]$$

**"Optimism in the face of uncertainty."** The bonus is a confidence-interval width from Hoeffding (Ch. 2 §2.3): it shrinks as $1/\sqrt{N_a}$ with more pulls and grows slowly with $t$ so no arm is abandoned forever.

▸ Regret bound: $O\!\left(\sum_{a:\Delta_a>0}\frac{\ln T}{\Delta_a}\right)$ — **logarithmic in $T$**, which matches the Lai–Robbins lower bound up to constants. UCB is essentially optimal.

**Thompson sampling.** Maintain a posterior over each arm's mean; sample one value per arm; play the argmax. Also achieves logarithmic regret, is trivially easy to implement (Beta–Bernoulli conjugacy), and **empirically outperforms UCB** in most practical settings. It naturally handles delayed feedback and batched decisions.

### In full MDPs

Much harder — you must explore *sequences* of actions, and the state you need to reach may be a hundred steps away.

- **Count-based bonuses:** $r^+ = \frac{\beta}{\sqrt{N(s)}}$; in continuous spaces, use pseudo-counts from a density model or hashing.
- **Curiosity / prediction error (ICM, RND):** reward states where a learned model predicts poorly. ▸ **The noisy-TV problem:** a genuinely stochastic element (static on a screen) is permanently unpredictable, so the agent watches it forever. RND partly fixes this by predicting the output of a *fixed random network* — deterministic, so aleatoric noise is learnable and only novelty persists.
- **Bootstrapped DQN / noisy networks:** posterior-sampling-style deep exploration.
- **Go-Explore:** archive visited states, return to promising ones, then explore from there. Solved Montezuma's Revenge, which had defeated every bonus-based method.

▸ **The hard-exploration benchmark to know:** Montezuma's Revenge, where reward is so sparse that random exploration essentially never finds it.

---

## 26.9 Reward design

▸ **Reward shaping theorem (Ng, Harada & Russell):** adding a potential-based shaping term
$$F(s,a,s') = \gamma\Phi(s')-\Phi(s)$$
leaves the optimal policy **unchanged**, for any function $\Phi$. Any non-potential-based shaping can change the optimum, usually badly.

This is the one theoretically safe way to add intermediate rewards, and it's a great thing to know: it says you may express "these states are promising" via a potential, but you may not reward *behaviours* without risking that the agent optimizes the behaviour instead of the goal.

**Reward hacking** — the agent maximizes the specified reward while violating its intent — is pervasive: boat-racing agents that circle collecting powerups instead of finishing, robots that exploit simulator physics, and (Ch. 16) language models that learn verbosity because raters prefer long answers. **Reward specification, not optimization, is usually the binding constraint on RL projects.**

---

## Check for Understanding

**All of RL rests on the Bellman equation — value now equals reward plus discounted value next — which converges because the Bellman operator is a $\gamma$-contraction in the sup norm; temporal-difference learning approximates it from samples by bootstrapping, trading bias for a large variance reduction; and the combination of bootstrapping, off-policy data, and function approximation is exactly the deadly triad that makes deep RL unstable.**

---

**Next:** [Chapter 27 — Deep Reinforcement Learning](27-deep-reinforcement-learning.md)
