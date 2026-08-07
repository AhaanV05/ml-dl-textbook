# Chapter 26 — Reinforcement Learning Foundations

> **Prerequisites:** Ch. 1 (§1.3 probability), Ch. 4.

> **New to the notation?** If symbols like $\in$, $\sum$, $\mathbb{E}$, $\nabla$, or $A^\top$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book.

### Symbols introduced in this chapter

Reinforcement learning brings its own alphabet, and almost none of it appears in the supervised chapters. Skim this once now; every entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $\mathcal{S}$, $\mathcal{A}$ | "script S, script A" | The set of all **states**, and the set of all **actions** |
| $s,\ a,\ r,\ s'$ | "s, a, r, s-prime" | One state, one action, one reward, and the **next** state |
| $`S_t,\ A_t,\ R_t`$ | "S-t, A-t, R-t" | The same things as **random variables** at time $t$. Capitals mean "not yet known" |
| $\pi(a \mid s)$ | "pi of a **given** s" | The **policy** — the probability of choosing action $a$ when in state $s$ |
| $P(s' \mid s,a)$ | "P of s-prime given s and a" | The **environment's** rule: where you land next, and how likely each landing is |
| $\gamma$ | "gamma" | The **discount factor** — what a reward one step in the future is worth today |
| $`G_t`$ | "G-t" | The **return**: the total discounted reward collected from time $t$ onward |
| $V^\pi(s)$ | "V-pi of s" | **How good state $s$ is**, assuming you follow policy $\pi$ from here on |
| $Q^\pi(s,a)$ | "Q-pi of s, a" | How good it is to take action $a$ in $s$ **and then** follow $\pi$ |
| $A^\pi(s,a)$ | "A-pi of s, a" | The **advantage**: how much better $a$ is than the policy's average action |
| $`V^*,\ Q^*,\ \pi^*`$ | "V-star, Q-star, pi-star" | The **best achievable** value, action-value, and policy. A star means optimal |
| $\mathcal{T}$ | "script T" | The **Bellman operator** — the machine that performs one "look ahead one step" pass |
| $`\lVert V \rVert_\infty`$ | "the sup norm of V" | The **largest single error anywhere**, across all states — worst case, not average |
| $`\delta_t`$ | "delta-t" | The **TD error** — surprise: what actually happened minus what you predicted |
| $\alpha$ | "alpha" | The **step size** of a value update. How much you move toward the new evidence |
| $\lambda$ | "lambda" | The **trace decay** — how far back in time credit for a surprise is spread |
| $`e_t(s)`$ | "e-t of s" | An **eligibility trace** — a fading memory of "I passed through here recently" |
| $`\rho_{t:T}`$ | "rho, t to T" | An **importance-sampling ratio** — the correction for learning from another policy's data |
| $\Phi(s)$ | "capital phi of s" | A **potential** — a hand-designed "how promising does this state look" score |
| $`N_a`$, $`\Delta_a`$ | "N-a, Delta-a" | How many times option $a$ has been tried; how much worse it is than the best option |
| $\mathbb{1}[\,\cdot\,]$ | "indicator" | **1 if the statement inside is true, 0 if false.** A switch |

**Every abbreviation in this chapter, spelled out.** Read the full form aloud once — most of these stop being intimidating the moment you hear what they stand for.

| Short | Full form |
|---|---|
| RL | Reinforcement Learning |
| MDP | Markov Decision Process |
| POMDP | Partially Observable Markov Decision Process |
| DP | Dynamic Programming |
| MC | Monte Carlo |
| TD | Temporal Difference |
| SARSA | State–Action–Reward–State–Action |
| UCB | Upper Confidence Bound |
| IS | Importance Sampling |
| GAE | Generalized Advantage Estimation |
| DQN | Deep Q-Network |
| PPO | Proximal Policy Optimization |
| ICM / RND | Intrinsic Curiosity Module / Random Network Distillation |
| i.i.d. | independent and identically distributed |

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

#### Unpacking the table, row by row

Five rows, five different ways RL breaks an assumption you have been relying on since Chapter 2. Take them slowly, because every difficulty in the next two chapters descends from one of them.

**Row 1 — "correct answer given" versus "scalar reward only."** In supervised learning, a training example is a pair: this image, that label. The label *is* the answer. In RL you are handed a single number — say $+1$ or $-3.7$ — and nothing else. Nobody says which action was the good one, or what the good one would have been.

> **Analogy.** Supervised learning is a maths tutor who marks your working and writes the correct line next to the wrong one. Reinforcement learning is a maths exam that returns only your total score. You know you got 62%. You do not know which questions you lost marks on, and you have to work it out by taking more exams.

**Row 2 — delayed feedback.** In chess, a losing move on turn 12 might only produce a reward (a loss, $-1$) on turn 60. There are 48 intervening moves, most of them fine. The reward signal has to travel backwards across all of them.

**Row 3 — the data distribution moves.** This is the row the book flags, and it deserves numbers. Suppose your policy currently drives a simulated car straight into a wall after four seconds. Then **100% of your training data is about the first four seconds.** You have never seen the road past the wall, so you cannot learn about it, so you cannot get past the wall — until exploration happens to carry you there, at which point your whole data distribution shifts and everything you learned about the early part may be re-weighted or invalidated.

**Row 4 — not independent and identically distributed (i.i.d.).** Consecutive video frames of a game are nearly identical. A minibatch of 32 consecutive frames contains roughly one frame's worth of information, not 32. Every convergence result in Chapters 2–5 assumed i.i.d. samples; RL violates it by construction, because time is the sampling order.

**Row 5 — exploration is mandatory.** In supervised learning, unsampled regions of input space are somebody else's problem. In RL, an action you never take is an action whose value you can never estimate — and its value might be enormous.

▸ **The one-sentence version of all five rows: in supervised learning you are given a dataset; in RL you *generate* your dataset, badly, using the very policy you are trying to fix.** That circularity is the whole subject.

> **Where the word "reinforcement" came from.** It is borrowed intact from animal psychology. **Edward Thorndike**, working with cats in puzzle boxes at the end of the 1890s, formulated what he called the **Law of Effect**: responses followed by a satisfying state of affairs become more likely to recur in that situation. "Reinforcement" as a technical term for a consequence that strengthens a behaviour comes out of that tradition and out of **Ivan Pavlov's** conditioning work of the same era. When RL researchers say an agent's behaviour is "reinforced," they are using a hundred-and-twenty-year-old term of art in almost exactly its original sense. The field's founding textbook — Richard Sutton and Andrew Barto's *Reinforcement Learning: An Introduction* — is explicit that the animal-learning literature was a direct source of the algorithms, not a decorative analogy.

> **The story behind "credit assignment."** The phrase is **Marvin Minsky's**, from his 1961 survey *Steps Toward Artificial Intelligence*. He identified it as the central obstacle for any system that learns from a delayed, aggregate signal: when a long sequence of decisions produces one outcome, which decision gets the credit? Minsky was writing before backpropagation, before Q-learning, before anything in this chapter existed — and he named the problem that all of it exists to solve.

---

## 26.2 Markov Decision Processes

▸ An MDP is the tuple $\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$: states, actions, transition kernel $P(s'\mid s,a)$, reward $R(s,a)$, discount $\gamma\in[0,1)$.

**The Markov property:** $`P(s_{t+1}\mid s_t,a_t) = P(s_{t+1}\mid s_1,a_1,\dots,s_t,a_t)`$. The current state contains everything relevant about the past.

▸ **This is an assumption about the state representation, not about the world.** A single video frame is not Markov (you can't tell velocity); a stack of four frames nearly is. When it fails you have a **POMDP**, and the standard fix is to make the state a function of history — a frame stack, an RNN hidden state, or a transformer context.

#### Reading the MDP tuple in plain English

$\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$ is read aloud as *"the tuple S, A, P, R, gamma."* The angle brackets $\langle\ \rangle$ mean nothing more than "here is a fixed list of ingredients" — they are the mathematical equivalent of a parts list on the side of a flat-pack box. **An MDP (Markov Decision Process) is not a theory. It is a specification of what you must write down before you are allowed to start.**

| Ingredient | Read aloud | What you actually have to supply |
|---|---|---|
| $\mathcal{S}$ | "script S" | Every situation the agent can be in |
| $\mathcal{A}$ | "script A" | Every move the agent can make |
| $P(s' \mid s,a)$ | "P of s-prime given s and a" | The world's physics: "if you're here and do that, where do you end up?" |
| $R(s,a)$ | "R of s, a" | The score: "how many points is that worth?" |
| $\gamma$ | "gamma" | How much you care about later versus now |

**Make it concrete — a two-state world.** Let $\mathcal{S} = \{\text{Hungry}, \text{Fed}\}$ and $\mathcal{A} = \{\text{Cook}, \text{Wait}\}$. Then $P$ is a small table:

- $P(\text{Fed} \mid \text{Hungry}, \text{Cook}) = 0.8$ — cooking usually works.
- $P(\text{Hungry} \mid \text{Hungry}, \text{Cook}) = 0.2$ — sometimes you burn it.
- $P(\text{Hungry} \mid \text{Hungry}, \text{Wait}) = 1.0$ — waiting never feeds you.

And $R(\text{Hungry},\text{Cook}) = -1$ (effort), $R(\text{Fed},\cdot) = +10$. That is a complete MDP. It has $|\mathcal{S}| \times |\mathcal{A}| = 4$ entries. Everything in this chapter applies to it exactly as written; the only thing that changes for Go is that the table has $10^{170}$ rows.

▸ **The "prime" in $s'$ just means "next."** It is not a derivative, not a transpose, not anything else. Read $s'$ as *"s-prime"* and think *"the state after this one."* The same convention gives $a'$ for "the next action." This one piece of notation appears in nearly every equation in Chapters 26 and 27.

#### The Markov property, decoded

$$P(s_{t+1}\mid s_t,a_t) = P(s_{t+1}\mid s_1,a_1,\dots,s_t,a_t)$$

Read the left side: *"the probability of the next state, given only the current state and current action."* Read the right side: *"the probability of the next state, given the entire history since the beginning of time."* The claim is that **these two are equal** — that the whole history tells you nothing the present doesn't already tell you.

> **Analogy.** Look at a chessboard mid-game. To decide your next move, does it help to know the order in which the pieces arrived at their squares? Almost never — the position is the position. Chess is (nearly) Markov. Now look at a single photograph of a tennis ball in mid-air. Is it rising or falling? You cannot tell. **One photograph is not Markov; two consecutive photographs almost are**, because the difference between them encodes velocity. That is precisely why Chapter 27's Atari agents stack four frames.

▸ **Sit with the book's warning: the Markov property is a property of your *state representation*, not of reality.** Reality is whatever it is. You get to choose what to call "the state," and by choosing badly you can make any problem non-Markov, or by choosing well you can make almost anything Markov. Poker is not Markov if the state is "my cards"; it is much closer to Markov if the state is "my cards plus the full betting history." **Fixing a broken Markov assumption is a data-engineering job, not a mathematics job.**

**POMDP** stands for **Partially Observable Markov Decision Process** — an MDP where you cannot see the true state, only a noisy or incomplete observation of it. Nearly every real problem is a POMDP. The three standard repairs, in increasing order of power: stack the last $k$ observations (cheap, works for velocity), carry a recurrent hidden state (an RNN, Ch. 9), or feed a transformer the whole visible history (Ch. 11). All three are the same move — **manufacture a state that is Markov by stuffing history into it.**

> **Where this came from.** **Andrey Markov** introduced what we now call Markov chains around 1906, and his motivation was an argument. The Russian mathematician Pavel Nekrasov had claimed that the law of large numbers required independence, and had attached philosophical significance to the claim. Markov set out to show this was false by constructing dependent sequences that still obeyed the law. To demonstrate that his chains described something real rather than a contrivance, he then did one of the great acts of manual labour in the history of statistics: he took the first 20,000 letters of Pushkin's *Eugene Onegin*, classified each as a vowel or a consonant **by hand**, and tabulated the transition frequencies. He was, without any intention of doing so, building the first statistical language model. The decision process built on top of his chains — the "D" in MDP — came much later, in the 1950s, from the dynamic-programming work of Richard Bellman and Ronald Howard.

#### Examples and non-examples: is this state representation Markov?

**✅ States that already carry everything the future needs**

| Example | Why it qualifies |
|---|---|
| A chess position including castling rights, the en-passant square, and the repetition count | Given all that, the legal futures do not depend on the move order that produced the position |
| Four stacked Atari frames | One frame hides velocity; four make it recoverable by differencing |
| Poker: your hole cards **plus** the full betting history of the hand | The betting is what carries the information about the opponents' hands |
| A robot's joint angles **and** joint velocities | Angles alone leave the dynamics underdetermined; adding velocities closes the gap |
| Grid position in a gridworld with fixed dynamics | There are no hidden variables left to remember |

**❌ Near-misses — representations that break the property**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A single Atari frame | Is the ball rising or falling? The frame cannot say | A **partial observation** — the setting is a POMDP |
| A chess position without castling rights | Two boards can look identical and have different legal moves | An incomplete state |
| Poker with state = "my two cards" | Everything informative has been thrown away | An observation, badly chosen |
| Dialogue state = "the last user message" | The earlier turns change what the right reply is | A truncated context window |
| "The world is complicated, so the problem isn't Markov" | Markov is a property of the state variable you wrote down, not of the universe | A modelling choice you have not made yet |
| Stacking 4 frames in a game where a power-up lasts 300 frames | The window is shorter than the memory the task requires | Still a POMDP — the fix has to match the timescale |

▸ **The boundary:** the question is never "does the world remember?" — it is "does **my state variable** already contain everything the history would have told me?" You choose the state, so you choose whether the property holds.

> **Common misconception.** *"The Markov property says the world is memoryless — that the past doesn't influence the future."* It says nothing of the kind, and reading it that way makes the entire framework look absurd, because of course the past influences the future. **The property is about your state representation: it claims that whatever the history would have told you is *already encoded* in the state you wrote down.** The past matters enormously; it simply does not need to be consulted separately, because it has been summarized. That is why "make it Markov" is always a data-engineering task — stack four frames, append joint velocities, carry the betting history — and never a mathematical one. The belief is tempting because "memoryless" is  the standard word for Markov *chains* in probability texts, where the state is handed to you and there is nothing to design. **In RL you design the state, and that changes what the word is doing.**

### Returns and discounting

▸ $$G_t = \sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$

**Why $\gamma$ exists — three independent reasons:**
1. **Convergence.** With bounded rewards $`|R|\le R_{\max}`$, $`|G_t|\le\frac{R_{\max}}{1-\gamma}<\infty`$. Undiscounted infinite sums may diverge.
2. **Uncertainty.** $\gamma$ is equivalent to a per-step $1-\gamma$ probability of episode termination.
3. **Variance reduction.** Discounting truncates the effective horizon, and long-horizon returns have enormous variance.

▸ **The effective horizon is $\frac{1}{1-\gamma}$.** $\gamma=0.99$ → 100 steps; $\gamma=0.999$ → 1000. This is the number to reason with when choosing $\gamma$, not $\gamma$ itself. Higher $\gamma$ is more farsighted *and* harder to learn.

#### Reading the return $`G_t`$ in plain English

$$G_t = \sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$

Read aloud: *"G-t equals the sum, over k from zero to infinity, of gamma-to-the-k times R-t-plus-k-plus-one."* Now in English: **"add up all the rewards from here to the end of time, but shrink each one a bit more the further away it is."**

The pieces:

- $`\sum_{k=0}^\infty`$ is a `for` loop over future time steps, starting at the next one (§0.3). $k$ is the loop counter; nothing else in the expression moves.
- $`R_{t+k+1}`$ is the reward received $k+1$ steps from now. The off-by-one is a convention: the reward for acting at time $t$ arrives at time $t+1$, because you have to act before the world can pay you.
- $\gamma^k$ is the **shrink factor**, applied $k$ times. Since $0 \le \gamma < 1$, raising it to a power makes it smaller.

**Put real numbers in.** Take $\gamma = 0.9$ and suppose every step pays exactly $1$:

$$G_t = 1 + 0.9 + 0.81 + 0.729 + \dots = \frac{1}{1-0.9} = 10$$

An infinite stream of $+1$ is worth exactly $10$. That is the **geometric series** $`\sum_k \gamma^k = \frac{1}{1-\gamma}`$, and it is where the bound $`\lvert G_t\rvert \le \frac{R_{\max}}{1-\gamma}`$ in reason 1 comes from — the worst possible return is "maximum reward, forever."

> **Analogy: $\gamma$ is an interest rate, in reverse.** A bank offering 11% would turn £9 into £10 in a year; equivalently, £10 arriving next year is worth about £9 today. Discounting a future reward by $\gamma = 0.9$ is exactly this operation — the **present value** calculation from finance, imported wholesale. A reward far in the future is worth less not because it matters less, but because you might not survive to collect it and because you cannot spend it now.

#### Why $\frac{1}{1-\gamma}$ deserves the name "horizon"

The claim is that $\gamma = 0.99$ means "about 100 steps of foresight." Here is why that is not arbitrary. Set $k = \frac{1}{1-\gamma}$ and compute the discount at that point:

$$\gamma^{1/(1-\gamma)} \approx e^{-1} \approx 0.368 \quad\text{for } \gamma \text{ near } 1$$

Check it: $0.99^{100} = 0.366$. And $0.999^{1000} = 0.368$. **The effective horizon is the distance at which a reward has faded to about 37% of face value.** Beyond it, rewards still count, but they are being quietly ignored: $0.99^{1000} \approx 0.000043$, which is four-hundredths of one percent — numerically indistinguishable from zero once you add sampling noise.

| $\gamma$ | Effective horizon | Value of an eternal $+1$ | Worth of a reward 500 steps away |
|---|---|---|---|
| $0.9$ | 10 steps | $10$ | $\approx 10^{-23}$ |
| $0.99$ | 100 steps | $100$ | $\approx 0.0066$ |
| $0.999$ | 1000 steps | $1000$ | $\approx 0.61$ |

▸ **Two practical consequences fall straight out of that table.** First, **raising $\gamma$ inflates the scale of your value function**, so a network that comfortably predicted values near $10$ must now predict values near $1000$ — a real cause of instability that is often misdiagnosed as an algorithmic problem. Second, **a reward that lies beyond your effective horizon is invisible**, so if your task's payoff comes 3000 steps in and you set $\gamma = 0.99$, the agent is not being stubborn — you have mathematically told it not to care.

**The three reasons for $\gamma$, restated.** Reason 1 (**convergence**) is mathematical hygiene: without $\gamma$, "total reward over infinite time" can be $\infty$ for every policy, and $\infty = \infty$ gives you no way to prefer one. Reason 2 (**uncertainty**) is the elegant one: a $\gamma$ of $0.99$ is *exactly equivalent* to an undiscounted problem in which the episode ends with probability $0.01$ at each step. Discounting is not a fudge; it is an honest statement that the future is not guaranteed. Reason 3 (**variance**) is the practical one, and it becomes the entire subject of §26.5.

### Value functions

▸ $$V^\pi(s) = \mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a) = \mathbb{E}_\pi[G_t\mid S_t=s, A_t=a]$$

**Advantage:** $A^\pi(s,a) = Q^\pi(s,a)-V^\pi(s)$ — how much better than average this action is. ▸ Note $`\mathbb{E}_{a\sim\pi}[A^\pi(s,a)]=0`$ by construction, which is why the advantage is the right quantity for policy gradients (Ch. 27 §27.5).

#### $V$, $Q$, and $A$ — decoded

These three functions are the whole vocabulary of value-based RL, and the difference between them is one of the most common sources of confusion in the subject. Read them aloud first.

$$V^\pi(s) = \mathbb{E}_\pi[G_t\mid S_t=s]$$

*"V-pi of s equals the expectation, under policy pi, of the return G-t, **given** that the state at time t is s."* In English: **"If I drop you here and you play by the rules of $\pi$ from now on, what score should you expect?"**

Every symbol:

- $\mathbb{E}$ is the probability-weighted average (§0.5). It is here because two things are random: the environment's dice ($P$) and the policy's own coin flips ($\pi$).
- The subscript on $`\mathbb{E}_\pi`$ says **what is generating the randomness** — "average over the trajectories that $\pi$ produces."
- The vertical bar $\mid$ means **"given,"** not "divide" and not "absolute value" (§0.9, Trap 5). It restricts the average to worlds where $`S_t = s`$.
- The superscript $\pi$ on $V^\pi$ is **not a power**. It is a label: "the value function *belonging to* policy $\pi$." A different policy has a different $V$. This is why $`V^*`$ (star) is meaningful — it labels the best one.

$Q^\pi(s,a)$ adds one more condition: *"given that the state is $s$ **and** the action you take right now is $a$."*

▸ **The difference between $V$ and $Q$ is exactly one action.** $Q$ lets you specify the first move and then hands control back to $\pi$; $V$ lets $\pi$ choose from the start. Formally $`V^\pi(s) = \sum_a \pi(a\mid s)\,Q^\pi(s,a)`$ — *"the value of a state is the policy-weighted average of the values of the actions available in it."*

**Put numbers on it.** You are in a state with three actions. Suppose $Q^\pi(s, \text{left}) = 12$, $Q^\pi(s, \text{straight}) = 4$, $Q^\pi(s, \text{right}) = 2$, and the current policy picks each with probability $\tfrac13$. Then:

$$V^\pi(s) = \tfrac13(12) + \tfrac13(4) + \tfrac13(2) = 6$$

and the advantages are

$$A(\text{left}) = 12 - 6 = +6,\qquad A(\text{straight}) = 4 - 6 = -2,\qquad A(\text{right}) = 2-6 = -4$$

Notice $\tfrac13(6) + \tfrac13(-2) + \tfrac13(-4) = 0$. **That is the claim $`\mathbb{E}_{a\sim\pi}[A^\pi(s,a)] = 0`$, and now you can see it is not a theorem so much as an accounting identity**: you subtracted the average, so the average of what's left is zero.

> **Analogy for the three functions.** $V(s)$ is the *par* for a hole in golf. $Q(s,a)$ is your expected total if you play this particular club off the tee and then play normally. $A(s,a)$ is **how many strokes that club choice gains or loses you relative to par.** Announcers quote "two under par," not the raw stroke count, for exactly the reason RL prefers advantages: the raw number is dominated by which hole you're on, and the interesting part is your decision.

▸ **Why the advantage is the right quantity for learning, in one line.** Suppose a state is worth $1000$ and all its actions are worth between $999$ and $1001$. The raw $Q$ values are nearly identical — a learning signal of $1001$ versus $1000$ is a 0.1% difference, and sampling noise will swamp it. But the advantages are $+1$ and $0$ — a *100%* difference. **Subtracting $V$ removes the part of the signal that carries no information about your choice**, which is variance reduction of the purest kind. This single idea is the baseline trick of Chapter 27 §27.4 and the reason GAE (§27.6) exists.

#### Examples and non-examples: reward, return, and value

Three words for three different objects, used interchangeably in casual conversation more often than any other trio in this book. Prising them apart is most of the vocabulary of the subject.

**✅ Reward $`R_t`$ — one scalar, handed over by the environment at one time step**

| Example | Why it qualifies |
|---|---|
| $+1$ for eating a pellet, $-1$ for touching a ghost | Emitted by the environment at a specific step; the agent does not compute it |
| $0$ on every step of a maze, then $+1$ at the exit | Sparse, but still exactly one number per step |
| $-1$ per timestep, to encourage finishing quickly | Rewards may be negative — the informative ones often are |

**✅ Return $`G_t`$ — the discounted sum of rewards from $t$ onward, along a single trajectory**

| Example | Why it qualifies |
|---|---|
| $`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+\dots`$ | A number attached to one realized rollout, luck included |
| The final score of one particular episode | One sample of a random variable |

**✅ Value $V^\pi(s)$ — the expectation of that return, over the randomness of both environment and policy**

| Example | Why it qualifies |
|---|---|
| "Play from here under $\pi$ a thousand times and average the returns" | An average over trajectories, not one trajectory |
| $`V^\pi(s) = \sum_a \pi(a\mid s)\,Q^\pi(s,a)`$ | Built out of expectations at every level |

**❌ Near-misses**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| "Reaching the goal pays 100, so the goal state is worth 100" | Value counts everything that happens *after*, discounted; at a terminal state that is nothing at all | A single reward, not a value |
| A state that pays well and then leads off a cliff | Value is about the whole tail, and this tail is catastrophic | A trap with good bait |
| The return from one lucky episode | A single sample of a random variable | A **Monte Carlo estimate** — unbiased, high variance |
| "$Q(s,a)$ is the reward for taking $a$" | It is the expected *return* after taking $a$ and then following $\pi$ forever | The action-value function |
| Reading $A(s,a)$ as "the value of that action" | It is measured **relative to the state's own average**, and averages to zero under $\pi$ | A centred value |
| The undiscounted total in a continuing task | It can be infinite, which makes comparison meaningless | Precisely why $\gamma < 1$ exists |

▸ **The boundary:** **reward is one number from the environment; return is a sum along one trajectory; value is the average of that sum over all trajectories.** Reward is given, return is observed, value is estimated.

> **Common misconception.** *"Value is basically reward — a high-value state is one that pays well."* A state can pay nothing and be enormously valuable, and it can pay handsomely and be worthless. The opening position of a chess game has reward exactly zero and immense value; the square one step from the edge in Cliff Walking pays the same $-1$ as every other square and is worth far less than its neighbours. **Value is a claim about the *future*, discounted and averaged; reward is a receipt for the present.** The confusion is tempting because in bandit problems — the setting almost everyone meets first — there *is* no future, so value collapses to expected immediate reward and the two words  do coincide. **Everything interesting about RL is exactly the gap that opens the moment there is a next state.**

> **Common misconception.** *"The reward function tells the agent what the goal is."* It does not describe a goal. It defines a **scoring rule that a relentless optimizer will read adversarially**, and the agent has no access to your intent — only to the number. This is why a 1990s simulated bicycle rewarded for progress toward a target learned to ride in tight circles near the start, and why OpenAI's *CoastRunners* boat learned to spin in a lagoon harvesting regenerating powerups, catching fire repeatedly, scoring about 20% above human players and never finishing the race (§26.9). Neither agent malfunctioned; both maximized exactly what was written. The belief is tempting because we *derive* reward functions from our goals, so it feels as though the goal survives the translation — but the translation is lossy in one direction only, and the optimizer will search precisely the region you forgot to constrain.

---

## 26.3 The Bellman equations

### The expectation equation — derived

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k\ge0}\gamma^kR_{t+k+1}\ \Big|\ S_t=s\right] = \mathbb{E}_\pi\left[R_{t+1}+\gamma\sum_{k\ge0}\gamma^kR_{t+k+2}\ \Big|\ S_t=s\right]$$

The inner sum is $`G_{t+1}`$, and by the Markov property its expectation given $`S_{t+1}=s'`$ is $V^\pi(s')$:

▸ $$\boxed{\ V^\pi(s) = \sum_a\pi(a\mid s)\sum_{s',r}P(s',r\mid s,a)\big[r+\gamma V^\pi(s')\big]\ }$$

▸ $$Q^\pi(s,a)=\sum_{s',r}P(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\Big]$$

**This is the single most important idea in RL: the value of now equals the immediate reward plus the discounted value of next.** Every algorithm in Chapters 26 and 27 is a way of solving or approximating this recursion.

#### Reading the Bellman equation in plain English

$$V^\pi(s) = \sum_a\pi(a\mid s)\sum_{s',r}P(s',r\mid s,a)\big[r+\gamma V^\pi(s')\big]$$

This is the most important formula in the chapter and it looks far worse than it is. It is **two nested `for` loops around one short bracket.** Read it from the inside out.

**The bracket first.** $\big[r + \gamma V^\pi(s')\big]$ is the whole idea: *"what I collect right now, plus a discounted version of how good the place I land is."* One immediate payment plus one estimate of everything after.

**The inner sum.** $`\sum_{s',r}P(s',r\mid s,a)[\cdot]`$ says: *"you don't know where you'll land, so average over every possible landing, weighted by how likely it is."* This is an expectation written out longhand (§0.5). $P(s',r\mid s,a)$ is read *"the probability of landing in $s'$ and receiving reward $r$, given that you were in $s$ and did $a$."*

**The outer sum.** $`\sum_a \pi(a\mid s)[\cdot]`$ says: *"you don't know which action you'll take either, because the policy is random, so average over the actions too, weighted by how often $\pi$ picks each."*

▸ **So the whole line reads: "the value of a state is the average, over every action you might take and every place you might land, of what you get plus what you'll go on to get."** Three words are doing all the work: **average**, **plus**, **discounted**.

**Now put real numbers in.** Set $\gamma = 0.9$. You are in state $s$; your policy flips a fair coin between two actions:

- **Action $a$**: pays $r=2$ immediately and always leads to a state worth $V^\pi(s') = 10$.
- **Action $b$**: pays $r=5$ immediately and always leads to a state worth $V^\pi(s'')=0$.

$$V^\pi(s) = \underbrace{0.5}_{\pi(a\mid s)}\big[2 + 0.9(10)\big] + \underbrace{0.5}_{\pi(b\mid s)}\big[5 + 0.9(0)\big] = 0.5(11) + 0.5(5) = \boxed{8}$$

Notice what just happened. **Action $b$ pays more than twice as much right now and is still the worse move**, because $a$ leads somewhere valuable. That gap — $11$ versus $5$ — is the entire reason RL is not just greedy reward maximization, and the reason the equation is *recursive* rather than a lookup table.

#### Why the recursion is legitimate — the derivation, slowly

The book's derivation splits the return into "the first reward" and "everything after":

$$G_t = R_{t+1} + \gamma\underbrace{\big(R_{t+2} + \gamma R_{t+3} + \dots\big)}_{= \ G_{t+1}}$$

**This is just factoring out one $\gamma$.** Every term after the first has at least one factor of $\gamma$; pull it out and what remains is the same expression starting one step later. That algebraic triviality is the whole trick.

The  non-trivial step is the next one: replacing $`\mathbb{E}[G_{t+1}\mid S_{t+1}=s']`$ with $V^\pi(s')$. **This is legal only because of the Markov property.** If the future depended on how you arrived at $s'$, then $V^\pi(s')$ would not be a well-defined number and the recursion would not close.

▸ **The Markov assumption is not a technicality — it is the load-bearing wall.** It is what lets a *single number per state* summarize an infinite future, which is what makes value functions storable, learnable, and finite. When Markov fails, the number $V(s)$ is an average over histories you cannot distinguish, and every algorithm downstream inherits that ambiguity as bias.

> **Where this came from.** **Richard Bellman** developed dynamic programming at the **RAND Corporation** around 1950–1953, working on multi-stage decision processes for the U.S. Air Force. He also coined the phrase **"curse of dimensionality"** — it appears in his 1957 book *Dynamic Programming* — to describe exactly the $\lvert\mathcal{S}\rvert^2\lvert\mathcal{A}\rvert$ blow-up noted at the end of §26.4.
>
> The name "dynamic programming" is itself a small mystery. In Bellman's autobiography he explained that he needed a title that would not attract the attention of the then Secretary of Defense, Charles Wilson, who was said to be hostile to anything sounding like mathematical *research*; "programming" at the time meant planning and scheduling (as in George Dantzig's *linear* programming), and "dynamic" was chosen as an adjective nobody could use pejoratively. **Accounts differ on how literally to take this.** The dates do not line up perfectly — Wilson took office in 1953, after Bellman's work had begun — so the story is generally treated as a retrospective embellishment of a real institutional pressure rather than a precise account. What is not disputed is that the name is a poor description of the method: dynamic programming is neither dynamic nor programming.
>
> **Ronald Howard** completed the picture in 1960 with *Dynamic Programming and Markov Processes*, which introduced **policy iteration** and welded Markov chains to Bellman's optimization — the moment the modern MDP appears in recognizable form.

### The optimality equations

▸ $$V^*(s)=\max_a\sum_{s',r}P(s',r\mid s,a)\big[r+\gamma V^*(s')\big]$$
▸ $$Q^*(s,a)=\sum_{s',r}P(s',r\mid s,a)\Big[r+\gamma\max_{a'}Q^*(s',a')\Big]$$

The $\max$ replaces the policy average. **These are nonlinear** (because of the max), so unlike the expectation equations they cannot be solved by linear algebra.

Given $`Q^*`$, the optimal policy is trivial: ▸ $`\pi^*(s)=\arg\max_aQ^*(s,a)`$. **This is why value-based methods are appealing — learn the value and the policy comes free.**

#### What the optimality equations actually say

Compare the two versions side by side. The **only** difference is what happens to the outer sum:

| Expectation equation | Optimality equation |
|---|---|
| $`\sum_a \pi(a\mid s)\,[\,\cdot\,]`$ | $`\max_a\,[\,\cdot\,]`$ |
| "average over what the policy would do" | "take the best one" |
| Describes **a** policy | Describes **the best** policy |
| Linear in $V$ — solvable by matrix inversion | Nonlinear — no closed form |

▸ **Swapping $`\sum_a \pi(a\mid s)`$ for $`\max_a`$ is the entire content of the optimality equations.** It is the difference between "how well will I do?" and "how well *could* I do?"

**Why the $\max$ makes it nonlinear, concretely.** A sum is linear: $\text{avg}(2x) = 2\,\text{avg}(x)$, and $\text{avg}(x+y) = \text{avg}(x)+\text{avg}(y)$. A maximum obeys the first rule but breaks the second. Take $x = (3, 1)$ and $y = (1, 3)$:

$$\max(x) + \max(y) = 3 + 3 = 6, \qquad \max(x+y) = \max(4,4) = 4$$

$6 \ne 4$. **A $\max$ is a kink, and a kink is not a straight line.** That is why the expectation equations can be solved exactly with linear algebra — they are $\lvert\mathcal{S}\rvert$ equations in $\lvert\mathcal{S}\rvert$ unknowns, and you can literally invert a matrix — while the optimality equations cannot, and must be reached by iteration instead.

#### Why "learn the value and the policy comes free" is such a big deal

$`\pi^*(s) = \arg\max_a Q^*(s,a)`$ reads: *"the best policy in state $s$ is: whichever action has the highest Q-star."* Recall from §0.3 that $\arg\max$ returns **the input, not the value** — $\max$ tells you the height of the peak, $\arg\max$ tells you where it is. Here you want the action, so it is $\arg\max$.

**Put numbers on it.** Suppose in some state, $`Q^*(s,\text{north}) = 8.2`$, $`Q^*(s,\text{south}) = 3.1`$, $`Q^*(s,\text{east}) = 8.9`$. Then $`\max_a Q^*(s,a) = 8.9`$ and $`\arg\max_a Q^*(s,a) = \text{east}`$. The policy is a one-line scan of a table.

> **Analogy.** A really good restaurant guide replaces the need for a chef. If somebody hands you a perfectly accurate score for every dish in every restaurant in the city, you don't need to develop taste — you just read off the highest number. $`Q^*`$ is that guide, and the greedy policy is "always order the top-scoring dish."

▸ **This is the appeal and the trap of value-based methods.** The appeal: you never have to represent a policy at all, only a value function. The trap: computing $`\arg\max_a`$ requires *enumerating every action*, which is instant for four Atari buttons and impossible for a robot arm with seven continuous joints. **That single line is the reason Chapter 27 splits into value-based methods (discrete actions) and policy-based methods (continuous actions).** When you cannot take an $\arg\max$, you must learn the policy directly.

---

## 26.4 Dynamic programming and the convergence proof

Assume $P$ and $R$ are known.

**Policy evaluation:** iterate $`V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s'}P[r+\gamma V_k(s')]`$.
**Policy improvement:** $`\pi'(s)=\arg\max_aQ^\pi(s,a)`$.
**Policy iteration:** alternate the two. Converges in finitely many iterations, since there are finitely many deterministic policies and each step strictly improves.
**Value iteration:** apply the optimality operator directly — one sweep of evaluation fused with improvement.

#### The four algorithms, decoded

"Assume $P$ and $R$ are known" means: **you have been handed the rulebook.** You know the physics of the world exactly. This is the easy case, and it almost never happens — but everything harder is built by degrading from here, so it is worth understanding properly.

**Policy evaluation** asks: *given a fixed policy, how good is every state?* The update

$$V_{k+1}(s)\leftarrow\sum_a\pi(a\mid s)\sum_{s'}P\big[r+\gamma V_k(s')\big]$$

is the Bellman expectation equation with the arrow $\leftarrow$ instead of $=$. That change matters. Recall §0.11: **$\leftarrow$ is an assignment, a line of code, not an equation to solve.** It says "compute the right-hand side using your *current* guesses $`V_k`$, and store the result as your new guess $`V_{k+1}`$." You are repeatedly plugging your estimates back into themselves.

**Policy improvement** asks: *given accurate values, what should I do?* $`\pi'(s) = \arg\max_a Q^\pi(s,a)`$ — look one step ahead using the values you just computed, and act greedily.

▸ **Policy iteration alternates the two, and the claim that it terminates is stronger than it first appears.** With $\lvert\mathcal{A}\rvert$ actions and $\lvert\mathcal{S}\rvert$ states there are $\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$ deterministic policies — an astronomically large but **finite** number. Each improvement step produces a policy that is strictly better (or you stop). A strictly increasing walk through a finite set must end. **That is the entire termination proof, and it is a counting argument, not an analytic one.**

> **Analogy for policy iteration.** You are learning a commute. **Evaluation** is driving your current route enough times to know how long each leg takes. **Improvement** is looking at those timings and saying "at that junction I should turn left instead." Then you evaluate the new route, spot another improvement, and so on. Because each new route is  faster and there are finitely many routes, you must eventually stop finding improvements — at which point you are on the fastest route.

**Value iteration** is the impatient version. Instead of evaluating a policy to convergence and *then* improving, it does one evaluation sweep with the $\max$ baked in. It never represents a policy at all until the very end. In practice it is what people implement, because a fully converged evaluation is wasted work when you are about to change the policy anyway.

### Why it converges: the contraction argument

Define the Bellman optimality operator $\mathcal{T}$:
$$(\mathcal{T}V)(s) = \max_a\sum_{s'}P(s'|s,a)\big[R(s,a)+\gamma V(s')\big]$$

▸ **Claim: $\mathcal{T}$ is a $\gamma$-contraction in the sup norm.**

$$\|\mathcal{T}V-\mathcal{T}W\|_\infty = \max_s\left|\max_a\sum_{s'}P[R+\gamma V(s')] - \max_a\sum_{s'}P[R+\gamma W(s')]\right|$$

Using $`|\max_af(a)-\max_ag(a)|\le\max_a|f(a)-g(a)|`$:

$$\le\max_{s,a}\ \gamma\sum_{s'}P(s'|s,a)\,|V(s')-W(s')| \ \le\ \gamma\max_{s'}|V(s')-W(s')| = \gamma\|V-W\|_\infty$$

using $`\sum_{s'}P=1`$. ∎

▸ **By the Banach fixed-point theorem, $\mathcal{T}$ has a unique fixed point $`V^*`$ and value iteration converges to it geometrically: $`\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty`$.**

**This is the theoretical foundation of all of RL, and it is why $\gamma<1$ matters mathematically rather than just practically.** It also shows the convergence rate degrades as $\gamma\to1$ — long horizons are provably harder.

**Complexity:** $O(|\mathcal{S}|^2|\mathcal{A}|)$ per sweep. Tabular DP is infeasible for real problems (backgammon has $10^{20}$ states, Go has $10^{170}$) — hence function approximation, and hence everything in Chapter 27.

#### Unpacking the contraction proof

This is the most theoretically loaded page in the chapter, and it rests on two ideas: a **norm** that measures worst-case error, and a **contraction** that shrinks it. Neither is hard.

**First, the sup norm.** $`\lVert V - W\rVert_\infty`$ is read *"the infinity-norm of V minus W,"* also called the **sup norm** or **max norm**. From §1.1.4:

$$\lVert V - W\rVert_\infty = \max_s \lvert V(s) - W(s)\rvert$$

*"Go through every state, compute how far apart the two value functions are there, and report the single largest disagreement."* Not the average disagreement — the **worst** one. This choice is deliberate: a guarantee about the worst state is a guarantee about every state.

**Second, the operator.** $\mathcal{T}$ is a **function whose input is an entire value function and whose output is another entire value function.** That is a strange kind of object the first time you meet it, so name it plainly: $\mathcal{T}$ is "one sweep of value iteration," packaged as a single symbol so you can do algebra with it. $(\mathcal{T}V)(s)$ means "apply one sweep to $V$, then read off the answer at state $s$."

**Third, the claim.** "$\mathcal{T}$ is a $\gamma$-contraction" means:

$$\lVert \mathcal{T}V - \mathcal{T}W\rVert_\infty \le \gamma\,\lVert V - W\rVert_\infty$$

*"Take any two value functions. Apply one sweep to each. They are now at least $\gamma$ times closer together than they were."* With $\gamma = 0.9$, one sweep shrinks every disagreement by at least 10%.

> **Analogy.** Photocopy a picture at 90% scale, then photocopy the copy, and again. Any two starting pictures, however different, converge to the same vanishing dot. Or, more precisely: take a map of the room you are standing in and drop it on the floor. **There is exactly one point on the map lying directly above the point of the room it represents** — and there is always exactly one, no matter how you crumple or rotate the map, as long as it is smaller than the room. That is the Banach fixed-point theorem, and $`V^*`$ is that point.

**Why the proof works, in words.** Two facts do all the labour:

1. $`\lvert\max_a f(a) - \max_a g(a)\rvert \le \max_a\lvert f(a)-g(a)\rvert`$ — *"the best of one list can't differ from the best of another by more than the biggest single difference between them."* This is the step that survives the nonlinearity of the $\max$.
2. $`\sum_{s'}P(s'\mid s,a) = 1`$ — probabilities sum to one, so averaging a set of numbers can never exceed the largest of them.

Everything else cancels. **The immediate reward $r$ appears identically on both sides and subtracts away**, which is why the $\gamma$ survives alone: the only thing that carries forward from one sweep to the next is the *discounted* future term.

▸ **Put numbers on the convergence rate.** Start with $`V_0 = 0`$ everywhere when the true values are up to $100$, so the initial error is $100$. How many sweeps until the worst state is within $0.01$? You need $\gamma^k \cdot 100 < 0.01$, i.e. $\gamma^k < 10^{-4}$:

| $\gamma$ | Effective horizon | Sweeps needed |
|---|---|---|
| $0.9$ | 10 | 88 |
| $0.99$ | 100 | 917 |
| $0.999$ | 1000 | 9206 |

**Ten times the horizon costs ten times the iterations.** This is the precise sense in which "long horizons are provably harder" — it is not a vague difficulty, it is a linear cost in $\frac{1}{1-\gamma}$ that shows up in every algorithm downstream, sampled or not.

**And the complexity line, made concrete.** $O(\lvert\mathcal{S}\rvert^2\lvert\mathcal{A}\rvert)$ per sweep: for each of $\lvert\mathcal{S}\rvert$ states, for each of $\lvert\mathcal{A}\rvert$ actions, sum over all $\lvert\mathcal{S}\rvert$ possible next states. Backgammon's $10^{20}$ states would require $10^{40}$ operations *per sweep*. At $10^{18}$ operations per second — roughly a large modern supercomputer — that is about $10^{22}$ seconds, some **ten thousand times the age of the universe**, for one sweep. **Tabular dynamic programming is not slow for backgammon. It is impossible, by a margin that no hardware improvement will ever touch.** Hence function approximation; hence Chapter 27.

> **Where this came from.** The **Banach fixed-point theorem** — also called the contraction mapping principle — was proved by **Stefan Banach** in his 1922 doctoral thesis, on integral equations. Banach was the central figure of the Lwów school of mathematics, whose members famously worked in a café, writing problems and solutions in a notebook kept by the waiters that became known as the *Scottish Book*. The theorem was intended to establish existence and uniqueness of solutions to differential and integral equations; it is now the reason your Q-learning agent converges. **Every convergence proof in tabular RL is a corollary of a 1922 result about integral equations**, which is a fair summary of how much of this field is borrowed.

---

## 26.5 Learning without a model

### Monte Carlo

Run a full episode, then set $`V(s)\leftarrow V(s)+\alpha[G_t - V(s)]`$.

Unbiased. **Zero bias, high variance.** Requires episodes to terminate. Cannot learn online.

### Temporal difference

▸ $$V(S_t)\leftarrow V(S_t)+\alpha\underbrace{\big[\,\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\,\big]}_{\text{TD error }\delta_t}$$

**Bootstrapping:** the target uses the current estimate $`V(S_{t+1})`$ rather than the actual return.

▸ **Biased (the target depends on a wrong estimate), but far lower variance** — one reward and one value estimate, instead of a sum of hundreds of random rewards. Learns online, from incomplete episodes, and empirically converges much faster.

**MC vs TD, the deepest distinction:** MC converges to the value estimate that minimizes squared error on the observed returns. **TD converges to the value function of the maximum-likelihood MDP implied by the data.** So TD exploits the Markov structure and MC does not — which is why TD wins when the Markov assumption holds and MC can be more robust when it doesn't.

#### Reading the temporal-difference update

Everything in §26.4 assumed you had been handed the rulebook $P$. **You have not been handed the rulebook.** "Learning without a model" means you must estimate values from experience alone — you get to *play*, and that is all.

Both methods share a shape you should learn to recognise on sight:

$$\text{new estimate} \leftarrow \text{old estimate} + \alpha\big[\,\text{target} - \text{old estimate}\,\big]$$

Read it: *"move your current guess a fraction $\alpha$ of the way toward the new evidence."* The bracket is the **error**; $\alpha$ is how seriously you take it.

▸ **This is an exponential moving average in disguise, and it is the same update as every online-learning rule in the book.** With $\alpha = 0.1$ you keep 90% of what you believed and take 10% of what you just saw. With $\alpha = 1$ you throw away your history entirely and believe only the last thing that happened. **The single design choice in all of value learning is what to put in the "target" slot.**

**Monte Carlo puts the actual return there.** Play the whole episode, see what you really scored, use that. Honest but slow — you must wait for the end, and the number you get is one sample of a very noisy random variable.

**Temporal difference puts a guess there:**

$$V(S_t)\leftarrow V(S_t)+\alpha\big[\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\big]$$

The TD target $`R_{t+1} + \gamma V(S_{t+1})`$ is *"the one reward I actually observed, plus my current guess about everything after that."* The bracket as a whole is the **TD error** $`\delta_t`$, read *"delta-t."*

▸ **The TD error is literally surprise: what happened, minus what you expected.** If $`\delta_t > 0`$, things went better than predicted and you raise your estimate. If $`\delta_t < 0`$, worse than predicted; lower it. If $`\delta_t = 0`$, the world behaved exactly as forecast and nothing needs to change.

> **Analogy (this is the standard one, and it is very good).** You leave the office at 6:00 and predict you will be home at 6:30. At 6:10 you reach the motorway and it is jammed — you now predict 6:50. **Do you wait until you get home to learn something?** Monte Carlo says yes: only the actual arrival time counts as data. Temporal difference says no — the moment your own forecast jumps from 6:30 to 6:50, that revision *is* the signal. You learned "leaving at 6:00 is worse than I thought" at 6:10, twenty minutes before the evidence technically arrived. **Bootstrapping is the formal version of trusting your own updated forecast.**

#### Bootstrapping, and why "biased but lower variance" is the right trade

**Bootstrapping** means using an estimate to update an estimate. It sounds circular and disreputable, and mathematically it *is* the reason all the trouble in §26.7 exists — but the payoff is enormous.

**Put numbers on the variance claim.** Suppose an episode is 100 steps long and each step's reward is random with standard deviation $1$. The Monte Carlo return sums 100 such random numbers, so (for roughly independent rewards) its variance is about $100$ and its standard deviation about $10$. The TD target contains **one** random reward plus a value estimate, so its standard deviation is about $1$.

▸ **A tenfold reduction in the noise of every single update.** Since the number of samples needed to resolve a signal scales with the *square* of the noise (§1.3.1), that is a hundredfold reduction in data required — bought at the price of a bias that shrinks as $V$ improves. **That is the deal, and it is why essentially nobody uses pure Monte Carlo.**

#### What "MC minimizes squared error, TD finds the maximum-likelihood MDP" means

This distinction sounds abstract until you see it produce two different answers on the same data. Here is the classic demonstration. You observe eight episodes in a world with two states, $A$ and $B$, and $\gamma = 1$:

| Episode | What happened |
|---|---|
| 1 | $A$, reward 0, then $B$, reward 0, end |
| 2–7 | $B$, reward 1, end (six times) |
| 8 | $B$, reward 0, end |

**What is $V(B)$?** Both methods agree: $B$ paid $1$ six times out of eight, so $V(B) = 0.75$.

**What is $V(A)$?** Now they disagree.

- **Monte Carlo** looks at the returns actually observed from $A$. There is exactly one such episode, and it returned $0$. So $\mathbf{V(A) = 0}$. This  does minimize squared error on the observed returns — you saw one number, and $0$ is the value that fits it perfectly.
- **Temporal difference** notices the *structure*: every time we were in $A$ we got $0$ and moved to $B$. So $V(A) = 0 + V(B) = \mathbf{0.75}$.

▸ **Both answers are correct answers to different questions.** MC answers "what happened from $A$?" TD answers "what does the model implied by all my data predict from $A$?" — and TD's answer is the one that uses episodes 2–8, which never visited $A$ at all. **That is the payoff of assuming Markov: information about $B$ transfers to $A$ for free.** If the world really is Markov, TD is using data that MC throws away. If it is not, TD is confidently propagating a false assumption, and MC's stubbornness becomes a virtue.

> **Where this came from.** The first working temporal-difference learner predates the name by twenty-five years. **Arthur Samuel**, at IBM in the 1950s, wrote a checkers program that improved by comparing its evaluation of the current position with its evaluation of a position several moves later, adjusting the earlier estimate toward the later one — bootstrapping, in all but name. His 1959 paper *Some Studies in Machine Learning Using the Game of Checkers* is also where the phrase **"machine learning"** enters the literature.
>
> The method was isolated, named, and analysed by **Richard Sutton** in his 1984 PhD work with Andrew Barto and in the 1988 paper *Learning to Predict by the Methods of Temporal Differences*. The vindication arrived in the early 1990s from **Gerald Tesauro at IBM**, whose **TD-Gammon** combined TD($\lambda$) with a neural network and learned backgammon almost entirely from self-play, reaching a level competitive with the best human players. Its most remarkable legacy is that **human experts changed their opening theory in response to the program's preferences** — a machine-learning system that taught its own domain something new, in 1992.

> **The story behind the dopamine connection.** In 1997, **Wolfram Schultz, Peter Dayan and Read Montague** published a paper in *Science* showing that the firing pattern of dopamine neurons in the primate midbrain matches the **temporal-difference error** with striking fidelity. The neurons fire when an unexpected reward arrives; after a cue reliably predicts the reward, they stop firing at the reward and fire at the *cue* instead; and if the predicted reward fails to appear, their firing rate **drops below baseline at exactly the expected time**. That is $`\delta_t`$, positive, shifted, and negative, in living tissue. An algorithm invented to make computers play backgammon turned out to describe a mechanism the brain had been running for a very long time — one of the few  bidirectional exchanges between machine learning and neuroscience.

### $n$-step and TD($\lambda$)

$$G_t^{(n)} = R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n})$$

$n=1$ is TD, $n=\infty$ is MC. Intermediate $n$ (3–10) is usually best — an explicit bias–variance dial.

**TD($\lambda$)** takes a geometrically weighted average of all $n$-step returns:
▸ $$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

Implemented online with **eligibility traces**: $`e_t(s)=\gamma\lambda e_{t-1}(s)+\mathbb{1}[S_t=s]`$, then update every state by $`\alpha\delta_te_t(s)`$.

▸ **The trace is a short-term memory of which states were recently visited, so one TD error can be assigned to all of them at once.** It is the elegant solution to credit assignment, and it reappears as GAE in Chapter 27 §27.6 — GAE *is* TD($\lambda$) applied to advantages.

#### Unpacking $n$-step returns and TD($\lambda$)

$$G_t^{(n)} = R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n})$$

Read aloud: *"the n-step return equals n real rewards, discounted, plus a discounted estimate of everything after that."* The superscript $(n)$ in parentheses is a **label, not a power** — it names which member of the family this is.

**The structure is: truth for a while, then a guess.**

| Estimator | Real rewards used | Where the guess starts |
|---|---|---|
| $n=1$ (TD) | 1 | immediately |
| $n=3$ | 3 | after three steps |
| $n=\infty$ (MC) | all of them | never |

▸ **$n$ is a dial between honesty and noise.** Every extra real reward you include removes a little bias (you're relying less on a possibly-wrong $V$) and adds a little variance (you're absorbing one more roll of the dice). Three to ten real steps is usually the sweet spot, and that empirical fact reappears untouched in Chapter 27's DQN improvements.

**Now TD($\lambda$).** Rather than committing to one $n$, average all of them:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

The weights are $(1-\lambda)$, $(1-\lambda)\lambda$, $(1-\lambda)\lambda^2$, … — a **geometric decay**. Check they are a legitimate weighting: $`(1-\lambda)\sum_{n\ge1}\lambda^{n-1} = (1-\lambda)\cdot\frac{1}{1-\lambda} = 1`$. Good — the weights sum to one, so this is a  weighted average and not a rescaling.

**Put numbers in with $\lambda = 0.9$:**

| $n$ | Weight $(1-\lambda)\lambda^{n-1}$ |
|---|---|
| 1 | $0.100$ |
| 2 | $0.090$ |
| 3 | $0.081$ |
| 10 | $0.039$ |
| 30 | $0.0047$ |

The 1-step return gets 10% of the vote, the 10-step return about 4%, and by $n=30$ the contribution is negligible. **$\frac{1}{1-\lambda} = 10$ is the effective averaging horizon**, exactly parallel to $\frac{1}{1-\gamma}$ for the discount.

The two extremes fall out immediately: at $\lambda = 0$ all the weight lands on $n=1$ and you have plain TD; at $\lambda \to 1$ the weight slides out to $n = \infty$ and you have Monte Carlo. **$\lambda$ is a continuous knob between the two algorithms of §26.5.**

#### What an eligibility trace actually does

$$e_t(s)=\gamma\lambda\, e_{t-1}(s)+\mathbb{1}[S_t=s]$$

Read: *"the trace for state $s$ decays by a factor $\gamma\lambda$ each step, and gets $+1$ added whenever you are actually standing in $s$."* The indicator $`\mathbb{1}[S_t = s]`$ is 1 if you are in $s$ right now and 0 otherwise (§0.5) — an `if` statement written as a symbol.

Then every state is updated by $`\alpha\,\delta_t\,e_t(s)`$: **one surprise, distributed across every state in proportion to how recently you were there.**

> **Analogy.** Walk across a floor with wet shoes. Each footprint starts dark and dries steadily. When something notable happens, you look down and blame each footprint in proportion to how wet it still is — the step you just took gets most of the credit, the one from ten paces back gets a little, and the one from a hundred paces back has dried and gets none. $\gamma\lambda$ is the drying rate.

▸ **Why this is the elegant solution to credit assignment.** The forward view ("average all $n$-step returns") requires waiting for the future. The backward view (traces) achieves the *same updates* while running strictly online, in one pass, with one extra number per state. **You never wait, and you never store the trajectory.** For an agent living in real time, that is the difference between an algorithm and a thought experiment.

**A worked micro-example.** With $\gamma = 0.9$, $\lambda = 0.8$, so $\gamma\lambda = 0.72$. You visit states $`s_1, s_2, s_3`$ in order, and at $`s_3`$ receive a large surprise $\delta = +10$ with $\alpha = 0.1$. The traces are $`e(s_3)=1`$, $`e(s_2)=0.72`$, $`e(s_1)=0.72^2 = 0.518`$. So the updates are $+1.0$, $+0.72$, and $+0.52$ respectively. **The reward reaches back three states in a single step**, where one-step TD would have needed three separate visits to propagate the same information.

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

**Expected SARSA:** replace $`Q(S_{t+1},A_{t+1})`$ with $`\sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a')`$. Removes the variance from sampling $`A_{t+1}`$; strictly better than SARSA at the same cost.

**Convergence guarantee (tabular):** Q-learning converges to $`Q^*`$ with probability 1 provided all state–action pairs are visited infinitely often and the step sizes satisfy the Robbins–Monro conditions $`\sum_t\alpha_t=\infty`$, $`\sum_t\alpha_t^2<\infty`$.

#### SARSA and Q-learning, decoded

Put the two updates side by side and cover everything except the target. That is the only thing that differs.

| Algorithm | Target | In words |
|---|---|---|
| SARSA | $`R_{t+1}+\gamma Q(S_{t+1},A_{t+1})`$ | "…plus the value of **what I will actually do next**" |
| Q-learning | $`R_{t+1}+\gamma \max_{a'}Q(S_{t+1},a')`$ | "…plus the value of **the best thing available next**" |

▸ **One uses the action you took; the other uses the action you wish you'd take.** Everything else about the two algorithms is identical.

**Where the name SARSA comes from:** the update needs the quintuple $`(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})`$ — **S**tate, **A**ction, **R**eward, **S**tate, **A**ction. The algorithm is named after its own argument list. **Q-learning needs only $`(S_t, A_t, R_{t+1}, S_{t+1})`$** — no next action, because it takes a $\max$ instead of asking what you did. That difference of one symbol is what makes Q-learning off-policy, and it is worth being able to state that compactly.

**On-policy versus off-policy, said plainly:**

- **On-policy** — "I learn about the policy I am using." The data and the thing being learned are the same object.
- **Off-policy** — "I use data generated by one policy to learn about a different one." The data is generated by a **behaviour policy** $b$; the thing being learned is the **target policy** $\pi$.

> **Analogy.** SARSA is a driving instructor who grades you on the drives you actually take, nervous overtakes and all — so the score reflects *you*, learner's mistakes included. Q-learning is a rally coach who watches your practice runs but scores the route as if a perfect driver were behind the wheel. The rally coach identifies the faster line; the driving instructor tells you which line is survivable **for someone who still makes mistakes.** In Cliff Walking, the cliff is real and you do still make mistakes, so the instructor's advice keeps you alive.

▸ **The Cliff Walking result is not a defect of either algorithm — it is the two of them answering different questions correctly.** If you will eventually stop exploring, you want Q-learning's answer. If you must perform well *while* exploring — a robot with real hardware, a live recommendation system, a medical policy — you want SARSA's.

#### Unpacking the Robbins–Monro conditions

$$\sum_t\alpha_t=\infty \qquad\text{and}\qquad \sum_t\alpha_t^2<\infty$$

Two conditions that look opposed and are not. Read them as **"big enough for long enough, but shrinking."**

- $`\sum_t \alpha_t = \infty`$ — *"the total distance you are able to travel is unlimited."* If the step sizes shrank too fast, you could run out of movement before reaching the answer, like a runner whose stride halves every step and who therefore never crosses the finish line.
- $`\sum_t \alpha_t^2 < \infty`$ — *"the accumulated noise is finite."* Errors contribute in proportion to $\alpha^2$ (variance scales with the square of a scale factor), so this says the random jitter you absorb over all time adds up to something bounded, and the estimate can settle.

**Does anything satisfy both?** Yes: $`\alpha_t = 1/t`$. The harmonic series $\sum 1/t$ diverges (slowly — it grows like $\ln t$), while $\sum 1/t^2 = \pi^2/6 \approx 1.645$ converges. It is a narrow window, and $1/t$ sits almost exactly in it.

| Schedule | $`\sum\alpha_t`$ | $`\sum\alpha_t^2`$ | Converges? |
|---|---|---|---|
| $`\alpha_t = 1/t`$ | $\infty$ ✓ | $1.645$ ✓ | **Yes** |
| $`\alpha_t = 1/t^2`$ | $1.645$ ✗ | finite ✓ | No — stops moving too soon |
| $`\alpha_t = \alpha`$ (constant) | $\infty$ ✓ | $\infty$ ✗ | No — never stops jittering |

▸ **And yet essentially every practical implementation uses a constant $\alpha$**, knowingly violating the second condition. The reason is that a constant step size never stops adapting, which is exactly what you want when the environment is non-stationary — and in RL the environment *is* effectively non-stationary, because your own policy keeps changing. **You trade guaranteed convergence to a point for permanent tracking of a moving target.** That is a deliberate and almost universal choice, and it is worth knowing that the theory and the practice differ here on purpose.

> **Where this came from.** The two conditions are from **Herbert Robbins and Sutton Monro's** 1951 paper *A Stochastic Approximation Method*, which asked a question with nothing to do with agents or rewards: how do you find the root of a function you can only observe through noise? Their answer — take shrinking steps in the direction the noisy measurement suggests — is the direct ancestor of stochastic gradient descent (Ch. 4) as well as of Q-learning. (A coincidence worth enjoying: the co-author's given name is Sutton, and the field's other Sutton, Richard, is unrelated.)

> **The story behind Q-learning.** **Chris Watkins** introduced it in his 1989 Cambridge PhD thesis, *Learning from Delayed Rewards*, and the letter $Q$ is simply the symbol he used for the quality function — there is no deeper meaning. What made it land was the proof, published with **Peter Dayan in 1992**, that the tabular algorithm converges to $`Q^*`$ with probability one. **An off-policy method that provably converges without ever needing to know the environment's dynamics was a  surprising result**, and it is why Q-learning rather than SARSA became the default. SARSA itself was published by **Gavin Rummery and Mahesan Niranjan in 1994**, under the less memorable name "modified connectionist Q-learning"; the acronym SARSA was suggested by Richard Sutton and stuck immediately, which is a small lesson in the value of a good name.

### Maximization bias

▸ $`\mathbb{E}[\max_a\hat Q(a)] \ge \max_a\mathbb{E}[\hat Q(a)]`$ by Jensen — **the max of noisy estimates is systematically too large.** Q-learning uses the same values to *select* and to *evaluate* the best action, so it is biased upward.

**Double Q-learning** fixes it by maintaining two estimates and decoupling the roles:
$$Q_1(s,a)\leftarrow Q_1(s,a)+\alpha\Big[r+\gamma\,Q_2\big(s',\ \arg\max_{a'}Q_1(s',a')\big)-Q_1(s,a)\Big]$$
One network picks the action, the other scores it. **This is the direct ancestor of Double DQN** (Ch. 27 §27.3).

#### Maximization bias, decoded

$$\mathbb{E}\big[\max_a\hat Q(a)\big] \ \ge\ \max_a\mathbb{E}\big[\hat Q(a)\big]$$

Read the left side: *"first take the maximum of your noisy estimates, then average over the noise."* Read the right side: *"first average away the noise, then take the maximum."* The hat on $\hat Q$ means **estimate**, not truth (§0.6). The claim is that doing the max first gives a systematically **larger** answer.

**Why, in one sentence: the maximum deliberately selects for whichever estimate happened to be lucky.**

**Put real numbers in.** Suppose two actions are  worth exactly $0$ each, and your estimates are unbiased but noisy — each is a draw from $\mathcal{N}(0,1)$. The right-hand side is $\max(0, 0) = 0$. The left-hand side is the expected maximum of two independent standard normals, which is $1/\sqrt{\pi} \approx 0.564$.

**Your value estimate for a worthless state is $+0.56$, and you did nothing wrong.** Both estimates were unbiased. The bias was created by the act of selection.

It gets worse with more choices:

| Number of equally-worthless actions | Expected value of the max |
|---|---|
| 2 | $\approx 0.56$ |
| 5 | $\approx 1.16$ |
| 10 | $\approx 1.54$ |
| 50 | $\approx 2.25$ |

> **Analogy.** Ask a hundred people to guess a stranger's age, then report *the highest guess* as your estimate. The individual guesses might be beautifully unbiased; the maximum of them is guaranteed to be too high, and the more guessers you poll, the worse it gets. **This is the winner's curse from auction theory** — the winning bid is, by construction, the bid from whoever most overestimated the item.

▸ **Bootstrapping turns this from an annoyance into a disease.** The inflated value of $s'$ becomes the target for $s$, whose inflated value becomes the target for its predecessor, and so on backwards. **Optimism compounds along the chain of states.** This is precisely why Q-learning with function approximation is famous for producing value estimates that climb steadily while actual performance does not.

**How Double Q-learning fixes it.** Notice that the single $\max$ is doing two jobs at once: *choosing* which action is best, and *reporting* how good it is. Both jobs use the same noisy numbers, so the noise that won the selection is the same noise that inflates the report.

$$Q_1(s,a)\leftarrow Q_1(s,a)+\alpha\Big[r+\gamma\,Q_2\big(s',\ \arg\max_{a'}Q_1(s',a')\big)-Q_1(s,a)\Big]$$

Read the inner part: *"ask $`Q_1`$ which action looks best, then ask $`Q_2`$ how good that action actually is."* Since $`Q_2`$'s noise is independent of $`Q_1`$'s, the action that got lucky in $`Q_1`$ has no reason to be lucky in $`Q_2`$ as well. **The selection and the evaluation no longer share a coin flip.**

> **Analogy.** Do not let the same person both nominate the winner and set the prize money. Have one panel pick, and an independent panel value.

> **Where this came from.** **Hado van Hasselt** introduced Double Q-learning in 2010, and carried it to deep networks as **Double DQN** in 2016 with Arthur Guez and David Silver at DeepMind. The 2016 paper's most persuasive contribution was diagnostic rather than algorithmic: it showed that the original DQN's value estimates on Atari were dramatically higher than the returns the agent was actually achieving — **the network was confidently predicting scores it never scored** — and that the one-line fix removed most of the gap.

---

## 26.7 Off-policy learning and importance sampling

To evaluate a target policy $\pi$ using data from a behaviour policy $b$:

▸ $$\rho_{t:T} = \prod_{k=t}^{T}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},\qquad V^\pi(s)=\mathbb{E}_b[\rho_{t:T}G_t\mid S_t=s]$$

▸ **The variance of a product of $T$ ratios explodes exponentially in $T$.** A single trajectory where $\pi$ and $b$ disagree early can have $\rho=10^6$ or $\rho=0$. This is why naive off-policy Monte Carlo is unusable beyond short horizons.

**Mitigations:** weighted (self-normalized) importance sampling — biased but dramatically lower variance and consistent; per-decision IS; and clipping the ratio, which is exactly what PPO does (Ch. 27 §27.7).

▸ **Q-learning avoids importance sampling entirely** because its target uses $`\max_{a'}`$, which doesn't depend on which policy generated the data. That is the real reason it is the workhorse of off-policy RL.

### The deadly triad

▸ **Function approximation + bootstrapping + off-policy training** — any two are safe; **all three together can diverge**, even in tiny linear problems (Baird's counterexample).

**Why:** bootstrapping means the target moves with the parameters; off-policy means the distribution of updates doesn't match the distribution the value function is being fit under; function approximation means an update at one state changes others. The composition need not be a contraction in any norm.

**This is the fundamental instability of deep RL**, and every trick in Chapter 27 — target networks, replay buffers, trust regions, gradient clipping, conservative updates — is a partial countermeasure.

#### Reading the importance-sampling ratio

$$\rho_{t:T} = \prod_{k=t}^{T}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}$$

Read aloud: *"rho, from t to T, equals the product over k of pi of A-k given S-k, divided by b of A-k given S-k."* The $\prod$ is a `for` loop that **multiplies** instead of adding (§0.3).

The idea underneath is simple and reasonable. You have data collected by policy $b$ (the **behaviour** policy — what you actually did), and you want to know how policy $\pi$ (the **target** policy — what you're curious about) would have fared. You cannot re-run history, so instead you **re-weight** it: if $\pi$ would have chosen this action three times as often as $b$ did, count this trajectory three times as heavily.

- Ratio $> 1$: $\pi$ likes this action more than $b$ did → **upweight** this experience.
- Ratio $< 1$: $\pi$ likes it less → **downweight** it.
- Ratio $= 0$: $\pi$ would never do this → **discard** the entire trajectory, and everything after it too.

> **Analogy.** You surveyed a crowd that turned out to be 80% students, but the population you care about is 20% students. You don't re-run the survey — you re-weight each student's answer by $\frac{0.2}{0.8} = 0.25$ and each non-student's by $\frac{0.8}{0.2} = 4$. Importance sampling is that correction, applied to a *sequence* of decisions rather than one.

▸ **And the sequence is where it dies.** Because the corrections multiply, the variance compounds exactly like the $\lambda^k$ of §1.1.2 and the $1.2^{50}$ of §1.1.4 — the same mathematics, a third time.

**Put real numbers on the explosion.** Two actions. $b$ picks each with probability $0.5$; $\pi$ picks the first with probability $0.9$ and the second with $0.1$. The per-step ratio is either $\frac{0.9}{0.5}=1.8$ or $\frac{0.1}{0.5}=0.2$, each equally likely under $b$. Its mean is $0.5(1.8)+0.5(0.2)=1$ exactly — **unbiased, as promised.**

Now compound it over a 50-step episode:

| Quantity | Value over 50 steps |
|---|---|
| Mean of $\rho$ | $1$ (exactly — still unbiased) |
| Largest possible $\rho$ | $1.8^{50} \approx 5.8\times10^{12}$ |
| Smallest possible $\rho$ | $0.2^{50} \approx 10^{-35}$ |
| Standard deviation of $\rho$ | $\approx 2.3\times10^{5}$ |

▸ **An estimator with mean 1 and standard deviation 230,000 is not an estimator; it is a lottery.** Almost every trajectory contributes essentially nothing, and once in a very long while one contributes everything. To average that down to a useful answer you would need on the order of $(2.3\times10^5)^2 \approx 5\times10^{10}$ trajectories — and the policies in this example were not even very different. **This is why naive off-policy Monte Carlo is unusable beyond short horizons**, stated with numbers rather than adjectives.

The three named mitigations, decoded:

- **Weighted (self-normalized) importance sampling** — divide by the sum of the weights rather than by the count. Introduces a small bias but drops the variance enormously, and stays **consistent** (the bias vanishes as data grows). Nearly always the right choice in practice.
- **Per-decision IS** — notice that a reward at step $k$ only needs correcting for the decisions *up to* step $k$, not the whole episode. Shorter products, smaller variance, free.
- **Clipping the ratio** — simply refuse to let $\rho$ exceed some bound. Crude, biased, and spectacularly effective. ▸ **This is exactly what PPO does** (Ch. 27 §27.7), which means the most widely deployed policy-gradient algorithm in the world is, at its core, a variance-control hack on a 1950s statistical technique.

▸ **And now the reason Q-learning matters so much.** Its target is $`\max_{a'}Q(s',a')`$. Look at what is *absent*: there is no $\pi(a'\mid s')$, no $b(a'\mid s')$, no ratio. The $\max$ does not care which policy generated the data, because it is not averaging over a policy at all. **Q-learning gets off-policy learning for free by never taking an expectation over actions**, and that structural accident is why it, rather than importance-sampled Monte Carlo, became the workhorse.

#### The deadly triad, decoded

Three ingredients, each individually reasonable:

| Ingredient | What it means | Why you want it |
|---|---|---|
| **Function approximation** | A network stands in for the table $Q(s,a)$ | The table has $10^{170}$ rows. There is no alternative |
| **Bootstrapping** | Targets are built from your own current estimates | The 100× variance reduction of §26.5 |
| **Off-policy training** | Learning about $\pi$ from data generated by $b$ | Replay buffers, sample reuse, learning the greedy policy while exploring |

▸ **Any two are provably safe. All three together can diverge — not "perform poorly," but blow up to infinity — even in a linear problem with seven states.** That last part is Baird's counterexample, and its smallness is the point: this is not a big-model phenomenon that more compute will fix.

**Why the composition breaks, in words.** Take them pairwise and each pair has a reason to be safe:

- Tabular + bootstrapping + off-policy = Q-learning, which converges (§26.6) because a table can change one entry without touching any other.
- Function approximation + bootstrapping + **on**-policy = TD with a network, which is well behaved because the states you update are the states you visit, so errors are corrected where they occur.
- Function approximation + **no** bootstrapping + off-policy = plain supervised regression on Monte Carlo returns, which is just least squares and cannot diverge.

Now put all three together. **Bootstrapping** means every update changes the target of other updates. **Function approximation** means an update at one state leaks into states you did not visit, because they share parameters. **Off-policy** means the states you keep updating are *not* the states whose errors are being measured — so the leaked errors are never audited. The result is a feedback loop with no thermostat: an over-estimate at an unvisited state raises the target at a visited one, which raises the parameters, which raises the unvisited estimate further.

> **Analogy.** A rumour spreads through an office (function approximation: everyone shares information). Each person's confidence is based on hearing it from others rather than from the source (bootstrapping: estimates built on estimates). And nobody ever checks with the one person who was actually there (off-policy: you never sample the states you are inflating). **Any two of these is survivable. All three and the rumour grows without bound.**

> **Where this came from.** The counterexample is **Leemon Baird's**, from a 1995 paper on residual algorithms; it is a seven-state Markov process with linear function approximation on which off-policy TD's parameters diverge to infinity, and it is small enough to work through by hand. The name **"deadly triad"** is Richard Sutton's, popularized in the second edition of Sutton and Barto. **The counterexample is thirty years old and no method has made it go away** — every technique in Chapter 27 manages the instability rather than eliminating it, which is the honest way to read that chapter.

---

## 26.8 Exploration

### The dilemma
Exploit what you know, or explore to learn more. **The information value of an action is not in its reward.**

### Multi-armed bandits — the clean case

**$\epsilon$-greedy.** Simple; **linear regret** $\Theta(\epsilon T)$ if $\epsilon$ is fixed, since you keep exploring forever. Decay $`\epsilon_t\propto1/t`$ for logarithmic regret.

**UCB1:**
▸ $$a_t=\arg\max_a\left[\hat\mu_a + c\sqrt{\frac{\ln t}{N_a}}\right]$$

**"Optimism in the face of uncertainty."** The bonus is a confidence-interval width from Hoeffding (Ch. 2 §2.3): it shrinks as $`1/\sqrt{N_a}`$ with more pulls and grows slowly with $t$ so no arm is abandoned forever.

▸ Regret bound: $`O\!\left(\sum_{a:\Delta_a>0}\frac{\ln T}{\Delta_a}\right)`$ — **logarithmic in $T$**, which matches the Lai–Robbins lower bound up to constants. UCB is essentially optimal.

**Thompson sampling.** Maintain a posterior over each arm's mean; sample one value per arm; play the argmax. Also achieves logarithmic regret, is trivially easy to implement (Beta–Bernoulli conjugacy), and **empirically outperforms UCB** in most practical settings. It naturally handles delayed feedback and batched decisions.

#### What "regret" means, and why it is the right scoreboard

Before decoding UCB, decode **regret**, since every claim in this section is stated in terms of it.

$$\text{Regret}(T) \;=\; \underbrace{T\mu^*}_{\text{what a genius would have scored}} \;-\; \underbrace{\mathbb{E}\Big[\textstyle\sum_{t=1}^{T} \text{reward}_t\Big]}_{\text{what you scored}}$$

*"How much worse did you do than someone who knew the answer from the start?"* $`\mu^*`$ is the mean payoff of the best arm and $`\Delta_a = \mu^* - \mu_a`$ is arm $a$'s **gap** — how much you lose each time you pull it. Regret is therefore just $`\sum_a \Delta_a \times (\text{times you pulled } a)`$.

▸ **The shape of the regret curve is what matters, not its height.** Regret that grows **linearly** in $T$ means you are losing a fixed amount per step forever — you never finish learning. Regret that grows **logarithmically** means the loss per step is going to zero; you make your mistakes early and then stop. **Logarithmic regret is the mathematical statement of "eventually figures it out."**

**Put numbers on the difference.** Run for $T = 10^6$ steps with two arms whose gap is $\Delta = 1$:

| Strategy | Regret at $T = 10^6$ |
|---|---|
| $\epsilon$-greedy, $\epsilon = 0.1$ fixed | $\approx 50{,}000$ |
| UCB1 | $\approx 100$ |

**Five hundred times better**, and the ratio keeps growing with $T$. The reason is stated in the book's own sentence: with fixed $\epsilon$ you *keep exploring forever*, wasting $\epsilon$ of every future step on options you have already established are worse. $\Theta(\epsilon T)$ is that waste, written down.

#### Reading the UCB rule

$$a_t=\arg\max_a\left[\hat\mu_a + c\sqrt{\frac{\ln t}{N_a}}\right]$$

Read aloud: *"pick the arm that maximizes: its estimated mean, plus c times the square root of, log t over N-a."* Every symbol:

- $`\hat\mu_a`$ — "mu-hat-a," your **current estimate** of arm $a$'s average payoff. The hat means estimate (§0.6).
- $`N_a`$ — how many times you have pulled arm $a$ so far.
- $t$ — how many pulls you have made in total, across all arms.
- $c$ — a constant controlling how adventurous you are.
- The square-root term — a **confidence-interval width**, a bonus for ignorance.

▸ **"Optimism in the face of uncertainty" means: judge every option by the best it could plausibly be, not by your best guess of what it is.** An arm you have never tried gets an enormous bonus, so you try it. If it disappoints, $`N_a`$ grows, the bonus shrinks, and you stop. **The strategy explores exactly as much as its own ignorance justifies, and not one pull more.**

**Watch the two forces.** The bonus shrinks as $`\frac{1}{\sqrt{N_a}}`$ — that is the standard-error law from §1.3.1, the same $\sqrt{n}$ that governs every estimate in the book. And it grows as $\sqrt{\ln t}$, very slowly, so that an arm ignored for a long time eventually gets one more look. **Neglect is punished, but only logarithmically.**

Numbers, with $c=1$ and $t = 1000$ (so $\ln t = 6.9$):

| Pulls of arm $a$ so far | Exploration bonus |
|---|---|
| 1 | $2.63$ |
| 10 | $0.83$ |
| 100 | $0.26$ |
| 10{,}000 | $0.026$ |

An arm you have pulled once carries a bonus of $2.6$ — enough to beat almost any established estimate. After ten thousand pulls the bonus is negligible and the arm is judged on its record alone. **Ten times more evidence buys you a bonus $\sqrt{10}\approx3.16$ times smaller**, which is the whole reason data is expensive.

> **Analogy.** Interviewing candidates. The one with a long track record you judge on the track record. The one who has done almost nothing yet, you judge on their ceiling — because the cost of one more interview is small and the upside is unknown. **Once you have interviewed them enough times, their ceiling and their record converge, and the reason to keep looking evaporates.**

**Reading the regret bound.** $`O\!\left(\sum_{a:\Delta_a>0}\frac{\ln T}{\Delta_a}\right)`$ says: for each *suboptimal* arm, you pay about $`\frac{\ln T}{\Delta_a}`$. Note the $`\Delta_a`$ is in the **denominator**, which looks backwards until you think about it: **an arm that is only slightly worse is expensive to rule out** (you need many samples to distinguish it) but each mistake costs little; an arm that is much worse is cheap to rule out but each mistake is costly. The two effects cancel to leave $`\ln T / \Delta_a`$, and the **Lai–Robbins lower bound** of 1985 proves no algorithm can do better. UCB is not merely good; it is essentially the end of the story for this problem.

> **Where "bandit" came from.** A slot machine was called a **one-armed bandit** — one lever, and it robs you. A row of them, each with unknown odds, is a *multi-armed* bandit. The formal problem was posed by **Herbert Robbins in 1952**, and the WWII-era interest in it was intense enough that **Peter Whittle** later joked that during the war the problem so consumed Allied analysts that someone suggested dropping it over Germany to sabotage German science. It is a joke, told as a joke, and worth knowing as an indicator of how  hard the problem was considered.

> **The story behind Thompson sampling.** **William R. Thompson** published the idea in *Biometrika* in **1933**, and it had nothing to do with slot machines: he was asking how to allocate patients between two medical treatments when you do not yet know which works better, so that fewer patients receive the inferior one. **It was then almost entirely ignored for about seventy-five years.** Interest revived around 2010, when empirical studies — notably by Olivier Chapelle and Lihong Li — found the 1933 method beating the carefully-engineered modern algorithms on real advertising data; regret bounds matching UCB's followed shortly after. **The oldest algorithm in the section is the one that works best in practice**, which is a useful corrective to the assumption that the literature moves monotonically forward.

### In full MDPs

Much harder — you must explore *sequences* of actions, and the state you need to reach may be a hundred steps away.

- **Count-based bonuses:** $r^+ = \frac{\beta}{\sqrt{N(s)}}$; in continuous spaces, use pseudo-counts from a density model or hashing.
- **Curiosity / prediction error (ICM, RND):** reward states where a learned model predicts poorly. ▸ **The noisy-TV problem:** a  stochastic element (static on a screen) is permanently unpredictable, so the agent watches it forever. RND partly fixes this by predicting the output of a *fixed random network* — deterministic, so aleatoric noise is learnable and only novelty persists.
- **Bootstrapped DQN / noisy networks:** posterior-sampling-style deep exploration.
- **Go-Explore:** archive visited states, return to promising ones, then explore from there. Solved Montezuma's Revenge, which had defeated every bonus-based method.

▸ **The hard-exploration benchmark to know:** Montezuma's Revenge, where reward is so sparse that random exploration essentially never finds it.

#### Why exploration in an MDP is a different problem entirely

In a bandit, every option is available on every turn. In an MDP, **the option you need may be a hundred correct decisions away**, and you have to make all hundred before you find out whether any of them was right.

**Put the number on it.** Imagine a corridor of 100 rooms; the treasure is in the last one; there are two actions, left and right; and you must go right every single time. A uniformly random policy reaches the treasure with probability

$$2^{-100} \approx 10^{-30}$$

At a million steps per second you would wait roughly $10^{16}$ years. **This is not "exploration is inefficient." This is "exploration never happens."** And a corridor of 100 rooms is a trivially simple environment.

> **Analogy.** A bandit is a menu — you can order anything, any night, and learn from it. An MDP is a mountain range: to learn what's behind the third ridge you must first cross the first two, and if you have never crossed them you have no idea the third exists. **Bandit exploration is about trying things. MDP exploration is about getting somewhere.**

▸ This is why every method in the list is really a way of **manufacturing a reward signal where the environment provides none** — turning "explore" into "collect these fake points," so that the machinery of §§26.3–26.6 can do the rest.

#### The exploration bonuses, decoded

**Count-based:** $r^+ = \frac{\beta}{\sqrt{N(s)}}$ — *"pay a bonus for visiting a state, inversely proportional to the square root of how often you have already been there."* This is UCB's bonus transplanted from arms to states, with the same $1/\sqrt{N}$ decay for the same reason. It works perfectly in gridworlds and fails immediately in Atari, because **no two pixel frames are ever identical, so every state has $N(s)=1$ forever.** Pseudo-counts (fit a density model and ask "how surprising is this state?") and hashing (bucket similar states together) are the two standard repairs.

**Curiosity / prediction error:** reward the agent for going where its own model is wrong, on the theory that model error means novelty. **ICM** is the Intrinsic Curiosity Module; **RND** is Random Network Distillation.

▸ **The noisy-TV problem, and why it is deeper than it looks.** Put a television showing static in the corridor. Static is * random* — no model, however good, can ever predict the next frame. So the prediction error at that state never falls, the curiosity bonus never decays, and **the agent sits and watches the television forever.** It has not malfunctioned; it has correctly maximized the objective you wrote.

The diagnosis is the distinction from Chapter 33: prediction error mixes **aleatoric** uncertainty (irreducible randomness in the world — the static) with **epistemic** uncertainty (your own ignorance — the unexplored corridor). Only the second is worth chasing.

**RND's fix is elegant.** Instead of predicting the environment's next state, predict the output of a **fixed, randomly initialized neural network** applied to the current state. That target is a deterministic function — there is no noise in it, by construction. So the error can always be driven to zero given enough visits, and a persistently high error can only mean *"I have not been here often."*

> **Analogy.** Rather than asking "can I predict what happens next?" — which is unanswerable in a casino — RND asks "have I seen this before?", by checking whether it can reproduce an arbitrary but consistent fingerprint of the state. **You cannot memorize static, but you can certainly memorize that you have stood in this room.**

**Go-Explore** attacks the problem from the opposite direction. Bonus-based methods assume that if a state is interesting you will find your way back to it — but the way back is itself a hundred-step sequence you may never repeat. Go-Explore therefore **archives promising states and returns to them directly** (by replaying the action sequence, or by resetting the simulator) before exploring onward. The insight it makes explicit — **"detachment" and "derailment," the twin failures of forgetting where the frontier was and being unable to get back to it** — is the most useful part of the paper even if you never use the algorithm.

> **Where this came from.** **Montezuma's Revenge** is an Atari 2600 platform game from 1984, and it became the field's standard measure of hard exploration for an unglamorous reason: **the original DQN scored zero on it.** Not "poorly" — zero, the score of an agent that never once found a reward. The game requires a long, precise sequence (climb down, get the key, cross the room, unlock the door) before any points appear at all. Go-Explore, developed by a team at **Uber AI Labs** including Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth Stanley and Jeff Clune, was the first method to comprehensively beat it, with results published in *Nature* in 2021 — roughly six years after DQN's zero.

---

## 26.9 Reward design

▸ **Reward shaping theorem (Ng, Harada & Russell):** adding a potential-based shaping term
$$F(s,a,s') = \gamma\Phi(s')-\Phi(s)$$
leaves the optimal policy **unchanged**, for any function $\Phi$. Any non-potential-based shaping can change the optimum, usually badly.

This is the one theoretically safe way to add intermediate rewards, and it's a great thing to know: it says you may express "these states are promising" via a potential, but you may not reward *behaviours* without risking that the agent optimizes the behaviour instead of the goal.

**Reward hacking** — the agent maximizes the specified reward while violating its intent — is pervasive: boat-racing agents that circle collecting powerups instead of finishing, robots that exploit simulator physics, and (Ch. 16) language models that learn verbosity because raters prefer long answers. **Reward specification, not optimization, is usually the binding constraint on RL projects.**

#### Unpacking potential-based shaping

The problem being solved: sparse rewards make learning nearly impossible (§26.8), so you would like to hand the agent some hints along the way. The danger: **hints change the objective, and the agent optimizes the objective you actually wrote.**

$$F(s,a,s') = \gamma\Phi(s')-\Phi(s)$$

Read aloud: *"F of s, a, s-prime equals gamma times capital-phi of s-prime, minus capital-phi of s."* $\Phi$ (**capital phi**) is a **potential function** — any function you invent that scores "how promising does this state look?" You are free to choose it however you like; the theorem holds for *any* $\Phi$.

▸ **The rule in words: you may only reward the agent for the *change* in your promise-score, never for the score itself, and never for the action taken.** Reward the improvement, not the position, and not the behaviour.

**A concrete choice.** In chess, let $\Phi(s)$ be your material advantage in pawns. Capture a knight and you move from $\Phi = 0$ to $\Phi = 3$, so (with $\gamma\approx1$) the shaping reward is $F \approx +3$. Lose a bishop and $F \approx -3$. The agent gets immediate feedback about material, instead of waiting sixty moves for a win or a loss.

**Why the optimal policy cannot change — the telescoping argument.** Add up the shaping rewards over a whole trajectory, discounted as usual:

$$\sum_{t\ge0}\gamma^t F(s_t,a_t,s_{t+1}) = \sum_{t\ge0}\Big[\gamma^{t+1}\Phi(s_{t+1}) - \gamma^{t}\Phi(s_t)\Big] = -\Phi(s_0)$$

**Every interior term cancels with its neighbour.** The $`+\gamma^{1}\Phi(s_1)`$ from $t=0$ is destroyed by the $`-\gamma^{1}\Phi(s_1)`$ from $t=1$, and so on down the line. What survives is $`-\Phi(s_0)`$, which depends only on where you *started* — and every policy starts in the same place.

▸ **So potential-based shaping adds the same constant to every policy's total return.** Adding a constant to every score cannot change which score is largest. The rankings are untouched, so $`\pi^*`$ is untouched, while the *learning signal* is transformed from one number at the end into a dense stream. **That is a  free lunch, and it is nearly the only one in this chapter.**

> **Analogy.** Give every hiker on a mountain £10 for each metre of altitude they gain and charge them £10 for each metre they lose. Whichever route they take, a hiker who ends at the summit has been paid exactly the same total — the height of the summit minus the height of the car park. **The payments make progress legible without making any route more profitable than another.** Now instead pay £10 for every metre *walked uphill*, with no charge for coming down, and you have invented a machine for making people walk up and down the same slope all day. That second scheme is non-potential-based shaping, and it is what goes wrong.

**Why any non-potential shaping is dangerous.** Because the terms no longer telescope, the total added reward depends on the *path*, so different policies receive different bonuses — and the agent will happily find the policy that maximizes the bonus rather than the goal.

> **Where this came from.** The theorem is **Andrew Ng, Daishi Harada and Stuart Russell's**, from a 1999 paper titled *Policy Invariance Under Reward Transformations*. It was motivated by a specific and well-known failure: a simulated bicycle agent, rewarded for making progress toward a goal, learned to **ride in small circles near the starting point**, collecting the progress reward over and over without ever approaching the destination. The theorem is the precise statement of what that reward function did wrong and what class of hints is safe.

> **The story behind reward hacking as a research topic.** The boat-racing example is real and worth picturing: in 2016 OpenAI published results from *CoastRunners*, a boat-racing game where points are awarded for hitting targets along the course. Their agent discovered a lagoon containing three regenerating targets, and learned to **drive in circles, repeatedly catching fire, crashing into other boats, and never finishing the race** — while scoring roughly 20% higher than human players. The agent was not broken. It found the highest-scoring policy, exactly as designed. **The lesson researchers took from it is that a reward function is not a description of what you want; it is a specification that will be read adversarially by a very persistent optimizer.**

---

## Did you know?

- **"Reinforcement" is a term of art borrowed intact from animal psychology.** It comes out of the tradition of Pavlov's conditioning work and Edward Thorndike's **Law of Effect** — the observation, from experiments with cats in puzzle boxes in the 1890s, that behaviours followed by satisfying consequences become more likely. RL researchers are using a hundred-and-twenty-year-old word in nearly its original sense.

- **Dynamic programming is neither dynamic nor programming.** Richard Bellman chose the name at RAND in the early 1950s, and in his autobiography he attributed the choice partly to institutional politics — the need for a title that would not read as mathematical *research* to a defence establishment hostile to it. "Programming" then meant scheduling, as in linear programming. **Accounts differ on how literally to take the story**, since the dates do not line up cleanly, but the name's uninformativeness is beyond dispute.

- **Bellman also coined "the curse of dimensionality."** It appears in his 1957 book *Dynamic Programming*, describing exactly the $\lvert\mathcal{S}\rvert^2\lvert\mathcal{A}\rvert$ blow-up that makes tabular methods useless.

- **The first application of Markov chains was to Russian poetry.** To show that his dependent sequences described something real, Andrey Markov classified the first 20,000 letters of Pushkin's *Eugene Onegin* as vowels or consonants **by hand** and tabulated the transitions. He built the first statistical language model around 1913, with no computer and no intention of building one.

- **Dopamine neurons compute a temporal-difference error.** A 1997 *Science* paper by Wolfram Schultz, Peter Dayan and Read Montague showed that midbrain dopamine neurons fire at unexpected rewards, shift their firing to the predictive cue once the cue is learned, and dip **below baseline** at the exact moment a predicted reward fails to arrive. That is $`\delta_t`$, positive and negative, in living tissue — an algorithm designed for backgammon turning out to describe neural hardware.

- **A backgammon program changed human opening theory.** Gerald Tesauro's TD-Gammon (IBM, early 1990s) learned almost entirely from self-play and reached a level competitive with the world's best players. Human experts subsequently **revised standard opening moves** to match the program's preferences — machine learning teaching its own domain something new, in 1992.

- **The phrase "machine learning" was coined in a paper about checkers.** Arthur Samuel's 1959 *Some Studies in Machine Learning Using the Game of Checkers* named the field — and his program was already doing temporal-difference learning, twenty-five years before anyone named that.

- **Thompson sampling was published in 1933 and then ignored for about seventy-five years.** William Thompson's *Biometrika* paper was about assigning patients to medical treatments, not slot machines. It resurfaced around 2010 when empirical studies found the 1933 method outperforming decades of carefully engineered alternatives on real advertising data.

- **"Bandit" is slot-machine slang.** A slot machine was a *one-armed bandit* — one lever, and it robs you. Peter Whittle later joked that the problem was so consuming during the Second World War that someone proposed dropping it over Germany as a sabotage measure. The joke is well known; it is, emphatically, a joke.

- **The $Q$ in Q-learning stands for nothing in particular.** Chris Watkins used $Q$ for the *quality* of a state–action pair in his 1989 Cambridge thesis, and the letter stuck. SARSA, by contrast, is named after its own argument list — State, Action, Reward, State, Action — and the acronym was suggested by Richard Sutton to replace the original and thoroughly forgettable "modified connectionist Q-learning."

- **Every convergence proof in tabular RL is a corollary of a 1922 thesis about integral equations.** Stefan Banach's fixed-point theorem was never intended to have anything to do with decisions, agents, or rewards.

- **Baird's counterexample is seven states wide and has never been solved.** Leemon Baird showed in 1995 that off-policy TD with linear function approximation can diverge to infinity on a tiny problem. Thirty years and enormous amounts of compute later, every deep RL method still *manages* the deadly triad rather than eliminating it.

- **The original DQN scored exactly zero on Montezuma's Revenge.** Not "poorly" — zero, across all its training, because random exploration essentially never reaches the game's first reward. It took roughly six more years and a fundamentally different approach (Go-Explore) to comprehensively beat it.

---

## Check for Understanding

**All of RL rests on the Bellman equation — value now equals reward plus discounted value next — which converges because the Bellman operator is a $\gamma$-contraction in the sup norm; temporal-difference learning approximates it from samples by bootstrapping, trading bias for a large variance reduction; and the combination of bootstrapping, off-policy data, and function approximation is exactly the deadly triad that makes deep RL unstable.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why is reinforcement learning harder than supervised learning?** Give the answer in terms of *who generates the data*, not in terms of algorithms.
2. **What does the Markov property actually claim**, and why is it a statement about your state representation rather than about the world? Use the tennis-ball photograph.
3. **Why does a discount factor exist at all?** Give all three reasons, and say what "effective horizon" means in steps rather than in $\gamma$.
4. **What is the Bellman equation, in one sentence with no symbols?** Then say why the recursion is only legitimate if the Markov property holds.
5. **Why does value iteration converge?** Explain "contraction" using the map-on-the-floor picture, and say why raising $\gamma$ from 0.99 to 0.999 costs you ten times the iterations.
6. **What is a TD error?** Explain it with the driving-home example, and say why bootstrapping cuts the noise of an update roughly tenfold on a hundred-step episode.
7. **In the eight-episode A/B example, why do Monte Carlo and TD give different values for state $A$** — and which one is right?
8. **What is the difference between SARSA and Q-learning, in terms of the cliff?** Say which one you would deploy on a real robot and why.
9. **Why is the maximum of several noisy estimates too high**, even when every individual estimate is unbiased? Connect it to the winner's curse.
10. **Why does the variance of an importance-sampling ratio explode with the horizon?** Say the number: mean 1, standard deviation in the hundreds of thousands over fifty steps.
11. **What are the three parts of the deadly triad**, and why is each pair of them safe while all three are not?
12. **What does "optimism in the face of uncertainty" mean**, and why does the UCB bonus shrink like $1/\sqrt{N}$ rather than $1/N$?
13. **Why can you safely reward the *change* in a promise-score but not the score itself?** Use the hikers-and-altitude picture, and explain what telescoping means without writing a sum.

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 27 — Deep Reinforcement Learning](27-deep-reinforcement-learning.md)
