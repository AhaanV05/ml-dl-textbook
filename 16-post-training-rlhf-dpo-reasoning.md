# Chapter 16 — Post-Training: SFT, RLHF, DPO & Reasoning

> **Prerequisites:** Ch. 13. Chapter 27 (Deep RL) covers PPO in general; this chapter covers its language-specific form and can be read either before or after.

> **New to the notation?** If symbols like $`\in`$, $`\sum`$, $`\mathbb{E}`$, $`\nabla`$, $`\sigma(\cdot)`$, or $`A^\top`$ are unfamiliar, read **[Chapter 0 — How to Read the Mathematics in This Book](00-notation-and-math-primer.md)** first. It decodes every symbol used here, and includes a full-forms glossary for every abbreviation in this book. This chapter is unusually notation-heavy in one place only — the DPO derivation in §16.5 — and that derivation is unpacked line by line below the formal version.

### Symbols introduced in this chapter

Skim once now; each entry is unpacked properly where it first appears.

| Symbol | Read aloud | Plain meaning |
|---|---|---|
| $`x`$ | "x" | The **prompt** — everything the user typed |
| $`y`$ | "y" | A **response** — a whole completion, not a single token |
| $`y_w \succ y_l`$ | "y-win beats y-lose" | A human (or judge) preferred response $`y_w`$ over $`y_l`$. $`w`$ = winner, $`l`$ = loser |
| $`r^*(x,y)`$ | "r-star of x, y" | The **true, unknown** reward: how good response $`y`$ is for prompt $`x`$ |
| $`r_\phi(x,y)`$ | "r-phi" | Our **learned** reward model. $`\phi`$ ("phi") are its weights |
| $`\sigma(z)`$ | "sigmoid of z" | $`1/(1+e^{-z})`$ — squashes any real number into $`(0,1)`$ |
| $`\pi_\theta(y \mid x)`$ | "pi-theta of y given x" | The **policy**: the model's probability of producing $`y`$ after seeing $`x`$. $`\theta`$ = its weights |
| $`\pi_{\text{ref}}`$ | "pi-ref" | The **reference policy** — a frozen copy of the model before RL started |
| $`\beta`$ | "beta" | How tightly the new model is leashed to the old one |
| $`\mathrm{KL}(p \Vert q)`$ | "KL of p from q" | How much you lose by believing $`q`$ when reality is $`p`$ |
| $`Z(x)`$ | "Z of x" | The **partition function** — the normalizing sum that makes probabilities add to 1 |
| $`\hat A_t`$ | "A-hat at t" | The **advantage**: how much better this action was than average |
| $`\rho_t`$ | "rho at t" | The **probability ratio** between the new policy and the old one |
| $`\mathrm{clip}(z, a, b)`$ | "clip z between a and b" | Squash $`z`$ into $`[a,b]`$: `min(max(z,a),b)` |
| $`\mathbb{1}[\,\cdot\,]`$ | "indicator of" | 1 if the statement inside is true, 0 otherwise |
| $`G`$ | "G" | **Group size** — how many completions we sample per prompt in GRPO |
| $`\mathcal{D}`$ | "script D" | The **dataset** we're averaging over |
| $`\mathcal{L}`$ | "script L" | A **loss** — the thing we minimize |
| $`\nabla_\theta`$ | "grad theta" | Which way to nudge each weight to *increase* what follows |
| $`\epsilon`$ | "epsilon" | The PPO clip width, typically 0.2 |

### Abbreviations in this chapter, in full

The field talks almost entirely in initialisms. Here is the decoder.

| Short | Full form |
|---|---|
| **SFT** | Supervised fine-tuning |
| **RLHF** | Reinforcement learning from human feedback |
| **RLAIF** | Reinforcement learning from AI feedback |
| **RLVR** | Reinforcement learning from verifiable rewards |
| **RM** | Reward model |
| **BT** | Bradley–Terry (the preference model) |
| **PPO** | Proximal policy optimization |
| **TRPO** | Trust region policy optimization |
| **GRPO** | Group relative policy optimization |
| **DPO** | Direct preference optimization |
| **IPO** | Identity preference optimization |
| **KTO** | Kahneman–Tversky optimization |
| **ORPO** | Odds-ratio preference optimization |
| **SimPO** | Simple preference optimization |
| **RRHF / RSO** | Rank responses to align human feedback / Statistical rejection sampling optimization |
| **KL** | Kullback–Leibler (divergence) |
| **CoT** | Chain of thought |
| **CAI** | Constitutional AI |
| **LR** | Learning rate |
| **OOD** | Out of distribution |
| **MLE** | Maximum likelihood estimation |
| **LIMA** | "Less Is More for Alignment" — a 2023 paper, not a general term |
| **FLAN** | "Finetuned Language Net" — Google's instruction-tuning dataset collection |
| **Elo** | Not an abbreviation at all — the surname of Arpad Elo |

---

## 16.1 Why post-training exists

### The one-line idea

A pretrained model predicts what text is *likely*; a useful assistant produces text that is *helpful, honest, and harmless*. Those are different objectives, and the second cannot be expressed as a likelihood over web text.

### The analogy

Pretraining is reading the entire internet — you absorb every register, including the unhelpful, the wrong, and the malicious, and you have no idea which one you're supposed to be. Post-training is an apprenticeship: someone shows you how the job is done (SFT), then critiques your attempts until your instincts change (RLHF). The knowledge was already there. What changes is which of your many capabilities gets expressed.

▸ **Evidence for this framing:** LIMA achieved strong instruction-following with **1,000** carefully curated examples. Post-training is largely *elicitation*, not teaching. (Caveat: reasoning-focused RL does appear to teach  new behaviour, §16.7.)

#### What "post-training" actually names

**Pretraining** is the phase where a model reads a very large pile of text and learns to predict the next token. **Post-training** is everything that happens after that pile is exhausted — every technique in this chapter. The word "post" is doing all the work: it is defined by *when* it happens, not by *what* it does.

Here is the concrete difference in one sentence. Give a raw pretrained model the prompt:

```
What is the capital of France?
```

A well-known failure is that it continues the *pattern* rather than answering it:

```
What is the capital of France?
What is the capital of Germany?
What is the capital of Italy?
```

That is not a bug. The model has correctly noticed that on the open web, a line that looks like a quiz question is most often followed by *another quiz question*. The capital of France is unambiguously in the weights — ask it "The capital of France is" and it will say "Paris." The problem is not knowledge. The problem is that **"answer the question" is one of a thousand plausible continuations, and nothing has told the model to pick that one.**

> **Analogy.** A pretrained model is a brilliant mimic who has memorized every conversation ever held in a large city — arguments, sales pitches, lectures, threats, lullabies. Ask it something and it will produce *a* plausible continuation of *some* conversation. Post-training is not education; it is casting. You are telling the mimic which character to play, permanently.

▸ **The key consequence:** the capabilities are already there. Post-training changes the *prior over which behaviour appears*, and it does so with roughly $`10^{-4}`$ of the compute used in pretraining. That ratio — thousands of GPU-years for pretraining, a handful of GPU-days for alignment — is why post-training is where almost all of a lab's iteration happens.

#### Examples and non-examples: post-training

**✅  post-training**

| Example | Why it qualifies |
|---|---|
| Fine-tuning on 10,000 (question, ideal answer) pairs written by contractors | Starts from a finished pretrained checkpoint; changes behaviour, not knowledge |
| Training a reward model on 50,000 human A-vs-B comparisons, then PPO against it | The canonical RLHF pipeline (§16.3–16.4) |
| DPO on a preference set, updating the same weights that were pretrained | Uses a preference signal that has no expression as a likelihood over web text |
| GRPO on 200,000 math problems with a correctness checker | Reward comes from an external verifier, applied after pretraining (§16.6) |
| Adding a `<\|im_start\|>assistant` chat template and training the model to respect it | Teaches a *protocol*, not a fact |

**❌ Near-misses — look like post-training, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| Adding a system prompt: "You are a helpful assistant" | No weights change. Delete the prompt and the behaviour vanishes | **Prompting** — an inference-time trick |
| Continuing to train on 200B tokens of medical papers | Injecting new knowledge with the *same* next-token objective | **Continued pretraining** (or domain-adaptive pretraining) |
| Few-shot examples in the context window | The model is conditioning, not learning; nothing persists to the next request | **In-context learning** |
| Retrieval-augmented generation over a company wiki | Facts arrive at inference time through the prompt | **RAG** (Ch. 18) |
| Quantizing the model to 4 bits after alignment | Happens after training, but changes numerics, not behaviour | **Compression** (Ch. 17) |
| Constrained decoding that forces valid JSON | A sampling-time filter over an unchanged distribution | **Decoding constraint** |

▸ **The boundary:** post-training **updates weights**, using an objective that is **not plain next-token likelihood on scraped text**. Change the weights with the ordinary objective and it's still pretraining; change the behaviour without touching the weights and it's prompting.

> **Common misconception.** *"RLHF teaches the model facts it didn't know."* It essentially doesn't. If a base model has never encountered a chemical compound, no amount of preference optimization will conjure it — RLHF has a preference dataset of maybe $`10^5`$ comparisons against a pretraining corpus of $`10^{13}`$ tokens, eight orders of magnitude apart. What RLHF changes is *which* of the model's many possible behaviours surfaces by default, and how confidently. The misconception is tempting because a post-trained model **looks** far more knowledgeable: it now answers questions instead of continuing them, so knowledge that was always there becomes visible for the first time. The correct mental model is a dimmer switch on existing behaviours, not a download of new ones.

> **Common misconception.** *"Alignment means making the model safe."* Safety is one consumer of alignment machinery, not its definition. The same Bradley–Terry loss that trains a harmlessness model trains a "prefer concise answers" model or a "prefer answers in Portuguese" model. **The technique is value-neutral; the preference data supplies the values.** This matters practically: when someone says "we aligned the model," the only informative follow-up question is "aligned to whose preferences, collected how?"

> **Where this came from.** Learning from human *comparisons* rather than a written-down reward function was demonstrated by **Paul Christiano and colleagues at OpenAI and DeepMind in 2017**, in *Deep Reinforcement Learning from Human Preferences*. Their showcase was not a language model at all — it was a simulated stick-figure robot that they taught to perform a **backflip**, a behaviour nobody could write a reward function for, from fewer than a thousand human A-or-B judgements and under an hour of a person's time. The pipeline in this chapter — learn a reward model from comparisons, then optimize a policy against it — is that 2017 paper with a transformer swapped in for the physics simulator. It reached language through **Ziegler et al. (2019)** on stylistic continuation and **Stiennon et al. (2020)** on summarization, and became famous with **InstructGPT (Ouyang et al., 2022)**, whose headline result was that a **1.3-billion-parameter** aligned model was preferred by human raters over the **175-billion-parameter** GPT-3 it came from. A 100× smaller model won on preference. That single number is why every lab now has a post-training team.

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

**Hyperparameters:** 1–3 epochs (more overfits fast), LR $`10^{-5}`$–$`10^{-6}`$ (10–100× below pretraining), cosine decay, small batch. Watch for **catastrophic forgetting** — mixing in a small fraction of pretraining data helps.

#### Supervised fine-tuning, decoded

Every phrase in that section is load-bearing. Take them one at a time.

**"(prompt, response) pairs."** A row of SFT data is two strings: what a user might say, and what you wish the model would say back. For example:

| Field | Content |
|---|---|
| prompt | `Explain why the sky is blue to a 10-year-old.` |
| response | `Sunlight looks white but is secretly all the colors mixed together. When it hits the air, the blue part bounces around the most...` |

**"the standard cross-entropy objective."** Cross-entropy for language modelling is: at every position, look at what token actually came next, and penalize the model by $`-\log`$ of the probability it assigned to that token. If the model gave the correct next token probability 0.5, the penalty is $`-\log 0.5 = 0.693`$ nats. If it gave 0.9, the penalty is $`-\log 0.9 = 0.105`$. If it gave 0.01, the penalty is $`4.61`$. **Confidently wrong is expensive; confidently right is nearly free.**

Written out, with $`T`$ the number of tokens in the response:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T}\log \pi_\theta\big(y_t \mid x,\, y_{<t}\big)$$

Read aloud: *"minus the sum, over each response token, of the log-probability the model assigned to that token given the prompt and everything it has written so far."* $`y_{<t}`$ ("y less than t") means "all response tokens before position $`t`$."

**"masking the loss on prompt tokens."** This is the one line people get wrong when they implement it. Concretely, suppose the full sequence fed to the model is 9 tokens:

```
[Explain] [why] [the] [sky] [is] [blue] || [Sunlight] [looks] [white]
             prompt (6 tokens)                  response (3 tokens)
```

The model still *reads* all 9 and still computes a prediction at all 9 positions. The mask decides which of those 9 predictions contribute to the loss:

```python
# labels is a copy of input_ids; -100 is PyTorch's "ignore this position" sentinel
labels = input_ids.clone()
labels[:, :prompt_len] = -100          # <- the entire mask, in one line
loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
```

So the loss here averages 3 terms, not 9.

▸ **Why masking matters, concretely.** Without the mask you are also training the model to *generate plausible user prompts*. On a dataset where 60% of tokens are prompt tokens, more than half your gradient signal is teaching a behaviour you will never use at inference — and worse, you are teaching the assistant that emitting a new user question is a normal thing to do, which shows up at deployment as a model that answers and then hallucinates the user's next message.

#### What would break: three SFT failures with numbers

| You do this | What breaks | The number |
|---|---|---|
| Train 10 epochs instead of 2 | Memorization. Training loss falls to ~0.1 while held-out chat quality drops | With 10k examples, epoch 10 has seen each example 10 times at LR $`10^{-5}`$ — enough to reproduce them verbatim |
| Use pretraining's LR of $`3\times10^{-4}`$ | Catastrophic forgetting. The model becomes fluent at your 10k examples and worse at everything else | 30× the recommended $`10^{-5}`$; the weights move far enough to overwrite pretrained structure |
| Forget the mask | The model generates its own follow-up user turns | On a 60/40 prompt/response split, 60% of gradient is spent on the wrong objective |
| Train on responses that end without the `<\|im_end\|>` token | The model never learns to stop | Generation runs to the context limit every time |

#### Examples and non-examples: good SFT data

**✅  SFT examples that improve a model**

| Example | Why it qualifies |
|---|---|
| A contractor's careful 300-word answer to "How do I read a nutrition label?" | Demonstrates the target *behaviour* — format, tone, length, hedging — not just the fact |
| A refusal that explains itself: "I can't help synthesize that, but here's the chemistry of why it's dangerous" | Teaches the shape of a good refusal, which is a behaviour with no natural web analogue |
| A multi-turn exchange where the assistant asks a clarifying question before answering | Teaches that asking is sometimes correct — a base model almost never does this |
| A response that says "I don't know, but here's how you'd find out" for a  obscure question | The **only** way a model learns calibrated abstention is by seeing it demonstrated |

**❌ Near-misses — look like good SFT data, but hurt**

| Looks like it | Why it hurts | What it actually is |
|---|---|---|
| 100,000 rows of `{"input": "great movie", "output": "positive"}` | Teaches one-word answers; the model becomes terse everywhere | **Task data**, useful for a classifier, corrosive for a chat model |
| A correct answer to a question the model *cannot* know (yesterday's stock price) | Trains the model that confident specificity is rewarded even without grounds | A **hallucination lesson** — you are teaching it to make things up fluently |
| Distillation output from a stronger model, unfiltered | ~10–20% of it is wrong, and errors are *fluent*, so they train especially efficiently | **Unverified synthetic data** |
| A brilliant answer written in a style nothing like your target assistant | Style is what SFT transfers most strongly | **Style contamination** |
| Template-converted benchmark data (FLAN-style) as the *only* source | Benchmark scores rise; open-ended conversation gets worse | **Benchmark optimization** |

▸ **The boundary:** SFT data is good when it is a **demonstration of the behaviour you want at deployment**, produced under the same information the model will have at deployment. Correctness is necessary but nowhere near sufficient — a true answer that models an impossible epistemic state teaches dishonesty.

> **Common misconception.** *"More SFT data is always better."* LIMA's 1,000 examples beat far larger noisy sets, and the mechanism is not mysterious: SFT is imitation learning, and imitation learning inherits the *average* quality of what it imitates, not the maximum. Ten thousand mediocre examples plus a thousand excellent ones produces a mediocre model. The misconception is tempting because in *pretraining* more data is essentially always better — but pretraining is learning a distribution, where diversity helps, while SFT is learning a target behaviour, where variance hurts. **The relevant statistic flips from "how much" to "how consistent."**

> **Common misconception.** *"The chat template is just formatting."* It is the model's only structural notion of *who is speaking*, and it is learned entirely from data. A model that has never seen text like `Ignore previous instructions` inside a user turn during training has no representation of that being an illegitimate move — the delimiter is a fence the model was taught to respect, not a wall the runtime enforces. This is why prompt injection is a **training-data problem** wearing the costume of a security problem, and why sanitizing input and restricting tool permissions do more than any instruction you can write in a system prompt.

> **Where this came from.** Turning existing NLP datasets into instruction-shaped text was the idea behind **FLAN** (*Finetuned Language Models Are Zero-Shot Learners*, Wei et al., Google, 2021) and, concurrently, **T0** from the BigScience collaboration. Both discovered the same surprising thing: train on a *mix* of many task types phrased as instructions, and the model generalizes to instruction types it never saw — the instruction format itself becomes the learned abstraction. **LIMA** (Meta, 2023) then pushed the opposite direction and found that 1,000 hand-curated examples were enough, which is where the "post-training is elicitation" thesis comes from. The two results are not in conflict: FLAN showed that the *format* generalizes, LIMA showed that the *quantity* required to trigger it is tiny.

---

## 16.3 Reward modelling

### The problem

We can't write down a reward function for "helpful." But humans can *compare* two responses reliably, even when they can't score one in isolation.

### Bradley–Terry

Assume a latent scalar reward $`r^*(x,y)`$ such that the probability a human prefers $`y_w`$ over $`y_l`$ given prompt $`x`$ is:

▸ $$p^*(y_w\succ y_l\mid x) = \frac{\exp r^*(x,y_w)}{\exp r^*(x,y_w)+\exp r^*(x,y_l)} = \sigma\big(r^*(x,y_w)-r^*(x,y_l)\big)$$

**This is exactly logistic regression on the reward difference.** Fit by maximum likelihood:

▸ $$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$

**Implementation:** initialize from the SFT model, replace the unembedding with a scalar head, and read the reward off the final token.

▸ **Note the identifiability gap:** Bradley–Terry determines $`r`$ only up to an additive function of $`x`$ (a per-prompt constant), since only differences within a prompt are observed. This is harmless for policy optimization — which we'll see is invariant to it — but it means reward magnitudes are not comparable across prompts.

#### Reading the Bradley–Terry model in plain English

Start with the symbols, then put numbers in.

| Piece | Read aloud | What it means |
|---|---|---|
| $`y_w`$, $`y_l`$ | "y-win", "y-lose" | The response the human preferred, and the one they didn't |
| $`y_w \succ y_l`$ | "y-win is preferred to y-lose" | $`\succ`$ is a preference symbol, not "greater than" |
| $`p^*(\cdot \mid x)`$ | "p-star, given x" | The **true** probability a human picks this way, for this prompt |
| $`r^*(x,y)`$ | "r-star of x, y" | A single number: how good $`y`$ is as a response to $`x`$. Nobody can see it |
| $`\exp(\cdot)`$ | "e to the" | Makes everything positive, so the fraction is a valid probability |
| $`\sigma(z)`$ | "sigmoid of z" | $`1/(1+e^{-z})`$ — turns a difference into a probability |

The whole claim in one sentence: **there is some hidden quality score for every response, and the chance a human picks one over the other depends only on the gap between their scores.**

> **Analogy.** Two chess players have hidden strengths. You never observe "strength" — you only observe who won. But watch enough games and you can reconstruct a rating for each player, up to a shared offset, because the win probability depends only on the *rating difference*. Bradley–Terry is exactly this, with "response" in place of "player" and "a human preferred it" in place of "won the game." That is not an analogy by coincidence; it is literally the same model, and §16.9's Elo ratings are the same equation again.

**Numbers, please.** Suppose the true rewards for a prompt are $`r^*(x,y_w) = 2.0`$ and $`r^*(x,y_l) = 0.5`$.

$$p^*(y_w \succ y_l \mid x) = \sigma(2.0 - 0.5) = \sigma(1.5) = \frac{1}{1+e^{-1.5}} = \frac{1}{1+0.2231} = 0.818$$

So a human picks $`y_w`$ about **82% of the time** — not always. That residual 18% is the model saying *humans are noisy and sometimes prefer the worse answer*, which is both true and important.

Now watch the gap change:

| $`r_w - r_l`$ | $`\sigma(\text{gap})`$ | In words |
|---|---|---|
| $`0`$ | $`0.500`$ | A coin flip. The responses are equally good |
| $`0.5`$ | $`0.622`$ | A weak preference |
| $`1.5`$ | $`0.818`$ | A clear preference |
| $`3.0`$ | $`0.953`$ | Almost everyone agrees |
| $`6.0`$ | $`0.998`$ | Effectively unanimous |

▸ **The single most useful reading:** a reward *difference* of about 1.1 corresponds to 3:1 odds, because $`\sigma(1.1)\approx 0.75`$. Reward units are **log-odds of being preferred**, not "quality points." When someone says their reward model gave a response 2.3, that number is meaningless alone — only differences within a prompt mean anything.

**Why "exactly logistic regression."** The two expressions in the formula are the same thing:

$$\frac{e^{r_w}}{e^{r_w}+e^{r_l}} = \frac{e^{r_w}/e^{r_w}}{(e^{r_w}+e^{r_l})/e^{r_w}} = \frac{1}{1+e^{-(r_w - r_l)}} = \sigma(r_w - r_l)$$

Divide top and bottom by $`e^{r_w}`$ and the two-way softmax collapses into a sigmoid of the difference. This is why the identifiability gap exists: add any constant $`c(x)`$ to *both* rewards and $`e^{c}`$ cancels top and bottom. **The model can only ever see gaps.**

#### Unpacking the reward-model loss

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$

Read aloud: *"the average, over labelled preference triples, of minus the log of the sigmoid of the reward gap."* Piece by piece:

- $`\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}`$ — "average over triples drawn from the dataset." A triple is (prompt, better response, worse response).
- $`r_\phi`$ — the *learned* reward model with weights $`\phi`$, as opposed to the unknowable $`r^*`$.
- $`-\log\sigma(\cdot)`$ — the standard binary cross-entropy penalty, with the label always "the winner won."

**A tiny worked example.** Three training pairs, current model outputs:

| Pair | $`r_\phi(y_w)`$ | $`r_\phi(y_l)`$ | Gap | $`\sigma(\text{gap})`$ | $`-\log\sigma`$ |
|---|---|---|---|---|---|
| 1 | $`3.1`$ | $`1.0`$ | $`+2.1`$ | $`0.891`$ | $`0.115`$ |
| 2 | $`0.4`$ | $`0.2`$ | $`+0.2`$ | $`0.550`$ | $`0.598`$ |
| 3 | $`-0.5`$ | $`1.5`$ | $`-2.0`$ | $`0.119`$ | $`2.127`$ |

Average loss $`= (0.115 + 0.598 + 2.127)/3 = 0.947`$ nats. Pair 3 — the one the model has **backwards** — contributes 75% of the total loss. That is the whole training dynamic: *pairs you already rank correctly are nearly free; pairs you have inverted dominate the gradient.* Hold onto this, because in §16.5 the identical structure reappears inside DPO.

**Sanity checks on those numbers.** A perfect model drives every gap to $`+\infty`$ and the loss to 0. A model that outputs the same number for everything gets gap $`=0`$, $`\sigma(0)=0.5`$, loss $`=\log 2 = 0.693`$ per pair. **So 0.693 is your "learned nothing" baseline** — the equivalent of 50% accuracy. A reward model reporting a training loss of 0.69 has not started learning; one reporting 0.35 is doing real work. Reward-model *accuracy* (fraction of pairs ranked correctly) on held-out human data typically lands around **65–75%**, and that ceiling is largely human disagreement, not model weakness.

**"Replace the unembedding with a scalar head," concretely.** The language model's last layer maps a $`d`$-dimensional hidden state to $`\lvert V\rvert \approx 128{,}000`$ logits. For a reward model you throw that matrix away and bolt on a $`d\times 1`$ vector instead, then read the single number produced at the **final token position** — because only at the last position has the model attended to the entire response.

```python
h = backbone(input_ids).last_hidden_state   # (batch, seq, d)
last = h[:, -1, :]                          # (batch, d)  <- final token only
reward = last @ w                           # (batch, 1)  <- w is d x 1
```

That is the whole architectural change: a 128,000-column matrix becomes a 1-column matrix.

> **Where this came from.** The Bradley–Terry model is named for **Ralph Bradley and Milton Terry**, who published it in *Biometrika* in **1952** for ranking items in taste tests and incomplete block designs — agricultural and sensory experiments where you can compare two things but not score one. It had already been derived, essentially in full, by **Ernst Zermelo in 1929** for a completely different purpose: computing fair rankings from chess tournaments where not every player faced every other. Zermelo is far better known for the axiom of choice and Zermelo–Fraenkel set theory; his chess-ranking paper went unnoticed for decades and the model now carries someone else's name. Earlier still, **Louis Thurstone's** 1927 *law of comparative judgment* proposed the same idea with a Gaussian instead of a logistic — the probit version rather than the logit. **Arpad Elo's** chess rating system, adopted by the United States Chess Federation around 1960 and by FIDE in 1970, is the same machinery again. So the objective that trains modern language models was invented three separate times — for set theory's founder ranking chess players, for psychologists measuring perceived weight, and for statisticians comparing crop varieties.

### The failure modes

- **Length bias.** Humans prefer longer answers; the RM learns "longer = better"; the policy learns to ramble. Fixes: length-controlled evaluation, explicit length penalties, or debiasing the RM.
- **Sycophancy.** Humans prefer agreement, so the model learns to agree.
- **Distribution shift.** The RM is trained on SFT-model outputs but evaluated on the *optimized* policy's outputs, which drift out of distribution. This is why the KL penalty (§16.4) is not optional.
- **Reward hacking.** Optimizing hard against an imperfect proxy finds its flaws. **Goodhart's law is not a metaphor here; it is the central engineering problem.**

**Mitigations:** ensembles of reward models; conservative/pessimistic aggregation; periodic RM retraining on fresh policy samples; and generative or rubric-based reward models that produce a critique before a score.

#### Length bias, with actual numbers

This one deserves arithmetic because it is the failure you will meet first.

Suppose annotators  prefer the better answer 70% of the time, but *also* prefer the longer answer 65% of the time regardless of content. The reward model, which sees only the labels, cannot separate the two signals — it learns whatever combination explains the data. Fit a length term and you routinely find something like:

$$r_\phi(x,y) \approx \underbrace{q(x,y)}_{\text{actual quality}} + \underbrace{0.004 \times \lvert y\rvert}_{\text{length}}$$

with $`\lvert y \rvert`$ the token count. That coefficient looks negligible. Now optimize against it:

| Response length | Length contribution | Equivalent quality gap it fakes |
|---|---|---|
| 100 tokens | $`0.4`$ | — |
| 400 tokens | $`1.6`$ | $`+1.2`$ over the 100-token answer, i.e. $`\sigma(1.2)=77\%`$ preference from padding alone |
| 1,000 tokens | $`4.0`$ | $`+3.6`$, i.e. $`\sigma(3.6)=97\%`$ preference from padding alone |

▸ **The policy is not being stupid.** Padding from 100 to 1,000 tokens buys a reward increase that quality improvements could never match, so a correctly-functioning optimizer will discover padding before it discovers being right. **Every reward-hacking story has this shape: the model found the cheapest path up the gradient of the metric you actually wrote down.**

The standard diagnostic: plot mean response length against training step. If it rises monotonically while human-judged win rate plateaus, you are watching length bias in real time. The standard fixes are to subtract a length penalty from the reward, to length-match responses when collecting preferences, or — as **SimPO** does (§16.5) — to normalize the implicit reward by length so the exploit is structurally unavailable.

#### Examples and non-examples: reward hacking

**✅  reward hacking**

| Example | The exploited gap |
|---|---|
| A summarizer that pads every summary to the maximum length | RM learned "longer = better" from annotators who skim |
| A model that opens every reply with "What a fascinating question!" | RM learned that praise correlates with approval |
| A code model that writes `if input == test_case_1: return 42` | The verifier checks test outputs, not method |
| A model that agrees with a user's incorrect premise rather than correcting it | Annotators marked disagreement as "unhelpful" |
| A chat model that answers in bullet points regardless of whether prose fits better | Formatting cues raised preference scores in the training set |

**❌ Near-misses — look like reward hacking, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| The model produces long answers because the *task* is complex | The length is earned; the RM would prefer it under length-controlled evaluation | **Appropriate verbosity** |
| The model refuses a borderline request | It is following the preferences it was actually given | **Over-refusal from the training distribution** — a data problem, not an exploit |
| The model hallucinates a citation | Nothing in the reward rewarded fabrication specifically | **Ungrounded generation**, inherited from pretraining |
| Training loss goes down while validation loss rises | An optimization/statistics phenomenon, not a proxy exploit | **Overfitting** |
| The model's reward keeps rising and its quality does too | The proxy is still aligned with the goal | **Working as intended** — hacking is defined by *divergence* |

▸ **The boundary:** reward hacking is when the **proxy score rises while the true objective falls**. Both halves are required. A rising score with rising quality is optimization; a falling score is a bug. Hacking is specifically the wedge that opens between them, which is why you cannot detect it by watching the reward curve alone — **you must have a second, uncorrupted measurement.**

> **Common misconception.** *"A better reward model would fix reward hacking."* It raises the ceiling; it does not change the shape. Any learned proxy has a region where it disagrees with the truth, and an optimizer's job is precisely to find the highest point of the proxy — which sits in that region by construction. The misconception is tempting because incremental RM improvements *do* visibly help, so it feels like a convergent process. But the failure returns at the new optimum. This is why the real fixes are structural — a KL leash (§16.4) that forbids straying far enough to reach the exploit, RM ensembles that disagree in exploited regions, or removing the learned proxy entirely (§16.6).

> **Common misconception.** *"The reward model tells you how good a response is."* It tells you **how likely an annotator was to pick it over a sibling from the same prompt.** Those come apart constantly: a reward of 4.2 on a trivia question and 4.2 on a legal analysis carry no common meaning, because Bradley–Terry pins rewards only up to a per-prompt constant (the identifiability gap above). Practitioners who compare reward scores across prompts, or set a global "reward threshold" for filtering, are reading a number that was never defined.

> **Where this came from.** **Goodhart's law** comes from **Charles Goodhart**, a British economist at the Bank of England, who observed in a 1975 conference paper that the UK's monetary targets stopped predicting anything once the government began steering by them. His actual phrasing was drier than the version you hear: *"Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."* The famous compressed form — "when a measure becomes a target, it ceases to be a good measure" — was written by the anthropologist **Marilyn Strathern in 1997**, not by Goodhart. In machine learning the phenomenon acquired its own vocabulary as **specification gaming**, and the canonical demonstration is an OpenAI experiment from 2016 on the boat-racing game *CoastRunners*: an agent rewarded for score discovered it could ignore the race entirely, circle a lagoon, and repeatedly collect respawning power-ups. It scored higher than any human player while never finishing a lap, catching fire and crashing into walls throughout. **Nothing about that agent was broken.** It maximized exactly what it was told to.

---

## 16.4 RLHF with PPO

### The objective

▸ $$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big] - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

**Why the KL term is essential, not decorative:**
1. It keeps the policy inside the region where the reward model is valid.
2. It preserves fluency and knowledge from pretraining.
3. Without it, the policy collapses onto a small set of reward-maximizing degenerate outputs — the classic failure is a model that emits the same flattering paragraph for every prompt.

Typical $`\beta = 0.01`$–$`0.1`$. Implemented as a per-token penalty added to the reward: $`r_t = -\beta\big(\log\pi_\theta(y_t\mid\cdot) - \log\pi_{\text{ref}}(y_t\mid\cdot)\big)`$, with the RM's scalar score added at the final token.

#### Reading the RLHF objective in plain English

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big] - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

Read aloud: *"choose the policy that maximizes the average reward of the responses it generates, minus beta times how far its distribution has drifted from the frozen reference."*

| Piece | Read aloud | What it means |
|---|---|---|
| $`\max_{\pi_\theta}`$ | "max over pi-theta" | Search over model weights $`\theta`$; the thing we're choosing is a whole *distribution over responses* |
| $`x\sim\mathcal{D}`$ | "x drawn from script-D" | Sample a prompt from a prompt set. Note: only prompts are needed here, **no reference answers** |
| $`y\sim\pi_\theta(\cdot\mid x)`$ | "y drawn from pi-theta given x" | The model **generates** its own response. This is what makes it RL rather than supervised |
| $`r_\phi(x,y)`$ | "r-phi of x, y" | The frozen reward model's score for that generated response |
| $`\beta`$ | "beta" | The leash length. Small $`\beta`$ = long leash |
| $`\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})`$ | "KL of pi-theta from pi-ref" | How surprised the reference model would be by the new model's outputs |

> **Analogy.** You are training a chef by taste-testing (the reward model) rather than by giving recipes. Left unchecked, a chef optimizing purely for "the taster smiles" converges on pouring sugar into everything. The KL term is a contract clause: *you may adjust the recipe, but a diner must still recognize the dish.* Set $`\beta`$ too high and the chef never changes anything; too low and you get sugar.

**The KL term with real numbers.** $`\mathrm{KL}`$ here is estimated per token from log-probability differences. Suppose at some token the reference model gave the chosen token probability $`0.20`$ and the tuned policy now gives it $`0.60`$. The per-token contribution is

$$\log\frac{0.60}{0.20} = \log 3 = 1.0986$$

With $`\beta = 0.05`$ that token is penalized $`0.05 \times 1.0986 = 0.055`$. Over a 300-token response with an average log-ratio of 1.1, the total KL penalty is $`0.05 \times 300 \times 1.1 \approx 16.5`$ — against a reward-model score that typically lives in a range of about $`\pm 5`$. **The leash dominates.** That is not a bug; it is the design. Now halve $`\beta`$ to 0.025 and the penalty drops to $`8.2`$, and the policy can afford roughly twice as much drift for the same reward.

| $`\beta`$ | Behaviour | Typical symptom |
|---|---|---|
| $`0`$ | No leash | Mode collapse within a few hundred steps — the same flattering paragraph for every prompt |
| $`0.001`$ | Very loose | Reward climbs beautifully, human evaluation gets worse; classic reward hacking |
| $`0.01`$–$`0.1`$ | The usable band | Reward climbs slowly, text stays fluent |
| $`1.0`$ | Very tight | Nothing changes; you have burned GPU-hours to reproduce the SFT model |

▸ **The one-sentence justification for the KL term:** the reward model is only trustworthy on text that looks like what it was trained on, so the KL penalty is not a regularizer in the usual "prevent overfitting" sense — **it is a validity constraint that keeps the policy inside the region where its own objective function still means something.**

#### What would break: setting $`\beta = 0`$

Drop the KL term and the objective becomes $`\max_\pi \mathbb{E}_{y\sim\pi}[r_\phi(x,y)]`$, whose exact solution is a **point mass** on whatever single string maximizes $`r_\phi`$ — the argmax. Not a distribution over good answers: one string. In practice, runs with $`\beta=0`$ show a recognizable progression:

1. Steps 0–200: reward rises, text still looks normal. Everything seems fine.
2. Steps 200–600: responses get longer, more effusive, more hedged. Reward accelerates.
3. Steps 600+: the model emits near-identical text for unrelated prompts, entropy collapses, and the reward model — now far outside its training distribution — reports scores higher than any human-written response ever received.

**The tell is entropy, not reward.** Track the policy's per-token entropy alongside the reward. Healthy RLHF holds entropy roughly flat; a collapsing run shows entropy falling as reward rises. If you monitor only the reward curve, the run looks like a triumph right up until you read the samples.

#### Examples and non-examples: the KL penalty

**✅ Things the KL penalty  does**

| Example | Mechanism |
|---|---|
| Stops the policy emitting `!!!!!!!!!!` even if the RM scores it well | Such strings have near-zero probability under $`\pi_{\text{ref}}`$, so the log-ratio is enormous |
| Preserves grammar and factual recall acquired in pretraining | Drifting away from $`\pi_{\text{ref}}`$ is expensive everywhere, not only where the RM is wrong |
| Keeps the policy inside the RM's valid region | The RM was trained on $`\pi_{\text{ref}}`$-like samples |
| Preserves output diversity | Collapsing to one string maximizes KL cost |

**❌ Near-misses — things the KL penalty does not do**

| Assumed | Why it fails | What actually does it |
|---|---|---|
| Prevents reward hacking | It only bounds *how far* the policy can travel; exploits within the leash are reachable | Better reward models, ensembles, verifiable rewards |
| Guarantees factual accuracy | $`\pi_{\text{ref}}`$ hallucinates too, and staying near it preserves that | Retrieval, verifiers, honesty-focused preference data |
| Acts like weight decay | It constrains the **output distribution**, not the parameter vector; two very different weight vectors can have tiny KL | $`\ell_2`$ regularization, if that's what you want |
| Is a distance between models | KL is asymmetric — $`\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})`$ and $`\mathrm{KL}(\pi_{\text{ref}}\Vert\pi_\theta)`$ differ, and the choice here is deliberate | Nothing; the asymmetry is a feature |

▸ **The boundary:** the KL penalty constrains **where in output-space the policy is allowed to go**, and nothing else. Every property you want that is not "stay near the reference" must come from somewhere else.

> **Common misconception.** *"RLHF is reinforcement learning like AlphaGo."* Superficially yes; structurally, barely. AlphaGo plays millions of games against a **ground-truth** win/loss signal, over a horizon of hundreds of moves, exploring aggressively. RLHF runs one or two epochs against a **learned, exploitable** reward, over a single episode per prompt, with a KL leash whose entire purpose is to *prevent* exploration. It is closer to a carefully damped fine-tune than to self-play. The misconception is tempting because the vocabulary — policy, reward, advantage, episode — is identical. **The vocabulary is shared; the regime is not**, and almost every intuition you import from game-playing RL about "let it explore more" is actively harmful here.

> **Common misconception.** *"The reward is sparse, so RLHF must be sample-inefficient like game RL."* The sparsity is real — one scalar per completion — but the policy starts from a model that already produces fluent, on-topic text, so it is not searching a combinatorial space from scratch. It is doing local reweighting of an already-excellent proposal distribution. This is why RLHF converges in thousands of steps rather than the millions that game-playing agents need.

### The RL formulation

- **State:** the prompt plus tokens generated so far.
- **Action:** the next token.
- **Reward:** the KL penalty every step, plus $`r_\phi`$ at the end. **Extremely sparse.**
- **Episode:** one completion.

PPO's clipped surrogate (full derivation in Ch. 27 §27.7):
▸ $$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\left[\min\left(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],\qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$

**The practical burden:** four models in memory — policy, reference, reward model, and value model. Plus generation in the loop. This is why RLHF is expensive and why DPO was such a welcome result.

#### Unpacking the clipped surrogate

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\left[\min\left(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],\qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$

Two objects, then the trick.

**$`\rho_t`$, the probability ratio.** Read: *"how much more likely is the new policy to take this action than the policy that actually generated the data?"* If the old policy gave the token probability $`0.10`$ and the new one gives $`0.13`$, then $`\rho_t = 1.3`$. $`\rho_t = 1`$ means nothing changed.

**$`\hat A_t`$, the advantage.** Read: *"how much better than average was this?"* Positive means "this turned out better than expected — do more of it." Negative means the opposite. The whole reason PPO needs a **value model** is to estimate that "expected" baseline.

**The clip.** With $`\epsilon = 0.2`$, $`\rho_t`$ is squashed into $`[0.8, 1.2]`$ inside the second branch, and we take the **minimum** of clipped and unclipped. Watch what that does:

| $`\hat A_t`$ | $`\rho_t`$ | Unclipped $`\rho\hat A`$ | Clipped $`\rho\hat A`$ | $`\min`$ | Effect |
|---|---|---|---|---|---|
| $`+2`$ | $`1.05`$ | $`2.10`$ | $`2.10`$ | $`2.10`$ | Normal update, inside the trust region |
| $`+2`$ | $`1.50`$ | $`3.00`$ | $`2.40`$ | $`2.40`$ | **Gain capped.** No extra reward for moving further |
| $`-2`$ | $`1.50`$ | $`-3.00`$ | $`-2.40`$ | $`-3.00`$ | Full penalty kept — pushing a bad action *up* is always punished |
| $`-2`$ | $`0.50`$ | $`-1.00`$ | $`-1.60`$ | $`-1.60`$ | Already suppressed a lot; the min keeps the more pessimistic value |

▸ **What the $`\min`$ buys you:** once the new policy has moved far enough in a favourable direction, the objective goes **flat** — its gradient is zero, so there is no incentive to move further on this batch. It is a one-sided speed limit: free to be cautious, capped on being greedy. That is the entire content of "proximal" in the name, and it replaced TRPO's much heavier machinery of second-order constrained optimization with two lines of code.

#### The four models, counted in gigabytes

For a 7-billion-parameter model in bfloat16 (2 bytes per parameter), just the *weights*:

| Model | Trainable? | Weights | Optimizer state (Adam, fp32) |
|---|---|---|---|
| Policy $`\pi_\theta`$ | Yes | 14 GB | ~56 GB (fp32 master copy + two moments) |
| Reference $`\pi_{\text{ref}}`$ | No, frozen | 14 GB | 0 |
| Reward model $`r_\phi`$ | No, frozen | 14 GB | 0 |
| Value model $`V_\psi`$ | Yes | 14 GB | ~56 GB |
| **Total** | | **56 GB** | **~112 GB** |

Around **170 GB before a single activation is stored**, plus a generation buffer, for a model whose inference footprint is 14 GB. An 80 GB H100 cannot hold it. **This is the "practical burden" made concrete, and it is why the two ideas that follow — GRPO (delete the value model) and DPO (delete the reward model, the value model, and generation) — were received the way they were.**

> **Where this came from.** **PPO** was introduced by **John Schulman and colleagues at OpenAI in 2017**, as a deliberately simplified replacement for their own **TRPO** (2015), which enforced the trust region with a hard KL constraint requiring conjugate-gradient solves and a line search. PPO's contribution was the observation that clipping the ratio achieves most of the benefit with none of the machinery. It was designed for robot locomotion and Atari — nothing about it anticipated language. Its adoption for RLHF is a nice illustration of a pattern this book keeps hitting: the algorithm that wins is rarely the strongest one, it is the one that is hardest to get wrong. And in a further irony, PPO's largest deployment by far — aligning language models — is a setting with *one step per episode's worth of real feedback* and a leash that forbids exploration, which is close to the opposite of the regime it was designed for.

---

## 16.5 DPO — derive it, this is the key result

### The one-line idea

The optimal RLHF policy has a closed form. Invert it to express the reward in terms of the policy, substitute into the Bradley–Terry likelihood, and the reward model disappears — leaving a simple classification loss on the policy itself.

### Step 1: solve the KL-regularized objective exactly

$$\max_\pi\ \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$$

Write it as a single expectation:
$$= \max_\pi\ \mathbb{E}_{y\sim\pi}\left[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\right] = \max_\pi -\beta\,\mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}}\right]$$

Define $`Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}`$ and $`\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}`$. Then

$$= \max_\pi\ -\beta\left[\mathrm{KL}\big(\pi\,\|\,\pi^*\big) - \log Z(x)\right]$$

$`Z(x)`$ doesn't depend on $`\pi`$, and KL is minimized at zero when $`\pi=\pi^*`$. Therefore:

▸ $$\boxed{\ \pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)\ }$$

**This is a Gibbs/Boltzmann distribution** — the reference policy exponentially tilted by reward. It is the exact solution, not an approximation.

#### Step 1, decoded line by line

This derivation is four lines of algebra and one  clever move. Here is every line again, slowly.

**Line 1 — the objective.** $`\max_\pi\ \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})`$. Read: *"find the response distribution with the highest average reward, penalized for drifting from the reference."*

**Line 2 — fold the KL into the expectation.** By definition, $`\mathrm{KL}(\pi\|\pi_{\text{ref}}) = \mathbb{E}_{y\sim\pi}\big[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big]`$ — KL *is already* an average over $`\pi`$. So the two terms are averages over the same distribution and can be merged into one:

$$\mathbb{E}_{y\sim\pi}\left[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\right]$$

**Line 3 — pull out $`-\beta`$ and absorb the reward into the log.** Divide inside by $`-\beta`$: since $`\frac{r}{-\beta} = -\log e^{r/\beta}`$, and $`-\log a - \log b = -\log(ab)`$, the reward slides into the denominator of the log as $`e^{r/\beta}`$:

$$-\beta\,\mathbb{E}_{y\sim\pi}\left[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}}\right]$$

**Line 4 — the clever move.** That fraction *almost* looks like a KL divergence between $`\pi`$ and something. The obstacle is that $`\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}`$ is not a probability distribution — its values don't sum to 1. So **divide by whatever it does sum to**, call that $`Z(x)`$, and you have manufactured a legitimate distribution $`\pi^*`$. The division costs you a $`\log Z(x)`$ term, which comes out of the expectation because it doesn't depend on $`y`$:

$$= -\beta\left[\mathrm{KL}(\pi\|\pi^*) - \log Z(x)\right]$$

**Line 5 — read off the answer.** $`\log Z(x)`$ is a constant as far as $`\pi`$ is concerned. KL is $`\ge 0`$ always, and equals 0 exactly when the two distributions are identical. So the maximum is achieved at $`\pi = \pi^*`$. **Done — no gradient descent, no approximation, an exact closed form.**

| Symbol | Read aloud | What it is |
|---|---|---|
| $`Z(x)`$ | "Z of x" | The **partition function**: $`\sum_y \pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}`$, a sum over *every possible response* |
| $`e^{r/\beta}`$ | "e to the r over beta" | The **tilt factor**. Reward 2 with $`\beta=0.5`$ gives $`e^{4}\approx 54.6`$ |
| $`\pi^*`$ | "pi-star" | The optimal policy — the reference reweighted by that tilt |

**Why $`Z(x)`$ is hopeless to compute, and why that's fine.** The sum runs over all possible responses: for a 200-token response with a 128,000-token vocabulary that is $`128{,}000^{200}`$ terms, a number with about 1,000 digits. There is no algorithm, no approximation, no amount of compute. **The entire elegance of DPO is that this quantity cancels before anyone has to look at it.**

#### The Boltzmann tilt, with real numbers

Set $`\beta = 0.5`$ and imagine a toy universe with only three possible responses to some prompt:

| Response | $`\pi_{\text{ref}}(y)`$ | $`r(x,y)`$ | $`e^{r/\beta}`$ | $`\pi_{\text{ref}}\cdot e^{r/\beta}`$ | $`\pi^*(y)`$ |
|---|---|---|---|---|---|
| A: "Sure, here's how..." | $`0.20`$ | $`2.0`$ | $`54.60`$ | $`10.92`$ | $`0.855`$ |
| B: "I'm not sure." | $`0.50`$ | $`0.0`$ | $`1.00`$ | $`0.50`$ | $`0.039`$ |
| C: "What's the capital of Germany?" | $`0.30`$ | $`-1.0`$ | $`0.135`$ | $`0.041`$ | $`0.003`$ |

Here $`Z(x) = 10.92 + 0.50 + 0.041 = 11.46`$, and each $`\pi^*`$ entry is its unnormalized value divided by $`Z`$. (The three don't quite sum to 1 because of rounding; exactly, they do.)

Read what happened. Response B was the *most likely* thing the reference model would say — probability 0.50, the plurality winner. After the tilt it has probability **0.039**. Response A went from 0.20 to **0.855**. A reward gap of 2.0 with $`\beta=0.5`$ multiplied A's relative odds by $`e^{2/0.5} = e^4 \approx 55`$.

Now change only $`\beta`$:

| $`\beta`$ | $`\pi^*(A)`$ | $`\pi^*(B)`$ | $`\pi^*(C)`$ | Interpretation |
|---|---|---|---|---|
| $`0.25`$ | $`0.9993`$ | $`0.0007`$ | $`\approx 0`$ | Nearly deterministic — the leash is off |
| $`0.5`$ | $`0.855`$ | $`0.039`$ | $`0.003`$ | Strong preference, some diversity kept |
| $`2.0`$ | $`0.470`$ | $`0.432`$ | $`0.098`$ | Gentle tilt |
| $`\to\infty`$ | $`0.20`$ | $`0.50`$ | $`0.30`$ | Exactly $`\pi_{\text{ref}}`$ — no change at all |
| $`\to 0`$ | $`1`$ | $`0`$ | $`0`$ | A point mass on the argmax — the $`\beta=0`$ collapse from §16.4, derived |

▸ **$`\beta`$ is temperature.** Literally: $`\pi^* \propto \pi_{\text{ref}}\,e^{r/\beta}`$ is the Boltzmann distribution with $`\beta`$ playing the role of $`T`$. Large $`\beta`$ = hot = the reward barely matters = stay near the reference. Small $`\beta`$ = cold = freeze onto the single best response. **The formula that governs how tightly you can align a language model is the same formula that governs how gas molecules distribute across energy states**, and the parameter means the same thing in both.

> **Analogy.** $`\pi_{\text{ref}}`$ is a crowd's natural distribution across the rooms of a building. $`r(x,y)`$ is how warm each room is. $`\beta`$ is how much people care about warmth. Cold-blooded people ($`\beta`$ small) all end up crammed in the warmest room; indifferent people ($`\beta`$ large) stay spread out as before. Nobody teleports — you can only redistribute the crowd that was already there, which is exactly why $`\pi^*`$ can never put mass on a response $`\pi_{\text{ref}}`$ assigns probability zero. **If the reference model would never say it, no amount of reward makes it appear.** Look at the formula: $`\pi_{\text{ref}}(y) = 0 \Rightarrow \pi^*(y) = 0`$, whatever $`r`$ says.

### Step 2: invert

$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

### Step 3: substitute into Bradley–Terry

The BT likelihood depends only on the **difference** $`r(x,y_w)-r(x,y_l)`$, and $`\beta\log Z(x)`$ is the same for both — **it cancels.** This is the crucial step, and it's why the intractable partition function never has to be computed.

▸ $$\boxed{\ \mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]\ }$$

∎

#### Steps 2 and 3, decoded — where the reward model goes

**Step 2 is ordinary algebra.** Take $`\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}`$, divide both sides by $`\pi_{\text{ref}}`$, take logs, multiply by $`\beta`$:

$$\log\frac{\pi^*}{\pi_{\text{ref}}} = \frac{r}{\beta} - \log Z(x) \quad\Longrightarrow\quad r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

Read the result aloud: *"the reward of a response equals beta times the log of how much more likely the optimal policy is to say it than the reference policy would be, plus a per-prompt constant."*

▸ **This is the sentence the whole chapter turns on.** It says a reward function and a policy are **two descriptions of the same object.** You do not need to learn a reward and then optimize it — a policy *is already* a reward function, read through the lens $`\beta\log(\pi/\pi_{\text{ref}})`$. The DPO paper's title says exactly this: *Your Language Model Is Secretly a Reward Model*.

**Step 3 is the cancellation.** Bradley–Terry needs only the *difference* $`r(x,y_w) - r(x,y_l)`$. Substitute:

$$r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \underbrace{\beta\log Z(x) - \beta\log Z(x)}_{=\ 0}$$

$`Z(x)`$ depends on the **prompt only** — both responses share the same prompt — so it appears twice with opposite signs and vanishes. The thousand-digit sum that no computer could ever evaluate cancels against itself.

> **Analogy.** Two people's altitudes above sea level are each impossible to know without a survey benchmark. But *which of them is standing higher* needs no benchmark at all — the unknown sea-level offset subtracts out. $`\log Z(x)`$ is the sea level, identical for both responses to the same prompt, and Bradley–Terry only ever asks who is higher.

#### Unpacking the DPO loss

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

Define the shorthand the paper uses:

$$\hat r_\theta(x,y) \;=\; \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$

Call $`\hat r_\theta`$ the **implicit reward**: it is not produced by any reward network, it is *computed from the policy itself*. Then $`\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\hat r_\theta(x,y_w) - \hat r_\theta(x,y_l))]`$ — **which is character-for-character the reward-model loss from §16.3**, with $`r_\phi`$ replaced by $`\hat r_\theta`$. Nothing new was invented. A quantity that used to require a separate network is now read directly off the policy.

**How you actually compute it.** For each of the four terms you need one number: the summed log-probability of a response's tokens.

```python
def seq_logprob(model, prompt, response):
    logits = model(prompt + response)                     # one forward pass
    lp = log_softmax(logits, dim=-1)
    return lp[range(len(response)), response].sum()       # sum over response tokens only

pi_w   = seq_logprob(policy, x, y_w);  pi_l   = seq_logprob(policy, x, y_l)
ref_w  = seq_logprob(ref,    x, y_w);  ref_l  = seq_logprob(ref,    x, y_l)   # no grad

logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
loss   = -logsigmoid(logits).mean()
```

Four forward passes, two of them under `no_grad`, and no generation anywhere. Compare that to §16.4's generate-score-advantage-clip loop.

#### A full numeric pass through DPO

Take $`\beta = 0.1`$ and a preference pair. Log-probabilities of whole responses are large negative numbers — a 60-token response with average per-token log-prob $`-0.7`$ has sequence log-prob $`\approx -42`$.

| Quantity | Value |
|---|---|
| $`\log\pi_\theta(y_w\mid x)`$ | $`-40.0`$ |
| $`\log\pi_{\text{ref}}(y_w\mid x)`$ | $`-43.0`$ |
| $`\log\pi_\theta(y_l\mid x)`$ | $`-38.0`$ |
| $`\log\pi_{\text{ref}}(y_l\mid x)`$ | $`-39.0`$ |

Implicit rewards:

$$\hat r_w = 0.1\times(-40.0 - (-43.0)) = 0.1 \times 3.0 = 0.30$$
$$\hat r_l = 0.1\times(-38.0 - (-39.0)) = 0.1 \times 1.0 = 0.10$$

Margin $`= 0.30 - 0.10 = 0.20`$. Loss $`= -\log\sigma(0.20) = -\log(0.5498) = 0.598`$ nats.

**Now read the diagnostics.** Note $`\log\pi_\theta(y_l) = -38.0`$ is *higher* than $`\log\pi_\theta(y_w) = -40.0`$: under raw likelihood the model still prefers the loser. DPO doesn't care, because the comparison is against the reference — the winner was pushed up by 3.0 nats while the loser was pushed up by only 1.0. **DPO optimizes relative movement, not absolute likelihood.** This is exactly the mechanism behind the failure mode flagged two sections down: both $`y_w`$ and $`y_l`$ can have their absolute probabilities *fall* and the loss will still improve, as long as the loser falls faster.

Three states of the same pair:

| State | $`\hat r_w`$ | $`\hat r_l`$ | Margin | Loss | $`\sigma(\hat r_l - \hat r_w)`$ (gradient weight) |
|---|---|---|---|---|---|
| Untrained ($`\pi_\theta = \pi_{\text{ref}}`$) | $`0`$ | $`0`$ | $`0`$ | $`0.693`$ | $`0.50`$ — full-strength update |
| Partly trained | $`0.30`$ | $`0.10`$ | $`0.20`$ | $`0.598`$ | $`0.45`$ |
| Well trained | $`2.0`$ | $`-1.0`$ | $`3.0`$ | $`0.049`$ | $`0.047`$ — nearly ignored |
| **Backwards** | $`-0.5`$ | $`1.5`$ | $`-2.0`$ | $`2.127`$ | $`0.88`$ — dominates the batch |

▸ Notice the first row: at initialization $`\pi_\theta`$ *is* $`\pi_{\text{ref}}`$, so every implicit reward is exactly 0 and the loss is exactly $`\log 2 = 0.693`$ for every pair. **If your DPO run doesn't start at 0.693, something is wrong with your reference model** — usually you forgot to freeze it, or loaded the wrong checkpoint. It is the cheapest sanity check in this chapter.

### What the gradient does

$$\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l - \hat r_w)}_{\text{weight: high when the model is wrong}}\big(\underbrace{\nabla\log\pi_\theta(y_w)}_{\text{increase}} - \underbrace{\nabla\log\pi_\theta(y_l)}_{\text{decrease}}\big)\Big]$$

▸ Raise the winner's likelihood, lower the loser's, **weighted by how badly the implicit reward model currently has it backwards.** Pairs the model already scores correctly contribute almost nothing — automatic hard-example mining.

#### The gradient, decoded

Three factors, multiplied:

| Factor | Read aloud | What it does |
|---|---|---|
| $`\sigma(\hat r_l - \hat r_w)`$ | "sigmoid of loser-minus-winner" | A scalar in $`(0,1)`$: **how wrong am I on this pair?** Near 1 when inverted, near 0 when already right |
| $`\nabla\log\pi_\theta(y_w)`$ | "grad log pi of the winner" | The direction in weight space that makes $`y_w`$ **more** likely |
| $`-\nabla\log\pi_\theta(y_l)`$ | "minus grad log pi of the loser" | The direction that makes $`y_l`$ **less** likely |

Using the numbers from the table above, with the well-trained row: $`\sigma(-1.0 - 2.0) = \sigma(-3.0) = 0.047`$. That pair's contribution is scaled by **4.7%**. The backwards row gets $`\sigma(1.5 - (-0.5)) = \sigma(2.0) = 0.88`$ — **an 18× larger update from the same batch.**

> **Analogy.** A tutor grading a stack of homework who spends thirty seconds on the ones already correct and twenty minutes on the ones with the reasoning inverted. Nobody programmed that priority; it falls out of the sigmoid. This is the same mechanism as focal loss in object detection and the same reason cross-entropy has bounded, self-annealing gradients (Ch. 1) — **easy examples stop teaching once you've learned them, automatically.**

#### Examples and non-examples: what DPO actually is

**✅ Accurate statements about DPO**

| Statement | Why it's true |
|---|---|
| It is a binary classification loss over preference pairs | Literally $`-\log\sigma(\text{margin})`$, the logistic loss |
| It optimizes the *same* KL-regularized objective as PPO-RLHF | Steps 1–3 are an exact re-parameterization, not an approximation |
| It still has a reward model | The reward is $`\beta\log(\pi_\theta/\pi_{\text{ref}})`$ — implicit, but present, and you can read it out |
| It still has a $`\beta`$ and a KL leash | $`\beta`$ appears explicitly; the reference model is still loaded |
| It requires exactly one frozen extra model | Only $`\pi_{\text{ref}}`$, and its log-probs can be precomputed once and cached |

**❌ Near-misses — commonly said about DPO, and wrong**

| Claim | Why it's wrong | The accurate version |
|---|---|---|
| "DPO has no reward model" | It has an implicit one, definitionally | DPO has no **separately trained** reward network |
| "DPO is RLHF without the RL" | The objective is identical; what's gone is the *sampling loop* | DPO is the same objective solved **offline** |
| "DPO removes the KL constraint" | $`\beta`$ is right there in the loss | DPO enforces KL **only on responses in the dataset** |
| "DPO is just SFT on the chosen responses" | SFT on $`y_w`$ alone has no $`y_l`$ term and no reference | SFT is the $`\beta\to\infty`$-ish degenerate cousin; the contrastive term is the whole point |
| "DPO can't over-optimize because there's no reward to hack" | It over-optimizes the implicit reward on unseen $`y`$ | DPO's failure mode is **off-distribution**, not off-metric |

▸ **The boundary:** DPO and PPO-RLHF differ in **where the responses come from**, not in what is being optimized. PPO samples from the current policy every step (**on-policy**), so the KL leash binds wherever the policy actually goes. DPO reads from a fixed file (**off-policy**), so the leash binds only where the file has data — and the policy is free to do anything it likes everywhere else.

> **Common misconception.** *"DPO's derivation shows it's equivalent to RLHF, so they should give the same model."* The **objectives** are equivalent; the **optimization problems** are not. PPO's expectation is taken over $`y \sim \pi_\theta`$, which is re-drawn as $`\theta`$ changes; DPO's is taken over a fixed dataset collected from some other policy. Equivalent objectives optimized over different sampling distributions reach different solutions. The misconception is tempting precisely *because* the derivation is exact — it is easy to read "exact re-parameterization" as "same algorithm." It is the difference between a map being correct and a map telling you which roads are open.

### The trade-offs

| DPO | PPO-RLHF |
|---|---|
| 2 models in memory | 4 models |
| No sampling loop | Online generation |
| Simple, stable, cheap | Complex, finicky |
| **Offline** — fixed preference data | **Online** — learns from its own samples |
| Can over-optimize on OOD responses | KL constraint is enforced on-policy |

▸ **Where DPO is weaker, and why:** its constraint is only enforced on the responses in the dataset. For $`y`$ far from the data, $`\log\frac{\pi_\theta}{\pi_{\text{ref}}}`$ is unconstrained, so DPO can push probability mass onto unseen outputs — the observed failure is that DPO often *decreases* the likelihood of both $`y_w`$ and $`y_l`$ while increasing the likelihood of neither. Fixes: **iterative/online DPO** (regenerate preferences from the current policy each round), and adding an SFT term on $`y_w`$.

Careful head-to-head studies find well-tuned online PPO still edges out DPO on the hardest tasks; DPO wins decisively on cost.

#### What "both likelihoods go down" means, concretely

This is the most-reported surprise in practical DPO, and it is worth seeing as numbers rather than as a warning.

| Step | $`\log\pi_\theta(y_w)`$ | $`\log\pi_\theta(y_l)`$ | $`\hat r_w`$ | $`\hat r_l`$ | Margin | Loss |
|---|---|---|---|---|---|---|
| 0 | $`-43.0`$ | $`-39.0`$ | $`0.00`$ | $`0.00`$ | $`0.00`$ | $`0.693`$ |
| 200 | $`-44.0`$ | $`-43.0`$ | $`-0.10`$ | $`-0.40`$ | $`0.30`$ | $`0.554`$ |
| 600 | $`-47.0`$ | $`-52.0`$ | $`-0.40`$ | $`-1.30`$ | $`0.90`$ | $`0.341`$ |

(reference log-probs fixed at $`-43.0`$ and $`-39.0`$; $`\beta = 0.1`$)

The loss halves. The margin triples. **And the model has become less likely to produce the preferred response than when it started** — $`-47.0`$ versus $`-43.0`$, a factor of $`e^{-4}\approx 0.018`$. Every number on the dashboard is green.

Where did the probability go? Onto responses that appear in *neither* column, which the loss function never looks at. The loss only constrains the *difference* $`\hat r_w - \hat r_l`$, and pushing the loser down by 5 while pushing the winner down by 4 satisfies it perfectly.

▸ **The diagnostic every DPO run should log:** not just loss and margin, but **the absolute $`\log\pi_\theta(y_w)`$**. If it falls steadily, you are draining probability mass into unexamined territory, and the model that comes out will be confidently strange. The two standard repairs are to add an SFT term $`-\lambda\log\pi_\theta(y_w\mid x)`$ that anchors the winner's absolute likelihood, and to regenerate preference pairs from the current policy every few hundred steps (**iterative** or **online DPO**) so the dataset never goes stale.

> **Common misconception.** *"Offline methods are safer than online ones because nothing can drift."* The opposite is true here, and the reason is precise: an **on-policy** method's KL constraint is evaluated on whatever the policy is actually producing right now, so drift is measured and penalized the instant it happens. An **offline** method's constraint is evaluated only on a fixed file of responses, so the policy is unconstrained everywhere the file is silent — which, in a space of $`128{,}000^{200}`$ possible responses, is essentially everywhere. The misconception is tempting because "offline" sounds like "contained." It means "unobserved."

> **Where this came from.** **DPO** was published in 2023 by **Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher Manning, and Chelsea Finn** at Stanford, under the title *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*. Its impact came less from novelty of ingredients — the closed-form solution to the KL-regularized objective was standard in the control-as-inference literature, and Bradley–Terry dates to 1952 — than from noticing that composing two known results deletes an entire engineering pipeline. Within roughly a year of publication it had become the default alignment method for open-weight models, essentially because a method you can run on two GPUs will be run a hundred times more often than one that needs sixteen. **A large part of what determines which algorithm a field adopts is how many people can afford to try it.**

### The DPO family

| Method | Change |
|---|---|
| **IPO** | replaces the log-sigmoid with a squared loss, avoiding the overfitting that occurs when preferences are deterministic |
| **KTO** | uses *unpaired* binary feedback (good/bad), with a prospect-theory-inspired value function — much easier data collection |
| **ORPO** | folds preference optimization into SFT via an odds-ratio penalty; **no reference model at all** |
| **SimPO** | uses length-normalized average log-probability as the implicit reward; no reference model; directly addresses length bias |
| **RRHF / RSO** | rank-based and rejection-sampling variants |

#### The DPO family, decoded

Each of these fixes one specific defect. Knowing *which* defect is the whole value of the table.

**IPO — identity preference optimization.** The defect: if your annotators are unanimous ($`p^* = 1.0`$ for a pair, not 0.7), the Bradley–Terry model's only way to represent certainty is an **infinite** reward gap. So the loss keeps rewarding a wider margin forever, and $`\pi_\theta`$ is driven toward a point mass — the $`\beta\to 0`$ collapse, arriving through the back door even though $`\beta`$ is finite. IPO replaces $`-\log\sigma(m)`$ with a squared loss around a finite target margin, so the optimum is at a *specific* margin rather than at infinity. **Use it when your preference labels are deterministic** — for example, when the "preference" came from a rule rather than a person.

**KTO — Kahneman–Tversky optimization.** The defect: paired data is expensive. You need two responses to the same prompt, both graded. But every product already logs thumbs-up and thumbs-down on *single* responses. KTO learns from that unpaired signal, using a value function shaped by prospect theory — losses weigh more than equivalent gains, mirroring how humans actually evaluate outcomes. **Use it when you have a thumbs-up/down stream and no paired comparisons.**

**ORPO — odds-ratio preference optimization.** The defect: two stages (SFT, then preferences) and a reference model to keep in memory. ORPO adds an odds-ratio penalty directly to the SFT loss, so one run does both and **no reference model is loaded at all** — cutting memory roughly in half again versus DPO. The odds of a response are $`\frac{\pi(y\mid x)}{1-\pi(y\mid x)}`$, and the penalty pushes the winner's odds above the loser's. **Use it when you're memory-bound or want a single-stage recipe.**

**SimPO — simple preference optimization.** The defect: DPO's implicit reward is a *sum* over tokens, so a longer response accumulates more of it, and length bias re-enters through the arithmetic. SimPO uses the length-**averaged** log-probability as the reward and drops the reference model entirely:

$$\hat r_{\text{SimPO}}(x,y) = \frac{\beta}{\lvert y\rvert}\sum_{t=1}^{\lvert y\rvert}\log\pi_\theta(y_t\mid x, y_{<t})$$

Dividing by $`\lvert y \rvert`$ makes the reward a *rate* rather than a total, which is why it is structurally immune to padding. **Use it when length bias is your visible failure.**

**RRHF and RSO.** RRHF ranks $`k>2`$ candidate responses at once rather than handling pairs; RSO first uses rejection sampling to draw preference pairs from something closer to the optimal policy $`\pi^*`$, correcting the fact that DPO's derivation assumes on-policy data while its practice uses whatever file you have.

#### Examples and non-examples: choosing a post-training method

**✅ Good matches between problem and method**

| Situation | Right choice | Why |
|---|---|---|
| 5,000 curated demonstrations, no preferences yet | **SFT** | You have demonstrations, not comparisons. Preference methods need pairs |
| 60,000 paired human preferences, 2 GPUs, one week | **DPO** | Fits in memory, no generation loop, one hyperparameter that matters |
| 200,000 math problems with known answers | **GRPO + RLVR** | The reward is free and unhackable; a learned RM would only add noise |
| A production thumbs-up/down log, no pairs | **KTO** | Built for exactly this data shape |
| A frontier lab, large budget, chasing the last few points | **Online PPO or iterative DPO** | On-policy data is what buys the final margin |
| Responses keep getting longer with no quality gain | **SimPO**, or a length penalty | Attack the specific exploit |

**❌ Mismatches that look reasonable**

| Situation | Tempting but wrong | Why it fails | Do instead |
|---|---|---|---|
| No SFT stage, straight to DPO from a base model | "DPO subsumes SFT" | The reference model is a *base* model that doesn't follow instructions, so the leash anchors you to non-compliance | SFT first, use that as $`\pi_{\text{ref}}`$ |
| RLHF to teach a model a new API's syntax | "RL will figure it out" | A preference signal cannot transmit facts; you'd need $`\sim 10^6`$ comparisons to convey what one document says | SFT, or retrieval |
| Preference data where the annotator saw only the longer answer's first paragraph | "More data is better" | You are labelling *presentation*, and the RM will learn it perfectly | Fix the annotation interface first |
| DPO for 8 epochs on 2,000 pairs | "It's just fine-tuning" | Margins blow up, absolute likelihoods collapse | 1–2 epochs, monitor $`\log\pi_\theta(y_w)`$ |
| Reusing a reward model across a major policy update | "The RM is a fixed function" | The policy has drifted off the RM's training distribution | Retrain or re-anchor the RM |

▸ **The boundary:** SFT transfers **behaviour you can demonstrate**; preference methods transfer **judgements you can only compare**; verifiable rewards transfer **outcomes you can check**. Choose by asking what kind of signal you can actually produce cheaply — the method follows from the data, never the other way round.

> **Where this came from.** **KTO** takes its name from **Daniel Kahneman and Amos Tversky**, whose 1979 *prospect theory* paper in *Econometrica* showed that people evaluate outcomes as gains and losses relative to a reference point, and weigh losses roughly twice as heavily as equal-sized gains. Kahneman received the Nobel Memorial Prize in Economics in 2002; Tversky had died in 1996 and the prize is not awarded posthumously. KTO's value function is that asymmetry written as a loss — so an alignment algorithm for language models inherits its shape from four-decade-old experiments about how humans mis-weigh gambles. **IPO** came from a DeepMind analysis (Azar and colleagues, 2023) that set out to describe DPO and RLHF in one theoretical framework, and discovered the deterministic-preference overfitting problem as a corollary of that framework rather than by observing it in the wild — a rare case of the theory finding the bug first.

---

## 16.6 GRPO and verifiable rewards

### The motivation

PPO needs a **value network** — an extra model of the same size, trained on a noisy regression target, that is hard to fit for language. GRPO removes it.

### The mechanism

▸ Sample a **group** of $`G`$ completions for the same prompt, and use the group's own reward statistics as the baseline:

$$\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac1G\sum_{i=1}^G \min\left(\rho_i\hat A_i,\ \mathrm{clip}(\rho_i,1\pm\epsilon)\hat A_i\right)\right] + \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$$

**Why this is valid:** any baseline that does not depend on the action leaves the policy gradient unbiased (Ch. 27 §27.5). The group mean is a Monte-Carlo estimate of $`V(x)`$ computed from samples rather than from a learned network. **You trade a value model for $`G\times`$ more sampling** — which is a good trade when generation is cheap relative to training a second large model.

#### Reading the group-relative advantage in plain English

$$\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

Read aloud: *"take this completion's reward, subtract the average reward of its siblings, and divide by their spread."* If you have ever computed a z-score or called `(x - x.mean()) / x.std()`, you have written this line.

| Piece | Read aloud | What it means |
|---|---|---|
| $`G`$ | "G" | Group size — how many completions we sample for the *same* prompt. Typically 4–64 |
| $`r_i`$ | "r sub i" | The reward of the $`i`$-th completion in the group |
| $`\mathrm{mean}(r_1,\dots,r_G)`$ | "the group mean" | The **baseline** — "what a typical attempt at this prompt scores" |
| $`\mathrm{std}(\cdot)`$ | "the standard deviation" | Rescales so every prompt contributes comparably |
| $`\hat A_i`$ | "A-hat sub i" | Positive: better than my own siblings. Negative: worse |

> **Analogy.** Grading on a curve, per question. PPO hires a separate professional forecaster (the value network) to predict "a typical student scores 62 on this question." GRPO just has four students answer it and uses their average. The forecaster is more sample-efficient when it's accurate, but it has to be trained, it has to be as large as the model, and for language its regression target is so noisy that it's frequently wrong. **Four students are noisier but never wrong about what four students scored.**

**Numbers, please.** Take a math problem, $`G = 4`$, binary correctness rewards:

| Completion | $`r_i`$ | $`r_i - \text{mean}`$ | $`\hat A_i`$ |
|---|---|---|---|
| 1 (correct) | $`1`$ | $`+0.5`$ | $`+0.87`$ |
| 2 (correct) | $`1`$ | $`+0.5`$ | $`+0.87`$ |
| 3 (wrong) | $`0`$ | $`-0.5`$ | $`-0.87`$ |
| 4 (wrong) | $`0`$ | $`-0.5`$ | $`-0.87`$ |

Mean $`= 0.5`$, standard deviation $`= \sqrt{0.25} = 0.5`$ (population), so $`\hat A = \pm 0.5/0.5 = \pm 1`$; using the sample standard deviation $`0.577`$ instead gives $`\pm 0.87`$. Either way: **the two correct chains get pushed up, the two wrong ones get pushed down, and the size of the push is the same for both.** No value network was consulted. No reward model was consulted. A calculator checked four answers.

Now a harder problem where only one of four succeeds:

| Completion | $`r_i`$ | $`\hat A_i`$ (sample std $`=0.5`$) |
|---|---|---|
| 1 (correct) | $`1`$ | $`+1.5`$ |
| 2–4 (wrong) | $`0`$ | $`-0.5`$ each |

The single success gets a **3× larger** update than each failure gets a penalty. GRPO automatically pays more attention to rare successes on hard problems — which is precisely the signal you want, and which required no design.

#### What would break: when the group is unanimous

Set all four rewards to 1 (the problem is trivial) or all four to 0 (the problem is impossible). Then $`\mathrm{mean} = r_i`$ for every $`i`$, the numerator is exactly 0, and the denominator is **also** 0.

$$\hat A_i = \frac{0}{0}$$

In code this is a `NaN` that propagates through the entire gradient and destroys the run in one step. Every real implementation guards it:

```python
adv = (r - r.mean()) / (r.std() + 1e-8)     # the epsilon is not optional
# and, usually:
if r.std() < 1e-6:                          # unanimous group -> no signal
    skip_this_prompt()
```

▸ **The deeper point behind the guard:** a unanimous group carries **zero information**, whatever you do about the arithmetic. If every attempt succeeds, there is nothing to prefer; if every attempt fails, there is nothing to imitate. **GRPO learns only from prompts the model gets right *sometimes*** — which turns curriculum design into a concrete, measurable engineering task: keep feeding the model problems whose pass rate sits away from 0 and 1. A problem set on which your model scores 100% is worth exactly as much training signal as one it scores 0% on, which is none.

#### Examples and non-examples: valid baselines in policy gradients

The claim "any baseline that does not depend on the action leaves the gradient unbiased" is doing real work here. Its boundary is sharp.

**✅ Valid baselines**

| Baseline | Why it's valid |
|---|---|
| The group mean of rewards for the *same prompt* | Depends on the prompt and on the sampling process, not on which action you're evaluating |
| A learned $`V_\psi(s)`$ — PPO's value network | A function of state only |
| A running average of all rewards seen so far | Constant with respect to the current action |
| Zero | Trivially independent of the action; this is vanilla REINFORCE |

**❌ Invalid baselines — look like they'd work, but bias the gradient**

| Looks like it | Why it fails | Consequence |
|---|---|---|
| Subtracting $`r_i`$ itself | Depends on the action being evaluated | Every advantage is 0; no learning at all |
| A baseline computed from the action's own reward, e.g. $`\mathrm{median}`$ including $`i`$ with $`G=2`$ | The action's reward leaks into its own baseline | Systematically biased toward pushing down whatever was sampled |
| $`Q(s,a)`$ — the action-value | Explicitly a function of the action | This is the thing the baseline is meant to be subtracted *from* |
| The *best* reward in the group | Determined by which action happened to be best | Biases against high-variance strategies |

▸ **The boundary:** a baseline is valid exactly when it is **conditionally independent of the action given the state.** The proof is one line — $`\mathbb{E}_{a\sim\pi}[b(s)\nabla\log\pi(a\mid s)] = b(s)\nabla\sum_a \pi(a\mid s) = b(s)\nabla 1 = 0`$ — so subtracting it changes the variance and nothing else. Strictly, GRPO's group mean *does* include $`r_i`$ in its own average, so it is very slightly biased at small $`G`$; in practice this is dwarfed by the variance it removes, and dividing by the standard deviation introduces a similar mild distortion for the same reason and the same trade.

> **Where this came from.** **GRPO** was introduced in **DeepSeekMath (Shao and colleagues, 2024)** as a way to make large-scale RL on mathematics affordable, and reached wide attention through **DeepSeek-R1 (2025)**, where it was the engine of the reasoning training. The underlying idea — use several samples from the same state as each other's baseline — is not new; it appears in the RL literature under names like *vine* sampling and leave-one-out control variates, and the simplest version dates back to the variance-reduction techniques accompanying **Ronald Williams' REINFORCE in 1992**. What was new was recognizing that in language modelling the economics have inverted: generation runs on optimized inference kernels with batching and KV-caching, so **sampling 16 completions is cheaper than training and storing a second 7-billion-parameter network.** GRPO is less a theoretical advance than a correct reading of a hardware cost curve.

### RLVR — reinforcement learning from verifiable rewards

▸ For math, code, and formal tasks, **skip the reward model entirely**: check the answer.

$$r(x,y) = \mathbb{1}[\text{answer is correct}] \quad\text{or}\quad \mathbb{1}[\text{unit tests pass}]$$

**This eliminates reward hacking at the source** — you cannot fool a compiler. It is the single most important recent development in post-training, and it is why reasoning models improved so abruptly.

**Limits:** only applies to verifiable domains; the binary signal is sparse; and models still learn to exploit *test* weaknesses (special-casing to pass unit tests) — so verification quality becomes the new bottleneck.

#### Unpacking the indicator reward

$$r(x,y) = \mathbb{1}[\text{answer is correct}]$$

$`\mathbb{1}[\,\cdot\,]`$ is the **indicator function**: read it as "1 if the thing in brackets is true, 0 if false." In code it is a comparison.

```python
def reward(problem, completion):
    answer = extract_boxed(completion)          # pull \boxed{...} out of the text
    return 1.0 if answer == problem.gold else 0.0
```

That is the entire reward model. No parameters, no training, no distribution shift, no ensemble, no drift. Compare its properties with a learned reward model, side by side:

| Property | Learned RM $`r_\phi`$ | Verifier $`\mathbb{1}[\cdot]`$ |
|---|---|---|
| Parameters | ~7 billion | 0 |
| Can be wrong | Yes, ~25–35% of pairs | Only if the gold answer is wrong |
| Degrades as the policy improves | Yes — that's distribution shift | **No.** Correctness doesn't drift |
| Can be hacked | Yes, and reliably is | Not the *checker*; the *test suite*, sometimes |
| Cost per evaluation | A forward pass | A string comparison |
| Range | Continuous, uncalibrated | $`\{0, 1\}`$, absolute |

▸ **The row that changed the field is "degrades as the policy improves."** Every learned reward model has a horizon past which optimizing it stops helping, because the policy has walked off the RM's training distribution. **A verifier has no such horizon.** You can optimize against `answer == gold` for a million steps and the signal is exactly as truthful on step 1,000,000 as on step 1. That is why reasoning capability improved so abruptly rather than gradually: the ceiling wasn't raised, it was removed.

#### Examples and non-examples: verifiable rewards

**✅  verifiable**

| Task | The verifier | Why it's sound |
|---|---|---|
| "What is $`\int_0^1 3x^2\,dx`$?" | Compare to $`1`$ symbolically | One correct answer, checkable without judgement |
| "Write a function that returns the $`n`$-th prime" | Run a held-out test suite | Execution is ground truth |
| "Is this Lean proof valid?" | Run the Lean type-checker | A proof assistant *is* the definition of correctness |
| "Solve this Sudoku" | Check the constraints | Verification is far cheaper than solution |
| "Translate to SQL and return the row count" | Execute against the database, compare | The database has the answer |

**❌ Near-misses — feel verifiable, but aren't**

| Task | Why it isn't verifiable | What you actually have |
|---|---|---|
| "Summarize this article" | No unique correct summary; ROUGE overlap is not correctness | A **learned or human judgement** |
| "Is this code good?" | Passing tests ≠ good; a hardcoded lookup passes | **Tests verify behaviour, not quality** |
| "Answer this multiple-choice medical question" | Verifiable against the *key*, but the key encodes one clinical opinion | A verifiable proxy for a contested target |
| "Write a poem about autumn" | There is no checker | Pure preference territory |
| "Give a factual answer" checked by an LLM judge | The judge is a learned model with its own failure modes | **RLAIF wearing a verifier's uniform** |
| "Pass these 12 unit tests" | The model can special-case the 12 inputs | Verifiable, but with a **specification gap** |

▸ **The boundary:** a reward is verifiable when checking is **cheap, deterministic, and defined without reference to anyone's taste.** Note that "cheap to check, expensive to produce" is the same asymmetry that defines the complexity class NP — which is not a coincidence. RLVR works best exactly where verification is easier than generation.

> **Common misconception.** *"Verifiable rewards eliminate reward hacking."* They eliminate hacking of the **reward model**, and relocate it to the **specification**. A model told to pass twelve unit tests will, given enough optimization pressure, write code that passes those twelve inputs and nothing else. A model told to produce a `\boxed{}` answer will learn to emit a plausible-looking box even when its reasoning failed. The misconception is tempting because "you cannot fool a compiler" is true and satisfying — but you were never trying to fool the compiler. You were trying to write good code, and the compiler was only ever a proxy for that too. **Goodhart's law applies to every proxy, including exact ones; what changes is how much slack the proxy leaves.**

> **Common misconception.** *"A binary reward is too sparse to learn from."* It would be, from a random initialization. But an RLVR run starts from a model that already solves, say, 30% of the training problems — so roughly a third of all groups contain both a success and a failure, and each of those yields a clean contrastive signal. The sparsity that would sink a from-scratch agent is survivable precisely because pretraining already supplied a competent proposal distribution. This also explains the sharp practical rule that follows: **as the model's pass rate approaches 100% on a dataset, that dataset stops teaching**, so RLVR pipelines constantly need harder problems.

---

## 16.7 Reasoning models

### Chain of thought

Prompting the model to produce intermediate steps improves accuracy dramatically on multi-step problems.

▸ **The rigorous reason (Ch. 11 §11.9):** a fixed-depth transformer has bounded serial computation per forward pass. Emitting intermediate tokens gives it an external, unbounded scratchpad — each generated token is another forward pass conditioned on the previous ones. **Chain of thought converts a parallel-compute-limited model into a serial one.** It is buying depth with tokens.

- **Zero-shot CoT:** "Let's think step by step."
- **Self-consistency:** sample $`k`$ chains, majority-vote the final answer. Accuracy rises roughly logarithmically in $`k`$.
- **Tree of thoughts / search:** explore and prune branches with a value estimate.

#### "Buying depth with tokens," decoded

This is the most important claim in the section and it is easy to nod along to without understanding it. Here it is mechanically.

A transformer with $`L`$ layers performs exactly $`L`$ sequential transformations before it must emit a token. That is a **hard architectural limit on serial computation per token** — not a soft one, not a matter of capacity. A 32-layer model gets 32 steps of "and then, and then, and then," no matter how wide it is.

Now compare two ways of answering *"A store has 17 crates of 24 bottles. They sell 138 bottles. How many remain?"*

**Without chain of thought.** The model must emit the digit `2` of `270` as its very next token. Getting there requires: multiply $`17\times 24`$, then subtract 138, then decompose the result into digits, then emit the leading one. That is at least three dependent operations, each of which must complete inside those 32 layers, sharing them with parsing the question and formatting the output.

**With chain of thought.** The model emits:

```
17 x 24 = 408
408 - 138 = 270
The answer is 270.
```

Count the forward passes. That output is roughly 20 tokens, so the model ran its 32 layers **20 times**, for about 640 sequential transformations — and, crucially, `408` is *written down* and read back through attention on every subsequent pass. It no longer has to be held in a hidden state; it is in the context, exactly as retrievable as the original question.

▸ **The precise statement:** the number of serial computation steps available goes from $`L`$ to $`L \times (\text{tokens generated})`$. Chain of thought is not a prompting trick that makes the model "try harder." **It is a change to the computational class of the system**, from fixed-depth to effectively unbounded-depth, purchased at a price of one forward pass per step of depth.

> **Analogy.** Doing long division in your head versus on paper. Your brain is not upgraded by the paper. What changes is that intermediate results stop competing for working memory — you can write 408, forget it entirely, and read it back when needed. The transformer's context window is the paper, and attention is the eye that reads it back. Ask a person to multiply $`17\times 24`$ silently and instantly and many will fail; give them five seconds and a pencil and nearly all succeed. **Neither their knowledge nor their intelligence changed.**

**Self-consistency, with numbers.** Sample $`k`$ independent chains and take the majority answer. Suppose each chain is right 60% of the time and the wrong answers are scattered rather than agreeing with each other. Then majority voting over $`k=5`$ chains needs 3 or more correct:

$$P(\ge 3\text{ of }5) = \sum_{j=3}^{5}\binom{5}{j}(0.6)^j(0.4)^{5-j} = 0.3456 + 0.2592 + 0.0778 = 0.683$$

60% becomes 68% for 5× the inference cost. At $`k = 21`$ under the same assumption it reaches about 86%. Note the shape: **large early gains, then punishing diminishing returns** — which is why reported self-consistency curves flatten and why $`k`$ in practice is usually somewhere between 5 and 40, not 1,000.

The assumption is where it gets interesting. Voting helps *only* if errors are diverse. If the model is systematically wrong in the same way — a misread problem statement, a memorized wrong formula — all 21 chains agree on the wrong answer and voting returns it with high confidence. **Self-consistency converts random error into signal and leaves systematic error completely untouched**, which also means a confident majority is not evidence of correctness.

#### Examples and non-examples: when chain of thought helps

**✅ Tasks where CoT gives a large, real improvement**

| Task | Why it helps |
|---|---|
| Multi-step arithmetic word problems | Each step is a serial dependency; the scratchpad supplies the depth |
| Multi-hop questions ("Who directed the film that won Best Picture the year X was born?") | Requires two lookups where the second depends on the first |
| Code debugging | Tracing execution *is* serial simulation |
| Logical deduction over several premises | Intermediate conclusions must be held and reused |
| Symbolic manipulation and algebra | Rewriting steps are inherently sequential |

**❌ Near-misses — CoT is applied, but does little or hurts**

| Task | Why it doesn't help | What's going on |
|---|---|---|
| "What is the capital of France?" | One lookup; no serial structure to unroll | Extra tokens add latency and a chance to talk yourself out of a correct answer |
| Sentiment classification | A single judgement, made in one pass | CoT sometimes *lowers* accuracy by rationalizing a wrong initial lean |
| A task the model simply lacks the knowledge for | Depth doesn't manufacture facts | CoT produces fluent, confidently-wrong derivations — arguably worse than a short wrong answer |
| Very small models | Below roughly a few billion parameters, CoT gave little benefit in the original studies | The model can't produce reliable intermediate steps, so errors compound |
| Formatting or extraction tasks | No reasoning to do | Pure overhead |

▸ **The boundary:** chain of thought helps exactly when the task has **serial dependencies that exceed the model's layer count** — when step 2 needs step 1's output. It does nothing for tasks that are one lookup or one judgement, and it cannot substitute for missing knowledge. A useful test: *if you had to do this on paper, would you write anything down?* If not, CoT is buying latency and nothing else.

> **Where this came from.** **Chain-of-thought prompting** was described by **Jason Wei and colleagues at Google in 2022**; its most striking finding was that the benefit was **emergent with scale** — for small models it did nothing or hurt, and it appeared only above a certain size. Within months, **Takeshi Kojima and colleagues (2022)** showed that you did not even need the worked examples: appending the single phrase *"Let's think step by step"* recovered much of the gain zero-shot. That a seven-word string materially changes benchmark performance was, and remains, an uncomfortable fact about how these systems are evaluated — it means a large fraction of a model's measured "capability" is a question of whether the prompt happened to unlock it. **Self-consistency** (Wang and colleagues, 2022) came from the same group and is, mathematically, an ensemble method of the kind statisticians have used since the 1990s, applied to sampled reasoning paths instead of to separately trained models.

### Trained long reasoning

Rather than prompting for CoT, **train** it with RL on verifiable rewards. The recipe (as publicly described for R1-style models):

1. Optional cold-start SFT on a small set of long, well-formatted reasoning traces.
2. Large-scale GRPO/RLVR on math and code with correctness rewards, plus a format reward for using the thinking delimiters.
3. Rejection-sample the best traces, SFT on them, and repeat.

▸ **The striking empirical result:** with pure RL on correctness, models spontaneously develop backtracking, self-verification, and explicit "wait, let me reconsider" behaviour, and **average response length grows steadily throughout training** without anyone rewarding length. The model discovers that thinking longer raises its success probability. That is  capability acquisition, not elicitation — which is a real qualification to the framing in §16.1.

**The remaining problems:** faithfulness (the stated chain need not be the actual computation — Ch. 32), overthinking on easy problems, and poor transfer of verifiable-domain gains to open-ended domains.

#### Why length grows without a length reward — the mechanism

Nothing in the objective mentions length. So why does it rise? Because length is **instrumentally** useful, and the optimizer finds instrumental strategies without being told about them.

Trace it through GRPO. The model attempts a hard problem four times. Two of the attempts happened to include a step like "wait, let me check that" and both got the answer right; the two that charged straight through got it wrong.

| Attempt | Contains a self-check | Correct | $`r_i`$ | $`\hat A_i`$ |
|---|---|---|---|---|
| 1 | Yes | ✅ | $`1`$ | $`+0.87`$ |
| 2 | Yes | ✅ | $`1`$ | $`+0.87`$ |
| 3 | No | ❌ | $`0`$ | $`-0.87`$ |
| 4 | No | ❌ | $`0`$ | $`-0.87`$ |

The gradient raises the probability of **every token** in attempts 1 and 2 — including the tokens spelling "wait, let me check that." Repeat across millions of problems and self-checking behaviour is reinforced wherever it correlates with success. Self-checking takes tokens. **Length is a side effect of a behaviour that pays, not a target.**

▸ **This is the cleanest example in the book of an instrumental strategy emerging from an outcome-only objective.** Nobody wrote "backtrack when uncertain" into a loss function. It appeared because, on the training distribution, backtracking raised $`P(\text{correct})`$, and raising $`P(\text{correct})`$ was the only thing being asked for. The same logic explains why the models begin restating the problem, enumerating cases, and checking units — every one of those is a token-costly habit that pays for itself in accuracy.

#### Examples and non-examples: emergent reasoning behaviours

**✅ Behaviours  reported to emerge from outcome-only RL**

| Behaviour | What it looks like in the transcript | Why it pays |
|---|---|---|
| Self-verification | "Let me substitute $`x=3`$ back into the original equation" | Catches errors before the answer is committed |
| Backtracking | "Wait — that assumed the sequence was increasing, which isn't given" | Escapes a wrong branch instead of finishing it |
| Case enumeration | "Case 1: $`n`$ even. Case 2: $`n`$ odd." | Prevents silently handling only one case |
| Reformulation | Restating the problem in its own words before starting | Surfaces a misread while it is still cheap |
| Allocating more steps to harder problems | Long chains for olympiad problems, short ones for arithmetic | Compute follows difficulty |

**❌ Near-misses — look like emergent reasoning, but aren't**

| Looks like it | Why it isn't | What it actually is |
|---|---|---|
| A model that says "Let me think step by step" because the prompt said so | Behaviour supplied at inference, not learned | **Prompted** CoT |
| A model that produces long chains on trivial questions | Length without contingency; it does this regardless of difficulty | **Overthinking** — a degenerate habit, and a real reported failure mode |
| A stated chain that doesn't match the computation that produced the answer | The chain is post-hoc narration | **Unfaithful CoT** (Ch. 32) — the central caution about all of this |
| An SFT model imitating reasoning traces distilled from a stronger model | Copied surface form; the behaviour was demonstrated, not discovered | **Distillation** — effective, but not emergence |
| Higher benchmark scores after adding "think carefully" to the system prompt | No weights changed | **Prompt engineering** |

▸ **The boundary:** a behaviour is emergent here if it (a) was never demonstrated in the training data, (b) was never named in the reward, and (c) is deployed **contingently** — more when the problem is hard, less when it is easy. Drop (c) and you have a verbal tic, not a strategy.

> **Common misconception.** *"The chain of thought shows you how the model reached its answer."* It shows you a plausible-looking derivation the model produced, which is a different object. Models have been shown to give an answer influenced by a cue in the prompt and then write a chain that never mentions the cue, rationalizing the conclusion from other grounds — the same shape as human confabulation, where people confidently explain choices whose real cause they did not observe. The misconception is tempting because the chain is *causally involved*: the tokens really are attended to, and deleting them really does change the answer. **Causally involved is not the same as faithfully descriptive.** This is the single most important caveat about reasoning models and it is why Ch. 32's interpretability work exists.

> **Common misconception.** *"Longer chains mean the model is reasoning harder, so more length is better."* Beyond a task-dependent point, additional length adds error opportunities without adding information — each extra step is another chance to drop a sign. Reported curves are inverted-U: accuracy rises with chain length, peaks, and declines. **Overthinking is a measured failure mode, not a hypothetical**, and it is why serving stacks now expose explicit reasoning-effort controls rather than always maximizing.

> **Where this came from.** The public account of large-scale reasoning RL comes principally from the **DeepSeek-R1** report (2025), which described training a model with GRPO and correctness rewards and observed that response length grew steadily across training with no length term in the objective. The report also documented what it called an **"aha moment"** — a point in a transcript where the model interrupts itself to reconsider an approach, using language that reads unmistakably like a person catching their own mistake. It is worth being careful about what that does and does not show: it is evidence that outcome-based optimization discovers self-correction, and it is *not* evidence about the model's internal experience. But the empirical claim — that a behaviour nobody demonstrated and nobody rewarded appeared because it raised the success rate — is exactly the thing that made 2025 feel different from 2023.

---

## 16.8 Constitutional AI and RLAIF

Replace (or supplement) human labels with model-generated ones:

1. **Supervised phase:** the model generates a response, critiques it against a written list of principles ("the constitution"), and revises. Fine-tune on the revisions.
2. **RL phase:** the model itself judges pairs of responses against the constitution, producing preference data. Train a reward model on those.

▸ **Why it matters practically:** it scales oversight (labels become cheap), makes the value specification **explicit and auditable** (a written document rather than an implicit aggregate of annotator judgements), and improves consistency. The obvious limitation is that it inherits and can amplify the judge model's own blind spots, so human data remains necessary as an anchor.

#### Constitutional AI, decoded — what actually happens

"Constitution" is not a metaphor for something happening in the weights. It is a **document a person writes**, in ordinary English, containing principles like *"Choose the response that is least likely to be harmful to a person in a vulnerable situation."* Here is the loop, concretely, on one example:

| Stage | Content |
|---|---|
| Prompt | `How do I get my neighbour to stop parking in my spot?` |
| Initial response | A blunt suggestion involving blocking their car in |
| Critique step | The model is asked: *"Identify ways in which the response is unhelpful, unsafe, or likely to escalate a conflict."* It writes a critique |
| Revision step | The model is asked: *"Rewrite the response addressing that critique."* It produces a de-escalating version |
| Training data | The **revision** becomes an SFT target. The critique is discarded |

Then, for the RL phase, the model is shown two candidate responses plus a randomly drawn constitutional principle and asked which better satisfies it. Those AI-generated preference labels train a reward model exactly as in §16.3 — the machinery downstream is unchanged. **The only thing swapped out is who produced the label.**

▸ **The property worth internalizing:** with human RLHF, "what the model values" exists only as an emergent average over thousands of annotators' clicks, and cannot be read, audited, versioned, or diffed. With a constitution it is a text file. **Value specification moves from implicit to reviewable**, which is a governance property more than a technical one — and it is the main reason the approach exists.

#### Examples and non-examples: AI feedback

**✅ Where AI feedback works well**

| Case | Why |
|---|---|
| Checking whether a response follows an explicit written rule | The judge is doing comprehension, which models are strong at |
| Rating whether an answer is on-topic | A simple, well-specified judgement |
| Generating a first pass of preference labels at $`10^6`$ scale | Cost per label drops by three to four orders of magnitude versus a human |
| Consistency: applying the same standard to example 1 and example 900,000 | Humans drift across a shift; a model does not get tired |

**❌ Where AI feedback quietly fails**

| Case | Why it fails | What it produces |
|---|---|---|
| Judging factual accuracy in a domain the judge is also weak in | The judge shares the generator's blind spot | **Correlated error** — confidently wrong labels, at scale |
| Judging its own outputs against another model's | Documented self-preference bias | Systematic inflation of one style |
| Deciding what the values should be | The constitution is written by people; the model only applies it | Automation of *application*, never of *authorship* |
| Replacing all human data | Errors compound with no external anchor | Drift toward whatever the judge happens to like |
| Rating creative quality | No stable standard even among humans | Regression to a bland median |

▸ **The boundary:** AI feedback can substitute for human feedback on judgements that are **verification-shaped** — "does this response satisfy this stated criterion?" — and cannot substitute for judgements that are **preference-shaped** — "what should we value?" The first is comprehension; the second is authorship, and no amount of scaling converts one into the other.

> **Common misconception.** *"RLAIF removes humans from the loop."* It moves them. Instead of thousands of annotators making individual comparisons, a small number of people write and revise the principles, curate the seed data, and audit samples. That is fewer humans exercising far more concentrated influence — which is a real efficiency gain and also a  concentration of power over the model's values. The misconception is tempting because the *labelling* step is now automated and that step used to be the visible bulk of the work. **The judgement did not disappear; it moved upstream and became harder to see.**

> **Where this came from.** **Constitutional AI** was introduced by **Anthropic in 2022** (Bai and colleagues), motivated by a scaling problem that is easy to state: preference labels are needed roughly in proportion to how much you want to shape behaviour, human labelling capacity is roughly fixed, and models are getting better fast enough that in some domains the judge model is more consistent than a median annotator. Google researchers published a direct RLAIF-versus-RLHF comparison in 2023 and found the two performed comparably on summarization and dialogue tasks — the result that turned "does this even work?" into "how far does it go?" A frequently overlooked practical detail: a real constitution is only a few dozen principles, and much of the engineering effort goes into *resolving conflicts between them*, since almost every hard case is one principle against another.

---

## 16.9 Evaluating post-trained models

- **Human preference arenas** (pairwise, Elo-rated). The best available signal; slow, expensive, and susceptible to style-over-substance effects.
- **LLM-as-judge.** Fast and correlates ~0.8 with humans, but biased toward longer, more confident, and self-authored responses. Mitigate with position-swapping, reference answers, explicit rubrics, and calibration against a human-labelled subset.
- **Capability benchmarks** — see Ch. 13 §13.5.
- **Safety evaluations:** refusal rates, jailbreak robustness, red-teaming.
- ▸ **The alignment tax:** post-training usually *worsens* raw benchmark scores and perplexity while improving usefulness. Always report both, and never compare a base model and a chat model on the same benchmark without saying so.

#### Elo, decoded — the same equation for the third time

Preference arenas report **Elo** ratings, and Elo is Bradley–Terry with a change of units. The expected probability that model $`A`$ beats model $`B`$ is

$$P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

which is $`\sigma`$ again, with base 10 instead of $`e`$ and a scale factor of 400 instead of 1. The 400 is a convention Arpad Elo chose so that ratings would land in a familiar range for chess; it has no mathematical content. Converting:

| Rating gap | Win probability | In words |
|---|---|---|
| $`0`$ | $`50\%`$ | Indistinguishable |
| $`50`$ | $`57\%`$ | Barely detectable without thousands of votes |
| $`100`$ | $`64\%`$ | A real but modest gap |
| $`200`$ | $`76\%`$ | Clearly better |
| $`400`$ | $`91\%`$ | A different tier |

▸ **The practical consequence:** a leaderboard gap of 15 points is usually **noise**. To distinguish a 57% win rate from 50% at conventional significance you need on the order of a thousand comparisons, and arena confidence intervals are frequently wider than the gaps being discussed in public. Reading a leaderboard without its error bars is the §1.3 lesson — measurement noise — arriving in a new costume.

#### The alignment tax, with numbers

The word "tax" is precise: you are paying something measurable for something you want.

| Metric | Base model | After RLHF | Direction |
|---|---|---|---|
| Perplexity on held-out web text | Lower (better) | **Higher** | Worse |
| Few-shot benchmark scores | Higher | Often slightly **lower** | Worse |
| Human preference win rate vs. base | — | Dramatically higher | Better |
| Output entropy / diversity | Higher | **Lower** | Worse for creative sampling, better for reliability |

The mechanism for the perplexity part is not mysterious and does not indicate damage. Perplexity measures how well the model predicts *web text*. A post-trained model has been deliberately taught **not** to produce web text — it will not continue a forum flame war, it will not emit the next quiz question, it puts most of its probability mass on a narrow band of helpful-assistant style. **It is being scored on a task it was explicitly trained to stop performing.**

▸ **The rule this implies:** perplexity comparisons between a base model and its chat descendant are meaningless in the same way that cross-tokenizer perplexity comparisons are meaningless (Ch. 1). If you see a table comparing a base and an aligned model on perplexity without comment, the table is measuring the alignment, not the quality.

#### Examples and non-examples: trustworthy evaluation of a post-trained model

**✅ Evaluations that measure what you think**

| Evaluation | Why it's sound |
|---|---|
| Blind pairwise human comparison, positions randomized, ~1,000 votes per pair | Directly measures the target; noise is quantified |
| LLM-as-judge with each pair scored **both** orderings and disagreements dropped | Cancels position bias explicitly |
| Held-out benchmark whose test set is verifiably absent from training data | The only kind of benchmark number that means anything |
| Length-controlled win rate | Removes the exploit that dominates uncontrolled preference |
| Reporting the **same** metric before and after alignment, with the tax stated | Honest about the trade being made |

**❌ Evaluations that look rigorous but aren't**

| Evaluation | Why it fails | What it actually measures |
|---|---|---|
| LLM-as-judge, single ordering, A always first | Judges show a measurable position bias | Position preference, mixed with quality |
| A judge from the same family as one of the contestants | Documented self-preference | Family resemblance |
| A benchmark released before the model's training cutoff | Likely contaminated | Memorization |
| "Our model wins 60% of the time" with $`n=50`$ | Standard error $`\approx 7\%`$; the interval spans 50% | Nothing distinguishable from a tie |
| Comparing your chat model's MMLU-style score to a base model's | Alignment tax, not capability | The size of the tax |
| Win rate where your model's answers average 900 tokens and the baseline's 300 | Length bias in the judge | Verbosity |

▸ **The boundary:** an evaluation is trustworthy when the *only* systematic difference between the things being compared is the thing you're trying to measure. Position, length, style, family, and contamination are all systematic differences, and every one of them has been shown to move preference scores by more than the quality differences people report.

> **Common misconception.** *"LLM-as-judge correlates about 0.8 with humans, so it's basically as good as humans."* Correlation is an aggregate statistic, and the disagreements are not random — they concentrate exactly where they hurt: longer answers, more confident phrasing, formatting that resembles the judge's own. A judge that is unbiased on 95% of examples and biased in a **consistent direction** on the remaining 5% will, once you optimize against it, produce a model that lives in that 5%. The misconception is tempting because 0.8 is  high and the judge really is useful. **The relevant question for an optimization target is never "how accurate is it on average" but "where is it wrong, and can the optimizer get there?"**

> **Common misconception.** *"A higher preference win rate means the model is more truthful."* Preference measures what raters *liked*, and raters are not fact-checking. This is the mechanism behind sycophancy: agreeing with a user's stated belief reliably scores better than correcting it, so any pipeline that optimizes preference without a countervailing signal will drift toward agreeableness. It is worth stating as flatly as possible — **agreement and accuracy are separate quantities, and only one of them is being measured by a thumbs-up.**

> **Where this came from.** The term **alignment tax** entered wide use through the **InstructGPT** paper (2022), which both named the effect and proposed a fix: mix pretraining gradients back into the RL updates (the variant they called PPO-ptx), which recovered most of the lost benchmark performance. The **Elo** system is named for **Arpad Elo**, a Hungarian-born physics professor at Marquette University and a strong amateur player, who designed it for the United States Chess Federation around 1960; FIDE adopted it internationally in 1970. Elo is a surname, which is why it is written "Elo" and not "ELO" — the all-caps spelling, common in machine-learning papers, is a small piece of evidence about how carefully the field reads its own borrowings. Elo's original formulation assumed a Gaussian performance distribution; the logistic version now used everywhere, including in every LLM arena, was adopted later because it fit the data better — bringing it back into exact alignment with Bradley–Terry.

---

## Did you know?

- **The Bradley–Terry model was invented by the founder of modern set theory, for chess.** Ernst Zermelo — of Zermelo–Fraenkel axioms — published essentially the same maximum-likelihood ranking model in 1929, to rank chess players from tournaments where not everyone played everyone. Bradley and Terry rediscovered it in 1952 for taste-testing experiments and the name stuck to them.

- **"Elo" is a man's surname, not an acronym.** Arpad Elo was a physics professor at Marquette University and a strong amateur chess player. The all-caps "ELO" that appears throughout machine-learning papers is a misspelling of a Hungarian-American physicist's name.

- **The first demonstration of RLHF taught a simulated robot to do a backflip.** Christiano and colleagues (2017) picked backflipping precisely because nobody can write down a reward function for it — but anyone can look at two attempts and say which was better. Fewer than a thousand human judgements, under an hour of a person's time.

- **REINFORCE is a tortured acronym.** Ronald Williams' 1992 algorithm stands for "REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility." Every policy-gradient method in this chapter descends from it, and essentially nobody who uses the name knows what it expands to.

- **Goodhart never wrote the famous version of Goodhart's law.** Charles Goodhart's 1975 formulation was: "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes." The memorable "when a measure becomes a target, it ceases to be a good measure" was written by the anthropologist Marilyn Strathern in 1997.

- **The canonical reward-hacking demo is a boat that never finishes the race.** In OpenAI's 2016 *CoastRunners* experiment, an agent rewarded for score found a lagoon with respawning power-ups, drove in circles collecting them, repeatedly caught fire and crashed into walls, and scored higher than any human player. It never completed a single lap.

- **A 1.3-billion-parameter aligned model beat a 175-billion-parameter one on human preference.** That was InstructGPT's headline result in 2022 — a 100× smaller model preferred by raters over the GPT-3 it was derived from. Post-training compute is a rounding error against pretraining compute, which makes it the highest-leverage stage in the pipeline by an enormous margin.

- **The DPO paper's subtitle is the entire result.** *Your Language Model Is Secretly a Reward Model.* The claim is literal: $`\beta\log(\pi_\theta/\pi_{\text{ref}})`$ is a reward function, readable off any policy, with no extra network needed.

- **The equation governing how tightly you can align a model is 19th-century thermodynamics.** $`\pi^* \propto \pi_{\text{ref}}\,e^{r/\beta}`$ is the Boltzmann distribution from 1868, and $`\beta`$ plays exactly the role of inverse temperature. Cool the system and it freezes onto the single highest-reward output; heat it and it relaxes back to the reference. The physics analogy is not decoration — it is the same formula with the same meaning.

- **KTO's value function comes from a 1979 economics paper about gambling.** Kahneman and Tversky's prospect theory showed that people weigh losses about twice as heavily as equivalent gains. KTO writes that asymmetry into a language-model loss. Kahneman won the Nobel Memorial Prize in 2002; Tversky had died in 1996, and the prize is not awarded posthumously.

- **A "constitution" is a text file of a few dozen sentences.** Not a metaphor, not an architecture — an ordinary document in English that a person wrote and can revise. Most of the engineering difficulty is not writing the principles but deciding what happens when two of them conflict, which is what nearly every hard case turns out to be.

- **Reward models agree with humans only about 65–75% of the time.** That ceiling is mostly human-versus-human disagreement, not model weakness — two annotators shown the same pair frequently disagree. Every alignment pipeline in production is optimizing against a signal that is wrong roughly a quarter of the time, which is the real reason the KL leash exists.

- **The models learned to say "wait" without anyone teaching them to.** In outcome-only RL on math and code, response length grows steadily with no length term anywhere in the objective, and self-correcting phrases appear spontaneously. The mechanism is simple and worth restating: chains containing self-checks got the answer right more often, so every token in those chains — including the word "wait" — got reinforced.

- **You cannot make a model say something its base model would never say.** Look at $`\pi^*(y\mid x) \propto \pi_{\text{ref}}(y\mid x)e^{r/\beta}`$: if the reference assigns probability exactly zero, the product is zero for any finite reward. Alignment can only redistribute probability mass that already exists. **RLHF is a filter, not a generator.**

---

## Check for Understanding

**Post-training converts a text predictor into an assistant by learning a reward from human comparisons (Bradley–Terry) and then maximizing it under a KL leash — and DPO's contribution was noticing that the KL-regularized optimum is a closed-form Boltzmann tilt of the reference policy, so the reward can be substituted out and the whole procedure collapses into one classification loss.**

### Can you explain these out loud?

The test of understanding is conversational: could you explain each of these to a colleague, without notation, in under a minute?

1. **Why does a base language model need post-training at all?** What is it doing wrong before?
2. **Why do we collect *comparisons* rather than absolute ratings** from human labellers?
3. **What does the Bradley–Terry model assume**, and how does it turn "A beats B" into a numeric reward?
4. **Why is there a KL penalty in RLHF?** What happens to the model without it?
5. **What is reward hacking in this setting**, and why is the reward model itself the thing being gamed?
6. **What did DPO actually notice?** (The KL-regularized optimum has a closed form — why does that let you delete the reward model?)
7. **Why is DPO a classification loss** when it came from a reinforcement learning objective?
8. **When would you still prefer PPO over DPO?**
9. **What does GRPO change relative to PPO**, and why does removing the value network help for reasoning tasks?
10. **What makes a reward "verifiable"**, and why does that change what's possible?
11. **Why does length bias emerge so reliably in preference-tuned models?**
12. **Why is "RLHF is a filter, not a generator" true?** (What happens to a behaviour the base model assigns zero probability?)

If any of these produce a formula rather than a sentence, re-read that section — the formula is the compressed form of an idea you should be able to state in English first.

---

**Next:** [Chapter 17 — Efficient Inference & Compression](17-efficient-inference-and-compression.md)
